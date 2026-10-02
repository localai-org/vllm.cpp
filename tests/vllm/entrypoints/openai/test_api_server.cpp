// Tests for the OpenAI HTTP api_server (M3.1 Task 4). Two layers:
//   1. HANDLER-DISPATCH (no socket, the primary coverage): drive the ApiServer
//      handle_* methods with request bodies over a small synthetic AsyncLLM and
//      assert the OpenAI response shape, SSE framing, error status codes and
//      that a completion serializes without a 500 from invalid UTF-8.
//   2. SOCKET SMOKE (real HTTP over an ephemeral port via cpp-httplib's client):
//      start the server on a background thread, issue real requests to /health,
//      /v1/models, a non-streaming + streaming /v1/completions and a
//      /v1/chat/completions, asserting the framing end to end.
//
// The synthetic model mirrors tests/vllm/entrypoints/openai/test_serving.cpp
// (tiny hybrid-MoE Qwen3.6 + the BPE fixture, vocab ids 0..21).
#include "vllm/entrypoints/openai/api_server.h"
#include "vllm/entrypoints/openai/video_api.h"
#include "vllm/multimodal/parakeet_transcription.h"
#include "vllm/multimodal/speech_engine.h"
#include "support/test_env.h"

#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

// SERVE-HTTP-TRANSPORT: the behavioral TCP_NODELAY assertion reads the accepted
// server socket's options in-process via /proc/self/fd (Linux only; the CPU
// test tier runs on Linux).
#if defined(__linux__)
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#endif

#include "vllm/config/device.h"
#include "vllm/config/scheduler.h"
#include "vllm/config/speculative.h"
#include "vllm/config/multimodal.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platform/console_shutdown.h"
#include "vllm/platform/process.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/spec_decode/metrics.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::SchedulerConfig;
using vllm::entrypoints::openai::ApiServer;
using vllm::entrypoints::openai::ConfigureUtilityEndpoints;
using vllm::entrypoints::openai::UtilityEndpointOptions;
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
using vllm::entrypoints::openai::OpenAIServingModels;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
using vllm::v1::AsyncLLM;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GPUModelRunner;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::LLMEngine;
using vllm::v1::MambaSpec;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// ─── Threads that can still report a failure (#584) ──────────────────────────
//
// Every socket case below runs the server on a background thread and asserts
// against it. A bare `std::thread` held across those assertions makes the file
// unable to report anything at all, through two separate std::terminate paths:
//
//   1. `~thread` on a JOINABLE thread calls std::terminate ([thread.thread.destr]).
//      A failing REQUIRE, or a `json::parse` on an unexpected body, unwinds past
//      the thread object and ends the process.
//   2. An exception escaping a thread's initial function is std::terminate too
//      ([except.handle]/9), so a throw inside `serve()` does the same.
//
// On MSVC std::terminate reaches `abort()`, which is `__fastfail`, which raises
// status 0xC0000409 and bypasses SEH by design. doctest's Windows handler never
// runs and its buffered stdout is discarded, so a NAMED assertion failure
// arrives in CI as an opaque exit code with no `Status:` and no `assertions:`
// line — which is exactly what #584 has printed on both Windows lanes.
//
// These two types close both paths. They do not claim to fix whatever #584's
// fast-fail actually is; they make the run able to say so.

// One thread, joined by the destructor on every path. The body runs inside a
// catch-all and the escaped exception is rethrown by `join()`, which is a
// synchronisation point, so the store and the load do not race. The DESTRUCTOR
// never rethrows: throwing while unwinding is the failure this type exists to
// prevent. `stop_request` runs before the join, for a body that waits on
// something and would otherwise never return.
class ScopedThread {
 public:
  template <typename Body>
  explicit ScopedThread(Body&& body, std::function<void()> stop_request = {})
      : stop_request_(std::move(stop_request)),
        escaped_(std::make_shared<std::exception_ptr>()),
        thread_([slot = escaped_, fn = std::forward<Body>(body)]() mutable {
          try {
            fn();
          } catch (...) {
            *slot = std::current_exception();
          }
        }) {}

  ScopedThread(const ScopedThread&) = delete;
  ScopedThread& operator=(const ScopedThread&) = delete;
  // Movable so a vector of them can exist; the body captured the exception slot
  // BY VALUE rather than capturing `this`, so a move leaves no dangling handle.
  ScopedThread(ScopedThread&&) = default;
  // Move ASSIGNMENT stays deleted: assigning onto a joinable thread is itself
  // std::terminate, and nothing here needs it.
  ScopedThread& operator=(ScopedThread&&) = delete;

  ~ScopedThread() { stop_and_join(); }

  // Join and surface an exception the body swallowed, at a point where doctest
  // can translate and name it.
  void join() {
    stop_and_join();
    if (escaped_ && *escaped_) {
      std::exception_ptr e = *escaped_;
      *escaped_ = nullptr;
      std::rethrow_exception(e);
    }
  }

 private:
  void stop_and_join() noexcept {
    if (!thread_.joinable()) return;
    if (stop_request_) {
      // A throwing stop action would defeat the whole point on the unwind path.
      try {
        stop_request_();
      } catch (...) {
      }
    }
    thread_.join();
  }

  std::function<void()> stop_request_;
  std::shared_ptr<std::exception_ptr> escaped_;
  std::thread thread_;  // declared last: constructed after the slot it reads
};

// `ScopedThread` for the shape that dominates this file — an ApiServer served on
// a background thread. It owns the `stop()` as well as the join, so a case that
// throws before its stop line is reached still ends.
//
// The stop action waits for the accept loop first. `httplib::Server::stop()` is
// a no-op while `is_running_` is false (`third_party/httplib/httplib.h:11460`),
// and `listen_internal` raises that flag only once it is in the loop (`:12027`),
// so stopping too early would leave the destructor blocked in `join()` forever —
// turning a fast-fail into a CI timeout, which is a worse instrument, not a
// better one. The bound is the same 500 x 2 ms the call sites already used.
//
// It owns the stop EXCLUSIVELY, and the call sites no longer call
// `h.server.stop()` themselves. A second `stop()` is not a no-op: it sees
// `is_running_` still true while the accept loop unwinds and `svr_sock_` already
// exchanged to INVALID_SOCKET, which trips `assert(svr_sock_ != INVALID_SOCKET)`
// at `httplib.h:11462` on every build that is not NDEBUG — which is this suite's
// own Linux build.
class ScopedServerThread {
 public:
  explicit ScopedServerThread(ApiServer& server)
      : thread_([&server] { server.serve(); },
                [&server] {
                  for (int i = 0; i < 500 && !server.is_running(); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                  server.stop();
                }) {}

  void join() { thread_.join(); }

 private:
  ScopedThread thread_;
};

// ─── Synthetic weights (mirrors test_serving.cpp) ────────────────────────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 22;
constexpr int kBlockSize = 32;
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

KVCacheConfig MakeKvConfig(const HfConfig& c) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh, DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(
          kBlockSize,
          std::vector<std::vector<int64_t>>{{conv_dim, Kw - 1},
                                            {Hv, Dv, Dk}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_apisrv_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},    {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11},  {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15},  {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

// In-vocab chat prompt seam (the fixture vocab is ids 0..21).
std::string InVocabChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<vllm::entrypoints::openai::ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&) {
  std::string p;
  for (const ChatMessage& m : messages)
    if (m.content.has_value()) p += *m.content;
  return p;
}

// A fully-wired serving stack + ApiServer over the synthetic engine.
struct ServerHarness {
  ServerHarness(const HfConfig& c, const Qwen3_5MoeWeights& w,
                const Tokenizer& tok,
                bool enable_force_include_usage = false,
                size_t max_concurrent_streams =
                    ApiServer::kDefaultMaxConcurrentStreams,
                vllm::entrypoints::openai::ChatPromptFn chat_prompt =
                    InVocabChatPrompt)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, kMaxModelLen * 8),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        async_engine(input_processor, scheduler, executor, output_processor,
                     Hasher()),
        models("test-model"),
        completion(async_engine, "test-model", enable_force_include_usage),
        chat(async_engine, "test-model", std::move(chat_prompt), "hermes",
             /*reasoning_parser_name=*/std::string(), enable_force_include_usage),
        server(completion, chat, models, "9.9.9", max_concurrent_streams) {}

  static SchedulerConfig MakeSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = kMaxModelLen * 8;
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kMaxModelLen;
    cfg.watermark = 0.0;
    return cfg;
  }
  static vllm::v1::BlockHasher Hasher() {
    static bool init = false;
    if (!init) {
      init_none_hash(sha256_cbor);
      init = true;
    }
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  AsyncLLM async_engine;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

#if defined(__linux__)
// Locate the ApiServer's accepted connection socket in this process and read its
// TCP_NODELAY. The test client and the server share one process (the server runs
// on a background thread over loopback), so the accepted socket fd is
// discoverable: it is the AF_INET socket whose local port is the server's listen
// port and whose peer port is the client's ephemeral port. That pair uniquely
// distinguishes it from the listening socket (which has no peer → getpeername
// fails) and from the client socket (whose local port is the client port).
// Returns the getsockopt TCP_NODELAY value, or -1 if the socket is not found.
int AcceptedSocketTcpNoDelay(int server_port, int client_port) {
  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) return -1;
  int result = -1;
  for (struct dirent* entry; (entry = readdir(dir)) != nullptr;) {
    char* end = nullptr;
    const long fd = std::strtol(entry->d_name, &end, 10);
    if (end == entry->d_name || *end != '\0') continue;

    sockaddr_in local{};
    socklen_t local_len = sizeof(local);
    if (getsockname(static_cast<int>(fd),
                    reinterpret_cast<sockaddr*>(&local), &local_len) != 0)
      continue;
    if (local.sin_family != AF_INET) continue;
    if (static_cast<int>(ntohs(local.sin_port)) != server_port) continue;

    sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    if (getpeername(static_cast<int>(fd),
                    reinterpret_cast<sockaddr*>(&peer), &peer_len) != 0)
      continue;  // listening socket has no peer
    if (peer.sin_family != AF_INET) continue;
    if (static_cast<int>(ntohs(peer.sin_port)) != client_port) continue;

    int nodelay = -1;
    socklen_t opt_len = sizeof(nodelay);
    if (getsockopt(static_cast<int>(fd), IPPROTO_TCP, TCP_NODELAY, &nodelay,
                   &opt_len) == 0) {
      result = nodelay;
    }
    break;
  }
  closedir(dir);
  return result;
}
#endif  // defined(__linux__)

// ─── SPEC-DECODE /metrics (SERVE-METRICS, #2770) ─────────────────────────────
// A serving stack whose engine actually SPECULATES, so the exposition the
// server hands a scraper can be compared against what the drafter really did.
//
// The runner is a stub rather than the GPUModelRunner the other harness uses:
// drafting needs an MTP head, the synthetic checkpoint above has none, and a
// harness that cannot draft would make the acceptance counter read zero for a
// reason that has nothing to do with the wiring under test.

// Deterministic drafting model double. Request output position `p` is Tok(p);
// the drafter proposes the true token except at every third position, so
// acceptance is partial and num_accepted_tokens != num_draft_tokens. Every id
// stays inside the fixture vocab (0..21) so the detokenizer can decode it.
class DraftingRunnerStub : public vllm::v1::ModelRunnerBase {
 public:
  explicit DraftingRunnerStub(int k) : k_(k) {}

  static int32_t Tok(int p) { return static_cast<int32_t>((p * 7 + 3) % 10); }
  int32_t Draft(int p) const { return (p % 3 == 2) ? Tok(p) + 1 : Tok(p); }

  std::optional<vllm::v1::ModelRunnerOutput> execute_model(
      const vllm::v1::SchedulerOutput& scheduler_output) override {
    stashed_ = scheduler_output;
    return std::nullopt;
  }

  vllm::v1::ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>&) override {
    vllm::v1::ModelRunnerOutput mro;
    int idx = 0;
    vllm::v1::DraftTokenIds fresh;
    for (const auto& [req_id, n] : stashed_.num_scheduled_tokens) {
      (void)n;
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      int& pos = out_pos_[req_id];

      std::vector<int32_t> drafts;
      const auto it = stashed_.scheduled_spec_decode_tokens.find(req_id);
      if (it != stashed_.scheduled_spec_decode_tokens.end()) drafts = it->second;

      std::vector<int32_t> emitted;
      if (drafts.empty()) {
        emitted.push_back(Tok(pos));
      } else {
        int accepted = 0;
        for (std::size_t j = 0; j < drafts.size(); ++j) {
          if (drafts[j] != Tok(pos + static_cast<int>(j))) break;
          ++accepted;
        }
        // What the ENGINE should end up reporting, counted at the only place
        // that knows the truth. The scrape is compared against these.
        drafts_verified.fetch_add(1);
        draft_tokens_verified.fetch_add(static_cast<int64_t>(drafts.size()));
        draft_tokens_accepted.fetch_add(accepted);
        for (int d = 0; d < accepted && d < kMaxTrackedDepth; ++d) {
          accepted_per_pos[static_cast<std::size_t>(d)].fetch_add(1);
        }
        for (int i = 0; i <= accepted; ++i) emitted.push_back(Tok(pos + i));
      }
      pos += static_cast<int>(emitted.size());
      mro.sampled_token_ids.push_back(std::move(emitted));

      std::vector<int32_t> next;
      next.reserve(static_cast<std::size_t>(k_));
      for (int i = 0; i < k_; ++i) next.push_back(Draft(pos + i));
      fresh.req_ids.push_back(req_id);
      fresh.draft_token_ids.push_back(std::move(next));
    }
    pending_ = std::move(fresh);
    return mro;
  }

  std::optional<vllm::v1::DraftTokenIds> take_draft_token_ids() override {
    std::optional<vllm::v1::DraftTokenIds> out = std::move(pending_);
    pending_.reset();
    return out;
  }

  // Written on the ENGINE thread and read by the TEST thread while the engine
  // is still alive, so atomic rather than plain: the sanitizer build runs this
  // suite and a plain int64_t here would be a genuine data race, not a
  // pedantic one.
  static constexpr int kMaxTrackedDepth = 8;
  std::atomic<int64_t> drafts_verified{0};
  std::atomic<int64_t> draft_tokens_verified{0};
  std::atomic<int64_t> draft_tokens_accepted{0};
  std::array<std::atomic<int64_t>, kMaxTrackedDepth> accepted_per_pos{};

 private:
  const int k_;
  vllm::v1::SchedulerOutput stashed_;
  std::map<std::string, int> out_pos_;
  std::optional<vllm::v1::DraftTokenIds> pending_;
};

// The same serving stack as ServerHarness, over a speculative scheduler and the
// drafting stub, with the ONE PrometheusStatLogger constructed the way
// server_main.cpp constructs it: k from the resolved speculative config.
struct SpecServerHarness {
  explicit SpecServerHarness(const HfConfig& c, const Tokenizer& tok, int k)
        // server_main.cpp:1340-1342 declares the logger BEFORE the engine, so
        // reverse destruction order joins AsyncLLM before the non-owning
        // pointer it holds is released. Same order here, for the same reason.
      : logger("test-model", kSpecMaxModelLen, /*engine_index=*/0, k),
        scheduler(MakeSpecSchedulerConfig(), MakeSpecKvConfig(), kSpecBlockSize,
                  /*enable_caching=*/true,
                  /*structured_output_manager=*/nullptr,
                  vllm::SpeculativeConfig::ResolveMtp(
                      /*mtp_num_hidden_layers=*/1, k)),
        runner(k),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        async_engine(input_processor, scheduler, executor, output_processor,
                     get_request_block_hasher(kSpecBlockSize, sha256_cbor),
                     /*shutdown_timeout_s=*/0, /*max_concurrent_batches=*/1,
                     /*structured_output_manager=*/nullptr,
                     /*check_for_draft_tokens=*/true),
        models("test-model"),
        completion(async_engine, "test-model", false),
        chat(async_engine, "test-model", InVocabChatPrompt, "hermes",
             /*reasoning_parser_name=*/std::string(), false),
        server(completion, chat, models, "9.9.9",
               ApiServer::kDefaultMaxConcurrentStreams) {
    async_engine.set_stat_logger(&logger);
    server.set_metrics_logger(&logger);
  }

  static constexpr int kSpecBlockSize = 16;
  static constexpr int kSpecMaxModelLen = 512;

  static SchedulerConfig MakeSpecSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = 2048;
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kSpecMaxModelLen;
    cfg.watermark = 0.0;
    return cfg;
  }
  static KVCacheConfig MakeSpecKvConfig() {
    KVCacheConfig kv;
    kv.num_blocks = 512;
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"layer"},
        std::make_shared<FullAttentionSpec>(kSpecBlockSize, /*num_kv_heads=*/1,
                                            /*head_size=*/1, DType::kF32));
    return kv;
  }

  // The ONE logger, constructed the way server_main.cpp constructs it:
  // served_model_name, max_model_len, engine 0, and the RESOLVED
  // num_speculative_tokens. Declared first so it outlives the engine.
  vllm::v1::metrics::PrometheusStatLogger logger;
  Scheduler scheduler;
  DraftingRunnerStub runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  AsyncLLM async_engine;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

// The value of `name{...}` as it appears in a prometheus text exposition, or
// -1 when the series is absent. A counter renders as `<name>_total`.
double ScrapeSeries(const std::string& body, const std::string& series) {
  const std::size_t at = body.find("\n" + series);
  if (at == std::string::npos) return -1.0;
  const std::size_t start = at + 1 + series.size();
  const std::size_t end = body.find('\n', start);
  if (end == std::string::npos) return -1.0;
  return std::stod(body.substr(start, end - start));
}

}  // namespace

// ─── 1. Handler-dispatch (no socket) ─────────────────────────────────────────
TEST_CASE("api_server: non-stream completion dispatch → 200 + OpenAI shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"model":"test-model","prompt":"hello","max_tokens":5,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);

  CHECK(r.status == 200);
  CHECK_FALSE(r.streaming);
  CHECK(r.content_type == "application/json");
  json j = json::parse(r.body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("model") == "test-model");
  CHECK(j.at("id").get<std::string>().rfind("cmpl-", 0) == 0);
  CHECK(j.at("choices").at(0).at("finish_reason") == "length");
  CHECK(j.at("usage").at("completion_tokens") == 5);
}

TEST_CASE("api_server: streaming completion dispatch → SSE chunks ending [DONE]") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":5,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);

  CHECK(r.status == 200);
  CHECK(r.streaming);
  CHECK(r.content_type == "text/event-stream");
  REQUIRE(r.sse_stream != nullptr);
  std::vector<std::string> chunks;
  std::string chunk;
  while (r.sse_stream->next(chunk)) chunks.push_back(chunk);
  REQUIRE(chunks.size() >= 2);
  CHECK(chunks.back() == "data: [DONE]\n\n");
  // Each non-terminal chunk is a valid `data: {json}\n\n` frame.
  for (size_t i = 0; i + 1 < chunks.size(); ++i) {
    REQUIRE(chunks[i].rfind("data: ", 0) == 0);
    REQUIRE(chunks[i].substr(chunks[i].size() - 2) == "\n\n");
    json j = json::parse(chunks[i].substr(6, chunks[i].size() - 8));
    CHECK(j.at("object") == "text_completion");
  }
}

// Ported from tests/entrypoints/openai/completion/test_completion.py:
// test_completion_stream_options @ e24d1b24.
TEST_CASE("api_server: completion include_usage emits a final native-ID frame") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body = R"({
    "prompt":"hello", "max_tokens":5, "temperature":0.0, "stream":true,
    "stream_options":{"include_usage":true,
                      "continuous_usage_stats":false}
  })";
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  REQUIRE(result.status == 200);
  REQUIRE(result.sse_stream != nullptr);

  std::vector<json> frames;
  bool done_seen = false;
  std::string chunk;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") {
      done_seen = true;
      continue;
    }
    frames.push_back(json::parse(chunk.substr(6, chunk.size() - 8)));
  }
  REQUIRE(done_seen);
  REQUIRE(frames.size() >= 2);
  const json& usage_frame = frames.back();
  CHECK(usage_frame.at("choices").empty());
  REQUIRE(usage_frame.contains("usage"));
  CHECK(usage_frame.at("usage").at("prompt_tokens").get<int>() > 0);
  CHECK(usage_frame.at("usage").at("completion_tokens") == 5);
  CHECK(usage_frame.at("usage").at("total_tokens").get<int>() ==
        usage_frame.at("usage").at("prompt_tokens").get<int>() + 5);
  for (size_t i = 0; i + 1 < frames.size(); ++i) {
    CHECK_FALSE(frames[i].at("choices").empty());
    CHECK_FALSE(frames[i].contains("usage"));
  }
}

TEST_CASE("api_server: completion continuous usage is cumulative and conditional") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  SUBCASE("include_usage gates continuous usage and the final frame") {
    ServerHarness h(c, w, Fixture());
    const std::string body = R"({
      "prompt":"hello", "max_tokens":5, "temperature":0.0, "stream":true,
      "stream_options":{"include_usage":true,
                        "continuous_usage_stats":true}
    })";
    ApiServer::DispatchResult result = h.server.handle_completions(body);
    REQUIRE(result.sse_stream != nullptr);
    int previous_completion_tokens = 0;
    bool final_usage_seen = false;
    std::string chunk;
    while (result.sse_stream->next(chunk)) {
      if (chunk == "data: [DONE]\n\n") continue;
      const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
      REQUIRE(frame.contains("usage"));
      const json& usage = frame.at("usage");
      const int completion_tokens = usage.at("completion_tokens");
      CHECK(completion_tokens >= previous_completion_tokens);
      CHECK(usage.at("total_tokens").get<int>() ==
            usage.at("prompt_tokens").get<int>() + completion_tokens);
      if (frame.at("choices").empty()) {
        final_usage_seen = true;
        CHECK(completion_tokens == 5);
      } else {
        CHECK(completion_tokens > previous_completion_tokens);
      }
      previous_completion_tokens = completion_tokens;
    }
    CHECK(final_usage_seen);
    CHECK(previous_completion_tokens == 5);
  }

  SUBCASE("continuous=true is ignored when include_usage=false") {
    ServerHarness h(c, w, Fixture());
    const std::string body = R"({
      "prompt":"hello", "max_tokens":3, "temperature":0.0, "stream":true,
      "stream_options":{"include_usage":false,
                        "continuous_usage_stats":true}
    })";
    ApiServer::DispatchResult result = h.server.handle_completions(body);
    REQUIRE(result.sse_stream != nullptr);
    std::string chunk;
    while (result.sse_stream->next(chunk)) {
      if (chunk == "data: [DONE]\n\n") continue;
      const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
      CHECK_FALSE(frame.contains("usage"));
      CHECK_FALSE(frame.at("choices").empty());
    }
  }
}

TEST_CASE("api_server: stream_options reject non-stream completion and chat") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  ApiServer::DispatchResult completion = h.server.handle_completions(R"({
    "prompt":"hello", "stream":false,
    "stream_options":{"include_usage":true}
  })");
  CHECK(completion.status == 400);
  CHECK(json::parse(completion.body).at("error").at("type") ==
        "BadRequestError");

  ApiServer::DispatchResult chat = h.server.handle_chat_completions(R"({
    "messages":[{"role":"user","content":"hello"}], "stream":false,
    "stream_options":{"continuous_usage_stats":true}
  })");
  CHECK(chat.status == 400);
  CHECK(json::parse(chat.body).at("error").at("type") ==
        "BadRequestError");
}

// Ported from tests/entrypoints/openai/completion/test_completion.py:259
// (test_completion_streaming), with W2's load-bearing arrival assertion: the
// first content frame is observed before the terminal frame instead of being
// replayed from a precomputed vector after generation completes.
TEST_CASE("api_server: live SSE first chunk arrives before generation completes") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":20,"temperature":0.0,"stream":true})";
  const auto started = std::chrono::steady_clock::now();
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  const auto dispatched = std::chrono::steady_clock::now();
  REQUIRE(result.sse_stream != nullptr);
  CHECK(result.sse_chunks.empty());  // no precomputed fake-SSE fallback

  std::string first;
  REQUIRE(result.sse_stream->next(first));
  const auto first_arrival = std::chrono::steady_clock::now();
  REQUIRE(first.rfind("data: ", 0) == 0);
  const json first_payload =
      json::parse(first.substr(6, first.size() - 8));
  CHECK(first_payload.at("choices").at(0).at("finish_reason").is_null());

  size_t chunk_count = 1;
  std::string chunk;
  while (result.sse_stream->next(chunk)) ++chunk_count;
  const auto completed = std::chrono::steady_clock::now();
  CHECK(chunk_count > 2);  // content frames plus [DONE]
  CHECK(dispatched < first_arrival);
  CHECK(first_arrival < completed);
  CHECK(started < completed);
}

TEST_CASE("api_server: chat dispatch → assistant message") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":4,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(body);

  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
}

// ─── #1681 — the REAL published Qwen3.8 chat template through the PRODUCTION
// /v1/chat/completions dispatch ──────────────────────────────────────────────
//
// tests/fixtures/qwen38_chat_template.jinja is the `chat_template` value of
// r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 @
// 36f717a22990e82c54c1d48ee77c491b87825680 (tokenizer_config.json; byte
// identical to that revision's chat_template.jinja). It gates on
//   {%- if enable_thinking is undefined or enable_thinking is true %}
// and `undefined` is a Jinja2 built-in test the vendored minja engine did not
// implement, so every chat request against the whole family answered HTTP 500.
//
// The ENTRY POINT is the point. A unit test on the renderer would have missed
// this the same way every gate in this tree did: the only client that drives
// them is `vllm-cli`, which renders no chat template at all. So the render runs
// inside ApiServer::handle_chat_completions, through the same production
// MakeChatTemplatePromptFn that server_main.cpp installs.
//
// ONE adaptation, named rather than hidden: the synthetic engine behind the
// dispatch carries a 22-token fixture vocabulary that cannot encode Qwen text,
// so the prompt function records the rendered prompt and hands the ENGINE an
// in-vocabulary string. The render, its failure mode and the HTTP status are
// all production.
namespace {

std::string ReadTestFixture(const std::string& name) {
  const std::string path = std::string(VLLM_TEST_FIXTURES_DIR) + "/" + name;
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "missing test fixture: " << path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The production renderer, wrapped so the captured prompt is readable and the
// engine still gets something its 22-token vocabulary can encode.
struct CapturingTemplatePrompt {
  std::shared_ptr<std::string> rendered = std::make_shared<std::string>();
  vllm::entrypoints::openai::ChatPromptFn fn;

  explicit CapturingTemplatePrompt(const std::string& template_str)
      : fn([inner = vllm::entrypoints::MakeChatTemplatePromptFn(
                template_str, /*bos_token=*/"", /*eos_token=*/"<|im_end|>"),
            out = rendered](
               const std::vector<ChatMessage>& messages,
               bool add_generation_prompt,
               const std::vector<
                   vllm::entrypoints::openai::ChatCompletionToolsParam>& tools,
               const nlohmann::ordered_json& chat_template_kwargs) {
          *out = inner(messages, add_generation_prompt, tools,
                       chat_template_kwargs);
          return std::string("hello");
        }) {}
};

}  // namespace

TEST_CASE("api_server: the real Qwen3.8 chat template renders through the "
          "production /v1/chat/completions dispatch (#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);

  const std::string body =
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(body);

  // The defect: 500 with "Unknown type for 'is' operator: undefined".
  INFO("dispatch body: " << r.body);
  CHECK(r.status == 200);
  CHECK(r.body.find("Unknown type for 'is' operator") == std::string::npos);
  // The template really ran: its own system header is in the rendered prompt.
  CHECK(prompt.rendered->find("<|im_start|>user\nhi<|im_end|>") !=
        std::string::npos);
}

// The other half of #1681. `is undefined` is only useful if the variable CAN be
// undefined, and before this row apply_chat_template bound `enable_thinking`
// on every render, so the test could never answer true and the Qwen3.8 family's
// own reasoning default was silently inverted against vLLM and SGLang.
TEST_CASE("api_server: an unsupplied chat template kwarg stays Jinja-undefined "
          "so Qwen3.8 renders its own reasoning default (#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);

  const std::string body =
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(body);

  REQUIRE(r.status == 200);
  // The `{%- if enable_thinking is undefined or enable_thinking is true %}`
  // branch and nothing else emits this sentence.
  CHECK(prompt.rendered->find("Reasoning effort is set to xhigh") !=
        std::string::npos);
}

// The body both competitor arms of #1574 were measured with. Before this row
// ChatCompletionRequest had no such field and the seam had nowhere to put it,
// so the flag was accepted by the JSON parser and dropped on the floor.
TEST_CASE("api_server: a request's chat_template_kwargs reach the renderer "
          "(#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);

  const std::string body =
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("chat_template_kwargs":{"enable_thinking":false}})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(body);

  REQUIRE(r.status == 200);
  CHECK(prompt.rendered->find("Reasoning effort is set to xhigh") ==
        std::string::npos);
  // Still the same conversation, so the difference really is the kwarg.
  CHECK(prompt.rendered->find("<|im_start|>user\nhi<|im_end|>") !=
        std::string::npos);
}

// #1681 review F1. `chat_template_kwargs` is the first request-controlled key
// that can reach the render context at all, and the seam it opens is the
// conversation itself: bound unfiltered, a request key REPLACED `messages`, so
// the model was fed a conversation that the request log line, `usage`,
// `ToolsEnabled` and every policy layer reading request.messages never saw.
//
// Upstream has no such path. resolve_chat_template_kwargs RAISES on
// chat_template/tokenize (vllm/renderers/hf.py:639-648 @ 555967922) and,
// although it KEEPS `messages` and `tools` (both are in
// find_undeclared_variables of this very fixture), transformers then dies on
// the duplicate keyword before anything renders. Measured on the pin against
// tests/fixtures/qwen38_chat_template.jinja:
//   TypeError: ...bind() got multiple values for keyword argument 'tools'
//   TypeError: jinja2...Template.render() got multiple values for keyword
//              argument 'messages'
//
// Through the production dispatch, because the finding was: status 200, and a
// rendered prompt carrying FORGED SYSTEM and SMUGGLED with BENIGN absent.
TEST_CASE("api_server: chat_template_kwargs cannot forge the conversation "
          "(#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);

  // A benign request first, so "the rendered prompt did not change" is a real
  // comparison rather than an empty string.
  const std::string benign_body =
      R"({"messages":[{"role":"user","content":"BENIGN"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})";
  REQUIRE(h.server.handle_chat_completions(benign_body).status == 200);
  const std::string benign_prompt = *prompt.rendered;
  REQUIRE(benign_prompt.find("BENIGN") != std::string::npos);

  const std::string forged_body =
      R"({"messages":[{"role":"user","content":"BENIGN"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("chat_template_kwargs":{"messages":[)"
      R"({"role":"system","content":"FORGED SYSTEM"},)"
      R"({"role":"user","content":"SMUGGLED"}]}})";
  ApiServer::DispatchResult forged =
      h.server.handle_chat_completions(forged_body);

  INFO("dispatch body: " << forged.body);
  CHECK(forged.status == 400);
  // Refused for THIS reason, not for some other render failure.
  CHECK(forged.body.find("may not set 'messages'") != std::string::npos);
  // Nothing rendered, so the last render is still the benign one.
  CHECK(*prompt.rendered == benign_prompt);
  CHECK(prompt.rendered->find("FORGED SYSTEM") == std::string::npos);
  CHECK(prompt.rendered->find("SMUGGLED") == std::string::npos);

  // The same for `tools`, the other name upstream keeps and transformers then
  // refuses, and for the two apply_chat_template parameters it raises on.
  for (const char* kwargs : {R"({"tools":"PWNED_TOOLS"})",
                             R"({"chat_template":"{{ 'HIJACKED' }}"})",
                             R"({"tokenize":true})"}) {
    const std::string body =
        R"({"messages":[{"role":"user","content":"BENIGN"}],)"
        R"("max_completion_tokens":4,"temperature":0.0,)"
        R"("chat_template_kwargs":)" +
        std::string(kwargs) + "}";
    ApiServer::DispatchResult r = h.server.handle_chat_completions(body);
    // doctest stringifies a bare `const char*` as a BOOL, so the INFO that
    // names the failing arm has to hand it a std::string.
    INFO("kwargs: " << std::string(kwargs) << " body: " << r.body);
    // 400, not 500: upstream's ValueError / TypeError reach
    // create_error_response's BadRequestError default
    // (serve/utils/error_response.py:16-21).
    CHECK(r.status == 400);
    CHECK(*prompt.rendered == benign_prompt);
  }

  // add_generation_prompt is the one renderer-owned name upstream neither
  // raises on nor honours: the request's own add_generation_prompt field is on
  // the OVERRIDE side of merge_kwargs and has already replaced the kwarg
  // (chat_completion/protocol.py:530-544, params.py:28-40). So the request
  // renders 200 WITH the assistant header, exactly as if it had not tried.
  const std::string agp_body =
      R"({"messages":[{"role":"user","content":"BENIGN"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("chat_template_kwargs":{"add_generation_prompt":false}})";
  ApiServer::DispatchResult agp = h.server.handle_chat_completions(agp_body);
  INFO("dispatch body: " << agp.body);
  CHECK(agp.status == 200);
  CHECK(prompt.rendered->find("<|im_start|>assistant") != std::string::npos);
}

// The /tokenize chat form renders through the SAME seam, so it has to see the
// same kwargs or the two disagree about what the model is fed
// (serve/tokenize/protocol.py:97,138).
TEST_CASE("api_server: /tokenize chat form honours chat_template_kwargs "
          "(#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);
  h.server.set_tokenizer(&Fixture(), /*max_model_len=*/kMaxModelLen);

  ApiServer::DispatchResult on = h.server.handle_tokenize(
      R"({"messages":[{"role":"user","content":"hi"}]})");
  REQUIRE(on.status == 200);
  const std::string with_default = *prompt.rendered;

  ApiServer::DispatchResult off = h.server.handle_tokenize(
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("chat_template_kwargs":{"enable_thinking":false}})");
  REQUIRE(off.status == 200);

  CHECK(with_default.find("Reasoning effort is set to xhigh") !=
        std::string::npos);
  CHECK(prompt.rendered->find("Reasoning effort is set to xhigh") ==
        std::string::npos);
}

// #1681 second review F1. The first review's filter refused the four names the
// ADAPTER supplies. It could not see the ~30 the ENGINE supplies: minja
// resolves a global, a filter and an is-test through the SAME Context chain
// (third_party/minja/minja.hpp Context::builtins / Context::make), and
// `context->set(key, ...)` writes into the CHILD, so ANY request key shadows
// them. `{"namespace": 1}` therefore broke line 1 of the shipped Qwen3.8
// template -- `{%- set image_count = namespace(value=0) %}` -- and every chat
// request carrying it answered HTTP 500.
//
// Upstream renders 200 for all of them. jinja2 keeps its globals, filters and
// tests OUT of the variable namespace, so `find_undeclared_variables` never
// reports one and `accept_vars` drops the kwarg before anything renders.
// Measured on jinja2 3.1.2 with the pin's own env (hf.py:598-606) over this
// fixture: of minja's 31 builtin names, `find_undeclared_variables |
// hf_base_params` keeps exactly ONE -- `raise_exception`, which transformers
// adds to the environment AFTER that parse and jinja2 supplies nowhere.
TEST_CASE("api_server: a chat_template_kwarg cannot shadow a renderer builtin "
          "(#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);

  const std::string benign_body =
      R"({"messages":[{"role":"user","content":"BENIGN"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})";
  REQUIRE(h.server.handle_chat_completions(benign_body).status == 200);
  const std::string benign_prompt = *prompt.rendered;
  REQUIRE(benign_prompt.find("BENIGN") != std::string::npos);

  // `namespace` is the one this was found on: the template's very first line
  // calls it, so a bound `1` makes the whole render fail. The rest are names
  // this fixture also uses -- as a global, as a filter, and as an is-test --
  // so the case covers all three minja lookup kinds.
  for (const char* kwargs :
       {R"({"namespace":1})", R"({"tojson":1})", R"({"length":1})",
        R"({"trim":1})", R"({"items":1})", R"({"string":1})",
        R"({"safe":1})", R"({"default":1})", R"({"range":1})",
        R"({"join":1})", R"({"upper":1})"}) {
    const std::string body =
        R"({"messages":[{"role":"user","content":"BENIGN"}],)"
        R"("max_completion_tokens":4,"temperature":0.0,)"
        R"("chat_template_kwargs":)" +
        std::string(kwargs) + "}";
    ApiServer::DispatchResult r = h.server.handle_chat_completions(body);
    INFO("kwargs: " << std::string(kwargs) << " status: " << r.status
                    << " body: " << r.body);
    CHECK(r.status == 200);
    // Dropped, exactly as upstream drops it: the prompt is byte-identical to
    // the one the same conversation rendered without the kwarg.
    CHECK(*prompt.rendered == benign_prompt);
  }

  // The single exception, and it is upstream's. `raise_exception` is the one
  // minja builtin jinja2 supplies nowhere, so it IS in this fixture's
  // find_undeclared_variables, upstream keeps it, and the request value
  // shadows transformers' own global. Every call site in this template sits
  // behind an error branch, so a well-formed conversation still renders -- on
  // both engines.
  const std::string re_body =
      R"({"messages":[{"role":"user","content":"BENIGN"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("chat_template_kwargs":{"raise_exception":1}})";
  ApiServer::DispatchResult re = h.server.handle_chat_completions(re_body);
  INFO("dispatch body: " << re.body);
  CHECK(re.status == 200);
}

// #1681 second review F2. A render failure the REQUEST caused was a 500 here
// and a 400 upstream, and our own two endpoints disagreed about the same body:
// /tokenize already answered 400 (api_server.cpp handle_tokenize) while
// /v1/chat/completions fell through to the generic std::exception arm.
//
// Upstream reaches 400 twice over. safe_apply_chat_template wraps ANY
// exception out of apply_chat_template into a ValueError
// (vllm/renderers/hf.py:785-789 @ 555967922), and create_error_response maps
// ValueError/TypeError (error_response.py:48-52) AND jinja2.TemplateError and
// its subclasses (error_response.py:61-65) to BadRequestError.
// The case name carries no comma on purpose: doctest's `-tc` filter splits on
// commas, so a comma makes a case unselectable and it reports `0 cases ran`
// under a green `SUCCESS!`.
TEST_CASE("api_server: a request-caused chat template render failure answers "
          "400 rather than 500 (#1681)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  CapturingTemplatePrompt prompt(ReadTestFixture("qwen38_chat_template.jinja"));
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  ApiServer::kDefaultMaxConcurrentStreams, prompt.fn);
  h.server.set_tokenizer(&Fixture(), /*max_model_len=*/kMaxModelLen);

  // A kwarg upstream KEEPS and hands to the template, whose own
  // raise_exception rejects the value. Client input, so 400.
  const std::string bad_kwarg =
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("chat_template_kwargs":{"reasoning_effort":"nonesuch"}})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(bad_kwarg);
  INFO("dispatch body: " << r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("BadRequestError") != std::string::npos);

  // The same template refusal reached without any kwarg at all: an unknown
  // role. Pre-existing, and the same missing mapping.
  const std::string bad_role =
      R"({"messages":[{"role":"banana","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})";
  ApiServer::DispatchResult role = h.server.handle_chat_completions(bad_role);
  INFO("dispatch body: " << role.body);
  CHECK(role.status == 400);

  // And the two endpoints now agree on the identical body, which is the
  // property that was broken: /tokenize was already 400.
  ApiServer::DispatchResult tok = h.server.handle_tokenize(
      R"({"messages":[{"role":"banana","content":"hi"}]})");
  CHECK(tok.status == role.status);
}

TEST_CASE("api_server: live chat SSE emits role, content, finish, and DONE") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":8,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult result = h.server.handle_chat_completions(body);
  REQUIRE(result.sse_stream != nullptr);

  std::string chunk;
  REQUIRE(result.sse_stream->next(chunk));
  json role = json::parse(chunk.substr(6, chunk.size() - 8));
  CHECK(role.at("choices").at(0).at("delta").at("role") == "assistant");
  CHECK(role.at("choices").at(0).at("finish_reason").is_null());

  size_t content_frames = 0;
  bool finish_seen = false;
  bool done_seen = false;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") {
      done_seen = true;
      continue;
    }
    json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    const json& choice = frame.at("choices").at(0);
    if (!choice.at("finish_reason").is_null()) finish_seen = true;
    if (choice.at("delta").contains("content")) ++content_frames;
  }
  CHECK(content_frames > 0);
  CHECK(finish_seen);
  CHECK(done_seen);
}

// Ported from tests/entrypoints/openai/chat_completion/test_chat.py:
// test_chat_completion_stream_options and
// test_enable_force_include_usage.py:test_chat_with_enable_force_include_usage.
TEST_CASE("api_server: chat continuous usage covers role content and final frame") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body = R"({
    "messages":[{"role":"user","content":"hello"}],
    "max_completion_tokens":5, "temperature":0.0, "stream":true,
    "stream_options":{"include_usage":true,
                      "continuous_usage_stats":true}
  })";
  ApiServer::DispatchResult result = h.server.handle_chat_completions(body);
  REQUIRE(result.sse_stream != nullptr);

  int previous_completion_tokens = 0;
  bool role_seen = false;
  bool final_usage_seen = false;
  std::string chunk;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") continue;
    const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    REQUIRE(frame.contains("usage"));
    const int completion_tokens =
        frame.at("usage").at("completion_tokens");
    CHECK(completion_tokens >= previous_completion_tokens);
    CHECK(frame.at("usage").at("total_tokens").get<int>() ==
          frame.at("usage").at("prompt_tokens").get<int>() +
              completion_tokens);
    if (frame.at("choices").empty()) {
      final_usage_seen = true;
      CHECK(completion_tokens == 5);
    } else {
      const json& delta = frame.at("choices").at(0).at("delta");
      if (delta.contains("role")) {
        role_seen = true;
        CHECK(completion_tokens == 0);
      }
    }
    previous_completion_tokens = completion_tokens;
  }
  CHECK(role_seen);
  CHECK(final_usage_seen);
  CHECK(previous_completion_tokens == 5);
}

TEST_CASE("api_server: chat final-only usage follows the finish choice") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body = R"({
    "messages":[{"role":"user","content":"hello"}],
    "max_completion_tokens":4, "temperature":0.0, "stream":true,
    "stream_options":{"include_usage":true,
                      "continuous_usage_stats":false}
  })";
  ApiServer::DispatchResult result = h.server.handle_chat_completions(body);
  REQUIRE(result.sse_stream != nullptr);
  bool finish_seen = false;
  bool usage_seen_after_finish = false;
  std::string chunk;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") continue;
    const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    if (frame.at("choices").empty()) {
      CHECK(finish_seen);
      REQUIRE(frame.contains("usage"));
      CHECK(frame.at("usage").at("completion_tokens") == 4);
      usage_seen_after_finish = true;
      continue;
    }
    CHECK_FALSE(frame.contains("usage"));
    if (!frame.at("choices").at(0).at("finish_reason").is_null()) {
      finish_seen = true;
    }
  }
  CHECK(finish_seen);
  CHECK(usage_seen_after_finish);
}

TEST_CASE("api_server: force include usage applies without request options") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/true);

  const std::string body = R"({
    "messages":[{"role":"user","content":"hello"}],
    "max_completion_tokens":4, "temperature":0.0, "stream":true
  })";
  ApiServer::DispatchResult result = h.server.handle_chat_completions(body);
  REQUIRE(result.sse_stream != nullptr);
  int final_completion_tokens = -1;
  std::string chunk;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") continue;
    const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    REQUIRE(frame.contains("usage"));
    if (frame.at("choices").empty()) {
      final_completion_tokens =
          frame.at("usage").at("completion_tokens").get<int>();
    }
  }
  CHECK(final_completion_tokens == 4);
}

TEST_CASE("api_server: disconnect before pending usage leaves no live request") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body = R"({
    "prompt":"hello", "max_tokens":4, "temperature":0.0, "stream":true,
    "stream_options":{"include_usage":true}
  })";
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  REQUIRE(result.sse_stream != nullptr);

  bool finish_seen = false;
  std::string chunk;
  while (!finish_seen && result.sse_stream->next(chunk)) {
    REQUIRE(chunk != "data: [DONE]\n\n");
    const json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    REQUIRE_FALSE(frame.at("choices").empty());
    finish_seen =
        !frame.at("choices").at(0).at("finish_reason").is_null();
  }
  REQUIRE(finish_seen);

  // The engine has already retired the terminal output, while the pull stream
  // still owns the unconsumed empty-choice usage frame. A client disconnect
  // at this exact boundary must be idempotent and leave no engine request.
  result.sse_stream->abort();
  result.sse_stream->abort();
  for (int i = 0; i < 500 && h.async_engine.has_unfinished_requests(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK_FALSE(h.async_engine.has_unfinished_requests());
  result.sse_stream.reset();
  CHECK_FALSE(h.async_engine.has_unfinished_requests());
}

TEST_CASE("api_server: disconnect aborts a live SSE request") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":30,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  REQUIRE(result.sse_stream != nullptr);
  std::string first;
  REQUIRE(result.sse_stream->next(first));
  result.sse_stream->abort();
  for (int i = 0; i < 500 && h.async_engine.has_unfinished_requests(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK_FALSE(h.async_engine.has_unfinished_requests());
}

TEST_CASE("api_server: malformed JSON → 400 error shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  ApiServer::DispatchResult r = h.server.handle_completions("{not json");
  CHECK(r.status == 400);
  json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("code") == 400);
}

// A prompt the engine can never serve is the CLIENT's mistake, so it is a 400,
// not the 500 the generic handler would report and not a silent finish reason.
// Upstream raises ValueError from _validate_prompt_len
// (input_processor.py:387-432) and maps it to BadRequestError / HTTP 400 in
// create_error_response (serve/utils/error_response.py:62-65). External PR #227.
TEST_CASE("api_server: prompt past max_model_len → 400 BadRequestError") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  // kMaxModelLen is 32 tokens; "hello world" is 2, so 20 repeats overruns it.
  std::string long_prompt;
  for (int i = 0; i < 20; ++i) long_prompt += "hello world";
  REQUIRE(static_cast<int>(
              Fixture().EncodeWithSpecialTokens(long_prompt).size()) >
          kMaxModelLen);

  json body;
  body["prompt"] = long_prompt;
  body["max_tokens"] = 3;
  body["temperature"] = 0.0;

  ApiServer::DispatchResult r = h.server.handle_completions(body.dump());
  CHECK(r.status == 400);
  json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("code") == 400);
  CHECK(j.at("error").at("message").get<std::string>().find(
            "maximum model length of 32") != std::string::npos);
  // Refused at admission: nothing was handed to the engine to wedge on.
  CHECK_FALSE(h.async_engine.has_unfinished_requests());

  // Same mapping on the chat route.
  json chat_body;
  chat_body["messages"] =
      json::array({{{"role", "user"}, {"content", long_prompt}}});
  chat_body["max_tokens"] = 3;
  chat_body["temperature"] = 0.0;
  ApiServer::DispatchResult cr =
      h.server.handle_chat_completions(chat_body.dump());
  CHECK(cr.status == 400);
  CHECK(json::parse(cr.body).at("error").at("type") == "BadRequestError");
}

TEST_CASE("api_server: unknown model → 404 error shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"model":"nope","prompt":"hello","max_tokens":3,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);
  CHECK(r.status == 404);
  json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "NotFoundError");
}

TEST_CASE("api_server: /v1/models, /health, /version dispatch") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  json models = json::parse(h.server.handle_models().body);
  CHECK(models.at("object") == "list");
  CHECK(models.at("data").at(0).at("id") == "test-model");
  CHECK(models.at("data").at(0).at("object") == "model");

  ApiServer::DispatchResult health = h.server.handle_health();
  CHECK(health.status == 200);

  json ver = json::parse(h.server.handle_version().body);
  CHECK(ver.at("version") == "9.9.9");
}

// ─── 1b. C8 utility + observability endpoints (SERVE-UTILITY-ENDPOINTS /
// SERVE-METRICS). Additive/opt-in: attaching a tokenizer/metrics/reset backing
// enables the routes; the schemas match vLLM 0.26. ──────────────────────────
TEST_CASE("api_server: /tokenize + /detokenize round-trip and schema") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_tokenizer(&Fixture(), /*max_model_len=*/kMaxModelLen);

  // tokenize (TokenizeCompletionRequest → TokenizeResponse). "hello world" is
  // within the tiny BPE fixture's vocab.
  ApiServer::DispatchResult tok = h.server.handle_tokenize(
      R"({"prompt":"hello world","add_special_tokens":false,"return_token_strs":true})");
  REQUIRE(tok.status == 200);
  json tj = json::parse(tok.body);
  CHECK(tj.contains("count"));
  CHECK(tj.contains("max_model_len"));
  CHECK(tj.at("max_model_len") == kMaxModelLen);
  REQUIRE(tj.at("tokens").is_array());
  CHECK(tj.at("count") == tj.at("tokens").size());
  CHECK(tj.at("token_strs").is_array());
  CHECK(tj.at("token_strs").size() == tj.at("tokens").size());

  // detokenize round-trips the token ids back through DetokenizeResponse.
  std::vector<int> ids = tj.at("tokens").get<std::vector<int>>();
  json detok_req;
  detok_req["tokens"] = ids;
  ApiServer::DispatchResult detok =
      h.server.handle_detokenize(detok_req.dump());
  REQUIRE(detok.status == 200);
  json dj = json::parse(detok.body);
  REQUIRE(dj.contains("prompt"));
  CHECK(dj.at("prompt").is_string());

  // return_token_strs omitted → null (schema default).
  json plain = json::parse(
      h.server.handle_tokenize(R"({"prompt":"hello"})").body);
  CHECK(plain.at("token_strs").is_null());

  // Raw-`prompt` form byte-identical to the pre-chat-form behavior: the exact
  // token ids are Fixture().EncodeWithSpecialTokens("hello") (add_special_tokens
  // default True for the completion form; protocol.py:28).
  {
    json raw = json::parse(
        h.server.handle_tokenize(R"({"prompt":"hello world"})").body);
    std::vector<int> expect = Fixture().EncodeWithSpecialTokens("hello world");
    CHECK(raw.at("tokens").get<std::vector<int>>() == expect);
    CHECK(raw.at("count") == expect.size());
  }

  // Errors: missing prompt → 400; detokenize missing tokens → 400.
  CHECK(h.server.handle_tokenize(R"({})").status == 400);
  CHECK(h.server.handle_detokenize(R"({})").status == 400);
}

// Chat-form /tokenize (serve/tokenize/protocol.py:50 TokenizeChatRequest):
// messages are rendered through the SAME chat template create_chat_completion
// uses (chat_.prompt_fn()) then tokenized, returning {count, max_model_len,
// tokens} identically to vLLM 0.26 serving_tokenization. The harness wires
// InVocabChatPrompt as that seam, so the expected ids are computed by rendering
// the messages through it and Encode()-ing the result (add_special_tokens
// default False for the chat form; protocol.py:78).
TEST_CASE("api_server: /tokenize chat form renders template + tokenizes") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_tokenizer(&Fixture(), /*max_model_len=*/kMaxModelLen);

  // A fixed messages array. InVocabChatPrompt concatenates the contents, so the
  // rendered prompt is "helloworld"; the expected ids come from tokenizing that
  // exact string with the chat-form default (add_special_tokens=False → Encode).
  const char* kChatBody =
      R"({"messages":[{"role":"user","content":"hello"},)"
      R"({"role":"assistant","content":"world"}]})";
  const std::vector<ChatMessage> kMessages = {
      ChatMessage{"user", std::string("hello")},
      ChatMessage{"assistant", std::string("world")}};
  const std::string rendered = InVocabChatPrompt(
      kMessages, /*add_generation_prompt=*/true, {},
      nlohmann::ordered_json::object());
  const std::vector<int> expect = Fixture().Encode(rendered);

  ApiServer::DispatchResult tok = h.server.handle_tokenize(kChatBody);
  REQUIRE(tok.status == 200);
  json tj = json::parse(tok.body);
  // Response shape matches vLLM (count / max_model_len / tokens / token_strs).
  CHECK(tj.contains("count"));
  CHECK(tj.at("max_model_len") == kMaxModelLen);
  REQUIRE(tj.at("tokens").is_array());
  // Chat-form tokens are EXACTLY the template-render → tokenize result.
  CHECK(tj.at("tokens").get<std::vector<int>>() == expect);
  CHECK(tj.at("count") == expect.size());
  CHECK(tj.at("token_strs").is_null());

  // return_token_strs is honored on the chat form too.
  json body_strs = json::parse(kChatBody);
  body_strs["return_token_strs"] = true;
  json ts = json::parse(h.server.handle_tokenize(body_strs.dump()).body);
  CHECK(ts.at("token_strs").is_array());
  CHECK(ts.at("token_strs").size() == expect.size());

  // The chat form agrees with the chat-completions path's OWN prompt tokens:
  // both render through chat_.prompt_fn() and tokenize the same string, so the
  // chat-form token ids equal what create_chat_completion would feed the engine
  // (add_generation_prompt is create_chat_completion's fixed True default).

  // add_generation_prompt=false path still tokenizes (InVocabChatPrompt ignores
  // the flag, but the request is accepted and produces the same ids here).
  {
    json b = json::parse(kChatBody);
    b["add_generation_prompt"] = false;
    ApiServer::DispatchResult r = h.server.handle_tokenize(b.dump());
    REQUIRE(r.status == 200);
    CHECK(json::parse(r.body).at("tokens").get<std::vector<int>>() == expect);
  }

  // check_generation_prompt: continue_final_message + add_generation_prompt both
  // true → 400 (protocol.py:120-128).
  {
    json b = json::parse(kChatBody);
    b["continue_final_message"] = true;
    b["add_generation_prompt"] = true;
    CHECK(h.server.handle_tokenize(b.dump()).status == 400);
  }

  // Empty messages array is a VALID chat request (renders to the empty prompt),
  // not a 400 — the pre-chat-form behavior rejected it; now it tokenizes.
  {
    ApiServer::DispatchResult r =
        h.server.handle_tokenize(R"({"messages":[]})");
    REQUIRE(r.status == 200);
    json rj = json::parse(r.body);
    CHECK(rj.at("tokens").get<std::vector<int>>() == Fixture().Encode(""));
  }
}

TEST_CASE("api_server: /metrics exposition through the server handler") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  vllm::v1::metrics::PrometheusStatLogger logger("test-model", kMaxModelLen);
  logger.SetCacheConfigInfo(1024, 1.0);
  h.server.set_metrics_logger(&logger);

  ApiServer::DispatchResult m = h.server.handle_metrics();
  CHECK(m.status == 200);
  CHECK(m.content_type == "text/plain; version=0.0.4; charset=utf-8");
  CHECK(m.body.find("vllm:num_requests_running") != std::string::npos);
  CHECK(m.body.find("vllm:prompt_tokens_total") != std::string::npos);
  CHECK(m.body.find("vllm:time_to_first_token_seconds_bucket") !=
        std::string::npos);
}

// ─── SERVE-METRICS residual: spec decode (#2770) ─────────────────────────────
// THE REACHABILITY GATE. The production entry point is GET /metrics on the
// shipped server; the value has to arrive there from a real engine step, not
// from a hand-fed SchedulerStats. This case drives a completion through
// ApiServer::handle_completions over a SPECULATING engine and then reads the
// counters back through ApiServer::handle_metrics, comparing them against what
// the drafter itself recorded.
//
// It reds if ANY of the three new production call sites is removed: the
// make_spec_decoding_stats call in Scheduler::update_from_output, the
// make_stats() republish, or the Record() fold in PrometheusStatLogger.
TEST_CASE(
    "api_server: /metrics exports the spec_decode families a speculating "
    "engine actually produced") {
  const HfConfig c = MakeConfig();
  constexpr int kK = 3;
  SpecServerHarness h(c, Fixture(), kK);

  // Long enough to run many verify steps, so acceptance is non-trivial.
  const std::string body =
      R"({"model":"test-model","prompt":"hello","max_tokens":24,)"
      R"("temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);
  REQUIRE(r.status == 200);
  CHECK(json::parse(r.body).at("usage").at("completion_tokens") == 24);

  // The engine really speculated. Without this the assertions below could all
  // hold at zero, which is the "well-formed catalog that never moved" failure
  // this row exists to end.
  REQUIRE(h.runner.drafts_verified.load() > 0);
  REQUIRE(h.runner.draft_tokens_accepted.load() > 0);
  CHECK(h.runner.draft_tokens_accepted.load() <
        h.runner.draft_tokens_verified.load());

  const std::string kLabels = R"({model_name="test-model",engine="0"})";
  // The last step's Record runs on the output-handler thread after the
  // collector was woken, so the scrape can legitimately race the final fold.
  // Poll to the drafter's own totals rather than sleeping a fixed interval.
  std::string metrics;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  for (;;) {
    ApiServer::DispatchResult m = h.server.handle_metrics();
    REQUIRE(m.status == 200);
    metrics = m.body;
    if (ScrapeSeries(metrics,
                     "vllm:spec_decode_num_drafts_total" + kLabels) ==
            static_cast<double>(h.runner.drafts_verified.load()) &&
        ScrapeSeries(metrics,
                     "vllm:spec_decode_num_draft_tokens_total" + kLabels) ==
            static_cast<double>(h.runner.draft_tokens_verified.load()) &&
        ScrapeSeries(metrics,
                     "vllm:spec_decode_num_accepted_tokens_total" + kLabels) ==
            static_cast<double>(h.runner.draft_tokens_accepted.load())) {
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // The four family names, in the spelling a scraper sees.
  CHECK(metrics.find("# TYPE vllm:spec_decode_num_drafts counter") !=
        std::string::npos);
  CHECK(metrics.find("# TYPE vllm:spec_decode_num_draft_tokens counter") !=
        std::string::npos);
  CHECK(metrics.find("# TYPE vllm:spec_decode_num_accepted_tokens counter") !=
        std::string::npos);
  CHECK(metrics.find(
            "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter") !=
        std::string::npos);
  CHECK(metrics.find("# HELP vllm:spec_decode_num_accepted_tokens Number of "
                     "accepted tokens.") != std::string::npos);

  // The VALUES. Equality, not `> 0`: a fold that dropped the per-request
  // aggregation or double-counted a step still passes a positivity check.
  CHECK(ScrapeSeries(metrics, "vllm:spec_decode_num_drafts_total" + kLabels) ==
        static_cast<double>(h.runner.drafts_verified.load()));
  CHECK(ScrapeSeries(metrics,
                     "vllm:spec_decode_num_draft_tokens_total" + kLabels) ==
        static_cast<double>(h.runner.draft_tokens_verified.load()));
  CHECK(ScrapeSeries(metrics,
                     "vllm:spec_decode_num_accepted_tokens_total" + kLabels) ==
        static_cast<double>(h.runner.draft_tokens_accepted.load()));

  // Per-position: k series, labelled 0..k-1, each carrying that depth's
  // acceptance count. This is the vector an acceptance-driven depth policy
  // reads, and the aggregate cannot substitute for it.
  for (int pos = 0; pos < kK; ++pos) {
    const std::string series =
        "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"test-"
        "model\",engine=\"0\",position=\"" + std::to_string(pos) + "\"}";
    CHECK(ScrapeSeries(metrics, series) ==
          static_cast<double>(
              h.runner.accepted_per_pos[static_cast<std::size_t>(pos)].load()));
  }
  // ...and no series for a depth the config never allowed.
  CHECK(metrics.find("position=\"" + std::to_string(kK) + "\"") ==
        std::string::npos);

  // Acceptance is derivable from the scrape alone, which is the whole point:
  // rate(accepted) / rate(draft_tokens) is vLLM's documented PromQL
  // (spec_decode/metrics.py:180-184).
  const double accepted =
      ScrapeSeries(metrics, "vllm:spec_decode_num_accepted_tokens_total" + kLabels);
  const double drafted =
      ScrapeSeries(metrics, "vllm:spec_decode_num_draft_tokens_total" + kLabels);
  REQUIRE(drafted > 0.0);
  CHECK(accepted / drafted > 0.0);
  CHECK(accepted / drafted < 1.0);
}

// The upstream GATE, mirrored: with no speculative config the families are not
// registered at all (spec_decode/metrics.py:211-213), so /metrics is silent
// about speculation rather than reporting four counters frozen at zero. A
// dashboard can then tell "this engine does not speculate" from "this engine
// speculates and accepts nothing", which are opposite findings.
TEST_CASE("api_server: /metrics omits the spec_decode families with no "
          "speculative config") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  // Constructed exactly as server_main.cpp constructs it for a non-speculative
  // engine: the default num_speculative_tokens of 0.
  vllm::v1::metrics::PrometheusStatLogger logger("test-model", kMaxModelLen);
  h.server.set_metrics_logger(&logger);

  ApiServer::DispatchResult m = h.server.handle_metrics();
  REQUIRE(m.status == 200);
  CHECK(m.body.find("vllm:num_requests_running") != std::string::npos);
  CHECK(m.body.find("vllm:spec_decode_num_drafts") == std::string::npos);
  CHECK(m.body.find("vllm:spec_decode_num_draft_tokens") == std::string::npos);
  CHECK(m.body.find("vllm:spec_decode_num_accepted_tokens") ==
        std::string::npos);
  CHECK(m.body.find("vllm:spec_decode_num_accepted_tokens_per_pos") ==
        std::string::npos);
}

TEST_CASE("api_server: /ping mirrors /health, /server_info shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  CHECK(h.server.handle_ping().status == 200);
  json info = json::parse(h.server.handle_server_info().body);
  CHECK(info.contains("vllm_config"));
  CHECK(info.contains("vllm_env"));
  CHECK(info.contains("system_env"));
}

TEST_CASE("api_server: /reset_prefix_cache → {\"success\": bool}") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  bool called = false;
  h.server.set_reset_prefix_cache(
      [&called](bool /*running*/, bool /*external*/) {
        called = true;
        return true;
      });
  json r = json::parse(h.server.handle_reset_prefix_cache(false, false).body);
  CHECK(called);
  CHECK(r.at("success") == true);
}

// GET /tokenizer_info (serve/tokenize/api_router.py:95-108 attach_router →
// serving.py:154-160 get_tokenizer_info → TokenizerInfoResponse). Surfaces the
// tokenizer_config.json-equivalent fields our BPE tokenizer can genuinely back;
// vLLM-only fields (raw chat_template string, HF init_kwargs, added-token
// normalized/single_word) are omitted by design. Ported from
// tests/entrypoints/openai/test_tokenization.py::test_get_tokenizer_info_*.
TEST_CASE("api_server: /tokenizer_info surfaces backed tokenizer metadata") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_tokenizer(&Fixture(), /*max_model_len=*/kMaxModelLen);

  ApiServer::DispatchResult info = h.server.handle_tokenizer_info();
  REQUIRE(info.status == 200);
  CHECK(info.content_type == "application/json");
  json j = json::parse(info.body);

  // tokenizer_class is required by TokenizerInfoResponse; the fixture is a
  // byte-level BPE, so the genuine family name is reported.
  CHECK(j.at("tokenizer_class") == "ByteLevelBPETokenizer");
  CHECK(j.at("model_max_length") == kMaxModelLen);
  CHECK(j.at("vocab_size") == Fixture().VocabSize());

  // added_tokens_decoder mirrors HF tokenizer_config.json: id → {content,
  // special, lstrip, rstrip}. The fixture declares three added tokens (ids
  // 19/20/21), two of them special.
  REQUIRE(j.contains("added_tokens_decoder"));
  const json& added = j.at("added_tokens_decoder");
  CHECK(added.size() == Fixture().AddedTokens().size());
  REQUIRE(added.contains("19"));
  CHECK(added.at("19").at("content") == "<|end|>");
  CHECK(added.at("19").at("special") == true);
  CHECK(added.at("20").at("content") == "<tool>");
  CHECK(added.at("20").at("special") == false);
  // Named gaps: the raw chat_template string is NOT surfaced (chat templates
  // live in the ChatPromptFn seam), nor the HF init_kwargs, nor the added-token
  // normalized/single_word flags.
  CHECK_FALSE(j.contains("chat_template"));
  CHECK_FALSE(added.at("19").contains("normalized"));

  // Without a tokenizer attached the handler is a 500 (the route itself is
  // only registered when a tokenizer is attached AND the flag is enabled).
  ServerHarness bare(c, w, Fixture());
  CHECK(bare.server.handle_tokenizer_info().status == 500);
}

// POST /abort_requests (serve/dev/rlhf/api_router.py:94-138): the abort
// callback aborts the listed request ids; the response is
// {"status":"aborted","aborted":<count>}. Malformed JSON → 400 {"detail":...}.
TEST_CASE("api_server: /abort_requests shape + callback wiring") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  std::vector<std::string> seen_ids;
  bool empty_seen = false;
  h.server.set_abort_requests(
      [&](const std::vector<std::string>& ids) -> int {
        if (ids.empty()) {
          empty_seen = true;
          return 7;  // "abort all" → callback-reported count
        }
        seen_ids = ids;
        return static_cast<int>(ids.size());
      });

  // Explicit ids → passed through verbatim; count == number of ids.
  json r = json::parse(
      h.server.handle_abort_requests(R"({"request_ids":["a","b","c"]})").body);
  CHECK(r.at("status") == "aborted");
  CHECK(r.at("aborted") == 3);
  CHECK(seen_ids == std::vector<std::string>{"a", "b", "c"});

  // Missing request_ids → empty list → "abort all" branch.
  json all = json::parse(h.server.handle_abort_requests(R"({})").body);
  CHECK(empty_seen);
  CHECK(all.at("status") == "aborted");
  CHECK(all.at("aborted") == 7);

  // Malformed JSON → 400 with vLLM's HTTPException detail shape.
  ApiServer::DispatchResult bad = h.server.handle_abort_requests("{not json");
  CHECK(bad.status == 400);
  CHECK(json::parse(bad.body).at("detail") == "Invalid JSON format");
}

// /abort_requests actually tears down an in-flight AsyncLLM request when the
// callback is wired to the engine abort path (async_llm.h:115 abort). Ported
// from tests/entrypoints/serve/dev/rlhf's abort behavior + test_async_llm abort.
TEST_CASE("api_server: /abort_requests aborts an in-flight engine request") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_abort_requests(
      [&](const std::vector<std::string>& ids) -> int {
        h.async_engine.abort(ids);
        return static_cast<int>(ids.size());
      });

  // Submit a long in-flight request directly to the engine under a known id so
  // the abort target is deterministic (the serving layer mints its own
  // "cmpl-<n>-0" engine ids).
  vllm::SamplingParams params;
  params.max_tokens = 30;
  params.temperature = 0.0;
  vllm::v1::AsyncRequest req =
      h.async_engine.add_request("abort-me", "hello", params);
  REQUIRE(h.async_engine.has_unfinished_requests());

  json r = json::parse(
      h.server.handle_abort_requests(R"({"request_ids":["abort-me"]})").body);
  CHECK(r.at("status") == "aborted");
  CHECK(r.at("aborted") == 1);

  for (int i = 0; i < 500 && h.async_engine.has_unfinished_requests(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK_FALSE(h.async_engine.has_unfinished_requests());
}

// ─── 2. Socket smoke test (real HTTP over an ephemeral port) ─────────────────
TEST_CASE("api_server: socket smoke — real HTTP requests over an ephemeral port") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  // Wait until the accept loop is up.
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(15, 0);

  // /health
  {
    auto res = client.Get("/health");
    REQUIRE(res);
    CHECK(res->status == 200);
  }
  // /v1/models
  {
    auto res = client.Get("/v1/models");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("data").at(0).at("id") == "test-model");
  }
  // Non-streaming /v1/completions
  {
    auto res = client.Post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":5,"temperature":0.0})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("object") == "text_completion");
    CHECK_FALSE(j.at("choices").at(0).at("text").get<std::string>().empty());
  }
  // Streaming /v1/completions — the body is the concatenated SSE frames.
  {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello","max_tokens":5,"temperature":0.0,"stream":true})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.rfind("data: ", 0) == 0);
    CHECK(res->body.find("data: [DONE]\n\n") != std::string::npos);
  }
  // /v1/chat/completions
  {
    auto res = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":4,"temperature":0.0})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("object") == "chat.completion");
    CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
  }

  server_thread.join();  // stops the server, then joins
}

// Route-registration gate over a real socket: /tokenizer_info is ABSENT (404)
// unless a tokenizer is attached AND the info flag is enabled (mirrors vLLM
// gating it behind enable_tokenizer_info_endpoint), and /abort_requests is
// ABSENT until the abort callback is attached. This is the RED-first evidence
// that the additive routes are opt-in and never perturb the base routing.
TEST_CASE("api_server: /tokenizer_info + /abort_requests are opt-in routes") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  SUBCASE("backings unattached → routes absent (404)") {
    ServerHarness h(c, w, Fixture());
    // Attach the tokenizer but leave the info flag OFF: /tokenize is enabled,
    // /tokenizer_info stays absent.
    h.server.set_tokenizer(&Fixture(), kMaxModelLen);
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());

    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(5, 0);
    auto info = client.Get("/tokenizer_info");
    REQUIRE(info);
    CHECK(info->status == 404);  // flag off → route not registered
    auto abort = client.Post("/abort_requests", R"({"request_ids":[]})",
                             "application/json");
    REQUIRE(abort);
    CHECK(abort->status == 404);  // no callback → route not registered

    server_thread.join();  // stops the server, then joins
  }

  SUBCASE("backings attached → routes serve (200)") {
    ServerHarness h(c, w, Fixture());
    h.server.set_tokenizer(&Fixture(), kMaxModelLen);
    h.server.set_tokenizer_info_enabled(true);
    int aborted_calls = 0;
    h.server.set_abort_requests(
        [&](const std::vector<std::string>& ids) -> int {
          ++aborted_calls;
          return static_cast<int>(ids.size());
        });
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());

    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(5, 0);
    auto info = client.Get("/tokenizer_info");
    REQUIRE(info);
    CHECK(info->status == 200);
    CHECK(json::parse(info->body).at("tokenizer_class") ==
          "ByteLevelBPETokenizer");
    auto abort = client.Post("/abort_requests", R"({"request_ids":["x","y"]})",
                             "application/json");
    REQUIRE(abort);
    CHECK(abort->status == 200);
    CHECK(json::parse(abort->body).at("aborted") == 2);
    CHECK(aborted_calls == 1);

    server_thread.join();  // stops the server, then joins
  }
}

// ─── 1c. PRODUCTION WIRING (CLAIM-C8-SERVE-PROD-WIRING) ──────────────────────
// ConfigureUtilityEndpoints is the SINGLE seam examples/server/main.cpp uses to
// light the C8 endpoints from the LIVE engine + tokenizer. This drives that exact
// seam over the synthetic engine and asserts the production server now serves each
// newly-wired route under vLLM 0.26's per-endpoint default gating. RED-first:
// without the seam the C8 routes 404; the seam turns on exactly what vLLM turns on
// by default, and gates /tokenizer_info + /abort_requests behind their flags.
TEST_CASE("api_server: ConfigureUtilityEndpoints wires the production C8 surface") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  // Run `body` against a client bound to a freshly-served harness, then stop.
  auto with_server = [](ServerHarness& h, auto&& body) {
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(5, 0);
    body(client);
    server_thread.join();  // stops the server, then joins
  };

  // RED: a default production server WITHOUT the wiring seam 404s every C8 route,
  // while the core routes are unaffected — the inertness baseline.
  SUBCASE("without the wiring seam every C8 route is 404 (RED); core routes 200") {
    ServerHarness h(c, w, Fixture());
    with_server(h, [](httplib::Client& client) {
      auto tok =
          client.Post("/tokenize", R"({"prompt":"hi"})", "application/json");
      REQUIRE(tok);
      CHECK(tok->status == 404);
      auto detok =
          client.Post("/detokenize", R"({"tokens":[0]})", "application/json");
      REQUIRE(detok);
      CHECK(detok->status == 404);
      auto info = client.Get("/tokenizer_info");
      REQUIRE(info);
      CHECK(info->status == 404);
      auto abort = client.Post("/abort_requests", R"({"request_ids":["x"]})",
                               "application/json");
      REQUIRE(abort);
      CHECK(abort->status == 404);
      // Core routes remain served + unchanged.
      auto health = client.Get("/health");
      REQUIRE(health);
      CHECK(health->status == 200);
      auto models = client.Get("/v1/models");
      REQUIRE(models);
      CHECK(models->status == 200);
      CHECK(json::parse(models->body).at("data").at(0).at("id") == "test-model");
    });
  }

  // GREEN, production defaults (both flags OFF): /tokenize + /detokenize serve
  // (on by default when a tokenizer exists), while /tokenizer_info and
  // /abort_requests stay 404 — exactly vLLM's default gating.
  SUBCASE("production defaults: tokenize/detokenize on, info+abort gated 404") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    with_server(h, [](httplib::Client& client) {
      auto tok = client.Post("/tokenize", R"({"prompt":"hello world"})",
                             "application/json");
      REQUIRE(tok);
      CHECK(tok->status == 200);
      json tj = json::parse(tok->body);
      CHECK(tj.at("max_model_len") == kMaxModelLen);
      std::vector<int> ids = tj.at("tokens").get<std::vector<int>>();
      json detok_req;
      detok_req["tokens"] = ids;
      auto detok =
          client.Post("/detokenize", detok_req.dump(), "application/json");
      REQUIRE(detok);
      CHECK(detok->status == 200);
      CHECK(json::parse(detok->body).at("prompt").is_string());
      // Flag/dev-mode-gated routes stay off.
      auto info = client.Get("/tokenizer_info");
      REQUIRE(info);
      CHECK(info->status == 404);
      auto abort = client.Post("/abort_requests", R"({"request_ids":["x"]})",
                               "application/json");
      REQUIRE(abort);
      CHECK(abort->status == 404);
    });
  }

  // GREEN, both flags ON: /tokenizer_info + /abort_requests are now served over a
  // real socket by the production seam.
  SUBCASE("flags on: tokenizer_info + abort_requests serve over the socket") {
    ServerHarness h(c, w, Fixture());
    UtilityEndpointOptions opts;
    opts.enable_tokenizer_info_endpoint = true;
    opts.enable_server_dev_mode = true;
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              opts);
    with_server(h, [](httplib::Client& client) {
      auto info = client.Get("/tokenizer_info");
      REQUIRE(info);
      CHECK(info->status == 200);
      CHECK(json::parse(info->body).at("tokenizer_class") ==
            "ByteLevelBPETokenizer");
      auto abort = client.Post("/abort_requests", R"({"request_ids":["nope"]})",
                               "application/json");
      REQUIRE(abort);
      CHECK(abort->status == 200);
      CHECK(json::parse(abort->body).at("status") == "aborted");
    });
  }

  // The production abort callback installed by ConfigureUtilityEndpoints reaches
  // the LIVE AsyncLLM: an explicit-id abort tears the in-flight request down and
  // reports the exact drop in unfinished requests (before − after). Driven
  // in-process (handle_abort_requests, no socket) so the count is deterministic.
  SUBCASE("dev-mode abort callback tears down the live request (exact count)") {
    ServerHarness h(c, w, Fixture());
    UtilityEndpointOptions opts;
    opts.enable_server_dev_mode = true;
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              opts);

    vllm::SamplingParams params;
    params.max_tokens = 30;
    params.temperature = 0.0;
    h.async_engine.add_request("abort-me", "hello", params);
    REQUIRE(h.async_engine.has_unfinished_requests());

    json r = json::parse(
        h.server.handle_abort_requests(R"({"request_ids":["abort-me"]})").body);
    CHECK(r.at("status") == "aborted");
    CHECK(r.at("aborted") == 1);  // production before/after delta
    for (int i = 0; i < 500 && h.async_engine.has_unfinished_requests(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK_FALSE(h.async_engine.has_unfinished_requests());

    // Empty request_ids → the abort-ALL residual → 0 aborted (AsyncLLM exposes
    // no active-request-id accessor; explicit-id abort is the supported path).
    json all = json::parse(
        h.server.handle_abort_requests(R"({"request_ids":[]})").body);
    CHECK(all.at("status") == "aborted");
    CHECK(all.at("aborted") == 0);
  }
}

// ─── SERVE-REQUEST-LENGTH-GUARD (#1541) ─────────────────────────────────────
// #1365's algorithmic fix removed the QUADRATIC term from the BPE merge loop
// (`67823aee2`: 64 KB in one pretoken, 23,620.695 ms -> 7.797 ms, exponent ~2
// -> ~1). It added no BOUND, and a linear cost against httplib's 100 MB
// CPPHTTPLIB_PAYLOAD_MAX_LENGTH is a smaller problem than a quadratic one
// rather than the absence of one. There is no authentication anywhere in
// src/vllm/entrypoints/, and /tokenize needs neither an engine nor a model, so
// that cost is paid on an HTTP worker for an unauthenticated caller.
//
// The guard REFUSES and never truncates: a silently shortened prompt returns
// model output for text the caller did not send. It sits at the REQUEST
// BOUNDARY, before any tokenization -- NOT in
// vllm::v1::InputProcessor::ValidatePromptLen, which needs the token count the
// expensive step produces and therefore cannot run before it.
//
// THE BOUND IS DERIVED, not chosen. The token texts of an encode concatenate
// back to the input, so a prompt of B bytes costs at least B / max_token_bytes
// tokens, where max_token_bytes is the longest STORED token text in the loaded
// vocabulary. A prompt longer than `max_model_len * max_token_bytes` therefore
// exceeds max_model_len tokens and is already unservable, so the bound refuses
// nothing ValidatePromptLen would have accepted. For this fixture
// max_token_bytes is 9 -- the added token "<|end|>of"; the longest vocab entry
// "Ġworld" is 7, since 'Ġ' is two UTF-8 bytes -- and kMaxModelLen is 32.
//
// Every case runs OVER A REAL SOCKET through the production wiring seam
// ConfigureUtilityEndpoints, because a guard that only fires when a test calls
// the parse function is the unpassed-parameter shape (.agents/reachability.md).
constexpr size_t kMaxPromptBytes = static_cast<size_t>(kMaxModelLen) * 9;

// Read the OpenAI ErrorResponse fields WITHOUT throwing when the body is not
// one: a subcase whose request was (wrongly) served answers 200 with a normal
// payload, and an .at() there would abort the whole TEST_CASE and hide every
// later subcase's own verdict.
std::string ErrType(const json& j) {
  return j.value("error", json::object()).value("type", std::string());
}
int ErrCode(const json& j) {
  return j.value("error", json::object()).value("code", 0);
}
std::string ErrMsg(const json& j) {
  return j.value("error", json::object()).value("message", std::string());
}

TEST_CASE("api_server: an oversized prompt is REFUSED at the request boundary") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  auto with_server = [](ServerHarness& h, auto&& body) {
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(30, 0);
    body(client);
    server_thread.join();  // stops the server, then joins
  };

  // The unauthenticated, engine-free surface: /tokenize needs no model and no
  // scheduler, so it is the one an anonymous caller reaches most cheaply.
  SUBCASE("/tokenize: one byte over the bound is 400 and NOTHING is tokenized") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    with_server(h, [](httplib::Client& client) {
      json req;
      req["prompt"] = std::string(kMaxPromptBytes + 1, 'h');
      auto r = client.Post("/tokenize", req.dump(), "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body);
      CHECK(ErrType(j) == "BadRequestError");
      CHECK(ErrCode(j) == 400);
      const std::string msg = ErrMsg(j);
      // The refusal NAMES the limit and what arrived, which is what makes it
      // actionable rather than a bare 400.
      CHECK(msg.find("289") != std::string::npos);
      CHECK(msg.find("288") != std::string::npos);
      CHECK(msg.find("bytes") != std::string::npos);
      // TRUNCATION DETECTOR. A guard that shortened the prompt to the bound
      // instead of refusing it would answer 200 carrying a token list for text
      // the caller never sent. There is no token list here at all.
      CHECK_FALSE(j.contains("tokens"));
      CHECK_FALSE(j.contains("count"));
    });
  }

  // The bound is INCLUSIVE and it does not truncate below itself: a prompt AT
  // the bound is tokenized whole, so the guard cannot be satisfied by a
  // shortening step hidden anywhere on the path.
  SUBCASE("/tokenize: a prompt AT the bound is tokenized in FULL") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    const std::string at_bound(kMaxPromptBytes, 'h');
    const size_t expected = Fixture().EncodeWithSpecialTokens(at_bound).size();
    REQUIRE(expected > 0);
    with_server(h, [&](httplib::Client& client) {
      json req;
      req["prompt"] = at_bound;
      auto r = client.Post("/tokenize", req.dump(), "application/json");
      REQUIRE(r);
      CHECK(r->status == 200);
      json j = json::parse(r->body);
      REQUIRE(j.contains("count"));
      CHECK(j.at("count").get<size_t>() == expected);
      CHECK(j.at("tokens").size() == expected);
    });
  }

  // /v1/completions already answers 400 on an over-long prompt -- but only
  // AFTER the encode, from ValidatePromptLen, and its message names the TOKEN
  // limit. The byte bound has to fire FIRST, and say so.
  SUBCASE("/v1/completions: refused on BYTES, before the encode") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    with_server(h, [](httplib::Client& client) {
      json req;
      req["model"] = "test-model";
      req["prompt"] = std::string(kMaxPromptBytes + 1, 'h');
      req["max_tokens"] = 1;
      req["temperature"] = 0.0;
      auto r = client.Post("/v1/completions", req.dump(), "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body);
      CHECK(ErrType(j) == "BadRequestError");
      const std::string msg = ErrMsg(j);
      CHECK(msg.find("289") != std::string::npos);
      CHECK(msg.find("288") != std::string::npos);
      CHECK(msg.find("bytes") != std::string::npos);
      // It is the BYTE refusal, not the post-encode token one.
      CHECK(msg.find("maximum model length") == std::string::npos);
    });
  }

  SUBCASE("/v1/chat/completions: refused on the summed message BYTES") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    with_server(h, [](httplib::Client& client) {
      // Split across two turns, so the sum is what is measured rather than any
      // single field.
      json req;
      req["model"] = "test-model";
      req["messages"] = json::array(
          {{{"role", "user"}, {"content", std::string(200, 'h')}},
           {{"role", "assistant"}, {"content", std::string(89, 'h')}}});
      req["max_tokens"] = 1;
      req["temperature"] = 0.0;
      auto r =
          client.Post("/v1/chat/completions", req.dump(), "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body);
      CHECK(ErrType(j) == "BadRequestError");
      const std::string msg = ErrMsg(j);
      CHECK(msg.find("289") != std::string::npos);
      CHECK(msg.find("288") != std::string::npos);
      CHECK(msg.find("bytes") != std::string::npos);
    });
  }

  // The bound is a property of the REQUEST, so it is decided before the model
  // is looked up -- mirroring vLLM, where validate_prompt_list_length is a
  // pydantic model_validator (completion/protocol.py:536-553) and therefore
  // runs at body validation, ahead of the router's check_model.
  SUBCASE("/v1/completions: the bound is decided before the model lookup") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    with_server(h, [](httplib::Client& client) {
      json req;
      req["model"] = "no-such-model";
      req["prompt"] = std::string(kMaxPromptBytes + 1, 'h');
      req["max_tokens"] = 1;
      auto r = client.Post("/v1/completions", req.dump(), "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);  // not the 404 an unknown model would earn
      CHECK(ErrMsg(json::parse(r->body)).find("288") != std::string::npos);
    });
  }
}


// The bound is DERIVED, so the derivation is what is pinned -- not the number.
// A literal 288 in the cases above would still pass if the bound came from
// somewhere else entirely.
TEST_CASE("api_server: the prompt bound IS max_model_len x MaxTokenBytes") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  // The fixture's longest STORED token text: the added token "<|end|>of".
  // "Ġworld" is 7, because the byte-level mark 'Ġ' (U+0120) is two UTF-8 bytes.
  REQUIRE(Fixture().MaxTokenBytes() == 9);

  SUBCASE("attached through the production seam: the product, exactly") {
    ServerHarness h(c, w, Fixture());
    ConfigureUtilityEndpoints(h.server, Fixture(), kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    CHECK(h.server.max_prompt_bytes() ==
          static_cast<size_t>(kMaxModelLen) * Fixture().MaxTokenBytes());
    CHECK(h.server.max_prompt_bytes() == kMaxPromptBytes);
  }

  SUBCASE("no tokenizer attached: unbounded, and the guard is inert") {
    ServerHarness h(c, w, Fixture());
    CHECK(h.server.max_prompt_bytes() == 0);
    // Inert, not "small": an oversized prompt reaches the engine and earns the
    // POST-encode token refusal instead, exactly as before this row.
    json req;
    req["prompt"] = std::string(kMaxPromptBytes + 1, 'h');
    req["max_tokens"] = 1;
    ApiServer::DispatchResult r = h.server.handle_completions(req.dump());
    CHECK(r.status == 400);
    CHECK(ErrMsg(json::parse(r.body)).find("maximum model length") !=
          std::string::npos);
  }

  // Upstream's own "no context length is known" state
  // (InputProcessor::ValidatePromptLen early-outs on max_model_len <= 0). A
  // bound cannot be derived from it, so there is none.
  SUBCASE("max_model_len <= 0: unbounded") {
    ServerHarness h(c, w, Fixture());
    h.server.set_tokenizer(&Fixture(), 0);
    CHECK(h.server.max_prompt_bytes() == 0);
    h.server.set_tokenizer(&Fixture(), kMaxModelLen);
    CHECK(h.server.max_prompt_bytes() == kMaxPromptBytes);
  }
}

// ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN. The chat handler carried a FIXED
// 200,000-character cap (VT_SERVER_MAX_PROMPT_CHARS) that was not derived from
// max_model_len, so a 262,144-token server refused prose above about 48k tokens
// with an HTTP 500. vLLM's only pre-tokenization bound is
// `max_input_tokens * tokenizer.max_chars_per_token`
// (vllm/renderers/params.py:342-365 at e126687a9a), refused as a 400; ours is
// this row's `max_model_len * MaxTokenBytes()`.
//
// Reproducing "over 200,000 characters but inside the context" needs a
// vocabulary whose tokens are LONG, or an engine with a 22k-token context. This
// fixture takes the first route: "h" repeated 2^k for k = 0..13, joined by
// power-of-two merges, so a run of 8,192 'h' is ONE token and 25 such tokens
// are 204,800 bytes. The derived bound is then 32 x 8192 = 262,144 bytes.
Tokenizer BuildLongTokenFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_apisrv_longtok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array();
  doc["normalizer"] = nullptr;
  // The same Split + ByteLevel pre-tokenizer as Fixture(): a run of letters is
  // one pretoken, so the merges below decide the token count alone.
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = json::object();
  json merges = json::array();
  std::string run = "h";
  for (int k = 0; k <= 13; ++k) {
    vocab[run] = k;
    if (k < 13) merges.push_back(json::array({run, run}));
    run += run;
  }
  // Fill the model's 22-id vocabulary, so every id the synthetic model can
  // sample detokenizes.
  const char* fill[] = {"e", "l", "o", "w", "r", "d", "Ġ", "1"};
  for (int i = 0; i < 8; ++i) vocab[fill[i]] = 14 + i;
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", merges}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& LongTokenFixture() {
  static const Tokenizer tok = BuildLongTokenFixture();
  return tok;
}

// Holds VT_SERVER_MAX_PROMPT_CHARS at one value for a scope and restores what
// the process had, so a developer shell that exports it cannot pick the arm.
class ScopedPromptCharsEnv {
 public:
  explicit ScopedPromptCharsEnv(const char* value) {
    const char* prev = std::getenv(kName);
    had_ = prev != nullptr;
    if (had_) prev_ = prev;
    vllm_test::SetEnv(kName, value);
  }
  ~ScopedPromptCharsEnv() {
    if (had_) {
      vllm_test::SetEnv(kName, prev_);
    } else {
      vllm_test::UnsetEnv(kName);
    }
  }
  ScopedPromptCharsEnv(const ScopedPromptCharsEnv&) = delete;
  ScopedPromptCharsEnv& operator=(const ScopedPromptCharsEnv&) = delete;

 private:
  static constexpr const char* kName = "VT_SERVER_MAX_PROMPT_CHARS";
  bool had_ = false;
  std::string prev_;
};

// Binds the harness's server to a loopback port, runs `body` with a client,
// then stops and joins the server.
template <class Body>
void WithPromptCapServer(ServerHarness& h, Body&& body) {
  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(120, 0);
  body(client);
  server_thread.join();  // stops the server, then joins
}

// One user message, greedy, one output token.
std::string PromptCapChatBody(const std::string& content) {
  json req;
  req["model"] = "test-model";
  req["messages"] = json::array({{{"role", "user"}, {"content", content}}});
  req["max_tokens"] = 1;
  req["temperature"] = 0.0;
  return req.dump();
}

TEST_CASE("api_server: the chat prompt cap never refuses what the context holds") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = LongTokenFixture();
  constexpr size_t kLongToken = 8192;
  REQUIRE(tok.MaxTokenBytes() == kLongToken);
  const size_t derived_bound = static_cast<size_t>(kMaxModelLen) * kLongToken;


  // 25 tokens, 204,800 bytes: above the old fixed 200,000, inside a 32-token
  // context, and below the derived 262,144-byte bound.
  const std::string fits(25 * kLongToken, 'h');
  REQUIRE(fits.size() > 200000);
  REQUIRE(fits.size() <= derived_bound);
  REQUIRE(tok.EncodeWithSpecialTokens(fits).size() == 25);

  SUBCASE("unset: a prompt over 200,000 bytes that fits the context is SERVED") {
    ScopedPromptCharsEnv env(nullptr);
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    REQUIRE(h.server.max_prompt_bytes() == derived_bound);
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(fits),
                           "application/json");
      REQUIRE(r);
      INFO("body: " << r->body.substr(0, 400));
      CHECK(r->status == 200);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrMsg(j).find("VT_SERVER_MAX_PROMPT_CHARS") == std::string::npos);
      // CHECK, not REQUIRE: a refusal here must not abort the TEST_CASE and
      // hide the later subcases' own verdicts.
      const bool served = j.is_object() && j.contains("choices") &&
                          j.at("choices").size() == 1 && j.contains("usage");
      CHECK(served);
      if (served) {
        CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
        CHECK(j.at("usage").at("prompt_tokens") == 25);
      }
    });
  }

  // The token refusal is untouched: 31 long tokens plus "e" and "l" are 33
  // tokens in 253,954 bytes, under the byte bound, so ValidatePromptLen decides
  // it with vLLM's _validate_prompt_len text
  // (vllm/v1/engine/input_processor.py:436-476).
  SUBCASE("unset: a prompt over max_model_len in TOKENS is still refused, 400") {
    ScopedPromptCharsEnv env(nullptr);
    const std::string over = std::string(31 * kLongToken, 'h') + "el";
    REQUIRE(over.size() <= derived_bound);
    REQUIRE(tok.EncodeWithSpecialTokens(over).size() == 33);
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(over),
                           "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrType(j) == "BadRequestError");
      CHECK(ErrCode(j) == 400);
      CHECK(ErrMsg(j).find("The decoder prompt (length 33) is longer than the "
                           "maximum model length of 32.") != std::string::npos);
      CHECK_FALSE(j.contains("choices"));
    });
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  // #1541's guarantee survives: a pathological body is refused by the derived
  // byte bound, before the template renders and before any encode.
  SUBCASE("unset: a 4 MiB prompt is refused by the byte bound, not tokenized") {
    ScopedPromptCharsEnv env(nullptr);
    const std::string huge(4u << 20, 'h');
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(huge),
                           "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrType(j) == "BadRequestError");
      const std::string msg = ErrMsg(j);
      CHECK(msg.find("prompt length 4194304 bytes exceeds the maximum allowed "
                     "prompt length of 262144 bytes") != std::string::npos);
      // The byte refusal, not the post-encode token one.
      CHECK(msg.find("maximum model length") == std::string::npos);
      CHECK_FALSE(j.contains("choices"));
    });
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  // An operator who SETS the variable keeps a ceiling, and a request over it is
  // the request's fault: 400, naming the variable, blaming no client.
  SUBCASE("set: an explicit operator ceiling refuses with 400, not 500") {
    ScopedPromptCharsEnv env("1000");
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(fits),
                           "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrType(j) == "BadRequestError");
      const std::string msg = ErrMsg(j);
      CHECK(msg.find("204800") != std::string::npos);
      CHECK(msg.find("VT_SERVER_MAX_PROMPT_CHARS=1000") != std::string::npos);
      CHECK(msg.find("Hermes") == std::string::npos);
      CHECK_FALSE(j.contains("choices"));
    });
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  // The ceiling's comparison is `>`: a rendered prompt of EXACTLY the ceiling is
  // served, and one byte over is refused. Either off-by-one (`>=`, or
  // `> ceiling + 1`) turns one half of this pair red.
  SUBCASE("set: a prompt exactly at the operator ceiling is served") {
    ScopedPromptCharsEnv env("204800");
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(fits),
                           "application/json");
      REQUIRE(r);
      INFO("body: " << r->body.substr(0, 400));
      CHECK(r->status == 200);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrMsg(j).find("VT_SERVER_MAX_PROMPT_CHARS") == std::string::npos);
      CHECK((j.is_object() && j.contains("choices")));
    });
    CHECK(h.input_processor.num_prompt_encodes() == 1);
  }
  SUBCASE("set: a prompt one byte over the operator ceiling is refused") {
    ScopedPromptCharsEnv env("204800");
    const std::string over = fits + "h";  // 204,801 bytes, 26 tokens
    REQUIRE(tok.EncodeWithSpecialTokens(over).size() == 26);
    ServerHarness h(c, w, tok);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", PromptCapChatBody(over),
                           "application/json");
      REQUIRE(r);
      CHECK(r->status == 400);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrType(j) == "BadRequestError");
      const std::string msg = ErrMsg(j);
      CHECK(msg.find("204801") != std::string::npos);
      CHECK(msg.find("VT_SERVER_MAX_PROMPT_CHARS=204800") != std::string::npos);
      CHECK_FALSE(j.contains("choices"));
    });
    CHECK(h.input_processor.num_prompt_encodes() == 0);
  }
}

// The default resolution itself. The end-to-end cases above cannot tell "no
// fixed ceiling" from "a fixed ceiling above the fixture's 262,144-byte derived
// bound", because the derived bound refuses such a prompt first. A real
// server's derived bound is tens of megabytes, so a finite default anywhere
// below it would refuse prompts the context holds, as the old 200,000 did.
// This case therefore pins the resolution: unset, empty and 0 are NO ceiling.
TEST_CASE("api_server: VT_SERVER_MAX_PROMPT_CHARS unset sets no fixed ceiling") {
  using vllm::entrypoints::openai::OperatorMaxPromptChars;
  {
    ScopedPromptCharsEnv env(nullptr);
    CHECK(OperatorMaxPromptChars() == 0);
  }
  {
    ScopedPromptCharsEnv env("");
    CHECK(OperatorMaxPromptChars() == 0);
  }
  {
    ScopedPromptCharsEnv env("0");
    CHECK(OperatorMaxPromptChars() == 0);
  }
  {
    ScopedPromptCharsEnv env("1000");
    CHECK(OperatorMaxPromptChars() == 1000);
  }
}

// F1 of the SERVE-REQUEST-LENGTH-GUARD amendment review. ApiServer's #1541
// guard sums `messages[].content` only, before the template renders, so bytes
// the template adds -- tool descriptions and schemas, assistant tool_calls
// arguments, per-message framing -- were never bounded before the encode. vLLM
// runs `_text_len_check` on the RENDERED prompt, template and tools included
// (vllm/renderers/params.py:342-370, reached through apply_pre_tokenization at
// params.py:386-399 @ e126687a9a). This fixture template renders the tool
// descriptions and the tool_calls arguments verbatim and adds nothing else, so
// the rendered length and token count are exact. The spy is
// InputProcessor::num_prompt_encodes(): a refusal before the encode leaves it 0.
std::string ToolsRenderingChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<vllm::entrypoints::openai::ChatCompletionToolsParam>& tools,
    const nlohmann::ordered_json&) {
  std::string p;
  for (const ChatMessage& m : messages) {
    if (m.content.has_value()) p += *m.content;
    if (m.tool_calls.has_value())
      for (const auto& call : *m.tool_calls) p += call.function.arguments;
  }
  for (const auto& t : tools)
    if (t.function.description.has_value()) p += *t.function.description;
  return p;
}

TEST_CASE("api_server: the derived bound applies to the RENDERED chat prompt") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = LongTokenFixture();
  constexpr size_t kLongToken = 8192;
  REQUIRE(tok.MaxTokenBytes() == kLongToken);
  const size_t derived_bound = static_cast<size_t>(kMaxModelLen) * kLongToken;
  ScopedPromptCharsEnv env(nullptr);

  // A small message, so the content-only #1541 guard passes, and `tool_bytes`
  // of 'h' in one tool description.
  auto tools_body = [](size_t tool_bytes) {
    json req = json::parse(PromptCapChatBody("h"));
    req["tools"] = json::array(
        {{{"type", "function"},
          {"function",
           {{"name", "f"}, {"description", std::string(tool_bytes, 'h')}}}}});
    return req.dump();
  };
  // A prior assistant turn whose tool_calls arguments carry `arg_bytes`.
  auto tool_calls_body = [](size_t arg_bytes) {
    json req = json::parse(PromptCapChatBody("h"));
    req["messages"] = json::array(
        {{{"role", "user"}, {"content", "h"}},
         {{"role", "assistant"},
          {"content", nullptr},
          {"tool_calls",
           json::array({{{"id", "call_0"},
                         {"type", "function"},
                         {"function",
                          {{"name", "f"},
                           {"arguments", std::string(arg_bytes, 'h')}}}}})}},
         {{"role", "tool"}, {"tool_call_id", "call_0"}, {"content", "h"}}});
    return req.dump();
  };
  auto check_byte_refusal = [&](const httplib::Result& r, size_t rendered) {
    REQUIRE(r);
    INFO("body: " << r->body.substr(0, 600));
    CHECK(r->status == 400);
    json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
    CHECK(ErrType(j) == "BadRequestError");
    CHECK(ErrCode(j) == 400);
    const std::string msg = ErrMsg(j);
    // vLLM's _text_len_check wording, in bytes (params.py:352-362).
    CHECK(msg.find("This model's maximum context length is 32 tokens.") !=
          std::string::npos);
    CHECK(msg.find("your prompt contains " + std::to_string(rendered) +
                   " bytes (more than " + std::to_string(derived_bound) +
                   " bytes, which is the upper bound for 32 input tokens)") !=
          std::string::npos);
    // Not the post-encode token refusal.
    CHECK(msg.find("The decoder prompt") == std::string::npos);
    CHECK_FALSE(j.contains("choices"));
  };

  SUBCASE("4 MiB in a tool description is refused BEFORE the encode") {
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", tools_body(4u << 20),
                           "application/json");
      check_byte_refusal(r, (4u << 20) + 1);
    });
    CHECK(h.input_processor.num_prompt_encodes() == 0);
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  SUBCASE("4 MiB in assistant tool_calls arguments is refused BEFORE the encode") {
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", tool_calls_body(4u << 20),
                           "application/json");
      check_byte_refusal(r, (4u << 20) + 2);
    });
    CHECK(h.input_processor.num_prompt_encodes() == 0);
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  // The boundary pair on the derived bound. Exactly 262,144 rendered bytes are
  // 32 tokens: not refused by the byte bound, so they reach the encode and the
  // token refusal decides them. One byte more is refused before the encode.
  SUBCASE("a rendered prompt exactly at the derived bound reaches the encode") {
    const size_t tool_bytes = derived_bound - 1;  // + the 1-byte message
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", tools_body(tool_bytes),
                           "application/json");
      REQUIRE(r);
      INFO("body: " << r->body.substr(0, 600));
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(ErrMsg(j).find("The decoder prompt (length 32)") !=
            std::string::npos);
      CHECK(ErrMsg(j).find("bytes (more than") == std::string::npos);
    });
    CHECK(h.input_processor.num_prompt_encodes() == 1);
  }
  SUBCASE("a rendered prompt one byte over the derived bound is refused") {
    const size_t tool_bytes = derived_bound;  // + the 1-byte message
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", tools_body(tool_bytes),
                           "application/json");
      check_byte_refusal(r, derived_bound + 1);
    });
    CHECK(h.input_processor.num_prompt_encodes() == 0);
  }
}

// Review B2 of the SERVE-REQUEST-LENGTH-GUARD amendment. The rendered-prompt
// check claims to run before EVERY encode create_chat_completion can reach, and
// the text path's spy (InputProcessor::num_prompt_encodes) sees only one of
// them. The other two encode outside InputProcessor's text overload:
//   * use_beam_search encodes with beam_tokenizer_->Encode(prompt), counted by
//     OpenAIServingChat::num_beam_prompt_encodes();
//   * the multimodal seam (MakeQwen3VLImageChatFn) re-renders the messages and
//     encodes inside chat_mm.cpp. The case wraps the REAL seam in a counter, so
//     a check moved below the seam's call site is seen as a seam entry.
// A reviewer moved the check below the beam-search block and every suite
// stayed green; these cases are the gate that mutation lacked. vLLM runs
// _text_len_check on the rendered prompt before any encode on either path
// (vllm/renderers/params.py:342-370,386-399 @ e126687a9a; beam search renders
// through the same renderer, chat_completion/serving.py:319-343).
TEST_CASE("api_server: the rendered-prompt bound runs before the beam and "
          "multimodal encodes") {
  namespace oai = vllm::entrypoints::openai;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = LongTokenFixture();
  constexpr size_t kLongToken = 8192;
  REQUIRE(tok.MaxTokenBytes() == kLongToken);
  const size_t derived_bound = static_cast<size_t>(kMaxModelLen) * kLongToken;
  ScopedPromptCharsEnv env(nullptr);

  // `tool_bytes` of 'h' in one tool description, so the content-only #1541
  // guard passes and only the rendered-prompt check can refuse.
  auto with_tools = [](json req, size_t tool_bytes) {
    req["tools"] = json::array(
        {{{"type", "function"},
          {"function",
           {{"name", "f"}, {"description", std::string(tool_bytes, 'h')}}}}});
    return req;
  };
  auto beam_body = [&](size_t tool_bytes) {
    json req = json::parse(PromptCapChatBody("h"));
    req["use_beam_search"] = true;
    req["n"] = 1;
    return with_tools(std::move(req), tool_bytes).dump();
  };
  auto check_byte_refusal = [&](const httplib::Result& r, size_t rendered) {
    REQUIRE(r);
    INFO("body: " << r->body.substr(0, 600));
    CHECK(r->status == 400);
    json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
    CHECK(ErrType(j) == "BadRequestError");
    const std::string msg = ErrMsg(j);
    CHECK(msg.find("your prompt contains " + std::to_string(rendered) +
                   " bytes (more than " + std::to_string(derived_bound) +
                   " bytes, which is the upper bound for 32 input tokens)") !=
          std::string::npos);
    CHECK(msg.find("The decoder prompt") == std::string::npos);
    CHECK_FALSE(j.contains("choices"));
  };

  // The control: an in-bound beam request reaches the beam encode once and is
  // served, so a counter that never moves cannot pass the refusal below.
  SUBCASE("an in-bound use_beam_search request reaches the beam encode") {
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    h.chat.set_beam_search_tokenizer(&tok, std::nullopt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", beam_body(1),
                           "application/json");
      REQUIRE(r);
      INFO("body: " << r->body.substr(0, 600));
      CHECK(r->status == 200);
      json j = json::parse(r->body, nullptr, /*allow_exceptions=*/false);
      CHECK(j.contains("choices"));
    });
    CHECK(h.chat.num_beam_prompt_encodes() == 1);
    CHECK(h.input_processor.num_prompt_encodes() == 0);
  }

  // One byte over the derived bound, on the default configuration.
  SUBCASE("use_beam_search over the bound is refused before the beam encode") {
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    h.chat.set_beam_search_tokenizer(&tok, std::nullopt);
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", beam_body(derived_bound),
                           "application/json");
      check_byte_refusal(r, derived_bound + 1);
    });
    CHECK(h.chat.num_beam_prompt_encodes() == 0);
    CHECK(h.input_processor.num_prompt_encodes() == 0);
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }

  // The same request with one image part, through the REAL production seam.
  // In bound, the seam is entered once and runs its own encode. The fixture
  // vocabulary cannot encode the seam's Qwen placeholder marker ("<" is not in
  // it), so that encode throws and the control answers 500 with the
  // tokenizer's own message: the observable proof that the seam's encode ran.
  // The codec runs only after that encode, so it stays at 0 in both arms.
  // Over the bound, the seam is never entered.
  size_t image_tool_bytes = 0;
  bool image_in_bound = false;
  SUBCASE("an in-bound image request enters the multimodal seam") {
    image_tool_bytes = 1;
    image_in_bound = true;
  }
  SUBCASE("an image request over the bound is refused before the multimodal seam") {
    image_tool_bytes = derived_bound;
  }
  if (image_tool_bytes != 0) {
    ServerHarness h(c, w, tok, false, ApiServer::kDefaultMaxConcurrentStreams,
                    ToolsRenderingChatPrompt);
    const vllm::MultiModalConfig default_cfg;
    const vllm::multimodal::BaseProcessingInfo info(
        default_cfg, oai::Qwen3VLChatSupportedMmLimits());
    vllm::multimodal::Qwen3VLProcessorConfig pcfg;
    pcfg.image_token_id = 3;
    const vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
    std::atomic<int> seam_calls{0};
    std::atomic<int> codec_calls{0};
    auto real_seam = oai::MakeQwen3VLImageChatFn(
        proc, tok, ToolsRenderingChatPrompt,
        [&codec_calls](const oai::DecodedMedia&) -> oai::DecodedImageRgb {
          codec_calls.fetch_add(1);
          throw std::runtime_error("codec reached");
        },
        info);
    h.chat.set_multimodal_chat_fn(
        [&seam_calls, real_seam](const std::vector<ChatMessage>& messages) {
          seam_calls.fetch_add(1);
          return real_seam(messages);
        });
    ConfigureUtilityEndpoints(h.server, tok, kMaxModelLen, h.async_engine,
                              UtilityEndpointOptions{});
    json req = json::parse(PromptCapChatBody("h"));
    req["messages"] = json::array(
        {{{"role", "user"},
          {"content",
           json::array({{{"type", "image_url"},
                         {"image_url",
                          {{"url", "data:image/x-raw-rgb;base64,AAAA"}}}},
                        {{"type", "text"}, {"text", "h"}}})}}});
    req = with_tools(std::move(req), image_tool_bytes);
    WithPromptCapServer(h, [&](httplib::Client& client) {
      auto r = client.Post("/v1/chat/completions", req.dump(),
                           "application/json");
      if (image_in_bound) {
        REQUIRE(r);
        INFO("body: " << r->body.substr(0, 600));
        CHECK(r->status == 500);
        CHECK(r->body.find("tokenizer: symbol") != std::string::npos);
      } else {
        check_byte_refusal(r, derived_bound + 1);
      }
    });
    CHECK(seam_calls.load() == (image_in_bound ? 1 : 0));
    CHECK(codec_calls.load() == 0);
    CHECK(h.input_processor.num_prompt_encodes() == 0);
    CHECK_FALSE(h.async_engine.has_unfinished_requests());
  }
}

// W2 port of test_async_llm.test_load at the HTTP boundary: concurrent workers
// submit into one AsyncLLM queue and complete as one scheduler batch; there is
// no server-wide engine mutex.
TEST_CASE("api_server: concurrent requests share AsyncLLM without state races") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  constexpr int kClients = 6;
  std::vector<int> statuses(kClients, -1);
  std::vector<std::string> texts(kClients);
  // Declared AFTER the vectors its bodies write into, so the joining destructor
  // runs BEFORE those vectors are destroyed. The previous order was safe only
  // because a joinable `std::thread` ended the process instead of unwinding.
  std::vector<ScopedThread> clients;
  clients.reserve(kClients);
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([&, i]() {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(30, 0);
      auto res = client.Post(
          "/v1/completions",
          R"({"prompt":"hello","max_tokens":4,"temperature":0.0})",
          "application/json");
      if (res) {
        statuses[static_cast<size_t>(i)] = res->status;
        try {
          json j = json::parse(res->body);
          texts[static_cast<size_t>(i)] =
              j.at("choices").at(0).at("text").get<std::string>();
        } catch (...) {
          statuses[static_cast<size_t>(i)] = -2;  // malformed body
        }
      }
    });
  }
  for (auto& t : clients) t.join();

  for (int i = 0; i < kClients; ++i) {
    CHECK(statuses[static_cast<size_t>(i)] == 200);
    CHECK_FALSE(texts[static_cast<size_t>(i)].empty());
  }
  // All greedy on the same prompt → identical deterministic output, which also
  // confirms no cross-request state bleed.
  for (int i = 1; i < kClients; ++i)
    CHECK(texts[static_cast<size_t>(i)] == texts[0]);

  server_thread.join();  // stops the server, then joins
}

TEST_CASE("api_server: configured persistent-stream capacity remains readable") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  constexpr size_t kStreamCapacity = 32;
  ServerHarness h(c, w, Fixture(), /*enable_force_include_usage=*/false,
                  kStreamCapacity);

  CHECK(h.server.http_worker_count() ==
        kStreamCapacity + ApiServer::kControlWorkerHeadroom);
  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  // cpp-httplib keeps a worker inside process_and_close_socket while a
  // keep-alive connection waits for its next request. Park exactly the
  // configured stream floor this way, then prove the bounded control reserve
  // still reads and answers another accepted socket. This is the deterministic
  // CPU reproduction of the c32 unread-socket failure mode.
  std::vector<std::unique_ptr<httplib::Client>> parked;
  parked.reserve(kStreamCapacity);
  for (size_t i = 0; i < kStreamCapacity; ++i) {
    auto client = std::make_unique<httplib::Client>("127.0.0.1", port);
    client->set_keep_alive(true);
    client->set_connection_timeout(5, 0);
    client->set_read_timeout(5, 0);
    auto response = client->Get("/health");
    REQUIRE(response);
    CHECK(response->status == 200);
    parked.push_back(std::move(client));
  }
  CHECK(parked.size() == kStreamCapacity);

  httplib::Client control("127.0.0.1", port);
  control.set_connection_timeout(5, 0);
  control.set_read_timeout(5, 0);
  auto response = control.Get("/health");
  REQUIRE(response);
  CHECK(response->status == 200);

  parked.clear();
  server_thread.join();  // stops the server, then joins
}

TEST_CASE("api_server: stream capacity must be positive") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  CHECK_THROWS_AS(ApiServer(h.completion, h.chat, h.models, "bad", 0),
                  std::invalid_argument);
}

TEST_CASE("api_server: legacy worker pool is an explicit diagnostic mode") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  ApiServer legacy(
      h.completion, h.chat, h.models, "legacy",
      ApiServer::kDefaultMaxConcurrentStreams,
      ApiServer::HttpWorkerPoolMode::kLegacyDynamic);
  CHECK(legacy.http_worker_count() == 0);
}

// SERVE-HTTP-TRANSPORT: mirror uvicorn/asyncio's default TCP_NODELAY on the
// accepted SSE-serving socket. vLLM serves through uvicorn over asyncio
// (`vllm/entrypoints/launcher.py:71,76`), and CPython asyncio disables Nagle on
// every accepted TCP stream socket (`asyncio/base_events.py:192-197`
// `_set_nodelay`, called from `asyncio/selector_events.py:950`). cpp-httplib
// defaults it off (`third_party/httplib/httplib.h:142`) and applies it to the
// accepted socket only when `tcp_nodelay_` is set (`httplib.h:12083`). This
// asserts the observable server-side transport effect, not the presence of the
// production call.
TEST_CASE(
    "api_server: accepted SSE socket has TCP_NODELAY (mirror uvicorn/asyncio)") {
#if defined(__linux__)
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  // A raw client socket (not httplib::Client) so the accepted server socket is
  // discoverable by matching the client's ephemeral peer port.
  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(client_fd >= 0);
  sockaddr_in server_addr{};
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(static_cast<uint16_t>(port));
  REQUIRE(inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) == 1);
  REQUIRE(::connect(client_fd, reinterpret_cast<sockaddr*>(&server_addr),
                    sizeof(server_addr)) == 0);

  // Drive one keep-alive request: the server accepts, applies its socket
  // options, answers, and parks the accepted socket in its keep-alive read loop
  // (the fd stays open for the scan below).
  const std::string request =
      "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n";
  REQUIRE(::send(client_fd, request.data(), request.size(), 0) ==
          static_cast<ssize_t>(request.size()));
  char buf[512];
  const ssize_t got = ::recv(client_fd, buf, sizeof(buf), 0);
  REQUIRE(got > 0);  // a served response proves the accept already happened

  sockaddr_in client_local{};
  socklen_t client_local_len = sizeof(client_local);
  REQUIRE(getsockname(client_fd, reinterpret_cast<sockaddr*>(&client_local),
                      &client_local_len) == 0);
  const int client_port = static_cast<int>(ntohs(client_local.sin_port));

  const int nodelay = AcceptedSocketTcpNoDelay(port, client_port);
  REQUIRE(nodelay >= 0);  // accepted socket was located
  CHECK(nodelay == 1);    // RED until ApiServer calls set_tcp_nodelay(true)

  ::close(client_fd);
  server_thread.join();  // stops the server, then joins
#endif  // defined(__linux__)
}

// ---------------------------------------------------------------------------
// MiniMax-H3 /v1/videos routes. These are ADDITIVE and OPT-IN: without a runner
// attached the handlers refuse, and the routes are never registered at all.
// ---------------------------------------------------------------------------

namespace {
std::string VideoBody() {
  return R"({"prompt":"a cat","num_inference_steps":4})";
}
}  // namespace

TEST_CASE("api_server: /v1/videos without a runner is a 500, not a crash") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  ApiServer::DispatchResult async_result = h.server.handle_videos(VideoBody());
  CHECK(async_result.status == 500);
  ApiServer::DispatchResult sync_result =
      h.server.handle_videos_sync(VideoBody());
  CHECK(sync_result.status == 500);
  // And an unknown job id is a 404 rather than an empty 200.
  CHECK(h.server.handle_video_status("video-0").status == 404);
}

TEST_CASE("api_server: /v1/videos/sync runs the runner and returns its path") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  std::string seen_prompt;
  int64_t seen_steps = 0;
  h.server.set_video_runner(
      [&](const vllm::openai::VideoRequest& req) -> std::string {
        seen_prompt = req.prompt;
        seen_steps = req.num_inference_steps;
        return "/tmp/out.mp4";
      });

  ApiServer::DispatchResult r = h.server.handle_videos_sync(VideoBody());
  REQUIRE(r.status == 200);
  CHECK(r.content_type == "application/json");
  nlohmann::json body = nlohmann::json::parse(r.body);
  CHECK(body.at("status") == "succeeded");
  CHECK(body.at("output_path") == "/tmp/out.mp4");
  // The parsed request actually reached the runner.
  CHECK(seen_prompt == "a cat");
  CHECK(seen_steps == 4);

  // The job is retrievable afterwards by id.
  ApiServer::DispatchResult status =
      h.server.handle_video_status(body.at("id").get<std::string>());
  REQUIRE(status.status == 200);
  CHECK(nlohmann::json::parse(status.body).at("status") == "succeeded");
}

TEST_CASE("api_server: a throwing runner fails the job, sync and async alike") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_video_runner(
      [](const vllm::openai::VideoRequest&) -> std::string {
        throw std::runtime_error("ffmpeg missing");
      });

  ApiServer::DispatchResult sync_result =
      h.server.handle_videos_sync(VideoBody());
  CHECK(sync_result.status == 500);
  CHECK(nlohmann::json::parse(sync_result.body)
            .at("error")
            .at("message")
            .get<std::string>() == "ffmpeg missing");

  // The async endpoint still accepts the job; the FAILURE surfaces on polling,
  // and crucially the worker thread does not terminate the process.
  ApiServer::DispatchResult async_result = h.server.handle_videos(VideoBody());
  REQUIRE(async_result.status == 200);
  const std::string id =
      nlohmann::json::parse(async_result.body).at("id").get<std::string>();
  std::string status;
  for (int i = 0; i < 400; ++i) {
    status = nlohmann::json::parse(h.server.handle_video_status(id).body)
                 .at("status")
                 .get<std::string>();
    if (status == "failed" || status == "succeeded") break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  CHECK(status == "failed");
}

TEST_CASE("api_server: /v1/videos rejects a malformed body with 400") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  h.server.set_video_runner(
      [](const vllm::openai::VideoRequest&) -> std::string {
        FAIL("the runner must not be reached for an invalid request");
        return {};
      });
  CHECK(h.server.handle_videos("{not json").status == 400);
  CHECK(h.server.handle_videos_sync("{not json").status == 400);
  // A body that parses but carries no prompt is equally a client error.
  CHECK(h.server.handle_videos_sync(R"({"num_inference_steps":4})").status == 400);
  // An OpenAI field we cannot read is a 400 too, never a silent default geometry.
  CHECK(h.server.handle_videos_sync(R"({"prompt":"x","size":"720p"})").status == 400);
  CHECK(h.server.handle_videos(R"({"prompt":"x","seconds":"soon"})").status == 400);
}

// ---------------------------------------------------------------------------
// OpenAI's Sora shape over the routes: the request aliases reach the runner, a
// model mismatch is stated rather than swallowed, and GET /v1/videos/{id}/content
// hands back the finished MP4 (the endpoint is unusable without it).
// ---------------------------------------------------------------------------

namespace {
// A scratch file that removes itself, standing in for the runner's muxed .mp4.
class ScratchFile {
 public:
  explicit ScratchFile(const std::string& contents) {
    static std::atomic<long long> counter{0};
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_cpp_video_" +
              std::to_string(std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count()) +
              "_" + std::to_string(counter.fetch_add(1)) + ".mp4"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }
  ~ScratchFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// A minimal MP4 `ftyp` box: BINARY, with embedded NULs, so a body that arrived
// truncated or text-mangled cannot compare equal by accident.
std::string FakeMp4Bytes() {
  static constexpr unsigned char kBytes[] = {
      0x00, 0x00, 0x00, 0x18, 'f',  't',  'y',  'p',  'm',  'p',  '4',
      '2',  0x00, 0x00, 0x00, 0x00, 'm',  'p',  '4',  '2',  'i',  's',
      'o',  'm',  0x00, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02};
  return std::string(reinterpret_cast<const char*>(kBytes), sizeof(kBytes));
}
}  // namespace

TEST_CASE("api_server: the OpenAI request aliases reach the runner unchanged") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  vllm::openai::VideoRequest seen;
  h.server.set_video_runner(
      [&](const vllm::openai::VideoRequest& req) -> std::string {
        seen = req;
        return "/tmp/out.mp4";
      });

  // The exact body an unmodified OpenAI client sends. Every value differs from
  // the field default, so this proves the parser ran end to end.
  ApiServer::DispatchResult r = h.server.handle_videos_sync(R"({
    "model": "test-model", "prompt": "a cat on a skateboard",
    "size": "1280x720", "seconds": "8", "input_reference": "/tmp/frame0.ppm"
  })");
  REQUIRE(r.status == 200);
  CHECK(seen.prompt == "a cat on a skateboard");
  CHECK(seen.width == 1280);
  CHECK(seen.height == 720);
  CHECK(seen.duration_seconds == doctest::Approx(8.0));
  CHECK(seen.input_reference_path == "/tmp/frame0.ppm");
  CHECK(seen.has_input_reference());
  CHECK(seen.model == "test-model");
  // The served model was named, so nothing is warned about.
  nlohmann::json body = nlohmann::json::parse(r.body);
  CHECK(body.at("model") == "test-model");
  CHECK_FALSE(body.contains("warning"));
}

TEST_CASE("api_server: every reference modality reaches the runner, or is refused") {
  // H3 has three reference modalities and OpenAI's schema carries one, so the
  // other two ride in `metadata`. What matters is that each one ARRIVES at the
  // runner (the library gates already prove a reference changes the output);
  // a reference that parsed and then never reached the pipeline is the failure
  // that looks like it worked.
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  vllm::openai::VideoRequest seen;
  int calls = 0;
  h.server.set_video_runner(
      [&](const vllm::openai::VideoRequest& req) -> std::string {
        seen = req;
        ++calls;
        return "/tmp/out.mp4";
      });

  SUBCASE("the metadata video + audio references arrive together") {
    REQUIRE(h.server
                .handle_videos_sync(R"({"prompt":"x","metadata":{
                    "input_reference_video":"/tmp/prev_job",
                    "input_reference_audio":"/tmp/voice.wav",
                    "trace_id":"abc-123"}})")
                .status == 200);
    CHECK(calls == 1);
    CHECK(seen.input_reference_video_dir == "/tmp/prev_job");
    CHECK(seen.input_reference_audio_path == "/tmp/voice.wav");
    CHECK(seen.metadata.at("trace_id") == "abc-123");  // free-form keys pass through
    CHECK_FALSE(seen.has_input_reference());
  }

  SUBCASE("the audio reference arrives alone, as inline bytes") {
    REQUIRE(h.server
                .handle_videos_sync(
                    R"({"prompt":"x","metadata":{"input_reference_audio":"data:audio/wav;base64,aGk="}})")
                .status == 200);
    CHECK(calls == 1);
    REQUIRE(seen.input_reference_audio_bytes.size() == 2);
    CHECK(seen.input_reference_audio_bytes[0] == 'h');
    CHECK_FALSE(seen.has_input_reference_video());
  }

  SUBCASE("fl2va + ref2va is a 400 and the runner is never reached") {
    ApiServer::DispatchResult r = h.server.handle_videos_sync(
        R"({"prompt":"x","input_reference":"/tmp/f0.ppm",
            "metadata":{"input_reference_audio":"/tmp/a.wav"}})");
    CHECK(r.status == 400);
    CHECK(calls == 0);  // nothing generated from a half-honoured request
    const std::string message =
        nlohmann::json::parse(r.body).at("error").at("message").get<std::string>();
    CHECK(message.find("exclusive") != std::string::npos);
  }
}

TEST_CASE("api_server: an unserved `model` warns on the job but still generates") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  bool ran = false;
  h.server.set_video_runner(
      [&](const vllm::openai::VideoRequest&) -> std::string {
        ran = true;
        return "/tmp/out.mp4";
      });

  // A Sora client cannot know the local video model's name, so refusing would
  // defeat the compatibility; ignoring would hide a real mismatch.
  ApiServer::DispatchResult r =
      h.server.handle_videos_sync(R"({"model":"sora-2-pro","prompt":"x"})");
  REQUIRE(r.status == 200);
  CHECK(ran);
  nlohmann::json body = nlohmann::json::parse(r.body);
  CHECK(body.at("status") == "succeeded");
  CHECK(body.at("model") == "sora-2-pro");
  REQUIRE(body.contains("warning"));
  const std::string warning = body.at("warning").get<std::string>();
  CHECK(warning.find("sora-2-pro") != std::string::npos);
  CHECK(warning.find("test-model") != std::string::npos);

  // The note survives on the polled record, not just the create response.
  ApiServer::DispatchResult status =
      h.server.handle_video_status(body.at("id").get<std::string>());
  REQUIRE(status.status == 200);
  CHECK(nlohmann::json::parse(status.body).at("warning") == warning);
}

TEST_CASE("api_server: GET /v1/videos/{id}/content serves the finished MP4") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // The scratch file outlives the server: ~ApiServer joins the async workers, so
  // it must be destroyed after them, not before.
  const std::string expected = FakeMp4Bytes();
  const ScratchFile mp4(expected);
  ServerHarness h(c, w, Fixture());

  SUBCASE("unknown id is a 404") {
    h.server.set_video_runner(
        [path = mp4.path()](const vllm::openai::VideoRequest&) -> std::string { return path; });
    ApiServer::DispatchResult r = h.server.handle_video_content("vid_nope");
    CHECK(r.status == 404);
    CHECK(nlohmann::json::parse(r.body).at("error").at("type") == "NotFoundError");
  }

  SUBCASE("an unfinished job is a 409 naming its status, never a truncated file") {
    // The runner blocks until released, so the async job is genuinely mid-flight.
    // Both captures are BY VALUE: the worker thread outlives this scope's locals.
    auto release = std::make_shared<std::atomic<bool>>(false);
    h.server.set_video_runner(
        [release, path = mp4.path()](const vllm::openai::VideoRequest&) -> std::string {
          while (!release->load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
          return path;
        });
    ApiServer::DispatchResult started =
        h.server.handle_videos(R"({"prompt":"a cat"})");
    REQUIRE(started.status == 200);
    const std::string id =
        nlohmann::json::parse(started.body).at("id").get<std::string>();

    ApiServer::DispatchResult r = h.server.handle_video_content(id);
    CHECK(r.status == 409);
    CHECK(r.content_type == "application/json");  // an error, not zero bytes of mp4
    const std::string message = nlohmann::json::parse(r.body)
                                    .at("error")
                                    .at("message")
                                    .get<std::string>();
    CHECK(message.find(id) != std::string::npos);
    // It says WHICH pending state, so the client knows to keep polling.
    CHECK((message.find("queued") != std::string::npos ||
           message.find("running") != std::string::npos));
    CHECK(r.body.find("ftyp") == std::string::npos);  // no bytes of the file leaked

    // Released, the SAME id now serves the bytes: 409 meant "not yet", not "no".
    release->store(true);
    std::string final_status;
    for (int i = 0; i < 400; ++i) {
      final_status = nlohmann::json::parse(h.server.handle_video_status(id).body)
                         .at("status")
                         .get<std::string>();
      if (final_status == "succeeded" || final_status == "failed") break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(final_status == "succeeded");
    ApiServer::DispatchResult after = h.server.handle_video_content(id);
    REQUIRE(after.status == 200);
    CHECK(after.body == FakeMp4Bytes());
  }

  SUBCASE("a failed job surfaces the failure, not an empty body") {
    h.server.set_video_runner(
        [](const vllm::openai::VideoRequest&) -> std::string {
          throw std::runtime_error("ffmpeg exited 1");
        });
    ApiServer::DispatchResult sync_result =
        h.server.handle_videos_sync(R"({"prompt":"a cat"})");
    REQUIRE(sync_result.status == 500);
    // The sync failure path still creates the job record, so its id is pollable.
    ApiServer::DispatchResult listed = h.server.handle_video_status("vid_1");
    REQUIRE(listed.status == 200);
    CHECK(nlohmann::json::parse(listed.body).at("status") == "failed");

    ApiServer::DispatchResult r = h.server.handle_video_content("vid_1");
    CHECK(r.status == 500);
    CHECK(nlohmann::json::parse(r.body)
              .at("error")
              .at("message")
              .get<std::string>()
              .find("ffmpeg exited 1") != std::string::npos);
  }

  SUBCASE("a succeeded job hands back the exact bytes as video/mp4") {
    h.server.set_video_runner(
        [path = mp4.path()](const vllm::openai::VideoRequest&) -> std::string { return path; });
    ApiServer::DispatchResult done =
        h.server.handle_videos_sync(R"({"prompt":"a cat"})");
    REQUIRE(done.status == 200);
    const std::string id = nlohmann::json::parse(done.body).at("id").get<std::string>();

    ApiServer::DispatchResult r = h.server.handle_video_content(id);
    REQUIRE(r.status == 200);
    CHECK(r.content_type == "video/mp4");
    CHECK(r.body.size() == expected.size());
    CHECK(r.body == expected);  // byte-exact, embedded NUL included
  }

  SUBCASE("an output that vanished is a 500, not a 200 with zero bytes") {
    std::string path;
    {
      const ScratchFile doomed(expected);
      path = doomed.path();
      h.server.set_video_runner(
          [path](const vllm::openai::VideoRequest&) -> std::string { return path; });
      ApiServer::DispatchResult done =
          h.server.handle_videos_sync(R"({"prompt":"a cat"})");
      REQUIRE(done.status == 200);
    }  // the file is removed here, while the job record still points at it
    ApiServer::DispatchResult r = h.server.handle_video_content("vid_1");
    CHECK(r.status == 500);
    CHECK(nlohmann::json::parse(r.body)
              .at("error")
              .at("message")
              .get<std::string>()
              .find("not readable") != std::string::npos);
  }
}

TEST_CASE("api_server: the /v1/videos routes do not exist without a runner") {
  // ADDITIVE + OPT-IN is load-bearing: a server built without video support must
  // be byte-identical to before, which only a REAL socket can prove (the handler
  // returning 500 says nothing about whether the route was registered).
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  auto with_server = [](ServerHarness& h, auto&& body) {
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(5, 0);
    body(client);
    server_thread.join();  // stops the server, then joins
  };

  SUBCASE("no runner: every video route 404s, and the core routes are unaffected") {
    ServerHarness h(c, w, Fixture());
    with_server(h, [](httplib::Client& client) {
      auto async_post =
          client.Post("/v1/videos", R"({"prompt":"a cat"})", "application/json");
      REQUIRE(async_post);
      CHECK(async_post->status == 404);  // 200 once a runner is attached
      auto sync_post = client.Post("/v1/videos/sync", R"({"prompt":"a cat"})",
                                   "application/json");
      REQUIRE(sync_post);
      CHECK(sync_post->status == 404);
      auto status = client.Get("/v1/videos/vid_1");
      REQUIRE(status);
      CHECK(status->status == 404);
      auto content = client.Get("/v1/videos/vid_1/content");
      REQUIRE(content);
      CHECK(content->status == 404);
      // Not OUR 404: the route is absent, so no ErrorResponse envelope is emitted.
      CHECK(content->body.find("NotFoundError") == std::string::npos);
      CHECK(status->body.find("NotFoundError") == std::string::npos);

      auto health = client.Get("/health");
      REQUIRE(health);
      CHECK(health->status == 200);
    });
  }

  SUBCASE("with a runner: all four routes serve, content included") {
    const std::string expected = FakeMp4Bytes();
    const ScratchFile mp4(expected);
    ServerHarness h(c, w, Fixture());
    h.server.set_video_runner(
        [path = mp4.path()](const vllm::openai::VideoRequest&) -> std::string { return path; });
    with_server(h, [&](httplib::Client& client) {
      auto created =
          client.Post("/v1/videos/sync",
                      R"({"model":"sora-2-pro","prompt":"a cat","size":"1280x720"})",
                      "application/json");
      REQUIRE(created);
      REQUIRE(created->status == 200);
      const std::string id =
          nlohmann::json::parse(created->body).at("id").get<std::string>();

      auto status = client.Get(("/v1/videos/" + id).c_str());
      REQUIRE(status);
      CHECK(status->status == 200);
      CHECK(nlohmann::json::parse(status->body).at("status") == "succeeded");

      auto content = client.Get(("/v1/videos/" + id + "/content").c_str());
      REQUIRE(content);
      REQUIRE(content->status == 200);
      CHECK(content->get_header_value("Content-Type") == "video/mp4");
      CHECK(content->body == expected);

      // Now the 404 IS ours: the route exists and the handler rejected the id.
      auto missing = client.Get("/v1/videos/vid_absent/content");
      REQUIRE(missing);
      CHECK(missing->status == 404);
      CHECK(nlohmann::json::parse(missing->body).at("error").at("type") ==
            "NotFoundError");
    });
  }
}

// ─── /v1/audio/transcriptions (ARCH-ONE-SURFACE ROW 1) ───────────────────────
// Task-conditional like /v1/videos: a TEXT server never registers the route; a
// serving-less (transcription-only) server registers it and NOT the generate
// routes — vLLM's supported_tasks-conditional registration
// (api_server.py:255-265) + speech_to_text/transcription semantics. The
// transcriber wraps the REAL library seam (ParakeetTranscriber) on the
// committed parakeet_e2e fixture, so the route is gated against the SAME
// pre-refactor transcript golden as the C ABI and the example.

namespace {

std::string ReadFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

struct AsrHarness {
  vllm::entrypoints::openai::OpenAIServingModels models{"parakeet-fixture"};
  ApiServer server{models, "test-version"};
  std::shared_ptr<vllm::multimodal::ParakeetTranscriber> transcriber;

  AsrHarness() {
    transcriber = std::make_shared<vllm::multimodal::ParakeetTranscriber>(
        vllm::multimodal::ParakeetTranscriber::FromDir(
            std::string(PARAKEET_E2E_FIXTURE_DIR) + "/ctc"));
    auto t = transcriber;
    server.set_transcriber([t](const uint8_t* wav, size_t n) {
      return t->TranscribeWavBytes(wav, n);
    });
  }
  std::string wav_bytes() const {
    return ReadFileBytes(std::string(PARAKEET_E2E_FIXTURE_DIR) + "/audio.wav");
  }
};

}  // namespace

TEST_CASE("api_server: transcriptions dispatch reproduces the golden") {
  AsrHarness h;
  const std::string wav = h.wav_bytes();

  // Default response_format ("json") -> TranscriptionResponse {"text": ...}.
  ApiServer::DispatchResult r = h.server.handle_audio_transcriptions(wav, "");
  CHECK(r.status == 200);
  CHECK(r.content_type == "application/json");
  CHECK(json::parse(r.body).at("text") == "atheat");

  // response_format=text -> the raw transcript as text/plain.
  r = h.server.handle_audio_transcriptions(wav, "text");
  CHECK(r.status == 200);
  CHECK(r.content_type == "text/plain; charset=utf-8");
  CHECK(r.body == "atheat");

  // Unsupported formats are named residuals -> 400.
  r = h.server.handle_audio_transcriptions(wav, "verbose_json");
  CHECK(r.status == 400);
  CHECK(json::parse(r.body).at("error").at("type") == "BadRequestError");

  // An empty upload -> 400.
  r = h.server.handle_audio_transcriptions("", "");
  CHECK(r.status == 400);

  // Undecodable audio -> 400 naming the cause.
  r = h.server.handle_audio_transcriptions("not a wav at all", "");
  CHECK(r.status == 400);

  // The serving-less server refuses the generate handlers with the
  // NotImplementedError mirror (the socket layer does not even register them).
  r = h.server.handle_completions("{}");
  CHECK(r.status == 500);
  CHECK(json::parse(r.body).at("error").at("message").get<std::string>().find(
            "does not support Completions") != std::string::npos);
  r = h.server.handle_chat_completions("{}");
  CHECK(r.status == 500);
}

TEST_CASE("api_server: transcriptions without a transcriber is a 500, not a crash") {
  vllm::entrypoints::openai::OpenAIServingModels models{"no-asr"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r = server.handle_audio_transcriptions("bytes", "");
  CHECK(r.status == 500);
  CHECK(json::parse(r.body).at("error").at("message").get<std::string>().find(
            "does not support Transcriptions") != std::string::npos);
}

TEST_CASE("api_server: transcriptions socket smoke (multipart), generate routes 404") {
  AsrHarness h;
  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);

    // Multipart upload, exactly the OpenAI wire shape.
    httplib::UploadFormDataItems items = {
        {"file", h.wav_bytes(), "audio.wav", "audio/wav"},
        {"response_format", "json", "", ""},
    };
    auto res = client.Post("/v1/audio/transcriptions", items);
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(json::parse(res->body).at("text") == "atheat");

    // A multipart body without the `file` part -> 400.
    httplib::UploadFormDataItems no_file = {
        {"response_format", "json", "", ""},
    };
    auto bad = client.Post("/v1/audio/transcriptions", no_file);
    REQUIRE(bad);
    CHECK(bad->status == 400);

    // The generate routes are NOT registered on a transcription-only server.
    auto completions = client.Post("/v1/completions", "{}", "application/json");
    REQUIRE(completions);
    CHECK(completions->status == 404);
    auto chat = client.Post("/v1/chat/completions", "{}", "application/json");
    REQUIRE(chat);
    CHECK(chat->status == 404);

    // Liveness + discovery still serve.
    auto health = client.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
    auto models_res = client.Get("/v1/models");
    REQUIRE(models_res);
    CHECK(models_res->status == 200);
    CHECK(json::parse(models_res->body).at("data").at(0).at("id") ==
          "parakeet-fixture");
  }

  server_thread.join();  // stops the server, then joins
}

TEST_CASE("api_server: the audio routes do not exist on a TEXT server") {
  // The reverse of the ASR socket smoke above, and the exact twin of "the
  // /v1/videos routes do not exist without a runner": task-conditional
  // registration means a TEXT-engine server (no transcriber attached) must
  // answer 404 from the ROUTE TABLE for /v1/audio/*. This pins the
  // `if (transcriber_)` registration gate itself — the direct-dispatch 500
  // test above cannot see route registration, so `if (true)` there would
  // register the route on every text server and only THIS test reds (the
  // mutated server answers 400/500 from the handler instead of 404).
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);

    // A well-formed multipart upload — exactly what the route would accept if
    // it existed — must fall through to httplib's 404, proving the route was
    // never registered (not merely that the handler rejected the payload).
    httplib::UploadFormDataItems items = {
        {"file", "RIFF fake", "audio.wav", "audio/wav"},
    };
    auto res = client.Post("/v1/audio/transcriptions", items);
    REQUIRE(res);
    CHECK(res->status == 404);

    // /v1/audio/translations is NOT routed anywhere yet (a named residual of
    // the ROW 1 fold): 404 on the text server documents that absence too.
    auto translations = client.Post("/v1/audio/translations", items);
    REQUIRE(translations);
    CHECK(translations->status == 404);

    // The text server still serves its own task, so the 404s above are about
    // the audio routes, not a dead server.
    auto health = client.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
  }

  server_thread.join();  // stops the server, then joins
}

// ─── ARCH-ONE-SURFACE ROW 8: the server's --device seam ──────────────────────
// The exact chain examples/server/main.cpp drives for `--device cpu`:
// vllm::DeviceFromString -> EngineParams.device -> LoadedEngine (SelectQueue's
// explicit arm) -> async_engine() -> the OpenAI serving stack — here over the
// synthetic in-memory model (no disk), asserting the device-selected engine
// SERVES and sits on the CPU queue. The policy matrix (cpu beats a registered
// accelerator; explicit cuda never falls back) is test_loaded_engine_dense.cpp;
// the C-ABI plumb is test_capi.cpp.
TEST_CASE("api_server: an explicit-cpu device-selected engine serves /v1/completions") {
  const HfConfig c = MakeConfig();
  vllm::entrypoints::EngineParams params;
  params.block_size = kBlockSize;
  params.num_blocks = 32;
  params.max_model_len = kMaxModelLen;
  params.max_num_seqs = 8;
  // The server's own parse of `--device cpu` (an unknown name throws there at
  // startup; pinned in test_loaded_engine_dense.cpp).
  params.device = vllm::DeviceFromString("cpu");
  vllm::entrypoints::LoadedEngine loaded(c, MakeWeights(c), BuildFixture(),
                                         params);
  // The observable seam: the runner of the explicitly-cpu engine is on the CPU
  // device (on a CUDA build this is the force-CPU pin; auto would select CUDA).
  CHECK(loaded.runner().device().type == vt::DeviceType::kCPU);

  OpenAIServingModels models("test-model");
  OpenAIServingCompletion completion(loaded.async_engine(), "test-model",
                                     /*enable_force_include_usage=*/false);
  OpenAIServingChat chat(loaded.async_engine(), "test-model", InVocabChatPrompt,
                         "hermes", /*reasoning_parser_name=*/std::string(),
                         /*enable_force_include_usage=*/false);
  ApiServer server(completion, chat, models, "9.9.9");

  const std::string body =
      R"({"model":"test-model","prompt":"hello","max_tokens":5,"temperature":0.0})";
  ApiServer::DispatchResult r = server.handle_completions(body);
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("choices").at(0).at("finish_reason") == "length");
  CHECK(j.at("usage").at("completion_tokens") == 5);
}

// ─── /v1/embeddings (ARCH-ONE-SURFACE ROW 6) ─────────────────────────────────
// Task-conditional like /v1/audio/transcriptions: a TEXT server never
// registers the route; a pooling (embedding) server registers it and NOT the
// generate routes — vLLM's supported_tasks-conditional registration
// (api_server.py:255-265) + pooling/embed/api_router.py:28 semantics. The
// embedder wraps the REAL engine path (LoadedEngine::FromModelDir on the
// committed llama_embed_e2e fixture -> LLMEngine::embed -> the registry
// forward + PoolingRunner step), the SAME path vllm_embed drives.

namespace {

struct EmbedHarness {
  vllm::entrypoints::openai::OpenAIServingModels models{"llama-embed-fixture"};
  ApiServer server{models, "test-version"};
  std::shared_ptr<vllm::entrypoints::LoadedEngine> loaded;
  std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();

  EmbedHarness() {
    vllm::entrypoints::EngineParams params;
    params.max_model_len = 64;
    loaded = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
        vllm::entrypoints::LoadedEngine::FromModelDir(
            std::string(LLAMA_EMBED_FIXTURE_DIR), params));
    auto engine = loaded;
    auto mu = mutex;
    auto counter = std::make_shared<std::atomic<uint64_t>>(0);
    server.set_embedder(
        [engine, mu, counter](const std::vector<std::string>& inputs) {
          std::lock_guard<std::mutex> lock(*mu);
          ApiServer::EmbeddingBatch batch;
          for (const std::string& text : inputs) {
            std::vector<int32_t> ids =
                engine->tokenizer().EncodeWithSpecialTokens(text);
            REQUIRE(!ids.empty());
            batch.prompt_tokens += static_cast<int64_t>(ids.size());
            vllm::RequestOutput ro = engine->engine().embed(
                std::move(ids), vllm::PoolingParams{},
                "embd-" + std::to_string(counter->fetch_add(1)));
            REQUIRE(ro.finished);
            REQUIRE(ro.pooling_output.has_value());
            batch.embeddings.push_back(std::move(*ro.pooling_output));
          }
          return batch;
        });
  }
};

}  // namespace

TEST_CASE("api_server: embeddings dispatch — OpenAI shape over the engine path") {
  EmbedHarness h;

  // ONE string input.
  ApiServer::DispatchResult r = h.server.handle_embeddings(
      R"({"model":"llama-embed-fixture","input":"the quick brown fox"})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("object") == "list");
  CHECK(j.at("model") == "llama-embed-fixture");
  CHECK(std::string(j.at("id")).rfind("embd-", 0) == 0);
  REQUIRE(j.at("data").size() == 1);
  CHECK(j.at("data").at(0).at("object") == "embedding");
  CHECK(j.at("data").at(0).at("index") == 0);
  REQUIRE(j.at("data").at(0).at("embedding").is_array());
  CHECK(j.at("data").at(0).at("embedding").size() == 64);  // hidden_size
  // Unit L2: the pooling normalize ran.
  double l2 = 0.0;
  for (const auto& v : j.at("data").at(0).at("embedding"))
    l2 += v.get<double>() * v.get<double>();
  CHECK(std::sqrt(l2) == doctest::Approx(1.0).epsilon(1e-5));
  CHECK(j.at("usage").at("prompt_tokens").get<int64_t>() > 0);
  CHECK(j.at("usage").at("total_tokens") == j.at("usage").at("prompt_tokens"));

  // ARRAY input: one embedding per string, input order.
  r = h.server.handle_embeddings(
      R"({"input":["the quick brown fox","the lazy dog"]})");
  CHECK(r.status == 200);
  j = json::parse(r.body);
  REQUIRE(j.at("data").size() == 2);
  CHECK(j.at("data").at(1).at("index") == 1);

  // Malformed / unsupported requests.
  CHECK(h.server.handle_embeddings("not json").status == 400);
  CHECK(h.server.handle_embeddings(R"({"model":"x"})").status == 404);
  CHECK(h.server.handle_embeddings(R"({"input":42})").status == 400);
  CHECK(h.server.handle_embeddings(R"({"input":[]})").status == 400);
  CHECK(h.server.handle_embeddings(R"({"input":[[1,2]]})").status == 400);
  CHECK(h.server
            .handle_embeddings(
                R"({"input":"x","encoding_format":"base64"})")
            .status == 400);
  CHECK(h.server.handle_embeddings(R"({"input":"x","dimensions":16})").status ==
        400);
}

TEST_CASE("api_server: embeddings without an embedder is a 500, not a crash") {
  vllm::entrypoints::openai::OpenAIServingModels models{"m"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r = server.handle_embeddings(R"({"input":"x"})");
  CHECK(r.status == 500);
  CHECK(json::parse(r.body).at("error").at("message") ==
        "The model does not support Embeddings API");
}

TEST_CASE("api_server: embeddings socket smoke; generate routes 404 on the "
          "embedding server") {
  EmbedHarness h;
  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);

    auto res = client.Post("/v1/embeddings",
                           R"({"input":"the quick brown fox"})",
                           "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(json::parse(res->body).at("data").at(0).at("embedding").size() == 64);

    // The generate routes are NOT registered on an embedding server (the
    // task-conditional registration, both directions).
    auto completions = client.Post("/v1/completions", "{}", "application/json");
    REQUIRE(completions);
    CHECK(completions->status == 404);
    auto chat = client.Post("/v1/chat/completions", "{}", "application/json");
    REQUIRE(chat);
    CHECK(chat->status == 404);

    // Liveness + discovery still serve.
    auto health = client.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
    auto models_res = client.Get("/v1/models");
    REQUIRE(models_res);
    CHECK(json::parse(models_res->body).at("data").at(0).at("id") ==
          "llama-embed-fixture");
  }

  server_thread.join();  // stops the server, then joins
}

TEST_CASE("api_server: /v1/embeddings does not exist on a TEXT server") {
  // The reverse pin, the exact twin of "the audio routes do not exist on a
  // TEXT server": task-conditional registration means a TEXT-engine server (no
  // embedder attached) must answer 404 from the ROUTE TABLE — a well-formed
  // request that the handler WOULD accept proves the route was never
  // registered (an `if (true)` registration mutation answers 200/400 from the
  // handler instead and only THIS test reds).
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);
    auto res = client.Post("/v1/embeddings", R"({"input":"hello"})",
                           "application/json");
    REQUIRE(res);
    CHECK(res->status == 404);
  }

  server_thread.join();  // stops the server, then joins
}

TEST_CASE("platform process: Windows command line preserves every argv byte") {
  const std::vector<std::wstring> argv = {
      L"ffmpeg", L"two words", L"C:\\path\\", L"a\"b", L""};
  const std::wstring expected =
      LR"cmd("ffmpeg" "two words" "C:\path\\" "a\"b" "")cmd";
  CHECK(vllm::platform::BuildWindowsCommandLine(argv) == expected);
}

#if !defined(_WIN32)
TEST_CASE("platform process: direct argv runner propagates the child exit") {
  CHECK(vllm::platform::RunProcessArgv({"/bin/sh", "-c", "exit 23"}) == 23);
}
#endif

TEST_CASE("platform shutdown: a stop request is thread-safe and idempotent") {
  std::atomic<int> stops{0};
  vllm::platform::ConsoleShutdown shutdown([&]() { ++stops; }, false);
  shutdown.RequestStop();
  shutdown.RequestStop();
  CHECK(stops.load() == 1);
}

#if defined(_WIN32)
TEST_CASE("platform shutdown: teardown drains an acquired console handler") {
  constexpr DWORD kWaitMs = 5000;
  HANDLE acquired = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  HANDLE resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  HANDLE before_drain = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  REQUIRE(acquired != nullptr);
  REQUIRE(resume != nullptr);
  REQUIRE(before_drain != nullptr);
  std::atomic<int> stops{0};
  std::atomic<bool> destroyed{false};
  auto shutdown = std::make_unique<vllm::platform::ConsoleShutdown>(
      [&]() { ++stops; });
  shutdown->SetBeforeDrainEventForTest(before_drain);
  // Both threads below block until `resume` is set, so their stop action is
  // that SetEvent: an unwind must not park the joining destructor forever.
  // `resume` is manual-reset, so setting an already-set event is a no-op and
  // the explicit SetEvent calls further down stay exactly as they were.
  ScopedThread handler(
      [&] {
        CHECK(vllm::platform::ConsoleShutdown::DispatchControlEventForTest(
            CTRL_BREAK_EVENT, acquired, resume));
      },
      [&] { SetEvent(resume); });
  const DWORD acquired_result = WaitForSingleObject(acquired, kWaitMs);
  if (acquired_result != WAIT_OBJECT_0) {
    SetEvent(resume);
    handler.join();
    shutdown.reset();
    CloseHandle(before_drain);
    CloseHandle(resume);
    CloseHandle(acquired);
    FAIL("console handler did not acquire state within timeout");
  }
  ScopedThread destroyer(
      [&] {
        shutdown.reset();
        destroyed.store(true, std::memory_order_release);
      },
      [&] { SetEvent(resume); });
  const DWORD drain_result = WaitForSingleObject(before_drain, kWaitMs);
  if (drain_result != WAIT_OBJECT_0) {
    SetEvent(resume);
    handler.join();
    destroyer.join();
    CloseHandle(before_drain);
    CloseHandle(resume);
    CloseHandle(acquired);
    FAIL("console teardown did not reach drain within timeout");
  }
  CHECK_FALSE(destroyed.load(std::memory_order_acquire));
  REQUIRE(SetEvent(resume) != 0);
  handler.join();
  destroyer.join();
  CHECK(destroyed.load(std::memory_order_acquire));
  CHECK(stops.load() == 1);
  CloseHandle(before_drain);
  CloseHandle(resume);
  CloseHandle(acquired);
}
#endif

// ── MULTIMODAL INPUT LIMITS, END TO END (#607 wave L2, #686) ────────────────
//
// The gate PR #685's reviewer owed to this wave: L1 ported the refusal and
// proved it throws, but nothing called it on a live request, so the claim "and
// it becomes HTTP 400" was unproven end to end. This is that proof, and it is
// deliberately written to distinguish 400 from BOTH of the wrong answers:
//
//   * NOT 500. `InputValidationError` is caught at api_server.cpp:252, AHEAD of
//     the generic `std::exception -> InternalServerError` arm. A refusal thrown
//     as any other type would land as a 500 — a client mistake reported as a
//     server fault — which is exactly why L1 reused the ONE validation type
//     instead of adding a multimodal-only one. Leg 3 below pins that by
//     throwing a bespoke type through the same seam and observing the 500.
//   * NOT a truncated 200. Before this wave the seam took the first image_url
//     part and `break`ed, so THREE images produced a perfectly ordinary 200
//     about one of them (#686). Leg 2 shows a within-limit request still
//     answering 200, so the 400 is a LIMIT decision, not "multimodal is off".
//
// RECORDED, because the RED run says something #686 does not: against the code
// before this wave, leg 2 (the validate-then-build shape) returned the truncated
// 200 exactly as #686 describes, but leg 1 — the REAL MakeQwen3VLImageChatFn —
// returned a FIVE HUNDRED. It injects one placeholder marker per image part but
// routes only the first image, so ExpandImagePlaceholders raised "more image
// placeholders than grids" and the client saw a server fault carrying an
// internal message. Both are wrong in the same way (neither is upstream's
// refusal) and both become the 400 below, but the issue's "served with one,
// silently" understates the production seam's case rather than overstating it.
//
// Upstream chain: chat_utils.py:662 (the per-item check) ->
// multimodal/processing/context.py:409-428 (VLLMValidationError) ->
// serve/utils/error_response.py:62-65 (BadRequestError / 400), all at
// 5559679229bc.
TEST_CASE("api_server: an over-limit multimodal chat request is HTTP 400") {
  namespace oai = vllm::entrypoints::openai;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  // A chat body carrying `n` image parts, in the OpenAI array-content form.
  auto ImageBody = [](int n) {
    json parts = json::array();
    for (int i = 0; i < n; ++i) {
      parts.push_back({{"type", "image_url"},
                       {"image_url",
                        {{"url", "data:image/x-raw-rgb;base64,AAAA"}}}});
    }
    parts.push_back({{"type", "text"}, {"text", "hello"}});
    const json body = {
        {"messages", json::array({{{"role", "user"}, {"content", parts}}})},
        {"max_completion_tokens", 4},
        {"temperature", 0.0}};
    return body.dump();
  };

  // The seam's own limits, folded with a DEFAULT MultiModalConfig (no
  // --limit-mm-per-prompt, no --language-model-only): image=1.
  const vllm::MultiModalConfig default_cfg;
  const vllm::multimodal::BaseProcessingInfo info(
      default_cfg, oai::Qwen3VLChatSupportedMmLimits());

  // ── LEG 1: the REAL production seam. MakeQwen3VLImageChatFn validates before
  // it decodes anything, so the processor and codec below are never reached on
  // this path — which is why a synthetic processor config is honest here: the
  // refusal is the whole code path under test.
  vllm::multimodal::Qwen3VLProcessorConfig pcfg;
  pcfg.image_token_id = 3;  // inside the fixture vocab (ids 0..21)
  const vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  oai::ImageCodecFn never_reached =
      [](const oai::DecodedMedia&) -> oai::DecodedImageRgb {
    FAIL("the codec must not run: the limit check refuses first");
    return {};
  };
  h.chat.set_multimodal_chat_fn(oai::MakeQwen3VLImageChatFn(
      proc, Fixture(), InVocabChatPrompt, never_reached, info));

  ApiServer::DispatchResult refused =
      h.server.handle_chat_completions(ImageBody(3));
  CHECK(refused.status == 400);
  // Explicitly NOT the 500 arm, and explicitly not a 200 body.
  CHECK(refused.status != 500);
  CHECK(refused.status != 200);
  {
    const json j = json::parse(refused.body);
    CHECK(j.at("error").at("type") == "BadRequestError");
    // Upstream's message text VERBATIM (context.py:421-423), reaching the wire.
    CHECK(j.at("error").at("message") ==
          "At most 1 image(s) may be provided in one prompt.");
    // A truncated 200 would have carried a completion instead.
    CHECK_FALSE(j.contains("choices"));
  }

  // ── LEG 2: the SAME server, a WITHIN-limit request, still 200. This is what
  // separates "the limit refused you" from "multimodal requests are off": the
  // seam here is the production SHAPE (validate, then build the engine input),
  // with the synthetic engine's in-vocab ids standing in for the real
  // processor's expansion, because the fixture vocab is 22 tokens wide and a
  // real Qwen image id (151655) would run off the end of it.
  h.chat.set_multimodal_chat_fn(
      [&info](const std::vector<ChatMessage>& messages)
          -> std::optional<vllm::multimodal::MultiModalInputs> {
        oai::ValidateChatMmLimits(info, messages);
        vllm::multimodal::MultiModalInputs mm;
        mm.prompt_token_ids = {13, 17};  // "hello", " world"
        vllm::multimodal::MultiModalFeatureSpec spec;
        spec.modality = "image";
        spec.offset = 0;
        spec.length = 1;
        mm.mm_features.push_back(std::move(spec));
        return mm;
      });
  ApiServer::DispatchResult served =
      h.server.handle_chat_completions(ImageBody(1));
  CHECK(served.status == 200);
  {
    const json j = json::parse(served.body);
    CHECK(j.at("object") == "chat.completion");
    CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
  }
  // ...and the same seam refuses three, so leg 1's 400 was not an artifact of
  // the production seam's own decode path.
  CHECK(h.server.handle_chat_completions(ImageBody(3)).status == 400);

  // ── LEG 3: the DISCRIMINATOR. A seam that throws anything OTHER than
  // InputValidationError lands as a 500. Without this leg, "status == 400"
  // could be satisfied by a handler that answered 400 to every seam failure,
  // and the type L1 chose would be doing no work.
  h.chat.set_multimodal_chat_fn(
      [](const std::vector<ChatMessage>&)
          -> std::optional<vllm::multimodal::MultiModalInputs> {
        throw std::runtime_error("a seam failure that is NOT a validation error");
      });
  ApiServer::DispatchResult faulted =
      h.server.handle_chat_completions(ImageBody(1));
  CHECK(faulted.status == 500);
  CHECK(json::parse(faulted.body).at("error").at("type") ==
        "InternalServerError");
}

TEST_CASE("api_server: --language-model-only answers an image request with 400") {
  // The flag's main observable effect, at the HTTP boundary. Upstream's
  // --language-model-only is not "the same server, minus some VRAM" — it is a
  // server that answers an image request with "At most 0 image(s) may be
  // provided in one prompt." (multimodal.py:78-80 -> :326-327 ->
  // context.py:409-428). The VRAM half is wave L3 and is NOT claimed here.
  namespace oai = vllm::entrypoints::openai;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::MultiModalConfig lm_only;
  lm_only.language_model_only = true;
  const vllm::multimodal::BaseProcessingInfo info(
      lm_only, oai::Qwen3VLChatSupportedMmLimits());

  vllm::multimodal::Qwen3VLProcessorConfig pcfg;
  pcfg.image_token_id = 3;
  const vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  oai::ImageCodecFn never_reached =
      [](const oai::DecodedMedia&) -> oai::DecodedImageRgb {
    FAIL("the codec must not run under --language-model-only");
    return {};
  };
  h.chat.set_multimodal_chat_fn(oai::MakeQwen3VLImageChatFn(
      proc, Fixture(), InVocabChatPrompt, never_reached, info));

  const json body = {
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url",
                                     {{"url",
                                       "data:image/x-raw-rgb;base64,AAAA"}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0}};
  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(body.dump());
  CHECK(r.status == 400);
  const json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("message") ==
        "At most 0 image(s) may be provided in one prompt. "
        "Set `--limit-mm-per-prompt` to increase this limit.");

  // A TEXT-only request on the same server is unaffected — the flag limits
  // multimodal INPUT, it does not turn the server off.
  const ApiServer::DispatchResult text = h.server.handle_chat_completions(
      R"({"messages":[{"role":"user","content":"hello"}],)"
      R"("max_completion_tokens":4,"temperature":0.0})");
  CHECK(text.status == 200);
}

// ---------------------------------------------------------------------------
// POST /v1/audio/speech (W6 of #672). ADDITIVE and OPT-IN, exactly like the
// video runner and the transcriber above: without a synthesizer attached the
// handler refuses and the route is never registered at all.
// ---------------------------------------------------------------------------

namespace {
std::string MusicBody() {
  return R"({"model":"minimax-music3","lyrics":"[Verse]\nMorning light\n",
             "description":"Genre: acoustic pop. BPM: 96.",
             "audio_duration":12.5,"num_inference_steps":4,"seed":7})";
}

// A 44100 Hz STEREO RIFF/WAVE, so the route's output can be checked for the
// rate and channel count Music3 declares rather than for "some bytes".
vllm::openai::SpeechResponse StereoWav(int64_t frames) {
  std::string wav;
  const uint32_t payload = static_cast<uint32_t>(frames * 2 * 2);
  const auto put32 = [&wav](uint32_t v) {
    for (int i = 0; i < 4; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  const auto put16 = [&wav](uint16_t v) {
    for (int i = 0; i < 2; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  wav += "RIFF";
  put32(36u + payload);
  wav += "WAVE";
  wav += "fmt ";
  put32(16u);
  put16(1u);
  put16(2u);       // stereo
  put32(44100u);   // the family's NATIVE rate
  put32(44100u * 4u);
  put16(4u);
  put16(16u);
  wav += "data";
  put32(payload);
  for (int64_t i = 0; i < frames * 2; ++i) put16(static_cast<uint16_t>(i & 0xFFFF));
  vllm::openai::SpeechResponse out;
  out.wav = wav;
  out.sample_rate = 44100;
  out.channels = 2;
  out.samples_per_channel = frames;
  return out;
}

// A 44-byte-header 16-bit PCM MONO WAV as a base64 `data:` URL, which is the
// only reference-clip encoding `ParseSpeechRequest` accepts. Built here, like
// its twin in test_speech_api.cpp, so the sample values stay visible to the
// assertions rather than sitting in a committed blob.
std::string MonoWavDataUri(const std::vector<int16_t>& samples, uint32_t rate) {
  std::string wav;
  const uint32_t payload = static_cast<uint32_t>(samples.size() * 2);
  const auto put32 = [&wav](uint32_t v) {
    for (int i = 0; i < 4; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  const auto put16 = [&wav](uint16_t v) {
    for (int i = 0; i < 2; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  wav += "RIFF";
  put32(36u + payload);
  wav += "WAVE";
  wav += "fmt ";
  put32(16u);
  put16(1u);  // PCM
  put16(1u);  // mono
  put32(rate);
  put32(rate * 2);
  put16(2u);
  put16(16u);
  wav += "data";
  put32(payload);
  for (const int16_t sample : samples) put16(static_cast<uint16_t>(sample));

  static const char* kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string b64;
  for (size_t i = 0; i < wav.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(wav[i]);
    const uint32_t b = i + 1 < wav.size() ? static_cast<unsigned char>(wav[i + 1]) : 0u;
    const uint32_t d = i + 2 < wav.size() ? static_cast<unsigned char>(wav[i + 2]) : 0u;
    const uint32_t triple = (a << 16) | (b << 8) | d;
    b64 += kAlphabet[(triple >> 18) & 0x3F];
    b64 += kAlphabet[(triple >> 12) & 0x3F];
    b64 += i + 1 < wav.size() ? kAlphabet[(triple >> 6) & 0x3F] : '=';
    b64 += i + 2 < wav.size() ? kAlphabet[triple & 0x3F] : '=';
  }
  return "data:audio/wav;base64," + b64;
}

uint32_t WavU32(const std::string& wav, size_t offset) {
  return static_cast<uint32_t>(static_cast<unsigned char>(wav[offset])) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(wav[offset + 3])) << 24);
}
uint16_t WavU16(const std::string& wav, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(wav[offset]) |
                               (static_cast<unsigned char>(wav[offset + 1]) << 8));
}
}  // namespace

TEST_CASE("api_server: /v1/audio/speech without a synthesizer is a 500, not a crash") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  // ADDITIVITY: with nothing attached the handler refuses and NO speech
  // envelope leaks — the body is the ordinary OpenAI error object, and the
  // route itself is never registered, so the socket answers 404 as before.
  ApiServer::DispatchResult r = h.server.handle_audio_speech(MusicBody());
  CHECK(r.status == 500);
  CHECK(r.content_type == "application/json");
  CHECK(r.body.find("No speech synthesizer configured") != std::string::npos);
  CHECK(r.body.find("RIFF") == std::string::npos);
  CHECK(r.body.find("minimax-music3") == std::string::npos);
}

TEST_CASE("api_server: /v1/audio/speech serves the family's 44100 Hz stereo WAV") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::openai::SpeechRequest seen;
  int64_t calls = 0;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  caps.requires_reference_audio = false;  // Music3 conditions on text alone
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest& req) {
        seen = req;
        ++calls;
        return StereoWav(1024);
      },
      caps);

  ApiServer::DispatchResult r = h.server.handle_audio_speech(MusicBody());
  CHECK(r.status == 200);
  // AUDIO BYTES, not a JSON envelope around them.
  CHECK(r.content_type == "audio/wav");
  REQUIRE(r.body.size() > 44);
  CHECK(r.body.compare(0, 4, "RIFF") == 0);
  CHECK(WavU16(r.body, 22) == 2);      // stereo
  CHECK(WavU32(r.body, 24) == 44100u); // the native rate, unresampled
  CHECK(WavU16(r.body, 34) == 16);     // 16-bit PCM
  CHECK(r.body.size() == 44u + 1024u * 2u * 2u);

  // Every field reached the seam, and each value DIFFERS from its default.
  CHECK(calls == 1);
  CHECK(seen.lyrics == "[Verse]\nMorning light\n");
  CHECK(seen.description == "Genre: acoustic pop. BPM: 96.");
  CHECK(seen.text.empty());
  CHECK(seen.audio_duration_s == doctest::Approx(12.5));
  CHECK(seen.num_inference_steps == 4);
  CHECK(seen.seed == 7);
  CHECK(seen.model == "minimax-music3");
}

TEST_CASE("api_server: requires_reference_audio REFUSES before the runner is called") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  int64_t calls = 0;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "indextts2";
  caps.sample_rate = 22050;
  caps.channels = 1;
  // TRUE — IndexTTS-2 has NO text-only synthesis, so an absent clip is a
  // refusal rather than a default voice. This is the value that DIFFERS from
  // Music3's, so a pass proves the flag was consulted.
  caps.requires_reference_audio = true;
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest&) {
        ++calls;
        return StereoWav(8);
      },
      caps);

  ApiServer::DispatchResult r = h.server.handle_audio_speech(R"({"input":"hello"})");
  CHECK(r.status == 400);
  CHECK(r.body.find("indextts2") != std::string::npos);
  CHECK(r.body.find("reference_audio") != std::string::npos);
  // THE POINT: nothing staged and nothing synthesized.
  CHECK(calls == 0);

  // And the SAME request against a family that needs no clip is served.
  ServerHarness other(c, w, Fixture());
  int64_t other_calls = 0;
  vllm::openai::SpeechCapabilities music;
  music.family = "minimax-music3";
  music.sample_rate = 44100;
  music.channels = 2;
  music.requires_reference_audio = false;
  other.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest&) {
        ++other_calls;
        return StereoWav(8);
      },
      music);
  ApiServer::DispatchResult served =
      other.server.handle_audio_speech(R"({"lyrics":"[Verse]\nx\n"})");
  CHECK(served.status == 200);
  CHECK(other_calls == 1);
}

TEST_CASE("api_server: /v1/audio/speech surfaces the family's own refusal verbatim") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  h.server.set_synthesizer(
      [](const vllm::openai::SpeechRequest&) -> vllm::openai::SpeechResponse {
        throw std::runtime_error(
            "MiniMax-Music3: the AUTOREGRESSIVE HEAD is not implemented (W2, issue #672)");
      },
      caps);
  ApiServer::DispatchResult r = h.server.handle_audio_speech(MusicBody());
  CHECK(r.status == 500);
  // The message NAMES the missing stage; a generic body would throw that away
  // and send the caller to read loader source.
  CHECK(r.body.find("AUTOREGRESSIVE HEAD") != std::string::npos);
  CHECK(r.body.find("#672") != std::string::npos);

  // A malformed body is a 400 BEFORE the runner, and an unsupported field is
  // refused rather than dropped.
  CHECK(h.server.handle_audio_speech("{not json").status == 400);
  CHECK(h.server.handle_audio_speech("{}").status == 400);
  CHECK(h.server.handle_audio_speech(R"({"input":"hi","voice":"alloy"})").status == 400);
  CHECK(h.server.handle_audio_speech(R"({"input":"hi","stream":true})").status == 400);
}

TEST_CASE("api_server: /v1/audio/speech route registration is ADDITIVE over a real socket") {
  // The handler-dispatch cases above prove the LOGIC; this one proves the ROUTE
  // TABLE, which is where additivity actually lives: without a synthesizer the
  // endpoint must not exist at all, so a server built before W6 and one built
  // after are indistinguishable to a client that never asks for speech.
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  auto with_socket = [](ServerHarness& h, auto&& body) {
    const int port = h.server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    ScopedServerThread server_thread(h.server);
    for (int i = 0; i < 500 && !h.server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(h.server.is_running());
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);
    body(client);
    server_thread.join();  // stops the server, then joins
  };

  SUBCASE("with NO speech family attached the route is 404 and nothing leaks") {
    ServerHarness h(c, w, Fixture());
    with_socket(h, [](httplib::Client& client) {
      auto res = client.Post("/v1/audio/speech", MusicBody(), "application/json");
      REQUIRE(res);
      // 404 from the route table, NOT a 500 from a handler — the endpoint does
      // not exist, exactly as before W6 existed.
      CHECK(res->status == 404);
      // No speech envelope of any kind reaches the wire.
      CHECK(res->body.find("minimax-music3") == std::string::npos);
      CHECK(res->body.find("synthesizer") == std::string::npos);
      CHECK(res->body.find("RIFF") == std::string::npos);
      // And the routes that were always there are untouched.
      auto health = client.Get("/health");
      REQUIRE(health);
      CHECK(health->status == 200);
      auto models = client.Get("/v1/models");
      REQUIRE(models);
      CHECK(models->status == 200);
    });
  }

  SUBCASE("with a speech family attached the route serves audio/wav bytes") {
    ServerHarness h(c, w, Fixture());
    vllm::openai::SpeechCapabilities caps;
    caps.family = "minimax-music3";
    caps.sample_rate = 44100;
    caps.channels = 2;
    caps.requires_reference_audio = false;
    h.server.set_synthesizer(
        [](const vllm::openai::SpeechRequest&) { return StereoWav(2048); }, caps);
    with_socket(h, [](httplib::Client& client) {
      auto res = client.Post("/v1/audio/speech", MusicBody(), "application/json");
      REQUIRE(res);
      CHECK(res->status == 200);
      CHECK(res->get_header_value("Content-Type") == "audio/wav");
      REQUIRE(res->body.size() == 44u + 2048u * 2u * 2u);
      CHECK(res->body.compare(0, 4, "RIFF") == 0);
      CHECK(WavU16(res->body, 22) == 2);       // stereo
      CHECK(WavU32(res->body, 24) == 44100u);  // the family's native rate
      // Every OTHER route is exactly where it was: attaching a synthesizer adds
      // one endpoint and moves none.
      auto health = client.Get("/health");
      REQUIRE(health);
      CHECK(health->status == 200);
      auto videos = client.Post("/v1/videos", R"({"prompt":"x"})", "application/json");
      REQUIRE(videos);
      CHECK(videos->status == 404);  // still unregistered, no video runner attached
    });
  }
}

// CONCURRENCY, entered through the production route (#2836,
// SERVE-SPEECH-ENGINE-SERIALIZE).
//
// `speech_engine.h` states of `Synthesize`: "Implementations serialize
// internally (staged weights are shared state)". Nothing held anyone to it, and
// this route is where that costs something: `handle_audio_speech` calls the
// synthesizer INLINE on whichever cpp-httplib worker served the request
// (`api_server.cpp:636`), it takes no lock, and the pool is TWELVE threads on a
// stock server -- `HttpWorkerCount` is `max_concurrent_streams + headroom`, and
// the defaults are 8 and 4 (`api_server.h:70-71`). So the shape #2712 called
// "the one that would make it live" is reachable on the default configuration,
// not behind a flag.
//
// This case measures that rather than arguing it: four real clients on four
// threads, over a real socket, through the registered route and the production
// mapping `vllm::openai::SynthesizeSpeechRequest`. The engine records how deep
// the seam ever got. The test_speech_engine sibling holds the same contract at
// the seam and is the one the sanitized lane builds; this one is here because a
// contract nothing production-side reaches is a class, not a capability.
namespace {

class RouteConcurrencyProbe final : public ::vllm::multimodal::SpeechEngine {
 public:
  std::string family() const override { return "concurrency-probe"; }
  int64_t sample_rate() const override { return 22050; }
  bool requires_reference_audio() const override { return false; }

  ::vllm::multimodal::SpeechResult SynthesizeLocked(
      const ::vllm::multimodal::SpeechGenParams&) override {
    const int depth = depth_.fetch_add(1) + 1;
    int seen = max_depth_.load();
    while (depth > seen && !max_depth_.compare_exchange_weak(seen, depth)) {
    }
    // Long enough that four requests dispatched together overlap when nothing
    // serialises them, and short enough that four serialised runs finish well
    // inside the client's 15 s read timeout. A real synthesis is minutes.
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    depth_.fetch_sub(1);
    calls_.fetch_add(1);
    ::vllm::multimodal::SpeechResult r;
    r.sample_rate = 22050;
    r.channels = 1;
    r.samples.assign(64, 0.25F);
    return r;
  }

  int max_depth() const { return max_depth_.load(); }
  int calls() const { return calls_.load(); }

 private:
  std::atomic<int> depth_{0};
  std::atomic<int> max_depth_{0};
  std::atomic<int> calls_{0};
};

}  // namespace

TEST_CASE("api_server: concurrent /v1/audio/speech requests do NOT enter one engine at once") {
  constexpr int kClients = 4;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  RouteConcurrencyProbe engine;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "concurrency-probe";
  caps.sample_rate = 22050;
  caps.channels = 1;
  caps.requires_reference_audio = false;
  // The PRODUCTION wiring, byte for byte what `server_main.cpp:1717-1722` and
  // `:1014-1019` install: the route's lambda calls the library mapping, which
  // calls the seam. Nothing here stands in for a step of that chain.
  h.server.set_synthesizer(
      [&engine](const vllm::openai::SpeechRequest& req) {
        return vllm::openai::SynthesizeSpeechRequest(engine, req);
      },
      caps);

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  std::atomic<int> ready{0};
  std::atomic<int> served{0};
  std::vector<std::thread> clients;
  clients.reserve(kClients);
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([port, &ready, &served] {
      httplib::Client client("127.0.0.1", port);
      client.set_connection_timeout(5, 0);
      client.set_read_timeout(15, 0);
      // Released together, so an overlap is a property of the server and not of
      // how fast each thread happened to start.
      ready.fetch_add(1);
      while (ready.load() < kClients) std::this_thread::yield();
      auto res = client.Post("/v1/audio/speech", MusicBody(), "application/json");
      if (res && res->status == 200 && res->body.compare(0, 4, "RIFF") == 0) {
        served.fetch_add(1);
      }
    });
  }
  for (std::thread& t : clients) t.join();
  server_thread.join();  // stops the server, then joins

  // EVERY request is still answered. Serialising must queue the second caller,
  // never refuse it: no upstream behaviour turns a concurrent speech request
  // into a 429, and a test that accepted fewer than four answers would be
  // passing on a regression.
  CHECK(served.load() == kClients);
  CHECK(engine.calls() == kClients);
  // THE ASSERTION. One is what the seam's own contract says; anything above it
  // is two cpp-httplib workers inside one engine's staged weights and one
  // `vt::Queue`.
  CHECK(engine.max_depth() == 1);
}

// The REQUEST-KEY contract, entered through the production route rather than
// through `ParseSpeechRequest`. `tests/vllm/entrypoints/openai/test_speech_api.cpp`
// localizes each refusal; this case proves the registered endpoint reaches them,
// which is the half a parser test cannot see — the parse call at
// `api_server.cpp:504` is one line, and deleting it leaves every parser test
// green while `POST /v1/audio/speech` honours nothing.
TEST_CASE("api_server: /v1/audio/speech refuses the dropped keys AT THE ROUTE") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::openai::SpeechRequest seen;
  int64_t calls = 0;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  caps.requires_reference_audio = false;
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest& req) {
        seen = req;
        ++calls;
        return StereoWav(8);
      },
      caps);

  // THE COST CASE OF #925, as the route sees it: `audio_duration` where OpenAI
  // puts it, beside an `extra_params` object. The duration used to be dropped
  // here and the family answered with 60 s of music behind a 200.
  ApiServer::DispatchResult split = h.server.handle_audio_speech(
      R"({"lyrics":"[Verse]\nx\n","extra_params":{"seed":7},"audio_duration":0.1})");
  CHECK(split.status == 200);
  CHECK(calls == 1);
  CHECK(seen.audio_duration_s == doctest::Approx(0.1));
  CHECK(seen.seed == 7);

  // Each newly refused key is a 400 whose body NAMES the key and what to send
  // instead, and NOTHING is synthesized.
  calls = 0;
  struct Case {
    const char* body;
    const char* names;
  };
  const Case cases[] = {
      {R"({"input":"hi","speaker":"alloy"})", "`speaker` is SGLang-Omni's alias for `voice`"},
      {R"({"lyrics":"x","instructions":"Genre: lo-fi"})",
       "`instructions` is SGLang-Omni's spelling"},
      {R"({"input":"hi","ref_audio":"/tmp/v.wav"})", "`ref_audio` is SGLang-Omni's spelling"},
      {R"({"input":"hi","ref_text":"hello"})",
       "`ref_text` (the transcript of a reference clip) is not supported"},
      {R"({"lyrics":"x","token_count":250})", "`token_count` is SGLang-Omni's LENGTH"},
      {R"({"lyrics":"x","duration_tokens":250})", "`duration_tokens` is SGLang-Omni's LENGTH"},
      {R"({"input":"hi","task_type":"CustomVoice"})",
       "`task_type` (`Base`/`CustomVoice`/`VoiceDesign`) is not supported"},
      {R"({"input":"hi","x_vector_only_mode":true})", "`x_vector_only_mode` is not supported"},
      {R"({"input":"hi","initial_codec_chunk_frames":4})",
       "`initial_codec_chunk_frames` is not supported"},
      // The guards that already existed, now seen through an `extra_params`
      // body, which is where they had stopped firing.
      {R"({"lyrics":"x","extra_params":{"seed":7},"audio_duration_s":0.1})",
       "`audio_duration_s` is not a request key"},
      {R"({"lyrics":"x","extra_params":{"seed":7},"temperature":0.7})",
       "`temperature` is not supported"},
      {R"({"lyrics":"x","extra_params":{"voice":"alloy"}})", "`voice` is not supported"},
  };
  for (const Case& one : cases) {
    CAPTURE(one.body);
    ApiServer::DispatchResult r = h.server.handle_audio_speech(one.body);
    CHECK(r.status == 400);
    CHECK(r.content_type == "application/json");
    CHECK(r.body.find(one.names) != std::string::npos);
    CHECK(r.body.find("RIFF") == std::string::npos);
  }
  // NOT ONE of the twelve reached the family.
  CHECK(calls == 0);
}

TEST_CASE("api_server: /v1/audio/speech honours the request keys OVER A REAL SOCKET") {
  // The route table plus the handler plus the parser, entered the way a user
  // enters them. #925 was a 42-minute job that a caller could not tell from the
  // one it asked for, so the claim worth pinning here is a claim about an HTTP
  // request and not about a function.
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::openai::SpeechRequest seen;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  caps.requires_reference_audio = false;
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest& req) {
        seen = req;
        return StereoWav(8);
      },
      caps);

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());
  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(15, 0);

  {
    auto res = client.Post(
        "/v1/audio/speech",
        R"({"lyrics":"[Verse]\nx\n","extra_params":{"seed":7},"audio_duration":0.1})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    // THE ASSERTION THE 42-MINUTE JOB NEEDED: the duration on the wire is the
    // duration the family was asked for, not the family's default.
    CHECK(seen.audio_duration_s == doctest::Approx(0.1));
    CHECK(seen.seed == 7);
  }
  {
    auto res = client.Post("/v1/audio/speech", R"({"lyrics":"x","instructions":"Genre: lo-fi"})",
                           "application/json");
    REQUIRE(res);
    CHECK(res->status == 400);
    CHECK(res->body.find("`instructions` is SGLang-Omni's spelling") != std::string::npos);
    CHECK(res->body.find("`description`") != std::string::npos);
  }
  server_thread.join();  // stops the server, then joins
}

// The CONTENT keys, at the ROUTE (#1336). The parser cases in
// `tests/vllm/entrypoints/openai/test_speech_api.cpp` localize each one; this
// case is the half they cannot see, because what a nested content key costs is
// paid downstream of the parser: the family never receives it, and the family
// is where two of the three refusals live.
TEST_CASE("api_server: /v1/audio/speech reads the CONTENT keys from both placements") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::openai::SpeechRequest seen;
  int64_t calls = 0;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  caps.requires_reference_audio = false;
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest& req) {
        seen = req;
        ++calls;
        return StereoWav(8);
      },
      caps);

  // A nested `text` REACHES the family. That matters because Music3 refuses
  // `text` by name (`minimax_music3_speech.cpp:440-446`) and its refusal ends
  // with the words "rather than having it silently dropped": before this
  // repair, nesting the key produced precisely that drop, with a 200 over it.
  // The harness synthesizer stands in for the family, so what is asserted here
  // is that the string ARRIVES, which is what the real refusal needs.
  ApiServer::DispatchResult r =
      h.server.handle_audio_speech(R"({"lyrics":"x","extra_params":{"text":"hello"}})");
  CHECK(r.status == 200);
  CHECK(calls == 1);
  CHECK(seen.text == "hello");

  // A nested `language` reaches the family too, where
  // `minimax_music3_speech.cpp:456-460` refuses it by name.
  r = h.server.handle_audio_speech(R"({"lyrics":"x","extra_params":{"language":"en"}})");
  CHECK(r.status == 200);
  CHECK(seen.language == "en");

  // The rest of the content set, over the route.
  r = h.server.handle_audio_speech(
      R"({"extra_params":{"lyrics":"[Verse]\nx\n","description":"pop","model":"minimax-music3"}})");
  CHECK(r.status == 200);
  CHECK(seen.lyrics == "[Verse]\nx\n");
  CHECK(seen.description == "pop");
  CHECK(seen.model == "minimax-music3");

  // PRECEDENCE is the one behaviour a caller can OBSERVE, and until now it was
  // pinned by the parser suite alone. `extra_params` wins at the route as well.
  r = h.server.handle_audio_speech(
      R"({"lyrics":"top","audio_duration":9.0,"extra_params":{"lyrics":"nested","audio_duration":0.5}})");
  CHECK(r.status == 200);
  CHECK(seen.lyrics == "nested");
  CHECK(seen.audio_duration_s == doctest::Approx(0.5));

  // The #925 boundary at the route: an unknown key in either placement is a
  // 200, not a 400. A strict whitelist would red these two.
  calls = 0;
  CHECK(h.server.handle_audio_speech(R"({"lyrics":"x","some_future_knob":1})").status == 200);
  CHECK(h.server.handle_audio_speech(R"({"lyrics":"x","extra_params":{"some_future_knob":1}})")
            .status == 200);
  CHECK(calls == 2);
}

// The SECOND-ORDER cost of the nested `reference_audio` drop, which is the one
// a caller could not have diagnosed. For a family whose
// `requires_reference_audio()` is true, `api_server.cpp:511-516` refuses BEFORE
// staging with "`reference_audio` ... is required". A nested clip was dropped
// by the parser, so the server answered that 400 to a caller who had supplied
// the clip, naming the field they had just sent.
TEST_CASE("api_server: a nested `reference_audio` satisfies a family that REQUIRES one") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  vllm::openai::SpeechRequest seen;
  int64_t calls = 0;
  vllm::openai::SpeechCapabilities caps;
  caps.family = "indextts2";
  caps.sample_rate = 22050;
  caps.channels = 1;
  caps.requires_reference_audio = true;  // IndexTTS-2.5 has no text-only synthesis
  h.server.set_synthesizer(
      [&](const vllm::openai::SpeechRequest& req) {
        seen = req;
        ++calls;
        return StereoWav(8);
      },
      caps);

  const std::string uri = MonoWavDataUri({16384, -16384}, 22050);

  // The control: with no clip at all the route still refuses, so this case
  // cannot pass by the guard having been removed.
  ApiServer::DispatchResult none = h.server.handle_audio_speech(R"({"input":"hi"})");
  CHECK(none.status == 400);
  CHECK(none.body.find("is required") != std::string::npos);
  CHECK(calls == 0);

  // TOP LEVEL: accepted before this repair and after it.
  ApiServer::DispatchResult flat = h.server.handle_audio_speech(
      R"({"input":"hi","reference_audio":")" + uri + R"("})");
  CHECK(flat.status == 200);
  CHECK(seen.reference_audio.size() == 2);

  // NESTED: a 400 naming the very field the caller sent, until now.
  ApiServer::DispatchResult nested = h.server.handle_audio_speech(
      R"({"input":"hi","extra_params":{"reference_audio":")" + uri + R"("}})");
  CHECK(nested.status == 200);
  CHECK(nested.body.find("is required") == std::string::npos);
  REQUIRE(seen.reference_audio.size() == 2);
  CHECK(seen.reference_sample_rate == 22050);
  CHECK(seen.reference_audio[0] == doctest::Approx(0.5));
  CHECK(seen.reference_audio[1] == doctest::Approx(-0.5));
  CHECK(calls == 2);
}

// ─── The SPEECH-ONLY server (#672) ──────────────────────────────────────────
//
// `vllm-server --speech-model <dir>` with NO `--model` is now a valid
// invocation, because upstream's own recipe is `sgl-omni serve --model
// MiniMaxAI/MiniMax-Music3` and nothing else — a music model is not an
// accessory to a text model. Requiring a text checkpoint beside a 28.5 GB music
// one made the documented recipe unrunnable on any box whose smallest text
// model is tens of gigabytes.
//
// This case pins the ROUTE TABLE that invocation produces, which is the half a
// handler-dispatch test cannot see. It is the exact twin of the ASR pair above
// ("transcriptions socket smoke … generate routes 404" / "the audio routes do
// not exist on a TEXT server"), and it uses the SAME construction the server's
// speech-only branch does: an `ApiServer` built from serving-models alone, with
// no completion and no chat handler, plus a synthesizer.
//
// It needs no checkpoint and no engine, so it runs in CI unconditionally.
TEST_CASE("api_server: a SPEECH-ONLY server serves speech and 404s the generate routes") {
  OpenAIServingModels models{"minimax-music3"};
  ApiServer server{models, "speech-only"};
  vllm::openai::SpeechCapabilities caps;
  caps.family = "minimax-music3";
  caps.sample_rate = 44100;
  caps.channels = 2;
  caps.requires_reference_audio = false;
  server.set_synthesizer([](const vllm::openai::SpeechRequest&) { return StereoWav(1024); },
                         caps);

  const int port = server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(server);
  for (int i = 0; i < 500 && !server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(server.is_running());

  {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(15, 0);

    // The route this server exists to serve.
    auto speech = client.Post("/v1/audio/speech", MusicBody(), "application/json");
    REQUIRE(speech);
    CHECK(speech->status == 200);
    CHECK(speech->get_header_value("Content-Type") == "audio/wav");
    REQUIRE(speech->body.size() == 44u + 1024u * 2u * 2u);
    CHECK(speech->body.compare(0, 4, "RIFF") == 0);
    CHECK(WavU16(speech->body, 22) == 2);       // STEREO
    CHECK(WavU32(speech->body, 24) == 44100u);  // the family's NATIVE rate

    // The TEXT routes are ABSENT, not present-and-broken. A well-formed body —
    // exactly what the route would accept if it existed — must fall through to
    // httplib's own 404, which is what proves the route was never registered
    // rather than that a handler rejected the payload.
    auto completions = client.Post("/v1/completions",
                                   R"({"model":"minimax-music3","prompt":"hi"})",
                                   "application/json");
    REQUIRE(completions);
    CHECK(completions->status == 404);
    auto chat = client.Post(
        "/v1/chat/completions",
        R"({"model":"minimax-music3","messages":[{"role":"user","content":"hi"}]})",
        "application/json");
    REQUIRE(chat);
    CHECK(chat->status == 404);
    // Not OUR 404 either: no error envelope is emitted, so nothing tells a
    // client the route half-exists.
    CHECK(completions->body.find("\"object\"") == std::string::npos);
    CHECK(chat->body.find("\"object\"") == std::string::npos);

    // The OTHER opt-in surfaces stay unregistered too — attaching a synthesizer
    // adds ONE endpoint, not a family of them.
    auto videos = client.Post("/v1/videos", R"({"prompt":"x"})", "application/json");
    REQUIRE(videos);
    CHECK(videos->status == 404);
    auto embeddings =
        client.Post("/v1/embeddings", R"({"input":"x"})", "application/json");
    REQUIRE(embeddings);
    CHECK(embeddings->status == 404);

    // Liveness and discovery still serve, so the 404s above are about the
    // routes and not about a dead server — and `/v1/models` reports the FAMILY,
    // which is what the speech-only branch defaults the served name to.
    auto health = client.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
    auto models_res = client.Get("/v1/models");
    REQUIRE(models_res);
    CHECK(models_res->status == 200);
    CHECK(json::parse(models_res->body).at("data").at(0).at("id") == "minimax-music3");
  }

  server_thread.join();  // stops the server, then joins
}

// ─── SAMPLE-PROMPT-LOGPROBS W2: the distribution reaches a client ────────────
//
// Ported from tests/entrypoints/openai/completion/test_completion.py:281-308
// (test_prompt_logprobs_completion) and :78 (`prompt_logprobs is None` when the
// field was not asked for) @ 555967922, adapted to a real socket against this
// file's synthetic engine instead of an `openai.AsyncOpenAI` client.
//
// UNAVOIDABLE HARNESS ADAPTATION, recorded: upstream's `-1` arm expects a
// BadRequestError, because `-1` widens to the vocabulary size and upstream's
// `--max-logprobs` defaults to 20 (sampling_params.py:806-814). Our ModelConfig
// carries no separate max_logprobs and the cap IS the vocabulary size
// (src/vllm/v1/engine/input_processor.cpp:143), so `-1` is served here rather
// than refused. That is the pre-existing, recorded deviation of the cap, not of
// this row; the `-2` arm below keeps upstream's refusal shape.
//
// This is the instrument #1775 asked for: `prompt_logprobs` teacher-forces the
// model along the request's OWN ids and returns the distribution at each prompt
// position, which is exactly the comparison the oracle side already ran.
TEST_CASE("api_server: prompt_logprobs reaches a real HTTP client") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(30, 0);

  // test_completion.py:78 — not asked for, so the field is present and null.
  SUBCASE("not requested → an explicit null, on both endpoints") {
    auto res = client.Post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":3,"temperature":0.0})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const json j = json::parse(res->body);
    REQUIRE(j.at("choices").at(0).contains("prompt_logprobs"));
    CHECK(j.at("choices").at(0).at("prompt_logprobs").is_null());

    auto chat = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":3,"temperature":0.0})",
        "application/json");
    REQUIRE(chat);
    REQUIRE(chat->status == 200);
    const json cj = json::parse(chat->body);
    REQUIRE(cj.contains("prompt_logprobs"));
    CHECK(cj.at("prompt_logprobs").is_null());
  }

  // test_completion.py:300-305 — requested, so every choice carries a non-empty
  // list. The VALUE assertions below are the point: a shape-only check passes on
  // an array of zeros, and zeros are exactly what a dropped payload looks like.
  SUBCASE("requested → a real per-position distribution") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":3,"temperature":0.0,"prompt_logprobs":4})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const json j = json::parse(res->body);
    const json& plp = j.at("choices").at(0).at("prompt_logprobs");
    REQUIRE(plp.is_array());
    CHECK(plp.size() > 0);
    // create_prompt_logprobs (v1/engine/logprobs.py:162-167): the first prompt
    // token has no predecessor, so its entry is None.
    CHECK(plp.at(0).is_null());
    REQUIRE(plp.size() >= 2);

    int positions_checked = 0;
    for (std::size_t i = 1; i < plp.size(); ++i) {
      const json& pos = plp.at(i);
      REQUIRE(pos.is_object());
      REQUIRE(pos.size() > 0);
      double mass = 0.0;
      double best = -1e30;
      int best_rank_seen = -1;
      bool saw_rank_one = false;
      for (auto it = pos.begin(); it != pos.end(); ++it) {
        // The key is a decimal token id.
        CHECK(it.key().find_first_not_of("-0123456789") == std::string::npos);
        const double lp = it.value().at("logprob").get<double>();
        // A log-probability is <= 0 and finite here (nothing is clamped in this
        // fixture), and it is NOT the 0.0 that a default-constructed payload has.
        CHECK(std::isfinite(lp));
        CHECK(lp <= 0.0);
        mass += std::exp(lp);
        if (lp > best) best = lp;
        const int rank = it.value().at("rank").get<int>();
        CHECK(rank >= 1);
        if (rank == 1) {
          saw_rank_one = true;
          best_rank_seen = rank;
        }
        CHECK(it.value().contains("decoded_token"));
      }
      // The entries are a SUBSET of one log_softmax distribution, so their
      // probabilities sum to at most 1 (up to float error). An array of zeros
      // would sum to `pos.size()` and fail here; an array of equal constants
      // would fail the strict-max check below.
      CHECK(mass <= 1.0 + 1e-3);
      CHECK(mass > 0.0);
      // rank 1 exists and its logprob IS the maximum of this position.
      CHECK(saw_rank_one);
      CHECK(best_rank_seen == 1);
      for (auto it = pos.begin(); it != pos.end(); ++it) {
        if (it.value().at("rank").get<int>() == 1) {
          CHECK(it.value().at("logprob").get<double>() == doctest::Approx(best));
        }
      }
      ++positions_checked;
    }
    CHECK(positions_checked >= 1);

    // test_completion.py:303-305 asserts `choices[1].prompt_logprobs` too. It is
    // ported here now that AsyncLLM — the engine this server runs on — fans an
    // n>1 request out into n children (#1816); before that it returned ONE
    // choice and this arm could only be gated over the SYNC LLMEngine, in
    // tests/vllm/entrypoints/openai/test_serving.cpp.
    auto fanned = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":3,"temperature":1.0,"top_k":1,)"
        R"("seed":7,"n":2,"prompt_logprobs":4})",
        "application/json");
    REQUIRE(fanned);
    REQUIRE(fanned->status == 200);
    const json fj = json::parse(fanned->body);
    REQUIRE(fj.at("choices").size() == 2);
    // completion/serving.py:588 sets it INSIDE the per-output loop, so the one
    // prompt payload repeats on every choice rather than riding only the first.
    CHECK(fj.at("choices").at(1).at("prompt_logprobs").is_array());
    CHECK(fj.at("choices").at(1).at("prompt_logprobs") ==
          fj.at("choices").at(0).at("prompt_logprobs"));
  }

  // chat_completion/serving.py:1070 — the chat payload is a TOP-LEVEL response
  // field, not a per-choice one.
  SUBCASE("chat: the payload is top-level on the response") {
    auto res = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":3,"temperature":0.0,"prompt_logprobs":2})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const json j = json::parse(res->body);
    REQUIRE(j.at("prompt_logprobs").is_array());
    CHECK(j.at("prompt_logprobs").size() > 0);
    CHECK(j.at("prompt_logprobs").at(0).is_null());
    // Not smuggled onto the choice — upstream's chat choice has no such field.
    CHECK_FALSE(j.at("choices").at(0).contains("prompt_logprobs"));
  }

  // completion/protocol.py:483-494 — the two refusals, as the 400 a client sees.
  SUBCASE("the request validators answer 400 over the socket") {
    auto streamed = client.Post(
        "/v1/completions",
        R"({"prompt":"hello","max_tokens":2,"prompt_logprobs":2,"stream":true})",
        "application/json");
    REQUIRE(streamed);
    CHECK(streamed->status == 400);
    CHECK(json::parse(streamed->body).at("error").at("message")
              .get<std::string>()
              .find("`prompt_logprobs` are not available when `stream=True`.") !=
          std::string::npos);

    auto negative = client.Post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":2,"prompt_logprobs":-2})",
        "application/json");
    REQUIRE(negative);
    CHECK(negative->status == 400);
    CHECK(json::parse(negative->body).at("error").at("message")
              .get<std::string>()
              .find("`prompt_logprobs` must be a positive value or -1.") !=
          std::string::npos);

    auto non_numeric = client.Post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":2,"prompt_logprobs":"2"})",
        "application/json");
    REQUIRE(non_numeric);
    CHECK(non_numeric->status == 400);
    CHECK(json::parse(non_numeric->body).at("error").at("message")
              .get<std::string>()
              .find("`prompt_logprobs` must be an integer.") != std::string::npos);
  }

  server_thread.join();  // stops the server, then joins
}

// ─── SAMPLE-N-ASYNC (#1816): the SERVED engine fans an n>1 request out ───────
//
// Ported from tests/entrypoints/openai/completion/test_completion.py:349
// (test_parallel_no_streaming) and :398 (test_parallel_streaming) @ 5559679229.
//
// The best_of subcases below have NO upstream source. `best_of` is not a field
// on upstream's CompletionRequest — 0.26 dropped it from the live path, and
// `grep -rn best_of tests/entrypoints/` at the pin returns nothing. It is a
// declared LOCAL extension implementing the classic OpenAI / vLLM-V0 contract
// (protocol.h:201-211), so those subcases gate our own semantics against our
// own deterministic fan-out. test_batch_completions:616 is NOT that source:
// its n=2 arm (:629-651) requires use_beam_search=True and never sets best_of.
//
// These run over a REAL SOCKET against ApiServer's registered /v1/completions
// route, which is the production entry point, and therefore over AsyncLLM —
// the engine every HTTP route runs on. The equivalent property is gated for the
// SYNC LLMEngine in test_serving.cpp ("serving_completion: n>1 returns n
// indexed, deterministic choices"); that engine is not the served one, which is
// exactly how #1816 stayed invisible.
//
// The fixture is deterministic by construction (temperature 1.0 + top_k 1
// collapses to the argmax, as test_serving.cpp's sync case does), so upstream's
// "n unique completions" and "subrequests finished at different times"
// assertions do not port — this tiny model has one argmax path. What ports is
// the part #1816 breaks: the COUNT, the per-choice index, a finish reason on
// every choice, and usage that sums the children while counting the prompt once.
TEST_CASE("api_server: an n>1 completion is fanned out over a real socket") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(30, 0);

  constexpr int kN = 3;        // upstream's n
  constexpr int kMaxTokens = 4;

  // test_completion.py:370-378 — n choices, choice.index == idx, and a finish
  // reason on every one of them.
  SUBCASE("non-streaming: n indexed choices, and usage sums the children") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":3,"temperature":1.0,)"
        R"("top_k":1,"seed":7})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const json j = json::parse(res->body);
    const json& choices = j.at("choices");
    REQUIRE(choices.is_array());
    // THE assertion #1816 names: the served engine answered with ONE choice.
    REQUIRE(choices.size() == static_cast<std::size_t>(kN));
    for (std::size_t i = 0; i < choices.size(); ++i) {
      CHECK(choices.at(i).at("index").get<int>() == static_cast<int>(i));
      REQUIRE(choices.at(i).at("finish_reason").is_string());
      CHECK(choices.at(i).at("finish_reason").get<std::string>() == "length");
      CHECK_FALSE(choices.at(i).at("text").get<std::string>().empty());
    }
    // completion/serving.py:599 (`num_generated_tokens += len(output.token_ids)`,
    // INSIDE the per-output loop) and :601 (`num_prompt_tokens +=
    // len(prompt_token_ids)`, OUTSIDE it) — the completions sum over the n
    // children while the prompt is counted once per prompt. A one-child answer
    // reads kMaxTokens here.
    const json& usage = j.at("usage");
    CHECK(usage.at("completion_tokens").get<int>() == kN * kMaxTokens);
    const int prompt_tokens = usage.at("prompt_tokens").get<int>();
    CHECK(prompt_tokens > 0);
    CHECK(usage.at("total_tokens").get<int>() ==
          prompt_tokens + kN * kMaxTokens);
  }

  // test_completion.py:419-424 — the samples are FLATTENED into one stream:
  // every chunk carries exactly one choice, tagged with the index it belongs
  // to, and the stream carries n finish reasons in total.
  SUBCASE("streaming: one choice per chunk, n finish reasons, indices 0..n-1") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":3,"temperature":1.0,)"
        R"("top_k":1,"seed":7,"stream":true})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const std::string& body = res->body;

    int finish_reason_count = 0;
    std::vector<int> tokens_seen(kN, 0);
    std::size_t pos = 0;
    bool saw_done = false;
    while ((pos = body.find("data: ", pos)) != std::string::npos) {
      const std::size_t end = body.find("\n\n", pos);
      REQUIRE(end != std::string::npos);
      const std::string payload = body.substr(pos + 6, end - (pos + 6));
      pos = end + 2;
      if (payload == "[DONE]") {
        saw_done = true;
        continue;
      }
      const json frame = json::parse(payload);
      if (!frame.contains("choices") || frame.at("choices").empty()) continue;
      // Upstream reads chunk.choices[0] per chunk: one choice per frame.
      REQUIRE(frame.at("choices").size() == 1);
      const json& choice = frame.at("choices").at(0);
      const int index = choice.at("index").get<int>();
      REQUIRE(index >= 0);
      REQUIRE(index < kN);
      ++tokens_seen[static_cast<std::size_t>(index)];
      if (choice.at("finish_reason").is_string()) ++finish_reason_count;
    }
    CHECK(saw_done);
    // test_completion.py:425-428 — n completions, each with a finish reason.
    CHECK(finish_reason_count == kN);
    for (int i = 0; i < kN; ++i) {
      CHECK(tokens_seen[static_cast<std::size_t>(i)] > 0);
    }
  }

  // OURS, not upstream's (see the header note): best_of asks the engine for
  // best_of children and the serving layer ranks them down to n. With no
  // fan-out there is nothing to rank, so this reads one choice.
  SUBCASE("best_of > n returns exactly n ranked choices") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":2,"best_of":4,)"
        R"("temperature":1.0,"top_k":1,"seed":7})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const json j = json::parse(res->body);
    REQUIRE(j.at("choices").size() == 2);
    CHECK(j.at("choices").at(0).at("index").get<int>() == 0);
    CHECK(j.at("choices").at(1).at("index").get<int>() == 1);
  }

  // ── best_of != n + stream DOWNGRADES to the non-streamed body ────────────
  // Ported from serving_completion.py:253-260 @ `56e96b37e4^` (the parent of
  // vllm#29090, which removed `best_of`; the pin's tree cannot carry the
  // anchor) plus completion/serving.py:269-278 @ `5559679229` for the delivery
  // shape, which the pin DOES still carry.
  //
  // Upstream computes `stream = request.stream and (best_of is None or n ==
  // best_of) and not use_beam_search` and, when that is false, runs the
  // NON-streaming aggregator and wraps its complete response in ONE SSE frame
  // followed by `[DONE]`. So the client that asked to stream gets
  // text/event-stream back, but the payload is a whole CompletionResponse with
  // the ranked top-n choices — never a partial delta, and never `best_of`
  // choices.
  //
  // This replaces a 400. The 400 read `{"n":1,"best_of":4,"stream":true}` —
  // the canonical OpenAI `best_of` call — as a client error, which upstream
  // has never done: the refusal at completion/serving.py:136-139 arrived in
  // `65a4da1504` (vllm#36160) four months after best_of was gone, so it is not
  // precedent for this field.
  //
  // Reads the single frame and asserts it is the aggregated response, not a
  // stream of deltas.
  const auto downgraded_frames = [](const std::string& body) {
    std::vector<json> frames;
    bool saw_done = false;
    std::size_t pos = 0;
    while ((pos = body.find("data: ", pos)) != std::string::npos) {
      const std::size_t end = body.find("\n\n", pos);
      REQUIRE(end != std::string::npos);
      const std::string payload = body.substr(pos + 6, end - (pos + 6));
      pos = end + 2;
      if (payload == "[DONE]") {
        saw_done = true;
        continue;
      }
      frames.push_back(json::parse(payload));
    }
    CHECK(saw_done);
    return frames;
  };

  SUBCASE("best_of > n with stream:true downgrades to n ranked choices") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":2,"best_of":4,)"
        R"("temperature":1.0,"top_k":1,"seed":7,"stream":true})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    // The TRANSPORT is still SSE. This is half of what the downgrade promises
    // and the half no other assertion here can see: the client asked for
    // `text/event-stream`, and completion/serving.py:269-278 answers a
    // downgraded request on that transport rather than switching to JSON.
    // serving_completion.cpp says an `application/json` body would break the
    // OpenAI SDKs that unconditionally parse the stream; this is that claim,
    // measured.
    CHECK(res->get_header_value("Content-Type") == "text/event-stream");
    const std::vector<json> frames = downgraded_frames(res->body);
    // ONE payload frame: the aggregated response, not a delta stream. A
    // streamed answer to this body emitted one frame PER TOKEN PER CHILD.
    REQUIRE(frames.size() == 1);
    const json& r = frames.at(0);
    // The complete non-streamed body: `text_completion`, usage, and exactly
    // the n ranked choices — NOT the best_of=4 the engine generated.
    CHECK(r.at("object").get<std::string>() == "text_completion");
    REQUIRE(r.at("choices").size() == 2);
    CHECK(r.at("choices").at(0).at("index").get<int>() == 0);
    CHECK(r.at("choices").at(1).at("index").get<int>() == 1);
    CHECK(r.contains("usage"));
    // A downgraded request still delivered whole text, not one delta: the
    // aggregator ran, so each choice carries its full continuation.
    for (const json& choice : r.at("choices")) {
      CHECK(choice.at("finish_reason").is_string());
    }
  }

  // The canonical OpenAI call: n defaults to 1, best_of asks for 4 candidates
  // and the endpoint returns the single best. Under the removed 400 this — the
  // most common best_of body there is — answered 400. Before this row it
  // answered 200.
  SUBCASE("n=1 best_of=4 stream:true downgrades to exactly one choice") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":1,"best_of":4,)"
        R"("temperature":1.0,"top_k":1,"seed":7,"stream":true})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const std::vector<json> frames = downgraded_frames(res->body);
    REQUIRE(frames.size() == 1);
    const json& r = frames.at(0);
    CHECK(r.at("object").get<std::string>() == "text_completion");
    REQUIRE(r.at("choices").size() == 1);
    CHECK(r.at("choices").at(0).at("index").get<int>() == 0);
    CHECK(r.at("choices").at(0).at("finish_reason").is_string());
  }

  // The NARROW half of the mirrored condition: `best_of == n` has nothing to
  // rank, so it streams normally — many delta frames, not one aggregated body.
  // Upstream's expression is `best_of is None or n == best_of`, and this is the
  // second disjunct. (`best_of` UNSET is the streaming subcase above.)
  SUBCASE("best_of == n with stream:true still streams deltas") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":2,"best_of":2,)"
        R"("temperature":1.0,"top_k":1,"seed":7,"stream":true})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const std::vector<json> frames = downgraded_frames(res->body);
    // Delta frames, one choice each, and MORE than the single aggregated body
    // the downgrade produces. This is what proves the condition is narrow.
    CHECK(frames.size() > 1);
    for (const json& frame : frames) {
      if (!frame.contains("choices") || frame.at("choices").empty()) continue;
      CHECK(frame.at("object").get<std::string>() == "text_completion");
      CHECK(frame.at("choices").size() == 1);
    }
  }

  // NOT an abort test: it polls a request nobody cancelled, so it measures the
  // NATURAL drain. What it gates is that all n children retire — a fan-out that
  // registered a child the OutputProcessor never finishes would hang here. The
  // abort path is gated by the disconnect case below, which never lets the
  // children reach max_tokens.
  SUBCASE("every child of an n>1 request retires when the request finishes") {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello world","max_tokens":4,"n":3,"temperature":1.0,)"
        R"("top_k":1,"seed":7})",
        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    for (int i = 0; i < 500 && h.async_engine.get_num_unfinished_requests() > 0;
         ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(h.async_engine.get_num_unfinished_requests() == 0);
  }

  server_thread.join();  // stops the server, then joins
}

// SAMPLE-N-ASYNC (#1816): a client that goes away mid-stream must take EVERY
// child with it. The id the client's stream holds is now the PARENT id, which
// is not a key in the OutputProcessor's request table at all — only
// external_req_ids_ maps it onto the n internal children. Without that
// resolution the abort matches nothing and all n children run on to
// max_tokens, burning KV blocks for a socket that is gone.
//
// The assertion is taken SYNCHRONOUSLY, with no poll loop: AsyncLLM::abort
// erases the child states under the same mutex get_num_unfinished_requests()
// reads, so the count is exact the instant abort() returns. A poll loop would
// pass on natural drain and measure nothing, which is the trap the previous
// version of the case above fell into. max_tokens is set far above what the
// first frame consumes so the children are provably still live at that instant
// — the pre-abort REQUIRE fails loudly rather than passing vacuously if the
// engine ever outran it.
//
// Dispatch goes through ApiServer::handle_completions, which is the body the
// registered /v1/completions route calls (api_server.cpp:1172), and the
// disconnect is SseStream::abort — what the route's content provider calls when
// the socket write fails and what ~CompletionSseStream calls on teardown.
TEST_CASE("api_server: a disconnect on an n>1 stream aborts every child") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  constexpr int kN = 3;
  const std::string body =
      R"({"prompt":"hello","max_tokens":30,"n":3,"temperature":1.0,)"
      R"("top_k":1,"seed":7,"stream":true})";
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  REQUIRE(result.sse_stream != nullptr);

  std::string first;
  REQUIRE(result.sse_stream->next(first));
  // n children live, not one: the fan-out ran and none of them has finished.
  REQUIRE(h.async_engine.get_num_unfinished_requests() == kN);

  result.sse_stream->abort();  // the client disconnected
  CHECK(h.async_engine.get_num_unfinished_requests() == 0);

  // Idempotent, and destruction adds nothing (~CompletionSseStream aborts too).
  result.sse_stream->abort();
  CHECK(h.async_engine.get_num_unfinished_requests() == 0);
  result.sse_stream.reset();
  CHECK(h.async_engine.get_num_unfinished_requests() == 0);
}

// SAMPLE-N-ASYNC (#1816): /v1/chat/completions NON-streaming n>1 became live
// with the fan-out and had no test on any route. Ported from
// tests/entrypoints/openai/chat_completion/test_completion.py's n>1 arm shape:
// n choices, choice.index == idx, a finish reason on each, and usage that sums
// the children while counting the rendered prompt once.
//
// The chat STREAMING arm is NOT gated here because it is broken: ChatSseStream
// reads outputs.front() only, so an n>1 chat stream shows ONE choice while the
// engine runs n children. That is #2120, and this row makes it worse rather
// than leaving it alone — see the spec's `## Owed`.
TEST_CASE("api_server: an n>1 chat completion returns n choices over a socket") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  ScopedServerThread server_thread(h.server);
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(30, 0);

  constexpr int kN = 3;
  constexpr int kMaxTokens = 3;
  auto res = client.Post(
      "/v1/chat/completions",
      R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":3,)"
      R"("n":3,"temperature":1.0,"top_k":1,"seed":7})",
      "application/json");
  REQUIRE(res);
  REQUIRE(res->status == 200);
  const json j = json::parse(res->body);
  const json& choices = j.at("choices");
  REQUIRE(choices.is_array());
  REQUIRE(choices.size() == static_cast<std::size_t>(kN));
  for (std::size_t i = 0; i < choices.size(); ++i) {
    CHECK(choices.at(i).at("index").get<int>() == static_cast<int>(i));
    REQUIRE(choices.at(i).at("finish_reason").is_string());
    CHECK(choices.at(i).at("message").at("role") == "assistant");
  }
  const json& usage = j.at("usage");
  CHECK(usage.at("completion_tokens").get<int>() == kN * kMaxTokens);
  const int prompt_tokens = usage.at("prompt_tokens").get<int>();
  CHECK(prompt_tokens > 0);
  CHECK(usage.at("total_tokens").get<int>() == prompt_tokens + kN * kMaxTokens);

  // ── the chat route RANKS best_of non-streamed (serving_chat.cpp:1015-1025) ─
  // Same SelectBestOf path the completions route uses. Without this the trim
  // is unreachable from any test: deleting `trim_best_of` left every suite
  // green, so the chat arm of SAMPLE-BEST-OF was landing ungated.
  {
    auto bo = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":3,)"
        R"("n":2,"best_of":4,"temperature":1.0,"top_k":1,"seed":7})",
        "application/json");
    REQUIRE(bo);
    REQUIRE(bo->status == 200);
    const json bj = json::parse(bo->body);
    // Exactly n=2, NOT the best_of=4 the engine generated.
    REQUIRE(bj.at("choices").size() == 2);
    CHECK(bj.at("choices").at(0).at("index").get<int>() == 0);
    CHECK(bj.at("choices").at(1).at("index").get<int>() == 1);
  }

  // ── the chat route does NOT downgrade best_of + stream ────────────────────
  // This pins a DELIBERATE, upstream-sourced asymmetry with /v1/completions,
  // which does downgrade. At `56e96b37e4^` upstream's ChatCompletionRequest
  // carries best_of (protocol.py:568, class at :528) and passes it into
  // SamplingParams (:892), yet serving_chat.py:355 streams on the RAW
  // request.stream with no best_of or beam-search guard — only the completions
  // route (serving_completion.py:253-260) ever downgraded. Mirroring vLLM
  // means keeping that difference, so it is asserted rather than left to be
  // rediscovered as a bug.
  //
  // Asserts ONLY that the request streamed — i.e. that the completions route's
  // single aggregated frame did NOT happen here. It deliberately does not
  // assert the frames' choice indices, which #2120 currently collapses to 0;
  // this check stays true once #2120 is fixed.
  {
    auto st = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":3,)"
        R"("n":2,"best_of":4,"temperature":1.0,"top_k":1,"seed":7,)"
        R"("stream":true})",
        "application/json");
    REQUIRE(st);
    REQUIRE(st->status == 200);
    int data_frames = 0;
    bool saw_done = false;
    std::size_t pos = 0;
    const std::string& sbody = st->body;
    while ((pos = sbody.find("data: ", pos)) != std::string::npos) {
      const std::size_t end = sbody.find("\n\n", pos);
      REQUIRE(end != std::string::npos);
      const std::string payload = sbody.substr(pos + 6, end - (pos + 6));
      pos = end + 2;
      if (payload == "[DONE]") {
        saw_done = true;
        continue;
      }
      const json frame = json::parse(payload);
      // A chat DELTA frame, never the aggregated completions-style body.
      CHECK(frame.at("object").get<std::string>() == "chat.completion.chunk");
      ++data_frames;
    }
    CHECK(saw_done);
    // Streamed, not downgraded: the completions route answers the equivalent
    // body with exactly ONE frame carrying the whole response.
    CHECK(data_frames > 1);
  }

  server_thread.join();  // stops the server, then joins
}

// ─── /v1/ner + /v1/systemone (MODEL-GLINER25 Phase 4f) ──────────────────────
//
// NER dispatch over a MOCK NerFn callback: the callback wraps the ONE library
// seam (Gliner2NerInference) — the SAME path vllm_gliner_ner drives — so HTTP
// and FFI cannot drift. The mock returns deterministic entities so the JSON
// shape is asserted exactly. The "without ner" case is a 500, not a crash,
// mirroring the embedder pattern.

namespace {

struct NerHarness {
  vllm::entrypoints::openai::OpenAIServingModels models{"gliner2-fixture"};
  ApiServer server{models, "test-version"};

  NerHarness() {
    server.set_ner(
        [](const std::string& text, const std::vector<std::string>& labels,
           float threshold, int64_t max_width) {
          (void)threshold;
          (void)max_width;
          ApiServer::NerResult result;
          result.prompt_tokens = 7;
          if (text == "john works at google" &&
              std::find(labels.begin(), labels.end(), "person") !=
                  labels.end()) {
            result.entities.push_back(
                {"person", "john", 0, 4, 0.95F});
          }
          if (text == "john works at google" &&
              std::find(labels.begin(), labels.end(), "organization") !=
                  labels.end()) {
            result.entities.push_back(
                {"organization", "google", 15, 21, 0.88F});
          }
          return result;
        });
  }
};

}  // namespace

TEST_CASE("api_server: NER dispatch — OpenAI shape over the mock callback") {
  NerHarness h;

  // Valid request with two labels.
  ApiServer::DispatchResult r = h.server.handle_ner(
      R"({"model":"gliner2-fixture","text":"john works at google",)"
      R"("labels":["person","organization"],"threshold":0.5})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("object") == "ner");
  CHECK(j.at("model") == "gliner2-fixture");
  CHECK(std::string(j.at("id")).rfind("ner-", 0) == 0);
  REQUIRE(j.at("entities").size() == 2);
  // Entities are sorted by the mock in insertion order.
  CHECK(j.at("entities").at(0).at("label") == "person");
  CHECK(j.at("entities").at(0).at("text") == "john");
  CHECK(j.at("entities").at(0).at("start") == 0);
  CHECK(j.at("entities").at(0).at("end") == 4);
  CHECK(j.at("entities").at(0).at("confidence") == doctest::Approx(0.95F));
  CHECK(j.at("entities").at(1).at("label") == "organization");
  CHECK(j.at("entities").at(1).at("text") == "google");
  CHECK(j.at("usage").at("prompt_tokens").get<int64_t>() == 7);
  CHECK(j.at("usage").at("total_tokens") == 7);

  // Single label — one entity.
  r = h.server.handle_ner(
      R"({"text":"john works at google","labels":["person"]})");
  CHECK(r.status == 200);
  j = json::parse(r.body);
  REQUIRE(j.at("entities").size() == 1);
  CHECK(j.at("entities").at(0).at("label") == "person");

  // No match — zero entities is a 200 with an empty array.
  r = h.server.handle_ner(R"({"text":"nothing here","labels":["person"]})");
  CHECK(r.status == 200);
  j = json::parse(r.body);
  CHECK(j.at("entities").size() == 0);
}

TEST_CASE("api_server: NER malformed requests") {
  NerHarness h;
  CHECK(h.server.handle_ner("not json").status == 400);
  CHECK(h.server.handle_ner(R"({"model":"x"})").status == 404);
  CHECK(h.server.handle_ner(R"({"text":"x"})").status == 400);  // no labels
  CHECK(h.server.handle_ner(R"({"labels":["x"]})").status == 400);  // no text
  CHECK(h.server.handle_ner(R"({"text":42,"labels":["x"]})").status == 400);
  CHECK(h.server.handle_ner(R"({"text":"x","labels":"notarray"})").status == 400);
  CHECK(h.server.handle_ner(R"({"text":"x","labels":[42]})").status == 400);
  CHECK(h.server.handle_ner(R"({"text":"x","labels":[]})").status == 400);
}

TEST_CASE("api_server: NER without a ner callback is a 500, not a crash") {
  vllm::entrypoints::openai::OpenAIServingModels models{"m"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r =
      server.handle_ner(R"({"text":"x","labels":["y"]})");
  CHECK(r.status == 500);
  CHECK(json::parse(r.body).at("error").at("message") ==
        "The model does not support NER");
}

TEST_CASE("api_server: systemone dispatch — jev-compatible answers shape") {
  NerHarness h;

  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"john works at google",)"
      R"("questions":{"person":{"type":"noul","instructions":"person"},)"
      R"("organization":{"type":"noul","instructions":"organization"}}})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("model") == "gliner2-fixture");
  REQUIRE(j.at("answers").is_object());
  REQUIRE(j.at("answers").contains("person"));
  REQUIRE(j.at("answers").contains("organization"));
  CHECK(j.at("answers").at("person").at("type") == "noul");
  CHECK(j.at("answers").at("person").at("noul") == doctest::Approx(0.95F));
  REQUIRE(j.at("answers").at("person").at("entities").size() == 1);
  CHECK(j.at("answers").at("person").at("entities").at(0).at("text") ==
        "john");
  CHECK(j.at("answers").at("organization").at("noul") ==
        doctest::Approx(0.88F));
  CHECK(j.at("usage").at("input_tokens").get<int64_t>() == 7);
  CHECK(j.at("usage").at("output_tokens").get<int64_t>() == 0);

  // state as a non-string (JSON value) — stringified internally.
  r = h.server.handle_systemone(
      R"({"state":42,"questions":{"x":{"type":"noul","instructions":"thing"}}})");
  CHECK(r.status == 200);
}

TEST_CASE("api_server: systemone malformed requests") {
  NerHarness h;
  CHECK(h.server.handle_systemone("not json").status == 400);
  CHECK(h.server.handle_systemone(R"({})").status == 400);  // no state
  CHECK(h.server.handle_systemone(R"({"state":"x"})").status == 400);  // no questions
  CHECK(h.server.handle_systemone(
            R"({"state":"x","questions":"notobj"})").status == 400);
  CHECK(h.server.handle_systemone(R"({"state":"x","questions":{}})").status ==
        400);
}

TEST_CASE("api_server: systemone without a ner or decision callback is a 500") {
  vllm::entrypoints::openai::OpenAIServingModels models{"m"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r =
      server.handle_systemone(R"({"state":"x","questions":{"y":{"type":"noul"}}})");
  CHECK(r.status == 500);
  CHECK(json::parse(r.body).at("error").at("message") ==
        "The model does not support SystemOne");
}

// MODEL-NIMBLE: the request-level seam owns the whole body, wins over the
// per-question path, and maps its std::invalid_argument to a 400.
TEST_CASE("api_server: systemone request-level seam") {
  vllm::entrypoints::openai::OpenAIServingModels models{"nimble-served"};
  ApiServer server{models, "test-version"};
  std::string seen;
  server.set_systemone_request(
      [&seen](const nlohmann::ordered_json& body) -> nlohmann::ordered_json {
        seen = body.dump();
        if (!body.contains("questions")) {
          throw std::invalid_argument("questions is required");
        }
        nlohmann::ordered_json out;
        out["answers"] = {{"q", {{"type", "noul"}, {"noul", 0.25}}}};
        out["usage"] = {{"input_tokens", 11}, {"output_tokens", 0}};
        return out;
      });
  // A body the per-question parser would refuse (unknown question type)
  // reaches the seam untouched: the seam owns validation.
  const std::string body =
      R"({"state":{"k":[1,"<b>"]},"questions":{"q":{"type":"custom"}}})";
  ApiServer::DispatchResult r = server.handle_systemone(body);
  CHECK(r.status == 200);
  CHECK(seen == nlohmann::ordered_json::parse(body).dump());
  json j = json::parse(r.body);
  CHECK(j.at("model") == "nimble-served");
  CHECK(j.at("answers").at("q").at("noul") == doctest::Approx(0.25));
  CHECK(j.at("usage").at("input_tokens").get<int64_t>() == 11);
  // The request's own model name is echoed, as openjev does.
  r = server.handle_systemone(R"({"model":"nimble","state":"s","questions":{}})");
  CHECK(json::parse(r.body).at("model") == "nimble");
  // std::invalid_argument is the request's fault.
  r = server.handle_systemone(R"({"state":"s"})");
  CHECK(r.status == 400);
  CHECK(json::parse(r.body).at("error").at("message") == "questions is required");
  CHECK(server.handle_systemone("not json").status == 400);
}

// ── kev full-compatibility: choice, score, permute, separate ─────────────────

TEST_CASE("api_server: systemone choice question") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"john works at google",)"
      R"("questions":{"pick":{"type":"choice",)"
      R"("instructions":"entity type",)"
      R"("criteria":{"person":null,"organization":null}}}})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  json& ans = j.at("answers").at("pick");
  CHECK(ans.at("type") == "choice");
  // person (0.95) > organization (0.88) after softmax → choice = "person"
  CHECK(ans.at("choice") == "person");
  REQUIRE(ans.at("probabilities").is_object());
  CHECK(ans.at("probabilities").contains("person"));
  CHECK(ans.at("probabilities").contains("organization"));
  CHECK(ans.at("confidence").is_number());
  // latency_ms present (kev serve.py returns it)
  CHECK(j.contains("latency_ms"));
}

TEST_CASE("api_server: systemone score question") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"john works at google",)"
      R"("questions":{"rate":{"type":"score",)"
      R"("criteria":["person","organization","other"]}}})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  json& ans = j.at("answers").at("rate");
  CHECK(ans.at("type") == "score");
  CHECK(ans.at("score").is_number());
  REQUIRE(ans.at("legend").is_object());
  CHECK(ans.at("legend").contains("0"));
  CHECK(ans.at("legend").contains("1"));
  CHECK(ans.at("legend").contains("2"));
  REQUIRE(ans.at("probabilities").is_object());
  CHECK(ans.at("probabilities").contains("0"));
  CHECK(ans.at("probabilities").contains("1"));
  CHECK(ans.at("probabilities").contains("2"));
  CHECK(ans.at("confidence").is_number());
}

TEST_CASE("api_server: systemone separate — N passes, summed tokens") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone_separate(
      R"({"state":"john works at google",)"
      R"("questions":{"person":{"type":"noul","instructions":"person"},)"
      R"("organization":{"type":"noul","instructions":"organization"}}})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("answers").at("person").at("type") == "noul");
  CHECK(j.at("answers").at("person").at("noul") ==
        doctest::Approx(0.95F));
  CHECK(j.at("answers").at("organization").at("noul") ==
        doctest::Approx(0.88F));
  // Two questions, each costs 7 prompt tokens in the mock → 14 total.
  CHECK(j.at("usage").at("input_tokens").get<int64_t>() == 14);
  CHECK(j.contains("latency_ms"));
}

TEST_CASE("api_server: systemone permute — choice under n_perm orders") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone_permute(
      R"({"request":{"state":"john works at google",)"
      R"("questions":{"pick":{"type":"choice",)"
      R"("instructions":"entity type",)"
      R"("criteria":{"person":null,"organization":null}}}},)"
      R"("question":"pick","n_perm":3,"seed":42})");
  CHECK(r.status == 200);
  json j = json::parse(r.body);
  REQUIRE(j.at("runs").is_array());
  CHECK(j.at("runs").size() == 3);
  // Each run carries order, probabilities, choice, latency_ms.
  for (const auto& run : j.at("runs")) {
    REQUIRE(run.at("order").is_array());
    REQUIRE(run.at("probabilities").is_object());
    CHECK(run.at("choice").is_string());
    CHECK(run.contains("latency_ms"));
  }
  // The mock NER is order-invariant, so the argmax must be stable.
  CHECK(j.at("argmax_stable").get<bool>() == true);
  REQUIRE(j.at("spread").is_object());
  CHECK(j.at("spread").contains("person"));
  CHECK(j.at("spread").contains("organization"));
}

TEST_CASE("api_server: systemone permute rejects non-choice questions") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone_permute(
      R"({"request":{"state":"john works at google",)"
      R"("questions":{"q":{"type":"noul"}}},)"
      R"("question":"q"})");
  CHECK(r.status == 400);
}

TEST_CASE("api_server: systemone permute without a ner callback is a 500") {
  vllm::entrypoints::openai::OpenAIServingModels models{"m"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r = server.handle_systemone_permute(
      R"({"request":{"state":"x","questions":{"q":{"type":"choice",)"
      R"("instructions":"i","criteria":{"a":null,"b":null}}}},"question":"q"})");
  CHECK(r.status == 500);
}

TEST_CASE("api_server: systemone separate without a ner callback is a 500") {
  vllm::entrypoints::openai::OpenAIServingModels models{"m"};
  ApiServer server{models, "test-version"};
  ApiServer::DispatchResult r = server.handle_systemone_separate(
      R"({"state":"x","questions":{"y":{"type":"noul"}}})");
  CHECK(r.status == 500);
}

TEST_CASE("api_server: systemone unknown question type is a 400") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"x","questions":{"q":{"type":"bogus"}}})");
  CHECK(r.status == 400);
}

TEST_CASE("api_server: systemone choice without criteria is a 400") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"x","questions":{"q":{"type":"choice"}}})");
  CHECK(r.status == 400);
}

TEST_CASE("api_server: systemone score with too few levels is a 400") {
  NerHarness h;
  ApiServer::DispatchResult r = h.server.handle_systemone(
      R"({"state":"x","questions":{"q":{"type":"score","criteria":["only"]}}})");
  CHECK(r.status == 400);
}
