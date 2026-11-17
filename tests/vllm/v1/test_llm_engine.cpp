// End-to-end tests for the V1 LLMEngine (M1.8 Task 6) — the whole loop wired:
// InputProcessor -> EngineCore(Scheduler + Executor over the real GPUModelRunner)
// -> OutputProcessor -> RequestOutput. Ported from vllm/v1/engine/llm_engine.py @
// e24d1b24 (add_request/step) + the LLM.generate driver.
//
// Drives a SMALL SYNTHETIC hybrid-MoE Qwen3.6 model on CPU (the real 35B greedy
// through the full paged loop on dgx is the milestone acceptance gate — dgx-
// pending). This proves the pieces actually CONNECT and produce a deterministic
// stream. Cases:
//   1. greedy determinism + termination: generate(prompt, greedy, max_tokens=N)
//      yields exactly N tokens, the loop terminates, and two runs of the same
//      prompt+greedy produce the SAME tokens.
//   2. 2-request concurrent batch: two requests added, stepped together, each
//      gets its own correct stream (matches its standalone single-request run).
//   3. streaming (DELTA) vs non-streaming (CUMULATIVE): the concatenated deltas
//      equal the cumulative full text for the same prompt+greedy.
//   4. stop on max_tokens (LENGTH finish) end to end.
//   5. stop on a stop STRING: the OutputProcessor's string-stop -> reqs_to_abort
//      -> EngineCore aborts -> the loop ends (finish_reason "stop").
//
// The synthetic model uses vocab_size == the tiny BPE fixture's assigned id count
// (0..23, no holes) so every greedy argmax is decodable. The KV cache config uses
// a UNIFIED block_size == max_model_len == hash_block_size (the hybrid KV
// coordinator requires every group's block_size == hash_block_size); with prompts
// far shorter than a block, prefix caching stays inert (no block ever completes).
#include "vllm/v1/engine/llm_engine.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/entrypoints/beam_search.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/metrics/prometheus.h"
#include "vllm/v1/metrics/stats.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
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
using vllm::v1::metrics::PrometheusStatLogger;
using vt::DType;

namespace {
// Restores the previous value at scope exit (KERNEL-GDN-CHUNKED-MIRROR T9/R10:
// a bare setenv here would make doctest's case order select a GDN algorithm).
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) { had_old_ = true; old_ = old; }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_) setenv(name_.c_str(), old_.c_str(), 1);
    else unsetenv(name_.c_str());
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};


// ─── Synthetic weights (mirrors tests/vllm/v1/worker/test_runner.cpp) ────────
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

// Vocab size == the tiny BPE fixture's assigned ids (0..23): every greedy argmax
// over the lm_head is decodable by the detokenizer (no vocab holes).
constexpr int kVocab = 24;
constexpr int kBlockSize = 32;      // == max_model_len == hash_block_size (hybrid)
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
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
  c.raw = json::object();  // no eos_token_id -> generation runs to max_tokens.
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

// Hybrid KV config: one full-attn group + one mamba (GDN) group, UNIFIED
// block_size == max_model_len == hash_block_size (hybrid coordinator constraint).
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

// The tiny oracle-verified BPE fixture from tests/vllm/v1/test_output_processor
// (ids 0..23, no holes): "hello"=13, " world"=17, "1"=8, "2"=9, ...
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_llmengine_tok_" + std::to_string(counter++) + ".json"))
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
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
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

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// Owns a fully-wired LLMEngine stack over the shared const config/weights/
// tokenizer. Members are declared in dependency order (Scheduler + runner ->
// Executor -> EngineCore; InputProcessor/OutputProcessor -> LLMEngine) so the
// by-reference seams stay valid for the stack's lifetime.
struct Harness {
  // `max_num_batched_tokens` 0 means "the default budget, big enough that no
  // prompt is ever chunked"; a small positive value forces chunked prefill.
  Harness(const HfConfig& c, const Qwen3_5MoeWeights& w, const Tokenizer& tok,
          int max_num_reqs = 8, int max_num_batched_tokens = 0)
      : scheduler(MakeSchedulerConfig(max_num_batched_tokens), MakeKvConfig(c),
                  kBlockSize, /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), max_num_reqs, kMaxModelLen,
               max_num_batched_tokens > 0 ? max_num_batched_tokens
                                          : kMaxModelLen * max_num_reqs),
        executor(runner),
        engine_core(scheduler, executor),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, engine_core, output_processor, Hasher()) {}

  static SchedulerConfig MakeSchedulerConfig(int max_num_batched_tokens = 0) {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = max_num_batched_tokens > 0
                                     ? max_num_batched_tokens
                                     : kMaxModelLen * 8;
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
    // block_size == hash_block_size; prompts are far shorter than a block, so no
    // block ever completes and this stays inert (prefix caching never fires).
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  EngineCore engine_core;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  LLMEngine engine;
};

// The PRODUCTION-server stack: identical model/config/tokenizer to Harness but
// the asynchronous AsyncLLM frontend (the engine examples/server/main.cpp holds)
// instead of the sync LLMEngine. Used to gate BeamSearchAsync (online.py driver)
// token-for-token against the sync BeamSearch (offline.py) over the same weights.
struct AsyncHarness {
  AsyncHarness(const HfConfig& c, const Qwen3_5MoeWeights& w,
               const Tokenizer& tok, int max_num_reqs = 8)
      : scheduler(Harness::MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), max_num_reqs, kMaxModelLen,
               /*max_num_batched_tokens=*/kMaxModelLen * max_num_reqs),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, scheduler, executor, output_processor,
               Harness::Hasher()) {}

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  vllm::v1::AsyncLLM engine;
};

// The {model_name, engine} label suffix every series carries for model "m".
constexpr const char* kL = "{model_name=\"m\",engine=\"0\"}";

// Parse the scalar a Prometheus text line carries: find "<series> " and read the
// value to end of line. `series` is the full "name{labels}" (or a
// histogram/counter suffixed name). Fails the test if the series is absent.
double MetricValue(const std::string& text, const std::string& series) {
  const std::string needle = series + " ";
  const size_t p = text.find(needle);
  REQUIRE_MESSAGE(p != std::string::npos, "series absent: " << series);
  const size_t v = p + needle.size();
  const size_t e = text.find('\n', v);
  return std::stod(text.substr(v, e - v));
}

// The observation count of a histogram family (its _count series).
int64_t HistogramCount(const std::string& text, const std::string& name) {
  return static_cast<int64_t>(
      MetricValue(text, name + "_count" + std::string(kL)));
}

// The observation SUM of a histogram family (its _sum series). This is what
// flips from 0 to positive once the QUEUED/SCHEDULED events populate the
// per-request timing intervals: before the event wiring the histogram is still
// observed once per finished request, but always with value 0.0, so _count is
// nonzero while _sum stays exactly 0.
double HistogramSum(const std::string& text, const std::string& name) {
  return MetricValue(text, name + "_sum" + std::string(kL));
}

}  // namespace

// ─── 1. Greedy determinism + termination ─────────────────────────────────────
TEST_CASE("llm_engine: greedy generate is deterministic and terminates at max_tokens") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  RequestOutput run1;
  RequestOutput run2;
  {
    Harness h(c, w, tok);
    run1 = h.engine.generate(prompt, Greedy(kN), "req");
    CHECK_FALSE(h.engine.has_unfinished_requests());  // loop terminated.
  }
  {
    Harness h(c, w, tok);  // fresh stack -> fresh KV/scheduler state.
    run2 = h.engine.generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  // Exactly N tokens produced (no eos configured -> length finish).
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kN);
  REQUIRE(run1.outputs[0].finish_reason.has_value());
  CHECK(*run1.outputs[0].finish_reason == "length");

  // Deterministic: same prompt + greedy -> identical token stream + text.
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run1.outputs[0].token_ids == run2.outputs[0].token_ids);
  CHECK(run1.outputs[0].text == run2.outputs[0].text);
}

// ─── 1b. n>1 parallel sampling: deterministic fan-out into n outputs ─────────
// SAMPLE-N (ROAD-V1-C7). A single request with n>1 is fanned out by the engine
// into n child sequences (llm_engine.py:280 ParentRequest), each aggregated back
// into ONE RequestOutput carrying n CompletionOutputs (parallel_sampling.py
// get_outputs / output_processor.py:326). UPSTREAM'S children share the prompt
// list — `copy(request)` (llm_engine.py:283) is SHALLOW. OURS each copy it,
// because EngineCoreRequest::prompt_token_ids is held BY VALUE (types.h:79);
// that asymmetry is a cost, not a semantic difference, and is tracked as #2145.
//
// The gate is deterministic-exact: vLLM FORBIDS n>1 under greedy sampling
// (sampling_params.py::_verify_greedy_sampling), so the clean n>1 determinism
// gate uses top_k=1 sampling — a LEGAL n>1 config whose sampling collapses to the
// argmax, i.e. every child is token-identical to the single greedy result. RED
// before the fan-out: an n>1 request returns exactly ONE output; GREEN after: n.
TEST_CASE("llm_engine: n>1 fans out into n token-identical deterministic outputs") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kTok = 6;    // tokens per sequence
  const int kSeqs = 4;   // n

  // A deterministic n==1 sampling config: top_k=1 collapses random sampling to
  // the argmax. Seeded so seeded children take the seed+index clone path.
  auto DeterministicN = [&](int n) {
    SamplingParams sp;
    sp.n = n;
    sp.temperature = 1.0;  // random sampling path (legal for n>1)
    sp.top_k = 1;          // only the argmax survives -> deterministic
    sp.seed = 1234;        // seeded -> each child gets seed+index
    sp.max_tokens = kTok;
    sp.output_kind = RequestOutputKind::kFinalOnly;  // aggregate the n outputs
    sp.PostInit();         // n>1 sampling is accepted (not greedy)
    return sp;
  };

  // Reference sequences from FRESH stacks (each run gets clean KV/scheduler state).
  std::vector<int32_t> greedy_ref;
  {
    Harness h(c, w, tok);
    greedy_ref = h.engine.generate(prompt, Greedy(kTok), "g").outputs.at(0).token_ids;
  }
  std::vector<int32_t> topk1_ref;  // same sampler path as the children
  {
    Harness h(c, w, tok);
    topk1_ref =
        h.engine.generate(prompt, DeterministicN(1), "s").outputs.at(0).token_ids;
  }
  REQUIRE(static_cast<int>(greedy_ref.size()) == kTok);
  REQUIRE(static_cast<int>(topk1_ref.size()) == kTok);

  // n>1: one RequestOutput aggregating n child completions.
  Harness h(c, w, tok);
  RequestOutput r = h.engine.generate(prompt, DeterministicN(kSeqs), "par");

  REQUIRE(r.finished);
  CHECK(r.request_id == "par");                 // aggregated under the PARENT id
  CHECK_FALSE(h.engine.has_unfinished_requests());
  // RED before fan-out: this is 1. GREEN after: exactly n.
  REQUIRE(static_cast<int>(r.outputs.size()) == kSeqs);

  // Each of the n completions carries its own index 0..n-1 and is token-identical
  // to the deterministic single-sequence result.
  std::set<int> seen;
  for (const vllm::CompletionOutput& o : r.outputs) {
    seen.insert(o.index);
    CHECK(static_cast<int>(o.token_ids.size()) == kTok);
    CHECK(o.token_ids == topk1_ref);   // same sampler path => identical
    CHECK(o.token_ids == greedy_ref);  // top_k=1 == greedy argmax
    REQUIRE(o.finish_reason.has_value());
    CHECK(*o.finish_reason == "length");
  }
  for (int i = 0; i < kSeqs; ++i) {
    CHECK_MESSAGE(seen.count(i) == 1, "missing completion index " << i);
  }
}

// ─── 1c. Beam search: end-to-end over the CPU engine (SAMPLE-BEAM) ───────────
// Beam search is an OUTER loop over the engine: each step runs one decode per beam
// (logprobs=2*beam_width, max_tokens=1), expands each beam to those next tokens,
// keeps the top-beam_width by the length-penalty score, and returns beam_width
// beams ordered by descending score (vllm/entrypoints/generate/beam_search/
// offline.py). Deterministic (greedy per-beam) ⇒ exact. The model-free scoring /
// selection / EOS / length-penalty gate is tests/vllm/entrypoints/test_beam_search
// .cpp; here we prove the driver runs the real engine and returns beam_width valid
// continuations, anchored to the deterministic greedy path: beam_width=1 beam
// search IS greedy (keep top-1 each step), so its tokens must equal plain greedy
// generate, and a wider beam must find >= cumulative logprob.
TEST_CASE("llm_engine: beam search returns beam_width scored continuations, bw=1 == greedy") {
  const HfConfig c = MakeConfig();  // synthetic Qwen3.6, no eos -> runs to length
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kSteps = 4;
  const std::optional<int> kNoEos = std::nullopt;  // config has no eos_token_id

  // Greedy reference (+ the exact prompt token ids the engine tokenizes "hello"
  // into, reused verbatim for the beam-search prompt so tokenization matches).
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> greedy_gen;
  {
    Harness h(c, w, tok);
    RequestOutput g = h.engine.generate("hello", Greedy(kSteps), "g");
    prompt_tokens = g.prompt_token_ids;
    greedy_gen = g.outputs.at(0).token_ids;
  }
  REQUIRE(!prompt_tokens.empty());
  REQUIRE(static_cast<int>(greedy_gen.size()) == kSteps);

  auto Params = [&](int beam_width) {
    vllm::BeamSearchParams p;
    p.beam_width = beam_width;
    p.max_tokens = kSteps;
    p.temperature = 0.0;  // greedy per-beam -> deterministic
    p.length_penalty = 1.0;
    p.ignore_eos = false;
    return p;
  };

  // beam_width == 1 collapses to greedy: keep top-1 of the 2 candidates each step
  // (score with length_penalty 1.0 and equal length == argmax cum_logprob).
  vllm::BeamSearchOutput bw1;
  {
    Harness h(c, w, tok);
    bw1 = vllm::BeamSearch(h.engine, prompt_tokens, Params(1), kNoEos, &tok);
  }
  REQUIRE(bw1.sequences.size() == 1);
  // Full tokens == prompt + generated; the generated tail must equal greedy.
  REQUIRE(bw1.sequences[0].tokens.size() == prompt_tokens.size() + kSteps);
  std::vector<int32_t> bw1_gen(
      bw1.sequences[0].tokens.begin() +
          static_cast<std::ptrdiff_t>(prompt_tokens.size()),
      bw1.sequences[0].tokens.end());
  CHECK(bw1_gen == greedy_gen);

  // beam_width == 3: exactly 3 outputs, each a valid full-length continuation of
  // the prompt, distinct, ordered by descending score.
  const int kBeam = 3;
  vllm::BeamSearchOutput bw3;
  {
    Harness h(c, w, tok);
    bw3 = vllm::BeamSearch(h.engine, prompt_tokens, Params(kBeam), kNoEos, &tok);
  }
  REQUIRE(static_cast<int>(bw3.sequences.size()) == kBeam);

  std::set<std::vector<int32_t>> seen;
  double prev_score = 1e18;
  for (const vllm::BeamSearchSequence& s : bw3.sequences) {
    // Valid continuation: prompt prefix + kSteps generated tokens.
    REQUIRE(s.tokens.size() == prompt_tokens.size() + kSteps);
    CHECK(std::equal(prompt_tokens.begin(), prompt_tokens.end(),
                     s.tokens.begin()));
    // Distinct beams.
    CHECK(seen.insert(s.tokens).second);
    // No eos configured -> length finish; text decoded (tokenizer supplied).
    REQUIRE(s.finish_reason.has_value());
    CHECK(*s.finish_reason == "length");
    CHECK(s.text.has_value());
    // Descending by beam-search score.
    const double score =
        vllm::get_beam_search_score(s.tokens, s.cum_logprob, kNoEos, 1.0);
    CHECK(score <= prev_score + 1e-9);
    prev_score = score;
  }

  // Beam search never does WORSE than greedy in cumulative logprob: the best
  // beam_width=3 beam's cum_logprob >= the beam_width=1 (greedy) beam's.
  CHECK(bw3.sequences[0].cum_logprob >= bw1.sequences[0].cum_logprob - 1e-6);
}

// ─── 1d. ASYNC beam search: BeamSearchAsync == BeamSearch, token-identical ────
// The CORE async gate (SAMPLE-BEAM async/production). BeamSearchAsync drives an
// AsyncLLM (the production-server engine, online.py) instead of the sync
// LLMEngine (offline.py), but calls the SAME model-free BeamSearchStep scoring.
// Over the SAME synthetic model/prompt/params, the async driver must return beams
// token-IDENTICAL to the sync driver — same tokens, same order, same scores,
// same text. This proves the async engine-drive did NOT change the algorithm.
TEST_CASE("llm_engine: BeamSearchAsync returns beams token-identical to sync BeamSearch") {
  const HfConfig c = MakeConfig();  // synthetic Qwen3.6, no eos -> runs to length
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kSteps = 4;
  const std::optional<int> kNoEos = std::nullopt;

  const std::vector<int32_t> prompt_tokens = tok.Encode("hello");
  REQUIRE(!prompt_tokens.empty());

  auto Params = [&](int beam_width) {
    vllm::BeamSearchParams p;
    p.beam_width = beam_width;
    p.max_tokens = kSteps;
    p.temperature = 0.0;  // greedy per-beam -> deterministic
    p.length_penalty = 1.0;
    p.ignore_eos = false;
    return p;
  };

  for (const int beam_width : {1, 2, 3}) {
    // Sync (offline) driver over a fresh LLMEngine stack.
    vllm::BeamSearchOutput sync_out;
    {
      Harness h(c, w, tok);
      sync_out = vllm::BeamSearch(h.engine, prompt_tokens, Params(beam_width),
                                  kNoEos, &tok);
    }
    // Async (online) driver over a fresh, identical AsyncLLM stack.
    vllm::BeamSearchOutput async_out;
    {
      AsyncHarness h(c, w, tok);
      async_out = vllm::BeamSearchAsync(h.engine, prompt_tokens,
                                        Params(beam_width), kNoEos, &tok);
    }

    // Token-identical: same beam count, and per beam same tokens / cum_logprob /
    // finish_reason / decoded text, in the SAME descending-score order.
    REQUIRE(async_out.sequences.size() == sync_out.sequences.size());
    REQUIRE(static_cast<int>(async_out.sequences.size()) == beam_width);
    for (std::size_t i = 0; i < sync_out.sequences.size(); ++i) {
      const vllm::BeamSearchSequence& s = sync_out.sequences[i];
      const vllm::BeamSearchSequence& a = async_out.sequences[i];
      CHECK(a.tokens == s.tokens);
      CHECK(a.cum_logprob == doctest::Approx(s.cum_logprob));
      CHECK(a.finish_reason == s.finish_reason);
      CHECK(a.text == s.text);
    }
  }
}

// ─── 2. Two-request concurrent batch ─────────────────────────────────────────
TEST_CASE("llm_engine: 2-request concurrent batch produces correct per-request streams") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  // Oracle: each request run on its OWN engine.
  RequestOutput solo_a;
  RequestOutput solo_b;
  {
    Harness h(c, w, tok);
    solo_a = h.engine.generate("hello", Greedy(kN), "A");
  }
  {
    Harness h(c, w, tok);
    solo_b = h.engine.generate("world", Greedy(kN), "B");
  }

  // Both concurrently on one engine, stepped together.
  Harness h(c, w, tok);
  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "world", Greedy(kN));

  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) finished[out.request_id] = out;
    }
  }

  REQUIRE(finished.count("A") == 1);
  REQUIRE(finished.count("B") == 1);
  // Each concurrent stream matches its standalone single-request run (the batch
  // path is per-request independent).
  CHECK(finished["A"].outputs[0].token_ids == solo_a.outputs[0].token_ids);
  CHECK(finished["B"].outputs[0].token_ids == solo_b.outputs[0].token_ids);
  CHECK(finished["A"].outputs[0].token_ids != finished["B"].outputs[0].token_ids);
}

// ─── 3. Streaming (DELTA) vs non-streaming (CUMULATIVE) equivalence ───────────
TEST_CASE("llm_engine: streaming deltas concatenate to the non-streaming full text") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  // Non-streaming: the finished CUMULATIVE output carries the full text.
  RequestOutput cumulative;
  {
    Harness h(c, w, tok);
    cumulative = h.engine.generate(prompt, Greedy(kN), "req");
  }

  // Streaming: DELTA output_kind — accumulate the per-step deltas.
  std::string streamed;
  std::vector<int32_t> streamed_ids;
  {
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(kN);
    sp.output_kind = RequestOutputKind::kDelta;
    h.engine.add_request("req", prompt, sp);
    while (h.engine.has_unfinished_requests()) {
      for (const RequestOutput& out : h.engine.step()) {
        if (!out.outputs.empty()) {
          streamed += out.outputs[0].text;
          for (int32_t t : out.outputs[0].token_ids) streamed_ids.push_back(t);
        }
      }
    }
  }

  REQUIRE(cumulative.outputs.size() == 1);
  CHECK(streamed == cumulative.outputs[0].text);
  CHECK(streamed_ids == cumulative.outputs[0].token_ids);
}

// ─── 4. Stop on max_tokens (LENGTH finish), driven a step at a time ───────────
TEST_CASE("llm_engine: max_tokens length-stops the request end to end") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 3;

  Harness h(c, w, tok);
  h.engine.add_request("req", "hello", Greedy(kN));

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);  // must terminate.
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  CHECK(static_cast<int>(result.outputs[0].token_ids.size()) == kN);
  CHECK(*result.outputs[0].finish_reason == "length");
  CHECK_FALSE(h.engine.has_unfinished_requests());
}

// ─── 5. Stop on a stop STRING (OutputProcessor -> reqs_to_abort -> abort) ──────
TEST_CASE("llm_engine: a stop string ends the request through the full loop") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";

  // First produce the deterministic full greedy text so we can pick a stop
  // substring that actually appears in it.
  std::string full_text;
  {
    Harness h(c, w, tok);
    RequestOutput r = h.engine.generate(prompt, Greedy(8), "probe");
    full_text = r.outputs[0].text;
  }
  REQUIRE(full_text.size() >= 2);
  // A stop string drawn from the middle of the produced text (skip the first
  // char so truncation leaves a non-trivial prefix).
  const std::string stop = full_text.substr(1, 1);

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(8);
  sp.stop = {stop};
  h.engine.add_request("req", prompt, sp);

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  REQUIRE(result.outputs[0].finish_reason.has_value());
  // The string stop is detected by the OutputProcessor (not the scheduler's
  // token-level check_stop) -> finish_reason "stop", stop_reason == the string.
  CHECK(*result.outputs[0].finish_reason == "stop");
  REQUIRE(result.outputs[0].stop_reason.has_value());
  CHECK(*result.outputs[0].stop_reason == stop);
  // The reqs_to_abort feedback tore the request down in the EngineCore too.
  CHECK_FALSE(h.engine.has_unfinished_requests());
  // ...AND reached the SCHEDULER: a string stop is invisible to the scheduler's
  // token-level check_stop, so ONLY the reqs_to_abort -> abort_requests feedback
  // removes it. has_unfinished_requests() above only reflects OutputProcessor
  // state (which erases on finish regardless); this asserts the abort genuinely
  // propagated to the scheduler, so the request is not silently left running
  // (leaked KV + wasted compute) if the feedback were dropped.
  CHECK(h.scheduler.get_num_unfinished_requests() == 0);
  // Output truncated before the stop string (include_stop_str_in_output=false).
  CHECK(result.outputs[0].text.find(stop) == std::string::npos);
}

// ─── 6. Live per-step metrics feed the Prometheus registry (SERVE-METRICS) ────
// The step loop folds this step's SchedulerStats + IterationStats into the
// attached PrometheusStatLogger (llm_engine.py:308-329). Before this wiring the
// registry stayed primed at zero forever (a scrape returned the schema, not the
// counts). This drives the CPU reference engine for several real steps and
// asserts the LIVE values are correct and evolve.
TEST_CASE("llm_engine: live per-step stats populate the Prometheus registry") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 4;  // max_tokens per request (length stop).

  Harness h(c, w, tok);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);

  const std::string kRun = std::string("vllm:num_requests_running") + kL;
  const std::string kWait = std::string("vllm:num_requests_waiting") + kL;
  const std::string kPrompt = std::string("vllm:prompt_tokens_total") + kL;
  const std::string kGen = std::string("vllm:generation_tokens_total") + kL;
  const std::string kSuccessLen =
      "vllm:request_success_total{model_name=\"m\",engine=\"0\",finished_reason="
      "\"length\"}";

  // Baseline: primed at zero, no histogram observations — exactly the state a
  // scrape saw when the step site never recorded (the pre-wiring behaviour).
  {
    const std::string t = logger.Expose();
    CHECK(MetricValue(t, kPrompt) == 0.0);
    CHECK(MetricValue(t, kGen) == 0.0);
    CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 0);
    CHECK(HistogramCount(t, "vllm:e2e_request_latency_seconds") == 0);
  }

  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "hello", Greedy(kN));

  // Step 1 = prefill: both requests enter the running batch and emit their first
  // token. Gauges track the batch; prompt tokens (1 per "hello") are counted and
  // one TTFT sample is observed per request.
  std::map<std::string, RequestOutput> finished;
  for (RequestOutput& o : h.engine.step()) {
    if (o.finished) finished[o.request_id] = o;
  }
  {
    const std::string t = logger.Expose();
    CHECK(MetricValue(t, kRun) == 2.0);
    CHECK(MetricValue(t, kWait) == 0.0);
    CHECK(MetricValue(t, kPrompt) == 2.0);   // 1 token per "hello" x2 prefilled.
    CHECK(MetricValue(t, kGen) == 2.0);      // first token per request.
    CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 2);
    CHECK(HistogramCount(t, "vllm:inter_token_latency_seconds") == 0);
  }

  // Drive to completion.
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& o : h.engine.step()) {
      if (o.finished) finished[o.request_id] = o;
    }
  }
  REQUIRE(finished.size() == 2);

  int64_t total_gen = 0;
  for (const auto& kv : finished) {
    total_gen += static_cast<int64_t>(kv.second.outputs[0].token_ids.size());
  }
  CHECK(total_gen == 2 * kN);  // deterministic greedy, length-stopped.

  const std::string t = logger.Expose();
  // Token counters == the EXACT counts the run produced.
  CHECK(MetricValue(t, kPrompt) == 2.0);
  CHECK(MetricValue(t, kGen) == static_cast<double>(total_gen));
  // Both requests finished with the "length" reason.
  CHECK(MetricValue(t, kSuccessLen) == 2.0);
  // All requests drained: the gauges fall back to zero.
  CHECK(MetricValue(t, kRun) == 0.0);
  CHECK(MetricValue(t, kWait) == 0.0);
  // Histogram sample counts: TTFT once per request; ITL once per decode token
  // (kN-1 per request, the first token being the prefill TTFT); e2e + TPOT once
  // per finished request; iteration-tokens observed on at least the prefill step.
  CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:inter_token_latency_seconds") == 2 * (kN - 1));
  CHECK(HistogramCount(t, "vllm:e2e_request_latency_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:request_time_per_output_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:iteration_tokens_total") >= 1);
  CHECK(HistogramCount(t, "vllm:request_generation_tokens") == 2);
}

TEST_CASE("llm_engine: computed prompt metric includes every tokenless chunk") {
  const auto c = MakeConfig();
  const auto w = MakeWeights(c);
  Harness h(c, w, Fixture(), 8, /*max_num_batched_tokens=*/2);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);
  h.engine.add_request("computed", std::vector<int32_t>(7, 1), Greedy(2));
  const std::string metric = std::string("vllm:prefill_kv_computed_tokens_total") + kL;
  for (const int completed : {2, 4, 6, 7}) {
    h.engine.step();
    CHECK(MetricValue(logger.Expose(), metric) == completed);
  }
  while (h.engine.has_unfinished_requests()) h.engine.step();
  CHECK(MetricValue(logger.Expose(), metric) == 7);
}

// ─── 7. Per-request queue/prefill/inference timing from EngineCoreEvents ──────
// The scheduler records QUEUED (add_request), SCHEDULED (batch admission) and
// PREEMPTED events per request; the OutputProcessor folds them into the
// queue/prefill/inference timing intervals update_from_finished_request observes
// (stats.py:459-476), feeding the Prometheus timing histograms. Before this
// wiring these histograms were STILL observed once per finished request, but
// always with value 0.0 — so their _sum stayed exactly 0. This drives the real
// CPU engine to completion and asserts the sums now flip POSITIVE and the
// intervals are correctly ordered. RED-first: reverting the event
// emission/consumption leaves every _sum below at 0.0 and each CHECK fails.
TEST_CASE("llm_engine: per-request queue/prefill/inference timing populates") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 4;  // max_tokens per request (length stop).

  Harness h(c, w, tok);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);

  const char* kQueue = "vllm:request_queue_time_seconds";
  const char* kPrefill = "vllm:request_prefill_time_seconds";
  const char* kInference = "vllm:request_inference_time_seconds";
  const char* kDecode = "vllm:request_decode_time_seconds";
  const char* kE2E = "vllm:e2e_request_latency_seconds";

  // Baseline: nothing observed yet, so every timing _sum is 0.
  {
    const std::string t = logger.Expose();
    CHECK(HistogramSum(t, kQueue) == 0.0);
    CHECK(HistogramSum(t, kPrefill) == 0.0);
    CHECK(HistogramSum(t, kInference) == 0.0);
  }

  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "hello", Greedy(kN));
  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& o : h.engine.step()) {
      if (o.finished) finished[o.request_id] = o;
    }
  }
  REQUIRE(finished.size() == 2);

  const std::string t = logger.Expose();
  // Every finished request contributes one observation to each timing family.
  CHECK(HistogramCount(t, kQueue) == 2);
  CHECK(HistogramCount(t, kPrefill) == 2);
  CHECK(HistogramCount(t, kInference) == 2);

  const double queue_sum = HistogramSum(t, kQueue);
  const double prefill_sum = HistogramSum(t, kPrefill);
  const double inference_sum = HistogramSum(t, kInference);
  const double decode_sum = HistogramSum(t, kDecode);
  const double e2e_sum = HistogramSum(t, kE2E);

  // THE FLIP: each interval is now a real positive duration (was exactly 0).
  CHECK(queue_sum > 0.0);
  CHECK(prefill_sum > 0.0);
  CHECK(inference_sum > 0.0);
  CHECK(decode_sum > 0.0);

  // Interval algebra (stats.py): inference = prefill + decode (both measured
  // from the SAME scheduled_ts / first_token_ts / last_token_ts endpoints), so
  // the aggregate sums satisfy it too, within float tolerance.
  CHECK(inference_sum == doctest::Approx(prefill_sum + decode_sum));
  // Ordering: the queued span precedes scheduling, inference sits inside e2e,
  // and prefill is a sub-interval of inference.
  CHECK(inference_sum <= e2e_sum);
  CHECK(prefill_sum <= inference_sum);
  CHECK(queue_sum > 0.0);
}

// ─── 7b. The SAME invariants on the ASYNC serving path (SERVE-METRICS, #277) ──
// Cases 6 and 7 drive `LLMEngine`. The shipped server does not: it serves from
// `AsyncLLM` (server_main.cpp -> loaded->async_engine()), whose output handler
// folded nothing into any logger — so a real deployment scraped a well-formed
// `vllm:*` catalog whose series never moved, which reads as "idle" rather than
// as "missing". This drives the ASYNC stack over the identical model, prompts
// and sampling params as case 6 and asserts the identical invariants, plus case
// 7's per-request timing algebra.
//
// Mirrors async_llm.py:662-665 (build IterationStats), :676-678 (thread it into
// process_outputs) and :697-702 (fold it + scheduler_stats into the logger).
//
// The scrape happens after `shutdown()` joins the output-handler thread. That is
// the quiescence point: unlike upstream's asyncio handler — which runs to its
// next `await` before a woken consumer resumes, so `record()` always precedes
// the consumer — our handler is a real thread, and a drained collector says
// nothing about whether the fold for that step has retired.
//
// RED before the wiring: every value below is 0 (the primed schema, not counts).
TEST_CASE("async_llm: live per-step stats populate the Prometheus registry") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 4;  // max_tokens per request (length stop), as in case 6.

  AsyncHarness h(c, w, tok);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);

  const std::string kRun = std::string("vllm:num_requests_running") + kL;
  const std::string kWait = std::string("vllm:num_requests_waiting") + kL;
  const std::string kPrompt = std::string("vllm:prompt_tokens_total") + kL;
  const std::string kGen = std::string("vllm:generation_tokens_total") + kL;
  const std::string kSuccessLen =
      "vllm:request_success_total{model_name=\"m\",engine=\"0\",finished_reason="
      "\"length\"}";
  const char* kQueue = "vllm:request_queue_time_seconds";
  const char* kPrefill = "vllm:request_prefill_time_seconds";
  const char* kInference = "vllm:request_inference_time_seconds";
  const char* kDecode = "vllm:request_decode_time_seconds";
  const char* kE2E = "vllm:e2e_request_latency_seconds";

  // Baseline: primed at zero — exactly the state a production scrape saw.
  {
    const std::string t = logger.Expose();
    CHECK(MetricValue(t, kPrompt) == 0.0);
    CHECK(MetricValue(t, kGen) == 0.0);
    CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 0);
    CHECK(HistogramCount(t, kE2E) == 0);
    CHECK(HistogramSum(t, kQueue) == 0.0);
  }

  // Both requests are admitted before either is drained, so they share the
  // prefill step exactly as case 6's pair does.
  vllm::v1::AsyncRequest a =
      h.engine.add_request("A", std::string("hello"), Greedy(kN));
  vllm::v1::AsyncRequest b =
      h.engine.add_request("B", std::string("hello"), Greedy(kN));

  int64_t total_gen = 0;
  for (const vllm::v1::AsyncRequest* r : {&a, &b}) {
    for (;;) {
      const RequestOutput o = h.engine.get_output(*r);
      REQUIRE(o.outputs.size() == 1);
      if (o.finished) {
        total_gen += static_cast<int64_t>(o.outputs[0].token_ids.size());
        break;
      }
    }
  }
  CHECK(total_gen == 2 * kN);  // deterministic greedy, length-stopped.

  // Join the output handler: every fold for every step has now retired.
  h.engine.shutdown();

  const std::string t = logger.Expose();
  // Token counters == the EXACT counts the run produced (case 6's invariant).
  CHECK(MetricValue(t, kPrompt) == 2.0);  // 1 token per "hello" x2 prefilled.
  CHECK(MetricValue(t, kGen) == static_cast<double>(total_gen));
  // Both requests finished with the "length" reason.
  CHECK(MetricValue(t, kSuccessLen) == 2.0);
  // All requests drained: the gauges track the batch back down to zero.
  CHECK(MetricValue(t, kRun) == 0.0);
  CHECK(MetricValue(t, kWait) == 0.0);
  // Histogram sample counts: TTFT once per request; ITL once per decode token;
  // e2e + TPOT once per finished request.
  CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:inter_token_latency_seconds") == 2 * (kN - 1));
  CHECK(HistogramCount(t, kE2E) == 2);
  CHECK(HistogramCount(t, "vllm:request_time_per_output_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:iteration_tokens_total") >= 1);
  CHECK(HistogramCount(t, "vllm:request_generation_tokens") == 2);

  // Case 7's invariant on the async path: the per-request timing intervals are
  // real positive durations, and inference == prefill + decode.
  CHECK(HistogramCount(t, kQueue) == 2);
  CHECK(HistogramCount(t, kPrefill) == 2);
  CHECK(HistogramCount(t, kInference) == 2);
  const double queue_sum = HistogramSum(t, kQueue);
  const double prefill_sum = HistogramSum(t, kPrefill);
  const double inference_sum = HistogramSum(t, kInference);
  const double decode_sum = HistogramSum(t, kDecode);
  const double e2e_sum = HistogramSum(t, kE2E);
  CHECK(queue_sum > 0.0);
  CHECK(prefill_sum > 0.0);
  CHECK(inference_sum > 0.0);
  CHECK(decode_sum > 0.0);
  CHECK(e2e_sum > 0.0);
  CHECK(inference_sum == doctest::Approx(prefill_sum + decode_sum));
  CHECK(inference_sum <= e2e_sum);
  CHECK(prefill_sum <= inference_sum);
  // TTFT is measured from the engine-core timestamp, so it must be positive —
  // a zero/unstamped timestamp reports it as -arrival_time.
  CHECK(HistogramSum(t, "vllm:time_to_first_token_seconds") > 0.0);
}

// The DEFAULT async path — no logger attached — must stay byte-identical. This
// is the whole inertness claim: the fold is opt-in, so an engine with no logger
// takes the same no-stats process_outputs call it took before this row.
TEST_CASE("async_llm: with no logger attached the token stream is unchanged") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 6;

  std::vector<int32_t> with_logger;
  {
    AsyncHarness h(c, w, tok);
    PrometheusStatLogger logger("m", kMaxModelLen);
    h.engine.set_stat_logger(&logger);
    const RequestOutput r =
        h.engine.generate(std::string("hello"), Greedy(kN), "req");
    REQUIRE(r.outputs.size() == 1);
    with_logger = r.outputs[0].token_ids;
    // A terminal collector value is published before the output handler folds
    // the same iteration into the logger. Detach is therefore a quiescence
    // barrier: after it returns this caller may destroy the non-owning logger
    // even while the engine itself remains alive.
    h.engine.set_stat_logger(nullptr);
  }
  std::vector<int32_t> without_logger;
  {
    AsyncHarness h(c, w, tok);
    const RequestOutput r =
        h.engine.generate(std::string("hello"), Greedy(kN), "req");
    REQUIRE(r.outputs.size() == 1);
    without_logger = r.outputs[0].token_ids;
  }
  CHECK(with_logger.size() == static_cast<std::size_t>(kN));
  CHECK(with_logger == without_logger);
}

// ─── ENG-ASYNC-SCHED depth-2 serving-path e2e (heap-corruption regression) ────
// Full production-server stack (LoadedEngine -> AsyncLLM -> InprocClient ->
// EngineCoreProc::run_busy_loop -> step_with_batch_queue, mcb=2) over the MoE
// arch, driven with ignore_eos PAST the natural EOS — the exact serving shape
// that aborted the 35B (`malloc(): unaligned tcache chunk`) with a heap
// corruption in prepare_inputs. The synthetic model here settles into token 17,
// which we make the eos: ignore_eos=false stops at ~7 tokens, ignore_eos=true
// keeps generating past it. On the CPU eager backend there is no real GPU
// overlap so this cannot reproduce the crash by itself (the true guard is
// test_runner's async-drain invariant); it locks the full async serving loop
// against LOGIC regressions on this path and documents the trigger. See the
// runner's async_forward_in_flight_ lifetime guard for the actual fix.
TEST_CASE("llm_engine: async depth-2 serving generates past ignore_eos (no corruption)") {
  HfConfig c = MakeConfig();
  c.raw["eos_token_id"] = 17;  // the model settles into token 17 (step ~7+)
  vllm::entrypoints::EngineParams params;

  // ignore_eos=false: the request stops at the natural EOS well before the cap
  // (the fix does not change output — decode is byte-identical, just correctly
  // sequenced against the overlapped GPU work).
  {
    vllm::entrypoints::LoadedEngine eng(c, MakeWeights(c), Fixture(), params);
    CHECK(eng.async_scheduling_enabled());       // depth-2 step_with_batch_queue
    CHECK(eng.max_concurrent_batches() == 2);
    SamplingParams sp = Greedy(24);
    sp.ignore_eos = false;
    RequestOutput r = eng.async_engine().generate(std::string("hello"), sp, "s");
    REQUIRE(r.finished);
    REQUIRE(r.outputs.size() == 1);
    CHECK(static_cast<int>(r.outputs[0].token_ids.size()) < 24);  // stopped at eos
  }

  // ignore_eos=true PAST the ~4-8-token trigger: the exact serving shape that
  // aborted the 35B. The full AsyncLLM -> step_with_batch_queue depth-2 loop must
  // run to the cap and return a clean terminal output (CPU cannot reproduce the
  // GPU-overlap crash; this guards the loop's LOGIC — the runner drain invariant
  // is gated in test_runner).
  {
    vllm::entrypoints::LoadedEngine eng(c, MakeWeights(c), Fixture(), params);
    SamplingParams sp = Greedy(16);
    sp.ignore_eos = true;
    RequestOutput r = eng.async_engine().generate(std::string("hello"), sp, "req");
    REQUIRE(r.finished);
    REQUIRE(r.outputs.size() == 1);
    CHECK(static_cast<int>(r.outputs[0].token_ids.size()) == 16);  // ran to cap
  }
}

// ─── 8. logprobs=-1 reaches the client intact (issue #231) ───────────────────
// "All logprobs". The sampler has a `num_logprobs == -1` branch that returns a
// raw-vocab LogprobsTensors with EMPTY ids and ranks (1:1 sampler.py:122-125)
// while `num_tokens_per_position` is the full vocab — so a live request that
// reached that branch died in the FIRST thing to touch the tensor:
// LogprobsTensors::slice_request (src/vllm/v1/outputs.cpp:31-37), called from
// scheduler.cpp:920-924, which assigns a num_positions*vocab range out of the
// two empty vectors' null begin(). LogprobsProcessor::UpdateSampleLogprobs
// (src/vllm/v1/engine/logprobs.cpp:51-77) indexes the same two arrays and would
// fault identically, but the process never gets there. Upstream reaches neither,
// because gpu_input_batch.py:434-440 widens the sentinel to vocab_size at
// admission; we propagated it instead.
//
// RED before the widening: SIGSEGV inside LogprobsTensors::slice_request.
TEST_CASE("llm_engine: logprobs=-1 returns a full-vocab logprobs dict") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 3;

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(kN);
  sp.logprobs = -1;  // all
  const RequestOutput r = h.engine.generate(std::string("hello"), sp, "req");

  REQUIRE(r.finished);
  REQUIRE(r.outputs.size() == 1);
  REQUIRE(r.outputs[0].logprobs.has_value());
  REQUIRE(r.outputs[0].logprobs->size() == static_cast<std::size_t>(kN));

  for (std::size_t i = 0; i < r.outputs[0].logprobs->size(); ++i) {
    const vllm::LogprobsOnePosition& pos = (*r.outputs[0].logprobs)[i];
    // "All" means every vocab entry is present, exactly once.
    CHECK(pos.entries.size() == static_cast<std::size_t>(kVocab));
    // The sampled token carries its own rank, and the row is a distribution.
    const vllm::Logprob* self = pos.find(r.outputs[0].token_ids[i]);
    REQUIRE(self != nullptr);
    CHECK(self->rank == 1);  // greedy -> the sampled token IS the argmax
    double mass = 0.0;
    for (const auto& [tid, lp] : pos.entries) {
      (void)tid;
      mass += std::exp(static_cast<double>(lp.logprob));
    }
    CHECK(mass == doctest::Approx(1.0).epsilon(1e-4));
  }
}

// A finite count is unchanged by the widening. The gathered row is
// [sampled | top-2], k+1 == 3 wide; greedy makes the sampled token the rank-1
// entry, so the dict-dedup collapses it to EXACTLY 2 distinct ids at rank 1 and
// rank 2. Asserted exactly, not as a `>= 1 && <= 3` range, which any narrower or
// empty-ish row would also satisfy.
//
// What this case deliberately does NOT claim to cover: the SAMPLER-side gather
// width. AppendLogprobsForNextPosition (logprobs.h:82) truncates to the
// REQUEST's own num_logprobs, so widening every request to vocab_size leaves the
// client-visible payload byte-identical — no engine-level assertion can see it.
// That width is pinned where it is observable, at the input-batch seam:
// tests/vllm/v1/worker/test_input_batch.cpp:641 (`max_num_logprobs == 4` for a
// finite request) and :687 (`num_logprobs.at("a") == 3`); mutating add_request to
// widen unconditionally turns both RED.
TEST_CASE("llm_engine: a finite logprobs count is unaffected by the -1 widening") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(3);
  sp.logprobs = 2;
  const RequestOutput r = h.engine.generate(std::string("hello"), sp, "req");

  REQUIRE(r.outputs.size() == 1);
  REQUIRE(r.outputs[0].logprobs.has_value());
  for (std::size_t i = 0; i < r.outputs[0].logprobs->size(); ++i) {
    const vllm::LogprobsOnePosition& pos = (*r.outputs[0].logprobs)[i];
    CHECK(pos.entries.size() == 2u);  // sampled(==top-1) + top-2
    const vllm::Logprob* self = pos.find(r.outputs[0].token_ids[i]);
    REQUIRE(self != nullptr);
    CHECK(self->rank == 1);  // greedy -> the sampled token IS the argmax
  }
}

// ─── 9. Prompt logprobs (SAMPLE-PROMPT-LOGPROBS, issue #223) ─────────────────
// The runner-side source ported from gpu_model_runner.py:5612-5719
// (`_get_prompt_logprobs_dict`). Everything below the runner was already
// landed and gated, so before this row `prompt_logprobs=k` produced a
// RequestOutput whose prompt_logprobs held ONLY the leading None that
// LogprobsProcessor::FromNewRequest seeds — the feature was a silent no-op.
// These cases are RED against that state.
//
// The tiny fixture has no eos, vocab 0..23, max_model_len 32.
namespace {

// The full prompt-logprob payload for `prompt_token_ids`, run to completion.
std::optional<vllm::PromptLogprobs> PromptLogprobsFor(
    const HfConfig& c, const Qwen3_5MoeWeights& w, const Tokenizer& tok,
    const std::vector<int32_t>& prompt_token_ids, int prompt_logprobs,
    int max_tokens = 2, int max_num_batched_tokens = 0) {
  Harness h(c, w, tok, /*max_num_reqs=*/8, max_num_batched_tokens);
  SamplingParams sp = Greedy(max_tokens);
  sp.prompt_logprobs = prompt_logprobs;
  const RequestOutput r = h.engine.generate(prompt_token_ids, sp, "req");
  REQUIRE(r.finished);
  return r.prompt_logprobs;
}

}  // namespace

// (a) SHAPE: one entry per prompt token, the first None, the rest carrying the
// prompt token itself. 1:1 vllm/v1/engine/logprobs.py:162-187.
TEST_CASE("llm_engine: prompt_logprobs yields one entry per prompt token") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  const int kK = 2;

  const std::optional<vllm::PromptLogprobs> plp =
      PromptLogprobsFor(c, w, tok, prompt, kK);

  REQUIRE(plp.has_value());
  // RED before the runner source: exactly 1 (the seeded leading None).
  REQUIRE(plp->size() == prompt.size());
  CHECK_FALSE((*plp)[0].has_value());  // nothing precedes the first token
  for (std::size_t i = 1; i < plp->size(); ++i) {
    REQUIRE_MESSAGE((*plp)[i].has_value(), "position " << i << " must be scored");
    const vllm::LogprobsOnePosition& pos = *(*plp)[i];
    // The prompt token itself is always present, with its true rank.
    const vllm::Logprob* self = pos.find(prompt[i]);
    REQUIRE_MESSAGE(self != nullptr, "prompt token missing at position " << i);
    CHECK(self->rank >= 1);
    // sampled + k top-k, deduped when the prompt token IS in the top-k.
    CHECK(pos.entries.size() >= 1);
    CHECK(pos.entries.size() <= static_cast<std::size_t>(kK) + 1);
  }
}

// (b) VALUES: prompt position i scores the token at i+1 against the model's
// distribution AT position i. That is the same distribution the sampler sees
// when the request's prompt is truncated to [0, i), so the two must agree — an
// independent check that arrives through the SAMPLED-logprobs path instead of
// the prompt one. The token compared is the truncated run's greedy pick, which
// is guaranteed present on both sides: it is rank 1 for the sampler, and the
// prompt row asked for every vocab entry.
TEST_CASE("llm_engine: a prompt logprob equals the sampled logprob at the same position") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};

  const std::optional<vllm::PromptLogprobs> plp =
      PromptLogprobsFor(c, w, tok, prompt, /*prompt_logprobs=*/-1);
  REQUIRE(plp.has_value());
  REQUIRE(plp->size() == prompt.size());

  for (std::size_t i = 1; i < prompt.size(); ++i) {
    // Same model, prompt truncated to [0, i): its ONE sampled position is
    // prompt position i-1 — the row that scores prompt[i].
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(1);
    sp.logprobs = 1;
    const std::vector<int32_t> head(prompt.begin(),
                                    prompt.begin() + static_cast<long>(i));
    const RequestOutput r = h.engine.generate(head, sp, "head");
    REQUIRE(r.outputs.size() == 1);
    REQUIRE(r.outputs[0].token_ids.size() == 1);
    REQUIRE(r.outputs[0].logprobs.has_value());
    REQUIRE_FALSE(r.outputs[0].logprobs->empty());
    const int32_t argmax = r.outputs[0].token_ids[0];
    const vllm::Logprob* sampled = (*r.outputs[0].logprobs)[0].find(argmax);
    REQUIRE(sampled != nullptr);

    const vllm::Logprob* from_prompt = (*plp)[i]->find(argmax);
    REQUIRE_MESSAGE(from_prompt != nullptr,
                    "argmax absent from the all-vocab prompt row at " << i);
    // The argmax of the distribution, so rank 1 on both sides.
    CHECK(sampled->rank == 1);
    CHECK(from_prompt->rank == 1);
    // Same distribution reached over a different sequence length, so compare
    // with a tolerance rather than bit-for-bit.
    CHECK(from_prompt->logprob == doctest::Approx(sampled->logprob).epsilon(1e-4));
  }
}

// (c) NORMALIZATION: prompt_logprobs=-1 widens to the whole vocab, and a row of
// log_softmax exponentiates to 1. Catches a raw-logits-instead-of-logprobs
// regression, which no shape assertion would.
TEST_CASE("llm_engine: prompt_logprobs=-1 rows are a normalized distribution") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};

  const std::optional<vllm::PromptLogprobs> plp =
      PromptLogprobsFor(c, w, tok, prompt, /*prompt_logprobs=*/-1);
  REQUIRE(plp.has_value());
  REQUIRE(plp->size() == prompt.size());

  for (std::size_t i = 1; i < plp->size(); ++i) {
    const vllm::LogprobsOnePosition& pos = *(*plp)[i];
    CHECK(pos.entries.size() == static_cast<std::size_t>(kVocab));
    double mass = 0.0;
    int best_rank_token_count = 0;
    for (const auto& [tid, lp] : pos.entries) {
      mass += std::exp(static_cast<double>(lp.logprob));
      if (lp.rank == 1) ++best_rank_token_count;
    }
    CHECK(mass == doctest::Approx(1.0).epsilon(1e-4));
    // Exactly one token holds rank 1 (the argmax), unless the prompt token IS
    // it — in which case the deduped dict carries that single entry once.
    CHECK(best_rank_token_count >= 1);
  }
}

// (d) CHUNKED PREFILL: the accumulation across chunks
// (gpu_model_runner.py:5646-5706) must land the identical tensor. Same prompt,
// a batched-token budget small enough to split it, byte-identical payload.
TEST_CASE("llm_engine: chunked prefill accumulates the identical prompt logprobs") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  const int kK = 2;

  // ─── THE STITCHING IS EXACT; THE ARITHMETIC UNDER IT NEED NOT BE ───────────
  //
  // This case is about the ACCUMULATION MACHINERY (gpu_model_runner.py:5646-5706)
  // — that rows produced in separate scheduler steps are stitched back into the
  // same tensor, at the same positions, for the same token ids. That claim is
  // arm-independent, and it stays gated EXACTLY, on the arm where "identical"
  // is achievable.
  //
  // It is not achievable on the other one, and that is upstream's arrangement
  // rather than a defect. This config's layers are `linear_attention`, so every
  // chunk boundary crosses a GDN prefill, and since KERNEL-GDN-CHUNKED-MIRROR
  // (#2612 / #2845) that runs vLLM's chunked WY decomposition, in which the
  // interactions across a boundary go through a BF16-ROUNDED state snapshot
  // (chunk_delta_h.py:178,352) instead of the intra-chunk f32 path. With
  // `max_num_batched_tokens=1` this case puts a boundary between EVERY pair of
  // prompt tokens, which is the maximum discontinuity the algorithm admits.
  // Upstream gates the same property on its own chunked kernel at 1e-3 state /
  // 2e-2 output rather than at equality
  // (tests/kernels/mamba/cpu/test_cpu_gdn_ops.py::
  //  test_chunk_gated_delta_rule_cpu_two_call_split @ e126687a9a).
  //
  // Measured here: sequential arm 0 exactly; chunked arm max|d| 3.28e-3, max
  // relative 1.2e-3.
  auto compare = [&](const char* flag, bool require_exact) {
    ScopedEnv pin("VT_GDN_CHUNKED", flag);
    const std::optional<vllm::PromptLogprobs> whole =
        PromptLogprobsFor(c, w, tok, prompt, kK);
    // A budget of 1 token per step forces one chunk per prompt token, which also
    // walks the num_logits <= 0 exact-prefill edge (:5668-5671).
    const std::optional<vllm::PromptLogprobs> chunked = PromptLogprobsFor(
        c, w, tok, prompt, kK, /*max_tokens=*/2, /*max_num_batched_tokens=*/1);

    REQUIRE(whole.has_value());
    REQUIRE(chunked.has_value());
    // Assert the height explicitly: comparing two EMPTY payloads would otherwise
    // pass vacuously against a runner that produces nothing.
    REQUIRE(whole->size() == prompt.size());
    REQUIRE(chunked->size() == prompt.size());
    double worst = 0.0;
    int compared = 0;
    for (std::size_t i = 0; i < whole->size(); ++i) {
      REQUIRE((*whole)[i].has_value() == (*chunked)[i].has_value());
      if (!(*whole)[i].has_value()) continue;
      const vllm::LogprobsOnePosition& a = *(*whole)[i];
      const vllm::LogprobsOnePosition& b = *(*chunked)[i];
      REQUIRE(a.entries.size() == b.entries.size());
      for (const auto& [tid, lp] : a.entries) {
        const vllm::Logprob* other = b.find(tid);
        // THE MACHINERY ASSERTIONS, EXACT ON BOTH ARMS: the same token ids land
        // at the same positions with the same ranks. A stitching defect moves
        // or drops rows; it does not perturb them by 1e-3.
        REQUIRE_MESSAGE(other != nullptr, "token " << tid << " missing at " << i);
        CHECK(other->rank == lp.rank);
        if (require_exact) {
          CHECK(other->logprob == lp.logprob);
        } else {
          CHECK(other->logprob == doctest::Approx(lp.logprob).epsilon(5e-3));
        }
        worst = std::max(worst, std::abs(static_cast<double>(other->logprob) - lp.logprob));
        ++compared;
      }
    }
    // A `const char*` ternary directly inside MESSAGE() degrades to a bool and
    // prints "1"; bind it to a std::string first.
    const std::string arm = require_exact ? "SEQUENTIAL" : "CHUNKED";
    MESSAGE(arm << " GDN arm: " << compared
            << " logprobs compared, worst |d| = " << worst);
    // A counted property: `assertions: 0 ... SUCCESS!` is a skip wearing a pass.
    CHECK(compared > 0);
  };
  compare("0", /*require_exact=*/true);
  compare("1", /*require_exact=*/false);
}

// (e) INERTNESS: a request that did not ask gets nothing, and asking must not
// move the sampled tokens. The prompt rows widen the lm_head gather, so this is
// the assertion that the widening stays confined to the request that asked.
TEST_CASE("llm_engine: prompt_logprobs is inert when unset and does not move tokens") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  const int kN = 6;

  RequestOutput off;
  {
    Harness h(c, w, tok);
    off = h.engine.generate(prompt, Greedy(kN), "req");
  }
  RequestOutput on;
  {
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(kN);
    sp.prompt_logprobs = 2;
    on = h.engine.generate(prompt, sp, "req");
  }

  CHECK_FALSE(off.prompt_logprobs.has_value());  // never asked -> never built
  REQUIRE(on.prompt_logprobs.has_value());
  REQUIRE(off.outputs.size() == 1);
  REQUIRE(on.outputs.size() == 1);
  CHECK(on.outputs[0].token_ids == off.outputs[0].token_ids);
}

// (f) CONCURRENCY: two requests in one batch, different k, each gets its own
// tensor at its own width. Guards the request-order slicing of the appended
// gather rows.
TEST_CASE("llm_engine: concurrent requests get their own prompt logprobs") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> p_a = {1, 2, 3, 4, 5};
  const std::vector<int32_t> p_b = {6, 7, 8};

  Harness h(c, w, tok);
  SamplingParams sp_a = Greedy(3);
  sp_a.prompt_logprobs = 1;
  SamplingParams sp_b = Greedy(3);
  sp_b.prompt_logprobs = 3;
  h.engine.add_request("a", p_a, sp_a);
  h.engine.add_request("b", p_b, sp_b);

  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& r : h.engine.step()) {
      if (r.finished) finished[r.request_id] = std::move(r);
    }
  }

  REQUIRE(finished.count("a") == 1);
  REQUIRE(finished.count("b") == 1);
  REQUIRE(finished["a"].prompt_logprobs.has_value());
  REQUIRE(finished["b"].prompt_logprobs.has_value());
  REQUIRE(finished["a"].prompt_logprobs->size() == p_a.size());
  REQUIRE(finished["b"].prompt_logprobs->size() == p_b.size());
  // Widths follow each request's own k, not the batch max.
  CHECK((*finished["a"].prompt_logprobs)[1]->entries.size() <= 2);
  CHECK((*finished["b"].prompt_logprobs)[1]->entries.size() <= 4);
}

// (g) THE ROUTE DECISION ITSELF. Case (e) compares prompt-logprobs-on against
// -off inside ONE build, so it can never see a change to the SHARED production
// route: force the full-logits path on every step and both arms move together,
// still agreeing. (Review finding 2 on PR #235: that exact mutation left the
// file 17/17 · 346 assertions, SUCCESS.) So assert the decision, not its
// symmetry — on a step where nobody asked, the forward must have gathered
// before lm_head and produced step_num_logits() rows, NOT one row per token.
// The prefill step of a 5-token prompt is 5 tokens against 1 sampler row, so
// the two counts are far apart and the assertion has teeth.
TEST_CASE("llm_engine: no prompt-logprob request keeps the lm_head gather on") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};

  // OFF: every forward this run must be the gathered one.
  int gathered_prefill_steps = 0;
  {
    Harness h(c, w, tok);
    h.engine.add_request("off", prompt, Greedy(3));
    while (h.engine.has_unfinished_requests()) {
      h.engine.step();
      if (h.runner.last_forward_num_reqs() == 0) continue;  // 0-token flush
      REQUIRE(h.runner.last_forward_rows() == h.runner.step_num_logits());
      if (h.runner.last_forward_num_actual_tokens() >
          h.runner.step_num_logits()) {
        ++gathered_prefill_steps;
      }
    }
  }
  // A step where the counts actually differ has to have happened, or the
  // assertion above would be vacuous.
  CHECK(gathered_prefill_steps >= 1);

  // ON: the same prefill must take the OTHER route — one lm_head row per
  // scheduled token — which is what makes the OFF assertion a decision and not
  // a tautology about ForwardLogits.
  int full_logits_steps = 0;
  {
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(3);
    sp.prompt_logprobs = 2;
    h.engine.add_request("on", prompt, sp);
    while (h.engine.has_unfinished_requests()) {
      h.engine.step();
      if (h.runner.last_forward_num_reqs() == 0) continue;
      if (h.runner.last_forward_rows() ==
              h.runner.last_forward_num_actual_tokens() &&
          h.runner.last_forward_num_actual_tokens() >
              h.runner.step_num_logits()) {
        ++full_logits_steps;
      }
    }
  }
  CHECK(full_logits_steps >= 1);
}

// (h) THE EXACT-PREFILL EDGE IN A MIXED BATCH. prepare_inputs deliberately
// keeps a final-chunk entry with num_rows == 0 (gpu_model_runner.py:5668-5673):
// the previous step consumed exactly num_prompt_tokens - 1 prompt tokens, so
// there is nothing left to score but the tensor still has to be EMITTED. That
// entry contributes NO gather indices, so `prompt_logprob_indices` is empty and
// the step keeps the gathered lm_head — while `prompt_logprob_rows` is not.
//
// Case (d) walks the same edge with ONE request, where the full-logits row
// count and the sampler row count coincide at 1 by accident. Put a SECOND
// request in the step and they no longer do. Found by review on PR #235: the
// runner asserted "a prompt-logprob step must carry full logits" on
// prompt_logprob_rows rather than on the slice it was about to take, so this
// step threw out of engine.step() and killed the whole batch — including the
// request that never asked for prompt logprobs.
//
// Budget 3: step 1 prefills 3 of A's 4 tokens; step 2 schedules A's last prompt
// token (a zero-row final chunk) beside 2 tokens of B, so the step runs on 3
// tokens with 2 sampler rows.
TEST_CASE("llm_engine: a zero-row final chunk beside another request still emits") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::vector<int32_t> p_a = {1, 2, 3, 4};
  const std::vector<int32_t> p_b = {5, 6, 7, 8};

  Harness h(c, w, tok, /*max_num_reqs=*/8, /*max_num_batched_tokens=*/3);
  SamplingParams sp_a = Greedy(2);
  sp_a.prompt_logprobs = 2;
  h.engine.add_request("a", p_a, sp_a);
  h.engine.add_request("b", p_b, Greedy(2));  // never asks

  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& r : h.engine.step()) {
      if (r.finished) finished[r.request_id] = std::move(r);
    }
  }

  // Both requests survive the step: the throw took the whole batch down.
  REQUIRE(finished.count("a") == 1);
  REQUIRE(finished.count("b") == 1);
  CHECK_FALSE(finished["b"].prompt_logprobs.has_value());
  REQUIRE(finished["a"].prompt_logprobs.has_value());
  REQUIRE(finished["a"].prompt_logprobs->size() == p_a.size());
  CHECK_FALSE((*finished["a"].prompt_logprobs)[0].has_value());
  for (std::size_t i = 1; i < p_a.size(); ++i) {
    REQUIRE_MESSAGE((*finished["a"].prompt_logprobs)[i].has_value(),
                    "position " << i << " must be scored");
  }
}

// ─── 9. logprob_token_ids reaches the client end to end (issue #264) ─────────
// Generative scoring: the caller names the ids it wants scored, and gets back
// exactly those plus the sampled token — no full-vocab sort, no top-k.
//
// This is the reachability gate for the whole feature: it only passes when the
// SamplingParams field, the InputBatch plumbing, the sampler gather AND the
// `num_logprobs` property (which the scheduler's slice gate and the
// LogprobsProcessor read) are all wired.
//
// RED before the port: outputs[0].logprobs has NO value — the scheduler gates
// the slice on the raw `logprobs` field, which this request leaves unset.
TEST_CASE("llm_engine: logprob_token_ids returns exactly the requested ids") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 3;
  const std::vector<int32_t> kWanted = {5, 11, 2};

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(kN);
  sp.logprob_token_ids = kWanted;  // `logprobs` deliberately left unset
  const RequestOutput r = h.engine.generate(std::string("hello"), sp, "req");

  REQUIRE(r.finished);
  REQUIRE(r.outputs.size() == 1);
  REQUIRE(r.outputs[0].logprobs.has_value());
  REQUIRE(r.outputs[0].logprobs->size() == static_cast<std::size_t>(kN));

  for (std::size_t i = 0; i < r.outputs[0].logprobs->size(); ++i) {
    const vllm::LogprobsOnePosition& pos = (*r.outputs[0].logprobs)[i];
    const int32_t sampled = r.outputs[0].token_ids[i];
    // Every requested id is present and finite (never the -inf padding).
    for (int32_t want : kWanted) {
      const vllm::Logprob* lp = pos.find(want);
      REQUIRE(lp != nullptr);
      CHECK(std::isfinite(lp->logprob));
    }
    // The sampled token is present too, at its FULL-VOCAB rank (greedy, so 1).
    const vllm::Logprob* self = pos.find(sampled);
    REQUIRE(self != nullptr);
    CHECK(self->rank == 1);
    // And NOTHING else: the requested ids plus the sampled token, deduped.
    std::set<int32_t> expected(kWanted.begin(), kWanted.end());
    expected.insert(sampled);
    CHECK(pos.entries.size() == expected.size());
    for (const auto& [tid, unused] : pos.entries) {
      (void)unused;
      CHECK(expected.count(tid) == 1);
    }
  }
}
