// Unit tests for the OFFLINE Batch API runner (SERVE-BATCH-API, W1) — the
// RunBatch orchestrator driven over the SAME small SYNTHETIC LLMEngine +
// OpenAIServingChat the serving tests use (test_serving.cpp::Harness), NO
// socket, NO file. We assert the JSONL round-trip: a multi-line
// BatchRequestInput JSONL -> the correct BatchRequestOutput rows IN ORDER,
// custom_id echoed, per-line error isolation (a malformed line becomes an error
// row and does NOT abort the batch), and the endpoint dispatch table.
//
// Ported from: vllm/entrypoints/openai/run_batch.py @ 555967922; upstream test
// tests/entrypoints/openai/test_run_batch.py (test_empty_file:375,
// test_completions:402, test_completions_invalid_input:432 — re-expressed at the
// library level since we have no `vllm run-batch` subprocess CLI yet; see
// specs/batch-api.md § Tests to port).
#include "vllm/entrypoints/openai/run_batch.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/entrypoints/beam_search.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::ChatCompletionResponseChoice;
using vllm::entrypoints::openai::ChatCompletionResponseStreamChoice;
using vllm::entrypoints::openai::ChatCompletionResult;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::CompletionRequest;
using vllm::entrypoints::openai::CompletionResult;
using vllm::entrypoints::openai::DefaultChatPromptFallback;
using vllm::entrypoints::openai::DeltaMessage;
using vllm::entrypoints::openai::get_tool_parser;
using vllm::entrypoints::openai::HermesToolParser;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
using vllm::BeamSearch;
using vllm::BeamSearchOutput;
using vllm::BeamSearchParams;
using vllm::CompletionOutput;
using vllm::entrypoints::openai::SelectBestOf;
using vllm::entrypoints::openai::ShapeChatDelta;
using vllm::entrypoints::openai::ShapeChatMessage;
using vllm::entrypoints::openai::ShapedChatMessage;
using vllm::entrypoints::openai::ApplyToolChoiceStructuredOutput;
using vllm::entrypoints::openai::StreamOptions;
using vllm::entrypoints::openai::ToolChoice;
using vllm::entrypoints::openai::ToolChoiceStructuralTagSpec;
using vllm::entrypoints::openai::ToolParser;
using vllm::entrypoints::openai::ToolsEnabled;
using vllm::entrypoints::openai::ReasoningParser;
using vllm::entrypoints::openai::get_reasoning_parser;
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
using vt::DType;

namespace {

// ─── Synthetic weights (mirrors tests/vllm/v1/test_llm_engine.cpp) ───────────
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

// Vocab ids 0..21 — the tiny BPE fixture MINUS the two half-emoji byte tokens
// (ids 22/23). Every id here decodes to valid UTF-8, so a DELTA text fragment is
// never a split multibyte sequence that json::dump() would reject. (A real
// incremental detokenizer buffers incomplete UTF-8; the synthetic argmax over
// random weights would otherwise split the 4-byte emoji across deltas.)
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
       ("vllm_serving_tok_" + std::to_string(counter++) + ".json"))
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

// Owns a fully-wired LLMEngine stack (mirrors test_llm_engine.cpp::Harness).
struct Harness {
  Harness(const HfConfig& c, const Qwen3_5MoeWeights& w, const Tokenizer& tok,
          int max_num_reqs = 8)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), max_num_reqs, kMaxModelLen,
               /*max_num_batched_tokens=*/kMaxModelLen * max_num_reqs),
        executor(runner),
        engine_core(scheduler, executor),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, engine_core, output_processor, Hasher()) {}

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
  EngineCore engine_core;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  LLMEngine engine;
};

// The PRODUCTION serving stack: the AsyncLLM frontend the HTTP server holds
// (examples/server/main.cpp), same model/config/tokenizer as Harness. Used to
// gate use_beam_search over the async serving path (BeamSearchAsync).
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

}  // namespace

namespace {

using vllm::entrypoints::openai::BatchRequestOutput;
using vllm::entrypoints::openai::OpenAIServingModels;
using vllm::entrypoints::openai::RunBatch;

// In-vocab chat-prompt seam (the fixture vocab is ids 0..21; a "role: content"
// join carries out-of-vocab bytes). Same seam test_serving.cpp injects — here it
// concatenates message content so "hello" tokenizes in-vocab.
std::string InVocabChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&) {
  std::string p;
  for (const ChatMessage& m : messages) {
    if (m.content.has_value()) p += *m.content;
  }
  return p;
}

// A well-formed /v1/chat/completions body over the in-vocab fixture.
json ChatBody() {
  return json{{"messages", json::array({json{{"role", "user"},
                                              {"content", "hello"}}})},
              {"max_completion_tokens", 4},
              {"temperature", 0.0}};
}

// The ORDER-ASSERTING prompt seam (ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61
// F1 / ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN): the serving layer hands the
// prompt fn the parsed `tools`, and the pinned renderer dumps their schemas
// into the prompt with sort_keys=False — so the schema's key order AT THIS
// SEAM is the order the prompt carries. The fn records the tools'
// `parameters` serialized order-preserving (ordered_json::dump keeps
// insertion order) and returns an in-vocab string for the engine, the same
// split the api_server capturing seam uses. What it captures is readable;
// what the engine tokenizes stays encodable.
struct CapturingToolsPrompt {
  std::shared_ptr<std::string> rendered = std::make_shared<std::string>();
  vllm::entrypoints::openai::ChatPromptFn fn;

  CapturingToolsPrompt()
      : fn([out = rendered](
               const std::vector<ChatMessage>& messages, bool,
               const std::vector<ChatCompletionToolsParam>& tools,
               const nlohmann::ordered_json&) {
          std::string captured;
          for (const ChatMessage& m : messages) {
            if (m.content.has_value()) captured += *m.content;
          }
          for (const ChatCompletionToolsParam& t : tools) {
            if (t.function.parameters.has_value()) {
              captured += "|";
              captured += t.function.parameters->dump();
            }
          }
          *out = captured;
          return std::string("hello");  // in-vocab for the fixture tokenizer
        }) {}
};

// One BatchRequestInput JSONL line.
std::string BatchLine(const std::string& custom_id, const std::string& url,
                      const json& body) {
  return json{{"custom_id", custom_id},
              {"method", "POST"},
              {"url", url},
              {"body", body}}
      .dump();
}

}  // namespace

// ─── test_empty_file (test_run_batch.py:375) — empty input -> empty output ────
TEST_CASE("run_batch: empty input yields empty output") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  RunBatch runner(&serving);

  CHECK(runner.RunLines("").empty());
  CHECK(runner.Run("").empty());
  // Blank / whitespace-only lines are skipped (run_batch.py:808-811).
  CHECK(runner.RunLines("\n  \n\n").empty());
}

// ─── test_completions (test_run_batch.py:402) — a multi-line chat batch yields
// schema-valid rows IN ORDER with custom_id echoed. RED-first: dropping a
// custom_id or misordering the rows fails these CHECKs. ────────────────────────
TEST_CASE("run_batch: chat batch yields ordered rows, custom_id echoed") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  OpenAIServingModels models("test-model");
  RunBatch runner(&serving, &models);

  const std::string input = BatchLine("request-1", "/v1/chat/completions",
                                      ChatBody()) +
                            "\n" +
                            BatchLine("request-2", "/v1/chat/completions",
                                      ChatBody()) +
                            "\n" +
                            BatchLine("request-3", "/v1/chat/completions",
                                      ChatBody());

  const std::vector<BatchRequestOutput> rows = runner.RunLines(input);
  REQUIRE(rows.size() == 3);
  const char* expected_ids[] = {"request-1", "request-2", "request-3"};
  for (size_t i = 0; i < rows.size(); ++i) {
    CHECK(rows[i].custom_id == expected_ids[i]);  // echoed, in order
    CHECK(!rows[i].custom_id.empty());            // RED-first: never dropped
    CHECK(rows[i].id.rfind("vllm-", 0) == 0);
    REQUIRE(rows[i].response.has_value());
    CHECK(rows[i].response->status_code == 200);
    CHECK(rows[i].response->request_id.rfind("vllm-batch-", 0) == 0);
    REQUIRE(rows[i].response->body.has_value());
    CHECK(rows[i].response->body->contains("choices"));
    CHECK(rows[i].response->body->at("object") == "chat.completion");
    CHECK_FALSE(rows[i].error.has_value());  // success -> error null
  }

  // The serialized JSONL round-trips: every line parses as a BatchRequestOutput
  // shape (id/custom_id/response/error), mirroring the upstream test's
  // BatchRequestOutput.model_validate_json(line) over each output line.
  const std::string out_jsonl = runner.Run(input);
  size_t nlines = 0;
  std::istringstream os(out_jsonl);
  std::string oline;
  while (std::getline(os, oline)) {
    if (oline.empty()) continue;
    const json parsed = json::parse(oline);  // throws if malformed
    CHECK(parsed.contains("id"));
    CHECK(parsed.contains("custom_id"));
    CHECK(parsed.contains("response"));
    CHECK(parsed.contains("error"));
    CHECK(parsed.at("custom_id") == expected_ids[nlines]);
    ++nlines;
  }
  CHECK(nlines == 3);
}

// ─── Per-line isolation: a MALFORMED line becomes an error row and the batch
// CONTINUES (recorded deviation from upstream's abort). RED-first: a runner that
// aborted, dropped the line, or misplaced the row fails these CHECKs. ──────────
TEST_CASE("run_batch: a malformed line yields an error row, batch continues") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  RunBatch runner(&serving);

  const std::string input =
      BatchLine("request-1", "/v1/chat/completions", ChatBody()) + "\n" +
      "this is not valid json {" + "\n" +
      BatchLine("request-3", "/v1/chat/completions", ChatBody());

  const std::vector<BatchRequestOutput> rows = runner.RunLines(input);
  REQUIRE(rows.size() == 3);  // the bad line did NOT abort the batch

  CHECK(rows[0].custom_id == "request-1");
  CHECK_FALSE(rows[0].error.has_value());  // success

  // Middle row: unparsable line -> error row (status 400, string error, empty
  // custom_id since it could not be recovered).
  CHECK(rows[1].custom_id.empty());
  REQUIRE(rows[1].error.has_value());
  CHECK(rows[1].error->is_string());
  REQUIRE(rows[1].response.has_value());
  CHECK(rows[1].response->status_code == 400);
  CHECK_FALSE(rows[1].response->body.has_value());

  CHECK(rows[2].custom_id == "request-3");
  CHECK_FALSE(rows[2].error.has_value());  // batch recovered
}

// ─── test_completions_invalid_input (test_run_batch.py:432) — a line missing the
// required custom_id (the upstream INVALID_INPUT_BATCH shape: `invalid_field`
// instead of `custom_id`). Upstream ABORTS the job; our library isolates it into
// an error row (recorded deviation). ──────────────────────────────────────────
TEST_CASE("run_batch: a line missing custom_id yields an error row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  RunBatch runner(&serving);

  const json bad = json{{"invalid_field", "request-1"},
                        {"method", "POST"},
                        {"url", "/v1/chat/completions"},
                        {"body", ChatBody()}};
  const BatchRequestOutput row = runner.RunLine(bad.dump());
  CHECK(row.custom_id.empty());
  REQUIRE(row.error.has_value());
  CHECK(row.error->is_string());
  REQUIRE(row.response.has_value());
  CHECK(row.response->status_code == 400);
}

// ─── Endpoint dispatch: an unknown URL -> the supported-endpoints error row
// (run_batch.py:832-842), custom_id still echoed. ─────────────────────────────
TEST_CASE("run_batch: an unknown url yields the supported-endpoints error row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  RunBatch runner(&serving);

  const BatchRequestOutput row =
      runner.RunLine(BatchLine("request-x", "/v1/nonsense", ChatBody()));
  CHECK(row.custom_id == "request-x");  // echoed even on error
  REQUIRE(row.error.has_value());
  REQUIRE(row.error->is_string());
  CHECK(row.error->get<std::string>().find("Supported endpoints") !=
        std::string::npos);
}

// ─── An /v1/embeddings line (a registered endpoint key with no serving handler
// wired at W1) -> the "does not support endpoint" error row (handler_getter ->
// None, run_batch.py:600-603). A named residual. ─────────────────────────────
TEST_CASE("run_batch: an embeddings url yields the unsupported-endpoint row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  RunBatch runner(&serving);

  const BatchRequestOutput row =
      runner.RunLine(BatchLine("emb-1", "/v1/embeddings", json::object()));
  CHECK(row.custom_id == "emb-1");
  REQUIRE(row.error.has_value());
  REQUIRE(row.error->is_string());
  CHECK(row.error->get<std::string>().find("does not support endpoint") !=
        std::string::npos);
}

// ─── check_model: an unknown model -> a 404 ErrorResponse error row (the
// ErrorResponse branch, run_batch.py:554-563) carrying the ErrorResponse object,
// not a plain string. ─────────────────────────────────────────────────────────
TEST_CASE("run_batch: an unknown model yields a 404 ErrorResponse row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  OpenAIServingModels models("test-model");
  RunBatch runner(&serving, &models);

  json body = ChatBody();
  body["model"] = "no-such-model";
  const BatchRequestOutput row =
      runner.RunLine(BatchLine("m-1", "/v1/chat/completions", body));
  CHECK(row.custom_id == "m-1");
  REQUIRE(row.response.has_value());
  CHECK(row.response->status_code == 404);
  REQUIRE(row.error.has_value());
  CHECK(row.error->is_object());  // ErrorResponse object, not a bare string
  CHECK(row.error->at("error").at("code") == 404);
}

// ─── The batch entry point keeps the request document's tool-schema key order
// (ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61 F1 +
// ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN). RunLine parses the line with
// nlohmann::json, which SORTS object keys, so DispatchChat must re-read
// `tools` from the body's ORIGINAL text (RestoreToolSchemaOrder) for the
// schema order to survive into the prompt — the pinned renderer dumps it
// with sort_keys=False. This case drives RunLine with a RAW line whose nested
// chat body is NON-alphabetical at every level (built as a raw string:
// BatchLine() would re-serialize through nlohmann::json and sort the keys,
// destroying the very order under test). It pins BOTH halves of the seam:
// (a) an object body dispatches to a 200 row — the committed
// RestoreToolSchemaOrder(body, request) call passed the json OBJECT where a
// string is expected and threw json.exception.type_error.302 on every object
// body; (b) the prompt seam receives the schema in the request document's
// key order, not the key-sorted form. RED evidence: (a) fails with the
// throwing call restored; (b) fails under a body.dump() argument — the dump is
// already key-sorted, so the order restoration reads sorted text and the
// seam no-ops. ──────────────────────────────────────────────────────────────
TEST_CASE("run_batch: a tools line keeps the document's schema key order in "
          "the prompt (ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61,"
          " ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);  // named: runner holds a reference
  Harness h(c, w, Fixture());
  CapturingToolsPrompt prompt;
  OpenAIServingChat serving(h.engine, "test-model", prompt.fn);
  RunBatch runner(&serving);

  // Non-alphabetical at every level: the tool wrapper (type before
  // function), the function (name/description/parameters), and the parameters
  // schema (type/properties/required, zeta before alpha).
  const std::string line =
      R"({"custom_id":"order-1","method":"POST","url":"/v1/chat/completions",)"
      R"("body":{"messages":[{"role":"user","content":"hi"}],)"
      R"("max_completion_tokens":4,"temperature":0.0,)"
      R"("tools":[{"type":"function","function":{"name":"get_weather",)"
      R"("description":"Get the weather for a city.",)"
      R"("parameters":{"type":"object","properties":{)"
      R"("zeta":{"type":"string"},"alpha":{"type":"string"}},)"
      R"("required":["zeta","alpha"]}}}]}})";

  // (a) No exception; the object body dispatches to a 200 row.
  const BatchRequestOutput row = runner.RunLine(line);
  CHECK(row.custom_id == "order-1");
  REQUIRE(row.response.has_value());
  CHECK(row.response->status_code == 200);
  CHECK_FALSE(row.error.has_value());
  REQUIRE(row.response->body.has_value());
  CHECK(row.response->body->contains("choices"));

  // (b) The prompt seam received the schema in the request document's key
  // order (compact ordered_json dump), not the key-sorted form the
  // nlohmann::json parse leaves behind without the order-preserving re-read.
  REQUIRE(prompt.rendered != nullptr);
  CHECK(prompt.rendered->find(
            "|{\"type\":\"object\",\"properties\":{\"zeta\":{\"type\":\"string\"},"
            "\"alpha\":{\"type\":\"string\"}},"
            "\"required\":[\"zeta\",\"alpha\"]}") != std::string::npos);
  CHECK(prompt.rendered->find(
            "|{\"properties\":{\"alpha\":{\"type\":\"string\"},"
            "\"zeta\":{\"type\":\"string\"}},"
            "\"required\":[\"zeta\",\"alpha\"],\"type\":\"object\"}") ==
        std::string::npos);
}
