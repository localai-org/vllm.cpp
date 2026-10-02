// Tests for the V1 InputProcessor text path
// (vllm/v1/engine/input_processor.py::process_inputs @ e24d1b24).
//
// M1.8 Task 2 is a pure host-logic port (no goldens). The InputProcessor is the
// "constructing unit" that runs the SamplingParams __post_init__ (PostInit) our
// M1.1 SamplingParams deferred. Cases:
//   (a) text prompt   -> EngineCoreRequest.prompt_token_ids == tokenizer.Encode
//                        (the byte-level BPE path); request_id + arrival_time set.
//   (b) PostInit ran  -> a near-zero temperature is clamped to _MAX_TEMP; a
//                        greedy (temperature 0) params has its top_p/top_k/min_p
//                        normalized. (Verify() alone would not normalize these.)
//   (c) invalid params-> Verify (run inside PostInit) throws on an out-of-range
//                        field (e.g. top_p > 1).
//   (d) max_tokens    -> unset (nullopt) defaults to max_model_len - prompt_len.
//   (e) eos/stop      -> primary eos is written to sampling_params.eos_token_id
//                        unless ignore_eos; a secondary eos id (a list config)
//                        is merged into stop_token_ids unless ignore_eos.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <memory>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/inputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::SamplingParams;
using vllm::SamplingType;
using vllm::multimodal::AudioKwargs;
using vllm::multimodal::ImageKwargs;
using vllm::multimodal::MultiModalFeatureSpec;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCoreRequest;
using vllm::v1::InputProcessor;
using vllm::v1::Request;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36";

// One tokenizer load for the whole binary (the golden tokenizer.json is ~20MB).
const Tokenizer& GoldenTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJson(kGoldenDir + "/tokenizer.json");
  return tok;
}

// Builds a minimal HfConfig carrying just what the InputProcessor reads:
// max_position_embeddings (-> max_model_len) and raw["eos_token_id"].
HfConfig MakeConfig(int64_t max_len, const json& eos_token_id) {
  HfConfig cfg;
  cfg.model_type = "qwen3_5_moe";
  cfg.hidden_size = 8;
  cfg.num_hidden_layers = 1;
  cfg.max_position_embeddings = max_len;
  cfg.raw = json::object();
  if (!eos_token_id.is_null()) {
    cfg.raw["eos_token_id"] = eos_token_id;
  }
  return cfg;
}

}  // namespace

TEST_CASE("process_inputs tokenizes the text prompt (byte-level BPE path)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, /*eos=*/json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  SamplingParams params;  // defaults: max_tokens=16
  EngineCoreRequest req = proc.process_inputs("req-1", prompt, params,
                                              /*arrival_time=*/123.0);

  CHECK(req.request_id == "req-1");
  CHECK(req.arrival_time == doctest::Approx(123.0));
  CHECK(req.prompt_token_ids == tok.Encode(prompt));
  CHECK_FALSE(req.prompt_token_ids.empty());
  // Default max_tokens is preserved when the caller sets it.
  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens == 16);
}

TEST_CASE("process_inputs runs SamplingParams PostInit (normalization)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SUBCASE("near-zero temperature is clamped up to _MAX_TEMP") {
    SamplingParams params;
    params.temperature = 0.005;  // in (0, _MAX_TEMP=1e-2)
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(vllm::kMaxTemp));
  }

  SUBCASE("greedy (temperature 0) normalizes top_p/top_k/min_p") {
    SamplingParams params;
    params.temperature = 0.0;
    params.top_p = 0.5;
    params.top_k = 20;
    params.min_p = 0.3;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(0.0));
    CHECK(req.sampling_params.top_p == doctest::Approx(1.0));
    CHECK(req.sampling_params.top_k == 0);
    CHECK(req.sampling_params.min_p == doctest::Approx(0.0));
    CHECK(req.sampling_params.Type() == SamplingType::kGreedy);
  }
}

TEST_CASE("process_inputs throws on invalid SamplingParams (Verify)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams bad;
  bad.top_p = 2.0;  // must be in (0, 1]
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", bad), std::runtime_error);

  SamplingParams greedy_multi;
  greedy_multi.temperature = 0.0;
  greedy_multi.n = 2;  // n must be 1 under greedy sampling
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", greedy_multi),
                  std::runtime_error);
}

TEST_CASE("process_inputs defaults max_tokens from max_model_len - prompt_len") {
  const Tokenizer& tok = GoldenTokenizer();
  const int64_t kMaxLen = 4096;
  HfConfig cfg = MakeConfig(kMaxLen, json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  const int prompt_len = static_cast<int>(tok.Encode(prompt).size());

  SamplingParams params;
  params.max_tokens = std::nullopt;  // "unset" -> generate up to max_model_len
  EngineCoreRequest req = proc.process_inputs("r", prompt, params);

  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens ==
        static_cast<int>(kMaxLen) - prompt_len);
}

TEST_CASE("process_inputs wires eos/stop token ids") {
  const Tokenizer& tok = GoldenTokenizer();

  SUBCASE("scalar eos is written to sampling_params.eos_token_id") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 151643);
    // A single eos id has no secondary ids -> stop_token_ids untouched.
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("ignore_eos leaves eos_token_id unset and stop_token_ids untouched") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("list eos: primary is eos_token_id, secondary merged into stop ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.stop_token_ids = {200, 999};  // 200 dedups against the eos ids
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 100);  // first element = primary
    // secondary eos ids {200, 300} unioned with existing {200, 999}.
    std::vector<int32_t> expected = {200, 300, 999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }

  // ─── generation_config.json eos ids (sampling_params.py:645-655) ───────────
  // Upstream's update_from_generation_config draws its secondary eos ids from
  // generation_config.json, NOT config.json. Gemma-4-26B ships
  //   config.json            eos: [1, 106]
  //   generation_config.json eos: [1, 106, 50]
  // so a port that reads only config.json never stops on 50.
  SUBCASE("generation_config eos ids merge into stop_token_ids") {
    HfConfig cfg = MakeConfig(4096, json::array({1, 106}));
    cfg.generation_config_eos_ids = {1, 106, 50};
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    // The PRIMARY eos still comes from config.json's first element: the
    // generation_config union must not reorder or displace it.
    CHECK(*req.sampling_params.eos_token_id == 1);
    std::vector<int32_t> expected = {50, 106};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }

  SUBCASE("generation_config eos ids still respect ignore_eos") {
    HfConfig cfg = MakeConfig(4096, json::array({1, 106}));
    cfg.generation_config_eos_ids = {1, 106, 50};
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    CHECK(req.sampling_params.stop_token_ids.empty());
    // ... but they DO feed min_tokens masking regardless (sampling_params.py:653).
    CHECK(req.sampling_params.all_stop_token_ids.count(50) == 1);
  }

  SUBCASE("list eos with ignore_eos does not touch stop_token_ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    params.stop_token_ids = {999};
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    std::vector<int32_t> expected = {999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }
}

// ─── ROAD-V1-C7 SAMPLE-LOGIT-FILTERS: bad_words + all_stop_token_ids wiring ───
TEST_CASE("process_inputs tokenizes bad_words into bad_words_token_ids") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  params.bad_words = {"hello"};
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);

  // update_from_tokenizer produces the per-word n-grams (sampling_params.py:659).
  REQUIRE(req.sampling_params.bad_words_token_ids.has_value());
  const auto& bwti = *req.sampling_params.bad_words_token_ids;
  REQUIRE_FALSE(bwti.empty());
  // The no-prefix variant is the encode of the lstripped word.
  CHECK(bwti[0] == tok.Encode("hello"));
  // Every produced id is in range.
  const int32_t max_id = tok.VocabSize() - 1;
  for (const auto& ngram : bwti) {
    for (int32_t id : ngram) {
      CHECK(id >= 0);
      CHECK(id <= max_id);
    }
  }
}

TEST_CASE("process_inputs seeds all_stop_token_ids (stop_token_ids + eos)") {
  const Tokenizer& tok = GoldenTokenizer();
  // A primary eos id via the tokenizer fallback is used when config eos is null;
  // pass an explicit eos list to make the assertion deterministic.
  HfConfig cfg = MakeConfig(4096, json({1000, 1001}));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  params.stop_token_ids = {55};
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);

  // all_stop_token_ids = stop_token_ids (PostInit) + eos ids (generation config).
  const auto& all = req.sampling_params.all_stop_token_ids;
  CHECK(all.count(55) == 1);
  CHECK(all.count(1000) == 1);
  CHECK(all.count(1001) == 1);
}

TEST_CASE("process_inputs: empty bad_words leaves bad_words_token_ids unset") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);
  SamplingParams params;  // no bad_words
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);
  CHECK_FALSE(req.sampling_params.bad_words_token_ids.has_value());
}

// ─── ROAD-V1-MM MM-SERVE-ENGINE: process_inputs_mm carries mm_features ───────
//
// The mm request path (input_processor.py:333-379): the prompt ids are the
// ALREADY placeholder-expanded stream (the serving-side processor ran) and the
// per-item mm_features ride onto the EngineCoreRequest alongside them. This is
// the CPU-verifiable half — everything UP TO the mm forward (which consumes
// mm_features on the GPU worker, MM-SERVE-E2E). RED-first: before this brick the
// engine had NO mm add_request path (process_inputs_tokens drops mm data);
// after, the request provably carries the mm handles + the expanded prompt.
TEST_CASE("process_inputs_mm carries mm_features + the expanded prompt") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(151643));
  InputProcessor proc(tok, cfg);

  // A placeholder-EXPANDED prompt: 2 real ids + 4 image_pad slots + 1 real id.
  const int32_t kImagePad = 151655;
  std::vector<int32_t> expanded = {5, 6, kImagePad, kImagePad,
                                   kImagePad, kImagePad, 7};

  // One image feature spec (the RouteImageRgb output shape): span [2,4] + an
  // opaque ImageKwargs handle + the mm-hash.
  MultiModalFeatureSpec spec;
  spec.mm_hash = "deadbeefcafe";
  spec.modality = "image";
  spec.offset = 2;
  spec.length = 4;
  auto kw = std::make_shared<ImageKwargs>();
  kw->num_patches = 16;
  kw->patch_feature_dim = 8;
  kw->image_grid_thw = {1, 8, 8};
  spec.data = kw;
  std::vector<MultiModalFeatureSpec> mm_features = {spec};

  SamplingParams params;  // default max_tokens=16
  EngineCoreRequest req = proc.process_inputs_mm(
      "mm-1", expanded, mm_features, params, /*arrival_time=*/42.0);

  // The EXPANDED prompt ids pass through verbatim (no tokenization).
  CHECK(req.request_id == "mm-1");
  CHECK(req.arrival_time == doctest::Approx(42.0));
  CHECK(req.prompt_token_ids == expanded);

  // The mm_features are carried onto the request (the handles + span + hash).
  REQUIRE(req.mm_features.size() == 1);
  CHECK(req.mm_features[0].modality == "image");
  CHECK(req.mm_features[0].mm_hash == "deadbeefcafe");
  CHECK(req.mm_features[0].offset == 2);
  CHECK(req.mm_features[0].length == 4);
  REQUIRE(req.mm_features[0].data != nullptr);
  CHECK(req.mm_features[0].data->num_patches == 16);
  CHECK(req.mm_features[0].data->image_grid_thw[1] == 8);

  // The eos/stop wiring is byte-for-byte the tokens path (scalar eos written).
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 151643);

  // Request::FromEngineCoreRequest threads mm_features onto the built Request —
  // the exact handoff LLMEngine/AsyncLLM::add_request(MultiModalInputs) perform
  // before enqueueing to the EngineCore (the encoder-cache / vision consumer).
  Request built = Request::FromEngineCoreRequest(req);
  REQUIRE(built.mm_features.size() == 1);
  CHECK(built.mm_features[0].mm_hash == "deadbeefcafe");
  CHECK(built.mm_features[0].length == 4);
  CHECK(built.prompt_token_ids == expanded);
}

TEST_CASE("process_inputs_mm with EMPTY mm_features == the tokens path") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(151643));
  InputProcessor proc(tok, cfg);

  std::vector<int32_t> ids = {10, 11, 12, 13};
  SamplingParams params;
  EngineCoreRequest mm_req = proc.process_inputs_mm(
      "r", ids, /*mm_features=*/{}, params, /*arrival_time=*/7.0);
  EngineCoreRequest tok_req =
      proc.process_inputs_tokens("r", ids, params, /*arrival_time=*/7.0);

  // INERTNESS: an empty mm_features request is identical to the tokens request
  // (same ids, same eos/max_tokens wiring) with an empty mm_features vector.
  CHECK(mm_req.mm_features.empty());
  CHECK(mm_req.prompt_token_ids == tok_req.prompt_token_ids);
  CHECK(mm_req.sampling_params.max_tokens == tok_req.sampling_params.max_tokens);
  CHECK(mm_req.sampling_params.eos_token_id ==
        tok_req.sampling_params.eos_token_id);
}

// ---------------------------------------------------------------------------
// _validate_prompt_len (input_processor.py:387-432 @ 555967922).
//
// A prompt the engine can never serve must be REJECTED at admission, not left
// to wedge the scheduler. Upstream raises ValueError, which the OpenAI server
// turns into HTTP 400 (serve/utils/error_response.py:62-65). We raise the typed
// InputValidationError so the server can map it the same way.
// ---------------------------------------------------------------------------
TEST_CASE("process_inputs rejects a prompt longer than max_model_len") {
  const Tokenizer& tok = GoldenTokenizer();
  // 8 tokens of context. Any real sentence exceeds it.
  HfConfig cfg = MakeConfig(/*max_len=*/8, json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt =
      "The capital of Germany is Berlin, and the capital of France is Paris.";
  REQUIRE(static_cast<int>(tok.EncodeWithSpecialTokens(prompt).size()) > 8);

  SamplingParams params;
  CHECK_THROWS_AS(proc.process_inputs("r", prompt, params),
                  vllm::v1::InputValidationError);

  // The message names the number a caller needs to act on
  // (input_processor.py:417-421).
  try {
    proc.process_inputs("r", prompt, params);
    FAIL("expected InputValidationError");
  } catch (const vllm::v1::InputValidationError& e) {
    const std::string msg = e.what();
    CHECK(msg.find("longer than the maximum model length of 8") !=
          std::string::npos);
    CHECK(msg.find("max_model_len") != std::string::npos);
  }
}

TEST_CASE(
    "process_inputs rejects a prompt EQUAL to max_model_len (no room to "
    "generate)") {
  // input_processor.py:423-432: prompt_len == max_prompt_len is also an error
  // for a generate runner, because at least one output token is requested.
  const Tokenizer& tok = GoldenTokenizer();
  const std::vector<int32_t> ids = {10, 11, 12, 13};
  HfConfig cfg = MakeConfig(/*max_len=*/4, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  CHECK_THROWS_AS(proc.process_inputs_tokens("r", ids, params),
                  vllm::v1::InputValidationError);
  try {
    proc.process_inputs_tokens("r", ids, params);
    FAIL("expected InputValidationError");
  } catch (const vllm::v1::InputValidationError& e) {
    CHECK(std::string(e.what()).find("requested output tokens") !=
          std::string::npos);
  }
}

TEST_CASE("process_inputs rejects an EMPTY prompt") {
  // input_processor.py:395-396: "The decoder prompt cannot be empty".
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  CHECK_THROWS_AS(proc.process_inputs_tokens("r", {}, params),
                  vllm::v1::InputValidationError);
}

TEST_CASE("process_inputs admits a prompt that fits, unchanged") {
  // The guard must not perturb the ordinary path: a short prompt under a large
  // context still produces the request it did before the check existed.
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  const EngineCoreRequest req = proc.process_inputs("r", "hello", params);
  CHECK(req.prompt_token_ids == tok.EncodeWithSpecialTokens("hello"));

  // The mm path applies the same check (upstream validates the decoder prompt
  // regardless of modality) and admits the same ids.
  const EngineCoreRequest mm = proc.process_inputs_mm(
      "r", tok.EncodeWithSpecialTokens("hello"), /*mm_features=*/{}, params);
  CHECK(mm.prompt_token_ids == req.prompt_token_ids);
}

TEST_CASE(
    "process_inputs prompt-length guard is inert when max_model_len is "
    "unknown") {
  // HARNESS ADAPTATION (no upstream counterpart): our HfConfig may carry no
  // max_position_embeddings at all (0), which means "unknown", not "zero
  // context". Upstream's ModelConfig always resolves a positive max_model_len so
  // it has no such state; checking against 0 would reject every prompt. Mirrors
  // upstream's skip_prompt_length_check early-out (input_processor.py:392-393).
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/0, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  params.max_tokens = 4;
  const EngineCoreRequest req =
      proc.process_inputs("r", "hello world", params);
  CHECK_FALSE(req.prompt_token_ids.empty());
}

TEST_CASE("InputProcessor honours an explicit max_model_len override") {
  // LoadedEngine resolves max_model_len from EngineParams::max_model_len (and,
  // when unset, auto-fits it to the KV pool). The processor must validate
  // against THAT resolved length, not the raw checkpoint context — otherwise
  // the admission guard and the startup guard disagree and a prompt the pool
  // cannot hold is still admitted. Upstream reads model_config.max_model_len,
  // which is the already-resolved value (input_processor.py:399-401).
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, json(nullptr));
  InputProcessor proc(tok, cfg, /*max_model_len_override=*/8);

  SamplingParams params;
  CHECK_THROWS_AS(
      proc.process_inputs(
          "r", "The capital of Germany is Berlin, and France's is Paris.",
          params),
      vllm::v1::InputValidationError);
}

// ─── #249: an out-of-range logprobs request must be REJECTED, not crash ──────
// GatherLogprobs partial_sorts k entries out of a vocab-sized index array, so an
// unbounded k walked off the end and killed the server process — reachable from
// a single unauthenticated POST with {"logprobs": 999999}. Upstream
// _validate_logprobs (sampling_params.py:755-770) REJECTS rather than clamps,
// and this mirrors that.
TEST_CASE("process_inputs rejects logprobs above the vocabulary cap (#249)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);
  const int vocab = static_cast<int>(tok.VocabSize());

  SUBCASE("sample logprobs far above vocab throws instead of crashing") {
    SamplingParams params;
    params.logprobs = 999999;
    CHECK_THROWS_AS(proc.process_inputs("r", "hi", params), std::runtime_error);
  }

  SUBCASE("prompt logprobs above vocab throws too") {
    SamplingParams params;
    params.prompt_logprobs = vocab + 1;
    CHECK_THROWS_AS(proc.process_inputs("r", "hi", params), std::runtime_error);
  }

  SUBCASE("exactly vocab is ACCEPTED — the cap is inclusive, as upstream's is") {
    SamplingParams params;
    params.logprobs = vocab;
    CHECK_NOTHROW(proc.process_inputs("r", "hi", params));
  }

  SUBCASE("an ordinary small request is unaffected") {
    SamplingParams params;
    params.logprobs = 5;
    CHECK_NOTHROW(proc.process_inputs("r", "hi", params));
  }
}

// ─── MODEL-TEV1 Phase 7: the tokenizer's EOS when the checkpoint names none ───
// ISSUE-LOCAL-01M3RTGVTN34YQFBR1KZH117XA, .agents/specs/tev1-eos-fallback.md.
// vLLM takes the PRIMARY eos from the tokenizer (renderers/base.py:310-317,
// tokenizer_config.json eos_token) and the secondary ids from
// generation_config.json, or from GenerationConfig.from_model_config when that
// file is absent (transformers_utils/config.py:1066-1090 @ 5559679229). These
// cases load real sibling files, as a checkpoint lays them out, and use the
// Qwen3.6 tokenizer: 248044 <|endoftext|>, 248046 <|im_end|>.
namespace {

class EosModelDir {
 public:
  EosModelDir(const std::string& config, const std::string& gen,
              const std::string& tokenizer_config) {
    static int counter = 0;
    dir_ = (std::filesystem::temp_directory_path() /
            ("vllm_eos_fallback_" + std::to_string(counter++)))
               .string();
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ + "/config.json", std::ios::binary) << config;
    if (!gen.empty()) {
      std::ofstream(dir_ + "/generation_config.json", std::ios::binary) << gen;
    }
    if (!tokenizer_config.empty()) {
      std::ofstream(dir_ + "/tokenizer_config.json", std::ios::binary)
          << tokenizer_config;
    }
  }
  ~EosModelDir() { std::filesystem::remove_all(dir_); }
  std::string config_path() const { return dir_ + "/config.json"; }

 private:
  std::string dir_;
};

// A nested Qwen3.5 wrapper, the shape both Tev1 checkpoints ship.
std::string NestedConfig(const std::string& top_level_eos) {
  return std::string(R"({"architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",)") + top_level_eos + R"(
    "text_config": {"model_type": "qwen3_5_text", "hidden_size": 8,
      "num_hidden_layers": 1, "vocab_size": 248320, "num_attention_heads": 2,
      "num_key_value_heads": 1, "max_position_embeddings": 4096,
      "eos_token_id": 248044}})";
}

EngineCoreRequest ProcessWith(const std::string& config_path,
                              SamplingParams params = SamplingParams{}) {
  const HfConfig cfg = vllm::LoadHfConfig(config_path);
  InputProcessor proc(GoldenTokenizer(), cfg);
  return proc.process_inputs("r", "hi", params);
}

}  // namespace

TEST_CASE("eos fallback: a checkpoint that lists its ids keeps its stop set") {
  // Pinned BEFORE the change: a top-level eos plus a generation_config.json,
  // and a tokenizer_config.json whose eos_token is a DIFFERENT id. Neither the
  // primary id nor stop_token_ids may move.
  EosModelDir model(NestedConfig(R"("eos_token_id": 248044,)"),
                    R"({"eos_token_id": [248044, 248045]})",
                    R"({"eos_token": "<|im_end|>"})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 248044);
  CHECK(req.sampling_params.stop_token_ids == std::vector<int32_t>{248045});
}

TEST_CASE("eos fallback: a top-level eos without generation_config.json keeps its stop set") {
  // from_model_config lets the OUTER eos_token_id win over text_config's, so a
  // nested checkpoint that names 248046 at the top and 248044 inside, with no
  // generation_config.json, stops on 248046 alone, before and after the change.
  EosModelDir model(NestedConfig(R"("eos_token_id": 248046,)"), "",
                    R"({"eos_token": "<|endoftext|>"})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 248046);
  CHECK(req.sampling_params.stop_token_ids.empty());
}

TEST_CASE("eos fallback: the Tev1 shape stops on the tokenizer's eos_token") {
  // No top-level eos, no generation_config.json, eos_token <|im_end|>: vLLM's
  // primary is 248046 and from_model_config adds the text config's 248044.
  EosModelDir model(NestedConfig(""), "", R"({"eos_token": "<|im_end|>"})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 248046);
  CHECK(req.sampling_params.stop_token_ids == std::vector<int32_t>{248044});

  SUBCASE("ignore_eos still suppresses both") {
    SamplingParams p;
    p.ignore_eos = true;
    const EngineCoreRequest r = ProcessWith(model.config_path(), p);
    CHECK_FALSE(r.sampling_params.eos_token_id.has_value());
    CHECK(r.sampling_params.stop_token_ids.empty());
  }
}

TEST_CASE("eos fallback: the object form of eos_token is read") {
  EosModelDir model(NestedConfig(""), "",
                    R"({"eos_token": {"content": "<|im_end|>", "special": true}})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 248046);
}

TEST_CASE("eos fallback: a present generation_config.json wins over the text config") {
  // try_get_generation_config reads the file when it exists and only falls
  // back to from_model_config when it does not.
  EosModelDir model(NestedConfig(""), R"({"eos_token_id": 248045})",
                    R"({"eos_token": "<|im_end|>"})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 248046);
  CHECK(req.sampling_params.stop_token_ids == std::vector<int32_t>{248045});
}

TEST_CASE("eos fallback: an eos_token that is not one token is not guessed") {
  EosModelDir model(NestedConfig(""), R"({"eos_token_id": 248045})",
                    R"({"eos_token": "not one token"})");
  const EngineCoreRequest req = ProcessWith(model.config_path());
  CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
  CHECK(req.sampling_params.stop_token_ids == std::vector<int32_t>{248045});
}
