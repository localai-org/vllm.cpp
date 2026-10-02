// MODEL-TEV1 Phase 6: Tev1 on /v1/systemone and vllm_decide
// (.agents/specs/tev1.md).
//
// Evidence, in order of how much it rests on the references:
//   1. tev1_goldens.inc is produced by RUNNING the references
//      (scripts/gen-tev1-goldens.py): Ollama's decision.Compile and Answer
//      from Go, and togethercomputer/tev1 decide.py payload() rendered by
//      transformers apply_chat_template on both checkpoints. Prompts compare
//      byte for byte and answers to 1e-12.
//   2. Every body Ollama refuses is refused here, and so is the 25-option
//      body that Ollama accepts and the model's contract does not.
//   3. Tev1Decide over a REAL LoadedEngine / AsyncLLM with a synthetic
//      Qwen3.5 dense "Tev1Model", and vllm_decide over that engine's handle,
//      must equal an INDEPENDENT recomputation from ForwardDense's last row
//      at the candidate ids.
// With VLLM_CPP_TEV1_TOKENIZER_DIR pointing at a checkpoint, the prompts must
// also tokenize to the reference's exact ids.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/engine_handle.h"
#include "qwen3_5_decision_fixture.h"
#include "tev1_goldens.inc"
#include "vllm.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/decision_scorer.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/tev1_inference.h"
#include "vllm/tokenizer/tokenizer.h"

namespace {

using ojson = nlohmann::ordered_json;
using qwen3_5_decision_fixture::ByteTokenizer;
using qwen3_5_decision_fixture::MakeConfig;
using qwen3_5_decision_fixture::MakeWeights;
using qwen3_5_decision_fixture::Q;
using vllm::HfConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::Tokenizer;

const ojson& Goldens() {
  static const ojson g = ojson::parse(kTev1Goldens);
  return g;
}

std::string RefusalOf(const ojson& body) {
  try {
    (void)vllm::tev1::CompileRequest(body);
  } catch (const vllm::tev1::RequestError& e) {
    return e.what();
  }
  return "";
}

// Same keys in the same order, strings and objects equal, numbers to 1e-12.
void CheckAnswer(const ojson& got, const ojson& want, double eps) {
  std::vector<std::string> got_keys, want_keys;
  for (auto it = got.begin(); it != got.end(); ++it) got_keys.push_back(it.key());
  for (auto it = want.begin(); it != want.end(); ++it) want_keys.push_back(it.key());
  CHECK(got_keys == want_keys);
  for (auto it = want.begin(); it != want.end(); ++it) {
    const ojson& w = it.value();
    const ojson& g = got[it.key()];
    if (w.is_number()) {
      CHECK(g.get<double>() == doctest::Approx(w.get<double>()).epsilon(eps));
    } else if (w.is_object() && !w.empty() && w.begin()->is_number()) {
      std::vector<std::string> gk, wk;
      for (auto p = g.begin(); p != g.end(); ++p) gk.push_back(p.key());
      for (auto p = w.begin(); p != w.end(); ++p) wk.push_back(p.key());
      CHECK(gk == wk);
      for (auto p = w.begin(); p != w.end(); ++p) {
        CHECK(g[p.key()].get<double>() ==
              doctest::Approx(p.value().get<double>()).epsilon(eps));
      }
    } else {
      CHECK(g == w);
    }
  }
}

EngineParams SyntheticParams() {
  EngineParams p;
  p.block_size = 32;
  p.num_blocks = 256;
  p.max_model_len = 2048;
  p.max_num_seqs = 8;
  return p;
}

std::unique_ptr<LoadedEngine> MakeTev1Engine() {
  const HfConfig c = MakeConfig("Tev1Model");
  return std::make_unique<LoadedEngine>(c, MakeWeights(c), Tokenizer(ByteTokenizer()),
                                        SyntheticParams());
}

// The independent answer: ForwardDense (full logits), its last row, the byte
// ids of "A".. (65..), then the shared answer at T=1.
ojson IndependentAnswers(const ojson& body, int64_t* input_tokens,
                         int candidate_shift = 0) {
  const HfConfig config = MakeConfig("Tev1Model");
  const vllm::Qwen3_5DenseWeights weights = MakeWeights(config);
  const vllm::tev1::Request req = vllm::tev1::CompileRequest(body);
  const std::vector<std::string> prompts = vllm::tev1::BuildPrompts(req);
  vt::Queue q = Q();
  ojson answers = ojson::object();
  *input_tokens = 0;
  for (size_t i = 0; i < req.fields.size(); ++i) {
    const auto& f = req.fields[i];
    const std::vector<int32_t> ids = ByteTokenizer().Encode(prompts[i]);
    *input_tokens += static_cast<int64_t>(ids.size());
    std::vector<int32_t> pos(ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    const std::vector<float> all =
        vllm::Qwen3_5DenseModel::ForwardDense(ids, pos, weights, config, q);
    const size_t last = (ids.size() - 1) * static_cast<size_t>(config.vocab_size);
    std::vector<double> cand;
    for (size_t k = 0; k < f.keys.size(); ++k) {
      cand.push_back(all[last + 65 + k + static_cast<size_t>(candidate_shift)]);
    }
    answers[f.name] = vllm::decision_scorer::AnswerFromLogits(f, cand, 1.0);
  }
  return answers;
}

// The engine runs the paged forward; the reference runs the dense one. They
// agree to float rounding, not bit for bit: the largest relative gap on these
// cases is 1.3e-3 (a 0.0377 probability).
constexpr double kEngineEps = 2e-3;

}  // namespace

// ── 1. Reference goldens ────────────────────────────────────────────────────

TEST_CASE("tev1.prompts.match_decide_py_through_both_chat_templates") {
  for (const ojson& c : Goldens()["cases"]) {
    const std::vector<std::string> prompts =
        vllm::tev1::BuildPrompts(vllm::tev1::CompileRequest(c["body"]));
    REQUIRE(prompts.size() == c["prompts"].size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(prompts[i] == c["prompts"][i].get<std::string>());
    }
  }
}

TEST_CASE("tev1.answers.match_ollama_Answer_through_the_decide_pipeline") {
  for (const ojson& c : Goldens()["cases"]) {
    // A logits source that returns the golden logits, and checks that the
    // pipeline asked for the byte ids of "A".. at the answer boundary.
    const auto golden_logits =
        [&c](const std::vector<std::vector<int32_t>>& prompt_ids,
             const std::vector<std::vector<int32_t>>& candidate_ids) {
          vllm::decision_scorer::CandidateLogits out;
          REQUIRE(prompt_ids.size() == c["logits"].size());
          for (size_t i = 0; i < candidate_ids.size(); ++i) {
            for (size_t k = 0; k < candidate_ids[i].size(); ++k) {
              CHECK(candidate_ids[i][k] == static_cast<int32_t>(65 + k));
            }
            out.logits.push_back(c["logits"][i].get<std::vector<double>>());
          }
          out.output_tokens = static_cast<int64_t>(prompt_ids.size());
          return out;
        };
    const vllm::Tev1Response got =
        vllm::Tev1DecideWith(ByteTokenizer(), 1 << 20, c["body"], golden_logits);
    const ojson& want = c["answers"];
    REQUIRE(got.answers.size() == want.size());
    for (auto it = want.begin(); it != want.end(); ++it) {
      INFO(it.key());
      CheckAnswer(got.answers[it.key()], it.value(), 1e-12);
    }
    CHECK(got.output_tokens == static_cast<int64_t>(want.size()));
  }
}

// ── 2. Refusals ─────────────────────────────────────────────────────────────

TEST_CASE("tev1.request.refuses_what_ollama_refuses_and_wider_than_24") {
  for (const ojson& r : Goldens()["refusals"]) {
    INFO(r["body"].dump().substr(0, 120));
    const std::string msg = RefusalOf(r["body"]);
    CHECK_FALSE(msg.empty());
    if (r["ollama_error"].is_null()) {
      // Ollama serves 26 candidates; the model documents 24 (decide.py).
      CHECK(msg.find("25 choices") != std::string::npos);
      CHECK(msg.find("2-24 options") != std::string::npos);
    }
  }
  // 24 is the widest accepted field.
  ojson wide = ojson::parse(
      R"({"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{}}}})");
  for (int i = 0; i < 24; ++i) wide["questions"]["a"]["criteria"]["k" + std::to_string(i)] = nullptr;
  CHECK(RefusalOf(wide).empty());
  // decide.py: every option needs a nonempty description.
  CHECK(RefusalOf(ojson::parse(
            R"({"state":"s","questions":{"a":{"type":"noul","instructions":"x","criteria":{"true":" "}}}})"))
            .find("nonempty description for every option") != std::string::npos);
  CHECK(RefusalOf(ojson::parse(R"([1])")).find("object") != std::string::npos);
}

TEST_CASE("tev1.decide.refuses_an_overlong_prompt_without_truncating") {
  const ojson body = Goldens()["cases"][0]["body"];
  std::string msg;
  try {
    (void)vllm::Tev1DecideWith(
        ByteTokenizer(), 100, body,
        [](const auto&, const auto&) -> vllm::decision_scorer::CandidateLogits {
          FAIL("the scorer must not run for a refused request");
          return {};
        });
  } catch (const vllm::tev1::RequestError& e) {
    msg = e.what();
  }
  // max_model_len 100 leaves 99 prompt tokens for the one sampled token.
  CHECK(msg.find("limit is 99. Nothing was truncated.") != std::string::npos);
}

// ── 3. Reachability through the engine and the C ABI ───────────────────────

TEST_CASE("tev1.decide.scores_through_the_engine_like_an_independent_forward") {
  auto engine = MakeTev1Engine();
  REQUIRE(engine->architecture() == "Tev1Model");
  for (const ojson& c : Goldens()["cases"]) {
    const ojson& body = c["body"];
    const ojson first_question = body["questions"].front();
    if (first_question.contains("criteria") && first_question["criteria"].size() > 3) {
      continue;  // the 24-option case is covered by the goldens; keep this fast
    }
    const vllm::Tev1Response got = vllm::Tev1Decide(
        engine->async_engine(), engine->tokenizer(), engine->max_model_len(), body);
    int64_t tokens = 0;
    const ojson want = IndependentAnswers(body, &tokens);
    INFO(body.dump().substr(0, 80));
    REQUIRE(got.answers.size() == want.size());
    for (auto it = want.begin(); it != want.end(); ++it) {
      CheckAnswer(got.answers[it.key()], it.value(), kEngineEps);
    }
    CHECK(got.input_tokens == tokens);
    CHECK(got.output_tokens == static_cast<int64_t>(want.size()));

    // The candidate ids matter: the neighbouring ids give a different answer.
    int64_t unused = 0;
    const ojson shifted = IndependentAnswers(body, &unused, /*candidate_shift=*/1);
    const ojson first = want.front();
    const ojson first_shifted = shifted.front();
    const char* key = first.contains("noul") ? "noul" : "confidence";
    CHECK(std::abs(first[key].get<double>() - first_shifted[key].get<double>()) > 1e-4);
  }
}

TEST_CASE("tev1.decide.is_reachable_through_vllm_decide") {
  vllm_engine* eng = vllm::capi::MakeEngineHandle(MakeTev1Engine());
  REQUIRE(eng != nullptr);
  const ojson body = Goldens()["cases"][1]["body"];
  char* out = nullptr;
  REQUIRE(vllm_decide(eng, body.dump().c_str(), &out) == VLLM_OK);
  REQUIRE(out != nullptr);
  const ojson got = ojson::parse(out);
  vllm_decide_free(out);
  int64_t tokens = 0;
  const ojson want = IndependentAnswers(body, &tokens);
  CHECK(got["model"] == "tev1");
  for (auto it = want.begin(); it != want.end(); ++it) {
    CheckAnswer(got["answers"][it.key()], it.value(), kEngineEps);
  }
  CHECK(got["usage"]["input_tokens"].get<int64_t>() == tokens);
  CHECK(got["usage"]["output_tokens"].get<int64_t>() == 3);

  // A refused request is VLLM_ERR_INVALID_ARGUMENT, not an engine fault.
  out = nullptr;
  CHECK(vllm_decide(eng, R"({"state":" ","questions":{"a":{"type":"noul","instructions":"x"}}})",
                    &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(out == nullptr);
  vllm_engine_free(eng);
}

TEST_CASE("tev1.decide.a_plain_qwen3_5_engine_is_refused_by_name") {
  const HfConfig c = MakeConfig("Qwen3_5ForConditionalGeneration");
  vllm_engine* eng = vllm::capi::MakeEngineHandle(std::make_unique<LoadedEngine>(
      c, MakeWeights(c), Tokenizer(ByteTokenizer()), SyntheticParams()));
  REQUIRE(eng != nullptr);
  char* out = nullptr;
  const std::string body = Goldens()["cases"][0]["body"].dump();
  CHECK(vllm_decide(eng, body.c_str(), &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("'Tev1Model'") != std::string::npos);
  vllm_engine_free(eng);
}

TEST_CASE("tev1.decide.Tev1Model_is_a_generation_model") {
  const auto& reg = vllm::RegistrationFor("Tev1Model");
  CHECK(reg.info.is_text_generation_model);
  CHECK_FALSE(reg.info.is_pooling_model);
}

// ── 4. Real tokenizer (opt-in) ──────────────────────────────────────────────

TEST_CASE("tev1.tokens.match_the_reference_ids_with_the_checkpoint_tokenizer") {
  const char* dir = std::getenv("VLLM_CPP_TEV1_TOKENIZER_DIR");
  if (dir == nullptr) {
    MESSAGE("skipped: set VLLM_CPP_TEV1_TOKENIZER_DIR to the tokenizer of "
            "togethercomputer/Tev1-4B-experimental @ 0b7becf0 or "
            "Tev1-0.8B-experimental @ 6bb2dff1");
    return;
  }
  std::ifstream in(std::filesystem::path(dir) / "tokenizer.json", std::ios::binary);
  REQUIRE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const Tokenizer tok = Tokenizer::FromHfJsonBytes(ss.str(), dir);
  for (const ojson& c : Goldens()["cases"]) {
    const vllm::tev1::Request req = vllm::tev1::CompileRequest(c["body"]);
    const std::vector<std::string> prompts = vllm::tev1::BuildPrompts(req);
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(tok.Encode(prompts[i]) == c["ids"][i].get<std::vector<int32_t>>());
      CHECK(vllm::decision_scorer::CandidateIds(tok, vllm::tev1::kAnswerBoundary,
                                                req.fields[i].keys.size(), "tev1") ==
            c["candidate_ids"][i].get<std::vector<int32_t>>());
    }
  }
}
