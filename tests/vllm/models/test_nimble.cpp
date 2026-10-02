// MODEL-NIMBLE: the Nimble decision lane (.agents/specs/nimble.md).
//
// Three kinds of evidence, in order of how much they rest on the reference
// (the synthetic model and tokenizer are qwen3_5_decision_fixture.h):
//   1. nimble_goldens.inc is produced by RUNNING the model author's prompt code
//      (parallel_schema.py via compiler.py) and openjev's scoring.answer
//      (scripts/gen-nimble-goldens.py). The prompt and answer cases compare
//      against it byte for byte and to 1e-12.
//   2. The request refusals are the reference's own error cases.
//   3. NimbleDecide, the seam /v1/systemone and vllm_decide both call, runs
//      over a synthetic Qwen3.5 dense model and a byte-level tokenizer, and its
//      answers must equal an INDEPENDENT recomputation from ForwardDense's
//      full-logits last row at the candidate ids.
// With VLLM_CPP_NIMBLE_TOKENIZER_DIR pointing at the checkpoint's tokenizer,
// the prompts must also tokenize to the reference's exact ids.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "nimble_goldens.inc"
#include "qwen3_5_decision_fixture.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nimble_inference.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace {

using ojson = nlohmann::ordered_json;
using vllm::HfConfig;
using vllm::Qwen3_5DenseWeights;
using vllm::tok::Tokenizer;

const ojson& Goldens() {
  static const ojson g = ojson::parse(kNimbleGoldens);
  return g;
}

HfConfig MakeConfig() { return qwen3_5_decision_fixture::MakeConfig("NimbleModel"); }
using qwen3_5_decision_fixture::ByteTokenizer;
using qwen3_5_decision_fixture::MakeWeights;
using qwen3_5_decision_fixture::Q;

std::string RefusalOf(const ojson& body) {
  try {
    (void)vllm::nimble::CompileRequest(body);
  } catch (const vllm::nimble::RequestError& e) {
    return e.what();
  }
  return "";
}

ojson Body(const char* text) { return ojson::parse(text); }

}  // namespace

// ── 1. Reference goldens ────────────────────────────────────────────────────

TEST_CASE("nimble.prompts.match_the_reference_prepare_prompts") {
  for (const ojson& c : Goldens()["cases"]) {
    const vllm::nimble::Request r = vllm::nimble::CompileRequest(c["body"]);
    const std::vector<std::string> prompts = vllm::nimble::BuildPrompts(r);
    REQUIRE(prompts.size() == c["prompts"].size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(prompts[i] == c["prompts"][i].get<std::string>());
    }
  }
}

TEST_CASE("nimble.answers.match_openjev_scoring") {
  for (const ojson& c : Goldens()["cases"]) {
    const vllm::nimble::Request r = vllm::nimble::CompileRequest(c["body"]);
    for (const ojson& a : c["answers"]) {
      const std::string name = a["field"].get<std::string>();
      const vllm::nimble::Field* field = nullptr;
      for (const auto& f : r.fields)
        if (f.name == name) field = &f;
      REQUIRE(field != nullptr);
      const ojson got = vllm::nimble::AnswerFromLogits(
          *field, a["logits"].get<std::vector<double>>(),
          a["temperature"].get<double>());
      const ojson& want = a["expected"];
      INFO(name << " T=" << a["temperature"].get<double>());
      // Same keys in the same order as the pydantic model.
      std::vector<std::string> got_keys, want_keys;
      for (auto it = got.begin(); it != got.end(); ++it) got_keys.push_back(it.key());
      for (auto it = want.begin(); it != want.end(); ++it) want_keys.push_back(it.key());
      CHECK(got_keys == want_keys);
      for (auto it = want.begin(); it != want.end(); ++it) {
        const ojson& w = it.value();
        const ojson& g = got[it.key()];
        if (w.is_number_float()) {
          CHECK(g.get<double>() == doctest::Approx(w.get<double>()).epsilon(1e-12));
        } else if (w.is_object() && !w.empty() && w.begin()->is_number()) {
          for (auto p = w.begin(); p != w.end(); ++p) {
            CHECK(g[p.key()].get<double>() ==
                  doctest::Approx(p.value().get<double>()).epsilon(1e-12));
          }
        } else {
          CHECK(g == w);
        }
      }
    }
  }
}

// ── 2. The reference's refusals ─────────────────────────────────────────────

TEST_CASE("nimble.request.refuses_what_the_reference_refuses") {
  CHECK(RefusalOf(Body(R"({"questions":{"a":{"type":"noul","instructions":"x"}}})"))
            .find("state") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"  ","questions":{"a":{"type":"noul","instructions":"x"}}})"))
            .find("Context must be a nonempty string") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{}})")).find("questions") !=
        std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"rank","instructions":"x"}}})"))
            .find("unknown type") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":" "}}})"))
            .find("nonempty description") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":"x","extra":1}}})"))
            .find("unsupported key 'extra'") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":"x","criteria":{"maybe":"m"}}}})"))
            .find("unsupported key 'maybe'") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{"only":null}}}})"))
            .find("2-64") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"score","instructions":"x","criteria":["low",2]}}})"))
            .find("strings") != std::string::npos);
  // 27 choices: the reference switches prompt contracts; this engine refuses
  // by name rather than approximating the extended arm.
  ojson wide = Body(R"({"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{}}}})");
  for (int i = 0; i < 27; ++i) wide["questions"]["a"]["criteria"]["k" + std::to_string(i)] = nullptr;
  const std::string msg = RefusalOf(wide);
  CHECK(msg.find("27 choices") != std::string::npos);
  CHECK(msg.find("extended_schema.py") != std::string::npos);
  // 26 is the widest accepted field.
  wide["questions"]["a"]["criteria"].erase("k26");
  CHECK(RefusalOf(wide).empty());
}

// ── 3. Reachability through NimbleDecide ────────────────────────────────────

TEST_CASE("nimble.decide.reads_the_last_row_candidate_logits_per_field") {
  const HfConfig config = MakeConfig();
  const double temperature = 2.179078721266035;
  auto model = vllm::MakeNimbleLoadedModel(MakeWeights(config), config,
                                           temperature, 8192, Q());
  const Tokenizer& tok = ByteTokenizer();
  const ojson body = Goldens()["cases"][0]["body"];

  const vllm::NimbleResponse got = vllm::NimbleDecide(*model, tok, body);

  // Independent recomputation: the FULL-logits ForwardDense, its last row, the
  // byte ids of "A", "B", "C" (65, 66, 67), then the openjev answer.
  const vllm::nimble::Request req = vllm::nimble::CompileRequest(body);
  const std::vector<std::string> prompts = vllm::nimble::BuildPrompts(req);
  const Qwen3_5DenseWeights weights = MakeWeights(config);
  vt::Queue q = Q();
  int64_t tokens = 0;
  std::vector<double> first_logits;
  REQUIRE(got.answers.size() == req.fields.size());
  for (size_t i = 0; i < req.fields.size(); ++i) {
    const auto& f = req.fields[i];
    const std::vector<int32_t> ids = tok.Encode(prompts[i]);
    tokens += static_cast<int64_t>(ids.size());
    std::vector<int32_t> pos(ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    const std::vector<float> all = vllm::Qwen3_5DenseModel::ForwardDense(
        ids, pos, weights, config, q);
    const size_t last = (ids.size() - 1) * static_cast<size_t>(config.vocab_size);
    std::vector<double> cand;
    for (size_t k = 0; k < f.keys.size(); ++k) cand.push_back(all[last + 65 + k]);
    if (i == 0) first_logits = cand;
    const ojson want = vllm::nimble::AnswerFromLogits(f, cand, temperature);
    const ojson& g = got.answers[f.name];
    INFO(f.name);
    REQUIRE(g["type"] == want["type"]);
    if (f.type == "noul") {
      CHECK(g["noul"].get<double>() == doctest::Approx(want["noul"].get<double>()).epsilon(1e-5));
    } else {
      for (const auto& key : f.keys) {
        CHECK(g["probabilities"][key].get<double>() ==
              doctest::Approx(want["probabilities"][key].get<double>()).epsilon(1e-5));
      }
      CHECK(g["confidence"].get<double>() ==
            doctest::Approx(want["confidence"].get<double>()).epsilon(1e-5));
    }
  }
  CHECK(got.input_tokens == tokens);
  // The answer is not the uniform prior, so the logits actually moved it.
  CHECK(std::abs(first_logits[0] - first_logits[1]) > 1e-6);

  // The engine's own temperature is the one applied: T=1 must differ.
  auto t1 = vllm::MakeNimbleLoadedModel(MakeWeights(config), config, 1.0, 8192, Q());
  const vllm::NimbleResponse got1 = vllm::NimbleDecide(*t1, tok, body);
  CHECK(std::abs(got1.answers["refund"]["noul"].get<double>() -
                 got.answers["refund"]["noul"].get<double>()) > 1e-6);
}

TEST_CASE("nimble.decide.refuses_an_overlong_prompt_without_truncating") {
  const HfConfig config = MakeConfig();
  auto model = vllm::MakeNimbleLoadedModel(MakeWeights(config), config, 1.0, 100, Q());
  const ojson body = Goldens()["cases"][0]["body"];
  std::string msg;
  try {
    (void)vllm::NimbleDecide(*model, ByteTokenizer(), body);
  } catch (const vllm::nimble::RequestError& e) {
    msg = e.what();
  }
  CHECK(msg.find("limit is 100. Nothing was truncated.") != std::string::npos);
}

TEST_CASE("nimble.decide.is_registered_as_NimbleModel") {
  const auto& reg = vllm::RegistrationFor("NimbleModel");
  CHECK(reg.info.is_pooling_model);
  CHECK_FALSE(reg.info.is_text_generation_model);
}

// ── 4. Real tokenizer (opt-in) ──────────────────────────────────────────────

TEST_CASE("nimble.tokens.match_the_reference_ids_with_the_checkpoint_tokenizer") {
  const char* dir = std::getenv("VLLM_CPP_NIMBLE_TOKENIZER_DIR");
  if (dir == nullptr) {
    MESSAGE("skipped: set VLLM_CPP_NIMBLE_TOKENIZER_DIR to the tokenizer of "
            "bespokelabs/Bespoke-Nimble-9B @ bd792f44");
    return;
  }
  std::ifstream in(std::filesystem::path(dir) / "tokenizer.json", std::ios::binary);
  REQUIRE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const Tokenizer tok = Tokenizer::FromHfJsonBytes(ss.str(), dir);
  for (const ojson& c : Goldens()["cases"]) {
    const std::vector<std::string> prompts =
        vllm::nimble::BuildPrompts(vllm::nimble::CompileRequest(c["body"]));
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(tok.Encode(prompts[i]) == c["ids"][i].get<std::vector<int32_t>>());
      std::vector<int32_t> cand;
      for (size_t k = 0; k < c["candidate_ids"][i].size(); ++k) {
        const std::vector<int32_t> letter =
            tok.Encode(std::string(1, static_cast<char>('A' + k)));
        REQUIRE(letter.size() == 1);
        cand.push_back(letter[0]);
      }
      CHECK(cand == c["candidate_ids"][i].get<std::vector<int32_t>>());
    }
  }
}
