// The REAL-WEIGHTS paged-engine gate for Flint on CPU — MODEL-TEXT-FLINT
// W-c (spec .agents/specs/flint-paged.md, issue
// ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77).
//
// Drives the published flint-Q4_K_M container through the FULL PAGED
// LLMEngine stack (LoadedEngine::FromModelDir on the .gguf file ->
// InputProcessor -> Scheduler -> the W-b paged forward arm -> Sampler ->
// OutputProcessor) with a BATCH of requests arriving CONTINUOUSLY: request
// 0 runs two steps alone, request 1 joins mid-flight, request 2 later —
// the scheduler batches mixed prefill/decode steps exactly as production
// serving arrives.
//
// The oracle is the CPU REFERENCE forward (FlintModel::Forward on the same
// weights, the row's oracle): its greedy chains are replayed step by step
// and each engine chain is adjudicated by the kolibri1 contract — a chain
// either replays the reference token-for-token, or its first divergence
// must be a NEAR-TIE: the reference's own top-2 logit margin at that step
// puts the engine's token within the house 2.5-nat band
// (test_flint_gguf.cpp / test_kolibri1_w3.cpp).
//
// Checkpoint-GATED: skips loudly when /mnt/models/flint-GGUF/ is not
// mounted (compiles + links everywhere, RUNS where the container is).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::HfConfig;
using vllm::FlintWeights;
#include "vllm/sampling_params.h"
#include "vt/backend.h"

#ifndef FLINT_GOLDENS
#define FLINT_GOLDENS "tests/vllm/models/flint_goldens_q4_k_m.json"
#endif

#ifndef FLINT_GGUF_PATH
#define FLINT_GGUF_PATH "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf"
#endif

namespace {

constexpr const char* kPublishedGguf = FLINT_GGUF_PATH;
// The house band (test_flint_gguf.cpp / the kolibri1 contract), NOT
// re-derived here.
constexpr double kLogitBand = 2.5;
constexpr int kMaxTokens = 24;
constexpr int kNumRequests = 3;

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// Greedy chain of the REFERENCE forward on the same weights, with the
// per-step full logits kept so a divergence can be adjudicated against the
// reference's own distribution.
struct RefChain {
  std::vector<int32_t> tokens;
  // Per generated step: the reference's logits over the vocab.
  std::vector<std::vector<float>> logits;
};

int32_t Argmax(const std::vector<float>& row) {
  int32_t best = 0;
  for (size_t i = 1; i < row.size(); ++i)
    if (row[i] > row[static_cast<size_t>(best)])
      best = static_cast<int32_t>(i);
  return best;
}

}  // namespace

TEST_CASE("flint paged engine: the published Q4_K_M serves continuous "
          "batches matching the reference chains") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  std::ifstream gf(FLINT_GOLDENS);
  REQUIRE_MESSAGE(gf, "goldens not found at " << FLINT_GOLDENS);
  nlohmann::json g;
  gf >> g;

  // ---- the reference chains (the oracle): greedy replay through
  // FlintModel::Forward on the same weights ----
  MESSAGE("flint paged engine: loading the weights for the reference "
          "replay...");
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const FlintWeights weights = vllm::LoadFlintFromGguf(gguf, config);

  std::vector<std::vector<int32_t>> prompt_ids;
  const auto& prompts = g.at("prompts");
  REQUIRE(static_cast<int>(prompts.size()) >= kNumRequests);
  for (int r = 0; r < kNumRequests; ++r) {
    std::vector<int32_t> ids;
    for (const auto& v : prompts[static_cast<size_t>(r)].at("input_ids"))
      ids.push_back(v.get<int32_t>());
    prompt_ids.push_back(std::move(ids));
  }

  MESSAGE("flint paged engine: replaying " << kNumRequests
           << " reference chains...");
  std::vector<RefChain> ref(static_cast<size_t>(kNumRequests));
  for (int r = 0; r < kNumRequests; ++r) {
    std::vector<int32_t> ids = prompt_ids[static_cast<size_t>(r)];
    RefChain& rc = ref[static_cast<size_t>(r)];
    for (int s = 0; s < kMaxTokens; ++s) {
      std::vector<int32_t> pos(static_cast<size_t>(ids.size()));
      for (size_t i = 0; i < pos.size(); ++i)
        pos[i] = static_cast<int32_t>(i);
      const std::vector<float> logits =
          vllm::FlintModel::Forward(ids, pos, weights, config);
      std::vector<float> last(logits.end() -
                              static_cast<std::ptrdiff_t>(config.vocab_size),
                              logits.end());
      rc.tokens.push_back(Argmax(last));
      rc.logits.push_back(std::move(last));
      ids.push_back(rc.tokens.back());
    }
  }

  // ---- the FULL PAGED ENGINE, continuous arrival ----
  MESSAGE("flint paged engine: loading the engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          kPublishedGguf, vllm::entrypoints::EngineParams{});

  std::vector<std::string> req_ids(static_cast<size_t>(kNumRequests));
  std::vector<std::vector<int32_t>> got(static_cast<size_t>(kNumRequests));
  std::vector<bool> finished(static_cast<size_t>(kNumRequests), false);

  // Continuous arrival: request 0 alone for two steps, then request 1 joins
  // mid-decode, then request 2 — the scheduler sees mixed decode/prefill
  // batches, exactly the production arrival shape.
  req_ids[0] = loaded->engine().add_request(
      "flint-paged-0", prompt_ids[0], Greedy(kMaxTokens));
  int arrived = 1;
  int steps = 0;
  while (loaded->engine().has_unfinished_requests()) {
    if (arrived < kNumRequests && steps == 2 * arrived) {
      req_ids[static_cast<size_t>(arrived)] =
          loaded->engine().add_request(
              "flint-paged-" + std::to_string(arrived),
              prompt_ids[static_cast<size_t>(arrived)],
              Greedy(kMaxTokens));
      ++arrived;
    }
    for (const vllm::RequestOutput& out : loaded->engine().step()) {
      for (size_t r = 0; r < req_ids.size(); ++r) {
        if (out.request_id == req_ids[r]) {
          got[r] = out.outputs[0].token_ids;
          if (out.finished) finished[r] = true;
        }
      }
    }
    ++steps;
  }
  MESSAGE("flint paged engine: " << steps << " engine steps, "
          << arrived << " requests arrived continuously");
  for (int r = 0; r < kNumRequests; ++r)
    CHECK(finished[static_cast<size_t>(r)]);

  // ---- adjudication: kolibri1 contract against the reference chains ----
  int exact = 0;
  for (int r = 0; r < kNumRequests; ++r) {
    const RefChain& rc = ref[static_cast<size_t>(r)];
    const std::vector<int32_t>& gt = got[static_cast<size_t>(r)];
    REQUIRE(static_cast<int>(gt.size()) == kMaxTokens);
    bool matches = true;
    int first_div = -1;
    for (int s = 0; s < kMaxTokens; ++s) {
      if (gt[static_cast<size_t>(s)] != rc.tokens[static_cast<size_t>(s)]) {
        first_div = s;
        matches = false;
        break;
      }
    }
    if (matches) {
      ++exact;
      MESSAGE("flint paged engine: request " << r << ": "
              << kMaxTokens << "/" << kMaxTokens
              << " token-exact vs the reference chain");
      continue;
    }
    // Near-tie adjudication at the FIRST divergence: the reference's own
    // top-2 margin there must put the engine's token within the band.
    const std::vector<float>& rl = rc.logits[static_cast<size_t>(first_div)];
    const float ref_best = rl[static_cast<size_t>(rc.tokens[static_cast<size_t>(first_div)])];
    const float eng = rl[static_cast<size_t>(gt[static_cast<size_t>(first_div)])];
    const double gap = static_cast<double>(ref_best) - static_cast<double>(eng);
    MESSAGE("flint paged engine: request " << r
            << ": first divergence at step " << first_div
            << " (engine " << gt[static_cast<size_t>(first_div)]
            << " vs reference " << rc.tokens[static_cast<size_t>(first_div)]
            << "), reference-margin gap " << gap << " nats");
    CHECK_MESSAGE(gap <= kLogitBand,
                  "request " << r << " diverged OUTSIDE the "
                  << kLogitBand << "-nat band (gap " << gap << ")");
  }
  MESSAGE("flint paged engine: " << exact << "/" << kNumRequests
          << " chains token-exact vs the reference forward");
  // DEBUG: detokenize engine output 0 for a human-readable signal.
  {
    vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromGguf(gguf);
    std::string text = tok.Decode(got[0]);
    std::fprintf(stderr, "ENGINE OUT 0: '%s'\n", text.c_str());
    std::vector<int32_t> refids = ref[0].tokens;
    std::string rtext = tok.Decode(refids);
    std::fprintf(stderr, "REFERENCE 0: '%s'\n", rtext.c_str());
  }
}
