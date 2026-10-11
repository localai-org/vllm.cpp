// Kolibri-1 n-gram spec-decode ENGINE-LEVEL reach probe (ISSUE-LOCAL-
// 01M4P22PX6BPRQAY1CK20FZQF6, branch row/kolibri-spec-probe).
//
// THE PRECISE BLOCKER, pinned. The tree's n-gram spec-decode machinery is
// generic engine code end to end (see docs/bench-evidence/
// kolibri1-spec-probe-20261011.md §1) — EXCEPT one qwen35-shaped seam in the
// loader: LoadedEngine::MakeKVCacheMaybeSpec
// (src/vllm/entrypoints/model_loader.cpp:1756-1763) builds
// vllm::MakeQwen3_5KVCacheSpec for ANY speculative config, ignoring the
// model's own make_kv_cache factory. For kolibri (FullAttentionSpec +
// SlidingWindowSpec, no recurrent group) that injects a GDN/MambaSpec group
// plus an fa_draft group the model does not have, and the runner refuses at
// src/vllm/v1/worker/gpu/runner.cpp:1074 ("runner: MambaSpec state shapes
// must be positive") during engine construction.
//
// This test pins the blocker exactly, so the integration that removes it
// flips this test deliberately rather than silently:
//
//   (a) the spec-OFF engine loads AND generates on CPU for kolibri (the
//       production serving path, PR #3422 — the probe proves it still runs);
//   (b) the spec-ON engine construction throws the precise error above —
//       proving the seam is the KV-spec branch and nothing else (the throw
//       fires inside FromModelDir, before any scheduling).
//
// The MEASURED acceptance + tok/s prototype lives in the bench-level harness
// tests/vllm/models/test_kolibri1_spec_probe.cpp (production proposer +
// production verify forward), which does not depend on this seam.
//
// Checkpoint-GATED: skips loudly when the real model is absent. CPU only.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

namespace {

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
// The runner refusal the qwen3.5-only KV branch triggers for kolibri.
constexpr const char* kExpectedBlocker =
    "runner: MambaSpec state shapes must be positive";

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("kolibri1 engine reach: spec-OFF generates on CPU; spec-ON is blocked at the qwen3.5 KV branch") {
  if (!std::filesystem::exists(std::string(kRealModelDir) +
                               "/model.safetensors.index.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;
  }
  std::ifstream gf(KOLIBRI1_GOLDENS);
  if (!gf) {
    MESSAGE("SKIP: KOLIBRI1_GOLDENS not readable: " << KOLIBRI1_GOLDENS);
    return;
  }
  const nlohmann::json g = nlohmann::json::parse(gf);
  const std::vector<int32_t> prompt =
      g.at("prompts")[0].at("input_ids").get<std::vector<int32_t>>();
  constexpr int kMaxTokens = 8;

  // (a) The spec-OFF engine: loads and generates on CPU for kolibri.
  vllm::entrypoints::EngineParams params_off;
  params_off.max_num_seqs = 8;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> off;
  REQUIRE_NOTHROW(off = vllm::entrypoints::LoadedEngine::FromModelDir(
                      kRealModelDir, params_off));
  const auto t0 = std::chrono::steady_clock::now();
  off->engine().add_request("off", prompt, Greedy(kMaxTokens));
  std::optional<vllm::RequestOutput> final;
  while (off->engine().has_unfinished_requests()) {
    for (const vllm::RequestOutput& item : off->engine().step()) {
      if (item.finished) final = item;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  REQUIRE(final.has_value());
  REQUIRE(final->outputs.size() == 1);
  const std::vector<int32_t>& chain = final->outputs[0].token_ids;
  MESSAGE("spec-OFF engine generated " << chain.size() << " tokens in "
                                        << std::chrono::duration<double>(t1 - t0).count()
                                        << " s on CPU; first tokens:");
  {
    std::string s;
    for (size_t i = 0; i < chain.size() && i < 8; ++i)
      s += std::to_string(chain[i]) + ",";
    MESSAGE("  " << s);
  }
  CHECK(chain.size() == static_cast<size_t>(kMaxTokens));
  // The spec-OFF arm is the production default: no drafts exist.
  CHECK(off->runner().spec_drafts_proposed() == 0);
  off.reset();

  // (b) The spec-ON engine: construction throws at the qwen3.5-only KV
  // branch. The message is pinned so a change to the seam (the integration,
  // or an accidental behavior change for qwen3.5) turns this red.
  vllm::entrypoints::EngineParams params_on;
  params_on.max_num_seqs = 8;
  params_on.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"ngram\",\"num_speculative_tokens\":5,"
      "\"prompt_lookup_min\":1,\"prompt_lookup_max\":4}");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> on;
  REQUIRE_THROWS_WITH_AS(
      on = vllm::entrypoints::LoadedEngine::FromModelDir(kRealModelDir,
                                                        params_on),
      doctest::Contains(kExpectedBlocker), std::exception);
  MESSAGE("spec-ON construction blocked at the pinned seam: \""
          << std::string(kExpectedBlocker) << "\"");
  MESSAGE("integration owed: dispatch MakeKVCacheMaybeSpec's widened branch "
          "on \"method needs spec state\" (use_eagle()) or \"the model's own "
          "KV config has a recurrent group\" — kolibri has none, so its own "
          "make_kv_cache is already spec-correct for n-gram");
}
