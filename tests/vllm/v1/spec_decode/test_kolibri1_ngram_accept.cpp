// Kolibri-1 n-gram spec-decode ACCEPTANCE MEASUREMENT (probe,
// ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6, branch row/kolibri-spec-probe).
//
// The measured answer to "what acceptance rate would an n-gram drafter get on
// kolibri's greedy chains?" — computed with the PRODUCTION proposer
// (src/vllm/v1/spec_decode/ngram_proposer.cpp, the ported
// FindLongestMatchedNgramAndProposeTokens) over the row's own greedy chains:
// the 8 W3 golden prompts (tests/vllm/models/kolibri1_goldens.json) and,
// when the decode bench has printed one, the bench chain.
//
// The simulation is EXACT, not statistical: greedy verification accepts a
// draft iff it equals the target argmax (the greedy accept rule of
// include/vllm/v1/spec_decode/rejection_sampler.h), so the committed tokens
// are always a prefix of the golden chain and the spec-decode run is
// byte-identical to plain greedy decode. The numbers below are therefore the
// TRUE acceptance the production drafter would measure on these chains —
// no model run is needed to state them.
//
// Per (k, window) config the test reports:
//   steps        verify passes the chain takes (plain decode: len(chain)-1)
//   tok/step     emitted tokens per verify pass (1 + accepted prefix)
//   step-ratio   plain steps / spec steps (the step-count multiplier)
//   acc/verify   accepted draft tokens per verify pass
//
// GATES:
//   (i)  EXACTNESS: the simulated spec chain equals the golden chain
//        byte-for-byte for every config — the exactness property the
//        engine-level gate (test_kolibri1_ngram_spec) then re-proves on the
//        real model.
//   (ii) PINNED RATES: the aggregate headline numbers are deterministic
//        (fixed chains, fixed algorithm), so they are pinned exactly; a
//        proposer mutation or a golden regeneration turns this red.
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/v1/spec_decode/ngram_proposer.h"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

using vllm::v1::spec_decode::FindLongestMatchedNgramAndProposeTokens;

namespace {

constexpr int kMaxModelLen = 262144;  // the checkpoint's max_position_embeddings

struct SimResult {
  int steps = 0;          // verify passes (the prefill pass is not counted)
  int emitted = 0;        // total committed tokens, the prefill token included
  int accepted = 0;       // accepted draft tokens
  int proposed = 0;       // draft tokens proposed
  bool exact = true;      // the committed tokens are a prefix of the chain
};

// One greedy request under n-gram spec decode, mirroring the engine's loop:
// prefill emits chain[0] with no drafts; every later step proposes from the
// committed context, verifies the draft prefix against the chain (the target
// argmax), and commits accepted+1 tokens (the final step is truncated by the
// generation limit, exactly as the engine's finish condition discards tokens
// past max_tokens).
SimResult Simulate(const std::vector<int32_t>& prompt,
                   const std::vector<int32_t>& chain, int min_n, int max_n,
                   int k) {
  SimResult r;
  std::vector<int32_t> context(prompt);
  int emitted = 0;
  if (!chain.empty()) {
    context.push_back(chain[0]);
    emitted = 1;
  }
  while (emitted < static_cast<int>(chain.size())) {
    const std::vector<int32_t> drafts = FindLongestMatchedNgramAndProposeTokens(
        context.data(), static_cast<int>(context.size()), min_n, max_n,
        kMaxModelLen, k);
    int accepted = 0;
    for (size_t i = 0; i < drafts.size(); ++i) {
      const int pos = emitted + static_cast<int>(i);
      if (pos < static_cast<int>(chain.size()) && drafts[i] == chain[pos]) {
        ++accepted;
      } else {
        break;  // stop at the first mismatch (the sticky-accept rule)
      }
    }
    const int room = static_cast<int>(chain.size()) - emitted;
    emitted += std::min(accepted + 1, room);
    context.assign(prompt.begin(), prompt.end());
    context.insert(context.end(), chain.begin(),
                   chain.begin() + emitted);
    r.steps += 1;
    r.accepted += accepted;
    r.proposed += static_cast<int>(drafts.size());
  }
  r.emitted = emitted;
  r.exact = (emitted == static_cast<int>(chain.size()));
  return r;
}

struct Chain {
  std::string name;
  std::vector<int32_t> prompt;
  std::vector<int32_t> chain;
};

std::vector<Chain> LoadChains() {
  std::vector<Chain> out;
  std::ifstream f(KOLIBRI1_GOLDENS);
  if (!f) return out;
  const nlohmann::json g = nlohmann::json::parse(f);
  for (size_t i = 0; i < g.at("prompts").size(); ++i) {
    const auto& p = g.at("prompts")[i];
    Chain c;
    c.name = "golden[" + std::to_string(i) + "] \"" +
             p.at("prompt").get<std::string>().substr(0, 40) + "\"";
    c.prompt = p.at("input_ids").get<std::vector<int32_t>>();
    c.chain = p.at("generated_ids").get<std::vector<int32_t>>();
    out.push_back(std::move(c));
  }
  return out;
}

}  // namespace

TEST_CASE("kolibri1 ngram acceptance on the W3 golden chains (production proposer)") {
  const std::vector<Chain> chains = LoadChains();
  REQUIRE_MESSAGE(!chains.empty(),
                  "KOLIBRI1_GOLDENS not readable: " << KOLIBRI1_GOLDENS);
  REQUIRE(chains.size() == 8);

  struct Cfg { int k; int mn; int mx; };
  const std::vector<Cfg> cfgs = {
      {1, 1, 1}, {2, 1, 4}, {3, 1, 4}, {4, 1, 4}, {5, 1, 4}, {8, 1, 4},
      {5, 2, 4}, {5, 5, 5},
  };

  MESSAGE("config | steps | plain | step-ratio | tok/step | acc/verify | "
          "accept-rate");
  for (const Cfg& c : cfgs) {
    int steps = 0, emitted = 0, accepted = 0, proposed = 0, plain = 0;
    for (const Chain& ch : chains) {
      const SimResult r = Simulate(ch.prompt, ch.chain, c.mn, c.mx, c.k);
      // GATE (i): exactness — the spec run reproduces the chain byte-for-byte.
      REQUIRE_MESSAGE(r.exact,
                      "chain broken by spec simulation: " << ch.name);
      steps += r.steps;
      emitted += r.emitted;
      accepted += r.accepted;
      proposed += r.proposed;
      plain += static_cast<int>(ch.chain.size()) - 1;
    }
    const double tok_per_step =
        steps > 0 ? static_cast<double>(emitted) / steps : 0.0;
    const double ratio =
        steps > 0 ? static_cast<double>(plain) / steps : 0.0;
    const double acc_rate =
        proposed > 0 ? static_cast<double>(accepted) / proposed : 0.0;
    MESSAGE("k=" << c.k << " window=" << c.mn << "-" << c.mx << " | " << steps
                  << " | " << plain << " | " << ratio << " | " << tok_per_step
                  << " | " << (steps > 0 ? static_cast<double>(accepted) / steps
                                         : 0.0)
                  << " | " << acc_rate);

    if (c.k == 5 && c.mn == 1 && c.mx == 4) {
      // GATE (ii): the pinned headline. 8 chains x 32 tokens: 248 plain
      // decode steps; the production proposer at k=5 window 1-4 takes 103
      // verify passes. The passes emit 248 tokens (accepted drafts + one
      // bonus/replacement each); with the 8 prefill tokens that is all 256.
      // Per-token accept rate 150/297; tokens per verify pass 248/103.
      CHECK(steps == 103);
      CHECK(emitted == 256);
      CHECK(accepted == 150);
      CHECK(proposed == 297);
      CHECK(tok_per_step == doctest::Approx(256.0 / 103.0));
      CHECK(ratio == doctest::Approx(248.0 / 103.0));
      CHECK(acc_rate == doctest::Approx(150.0 / 297.0));
    }
    if (c.k == 8 && c.mn == 1 && c.mx == 4) {
      CHECK(steps == 96);
      CHECK(emitted == 256);
      CHECK(accepted == 157);
      CHECK(proposed == 382);
    }
    if (c.k == 5 && c.mn == 5 && c.mx == 5) {
      // The upstream default window (5/5) is markedly worse on these chains.
      CHECK(steps == 147);
      CHECK(emitted == 256);
      CHECK(accepted == 107);
      CHECK(proposed == 124);
    }
  }
}

TEST_CASE("kolibri1 ngram acceptance: per-chain spread (window 1-4, k=5)") {
  const std::vector<Chain> chains = LoadChains();
  REQUIRE(!chains.empty());
  for (const Chain& ch : chains) {
    const SimResult r = Simulate(ch.prompt, ch.chain, 1, 4, 5);
    const int plain = static_cast<int>(ch.chain.size()) - 1;
    MESSAGE(ch.name << ": prompt " << ch.prompt.size() << " chain "
                    << ch.chain.size() << " -> steps " << r.steps << "/"
                    << plain << " tok/step "
                    << (r.steps > 0
                            ? static_cast<double>(r.emitted) / r.steps
                            : 0.0)
                    << " accepted " << r.accepted << "/" << r.proposed);
    REQUIRE(r.exact);
  }
}
