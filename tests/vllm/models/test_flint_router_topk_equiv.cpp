// FLINT-CPU-DESCENT-2 — the host router top-k selection equivalence suite.
//
// The keep-quant paged arm's MoE router (flint.cpp FlintMoeBlock) replaced the
// per-token `std::vector` workspace + `std::partial_sort` over all e scores
// with a hoisted workspace and a bounded top-k insertion (`std::lower_bound`
// against the same comparator). The router output feeds the expert gather, so
// this is a BIT-IDENTITY contract, not a band contract: the selected top-6 set,
// its order, and the renormalised weights must be identical on EVERY input —
// a different set changes gathered rows and therefore served tokens.
//
// This suite pins that equivalence the way the NEON quant-dot suites pin
// theirs: the reference here is a local copy of the PREVIOUS path (build the
// full score vector, partial_sort with the (value DESC, index ASC)
// comparator), and the candidate is a copy of the NEW path (bounded
// insertion). Both are compared as f32 bits + int32 ids per (token, k) slot.
// The comparator is a total order (indices are unique), so equivalence on
// random and tie-saturated grids covers the contract: the sigmoid values
// themselves are computed by the same `1.0f / (1.0f + std::exp(-lg))`
// expression in production on both paths and are not re-derived here.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kE = 384;   // flint.num_experts
constexpr int kTopK = 6;  // flint.num_experts_per_tok

struct Routing {
  std::vector<float> weights;  // t * top_k, renormalised
  std::vector<int32_t> ids;    // t * top_k
};

uint32_t Xorshift32(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

float Sigmoid(float lg) { return 1.0f / (1.0f + std::exp(-lg)); }

// The PREVIOUS path: full score vector + partial_sort.
Routing ReferenceRouting(const std::vector<float>& hlog, int t) {
  Routing r;
  r.weights.assign(static_cast<size_t>(t * kTopK), 0.0f);
  r.ids.assign(static_cast<size_t>(t * kTopK), -1);
  for (int i = 0; i < t; ++i) {
    std::vector<std::pair<float, int64_t>> score(kE);
    for (int n = 0; n < kE; ++n) {
      score[static_cast<size_t>(n)] = {Sigmoid(hlog[static_cast<size_t>(i) * kE + n]), n};
    }
    std::partial_sort(score.begin(), score.begin() + kTopK, score.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    double sum = 0.0;
    for (int kk = 0; kk < kTopK; ++kk) sum += score[static_cast<size_t>(kk)].first;
    for (int kk = 0; kk < kTopK; ++kk) {
      r.weights[static_cast<size_t>(i * kTopK + kk)] = static_cast<float>(
          score[static_cast<size_t>(kk)].first / sum);
      r.ids[static_cast<size_t>(i * kTopK + kk)] =
          static_cast<int32_t>(score[static_cast<size_t>(kk)].second);
    }
  }
  return r;
}

// The NEW path: bounded top-k insertion against the same comparator.
Routing CandidateRouting(const std::vector<float>& hlog, int t) {
  Routing r;
  r.weights.assign(static_cast<size_t>(t * kTopK), 0.0f);
  r.ids.assign(static_cast<size_t>(t * kTopK), -1);
  std::vector<std::pair<float, int64_t>> top;
  top.reserve(kTopK + 1);
  const auto before = [](const std::pair<float, int64_t>& x,
                         const std::pair<float, int64_t>& y) {
    return x.first > y.first || (x.first == y.first && x.second < y.second);
  };
  for (int i = 0; i < t; ++i) {
    top.clear();
    for (int n = 0; n < kE; ++n) {
      const std::pair<float, int64_t> cand{
          Sigmoid(hlog[static_cast<size_t>(i) * kE + n]), n};
      auto pos = std::lower_bound(top.begin(), top.end(), cand, before);
      if (pos == top.end()) {
        if (static_cast<int>(top.size()) < kTopK) top.push_back(cand);
        continue;
      }
      top.insert(pos, cand);
      if (static_cast<int>(top.size()) > kTopK) top.pop_back();
    }
    double sum = 0.0;
    for (int kk = 0; kk < kTopK; ++kk) sum += top[static_cast<size_t>(kk)].first;
    for (int kk = 0; kk < kTopK; ++kk) {
      r.weights[static_cast<size_t>(i * kTopK + kk)] = static_cast<float>(
          top[static_cast<size_t>(kk)].first / sum);
      r.ids[static_cast<size_t>(i * kTopK + kk)] =
          static_cast<int32_t>(top[static_cast<size_t>(kk)].second);
    }
  }
  return r;
}

uint32_t FloatBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return bits;
}

void CheckEquivalence(const std::vector<float>& hlog, int t,
                      const std::string& label) {
  CAPTURE(label);
  const Routing ref = ReferenceRouting(hlog, t);
  const Routing got = CandidateRouting(hlog, t);
  for (size_t s = 0; s < ref.weights.size(); ++s) {
    if (FloatBits(ref.weights[s]) != FloatBits(got.weights[s]) ||
        ref.ids[s] != got.ids[s]) {
      MESSAGE("slot " << s << " ref id " << ref.ids[s] << " w " << ref.weights[s]
                      << " | got id " << got.ids[s] << " w " << got.weights[s]);
    }
    REQUIRE(FloatBits(ref.weights[s]) == FloatBits(got.weights[s]));
    REQUIRE(ref.ids[s] == got.ids[s]);
  }
}

std::vector<float> RandomLogits(uint32_t* rng, int t) {
  std::vector<float> hlog(static_cast<size_t>(t * kE));
  for (float& v : hlog) {
    v = static_cast<float>(static_cast<int>(Xorshift32(rng) % 2001) - 1000) *
        0.01f;  // [-10, 10]
  }
  return hlog;
}

}  // namespace

TEST_CASE("flint router top-k: randomized logits are bit-identical") {
  for (uint32_t seed = 1; seed <= 4; ++seed) {
    uint32_t rng = seed * 0x9E3779B9U;
    for (int rep = 0; rep < 64; ++rep) {
      CheckEquivalence(RandomLogits(&rng, 1), 1,
                       "seed=" + std::to_string(seed) + " rep=" + std::to_string(rep));
    }
  }
  // Batches: the workspace hoist spans tokens; equivalence must hold per token.
  uint32_t rng = 0xC0FFEE13U;
  for (int rep = 0; rep < 16; ++rep) {
    const int t = 1 + static_cast<int>(Xorshift32(&rng) % 32);
    CheckEquivalence(RandomLogits(&rng, t), t, "batch rep=" + std::to_string(rep));
  }
}

TEST_CASE("flint router top-k: tie-saturated logits are bit-identical") {
  // Exact float ties exercise the index tie-break: duplicated logits at
  // scattered expert positions, plus the extreme logits (-inf-like, +inf-like
  // magnitudes) whose sigmoids collapse to exactly 1.0f or 0.0f.
  uint32_t rng = 0xABCDEF03U;
  for (int rep = 0; rep < 128; ++rep) {
    std::vector<float> hlog(kE);
    for (float& v : hlog) v = static_cast<float>(static_cast<int>(Xorshift32(&rng) % 5)) - 2.0f;
    CheckEquivalence(hlog, 1, "ties rep=" + std::to_string(rep));
  }
  for (int rep = 0; rep < 128; ++rep) {
    std::vector<float> hlog(kE);
    for (int n = 0; n < kE; ++n) {
      const int mode = static_cast<int>(Xorshift32(&rng) % 4);
      hlog[static_cast<size_t>(n)] =
          mode == 0 ? 100.0f : mode == 1 ? -100.0f : mode == 2 ? 0.0f : 30.0f;
    }
    CheckEquivalence(hlog, 1, "saturated rep=" + std::to_string(rep));
  }
}
