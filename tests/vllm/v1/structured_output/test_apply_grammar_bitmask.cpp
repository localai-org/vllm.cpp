// M3.4 Task 3 — apply_grammar_bitmask (the runner sample-path grammar mask).
//
// Ported from vllm/v1/structured_output/utils.py::apply_grammar_bitmask @
// e24d1b24. Drives the free function directly over a [num_logits, vocab] f32
// logits tensor (the exact tensor the runner gathers before Sampler::forward):
// unpack the compacted bitmask (bit set == token allowed), reorder its rows from
// structured_output_request_ids order to the dense batch order (req_ids), and set
// every FORBIDDEN token's logit to -inf (reusing the M1.7 apply_allowed_token_ids
// -inf masking). Cases mirror the Task-3 test list:
//   (a) a row that allows ONLY token K forces the greedy-sampled token to K;
//   (b) an empty grammar_output is a no-op;
//   (c) the row REORDER — in a 2-req batch where only dense row 1 is structured,
//       only row 1 is masked, row 0 untouched;
//   (d) the bit semantics — a bitmask forbidding exactly one token sets only that
//       logit to -inf, leaving the rest;
//   (e) multi-word vocab (>32) exercises the right word/bit indexing.
#include "vllm/v1/worker/gpu/runner.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/sample/metadata.h"
#include "vllm/v1/sample/sampler.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::v1::apply_grammar_bitmask;
using vllm::v1::GrammarOutput;
using vllm::v1::Sampler;
using vllm::v1::SamplingMetadata;
using vllm::v1::TokenBitmask;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

int Words(int vocab) { return (vocab + 31) / 32; }

// A bitmask over `num_seqs` rows, each `vocab`-wide, initialized all-FORBIDDEN
// (every bit clear). Callers set the allowed bits per row via Allow().
TokenBitmask ZeroBitmask(int num_seqs, int vocab) {
  TokenBitmask bm;
  bm.num_seqs = num_seqs;
  bm.num_words = Words(vocab);
  bm.data.assign(static_cast<size_t>(num_seqs) * static_cast<size_t>(bm.num_words), 0);
  return bm;
}
// A bitmask over `num_seqs` rows, each `vocab`-wide, initialized all-ALLOWED.
TokenBitmask FullBitmask(int num_seqs, int vocab) {
  TokenBitmask bm;
  bm.num_seqs = num_seqs;
  bm.num_words = Words(vocab);
  bm.data.assign(static_cast<size_t>(num_seqs) * static_cast<size_t>(bm.num_words), -1);
  return bm;
}
// Set bit for token t in row `row` to ALLOWED.
void Allow(TokenBitmask& bm, int row, int t) {
  bm.data[static_cast<size_t>(row) * static_cast<size_t>(bm.num_words) +
          static_cast<size_t>(t >> 5)] |= (int32_t{1} << (t & 31));
}
// Clear bit for token t in row `row` to FORBIDDEN.
void Forbid(TokenBitmask& bm, int row, int t) {
  bm.data[static_cast<size_t>(row) * static_cast<size_t>(bm.num_words) +
          static_cast<size_t>(t >> 5)] &= ~(int32_t{1} << (t & 31));
}

Tensor Logits(std::vector<float>& v, int64_t n, int64_t vocab) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {n, vocab});
}

bool IsNegInf(float x) { return std::isinf(x) && x < 0.0f; }

const std::map<std::string, std::vector<int32_t>> kNoSpec;  // T0: no spec-decode
}  // namespace

// ---------------------------------------------------------------------------
// (a) A bitmask row that allows ONLY token K forces the greedy-sampled token to
//     K, even though K was NOT the pre-mask argmax.
TEST_CASE("apply_grammar_bitmask: allow-only-K forces the sampled token to K") {
  const int vocab = 8;
  const int K = 5;
  // Pre-mask argmax is token 3 (value 9.0); K=5 is NOT the argmax.
  std::vector<float> logits = {0.1f, 0.2f, 0.3f, 9.0f, 0.4f, 1.0f, 0.5f, 0.6f};
  Tensor tl = Logits(logits, 1, vocab);

  GrammarOutput go;
  go.structured_output_request_ids = {"r0"};
  go.grammar_bitmask = ZeroBitmask(1, vocab);
  Allow(go.grammar_bitmask, 0, K);

  Queue q = Q();
  apply_grammar_bitmask(go, {"r0"}, kNoSpec, q, tl);

  // Every non-K logit is -inf; K is untouched.
  for (int t = 0; t < vocab; ++t) {
    if (t == K) {
      CHECK(logits[static_cast<size_t>(t)] == doctest::Approx(1.0f));
    } else {
      CHECK(IsNegInf(logits[static_cast<size_t>(t)]));
    }
  }

  // Greedy sampling now picks K.
  SamplingMetadata sm;
  sm.all_greedy = true;
  Sampler sampler;
  auto out = sampler.forward(q, tl, sm);
  REQUIRE(out.sampled_token_ids.size() == 1);
  REQUIRE(out.sampled_token_ids[0].size() == 1);
  CHECK(out.sampled_token_ids[0][0] == K);
}

// ---------------------------------------------------------------------------
// (b) An empty grammar_output (no structured request ids) is a no-op.
TEST_CASE("apply_grammar_bitmask: empty grammar_output leaves logits unchanged") {
  const int vocab = 8;
  std::vector<float> logits = {0.1f, 0.2f, 0.3f, 9.0f, 0.4f, 1.0f, 0.5f, 0.6f};
  const std::vector<float> before = logits;
  Tensor tl = Logits(logits, 1, vocab);

  GrammarOutput go;  // no structured ids, empty bitmask (num_seqs == 0)
  Queue q = Q();
  apply_grammar_bitmask(go, {"r0"}, kNoSpec, q, tl);

  CHECK(logits == before);
}

// ---------------------------------------------------------------------------
// (c) The row REORDER: a 2-req batch where only the SECOND req (dense row 1) is
//     structured -> only row 1 is masked, row 0 untouched.
TEST_CASE("apply_grammar_bitmask: reorder masks only the structured dense row") {
  const int vocab = 4;
  // Row 0 (non-structured) and row 1 (structured) identical pre-mask.
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f,   // dense row 0 ("r0")
                               1.0f, 2.0f, 3.0f, 4.0f};  // dense row 1 ("r1")
  const std::vector<float> before = logits;
  Tensor tl = Logits(logits, 2, vocab);

  // Only "r1" is structured; its bitmask (1 compacted row) allows only token 0.
  GrammarOutput go;
  go.structured_output_request_ids = {"r1"};
  go.grammar_bitmask = ZeroBitmask(1, vocab);
  Allow(go.grammar_bitmask, 0, 0);

  Queue q = Q();
  apply_grammar_bitmask(go, {"r0", "r1"}, kNoSpec, q, tl);

  // Row 0 untouched.
  for (int t = 0; t < vocab; ++t)
    CHECK(logits[static_cast<size_t>(t)] == doctest::Approx(before[static_cast<size_t>(t)]));
  // Row 1: only token 0 survives; tokens 1..3 are -inf.
  CHECK(logits[static_cast<size_t>(vocab) + 0] == doctest::Approx(1.0f));
  for (int t = 1; t < vocab; ++t)
    CHECK(IsNegInf(logits[static_cast<size_t>(vocab) + static_cast<size_t>(t)]));
}

// ---------------------------------------------------------------------------
// (d) The bit semantics: a bitmask forbidding exactly one token sets only that
//     logit to -inf and leaves the rest (bit set == allowed).
TEST_CASE("apply_grammar_bitmask: forbidding one token masks only that logit") {
  const int vocab = 8;
  const int FORBID = 3;
  std::vector<float> logits = {0.1f, 0.2f, 0.3f, 9.0f, 0.4f, 1.0f, 0.5f, 0.6f};
  const std::vector<float> before = logits;
  Tensor tl = Logits(logits, 1, vocab);

  GrammarOutput go;
  go.structured_output_request_ids = {"r0"};
  go.grammar_bitmask = FullBitmask(1, vocab);  // all allowed...
  Forbid(go.grammar_bitmask, 0, FORBID);       // ...except token 3

  Queue q = Q();
  apply_grammar_bitmask(go, {"r0"}, kNoSpec, q, tl);

  for (int t = 0; t < vocab; ++t) {
    if (t == FORBID) {
      CHECK(IsNegInf(logits[static_cast<size_t>(t)]));
    } else {
      CHECK(logits[static_cast<size_t>(t)] == doctest::Approx(before[static_cast<size_t>(t)]));
    }
  }
}

// ---------------------------------------------------------------------------
// (e) Multi-word vocab (>32): the allowed token lives in word 1, exercising the
//     word/bit indexing (token 40 -> word 1, bit 8).
TEST_CASE("apply_grammar_bitmask: multi-word vocab indexes the right word/bit") {
  const int vocab = 70;  // 3 words (ceil(70/32) == 3)
  const int K = 40;      // word 1, bit 8
  std::vector<float> logits(static_cast<size_t>(vocab), 0.5f);
  logits[10] = 9.0f;  // pre-mask argmax at 10 (word 0), NOT K
  Tensor tl = Logits(logits, 1, vocab);

  GrammarOutput go;
  go.structured_output_request_ids = {"r0"};
  go.grammar_bitmask = ZeroBitmask(1, vocab);
  Allow(go.grammar_bitmask, 0, K);

  Queue q = Q();
  apply_grammar_bitmask(go, {"r0"}, kNoSpec, q, tl);

  for (int t = 0; t < vocab; ++t) {
    if (t == K) {
      CHECK(logits[static_cast<size_t>(t)] == doctest::Approx(0.5f));
    } else {
      CHECK(IsNegInf(logits[static_cast<size_t>(t)]));
    }
  }

  SamplingMetadata sm;
  sm.all_greedy = true;
  Sampler sampler;
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == K);
}

TEST_CASE("apply_grammar_bitmask: mixed MTP batch masks every verify and bonus row") {
  constexpr int vocab = 16;
  // Dense order: plain k2 (rows 0..2), B k1 (3..4), A k3 (5..8).
  std::vector<float> logits(9 * vocab, 3.0f);
  Tensor tl = Logits(logits, 9, vocab);
  GrammarOutput go;
  go.structured_output_request_ids = {"A", "B"};
  go.grammar_bitmask = ZeroBitmask(6, vocab);
  for (int row = 0; row < 6; ++row) Allow(go.grammar_bitmask, row, row + 1);
  const std::map<std::string, std::vector<int32_t>> spec = {
      {"plain", {10, 11}}, {"B", {12}}, {"A", {13, 14, 15}}};
  Queue q = Q();
  apply_grammar_bitmask(go, {"plain", "B", "A"}, spec, q, tl);
  const std::vector<int> allowed{-1, -1, -1, 5, 6, 1, 2, 3, 4};
  for (int row = 0; row < 9; ++row)
    for (int token = 0; token < vocab; ++token) {
      const float value = logits[row * vocab + token];
      if (allowed[row] == -1 || allowed[row] == token)
        CHECK(value == 3.0f);
      else
        CHECK(IsNegInf(value));
    }
}
