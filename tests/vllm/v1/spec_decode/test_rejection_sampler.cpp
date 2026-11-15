// Ported from (test-porting protocol .agents/test-porting.md):
//   * tests/v1/sample/test_rejection_sampler.py @ e24d1b24 —
//       test_perfect_match          :133
//       test_early_mismatch         :154
//       test_multiple_sequences     :179
//       test_single_token_sequence  :204
//       test_empty_sequence         :225
//       test_multiple_mismatches    :246
//       test_parametrized_cases     :288
//     (the legacy-sampler suite; its ASSERTIONS are realized here against our
//     MRV2-shaped RejectionSampler — test-porting rule 3.)
//   * tests/v1/spec_decode/test_rejection_sampler_utils.py @ e24d1b24 —
//       test_greedy_rejection_sample        :183 (k in {1, 3})
//       test_placeholder_draft_token_rejected :285
//     (the stochastic :141, synthetic :215 and block-verification :325/:372
//      cases are SKIPPED until M-mtp-3 — see the SKIPPED note at the bottom.)
//
// The upstream cases construct logits whose argmax at expanded row j is
// output_tokens[r][j], schedule spec_tokens[r] as the drafts, and assert the
// emitted token stream. Upstream's fixed-width [num_reqs, k+1] output pads
// rejected positions with PLACEHOLDER_TOKEN_ID (-1,
// vllm/v1/sample/rejection_sampler.py:30); our RejectionSamplerOutput returns
// ragged per-request vectors of exactly num_sampled tokens, so a padded
// upstream row [a, b, -1, -1] becomes {a, b} plus num_sampled == 2. Both forms
// are asserted.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::v1::RejectionSampler;
using vllm::v1::RejectionSamplerOutput;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

constexpr int kVocab = 16;

// One verify-step fixture, built exactly the way upstream's
// `create_logits_tensor` + `create_spec_decode_metadata` do:
//   * `target_tokens[r]` are the tokens the TARGET argmaxes at request r's
//     expanded rows (length 1 + k_r);
//   * `spec_tokens[r]` are the k_r draft tokens the scheduler proposed.
// The draft_sampled array mirrors `input_ids[logits_indices]`: row cu[r] holds
// the previous step's token (never compared — upstream reads draft_sampled at
// logit_idx + 1, rejection_sampler_utils.py:534) and rows cu[r]+1.. hold the
// drafts.
struct VerifyStep {
  std::vector<float> logits;
  std::vector<int32_t> draft_sampled;
  std::vector<int32_t> cu_num_logits;
  int64_t num_logits = 0;
};

VerifyStep MakeStep(const std::vector<std::vector<int32_t>>& spec_tokens,
                    const std::vector<std::vector<int32_t>>& target_tokens) {
  REQUIRE(spec_tokens.size() == target_tokens.size());
  VerifyStep s;
  s.cu_num_logits.push_back(0);
  int32_t total = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    REQUIRE(target_tokens[r].size() == spec_tokens[r].size() + 1);
    total += static_cast<int32_t>(target_tokens[r].size());
    s.cu_num_logits.push_back(total);
  }
  s.num_logits = total;
  s.logits.assign(static_cast<size_t>(total) * kVocab, 0.0f);
  s.draft_sampled.assign(static_cast<size_t>(total), 0);
  int32_t row = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    // Row cu[r] input id: an arbitrary previously-sampled token (never read).
    s.draft_sampled[static_cast<size_t>(row)] = 0;
    for (size_t j = 0; j < target_tokens[r].size(); ++j) {
      // Make target_tokens[r][j] the strict argmax of expanded row cu[r]+j.
      s.logits[static_cast<size_t>(row + static_cast<int32_t>(j)) * kVocab +
               static_cast<size_t>(target_tokens[r][j])] = 10.0f;
      if (j < spec_tokens[r].size()) {
        s.draft_sampled[static_cast<size_t>(row) + j + 1] = spec_tokens[r][j];
      }
    }
    row += static_cast<int32_t>(target_tokens[r].size());
  }
  return s;
}

RejectionSamplerOutput Run(const std::vector<std::vector<int32_t>>& spec_tokens,
                           const std::vector<std::vector<int32_t>>& target_tokens,
                           int num_speculative_steps) {
  VerifyStep s = MakeStep(spec_tokens, target_tokens);
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  return sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits);
}

// The upstream fixed-width form: pad each request's emitted tokens to
// max_len with PLACEHOLDER_TOKEN_ID (-1).
std::vector<std::vector<int32_t>> Padded(const RejectionSamplerOutput& out, size_t width) {
  std::vector<std::vector<int32_t>> rows;
  for (const auto& toks : out.sampled_token_ids) {
    std::vector<int32_t> row = toks;
    row.resize(width, -1);
    rows.push_back(row);
  }
  return rows;
}

}  // namespace

// ---------------------------------------------------------------------------
// test_perfect_match (test_rejection_sampler.py:133): every draft matches, so
// the emitted stream is the drafts plus the bonus token.
TEST_CASE("rejection_sampler: perfect_match emits every draft plus the bonus token") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 2, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids.size() == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 4});
  CHECK(out.num_sampled[0] == 4);
  CHECK(out.num_rejected[0] == 0);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 3, 4});
}

// test_early_mismatch (:154): mismatch at position 1 -> emit the target argmax
// there and STOP; nothing after it is accepted.
TEST_CASE("rejection_sampler: early_mismatch emits the target argmax and stops") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 5, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 5});
  CHECK(out.num_sampled[0] == 2);
  // 3 drafts scheduled, 1 accepted -> 2 rejected (the scheduler rolls back by 2).
  CHECK(out.num_rejected[0] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 5, -1, -1});
}

// test_multiple_sequences (:179): two requests with DIFFERENT k_i (2 and 1).
TEST_CASE("rejection_sampler: multiple_sequences with different per-request k") {
  const RejectionSamplerOutput out = Run({{1, 2}, {3}}, {{1, 2, 5}, {3, 4}}, /*k=*/2);
  CHECK(out.sampled_token_ids.size() == 2);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{3, 4});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_rejected[1] == 0);
  CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, -1});
}

// test_single_token_sequence (:204): k == 1, accepted.
TEST_CASE("rejection_sampler: single_token_sequence (k=1) accepts and emits the bonus") {
  const RejectionSamplerOutput out = Run({{1}}, {{1, 2}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2});
  CHECK(out.num_sampled[0] == 2);
  CHECK(out.num_rejected[0] == 0);
}

// test_empty_sequence (:225): NO drafts — the k == 0 reduction. This is the
// byte-identity anchor: the emitted token is exactly the plain greedy argmax.
TEST_CASE("rejection_sampler: empty_sequence (k=0) reduces to the plain greedy argmax") {
  const RejectionSamplerOutput out = Run({{}}, {{5}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.num_rejected[0] == 0);
}

// test_multiple_mismatches (:246): both requests reject, at different positions.
TEST_CASE("rejection_sampler: multiple_mismatches reject independently per request") {
  const RejectionSamplerOutput out =
      Run({{1, 2, 3}, {4, 5, 6}}, {{1, 2, 7, 6}, {4, 8, 6, 9}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 7});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{4, 8});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 1);
  CHECK(out.num_rejected[1] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 7, -1});
  CHECK(Padded(out, 4)[1] == std::vector<int32_t>{4, 8, -1, -1});
}

// test_parametrized_cases (:288): the three upstream parametrizations.
TEST_CASE("rejection_sampler: parametrized cases (perfect / first-mismatch / mixed)") {
  SUBCASE("perfect match with bonus") {
    const RejectionSamplerOutput out = Run({{1, 2}}, {{1, 2, 3}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 3});
  }
  SUBCASE("first mismatch") {
    const RejectionSamplerOutput out = Run({{1}}, {{2, 3}}, /*k=*/1);
    CHECK(Padded(out, 2)[0] == std::vector<int32_t>{2, -1});
    CHECK(out.num_sampled[0] == 1);
    CHECK(out.num_rejected[0] == 1);
  }
  SUBCASE("mixed matches") {
    const RejectionSamplerOutput out = Run({{1, 2}, {3, 4}}, {{1, 5, 6}, {3, 4, 7}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 5, -1});
    CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, 7});
  }
}

// ---------------------------------------------------------------------------
// test_greedy_rejection_sample (test_rejection_sampler_utils.py:183, k in {1,3}):
// "Verify that greedy (temperature=0) always outputs the target argmax at every
// accepted position." Upstream drives one shared target distribution across many
// trials; we drive the same invariant over an exhaustive draft-pattern sweep,
// which additionally pins WHERE the run stops.
TEST_CASE("rejection_sampler: greedy_rejection_sample — every emitted token is the target argmax") {
  for (int k : {1, 3}) {
    // A fixed target argmax sequence; sweep every subset of matching drafts.
    const std::vector<int32_t> target_seq = {3, 7, 2, 9};  // length 4 >= k+1
    std::vector<int32_t> target(target_seq.begin(), target_seq.begin() + k + 1);
    const int num_patterns = 1 << k;
    for (int pattern = 0; pattern < num_patterns; ++pattern) {
      std::vector<int32_t> drafts(static_cast<size_t>(k));
      for (int i = 0; i < k; ++i) {
        // bit set => the draft matches the target argmax at position i.
        drafts[static_cast<size_t>(i)] =
            (pattern >> i) & 1 ? target[static_cast<size_t>(i)]
                               : (target[static_cast<size_t>(i)] + 1) % kVocab;
      }
      const RejectionSamplerOutput out = Run({drafts}, {target}, k);
      // Expected accepted length = the number of leading matching drafts.
      int expect_len = 0;
      while (expect_len < k && drafts[static_cast<size_t>(expect_len)] ==
                                   target[static_cast<size_t>(expect_len)]) {
        ++expect_len;
      }
      CAPTURE(k);
      CAPTURE(pattern);
      CHECK(out.num_sampled[0] == expect_len + 1);
      CHECK(out.num_rejected[0] == k - expect_len);
      REQUIRE(out.sampled_token_ids[0].size() == static_cast<size_t>(expect_len) + 1);
      // THE INVARIANT: every emitted token equals the target argmax at its row.
      for (int j = 0; j <= expect_len; ++j) {
        CHECK(out.sampled_token_ids[0][static_cast<size_t>(j)] ==
              target[static_cast<size_t>(j)]);
      }
    }
  }
}

// test_placeholder_draft_token_rejected (:285): a -1 placeholder draft id must
// be rejected without any out-of-bounds logits read.
TEST_CASE("rejection_sampler: placeholder draft token (-1) is rejected") {
  const RejectionSamplerOutput out = Run({{-1}}, {{5, 6}}, /*k=*/1);
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_rejected[0] == 1);
}

// A chunked-prefilling row samples and rejects nothing
// (_get_num_sampled_and_rejected_kernel, gpu/input_batch.py:421-433). Row 0 here
// both CARRIES DRAFTS (k = 2) and is prefilling, which is the shape the rule is
// about; a prefilling row with no drafts would zero counts that are already zero.
//
// THE SAMPLER IS THE ONLY HALF THIS ASSERTS. `GPUModelRunner` does not pass these
// two numbers to the propose: it writes `num_accepted_tokens = max(0, 1) = 1` for
// this row and `propose_after_verify` re-derives `num_sampled = 1`,
// `num_rejected = k` from that, where upstream passes the kernel's 0 and 0
// through (`vllm/v1/worker/gpu/model_runner.py:1144` -> `:1533-1546`). That
// divergence is pre-existing, A2-2 gave it a second entry point, and it is
// recorded under `## Owed` in `.agents/specs/dflash2-async-spec-sampler.md`.
TEST_CASE("rejection_sampler: a chunked-prefilling row reports 0 sampled and 0 rejected") {
  VerifyStep s = MakeStep({{1, 2}, {3}}, {{1, 9, 5}, {3, 4}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  const RejectionSamplerOutput out =
      sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits, {1, 0});
  CHECK(out.num_sampled[0] == 0);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[1] == 0);
}

// ─── SPEC-DFLASH2 A2-2 (#2802): THE DEVICE-RESIDENT SPLIT ───────────────────
//
// `forward` used to be one call that ran the accept walk, copied both outputs to
// the host and drained the queue. A2-2 splits it so the walk's outputs can stay
// on the device and the caller decides where the wait goes — a copy-queue event
// instead of a main-queue drain, which is one of the two compute-stream drains a
// speculative step pays. Upstream never pays the first one: its
// `RejectionSampler.__call__` returns DEVICE tensors
// (rejection_sampler.py:262-272 @ pin 5559679229) and the D2H is issued later by
// `AsyncOutput` on the copy stream (model_runner.py:1492-1499).
//
// WHAT THE CPU TIER CAN GATE HERE IS TOKEN IDENTITY, AND ONLY THAT. On this
// backend `Copy` is a memcpy and every event is a null-handle no-op, so nothing
// here can observe an overlap, and nothing here claims one — that is G3/G4 at
// A2-5 and needs a GPU. What it CAN observe is that the split did not move a
// token: every id, every `num_sampled` and every `num_rejected` that comes out
// of `verify` + `CopyToHost` + `finalize` must equal what `forward` produces on
// the identical step.
//
// RED-first: before the change none of `RejectionSamplerDeviceOutput`,
// `verify`, `CopyToHost` or `finalize` existed and this binary did not compile.
// After the change, mutating `finalize`'s `num_rejected` to `row_logits - 1`, or
// its emitted-token loop bound from `ns` to `ns - 1`, reds these cases while
// every pre-existing case in this file stays green — because those all go
// through `forward`, which is the same two halves in one call.
namespace {

// Drive the SPLIT halves by hand, exactly as the runner's copy-queue route does:
// issue the walk, copy the two outputs, wait, reduce on the host.
RejectionSamplerOutput RunSplit(const VerifyStep& s, int num_speculative_steps,
                                const std::vector<char>& is_chunked_prefilling) {
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  vllm::v1::RejectionSamplerDeviceOutput dev =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  const int64_t rows = dev.num_reqs();
  const int64_t width = dev.width();
  std::vector<int32_t> host_sampled(static_cast<size_t>(rows * width));
  std::vector<int32_t> host_num_sampled(static_cast<size_t>(rows));
  dev.CopyToHost(q, host_sampled.data(), host_num_sampled.data());
  vt::GetBackend(dev.device().type).Synchronize(q);
  return RejectionSampler::finalize(host_sampled, width, host_num_sampled,
                                    s.cu_num_logits, is_chunked_prefilling);
}

// `forward` on the identical step, for the comparison.
RejectionSamplerOutput RunWhole(const VerifyStep& s, int num_speculative_steps,
                                const std::vector<char>& is_chunked_prefilling) {
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  return sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits,
                         is_chunked_prefilling);
}

// Assert the two results id for id, not vector for vector, so a failure names
// the request and the position rather than saying "these differ".
void CheckSameTokens(const RejectionSamplerOutput& split,
                     const RejectionSamplerOutput& whole) {
  REQUIRE(split.num_sampled.size() == whole.num_sampled.size());
  REQUIRE(split.sampled_token_ids.size() == whole.sampled_token_ids.size());
  for (size_t r = 0; r < whole.num_sampled.size(); ++r) {
    INFO("request ", r);
    CHECK(split.num_sampled[r] == whole.num_sampled[r]);
    CHECK(split.num_rejected[r] == whole.num_rejected[r]);
    REQUIRE(split.sampled_token_ids[r].size() == whole.sampled_token_ids[r].size());
    for (size_t j = 0; j < whole.sampled_token_ids[r].size(); ++j) {
      INFO("token ", j);
      CHECK(split.sampled_token_ids[r][j] == whole.sampled_token_ids[r][j]);
    }
  }
}

}  // namespace

// A2-2.1 — the single-request shapes this file already covers, across the split.
TEST_CASE("A2-2: the split emits the same tokens as forward, one request") {
  SUBCASE("perfect match") {
    const VerifyStep s = MakeStep({{1, 2, 3}}, {{1, 2, 3, 4}});
    const RejectionSamplerOutput split = RunSplit(s, 3, {});
    CheckSameTokens(split, RunWhole(s, 3, {}));
    // Non-vacuous: the step really did emit four tokens, so the comparison above
    // was over something. Two empty results would compare equal.
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 4});
  }
  SUBCASE("early mismatch") {
    const VerifyStep s = MakeStep({{1, 2, 3}}, {{1, 5, 3, 4}});
    const RejectionSamplerOutput split = RunSplit(s, 3, {});
    CheckSameTokens(split, RunWhole(s, 3, {}));
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{1, 5});
    REQUIRE(split.num_rejected[0] == 2);
  }
  SUBCASE("placeholder draft id") {
    const VerifyStep s = MakeStep({{-1}}, {{5, 6}});
    const RejectionSamplerOutput split = RunSplit(s, 1, {});
    CheckSameTokens(split, RunWhole(s, 1, {}));
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{5});
  }
}

// A2-2.2 — THE MIXED STEP, and it is the case that matters.
//
// `num_reqs == 1` cannot separate a per-step reading of anything from a
// per-request one, which is the failure this repository has already paid for
// (#2710, and the note at the top of tests/vllm/v1/worker/
// test_combine_row_predicate.cpp). So the split is exercised on a batch whose
// rows carry DIFFERENT k: one row drafts nothing at all (k = 0, one expanded row
// — the shape the plain sampler would have handled), one accepts every draft,
// one rejects at its first. The expanded tensor is 1 + 3 + 4 = 8 rows for 3
// requests, so a reduction that used `num_reqs` where it needed `cu_num_logits`
// reads the wrong rows and this case says which request and which position.
TEST_CASE("A2-2: the split is token-identical on a MIXED step, num_reqs > 1") {
  const VerifyStep s = MakeStep({{}, {1, 2}, {7, 8, 9}},
                                {{4}, {1, 2, 6}, {7, 3, 9, 5}});
  const RejectionSamplerOutput split = RunSplit(s, 3, {});
  const RejectionSamplerOutput whole = RunWhole(s, 3, {});
  CheckSameTokens(split, whole);

  // Non-vacuous, and the three rows are genuinely different shapes:
  //   row 0: k = 0            -> one token, the plain greedy argmax
  //   row 1: k = 2, all accepted -> the two drafts plus the bonus
  //   row 2: k = 3, reject at 1  -> the first draft plus the target's argmax
  REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{4});
  REQUIRE(split.num_rejected[0] == 0);
  REQUIRE(split.sampled_token_ids[1] == std::vector<int32_t>{1, 2, 6});
  REQUIRE(split.num_rejected[1] == 0);
  REQUIRE(split.sampled_token_ids[2] == std::vector<int32_t>{7, 3});
  REQUIRE(split.num_rejected[2] == 2);
  // The rows the sampler was handed are NOT one per request, which is the whole
  // reason the route predicate is a per-step one.
  REQUIRE(s.num_logits == 8);
}

// A2-2.3 — the chunked-prefill zeroing lives in `finalize`, the HOST half, and
// must survive the split. It never reaches the kernel, so this is the one place
// the split could have dropped a rule silently.
TEST_CASE("A2-2: the split keeps the chunked-prefill zeroing, on a mixed step") {
  const VerifyStep s = MakeStep({{1, 2}, {3}}, {{1, 9, 5}, {3, 4}});
  const std::vector<char> prefilling{1, 0};
  const RejectionSamplerOutput split = RunSplit(s, 2, prefilling);
  CheckSameTokens(split, RunWhole(s, 2, prefilling));
  REQUIRE(split.num_sampled[0] == 0);
  REQUIRE(split.num_rejected[0] == 0);
  REQUIRE(split.num_sampled[1] == 2);
}

// A2-2.4 — `verify` leaves its outputs ON THE DEVICE and waits for nothing. The
// CPU backend cannot show the overlap that buys, but it CAN show the shape the
// runner's copy-queue route depends on: the walk is issued, the result is a
// live object carrying [num_reqs, width], and the bytes only appear on the host
// when the caller asks for them.
TEST_CASE("A2-2: verify returns device-resident outputs, shaped num_reqs x width") {
  const VerifyStep s = MakeStep({{}, {1, 2}}, {{4}, {1, 2, 6}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  vllm::v1::RejectionSamplerDeviceOutput dev =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  CHECK(dev.num_reqs() == 2);
  // Upstream's row stride: num_speculative_steps + 1
  // (rejection_sampler_utils.py:1026-1028).
  CHECK(dev.width() == 3);
  CHECK(dev.device().type == DeviceType::kCPU);

  // The download is a separate act, and it can happen more than once from the
  // same device result — which is what lets the runner put the wait on a queue
  // the walk did not run on.
  std::vector<int32_t> a(6), b(6), na(2), nb(2);
  dev.CopyToHost(q, a.data(), na.data());
  dev.CopyToHost(q, b.data(), nb.data());
  vt::GetBackend(dev.device().type).Synchronize(q);
  CHECK(a == b);
  CHECK(na == nb);
  CHECK(na[0] == 1);  // k = 0 row: exactly the one greedy argmax
  CHECK(na[1] == 3);  // k = 2, all accepted: two drafts plus the bonus
}

// A2-2.5 — the device result is MOVE-ONLY and owns what the kernel touches.
// Moving it must not double-free and must not lose the buffers, because the
// runner's copy-queue route holds it across the fork/copy/wait window.
TEST_CASE("A2-2: the device result moves without losing or double-freeing its buffers") {
  const VerifyStep s = MakeStep({{1, 2}}, {{1, 2, 6}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  vllm::v1::RejectionSamplerDeviceOutput src =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  vllm::v1::RejectionSamplerDeviceOutput dst = std::move(src);
  CHECK(src.num_reqs() == 0);  // moved-from is empty and safe to destroy
  CHECK(dst.num_reqs() == 1);
  CHECK(dst.width() == 3);
  std::vector<int32_t> out(3), ns(1);
  dst.CopyToHost(q, out.data(), ns.data());
  vt::GetBackend(dst.device().type).Synchronize(q);
  CHECK(ns[0] == 3);
  CHECK(out == std::vector<int32_t>{1, 2, 6});
}

// ─── A2-2 REPAIR: THE ACCEPT WALK'S SCRATCH IS PER CALL, NOT PER PROCESS ─────
//
// A fresh review found the ownership paragraph on `RejectionSamplerDeviceOutput`
// false: the accept walk reads SIX buffers and the object owned four. The one
// that mattered was the per-row argmax, which the CUDA arm kept in a file-scope
// grow-only global (`g_reject_argmax`) and `cudaFree`d from
// `EnsureRejectArgmaxScratch` whenever a later call needed more rows. `verify`
// returns with both kernels still queued, so that free lands under a still-
// queued accept kernel, and `target_argmax[start + i]` then reads whatever took
// the allocation's place. Speculative decoding is lossless, so the outcome is a
// wrong accept prefix and wrong emitted ids with no exception anywhere — reason
// A's class exactly (#1366). It is reachable with two runners in one process
// today, and reachable from one engine's own next step as soon as A2-4/A2-5 move
// the wait past the propose, which is what this type exists for.
//
// WHAT THIS TIER CAN AND CANNOT SHOW. On CPU the kernel has already finished
// when `verify` returns, so a shared scratch corrupts nothing here and no CPU
// test can turn the defect into a wrong token. What IS observable is the
// aliasing itself: two device results that are alive at the same time must hold
// DIFFERENT scratch. That is the property the fix installs, and it is the one a
// mutation can flip.
//
// RED-first, by mutation, since the buffer is new: making `verify` hand both
// calls one shared buffer (a function-static `int32_t*` grown on demand — the
// exact shape the CUDA global had) reds the pointer assertion below while every
// token case in this file stays green, which is the point: the tokens never
// moved on this backend either.
TEST_CASE("A2-2 repair: two live verifies own SEPARATE argmax scratch") {
  // The small step's argmaxes are 9 and 5, which appear NOWHERE in the big
  // step's [1, 2, 3, 7, 4, 5, 6, 8] at the same row. That is deliberate: with
  // the small step's obvious fixture ({{1}}, {{1, 2}}) its first two argmaxes
  // are 1 and 2, which are also the big step's first two, so the value
  // assertions below held under a shared buffer by pure fixture collision and
  // only the pointer inequality detected it.
  const VerifyStep small = MakeStep({{9}}, {{9, 5}});
  const VerifyStep big = MakeStep({{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3, 7}, {4, 5, 6, 8}});
  Queue q = Q();
  Tensor small_logits = Tensor::Contiguous(const_cast<float*>(small.logits.data()), DType::kF32,
                                           Cpu(), {small.num_logits, static_cast<int64_t>(kVocab)});
  Tensor big_logits = Tensor::Contiguous(const_cast<float*>(big.logits.data()), DType::kF32, Cpu(),
                                         {big.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(3);

  // The first walk is issued and NOT waited on, exactly as the copy-queue route
  // leaves it. The second is larger, which is precisely the input that made the
  // old grow-only global free the first one's buffer.
  vllm::v1::RejectionSamplerDeviceOutput first =
      sampler.verify(q, small_logits, small.draft_sampled, small.cu_num_logits);
  vllm::v1::RejectionSamplerDeviceOutput second =
      sampler.verify(q, big_logits, big.draft_sampled, big.cu_num_logits);

  REQUIRE(first.target_argmax_scratch() != nullptr);
  REQUIRE(second.target_argmax_scratch() != nullptr);
  CHECK(first.target_argmax_scratch() != second.target_argmax_scratch());

  // Non-vacuous: the scratch is the walk's real input, not an unused allocation.
  // On this backend it is host-readable, so assert the values the accept walk
  // read — the per-row argmaxes of each step's OWN logits, which is what a
  // shared buffer would have lost for the first result.
  //
  // WHICH ASSERTION DETECTS THE DEFECT. The pointer inequality above is the
  // deterministic one and it is the claim to make. The two value reads below
  // corroborate it: under a shared grow-only buffer they read a block the grow
  // has already freed, so what they see is whatever the allocator left there —
  // red in practice, not red by construction. Do not quote a failing-assertion
  // COUNT for that mutation; a read of freed memory does not have one.
  const int32_t* first_argmax = static_cast<const int32_t*>(first.target_argmax_scratch());
  CHECK(first_argmax[0] == 9);  // row 0 argmax: request 0's first target token
  CHECK(first_argmax[1] == 5);  // row 1 argmax: the bonus row
  const int32_t* second_argmax = static_cast<const int32_t*>(second.target_argmax_scratch());
  // The first two rows differ from the first step's 9 and 5, so the two value
  // assertions above cannot be satisfied by the big step's buffer. That the
  // fixtures do not collide is asserted here rather than trusted.
  CHECK(second_argmax[0] == 1);
  CHECK(second_argmax[1] == 2);
  CHECK(second_argmax[3] == 7);
  CHECK(second_argmax[4] == 4);
  CHECK(second_argmax[7] == 8);

  // And both results still decode correctly after the interleaving, read in the
  // order that would have been corrupted (the OLDER one last).
  std::vector<int32_t> big_sampled(static_cast<size_t>(second.num_reqs() * second.width()));
  std::vector<int32_t> big_ns(static_cast<size_t>(second.num_reqs()));
  second.CopyToHost(q, big_sampled.data(), big_ns.data());
  std::vector<int32_t> small_sampled(static_cast<size_t>(first.num_reqs() * first.width()));
  std::vector<int32_t> small_ns(static_cast<size_t>(first.num_reqs()));
  first.CopyToHost(q, small_sampled.data(), small_ns.data());
  vt::GetBackend(first.device().type).Synchronize(q);
  CHECK(small_ns[0] == 2);
  CHECK(big_ns[0] == 4);
  CHECK(big_ns[1] == 4);
  const RejectionSamplerOutput small_out = RejectionSampler::finalize(
      small_sampled, first.width(), small_ns, small.cu_num_logits, {});
  CHECK(small_out.sampled_token_ids[0] == std::vector<int32_t>{9, 5});
}

// ─── A2-2 REPAIR: THE EVENT CHOREOGRAPHY, RUN RATHER THAN READ ──────────────
//
// A fresh review found the copy-queue route's event sequence untested on every
// tier: `RunSplit` above waits with `Backend::Synchronize(q)` and never touches
// `RecordEvent`, `QueueWaitEvent`, `SynchronizeEvent` or a second queue, so the
// five-call sequence the runner actually issues existed only in inspection.
//
// This case issues that sequence in the runner's own order: record a fork event
// on one queue, make a second queue wait it, issue both D2H copies on the second
// queue, record a ready event there, block the host on that event alone.
//
// WHAT IT GATES, EXACTLY, AND WHAT IT CANNOT. It gates that the five calls
// compile against the real signatures, are well formed against the real buffers,
// and lose no tokens: the result is byte-identical to `forward`'s.
//
// It does NOT gate that the copy went anywhere. `CpuBackend::CreateQueue`
// returns `Queue{Device{kCPU, 0}, nullptr}` (src/vt/cpu/cpu_backend.cpp), which
// is byte-identical to this file's own `Q()` apart from the `id` field that no
// backend call reads. So `copy_q` and `main_q` are the SAME device and the SAME
// null handle, and `RecordEvent` / `QueueWaitEvent` / `SynchronizeEvent` are all
// the `vt::Backend` base no-ops (src/vt/backend.cpp). "The main queue is never
// synchronized" is therefore not a property this tier can hold — synchronizing
// `copy_q` here IS synchronizing `main_q` — and issuing the copies on `main_q`
// instead leaves this case green. That was measured, not assumed. The two-queue
// structure, and any overlap it buys, is G3/G4 at A2-5 and needs a GPU.
//
// The assertion below states that limit executably rather than in prose, so a
// reader who trusts the comment and a reader who runs the test learn the same
// thing.
TEST_CASE("A2-2 repair: the fork/copy/ready call sequence is well formed and token-identical") {
  const VerifyStep s = MakeStep({{}, {1, 2}, {7, 8, 9}},
                                {{4}, {1, 2, 6}, {7, 3, 9, 5}});
  vt::Backend& backend = vt::GetBackend(DeviceType::kCPU);
  Queue main_q = Q();
  Queue copy_q = backend.CreateQueue();
  vt::Event fork = backend.CreateEvent();
  vt::Event ready = backend.CreateEvent();

  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(3);
  std::vector<int32_t> host_sampled;
  std::vector<int32_t> host_num_sampled;
  int64_t width = 0;
  {
    vllm::v1::RejectionSamplerDeviceOutput dev =
        sampler.verify(main_q, logits, s.draft_sampled, s.cu_num_logits);
    const int64_t rows = dev.num_reqs();
    width = dev.width();
    host_sampled.assign(static_cast<size_t>(rows * width), 0);
    host_num_sampled.assign(static_cast<size_t>(rows), 0);

    // THE LIMIT, ASSERTED. On this tier the "second" queue is the same device and
    // the same handle as the main one, so nothing below can distinguish them and
    // this case makes no claim that it does.
    CHECK(copy_q.device.type == main_q.device.type);
    CHECK(copy_q.device.index == main_q.device.index);
    CHECK(copy_q.handle == main_q.handle);

    backend.RecordEvent(fork, main_q);
    backend.QueueWaitEvent(copy_q, fork);
    dev.CopyToHost(copy_q, host_sampled.data(), host_num_sampled.data());
    backend.RecordEvent(ready, copy_q);
    backend.SynchronizeEvent(ready);
    // `dev` is destroyed HERE: after the wait and while both queues are still
    // alive. That is the order its destructor requires — it drains the queues its
    // work is on before it frees (see rejection_sampler.h) — and it is the order
    // A2-4 has to keep when it moves the wait.
  }

  const RejectionSamplerOutput out = RejectionSampler::finalize(
      host_sampled, width, host_num_sampled, s.cu_num_logits, {});
  CheckSameTokens(out, RunWhole(s, 3, {}));
  // Non-vacuous, and the same three shapes A2-2.2 uses.
  REQUIRE(out.sampled_token_ids[0] == std::vector<int32_t>{4});
  REQUIRE(out.sampled_token_ids[1] == std::vector<int32_t>{1, 2, 6});
  REQUIRE(out.sampled_token_ids[2] == std::vector<int32_t>{7, 3});
  REQUIRE(out.num_rejected[2] == 2);

  backend.DestroyEvent(fork);
  backend.DestroyEvent(ready);
  backend.DestroyQueue(copy_q);
}

// ─── A2-2 REPAIR: THE DESTRUCTOR DRAINS BEFORE IT FREES ─────────────────────
//
// A fresh review found the ownership invariant said WHICH buffers the object
// owns and never said WHEN it may free them. `Release` called `Free` on all
// three device buffers with no wait of any kind, which is the `g_reject_argmax`
// defect relocated from the backend into this object's own scope.
//
// The caller that makes it real already exists. On the copy-queue route in
// `GPUModelRunner::sample_tokens_async` the device result is a block-scoped
// local, and A2-4's stated job is to move `SynchronizeEvent(verify_ready_event_)`
// past `propose_drafts`. Move it past that closing brace and the destructor
// frees `sampled_`, `num_sampled_` and `target_argmax_` while the D2H copy is
// still writing them: a garbage accept prefix, wrong emitted ids, nothing
// raised, and invisible to every token gate here because the verify is lossless.
//
// So the destructor now drains the queues the object's own work is on — the
// verify queue, and the queue of the last `CopyToHost` when that differs —
// before it frees. On CPU `Backend::Synchronize` is the base no-op, so no CPU
// test can turn the defect into a wrong token. What IS observable is the CALL
// ORDER, and that is the property the fix installs.
//
// RED-first, by mutation: deleting the two `Synchronize` calls from `Release`
// (its shape before this repair) makes `log[0]` a `free` and reds the three
// ordering assertions across the two cases below — measured, 2 cases and 3
// assertions failed, exit 1 — while every token case in this file stays green.
namespace {

// Forwards every backend call to the registered CPU backend and records the
// order of the two this case is about. The op table dispatches by DEVICE TYPE
// and not through `vt::Backend` (src/vt/ops.cpp), so the real CPU accept-walk
// kernel still runs underneath the spy.
class OrderSpyBackend final : public vt::Backend {
 public:
  explicit OrderSpyBackend(vt::Backend& inner) : inner_(inner) {}

  std::vector<std::string> log;

  void* Alloc(size_t bytes) override { return inner_.Alloc(bytes); }
  void Free(void* p) override {
    log.emplace_back("free");
    inner_.Free(p);
  }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    inner_.Memset(q, p, value, bytes);
  }
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    log.emplace_back("copy");
    inner_.Copy(q, dst, src, bytes);
  }
  Queue CreateQueue() override { return inner_.CreateQueue(); }
  void Synchronize(Queue& q) override {
    log.emplace_back("sync:" + std::to_string(q.id));
    inner_.Synchronize(q);
  }
  bool UnifiedMemory() const override { return inner_.UnifiedMemory(); }

 private:
  vt::Backend& inner_;
};

// Installs a backend for kCPU and restores whatever was there, so the rest of
// this binary is unaffected however the case exits.
class CpuBackendSwap {
 public:
  explicit CpuBackendSwap(vt::Backend& replacement)
      : prev_(&vt::GetBackend(DeviceType::kCPU)) {
    vt::RegisterBackend(DeviceType::kCPU, &replacement);
  }
  ~CpuBackendSwap() { vt::RegisterBackend(DeviceType::kCPU, prev_); }
  CpuBackendSwap(const CpuBackendSwap&) = delete;
  CpuBackendSwap& operator=(const CpuBackendSwap&) = delete;

 private:
  vt::Backend* prev_;
};

}  // namespace

TEST_CASE("A2-2 repair: the device result drains BOTH its queues before it frees") {
  const VerifyStep s = MakeStep({{1, 2}}, {{1, 2, 6}});
  vt::Backend& real = vt::GetBackend(DeviceType::kCPU);
  OrderSpyBackend spy(real);
  Queue main_q = Q();
  Queue copy_q = real.CreateQueue();
  REQUIRE(main_q.id != copy_q.id);  // distinguishable HERE even though the
                                    // backend cannot act on the difference
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(3);
  std::vector<int32_t> host_sampled;
  std::vector<int32_t> host_num_sampled;
  int64_t width = 0;

  {
    CpuBackendSwap swap(spy);
    vllm::v1::RejectionSamplerDeviceOutput dev =
        sampler.verify(main_q, logits, s.draft_sampled, s.cu_num_logits);
    width = dev.width();
    host_sampled.assign(static_cast<size_t>(dev.num_reqs() * dev.width()), 0);
    host_num_sampled.assign(static_cast<size_t>(dev.num_reqs()), 0);
    dev.CopyToHost(copy_q, host_sampled.data(), host_num_sampled.data());
    // Everything up to here is setup; the destructor is what this case asserts.
    spy.log.clear();
  }  // `dev` dies here, still under the swap and with both queues alive.

  // The drain comes FIRST, and it names both queues the object's work is on.
  REQUIRE(spy.log.size() >= 3);
  CHECK(spy.log[0] == "sync:" + std::to_string(main_q.id));
  CHECK(spy.log[1] == "sync:" + std::to_string(copy_q.id));
  CHECK(spy.log[2] == "free");
  // Nothing is freed early and drained afterwards.
  size_t frees = 0;
  bool sync_after_first_free = false;
  for (const std::string& entry : spy.log) {
    if (entry == "free") {
      ++frees;
    } else if (frees > 0 && entry.rfind("sync:", 0) == 0) {
      sync_after_first_free = true;
    }
  }
  CHECK(frees >= 3);  // sampled, num_sampled, target_argmax
  CHECK_FALSE(sync_after_first_free);

  // And the tokens the copy delivered are still the right ones, so the drain is
  // not standing in for a case that stopped computing anything.
  CHECK(host_num_sampled[0] == 3);
  const RejectionSamplerOutput out = RejectionSampler::finalize(
      host_sampled, width, host_num_sampled, s.cu_num_logits, {});
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 6});
}

TEST_CASE("A2-2 repair: a device result that was never copied drains ONE queue") {
  const VerifyStep s = MakeStep({{1, 2}}, {{1, 2, 6}});
  vt::Backend& real = vt::GetBackend(DeviceType::kCPU);
  OrderSpyBackend spy(real);
  Queue main_q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(3);

  {
    CpuBackendSwap swap(spy);
    vllm::v1::RejectionSamplerDeviceOutput dev =
        sampler.verify(main_q, logits, s.draft_sampled, s.cu_num_logits);
    spy.log.clear();
  }

  REQUIRE(spy.log.size() >= 2);
  CHECK(spy.log[0] == "sync:" + std::to_string(main_q.id));
  CHECK(spy.log[1] == "free");  // exactly one drain: there is no copy queue
}

// SKIPPED (test-porting rule 6), tracked to M-mtp-3 (spec §5):
//   * test_stochastic_rejection_sample (test_rejection_sampler_utils.py:141) and
//     test_synthetic_rejection_sample (:215) — the Gumbel / probability-ratio
//     path is out of scope for I3 (greedy only).
//   * test_block_verification_rejection_sample (:325) and
//     test_block_verification_matches_standard (:372) — block verification.
//   * tests/v1/worker/test_gpu_rejection_sampler_i64.py:109 (>2^31 logits-buffer
//     indexing) — needs the block-verification buffer layout.

// QWEN38-DFLASH2-STAGE-C W3: the loop's accept rule is the bundle's greedy
// cascade at the DFlash2 block shape (k = 8, the drafter's [anchor, MASK x 7]
// row count). Two cases pin the two commit contracts the loop relies on:
//   * full accept: the longest accepted prefix is the whole 8-row draft and the
//     commit is draft[0..7] + the verify bonus (dflash2_serving.py
//     `_accept_greedy` / `com = [pending] + drafts[:m] + row_ids[mi_u]`);
//   * reject at depth 3: the commit is the 3-token accepted prefix + the first
//     differing VERIFY token, and num_rejected = 5 — the GDN rollback length
//     the stage-B commit-or-restore restores on the next step.
TEST_CASE("rejection_sampler: dflash2 k=8 cascade — full accept commits draft plus bonus") {
  const RejectionSamplerOutput out =
      Run({{1, 2, 3, 4, 5, 6, 7, 8}},
          {{1, 2, 3, 4, 5, 6, 7, 8, 9}},
          /*k=*/8);
  REQUIRE(out.sampled_token_ids.size() == 1);
  CHECK(out.sampled_token_ids[0] ==
        std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  CHECK(out.num_sampled[0] == 9);  // 8 accepted drafts + the bonus
  CHECK(out.num_rejected[0] == 0);
}

TEST_CASE("rejection_sampler: dflash2 k=8 cascade — reject at depth 3 commits prefix plus target argmax") {
  // Drafts 1 2 3 match; draft 4 does not (target argmax there is 9).
  // Nothing past the first differing position may be accepted, even though
  // drafts 5..8 coincidentally match the target rows after 9.
  const RejectionSamplerOutput out =
      Run({{1, 2, 3, 4, 5, 6, 7, 8}},
          {{1, 2, 3, 9, 5, 6, 7, 8, 6}},
          /*k=*/8);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 9});
  CHECK(out.num_sampled[0] == 4);   // 3 accepted drafts + the replacement
  CHECK(out.num_rejected[0] == 5);  // the GDN rollback length
  CHECK(Padded(out, 9)[0] ==
        std::vector<int32_t>{1, 2, 3, 9, -1, -1, -1, -1, -1});
}
