// SPEC-MTP I5b — the drafter prefill input-prep (shift-splice) unit test.
//
// Ported from vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
// `_prepare_prefill_inputs_kernel` (:469-549) + `prepare_prefill_inputs`
// (:552-588) @ e24d1b24. There is no dedicated upstream UNIT test for this
// kernel (it is covered via the e2e spec-decode suite); as in
// tests/vllm/v1/worker/test_combine_tokens.cpp, the oracle is therefore derived
// DIRECTLY from the kernel algorithm (per request: shift input_ids left one
// within the query span, splice the just-sampled next token into the freed last
// slot, query_len -= num_rejected, copy positions unchanged, reuse the target
// query_start_loc / seq_lens), and every expected array is written out by hand.
//
// The four representative cases the I5b gate calls for:
//   * perfect-accept  (k accepted -> shift by k+1)
//   * early-mismatch  (accept j -> shift by j+1, query_len -= (k-j))     [k=1 and k=3]
//   * single-token    (k=0 -> plain next token)
//   * multi-request   (batch with different k_i)
// plus the chunked-prefill branch (num_sampled==0 -> next_prefill_tokens) and the
// CUDA-graph request-count padding.
//
// RED-first (proven by experiment, see the I5b report): with a NO-OP / no-shift
// stub for prepare_prefill_inputs every "draft span == shifted+spliced"
// assertion below fails — the drafter would consume the un-shifted verify tokens
// and never see the just-sampled next token. That is the RED; the real routine
// is the GREEN.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <vector>

#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.h"

using vllm::v1::prepare_prefill_inputs;
using vllm::v1::SpecPrefillInputs;

namespace {
// Identity idx_mapping (our persistent batch is condensed dense — the same
// convention as the rejection-sampler / combine tests).
std::vector<int32_t> Identity(int num_reqs) {
  std::vector<int32_t> m(static_cast<size_t>(num_reqs));
  std::iota(m.begin(), m.end(), 0);
  return m;
}
}  // namespace

// ─── perfect-accept: k=1, all drafts accepted -> shift by k+1=2 ──────────────
TEST_CASE("prepare_prefill: perfect-accept k=1 shifts and splices the bonus") {
  // One decoding request, verify span = 1 + k = 2 tokens [t0=10, t1=11] at
  // positions [5,6]. All k=1 drafts accepted: num_sampled = accepted+1 = 2,
  // num_rejected = 0, last_sampled (the bonus) = 99.
  const std::vector<int32_t> in_ids = {10, 11};
  const std::vector<int64_t> pos = {5, 6};
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<int32_t> seq_lens = {20};
  const std::vector<int32_t> num_sampled = {2};
  const std::vector<int32_t> num_rejected = {0};
  const std::vector<int32_t> last_sampled = {99};
  const std::vector<int32_t> next_prefill = {0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, /*max_num_reqs=*/1);

  // Draft span [t1, bonus] = [11, 99]: t0 dropped, rest shifted left, bonus spliced.
  CHECK(out.input_ids == std::vector<int32_t>{11, 99});
  // Positions unchanged.
  CHECK(out.positions == std::vector<int64_t>{5, 6});
  CHECK(out.query_start_loc == std::vector<int32_t>{0, 2});
  CHECK(out.seq_lens == std::vector<int32_t>{20});
  CHECK(out.last_token_indices == std::vector<int64_t>{1});  // query_start + query_len - 1
  CHECK(out.current_draft_step == 0);
}

// ─── early-mismatch k=1: accept j=0 -> shift by j+1=1, query_len -= 1 ─────────
TEST_CASE("prepare_prefill: early-mismatch k=1 (j=0) emits just the replacement") {
  const std::vector<int32_t> in_ids = {10, 11};
  const std::vector<int64_t> pos = {5, 6};
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<int32_t> seq_lens = {20};
  // 0 accepted: num_sampled = 1, num_rejected = k - j = 1. next token = the
  // target-argmax replacement = 99.
  const std::vector<int32_t> num_sampled = {1};
  const std::vector<int32_t> num_rejected = {1};
  const std::vector<int32_t> last_sampled = {99};
  const std::vector<int32_t> next_prefill = {0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, 1);

  // query_len = 2 - 1 = 1: the span of interest is [99] at slot 0; slot 1 is
  // don't-care padding (holds the target token 11 — never sampled, overwritten
  // next step).
  CHECK(out.input_ids[0] == 99);
  CHECK(out.last_token_indices == std::vector<int64_t>{0});  // query_start + 1 - 1
  CHECK(out.positions == std::vector<int64_t>{5, 6});
  CHECK(out.query_start_loc == std::vector<int32_t>{0, 2});  // metadata UNCHANGED by nr
}

// ─── early-mismatch k=3: accept j=1 -> shift by j+1=2, query_len -= (k-j)=2 ───
// The general (k>1) shape: not the k=1 production path, but it pins the
// query_len -= num_rejected arithmetic for j strictly between 0 and k.
TEST_CASE("prepare_prefill: early-mismatch k=3 (j=1) shifts by j+1 and reduces query_len") {
  const std::vector<int32_t> in_ids = {20, 21, 22, 23};  // verify span 1+k=4
  const std::vector<int64_t> pos = {7, 8, 9, 10};
  const std::vector<int32_t> qsl = {0, 4};
  const std::vector<int32_t> seq_lens = {30};
  // 1 accepted of 3: num_sampled = 2, num_rejected = 3 - 1 = 2. next token = 88.
  const std::vector<int32_t> num_sampled = {2};
  const std::vector<int32_t> num_rejected = {2};
  const std::vector<int32_t> last_sampled = {88};
  const std::vector<int32_t> next_prefill = {0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, 1);

  // query_len = 4 - 2 = 2. Draft slot0 = t1 = 21 (shift), slot1 = 88 (splice);
  // slots 2,3 don't-care tail (22,23).
  CHECK(out.input_ids[0] == 21);
  CHECK(out.input_ids[1] == 88);
  CHECK(out.last_token_indices == std::vector<int64_t>{1});  // query_start + 2 - 1
}

// ─── single-token: k=0 -> plain next token, no shift ─────────────────────────
TEST_CASE("prepare_prefill: single-token k=0 is a plain next-token splice") {
  const std::vector<int32_t> in_ids = {30};  // verify span 1 (no drafts)
  const std::vector<int64_t> pos = {4};
  const std::vector<int32_t> qsl = {0, 1};
  const std::vector<int32_t> seq_lens = {15};
  const std::vector<int32_t> num_sampled = {1};  // 0 accepted + 1 bonus
  const std::vector<int32_t> num_rejected = {0};
  const std::vector<int32_t> last_sampled = {77};
  const std::vector<int32_t> next_prefill = {0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, 1);

  // query_len = 1: shift loop empty, slot0 = next token 77.
  CHECK(out.input_ids == std::vector<int32_t>{77});
  CHECK(out.positions == std::vector<int64_t>{4});
  CHECK(out.last_token_indices == std::vector<int64_t>{0});
}

// ─── multi-request: batch with different k_i ─────────────────────────────────
TEST_CASE("prepare_prefill: multi-request batch, different k_i") {
  // req0: k=1 perfect accept; req1: k=1 mismatch (j=0); req2: k=0 single.
  const std::vector<int32_t> in_ids = {10, 11, 12, 13, 14};
  const std::vector<int64_t> pos = {5, 6, 3, 4, 9};
  const std::vector<int32_t> qsl = {0, 2, 4, 5};
  const std::vector<int32_t> seq_lens = {20, 21, 22};
  const std::vector<int32_t> num_sampled = {2, 1, 1};
  const std::vector<int32_t> num_rejected = {0, 1, 0};
  const std::vector<int32_t> last_sampled = {99, 98, 97};
  const std::vector<int32_t> next_prefill = {0, 0, 0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(3), last_sampled, next_prefill,
      num_sampled, num_rejected, /*max_num_reqs=*/3);

  // req0 [0,2): shift -> slot0=t1=11, slot1=bonus 99.
  // req1 [2,4): query_len=1 -> slot2=replacement 98; slot3 don't-care tail = 13.
  // req2 [4,5): slot4=next token 97.
  CHECK(out.input_ids == std::vector<int32_t>{11, 99, 98, 13, 97});
  CHECK(out.positions == std::vector<int64_t>{5, 6, 3, 4, 9});  // unchanged
  CHECK(out.query_start_loc == std::vector<int32_t>{0, 2, 4, 5});
  CHECK(out.seq_lens == std::vector<int32_t>{20, 21, 22});
  CHECK(out.last_token_indices == std::vector<int64_t>{1, 2, 4});
  CHECK(out.current_draft_step == 0);
}

// ─── chunked-prefill: num_sampled==0 -> splice next_prefill_tokens ───────────
TEST_CASE("prepare_prefill: chunked-prefill row splices next_prefill_tokens") {
  const std::vector<int32_t> in_ids = {40, 41, 42};  // a 3-token prefill chunk
  const std::vector<int64_t> pos = {0, 1, 2};
  const std::vector<int32_t> qsl = {0, 3};
  const std::vector<int32_t> seq_lens = {3};
  // Chunked-prefill row: RejectionSampler zeroed BOTH num_sampled and num_rejected.
  const std::vector<int32_t> num_sampled = {0};
  const std::vector<int32_t> num_rejected = {0};
  const std::vector<int32_t> last_sampled = {-1};  // must NOT be read on this branch
  const std::vector<int32_t> next_prefill = {55};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, 1);

  // query_len = 3 (no rejection): shift slot0=t1=41, slot1=t2=42, slot2=next
  // prefill 55.
  CHECK(out.input_ids == std::vector<int32_t>{41, 42, 55});
  CHECK(out.last_token_indices == std::vector<int64_t>{2});
}

// ─── CUDA-graph request-count padding ────────────────────────────────────────
TEST_CASE("prepare_prefill: pads request-indexed arrays to max_num_reqs") {
  const std::vector<int32_t> in_ids = {10, 11};
  const std::vector<int64_t> pos = {5, 6};
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<int32_t> seq_lens = {20};
  const std::vector<int32_t> num_sampled = {2};
  const std::vector<int32_t> num_rejected = {0};
  const std::vector<int32_t> last_sampled = {99};
  const std::vector<int32_t> next_prefill = {0};

  const SpecPrefillInputs out = prepare_prefill_inputs(
      in_ids, pos, qsl, seq_lens, Identity(1), last_sampled, next_prefill,
      num_sampled, num_rejected, /*max_num_reqs=*/4);

  // query_start_loc padded to the total token count (2) past req0; seq_lens and
  // last_token_indices zero-padded past num_reqs=1.
  CHECK(out.query_start_loc == std::vector<int32_t>{0, 2, 2, 2, 2});
  CHECK(out.seq_lens == std::vector<int32_t>{20, 0, 0, 0});
  CHECK(out.last_token_indices == std::vector<int64_t>{1, 0, 0, 0});
}


TEST_CASE("prepare_prefill: executed native vision MTP1 input contract") {
  std::ifstream stream(std::filesystem::path(__FILE__).parent_path() /
                       "fixtures/native_vision_mtp/transition.json");
  REQUIRE(stream.good());
  const auto fixture = nlohmann::json::parse(stream);
  const auto ids = fixture.at("input_ids").get<std::vector<int32_t>>();
  const auto positions = fixture.at("positions").get<std::vector<int64_t>>();
  const int32_t tokens = static_cast<int32_t>(ids.size());
  const auto prepared = prepare_prefill_inputs(
      ids, positions, {0, tokens}, {tokens}, Identity(1),
      fixture.at("last_sampled").get<std::vector<int32_t>>(),
      fixture.at("next_prefill_tokens").get<std::vector<int32_t>>(),
      fixture.at("num_sampled").get<std::vector<int32_t>>(),
      fixture.at("num_rejected").get<std::vector<int32_t>>(), 1);
  CHECK(prepared.input_ids == fixture.at("expected_draft_input_ids").get<std::vector<int32_t>>());
  CHECK(prepared.positions == fixture.at("expected_draft_positions").get<std::vector<int64_t>>());
  CHECK(prepared.last_token_indices == std::vector<int64_t>{tokens - 1});
  CHECK(prepared.query_start_loc == std::vector<int32_t>{0, tokens});
  CHECK(prepared.seq_lens == std::vector<int32_t>{tokens});
}
