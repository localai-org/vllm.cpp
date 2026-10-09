// Tests for the persistent InputBatch + CachedRequestState (M1.5 Task 2) — the
// num_reqs-major per-slot arrays the model runner keeps alive across steps.
//
// Ported from vllm/v1/worker/gpu_input_batch.py @ e24d1b24, with the behavioral
// oracle taken from tests/v1/worker/test_gpu_input_batch.py (the
// add_request / _remove_requests / condense / _compare_objs pattern:
// test_sampling_metadata_in_input_batch adds a batch, removes a subset, then
// condense()s and checks the dense state). Those upstream tests drive a random
// batch through SamplingMetadata; here — SamplingMetadata is not yet landed
// (M1.7) — we assert the same underlying per-slot array + req_id_to_index +
// block-table densification directly, plus the MRV2-contract from_new_request
// seed (prefill_token_ids). See the header for the deferred slot state.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/worker/gpu/input_batch.h"

using vllm::SamplingParams;
using vllm::v1::CachedRequestState;
using vllm::v1::InputBatch;
using vllm::v1::NewRequestData;

namespace {

// Build a single-group CachedRequestState directly (the V1 add-request input).
CachedRequestState make_req(const std::string& req_id,
                            std::vector<int32_t> prompt,
                            std::vector<int32_t> output,
                            std::vector<int> block_ids,
                            SamplingParams sp = SamplingParams{}) {
  CachedRequestState state;
  state.req_id = req_id;
  state.prompt_token_ids = std::move(prompt);
  state.output_token_ids = std::move(output);
  state.sampling_params = sp;
  state.block_ids = {std::move(block_ids)};
  state.num_computed_tokens = state.num_prompt_tokens;  // overwritten below
  state.finalize();
  return state;
}

InputBatch make_batch(int max_num_reqs = 8, int max_model_len = 64,
                      int num_speculative_steps = 0) {
  return InputBatch(/*max_num_reqs=*/max_num_reqs,
                    /*max_model_len=*/max_model_len,
                    /*max_num_batched_tokens=*/max_model_len,
                    /*vocab_size=*/1024, /*block_sizes=*/{16},
                    /*kernel_block_sizes=*/{16},
                    /*num_speculative_steps=*/num_speculative_steps);
}

}  // namespace

TEST_CASE("add_request fills a slot: token ids, counts, block rows, sampling") {
  InputBatch batch = make_batch();

  CachedRequestState req = make_req("r0", {10, 11, 12}, {20, 21}, {3, 7});
  req.num_computed_tokens = 3;
  const int idx = batch.add_request(req);

  CHECK(idx == 0);
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.at("r0") == 0);
  CHECK(batch.req_ids[0].has_value());
  CHECK(*batch.req_ids[0] == "r0");

  // token_ids_cpu seeded prompt then output (== prefill_token_ids).
  CHECK(batch.token_id(0, 0) == 10);
  CHECK(batch.token_id(0, 1) == 11);
  CHECK(batch.token_id(0, 2) == 12);
  CHECK(batch.token_id(0, 3) == 20);
  CHECK(batch.token_id(0, 4) == 21);

  CHECK(batch.num_prompt_tokens[0] == 3);
  CHECK(batch.num_tokens_no_spec[0] == 5);  // 3 prompt + 2 output
  CHECK(batch.num_computed_tokens_cpu[0] == 3);

  // Block-table rows added for group 0.
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 2);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 3);
  CHECK(batch.block_table[0].cpu_block_id(0, 1) == 7);

  // Default sampling params (temperature 1.0) => random, no top_p/top_k.
  CHECK(batch.all_random());
  CHECK_FALSE(batch.all_greedy());
  CHECK(batch.no_top_p());
  CHECK(batch.no_top_k());
  CHECK(batch.no_penalties());
}

TEST_CASE("add_request assigns sequential slots and tracks num_reqs") {
  InputBatch batch = make_batch();
  CHECK(batch.add_request(make_req("a", {1}, {}, {0})) == 0);
  CHECK(batch.add_request(make_req("b", {2}, {}, {1})) == 1);
  CHECK(batch.add_request(make_req("c", {3}, {}, {2})) == 2);
  CHECK(batch.num_reqs() == 3);
}

TEST_CASE("sampling predicates reflect per-request params") {
  InputBatch batch = make_batch();

  SamplingParams greedy;
  greedy.temperature = 0.0;  // Type() => greedy
  SamplingParams topk;
  topk.top_k = 5;
  SamplingParams topp;
  topp.top_p = 0.9;
  SamplingParams pen;
  pen.presence_penalty = 0.5;

  batch.add_request(make_req("g", {1}, {}, {0}, greedy));
  CHECK(batch.all_greedy());
  CHECK_FALSE(batch.all_random());
  CHECK(batch.temperature_cpu[0] == doctest::Approx(0.0));

  batch.add_request(make_req("k", {2}, {}, {1}, topk));
  CHECK_FALSE(batch.no_top_k());
  CHECK(batch.top_k_cpu[1] == 5);
  // Mixed greedy + random now.
  CHECK_FALSE(batch.all_greedy());
  CHECK_FALSE(batch.all_random());

  batch.add_request(make_req("p", {3}, {}, {2}, topp));
  CHECK_FALSE(batch.no_top_p());
  CHECK(batch.top_p_cpu[2] == doctest::Approx(0.9));

  batch.add_request(make_req("n", {4}, {}, {3}, pen));
  CHECK_FALSE(batch.no_penalties());
}

TEST_CASE("remove_request frees the slot and returns its index") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {5}));
  batch.add_request(make_req("b", {2}, {}, {6}));

  const std::optional<int> removed = batch.remove_request("a");
  REQUIRE(removed.has_value());
  CHECK(*removed == 0);
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.count("a") == 0);
  CHECK_FALSE(batch.req_ids[0].has_value());
  // Block row cleared.
  CHECK(batch.block_table[0].num_blocks_per_row[0] == 0);

  // Removing an unknown req returns nullopt.
  CHECK_FALSE(batch.remove_request("zzz").has_value());
}

TEST_CASE("condense: remove the middle request then densify [0, num_reqs)") {
  InputBatch batch = make_batch();
  // Three requests with distinct blocks + token ids.
  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200, 201}, {}, {20, 21});
  r1.num_computed_tokens = 2;
  CachedRequestState r2 = make_req("r2", {300}, {}, {30});
  r2.num_computed_tokens = 1;
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);
  REQUIRE(batch.num_reqs() == 3);

  // Remove the middle request (slot 1).
  const std::optional<int> removed = batch.remove_request("r1");
  REQUIRE(removed.has_value());
  CHECK(*removed == 1);
  CHECK(batch.num_reqs() == 2);

  batch.condense();

  // r2 (was slot 2) slides down into the freed slot 1; [0, 2) is now dense.
  CHECK(batch.num_reqs() == 2);
  CHECK(static_cast<int>(batch.req_ids.size()) == 2);
  CHECK(batch.req_id_to_index.at("r0") == 0);
  CHECK(batch.req_id_to_index.at("r2") == 1);
  CHECK(batch.req_id_to_index.count("r1") == 0);
  REQUIRE(batch.req_ids[1].has_value());
  CHECK(*batch.req_ids[1] == "r2");

  // r2's per-slot arrays moved into slot 1.
  CHECK(batch.token_id(1, 0) == 300);
  CHECK(batch.num_prompt_tokens[1] == 1);
  CHECK(batch.num_tokens_no_spec[1] == 1);
  CHECK(batch.num_computed_tokens_cpu[1] == 1);

  // r2's block-table row moved into slot 1.
  CHECK(batch.block_table[0].num_blocks_per_row[1] == 1);
  CHECK(batch.block_table[0].cpu_block_id(1, 0) == 30);

  // Slot 0 (r0) is untouched.
  CHECK(*batch.req_ids[0] == "r0");
  CHECK(batch.token_id(0, 0) == 100);
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 10);
}

// ─── async-scheduling per-slot state (ENG-ASYNC-SCHED W3 runner leaf) ────────
TEST_CASE("add_request seeds prefill_len; last_sampled only for resumed reqs") {
  InputBatch batch = make_batch();

  // Fresh prefill (num_computed == 0): prefill_len == num_tokens (prompt+output),
  // last_sampled stays 0 (combine never reads it during prefill).
  CachedRequestState fresh = make_req("fresh", {10, 11, 12}, {}, {3});
  fresh.num_computed_tokens = 0;
  const int i0 = batch.add_request(fresh);
  CHECK(batch.prefill_len[static_cast<size_t>(i0)] == 3);        // 3 prompt
  CHECK(batch.last_sampled_tokens[static_cast<size_t>(i0)] == 0);

  // Resumed / PD-disagg (0 < num_computed <= prefill_len): last_sampled seeded
  // with the token at num_computed-1 so the first decode reads the right id.
  CachedRequestState resumed = make_req("resumed", {20, 21, 22}, {23, 24}, {7});
  resumed.num_computed_tokens = 4;  // prefill_len = 5 (prompt3 + output2)
  const int i1 = batch.add_request(resumed);
  CHECK(batch.prefill_len[static_cast<size_t>(i1)] == 5);
  // token at index num_computed-1 == 3 -> the seed row is [20,21,22,23,24] -> 23.
  CHECK(batch.last_sampled_tokens[static_cast<size_t>(i1)] == 23);
}

TEST_CASE("condense moves last_sampled_tokens + prefill_len with the request") {
  InputBatch batch = make_batch();
  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200, 201}, {}, {20});
  r1.num_computed_tokens = 2;
  // r2 resumed so its last_sampled is a distinctive non-zero seed.
  CachedRequestState r2 = make_req("r2", {300, 301}, {302}, {30});
  r2.num_computed_tokens = 3;  // prefill_len 3 -> seed token at idx 2 == 302
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);
  REQUIRE(batch.last_sampled_tokens[2] == 302);
  REQUIRE(batch.prefill_len[2] == 3);

  batch.remove_request("r1");
  batch.condense();  // r2 (slot 2) slides into freed slot 1

  CHECK(batch.req_id_to_index.at("r2") == 1);
  CHECK(batch.last_sampled_tokens[1] == 302);  // moved with the request
  CHECK(batch.prefill_len[1] == 3);
  // Slot 0 untouched.
  CHECK(batch.prefill_len[0] == 1);
}

TEST_CASE("swap_states swaps last_sampled_tokens + prefill_len") {
  InputBatch batch = make_batch();
  CachedRequestState a = make_req("a", {1, 2}, {3}, {10});
  a.num_computed_tokens = 3;  // prefill_len 3, seed idx2 == 3
  CachedRequestState b = make_req("b", {5}, {}, {20});
  b.num_computed_tokens = 0;  // prefill_len 1, last_sampled 0
  batch.add_request(a);  // slot 0
  batch.add_request(b);  // slot 1
  REQUIRE(batch.last_sampled_tokens[0] == 3);
  REQUIRE(batch.prefill_len[0] == 3);

  batch.swap_states(0, 1);

  // The reorder primitive must carry the async state with the row.
  CHECK(batch.req_id_to_index.at("a") == 1);
  CHECK(batch.req_id_to_index.at("b") == 0);
  CHECK(batch.last_sampled_tokens[1] == 3);
  CHECK(batch.prefill_len[1] == 3);
  CHECK(batch.last_sampled_tokens[0] == 0);
  CHECK(batch.prefill_len[0] == 1);
}

// ─── SPEC-DFLASH2 A2-3: the per-req_state DRAFT BUFFER ──────────────────────
//
// Mirrors `RequestStates.draft_tokens` (vllm/v1/worker/gpu/states.py:71-77 @ pin
// 5559679229bc961848b121ccdeaa8fa5d79bec98), the persistent
// [num_req_states, num_speculative_steps] tensor the combine's draft scatter
// reads. The per-slot valid count beside it is OURS and has no upstream twin:
// the anchor A2-3 gave for it, `gpu_model_runner.py:883-895`, is the LEGACY
// runner's n-gram-GPU D2H buffer set at that pin, and the new runner takes each
// row's draft length from the scheduler's list instead
// (`vllm/v1/worker/gpu/model_runner.py:941-949`) — which we cannot, because
// under async scheduling that list is `-1` placeholders.
//
// WHY THE SIZING IS AN ASSERTION AND NOT AN IMPLEMENTATION DETAIL. The combine's
// scatter indexes `draft_tokens[req_state_idx * stride + b]`, and its CUDA
// counterpart (`src/vt/cuda/cuda_combine_tokens.cu`) has NOTHING bounding that
// read — the row spec records that gap and tells this wave to close it by sizing
// the buffer by the req_state POOL. A buffer sized by this step's REQUEST COUNT
// lets a high slot index past the allocation with every host check satisfied.
// The loud outcome is an illegal access; the quiet one is garbage in the draft
// slots, which costs acceptance and NOTHING ELSE because speculative decoding is
// lossless — the class of defect no token gate in this tree can see.
TEST_CASE("A2-3: the draft buffer is sized by the req_state POOL, not by num_reqs") {
  InputBatch batch = make_batch(/*max_num_reqs=*/8, /*max_model_len=*/64,
                                /*num_speculative_steps=*/3);
  // Nothing admitted yet: the rows exist for every slot the pool can hand out.
  CHECK(batch.num_reqs() == 0);
  CHECK(batch.num_speculative_steps == 3);
  CHECK(batch.draft_tokens.size() == 8u * 3u);
  CHECK(batch.num_valid_draft_tokens.size() == 8u);

  // One request admitted does not shrink it, which is the whole point: the
  // scatter reads by req_state slot and slot 7 stays addressable.
  CachedRequestState r0 = make_req("r0", {1, 2}, {}, {10});
  r0.num_computed_tokens = 0;
  batch.add_request(r0);
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.draft_tokens.size() == 8u * 3u);
}

// A runner with no speculator carries a ZERO-WIDTH buffer, exactly as upstream
// builds `torch.zeros(max_num_reqs, 0)` when num_speculative_steps is 0. The
// non-speculative path therefore allocates nothing and reads nothing, which is
// what keeps this wave byte-identical for every model shipping today.
TEST_CASE("A2-3: a non-speculative batch carries a zero-width draft buffer") {
  InputBatch batch = make_batch();  // num_speculative_steps defaults to 0
  CHECK(batch.num_speculative_steps == 0);
  CHECK(batch.draft_tokens.empty());
  CHECK(batch.num_valid_draft_tokens.size() == 8u);
}

// states.py:113 — `self.draft_tokens[req_idx].zero_()` in add_request. A freed
// slot is handed to the NEXT request, and without this the new request inherits
// the previous occupant's drafts. Those drafts verify against a sequence they
// were never proposed for: lossless verify means they are simply rejected, so
// the emitted tokens stay correct and only acceptance falls. Nothing raises.
TEST_CASE("A2-3: add_request zeroes the slot's draft row and its valid count") {
  InputBatch batch = make_batch(/*max_num_reqs=*/4, /*max_model_len=*/64,
                                /*num_speculative_steps=*/2);
  CachedRequestState first = make_req("first", {1, 2}, {}, {10});
  first.num_computed_tokens = 0;
  const int slot = batch.add_request(first);
  REQUIRE(slot == 0);

  // Give the slot drafts, as a propose would.
  batch.draft_tokens[0] = 77;
  batch.draft_tokens[1] = 88;
  batch.num_valid_draft_tokens[0] = 2;

  batch.remove_request("first");
  batch.condense();
  CachedRequestState second = make_req("second", {5, 6}, {}, {20});
  second.num_computed_tokens = 0;
  const int reused = batch.add_request(second);
  REQUIRE(reused == 0);  // the freed hole is refilled first

  CHECK(batch.draft_tokens[0] == 0);
  CHECK(batch.draft_tokens[1] == 0);
  CHECK(batch.num_valid_draft_tokens[0] == 0);
}

// The abort/finish reorder the combine's `idx_mapping` indirection exists for.
// After a condense the surviving request occupies a DIFFERENT req_state slot, and
// the scatter reads its drafts by slot. A row left behind is the previous
// occupant's drafts read for this request — again lossless, again invisible.
TEST_CASE("A2-3: condense moves the draft row and its count with the request") {
  InputBatch batch = make_batch(/*max_num_reqs=*/4, /*max_model_len=*/64,
                                /*num_speculative_steps=*/2);
  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200}, {}, {20});
  r1.num_computed_tokens = 1;
  CachedRequestState r2 = make_req("r2", {300}, {}, {30});
  r2.num_computed_tokens = 1;
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);

  // Distinct drafts per slot so a row that fails to move is not mistaken for one
  // that moved to an identical value.
  batch.draft_tokens[0] = 10;
  batch.draft_tokens[1] = 11;
  batch.num_valid_draft_tokens[0] = 2;
  batch.draft_tokens[2] = 20;
  batch.draft_tokens[3] = 21;
  batch.num_valid_draft_tokens[1] = 2;
  batch.draft_tokens[4] = 30;
  batch.draft_tokens[5] = 31;
  batch.num_valid_draft_tokens[2] = 1;

  batch.remove_request("r1");
  batch.condense();  // r2 (slot 2) slides into freed slot 1

  REQUIRE(batch.req_id_to_index.at("r2") == 1);
  CHECK(batch.draft_tokens[2] == 30);
  CHECK(batch.draft_tokens[3] == 31);
  CHECK(batch.num_valid_draft_tokens[1] == 1);
  // Slot 0 untouched.
  CHECK(batch.draft_tokens[0] == 10);
  CHECK(batch.num_valid_draft_tokens[0] == 2);
}

TEST_CASE("A2-3: swap_states swaps the draft rows and their counts") {
  InputBatch batch = make_batch(/*max_num_reqs=*/4, /*max_model_len=*/64,
                                /*num_speculative_steps=*/2);
  CachedRequestState a = make_req("a", {1}, {}, {10});
  a.num_computed_tokens = 1;
  CachedRequestState b = make_req("b", {5}, {}, {20});
  b.num_computed_tokens = 1;
  batch.add_request(a);  // slot 0
  batch.add_request(b);  // slot 1

  batch.draft_tokens[0] = 41;
  batch.draft_tokens[1] = 42;
  batch.num_valid_draft_tokens[0] = 2;
  batch.draft_tokens[2] = 51;
  batch.draft_tokens[3] = 52;
  batch.num_valid_draft_tokens[1] = 1;

  batch.swap_states(0, 1);

  CHECK(batch.req_id_to_index.at("a") == 1);
  CHECK(batch.draft_tokens[2] == 41);
  CHECK(batch.draft_tokens[3] == 42);
  CHECK(batch.num_valid_draft_tokens[1] == 2);
  CHECK(batch.draft_tokens[0] == 51);
  CHECK(batch.draft_tokens[1] == 52);
  CHECK(batch.num_valid_draft_tokens[0] == 1);
}

// ─── ENG-ASYNC-SCHED W4: the structural-op log a device mirror replays ───────
//
// On a discrete GPU last_sampled_tokens lives on the device, so the host cannot
// perform condense's row move or swap_states' row swap itself — it does not hold
// the values any more. It records what it did; the runner replays the record
// onto the device buffer in stream order. These cases pin the record, because a
// missing or misordered op silently feeds the NEXT step's combine the wrong
// request's token, which shows up as a corrupted output stream and nothing else.
TEST_CASE("W4: add_request records the last_sampled seed with its value") {
  InputBatch batch = make_batch();
  using Op = InputBatch::LastSampledOp;

  CachedRequestState fresh = make_req("fresh", {10, 11, 12}, {}, {3});
  fresh.num_computed_tokens = 0;
  const int i0 = batch.add_request(fresh);

  REQUIRE(batch.last_sampled_ops.size() == 1);
  CHECK(batch.last_sampled_ops[0].kind == Op::kSeed);
  CHECK(batch.last_sampled_ops[0].a == i0);
  CHECK(batch.last_sampled_ops[0].value == 0);

  // A resumed request seeds a real token, and the op must carry that VALUE (it
  // is the one op kind whose data the device cannot derive from indices).
  CachedRequestState resumed = make_req("resumed", {20, 21, 22}, {23, 24}, {7});
  resumed.num_computed_tokens = 4;
  const int i1 = batch.add_request(resumed);

  REQUIRE(batch.last_sampled_ops.size() == 2);
  CHECK(batch.last_sampled_ops[1].kind == Op::kSeed);
  CHECK(batch.last_sampled_ops[1].a == i1);
  CHECK(batch.last_sampled_ops[1].value == 23);
  CHECK(batch.last_sampled_ops[1].value ==
        batch.last_sampled_tokens[static_cast<size_t>(i1)]);
}

TEST_CASE("W4: condense records the row move, swap_states records the swap") {
  InputBatch batch = make_batch();
  using Op = InputBatch::LastSampledOp;

  CachedRequestState r0 = make_req("r0", {100}, {}, {10});
  r0.num_computed_tokens = 1;
  CachedRequestState r1 = make_req("r1", {200, 201}, {}, {20});
  r1.num_computed_tokens = 2;
  CachedRequestState r2 = make_req("r2", {300, 301}, {302}, {30});
  r2.num_computed_tokens = 3;
  batch.add_request(r0);
  batch.add_request(r1);
  batch.add_request(r2);

  // Drain the admission seeds the way the runner does, so what follows is only
  // the structural edits under test.
  batch.last_sampled_ops.clear();

  batch.remove_request("r1");
  batch.condense();  // r2 (slot 2) slides into the freed slot 1

  REQUIRE(batch.last_sampled_ops.size() == 1);
  CHECK(batch.last_sampled_ops[0].kind == Op::kMove);
  CHECK(batch.last_sampled_ops[0].a == 1);  // destination
  CHECK(batch.last_sampled_ops[0].b == 2);  // source
  // Replaying that move on a mirror must reproduce what the host array holds.
  CHECK(batch.last_sampled_tokens[1] == 302);

  batch.last_sampled_ops.clear();
  batch.swap_states(0, 1);

  REQUIRE(batch.last_sampled_ops.size() == 1);
  CHECK(batch.last_sampled_ops[0].kind == Op::kSwap);
  CHECK(batch.last_sampled_ops[0].a == 0);
  CHECK(batch.last_sampled_ops[0].b == 1);
}

TEST_CASE("W4: replaying the op log reproduces the host array exactly") {
  // The whole contract in one place: run a realistic admit / finish / reorder
  // sequence, replay the recorded ops onto an INDEPENDENT array with the same
  // semantics the device kernel implements, and require the two to agree. This
  // is what catches an op that is recorded in the wrong ORDER — each op
  // individually looks right, and only the composition disagrees.
  InputBatch batch = make_batch();
  using Op = InputBatch::LastSampledOp;
  std::vector<int32_t> mirror(batch.last_sampled_tokens.size(), 0);

  auto replay = [&] {
    for (const Op& op : batch.last_sampled_ops) {
      const size_t a = static_cast<size_t>(op.a);
      const size_t b = static_cast<size_t>(op.b);
      if (op.kind == Op::kSeed) {
        mirror[a] = op.value;
      } else if (op.kind == Op::kMove) {
        mirror[a] = mirror[b];
      } else {
        std::swap(mirror[a], mirror[b]);
      }
    }
    batch.last_sampled_ops.clear();
  };

  CachedRequestState a = make_req("a", {1, 2}, {3}, {10});
  a.num_computed_tokens = 3;  // seeds 3
  CachedRequestState b = make_req("b", {5, 6}, {7}, {20});
  b.num_computed_tokens = 3;  // seeds 7
  CachedRequestState c = make_req("c", {8, 9}, {11}, {30});
  c.num_computed_tokens = 3;  // seeds 11
  batch.add_request(a);
  batch.add_request(b);
  batch.add_request(c);
  replay();
  CHECK(mirror == batch.last_sampled_tokens);

  // Reorder, then finish the middle request and condense: the move must be
  // replayed AFTER the swap, or the mirror picks up the pre-swap occupant.
  batch.swap_states(0, 2);
  batch.remove_request("b");
  batch.condense();
  replay();
  CHECK(mirror == batch.last_sampled_tokens);
}

TEST_CASE("condense is a no-op when only the last request was removed") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {0}));
  batch.add_request(make_req("b", {2}, {}, {1}));

  // Remove the LAST slot: no active request lives above the hole, so condense
  // only trims (no move); req_id_to_index for the survivor is unchanged.
  batch.remove_request("b");
  batch.condense();
  CHECK(batch.num_reqs() == 1);
  CHECK(batch.req_id_to_index.at("a") == 0);
  CHECK(static_cast<int>(batch.req_ids.size()) == 1);
}

TEST_CASE("add after remove fills the freed hole before appending") {
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {0}));
  batch.add_request(make_req("b", {2}, {}, {1}));
  batch.add_request(make_req("c", {3}, {}, {2}));

  batch.remove_request("b");  // frees slot 1
  // Next add fills slot 1 (the hole), not slot 3.
  const int idx = batch.add_request(make_req("d", {4}, {}, {9}));
  CHECK(idx == 1);
  CHECK(batch.req_id_to_index.at("d") == 1);
  CHECK(batch.block_table[0].cpu_block_id(1, 0) == 9);
  // No stray hole remains for condense to fill.
  batch.condense();
  CHECK(batch.num_reqs() == 3);
}

TEST_CASE("CachedRequestState.from_new_request seeds output from prefill (MRV2)") {
  // MRV2 contract: prefill_token_ids == all_token_ids (prompt + output). The
  // state's output_token_ids is its tail beyond the prompt.
  NewRequestData nr;
  nr.req_id = "r";
  nr.prompt_token_ids = std::vector<int32_t>{1, 2, 3};
  nr.sampling_params = SamplingParams{};
  nr.block_ids = {{7}};
  nr.num_computed_tokens = 3;
  nr.prefill_token_ids = std::vector<int32_t>{1, 2, 3, 4, 5};  // prompt + 2 out

  CachedRequestState state = CachedRequestState::from_new_request(nr);
  CHECK(state.req_id == "r");
  CHECK(state.num_prompt_tokens == 3);
  CHECK(state.output_token_ids == std::vector<int32_t>{4, 5});
  CHECK(state.num_tokens() == 5);
  CHECK(state.num_computed_tokens == 3);

  // Feeding it through add_request reproduces the prefill_token_ids seed.
  InputBatch batch = make_batch();
  batch.add_request(state);
  for (int i = 0; i < 5; ++i) {
    CHECK(batch.token_id(0, i) == i + 1);
  }
  CHECK(batch.block_table[0].cpu_block_id(0, 0) == 7);
}

TEST_CASE("CachedRequestState.get_token_id and num_tokens") {
  CachedRequestState state;
  state.prompt_token_ids = {10, 11};
  state.output_token_ids = {12};
  state.finalize();
  CHECK(state.num_prompt_tokens == 2);
  CHECK(state.num_tokens() == 3);
  CHECK(state.get_token_id(0) == 10);
  CHECK(state.get_token_id(1) == 11);
  CHECK(state.get_token_id(2) == 12);
  CHECK(state.get_token_id(3) == -1);  // past the end
}

// ─── SamplingMetadata batch-change-gated rebuild (rescan §6 item e) ──────────

TEST_CASE("make_sampling_metadata rebuild is gated on batch change") {
  // Upstream refresh_metadata rebuilds self.sampling_metadata ONLY when the
  // batch changes; the correctness invariant the cache must preserve is that the
  // returned metadata always reflects the current dense [0,num_reqs) prefix.
  // frequency_penalties is always sized num_reqs (built regardless of penalties).
  InputBatch batch = make_batch();
  batch.add_request(make_req("a", {1}, {}, {0}));
  CHECK(batch.make_sampling_metadata().frequency_penalties.size() == 1);

  // A second add MUST invalidate the cache -> size 2 (a non-invalidating cache
  // would return the stale size-1 metadata here — the RED this guards).
  batch.add_request(make_req("b", {2}, {}, {1}));
  CHECK(batch.make_sampling_metadata().frequency_penalties.size() == 2);

  // Repeated calls without a batch change return the SAME cached object with
  // identical content (the cache hit).
  const auto& m1 = batch.make_sampling_metadata();
  const auto& m2 = batch.make_sampling_metadata();
  CHECK(&m1 == &m2);
  CHECK(m1.frequency_penalties.size() == 2);

  // remove + condense MUST invalidate -> the dense prefix shrinks to 1.
  batch.remove_request("a");
  batch.condense();
  CHECK(batch.make_sampling_metadata().frequency_penalties.size() == 1);
}

TEST_CASE("make_sampling_metadata stays fresh when penalties are active") {
  // The one caching deviation: with penalties active the metadata embeds the
  // per-step-mutable output_token_ids (a copy in our port), so make_sampling_
  // metadata rebuilds every call. Verify the penalty branch produces a
  // current-state result.
  InputBatch batch = make_batch();
  SamplingParams pen;
  pen.repetition_penalty = 1.2;  // -> no_penalties() == false
  batch.add_request(make_req("a", {1, 2, 3}, {4}, {0}, pen));

  const auto& md = batch.make_sampling_metadata();
  CHECK_FALSE(md.no_penalties);
  REQUIRE(md.output_token_ids.size() == 1);
  CHECK(md.output_token_ids[0] == std::vector<int32_t>{4});
  // frequency/presence/repetition penalty slices are sized to the batch.
  CHECK(md.repetition_penalties.size() == 1);
}

TEST_CASE("PR11 sampling positions survive progress, reorder and resume") {
  InputBatch batch = make_batch();
  SamplingParams params; params.seed = 123;
  auto a = make_req("a", {1, 2}, {8, 9}, {1}, params);
  auto b = make_req("b", {3}, {}, {2}, params);
  batch.add_request(a); batch.add_request(b);
  CHECK(batch.make_sampling_metadata().output_token_positions == std::vector<uint64_t>{2, 0});
  ++batch.num_tokens_no_spec[0]; // accepted token, with unchanged static sampling metadata
  CHECK(batch.make_sampling_metadata().output_token_positions == std::vector<uint64_t>{3, 0});
  batch.swap_states(0, 1);
  CHECK(batch.make_sampling_metadata().output_token_positions == std::vector<uint64_t>{0, 3});
  batch.remove_request("b"); batch.condense();
  CHECK(batch.make_sampling_metadata().output_token_positions == std::vector<uint64_t>{3});
  batch.remove_request("a"); batch.condense();
  a.output_token_ids.push_back(10); batch.add_request(a);
  CHECK(batch.make_sampling_metadata().output_token_positions == std::vector<uint64_t>{3});
}

// ===========================================================================
// SPEC-MTP (I2) InputBatch spec-decode ABI: num_accepted_tokens (seeded to 1)
// and update_req_spec_token_ids. These are the fields/updaters the rejection
// sampler (I3) and verify/propose runner (I5) build on. Grounded in
// gpu_input_batch.py:240-243,467,484-509,662,787 @ e24d1b24.
// ===========================================================================

TEST_CASE("add_request seeds num_accepted_tokens to 1 (default one token/step)") {
  InputBatch batch = make_batch();
  int s0 = batch.add_request(make_req("a", {1, 2}, {}, {0}));
  int s1 = batch.add_request(make_req("b", {3}, {4}, {1}));
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(s0)] == 1);
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(s1)] == 1);
}

TEST_CASE("update_req_spec_token_ids splices drafts and records spec_token_ids") {
  InputBatch batch = make_batch();
  // prompt {10,11,12} + 1 sampled output {13} -> num_tokens_no_spec == 4.
  int slot = batch.add_request(make_req("r0", {10, 11, 12}, {13}, {0}));
  REQUIRE(batch.num_tokens_no_spec[static_cast<size_t>(slot)] == 4);

  std::map<std::string, std::vector<int32_t>> scheduled = {{"r0", {77, 88}}};
  batch.update_req_spec_token_ids(slot, "r0", scheduled);

  // The drafts land after the non-spec prefix (columns 4,5) and are recorded.
  CHECK(batch.spec_token_ids[static_cast<size_t>(slot)] ==
        std::vector<int32_t>{77, 88});
  CHECK(batch.token_id(slot, 4) == 77);
  CHECK(batch.token_id(slot, 5) == 88);
}

TEST_CASE("update_req_spec_token_ids clears drafts when the request has none") {
  InputBatch batch = make_batch();
  int slot = batch.add_request(make_req("r0", {10, 11}, {12}, {0}));
  // First install some drafts.
  batch.update_req_spec_token_ids(slot, "r0", {{"r0", {5, 6}}});
  REQUIRE(batch.spec_token_ids[static_cast<size_t>(slot)].size() == 2);
  // A step with no scheduled drafts for r0 clears the stale list.
  batch.update_req_spec_token_ids(slot, "r0", {});
  CHECK(batch.spec_token_ids[static_cast<size_t>(slot)].empty());
}

TEST_CASE("num_accepted_tokens moves with the request under swap/condense") {
  InputBatch batch = make_batch();
  int s0 = batch.add_request(make_req("a", {1}, {}, {0}));
  int s1 = batch.add_request(make_req("b", {2}, {}, {1}));
  int s2 = batch.add_request(make_req("c", {3}, {}, {2}));
  // Simulate a rejection-sampler write (I3 will do this after verification).
  batch.num_accepted_tokens[static_cast<size_t>(s0)] = 2;
  batch.num_accepted_tokens[static_cast<size_t>(s2)] = 3;
  (void)s1;

  // swap slot0<->slot2: c (was slot2, val 3) -> slot0; a (was slot0, val 2) -> slot2.
  batch.swap_states(s0, s2);
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(s0)] == 3);  // c
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(s2)] == 2);  // a

  // Remove b (slot1) and condense: the highest active row (a, slot2, val 2) moves
  // down into the freed hole, carrying its accepted count; c stays at slot0.
  batch.remove_request("b");
  batch.condense();
  const int idx_a = batch.req_id_to_index.at("a");
  const int idx_c = batch.req_id_to_index.at("c");
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(idx_a)] == 2);
  CHECK(batch.num_accepted_tokens[static_cast<size_t>(idx_c)] == 3);
}

// ─── ROAD-V1-C7 SAMPLE-CORE / SAMPLE-LOGPROBS / SAMPLE-LOGIT-FILTERS wiring ───
// The controls flow SamplingParams -> add_request per-slot state ->
// build_sampling_metadata -> SamplingMetadata (which the Sampler consumes). These
// gate the WIRING (the transforms themselves are gated in test_sampler.cpp).
namespace {
CachedRequestState make_req_sp(const std::string& req_id, vllm::SamplingParams sp) {
  sp.PostInit();
  CachedRequestState state;
  state.req_id = req_id;
  state.prompt_token_ids = {1, 2, 3};
  state.sampling_params = sp;
  state.block_ids = {{0}};
  state.finalize();
  state.num_computed_tokens = state.num_prompt_tokens;
  return state;
}
}  // namespace

TEST_CASE("C7 wiring: logit_bias / allowed_token_ids / bad_words / min_p / "
          "min_tokens / logprobs reach SamplingMetadata") {
  InputBatch batch = make_batch(/*max_num_reqs=*/8, /*max_model_len=*/64);

  vllm::SamplingParams sp;
  sp.temperature = 0.7;  // random (so top_p/min_p not zeroed by greedy path)
  sp.min_p = 0.3;
  sp.min_tokens = 5;
  sp.stop_token_ids = {42};
  sp.logit_bias = std::map<int32_t, float>{{100, 3.5f}, {7, -2.0f}};
  sp.allowed_token_ids = std::vector<int32_t>{5, 9, 900};
  sp.bad_words = {"foo"};
  // Engine-side tokenization result (UpdateFromTokenizer); provide directly.
  sp.bad_words_token_ids =
      std::vector<std::vector<int32_t>>{{2, 3}, {11}};
  sp.logprobs = 4;
  const int idx = batch.add_request(make_req_sp("r0", sp));

  const auto& md = batch.make_sampling_metadata();

  // min_p slice populated (row idx == 0.3).
  REQUIRE(md.min_p.size() == 1);
  CHECK(md.min_p[static_cast<size_t>(idx)] == doctest::Approx(0.3f));

  // min_tokens: MinTokensState{5, {42}} (all_stop_token_ids seeded in PostInit).
  REQUIRE(md.min_tokens.count(idx) == 1);
  CHECK(md.min_tokens.at(idx).min_tokens == 5);
  CHECK(md.min_tokens.at(idx).stop_token_ids == std::set<int32_t>{42});

  // logit_bias map for the request.
  REQUIRE(md.logit_bias.count(idx) == 1);
  CHECK(md.logit_bias.at(idx).at(100) == doctest::Approx(3.5f));
  CHECK(md.logit_bias.at(idx).at(7) == doctest::Approx(-2.0f));

  // allowed_token_ids_mask: TRUE == exclude; allowed ids are FALSE.
  REQUIRE(md.allowed_token_ids_mask.has_value());
  const auto& row = (*md.allowed_token_ids_mask)[static_cast<size_t>(idx)];
  CHECK(row[5] == 0);
  CHECK(row[9] == 0);
  CHECK(row[900] == 0);
  CHECK(row[6] == 1);   // not allowed -> excluded
  CHECK(row[0] == 1);

  // bad_words_token_ids passed through by req index.
  REQUIRE(md.bad_words_token_ids.count(idx) == 1);
  CHECK(md.bad_words_token_ids.at(idx) ==
        std::vector<std::vector<int32_t>>{{2, 3}, {11}});

  // logprobs count reaches max_num_logprobs.
  CHECK(md.max_num_logprobs.has_value());
  CHECK(*md.max_num_logprobs == 4);

  // output_token_ids now required (min_tokens is an output-consuming proc).
  REQUIRE(md.output_token_ids.size() == 1);
}

TEST_CASE("C7 wiring: default request keeps every filter empty (inertness)") {
  InputBatch batch = make_batch();
  vllm::SamplingParams sp;  // defaults: greedy (temp 1 -> random actually)
  sp.temperature = 0.0;     // greedy
  const int idx = batch.add_request(make_req_sp("r0", sp));
  (void)idx;
  const auto& md = batch.make_sampling_metadata();
  CHECK(md.min_p.empty());
  CHECK(md.min_tokens.empty());
  CHECK(md.logit_bias.empty());
  CHECK(md.bad_words_token_ids.empty());
  CHECK_FALSE(md.allowed_token_ids_mask.has_value());
  CHECK_FALSE(md.max_num_logprobs.has_value());
  CHECK(batch.no_min_p());
  CHECK(batch.no_allowed_token_ids());
}

// `logprobs=-1` means "all logprobs". Upstream widens it to vocab_size at
// admission (gpu_input_batch.py:434-440) so that every value in num_logprobs is
// a concrete count and one gathered shape reaches every consumer; the sampler's
// `num_logprobs == -1` branch is then unreachable on the V1 path.
//
// We used to propagate the sentinel instead (a recorded deviation), which routed
// live requests into that branch and crashed the engine — issue #231. This case
// asserts the mirrored behaviour, so the old assertion (`max_num_logprobs ==
// -1`) is replaced rather than merely relaxed.
TEST_CASE("C7 wiring: -1 logprobs widens to vocab_size, and wins the max") {
  InputBatch batch = make_batch();
  vllm::SamplingParams a;
  a.temperature = 0.0;
  a.logprobs = 3;
  vllm::SamplingParams b;
  b.temperature = 0.0;
  b.logprobs = -1;  // all
  batch.add_request(make_req_sp("a", a));
  batch.add_request(make_req_sp("b", b));

  // Widened at admission: the map holds a count, never the sentinel.
  REQUIRE(batch.num_logprobs.count("b") == 1);
  CHECK(batch.num_logprobs.at("b") == batch.vocab_size);
  CHECK(batch.num_logprobs.at("a") == 3);

  // "All" is simply the largest count, so it wins the max on its own.
  const auto& md = batch.make_sampling_metadata();
  REQUIRE(md.max_num_logprobs.has_value());
  CHECK(*md.max_num_logprobs == batch.vocab_size);
}

// A lone `-1` request must still widen — the max is not what does the widening.
TEST_CASE("C7 wiring: a lone -1 logprobs request still carries vocab_size") {
  InputBatch batch = make_batch();
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.logprobs = -1;
  batch.add_request(make_req_sp("only", sp));
  REQUIRE(batch.num_logprobs.count("only") == 1);
  CHECK(batch.num_logprobs.at("only") == batch.vocab_size);
  const auto& md = batch.make_sampling_metadata();
  REQUIRE(md.max_num_logprobs.has_value());
  CHECK(*md.max_num_logprobs == batch.vocab_size);
}

TEST_CASE("C7 wiring: index-keyed controls follow the row through swap/condense") {
  InputBatch batch = make_batch();
  vllm::SamplingParams sa;
  sa.temperature = 0.0;
  sa.logit_bias = std::map<int32_t, float>{{1, 9.0f}};
  sa.allowed_token_ids = std::vector<int32_t>{1};
  vllm::SamplingParams sb;
  sb.temperature = 0.0;
  vllm::SamplingParams sc;
  sc.temperature = 0.0;
  sc.min_tokens = 2;
  sc.stop_token_ids = {5};
  int s0 = batch.add_request(make_req_sp("a", sa));
  int s1 = batch.add_request(make_req_sp("b", sb));
  int s2 = batch.add_request(make_req_sp("c", sc));
  (void)s1;

  batch.swap_states(s0, s2);  // a's logit_bias -> slot2; c's min_tokens -> slot0
  CHECK(batch.logit_bias.count(s2) == 1);
  CHECK(batch.logit_bias.at(s2).at(1) == doctest::Approx(9.0f));
  CHECK(batch.min_tokens.count(s0) == 1);
  CHECK(batch.min_tokens.at(s0).min_tokens == 2);

  batch.remove_request("b");
  batch.condense();
  const int ia = batch.req_id_to_index.at("a");
  const int ic = batch.req_id_to_index.at("c");
  CHECK(batch.logit_bias.at(ia).at(1) == doctest::Approx(9.0f));
  CHECK(batch.min_tokens.at(ic).min_tokens == 2);
  // The allowed-ids mask row for "a" still keeps only token 1.
  REQUIRE_FALSE(batch.allowed_token_ids_mask.empty());
  CHECK(batch.allowed_token_ids_mask[static_cast<size_t>(ia)][1] == 0);
  CHECK(batch.allowed_token_ids_mask[static_cast<size_t>(ia)][0] == 1);
}

// ─── logprob_token_ids plumbing (gpu_input_batch.py:273,443-444,574,934-951) ──
// The map is keyed by req_ID in the InputBatch and by req_INDEX in the
// SamplingMetadata, exactly as upstream: id-keying is what makes condense() free
// (like the sibling num_logprobs map), and make_sampling_metadata re-derives the
// indices from req_id_to_index over the LIVE batch only.
//
// RED before the port: batch.logprob_token_ids does not exist and
// md.logprob_token_ids is never set.
TEST_CASE("logprob_token_ids: ids reach SamplingMetadata keyed by req index") {
  InputBatch batch = make_batch();
  vllm::SamplingParams a;
  a.temperature = 0.0;
  vllm::SamplingParams b;
  b.temperature = 0.0;
  b.logprob_token_ids = std::vector<int32_t>{7, 42};
  batch.add_request(make_req_sp("a", a));
  batch.add_request(make_req_sp("b", b));

  // Tracked by req_id at admission; "a" never appears.
  REQUIRE(batch.logprob_token_ids.count("b") == 1);
  CHECK(batch.logprob_token_ids.at("b") == std::vector<int32_t>{7, 42});
  CHECK(batch.logprob_token_ids.count("a") == 0);

  const auto& md = batch.make_sampling_metadata();
  REQUIRE(md.logprob_token_ids.has_value());
  const int ib = batch.req_id_to_index.at("b");
  REQUIRE(md.logprob_token_ids->count(ib) == 1);
  CHECK(md.logprob_token_ids->at(ib) == std::vector<int32_t>{7, 42});
  CHECK(md.logprob_token_ids->count(batch.req_id_to_index.at("a")) == 0);

  // logprob_token_ids alone does NOT populate max_num_logprobs: upstream only
  // fills num_logprobs when sampling_params.logprobs is set (:435-440), and the
  // sampler's `or` at sampler.py:86 is what makes the snapshot fire anyway.
  CHECK_FALSE(md.max_num_logprobs.has_value());
}

// remove_request pops the entry (gpu_input_batch.py:574), and the ids follow the
// request through condense because the map is req_id-keyed.
TEST_CASE("logprob_token_ids: removal pops the entry, condense leaves it alone") {
  InputBatch batch = make_batch();
  vllm::SamplingParams a;
  a.temperature = 0.0;
  a.logprob_token_ids = std::vector<int32_t>{1};
  vllm::SamplingParams b;
  b.temperature = 0.0;
  b.logprob_token_ids = std::vector<int32_t>{2, 3};
  batch.add_request(make_req_sp("a", a));
  batch.add_request(make_req_sp("b", b));

  batch.remove_request("a");
  CHECK(batch.logprob_token_ids.count("a") == 0);
  batch.condense();
  REQUIRE(batch.logprob_token_ids.count("b") == 1);
  CHECK(batch.logprob_token_ids.at("b") == std::vector<int32_t>{2, 3});

  const auto& md = batch.make_sampling_metadata();
  REQUIRE(md.logprob_token_ids.has_value());
  CHECK(md.logprob_token_ids->at(batch.req_id_to_index.at("b")) ==
        std::vector<int32_t>{2, 3});
}

// Inertness: a request that does not set the field leaves the optional UNSET,
// mirroring `logprob_token_ids_by_index = None` at gpu_input_batch.py:935-936.
TEST_CASE("logprob_token_ids: a default request leaves the optional unset") {
  InputBatch batch = make_batch();
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  batch.add_request(make_req_sp("r0", sp));
  const auto& md = batch.make_sampling_metadata();
  CHECK(batch.logprob_token_ids.empty());
  CHECK_FALSE(md.logprob_token_ids.has_value());
}
