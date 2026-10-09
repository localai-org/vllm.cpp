// Tests for the V1 Scheduler.schedule() core token-budget algorithm
// (vllm/v1/core/sched/scheduler.py @ e24d1b24).
//
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24 (the create_scheduler /
// create_requests helpers + the schedule() cases), adapted to the T0 subset
// (behavioral only, no model runner — chunked prefill is driven by schedule()'s
// own _update_after_schedule advancing num_computed_tokens; a running decode is
// simulated by appending an output token, which is what update_from_output will
// do once M1.4 Task 4 lands):
//   - test_schedule            : FCFS admission — every waiting request is
//                                scheduled as a new req with min(prompt, budget)
//                                tokens; waiting drains into running.
//   - test_schedule (budget)   : two requests share max_num_batched_tokens
//                                (chunked-prefill split falls out of the budget).
//   - chunked prefill          : a prompt longer than the budget is split across
//                                multiple schedule() calls (num_computed advances
//                                each step) — mirrors the chunked-prefill intent
//                                of test_schedule_partial_requests without the
//                                encoder budget.
//   - running decode           : after prefill completes, the request is
//                                scheduled as a cached (diff) req with 1 token.
//   - test_preempt_during_execution : filling the KV cache makes allocate_slots
//                                return nullopt for a running request, so the
//                                FCFS tail is preempted (status PREEMPTED, KV
//                                freed, re-queued to the waiting front).
//   - max_num_seqs cap         : only max_num_seqs requests run concurrently.
#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/multimodal/inputs.h"  // MultiModalFeatureSpec (P2, #2379)
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::SchedulerPolicy;
using vllm::SpeculativeConfig;
using vllm::v1::DraftTokenIds;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::FinishReason;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::RequestStatus;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// M1.3's KVCacheManager only supports the prefix-caching coordinator
// (enable_caching == false is deferred), so the ported scheduler tests run with
// caching ON. Requests carry the cbor block hasher (block_hashes populate at
// block-size granularity); prompts are distinct per request ([i]*n), so there
// is no cross-request prefix reuse and admission/preemption stay deterministic.

// Mirror of test utils.create_scheduler (T0 subset).
std::unique_ptr<Scheduler> CreateScheduler(int max_num_seqs = 16,
                                           int max_num_batched_tokens = 8192,
                                           bool enable_chunked_prefill = true,
                                           int num_blocks = 10000,
                                           int block_size = 16,
                                           int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = enable_chunked_prefill;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;  // deterministic admission/preemption mechanics.

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// create_scheduler(num_speculative_tokens=k): a scheduler configured with an MTP
// speculator so num_lookahead_tokens == k and the spec-decode paths are engaged.
// Mirrors the upstream test helper's num_speculative_tokens kwarg (the MTP head
// depth n_predict is 1, so a user k of 1 is the natural default; larger k stay a
// multiple of n_predict). This is the only difference from CreateScheduler — the
// SpeculativeConfig turns the plumbing ON.
std::unique_ptr<Scheduler> CreateSpecScheduler(
    int num_speculative_tokens, int max_num_seqs = 16,
    int max_num_batched_tokens = 8192, bool enable_chunked_prefill = true,
    int num_blocks = 10000, int block_size = 16, int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = enable_chunked_prefill;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  // n_predict = 1 (the MTP head depth); the user requests k drafts per step.
  SpeculativeConfig spec =
      SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1,
                                    num_speculative_tokens);
  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true,
                                     /*structured_output_manager=*/nullptr,
                                     std::move(spec));
}

// Mirror of test utils.create_requests (T0 subset): prompts of `num_tokens`
// tokens, req ids "0".."n-1" (or the provided ids). Returns owning requests.
std::vector<std::unique_ptr<Request>> CreateRequests(
    int num_requests, int num_tokens = 10,
    const std::vector<std::string>& req_ids = {}) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < num_requests; ++i) {
    std::string id = req_ids.empty() ? std::to_string(i) : req_ids[i];
    std::vector<int32_t> prompt(num_tokens, static_cast<int32_t>(i));
    requests.push_back(std::make_unique<Request>(
        id, prompt, params, /*arrival_time=*/0.0, block_hasher));
  }
  return requests;
}

// Add a request and return the raw (non-owning) pointer for inspection.
Request* AddRequest(Scheduler& sched, std::unique_ptr<Request> req) {
  Request* raw = req.get();
  sched.add_request(std::move(req));
  return raw;
}

// Build a ModelRunnerOutput from an ordered list of (req_id, sampled_tokens),
// mirroring what the model runner (M1.5) returns: req_ids order + the
// req_id_to_index map + the ragged sampled_token_ids. An empty token list models
// a request still in chunked prefill (no sample this step).
ModelRunnerOutput MakeRunnerOutput(
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& per_req) {
  ModelRunnerOutput mro;
  int idx = 0;
  for (const auto& [id, toks] : per_req) {
    mro.req_ids.push_back(id);
    mro.req_id_to_index[id] = idx++;
    mro.sampled_token_ids.push_back(toks);
  }
  return mro;
}

}  // namespace

// ---------------------------------------------------------------------------
// FCFS admission: every waiting request is scheduled as a new req with its full
// prompt (under budget); waiting drains into running. (test_schedule)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: FCFS admits every request under budget") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10, /*num_tokens=*/10);
  std::vector<Request*> raw;
  for (auto& r : requests) raw.push_back(AddRequest(*scheduler, std::move(r)));

  auto output = scheduler->schedule();

  CHECK(output.scheduled_new_reqs.size() == 10);
  CHECK(output.scheduled_cached_reqs.num_reqs() == 0);
  CHECK(output.finished_req_ids.empty());
  // Every request scheduled its full prompt.
  for (const auto& [req_id, num_tokens] : output.num_scheduled_tokens) {
    CHECK(num_tokens == 10);
  }
  CHECK(output.total_num_scheduled_tokens == 100);

  // Waiting drained into running, in order.
  CHECK(scheduler->waiting->empty());
  REQUIRE(scheduler->running.size() == 10);
  for (int i = 0; i < 10; ++i) CHECK(scheduler->running[i] == raw[i]);
}

// ---------------------------------------------------------------------------
// Budget sharing + chunked-prefill split: two requests share the per-step
// budget; the second is chunked to fit. (budget of test_schedule / chunked
// prefill semantics)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: two requests share max_num_batched_tokens") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/20,
                                 {"a", "b"});
  AddRequest(*scheduler, std::move(requests[0]));
  AddRequest(*scheduler, std::move(requests[1]));

  auto output = scheduler->schedule();

  // "a" fills its 20 tokens; "b" gets the remaining 10 (chunked).
  CHECK(output.num_scheduled_tokens.at("a") == 20);
  CHECK(output.num_scheduled_tokens.at("b") == 10);
  CHECK(output.total_num_scheduled_tokens == 30);
  CHECK(output.scheduled_new_reqs.size() == 2);
}

// ---------------------------------------------------------------------------
// Budget-filling co-schedule (SERVE-GATE c2 parity). Two prompts that EXACTLY
// fill the per-step token budget (2 x 1024 == 2048, the online-gate workload's
// in1024/--max-num-batched-tokens 2048) are BOTH scheduled in a SINGLE step:
// each gets its full 1024 tokens, neither is chunked, and the budget lands at 0.
// This mirrors upstream scheduler.py:640-1013 (the waiting loop keeps admitting
// while token_budget > 0 with no per-step / partial-prefill cap — default
// max_num_partial_prefills == 1, long_prefill_token_threshold == 0), so our
// waiting loop (scheduler.cpp:234-298) co-schedules identically. RED if we ever
// serialized budget-filling prefills; GREEN proves we mirror vLLM's co-schedule.
// Ledger: the component's bimodal c2 TTFT (~0.45 s vs ~0.9 s) is therefore pure
// ARRIVAL phasing (whether both reqs are enqueued at the same schedule() call),
// not a scheduler divergence — see the next case for the staggered arrival.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: two budget-filling prefills co-schedule (c2 parity)") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/32,
                                   /*max_num_batched_tokens=*/2048);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/1024,
                                 {"a", "b"});
  Request* a = AddRequest(*scheduler, std::move(requests[0]));
  Request* b = AddRequest(*scheduler, std::move(requests[1]));

  auto output = scheduler->schedule();

  // Both prefills run in ONE step, each at its full 1024 tokens (no chunk).
  CHECK(output.num_scheduled_tokens.at("a") == 1024);
  CHECK(output.num_scheduled_tokens.at("b") == 1024);
  CHECK(output.total_num_scheduled_tokens == 2048);  // budget filled exactly.
  CHECK(output.scheduled_new_reqs.size() == 2);      // neither serialized.
  CHECK(output.scheduled_cached_reqs.num_reqs() == 0);
  CHECK(scheduler->waiting->empty());
  REQUIRE(scheduler->running.size() == 2);
  CHECK(scheduler->running[0] == a);
  CHECK(scheduler->running[1] == b);
  // Both completed prefill this step -> neither is a lingering prefill chunk.
  CHECK_FALSE(a->is_prefill_chunk);
  CHECK_FALSE(b->is_prefill_chunk);
}

// ---------------------------------------------------------------------------
// Staggered arrival (c2 TTFT lottery mechanism). If the second 1024-token
// request is not yet enqueued when the first is scheduled, the first prefills
// ALONE this step (the ~0.45 s TTFT mode). On the next step the first request's
// single decode token co-schedules with the second request's full 1024-token
// prefill inside the same 2048 budget (1 + 1024 <= 2048) — the late prefill is
// NOT refused/serialized, it simply landed one step later (the ~0.9 s mode).
// This is the arrival-phasing "lottery" behind the component's bimodal c2 TTFT;
// it is identical to upstream, so the leg-to-leg 3/3-vs-6/0 mixture is timing,
// not a scheduler policy difference (no scheduler change is warranted).
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: a late prefill co-schedules with a running decode") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/32,
                                   /*max_num_batched_tokens=*/2048);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/1024,
                                 {"a", "b"});
  Request* a = AddRequest(*scheduler, std::move(requests[0]));

  // Step 1: only "a" is enqueued -> it prefills ALONE (the fast TTFT mode).
  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.at("a") == 1024);
  CHECK(out0.num_scheduled_tokens.size() == 1);
  CHECK(out0.total_num_scheduled_tokens == 1024);  // half the budget unused.
  REQUIRE(a->num_computed_tokens == 1024);

  // "a" samples its first token (runner would return one on the next step).
  a->AppendOutputToken(0);
  // "b" arrives after "a" was already dispatched.
  Request* b = AddRequest(*scheduler, std::move(requests[1]));

  // Step 2: "a" decodes 1 token AND "b" prefills its full 1024 in the SAME step.
  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.at("a") == 1);      // running decode.
  CHECK(out1.num_scheduled_tokens.at("b") == 1024);   // late prefill co-scheduled.
  CHECK(out1.total_num_scheduled_tokens == 1025);
  CHECK(out1.scheduled_new_reqs.size() == 1);         // only "b" is new.
  CHECK(out1.scheduled_new_reqs[0].req_id == "b");
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);  // "a" as a cached diff.
  REQUIRE(scheduler->running.size() == 2);
  CHECK(scheduler->waiting->empty());
  (void)b;
}

// ---------------------------------------------------------------------------
// Chunked prefill split across steps: a long prompt (80) with budget 30 is
// scheduled 30 / 30 / 20 over three steps, num_computed advancing each step.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: chunked prefill splits a long prompt") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/80, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.at("0") == 30);
  CHECK(out0.scheduled_new_reqs.size() == 1);  // first chunk => new req
  CHECK(req->num_computed_tokens == 30);

  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.at("0") == 30);
  CHECK(out1.scheduled_new_reqs.empty());  // subsequent chunks => cached diff
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(req->num_computed_tokens == 60);

  auto out2 = scheduler->schedule();
  CHECK(out2.num_scheduled_tokens.at("0") == 20);
  CHECK(req->num_computed_tokens == 80);
  CHECK_FALSE(req->is_prefill_chunk);  // prefill complete.
}

// ---------------------------------------------------------------------------
// Running decode: after prefill completes, an appended output token (simulating
// the model runner sample) makes the request schedule as a cached (diff) req
// with exactly 1 new token. (SchedulerOutput new-vs-cached shape.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: running request decodes as a cached req") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out0 = scheduler->schedule();  // prefill
  CHECK(out0.scheduled_new_reqs.size() == 1);
  CHECK(out0.num_scheduled_tokens.at("0") == 4);
  REQUIRE(req->num_computed_tokens == 4);

  // Simulate the runner sampling one token.
  req->AppendOutputToken(0);

  auto out1 = scheduler->schedule();  // decode
  CHECK(out1.scheduled_new_reqs.empty());
  CHECK(out1.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(out1.scheduled_cached_reqs.req_ids[0] == "0");
  CHECK(out1.num_scheduled_tokens.at("0") == 1);
  CHECK(scheduler->running.size() == 1);
}

// ---------------------------------------------------------------------------
// Preemption on KV exhaustion: with only 10 usable blocks, two 80-token
// requests (5 blocks each) fill the cache; when the first needs a 6th block for
// its decode, allocate_slots returns nullopt and the FCFS tail (the second
// request) is preempted. (test_preempt_during_execution)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: KV exhaustion preempts the FCFS tail") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/100,
                                   /*enable_chunked_prefill=*/true,
                                   /*num_blocks=*/11, /*block_size=*/16);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/80,
                                 {"0", "1"});
  Request* req0 = requests[0].get();
  Request* req1 = requests[1].get();

  // Schedule the first request (5 blocks for 80 tokens).
  AddRequest(*scheduler, std::move(requests[0]));
  auto out0 = scheduler->schedule();
  CHECK(out0.num_scheduled_tokens.size() == 1);
  REQUIRE(out0.scheduled_new_reqs.size() == 1);
  CHECK(out0.scheduled_new_reqs[0].block_ids[0].size() == 5);

  // Schedule the second request while the first still runs (5 more blocks).
  AddRequest(*scheduler, std::move(requests[1]));
  auto out1 = scheduler->schedule();
  CHECK(out1.num_scheduled_tokens.size() == 1);
  REQUIRE(out1.scheduled_new_reqs.size() == 1);
  CHECK(out1.scheduled_new_reqs[0].block_ids[0].size() == 5);

  // The first request samples a token -> it needs a 6th block next step, but
  // the KV cache is full, so the second request is preempted.
  req0->AppendOutputToken(0);
  auto out2 = scheduler->schedule();

  CHECK(scheduler->running.size() == 1);
  CHECK(scheduler->running[0] == req0);
  CHECK(req1->status == RequestStatus::kPreempted);
  // The preempted request is re-queued to the waiting front, KV freed.
  CHECK(scheduler->waiting->size() == 1);
  CHECK(scheduler->waiting->peek_request() == req1);
  CHECK(req1->num_computed_tokens == 0);
  CHECK(out2.num_scheduled_tokens.count("0") == 1);
  CHECK(out2.num_scheduled_tokens.count("1") == 0);
}

// ---------------------------------------------------------------------------
// SERVE-RESPONSE-METRICS: the scheduler records the per-request engine-core
// events the frontend derives queue/prefill/inference timing + the preemption
// counter from — QUEUED at add_request (scheduler.py:2135), SCHEDULED when a
// request is admitted to the running batch (:1003), PREEMPTED when the FCFS tail
// is evicted under KV pressure (:1221), all gated on log_stats. This reuses the
// KV-exhaustion preemption mechanics above and asserts the events fire at the
// right sites, in order. RED-first: before the event wiring Request.events is
// always empty and every assertion below fails.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: records QUEUED/SCHEDULED/PREEMPTED engine-core events") {
  using vllm::v1::EngineCoreEventType;
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/100,
                                   /*enable_chunked_prefill=*/true,
                                   /*num_blocks=*/11, /*block_size=*/16);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/80,
                                 {"0", "1"});
  Request* req0 = requests[0].get();
  Request* req1 = requests[1].get();

  // QUEUED recorded immediately at add_request.
  AddRequest(*scheduler, std::move(requests[0]));
  REQUIRE(req0->events.size() == 1);
  CHECK(req0->events[0].type == EngineCoreEventType::kQueued);

  // Admitting req0 to the running batch records a SCHEDULED after the QUEUED,
  // ordered queued <= scheduled.
  auto out0 = scheduler->schedule();
  REQUIRE(req0->events.size() == 2);
  CHECK(req0->events[1].type == EngineCoreEventType::kScheduled);
  CHECK(req0->events[0].timestamp <= req0->events[1].timestamp);

  AddRequest(*scheduler, std::move(requests[1]));
  auto out1 = scheduler->schedule();
  REQUIRE(req1->status == RequestStatus::kRunning);
  // req1: QUEUED (add) then SCHEDULED (this admission). req0 stays at 2 (an
  // already-running decode request is NOT re-SCHEDULED).
  REQUIRE(req1->events.size() == 2);
  CHECK(req1->events[0].type == EngineCoreEventType::kQueued);
  CHECK(req1->events[1].type == EngineCoreEventType::kScheduled);
  CHECK(req0->events.size() == 2);

  // req0 samples a token -> it needs a 6th block next step, KV is full, so the
  // FCFS tail (req1) is preempted, recording a PREEMPTED event on top.
  req0->AppendOutputToken(0);
  auto out2 = scheduler->schedule();
  REQUIRE(req1->status == RequestStatus::kPreempted);
  REQUIRE(req1->events.size() == 3);
  CHECK(req1->events[2].type == EngineCoreEventType::kPreempted);
  // The preemption shares this step's scheduled timestamp -> ordered after the
  // request's own SCHEDULED.
  CHECK(req1->events[1].timestamp <= req1->events[2].timestamp);
  // num_preemptions bumped 1:1 with the event.
  CHECK(req1->num_preemptions == 1);
}

// ---------------------------------------------------------------------------
// Resumed-as-new (MRV2 output fold): a preempted request that is re-scheduled
// after KV frees appears in scheduled_new_reqs (NOT the cached diff), carrying
// prefill_token_ids == its full token ids. This exercises the V2 output tail
// (upstream scheduler.py `if self.use_v2_model_runner:` branch): resumed reqs
// are folded into scheduled_new_reqs and re-sent as FULL new reqs so the V2 gpu
// runner (M1.5) can re-seed its per-request all_token_ids from prefill_token_ids.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: resumed (preempted) request re-enters as new") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/100,
                                   /*enable_chunked_prefill=*/true,
                                   /*num_blocks=*/11, /*block_size=*/16);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/80,
                                 {"0", "1"});
  Request* req0 = requests[0].get();
  Request* req1 = requests[1].get();

  // Fill the KV cache with two 80-token reqs (5 blocks each).
  AddRequest(*scheduler, std::move(requests[0]));
  scheduler->schedule();
  AddRequest(*scheduler, std::move(requests[1]));
  scheduler->schedule();

  // req0 samples a token -> needs a 6th block; KV is full, so the FCFS tail
  // (req1) is preempted (KV freed, re-queued to the waiting front).
  req0->AppendOutputToken(0);
  scheduler->schedule();
  REQUIRE(req1->status == RequestStatus::kPreempted);
  REQUIRE(scheduler->waiting->peek_request() == req1);

  // Free req0's KV so the preempted req1 has room to be re-scheduled.
  scheduler->finish_requests("0", RequestStatus::kFinishedStopped);

  auto out = scheduler->schedule();

  // The V2 fold: req1 resumes as a NEW req (folded from scheduled_resumed_reqs),
  // NOT through the cached diff.
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "1");
  CHECK(out.scheduled_cached_reqs.num_reqs() == 0);
  // resumed_req_ids in the cached diff stays empty (resumed folded into new).
  CHECK(out.scheduled_cached_reqs.resumed_req_ids.empty());
  CHECK(req1->status == RequestStatus::kRunning);

  // prefill_token_ids carries req1's full token ids so the V2 runner re-seeds
  // its state; req1 never decoded, so this is its 80-token prompt.
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(*seed == req1->AllTokenIds());
  CHECK(seed->size() == 80);
}

// ---------------------------------------------------------------------------
// A first-time (WAITING->RUNNING) new req also carries prefill_token_ids == its
// full token ids under the MRV2 path (upstream V2 branch seeds every new req).
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: new req carries prefill_token_ids seed") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/10, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out = scheduler->schedule();

  REQUIRE(out.scheduled_new_reqs.size() == 1);
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(*seed == req->AllTokenIds());
  CHECK(seed->size() == 10);
}

// ---------------------------------------------------------------------------
// max_num_seqs cap: only max_num_seqs requests run concurrently. (test_schedule
// admission bound.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: max_num_seqs caps concurrent requests") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/2);
  auto requests = CreateRequests(/*num_requests=*/3, /*num_tokens=*/4);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto output = scheduler->schedule();

  CHECK(output.scheduled_new_reqs.size() == 2);
  CHECK(scheduler->running.size() == 2);
  CHECK(scheduler->waiting->size() == 1);
}

// ---------------------------------------------------------------------------
// add_request / finish_requests / counts (T0 subset). (test_add_requests,
// test_finish_request, test_get_num_unfinished_requests.)
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler: add_request registers into waiting + requests map") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10);
  for (int i = 0; i < 10; ++i) {
    std::string id = requests[i]->request_id;
    AddRequest(*scheduler, std::move(requests[i]));
    CHECK(scheduler->requests.count(id) == 1);
    CHECK(static_cast<int>(scheduler->waiting->size()) == i + 1);
  }
  CHECK(scheduler->get_num_unfinished_requests() == 10);
}

TEST_CASE("Scheduler: finish_requests removes from waiting + requests map") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/10);
  std::vector<std::string> ids;
  for (auto& r : requests) {
    ids.push_back(r->request_id);
    AddRequest(*scheduler, std::move(r));
  }

  for (int i = 0; i < 10; ++i) {
    scheduler->finish_requests(ids[i], RequestStatus::kFinishedAborted);
    CHECK(scheduler->requests.count(ids[i]) == 0);
    CHECK(static_cast<int>(scheduler->waiting->size()) == 9 - i);
  }
}

// ===========================================================================
// update_from_output — the schedule -> execute -> update loop (M1.4 Task 4).
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24: the
// test_update_from_output family, test_stop_via_update_from_output (eos /
// stop_token_ids / max_tokens / min_tokens), and the partial-prefill
// "no sampled tokens" case (test_schedule_partial_requests).
// ===========================================================================

// A sampled token is appended and the request advances (no stop, still running).
// (test_update_from_output.)
TEST_CASE("Scheduler.update_from_output: appends the sampled token") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto sched_out = scheduler->schedule();  // prefill 4 tokens
  REQUIRE(req->num_computed_tokens == 4);

  auto mro = MakeRunnerOutput({{"0", {42}}});
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  // The token was appended; the request keeps running (max_tokens 16, no eos).
  REQUIRE(req->output_token_ids.size() == 1);
  CHECK(req->output_token_ids[0] == 42);
  CHECK(req->status == RequestStatus::kRunning);
  CHECK_FALSE(req->IsFinished());

  // EngineCoreOutputs shape: one output carrying the new token, no finish.
  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  CHECK(eco.outputs[0].new_token_ids == std::vector<int32_t>{42});
  CHECK_FALSE(eco.outputs[0].finish_reason.has_value());
  CHECK_FALSE(eco.outputs[0].Finished());
  CHECK(scheduler->running.size() == 1);
  CHECK(scheduler->finished_req_ids.empty());
}

// An EOS token finishes the request: FINISHED_STOPPED, KV freed, id in
// finished_req_ids, erased from the requests map, finish_reason == stop.
// (test_stop_via_update_from_output — eos.)
TEST_CASE("Scheduler.update_from_output: eos token stops the request") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.eos_token_id = 7;

  auto sched_out = scheduler->schedule();
  auto mro = MakeRunnerOutput({{"0", {7}}});
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  CHECK(eco.outputs[0].new_token_ids == std::vector<int32_t>{7});
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kStop);
  CHECK(eco.outputs[0].Finished());
  // Freed + removed everywhere.
  CHECK(scheduler->finished_req_ids.count("0") == 1);
  CHECK(scheduler->requests.count("0") == 0);  // erased from the owning map
  CHECK(scheduler->running.empty());
}

// max_tokens reached -> FINISHED_LENGTH_CAPPED / finish_reason == length.
// (test_stop_via_update_from_output — max_tokens.)
TEST_CASE("Scheduler.update_from_output: max_tokens length-caps the request") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.max_tokens = 1;  // one output token is already the cap

  auto sched_out = scheduler->schedule();
  auto mro = MakeRunnerOutput({{"0", {5}}});  // not eos, not a stop token
  EngineCoreOutputs eco = scheduler->update_from_output(sched_out, mro);

  REQUIRE(eco.outputs.size() == 1);
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kLength);
  CHECK(scheduler->finished_req_ids.count("0") == 1);
  CHECK(scheduler->requests.count("0") == 0);
}

// min_tokens gates the EOS stop: the same eos token does NOT stop until at least
// min_tokens output tokens exist. (test_stop_via_update_from_output — min_tokens.)
TEST_CASE("Scheduler.update_from_output: min_tokens prevents an early eos stop") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.eos_token_id = 7;
  req->sampling_params.min_tokens = 2;
  req->sampling_params.max_tokens = 16;

  // First eos: only 1 output token < min_tokens(2) -> not stopped.
  auto out0 = scheduler->schedule();
  auto eco0 = scheduler->update_from_output(out0, MakeRunnerOutput({{"0", {7}}}));
  CHECK_FALSE(req->IsFinished());
  REQUIRE(eco0.outputs.size() == 1);
  CHECK_FALSE(eco0.outputs[0].finish_reason.has_value());

  // Second eos: 2 output tokens >= min_tokens(2) -> stopped.
  auto out1 = scheduler->schedule();  // decode step (num_new == 1)
  auto eco1 = scheduler->update_from_output(out1, MakeRunnerOutput({{"0", {7}}}));
  REQUIRE(eco1.outputs.size() == 1);
  REQUIRE(eco1.outputs[0].finish_reason.has_value());
  CHECK(*eco1.outputs[0].finish_reason == FinishReason::kStop);
  CHECK(scheduler->requests.count("0") == 0);
}

// A stop_token_ids match stops the request and carries the token id as
// stop_reason. (test_stop_via_update_from_output — stop_token_ids.)
TEST_CASE("Scheduler.update_from_output: stop_token_ids match sets stop_reason") {
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/4, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));
  req->sampling_params.stop_token_ids = {99};
  req->sampling_params.max_tokens = 16;

  auto sched_out = scheduler->schedule();
  auto eco = scheduler->update_from_output(sched_out, MakeRunnerOutput({{"0", {99}}}));

  REQUIRE(eco.outputs.size() == 1);
  REQUIRE(eco.outputs[0].finish_reason.has_value());
  CHECK(*eco.outputs[0].finish_reason == FinishReason::kStop);
  REQUIRE(eco.outputs[0].stop_reason.has_value());
  CHECK(*eco.outputs[0].stop_reason == "99");
  CHECK(scheduler->requests.count("0") == 0);
}

// A request still in chunked prefill receives an empty sampled-token list and
// gets no output, staying running with its output unchanged.
// (test_schedule_partial_requests intent.)
TEST_CASE("Scheduler.update_from_output: prefilling request gets no token") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/30);
  auto requests = CreateRequests(/*num_requests=*/1, /*num_tokens=*/80, {"0"});
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto sched_out = scheduler->schedule();  // first chunk (30 of 80)
  REQUIRE(req->num_computed_tokens == 30);
  REQUIRE(req->is_prefill_chunk);

  // The runner returns an empty token list for a still-prefilling request.
  auto eco = scheduler->update_from_output(sched_out, MakeRunnerOutput({{"0", {}}}));

  CHECK(eco.outputs.empty());  // no partial-prefill output
  CHECK(req->output_token_ids.empty());
  CHECK(req->num_computed_tokens == 30);  // update_from_output does not advance it
  CHECK(req->status == RequestStatus::kRunning);
  CHECK(scheduler->requests.count("0") == 1);
  CHECK(scheduler->finished_req_ids.empty());
}

// EngineCoreOutputs shape with a mixed batch: two decoding requests each yield
// one output; a third still-prefilling request yields none.
TEST_CASE("Scheduler.update_from_output: batched outputs shape") {
  auto scheduler = CreateScheduler(/*max_num_seqs=*/16,
                                   /*max_num_batched_tokens=*/40);
  // "a"/"b" are short (fully prefill this step); "c" is long (chunked).
  auto reqs = CreateRequests(/*num_requests=*/2, /*num_tokens=*/4, {"a", "b"});
  AddRequest(*scheduler, std::move(reqs[0]));
  AddRequest(*scheduler, std::move(reqs[1]));
  auto longreq = CreateRequests(/*num_requests=*/1, /*num_tokens=*/60, {"c"});
  AddRequest(*scheduler, std::move(longreq[0]));

  auto sched_out = scheduler->schedule();
  REQUIRE(sched_out.num_scheduled_tokens.at("a") == 4);
  REQUIRE(sched_out.num_scheduled_tokens.at("b") == 4);
  REQUIRE(sched_out.num_scheduled_tokens.at("c") == 32);  // chunked remainder

  auto eco = scheduler->update_from_output(
      sched_out, MakeRunnerOutput({{"a", {1}}, {"b", {2}}, {"c", {}}}));

  // "a" and "b" produced a token; "c" (still prefilling) produced none.
  CHECK(eco.outputs.size() == 2);
  CHECK(eco.engine_index == 0);
}

// ===========================================================================
// Priority scheduling (W4 / ENG-PRIORITY-SCHED).
// Ported from tests/v1/core/test_scheduler.py @ e24d1b24: the 11
// test_priority_scheduling_* cases (:2382-2856) +
// test_priority_scheduling_preemption_and_resumption_when_out_of_kv (:2978),
// with the create_scheduler_with_priority / create_requests_with_priority
// helpers (:2155,:2278).
//
// DEVIATION (recorded): our M1.3 KVCacheManager only supports the
// prefix-caching coordinator (enable_caching == false is deferred), so these
// run with caching ON — exactly like the FCFS scheduler tests above. Upstream's
// priority tests run with caching OFF (create_scheduler_with_priority default
// enable_prefix_caching=False). To keep caching ON behaviorally equivalent to
// caching OFF for the block-math (preemption/resumption) cases, every request is
// given a DISTINCT prompt (via a distinct `starting_idx`) so there is no
// cross-request prefix reuse — the tests exercise priority ORDER / victim
// selection, not the prefix cache. The EC/KV-connector variant (:3769) is not
// ported (no connectors); only the V2-model-runner, no-connector path is
// covered (that is our only output path).
// ===========================================================================

namespace {

// Mirror of test utils.create_scheduler_with_priority (T0 subset): identical to
// CreateScheduler but policy = priority.
std::unique_ptr<Scheduler> CreateSchedulerWithPriority(
    int max_num_seqs = 16, int max_num_batched_tokens = 8192,
    int num_blocks = 10000, int block_size = 16, int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  cfg.policy = SchedulerPolicy::kPriority;  // enable priority scheduling.

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// Mirror of test utils.create_requests_with_priority (T0 subset): each request
// carries a priority + arrival time; prompt = [i + starting_idx] * num_tokens
// (distinct per i+starting_idx). Default req ids "{i+starting_idx}".
std::vector<std::unique_ptr<Request>> CreateRequestsWithPriority(
    int num_requests, const std::vector<int>& priorities,
    const std::vector<double>& arrival_times = {}, int num_tokens = 10,
    int max_tokens = 16, const std::vector<std::string>& req_ids = {},
    int starting_idx = 0) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = max_tokens;

  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < num_requests; ++i) {
    std::string id =
        req_ids.empty() ? std::to_string(i + starting_idx) : req_ids[i];
    const double arrival =
        arrival_times.empty() ? static_cast<double>(i) : arrival_times[i];
    std::vector<int32_t> prompt(num_tokens,
                                static_cast<int32_t>(i + starting_idx));
    requests.push_back(std::make_unique<Request>(
        id, prompt, params, arrival, block_hasher, priorities[i]));
  }
  return requests;
}

// Collect the scheduled_new_reqs req ids in schedule order.
std::vector<std::string> ScheduledNewIds(const vllm::v1::SchedulerOutput& out) {
  std::vector<std::string> ids;
  for (const auto& nr : out.scheduled_new_reqs) ids.push_back(nr.req_id);
  return ids;
}

}  // namespace

// test_priority_scheduling_basic_ordering: lower priority value first.
TEST_CASE("Priority scheduling: basic priority ordering") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/3, /*priorities=*/{2, 0, 1},
      /*arrival_times=*/{1.0, 2.0, 3.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 3);
  // req_1 (prio 0), req_2 (prio 1), req_0 (prio 2).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "2", "0"});
}

// test_priority_scheduling_arrival_time_tiebreaker: equal priority -> arrival.
TEST_CASE("Priority scheduling: arrival time tiebreaker") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/3, /*priorities=*/{1, 1, 1},
      /*arrival_times=*/{3.0, 1.0, 2.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 3);
  // req_1 (1.0), req_2 (2.0), req_0 (3.0).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "2", "0"});
}

// test_priority_scheduling_mixed_priority_and_arrival.
TEST_CASE("Priority scheduling: mixed priority and arrival") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(
      /*num_requests=*/4, /*priorities=*/{2, 1, 1, 0},
      /*arrival_times=*/{1.0, 3.0, 2.0, 4.0});
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 4);
  // req_3 (prio 0), req_2 (prio 1, arr 2), req_1 (prio 1, arr 3), req_0 (prio 2).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"3", "2", "1", "0"});
}

// test_priority_scheduling_preemption: under KV pressure the lowest-PRIORITY
// running request is preempted, not the FCFS tail.
TEST_CASE("Priority scheduling: preempts the lowest-priority running request") {
  const int block_size = 16;
  const int num_blocks = 6;             // 1 null -> 5 usable
  const int num_tokens = block_size * 2;  // 32 tokens = exactly 2 blocks
  auto scheduler = CreateSchedulerWithPriority(
      /*max_num_seqs=*/3, /*max_num_batched_tokens=*/200, num_blocks, block_size);

  // Phase 1: low-priority request starts running (distinct prompt via idx 0).
  auto lo = CreateRequestsWithPriority(1, {5}, {1.0}, num_tokens, /*max_tokens=*/16,
                                       {"lo1"}, /*starting_idx=*/0);
  Request* lo1 = lo[0].get();
  AddRequest(*scheduler, std::move(lo[0]));
  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  scheduler->update_from_output(out, MakeRunnerOutput({{"lo1", {100}}}));

  // Phase 2: high-priority request arrives (distinct prompt via idx 1).
  auto hi = CreateRequestsWithPriority(1, {0}, {2.0}, num_tokens, /*max_tokens=*/16,
                                       {"hi1"}, /*starting_idx=*/1);
  AddRequest(*scheduler, std::move(hi[0]));
  out = scheduler->schedule();
  bool hi_admitted = false;
  for (const auto& nr : out.scheduled_new_reqs)
    if (nr.req_id == "hi1") hi_admitted = true;
  CHECK(hi_admitted);
  CHECK(scheduler->running.size() == 2);
  scheduler->update_from_output(out,
                                MakeRunnerOutput({{"lo1", {101}}, {"hi1", {100}}}));

  // Phase 3: hi1 needs a 3rd block, 0 free -> preempt the lowest-priority
  // running request = lo1 (priority 5 > 0).
  out = scheduler->schedule();
  CHECK(lo1->status == RequestStatus::kPreempted);
  bool hi_running = false;
  for (Request* r : scheduler->running)
    if (r->request_id == "hi1") hi_running = true;
  CHECK(hi_running);
}

// test_priority_scheduling_no_preemption_when_space_available.
TEST_CASE("Priority scheduling: no preemption when space is available") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/3,
                                               /*max_num_batched_tokens=*/200);
  auto lows = CreateRequestsWithPriority(2, {5, 5}, {1.0, 2.0}, /*num_tokens=*/30,
                                         /*max_tokens=*/16, {"lo1", "lo2"},
                                         /*starting_idx=*/0);
  for (auto& r : lows) AddRequest(*scheduler, std::move(r));
  auto out = scheduler->schedule();
  scheduler->update_from_output(
      out, MakeRunnerOutput({{"lo1", {100}}, {"lo2", {100}}}));

  auto hi = CreateRequestsWithPriority(1, {0}, {3.0}, /*num_tokens=*/30,
                                       /*max_tokens=*/16, {"hi1"},
                                       /*starting_idx=*/2);
  AddRequest(*scheduler, std::move(hi[0]));
  out = scheduler->schedule();

  CHECK(out.scheduled_new_reqs.size() == 1);  // hi1 admitted, no preemption
  CHECK(scheduler->running.size() == 3);
  CHECK(scheduler->waiting->empty());
}

// test_priority_scheduling_preemption_victim_selection: waiting queue order.
TEST_CASE("Priority scheduling: victim selection / waiting queue order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(3, {3, 2, 0}, {1.0, 2.0, 3.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "2");  // highest priority

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->priority == 2);
  CHECK(waiting[1]->priority == 3);
  CHECK(waiting[0]->request_id == "1");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_equal_priority_preemption: arrival tiebreak in wait.
TEST_CASE("Priority scheduling: equal priority -> arrival tiebreak in waiting") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(3, {2, 2, 2}, {3.0, 1.0, 2.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "1");  // earliest arrival (1.0)

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->arrival_time == 2.0);
  CHECK(waiting[1]->arrival_time == 3.0);
  CHECK(waiting[0]->request_id == "2");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_waiting_queue_order.
TEST_CASE("Priority scheduling: waiting queue maintains priority order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  auto requests = CreateRequestsWithPriority(4, {3, 1, 2, 0},
                                             {1.0, 2.0, 3.0, 4.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == "3");  // priority 0

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 3);
  std::vector<std::string> ids;
  std::vector<int> prios;
  for (Request* r : waiting) {
    ids.push_back(r->request_id);
    prios.push_back(r->priority);
  }
  CHECK(ids == std::vector<std::string>{"1", "2", "0"});
  CHECK(prios == std::vector<int>{1, 2, 3});
}

// test_priority_scheduling_fcfs_fallback: equal priority -> FCFS by arrival.
TEST_CASE("Priority scheduling: FCFS fallback when priorities equal") {
  auto scheduler = CreateSchedulerWithPriority();
  auto requests = CreateRequestsWithPriority(4, {1, 1, 1, 1},
                                             {4.0, 1.0, 3.0, 2.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 4);
  // Arrival order: req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0).
  CHECK(ScheduledNewIds(out) == std::vector<std::string>{"1", "3", "2", "0"});
}

// test_priority_scheduling_with_limited_slots.
TEST_CASE("Priority scheduling: max_num_seqs limits to highest priorities") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/2,
                                               /*max_num_batched_tokens=*/1000);
  auto requests = CreateRequestsWithPriority(4, {3, 1, 2, 0},
                                             {1.0, 2.0, 3.0, 4.0},
                                             /*num_tokens=*/10);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  auto out = scheduler->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 2);
  auto ids = ScheduledNewIds(out);
  // The 2 highest priorities: req_3 (prio 0), req_1 (prio 1).
  CHECK(std::find(ids.begin(), ids.end(), "3") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "1") != ids.end());

  auto waiting = scheduler->waiting->ToList();
  REQUIRE(waiting.size() == 2);
  CHECK(waiting[0]->priority == 2);
  CHECK(waiting[1]->priority == 3);
  CHECK(waiting[0]->request_id == "2");
  CHECK(waiting[1]->request_id == "0");
}

// test_priority_scheduling_heap_property: scheduling one at a time (finishing
// each) drains requests in ascending-priority order.
TEST_CASE("Priority scheduling: schedules in ascending priority order") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/1);
  const std::vector<int> priorities = {5, 1, 8, 3, 2, 7, 4, 6};
  std::vector<double> arrivals;
  for (std::size_t i = 0; i < priorities.size(); ++i)
    arrivals.push_back(static_cast<double>(i));
  auto requests = CreateRequestsWithPriority(
      static_cast<int>(priorities.size()), priorities, arrivals,
      /*num_tokens=*/10);
  // Map req id -> priority for the recording step.
  std::map<std::string, int> id_to_priority;
  for (auto& r : requests) id_to_priority[r->request_id] = r->priority;
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  std::vector<int> scheduled_priorities;
  while (!scheduler->waiting->empty()) {
    auto out = scheduler->schedule();
    if (!out.scheduled_new_reqs.empty()) {
      const std::string& id = out.scheduled_new_reqs[0].req_id;
      scheduled_priorities.push_back(id_to_priority[id]);
      scheduler->update_from_output(out, MakeRunnerOutput({{id, {100}}}));
      scheduler->finish_requests(id, RequestStatus::kFinishedStopped);
    }
  }

  std::vector<int> expected = priorities;
  std::sort(expected.begin(), expected.end());
  CHECK(scheduled_priorities == expected);
}

// test_priority_scheduling_preemption_and_resumption_when_out_of_kv (V2 model
// runner, no connector variant).
TEST_CASE("Priority scheduling: preemption then resumption when out of KV") {
  auto scheduler = CreateSchedulerWithPriority(/*max_num_seqs=*/2,
                                               /*max_num_batched_tokens=*/200,
                                               /*num_blocks=*/5,
                                               /*block_size=*/16);

  // request_low: priority 1, 30 tokens (distinct prompt via idx 0).
  auto lo = CreateRequestsWithPriority(1, {1}, {0.0}, /*num_tokens=*/30,
                                       /*max_tokens=*/16, {}, /*starting_idx=*/0);
  const std::string low_id = lo[0]->request_id;
  Request* low = lo[0].get();
  AddRequest(*scheduler, std::move(lo[0]));

  // 1st schedule.
  auto out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 1);
  // 1st decode.
  scheduler->update_from_output(out, MakeRunnerOutput({{low_id, {100}}}));

  // request_high: priority 0, 32 tokens (distinct prompt via idx 1).
  auto hi = CreateRequestsWithPriority(1, {0}, {1.0}, /*num_tokens=*/32,
                                       /*max_tokens=*/16, {}, /*starting_idx=*/1);
  const std::string high_id = hi[0]->request_id;
  AddRequest(*scheduler, std::move(hi[0]));

  // 2nd schedule: KV cache becomes full (0 free blocks).
  out = scheduler->schedule();
  CHECK(scheduler->kv_cache_manager->block_pool.get_num_free_blocks() == 0);
  CHECK(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 2);
  // 2nd decode.
  scheduler->update_from_output(
      out, MakeRunnerOutput({{low_id, {100}}, {high_id, {100}}}));

  // 3rd schedule: triggers priority preemption of request_low.
  out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.empty());
  CHECK(out.scheduled_cached_reqs.num_reqs() == 1);
  CHECK(out.scheduled_cached_reqs.req_ids[0] == high_id);
  CHECK(scheduler->requests.at(low_id)->status == RequestStatus::kPreempted);
  CHECK(scheduler->waiting->size() == 1);
  CHECK(scheduler->running.size() == 1);
  // 3rd decode: low produced nothing (preempted), high produced a token.
  scheduler->update_from_output(
      out, MakeRunnerOutput({{low_id, {}}, {high_id, {100}}}));
  // Finish high so the preempted low can resume.
  scheduler->finish_requests(high_id, RequestStatus::kFinishedStopped);

  // 4th schedule: request_low resumes. V2: folded into scheduled_new_reqs
  // carrying its full token ids in prefill_token_ids (30 prompt + 2 decoded).
  out = scheduler->schedule();
  CHECK(scheduler->waiting->empty());
  CHECK(scheduler->running.size() == 1);
  CHECK(out.scheduled_cached_reqs.num_reqs() == 0);
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.scheduled_new_reqs[0].req_id == low_id);
  const auto& seed = out.scheduled_new_reqs[0].prefill_token_ids;
  REQUIRE(seed.has_value());
  CHECK(seed->size() == 32);
  CHECK((*seed)[31] == 100);
  (void)low;
}

// ===========================================================================
// SPEC-MTP (I2) scheduler spec-decode plumbing. Ported from
// tests/v1/core/test_scheduler.py @ e24d1b24:
//   - test_schedule_spec_decoding_stats (:1042-1150) — the scheduling + rollback
//     arithmetic (the SpecDecodingStats telemetry assertions are DROPPED: no
//     SpecDecodingStats type at I2, per test-porting rule 3 realize-against-ours);
//   - test_spec_decoding_stats_empty_output (:1152-1210) — empty-output regression
//     (num_accepted floored at 0, no num_computed underflow);
//   - test_no_spec_tokens_scheduled_for_prefill_chunks (:1212-1303).
// The first-decode-step PADDING cases (test_spec_decode_padding_*) are SKIPPED —
// that feature (num_spec_tokens_to_schedule / placeholder -1 drafts) is DEFERRED
// past I2 (see scheduler.h / mtp-spec-decode.md §OMITTED). test-porting rule 6.
// ===========================================================================

namespace {
// Feed a full ModelRunnerOutput back to the scheduler from an ordered list of
// per-request sampled tokens over the scheduled requests (in num_scheduled_tokens
// key order, matching how the runner indexes the batch).
void FeedModelOutput(Scheduler& sched, const SchedulerOutput& out,
                     const std::vector<std::vector<int32_t>>& sampled) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> per_req;
  std::size_t i = 0;
  for (const auto& [req_id, n] : out.num_scheduled_tokens) {
    (void)n;
    per_req.emplace_back(req_id, i < sampled.size() ? sampled[i]
                                                    : std::vector<int32_t>{});
    ++i;
  }
  sched.update_from_output(out, MakeRunnerOutput(per_req));
}
}  // namespace

TEST_CASE(
    "Scheduler.schedule: speculative decode schedules 1+k tokens and rolls back "
    "num_computed on rejection") {
  // Ported cases from test_schedule_spec_decoding_stats parametrization.
  struct Case {
    std::vector<std::vector<int32_t>> spec_tokens;   // per request
    std::vector<std::vector<int32_t>> output_tokens;  // per request
  };
  const std::vector<Case> cases = {
      {{{1, 2, 3}}, {{1, 2, 3, 4}}},          // perfect match
      {{{1, 2, 3}}, {{1, 5}}},                // early mismatch
      {{{1, 2}, {3}}, {{1, 2, 5}, {3, 4}}},   // multiple sequences
      {{{1}}, {{1, 2}}},                      // single token sequence
      {{{}}, {{5}}},                          // empty sequence
      {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 7}, {4, 8}}},  // multiple mismatches
  };

  for (const auto& c : cases) {
    int k = 1;
    for (const auto& s : c.spec_tokens) {
      k = std::max(k, static_cast<int>(s.size()));
    }
    auto scheduler = CreateSpecScheduler(/*num_speculative_tokens=*/k);
    const int num_reqs = static_cast<int>(c.spec_tokens.size());
    auto requests = CreateRequests(num_reqs, /*num_tokens=*/1);
    std::vector<std::string> req_ids;
    std::vector<Request*> raw;
    for (auto& r : requests) {
      req_ids.push_back(r->request_id);
      raw.push_back(AddRequest(*scheduler, std::move(r)));
    }

    // Step 1: a plain decode (prefill of the 1-token prompt). No drafts yet.
    auto out = scheduler->schedule();
    CHECK(static_cast<int>(out.scheduled_new_reqs.size()) == num_reqs);
    CHECK(out.total_num_scheduled_tokens == num_reqs);
    for (const auto& id : req_ids) {
      CHECK(out.num_scheduled_tokens.at(id) == 1);
      CHECK(out.scheduled_spec_decode_tokens.count(id) == 0);
    }

    // Sample one token per request, then install the drafts for the next step.
    std::vector<std::vector<int32_t>> sampled(num_reqs, std::vector<int32_t>{0});
    FeedModelOutput(*scheduler, out, sampled);
    DraftTokenIds drafts;
    drafts.req_ids = req_ids;
    drafts.draft_token_ids = c.spec_tokens;
    scheduler->update_draft_token_ids(drafts);

    for (int i = 0; i < num_reqs; ++i) {
      Request* r = raw[i];
      CHECK(r->num_computed_tokens == 1);        // the prompt token
      CHECK(r->NumTokens() == 2);                // prompt + sampled
      CHECK(r->NumTokensWithSpec() ==
            2 + static_cast<int>(c.spec_tokens[i].size()));
    }

    // Step 2: schedule the drafts for verification (1 + k_i tokens per request).
    out = scheduler->schedule();
    CHECK(out.scheduled_new_reqs.empty());
    int expected_total = num_reqs;
    for (const auto& s : c.spec_tokens) {
      expected_total += static_cast<int>(s.size());
    }
    CHECK(out.total_num_scheduled_tokens == expected_total);
    std::vector<int> computed_before(num_reqs);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& id = req_ids[i];
      const int k_i = static_cast<int>(c.spec_tokens[i].size());
      CHECK(out.num_scheduled_tokens.at(id) == 1 + k_i);
      if (k_i > 0) {
        REQUIRE(out.scheduled_spec_decode_tokens.count(id) == 1);
        CHECK(static_cast<int>(out.scheduled_spec_decode_tokens.at(id).size()) ==
              k_i);
        CHECK(out.scheduled_spec_decode_tokens.at(id) == c.spec_tokens[i]);
      } else {
        CHECK(out.scheduled_spec_decode_tokens.count(id) == 0);
      }
      // The drafts were consumed off the request.
      CHECK(raw[i]->spec_token_ids.empty());
      computed_before[i] = raw[i]->num_computed_tokens;
      CHECK(computed_before[i] == 2 + k_i);  // 1 + (1 + k) scheduled this step
    }

    // Step 2 output: verify the rollback rewinds num_computed by num_rejected.
    FeedModelOutput(*scheduler, out, c.output_tokens);
    for (int i = 0; i < num_reqs; ++i) {
      const int k_i = static_cast<int>(c.spec_tokens[i].size());
      const int num_accepted = std::max(
          static_cast<int>(c.output_tokens[i].size()) - 1, 0);
      const int num_rejected = k_i - num_accepted;
      if (k_i > 0) {
        // num_computed was > 0, so the rollback applies.
        CHECK(raw[i]->num_computed_tokens == computed_before[i] - num_rejected);
      } else {
        // No drafts scheduled -> no rollback; num_computed unchanged this step.
        CHECK(raw[i]->num_computed_tokens == computed_before[i]);
      }
      CHECK(raw[i]->num_computed_tokens >= 0);
    }
  }
}

// ---------------------------------------------------------------------------
// The STATS half of upstream's test_schedule_spec_decoding_stats
// (tests/v1/core/test_scheduler.py:1307-1439): the same six parameterised cases
// and the same `expected` tuple, read off SchedulerStats rather than off the
// request. The case above already asserts the scheduling and the rollback; this
// one asserts what /metrics is fed (#2770).
//
// It enters through the two production functions the exposition depends on,
// `update_from_output` and `make_stats`, and asserts BOTH polarities: empty
// after a step that verified no draft, populated after one that did. RED before
// this row's change: SchedulerStats had no spec_decoding_stats field at all.
// ---------------------------------------------------------------------------
TEST_CASE(
    "Scheduler: spec-decoding stats count drafts, draft tokens, accepted "
    "tokens and per-position acceptance") {
  struct Case {
    std::vector<std::vector<int32_t>> spec_tokens;
    std::vector<std::vector<int32_t>> output_tokens;
    int64_t num_drafts;
    int64_t num_draft_tokens;
    int64_t num_accepted_tokens;
    std::vector<int64_t> num_accepted_tokens_per_pos;
  };
  // Upstream's parametrize list verbatim, expected == (num_drafts,
  // num_draft_tokens, num_accepted_tokens, num_accepted_tokens_per_pos).
  const std::vector<Case> cases = {
      {{{1, 2, 3}}, {{1, 2, 3, 4}}, 1, 3, 3, {1, 1, 1}},   // perfect match
      {{{1, 2, 3}}, {{1, 5}}, 1, 3, 1, {1, 0, 0}},         // early mismatch
      {{{1, 2}, {3}}, {{1, 2, 5}, {3, 4}}, 2, 3, 3, {2, 1}},  // multiple seqs
      {{{1}}, {{1, 2}}, 1, 1, 1, {1}},                     // single token
      {{{}}, {{5}}, 0, 0, 0, {0}},                         // empty sequence
      {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 7}, {4, 8}}, 2, 6, 3, {2, 1, 0}},
  };

  for (const auto& c : cases) {
    int k = 1;
    for (const auto& s : c.spec_tokens) {
      k = std::max(k, static_cast<int>(s.size()));
    }
    auto scheduler = CreateSpecScheduler(/*num_speculative_tokens=*/k);
    const int num_reqs = static_cast<int>(c.spec_tokens.size());
    auto requests = CreateRequests(num_reqs, /*num_tokens=*/1);
    std::vector<std::string> req_ids;
    for (auto& r : requests) {
      req_ids.push_back(r->request_id);
      AddRequest(*scheduler, std::move(r));
    }

    // Step 1 — a decode that verifies nothing. Upstream: "No draft or accepted
    // tokens counted yet".
    auto out = scheduler->schedule();
    FeedModelOutput(*scheduler, out,
                    std::vector<std::vector<int32_t>>(
                        num_reqs, std::vector<int32_t>{0}));
    CHECK_FALSE(scheduler->make_stats().spec_decoding_stats.has_value());

    DraftTokenIds drafts;
    drafts.req_ids = req_ids;
    drafts.draft_token_ids = c.spec_tokens;
    scheduler->update_draft_token_ids(drafts);

    // Step 2 — verify the drafts.
    out = scheduler->schedule();
    FeedModelOutput(*scheduler, out, c.output_tokens);
    const auto stats = scheduler->make_stats().spec_decoding_stats;

    if (c.num_drafts == 0) {
      // The empty-sequence case. make_spec_decoding_stats returns empty for a
      // request that verified no draft, so a step in which NO request drafted
      // reports nothing at all rather than a zero-valued aggregate
      // (scheduler.py:2722-2723).
      CHECK_FALSE(stats.has_value());
      continue;
    }
    REQUIRE(stats.has_value());
    CHECK(stats->num_drafts == c.num_drafts);
    CHECK(stats->num_draft_tokens == c.num_draft_tokens);
    CHECK(stats->num_accepted_tokens == c.num_accepted_tokens);
    CHECK(stats->num_accepted_tokens_per_pos ==
          c.num_accepted_tokens_per_pos);
    // The per-position vector is sized by the CONFIG, not by what was drafted
    // (metrics.py:37-38), which is what makes the Prometheus label cardinality
    // fixed at construction.
    CHECK(stats->num_spec_tokens == k);
    CHECK(static_cast<int>(stats->num_draft_tokens_per_pos.size()) == k);

    // A third step that verifies nothing CLEARS the stash: a stale aggregate
    // republished every step would multiply the counters without bound.
    out = scheduler->schedule();
    FeedModelOutput(*scheduler, out,
                   std::vector<std::vector<int32_t>>(
                       num_reqs, std::vector<int32_t>{9}));
    CHECK_FALSE(scheduler->make_stats().spec_decoding_stats.has_value());
  }
}

TEST_CASE(
    "Scheduler.update_from_output: empty spec output does not underflow "
    "num_computed (regression)") {
  // Ported from test_spec_decoding_stats_empty_output.
  auto scheduler = CreateSpecScheduler(/*num_speculative_tokens=*/3);
  auto requests = CreateRequests(1, /*num_tokens=*/1);
  const std::string req_id = requests[0]->request_id;
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out = scheduler->schedule();  // prefill
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  FeedModelOutput(*scheduler, out, {{0}});

  DraftTokenIds drafts;
  drafts.req_ids = {req_id};
  drafts.draft_token_ids = {{1, 2, 3}};
  scheduler->update_draft_token_ids(drafts);

  out = scheduler->schedule();  // schedule the 3 drafts for validation
  REQUIRE(out.scheduled_spec_decode_tokens.count(req_id) == 1);
  CHECK(out.scheduled_spec_decode_tokens.at(req_id).size() == 3);
  const int computed_before = req->num_computed_tokens;

  // Empty output (aborted/errored). Upstream's rollback guard is
  // `scheduled_spec AND (generated OR num_sampled_per_step == 0)` — with an empty
  // `generated` and num_sampled_per_step == 1 the block is SKIPPED entirely, so
  // num_computed is left unchanged (no rollback). The regression this guards is
  // that num_accepted = max(len([]) - 1, 0) = 0 (not -1) must never underflow /
  // throw: we exercise the path and confirm no negative num_computed results.
  FeedModelOutput(*scheduler, out, {{}});
  CHECK(req->num_computed_tokens == computed_before);
  CHECK(req->num_computed_tokens >= 0);
}

TEST_CASE(
    "Scheduler: draft tokens are ignored for prefill chunks") {
  // Ported from test_no_spec_tokens_scheduled_for_prefill_chunks. Budget 50,
  // 80-token prompt -> chunked prefill; drafts provided mid-prefill must be
  // dropped (no spec tokens scheduled with the remaining prefill).
  auto scheduler = CreateSpecScheduler(/*num_speculative_tokens=*/3,
                                       /*max_num_seqs=*/16,
                                       /*max_num_batched_tokens=*/50);
  auto requests = CreateRequests(1, /*num_tokens=*/80);
  const std::string req_id = requests[0]->request_id;
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  // First chunk: 50 of 80 tokens.
  auto out = scheduler->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  CHECK(out.num_scheduled_tokens.at(req_id) == 50);
  FeedModelOutput(*scheduler, out, {{}});  // still prefilling, no sampled token
  CHECK(req->is_prefill_chunk);

  // Drafts arrive mid-prefill -> ignored (cleared).
  DraftTokenIds drafts;
  drafts.req_ids = {req_id};
  drafts.draft_token_ids = {{1, 2, 3}};
  scheduler->update_draft_token_ids(drafts);
  CHECK(req->spec_token_ids.empty());

  // Second chunk: the remaining 30 prefill tokens ONLY (not 30 + 3 spec).
  out = scheduler->schedule();
  CHECK(out.num_scheduled_tokens.at(req_id) == 30);
  CHECK(out.scheduled_spec_decode_tokens.count(req_id) == 0);
  FeedModelOutput(*scheduler, out, {{42}});  // prefill complete, sampled token

  // Now drafts are accepted (prefill complete).
  scheduler->update_draft_token_ids(drafts);
  REQUIRE(req->spec_token_ids.size() == 3);
  CHECK(req->spec_token_ids == std::vector<int32_t>{1, 2, 3});

  // Third schedule: decode + 3 spec tokens = 4.
  out = scheduler->schedule();
  CHECK(out.num_scheduled_tokens.at(req_id) == 4);
  REQUIRE(out.scheduled_spec_decode_tokens.count(req_id) == 1);
  CHECK(out.scheduled_spec_decode_tokens.at(req_id).size() == 3);
}

TEST_CASE(
    "Scheduler.update_draft_token_ids_in_output (#1824): trims to the "
    "scheduled count, pads a short row with -1, and skips unknown requests") {
  // Ported from scheduler.py:2072-2107 @ 555967922 (SPEC-DFLASH2 W7). The
  // async draft-in-output path rewrites the drafts INSIDE a SchedulerOutput
  // (the scheduler-side request state holds only placeholders under async
  // scheduling); the grammar validate_tokens arm stays deferred exactly as in
  // the sync update_draft_token_ids above (no per-request validate seam yet).
  auto scheduler = CreateSpecScheduler(/*num_speculative_tokens=*/3);
  auto requests = CreateRequests(1, /*num_tokens=*/4);
  const std::string req_id = requests[0]->request_id;
  (void)AddRequest(*scheduler, std::move(requests[0]));

  auto out = scheduler->schedule();  // prefill
  FeedModelOutput(*scheduler, out, {{0}});
  DraftTokenIds drafts;
  drafts.req_ids = {req_id};
  drafts.draft_token_ids = {{1, 2, 3}};
  scheduler->update_draft_token_ids(drafts);

  out = scheduler->schedule();  // schedules the 3 drafts
  REQUIRE(out.scheduled_spec_decode_tokens.count(req_id) == 1);
  REQUIRE(out.scheduled_spec_decode_tokens.at(req_id).size() == 3);

  // (a) The worker returned MORE drafts than were scheduled (chunked-prefill
  // trim, scheduler.py:2091-2094): trimmed to the scheduled count.
  DraftTokenIds more;
  more.req_ids = {req_id};
  more.draft_token_ids = {{7, 8, 9, 10, 11}};
  scheduler->update_draft_token_ids_in_output(more, out);
  CHECK(out.scheduled_spec_decode_tokens.at(req_id) ==
        std::vector<int32_t>{7, 8, 9});
  CHECK(out.num_invalid_spec_tokens.empty());

  // (b) FEWER drafts than scheduled: padded to the scheduled count with -1
  // and the invalid tail recorded (scheduler.py:2099-2103).
  DraftTokenIds fewer;
  fewer.req_ids = {req_id};
  fewer.draft_token_ids = {{7}};
  scheduler->update_draft_token_ids_in_output(fewer, out);
  CHECK(out.scheduled_spec_decode_tokens.at(req_id) ==
        std::vector<int32_t>{7, -1, -1});
  REQUIRE(out.num_invalid_spec_tokens.count(req_id) == 1);
  CHECK(out.num_invalid_spec_tokens.at(req_id) == 2);

  // (c) An unknown request id is skipped (scheduler.py:2082-2085), and a known
  // request with NO scheduled spec entry is skipped too (:2087-2089).
  DraftTokenIds ghost;
  ghost.req_ids = {"ghost", req_id};
  ghost.draft_token_ids = {{1, 2, 3}, {4, 5, 6}};
  CHECK_NOTHROW(scheduler->update_draft_token_ids_in_output(ghost, out));
  CHECK(out.scheduled_spec_decode_tokens.count("ghost") == 0);
  CHECK(out.scheduled_spec_decode_tokens.at(req_id) ==
        std::vector<int32_t>{4, 5, 6});
  // Each call REPLACES num_invalid_spec_tokens (upstream builds a fresh dict
  // and assigns it, scheduler.py:2075,2107) — (b)'s entry does not survive (c).
  CHECK(out.num_invalid_spec_tokens.empty());
}

TEST_CASE(
    "Scheduler: no speculator -> spec-decode paths inert (default-off)") {
  // Default-off proof: a scheduler built WITHOUT a SpeculativeConfig never sees a
  // draft (the runner never produces one), so spec_token_ids stays empty on every
  // request, num_tokens_with_spec == num_tokens, and scheduled_spec_decode_tokens
  // is empty on every step — byte-identical to the pre-SPEC-MTP scheduler. Drive
  // a normal prefill+decode and confirm the spec map is empty and the decode
  // schedules exactly one token.
  auto scheduler = CreateScheduler();
  auto requests = CreateRequests(1, /*num_tokens=*/4);
  const std::string req_id = requests[0]->request_id;
  Request* req = AddRequest(*scheduler, std::move(requests[0]));

  auto out = scheduler->schedule();  // prefill 4 tokens
  CHECK(out.scheduled_spec_decode_tokens.empty());
  FeedModelOutput(*scheduler, out, {{0}});
  CHECK(req->spec_token_ids.empty());
  CHECK(req->NumTokensWithSpec() == req->NumTokens());

  out = scheduler->schedule();  // decode: exactly 1 token, no spec
  CHECK(out.num_scheduled_tokens.at(req_id) == 1);
  CHECK(out.scheduled_spec_decode_tokens.empty());
}

// ---------------------------------------------------------------------------
// The per-step token budget is what SPLITS a concurrent prefill wave (#669).
//
// This pins the c4 x 1024-token shape that the TTFT budget sweep runs, at two
// values of --max-num-batched-tokens, because the sweep produced a
// budget-INVARIANT wall time on our arm and the natural reading of that is "the
// budget never reaches the scheduler". It does. What the sweep measured is that
// 1 x 4096 and 2 x 2048 are the same total prefill work, not that the split
// failed to happen -- and this test is the thing that keeps that distinction
// falsifiable, since no timing gate can tell the two apart.
//
// Both arms mirror upstream's waiting loop exactly:
//   budget init                scheduler.cpp:465   <- scheduler.py:447
//   `token_budget > 0` guard   scheduler.cpp:642   <- scheduler.py:671
//   num_new = tokens - computed scheduler.cpp:688  <- scheduler.py:825
//   min(num_new, token_budget) scheduler.cpp:697   <- scheduler.py:859-860
//   token_budget -= num_new    scheduler.cpp:742   <- scheduler.py:1018-1019
// (upstream anchors @ 5559679229bc961848b121ccdeaa8fa5d79bec98.)
//
// Note which line ENDS the 2048 step: the loop guard, not the chunking clamp.
// Two 1024-token prompts take the budget to exactly 0, so the third request is
// never even peeked and there is NO partial third chunk -- upstream stops at the
// same place for the same reason (scheduler.py:671).
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: the token budget splits a 4x1024 prefill wave") {
  const int kPrompt = 1024;

  SUBCASE("budget 8192 admits the whole wave in ONE 4096-token step") {
    auto scheduler = CreateScheduler(/*max_num_seqs=*/256,
                                     /*max_num_batched_tokens=*/8192,
                                     /*enable_chunked_prefill=*/true,
                                     /*num_blocks=*/100000, /*block_size=*/16,
                                     /*max_model_len=*/8192);
    auto requests = CreateRequests(/*num_requests=*/4, kPrompt,
                                   {"a", "b", "c", "d"});
    for (auto& r : requests) AddRequest(*scheduler, std::move(r));

    auto out = scheduler->schedule();

    CHECK(out.total_num_scheduled_tokens == 4 * kPrompt);  // 4096 in one step.
    CHECK(out.num_scheduled_tokens.size() == 4);
    for (const char* id : {"a", "b", "c", "d"}) {
      CHECK(out.num_scheduled_tokens.at(id) == kPrompt);  // nobody is chunked.
    }
    CHECK(out.scheduled_new_reqs.size() == 4);
    CHECK(scheduler->waiting->empty());
  }

  SUBCASE("budget 2048 splits the SAME wave into two 2048-token steps") {
    auto scheduler = CreateScheduler(/*max_num_seqs=*/256,
                                     /*max_num_batched_tokens=*/2048,
                                     /*enable_chunked_prefill=*/true,
                                     /*num_blocks=*/100000, /*block_size=*/16,
                                     /*max_model_len=*/8192);
    auto requests = CreateRequests(/*num_requests=*/4, kPrompt,
                                   {"a", "b", "c", "d"});
    for (auto& r : requests) AddRequest(*scheduler, std::move(r));

    // Step 0: "a" and "b" fill the budget exactly; "c"/"d" stay WAITING.
    auto out0 = scheduler->schedule();
    CHECK(out0.total_num_scheduled_tokens == 2048);
    CHECK(out0.num_scheduled_tokens.size() == 2);
    CHECK(out0.num_scheduled_tokens.at("a") == kPrompt);
    CHECK(out0.num_scheduled_tokens.at("b") == kPrompt);
    // The loop guard stopped it, so there is no partial third chunk.
    CHECK(out0.num_scheduled_tokens.count("c") == 0);
    CHECK(out0.num_scheduled_tokens.count("d") == 0);
    CHECK(scheduler->waiting->size() == 2);

    // "a" and "b" completed prefill, so each samples its first token and moves
    // into decode.
    scheduler->update_from_output(
        out0, MakeRunnerOutput({{"a", {7}}, {"b", {7}}}));

    // Step 1: the RUNNING loop is served first (scheduler.cpp:478 <-
    // scheduler.py:473), so a/b's two decode tokens come off the budget before
    // the waiting loop runs. "c" then takes a full chunk and "d" gets only what
    // is left -- 2048 - 2 - 1024 = 1022. THIS is the clamp at scheduler.cpp:697
    // (<- scheduler.py:859-860) firing: a genuine mid-prompt split, which the
    // 8192 arm never performs. Pinning the exact 1022 is the point: an
    // off-by-one in the budget arithmetic changes it, and no timing gate can.
    auto out1 = scheduler->schedule();
    CHECK(out1.num_scheduled_tokens.at("a") == 1);  // decode
    CHECK(out1.num_scheduled_tokens.at("b") == 1);  // decode
    CHECK(out1.num_scheduled_tokens.at("c") == kPrompt);
    CHECK(out1.num_scheduled_tokens.at("d") == 1022);  // chunked remainder
    CHECK(out1.total_num_scheduled_tokens == 2048);    // budget filled exactly
    CHECK(scheduler->waiting->empty());
    // "d" is mid-prompt, so it samples nothing this step.
    Request* d = nullptr;
    for (Request* r : scheduler->running) {
      if (r->request_id == "d") d = r;
    }
    REQUIRE(d != nullptr);
    CHECK(d->is_prefill_chunk);

    scheduler->update_from_output(
        out1, MakeRunnerOutput({{"a", {7}}, {"b", {7}}, {"c", {7}}, {"d", {}}}));

    // Step 2: "d" finishes its prompt with the 2-token tail it was left.
    auto out2 = scheduler->schedule();
    CHECK(out2.num_scheduled_tokens.at("d") == 2);
    CHECK(out2.total_num_scheduled_tokens == 5);  // a,b,c decode + d's tail
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ENG-MM-INPUT-PIPELINE P2 (#2379) — the ENCODER half of the scheduler.
//
// Ported from tests/v1/core/test_scheduler.py::test_schedule_partial_requests
// and test_encoder_cache_* @ 5559679229, reduced to the paths this tree has:
// no encoder-decoder manager, no EC connector, no EAGLE shift.
//
// Before this row every one of these observables was hardcoded: the scheduler
// wrote `num_encoder_tokens = 0`, left `scheduled_encoder_inputs` empty and
// never called `free_encoder_mm_hashes`, and `EncoderCacheManager` — fully
// ported — was constructed nowhere in the tree.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// A scheduler with the encoder knobs the mm cases need. Separate from
// CreateScheduler so every existing case keeps the exact config it had.
std::unique_ptr<Scheduler> CreateMmScheduler(int max_num_batched_tokens,
                                             int encoder_compute_budget,
                                             int encoder_cache_size,
                                             bool disable_chunked_mm_input = false,
                                             int max_num_seqs = 16,
                                             int num_blocks = 10000,
                                             int block_size = 16) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.max_num_encoder_input_tokens = encoder_compute_budget;
  cfg.encoder_cache_size = encoder_cache_size;
  cfg.disable_chunked_mm_input = disable_chunked_mm_input;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// One placeholder span. `mm_hash` is what the encoder cache is keyed by, so two
// items that share it are ONE encoder run — which is the property the
// same-image-twice case below turns on.
vllm::multimodal::MultiModalFeatureSpec Item(const std::string& mm_hash,
                                             int offset, int length) {
  vllm::multimodal::MultiModalFeatureSpec spec;
  spec.mm_hash = mm_hash;
  spec.modality = "image";
  spec.offset = offset;
  spec.length = length;
  return spec;
}

// A request whose prompt carries the given placeholder spans. The prompt ids
// themselves are irrelevant to the scheduler — it budgets on the SPANS.
std::unique_ptr<Request> MmRequest(
    const std::string& id, int num_tokens,
    std::vector<vllm::multimodal::MultiModalFeatureSpec> items) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  auto block_hasher = get_request_block_hasher(16, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  std::vector<int32_t> prompt(static_cast<size_t>(num_tokens), 7);
  auto req = std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                       block_hasher);
  req->mm_features = std::move(items);
  return req;
}

}  // namespace

TEST_CASE("Scheduler.schedule: a text step schedules NO encoder input and frees none") {
  // The neutrality floor. Every encoder observable stays at the value it had
  // before this row, and it stays there because the request carries no items —
  // not because the code is absent.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/64,
                                     /*encoder_compute_budget=*/64,
                                     /*encoder_cache_size=*/64);
  auto requests = CreateRequests(/*num_requests=*/2, /*num_tokens=*/8);
  for (auto& r : requests) AddRequest(*scheduler, std::move(r));

  const SchedulerOutput out = scheduler->schedule();
  CHECK(out.scheduled_new_reqs.size() == 2);
  CHECK(out.scheduled_encoder_inputs.empty());
  CHECK(out.free_encoder_mm_hashes.empty());
  CHECK(out.total_num_scheduled_tokens == 16);
  // The payload hop is a no-op for a text request rather than absent.
  for (const auto& n : out.scheduled_new_reqs) CHECK(n.mm_features.empty());
}

TEST_CASE("Native Qwen3.5 encoder admission: scope preserves other engines") {
  vllm::HfConfig config;
  config.hidden_size = 5120;
  config.raw["quantization_config"] = {{"quant_method", "exl3"}};
  vllm::MultiModalConfig mm;
  const auto apply = [&](std::string_view arch, vt::DeviceType device) {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 4;
    cfg.max_num_batched_tokens = 1600;
    vllm::entrypoints::ConfigureNativeQwen3_5EncoderBudget(cfg, config, arch, device, mm);
    return cfg;
  };
  const auto unchanged = [&](const SchedulerConfig& cfg) {
    CHECK(cfg.max_num_encoder_input_tokens == 2048);
    CHECK(cfg.encoder_cache_size == 2048);
    CHECK(cfg.max_num_batched_tokens == 1600);
  };
  unchanged(apply("Qwen3_5ForCausalLM", vt::DeviceType::kXPU));
  unchanged(apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kCPU));
  mm.language_model_only = true;
  unchanged(apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU));
  mm.language_model_only = false;
  mm.limit_per_prompt["image"] = 0;
  unchanged(apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU));
  mm.limit_per_prompt["image"] = 2;
  config.raw["quantization_config"]["quant_method"] = "gptq";
  unchanged(apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU));
  config.raw["quantization_config"]["quant_method"] = "exl3";
  const auto enabled = apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU);
  CHECK(enabled.max_num_encoder_input_tokens == 8192);
  CHECK(enabled.encoder_cache_size == 32768);
  CHECK(enabled.max_num_batched_tokens == 1600);
  mm.limit_per_prompt["image"] = 1;
  const auto single = apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU);
  CHECK(single.max_num_encoder_input_tokens == 4096);
  CHECK(single.encoder_cache_size == 16384);
  config.raw["vision_config"]["patch_size"] = 0;
  CHECK_THROWS_AS(apply("Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU),
                  std::runtime_error);
}

TEST_CASE("Native Qwen3.5 encoder admission: C4 limit queues excess image and text clients") {
  vllm::HfConfig config;
  config.raw["quantization_config"]={{"quant_method","exl3"}};
  vllm::MultiModalConfig mm;
  const auto resolve=[&](int count, std::string_view arch, vt::DeviceType device) {
    return vllm::entrypoints::ResolveNativeQwen3_5MaxNumSeqs(count,config,arch,device,mm);
  };
  const std::string arch="Qwen3_5ForConditionalGeneration";
  for (int count : {1,2,4,5,32}) {
    CAPTURE(count);
    CHECK(resolve(count,arch,vt::DeviceType::kXPU)==std::min(count,4));
  }
  CHECK(resolve(32,"Qwen3_5ForCausalLM",vt::DeviceType::kXPU)==32);
  CHECK(resolve(32,arch,vt::DeviceType::kCPU)==32);
  mm.language_model_only=true;
  CHECK(resolve(32,arch,vt::DeviceType::kXPU)==32);
  mm.language_model_only=false;
  mm.limit_per_prompt["image"]=0;
  CHECK(resolve(32,arch,vt::DeviceType::kXPU)==32);
  mm.limit_per_prompt["image"]=2;
  config.raw["quantization_config"]["quant_method"]="gptq";
  CHECK(resolve(32,arch,vt::DeviceType::kXPU)==32);
  config.raw["quantization_config"]["quant_method"]="exl3";
  CHECK_THROWS_AS(resolve(0,arch,vt::DeviceType::kXPU),std::runtime_error);
  SchedulerConfig cfg;
  cfg.max_num_seqs=resolve(32,arch,vt::DeviceType::kXPU);
  cfg.max_num_batched_tokens=1600; cfg.max_model_len=64;
  vllm::entrypoints::ConfigureNativeQwen3_5EncoderBudget(cfg,config,arch,vt::DeviceType::kXPU,mm);
  CHECK(cfg.encoder_cache_size==32768);
  KVCacheConfig kv; kv.num_blocks=128;
  kv.kv_cache_groups.emplace_back(std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16,1,1,DType::kF32));
  Scheduler scheduler(cfg,kv,16,/*enable_caching=*/false);
  for (int i=0;i<6;++i) {
    const auto id=std::to_string(i);
    AddRequest(scheduler,MmRequest(id,16,i%2==0 ?
        std::vector<vllm::multimodal::MultiModalFeatureSpec>{Item(id,4,4)} :
        std::vector<vllm::multimodal::MultiModalFeatureSpec>{}));
  }
  const auto first=scheduler.schedule();
  REQUIRE(first.scheduled_new_reqs.size()==4);
  CHECK(first.total_num_scheduled_tokens==64);
  CHECK(first.scheduled_encoder_inputs.size()==2);
  for (const auto& request : first.scheduled_new_reqs)
    scheduler.finish_requests(request.req_id,RequestStatus::kFinishedStopped);
  const auto second=scheduler.schedule();
  REQUIRE(second.scheduled_new_reqs.size()==2);
  CHECK(second.total_num_scheduled_tokens==32);
  CHECK(second.scheduled_encoder_inputs.size()==1);
  for (const auto& request : second.scheduled_new_reqs)
    scheduler.finish_requests(request.req_id,RequestStatus::kFinishedStopped);
  CHECK(scheduler.schedule().total_num_scheduled_tokens==0);
}

TEST_CASE("Native Qwen3.5 encoder admission: maximum images progress through 1600-token chunks") {
  for (const int concurrency : {1, 4}) {
    CAPTURE(concurrency);
    SchedulerConfig cfg;
    cfg.max_num_seqs = concurrency;
    cfg.max_num_batched_tokens = 1600;
    cfg.max_model_len = 16384;
    vllm::HfConfig config;
    config.hidden_size = 5120;
    config.raw["quantization_config"] = {{"quant_method", "exl3"}};
    vllm::MultiModalConfig mm;
    mm.limit_per_prompt["image"] = 2;
    vllm::entrypoints::ConfigureNativeQwen3_5EncoderBudget(
        cfg, config, "Qwen3_5ForConditionalGeneration", vt::DeviceType::kXPU, mm);
    KVCacheConfig kv;
    kv.num_blocks = 10000;
    kv.kv_cache_groups.emplace_back(std::vector<std::string>{"layer"},
        std::make_shared<FullAttentionSpec>(16, 1, 1, DType::kF32));
    Scheduler scheduler(cfg, kv, 16, /*enable_caching=*/false);
    std::vector<Request*> requests;
    for (int i = 0; i < concurrency; ++i) {
      const auto id = std::to_string(i);
      requests.push_back(AddRequest(scheduler, MmRequest(id, 8400,
          {Item(id + "-first", 4, 4096), Item(id + "-second", 4110, 4096)})));
    }
    int encoded = 0;
    int scheduled = 0;
    // Bounded deterministic scheduler steps, no execution thread or host wait.
    for (int step = 0; step < 24 && scheduled < concurrency * 8400; ++step) {
      CAPTURE(step);
      CAPTURE(scheduled);
      const auto out = scheduler.schedule();
      REQUIRE(out.total_num_scheduled_tokens > 0);
      CHECK(out.total_num_scheduled_tokens <= 1600);
      scheduled += out.total_num_scheduled_tokens;
      for (const auto& [id, inputs] : out.scheduled_encoder_inputs)
        encoded += static_cast<int>(inputs.size());
    }
    CHECK(scheduled == concurrency * 8400);
    CHECK(encoded == concurrency * 2);
    for (const auto* request : requests) CHECK(request->num_computed_tokens == 8400);
  }
}

TEST_CASE("Scheduler.schedule: an image request carries mm_features to the worker") {
  // NewRequestData::from_request is the hop that used to drop the payload.
  auto scheduler = CreateMmScheduler(64, 64, 64);
  AddRequest(*scheduler, MmRequest("img", /*num_tokens=*/12,
                                   {Item("hash-a", /*offset=*/4, /*length=*/4)}));

  const SchedulerOutput out = scheduler->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 1);
  REQUIRE(out.scheduled_new_reqs[0].mm_features.size() == 1);
  CHECK(out.scheduled_new_reqs[0].mm_features[0].mm_hash == "hash-a");
  CHECK(out.scheduled_new_reqs[0].mm_features[0].offset == 4);
  CHECK(out.scheduled_new_reqs[0].mm_features[0].length == 4);

  // ...and the encoder admission named the item to run.
  REQUIRE(out.scheduled_encoder_inputs.count("img") == 1);
  REQUIRE(out.scheduled_encoder_inputs.at("img").size() == 1);
  CHECK(out.scheduled_encoder_inputs.at("img")[0] == 0);
  CHECK(out.total_num_scheduled_tokens == 12);
}

TEST_CASE("Scheduler.schedule: the same mm_hash twice is ONE encoder run") {
  // check_and_update_cache keys on the CONTENT hash, so a prompt that repeats
  // one image pays for one encoder pass. Keying on (req_id, input_id) would
  // schedule both and spend the budget twice.
  auto scheduler = CreateMmScheduler(64, 64, 64);
  AddRequest(*scheduler, MmRequest("twice", /*num_tokens=*/16,
                                   {Item("same", 2, 4), Item("same", 8, 4)}));

  const SchedulerOutput out = scheduler->schedule();
  REQUIRE(out.scheduled_encoder_inputs.count("twice") == 1);
  CHECK(out.scheduled_encoder_inputs.at("twice").size() == 1);
  CHECK(out.scheduled_encoder_inputs.at("twice")[0] == 0);
}

TEST_CASE("Scheduler.schedule: an already-encoded item is not re-scheduled") {
  // The second step covers the same span again (chunked prefill), and the cache
  // answers for it. A scheduler that re-scheduled would re-run the tower on
  // every chunk of a long prompt.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/8, 64, 64);
  AddRequest(*scheduler, MmRequest("img", /*num_tokens=*/16,
                                   {Item("hash-a", /*offset=*/0, /*length=*/4)}));

  const SchedulerOutput first = scheduler->schedule();
  CHECK(first.num_scheduled_tokens.at("img") == 8);
  REQUIRE(first.scheduled_encoder_inputs.count("img") == 1);

  const SchedulerOutput second = scheduler->schedule();
  CHECK(second.num_scheduled_tokens.at("img") == 8);
  CHECK(second.scheduled_encoder_inputs.empty());
}

// ---------------------------------------------------------------------------
// THE STRADDLING ITEM — the case the whole three-hop landing exists to keep
// safe. Upstream never chunks an ENCODER: when an item cannot be admitted, the
// admission TRUNCATES the decoder chunk to stop before the item's span. Without
// that truncation a chunk covers placeholder rows whose encoder never ran, and
// the runner's gather raises "Encoder cache miss" — which on the serving path
// is an HTTP 500 for an ordinary image request under a small token budget.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: an unschedulable encoder input TRUNCATES the chunk before its span") {
  // Encoder compute budget 2 < the item's 4 placeholder tokens, so the item
  // cannot run this step at all. The step must stop at offset 6.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/32,
                                     /*encoder_compute_budget=*/2,
                                     /*encoder_cache_size=*/64);
  AddRequest(*scheduler, MmRequest("img", /*num_tokens=*/16,
                                   {Item("hash-a", /*offset=*/6, /*length=*/4)}));

  const SchedulerOutput out = scheduler->schedule();
  // WITHOUT the clamp this is 16 and the step covers rows 6..9 with no encoder
  // output behind them.
  CHECK(out.num_scheduled_tokens.at("img") == 6);
  CHECK(out.scheduled_encoder_inputs.empty());
  // The tokens that WERE scheduled stop exactly at the span.
  CHECK(out.total_num_scheduled_tokens == 6);
}

TEST_CASE("Scheduler.schedule: the encoder CACHE, not only the budget, truncates") {
  // Same clamp, reached through the other arm: the compute budget is ample and
  // the cache is too small to hold the item. Two arms, one guard — a fix that
  // only consulted the budget would leave this one covering unencoded rows.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/32,
                                     /*encoder_compute_budget=*/64,
                                     /*encoder_cache_size=*/2);
  AddRequest(*scheduler, MmRequest("img", /*num_tokens=*/16,
                                   {Item("hash-a", /*offset=*/6, /*length=*/4)}));

  const SchedulerOutput out = scheduler->schedule();
  CHECK(out.num_scheduled_tokens.at("img") == 6);
  CHECK(out.scheduled_encoder_inputs.empty());
}

TEST_CASE("Scheduler.schedule: disable_chunked_mm_input rolls back to before the item") {
  // The stricter arm (scheduler.py:1470). The budget is ample, so the item WOULD
  // be admitted; what refuses is that the token budget covers only PART of its
  // span. 10 tokens would reach row 9 and the span is [6, 10) — so the step
  // rolls back to 6 rather than splitting the item.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/9,
                                     /*encoder_compute_budget=*/64,
                                     /*encoder_cache_size=*/64,
                                     /*disable_chunked_mm_input=*/true);
  AddRequest(*scheduler, MmRequest("img", /*num_tokens=*/16,
                                   {Item("hash-a", /*offset=*/6, /*length=*/4)}));

  const SchedulerOutput out = scheduler->schedule();
  CHECK(out.num_scheduled_tokens.at("img") == 6);
  CHECK(out.scheduled_encoder_inputs.empty());

  // The DEFAULT arm on the same shape splits the span and admits the encoder,
  // which is upstream's default and is safe because the encoder ran.
  auto lenient = CreateMmScheduler(/*max_num_batched_tokens=*/9, 64, 64,
                                   /*disable_chunked_mm_input=*/false);
  AddRequest(*lenient, MmRequest("img", 16, {Item("hash-a", 6, 4)}));
  const SchedulerOutput out2 = lenient->schedule();
  CHECK(out2.num_scheduled_tokens.at("img") == 9);
  REQUIRE(out2.scheduled_encoder_inputs.count("img") == 1);
}

TEST_CASE("Scheduler: a finished request's encoder entry becomes freeable and is EVICTED under pressure") {
  // free_encoder_mm_hashes is the worker's only signal to drop a vision
  // embedding. It reports what the cache actually EVICTED, not what became
  // unreferenced, so a cache with room reports nothing and the worker keeps the
  // entry for the next request that shares the hash.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/64,
                                     /*encoder_compute_budget=*/64,
                                     /*encoder_cache_size=*/4);
  AddRequest(*scheduler, MmRequest("a", /*num_tokens=*/8, {Item("hash-a", 0, 4)}));
  const SchedulerOutput first = scheduler->schedule();
  REQUIRE(first.scheduled_encoder_inputs.count("a") == 1);
  CHECK(first.free_encoder_mm_hashes.empty());

  scheduler->finish_requests("a", vllm::v1::RequestStatus::kFinishedAborted);

  // A second request needs the whole 4-slot cache for a DIFFERENT image, so the
  // first entry is evicted and named.
  AddRequest(*scheduler, MmRequest("b", /*num_tokens=*/8, {Item("hash-b", 0, 4)}));
  const SchedulerOutput second = scheduler->schedule();
  REQUIRE(second.scheduled_encoder_inputs.count("b") == 1);
  REQUIRE(second.free_encoder_mm_hashes.size() == 1);
  CHECK(second.free_encoder_mm_hashes[0] == "hash-a");
}

// ---------------------------------------------------------------------------
// PREEMPTION frees the victim's encoder references — `_preempt_request`'s
// `self.encoder_cache_manager.free(request)` (scheduler.py:1213), which sits
// beside the block free and had NO test: deleting the call left this whole file
// and `test_openai_api_server_mm_forward` green, because no case constructed a
// preempted request that carried `mm_features`.
//
// WHAT GOES WRONG WITHOUT IT, precisely. A preemption resets
// `num_computed_tokens` to 0, and `_free_encoder_inputs` frees an entry only
// once `offset + length <= num_computed_tokens`. So the victim's references
// survive the preemption and nothing drops them until the request either
// re-passes the span or finishes. That is an UNRECLAIMABLE WINDOW, not an
// unbounded leak: on resume `check_and_update_cache` re-references the same
// entry idempotently, and the entry frees normally once the span is re-passed.
// For the length of the window the slots are pinned, and `can_allocate` refuses
// admissions the cache in fact has room for.
//
// WHY THE WAITING LOOP MUST NOT RUN, AND WHAT ACTUALLY STOPS IT. The loop would
// re-visit the victim (a preemption prepends it to the waiting FRONT) and either
// re-reference the entry through `CheckAndUpdateCache` or drop it through the
// `allocate_slots` failure untouch (`free_request_encoder_inputs`,
// scheduler.cpp:939; scheduler.py:938-939) — and BOTH arms converge on the same
// counter whether or not `preempt_request` freed anything, which is exactly a
// test that cannot see the defect. What keeps the loop out is the guard, not the
// token budget: `Scheduler::schedule` wraps the whole waiting section in
// `if (preempted_reqs.empty())` (scheduler.cpp:849), which sits BEFORE
// `while (!waiting->empty() && token_budget > 0)` (scheduler.cpp:855). A step
// that preempted anything can never reach the loop, whatever the budget is.
// Upstream is identical at the pin (5559679229): `if not preempted_reqs and
// self._pause_state == PauseState.UNPAUSED:` (scheduler.py:668). So the sizing
// below buys the step's SHAPE — one decode chunk, one preemption, one 4-embed
// item — and it does not buy the loop's skip. Re-sizing it does not disarm this
// case, and a reviewer who widens the budget will still see both assertions go
// red when the free in `preempt_request` is deleted.
// ---------------------------------------------------------------------------
TEST_CASE("Scheduler.schedule: preemption RELEASES the victim's encoder cache slots") {
  // Budget 16 == one block, so a single decode chunk spends the whole step.
  // 8 blocks (one is the null block) leaves 7 usable: 5 for the hog's 80-token
  // prompt and 2 for the image request's 32. The encoder cache holds exactly one
  // 4-embed item, so "released" and "still pinned" are the only two states.
  auto scheduler = CreateMmScheduler(/*max_num_batched_tokens=*/16,
                                     /*encoder_compute_budget=*/64,
                                     /*encoder_cache_size=*/4,
                                     /*disable_chunked_mm_input=*/false,
                                     /*max_num_seqs=*/16,
                                     /*num_blocks=*/8, /*block_size=*/16);
  // The KV hog is TEXT and its prompt is all zeros, while MmRequest's is all
  // sevens: distinct prompts, so prefix caching cannot let one ride on the
  // other's blocks.
  auto hog_owner = CreateRequests(/*num_requests=*/1, /*num_tokens=*/80, {"hog"});
  Request* hog = hog_owner[0].get();
  AddRequest(*scheduler, std::move(hog_owner[0]));
  for (int i = 0; i < 5; ++i) {
    const SchedulerOutput out = scheduler->schedule();
    REQUIRE(out.num_scheduled_tokens.at("hog") == 16);
  }
  REQUIRE(hog->num_computed_tokens == 80);

  // The image request is admitted BEHIND the hog, so it is the FCFS tail and
  // therefore the preemption victim. Its item spans [0, 4), inside the first
  // 16-token chunk, so this step is the one that runs the encoder.
  Request* img = AddRequest(
      *scheduler, MmRequest("img", /*num_tokens=*/32,
                            {Item("hash-a", /*offset=*/0, /*length=*/4)}));
  const SchedulerOutput admit = scheduler->schedule();
  REQUIRE(admit.scheduled_encoder_inputs.count("img") == 1);
  // The item is allocated: the whole 4-slot cache is spoken for, and none of it
  // is reclaimable while the reference stands.
  REQUIRE(scheduler->encoder_cache_manager->num_free_slots() == 0);
  REQUIRE(scheduler->encoder_cache_manager->num_freeable_slots() == 0);

  // Finish the image request's prefill so both requests hold their full block
  // count and the cache is exactly full.
  const SchedulerOutput fill = scheduler->schedule();
  REQUIRE(fill.num_scheduled_tokens.at("img") == 16);
  REQUIRE(img->num_computed_tokens == 32);

  // The hog now has 16 sampled tokens the scheduler has not seen: one full
  // budget's worth, needing a sixth block that the KV cache cannot give. The
  // FCFS tail (the image request) is preempted to pay for it.
  for (int i = 0; i < 16; ++i) hog->AppendOutputToken(0);
  const SchedulerOutput out = scheduler->schedule();
  REQUIRE(out.num_scheduled_tokens.at("hog") == 16);
  REQUIRE(img->status == RequestStatus::kPreempted);
  REQUIRE(img->num_computed_tokens == 0);
  // INSTRUMENT PRECONDITION: this step preempted, so the `preempted_reqs.empty()`
  // guard (scheduler.cpp:849) skipped the waiting loop and nothing touched the
  // victim's encoder references after `preempt_request`. Without that skip the
  // two assertions below would read the resume rather than the preemption.
  REQUIRE(out.num_scheduled_tokens.count("img") == 0);
  REQUIRE(scheduler->waiting->peek_request() == img);

  // THE ASSERTION. The victim's reference is gone, so the entry is reclaimable
  // again and the counter is back at the value it had before the allocation
  // (cache_size == 4). Without the free in preempt_request this reads 0.
  CHECK(scheduler->encoder_cache_manager->num_freeable_slots() == 4);
  // And the consequence that makes it matter: an admission the pinned slots
  // would have refused now succeeds. can_allocate has no free slot to give, so
  // it can only say yes by reclaiming the preempted request's entry.
  CHECK(scheduler->encoder_cache_manager->CanAllocate(
      /*num_embeds=*/4, /*encoder_compute_budget=*/64,
      /*num_embeds_to_schedule=*/0));
}
