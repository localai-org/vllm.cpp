// Ported from: vllm/v1/core/sched/scheduler.py @ e24d1b24
//
// Scope (M1.4 Task 3): the V1 Scheduler's core token-budget algorithm — the
// unified schedule() loop the EngineCore step drives. There is no "prefill" vs
// "decode" distinction: each Request tracks num_computed_tokens vs num_tokens,
// and each step assigns tokens under a per-step budget so num_computed catches
// up to num_tokens. This covers chunked prefill, prefix caching and (deferred)
// speculative / jump decoding uniformly.
//
// Ported T0 behavior (matched line-by-line with upstream schedule()):
//   * running-first loop: num_new_tokens = num_tokens_with_spec +
//     num_output_placeholders - num_computed_tokens, clamped by the
//     long_prefill threshold, the token budget, and (max_model_len -
//     num_computed - num_sampled_tokens_per_step); allocate_slots -> nullopt
//     triggers FCFS-tail preemption (_preempt_request) retried until the
//     request fits or it is itself preempted.
//   * waiting loop (only when no preemption happened this step, and under both
//     the token budget and max_num_running_reqs): get_computed_blocks prefix
//     hit, chunked-prefill split via min(num_new_tokens, token_budget),
//     allocate_slots with full_sequence_must_fit = scheduler_reserve_full_isl,
//     admit as scheduled_new (WAITING) or scheduled_resumed (PREEMPTED).
//   * SchedulerOutput assembly (MRV2 / V2 model-runner path — upstream
//     scheduler.py `if self.use_v2_model_runner:` branch): resumed
//     (PREEMPTED->RUNNING) reqs fold into scheduled_new_reqs and are re-sent as
//     FULL NewRequestData carrying prefill_token_ids = request.AllTokenIds()
//     (the V2 gpu runner asserts prefill_token_ids on every new req and
//     re-seeds its all_token_ids from it); already-running reqs get the
//     CachedRequestData diff (_make_cached_request_data) WITHOUT the MRV1
//     all_token_ids / prev_step_scheduled_req_ids bookkeeping (the V2 runner
//     never reads CachedRequestData.all_token_ids). Plus the num_scheduled_
//     tokens map + total, num_common_prefix_blocks, finished_req_ids; then
//     _update_after_schedule advances num_computed_tokens (so the next step
//     continues a chunked prefill) and flips is_prefill_chunk. The MRV1 path
//     (resumed-as-cached) is NOT ported.
//   * add_request / finish_requests / get_num_unfinished_requests /
//     get_request_counts / has_finished_requests (T0 subset).
//
// DEFERRED (marked; matches upstream structure so re-adding is mechanical):
//   - Encoder inputs (_try_schedule_encoder_inputs), the encoder cache manager
//     and the encoder compute budget are LANDED (ENG-MM-INPUT-PIPELINE P2,
//     #2379). Every one of them is predicated on Request::mm_features being
//     non-empty, so a text-only engine never constructs an encoder input, never
//     spends encoder budget, and emits an empty scheduled_encoder_inputs /
//     free_encoder_mm_hashes exactly as before. STILL DEFERRED inside that
//     port: the encoder-DECODER manager subclass (Whisper cross-attention),
//     the EC connector's remote encoder cache, and the EAGLE +1 look-ahead
//     shift (use_eagle is false here).
//   - SPEC-MTP (I2, scheduler-half) LANDED the core spec-decode scheduler
//     plumbing: num_lookahead_tokens (derived from an optional SpeculativeConfig,
//     0 by default), num_tokens_with_spec in the running-loop budget, the
//     per-step scheduled_spec_decode_tokens map (1 + len(spec_token_ids) tokens
//     scheduled for a running request carrying drafts, scheduler.py:593-609),
//     the num_computed_tokens rollback on rejection (scheduler.py:1580-1612), and
//     update_draft_token_ids (scheduler.py:1937). All INERT when no
//     SpeculativeConfig is supplied (num_lookahead_tokens == 0, every request's
//     spec_token_ids stays empty), so the default path is byte-identical.
//     SPEC-DFLASH2 W7 (#1824) un-deferred the ASYNC draft-in-output path:
//     num_spec_tokens_to_schedule / num_invalid_spec_tokens on SchedulerOutput,
//     update_draft_token_ids_in_output (scheduler.py:2072-2107), the
//     async_tokens_to_discard rollback guard (:1670-1675) and the preempt-time
//     spec_token_ids clear (:1217-1218). STILL DEFERRED: the sync scheduler's
//     first-decode-step spec PADDING (pad_spec_decode, scheduler.py:827-843 — a
//     cudagraph-uniformity optimization) and the dynamic-SD lookup.
//   - num_output_placeholders / async scheduling (the early-continue on
//     max_tokens, next_decode_eligible_step) — treated as 0 / inert.
//   - DP prefill balancing (throttle_prefills / defer_prefills /
//     prefill_capacity_bound), PAUSE state, streaming/resumable sessions,
//     skipped_waiting, priority scheduling (SchedulingPolicy::kPriority throws).
//   - Structured-output grammar, LoRA gating, KV-connector / EC-connector,
//     mamba block-aligned split, defer_block_free / deferred_frees, kv-cache
//     metrics, MFU perf metrics, kv_cache_events. (log-stats events /
//     take_events are un-deferred for SERVE-RESPONSE-METRICS: schedule() records
//     QUEUED/SCHEDULED/PREEMPTED per request, drained onto EngineCoreOutput.)
//   This ports the MRV2 (V2 model runner, use_v2_model_runner = true) output
//   path — resumed reqs fold into new_reqs carrying prefill_token_ids — to
//   match the MRV2 gpu runner M1.5 lands. The MRV1 output path (resumed reqs
//   stay separate and flow through the cached diff) is NOT ported.
#ifndef VLLM_V1_CORE_SCHED_SCHEDULER_H_
#define VLLM_V1_CORE_SCHED_SCHEDULER_H_

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/config/speculative.h"
#include "vllm/v1/core/encoder_cache_manager.h"  // EncoderCacheManager (P2, #2379)
#include "vllm/distributed/kv_events.h"  // KVEventsConfig, EventPublisher
#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/recurrent_prefix_policy.h"
#include "vllm/v1/core/sched/request_queue.h"
#include "vllm/v1/engine/types.h"  // ModelRunnerOutput, EngineCoreOutputs
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/spec_decode/metrics.h"  // spec_decode::SpecDecodingStats

namespace vllm::v1::kv_offload {
class KVConnector;  // vllm/v1/kv_offload/kv_connector.h (KV-CONNECTORS W5 ABI)
}  // namespace vllm::v1::kv_offload

namespace vllm::v1 {

class StructuredOutputManager;  // vllm/v1/structured_output/manager.h

// The V1 Scheduler (T0 subset). Owns the Request lifetime (a map of owning
// unique_ptr); running_ and waiting_ hold non-owning raw pointers into it,
// exactly as upstream where the running list and waiting queue alias the same
// Request instances stored in self.requests.
class Scheduler {
 public:
  // Build the scheduler + its KVCacheManager. Mirrors the fields the upstream
  // ctor reads out of vllm_config for the T0 path: the scheduler_config knobs
  // (max_num_seqs, max_num_batched_tokens/max_num_scheduled_tokens,
  // long_prefill_token_threshold, enable_chunked_prefill, policy, watermark,
  // scheduler_reserve_full_isl, max_model_len) + the kv_cache_config, block
  // size, and the prefix-caching flag (upstream cache_config.enable_prefix_
  // caching). enable_caching is passed straight to the KVCacheManager.
  // structured_output_manager (upstream core.py:153 passes it into the
  // Scheduler ctor): the engine's StructuredOutputManager, used by
  // get_grammar_bitmask + the update_from_output FSM advance. Optional (null) to
  // stay backward-compatible with the M1.4/M1.8 tests that build a bare
  // scheduler — when null, structured output is a no-op (get_grammar_bitmask
  // returns nullopt; no grammar advance).
  //
  // speculative_config (upstream vllm_config.speculative_config, read at
  // scheduler.py:275-292): OPTIONAL. std::nullopt (the default) is the production
  // no-speculator path — num_lookahead_tokens stays 0, so every spec-decode path
  // is inert and the scheduler is byte-identical to the pre-SPEC-MTP engine. When
  // present (e.g. SpeculativeConfig::ResolveMtp), num_lookahead_tokens is derived
  // from it (SpeculativeConfig::NumLookaheadTokens) and threaded into
  // allocate_slots so the verify slots are reserved ahead of time.
  //
  // kv_events_config (upstream vllm_config.kv_events_config, read at
  // scheduler.py:86,116-119,155-158): OPTIONAL, and nullptr — the default, and
  // what every non-event call site passes — is upstream's `None`. It resolves
  // enable_kv_cache_events to FALSE, which is exactly the value this ctor used
  // to hard-code, so the KVCacheManager, the BlockPool and update_from_output
  // all behave byte-identically and the publisher is a no-op NullEventPublisher.
  // The pointee is read only during construction (to derive the flag and build
  // the publisher) and is NOT retained, so it need not outlive the call.
  // data_parallel_rank mirrors parallel_config.data_parallel_index: the
  // publisher stamps it onto every batch it emits.
  // hash_block_size (scheduler.py:76,268-270) is the granularity a request's
  // block hashes are computed at. It is NOT always block_size: a KV cache group
  // may page SMALLER than the engine's block table strides, and upstream
  // resolves the two separately (engine/core.py:158-170). 0 is upstream's
  // `None` — "same as block_size" — which is what every existing call site
  // means and what keeps them byte-identical.
  Scheduler(SchedulerConfig scheduler_config, KVCacheConfig kv_cache_config,
            int block_size, bool enable_caching = false,
            StructuredOutputManager* structured_output_manager = nullptr,
            std::optional<SpeculativeConfig> speculative_config = std::nullopt,
            const distributed::KVEventsConfig* kv_events_config = nullptr,
            int data_parallel_rank = 0, int hash_block_size = 0);

  // VIRTUAL destructor — REQUIRED, not cosmetic. `AsyncScheduler` derives from
  // this class and production/test code owns the derived object through a
  // `std::unique_ptr<Scheduler>` base pointer
  // (`src/vllm/entrypoints/model_loader.cpp::MakeScheduler`,
  // `include/vllm/entrypoints/model_loader.h::scheduler_`, tests/vllm/v1/
  // test_async_admission_timing.cpp:167-170). Deleting a derived object through
  // a base pointer whose destructor is non-virtual is UNDEFINED BEHAVIOUR
  // ([expr.delete]/3) — the derived destructor never runs. GCC does not
  // diagnose it; Clang's -Wdelete-non-abstract-non-virtual-dtor caught it while
  // building on macOS (BACKEND-METAL-MLX W0 item 4). This is a latent
  // correctness fix on EVERY platform, not a macOS accommodation, so it is
  // FIXED rather than suppressed. Declaring it also suppresses the implicit
  // move operations, which is harmless here: `Scheduler` owns a
  // `unique_ptr<KVCacheManager>` and was already documented non-movable.
  virtual ~Scheduler() = default;

  // add_request: enqueue a new request into waiting_ and register it in the
  // requests_ map (the scheduler takes ownership). (Upstream add_request T0
  // branch — no streaming/resumable/connector.)
  void add_request(std::unique_ptr<Request> request);

  // finish_requests: handle an external finish/abort signal for one request.
  // Removes it from running_/waiting_, sets the finished status, records it in
  // finished_req_ids_, frees its KV blocks, and erases it from requests_
  // (freeing the Request). No-op for an unknown or already-finished id.
  // (Upstream finish_requests + _free_request, T0 subset.)
  void finish_requests(const std::string& request_id,
                       RequestStatus finished_status);

  // schedule(): the core token-budget algorithm. See the file header.
  SchedulerOutput schedule();

  // get_grammar_bitmask (scheduler.py:1477-1499): collect the scheduled
  // structured-output request ids (skipping requests still in a prefill chunk),
  // in num_scheduled_tokens order, and ask the StructuredOutputManager to fill
  // their per-step token bitmask. Returns nullopt when there are no structured
  // reqs this step (or no manager is wired). The rows of the returned bitmask are
  // ordered exactly as GrammarOutput.structured_output_request_ids.
  std::optional<GrammarOutput> get_grammar_bitmask(
      const SchedulerOutput& scheduler_output);

  // update_from_output: the final leg of the schedule -> execute -> update loop.
  // For each scheduled request (iterating scheduler_output.num_scheduled_tokens,
  // matching upstream's `for req_id ... in num_scheduled_tokens.items()`), it
  // reads that request's sampled token(s) from model_runner_output via
  // req_id_to_index, appends them one at a time running check_stop after each
  // (trimming any tokens past a stop), and on a stop sets the finished status,
  // frees the request's KV blocks, records its id in finished_req_ids, removes it
  // from running_/waiting_ and erases it from the requests map. It builds one
  // EngineCoreOutput per request that produced tokens or finished, and returns
  // the batched EngineCoreOutputs for the step. A request still in chunked
  // prefill receives an empty token list from the runner and stays running with
  // no output. (Upstream scheduler.py::update_from_output, T0 subset.)
  //
  // DEFERRED (matches upstream structure): logprobs / prompt_logprobs,
  // speculative-decode acceptance & num_computed rollback, structured-output
  // grammar advance, pooling outputs, encoder-input free, KV-connector finish /
  // invalid-block reload, routed-experts, num_nans_in_logits, spec/perf/spec-
  // decoding stats, resumable / streaming sessions (so _handle_stopped_request
  // is always "finished" here), and the per-client_index output fan-out (T0 has
  // a single client, so a flat EngineCoreOutputs is returned rather than
  // dict[client_index, EngineCoreOutputs]).
  EngineCoreOutputs update_from_output(
      const SchedulerOutput& scheduler_output,
      const ModelRunnerOutput& model_runner_output);

  // KV-OFFLOAD W4: attach the scheduler-facing half of a KV connector. A null
  // connector (the default) is ZERO behaviour change — offloading is opt-in and
  // provably inert when off, so every existing model regression is unaffected.
  // The scheduler does NOT own the connector; the caller keeps it alive for the
  // scheduler's lifetime.
  void set_kv_connector(kv_offload::KVConnector* connector) {
    kv_connector_ = connector;
  }
  kv_offload::KVConnector* kv_connector() const {
    return kv_connector_;
  }

  // KV-EVENTS W3: replace the publisher the ctor built from the config.
  //
  // Upstream has no such setter — it never needs one, because its only non-null
  // publisher (ZmqEventPublisher) can be constructed from configuration alone.
  // Ours cannot yet: the ZMQ transport is DEFERRED for want of a socket
  // dependency, so the only publisher that can be OBSERVED is
  // CollectingEventPublisher, and teaching EventPublisherFactory a
  // `publisher="collecting"` kind would invent a --kv-events-config value vLLM
  // does not have, making our config surface diverge for the sake of a test.
  // This is the same injection shape set_kv_connector uses above, and for the
  // same reason. Production never calls it: the config-driven path stays 1:1.
  // The scheduler OWNS the publisher (unlike the connector).
  void set_kv_event_publisher(
      std::unique_ptr<distributed::EventPublisher> publisher) {
    kv_event_publisher_ = std::move(publisher);
  }
  distributed::EventPublisher* kv_event_publisher() const {
    return kv_event_publisher_.get();
  }

  // shutdown (scheduler.py:2456-2459): release the scheduler's external
  // resources. Today that is only the KV-event publisher; upstream also shuts
  // its KV and EC connectors down here, which are separate rows.
  //
  // NOTE: nothing in production calls this yet — upstream calls it from
  // EngineCore.shutdown, and our EngineCore has no shutdown path at all. Both
  // shipped publishers make that harmless (Null's shutdown is an empty body,
  // Collecting's sets a flag); the real consumer is the deferred ZMQ
  // publisher's background thread, which arrives with the transport.
  // Idempotent, and safe to call on a scheduler that has no events configured.
  virtual void shutdown();

  // update_draft_token_ids (scheduler.py:1937-1957): install the drafter's
  // freshly-proposed spec token ids onto their requests, to be scheduled for
  // verification on the NEXT step. For each (req_id, draft ids) pair: skip
  // unknown / finished requests; a request still in a prefill CHUNK ignores the
  // drafts (clearing any stale spec_token_ids); otherwise request.spec_token_ids
  // is set to the proposed ids (a structured-output request first validates them
  // against its grammar — deferred here, no-op without a manager). Called by the
  // engine core (core.py:511-517) with the runner's out-of-band
  // take_draft_token_ids output. Inert unless a speculator is configured (the
  // runner never produces drafts otherwise).
  void update_draft_token_ids(const DraftTokenIds& draft_token_ids);

  // update_draft_token_ids_in_output (scheduler.py:2072-2107, SPEC-DFLASH2 W7
  // #1824): the ASYNC variant. Under async scheduling the request state holds
  // only -1 placeholders (the worker keeps the real drafts), so when the
  // deferred (structured-output) sampling path needs real draft values for its
  // grammar bitmask, the worker's drafts are rewritten INTO the
  // SchedulerOutput's scheduled_spec_decode_tokens: each row is trimmed to the
  // scheduled count, a short row is padded back to it with -1, and the invalid
  // tail is recorded in scheduler_output.num_invalid_spec_tokens (REPLACED
  // whole per call). Unknown / finished requests and requests with no
  // scheduled entry are skipped. The grammar validate_tokens arm is deferred
  // exactly as in update_draft_token_ids above.
  void update_draft_token_ids_in_output(const DraftTokenIds& draft_token_ids,
                                        SchedulerOutput& scheduler_output);

  // async_scheduling: whether this scheduler is the async-scheduling class.
  // The engine core's post_step reads it to skip the out-of-band draft pull
  // under async scheduling (core.py:617 `not self.async_scheduling`; upstream
  // reads the resolved config — here the resolution PRODUCT is the scheduler
  // class, model_loader.cpp::MakeScheduler, so the class answers).
  virtual bool async_scheduling() const { return false; }

  // get_num_unfinished_requests: len(waiting) + len(running) (T0 subset).
  int get_num_unfinished_requests() const;
  // get_request_counts: (num_running, num_waiting).
  std::pair<int, int> get_request_counts() const;
  // has_finished_requests: whether any request finished since the last step.
  bool has_finished_requests() const;

  // The resolved scheduling policy this scheduler was constructed with
  // (SchedulerConfig.policy). Exposed so the ENG-SGLANG-BEHAVIOR-FLAG enablement
  // gate can assert that `lpm` (or `priority`) selected through the C ABI / C++
  // API / server flag actually reached the scheduler. Output-neutral (admission
  // ORDER only); see order_waiting_queue().
  SchedulerPolicy policy() const { return scheduler_config_.policy; }

  // --- Public state the ported tests inspect directly (mirrors upstream's
  // accessible attributes). --------------------------------------------------

  // req_id -> owning Request. The scheduler owns Request lifetime.
  std::map<std::string, std::unique_ptr<Request>> requests;
  // Running requests, in scheduling order (non-owning; the FCFS tail is
  // running.back(), the first preemption victim).
  std::vector<Request*> running;
  // The waiting (FCFS) queue (non-owning pointers).
  std::unique_ptr<RequestQueue> waiting;
  // Request ids finished between the previous and current step (flushed each
  // step by _update_after_schedule).
  std::set<std::string> finished_req_ids;
  // Ids preempted since the last schedule() (flushed each step).
  std::set<std::string> reset_preempted_req_ids;

  // Resolved scheduling constraints (upstream ctor).
  int max_num_running_reqs;
  int max_num_scheduled_tokens;
  int max_model_len;

  // The KV cache manager the scheduler drives (heap-owned; non-movable).
  std::unique_ptr<KVCacheManager> kv_cache_manager;

  // ENG-MM-INPUT-PIPELINE P2 (#2379): the encoder cache the multimodal
  // admission path budgets against (scheduler.py:230-238). Upstream builds a
  // manager unconditionally and gives it `cache_size = 0` when the model has no
  // multimodal budget; ours is built the same way from
  // SchedulerConfig::encoder_cache_size. It is only ever CONSULTED for a
  // request whose mm_features are non-empty, so a text engine's steps do not
  // touch it. Public because the ported encoder tests inspect it, mirroring
  // upstream's plain attribute.
  std::unique_ptr<core::EncoderCacheManager> encoder_cache_manager;

  // The rolling prefix-cache hit rate over the most recent 1000 REQUESTS.
  // schedule() folds each step's take-and-swap delta in. This is the engine's
  // answer to "prove the cache is being hit", which the benchmark protocol
  // requires of every caching/offloading arm.
  const CachingMetrics& prefix_cache_metrics() const {
    return prefix_cache_metrics_;
  }

  // make_stats (scheduler.py:2399-2436): snapshot this step's always-on
  // scheduler state for the Prometheus stat logger — running/waiting counts,
  // KV-cache usage, and the per-step prefix-cache token delta. Upstream drains
  // make_prefix_cache_stats() here and the frontend logger observes it into
  // CachingMetrics; our schedule() already drains+observes that delta (it owns
  // the CachingMetrics window, which non-logging callers read), so this returns
  // the SAME per-step delta stashed by schedule() — identical values, no double
  // take-and-swap. Const: reads the stash, never drains. The advanced fields
  // (skipped-waiting, connector/spec/cudagraph/perf, eviction events) are
  // deferred with their config-gated metric families (see loggers.h).
  SchedulerStats make_stats() const;

  // Emit the chunked-prefill progress lines for the step just executed
  // (VT_SERVER_PREFILL_PROGRESS / VT_SERVER_VERBOSE). Called by EngineCore
  // AFTER execute_model, because schedule() advances num_computed_tokens before
  // the GPU has run and timing it there reports the rate of an empty step. A
  // pure diagnostic: no engine state is read or written beyond the request map,
  // and the whole body is a no-op when the knob is off. NOT upstream — vLLM logs
  // this from its Python front end.
  void LogPrefillAfterExecute(const SchedulerOutput& scheduler_output);

 protected:
  // _update_after_schedule (scheduler.py:1166-1213): advance num_computed_tokens
  // for every scheduled request, refresh is_prefill_chunk (num_computed <
  // num_tokens + num_output_placeholders), set has_structured_output_requests,
  // and flush the finished/preempted id sets. Takes scheduler_output by mutable
  // ref (it writes has_structured_output_requests). protected + VIRTUAL so
  // AsyncScheduler can extend it (add the output placeholders after super()).
  virtual void update_after_schedule(SchedulerOutput& scheduler_output);

  // _update_request_with_output (scheduler.py:1886-1902): append each generated
  // token to the request, run check_stop after each, and trim any tokens past a
  // stop. Returns {kept_token_ids, stopped}. Called by update_from_output only
  // when the runner produced tokens (upstream `if new_token_ids:`). protected +
  // VIRTUAL so AsyncScheduler can wrap it (stale-frame discard + placeholder
  // drain + block caching, async_scheduler.py:51-75).
  virtual std::pair<std::vector<int32_t>, bool> update_request_with_output(
      Request& request, std::vector<int32_t> new_token_ids);

  // Members the AsyncScheduler subclass reads. num_sampled_tokens_per_step_ is
  // the per-step sampled-token count (1 at T0, 0 for diffusion); current_step_
  // is the scheduler iteration counter (PP cadence).
  int num_sampled_tokens_per_step() const { return num_sampled_tokens_per_step_; }
  int current_step() const { return current_step_; }

 private:
  // KV-OFFLOAD W4: the scheduler-facing KV connector (non-owning, opt-in, null by
  // default). See set_kv_connector.
  kv_offload::KVConnector* kv_connector_ = nullptr;

  // KV-EVENTS W3 (scheduler.py:116-119,155-158). enable_kv_cache_events_ is
  // false unless a KVEventsConfig was supplied AND enables them; it is what the
  // KVCacheManager (and through it the BlockPool) is constructed with, so with
  // the default it is exactly the previously hard-coded `false`.
  // kv_event_publisher_ is never null — EventPublisherFactory returns a
  // NullEventPublisher when events are off — so the publish path needs no null
  // check, matching upstream where the attribute is always set.
  bool enable_kv_cache_events_ = false;
  std::unique_ptr<distributed::EventPublisher> kv_event_publisher_;

  // Backing store for prefix_cache_metrics(); written only by schedule().
  CachingMetrics prefix_cache_metrics_;

  // This step's take-and-swap prefix-cache delta, stashed by schedule() when it
  // drains make_prefix_cache_stats() (the SAME observation fed to CachingMetrics
  // and, via make_stats(), to SchedulerStats.prefix_cache_stats). Reset to empty
  // on a step that produced no observation so make_stats() never re-reports a
  // stale delta.
  PrefixCacheStats last_prefix_cache_stats_;
  int64_t last_prefill_kv_computed_tokens_ = 0;

  // This step's speculative-decoding aggregate, stashed by update_from_output
  // for make_stats() (#2770). Same arrangement, and the same reason, as
  // last_prefix_cache_stats_ above: EngineCore calls make_stats() after
  // update_from_output returns, so the value cannot be passed down the call the
  // way scheduler.py:2211 passes it. EMPTY on every step that verified no
  // draft, which is what keeps SchedulerStats::spec_decoding_stats an honest
  // "this step speculated" signal rather than a running total.
  std::optional<spec_decode::SpecDecodingStats> last_spec_decoding_stats_;

  // make_spec_decoding_stats (scheduler.py:2714-2731): fold ONE request's
  // verify result into the step aggregate, constructing it on first use.
  // Returns empty when stats are off or the request verified no draft, which is
  // upstream's own return shape. `num_invalid_spec_tokens` is
  // SchedulerOutput's map of drafts the drafter could not fill (scheduled as -1
  // placeholders); they were never really drafted and are subtracted here.
  std::optional<spec_decode::SpecDecodingStats> make_spec_decoding_stats(
      std::optional<spec_decode::SpecDecodingStats> spec_decoding_stats,
      int num_draft_tokens, int num_accepted_tokens,
      const std::map<std::string, int>& num_invalid_spec_tokens,
      const std::string& request_id) const;

  // _preempt_request (scheduler.py:1203-1224): free the request's KV, mark it
  // PREEMPTED, reset its computed tokens, and re-queue it to the FRONT of waiting
  // (FCFS retry). The request must already have been popped from running by the
  // caller. `timestamp` is schedule()'s shared step reading, used to stamp the
  // PREEMPTED event when log_stats_ is on.
  void preempt_request(Request* request, double timestamp);

  // _try_schedule_encoder_inputs (scheduler.py:1379, grep -c == 1). Decide which
  // of `request`'s multimodal items must have their ENCODER run this step, and
  // CLAMP `num_new_tokens` so the decoder never runs past an item whose encoder
  // could not be scheduled.
  //
  // THE CLAMP IS THE CORRECTNESS PART, not an optimization. Upstream never
  // chunks an encoder input: an item is encoded whole the first time any part of
  // its placeholder span is scheduled, and every later chunk hits the cache. The
  // one way a decoder chunk can cover placeholder rows with no encoder output is
  // when the item could NOT be scheduled (budget or cache exhausted), and this
  // function's fallback — truncate to just before `start_pos`, or refuse the
  // request this step when prefix caching already carried num_computed_tokens
  // past it — is what stops that. Without it the runner's gather raises
  // "Encoder cache miss", which on the serving path is an HTTP 500 for a
  // perfectly ordinary image request under a small token budget.
  //
  // Returns the mm_features INDICES to encode. `num_new_tokens` and
  // `encoder_compute_budget` are IN/OUT, exactly as upstream returns new values
  // for both. A request with no mm_features returns {} and touches neither.
  std::vector<int> try_schedule_encoder_inputs(const Request& request,
                                               int num_computed_tokens,
                                               int* num_new_tokens,
                                               int* encoder_compute_budget);

  // _free_encoder_inputs (scheduler.py:2017, grep -c == 1). Drop this request's
  // reference to every encoder input whose placeholder span is now fully behind
  // num_computed_tokens — i.e. already folded into the decoder KV, so no later
  // step can gather it again. Called from update_from_output, AFTER the step
  // executed (upstream's comment at :1699 says exactly why). No-op for a request
  // with no mm_features.
  void free_encoder_inputs(const Request& request);

  // encoder_cache_manager.free(request) (encoder_cache_manager.py:243): drop
  // every reference this request holds, on finish or abort. Ours has no
  // request-shaped overload, so the scheduler spells the loop.
  void free_request_encoder_inputs(const Request& request);

  // ENG-SGLANG-BEHAVIOR-FLAG (SW1): under the `lpm` policy, reorder the waiting
  // queue by descending longest cached-prefix match before the admission loop,
  // so high-cache-hit requests are admitted first (maximizing hit rate). Ported
  // from SGLang SchedulePolicy.calc_priority / _sort_by_longest_prefix
  // (schedule_policy.py:176,205 @ v0.5.15 f63458b): a STABLE sort by
  // -num_matched_prefix_tokens (ties keep FCFS/arrival order). Mirrors SGLang's
  // large-queue fallback (schedule_policy.py:229-233): when the queue exceeds
  // kLpmMaxWaitingQueue the expensive per-request match+sort is skipped (order
  // stays fcfs). No-op unless the policy is kLPM AND prefix caching is on
  // (an lpm run with caching off has nothing to match, so it stays fcfs). It is
  // OUTPUT-NEUTRAL: only admission order changes; the pure match lookup has no
  // side effects (no stats, no LRU touch).
  //
  // SW2 — in-batch prefix-collision de-prioritization (SGLang
  // SchedulePolicy._compute_prefix_matches, schedule_policy.py:253-301):
  // refines the LPM sort. Two waiting requests that share an UNCACHED prefix
  // (each with a small real-cache match) would otherwise both be admitted the
  // same step and REDUNDANTLY compute that prefix. SW2 lets the FIRST such
  // request through and DE-PRIORITIZES every later collider (sorts it behind all
  // non-colliding requests, mirroring the `float("inf")` sort key at
  // schedule_policy.py:311); the collider waits one round for the first to
  // populate the APC and then HITS it. Detected by reusing OUR block-hash APC
  // keys (Request::block_hashes — the same keys the admission loop matches
  // against) against an ephemeral per-step "seen" set of already-admitted-into-
  // the-sort prefixes — NOT a second radix trie. Still OUTPUT-NEUTRAL: only
  // admission order changes; each request computes the same tokens (a deferred
  // collider recomputes FEWER, reusing the now-cached prefix, but produces the
  // identical token stream). Inert unless two waiting requests actually collide.
  void maybe_reorder_waiting_for_lpm();

  // SGLang's expensive-prefix-sort cutoff (schedule_policy.py:230): above this
  // many waiting requests, lpm degrades to fcfs for the step.
  static constexpr std::size_t kLpmMaxWaitingQueue = 128;

  // SW2 in-batch prefix-caching thresholds (TOKENS), mirroring SGLang's env
  // defaults (schedule_policy.py:76,83). kInBatchCheckThreshold: only a request
  // whose REAL cached-prefix match is <= this is subject to the in-batch check
  // (a request already well-served by the cache is never de-prioritized).
  // kInBatchDeprioritizeThreshold: a request whose in-batch (waiting-queue)
  // prefix match is >= this is de-prioritized. Applied at block granularity
  // (in-batch match tokens = matched blocks * block_size_).
  static constexpr int kInBatchCheckThreshold = 32;
  static constexpr int kInBatchDeprioritizeThreshold = 32;

  // _make_cached_request_data: build the diff payload for the already-running
  // (running_reqs) + resumed-from-preemption (resumed_reqs) requests. MRV2:
  // use_pp is false so new_token_ids stays empty, and all_token_ids stays empty
  // (the V2 runner never reads it). The V2 schedule() tail clears resumed_reqs
  // before this call (resumed reqs are re-sent as new reqs), so resumed_reqs is
  // empty in the V2 path — kept in the signature for upstream fidelity.
  CachedRequestData make_cached_request_data(
      const std::vector<Request*>& running_reqs,
      const std::vector<Request*>& resumed_reqs,
      const std::map<std::string, int>& num_scheduled_tokens,
      const std::map<std::string, KVCacheBlocks>& req_to_new_blocks);

  // The engine's StructuredOutputManager (upstream scheduler.py:94), or null when
  // structured output is not wired (M1.4/M1.8 tests). Non-owning.
  StructuredOutputManager* structured_output_manager_ = nullptr;

  SchedulerConfig scheduler_config_;
  KVCacheConfig kv_cache_config_;

  int long_prefill_token_threshold_;
  bool enable_chunked_prefill_;
  bool scheduler_reserve_full_isl_;
  // KV-cache block size (tokens per block). Held so SW2's in-batch collision
  // check can convert its block-granular match into tokens for the threshold
  // comparison (see maybe_reorder_waiting_for_lpm).
  int block_size_ = 0;
  bool align_recurrent_prefix_publication_ = false;
  RecurrentPrefixAlignmentMode recurrent_prefix_alignment_mode_ =
      RecurrentPrefixAlignmentMode::kAuto;
  // max_num_encoder_input_tokens (scheduler.py:226-228): the PER-STEP encoder
  // compute budget in encoder tokens, reset at the top of every schedule().
  int max_num_encoder_input_tokens_ = 0;
  // disable_chunked_mm_input (scheduler_config.disable_chunked_mm_input, read at
  // scheduler.py:1470): when set, a step may not cover only PART of a
  // multimodal item's placeholder span — the admission rolls back to before the
  // item instead. Default false, which is upstream's default and the arm the
  // budget/cache fallback above covers.
  bool disable_chunked_mm_input_ = false;
  // num_lookahead_tokens (scheduler.py:275-292): reserved verify slots per running
  // request. 0 when no SpeculativeConfig is supplied (default), else derived from
  // it in the ctor via SpeculativeConfig::NumLookaheadTokens (k for MTP). Threaded
  // into allocate_slots (schedule() lines below).
  int num_lookahead_tokens_ = 0;
  // num_spec_tokens (scheduler.py:241 `self.num_spec_tokens =
  // vllm_config.num_speculative_tokens`): the resolved per-step draft count.
  // 0 when no SpeculativeConfig is supplied — which keeps
  // SchedulerOutput::num_spec_tokens_to_schedule 0 and the AsyncScheduler's
  // placeholder assignment inert on the default path. SPEC-DFLASH2 W7 (#1824).
  int num_spec_tokens_ = 0;
  // log_stats (scheduler.py: `self.log_stats = ... not disable_log_stats`,
  // default True): gate for recording the per-request QUEUED/SCHEDULED/PREEMPTED
  // engine-core events. ON by default — upstream disable_log_stats defaults
  // False, and (as with the KVCacheManager log_stats above) the observability
  // surface is always-on; the events are inert unless a frontend stat logger
  // consumes them. When off, no event is ever appended, so the request path is
  // byte-identical.
  bool log_stats_ = true;
  // T0: 1 (not a diffusion model).
  int num_sampled_tokens_per_step_ = 1;
  // Scheduler iteration counter (upstream current_step; inert at T0 since
  // next_decode_eligible_step is deferred/0).
  int current_step_ = 0;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_SCHEDULER_H_
