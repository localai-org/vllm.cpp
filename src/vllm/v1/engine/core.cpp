// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (add_request, abort_requests, step — the T0 subset). See core.h for scope,
// deviations and deferrals.
#include "vllm/v1/engine/core.h"

#include <cstdio>
#include <cstdlib>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "vllm/v1/core/sched/output.h"          // GrammarOutput
#include "vllm/v1/structured_output/manager.h"  // StructuredOutputManager

namespace vllm::v1 {

void EngineCore::add_request(std::unique_ptr<Request> request) {
  // core.py:870-876 (preprocess_add_request): compile the request's grammar
  // before scheduling it. Upstream runs this in the input-processing thread; at
  // T0 it is synchronous here. No-op for a non-structured request or when no
  // manager is wired.
  if (structured_output_manager_ != nullptr &&
      request->use_structured_output()) {
    structured_output_manager_->grammar_init(*request);
  }
  // core.py:403 self.scheduler.add_request(request). The upstream request_id
  // type / pooling-task / kv_transfer validation and the abort_immediately
  // hook are deferred (see the file header).
  scheduler_.add_request(std::move(request));
}

void EngineCore::abort_requests(const std::vector<std::string>& request_ids) {
  // core.py:415 self.scheduler.finish_requests(request_ids,
  // RequestStatus.FINISHED_ABORTED). Our finish_requests takes one id, so
  // iterate (a no-op for unknown / already-finished ids, matching upstream).
  for (const std::string& request_id : request_ids) {
    scheduler_.finish_requests(request_id, RequestStatus::kFinishedAborted);
  }
}

std::pair<std::map<int, EngineCoreOutputs>, bool> EngineCore::step() {
  // core.py:488 if not self.scheduler.has_requests(): return {}, False
  // Our Scheduler has no has_requests(); inline the interface.py default
  // (connector pending-push term deferred — no connector at T0).
  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (!has_requests) {
    return {{}, false};
  }

  // core.py:490 scheduler_output = self.scheduler.schedule(...)
  // (the _should_throttle_prefills() arg is deferred — DP prefill balancing).
  SchedulerOutput scheduler_output = scheduler_.schedule();

  // core.py:491-499 execute the forward, then sample. The MRV2 runner's
  // execute_model returns None ("forward done"), so we always call sample_tokens.
  std::optional<ModelRunnerOutput> model_output =
      executor_.execute_model(scheduler_output);
  // Chunked-prefill progress, AFTER the forward so its elapsed_s is real wall
  // time rather than the cost of scheduling. No-op unless
  // VT_SERVER_PREFILL_PROGRESS / VT_SERVER_VERBOSE is on.
  scheduler_.LogPrefillAfterExecute(scheduler_output);
  // core.py:492 grammar_output = self.scheduler.get_grammar_bitmask(...). Nullopt
  // when no structured request is scheduled (or no manager is wired); threaded to
  // sample_tokens (Task 3 consumes it).
  const std::optional<GrammarOutput> grammar_output =
      scheduler_.get_grammar_bitmask(scheduler_output);
  if (!model_output.has_value()) {
    model_output = executor_.sample_tokens(grammar_output);
  }

  // core.py:503 self._process_aborts_queue() — deferred (no in-flight aborts at
  // T0; synchronous execution leaves no window).

  // core.py:504-506 engine_core_outputs = scheduler.update_from_output(...).
  // Our update_from_output returns a single flat EngineCoreOutputs (T0 single
  // client); wrap it in the per-client map to keep the upstream return shape.
  // A chunked prefill can perform a prefix lookup without producing tokens.
  // Keep that stats-only observation so the frontend does not lose its query
  // and hit counters. Finished-request-only entries remain deferred.
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(scheduler_output, *model_output);
  // Attach this step's scheduler snapshot + the engine_core_timestamp the
  // frontend threads into IterationStats for TTFT/ITL (core.py builds
  // EngineCoreOutputs(scheduler_stats=scheduler.make_stats(), timestamp=...)).
  // Inert on the no-logger path (the frontend never reads it).
  engine_core_outputs.scheduler_stats = scheduler_.make_stats();
  engine_core_outputs.timestamp = MonotonicSeconds();
  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty() ||
      engine_core_outputs.scheduler_stats.prefix_cache_stats.queries > 0) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }

  // core.py:509-517 post_step: feed the drafter's out-of-band proposal back to
  // the scheduler for the next step. Inert unless a speculator is configured.
  const bool model_executed = scheduler_output.total_num_scheduled_tokens > 0;
  post_step(model_executed);

  // core.py:508 return outputs, scheduler_output.total_num_scheduled_tokens > 0.
  return {std::move(outputs_by_client), model_executed};
}

void EngineCore::post_step(bool model_executed) {
  // core.py:509-517 (:617 at 555967922). Under async scheduling the draft
  // token ids are updated in the worker process instead (SPEC-DFLASH2 W7,
  // #1824): the AsyncScheduler ships -1 placeholders and the runner fills them
  // from its own propose, so the out-of-band pull below must NOT run — it
  // would overwrite the placeholders with values the scheduler must never
  // carry under async. The guard cannot be "which step function called me":
  // EngineCoreProc's busy loop AND the depth-1 LLMEngine::step both reach
  // here whatever the resolution, so it mirrors upstream's
  // `not self.async_scheduling` (the scheduler class IS the resolved value,
  // model_loader.cpp::MakeScheduler).
  if (!check_for_draft_tokens_ || scheduler_.async_scheduling() ||
      !model_executed) {
    return;
  }
  std::optional<DraftTokenIds> draft_token_ids = executor_.take_draft_token_ids();
  static const bool spec_post_trace = std::getenv("VT_SPEC_TRACE") != nullptr;
  if (spec_post_trace) {
    std::fprintf(stderr, "[spec-post_step] pulled=%d rows=%zu\n",
                 draft_token_ids.has_value() ? 1 : 0,
                 draft_token_ids.has_value() ? draft_token_ids->req_ids.size() : 0);
  }
  if (draft_token_ids.has_value()) {
    scheduler_.update_draft_token_ids(*draft_token_ids);
  }
}

std::pair<std::map<int, EngineCoreOutputs>, bool>
EngineCore::step_with_batch_queue() {
  // core.py:519-632. Fulfilling the batch queue has priority over consuming an
  // output: schedule + execute a new batch (if room + work), then block on the
  // OLDEST queued batch. Our executor resolves eagerly, so "execute" here means
  // the forward + sample ran synchronously and the ModelRunnerOutput is already
  // in hand (the AsyncScheduler placeholder accounting is what lets step N+1 be
  // scheduled before N's output is consumed — see core.h).
  assert(batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
         "step_with_batch_queue called with a full batch queue");

  bool model_executed = false;
  std::optional<SchedulerOutput> deferred_scheduler_output;

  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (has_requests) {
    // core.py:547 schedule (non-blocking; may return an empty batch).
    SchedulerOutput scheduler_output = scheduler_.schedule();
    // core.py:549-551 execute_model(non_block=True). MRV2: nullopt (forward
    // done). A failed eager forward throws here, through the engine-fatal guard.
    std::optional<ModelRunnerOutput> exec_out =
        executor_.execute_model(scheduler_output);
    // Same diagnostic as the synchronous path above.
    scheduler_.LogPrefillAfterExecute(scheduler_output);
    // core.py:552-553 model_executed = total_num_scheduled_tokens > 0
    // (is_ec_consumer is always true for us — no EC transfer).
    model_executed = scheduler_output.total_num_scheduled_tokens > 0;

    std::unique_ptr<AsyncModelRunnerOutput> async_out;
    if (!model_executed) {
      // core.py:555-557 no sampling required — carry the (empty) forward result
      // as an already-materialized async output.
      async_out = std::make_unique<ReadyModelRunnerOutput>(
          exec_out.value_or(ModelRunnerOutput{}));
    } else if (!scheduler_output.pending_structured_output_tokens) {
      // core.py:559-567 not waiting on any tokens — get the grammar bitmask and
      // sample immediately. sample_tokens_async issues the sampled-id copy but,
      // under async scheduling, DEFERS the host wait to get_output() at consume
      // time (below), so the copy overlaps the next step's forward; a synchronous
      // runner returns an already-materialized ReadyModelRunnerOutput.
      const std::optional<GrammarOutput> grammar_output =
          scheduler_.get_grammar_bitmask(scheduler_output);
      async_out = executor_.sample_tokens_async(grammar_output);
    } else {
      // core.py:568-571 defer sampling until the prior step's output is
      // processed (a structured request is waiting on in-flight tokens).
      deferred_scheduler_output = scheduler_output;
    }

    if (!deferred_scheduler_output.has_value()) {
      // core.py:573-575 add this step's result to the queue (front).
      assert(async_out != nullptr);
      batch_queue_.push_front(
          BatchQueueItem{std::move(async_out), std::move(scheduler_output)});
      // core.py:576-581 don't block on the next result unless the queue is full
      // or there are no more requests to schedule.
      const bool more_requests =
          scheduler_.get_num_unfinished_requests() > 0 ||
          scheduler_.has_finished_requests();
      if (batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
          (model_executed || more_requests)) {
        return {{}, model_executed};
      }
    }
  } else if (batch_queue_.empty()) {
    // core.py:583-587 queue empty and no requests — nothing to do.
    return {{}, false};
  }

  // core.py:589-590 block until the next result is available (pop the OLDEST).
  BatchQueueItem item = std::move(batch_queue_.back());
  batch_queue_.pop_back();

  // core.py:589-590 future.result() → get_output(): resolve the (possibly
  // deferred) sampled-id copy NOW — the ONE host block, and it is off the model's
  // critical path because it overlapped the batch we just scheduled+forwarded
  // above. A synchronous runner's output is already materialized (no wait).
  ModelRunnerOutput model_output = item.async_output->get_output();

  // core.py:604-607 update the scheduler from the popped batch's output.
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(item.scheduler_output, model_output);

  // core.py:609-630 grammar deferral: now that the prior output is processed,
  // compute the deferred batch's bitmask, sample it, and append it to the queue.
  // (The runner's execute_model stash from this step is still valid — no other
  // execute_model ran between it and here.)
  if (deferred_scheduler_output.has_value()) {
    // core.py:718-731 (SPEC-DFLASH2 W7, #1824): with drafts under async
    // scheduling, the deferred batch's scheduled_spec_decode_tokens still
    // holds the -1 placeholders (the worker fill patches only its own copy).
    // Pull the worker's real drafts and rewrite them into the deferred output
    // so the grammar bitmask below reads real token ids; a slot the worker
    // could not fill is -1-padded and recorded in num_invalid_spec_tokens for
    // the bitmask computation to skip. No-op without a speculator.
    if (check_for_draft_tokens_) {
      std::optional<DraftTokenIds> draft_token_ids =
          executor_.take_draft_token_ids();
      if (draft_token_ids.has_value()) {
        scheduler_.update_draft_token_ids_in_output(*draft_token_ids,
                                                    *deferred_scheduler_output);
      }
    }
    const std::optional<GrammarOutput> grammar_output =
        scheduler_.get_grammar_bitmask(*deferred_scheduler_output);
    std::unique_ptr<AsyncModelRunnerOutput> sampled =
        executor_.sample_tokens_async(grammar_output);
    batch_queue_.push_front(
        BatchQueueItem{std::move(sampled), std::move(*deferred_scheduler_output)});
  }

  // Attach this step's scheduler snapshot + engine-core timestamp, identically
  // to the synchronous step() above. Upstream stamps BOTH inside the path the
  // two step functions share — scheduler_stats in Scheduler.update_from_output
  // (scheduler.py:1938-1951) and timestamp in EngineCoreOutputs.__post_init__
  // (engine/__init__.py:249-251) — so upstream's step_with_batch_queue
  // (core.py:622-720) stamps nothing extra precisely because it already has
  // them.
  //
  // Until #277 this path stamped NEITHER: the async-scheduling serving stack
  // (LoadedEngine resolves max_concurrent_batches=2 whenever the runner
  // supports it) published a default-constructed SchedulerStats — every gauge
  // 0 — and a timestamp of 0.0, which turns every TTFT/e2e observation into
  // `-arrival_time`. The former VT_TTFT_DUMP-only timestamp stamp is subsumed
  // here; the diagnostic reads exactly the value it always did.
  engine_core_outputs.scheduler_stats = scheduler_.make_stats();
  engine_core_outputs.timestamp = MonotonicSeconds();

  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty() ||
      engine_core_outputs.scheduler_stats.prefix_cache_stats.queries > 0) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }
  // core.py:632 return engine_core_outputs, model_executed.
  return {std::move(outputs_by_client), model_executed};
}

}  // namespace vllm::v1
