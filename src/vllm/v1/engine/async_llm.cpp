// Ported from: vllm/v1/engine/async_llm.py @ e24d1b24
// (add_request :280-410, generate :524-635, _run_output_handler :637-707,
// abort :709-745). See async_llm.h for scope and the in-process deviation.
#include "vllm/v1/engine/async_llm.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

#include "vllm/v1/engine/parallel_sampling.h"
#include "vllm/v1/metrics/loggers.h"  // PrometheusStatLogger::Record
#include "vllm/v1/metrics/stats.h"    // IterationStats
#include "vllm/v1/request.h"

namespace vllm::v1 {

AsyncLLM::AsyncLLM(InputProcessor& input_processor, Scheduler& scheduler,
                   Executor& executor, OutputProcessor& output_processor,
                   BlockHasher block_hasher, int shutdown_timeout_s,
                   int max_concurrent_batches,
                   StructuredOutputManager* structured_output_manager,
                   bool check_for_draft_tokens)
    : input_processor_(input_processor),
      output_processor_(output_processor),
      block_hasher_(std::move(block_hasher)),
      engine_core_(scheduler, executor, structured_output_manager,
                   max_concurrent_batches, shutdown_timeout_s,
                   check_for_draft_tokens),
      output_handler_() {
  // Start only after every member (including the stop flags) is constructed;
  // the thread blocks on get_output until the first request arrives.
  output_handler_ = std::thread(&AsyncLLM::RunOutputHandler, this);
}

AsyncLLM::~AsyncLLM() { shutdown(); }

void AsyncLLM::set_stat_logger(metrics::PrometheusStatLogger* logger) {
  std::lock_guard<std::mutex> lock(stat_logger_mutex_);
  stat_logger_ = logger;
}

AsyncRequest AsyncLLM::add_request(const std::string& request_id,
                                   const std::string& prompt,
                                   SamplingParams params, int priority) {
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request submitted to a stopped AsyncLLM");
  }

  // async_llm.py:326-350: InputProcessor first, then one collector keyed by
  // the internal request id. T0 has no request-id remap/parallel sampling.
  EngineCoreRequest request = input_processor_.process_inputs(
      request_id, prompt, std::move(params), /*arrival_time=*/std::nullopt,
      priority);
  auto collector = std::make_shared<RequestOutputCollector>(
      request.sampling_params.output_kind, request.request_id);

  // async_llm.py:382-399 parallel-sampling fan-out. n == 1 falls through to the
  // single-sequence path below (byte-identical); n > 1 expands into n children
  // that share this one collector and each COPY the prompt tokens.
  // `EngineCoreRequest child = request` is a deep copy here, not upstream's
  // shallow `copy(request)` (async_llm.py:393), because
  // EngineCoreRequest::prompt_token_ids is a std::vector<int32_t> BY VALUE
  // (types.h:79) — O(n * prompt_len). Tracked as #2145; the cheap mirror is a
  // shared immutable token buffer, which is not a change this row makes.
  if (request.sampling_params.n > 1) {
    return PublishParallelSampling(request, prompt, std::move(collector));
  }

  // Build the core Request before publishing the OutputProcessor state so an
  // allocation/factory failure cannot leave a consumer waiting forever.
  auto core_request = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));

  // async_llm.py:_add_request :400-410: OutputProcessor registration MUST
  // precede the EngineCore enqueue; an immediately produced frame then always
  // finds its request state/collector.
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    // Serialize the final admission check, frontend registration, and core
    // enqueue with shutdown's abort-all transition. A submitter may have
    // passed the fast check above while processing inputs; it must not publish
    // a new collector after shutdown has already swept the request table.
    if (shutdown_started_.load() || errored_.load() ||
        engine_core_.engine_dead()) {
      throw EngineDeadError("request submitted to a stopped AsyncLLM");
    }
    output_processor_.add_request(request, prompt, /*request_index=*/0,
                                  collector);
    try {
      engine_core_.add_request_async(std::move(core_request));
    } catch (...) {
      // Queue allocation is the only expected failure here. Roll back the
      // frontend state so the id remains reusable and no collector hangs.
      (void)output_processor_.abort_requests({request.request_id});
      throw;
    }
  }

  return AsyncRequest{request.request_id, std::move(collector)};
}

AsyncRequest AsyncLLM::add_request(const std::string& request_id,
                                   std::vector<int32_t> prompt_token_ids,
                                   SamplingParams params, int priority) {
  // TokensPrompt path: identical to the string add_request except the request is
  // built from prompt_token_ids directly (no tokenization) and the OutputProcessor
  // gets no prompt string (std::nullopt). See llm_engine.cpp's mirror overload.
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request submitted to a stopped AsyncLLM");
  }

  EngineCoreRequest request = input_processor_.process_inputs_tokens(
      request_id, std::move(prompt_token_ids), std::move(params),
      /*arrival_time=*/std::nullopt, priority);
  auto collector = std::make_shared<RequestOutputCollector>(
      request.sampling_params.output_kind, request.request_id);

  // Parallel-sampling fan-out (see the string overload).
  if (request.sampling_params.n > 1) {
    return PublishParallelSampling(request, /*prompt=*/std::nullopt,
                                   std::move(collector));
  }

  auto core_request = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));

  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    if (shutdown_started_.load() || errored_.load() ||
        engine_core_.engine_dead()) {
      throw EngineDeadError("request submitted to a stopped AsyncLLM");
    }
    output_processor_.add_request(request, /*prompt=*/std::nullopt,
                                  /*request_index=*/0, collector);
    try {
      engine_core_.add_request_async(std::move(core_request));
    } catch (...) {
      (void)output_processor_.abort_requests({request.request_id});
      throw;
    }
  }

  return AsyncRequest{request.request_id, std::move(collector)};
}

std::vector<AsyncRequest> AsyncLLM::add_request_wave(
    std::vector<AsyncStringRequestInput> requests) {
  if (requests.empty()) return {};
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request wave submitted to a stopped AsyncLLM");
  }

  std::set<std::string> request_ids;
  for (const AsyncStringRequestInput& input : requests) {
    if (!request_ids.insert(input.request_id).second) {
      throw std::invalid_argument("duplicate request id in wave: " +
                                  input.request_id);
    }
  }

  std::vector<PreparedRequest> prepared;
  prepared.reserve(requests.size());
  for (AsyncStringRequestInput& input : requests) {
    EngineCoreRequest request = input_processor_.process_inputs(
        input.request_id, input.prompt, std::move(input.params),
        /*arrival_time=*/std::nullopt, input.priority);
    auto collector = std::make_shared<RequestOutputCollector>(
        request.sampling_params.output_kind, request.request_id);
    auto core_request = std::make_unique<Request>(
        Request::FromEngineCoreRequest(request, block_hasher_));
    prepared.push_back(PreparedRequest{
        std::move(request), std::move(input.prompt), std::move(collector),
        std::move(core_request)});
  }
  return PublishPreparedWave(std::move(prepared));
}

std::vector<AsyncRequest> AsyncLLM::add_request_wave(
    std::vector<AsyncTokensRequestInput> requests) {
  if (requests.empty()) return {};
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request wave submitted to a stopped AsyncLLM");
  }

  std::set<std::string> request_ids;
  for (const AsyncTokensRequestInput& input : requests) {
    if (!request_ids.insert(input.request_id).second) {
      throw std::invalid_argument("duplicate request id in wave: " +
                                  input.request_id);
    }
  }

  std::vector<PreparedRequest> prepared;
  prepared.reserve(requests.size());
  for (AsyncTokensRequestInput& input : requests) {
    EngineCoreRequest request = input_processor_.process_inputs_tokens(
        input.request_id, std::move(input.prompt_token_ids),
        std::move(input.params), /*arrival_time=*/std::nullopt, input.priority);
    auto collector = std::make_shared<RequestOutputCollector>(
        request.sampling_params.output_kind, request.request_id);
    auto core_request = std::make_unique<Request>(
        Request::FromEngineCoreRequest(request, block_hasher_));
    prepared.push_back(PreparedRequest{
        std::move(request), std::nullopt, std::move(collector),
        std::move(core_request)});
  }
  return PublishPreparedWave(std::move(prepared));
}

std::vector<AsyncRequest> AsyncLLM::PublishPreparedWave(
    std::vector<PreparedRequest> prepared) {
  std::vector<AsyncRequest> result;
  std::vector<std::string> rollback_ids;
  std::vector<std::unique_ptr<Request>> core_requests;
  result.reserve(prepared.size());
  rollback_ids.reserve(prepared.size());
  core_requests.reserve(prepared.size());
  for (PreparedRequest& item : prepared) {
    result.push_back(AsyncRequest{item.request.request_id, item.collector});
    rollback_ids.push_back(item.request.request_id);
    core_requests.push_back(std::move(item.core_request));
  }

  return PublishAsyncRequestWaveIfAlive(
      output_processor_mutex_,
      [&]() {
        return !shutdown_started_.load() && !errored_.load() &&
               !engine_core_.engine_dead();
      },
      [&]() -> std::vector<AsyncRequest> {
        // Reject every collision before creating the first new frontend state.
        // This keeps a colliding pre-existing request outside the rollback set.
        for (const PreparedRequest& item : prepared) {
          if (output_processor_.has_request(item.request.request_id)) {
            throw std::invalid_argument("duplicate live request id: " +
                                        item.request.request_id);
          }
        }

        std::size_t rollback_count = 0;
        try {
          for (std::size_t i = 0; i < prepared.size(); ++i) {
            // Include the current id in rollback before registration:
            // add_request may have inserted one of its maps before a later
            // allocation fails.
            rollback_count = i + 1;
            PreparedRequest& item = prepared[i];
            output_processor_.add_request(item.request, item.prompt,
                                          /*request_index=*/0, item.collector);
          }
          engine_core_.add_requests_async(std::move(core_requests));
        } catch (...) {
          rollback_ids.resize(rollback_count);
          output_processor_.rollback_requests(rollback_ids);
          throw;
        }
        return std::move(result);
      });
}

AsyncRequest AsyncLLM::add_request(const std::string& request_id,
                                   multimodal::MultiModalInputs mm_inputs,
                                   SamplingParams params, int priority) {
  // Multimodal path: identical to the tokens add_request except the request is
  // built via process_inputs_mm (placeholder-EXPANDED ids + mm_features carried
  // onto the EngineCoreRequest / Request). The OutputProcessor gets no prompt
  // string. See llm_engine.cpp's mirror mm overload.
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request submitted to a stopped AsyncLLM");
  }

  EngineCoreRequest request = input_processor_.process_inputs_mm(
      request_id, std::move(mm_inputs.prompt_token_ids),
      std::move(mm_inputs.mm_features), std::move(params),
      /*arrival_time=*/std::nullopt, priority);
  auto collector = std::make_shared<RequestOutputCollector>(
      request.sampling_params.output_kind, request.request_id);

  // Parallel-sampling fan-out: each child COPIES the parent EngineCoreRequest,
  // mm_features included. What is and is not shared, exactly: the ENCODER
  // PAYLOAD is shared, because MultiModalFeatureSpec holds it behind
  // std::shared_ptr<ImageKwargs>/<AudioKwargs> (multimodal/inputs.h:80-81) and
  // the copy only bumps the refcount. The spec VECTOR and its two strings per
  // spec are copied per child (types.h:93), as is prompt_token_ids — and on
  // this path that is the placeholder-EXPANDED prompt, so #2145's per-child
  // prompt copy costs MORE here than on the text path, not less.
  if (request.sampling_params.n > 1) {
    return PublishParallelSampling(request, /*prompt=*/std::nullopt,
                                   std::move(collector));
  }

  auto core_request = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));

  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    if (shutdown_started_.load() || errored_.load() ||
        engine_core_.engine_dead()) {
      throw EngineDeadError("request submitted to a stopped AsyncLLM");
    }
    output_processor_.add_request(request, /*prompt=*/std::nullopt,
                                  /*request_index=*/0, collector);
    try {
      engine_core_.add_request_async(std::move(core_request));
    } catch (...) {
      (void)output_processor_.abort_requests({request.request_id});
      throw;
    }
  }

  return AsyncRequest{request.request_id, std::move(collector)};
}

AsyncRequest AsyncLLM::PublishParallelSampling(
    const EngineCoreRequest& request, std::optional<std::string> prompt,
    std::shared_ptr<RequestOutputCollector> collector) {
  // async_llm.py:386-399. One shared ParentRequest, n children named
  // "{idx}_{parent}" with n == 1 sampling params (seed + idx when the parent is
  // seeded), all registered against the ONE collector this request hands back.
  const std::string parent_id = request.request_id;
  auto parent = std::make_shared<ParentRequest>(request);
  const int n = request.sampling_params.n;

  // Build every child and its core Request BEFORE taking the lock, mirroring
  // the single-sequence path's ordering rule: nothing that can throw runs
  // between the first frontend registration and the core enqueue.
  std::vector<EngineCoreRequest> children;
  std::vector<std::unique_ptr<Request>> core_requests;
  children.reserve(static_cast<std::size_t>(n));
  core_requests.reserve(static_cast<std::size_t>(n));
  for (int idx = 0; idx < n; ++idx) {
    std::pair<std::string, SamplingParams> child_info =
        parent->get_child_info(idx);
    // A DEEP copy: EngineCoreRequest holds prompt_token_ids by value
    // (types.h:79), so each child owns its own copy of the prompt, and
    // FromEngineCoreRequest below makes a second one per child. Upstream's
    // `copy(request)` (:393) is SHALLOW — the n children share one Python list
    // and the last child reuses the parent object outright — so we move
    // O(n * prompt_len) bytes where upstream moves none. Deliberate here: the
    // cheap mirror needs a shared immutable token buffer on EngineCoreRequest,
    // which every engine path reads. Owed, #2145.
    EngineCoreRequest child = request;
    child.request_id = std::move(child_info.first);
    child.sampling_params = std::move(child_info.second);
    core_requests.push_back(std::make_unique<Request>(
        Request::FromEngineCoreRequest(child, block_hasher_)));
    children.push_back(std::move(child));
  }

  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    // The same admission check the single-sequence overloads make under this
    // lock: a submitter that passed the fast check must not publish after
    // shutdown has already swept the request table.
    if (shutdown_started_.load() || errored_.load() ||
        engine_core_.engine_dead()) {
      throw EngineDeadError("request submitted to a stopped AsyncLLM");
    }
    // Reject every collision BEFORE creating the first new frontend state, the
    // same guard PublishPreparedWave makes above. add_request throws on a
    // "duplicate live request id", and the rollback below erases by id — so
    // without this pre-check a colliding child id would put the PRE-EXISTING
    // request into the rollback set and destroy state its own caller still
    // holds. Unreachable over HTTP today (the serving layer mints the parent
    // id), but the fan-out is also an ABI seam where the caller names the id.
    for (const EngineCoreRequest& child : children) {
      if (output_processor_.has_request(child.request_id)) {
        throw std::invalid_argument("duplicate live request id: " +
                                    child.request_id);
      }
    }

    std::vector<std::string> registered;
    registered.reserve(children.size());
    try {
      // :401-414 _add_request: OutputProcessor registration precedes the
      // EngineCore enqueue, so an immediately produced frame finds its state.
      for (std::size_t idx = 0; idx < children.size(); ++idx) {
        // Include the id BEFORE registering: add_request can insert one of its
        // maps and then throw.
        registered.push_back(children[idx].request_id);
        output_processor_.add_request(children[idx], prompt,
                                      /*request_index=*/static_cast<int>(idx),
                                      collector, parent);
      }
      // DEVIATION from upstream's per-child `await engine_core.add_request_async`
      // (:413): one batched enqueue. A mid-loop enqueue failure would otherwise
      // leave earlier children running in EngineCore with no frontend state,
      // which OutputProcessor::process_outputs ignores forever
      // (output_processor.cpp:401-405) — the request would burn KV blocks until
      // shutdown. The batch keeps the admission a single transaction, as
      // PublishPreparedWave already does.
      engine_core_.add_requests_async(std::move(core_requests));
    } catch (...) {
      // Roll the whole parent back: every registered child and the parent entry
      // it published, without manufacturing terminal output for a request the
      // caller never saw.
      output_processor_.rollback_requests(registered);
      throw;
    }
  }

  // The PARENT id, not a child's: it is what RequestOutput::request_id carries
  // (RequestState::make_request_output reads parent_req->external_req_id()) and
  // what abort() must be given to reach all n children.
  return AsyncRequest{parent_id, std::move(collector)};
}

RequestOutput AsyncLLM::get_output(const AsyncRequest& request) {
  if (request.collector == nullptr) {
    throw std::invalid_argument("AsyncLLM request has no collector");
  }
  return request.collector->get();
}

std::optional<RequestOutput> AsyncLLM::get_output_nowait(
    const AsyncRequest& request) {
  if (request.collector == nullptr) {
    throw std::invalid_argument("AsyncLLM request has no collector");
  }
  return request.collector->get_nowait();
}

std::optional<RequestOutput> AsyncLLM::get_output_for(
    const AsyncRequest& request, std::chrono::milliseconds timeout) {
  if (request.collector == nullptr) {
    throw std::invalid_argument("AsyncLLM request has no collector");
  }
  return request.collector->get_for(timeout);
}

RequestOutput AsyncLLM::generate(const std::string& prompt,
                                 SamplingParams params,
                                 const std::string& request_id, int priority) {
  AsyncRequest request =
      add_request(request_id, prompt, std::move(params), priority);
  try {
    for (;;) {
      // async_llm.py:572 drains the single-slot collector without blocking
      // when the output handler has already produced a value.
      std::optional<RequestOutput> ready = get_output_nowait(request);
      RequestOutput output = ready.has_value()
                                 ? std::move(*ready)
                                 : get_output(request);
      if (output.finished) return output;
    }
  } catch (...) {
    abort(request.request_id);
    throw;
  }
}

RequestOutput AsyncLLM::generate(std::vector<int32_t> prompt_token_ids,
                                 SamplingParams params,
                                 const std::string& request_id, int priority) {
  AsyncRequest request = add_request(request_id, std::move(prompt_token_ids),
                                     std::move(params), priority);
  try {
    for (;;) {
      std::optional<RequestOutput> ready = get_output_nowait(request);
      RequestOutput output = ready.has_value()
                                 ? std::move(*ready)
                                 : get_output(request);
      if (output.finished) return output;
    }
  } catch (...) {
    abort(request.request_id);
    throw;
  }
}

RequestOutput AsyncLLM::generate(multimodal::MultiModalInputs mm_inputs,
                                 SamplingParams params,
                                 const std::string& request_id, int priority) {
  // Multimodal single-request driver (mirrors the tokens generate loop). The
  // mm forward on the GPU worker consumes the carried mm_features (MM-SERVE-E2E).
  AsyncRequest request = add_request(request_id, std::move(mm_inputs),
                                     std::move(params), priority);
  try {
    for (;;) {
      std::optional<RequestOutput> ready = get_output_nowait(request);
      RequestOutput output = ready.has_value()
                                 ? std::move(*ready)
                                 : get_output(request);
      if (output.finished) return output;
    }
  } catch (...) {
    abort(request.request_id);
    throw;
  }
}

void AsyncLLM::abort(const std::string& request_id) {
  abort(std::vector<std::string>{request_id});
}

void AsyncLLM::abort(const std::vector<std::string>& request_ids) {
  std::vector<std::string> core_request_ids;
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    core_request_ids = output_processor_.abort_requests(
        request_ids, /*produce_final_output=*/true);
  }
  engine_core_.abort_requests_async(core_request_ids);
}

int AsyncLLM::get_num_unfinished_requests() const {
  std::lock_guard<std::mutex> lock(output_processor_mutex_);
  return output_processor_.get_num_unfinished_requests();
}

void AsyncLLM::RunOutputHandler() {
  try {
    for (;;) {
      // async_llm.py:651: one blocking EngineCoreOutputs pull.
      EngineCoreOutputs outputs = engine_core_.get_output();

      // async_llm.py:648-652 logger_ref[0] — read ONCE per iteration so the
      // build decision and the fold below cannot disagree if the logger is
      // detached mid-step.
      // Keep the attachment stable through the complete stats fold. In
      // particular, a terminal output is visible to generate() before Record
      // below finishes; holding this lock makes a concurrent detach wait until
      // the last possible use of the old non-owning pointer has retired.
      std::unique_lock<std::mutex> logger_lock(stat_logger_mutex_);
      metrics::PrometheusStatLogger* logger = stat_logger_;

      // async_llm.py:664-665
      //   iteration_stats = IterationStats() if (log_stats and num_outputs)
      // — `log_stats` is "a logger is attached" here, exactly as the sync site
      // reads it (llm_engine.cpp:190-193).
      //
      // DIAGNOSTIC (VT_TTFT_DUMP): keeps its own independent trigger, so a
      // no-logger run with the env unset is instruction-identical to before
      // this row (process_outputs takes its byte-identical no-stats path).
      static const bool kTrackAsyncStats =
          std::getenv("VT_TTFT_DUMP") != nullptr;
      const bool track_stats =
          (logger != nullptr && !outputs.outputs.empty()) || kTrackAsyncStats;

      IterationStats iteration_stats;
      OutputProcessorOutput processed;
      {
        std::lock_guard<std::mutex> lock(output_processor_mutex_);
        // RequestOutputs are pushed to their collectors by OutputProcessor;
        // the synchronous return list must therefore stay empty.
        //
        // async_llm.py:676-678 process_outputs(outputs_slice, outputs.timestamp,
        // iteration_stats). Our process_outputs reads the engine-core timestamp
        // off the EngineCoreOutputs itself (output_processor.cpp:367).
        if (track_stats) {
          processed = output_processor_.process_outputs(outputs,
                                                        &iteration_stats);
        } else {
          processed = output_processor_.process_outputs(outputs);
        }
      }
      // Stop-string finishes detected by the detokenizer must be reflected in
      // EngineCore after leaving the OutputProcessor critical section.
      engine_core_.abort_requests_async(processed.reqs_to_abort);

      // async_llm.py:697-702 — fold this step's SchedulerStats + IterationStats
      // into the registry. A stats-only prefix lookup from a tokenless prefill
      // chunk must also reach the logger; its empty IterationStats contributes
      // no token or completion timing sample.
      //
      // Deliberately OUTSIDE output_processor_mutex_: the logger's own mutex is
      // then a leaf lock that can never take part in a cycle with the
      // output-processor lock or a collector's condition variable.
      if (logger != nullptr &&
          (!outputs.outputs.empty() || outputs.scheduler_stats.prefix_cache_stats.queries > 0)) {
        logger->Record(outputs.scheduler_stats, iteration_stats);
      }
    }
  } catch (...) {
    if (!stopping_.load()) {
      // Restore the upstream fatal log (vllm/v1/engine/async_llm.py:703-705:
      // logger.exception("AsyncLLM output_handler failed.") before
      // output_processor.propagate_error(e)). std::cerr only, so the witness
      // survives a SIGKILL escalation.
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception& e) {
        std::cerr << "async-llm: output handler saw engine death: " << e.what()
                  << "\n";
      } catch (...) {
      }
      errored_.store(true);
      std::lock_guard<std::mutex> lock(output_processor_mutex_);
      output_processor_.propagate_error(std::current_exception());
    }
  }
}

void AsyncLLM::shutdown() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true)) return;

  stopping_.store(true);
  // An ADD can still be queued when shutdown wins the race before the engine
  // thread's first busy-loop iteration. EngineCoreProc then observes shutdown
  // with an empty Scheduler and exits without consuming that ADD, so relying
  // only on core abort outputs would strand its collector. Abort the frontend
  // states first: every consumer is woken exactly once regardless of whether
  // the core has admitted its request yet. The queued core abort remains a
  // harmless no-op for not-yet-admitted/already-finished IDs.
  std::vector<std::string> core_request_ids;
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    core_request_ids = output_processor_.abort_all_requests(
        /*produce_final_output=*/true);
  }
  engine_core_.abort_requests_async(core_request_ids);

  // EngineCoreProc abort/drain semantics enqueue all terminal frames before
  // its thread exits for already-admitted requests. Once joined, append a
  // sentinel to wake the output handler after it drains any residual frames.
  engine_core_.shutdown();
  engine_core_.proc().send_engine_dead("AsyncLLM shutdown");
  if (output_handler_.joinable()) output_handler_.join();
}

}  // namespace vllm::v1
