// Ported from: vllm/v1/engine/llm_engine.py @ e24d1b24 (LLMEngine.add_request /
// step) + vllm/entrypoints/offline_utils.py::_run_engine @ e24d1b24 (the
// generate driver loop). See llm_engine.h for scope, wiring and deviations.
#include "vllm/v1/engine/llm_engine.h"

#include <memory>
#include <optional>
#include <utility>

#include "vllm/v1/engine/parallel_sampling.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/metrics/stats.h"
#include "vllm/v1/request.h"

namespace vllm::v1 {

LLMEngine::LLMEngine(InputProcessor& input_processor, EngineCore& engine_core,
                     OutputProcessor& output_processor, BlockHasher block_hasher)
    : input_processor_(input_processor),
      engine_core_(engine_core),
      output_processor_(output_processor),
      block_hasher_(std::move(block_hasher)) {}

std::string LLMEngine::add_request(const std::string& request_id,
                                   const std::string& prompt,
                                   SamplingParams params, int priority) {
  // llm_engine.py:250 request = self.input_processor.process_inputs(...). The
  // text path: validate (SamplingParams PostInit) + tokenize + build the message.
  EngineCoreRequest request = input_processor_.process_inputs(
      request_id, prompt, std::move(params), /*arrival_time=*/std::nullopt,
      priority);
  const std::string req_id = request.request_id;

  // llm_engine.py:278-293 parallel-sampling fan-out. n==1 falls through to the
  // original single-sequence path below (byte-identical); n>1 expands into n
  // child requests that COPY the prompt tokens (sharing only the prefill KV,
  // via prefix caching), each with its own decode state + RNG offset,
  // aggregated back into one RequestOutput by the OutputProcessor's
  // ParentRequest. The token COPY is per child because
  // EngineCoreRequest::prompt_token_ids is a std::vector<int32_t> by value
  // (types.h:79); upstream's copy(request) (llm_engine.py:283) is shallow.
  // Tracked as #2145.
  if (request.sampling_params.n > 1) {
    FanOutParallelSampling(request, prompt);
    return req_id;
  }

  // llm_engine.py:274 self.output_processor.add_request(request, prompt_text,
  // None, 0) — BEFORE engine_core.add_request. Register the RequestState (with
  // our incremental detokenizer) so process_outputs can detokenize its stream.
  output_processor_.add_request(request, prompt, /*request_index=*/0);

  // llm_engine.py:276 self.engine_core.add_request(request). Upstream hands the
  // EngineCoreRequest to core.py, which builds the Request via
  // from_engine_core_request(request, block_hasher); our EngineCore.add_request
  // takes the built Request, so construct it here with the engine's block_hasher.
  auto req = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));
  engine_core_.add_request(std::move(req));
  return req_id;
}

std::string LLMEngine::add_request(const std::string& request_id,
                                   std::vector<int32_t> prompt_token_ids,
                                   SamplingParams params, int priority) {
  // TokensPrompt path: build the request from prompt_token_ids directly (no
  // tokenization). Mirrors the string add_request step-for-step otherwise; the
  // output processor gets no prompt string (std::nullopt) since none was given.
  EngineCoreRequest request = input_processor_.process_inputs_tokens(
      request_id, std::move(prompt_token_ids), std::move(params),
      /*arrival_time=*/std::nullopt, priority);
  const std::string req_id = request.request_id;

  // Parallel-sampling fan-out (see the string overload). n==1 keeps the original
  // single-sequence path byte-identical.
  if (request.sampling_params.n > 1) {
    FanOutParallelSampling(request, /*prompt=*/std::nullopt);
    return req_id;
  }

  output_processor_.add_request(request, /*prompt=*/std::nullopt,
                                /*request_index=*/0);

  auto req = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));
  engine_core_.add_request(std::move(req));
  return req_id;
}

std::string LLMEngine::add_request(const std::string& request_id,
                                   multimodal::MultiModalInputs mm_inputs,
                                   SamplingParams params, int priority) {
  // Multimodal path: the MultiModalInputs already holds the placeholder-EXPANDED
  // prompt ids + the per-item mm_features (the serving-side processor ran). Build
  // the EngineCoreRequest via process_inputs_mm (no tokenization; carries
  // mm_features), then mirror the tokens overload step-for-step.
  EngineCoreRequest request = input_processor_.process_inputs_mm(
      request_id, std::move(mm_inputs.prompt_token_ids),
      std::move(mm_inputs.mm_features), std::move(params),
      /*arrival_time=*/std::nullopt, priority);
  const std::string req_id = request.request_id;

  // Parallel-sampling fan-out: each child copies the parent EngineCoreRequest,
  // mm_features included. The ENCODER PAYLOAD is genuinely shared — a
  // MultiModalFeatureSpec holds it behind std::shared_ptr<ImageKwargs>/
  // <AudioKwargs> (multimodal/inputs.h:80-81) and the copy bumps a refcount.
  // The spec VECTOR (types.h:93) and prompt_token_ids (types.h:79) are copied
  // per child; on this path prompt_token_ids is the placeholder-EXPANDED
  // prompt, so #2145's per-child copy costs more here than on the text path.
  if (request.sampling_params.n > 1) {
    FanOutParallelSampling(request, /*prompt=*/std::nullopt);
    return req_id;
  }

  output_processor_.add_request(request, /*prompt=*/std::nullopt,
                                /*request_index=*/0);

  auto req = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));
  engine_core_.add_request(std::move(req));
  return req_id;
}

std::string LLMEngine::add_pooling_request(const std::string& request_id,
                                           std::vector<int32_t> prompt_token_ids,
                                           PoolingParams pooling_params,
                                           int priority) {
  // ARCH-ONE-SURFACE ROW 6 — the POOLING-task add. Mirrors the tokens
  // add_request step-for-step; the SamplingParams are a benign greedy default
  // (temperature 0, max_tokens 1) because the InputBatch admit reads them but
  // the sampler is NEVER invoked on a pooling model's step (the runner routes
  // to pool_tokens, model_runner.py:1586-1607 mirror). The PoolingParams ride
  // the EngineCoreRequest into Request::pooling_params, which arms the
  // scheduler's pooling stop (scheduler.py:1718-1721).
  SamplingParams greedy;
  greedy.temperature = 0.0;
  greedy.max_tokens = 1;
  EngineCoreRequest request = input_processor_.process_inputs_tokens(
      request_id, std::move(prompt_token_ids), std::move(greedy),
      /*arrival_time=*/std::nullopt, priority);
  if (!pooling_params.task.has_value()) {
    pooling_params.task = PoolingTask::kEmbed;
  }
  if (!pooling_params.use_activation.has_value()) {
    pooling_params.use_activation = true;  // pooling_runner.py:38 F.normalize
  }
  request.pooling_params = std::move(pooling_params);
  const std::string req_id = request.request_id;

  output_processor_.add_request(request, /*prompt=*/std::nullopt,
                                /*request_index=*/0);

  auto req = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));
  engine_core_.add_request(std::move(req));
  return req_id;
}

void LLMEngine::FanOutParallelSampling(const EngineCoreRequest& request,
                                       std::optional<std::string> prompt) {
  // llm_engine.py:280-291. Build the shared ParentRequest, then register n child
  // requests: each child gets id "{idx}_{parent}", n==1 sampling params (seeded
  // children get seed+idx), the same prompt CONTENT (copied per child, not
  // shared — see the deep-copy note below and #2145), and the shared parent so
  // OutputProcessor aggregates the n child CompletionOutputs into one
  // RequestOutput. Output-processor add BEFORE engine-core add, mirroring the
  // single-sequence order (:274-276).
  auto parent = std::make_shared<ParentRequest>(request);
  const int n = request.sampling_params.n;
  for (int idx = 0; idx < n; ++idx) {
    std::pair<std::string, SamplingParams> child_info =
        parent->get_child_info(idx);
    // A DEEP copy: EngineCoreRequest holds prompt_token_ids by value
    // (types.h:79). Upstream's `copy(request)` (llm_engine.py:283) is SHALLOW.
    // The same asymmetry the async fan-out carries; see async_llm.cpp. #2145.
    EngineCoreRequest child = request;
    child.request_id = child_info.first;
    child.sampling_params = std::move(child_info.second);

    output_processor_.add_request(child, prompt, /*request_index=*/idx,
                                  /*queue=*/nullptr, parent);
    auto child_req = std::make_unique<Request>(
        Request::FromEngineCoreRequest(child, block_hasher_));
    engine_core_.add_request(std::move(child_req));
  }
}

std::vector<RequestOutput> LLMEngine::step() {
  // llm_engine.py:303 outputs = self.engine_core.get_output(). Our EngineCore
  // fuses the core's schedule/execute/sample/update into step(), returning the
  // per-client outputs map (T0: at most one entry) + model_executed.
  std::pair<std::map<int, EngineCoreOutputs>, bool> stepped = engine_core_.step();
  std::map<int, EngineCoreOutputs>& outputs_by_client = stepped.first;

  std::vector<RequestOutput> request_outputs;
  for (auto& entry : outputs_by_client) {
    EngineCoreOutputs& engine_core_outputs = entry.second;

    // llm_engine.py:308 iteration_stats = IterationStats() if self.log_stats.
    // Built only when a stat logger is attached; otherwise process_outputs runs
    // its byte-identical no-stats path.
    IterationStats iteration_stats;
    IterationStats* iteration_stats_ptr =
        stat_logger_ != nullptr ? &iteration_stats : nullptr;

    // llm_engine.py:309 process_outputs — detokenize + string-stop + assemble
    // (+ fold IterationStats when logging).
    OutputProcessorOutput processed = output_processor_.process_outputs(
        engine_core_outputs, iteration_stats_ptr);

    // llm_engine.py:318 abort any reqs that finished due to stop strings (the
    // detokenizer stopped them but EngineCore did not signal it itself).
    engine_core_.abort_requests(processed.reqs_to_abort);

    // Record prefix lookups even when this chunk has no sampled token.
    // IterationStats is empty for that observation, so it adds no token or
    // completion timing sample.
    if (stat_logger_ != nullptr &&
        (!engine_core_outputs.outputs.empty() ||
         engine_core_outputs.scheduler_stats.prefix_cache_stats.queries > 0)) {
      stat_logger_->Record(engine_core_outputs.scheduler_stats, iteration_stats);
    }

    for (RequestOutput& ro : processed.request_outputs) {
      request_outputs.push_back(std::move(ro));
    }
  }
  return request_outputs;
}

void LLMEngine::abort_request(const std::string& request_id) {
  // llm_engine.py:230 abort_request: drop the request from the output processor
  // (so has_unfinished_requests() no longer counts it) and from the engine core
  // (scheduler finish_requests -> FINISHED_ABORTED). Upstream aborts the output
  // processor first, then the engine core; both are no-ops for unknown ids.
  const std::vector<std::string> ids = {request_id};
  output_processor_.abort_requests(ids);
  engine_core_.abort_requests(ids);
}

RequestOutput LLMEngine::generate(const std::string& prompt,
                                  SamplingParams params,
                                  const std::string& request_id, int priority) {
  // Offline driver for ONE request. CRITICAL: wait only until *this* request_id
  // finishes — not has_unfinished_requests() globally. Otherwise a concurrent
  // chat/async job (e.g. huge Hermes SOUL) pins every blocking generate forever.
  add_request(request_id, prompt, std::move(params), priority);
  RequestOutput result;
  int idle_steps = 0;
  int steps = 0;
  constexpr int kMaxIdleSteps = 100000;  // safety; max_tokens should stop sooner
  while (true) {
    std::vector<RequestOutput> step_outputs = step();
    ++steps;
    bool saw_self = false;
    bool self_finished = false;
    for (RequestOutput& out : step_outputs) {
      if (out.request_id != request_id) continue;
      saw_self = true;
      if (out.finished) {
        result = std::move(out);
        self_finished = true;
      }
    }
    if (self_finished) break;
    if (!saw_self) {
      ++idle_steps;
      if (idle_steps >= kMaxIdleSteps) {
        abort_request(request_id);
        result.request_id = request_id;
        result.finished = true;
        break;
      }
    } else {
      idle_steps = 0;
    }
    // If our request vanished without a finished output, stop.
    if (!has_unfinished_requests() && !self_finished) {
      break;
    }
  }
  (void)steps;
  return result;
}

RequestOutput LLMEngine::generate(std::vector<int32_t> prompt_token_ids,
                                  SamplingParams params,
                                  const std::string& request_id, int priority) {
  add_request(request_id, std::move(prompt_token_ids), std::move(params),
              priority);
  RequestOutput result;
  int idle_steps = 0;
  constexpr int kMaxIdleSteps = 100000;
  while (true) {
    std::vector<RequestOutput> step_outputs = step();
    bool saw_self = false;
    bool self_finished = false;
    for (RequestOutput& out : step_outputs) {
      if (out.request_id != request_id) continue;
      saw_self = true;
      if (out.finished) {
        result = std::move(out);
        self_finished = true;
      }
    }
    if (self_finished) break;
    if (!saw_self) {
      ++idle_steps;
      if (idle_steps >= kMaxIdleSteps) {
        abort_request(request_id);
        result.request_id = request_id;
        result.finished = true;
        break;
      }
    } else {
      idle_steps = 0;
    }
    if (!has_unfinished_requests() && !self_finished) break;
  }
  return result;
}

RequestOutput LLMEngine::embed(std::vector<int32_t> prompt_token_ids,
                               PoolingParams pooling_params,
                               const std::string& request_id, int priority) {
  add_pooling_request(request_id, std::move(prompt_token_ids),
                      std::move(pooling_params), priority);
  RequestOutput result;
  while (true) {
    std::vector<RequestOutput> step_outputs = step();
    bool self_finished = false;
    for (RequestOutput& out : step_outputs) {
      if (out.request_id == request_id && out.finished) {
        result = std::move(out);
        self_finished = true;
      }
    }
    if (self_finished) break;
    if (!has_unfinished_requests()) break;
  }
  return result;
}

RequestOutput LLMEngine::generate(multimodal::MultiModalInputs mm_inputs,
                                  SamplingParams params,
                                  const std::string& request_id, int priority) {
  add_request(request_id, std::move(mm_inputs), std::move(params), priority);
  RequestOutput result;
  int idle_steps = 0;
  constexpr int kMaxIdleSteps = 100000;
  while (true) {
    std::vector<RequestOutput> step_outputs = step();
    bool saw_self = false;
    bool self_finished = false;
    for (RequestOutput& out : step_outputs) {
      if (out.request_id != request_id) continue;
      saw_self = true;
      if (out.finished) {
        result = std::move(out);
        self_finished = true;
      }
    }
    if (self_finished) break;
    if (!saw_self) {
      ++idle_steps;
      if (idle_steps >= kMaxIdleSteps) {
        abort_request(request_id);
        result.request_id = request_id;
        result.finished = true;
        break;
      }
    } else {
      idle_steps = 0;
    }
    if (!has_unfinished_requests() && !self_finished) break;
  }
  return result;
}

}  // namespace vllm::v1
