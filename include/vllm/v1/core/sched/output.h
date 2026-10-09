// Ported from: vllm/v1/core/sched/output.py @ e24d1b24
//
// Scope (M1.4 Task 2): the value types the V1 Scheduler PRODUCES each step and
// the model runner (M1.5) CONSUMES — the new-request full payload, the
// running-request diff payload, and the SchedulerOutput envelope that carries
// them across the (upstream: process) boundary. Behavioral only: plain value
// carriers, no CUDA / model. Field names + the new-vs-cached diff semantics are
// mirrored 1:1 with upstream.
//
// THE new-vs-cached DIFF PROTOCOL (mirrored 1:1):
//   * scheduled_new_reqs: requests scheduled for the FIRST time carry their
//     FULL data (NewRequestData) — prompt, sampling params, the complete
//     per-group block_ids, num_computed_tokens. The worker caches this so it is
//     never re-sent.
//   * scheduled_cached_reqs: requests scheduled BEFORE carry only the DIFF
//     (CachedRequestData) — parallel arrays over req_ids: the NEWLY allocated
//     block_ids per request per group (appended to the cached block table,
//     unless the req_id is in resumed_req_ids in which case they REPLACE it),
//     the updated num_computed_tokens, num_output_tokens, and (PP-only)
//     new_token_ids. The heavy per-request state already lives in the worker.
//
// DEFERRED upstream state (marked; T0 never populates these):
//   NewRequestData: pooling_params, lora_request, prompt_embeds,
//     prompt_is_token_ids. mm_features IS carried now — ENG-MM-INPUT-PIPELINE
//     P2 (#2379) un-deferred it; see the field note. prefill_token_ids IS
//     carried now (the
//     MRV2 / V2 model-runner path — see the field note); the MRV1 path is not
//     ported.
//   SchedulerOutput trailing optionals, OMITTED here (a later unit slots them
//     back in without reshaping the struct): preempted_req_ids /
//     new_block_ids_to_zero (v2 model runner), num_invalid_spec_tokens /
//     num_spec_tokens_to_schedule (spec decode), kv_connector_metadata /
//     ec_connector_metadata (KV/EC transfer). GrammarOutput (structured output)
//     is likewise omitted.
//
// DEVIATIONS, recorded:
//   - Upstream NewRequestData.prompt_token_ids is `list[int] | None` and
//     .sampling_params is `SamplingParams | None`; represented here as
//     std::optional<...>. from_request always populates both in the T0
//     pure-token / generation path (pooling is deferred), but the optional
//     preserves the upstream nullability.
//   - block_ids is `tuple[list[int], ...]` upstream (one list per KV cache
//     group); here std::vector<std::vector<int>>, exactly what
//     KVCacheBlocks::get_block_ids() / KVCacheManager::get_block_ids() return.
//   - CachedRequestData.new_block_ids entries are
//     `tuple[list[int], ...] | None` upstream (None => no new blocks this step,
//     from get_block_ids(allow_none=True)); here
//     std::optional<std::vector<std::vector<int>>>.
//   - Upstream caches _req_id_to_num_output_tokens (a cached_property) for O(1)
//     is_context_phase lookups; here is_context_phase does a linear scan over
//     req_ids (T0 batches are small; the value types are rebuilt fresh each
//     step so no cache-invalidation concern). Semantics are identical.
//   - all_token_ids is the MRV1-only connector-propagation map (dict[str,
//     list[int]] upstream); carried for shape fidelity, left empty in T0.
//   - The scheduler assembles CachedRequestData field-by-field via
//     scheduler.py::_make_cached_request_data (lands in Task 3); output.py
//     itself exposes ONLY make_empty (no from_request/append constructor). The
//     public aggregate here matches that — the diff-building loop is Task 3.
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/multimodal/inputs.h"  // multimodal::MultiModalFeatureSpec
#include "vllm/sampling_params.h"
#include "vllm/v1/structured_output/backend_types.h"  // TokenBitmask

namespace vllm::v1 {

struct Request;  // vllm/v1/request.h (from_request source; cpp includes it).

// NewRequestData: the FULL payload for a request scheduled for the first time.
// (Upstream NewRequestData dataclass.)
struct NewRequestData {
  std::string req_id;
  // Upstream list[int] | None — always populated in the T0 token path.
  std::optional<std::vector<int32_t>> prompt_token_ids;
  // Upstream SamplingParams | None — always populated for generation requests.
  std::optional<SamplingParams> sampling_params;
  // Per-KV-cache-group block ids (upstream tuple[list[int], ...]).
  std::vector<std::vector<int>> block_ids;
  int num_computed_tokens = 0;
  // MRV2 (V2 model runner) full token-id seed: prompt+output at schedule time.
  // The V2 gpu runner asserts this is present for every new req and seeds its
  // per-request all_token_ids state from it (model_runner add_requests:
  // all_token_ids = prefill_token_ids). The MRV1 path passed nullopt; the V2
  // path passes request.AllTokenIds(). Upstream list[int] | None.
  std::optional<std::vector<int32_t>> prefill_token_ids;
  // mm_features (NewRequestData.mm_features, output.py; copied at
  // gpu_model_runner.py:1293 `mm_features=new_req_data.mm_features`, grep -c ==
  // 1). The processed multimodal placeholder specs the WORKER needs: the
  // encoder step runs the tower over them, the gather splices the results, and
  // the MRoPE init reads their spans. Request.mm_features was already carried
  // for the prefix-cache extra keys; this is the hop that lets it leave the
  // scheduler. EMPTY on every text request, and the runner's mm path is
  // predicated on it being non-empty somewhere in the step.
  // ENG-MM-INPUT-PIPELINE P2 (#2379).
  std::vector<multimodal::MultiModalFeatureSpec> mm_features;

  // Prompt hashes identify immutable recurrent snapshots at completed boundaries.
  std::vector<std::string> prompt_block_hashes = {};

  // from_request: build the full payload from the Request + its allocated
  // per-group block ids. Copies req_id, prompt_token_ids, sampling_params,
  // block_ids, num_computed_tokens, mm_features, and the (optional)
  // prefill_token_ids seed.
  // The V2 model-runner path passes prefill_token_ids = request.AllTokenIds();
  // callers that do not need the seed omit it (nullopt). (Upstream
  // NewRequestData.from_request; mm/pooling/lora/prompt_embeds DEFERRED.)
  static NewRequestData from_request(
      const Request& request, std::vector<std::vector<int>> block_ids,
      std::optional<std::vector<int32_t>> prefill_token_ids = std::nullopt);
};

// CachedRequestData: the DIFF-only payload for requests scheduled before.
// Parallel arrays indexed by position over req_ids. (Upstream CachedRequestData
// dataclass.)
struct CachedRequestData {
  std::vector<std::string> req_ids;
  // req_ids in this set have their block table REPLACED by new_block_ids
  // (resumed from preemption); those not in it have new_block_ids APPENDED.
  std::set<std::string> resumed_req_ids;
  // PP-only: the sampled token ids to forward. Empty when PP is not used.
  std::vector<std::vector<int32_t>> new_token_ids;
  // MRV1-only connector propagation (req_id -> full token ids). Empty in T0.
  std::map<std::string, std::vector<int32_t>> all_token_ids;
  // Per request: the NEWLY allocated per-group block ids, or nullopt when no
  // new blocks were allocated this step (upstream get_block_ids(allow_none)).
  std::vector<std::optional<std::vector<std::vector<int>>>> new_block_ids;
  std::vector<int> num_computed_tokens;
  std::vector<int> num_output_tokens;

  // num_reqs (property): len(req_ids).
  int num_reqs() const { return static_cast<int>(req_ids.size()); }

  // is_context_phase: true iff the request is present and still in its prefill
  // (context) phase, i.e. num_output_tokens == 0 for that req_id. (Upstream
  // is_context_phase via the _req_id_to_num_output_tokens cache; linear scan
  // here — see the header DEVIATIONS note.)
  bool is_context_phase(const std::string& req_id) const;

  // make_empty: the no-cached-requests diff.
  static CachedRequestData make_empty();
};

// SchedulerOutput: the per-step envelope the model runner consumes. (Upstream
// SchedulerOutput dataclass — T0 field subset; trailing optionals OMITTED, see
// the header DEFERRED note.)
struct SchedulerOutput {
  // Requests scheduled for the first time (full data, worker-cached).
  std::vector<NewRequestData> scheduled_new_reqs;
  // Requests scheduled before (diff only).
  CachedRequestData scheduled_cached_reqs;

  // req_id -> number of tokens scheduled for it this step.
  std::map<std::string, int> num_scheduled_tokens;
  // sum(num_scheduled_tokens.values()).
  int total_num_scheduled_tokens = 0;

  // req_id -> spec decode token ids. DEFERRED semantics: always empty in T0.
  std::map<std::string, std::vector<int32_t>> scheduled_spec_decode_tokens;
  // req_id -> encoder input indices to process. DEFERRED semantics: empty in T0.
  std::map<std::string, std::vector<int>> scheduled_encoder_inputs;
  // Number of common prefix blocks per KV cache group (for cascade attention).
  std::vector<int> num_common_prefix_blocks;

  // Request ids finished between the previous and current step (free cached
  // worker state for these).
  std::set<std::string> finished_req_ids;
  // mm_hash strings the encoder cache actually EVICTED since the previous step
  // (EncoderCacheManager::GetFreedMmHashes, encoder_cache_manager.py:255). The
  // worker drops those entries from its own mm_hash-keyed cache. Empty on every
  // text step.
  std::vector<std::string> free_encoder_mm_hashes;

  // Whether any structured-output request is scheduled this step and past its
  // prefill (upstream SchedulerOutput.has_structured_output_requests, set in
  // _update_after_schedule scheduler.py:1186). Gates get_grammar_bitmask.
  bool has_structured_output_requests = false;

  // Whether async scheduling must DEFER sampling this step because a structured
  // request is still waiting on the previous step's (in-flight, placeholder)
  // tokens before its grammar bitmask can be computed. Set by
  // AsyncScheduler::update_after_schedule (async_scheduler.py:31-33); read by
  // EngineCore::step_with_batch_queue (core.py:559-570). Always false under the
  // synchronous Scheduler. Un-deferred from the output.h DEFERRED list for
  // ENG-ASYNC-SCHED (spec async-serving.md W3).
  bool pending_structured_output_tokens = false;

  // num_spec_tokens_to_schedule (output.py:263 @ 555967922; set at
  // scheduler.py:1123-1156): the draft-token count the NEXT step's async
  // placeholders take — AsyncScheduler::update_after_schedule assigns
  // [-1] * this to each scheduled decode request (async_scheduler.py:24-25,
  // 43-45). The flat num_spec_tokens here (the dynamic-SD lookup that can
  // shrink it per step is deferred). 0 whenever no speculator is configured,
  // which keeps the default path byte-identical. SPEC-DFLASH2 W7 (#1824).
  int num_spec_tokens_to_schedule = 0;

  // num_invalid_spec_tokens (output.py:244): req_id -> the count of scheduled
  // spec tokens the draft-in-output rewrite could not fill (worker returned
  // fewer than scheduled / grammar-filtered) and padded with -1
  // (Scheduler::update_draft_token_ids_in_output, scheduler.py:2072-2107).
  // Each rewrite REPLACES the whole map. Empty unless that rewrite ran.
  std::map<std::string, int> num_invalid_spec_tokens;

  // make_empty: an empty step output.
  static SchedulerOutput make_empty();
};

// GrammarOutput (output.py:262-267): the per-step structured-output payload the
// scheduler produces (get_grammar_bitmask) and the runner consumes
// (apply_grammar_bitmask, Task 3). The bitmask rows are ordered exactly as
// structured_output_request_ids.
struct GrammarOutput {
  // ids of structured output requests.
  std::vector<std::string> structured_output_request_ids;
  // Bitmask ordered as structured_output_request_ids.
  TokenBitmask grammar_bitmask;
};

}  // namespace vllm::v1
