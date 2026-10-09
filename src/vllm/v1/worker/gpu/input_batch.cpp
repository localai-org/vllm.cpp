// Ported from: vllm/v1/worker/gpu_input_batch.py @ e24d1b24
// See include/vllm/v1/worker/gpu/input_batch.h for the scope, the V1-algorithm
// / MRV2-contract note, and the deferred slot state.

#include "vllm/v1/worker/gpu/input_batch.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vllm::v1 {

// ─── CachedRequestState ─────────────────────────────────────────────────────

CachedRequestState CachedRequestState::from_new_request(
    const NewRequestData& new_req) {
  CachedRequestState state;
  state.req_id = new_req.req_id;
  if (new_req.prompt_token_ids.has_value()) {
    state.prompt_token_ids = *new_req.prompt_token_ids;
  }
  // T0 generation path always carries sampling_params (pooling DEFERRED).
  state.sampling_params = new_req.sampling_params.value();
  state.block_ids = new_req.block_ids;
  state.num_computed_tokens = new_req.num_computed_tokens;
  // ENG-MM-INPUT-PIPELINE P2 (#2379), the worker end of the hop
  // NewRequestData opened (gpu_model_runner.py:1293). Empty on text.
  state.mm_features = new_req.mm_features;
  state.prompt_block_hashes = new_req.prompt_block_hashes;
  state.finalize();

  // MRV2 contract: prefill_token_ids == all_token_ids (prompt + output at
  // schedule time). The output tokens are its tail beyond the prompt, so the
  // per-slot token_ids_cpu seed (prompt then output) reproduces exactly the
  // scheduler's prefill_token_ids. (The V1 runner instead reconstructed
  // all_token_ids per-req — that MRV1 path is dead here.)
  if (new_req.prefill_token_ids.has_value()) {
    const std::vector<int32_t>& all_ids = *new_req.prefill_token_ids;
    for (size_t i = static_cast<size_t>(state.num_prompt_tokens);
         i < all_ids.size(); ++i) {
      state.output_token_ids.push_back(all_ids[i]);
    }
  }
  return state;
}

int CachedRequestState::get_token_id(int idx) const {
  if (idx < num_prompt_tokens) {
    return prompt_token_ids[static_cast<size_t>(idx)];
  }
  if (idx - num_prompt_tokens < static_cast<int>(output_token_ids.size())) {
    return output_token_ids[static_cast<size_t>(idx - num_prompt_tokens)];
  }
  return -1;
}

// ─── InputBatch::RemovedTracker ─────────────────────────────────────────────

void InputBatch::RemovedTracker::ensure_sorted() {
  if (!is_sorted_) {
    // Descending, so back() is the lowest index (upstream _ensure_removed_sorted
    // sorts reverse=True and pops from the end).
    std::sort(removed_.begin(), removed_.end(), std::greater<int>());
    is_sorted_ = true;
  }
}

void InputBatch::RemovedTracker::removed_append(int index) {
  removed_.push_back(index);
  is_sorted_ = false;
}

std::optional<int> InputBatch::RemovedTracker::peek_removed() {
  if (removed_.empty()) {
    return std::nullopt;
  }
  ensure_sorted();
  return removed_.back();
}

std::optional<int> InputBatch::RemovedTracker::pop_removed() {
  if (removed_.empty()) {
    return std::nullopt;
  }
  ensure_sorted();
  const int value = removed_.back();
  removed_.pop_back();
  return value;
}

const std::vector<int>& InputBatch::RemovedTracker::removed() {
  ensure_sorted();
  return removed_;
}

// ─── InputBatch ─────────────────────────────────────────────────────────────

namespace {
// Upstream builds MultiGroupBlockTable with per-group max_num_blocks derived
// from max_model_len; block_sizes / kernel_block_sizes are the caller's groups.
MultiGroupBlockTable make_block_table(
    int max_num_reqs, int max_model_len, int max_num_batched_tokens,
    std::vector<int> block_sizes, std::vector<int> kernel_block_sizes,
    std::optional<std::vector<int>> max_num_blocks_per_req,
    std::optional<std::vector<SlotMappingMode>> slot_mapping_modes) {
  return MultiGroupBlockTable(max_num_reqs, max_model_len,
                              max_num_batched_tokens, std::move(block_sizes),
                              std::move(kernel_block_sizes),
                              std::move(max_num_blocks_per_req),
                              /*cp_kv_cache_interleave_size=*/1,
                              std::move(slot_mapping_modes));
}
}  // namespace

InputBatch::InputBatch(int max_num_reqs, int max_model_len,
                       int max_num_batched_tokens, int vocab_size,
                       std::vector<int> block_sizes,
                       std::vector<int> kernel_block_sizes,
                       int num_speculative_steps,
                       std::optional<std::vector<int>> max_num_blocks_per_req,
                       std::optional<std::vector<SlotMappingMode>> slot_mapping_modes)
    : max_num_reqs(max_num_reqs),
      max_model_len(max_model_len),
      max_num_batched_tokens(max_num_batched_tokens),
      vocab_size(vocab_size),
      num_speculative_steps(num_speculative_steps),
      block_table(make_block_table(max_num_reqs, max_model_len,
                                   max_num_batched_tokens, std::move(block_sizes),
                                   std::move(kernel_block_sizes),
                                   std::move(max_num_blocks_per_req),
                                   std::move(slot_mapping_modes))) {
  const size_t n = static_cast<size_t>(max_num_reqs);
  token_ids_cpu.assign(n * static_cast<size_t>(max_model_len), 0);
  num_tokens_no_spec.assign(n, 0);
  num_prompt_tokens.assign(n, 0);
  num_computed_tokens_cpu.assign(n, 0);
  last_sampled_tokens.assign(n, 0);
  prefill_len.assign(n, 0);
  // SPEC-DFLASH2 A2-3 (#2911): sized by the req_state POOL and not by any step's
  // request count (states.py:71-77). See the header for why that is the
  // correctness requirement rather than the convenient choice: the scatter's
  // CUDA arm indexes this by req_state slot with nothing bounding the read.
  // num_speculative_steps == 0 leaves it empty, which is upstream's zero-width
  // tensor and what keeps every non-speculative runner unchanged.
  draft_tokens.assign(n * static_cast<size_t>(num_speculative_steps), 0);
  num_valid_draft_tokens.assign(n, 0);
  temperature_cpu.assign(n, 0.0f);
  top_p_cpu.assign(n, 0.0f);
  top_k_cpu.assign(n, 0);
  frequency_penalties_cpu.assign(n, 0.0f);
  presence_penalties_cpu.assign(n, 0.0f);
  repetition_penalties_cpu.assign(n, 0.0f);
  min_p_cpu.assign(n, 0.0f);
  seeds.assign(n, std::nullopt);
  // Speculative decoding: by default 1 token is generated per request
  // (gpu_input_batch.py:240-243). Overwritten by the rejection sampler after a
  // verify step; stays 1 on the default (no-speculator) path.
  num_accepted_tokens.assign(n, 1);
}

int InputBatch::register_add_request() {
  // Fill the next empty index if there is one; append to the end otherwise.
  if (std::optional<int> hole = removed_tracker_.pop_removed()) {
    return *hole;
  }
  return num_reqs();
}

int InputBatch::get_active_token_count(int req_index) const {
  return num_tokens_no_spec[static_cast<size_t>(req_index)] +
         static_cast<int>(spec_token_ids[static_cast<size_t>(req_index)].size());
}

int InputBatch::add_request(const CachedRequestState& request) {
  sampling_metadata_dirty_ = true;  // batch changed -> refresh_metadata rebuild
  const int req_index = register_add_request();
  const std::string& req_id = request.req_id;

  if (req_index == static_cast<int>(req_ids.size())) {
    req_ids.push_back(req_id);
    req_output_token_ids.push_back(request.output_token_ids);
    spec_token_ids.emplace_back();
  } else {
    req_ids[static_cast<size_t>(req_index)] = req_id;
    req_output_token_ids[static_cast<size_t>(req_index)] =
        request.output_token_ids;
    spec_token_ids[static_cast<size_t>(req_index)].clear();
  }
  req_id_to_index[req_id] = req_index;

  // Seed the token buffer: prompt then output (== prefill_token_ids).
  const int num_prompt = request.num_prompt_tokens;
  num_prompt_tokens[static_cast<size_t>(req_index)] = num_prompt;
  const size_t row = static_cast<size_t>(req_index) *
                     static_cast<size_t>(max_model_len);
  for (int i = 0; i < num_prompt; ++i) {
    token_ids_cpu[row + static_cast<size_t>(i)] = request.prompt_token_ids[
        static_cast<size_t>(i)];
  }
  for (size_t i = 0; i < request.output_token_ids.size(); ++i) {
    token_ids_cpu[row + static_cast<size_t>(num_prompt) + i] =
        request.output_token_ids[i];
  }
  // Number of tokens without spec-decode tokens.
  num_tokens_no_spec[static_cast<size_t>(req_index)] = request.num_tokens();
  num_computed_tokens_cpu[static_cast<size_t>(req_index)] =
      request.num_computed_tokens;
  // Speculative decoding: 1 token generated by default (gpu_input_batch.py:467).
  num_accepted_tokens[static_cast<size_t>(req_index)] = 1;
  block_table.add_row(request.block_ids, req_index);

  // Async-scheduling state (states.py::add_request:105-122). prefill_len is the
  // token count known at admission (prompt + pre-existing output), fixed for the
  // request's life. Seed last_sampled_tokens ONLY for a resumed / PD-disagg
  // request (0 < num_computed <= prefill_len) so its first decode step reads the
  // correct input id via combine; a fresh prefill (num_computed == 0) never has
  // combine read it, so it stays 0. Both are inert unless the async runner path
  // is engaged.
  const int prefill = request.num_tokens();
  prefill_len[static_cast<size_t>(req_index)] = prefill;
  if (0 < request.num_computed_tokens &&
      request.num_computed_tokens <= prefill) {
    last_sampled_tokens[static_cast<size_t>(req_index)] =
        token_ids_cpu[row + static_cast<size_t>(request.num_computed_tokens - 1)];
  } else {
    last_sampled_tokens[static_cast<size_t>(req_index)] = 0;
  }
  // W4: record the seed so a device mirror starts from the same value. The value
  // is host-known here, so this is the one op kind that carries data.
  last_sampled_ops.push_back(LastSampledOp{
      LastSampledOp::kSeed, req_index, 0,
      last_sampled_tokens[static_cast<size_t>(req_index)]});
  // SPEC-DFLASH2 A2-3 (#2911): `self.draft_tokens[req_idx].zero_()`
  // (states.py:113). A freed slot is handed to the NEXT request, so without this
  // the new occupant inherits the previous one's drafts. Those verify against a
  // sequence they were never proposed for; lossless verify then REJECTS them, so
  // the emitted tokens stay correct and only acceptance falls — nothing raises,
  // which is why it is zeroed here rather than left to a reader to notice.
  if (num_speculative_steps > 0) {
    const size_t draft_row = static_cast<size_t>(req_index) *
                             static_cast<size_t>(num_speculative_steps);
    std::fill(draft_tokens.begin() + static_cast<std::ptrdiff_t>(draft_row),
              draft_tokens.begin() + static_cast<std::ptrdiff_t>(draft_row) +
                  num_speculative_steps,
              0);
  }
  num_valid_draft_tokens[static_cast<size_t>(req_index)] = 0;

  // Sampling metadata (pooling DEFERRED — T0 always has sampling_params).
  const SamplingParams& sp = request.sampling_params;
  if (sp.Type() == SamplingType::kGreedy) {
    // Avoid a later division-by-zero in apply_temperature.
    temperature_cpu[static_cast<size_t>(req_index)] = 0.0f;
    greedy_reqs[req_id] = 1;
  } else {
    temperature_cpu[static_cast<size_t>(req_index)] =
        static_cast<float>(sp.temperature);
    random_reqs[req_id] = 1;
  }

  top_p_cpu[static_cast<size_t>(req_index)] = static_cast<float>(sp.top_p);
  if (sp.top_p < 1.0) {
    top_p_reqs[req_id] = 1;
  }
  int top_k = sp.top_k;
  if (0 < top_k && top_k < vocab_size) {
    top_k_reqs[req_id] = 1;
  } else {
    top_k = vocab_size;
  }
  top_k_cpu[static_cast<size_t>(req_index)] = top_k;

  frequency_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.frequency_penalty);
  if (sp.frequency_penalty != 0.0) {
    frequency_penalties_reqs[req_id] = 1;
  }
  presence_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.presence_penalty);
  if (sp.presence_penalty != 0.0) {
    presence_penalties_reqs[req_id] = 1;
  }
  repetition_penalties_cpu[static_cast<size_t>(req_index)] =
      static_cast<float>(sp.repetition_penalty);
  if (sp.repetition_penalty != 1.0) {
    repetition_penalties_reqs[req_id] = 1;
  }

  // Per-request RNG seed (upstream request.generator == sampling_params.seed).
  seeds[static_cast<size_t>(req_index)] = sp.seed;

  // ─── ROAD-V1-C7 SAMPLE-CORE / SAMPLE-LOGIT-FILTERS per-slot controls ───────
  // min_p (gpu_input_batch.py:414-419-style; MinPLogitsProcessor).
  min_p_cpu[static_cast<size_t>(req_index)] = static_cast<float>(sp.min_p);
  if (sp.min_p > 0.0) {
    min_p_reqs[req_id] = 1;
  } else {
    min_p_reqs.erase(req_id);
  }

  // min_tokens: only tracked when a floor is set (MinTokensLogitsProcessor). The
  // stop set is params.all_stop_token_ids (stop_token_ids + engine eos ids).
  min_tokens.erase(req_index);
  if (sp.min_tokens > 0) {
    min_tokens[req_index] = MinTokensState{
        sp.min_tokens,
        std::set<int32_t>(sp.all_stop_token_ids.begin(),
                          sp.all_stop_token_ids.end())};
  }

  // logit_bias (LogitBiasLogitsProcessor; gpu_input_batch.py analogue).
  logit_bias.erase(req_index);
  if (sp.logit_bias.has_value() && !sp.logit_bias->empty()) {
    logit_bias[req_index] = *sp.logit_bias;
  }

  // custom logits processor (ROAD-V1-C7 `custom_logit_processor`): a host
  // callback carried on the SamplingParams. Track only when one is registered.
  logits_processors.erase(req_index);
  if (sp.logits_processor.fn != nullptr) {
    logits_processors[req_index] = sp.logits_processor;
  }

  // num_logprobs (gpu_input_batch.py:434-440). The `-1` ("all logprobs")
  // sentinel is WIDENED to vocab_size here, exactly as upstream does, so that
  // downstream sees one shape whatever the request asked for. See
  // specs/logprobs-all-sentinel.md.
  num_logprobs.erase(req_id);
  if (sp.logprobs.has_value()) {
    num_logprobs[req_id] = *sp.logprobs == -1 ? vocab_size : *sp.logprobs;
  }

  // logprob_token_ids (gpu_input_batch.py:443-444): the EXPLICIT vocab ids to
  // score. Tracked separately from num_logprobs and, exactly as upstream, it
  // does NOT populate num_logprobs — the sampler's `or` at sampler.py:86 is
  // what makes the raw-logprobs snapshot fire for a request that sets only
  // this. Absent for every request that leaves the field unset.
  logprob_token_ids.erase(req_id);
  if (sp.logprob_token_ids.has_value()) {
    logprob_token_ids[req_id] = *sp.logprob_token_ids;
  }

  // num_prompt_logprobs (gpu_model_runner.py:1305-1310). Unlike num_logprobs
  // above, the `-1` ("all") sentinel is WIDENED to vocab_size here exactly as
  // upstream does: the prompt path feeds GatherLogprobs, which needs a concrete
  // column count, and it never reaches the Sampler that consumes our sentinel.
  num_prompt_logprobs.erase(req_id);
  if (sp.prompt_logprobs.has_value()) {
    num_prompt_logprobs[req_id] =
        *sp.prompt_logprobs == -1 ? vocab_size : *sp.prompt_logprobs;
  }

  // allowed_token_ids (gpu_input_batch.py:446-467). Lazily allocate the
  // [max_num_reqs][vocab] exclude mask; set the row all-TRUE (exclude) then clear
  // the allowed ids to FALSE (keep).
  has_allowed_token_ids.erase(req_id);
  if (allowed_token_ids_mask.empty() ||
      static_cast<int>(allowed_token_ids_mask[static_cast<size_t>(req_index)]
                           .size()) == vocab_size) {
    // Reset this row to all-false (keep every token) when the mask already
    // exists; the block below re-fills it for a request that restricts ids.
    if (!allowed_token_ids_mask.empty()) {
      std::fill(allowed_token_ids_mask[static_cast<size_t>(req_index)].begin(),
                allowed_token_ids_mask[static_cast<size_t>(req_index)].end(),
                static_cast<uint8_t>(0));
    }
  }
  if (sp.allowed_token_ids.has_value() && !sp.allowed_token_ids->empty()) {
    has_allowed_token_ids[req_id] = 1;
    if (allowed_token_ids_mask.empty()) {
      allowed_token_ids_mask.assign(
          static_cast<size_t>(max_num_reqs),
          std::vector<uint8_t>(static_cast<size_t>(vocab_size), 0));
    }
    std::vector<uint8_t>& mask_row =
        allowed_token_ids_mask[static_cast<size_t>(req_index)];
    std::fill(mask_row.begin(), mask_row.end(), static_cast<uint8_t>(1));
    for (int32_t tid : *sp.allowed_token_ids) {
      if (0 <= tid && tid < vocab_size) {
        mask_row[static_cast<size_t>(tid)] = 0;  // FALSE == keep this token
      }
    }
  }

  // bad_words (gpu_input_batch.py:469-471). Tokenized engine-side into
  // sampling_params.bad_words_token_ids (InputProcessor::UpdateFromTokenizer).
  bad_words_token_ids.erase(req_index);
  if (sp.bad_words_token_ids.has_value() && !sp.bad_words_token_ids->empty()) {
    bad_words_token_ids[req_index] = *sp.bad_words_token_ids;
  }

  return req_index;
}

const SamplingMetadata& InputBatch::make_sampling_metadata() const {
  // Batch-change-gated rebuild (rescan §6 item e), mirroring upstream
  // refresh_metadata (gpu_input_batch.py:812-830): rebuild the cached metadata
  // only when a batch mutation set sampling_metadata_dirty_. Deviation: our port
  // COPIES output_token_ids where upstream holds a live reference, so when
  // penalties are active (!no_penalties) — the only case that embeds the
  // per-step-growing output tokens — we rebuild every call to keep them fresh.
  // For the greedy / no-penalties gate workload the metadata depends only on the
  // (unchanged) request set + static sampling params, so the cache is
  // bit-identical to a fresh build. ROAD-V1-C7: min_tokens / bad_words also embed
  // the per-step-growing output tokens, so rebuild when either is active. A
  // custom logits processor is passed the per-step-growing output tokens too, so
  // rebuild when one is registered.
  if (sampling_metadata_dirty_ || !no_penalties() || !min_tokens.empty() ||
      !bad_words_token_ids.empty() || !logits_processors.empty()) {
    sampling_metadata_cache_ = build_sampling_metadata();
    sampling_metadata_dirty_ = false;
  }
  // Sampling offsets change even when all other metadata remains cached. Use
  // the logical accepted token count, never a sampling-call counter: chunked
  // prefill draws that are discarded must not advance a seeded request.
  auto& positions = sampling_metadata_cache_.output_token_positions;
  positions.resize(static_cast<size_t>(num_reqs()));
  for (int i = 0; i < num_reqs(); ++i) {
    positions[size_t(i)] = static_cast<uint64_t>(std::max(0, num_tokens_no_spec[size_t(i)] - num_prompt_tokens[size_t(i)]));
  }
  return sampling_metadata_cache_;
}

SamplingMetadata InputBatch::build_sampling_metadata() const {
  // Port of gpu_input_batch.py::_make_sampling_metadata (@ e24d1b24). Fills the
  // dense [0, num_reqs) prefix, matching upstream's field-fill order + the
  // "skip the copy when not needed" None/[]-defaults.
  const int n = num_reqs();
  const size_t nn = static_cast<size_t>(n);
  SamplingMetadata md;

  md.all_greedy = all_greedy();
  md.all_random = all_random();
  md.no_penalties = no_penalties();

  // temperature: None when all_greedy, else the [:num_reqs] slice
  // (gpu_input_batch.py:834-839).
  if (!md.all_greedy) {
    md.temperature = std::vector<float>(temperature_cpu.begin(),
                                        temperature_cpu.begin() + nn);
  }
  // top_p / top_k: None when the corresponding predicate is empty
  // (gpu_input_batch.py:919-920).
  if (!no_top_p()) {
    md.top_p =
        std::vector<float>(top_p_cpu.begin(), top_p_cpu.begin() + nn);
  }
  if (!no_top_k()) {
    md.top_k =
        std::vector<int32_t>(top_k_cpu.begin(), top_k_cpu.begin() + nn);
  }

  // Penalties are always sliced [:num_reqs] in the returned metadata
  // (gpu_input_batch.py:925-927); the device-copy is what upstream gates on
  // no_penalties, not the slice itself.
  md.frequency_penalties = std::vector<float>(
      frequency_penalties_cpu.begin(), frequency_penalties_cpu.begin() + nn);
  md.presence_penalties = std::vector<float>(
      presence_penalties_cpu.begin(), presence_penalties_cpu.begin() + nn);
  md.repetition_penalties = std::vector<float>(
      repetition_penalties_cpu.begin(), repetition_penalties_cpu.begin() + nn);

  // prompt_token_ids: only when penalties (or a token-id-consuming proc, always
  // false at T0) need them (gpu_input_batch.py:861-876). Ragged per-req prompt
  // slice of token_ids_cpu[:, :num_prompt_tokens].
  const bool needs_prompt_token_ids = !md.no_penalties;
  if (needs_prompt_token_ids) {
    std::vector<std::vector<int32_t>> prompts(nn);
    for (int i = 0; i < n; ++i) {
      const int np = num_prompt_tokens[static_cast<size_t>(i)];
      const size_t row =
          static_cast<size_t>(i) * static_cast<size_t>(max_model_len);
      prompts[static_cast<size_t>(i)].assign(
          token_ids_cpu.begin() + static_cast<std::ptrdiff_t>(row),
          token_ids_cpu.begin() + static_cast<std::ptrdiff_t>(row) + np);
    }
    md.prompt_token_ids = std::move(prompts);
  }

  // output_token_ids: only when a proc needs them (gpu_input_batch.py:906-922).
  // needs_output_token_ids = not no_penalties OR bad_words present OR a
  // logitsproc needs them (min_tokens is such a proc — it compares output_len to
  // the floor). thinking-budget stays deferred.
  const bool needs_output_token_ids =
      !md.no_penalties || !bad_words_token_ids.empty() || !min_tokens.empty() ||
      !logits_processors.empty();
  if (needs_output_token_ids) {
    md.output_token_ids.resize(nn);
    for (int i = 0; i < n; ++i) {
      const auto& row = req_output_token_ids[static_cast<size_t>(i)];
      if (row.has_value()) {
        md.output_token_ids[static_cast<size_t>(i)] = *row;
      }
    }
  }

  // spec_token_ids: pass the dense prefix (always empty lists at T0). Upstream
  // passes self.spec_token_ids directly (gpu_input_batch.py:929).
  md.spec_token_ids = std::vector<std::vector<int32_t>>(
      spec_token_ids.begin(), spec_token_ids.begin() + nn);

  // generators (gpu_input_batch.py:921, sourced :413-414 from request.generator
  // == sampling_params.seed): req_index -> seed for every seeded request in the
  // dense prefix. WIRED at M1.8 Task 4 from the per-slot `seeds` array; requests
  // without a seed are absent (they use the sampler's batch-default RNG,
  // upstream NOTE :251-252). CLOSES the M1.7 seed carry.
  for (int i = 0; i < n; ++i) {
    if (seeds[static_cast<size_t>(i)].has_value()) {
      md.generators[i] = static_cast<uint64_t>(*seeds[static_cast<size_t>(i)]);
    }
  }

  // ─── ROAD-V1-C7 SAMPLE-CORE / SAMPLE-LOGPROBS / SAMPLE-LOGIT-FILTERS ───────
  // max_num_logprobs (gpu_input_batch.py:950 / :1150-1151): max requested count
  // across the batch, or None. Default (no request asked) => None => the sampler
  // computes no logprobs (byte-identical to before).
  md.max_num_logprobs = max_num_logprobs();

  // logprob_token_ids (gpu_input_batch.py:934-951): re-key req_id -> req_INDEX
  // over the LIVE batch, skipping any id no longer in req_id_to_index. Upstream
  // leaves the field None when its own map is empty (`if self.logprob_token_ids:`
  // at :936), and `if sampling_metadata.logprob_token_ids:` at sampler.py:114
  // is false for BOTH None and {} — so an empty result stays unset here, which
  // keeps the default request byte-identical.
  if (!logprob_token_ids.empty()) {
    std::map<int, std::vector<int32_t>> by_index;
    for (const auto& [req_id, token_ids] : logprob_token_ids) {
      const auto it = req_id_to_index.find(req_id);
      if (it != req_id_to_index.end()) by_index[it->second] = token_ids;
    }
    if (!by_index.empty()) md.logprob_token_ids = std::move(by_index);
  }

  // allowed_token_ids_mask (gpu_input_batch.py:924-932): None unless some request
  // restricts ids; else the dense [:num_reqs] EXCLUDE-mask rows.
  if (!no_allowed_token_ids() && !allowed_token_ids_mask.empty()) {
    md.allowed_token_ids_mask = std::vector<std::vector<uint8_t>>(
        allowed_token_ids_mask.begin(), allowed_token_ids_mask.begin() + nn);
  }

  // bad_words_token_ids (gpu_input_batch.py:960): req_index -> n-grams, passed
  // through directly (the sampler indexes by req).
  md.bad_words_token_ids = bad_words_token_ids;

  // min_tokens / logit_bias / min_p (the T0 builtins). min_p is None-skipped like
  // top_p (empty vector => sampler no-ops the row); we pass the dense slice only
  // when some row is active, matching the "skip when unneeded" pattern.
  md.min_tokens = min_tokens;
  md.logit_bias = logit_bias;
  md.logits_processors = logits_processors;
  if (!no_min_p()) {
    md.min_p = std::vector<float>(min_p_cpu.begin(), min_p_cpu.begin() + nn);
  }

  return md;
}

std::optional<int> InputBatch::max_num_logprobs() const {
  // gpu_input_batch.py:1150-1151: max(num_logprobs.values()) or None. Every
  // value is already a concrete count — add_request widened `-1` to vocab_size
  // — so a request asking for "all" simply carries the largest count and wins
  // this max on its own, with no sentinel to propagate.
  if (num_logprobs.empty()) return std::nullopt;
  int best = 0;
  for (const auto& [req_id, k] : num_logprobs) {
    (void)req_id;
    best = std::max(best, k);
  }
  return best;
}

std::optional<int> InputBatch::remove_request(const std::string& req_id) {
  const auto it = req_id_to_index.find(req_id);
  if (it == req_id_to_index.end()) {
    return std::nullopt;
  }
  sampling_metadata_dirty_ = true;  // batch changed -> refresh_metadata rebuild
  const int req_index = it->second;
  req_id_to_index.erase(it);

  removed_tracker_.removed_append(req_index);
  req_ids[static_cast<size_t>(req_index)] = std::nullopt;
  req_output_token_ids[static_cast<size_t>(req_index)] = std::nullopt;
  spec_token_ids[static_cast<size_t>(req_index)].clear();
  seeds[static_cast<size_t>(req_index)] = std::nullopt;
  block_table.clear_row(req_index);

  // Discard from the sampling-predicate sets (LoRA / generators / logprobs /
  // pooling / structured-output DEFERRED).
  greedy_reqs.erase(req_id);
  random_reqs.erase(req_id);
  top_p_reqs.erase(req_id);
  top_k_reqs.erase(req_id);
  frequency_penalties_reqs.erase(req_id);
  presence_penalties_reqs.erase(req_id);
  repetition_penalties_reqs.erase(req_id);
  // ROAD-V1-C7 per-slot controls (gpu_input_batch.py:573-582). req_id-keyed
  // sets erase by id; req_index-keyed maps pop by index; the allowed-ids mask
  // row is cleared to all-false (keep).
  min_p_reqs.erase(req_id);
  has_allowed_token_ids.erase(req_id);
  num_logprobs.erase(req_id);
  logprob_token_ids.erase(req_id);  // gpu_input_batch.py:574
  // gpu_model_runner.py:1199 pops num_prompt_logprobs with the request state.
  num_prompt_logprobs.erase(req_id);
  min_tokens.erase(req_index);
  logit_bias.erase(req_index);
  logits_processors.erase(req_index);
  bad_words_token_ids.erase(req_index);
  if (!allowed_token_ids_mask.empty()) {
    std::fill(allowed_token_ids_mask[static_cast<size_t>(req_index)].begin(),
              allowed_token_ids_mask[static_cast<size_t>(req_index)].end(),
              static_cast<uint8_t>(0));
  }
  return req_index;
}

namespace {
// Move an index-keyed map value from `from` to `to`, popping any prior `to`
// (mirrors gpu_input_batch.py:828-830 condense's pop-and-reinsert).
template <typename V>
void MoveDictValue(std::map<int, V>& m, int from, int to) {
  auto it = m.find(from);
  if (it == m.end()) {
    m.erase(to);
    return;
  }
  m[to] = std::move(it->second);
  m.erase(it);
}
// Swap two index-keyed map values (mirrors swap_dict_values,
// gpu_input_batch.py:692).
template <typename V>
void SwapDictValues(std::map<int, V>& m, int i1, int i2) {
  auto a = m.find(i1);
  auto b = m.find(i2);
  const bool has_a = a != m.end();
  const bool has_b = b != m.end();
  if (has_a && has_b) {
    std::swap(a->second, b->second);
  } else if (has_a) {
    m[i2] = std::move(a->second);
    m.erase(i1);
  } else if (has_b) {
    m[i1] = std::move(b->second);
    m.erase(i2);
  }
}
}  // namespace

void InputBatch::condense() {
  const int num = num_reqs();

  // empty_req_indices is the removed list, sorted DESCENDING; it shrinks as we
  // pop (a live reference, exactly like upstream's batch_update_builder.removed).
  const std::vector<int>& empty_req_indices = removed_tracker_.removed();
  if (empty_req_indices.empty()) {
    // All removed slots were replaced by adds, or nothing was removed.
    return;
  }
  if (num == 0) {
    // The batched state is empty.
    req_ids.clear();
    req_output_token_ids.clear();
    spec_token_ids.clear();
    removed_tracker_ = RemovedTracker();
    return;
  }

  const auto is_empty = [&](int idx) {
    return std::find(empty_req_indices.begin(), empty_req_indices.end(), idx) !=
           empty_req_indices.end();
  };

  // NOTE(woosuk): assumes empty_req_indices is sorted in descending order.
  int last_req_index = num + static_cast<int>(empty_req_indices.size()) - 1;
  while (removed_tracker_.has_removed()) {
    // Find the largest non-empty index.
    while (is_empty(last_req_index)) {
      --last_req_index;
    }

    // Find the smallest empty index.
    const std::optional<int> empty_peek = removed_tracker_.peek_removed();
    const int empty_index = *empty_peek;
    if (empty_index >= last_req_index) {
      break;
    }

    // Move the active request at last_req_index down into empty_index. This
    // reorders the dense [0,num_reqs) prefix -> the sampling metadata must be
    // rebuilt (upstream tracks this as a batch_update_builder.moved). Set only
    // on an actual move so the no-op condense (called every step) preserves the
    // cache.
    sampling_metadata_dirty_ = true;
    removed_tracker_.pop_removed();
    const std::optional<std::string> req_id =
        req_ids[static_cast<size_t>(last_req_index)];
    req_ids[static_cast<size_t>(empty_index)] = req_id;
    req_ids[static_cast<size_t>(last_req_index)] = std::nullopt;
    // Move slot-to-slot directly (empty_index < last_req_index, always
    // distinct). Routing the move through a local
    // std::optional<std::vector<int32_t>> trips gcc-14's
    // -Werror=maybe-uninitialized when the later resize() inlines into
    // condense() (a false positive through the optional's raw storage); the
    // direct move keeps the semantics and removes the flagged object.
    req_output_token_ids[static_cast<size_t>(empty_index)] =
        std::move(req_output_token_ids[static_cast<size_t>(last_req_index)]);
    req_output_token_ids[static_cast<size_t>(last_req_index)] = std::nullopt;
    req_id_to_index[*req_id] = empty_index;

    const int num_tokens = get_active_token_count(last_req_index);

    std::swap(spec_token_ids[static_cast<size_t>(last_req_index)],
              spec_token_ids[static_cast<size_t>(empty_index)]);
    spec_token_ids[static_cast<size_t>(last_req_index)].clear();

    const size_t empty_row = static_cast<size_t>(empty_index) *
                             static_cast<size_t>(max_model_len);
    const size_t last_row = static_cast<size_t>(last_req_index) *
                            static_cast<size_t>(max_model_len);
    for (int i = 0; i < num_tokens; ++i) {
      token_ids_cpu[empty_row + static_cast<size_t>(i)] =
          token_ids_cpu[last_row + static_cast<size_t>(i)];
    }
    num_tokens_no_spec[static_cast<size_t>(empty_index)] =
        num_tokens_no_spec[static_cast<size_t>(last_req_index)];
    num_prompt_tokens[static_cast<size_t>(empty_index)] =
        num_prompt_tokens[static_cast<size_t>(last_req_index)];
    num_computed_tokens_cpu[static_cast<size_t>(empty_index)] =
        num_computed_tokens_cpu[static_cast<size_t>(last_req_index)];
    // Spec-decode accepted-token count moves with the request
    // (gpu_input_batch.py:787).
    num_accepted_tokens[static_cast<size_t>(empty_index)] =
        num_accepted_tokens[static_cast<size_t>(last_req_index)];
    // Async-scheduling per-slot state moves with the request (keeps it aligned
    // to the dense req_state index combine reads).
    last_sampled_tokens[static_cast<size_t>(empty_index)] =
        last_sampled_tokens[static_cast<size_t>(last_req_index)];
    // W4: the moved VALUE may only exist on the device mirror, so record the
    // move by index and let the device replay it in stream order.
    last_sampled_ops.push_back(
        LastSampledOp{LastSampledOp::kMove, empty_index, last_req_index, 0});
    prefill_len[static_cast<size_t>(empty_index)] =
        prefill_len[static_cast<size_t>(last_req_index)];
    // SPEC-DFLASH2 A2-3 (#2911): the draft row moves with the request, for the
    // same reason as every array above it — the combine's scatter and the async
    // placeholder fill both read it by req_state slot, and a row left behind is
    // the PREVIOUS occupant's drafts read for this request. Upstream needs no
    // equivalent because it never condenses (states.py:132 returns the slot to a
    // free list and the index is stable for the request's life); this is the
    // price of our condensed-dense batch, not a difference in what the state
    // means. Unlike `last_sampled_tokens` the values are host-known here — the
    // propose writes them on the host — so this is a plain row copy and needs no
    // replay-log entry.
    if (num_speculative_steps > 0) {
      const size_t to = static_cast<size_t>(empty_index) *
                        static_cast<size_t>(num_speculative_steps);
      const size_t from = static_cast<size_t>(last_req_index) *
                          static_cast<size_t>(num_speculative_steps);
      for (int c = 0; c < num_speculative_steps; ++c) {
        draft_tokens[to + static_cast<size_t>(c)] =
            draft_tokens[from + static_cast<size_t>(c)];
      }
    }
    num_valid_draft_tokens[static_cast<size_t>(empty_index)] =
        num_valid_draft_tokens[static_cast<size_t>(last_req_index)];
    block_table.move_row(last_req_index, empty_index);

    // Sampling metadata (LoRA / generators / allowed-token-ids / bad-words
    // DEFERRED).
    temperature_cpu[static_cast<size_t>(empty_index)] =
        temperature_cpu[static_cast<size_t>(last_req_index)];
    top_p_cpu[static_cast<size_t>(empty_index)] =
        top_p_cpu[static_cast<size_t>(last_req_index)];
    top_k_cpu[static_cast<size_t>(empty_index)] =
        top_k_cpu[static_cast<size_t>(last_req_index)];
    frequency_penalties_cpu[static_cast<size_t>(empty_index)] =
        frequency_penalties_cpu[static_cast<size_t>(last_req_index)];
    presence_penalties_cpu[static_cast<size_t>(empty_index)] =
        presence_penalties_cpu[static_cast<size_t>(last_req_index)];
    repetition_penalties_cpu[static_cast<size_t>(empty_index)] =
        repetition_penalties_cpu[static_cast<size_t>(last_req_index)];
    seeds[static_cast<size_t>(empty_index)] =
        seeds[static_cast<size_t>(last_req_index)];
    seeds[static_cast<size_t>(last_req_index)] = std::nullopt;

    // ROAD-V1-C7 per-slot controls move with the row (gpu_input_batch.py
    // :819-830). min_p is an array; the index-keyed maps pop-and-reinsert; the
    // allowed-ids mask row is copied then the vacated row cleared. The req_id-
    // keyed predicate sets (min_p_reqs / has_allowed_token_ids / num_logprobs /
    // logprob_token_ids / num_prompt_logprobs) need no move — they survive
    // reindexing.
    min_p_cpu[static_cast<size_t>(empty_index)] =
        min_p_cpu[static_cast<size_t>(last_req_index)];
    MoveDictValue(min_tokens, last_req_index, empty_index);
    MoveDictValue(logit_bias, last_req_index, empty_index);
    MoveDictValue(logits_processors, last_req_index, empty_index);
    MoveDictValue(bad_words_token_ids, last_req_index, empty_index);
    if (!allowed_token_ids_mask.empty()) {
      allowed_token_ids_mask[static_cast<size_t>(empty_index)] =
          allowed_token_ids_mask[static_cast<size_t>(last_req_index)];
      std::fill(
          allowed_token_ids_mask[static_cast<size_t>(last_req_index)].begin(),
          allowed_token_ids_mask[static_cast<size_t>(last_req_index)].end(),
          static_cast<uint8_t>(0));
    }

    // Decrement last_req_index since it is now empty.
    --last_req_index;
  }

  // Trim the dynamic lists to the batch size.
  req_ids.resize(static_cast<size_t>(num));
  req_output_token_ids.resize(static_cast<size_t>(num));
  spec_token_ids.resize(static_cast<size_t>(num));

  // Upstream clears the removed tracking in refresh_metadata (batch_update
  // reset), which is deferred to M1.7; reset it here so any trailing removed
  // indices (trimmed above) do not leak into the next step's bookkeeping.
  removed_tracker_ = RemovedTracker();
}

void InputBatch::swap_states(int i1, int i2) {
  // Port of gpu_input_batch.py::swap_states (@ e24d1b24). See input_batch.h for
  // the deferred (T0-empty) fields skipped here.
  if (i1 == i2) {
    return;
  }
  sampling_metadata_dirty_ = true;  // reorders the dense prefix -> rebuild
  const std::optional<std::string> old_id_i1 = req_ids[static_cast<size_t>(i1)];
  const std::optional<std::string> old_id_i2 = req_ids[static_cast<size_t>(i2)];

  // Only swap the active token prefix (max active count of the two rows).
  const int max_active = std::max(get_active_token_count(i1),
                                  get_active_token_count(i2));

  std::swap(req_ids[static_cast<size_t>(i1)], req_ids[static_cast<size_t>(i2)]);
  std::swap(req_output_token_ids[static_cast<size_t>(i1)],
            req_output_token_ids[static_cast<size_t>(i2)]);
  std::swap(spec_token_ids[static_cast<size_t>(i1)],
            spec_token_ids[static_cast<size_t>(i2)]);

  if (old_id_i1.has_value()) {
    req_id_to_index[*old_id_i1] = i2;
  }
  if (old_id_i2.has_value()) {
    req_id_to_index[*old_id_i2] = i1;
  }

  std::swap(num_tokens_no_spec[static_cast<size_t>(i1)],
            num_tokens_no_spec[static_cast<size_t>(i2)]);
  std::swap(num_prompt_tokens[static_cast<size_t>(i1)],
            num_prompt_tokens[static_cast<size_t>(i2)]);
  std::swap(num_computed_tokens_cpu[static_cast<size_t>(i1)],
            num_computed_tokens_cpu[static_cast<size_t>(i2)]);
  // Spec-decode accepted-token count (gpu_input_batch.py:662).
  std::swap(num_accepted_tokens[static_cast<size_t>(i1)],
            num_accepted_tokens[static_cast<size_t>(i2)]);
  // Async-scheduling per-slot state (moves with the row in the decode-first
  // reorder, so combine's dense req_state index stays correct).
  std::swap(last_sampled_tokens[static_cast<size_t>(i1)],
            last_sampled_tokens[static_cast<size_t>(i2)]);
  // W4: same reasoning as the condense move — indices, not values.
  last_sampled_ops.push_back(LastSampledOp{LastSampledOp::kSwap, i1, i2, 0});
  std::swap(prefill_len[static_cast<size_t>(i1)],
            prefill_len[static_cast<size_t>(i2)]);
  // SPEC-DFLASH2 A2-3 (#2911): the draft rows swap with their requests. The
  // decode-first reorder runs on a SPECULATIVE step by construction (it is the
  // spec-as-decode split), so this is the reorder most likely to move a row that
  // actually carries drafts.
  if (num_speculative_steps > 0) {
    const size_t d1 =
        static_cast<size_t>(i1) * static_cast<size_t>(num_speculative_steps);
    const size_t d2 =
        static_cast<size_t>(i2) * static_cast<size_t>(num_speculative_steps);
    for (int c = 0; c < num_speculative_steps; ++c) {
      std::swap(draft_tokens[d1 + static_cast<size_t>(c)],
                draft_tokens[d2 + static_cast<size_t>(c)]);
    }
  }
  std::swap(num_valid_draft_tokens[static_cast<size_t>(i1)],
            num_valid_draft_tokens[static_cast<size_t>(i2)]);

  // Swap the active token prefix of the two rows (upstream copies only
  // max_active_token_count columns).
  const size_t row1 =
      static_cast<size_t>(i1) * static_cast<size_t>(max_model_len);
  const size_t row2 =
      static_cast<size_t>(i2) * static_cast<size_t>(max_model_len);
  for (int c = 0; c < max_active; ++c) {
    std::swap(token_ids_cpu[row1 + static_cast<size_t>(c)],
              token_ids_cpu[row2 + static_cast<size_t>(c)]);
  }

  block_table.swap_row(i1, i2);

  // Sampling params (autoregressive models; pooling DEFERRED).
  std::swap(temperature_cpu[static_cast<size_t>(i1)],
            temperature_cpu[static_cast<size_t>(i2)]);
  std::swap(top_p_cpu[static_cast<size_t>(i1)],
            top_p_cpu[static_cast<size_t>(i2)]);
  std::swap(top_k_cpu[static_cast<size_t>(i1)],
            top_k_cpu[static_cast<size_t>(i2)]);
  std::swap(frequency_penalties_cpu[static_cast<size_t>(i1)],
            frequency_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(presence_penalties_cpu[static_cast<size_t>(i1)],
            presence_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(repetition_penalties_cpu[static_cast<size_t>(i1)],
            repetition_penalties_cpu[static_cast<size_t>(i2)]);
  std::swap(seeds[static_cast<size_t>(i1)], seeds[static_cast<size_t>(i2)]);

  // ROAD-V1-C7 per-slot controls (gpu_input_batch.py:686-700). min_p is an
  // array swap; the index-keyed maps use swap_dict_values; the allowed-ids mask
  // rows swap. The req_id-keyed predicate sets are keyed by id, so no swap.
  std::swap(min_p_cpu[static_cast<size_t>(i1)],
            min_p_cpu[static_cast<size_t>(i2)]);
  SwapDictValues(min_tokens, i1, i2);
  SwapDictValues(logit_bias, i1, i2);
  SwapDictValues(logits_processors, i1, i2);
  SwapDictValues(bad_words_token_ids, i1, i2);
  if (!allowed_token_ids_mask.empty()) {
    std::swap(allowed_token_ids_mask[static_cast<size_t>(i1)],
              allowed_token_ids_mask[static_cast<size_t>(i2)]);
  }
}

void InputBatch::update_req_spec_token_ids(
    int req_index, const std::string& req_id,
    const std::map<std::string, std::vector<int32_t>>& scheduled_spec_tokens) {
  // gpu_input_batch.py:484-509. Clear any prior drafts first: when spec decoding
  // is combined with structured output the scheduler can drop non-conforming
  // drafts, leaving scheduled_spec_decode_tokens empty for this request even
  // though speculation is on.
  std::vector<int32_t>& cur_spec_token_ids =
      spec_token_ids[static_cast<size_t>(req_index)];
  cur_spec_token_ids.clear();

  auto it = scheduled_spec_tokens.find(req_id);
  if (it == scheduled_spec_tokens.end() || it->second.empty()) {
    return;
  }
  const std::vector<int32_t>& ids = it->second;

  // Splice the draft ids into token_ids_cpu after the non-spec token prefix. For
  // async scheduling these are placeholders that _prepare_input_ids overwrites;
  // for the synchronous path they are the actual verify inputs. (is_token_ids /
  // multimodal token-vs-embed tracking is DEFERRED — T0 is always token ids.)
  const int start_index = num_tokens_no_spec[static_cast<size_t>(req_index)];
  const size_t row =
      static_cast<size_t>(req_index) * static_cast<size_t>(max_model_len);
  for (size_t i = 0; i < ids.size(); ++i) {
    token_ids_cpu[row + static_cast<size_t>(start_index) + i] = ids[i];
  }
  cur_spec_token_ids.insert(cur_spec_token_ids.end(), ids.begin(), ids.end());
}

}  // namespace vllm::v1
