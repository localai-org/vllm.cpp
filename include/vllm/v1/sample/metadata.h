// Ported from: vllm/v1/sample/metadata.py @ e24d1b24
//
// SamplingMetadata — the per-slot sampling state the V1 Sampler (M1.7) consumes,
// built once per step by InputBatch::make_sampling_metadata() (the port of
// gpu_input_batch.py::_make_sampling_metadata). Field NAMES mirror upstream 1:1.
//
// ─── T0 field subset ────────────────────────────────────────────────────────
// The scalar/vector fields are the num_reqs-dense slices of the InputBatch
// per-slot arrays. Upstream keeps them as device tensors sliced `[:num_reqs]`;
// here each is a host std::vector already truncated to num_reqs. The
// None-when-not-needed optionals (temperature/top_p/top_k/prompt_token_ids/
// allowed_token_ids_mask) preserve upstream's "skip the copy" semantics so the
// sampler's branching ports unchanged.
//
// ─── logitsprocs → flat fields (recorded deviation) ─────────────────────────
// Upstream carries a `logitsprocs: LogitsProcessors` plugin object graph plus
// `logprob_token_ids`, `spec_token_ids`, `thinking_budget_state_holder`. We do
// NOT port the plugin interface; instead the three T0 builtins
// (vllm/v1/sample/logits_processor/builtin.py) are represented as flat inputs —
// `min_tokens`, `logit_bias`, `min_p` — since Task 3 ports the three builtins
// directly as functions rather than as a plugin dispatch. The remaining plugin
// members are marked stubs below (defaulted empty/None) with their upstream cite.
#ifndef VLLM_V1_SAMPLE_METADATA_H_
#define VLLM_V1_SAMPLE_METADATA_H_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "vllm/logits_processor_callback.h"

namespace vllm::v1 {

// Per-request min-tokens state (the flattened MinTokensLogitsProcessor input,
// vllm/v1/sample/logits_processor/builtin.py::MinTokensLogitsProcessor). While
// `output_len < min_tokens`, the sampler masks every id in `stop_token_ids`
// (eos + stop_token_ids, i.e. upstream `params.all_stop_token_ids`) to -inf.
struct MinTokensState {
  int min_tokens = 0;
  std::set<int32_t> stop_token_ids;
};

// SamplingMetadata (vllm/v1/sample/metadata.py::SamplingMetadata) — T0 subset.
struct SamplingMetadata {
  // None when all_greedy (upstream skips the temperature copy). Else [num_reqs].
  std::optional<std::vector<float>> temperature;
  bool all_greedy = true;
  bool all_random = false;

  // None when no_top_p / no_top_k. Else [num_reqs].
  std::optional<std::vector<float>> top_p;
  std::optional<std::vector<int32_t>> top_k;

  // req_index -> per-request RNG seed. Upstream is `dict[int, torch.Generator]`;
  // we don't have torch.Generator, so we carry the seed (the actual seeded RNG
  // lands in Task 2's random_sample). Requests without their own seed are absent
  // from the map (upstream NOTE at gpu_input_batch.py:251-252).
  std::map<int, uint64_t> generators;

  // Accepted output position for each live request. Seeded draws are keyed by
  // (request seed, position), independent of batch row, chunking and reordering.
  // Empty is position zero for direct, single-step sampler callers.
  std::vector<uint64_t> output_token_positions;

  // None => no logprobs; 0 => sampled-token logprob only; k => top-k; -1 => all.
  std::optional<int> max_num_logprobs;

  bool no_penalties = true;
  // None unless penalties (or a token-id-consuming proc) need it. Ragged per-req
  // prompt token ids (upstream: a padded [num_reqs, max_prompt_len] i32 tensor).
  std::optional<std::vector<std::vector<int32_t>>> prompt_token_ids;
  // [num_reqs] each (dense slices of the InputBatch penalty arrays).
  std::vector<float> frequency_penalties;
  std::vector<float> presence_penalties;
  std::vector<float> repetition_penalties;

  // Per-request generated tokens so far (empty when no proc needs them, matching
  // upstream's needs_output_token_ids gate).
  std::vector<std::vector<int32_t>> output_token_ids;

  // None unless a request restricts allowed ids. Upstream is a 2D bool tensor
  // [num_reqs, vocab]; represented here as row-major bool rows [num_reqs][vocab].
  std::optional<std::vector<std::vector<uint8_t>>> allowed_token_ids_mask;

  // req_index -> list of bad-words token-id n-grams
  // (vllm/v1/sample/ops/bad_words.py::apply_bad_words input).
  std::map<int, std::vector<std::vector<int32_t>>> bad_words_token_ids;

  // ─── T0 builtin logits-processor inputs (flat; see header deviation) ───────
  // req_index -> min-tokens state (MinTokensLogitsProcessor).
  std::map<int, MinTokensState> min_tokens;
  // req_index -> (token_id -> additive bias) (LogitBiasLogitsProcessor).
  std::map<int, std::map<int32_t, float>> logit_bias;
  // [num_reqs] min-p thresholds (MinPLogitsProcessor); 0 disables per row.
  std::vector<float> min_p;

  // req_index -> custom host logits-processor callback (ROAD-V1-C7
  // `custom_logit_processor`). Empty => none (the byte-identical default). Each
  // callback is invoked once per decode step, in the sampler's non-argmax-
  // invariant logits-processor stage (after min_tokens / logit_bias, before
  // penalties — mirroring vLLM's custom-logitsproc application order,
  // sampler.py:399), over this request's mutable logits row. Requests with no
  // registered processor are absent from the map. See logits_processor_callback.h.
  std::map<int, LogitsProcessorCallback> logits_processors;

  // logprob_token_ids (metadata.py:49, `dict[int, list[int]] | None`): req_INDEX
  // -> the EXPLICIT vocab ids to score, for generative scoring. The sampler
  // returns the sampled token plus exactly these ids instead of a top-k, and
  // when a request ALSO carries a logprobs count the explicit ids WIN
  // (sampler.py:133-136). Fed by InputBatch::make_sampling_metadata, which
  // re-keys its own req_id-keyed map. Unset — or set-but-empty, both falsy
  // where upstream tests it — means the ordinary top-k path.
  std::optional<std::map<int, std::vector<int32_t>>> logprob_token_ids;

  // ─── STUBS (marked; defaulted empty/None at T0) ────────────────────────────
  // Upstream `logitsprocs: LogitsProcessors` plugin graph — NOT ported (the
  // three T0 builtins are the flat fields above). No field.
  //
  // Upstream `spec_token_ids: list[list[int]] | None` (speculative decode).
  // Always empty lists at T0 (kept so the sampler's spec branch ports).
  std::optional<std::vector<std::vector<int32_t>>> spec_token_ids;
  // Upstream `thinking_budget_state_holder` — deferred; no field/flag ported.
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_METADATA_H_
