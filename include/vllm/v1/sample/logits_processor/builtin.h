// Ported from: vllm/v1/sample/logits_processor/builtin.py @ e24d1b24
// (MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor).
//
// The three T0 builtin logits processors, ported as direct functions over the
// logits [num_reqs, vocab] (row-major f32) rather than the plugin object graph
// (see the metadata.h deviation note). Each carries its upstream
// `is_argmax_invariant()` as a constexpr — this drives the Sampler pipeline
// placement (M1.7 Task 4): non-argmax-invariant procs run BEFORE the greedy
// argmax snapshot; argmax-invariant procs run inside sample() after temperature.
#ifndef VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_
#define VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_

#include <cstdint>
#include <map>
#include <vector>

#include "vllm/v1/sample/metadata.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// MinTokensLogitsProcessor.is_argmax_invariant() -> False (censoring stop tokens
// can change the greedy argmax).
inline constexpr bool kMinTokensArgmaxInvariant = false;
// LogitBiasLogitsProcessor.is_argmax_invariant() -> False (bias can rebalance the
// argmax).
inline constexpr bool kLogitBiasArgmaxInvariant = false;
// MinPLogitsProcessor.is_argmax_invariant() -> True (min-p never masks the
// max-prob token, so greedy is unaffected).
inline constexpr bool kMinPArgmaxInvariant = true;

// MinTokensLogitsProcessor.apply. For each tracked request whose generated
// length is still below its min_tokens floor, mask every id in its stop-token
// set (eos + stop_token_ids) to -inf. `output_token_ids[i]` gives the generated
// length for request i.
void apply_min_tokens(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, MinTokensState>& min_tokens,
                      const std::vector<std::vector<int32_t>>& output_token_ids);

// LogitBiasLogitsProcessor.apply. Adds the per-(request, token) additive bias:
// logits[i, tok] += bias for every entry in `logit_bias`.
void apply_logit_bias(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, std::map<int32_t, float>>& logit_bias);

// Active GPU sampler's expanded verification rows: allowed-token masking,
// logit bias, then the min-token stop mask. `cu_num_logits` maps requests to rows;
// `output_token_positions[r] + row_depth` is the provisional output length.
void apply_speculative_logit_filters(
    vt::Queue& q, vt::Tensor& logits, const SamplingMetadata& metadata,
    const std::vector<int32_t>& cu_num_logits);

// Full target processor order for verification. Row depth d sees the committed
// output history followed by exactly d provisional drafts; the anchor at each
// request's first row is already in the committed history and is not appended.
// EOS masking uses accepted output positions, rather than an optimistic length.
void apply_speculative_logits_processors(
    vt::Queue& q, vt::Tensor& logits, const SamplingMetadata& metadata,
    const std::vector<int32_t>& cu_num_logits,
    const std::vector<int32_t>& draft_input_ids);

// MinPLogitsProcessor.apply. Per row: mask tokens whose softmax probability is
// below min_p[i] * max_prob to -inf. Rows with min_p[i] == 0 are unaffected.
void apply_min_p(vt::Queue& q, vt::Tensor& logits, const std::vector<float>& min_p);

// Custom host logits processors (ROAD-V1-C7 `custom_logit_processor`). For each
// (req_index -> callback) entry, invoke the callback once over the request's
// mutable logits row [vocab], passing the request's generated output token ids so
// far. Non-argmax-invariant: runs at the end of the non-argmax-invariant stage
// (after min_tokens / logit_bias, before penalties), mirroring vLLM's
// custom-logitsproc placement (sampler.py:399). The callbacks read+mutate logits
// on the HOST: this syncs the queue first, then (on a non-unified backend) stages
// the logits down to host memory, runs the callbacks, and copies the edited
// logits back. On a unified-memory backend (CPU / GB10) the row is edited in
// place with no staging copy. Empty `procs` => no-op (byte-identical default).
void apply_logits_processors(
    vt::Queue& q, vt::Tensor& logits,
    const std::map<int, LogitsProcessorCallback>& procs,
    const std::vector<std::vector<int32_t>>& output_token_ids);

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_LOGITS_PROCESSOR_BUILTIN_H_
