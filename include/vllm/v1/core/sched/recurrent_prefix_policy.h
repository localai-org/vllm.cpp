#ifndef VLLM_V1_CORE_SCHED_RECURRENT_PREFIX_POLICY_H_
#define VLLM_V1_CORE_SCHED_RECURRENT_PREFIX_POLICY_H_

#include <stdexcept>
#include <string_view>

namespace vllm::v1 {

enum class RecurrentPrefixAlignmentMode { kOff, kAuto, kForce };

// Preserve the explicit 0/1 rollback controls. Unset selects auto; invalid
// spellings must not silently select a different scheduling policy.
inline RecurrentPrefixAlignmentMode ParseRecurrentPrefixAlignmentMode(
    const char* value) {
  if (!value || std::string_view(value) == "auto")
    return RecurrentPrefixAlignmentMode::kAuto;
  if (std::string_view(value) == "off" || std::string_view(value) == "0")
    return RecurrentPrefixAlignmentMode::kOff;
  if (std::string_view(value) == "force" || std::string_view(value) == "1")
    return RecurrentPrefixAlignmentMode::kForce;
  throw std::invalid_argument("VT_RECURRENT_PREFIX_ALIGN must be auto, force/1, or off/0");
}

// Use the ORIGINAL prompt length and the configured per-step token capacity.
// Do not use the current free batch budget, remaining prompt, or a concurrency-
// dependent long-prefill threshold. Short prompts may lose a forced warm hit;
// force remains available for cache-heavy short-prompt workloads.
inline bool ShouldAlignRecurrentPrefix(RecurrentPrefixAlignmentMode mode,
                                       int prompt_tokens,
                                       int nominal_chunk_tokens) {
  return mode == RecurrentPrefixAlignmentMode::kForce ||
         (mode == RecurrentPrefixAlignmentMode::kAuto &&
          nominal_chunk_tokens > 0 && prompt_tokens > nominal_chunk_tokens);
}

// Reach the longest prefix the existing coordinator may safely restore.
// Recompute the final prompt token; shared draft pages require one more page.
// Only shorten the chunk that crosses this single boundary. Never manufacture
// intermediate recurrent state or enlarge the caller's token/encoder budget.
inline int ClampRecurrentPrefixPublication(int prompt_tokens, int computed,
                                          int candidate, int block_tokens,
                                          bool drop_draft_page) {
  if (prompt_tokens <= 0 || computed < 0 || candidate <= 0 || block_tokens <= 0)
    return candidate;
  const int pages = (prompt_tokens - 1) / block_tokens -
                    (drop_draft_page ? 1 : 0);
  if (pages <= 0) return candidate;
  const int boundary = pages * block_tokens;
  if (computed < boundary && candidate > boundary - computed)
    return boundary - computed;
  return candidate;
}

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SCHED_RECURRENT_PREFIX_POLICY_H_
