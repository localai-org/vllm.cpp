#ifndef VLLM_V1_ATTENTION_METADATA_VALIDATION_H_
#define VLLM_V1_ATTENTION_METADATA_VALIDATION_H_

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/backend.h"

namespace vllm::v1 {

// Check the host arrays before an upload uses the declared tensor shapes. CPU
// mirror fields are optional here: graph padding only refreshes device fields.
inline void ValidateAttentionUploadMetadata(const CommonAttentionMetadata& am,
                                            int64_t tokens) {
  if (tokens <= 0 || am.num_actual_tokens != tokens || am.num_reqs <= 0 ||
      am.block_table_num_cols <= 0 ||
      static_cast<int64_t>(am.slot_mapping.size()) != tokens ||
      static_cast<int64_t>(am.seq_lens.size()) != am.num_reqs ||
      static_cast<int64_t>(am.query_start_loc.size()) != am.num_reqs + 1 ||
      static_cast<int64_t>(am.block_table_tensor.size()) !=
          static_cast<int64_t>(am.num_reqs) * am.block_table_num_cols)
    throw std::invalid_argument("attention metadata: host upload shape mismatch");
  if (am.query_start_loc.front() != 0 || am.query_start_loc.back() != tokens)
    throw std::invalid_argument("attention metadata: query offsets do not span tokens");
  for (int request = 0; request < am.num_reqs; ++request) {
    const int first = am.query_start_loc[request];
    const int end = am.query_start_loc[request + 1];
    if (first < 0 || end < first || end > tokens || am.seq_lens[request] < end - first)
      throw std::invalid_argument("attention metadata: invalid query or sequence length");
  }
  for (const int64_t slot : am.slot_mapping)
    if (slot < -1)
      throw std::invalid_argument("attention metadata: invalid KV write slot");
}

// A text request may use permuted or shared physical pages. For every active
// write, the physical slot must agree with the read table at that token's
// logical position. -1 is a valid skipped write (for example, inert padding).
inline void ValidateAttentionPageSlots(const CommonAttentionMetadata& am,
                                       const std::vector<int32_t>& positions,
                                       int64_t block_size, int64_t num_blocks) {
  ValidateAttentionUploadMetadata(am, static_cast<int64_t>(positions.size()));
  if (block_size <= 0 || num_blocks <= 0)
    throw std::invalid_argument("attention metadata: invalid KV cache capacity");
  for (int request = 0; request < am.num_reqs; ++request) {
    const int64_t length = am.seq_lens[request];
    const int first = am.query_start_loc[request];
    const int end = am.query_start_loc[request + 1];
    if (first == end) continue;  // Unscheduled rows are never read or written.
    if (length > static_cast<int64_t>(am.block_table_num_cols) * block_size)
      throw std::invalid_argument("attention metadata: read table lacks visible pages");
    const int64_t row = static_cast<int64_t>(request) * am.block_table_num_cols;
    for (int64_t page = 0; page < (length + block_size - 1) / block_size; ++page) {
      const int32_t block = am.block_table_tensor[row + page];
      if (block < 0 || block >= num_blocks)
        throw std::invalid_argument("attention metadata: read page outside KV cache");
    }
    for (int token = first; token < end; ++token) {
      const int64_t slot = am.slot_mapping[token];
      if (slot == -1) continue;
      // PagedAttention derives its cache position from the request's sequence
      // length and query offsets. RoPE positions may have a different origin.
      const int64_t position = length - (end - first) + token - first;
      const int64_t page = position / block_size;
      const int64_t expected =
          static_cast<int64_t>(am.block_table_tensor[row + page]) * block_size +
          position % block_size;
      if (slot != expected)
        throw std::invalid_argument("attention metadata: KV read/write slot mismatch");
    }
  }
}

}  // namespace vllm::v1

#endif  // VLLM_V1_ATTENTION_METADATA_VALIDATION_H_
