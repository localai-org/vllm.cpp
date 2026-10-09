#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vt {
// Pinned FP16 producer geometry: Hk16/Hv48/D128/chunk64, up to four sequences.
// Physical capacity includes each sequence's 63-token virtual tail. This is not
// permission to consume padded tokens or reset an existing state.
inline constexpr int kGdnFp16MaxTokens = 4096;
inline constexpr size_t kGdnFp16ReservationBytes = 220 * 1024 * 1024;
struct GdnFp16C1Plan {
  int tokens, capacity;
  size_t q_offset, k_offset, v_offset, a_matrix_offset, w_offset, u_offset;
  size_t raw_a_offset, beta_offset, bias_offset, index_offset, initial_offset;
  size_t bytes;
};

inline GdnFp16C1Plan PlanGdnFp16Batch(int64_t tokens, int64_t sequences) {
  if (sequences < 1 || sequences > 4 || tokens < sequences || tokens > kGdnFp16MaxTokens)
    throw std::invalid_argument("GDN FP16 requires 1-4 nonempty sequences and at most4096 tokens");
  const int capacity = int(tokens) + int(sequences) * 63;
  size_t cursor = 0;
  auto take = [&](size_t bytes) {
    const auto offset = cursor;
    cursor += (bytes + 63) / 64 * 64;
    return offset;
  };
  const size_t q_bytes = size_t(capacity) * 16 * 128 * 2;
  const size_t v_bytes = size_t(capacity) * 48 * 128 * 2;
  const size_t matrix_bytes = size_t(capacity) * 48 * 64 * 2;
  const size_t wu_bytes = size_t(capacity) * 48 * 128 * 2;
  const size_t gate_bytes = size_t(capacity) * 48 * 4;
  const auto q = take(q_bytes), k = take(q_bytes), v = take(v_bytes);
  const auto a_matrix = take(matrix_bytes), w = take(wu_bytes), u = take(wu_bytes);
  const auto raw_a = take(gate_bytes), beta = take(gate_bytes);
  const auto bias = take(48 * 2), index = take(size_t(sequences) * sizeof(int32_t)),
             initial = take(size_t(sequences));
  if (cursor > kGdnFp16ReservationBytes)
    throw std::logic_error("GDN FP16 C1 layout exceeds the stable reservation");
  return {int(tokens), capacity, q, k, v, a_matrix, w, u, raw_a, beta,
          bias, index, initial, cursor};
}

inline GdnFp16C1Plan PlanGdnFp16C1(int64_t tokens) {
  return PlanGdnFp16Batch(tokens, 1);
}
}  // namespace vt
