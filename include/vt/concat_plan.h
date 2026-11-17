// Checked contiguous-row concatenation descriptors shared by portable model glue
// and typed VT device packing. No allocation, command submission or writes.
#pragma once
#include "vt/tensor.h"
#include <cstddef>
#include <cstdint>
#include <limits>
namespace vt {
struct ConcatRowsLayout { int64_t rows, hidden; size_t row_bytes, input_bytes; };
inline ConcatRowsLayout ValidateConcatRows(const Queue& q, const Tensor& out,
                                          const Tensor& left, const Tensor& right) {
  VT_CHECK(out.device == q.device && left.device == q.device && right.device == q.device,
           "concat rows: device mismatch");
  VT_CHECK(out.rank == 2 && left.rank == 2 && right.rank == 2,
           "concat rows: rank must be two");
  VT_CHECK(left.dtype == right.dtype && out.dtype == left.dtype &&
               (left.dtype == vt::DType::kF16 || left.dtype == vt::DType::kBF16 ||
                left.dtype == vt::DType::kF32), "concat rows: floating storage mismatch");
  const int64_t rows = left.shape[0], hidden = left.shape[1];
  VT_CHECK(rows >= 0 && hidden > 0 && hidden <= std::numeric_limits<int64_t>::max() / 2,
           "concat rows: invalid shape or width overflow");
  VT_CHECK(right.shape[0] == rows && right.shape[1] == hidden &&
               out.shape[0] == rows && out.shape[1] == 2 * hidden,
           "concat rows: incompatible shapes");
  const auto element_bytes = vt::SizeOf(left.dtype);
  // Also bound Tensor::Numel's signed arithmetic before IsContiguous/Bytes.
  VT_CHECK(uint64_t(rows) <= uint64_t(std::numeric_limits<int64_t>::max()) / uint64_t(2 * hidden) &&
               uint64_t(2 * hidden) <= std::numeric_limits<size_t>::max() / element_bytes &&
               uint64_t(rows) <= std::numeric_limits<size_t>::max() / element_bytes / uint64_t(2 * hidden),
           "concat rows: byte size overflow");
  VT_CHECK(left.IsContiguous() && right.IsContiguous() && out.IsContiguous(),
           "concat rows: contiguous inputs and destination required");
  if (!rows) return {rows, hidden, 0, 0};
  VT_CHECK(left.data && right.data && out.data, "concat rows: null storage");
  const size_t row_bytes = size_t(hidden) * element_bytes;
  const size_t input_bytes = size_t(rows) * row_bytes;
  const auto start = [](const void* data, size_t bytes) {
    const auto value = reinterpret_cast<uintptr_t>(data);
    VT_CHECK(value <= std::numeric_limits<uintptr_t>::max() - bytes,
             "concat rows: address range overflow");
    return value;
  };
  const auto dst = start(out.data, 2 * input_bytes);
  const auto a = start(left.data, input_bytes), b = start(right.data, input_bytes);
  VT_CHECK((dst + 2 * input_bytes <= a || a + input_bytes <= dst) &&
               (dst + 2 * input_bytes <= b || b + input_bytes <= dst),
           "concat rows: output aliases normalized input");
  return {rows, hidden, row_bytes, input_bytes};
}
}  // namespace vt
