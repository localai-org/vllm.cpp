// Internal MTP materialization seam; non-owning views keep the caller's DBufs.
#pragma once
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/concat_plan.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu_copy.h"
#endif
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace vllm::detail {
inline constexpr const char* kMtpConcatStridedDefault = "0";
inline bool MtpConcatStridedEnabled() {
  const char* value = std::getenv("VLLM_QWEN35_MTP_CONCAT_STRIDED");
  if (!value) value = kMtpConcatStridedDefault;
  VT_CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
           "VLLM_QWEN35_MTP_CONCAT_STRIDED must be 0/1");
  return std::strcmp(value, "1") == 0;
}

// The sole typed-packing alternative. Both switches are diagnostic; packing
// takes precedence when explicitly enabled. Small rows always keep old copies.
inline constexpr const char* kMtpConcatPackDefault = "0";
inline bool MtpConcatPackEnabled() {
  const char* value = std::getenv("VLLM_QWEN35_MTP_CONCAT_PACK");
  if (!value) value = kMtpConcatPackDefault;
  VT_CHECK(std::strcmp(value,"0") == 0 || std::strcmp(value,"1") == 0,
           "VLLM_QWEN35_MTP_CONCAT_PACK must be 0/1");
  return std::strcmp(value,"1") == 0;
}

// Validate all three descriptors and aliases before the first write. The old
// model seam produces contiguous tensors; do not invent a strided-input API.
// Both routes preserve same-dtype raw storage, including FP16 NaN payloads.
inline bool ConcatMtpNormalized(vt::Backend& backend, vt::Queue& q, vt::Tensor& out,
                                const vt::Tensor& left,
                                const vt::Tensor& right, bool use_strided, bool use_pack = false) {
  const auto layout = vt::ValidateConcatRows(q, out, left, right);
  const auto rows = layout.rows, hidden = layout.hidden;
  const auto row_bytes = layout.row_bytes;
  if (!rows) return false;
  const bool admitted = (use_strided || use_pack) && q.device.type == vt::DeviceType::kXPU &&
      left.dtype == vt::DType::kF16 && hidden == 5120 && rows >= 512 && rows <= 4096;
  auto* cat = static_cast<unsigned char*>(out.data);
#ifdef VLLM_CPP_XPU
  if (admitted && use_pack) {
    vt::xpu::ConcatRowsF16(q, out, left, right);
    return true;
  }
#endif
  if (admitted && use_strided) {
    auto first = left;
    first.data = cat;
    first.stride[0] = 2 * hidden;
    auto second = first;
    second.data = cat + row_bytes;
    vt::Copy(q, first, left);
    vt::Copy(q, second, right);
    return true;
  }
  for (int64_t row = 0; row < rows; ++row) {
    const size_t offset = size_t(row) * row_bytes;
    backend.Copy(q, cat + 2 * offset,
                 static_cast<const unsigned char*>(left.data) + offset, row_bytes);
    backend.Copy(q, cat + 2 * offset + row_bytes,
                 static_cast<const unsigned char*>(right.data) + offset, row_bytes);
  }
  return false;
}

// Programmatic isolated diagnostics only. Ordinary serving leaves this null.
// The callback owns any explicit readback/wait; the product seam adds neither.
using MtpHeadTensorObserver = void (*)(vt::Queue&, const char*, const vt::Tensor&);
inline thread_local MtpHeadTensorObserver mtp_head_tensor_observer = nullptr;
class MtpHeadTensorObserverScope {
 public:
  explicit MtpHeadTensorObserverScope(MtpHeadTensorObserver observer)
      : old_(mtp_head_tensor_observer) { mtp_head_tensor_observer = observer; }
  ~MtpHeadTensorObserverScope() { mtp_head_tensor_observer = old_; }
  MtpHeadTensorObserverScope(const MtpHeadTensorObserverScope&) = delete;
  MtpHeadTensorObserverScope& operator=(const MtpHeadTensorObserverScope&) = delete;
 private:
  MtpHeadTensorObserver old_;
};
}  // namespace vllm::detail
