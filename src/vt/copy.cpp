#include "vt/ops.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace vt {
namespace {
bool Float(DType d) { return d == DType::kF16 || d == DType::kBF16 || d == DType::kF32; }
int64_t Offset(const Tensor& t, int64_t i) {
  int64_t offset = 0;
  for (int d = t.rank - 1; d >= 0; --d) { offset += (i % t.shape[d]) * t.stride[d]; i /= t.shape[d]; }
  return offset;
}
void CpuCopy(Queue&, Tensor& out, const Tensor& in) {
  const size_t in_size = SizeOf(in.dtype), out_size = SizeOf(out.dtype);
  // Snapshot also covers transposed aliases and in-place widening/narrowing.
  std::vector<unsigned char> snapshot(in.Numel() * in_size);
  for (int64_t i = 0; i < in.Numel(); ++i)
    std::memcpy(snapshot.data() + i * in_size,
                static_cast<const char*>(in.data) + Offset(in, i) * in_size, in_size);
  for (int64_t i = 0; i < out.Numel(); ++i) {
    auto* dst = static_cast<char*>(out.data) + Offset(out, i) * out_size;
    const auto* src = snapshot.data() + i * in_size;
    if (out.dtype == in.dtype) { std::memcpy(dst, src, out_size); continue; }
    float f;
    if (in.dtype == DType::kF32) std::memcpy(&f, src, 4);
    else { uint16_t v; std::memcpy(&v, src, 2); f = in.dtype == DType::kF16 ? F16ToF32(v) : BF16ToF32(v); }
    if (out.dtype == DType::kF32) std::memcpy(dst, &f, 4);
    else { uint16_t v = out.dtype == DType::kF16 ? F32ToF16(f) : F32ToBF16(f); std::memcpy(dst, &v, 2); }
  }
}
struct Registrar {
  Registrar() { RegisterOp(OpId::kCopy, DeviceType::kCPU, reinterpret_cast<void*>(static_cast<CopyFn>(&CpuCopy))); }
};
[[maybe_unused]] Registrar registrar;
}
void Copy(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.device == q.device && in.device == q.device, "copy: device mismatch");
  VT_CHECK(out.rank == in.rank && in.rank >= 0 && in.rank <= kMaxRank, "copy: rank mismatch");
  VT_CHECK(!IsBlockQuant(in.dtype) && !IsBlockQuant(out.dtype), "copy: elementwise dtypes required");
  VT_CHECK(in.dtype == out.dtype || (Float(in.dtype) && Float(out.dtype)), "copy: unsupported conversion");
  std::vector<std::pair<int64_t, int64_t>> dims;
  for (int d = 0; d < in.rank; ++d) {
    VT_CHECK(in.shape[d] >= 0 && out.shape[d] == in.shape[d], "copy: shape mismatch");
    VT_CHECK(in.stride[d] >= 0 && out.stride[d] >= 0, "copy: negative stride");
    if (out.shape[d] > 1) dims.emplace_back(out.stride[d], out.shape[d]);
  }
  int64_t span = 1;
  std::sort(dims.begin(), dims.end());
  for (auto [stride, size] : dims) {
    VT_CHECK(stride >= span, "copy: overlapping output elements");
    VT_CHECK(stride <= (std::numeric_limits<int64_t>::max() - span) / (size - 1), "copy: stride overflow");
    span += (size - 1) * stride;
  }
  reinterpret_cast<CopyFn>(GetOp(OpId::kCopy, q.device.type))(q, out, in);
}
}  // namespace vt
