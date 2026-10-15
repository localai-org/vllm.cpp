// See `include/vllm/model_executor/models/dense_exl3_linear.h`. This
// translation unit exists so that the scheme header, and through it
// `dense_attn_block.h`, is included HERE and not in `qwen3_5.cpp`.
#include "vllm/model_executor/models/dense_exl3_linear.h"

#include <cstring>
#include <limits>

#include "vllm/model_executor/layers/quantization/exl3.h"

namespace vllm {
Exl3GroupedWeight MergeExl3Weights(const std::vector<const Exl3Weight*>& sources,
                                  std::string name) {
  VT_CHECK(!sources.empty() && sources.size() <= static_cast<size_t>(INT32_MAX),
           "exl3 merge: invalid source count");
  VT_CHECK(sources.front() != nullptr, "exl3 merge: null source");
  const int bits = sources.front()->Bits();
  const int64_t kt = sources.front()->trellis.shape[0];
  VT_CHECK(kt > 0 && kt <= INT32_MAX / 16 && kt % 8 == 0 && (bits == 4 || bits == 6),
           "exl3 merge: requires aligned K and mul1 4/6bpw");
  const int64_t k = kt * 16;
  Exl3GroupedWeight merged;
  merged.name = std::move(name);
  merged.codebook = 2;
  merged.output_offsets.push_back(0);
  for (const Exl3Weight* w : sources) {
    VT_CHECK(w != nullptr, "exl3 merge: null source");
    VT_CHECK(w->trellis.rank == 3 && w->trellis.shape[0] == kt &&
                 w->trellis.shape[1] > 0 && w->trellis.shape[1] <= INT32_MAX / 16,
             "exl3 merge: incompatible source tile dimensions");
    const int64_t n = w->OutFeatures();
    VT_CHECK(w->codebook == 2 && w->Bits() == bits &&
                 n > 0 && n % 128 == 0 && n <= INT32_MAX - merged.output_offsets.back(),
             "exl3 merge: incompatible source geometry/codebook");
    VT_CHECK(w->trellis.dtype == vt::DType::kI8 &&
                 w->trellis.bytes.size() == static_cast<uint64_t>(k / 16) * (n / 16) * (32 * bits) &&
                 w->suh.dtype == vt::DType::kF16 && w->suh.rank == 1 && w->suh.shape[0] == k &&
                 w->suh.bytes.size() == static_cast<uint64_t>(k) * 2 &&
                 w->svh.dtype == vt::DType::kF16 && w->svh.rank == 1 && w->svh.shape[0] == n &&
                 w->svh.bytes.size() == static_cast<uint64_t>(n) * 2,
             "exl3 merge: source host layout/byte span is invalid");
    merged.output_offsets.push_back(merged.output_offsets.back() + n);
  }
  const int64_t n = merged.output_offsets.back();
  auto allocate = [](OwnedTensor& t, vt::DType dtype, std::initializer_list<int64_t> shape) {
    t.dtype = dtype;
    t.rank = static_cast<int>(shape.size());
    size_t bytes = vt::SizeOf(dtype);
    int axis = 0;
    for (int64_t dim : shape) {
      VT_CHECK(dim > 0 && static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max() / bytes,
               "exl3 merge: byte size overflow");
      bytes *= static_cast<size_t>(dim);
      t.shape[axis++] = dim;
    }
    t.bytes.resize(bytes);
  };
  allocate(merged.trellis, vt::DType::kI8, {k / 16, n / 16, 32 * bits});
  allocate(merged.suh, vt::DType::kF16, {static_cast<int64_t>(sources.size()), k});
  allocate(merged.svh, vt::DType::kF16, {n});
  allocate(merged.source_map, vt::DType::kI32, {n / 128});
  const size_t merged_row = static_cast<size_t>(n / 16) * (32 * bits);
  for (size_t s = 0; s < sources.size(); ++s) {
    const Exl3Weight& w = *sources[s];
    const int64_t offset = merged.output_offsets[s];
    const size_t source_row = static_cast<size_t>(w.OutFeatures() / 16) * (32 * bits);
    for (int64_t kt = 0; kt < k / 16; ++kt)
      std::memcpy(merged.trellis.bytes.data() + kt * merged_row + (offset / 16) * (32 * bits),
                  w.trellis.bytes.data() + kt * source_row, source_row);
    std::memcpy(merged.suh.bytes.data() + s * k * 2, w.suh.bytes.data(), k * 2);
    std::memcpy(merged.svh.bytes.data() + offset * 2, w.svh.bytes.data(), w.OutFeatures() * 2);
    const int32_t id = static_cast<int32_t>(s);
    for (int64_t nb = offset / 128; nb < merged.output_offsets[s + 1] / 128; ++nb)
      std::memcpy(merged.source_map.bytes.data() + nb * sizeof(id), &id, sizeof(id));
  }
  return merged;
}

namespace dense_exl3 {

dense_attn::DBuf GroupedLinear(dense_attn::Dev d, const vt::Tensor& x,
                               const std::vector<const Exl3Weight*>& sources,
                               Exl3GroupedWeight& grouped) {
  if (grouped.Empty()) {
    std::string name;
    for (const auto* source : sources) {
      VT_CHECK(source != nullptr, "exl3 grouped linear: null source");
      if (!name.empty()) name += "+";
      name += source->name;
    }
    grouped = MergeExl3Weights(sources, std::move(name));
  }
  return dense_attn::Exl3GroupedMatmulD(d, x, grouped);
}

dense_attn::DBuf Linear(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_w, const Exl3Weight& exl3_w,
                        vt::DType out_dtype) {
  return layers::MakeLinearMethod(bf16_w, exl3_w)->Apply(d, x, out_dtype);
}

dense_attn::DBuf GateUp(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_gate_up, const Exl3Weight& gate,
                        const Exl3Weight& up, int64_t intermediate,
                        Exl3GroupedWeight* grouped) {
  return layers::MakeMlpGateUpMethod(bf16_gate_up, gate, up, intermediate, grouped)
      ->Apply(d, x);
}

}  // namespace dense_exl3
}  // namespace vllm
