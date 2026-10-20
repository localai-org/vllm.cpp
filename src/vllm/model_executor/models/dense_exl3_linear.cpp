// See `include/vllm/model_executor/models/dense_exl3_linear.h`. This
// translation unit exists so that the scheme header, and through it
// `dense_attn_block.h`, is included HERE and not in `qwen3_5.cpp`.
#include "vllm/model_executor/models/dense_exl3_linear.h"

#include <cstring>
#include <limits>
#include <cstdlib>
#include <string_view>

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

std::optional<dense_attn::DBuf> PreparedSiluMlp(dense_attn::Dev d,
    const vt::Tensor& x, const Exl3Weight& gate, const Exl3Weight& up,
    const Exl3Weight& down, Exl3GroupedWeight& grouped) {
  using vt::DType;
  if (d.q.device.type != vt::DeviceType::kXPU || d.activation_dtype != DType::kF16 ||
      x.dtype != DType::kF16 || x.rank != 2 || x.shape[0] < 512 || x.shape[0] > 4096 ||
      gate.Empty() || up.Empty() || down.Empty() ||
      gate.codebook != 2 || up.codebook != 2 || down.codebook != 2 ||
      gate.OutFeatures() != 17408 || up.OutFeatures() != 17408 ||
      down.InFeatures() != 17408 || down.OutFeatures() != 5120) return std::nullopt;
  const int64_t rows = x.shape[0], k = down.InFeatures(), n = down.OutFeatures();
  const int panel = vt::Exl3W8A8ModelPanelColumns();
  (void)vt::PlanExl3W8A8(rows, k, n, 1, down.Bits(), panel);
  // The existing packed group retains separate gate/up transforms and owners.
  auto gu = GroupedLinear(d, x, {&gate, &up}, grouped);
  auto trellis = dense_attn::ResidentWeight(d, down.trellis);
  auto suh = dense_attn::ResidentWeight(d, down.suh);
  // Same immutable source storage as the ordinary single-weight EXL3 wrapper.
  suh = vt::Tensor::Contiguous(suh.data, DType::kF16, d.q.device, {1, k});
  auto svh = dense_attn::ResidentWeight(d, down.svh);
  if (down.single_source_map.rank == 0) {
    down.single_source_map.dtype = DType::kI32;
    down.single_source_map.rank = 1;
    down.single_source_map.shape[0] = n / 128;
  }
  auto map = dense_attn::ResidentWeight(d, down.single_source_map, {}, [&](vt::Tensor& t) {
    d.b.Memset(d.q, t.data, 0, t.Bytes());
  });
  dense_attn::DBuf out(d, DType::kF16, {rows, n});
  const vt::Exl3GroupedLinearArgs args{down.Bits(), down.codebook, down.name.c_str(), panel};
  const char* table = std::getenv("VT_XPU_SILU_FP16_TABLE");
  VT_CHECK(!table || std::string_view(table) == "0" || std::string_view(table) == "1",
           "Invalid VT_XPU_SILU_FP16_TABLE");
  const char* typed = std::getenv("VT_XPU_SILU_FP16_TYPED");
  if ((!table || std::string_view(table) == "1") &&
      (!typed || std::string_view(typed) != "0") &&
      vt::detail::TryExl3GroupedW8A8SiluModel(d.q, out.t(), gu.t(), trellis, suh, svh,
          map, args, down.single_source_map.d_dev, down.w8a8_model_map)) return out;
  // A refusal has not published out. Reuse that SAME owner and the completed
  // gate/up tensor, so fallback neither repeats gate/up nor retains another Y.
  dense_attn::DBuf act(d, DType::kF16, {rows, k});
  vt::SiluAndMul(d.q, act.t(), gu.t());
  // Match ordinary GateUp's lifetime: this same-queue consumer now owns all
  // reads. Return the completed producer's owner to the existing queue-safe
  // pool before the down projection, with no new fence or recomputed gate/up.
  gu = dense_attn::DBuf{};
  vt::detail::Exl3GroupedW8A8Model(d.q, out.t(), act.t(), trellis, suh, svh, map,
      args, down.single_source_map.d_dev, down.w8a8_model_map);
  return out;
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
