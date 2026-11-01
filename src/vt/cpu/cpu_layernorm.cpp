// CPU LayerNorm / ReLU / Add — the reference (and CI-runnable) side of the
// cross-family dense primitives introduced by the OPT (`OPTForCausalLM`)
// bring-up. Same per-element math and the same f32-accumulation contract as
// src/vt/cuda/cuda_layernorm.cu (the block-reduction ORDER differs, exactly as
// it does for every other CPU/CUDA vt op pair).
//
// Like its CUDA sibling this is a SELF-REGISTERING translation unit (the
// `src/vt/cpu/cpu_ops.cpp` Registrar idiom), so adding the ops edited no
// existing kernel file — only the op-table declarations in `include/vt/ops.h`
// and the validating wrappers in `src/vt/ops.cpp`.
//
// Ported FROM (semantics, 1:1):
//   * LayerNorm — ATen `native_layer_norm` (`aten/src/ATen/native/
//     layer_norm.cpp`), what `nn.LayerNorm` computes for vLLM
//     `vllm/model_executor/models/opt.py:146-148,164-166,248-251`: BIASED (1/N)
//     variance, f32 accumulation for reduced-width inputs, affine
//     `y = (x-mean)*rstd*w + b`, one rounding on store.
//   * Relu — `get_act_fn("relu")` (opt.py:156).
//   * Add — `torch.add`, elementwise plus the rank-1 row-broadcast `nn.Linear`
//     bias form (opt.py:90-104,149-163,178,191,279).
#include <cmath>

#include "cpu_threadpool.h"
#include "vt/ops.h"
#include "vt/unaligned.h"

namespace vt::cpu {
namespace {

// AN INPUT TENSOR'S BYTES START AT AN ARBITRARY ADDRESS, and this file used to
// assume otherwise. A weight loaded by `LoadBf16Direct` can BORROW the mmap'd
// safetensors payload instead of copying it (`BorrowStTensorBytes`,
// ENG-LOAD-DIRECT-UPLOAD #150), and a safetensors tensor's offset is the running
// byte total of everything ahead of it plus the unpadded `8 + header_len`
// prologue — so it can be ODD. Forming a `uint16_t*` to it is undefined even
// where x86 executes the load, which is issue
// [#627](https://github.com/mudler/vllm.cpp/issues/627) and, before it,
// `ea4deb203` ("Use defined arbitrary-address tensor reads") — the commit that
// gave `vt::cpu::LoadF32` (cpu_ops.cpp:32) this exact shape for this exact
// reason. This copy was written after that commit and did not inherit it.
// `vt::LoadUnaligned` is a `memcpy`: the same bytes, the same value, no
// numerics moved.
//
// The STORE side deliberately keeps its typed pointer, mirroring
// `vt::cpu::StoreF32` (cpu_ops.cpp:43): an output is an engine-allocated
// buffer, never a borrowed file mapping.
float LoadF32At(const Tensor& t, int64_t i) {
  const auto* address = static_cast<const uint8_t*>(t.data) + i * SizeOf(t.dtype);
  switch (t.dtype) {
    case DType::kF32: return LoadUnaligned<float>(address);
    case DType::kF16: return F16ToF32(LoadUnaligned<uint16_t>(address));
    case DType::kBF16: return BF16ToF32(LoadUnaligned<uint16_t>(address));
    default: VT_CHECK(false, "cpu layer_norm: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "cpu layer_norm: unsupported output dtype");
  }
}

// out[r, :] = (x[r,:] - mean) * rsqrt(var + eps) * weight + bias, per row over
// the LAST dim. Two passes in f32 (mean, then squared deviations ABOUT that
// mean) — the numerically stable form, matching the CUDA kernel exactly.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t base = r * d;
      float sum = 0.0f;
      for (int64_t i = 0; i < d; ++i) sum += LoadF32At(x, base + i);
      const float mean = sum / static_cast<float>(d);
      float sq = 0.0f;
      for (int64_t i = 0; i < d; ++i) {
        const float dv = LoadF32At(x, base + i) - mean;
        sq += dv * dv;
      }
      const float rstd = 1.0f / std::sqrt(sq / static_cast<float>(d) + args.eps);
      for (int64_t i = 0; i < d; ++i) {
        float v = (LoadF32At(x, base + i) - mean) * rstd;
        if (weight != nullptr) v *= LoadF32At(*weight, i);
        if (bias != nullptr) v += LoadF32At(*bias, i);
        StoreF32At(out, base + i, v);
      }
    }
  });
}

void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  const int64_t d = x.rank == 0 ? 1 : x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : n / d;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0 * d; i < r1 * d; ++i) {
      const float v = LoadF32At(x, i);
      StoreF32At(out, i, v > 0.0f ? v : 0.0f);
    }
  });
}

void GeluTanhKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  const int64_t d = x.rank == 0 ? 1 : x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : n / d;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0 * d; i < r1 * d; ++i) {
      const float v = LoadF32At(x, i);
      const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
      StoreF32At(out, i, 0.5f * v * (1.0f + std::tanh(inner)));
    }
  });
}

void GeluErfKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  const int64_t d = x.rank == 0 ? 1 : x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : n / d;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0 * d; i < r1 * d; ++i) {
      const float v = LoadF32At(x, i);
      StoreF32At(out, i, 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f)));
    }
  });
}

void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  // rank-1 b over a's last dim == a nn.Linear bias row-broadcast; otherwise
  // plain elementwise.
  const bool bcast = b.rank == 1 && a.rank != 1;
  // Split the output element range, not the rows: the per-element expression
  // (including the bias broadcast index) is unchanged, so this is
  // bit-identical, and a decode step with one row no longer runs the whole
  // vector on the caller (spec § W5).
  ParallelForRows(CurrentThreadpool(), n, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i)
      StoreF32At(out, i, LoadF32At(a, i) + LoadF32At(b, bcast ? (i % d) : i));
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLayerNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kGeluTanh, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluTanhKernel)));
    RegisterOp(OpId::kGeluErf, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluErfKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
