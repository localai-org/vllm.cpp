// EXL3 (exllamav3 trellis) — the quantization scheme, on the shared linear seam.
//
// UPSTREAM, and the two halves come from DIFFERENT places on purpose:
//   THE SEAM is vLLM's. `vllm/model_executor/layers/quantization/base_config.py:87-180`
//     (`QuantizationConfig` + `get_quant_method`) and
//     `vllm/model_executor/layers/linear.py:141-181` (`LinearMethodBase`) define
//     where a scheme plugs in, and this header mirrors them exactly as fp8.h and
//     compressed_tensors/schemes/nvfp4.h do.
//   THE FORMAT is exllamav3's, because vLLM registers no EXL3 at the parity pin
//     `5559679229bc961848b121ccdeaa8fa5d79bec98` — the fallback case AGENTS.md
//     admits, with the pin recorded in `.agents/oracles/exllamav3.md`
//     (`2398c05635fbbad01a0a51dce63c85c6c8a8450e`, tag v1.4.3, MIT).
//     `exllamav3/modules/quant/exl3.py:16-40` owns the four tensors and
//     `:183-214` the runtime form this method computes.
//
// WHY THIS FILE EXISTS (QUANT-EXL3 W1, #2181). The trellis kernels have existed
// since `MODEL-DSV4-EXL3` W2 and are device-proven on GB10, but their only
// consumer was `src/vllm/model_executor/models/deepseek_v4.cpp` — a model-private
// arm, so no other architecture could reach the scheme and no stock EXL3
// checkpoint could load. That is the parallel-path shape AGENTS.md forbids.
//
// The compute is ONE `vt::Exl3Gemm`, which is already the whole fused linear:
//   C = had_r_128( had_r_128(A, pre_scale=suh) @ reconstruct(trellis), post_scale=svh )
// algebraically `A @ Exl3DequantLinear(trellis, suh, svh)` (`exl3.py:183-214` vs
// `:227-237`). Nothing here re-derives the format; this header is the BINDING.
#pragma once

#include <memory>
#include <string>

#include "vllm/model_executor/layers/linear.h"

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace layers {

// `Exl3Weight` itself lives beside `Nvfp4Weight` in
// `models/qwen3_5_weights.h`, for the same reason that one does: a quantized
// weight is data a model container holds, and declaring it here would make
// `qwen3.h` include `linear.h` -> `dense_attn_block.h` -> `qwen3.h`.

// The EXL3 linear method. `Apply` is one `vt::Exl3Gemm`, with the activation
// staged to fp16 on the way in.
//
// WHY THE STAGING IS NOT OPTIONAL. `Exl3Gemm` reads `a` as fp16 and nothing
// else — the CPU arm calls `HadRows(HadIo::kHalfHalf, ...)`
// (`cpu_exl3_kernels.cpp:205`) and the device arm stages `a_had` in fp16 —
// because exllamav3 runs the whole linear in fp16. A residual stream in bf16 or
// f32 therefore pays one `vt::CastF16` per call. That cast is a general op
// rather than a private helper here, and it is the third sibling of the
// `CastBf16`/`CastF32` pair the tree already had.
//
// THE OUTPUT DTYPE IS THE CALLER'S, never inherited from the kernel. `Exl3Gemm`
// writes f16 or f32 (`ops.h`), so an f32 request is written straight and a bf16
// request is written f32 and cast once — the destination the model dtype names,
// which is the polarity AGENTS.md §"Inherit vLLM defaults" requires and which a
// token gate cannot check for you.
class Exl3LinearMethod : public LinearMethodBase {
 public:
  explicit Exl3LinearMethod(const Exl3Weight* w) : w_(w) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    // ONE implementation, `dense_attn::Exl3MatmulD` in `dense_attn_block.h`.
    // It lives beside `ResidentWeight` because it needs it, and a scheme header
    // cannot include that one back (`linear.h` already includes it, so the
    // reverse edge would close a cycle). This method is the seam's thin binding
    // to that function — never a second copy.
    return dense_attn::Exl3MatmulD(d, x, *w_, out_dtype);
  }

  const char* Name() const override { return "exl3-trellis"; }

 private:
  const Exl3Weight* w_;
};

// The gate_up half of the MLP, on the shared `MlpGateUpMethodBase` seam.
//
// Scoped XPU FP16 SmallM uses one packed merge when supplied a model-lifetime
// group owner. Trellis output tiles concatenate separately for each K tile;
// gate/up keep distinct input transforms. Legacy callers use separate GEMMs.
class Exl3MlpGateUpMethod : public MlpGateUpMethodBase {
 public:
  Exl3MlpGateUpMethod(const Exl3Weight* gate, const Exl3Weight* up,
                      Exl3GroupedWeight* grouped = nullptr)
      : gate_(gate), up_(up), grouped_(grouped) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    const int64_t I = gate_->OutFeatures();
    VT_CHECK(up_->OutFeatures() == I,
             "exl3 gate_up: gate is [.., " + std::to_string(I) + "] but up is [.., " +
                 std::to_string(up_->OutFeatures()) + "]");
    // `vt::MoeSiluMul` rather than `vt::SiluAndMul`, and it is the op written
    // for this shape: SiluAndMul consumes ONE [M, 2I] operand with gate rows
    // first, which is what the MERGED arms hand it, while this one "takes the
    // two separately-produced projections so no concat/copy is needed"
    // (`ops.h`). The operator rounds SiLU through the gate dtype before
    // multiplying by up. Preserve the model's explicit activation precision
    // at both projections and the output; callers without a policy retain
    // the legacy BF16 behavior.
    const vt::DType dtype = d.activation_dtype.value_or(vt::DType::kBF16);
    if (grouped_ && d.q.device.type == vt::DeviceType::kXPU &&
        dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16) {
      if (grouped_->Empty())
        *grouped_ = MergeExl3Weights({gate_, up_}, gate_->name + "+" + up_->name);
      DBuf gu = dense_attn::Exl3GroupedMatmulD(d, x, *grouped_);
      DBuf act(d, dtype, {M, I});
      vt::SiluAndMul(d.q, act.t(), gu.t());
      return act;
    }
    DBuf g = dense_attn::Exl3MatmulD(d, x, *gate_, dtype);
    DBuf u = dense_attn::Exl3MatmulD(d, x, *up_, dtype);
    DBuf act(d, dtype, {M, I});
    vt::MoeSiluMul(d.q, act.t(), g.t(), u.t());
    return act;
  }

  const char* Name() const override { return "exl3-gate-up"; }

 private:
  const Exl3Weight* gate_;
  const Exl3Weight* up_;
  Exl3GroupedWeight* grouped_;
};

inline std::unique_ptr<MlpGateUpMethodBase> MakeMlpGateUpMethod(
    const OwnedTensor& bf16_gate_up, const Exl3Weight& gate, const Exl3Weight& up,
    int64_t intermediate, Exl3GroupedWeight* grouped = nullptr) {
  if (!gate.Empty()) return std::make_unique<Exl3MlpGateUpMethod>(&gate, &up, grouped);
  return std::make_unique<UnquantizedMlpGateUpMethod>(&bf16_gate_up, intermediate);
}

// get_quant_method analogue, same shape as the fp8 and NVFP4 factories and
// overloaded on the weight type: a non-empty EXL3 weight selects the trellis
// method, everything else falls to bf16. The scheme is chosen ONCE, at load,
// from the checkpoint's populated weights — never per forward call by a
// tensor-name probe (base_config.h records why that matters).
inline std::unique_ptr<LinearMethodBase> MakeLinearMethod(const OwnedTensor& bf16_w,
                                                          const Exl3Weight& exl3_w) {
  if (!exl3_w.Empty()) return std::make_unique<Exl3LinearMethod>(&exl3_w);
  return std::make_unique<UnquantizedLinearMethod>(&bf16_w);
}

}  // namespace layers
}  // namespace vllm
