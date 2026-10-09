// The EXL3 linear seam, reachable from a translation unit that cannot include
// `dense_attn_block.h` — MODEL-QWEN35-EXL3 (#2495 items 3 and 5).
//
// WHY THIS FILE EXISTS, and it is a build constraint rather than a design
// choice. `layers::Exl3LinearMethod` lives in
// `layers/quantization/exl3.h`, which includes `layers/linear.h`, which
// includes `models/dense_attn_block.h`. That header defines
// `dense_attn::ResidentWeight` and `dense_attn::ResidentWeightF32`, and
// `qwen3_5.cpp` defines its OWN same-named helpers in an anonymous namespace
// while calling them unqualified on a `dense_attn::Dev`. Argument-dependent
// lookup then finds both and the whole file stops compiling — 40 ambiguous
// calls in code this row does not own. `dense_fp8_block_gemm.h` records the
// same collision one line up in the same file and answers it with templates;
// this answers it with a translation-unit boundary, because the callee here is
// a scheme method rather than a kernel body.
//
// THERE IS NO SECOND IMPLEMENTATION. Each function below is one line that
// forwards to the shared factory in `layers/quantization/exl3.h`, which binds
// `Exl3LinearMethod` / `Exl3MlpGateUpMethod`, which call the ONE
// `dense_attn::Exl3MatmulD`. The factory argument order is preserved verbatim,
// so a caller still asks the get_quant_method question — "is the trellis
// weight populated?" — and never a tensor-name probe.
#pragma once

#include <cstdint>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/qwen3_5_weights.h"    // Exl3Weight, OwnedTensor
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {
namespace dense_exl3 {

// `layers::MakeLinearMethod(bf16_w, exl3_w)->Apply(d, x, out_dtype)`.
// A non-empty `exl3_w` selects the trellis method; everything else falls to the
// unquantized one, exactly as the fp8 and NVFP4 factories do.
dense_attn::DBuf Linear(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_w, const Exl3Weight& exl3_w,
                        vt::DType out_dtype);

// Scoped XPU FP16 SmallM/W8A8; model-owned cache is bound to these immutable
// sources for its entire lifetime. Returns one [M,sum(N)] contiguous owner.
dense_attn::DBuf GroupedLinear(dense_attn::Dev d, const vt::Tensor& x,
                               const std::vector<const Exl3Weight*>& sources,
                               Exl3GroupedWeight& grouped);

// `layers::MakeMlpGateUpMethod(bf16_gate_up, gate, up, intermediate)->Apply(d, x)`.
// The EXL3 arm returns [M, I] in d.activation_dtype, or BF16 when unspecified.
dense_attn::DBuf GateUp(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_gate_up, const Exl3Weight& gate,
                        const Exl3Weight& up, int64_t intermediate,
                        Exl3GroupedWeight* grouped = nullptr);

}  // namespace dense_exl3
}  // namespace vllm
