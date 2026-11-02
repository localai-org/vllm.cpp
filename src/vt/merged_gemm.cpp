// vt::MergedGemm — dispatch for the declarative MERGED-GEMM GROUP descriptor
// (merged_gemm.h). Realizes a group either through its fused fast op (the promoted
// shared kernel, selected by dtype/arity/epilogue) or through the byte-exact Tier-0
// composite of standalone vt:: ops — the same tiering FusedRecipe uses.
#include "vt/merged_gemm.h"

#include <cmath>
#include <limits>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt {

void MergedGemmDense(Queue& q, const MergedGemmGroup& desc, Tensor* products,
                     const Tensor& act, const Tensor* weights, int count,
                     Tensor* epilogue_out) {
  VT_CHECK(count == desc.arity && count >= 2 && products != nullptr && weights != nullptr,
           "MergedGemmDense: descriptor and shard count must match");
  VT_CHECK(desc.fast_op == kNoMergedFastOp,
           "MergedGemmDense: no dense-shard fused provider is registered");
  const bool swiglu = desc.epilogue == MergedEpilogue::kSiluMulClamp;
  VT_CHECK(desc.epilogue == MergedEpilogue::kNone || (swiglu && count == 2),
           "MergedGemmDense: unsupported dense epilogue");
  VT_CHECK(swiglu == (epilogue_out != nullptr),
           "MergedGemmDense: SwiGLU alone requires an epilogue output");
  for (int i = 0; i < count; ++i) MatmulBT(q, products[i], act, weights[i]);
  if (swiglu) MoeSiluMul(q, *epilogue_out, products[0], products[1]);
}

void MergedGemm(Queue& q, const MergedGemmGroup& desc, Tensor& out, const Tensor& act,
                const Tensor& gate_w, const Tensor& up_w, const Tensor& expert_ids,
                float epilogue_scalar, bool force_composite) {
  VT_CHECK(desc.arity == 2 && desc.epilogue == MergedEpilogue::kSiluMulClamp,
           "MergedGemm: only the arity-2 clamped-SwiGLU group is realized today");

  // Fast path: the descriptor's fused kernel, registered for this device. This is the
  // one-launch realization every keep-quant MoE arch inherits automatically.
  const bool have_fast =
      desc.fast_op != kNoMergedFastOp &&
      OpRegistered(static_cast<OpId>(desc.fast_op), q.device.type);
  if (have_fast && !force_composite) {
    MoeGateUpSwiGLUGrouped(q, out, act, gate_w, up_w, expert_ids, epilogue_scalar);
    return;
  }

  // Tier-0 COMPOSITE (the portable reference golden): the standalone-op sequence
  //   g = MatmulBTQuantGrouped(gate_w);  u = MatmulBTQuantGrouped(up_w);
  //   out = min(g,limit)·sigmoid(min(g,limit))·clamp(u,±limit)
  // BYTE-EXACT to what the fused kernel computes. It runs on the CPU reference tier;
  // an accelerated backend registers the fast op above rather than expanding this.
  VT_CHECK(q.device.type == DeviceType::kCPU,
           "MergedGemm: the Tier-0 composite is the CPU reference tier — a non-CPU "
           "device must register the group's fast op (kMoeGateUpSwiGLUGrouped)");
  const int64_t P = out.shape[0];
  const int64_t N = out.shape[1];
  std::vector<float> g(static_cast<size_t>(P) * N);
  std::vector<float> u(static_cast<size_t>(P) * N);
  Tensor gt = Tensor::Contiguous(g.data(), DType::kF32, out.device, {P, N});
  Tensor ut = Tensor::Contiguous(u.data(), DType::kF32, out.device, {P, N});
  MatmulBTQuantGrouped(q, gt, act, gate_w, expert_ids);
  MatmulBTQuantGrouped(q, ut, act, up_w, expert_ids);
  float* o = static_cast<float*>(out.data);
  const float limit = epilogue_scalar;
  for (size_t i = 0; i < g.size(); ++i) {
    const float gate = std::fmin(g[i], limit);
    const float up = std::fmin(std::fmax(u[i], -limit), limit);
    o[i] = gate * (1.0F / (1.0F + std::exp(-gate))) * up;
  }
}


// The block-wise FP8 merged group (merged_gemm.h kFp8BlockGateUpSwiGLU /
// kFp8BlockQkv, #1189 M6). Same tiering as MergedGemm above: the descriptor's
// fused kernel where a device has registered one, else the composite — ONE
// mainloop-scaled block GEMM over the N-concatenated operand, then the group's
// elementwise tail.
void MergedGemmFp8Block(Queue& q, const MergedGemmGroup& desc, Tensor& merged,
                        Tensor* out, const Tensor& a_fp8, const Tensor& a_scale,
                        const Tensor& w_fp8, const Tensor& w_scale, int block_n,
                        int block_k, float epilogue_scalar) {
  VT_CHECK(desc.arity >= 2,
           "MergedGemmFp8Block: a merged group needs at least two shards");
  VT_CHECK(desc.epilogue == MergedEpilogue::kNone ||
               desc.epilogue == MergedEpilogue::kSiluMulClamp,
           "MergedGemmFp8Block: only the kNone and clamped-SwiGLU epilogues are "
           "realized on the block-wise FP8 arm");
  VT_CHECK(merged.rank == 2 && w_fp8.rank == 2 && w_scale.rank == 2,
           "MergedGemmFp8Block: merged output and weight operands are rank-2");
  VT_CHECK(merged.shape[1] == w_fp8.shape[0],
           "MergedGemmFp8Block: the merged output's N does not match the "
           "N-concatenated weight's rows");

  // The fast path a future single-launch kernel takes. kNoMergedFastOp today on
  // both block groups, so this is unreachable until #1189 M5 registers one; it
  // is written here rather than at the model site so the model site never
  // learns the tiering rule.
  if (desc.fast_op != kNoMergedFastOp &&
      OpRegistered(static_cast<OpId>(desc.fast_op), q.device.type)) {
    VT_CHECK(false,
             "MergedGemmFp8Block: the group names a fused fast op that this "
             "build registers but cannot dispatch yet");
  }

  MatmulFp8BlockScaled(q, merged, a_fp8, a_scale, w_fp8, w_scale, block_n,
                       block_k);

  if (desc.epilogue == MergedEpilogue::kNone) {
    VT_CHECK(out == nullptr,
             "MergedGemmFp8Block: a kNone group has no epilogue output");
    return;
  }
  VT_CHECK(out != nullptr,
           "MergedGemmFp8Block: the clamped-SwiGLU epilogue needs an output");
  VT_CHECK(desc.arity == 2,
           "MergedGemmFp8Block: the clamped-SwiGLU epilogue is arity-2");
  VT_CHECK(merged.shape[1] == 2 * out->shape[1] &&
               merged.shape[0] == out->shape[0],
           "MergedGemmFp8Block: the SwiGLU output must be half the merged "
           "output's width and the same height");
  // limit == +inf reduces kSiluMulClamp to the plain silu(gate)*up SwiGLU, as
  // merged_gemm.h's enumerator records, and vt::SiluAndMul IS that reduction
  // over the contiguous [M,2I] buffer. A finite limit would need a clamped
  // kernel, which no backend binds for this arm, so it is refused rather than
  // dropped.
  VT_CHECK(std::isinf(epilogue_scalar) && epilogue_scalar > 0.0F,
           "MergedGemmFp8Block: the block-wise FP8 SwiGLU tail is the "
           "limit=+inf reduction of kSiluMulClamp; no clamped kernel is bound "
           "on this arm");
  SiluAndMul(q, *out, merged);
}

}  // namespace vt
