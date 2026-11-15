// DFlash draft model forward (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported from
// vllm/model_executor/models/qwen3_dflash.py @ 555967922. See qwen3_dflash.h.
//
// The ONE new brick is the attention: full-attention layers route through
// vt::DFlashBlockAttention with args.causal=false (BIDIRECTIONAL in-block); SWA
// layers use args.causal=true + the window. Every other op (embed, merged-qkv
// GEMM, per-head q/k RMSNorm, NeoX RoPE, SwiGLU, standard add+RMSNorm, lm_head) is
// reused from the landed Qwen3-dense block ops (dense_attn_block.h / vt::).
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
// MODEL-DFLASH2-EXL3 (#2495 item 7): the EXL3 arms of the same two seams. This
// file already includes `dense_attn_block.h` and says `using namespace
// dense_attn`, so it has none of the ADL collision that made `qwen3_5.cpp`
// reach the scheme through a translation-unit boundary (`dense_exl3_linear.h`).
#include "vllm/model_executor/layers/quantization/exl3.h"  // MakeLinearMethod / MakeMlpGateUpMethod
// MODEL-DFLASH2-NVFP4 (#2758): the NVFP4 W4A16 arms of the SAME two seams. The
// three overloads live in one namespace and are picked apart by the packed
// weight's type, so a draft binds exactly one method per projection.
#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/Reshape/MakeRopeArgs
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // #1628: the shared NVFP4 W4A16 logits GEMM
#include "vllm/model_executor/models/dense_exl3_linear.h"  // #2495 item 6: the ONE EXL3 linear seam
#include "vllm/model_executor/models/qwen3_dflash_internal.h"  // W11 (#1890): the block-attn route
#include "vllm/platforms/interface.h"                     // platforms::GetPlatform (static-graph gate)
#include "vt/backend.h"
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W5: the shared capture seam
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight, Reshape, MakeRopeArgs

constexpr int64_t kPadSlotId = -1;  // vLLM PAD_SLOT_ID (attention/backends/utils.py:45)

// ---------------------------------------------------------------------------
// SPEC-DFLASH2-QUANT-LMHEAD (#1628) - the ONE draft logits GEMM.
//
// The draft owns no head; it runs the TARGET's over its own hidden states, and
// the target's head has TWO storage owners (`LoadDenseLmHead`, qwen3_5_dense.h):
// raw-NK bf16, or ModelOpt/compressed-tensors NVFP4 kept PACKED. Every block
// forward below used to read only the first, so a target whose head is NVFP4 was
// refused at load by stored dtype.
//
// The packed arm goes through `dense_nvfp4::MatmulNvfp4W4A16D`. That matters
// here more than anywhere else: the DFlash2 candidate selector's whole input is
// the TARGET head's exact top-K, so the draft has to compute it with the head
// the target computes with, by the computation the target uses.
// `tests/vllm/models/test_qwen3_dflash2_draft.cpp` measures that equality
// against `Qwen3_5MTPModel::ComputeLogits` - the OTHER draft that shares the
// target's head - rather than asserting it from the code.
//
// IT IS NOT THE SAME FUNCTION, and this comment used to say it was "extracted
// VERBATIM". Read against the tree that is false twice over. The target's own
// logits take `MatmulNvfp4F32D` (qwen3_5.cpp:3106, reached from :3139), a
// different dispatcher; and the two `MatmulNvfp4MarlinD` bodies underneath them
// (qwen3_5.cpp:2849-2903 vs dense_nvfp4_gemm.h:505-560) differ in four ways.
// They hold SEPARATE function-local `static void* ws` workspaces. qwen3_5.cpp
// `Memset`s its workspace on EVERY call while the shared one zeroes it ONCE, on
// a documented kernel-self-reset invariant. The shared one threads
// `w.group_size` / `w.is_mxfp4` where qwen3_5.cpp hardcodes 16 / false. And the
// shared one increments `MutableW4A16Stats()`, which qwen3_5.cpp does not.
//
// The CONCLUSION survives, and the reason is narrower than "same code": FOR A
// HEAD the last three differences are unreachable. `LoadCtNvfp4Raw` and
// `LoadNvfp4AnyNaming` set neither `group_size` nor `is_mxfp4`, so both stay at
// their defaults 16 / false -- exactly what qwen3_5.cpp hardcodes and what
// `vt::MoeMarlinArgs` already defaults to -- and the stats counter is
// observational. What is left is the workspace-zero POLICY, which is the one
// thing a CUDA run has to check; `.agents/specs/dflash2-spec-decode.md`
// `## Owed` O29 names it as such.
//
// Upstream reaches the same place and needs no branch, because its head is an
// `nn.Module`: `compute_candidates` @ the MERGED vllm-project/vllm#52816 head
// `b389ac29465b33f9e9c534df221ea3c129e9793f` calls
// `LogitsProcessor.get_top_k_tokens(self.lm_head, ...)` ->
// `_apply_head` -> `lm_head.quant_method.apply` (logits_processor.py:241-286,
// :132-142), which IS the target's own logits path. The quant-method refusal
// that file carried at `66e5414c` - the one `RefuseQuantizedDflash2LmHead`
// mirrors - is gone from the merged version entirely.
DBuf DflashLogitsF32D(Dev d, const Tensor& x, const Qwen3DFlashWeights& weights,
                      int64_t vocab, int64_t hidden_size) {
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 6): a TRELLIS target head, computed with
  // packed, through the one EXL3 linear seam. FIRST, so every logits site in
  // this tree orders its arms `exl3 -> fp4 -> bf16` and three readers of one
  // head cannot disagree about precedence (`DenseLogitsF32D`, qwen3_5.cpp;
  // `Qwen3_5MTPModel::ComputeLogits`). f32 out: the trellis GEMM writes f32
  // directly, so nothing is widened on the way to the selector.
  if (!weights.lm_head_exl3.Empty()) {
    return dense_exl3::Linear(d, x, weights.lm_head, weights.lm_head_exl3,
                              DType::kF32);
  }
  if (!weights.lm_head_fp4.Empty()) {
    // The W4A16 dispatcher refuses a true-W4A4 weight itself; this says WHY in
    // this lane's terms, because falling into the fp4-activation GEMM the target
    // head does not take is the silent-wrong `## Risks/decisions` D12 is about.
    VT_CHECK(!weights.lm_head_fp4.IsTrueW4A4(),
             "dflash: the shared lm_head is NVFP4 with an ACTIVATION scale "
             "(true W4A4) and this draft's logits GEMM is the W4A16 dispatcher "
             "the target's own W4A16 head takes. Unset VT_MODELOPT_W4A4. Issue "
             "#1628 (https://github.com/mudler/vllm.cpp/issues/1628).");
    return dense_nvfp4::MatmulNvfp4W4A16D(d, x, weights.lm_head_fp4, DType::kF32);
  }
  // EXACTLY ONE owner is populated, and neither being populated is a LOADER
  // defect rather than a user error -- so it is named here instead of read as an
  // empty tensor. Found by this row's own reachability mutation: deleting the
  // packed branch above made the packed case reach `ResidentWeight` with an
  // empty `lm_head` and SEGFAULT, which is a red the suite cannot explain. A
  // named refusal is the same red with the reason attached.
  VT_CHECK(!weights.LmHead().Empty(),
           "dflash: the draft's SHARED lm_head is empty in ALL THREE owners. The "
           "draft runs the TARGET's head, so the loader fills exactly one of "
           "`lm_head` (raw-NK bf16), `lm_head_fp4` (packed NVFP4) and "
           "`lm_head_exl3` (packed EXL3 trellis) -- see LoadDflashSharedLmHead "
           "(qwen3_dflash.h), issues #1628 "
           "(https://github.com/mudler/vllm.cpp/issues/1628) and #2495 "
           "(https://github.com/mudler/vllm.cpp/issues/2495).");
  // QWEN38-DFLASH2-STAGE-C W3: a GGUF head now arrives PACKED (kept quantized,
  // the #1628 admitted state) and takes the same block-quant MatmulBT dispatch
  // the target's own logits GEMM takes (DenseLogitsF32D's keep-quant arm,
  // qwen3_5.cpp). f32 out, like every other arm of this function.
  if (vt::IsBlockQuant(weights.LmHead().dtype)) {
    Tensor lm = ResidentWeight(d, weights.LmHead());
    DBuf logits(d, DType::kF32, {x.shape[0], vocab});
    vt::MatmulBT(d.q, logits.t(), x, lm);
    return logits;
  }
  Tensor lm = ResidentWeight(d, weights.LmHead(), {vocab, hidden_size});
  DBuf logits(d, DType::kF32, {x.shape[0], vocab});
  vt::MatmulBT(d.q, logits.t(), x, lm);
  return logits;
}

// ---------------------------------------------------------------------------
// SPEC-DFLASH2 W2 (#1314) — the grouped dynamic depthwise convolution that wraps
// each attention and each MLP sublayer of a DFlash2 draft block.
//
// BEYOND-PIN, from `DFlashGroupedConv.prepare` / `.finish` and
// `DFlash2Qwen3DecoderLayer.forward`
// (vllm/model_executor/models/qwen3_dflash2.py @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`). Upstream's decoder layer is:
//
//     hidden, coefficients = self.attention_conv.prepare(hidden)
//     hidden               = self.self_attn(positions, hidden)
//     hidden               = self.attention_conv.finish(hidden, coefficients)
//     hidden, residual     = self.post_attention_layernorm(hidden, residual)
//     hidden, coefficients = self.mlp_conv.prepare(hidden)
//     hidden               = self.mlp(hidden)
//     hidden               = self.mlp_conv.finish(hidden, coefficients)
//
// TWO things about the shape of this pair are load-bearing and neither is
// visible in a token gate, because a DFlash2 draft whose conv is wrong still
// emits the TARGET's tokens (the verify is lossless) and loses only acceptance:
//
//  * ONE projection, TWO sides. `prepare` projects the sublayer INPUT once into
//    `[T, 2, taps, num_groups]` and convolves with side 0; `finish` reuses that
//    SAME buffer with side 1, over the sublayer OUTPUT. So the finish
//    coefficients are a function of the input, not of the output, and computing
//    them again after the sublayer would be a different model.
//  * The conv's block is the QUERY block, `1 + k`, and the taps are zeroed
//    across its boundary, which is what `weights.conv_block_size` carries.
//
// `stream` is convolved IN PLACE (the DBuf is replaced by the conv output),
// mirroring upstream's rebinding of `hidden_states`.
DBuf DflashConvPrepare(Dev d, const Qwen3DFlashConvWeights& cw,
                       const Qwen3DFlashWeights& weights, const HfConfig& config,
                       DBuf* stream) {
  const int64_t T = stream->t().shape[0];
  const int64_t H = config.hidden_size;
  const int64_t taps = weights.conv_taps;
  const int64_t groups = H / weights.conv_group_size;
  // ONE projection of the sublayer input -> [T, 2, taps, num_groups]. The GEMM
  // writes a flat [T, 2*taps*num_groups] view of the same buffer, which is the
  // reshape upstream spells as `.reshape(hidden.shape[0], 2, taps, num_groups)`.
  DBuf coef(d, DType::kBF16, {T, 2, taps, groups});
  {
    Tensor flat = Reshape(coef.t(), {T, 2 * taps * groups});
    Tensor wp = ResidentWeight(d, cw.kernel_projection);
    vt::MatmulBT(d.q, flat, stream->t(), wp);
  }
  DBuf out(d, DType::kBF16, {T, H});
  Tensor base = ResidentWeight(d, cw.base_kernel, {2, taps, H});
  vt::DFlashGroupedConvArgs a;
  a.block_size = weights.conv_block_size;
  a.taps = taps;
  a.num_groups = groups;
  a.group_size = weights.conv_group_size;
  a.side = 0;  // prepare
  vt::DFlashGroupedConv(d.q, out.t(), stream->t(), coef.t(), base, a);
  *stream = std::move(out);
  return coef;
}

void DflashConvFinish(Dev d, const Qwen3DFlashConvWeights& cw,
                      const Qwen3DFlashWeights& weights, const HfConfig& config,
                      DBuf* stream, const DBuf& coef) {
  const int64_t T = stream->t().shape[0];
  const int64_t H = config.hidden_size;
  const int64_t taps = weights.conv_taps;
  const int64_t groups = H / weights.conv_group_size;
  DBuf out(d, DType::kBF16, {T, H});
  Tensor base = ResidentWeight(d, cw.base_kernel, {2, taps, H});
  vt::DFlashGroupedConvArgs a;
  a.block_size = weights.conv_block_size;
  a.taps = taps;
  a.num_groups = groups;
  a.group_size = weights.conv_group_size;
  a.side = 1;  // finish, reading the SAME coefficients the prepare projected
  vt::DFlashGroupedConv(d.q, out.t(), stream->t(), coef.t(), base, a);
  *stream = std::move(out);
}

// The conv masks its taps by `row index mod conv_block_size`, exactly as
// upstream's `torch.arange(hidden_states.shape[0]) % block_size` does. That is
// the intra-block offset ONLY while every request block is contiguous and
// `conv_block_size`-aligned, which is the uniform (1+k) DFlash batch. A ragged
// batch would silently mask the wrong taps -- acceptance-only and token-invisible
// -- so it is refused here rather than discovered on a gate host.
void CheckDflashConvBatch(const Qwen3DFlashWeights& weights, const std::vector<int32_t>& cu) {
  VT_CHECK(weights.conv_block_size > 0,
           "qwen3_dflash2: conv_block_size must be set (1 + num_speculative_tokens)");
  for (size_t r = 0; r + 1 < cu.size(); ++r) {
    VT_CHECK(cu[r] % weights.conv_block_size == 0 &&
                 cu[r + 1] - cu[r] == static_cast<int32_t>(weights.conv_block_size),
             "qwen3_dflash2: the grouped convolution needs a uniform "
             "conv_block_size-aligned query block per request");
  }
}


// Device-resident per-layer context K/V (D7). The D5 path downloaded each layer's
// projected K/V to host (2 D->H copies/layer) and re-uploaded them in the block
// forward's [context;block] host interleave. This helper keeps the projected K/V
// ON DEVICE as bf16 [num_ctx, kv_size] buffers, so the block forward can build the
// combined sequence with device vt::IndexCopy instead of host round-trips. The
// float ops (cast/RMSNorm/GEMM/k-norm/RoPE) are IDENTICAL to the D5 path and run in
// the same order, so the stored K/V bits are bit-identical to the D5 download.
// Mirrors precompute_and_store_context_kv (qwen3_dflash.py:548-619), minus the
// paged-cache write (our within-step store is these DBufs).
struct ContextKVDev {
  std::vector<DBuf> k;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (normed+RoPE'd)
  std::vector<DBuf> v;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (raw)
  int64_t num_ctx = 0;
};

// MODEL-DFLASH2-NVFP4 (#2758): the THREE-ARM method choice, in ONE place.
//
// `layers::MakeLinearMethod` has an overload per quantization and each answers
// "unquantized" when its own packed weight is empty, so calling two of them in
// sequence is how a third arm is added without either overload learning about
// the other. The order is `exl3 -> nvfp4 -> bf16`, which is the order the
// loader's rungs run in and the order `LoadDflashSharedLmHead` already uses for
// the shared head.
//
// WHY A FUNCTION AND NOT A BRANCH REPEATED NINE TIMES: #2202 records what the
// three-copies-of-one-choice shape cost this file already -- the merged qkv
// seam landed in the cold body and the two hot ones kept re-reading the
// activation for a year. A third arm copied into all three bodies is the same
// defect waiting.
std::unique_ptr<layers::LinearMethodBase> DflashLinear(const OwnedTensor& bf16,
                                                       const Exl3Weight& exl3,
                                                       const Nvfp4Weight& fp4) {
  if (!exl3.Empty()) return layers::MakeLinearMethod(bf16, exl3);
  return layers::MakeLinearMethod(bf16, fp4);
}

std::unique_ptr<layers::MlpGateUpMethodBase> DflashMlpGateUp(
    const OwnedTensor& bf16_gate_up, const Exl3Weight& gate_exl3,
    const Exl3Weight& up_exl3, const Nvfp4Weight& gate_fp4,
    const Nvfp4Weight& up_fp4, int64_t intermediate) {
  if (!gate_exl3.Empty())
    return layers::MakeMlpGateUpMethod(bf16_gate_up, gate_exl3, up_exl3, intermediate);
  return layers::MakeMlpGateUpMethod(bf16_gate_up, gate_fp4, up_fp4, intermediate);
}

// MODEL-DFLASH2-EXL3 (#2495 item 7): the draft's q/k/v projections, for all
// three inline layer bodies.
//
// WHY ONE FUNCTION AND NOT A BRANCH REPEATED THREE TIMES. The three bodies each
// carried their own copy of the merged-versus-sliced bf16 choice, and #2202
// records what that cost: the merged seam landed in the cold body and the two
// hot ones kept re-reading the activation three times for a year. A third arm
// copied into all three is the same defect waiting. The bf16 arms below are the
// ones those bodies already ran, moved rather than rewritten -- same ops, same
// order, same `MergedQkvEnabled()` question.
//
// THE EXL3 ARM CANNOT BE MERGED, and `MergedQkvEnabled()` is therefore not
// asked on it. There is no merged trellis operand to read: joining two
// trellises on the output dimension interleaves per input tile, which is a real
// transform (`quantization/exl3.h`), and the artifact ships three separately
// fitted tensors with three separately fitted `svh` vectors.
//
// `layer.qkv_proj` is handed to all three factory calls as the unquantized
// fallback and is EMPTY on this arm. It is never dereferenced, because
// `MakeLinearMethod` binds the trellis method whenever the trellis is
// non-empty, and the loader reads all seven per-layer trellises unconditionally
// once `fc` classified the checkpoint -- so `IsExl3()` true means all three of
// these are populated, and a checkpoint that quantized only `fc` threw at load.
void ProjectDflashQkv(Dev d, const Qwen3DFlashWeights& weights,
                      const Qwen3DFlashLayerWeights& layer, const Tensor& x,
                      int64_t qdim, int64_t kdim, DBuf* q, DBuf* k, DBuf* v) {
  const int64_t T = x.shape[0];
  // MODEL-DFLASH2-NVFP4 (#2758) joins this arm rather than adding a second one:
  // an NVFP4 draft has no merged `qkv_proj` operand either. NVFP4 CAN row-stack
  // -- `dense_loaders::LoadMergedCtNvfp4W4A16` does it where the producer emits
  // a merged operand -- but this producer emitted q, k and v as three modules
  // with three independently fitted `weight_scale_2` scalars, and the target's
  // own ModelOpt path in this tree reads q/k/v/o unmerged for the same reason.
  // So `MergedQkvEnabled()` is not asked on either packed arm.
  if (weights.IsExl3() || weights.IsNvfp4()) {
    *q = DflashLinear(layer.qkv_proj, layer.q_proj_exl3, layer.q_proj_fp4)
             ->Apply(d, x, DType::kBF16);
    *k = DflashLinear(layer.qkv_proj, layer.k_proj_exl3, layer.k_proj_fp4)
             ->Apply(d, x, DType::kBF16);
    *v = DflashLinear(layer.qkv_proj, layer.v_proj_exl3, layer.v_proj_fp4)
             ->Apply(d, x, DType::kBF16);
    return;
  }
  *q = DBuf(d, DType::kBF16, {T, qdim});
  *k = DBuf(d, DType::kBF16, {T, kdim});
  *v = DBuf(d, DType::kBF16, {T, kdim});
  // Merged QKVParallelLinear: D1 folds the shared-input q/k/v GEMMs to ONE
  // MatmulBT over the merged owner + a contiguous QkvSplit (MergedQkvEnabled(),
  // VT_QWEN3_QKV_MERGE default ON; =0 = byte-identical 3-shard). RoPE handling
  // is UNCHANGED at every call site -- still RopeNeox, no RopeFromCache swap.
  Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
  if (MergedQkvEnabled()) {
    DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
    vt::MatmulBT(d.q, qkv.t(), x, wqkv);
    vt::QkvSplit(d.q, q->t(), k->t(), v->t(), qkv.t());
  } else {
    vt::MatmulBT(d.q, q->t(), x, wqkv.Slice(0, 0, qdim));
    vt::MatmulBT(d.q, k->t(), x, wqkv.Slice(0, qdim, qdim + kdim));
    vt::MatmulBT(d.q, v->t(), x, wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim));
  }
}

// SPEC-DFLASH2 W8 (#1838): the projection core over a DEVICE bf16 features
// tensor. The f32-host entry below is a marshaling shell over this — the cast
// it runs (f32 -> bf16) recovers exactly the bf16 bits the runner's aux tap
// carried, so feeding those bits directly is bit-identical.
ContextKVDev PrecomputeContextKVDeviceBf16(Dev d, const Tensor& ctxb_bf16,
                                           const Tensor& cpos, int64_t C,
                                           const Qwen3DFlashWeights& weights,
                                           const HfConfig& config) {
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const float eps = static_cast<float>(config.rms_norm_eps);

  ContextKVDev out;
  out.num_ctx = C;
  if (C == 0) return out;

  // normed = RMSNorm(context_states, hidden_norm) — the ONE shared hidden_norm
  // over the combined target features (qwen3_dflash.py:505-520).
  Tensor w_hn = ResidentWeight(d, weights.hidden_norm, {H});
  DBuf normed(d, DType::kBF16, {C, H});
  vt::RmsNorm(d.q, normed.t(), ctxb_bf16, w_hn, vt::RmsNormArgs{eps, false});

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    DBuf k, v;
    if (weights.IsExl3() || weights.IsNvfp4()) {
      // MODEL-DFLASH2-EXL3 (#2495 item 7) and MODEL-DFLASH2-NVFP4 (#2758):
      // neither packed draft has a merged qkv owner to slice, so k and v are
      // their own packed weights and their own GEMMs.
      k = DflashLinear(layer.qkv_proj, layer.k_proj_exl3, layer.k_proj_fp4)
              ->Apply(d, normed.t(), DType::kBF16);
      v = DflashLinear(layer.qkv_proj, layer.v_proj_exl3, layer.v_proj_fp4)
              ->Apply(d, normed.t(), DType::kBF16);
    } else {
      Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      k = DBuf(d, DType::kBF16, {C, kdim});
      v = DBuf(d, DType::kBF16, {C, kdim});
      vt::MatmulBT(d.q, k.t(), normed.t(), wk);
      vt::MatmulBT(d.q, v.t(), normed.t(), wv);
    }
    // K-norm over head_dim, then NeoX RoPE on K at the context positions (V raw).
    Tensor k2 = Reshape(k.t(), {C * Hkv, Dh});
    Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
    Tensor k3 = Reshape(k.t(), {C, Hkv, Dh});
    DBuf rope_scratch(d, DType::kBF16, {C, Hkv, Dh});
    rope_scratch.Zero(d);
    Tensor scratch3 = rope_scratch.t();
    vt::RopeNeox(d.q, k3, scratch3, cpos, MakeRopeArgs(config));
    out.k.push_back(std::move(k));  // bf16 [C, kdim] contiguous (RoPE'd view aliases it)
    out.v.push_back(std::move(v));  // bf16 [C, kdim] raw
  }
  return out;
}

// The f32-host entry every pre-W8 caller used: upload, cast to bf16, and run
// the SAME core. Kept as the marshaling shell (D3/D9/D11 parity surfaces and
// the host append feed host floats).
ContextKVDev PrecomputeContextKVDevice(Dev d, const float* context_states,
                                       const int32_t* context_positions, int64_t C,
                                       const Qwen3DFlashWeights& weights,
                                       const HfConfig& config) {
  const int64_t H = config.hidden_size;
  ContextKVDev out;
  out.num_ctx = C;
  if (C == 0) return out;
  DBuf ctx32(d, DType::kF32, {C, H}, context_states);
  DBuf ctxb(d, DType::kBF16, {C, H});
  vt::CastBf16(d.q, ctxb.t(), ctx32.t());
  DBuf cpos(d, DType::kI32, {C}, context_positions);
  return PrecomputeContextKVDeviceBf16(d, ctxb.t(), cpos.t(), C, weights, config);
}

}  // namespace

DflashPrepareOutputs PrepareDflashInputs(const DflashPrepareBatch& b) {
  // Pure-integer host port of _prepare_dflash_inputs_kernel (dflash/speculator.py:
  // 472-618). Every store the Triton kernel makes is reproduced here; there is no
  // float math, so this is bit-exact by construction.
  const int32_t num_reqs = static_cast<int32_t>(b.idx_mapping.size());
  VT_CHECK(num_reqs > 0, "prepare_dflash_inputs: num_reqs must be > 0");
  VT_CHECK(static_cast<int32_t>(b.target_query_start_loc.size()) == num_reqs + 1,
           "prepare_dflash_inputs: target_query_start_loc must be [num_reqs+1]");
  const int32_t nqpr = b.num_query_per_req;
  const int32_t nspec = b.num_speculative_steps;
  const int32_t stride = b.block_table_stride;
  const int32_t bs = b.block_size;
  const int64_t num_target_tokens = b.target_query_start_loc.back();

  DflashPrepareOutputs o;
  o.input_ids.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_positions.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_start_loc.assign(static_cast<size_t>(b.max_num_reqs) + 1, 0);
  o.seq_lens.assign(static_cast<size_t>(b.max_num_reqs), 0);
  o.query_slot_mapping.assign(static_cast<size_t>(b.max_num_tokens), 0);
  o.context_positions.assign(static_cast<size_t>(num_target_tokens), 0);
  o.context_slot_mapping.assign(static_cast<size_t>(num_target_tokens), 0);
  o.sample_indices.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_pos.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_idx_mapping.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);

  const int32_t sample_off = b.sample_from_anchor ? 0 : 1;

  for (int32_t r = 0; r < num_reqs; ++r) {
    const int32_t req_state_idx = b.idx_mapping[static_cast<size_t>(r)];
    const int32_t ctx_start = b.target_query_start_loc[static_cast<size_t>(r)];
    const int32_t ctx_end = b.target_query_start_loc[static_cast<size_t>(r) + 1];
    const int32_t num_ctx = ctx_end - ctx_start;
    const int32_t num_rejected = b.num_rejected[static_cast<size_t>(r)];
    const int32_t valid_ctx_end = ctx_end - num_rejected;
    const int32_t num_sampled = b.num_sampled[static_cast<size_t>(r)];
    const int32_t bonus_token =
        num_sampled > 0 ? b.last_sampled[static_cast<size_t>(req_state_idx)]
                        : b.next_prefill_tokens[static_cast<size_t>(req_state_idx)];
    const int64_t last_valid_pos =
        b.target_positions[static_cast<size_t>(valid_ctx_end) - 1];
    const int32_t query_base = r * nqpr;

    // --- Context positions / slots (j in [0, num_ctx)) ---
    for (int32_t j = 0; j < num_ctx; ++j) {
      const int64_t ctx_pos = b.target_positions[static_cast<size_t>(ctx_start + j)];
      int32_t ctx_block_num = static_cast<int32_t>(ctx_pos / bs);
      if (ctx_block_num > stride - 1) ctx_block_num = stride - 1;
      const int64_t ctx_block_id =
          b.block_table[static_cast<size_t>(r) * stride + ctx_block_num];
      const int64_t ctx_slot = ctx_block_id * bs + (ctx_pos % bs);
      o.context_positions[static_cast<size_t>(ctx_start + j)] = ctx_pos;
      o.context_slot_mapping[static_cast<size_t>(ctx_start + j)] = ctx_slot;
    }

    // --- Query positions / input_ids / slots + sample maps (offset in [0,nqpr)) ---
    for (int32_t off = 0; off < nqpr; ++off) {
      const int64_t query_pos = last_valid_pos + 1 + off;
      const int32_t query_idx = query_base + off;
      const int32_t input_id = (off == 0) ? bonus_token : b.parallel_drafting_token_id;
      int32_t q_block_num = static_cast<int32_t>(query_pos / bs);
      if (q_block_num > stride - 1) q_block_num = stride - 1;
      const int64_t q_block_id =
          b.block_table[static_cast<size_t>(r) * stride + q_block_num];
      const int64_t q_slot = q_block_id * bs + (query_pos % bs);
      o.input_ids[static_cast<size_t>(query_idx)] = input_id;
      o.query_positions[static_cast<size_t>(query_idx)] =
          std::min<int64_t>(query_pos, b.max_model_len - 1);
      o.query_slot_mapping[static_cast<size_t>(query_idx)] = q_slot;
      if (off >= sample_off) {
        const int32_t sample_idx = r * nspec + (off - sample_off);
        const int64_t spos = b.sample_from_anchor ? query_pos + 1 : query_pos;
        o.sample_indices[static_cast<size_t>(sample_idx)] = query_idx;
        o.sample_pos[static_cast<size_t>(sample_idx)] = spos;
        o.sample_idx_mapping[static_cast<size_t>(sample_idx)] = req_state_idx;
      }
    }

    o.query_start_loc[static_cast<size_t>(r)] = query_base;
    o.seq_lens[static_cast<size_t>(r)] = static_cast<int32_t>(last_valid_pos) + 1 + nqpr;
  }

  // --- Padding for CUDA-graph replay safety (kernel block_idx==0, req==last) ---
  const int32_t last_query_end = num_reqs * nqpr;
  for (int32_t i = num_reqs; i <= b.max_num_reqs; ++i)
    o.query_start_loc[static_cast<size_t>(i)] = last_query_end;
  // seq_lens[num_reqs, max_num_reqs) already 0 from assign.
  for (int32_t i = num_reqs * nspec; i < b.max_num_reqs * nspec; ++i) {
    o.sample_indices[static_cast<size_t>(i)] = 0;
    o.sample_pos[static_cast<size_t>(i)] = 0;
    o.sample_idx_mapping[static_cast<size_t>(i)] = -1;
  }
  for (int32_t i = num_reqs * nqpr; i < b.max_num_tokens; ++i)
    o.query_slot_mapping[static_cast<size_t>(i)] = kPadSlotId;
  return o;
}

Qwen3DFlashModel::DflashCombinedDevice Qwen3DFlashModel::CombineAuxFeaturesDevice(
    const vt::Tensor& aux_bf16, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Fin = H * weights.num_taps;
  VT_CHECK(aux_bf16.dtype == DType::kBF16,
           "qwen3_dflash fc (device): the aux tap must be bf16 — the pre-W8 host "
           "loop assumed the same dtype silently (SPEC-DFLASH2 W8, #1838)");
  VT_CHECK(aux_bf16.rank == 2 && aux_bf16.shape[1] == Fin,
           "qwen3_dflash fc (device): aux must be [T, H*num_taps]");
  const int64_t T = aux_bf16.shape[0];
  // MODEL-DFLASH2-EXL3 (#2495 item 7): ONE call site, both arms. The factory
  // binds `UnquantizedLinearMethod` when the trellis is empty, and its `Apply`
  // is exactly the `ResidentWeight` + `DBuf` + `vt::MatmulBT` sequence this line
  // replaces -- byte-for-byte the same three ops on a bf16 draft.
  DBuf comb = layers::MakeLinearMethod(weights.fc, weights.fc_exl3)
                  ->Apply(d, aux_bf16, DType::kBF16);
  VT_CHECK(comb.t().shape[0] == T && comb.t().shape[1] == H,
           "qwen3_dflash fc (device): the combine must produce [T, H]");
  DflashCombinedDevice out;
  out.tensor = comb.t();
  out.keep = comb.ReleaseShared();
  return out;
}

std::vector<float> Qwen3DFlashModel::CombineAuxFeatures(const std::vector<float>& aux_features,
                                                        int64_t T,
                                                        const Qwen3DFlashWeights& weights,
                                                        const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Fin = H * weights.num_taps;
  VT_CHECK(static_cast<int64_t>(aux_features.size()) == T * Fin,
           "qwen3_dflash fc: aux_features must be [T, H*num_taps]");
  // aux is [T, H*num_taps] f32 -> cast to bf16 -> the SAME device fc core (W8)
  // -> [T,H] bf16 -> f32 download. Bit-identical to the pre-W8 body: the cast
  // sequence is unchanged and the GEMM is the same call.
  DBuf aux32(d, DType::kF32, {T, Fin}, aux_features.data());
  DBuf auxb(d, DType::kBF16, {T, Fin});
  vt::CastBf16(d.q, auxb.t(), aux32.t());
  const DflashCombinedDevice comb = CombineAuxFeaturesDevice(auxb.t(), weights, config, queue);
  DBuf comb32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, comb32.t(), comb.tensor);
  std::vector<float> out(static_cast<size_t>(T) * H);
  comb32.Download(d, out.data());
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogits(
    const std::vector<int32_t>& input_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_dflash: positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(T),
           "qwen3_dflash: cu_seqlens must span [0,T]");
  VT_CHECK(weights.layers.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3_dflash: one layer weight per config.num_hidden_layers");
  if (weights.IsDflash2()) CheckDflashConvBatch(weights, cu);

  // Embed: hidden[T,H] bf16 = embed_tokens[input_ids]; mask slots take
  // embed_tokens[mask_token_id] naturally (in-vocab), or the dedicated mask
  // embedding when present (qwen3_dflash.py:432-438).
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.EmbedTable(), {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {T}, input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    // Substitute the dedicated mask embedding for mask_token_id rows.
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(T) * H);
    {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t r = 0; r < T; ++r)
      if (input_ids[static_cast<size_t>(r)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(r * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {T, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {T}, positions.data());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    // input_layernorm (std add+RMSNorm): dhn = norm(hidden + res); res updated.
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // SPEC-DFLASH2 W2 (#1314): attention_conv.prepare, before the sublayer.
    DBuf attn_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      attn_coef = DflashConvPrepare(d, layer.attention_conv, weights, config, &dhn);

    // attention over the context-free block (routes through DFlashBlockAttention).
    // Reuse the block helper but feed the real positions to RoPE.
    DBuf attn = [&]() -> DBuf {
      const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
      DBuf q, k, v;
      ProjectDflashQkv(d, weights, layer, dhn.t(), qdim, kdim, &q, &k, &v);
      Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
      Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
      Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
      Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
      Tensor wqn = ResidentWeight(d, layer.q_norm, {Dh});
      Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
      vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
      vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
      vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));
      Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
      DBuf a(d, DType::kBF16, {T, Hq, Dh});
      vt::DFlashBlockAttentionArgs pa;
      pa.scale = scale;
      pa.causal = layer.attn_mode.causal;
      pa.sliding_window = layer.attn_mode.sliding_window;
      pa.cu_seqlens = cu.data();
      pa.num_reqs = static_cast<int>(cu.size()) - 1;
      vt::DFlashBlockAttention(d.q, a.t(), q3, k3, v3, pa);
      Tensor o_in = Reshape(a.t(), {T, Hq * Dh});
      // MODEL-DFLASH2-EXL3 (#2495 item 7): ONE call site, both arms. On a bf16
      // draft `MakeLinearMethod` binds `UnquantizedLinearMethod`, whose `Apply`
      // IS the ResidentWeight + DBuf + MatmulBT triple this replaces.
      return DflashLinear(layer.o_proj, layer.o_proj_exl3, layer.o_proj_fp4)
          ->Apply(d, o_in, DType::kBF16);
    }();

    // SPEC-DFLASH2 W2: attention_conv.finish, over the sublayer OUTPUT with the
    // coefficients the prepare projected off the sublayer INPUT.
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.attention_conv, weights, config, &attn, attn_coef);

    // post_attention_layernorm (std add+RMSNorm).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

    // SwiGLU MLP: gate_up GEMM -> SiluAndMul -> down GEMM. gate_up+SiluAndMul run
    // through the SHARED bf16 gate-up MLP seam (layers::UnquantizedMlpGateUpMethod)
    // — byte-for-byte the same op sequence the inline path ran, now on the same
    // exemplar as qwen3.cpp MlpBlock. (Tier-A1 fold, arch-fusion-fold-plan.)
    const int64_t I = config.intermediate_size;
    // SPEC-DFLASH2 W2: mlp_conv.prepare / .finish around the MLP sublayer.
    DBuf mlp_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      mlp_coef = DflashConvPrepare(d, layer.mlp_conv, weights, config, &dh2);
    // MODEL-DFLASH2-EXL3 (#2495 item 7): the SAME seam, now selected by the
    // factory rather than hard-bound. On a bf16 draft it binds the identical
    // `UnquantizedMlpGateUpMethod` and nothing about the ops moves.
    DBuf act = DflashMlpGateUp(layer.gate_up_proj, layer.gate_proj_exl3,
                                    layer.up_proj_exl3, layer.gate_proj_fp4,
                                    layer.up_proj_fp4, I)
                   ->Apply(d, dh2.t());
    DBuf down = DflashLinear(layer.down_proj, layer.down_proj_exl3, layer.down_proj_fp4)
                    ->Apply(d, act.t(), DType::kBF16);
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.mlp_conv, weights, config, &down, mlp_coef);
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(T) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {T, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(T) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }

  DBuf logits = DflashLogitsF32D(d, dnorm.t(), weights, vocab, H);
  std::vector<float> out(static_cast<size_t>(T) * vocab);
  logits.Download(d, out.data());
  return out;
}

Qwen3DFlashModel::ContextKV Qwen3DFlashModel::PrecomputeContextKV(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t C = static_cast<int64_t>(context_positions.size());
  VT_CHECK(static_cast<int64_t>(context_states.size()) == C * H,
           "PrecomputeContextKV: context_states must be [num_ctx, H]");

  ContextKV ckv;
  ckv.num_ctx = C;
  if (C == 0) {
    ckv.k.assign(static_cast<size_t>(config.num_hidden_layers), {});
    ckv.v.assign(static_cast<size_t>(config.num_hidden_layers), {});
    return ckv;
  }

  // Device-resident projection (D7); download each layer's K/V to the host host
  // ContextKV the CPU/parity gates read. Bit-identical to the old inline path
  // (same ops, same order) — this public host API exists ONLY for the D3 kvprep
  // gates; production reaches the device buffers directly (ForwardBlockLogitsWithContext).
  ContextKVDev dev = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(), C, weights, config);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    std::vector<float> kh(static_cast<size_t>(C) * Hkv * Dh);
    std::vector<float> vh(static_cast<size_t>(C) * Hkv * Dh);
    DBuf tk(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tk.t(), Reshape(dev.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tk.Download(d, kh.data());
    DBuf tv(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tv.t(), Reshape(dev.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tv.Download(d, vh.data());
    ckv.k.push_back(std::move(kh));
    ckv.v.push_back(std::move(vh));
  }
  return ckv;
}

// Shared core (D9): the [context; block] block forward GIVEN a device-resident
// per-layer context K/V (ContextKVDev). Both ForwardBlockLogitsWithContext (which
// re-projects the whole context every step) and ForwardBlockLogitsWithPrecomputedKV
// (which uploads the persistent append-only store) build the ckv and delegate here,
// so the two paths are byte-identical downstream of how ckv's bits were obtained.
// #2202: the LEVEL-3 op split inside the batched draft forward.
//
// `VT_SPEC_TRACE=2` attributes the draft phase to `pre / fwd / select / walk`
// and put `fwd` at 76% of it. What `fwd` is MADE OF is unmeasured: after L1
// (contiguous context copy) and L2 (per-request query tiling) roughly 19-21 ms
// of it is still unexplained, and every lever guessed at without this split has
// been wrong. This is the instrument that ends the guessing.
//
// Level 3 only, and it SYNCHRONISES at each seam, so it serialises the forward
// and inflates the absolutes exactly as level 2 does. Read the SHARES, not the
// milliseconds. Inert at levels 0-2: one getenv, latched.
struct DflashOpSplit {
  double norm = 0, conv = 0, qkv = 0, qknorm_rope = 0, ctx_scatter = 0, attn = 0,
         o_proj = 0, mlp = 0, head = 0;
  int64_t layers = 0;
  bool on = false;
  vt::Backend* b = nullptr;
  vt::Queue* q = nullptr;
  std::chrono::steady_clock::time_point mark;

  void Begin(Dev& d) {
    if (!on) return;
    b = &d.b;
    q = &d.q;
    b->Synchronize(*q);
    mark = std::chrono::steady_clock::now();
  }
  // Close the open segment into `slot` and reopen at the same instant.
  void Lap(double& slot) {
    if (!on) return;
    b->Synchronize(*q);
    const auto now = std::chrono::steady_clock::now();
    slot += std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
  }
  void Report(int64_t Tq, int num_reqs) const {
    if (!on) return;
    const double t = norm + conv + qkv + qknorm_rope + ctx_scatter + attn + o_proj + mlp + head;
    if (t <= 0.0) return;
    std::fprintf(stderr,
                 "[fwd-ops] P=%d Tq=%lld L=%lld total=%.2fms norm=%.1f%% conv=%.1f%% "
                 "qkv=%.1f%% qknorm_rope=%.1f%% ctx_scatter=%.1f%% attn=%.1f%% "
                 "o_proj=%.1f%% mlp=%.1f%% head=%.1f%%\n",
                 num_reqs, static_cast<long long>(Tq), static_cast<long long>(layers), t,
                 100 * norm / t, 100 * conv / t, 100 * qkv / t, 100 * qknorm_rope / t,
                 100 * ctx_scatter / t, 100 * attn / t, 100 * o_proj / t, 100 * mlp / t,
                 100 * head / t);
  }
};

static bool DflashOpSplitEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_SPEC_TRACE");
    return e != nullptr && std::atoi(e) >= 3;
  }();
  return on;
}

static std::vector<float> ForwardWithCtxKVDev(
    Dev d, const ContextKVDev& ckv, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out,
    Qwen3DFlashModel::DflashBlockDeviceOut* device_out = nullptr) {
  VT_CHECK(device_out == nullptr || final_out == nullptr,
           "ForwardWithCtxKVDev: device_out and final_out are one hidden two ways — "
           "resident or downloaded — and no caller wants both (SPEC-DFLASH2 W8, #1837)");
  const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  VT_CHECK(static_cast<int64_t>(block_positions.size()) == Tq,
           "ForwardWithCtxKVDev: block_positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(Tq),
           "ForwardWithCtxKVDev: cu must span [0,Tq]");
  VT_CHECK(static_cast<int>(ctx_cu.size()) == num_reqs + 1 && ctx_cu.front() == 0,
           "ForwardWithCtxKVDev: ctx_cu must be [num_reqs+1]");
  const int64_t C = ckv.num_ctx;
  VT_CHECK(ctx_cu.back() == static_cast<int32_t>(C),
           "ForwardWithCtxKVDev: ctx_cu.back() must equal num_ctx");
  if (weights.IsDflash2()) CheckDflashConvBatch(weights, cu);

  // Combined [context; block] per-request layout for the attention (cu_comb), plus
  // the DEVICE index maps (D7) that place context and block K/V rows into the
  // combined buffer with vt::IndexCopy — replacing the D5 host download +
  // std::vector interleave + re-upload. These are tiny integer maps computed once
  // from the cu vectors and uploaded once. Since W12 D1 (#2087) the QUERY never
  // enters the combined buffer, so there is no output IndexSelect either: the op
  // reads the block queries where they already are.
  const int64_t Ncomb = C + Tq;
  std::vector<int32_t> cu_comb(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t cl = ctx_cu[static_cast<size_t>(r) + 1] - ctx_cu[static_cast<size_t>(r)];
    const int32_t bl = cu[static_cast<size_t>(r) + 1] - cu[static_cast<size_t>(r)];
    cu_comb[static_cast<size_t>(r) + 1] = cu_comb[static_cast<size_t>(r)] + cl + bl;
  }
  // ctx_dest[j] = combined row for context source row j (ctx_cu order).
  // blk_idx[i]  = combined row for block source row i (cu order); scatters the
  // block K/V in (IndexCopy: comb[blk_idx[i]] = block[i]). It is BY CONSTRUCTION
  // the per-request suffix `cu_comb[r+1] - (cu[r+1]-cu[r]) ...`, which is the
  // layout `DFlashBlockAttentionArgs::cu_seqlens_q` assumes.
  std::vector<int32_t> ctx_dest(static_cast<size_t>(C));
  std::vector<int32_t> blk_idx(static_cast<size_t>(Tq));
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t c0 = ctx_cu[static_cast<size_t>(r)], c1 = ctx_cu[static_cast<size_t>(r) + 1];
    const int32_t b0 = cu[static_cast<size_t>(r)], b1 = cu[static_cast<size_t>(r) + 1];
    const int32_t base = cu_comb[static_cast<size_t>(r)];
    for (int32_t j = c0; j < c1; ++j) ctx_dest[static_cast<size_t>(j)] = base + (j - c0);
    const int32_t bbase = base + (c1 - c0);
    for (int32_t i = b0; i < b1; ++i) blk_idx[static_cast<size_t>(i)] = bbase + (i - b0);
  }
  DBuf ctx_dest_d(d, DType::kI32, {C}, ctx_dest.data());
  DBuf blk_idx_d(d, DType::kI32, {Tq}, blk_idx.data());

  // Embed block tokens; substitute the dedicated mask embedding when present.
  DBuf hidden(d, DType::kBF16, {Tq, H});
  {
    Tensor dtab = ResidentWeight(d, weights.EmbedTable(), {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(Tq) * H);
    {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t rr = 0; rr < Tq; ++rr)
      if (block_input_ids[static_cast<size_t>(rr)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(rr * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {Tq, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {Tq, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {Tq}, block_positions.data());

  DflashOpSplit ops;
  ops.on = DflashOpSplitEnabled();
  ops.layers = config.num_hidden_layers;
  ops.Begin(d);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());
    ops.Lap(ops.norm);

    // SPEC-DFLASH2 W2 (#1314): attention_conv.prepare, before the sublayer.
    DBuf attn_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      attn_coef = DflashConvPrepare(d, layer.attention_conv, weights, config, &dhn);
    ops.Lap(ops.conv);

    // Block q/k/v: same per-layer path as the context-free forward.
    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    DBuf q, k, v;
    // #2202 put the MERGED QKV seam in this body, which had bypassed it: the
    // path production actually runs at c>1 kept re-reading the activation three
    // times and issuing three GEMMs where one does. MODEL-DFLASH2-EXL3 (#2495
    // item 7) then moved the whole choice into `ProjectDflashQkv`, so the three
    // bodies cannot drift apart again the way they did for a year.
    ProjectDflashQkv(d, weights, layer, dhn.t(), qdim, kdim, &q, &k, &v);
    ops.Lap(ops.qkv);
    Tensor q2 = Reshape(q.t(), {Tq * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {Tq * Hkv, Dh});
    Tensor q3 = Reshape(q.t(), {Tq, Hq, Dh});
    Tensor k3 = Reshape(k.t(), {Tq, Hkv, Dh});
    vt::RmsNorm(d.q, q2, q2, ResidentWeight(d, layer.q_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, ResidentWeight(d, layer.k_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));
    ops.Lap(ops.qknorm_rope);

    // Build the combined [context; block] q/k/v ON DEVICE (D7): scatter the layer's
    // device context K/V and this block's q/k/v into the packed combined buffer via
    // vt::IndexCopy — NO D->H download, NO host std::vector interleave, NO re-upload.
    // The bf16 values are bit-identical to the D5 f32-roundtrip path (bf16->f32->bf16
    // is an identity round-trip), so DFlashBlockAttention sees identical inputs.
    Tensor v3 = Reshape(v.t(), {Tq, Hkv, Dh});
    DBuf kcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    DBuf vcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    Tensor kcb3 = kcb.t(), vcb3 = vcb.t();
    if (C > 0) {  // this layer's device context K/V -> combined at ctx_dest
      Tensor ck2 = Reshape(ckv.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cv2 = Reshape(ckv.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cdst = ctx_dest_d.t();
      vt::IndexCopy(d.q, kcb3, ck2, cdst);
      vt::IndexCopy(d.q, vcb3, cv2, cdst);
    }
    {  // block k/v -> combined at blk_idx
      Tensor bidx = blk_idx_d.t();
      vt::IndexCopy(d.q, kcb3, k3, bidx);
      vt::IndexCopy(d.q, vcb3, v3, bidx);
    }
    // SPEC-DFLASH2 W12 D1 (#2087). The QUERY stays [Tq,...] while K/V span the
    // combined [context; block] sequence: `pa.cu_seqlens_q = cu` tells the op that
    // request r's (1+k) queries are the SUFFIX of its combined key run, which is
    // exactly where `blk_idx` put them. What this deletes is not a tidy-up:
    // before it, the op's grid ran over all `Ncomb` rows, so the draft computed an
    // attention output for EVERY context row of EVERY request in the batch and then
    // threw them away — `sum_r (ctx_r + 1 + k)^2` pairs per layer per step against
    // `(1+k) x C`, ~150x per row at the campaign's context. Gone with it: the
    // `[Ncomb,Hq,Dh]` query buffer and its zeroing memset, the `[Ncomb,Hq,Dh]`
    // output buffer, the query IndexCopy and the output IndexSelect.
    //
    // The surviving rows' arithmetic is UNCHANGED — same keys, same order, same
    // mask bound, same f32 recurrence — so this is bit-identical, and the CPU
    // fixtures in tests/vllm/v1/spec_decode/test_dflash_propose.cpp gate it as an
    // exact equality rather than a tolerance.
    DBuf a(d, DType::kBF16, {Tq, Hq * Dh});
    Tensor a3 = Reshape(a.t(), {Tq, Hq, Dh});
    vt::DFlashBlockAttentionArgs pa;
    pa.scale = scale;
    pa.causal = layer.attn_mode.causal;
    pa.sliding_window = layer.attn_mode.sliding_window;
    pa.cu_seqlens = cu_comb.data();
    pa.cu_seqlens_q = cu.data();
    pa.num_reqs = num_reqs;
    // #2089: the P>1 lane's counter. Read off the tensors that are about to be
    // passed, so a change to the launch shape moves the number.
    detail::NoteDflashCombinedAttn(q3.shape[0], kcb3.shape[0]);
    ops.Lap(ops.ctx_scatter);
    vt::DFlashBlockAttention(d.q, a3, q3, kcb3, vcb3, pa);
    ops.Lap(ops.attn);
    // MODEL-DFLASH2-EXL3 (#2495 item 7): the o_proj phase boundary MOVED by one
    // op. The weight's residency call used to be charged to `attn` because it
    // sat above this lap; `Apply` owns it now, so it is charged to `o_proj`.
    // That is the more honest attribution and it is recorded rather than
    // silent, because a phase table nobody warned is a phase table somebody
    // will read as a regression.
    DBuf attn = DflashLinear(layer.o_proj, layer.o_proj_exl3, layer.o_proj_fp4)
                    ->Apply(d, a.t(), DType::kBF16);
    ops.Lap(ops.o_proj);

    // SPEC-DFLASH2 W2 (#1314): attention_conv.finish. Its prepare ran above, on
    // the input_layernorm output, before the qkv projection.
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.attention_conv, weights, config, &attn, attn_coef);

    // post_attention_layernorm + SwiGLU MLP (unchanged from ForwardBlockLogits).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
    const int64_t I = config.intermediate_size;
    DBuf mlp_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      mlp_coef = DflashConvPrepare(d, layer.mlp_conv, weights, config, &dh2);
    // #2202 put the SHARED gate-up MLP seam in this body: the Tier-A1 fold
    // (`18ed6f038`) took `ForwardBlockLogits` and left both hot bodies on the
    // hand-roll. Its closing sentence said the point was that "a quantized
    // gate-up arm can now reach these bodies at all"; MODEL-DFLASH2-EXL3
    // (#2495 item 7) is the arm that does, and the factory is what selects it.
    DBuf act = DflashMlpGateUp(layer.gate_up_proj, layer.gate_proj_exl3,
                                    layer.up_proj_exl3, layer.gate_proj_fp4,
                                    layer.up_proj_fp4, I)
                   ->Apply(d, dh2.t());
    DBuf down = DflashLinear(layer.down_proj, layer.down_proj_exl3, layer.down_proj_fp4)
                    ->Apply(d, act.t(), DType::kBF16);
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.mlp_conv, weights, config, &down, mlp_coef);
    ops.Lap(ops.mlp);
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(Tq) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }


  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {Tq, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {Tq, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(Tq) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }
  DBuf logits = DflashLogitsF32D(d, dnorm.t(), weights, vocab, H);
  ops.Lap(ops.head);   // final norm + the [Tq, vocab] logits GEMM
  ops.Report(Tq, num_reqs);
  // SPEC-DFLASH2 W8 (#1837): the DEVICE hand-off — the same dnorm and logits
  // this function always computed, released to the caller instead of
  // downloaded. The host return is deliberately empty: downloading the full
  // f32 logits every step is the round trip the wave removes.
  if (device_out != nullptr) {
    device_out->hidden = dnorm.t();
    device_out->keep_hidden = dnorm.ReleaseShared();
    device_out->logits = logits.t();
    device_out->keep_logits = logits.ReleaseShared();
    return {};
  }
  std::vector<float> out(static_cast<size_t>(Tq) * vocab);
  logits.Download(d, out.data());
  return out;
}

// Upload a persistent host bf16 PrecomputedContextKV into per-layer device buffers,
// producing the SAME ContextKVDev the full recompute (PrecomputeContextKVDevice) would
// build — the stored bf16 bits ARE the projection output (bit-identical, D9).
static ContextKVDev UploadContextKV(Dev d,
                                    const Qwen3DFlashModel::PrecomputedContextKV& store,
                                    const HfConfig& config) {
  const int64_t C = store.num_ctx;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  ContextKVDev out;
  out.num_ctx = C;
  VT_CHECK(store.k.size() == static_cast<size_t>(config.num_hidden_layers) &&
               store.v.size() == static_cast<size_t>(config.num_hidden_layers),
           "UploadContextKV: store must hold one K/V per hidden layer");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    VT_CHECK(static_cast<int64_t>(store.k[static_cast<size_t>(l)].size()) == C * kdim &&
                 static_cast<int64_t>(store.v[static_cast<size_t>(l)].size()) == C * kdim,
             "UploadContextKV: per-layer K/V size must be num_ctx*kv_dim");
    DBuf k(d, DType::kBF16, {C, kdim},
           C > 0 ? store.k[static_cast<size_t>(l)].data() : nullptr);
    DBuf v(d, DType::kBF16, {C, kdim},
           C > 0 ? store.v[static_cast<size_t>(l)].data() : nullptr);
    out.k.push_back(std::move(k));
    out.v.push_back(std::move(v));
  }
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithContext(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const std::vector<int32_t>& ctx_cu, const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions, const std::vector<int32_t>& cu,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // Per-step FULL recompute of the context K/V (D5/D7 path): re-projects the ENTIRE
  // growing context every step (O(context^2) total). The D9 persistent path
  // (ForwardBlockLogitsWithPrecomputedKV) replaces this with an append-only store.
  ContextKVDev ckv = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(),
                                               static_cast<int64_t>(context_positions.size()),
                                               weights, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

void Qwen3DFlashModel::AppendContextKVHost(PrecomputedContextKV& store,
                                           const std::vector<float>& new_features,
                                           const std::vector<int32_t>& new_positions,
                                           const Qwen3DFlashWeights& weights,
                                           const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  const int64_t L = config.num_hidden_layers;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  if (store.k.empty()) {
    store.k.assign(static_cast<size_t>(L), {});
    store.v.assign(static_cast<size_t>(L), {});
  }
  VT_CHECK(store.k.size() == static_cast<size_t>(L) && store.v.size() == static_cast<size_t>(L),
           "AppendContextKVHost: store layer count mismatch");
  VT_CHECK(static_cast<int64_t>(new_features.size()) == count * H,
           "AppendContextKVHost: new_features must be [count, H]");
  if (count == 0) return;
  // Project ONLY the `count` new rows (positions == their absolute positions), reusing
  // the EXACT per-row projection the full recompute runs, then download the bf16 K/V
  // and append. Per-row independence (hidden_norm/KV-GEMM/k_norm/RoPE) => these bits
  // equal what a full C-row recompute would produce for these same rows.
  ContextKVDev dev = PrecomputeContextKVDevice(d, new_features.data(), new_positions.data(),
                                               count, weights, config);
  for (int64_t l = 0; l < L; ++l) {
    std::vector<uint16_t> kh(static_cast<size_t>(count) * kdim);
    std::vector<uint16_t> vh(static_cast<size_t>(count) * kdim);
    dev.k[static_cast<size_t>(l)].Download(d, kh.data());  // raw bf16 bits
    dev.v[static_cast<size_t>(l)].Download(d, vh.data());
    std::vector<uint16_t>& sk = store.k[static_cast<size_t>(l)];
    std::vector<uint16_t>& sv = store.v[static_cast<size_t>(l)];
    sk.insert(sk.end(), kh.begin(), kh.end());
    sv.insert(sv.end(), vh.begin(), vh.end());
  }
  store.num_ctx += count;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
    const PrecomputedContextKV& ckv_host, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  ContextKVDev ckv = UploadContextKV(d, ckv_host, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

// ============================ D11 Part A + D13 Part C substrate ============
// DEVICE-RESIDENT append-only draft-KV store, now a FIXED-CAPACITY PAGED cache (D13
// Part C substrate). Per draft layer a persistent pool [max_pages, block_size, Hkv, Dh]
// holds the projected bf16 K/V; a context row at absolute position p lives at flat slot
// p (page p/block_size, offset p%block_size), so the block_table is the IDENTITY and
// appending rows [L,L+count) is an IndexCopy scatter to slots [L,L+count). The store's
// shapes are STATIC (only seq_lens/block_table DATA changes as the context grows), so the
// Part B vt::DFlashPagedBlockAttention kernel reads it capture-safely and the Part C
// draft-step CUDA graph can be captured over these persistent buffers. BIT-IDENTICAL to
// the D9/D11 contiguous store by per-row projection independence (same
// PrecomputeContextKVDevice, same ascending-position append order) — the bf16 bits are
// merely placed at fixed paged slots instead of appended chunks; tokens+acceptance are
// unchanged.
constexpr int64_t kDflashPageSize = 16;  // rows per paged context page (block_size)

// #1919: the store's capacity used to be `kDflashMaxCtxSlots = 4096` right here,
// a compile-time constant unrelated to the `max_model_len` the engine advertises
// and admits. It is now RESOLVED (ResolveCtxStoreSizing below) and passed in.
// What remains a constant is the BYTE BUDGET the resolution is capped at,
// because `max_model_len` alone can be absurd: the pool is per request, per
// draft layer, bf16, K and V, so a 262144-token context costs about a gigabyte
// per concurrent request and unbounded device residency has OOM-rebooted this
// box (#1647).
//
// THE BUDGET IS THE AGGREGATE, because the residency is. One store is built per
// BATCH ROW (`runner.cpp`, the reused-slot rebuild), so what the device holds is
// `bytes_per_request * max_num_reqs`, and `gpu_memory_utilization` accounts none
// of it. A 256 MiB PER-REQUEST budget was therefore an 8 GiB peak at the
// `--max-num-seqs 32` `docs/USAGE.md` itself shows — the same
// unbounded-residency shape #1647 names, one indirection further out, and a term
// large enough to move a concurrency ladder that does not know it is there.
//
// 8 GiB is a CHOICE and not a measurement, and it is deliberately the aggregate
// the 256 MiB per-request shape ALREADY allowed at that documented
// `--max-num-seqs 32`: the default is behaviour-preserving there, it spends less
// below that concurrency and refuses to spend more above it, and what changed is
// that the number now bounds what the device actually holds. The startup line
// states the resolved per-request cost AND that aggregate, so an operator sees
// the term rather than discovering it. `VT_DFLASH_CTX_MAX_TOKENS` overrides the
// cap in TOKENS per request.
constexpr int64_t kDflashCtxTotalBudgetBytes = 8LL * 1024 * 1024 * 1024;

// SPEC-DFLASH2 W11 (#1890): route the draft block's ATTENTION through the SHARED
// paged seam (vt::ReshapeAndCache into the store's own pages, then
// vt::PagedAttention over [0, C+Tq)) instead of the bespoke
// vt::DFlashPagedBlockAttention op, whose block K/V live in no paged cache and
// therefore cannot reach any split-KV lane. DEFAULT ON
// (parity-enablers-ship-as-defaults); =0 restores the bespoke op for a
// same-binary A/B. BIT-IDENTICAL on CPU — the two kernels are the same
// three-pass online softmax in the same j-ascending order over the same bf16
// bits, which the wave's mask-parity case asserts byte-for-byte. On CUDA it is
// the near-tie class the lane already carries (VT_DFLASH_ATTN_BLOCK=1 is the
// existing bit-identical rollback for this op). MUST match
// cuda_paged_attn.cu Fa2DflashBlockEnabled(), which admits the routed read onto
// the FA-2 split-KV lane. Read fresh (host path, once per draft forward).
bool DflashBlockPagedRouteEnabled() {
  const char* e = std::getenv("VT_FA2_DFLASH_BLOCK");
  return e == nullptr || e[0] != '0';
}

struct DflashDeviceKVStore {
  // Per draft layer: a persistent bf16 paged pool [max_pages, block_size, Hkv, Dh].
  std::vector<DBuf> pool_k;
  std::vector<DBuf> pool_v;
  std::unique_ptr<DBuf> block_table;  // [1, max_pages] i32 identity (persistent)
  std::unique_ptr<DBuf> seq_lens;     // [1] i32 = num_ctx (persistent, updated on append)
  int64_t num_layers = 0;
  int64_t num_ctx = 0;
  int64_t max_pages = 0;
  int64_t block_size = 0;
  int64_t kdim = 0;  // Hkv*Dh

  // D13 Part C — per-request CUDA graph over the persistent (1+k) paged draft step.
  // All graph inputs are persistent device buffers: g_hidden (embed target, refreshed
  // OUTSIDE the graph each step), g_dpos (block positions, refreshed in place), g_cu
  // (cu_seqlens {0,Tq}, constant), plus the store's own pools/block_table/seq_lens.
  // The growing context enters purely through the in-place seq_lens VALUE, so the same
  // captured graph replays as the context grows. g_logits holds the graph output.
  std::unique_ptr<DBuf> g_hidden;   // [Tq, H] bf16
  std::unique_ptr<DBuf> g_dpos;     // [Tq] i32
  std::unique_ptr<DBuf> g_cu;       // [2] i32 {0, Tq}
  std::unique_ptr<DBuf> g_logits;   // [Tq, vocab] f32 (persistent graph output)
  // SPEC-DFLASH2 W8 (#1837): the post-final-norm hidden, captured beside the
  // logits for a DFlash2 draft (the candidate selector's projection input).
  // nullptr for a DFlash1 draft, whose capture is byte-identical to pre-W8.
  std::unique_ptr<DBuf> g_final_hidden;  // [Tq, H] bf16 (persistent graph output)
  // ENG-CUDAGRAPH-BREAK W5 (#1335): the instantiated graph, the ownership of its
  // handle, its release and its `captured()` state live in the SHARED SEAM
  // instead of in a raw `void*` plus a `Backend*` this store kept alive only so
  // its destructor could call `DestroyGraph`. `vt::BreakableGraph` releases every
  // segment it holds through `Backend::DestroyGraph`, which is the routing that
  // lets ENG-CUDAGRAPH-DEDUP (#1162) interpose at the backend later without
  // editing this file.
  // SPEC-DFLASH2 W11 (#1890): the two extra persistent inputs the PAGED-SEAM
  // route reads, refreshed IN PLACE outside any capture exactly as g_dpos is.
  // g_slot_map carries the speculative write slots [C, C+Tq) and g_seq_ext the
  // extended context bound C+Tq the attention reads; the growing context still
  // enters the captured graph purely as a device VALUE. Allocated only on the
  // routed arm, so a VT_FA2_DFLASH_BLOCK=0 store is byte-for-byte the pre-W11
  // one.
  std::unique_ptr<DBuf> g_slot_map;  // [Tq] i64 (paged write slots)
  std::unique_ptr<DBuf> g_seq_ext;   // [1]  i32 = num_ctx + Tq
  vt::BreakableGraph g_graph;
  int64_t g_tq = -1;                // captured (1+k); -1 = not yet
  int g_state = 0;                  // 0 cold, 1 warm (pool warmed, capture next), 2 captured
  // W11: the ROUTE the graph was captured under. A capture bakes the kernel
  // sequence, so a step that classifies differently must recapture rather than
  // replay the other lane — the same handling g_final_hidden already gets. -1 =
  // nothing captured yet.
  int g_route = -1;
};

// SPEC-DFLASH2 W11 (#1890): the route counters declared in
// qwen3_dflash_internal.h. Definition here, in the one translation unit that
// takes the decision, for the reason cudagraph_dispatch.cpp gives for the same
// shape: a header-defined mutable global gets one copy per translation unit and
// a gate that reads one copy while the forward writes another is the
// broken-instrument failure.
namespace detail {
namespace {
DflashBlockRouteStats& RouteStats() {
  static DflashBlockRouteStats s;
  return s;
}
}  // namespace
DflashBlockRouteStats GetDflashBlockRouteStats() { return RouteStats(); }

std::string FormatDflashBlockRouteStats(const DflashBlockRouteStats& s) {
  char buf[256];
  const int n = std::snprintf(
      buf, sizeof(buf),
      "[dflash-route] paged_seam=%lld block_kernel=%lld combined=%lld "
      "last_combined_q=%lld last_combined_k=%lld",
      static_cast<long long>(s.paged_seam_calls),
      static_cast<long long>(s.block_kernel_calls),
      static_cast<long long>(s.materialized_combined_calls),
      static_cast<long long>(s.last_combined_query_rows),
      static_cast<long long>(s.last_combined_key_rows));
  return n > 0 ? std::string(buf, static_cast<size_t>(n) < sizeof(buf)
                                      ? static_cast<size_t>(n)
                                      : sizeof(buf) - 1)
               : std::string();
}
void ResetDflashBlockRouteStats() { RouteStats() = DflashBlockRouteStats{}; }
void NoteDflashBlockRoute(DflashBlockAttnRoute route) {
  switch (route) {
    case DflashBlockAttnRoute::kPagedSeam: ++RouteStats().paged_seam_calls; break;
    case DflashBlockAttnRoute::kMaterializedCombined:
      ++RouteStats().materialized_combined_calls;
      break;
    default: ++RouteStats().block_kernel_calls; break;
  }
}
void NoteDflashCombinedAttn(int64_t query_rows, int64_t key_rows) {
  NoteDflashBlockRoute(DflashBlockAttnRoute::kMaterializedCombined);
  RouteStats().last_combined_query_rows = query_rows;
  RouteStats().last_combined_key_rows = key_rows;
}
}  // namespace detail

// #1919: the capacity resolution. Pure host arithmetic over the draft geometry
// and the engine's own context, so the CPU gate covers CUDA exactly.
//
// `want` mirrors upstream's per-step draft bound
// `min(max_seq_len + num_query_per_req, max_model_len)`
// (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:331-333`) read from the
// other side: the store must hold the whole advertised context PLUS the (1+k)
// query block the W11 paged route writes at slots `[C, C+Tq)`.
Qwen3DFlashModel::DflashCtxStoreSizing Qwen3DFlashModel::ResolveCtxStoreSizing(
    const HfConfig& config, int64_t max_model_len, int64_t num_query_per_req,
    int64_t max_num_reqs) {
  const auto round_up = [](int64_t n) {
    return ((n + kDflashPageSize - 1) / kDflashPageSize) * kDflashPageSize;
  };
  const auto round_down = [](int64_t n) { return (n / kDflashPageSize) * kDflashPageSize; };

  DflashCtxStoreSizing z;
  z.page_size = kDflashPageSize;
  // K and V, every draft layer, one context row.
  z.bytes_per_slot = config.num_hidden_layers *
                     (config.num_key_value_heads * config.head_dim) *
                     static_cast<int64_t>(sizeof(uint16_t)) * 2;
  if (z.bytes_per_slot <= 0) z.bytes_per_slot = 1;
  z.want_slots = round_up(std::max<int64_t>(max_model_len, 0) +
                          std::max<int64_t>(num_query_per_req, 0));
  if (z.want_slots < kDflashPageSize) z.want_slots = kDflashPageSize;

  // One store per BATCH ROW, so the budget is divided by the rows that can hold
  // one at the same time. A zero or negative count would divide the whole
  // aggregate into one request, which is the per-request budget this parameter
  // exists to remove, so it floors at one.
  z.max_num_reqs = std::max<int64_t>(max_num_reqs, 1);
  z.budget_bytes = kDflashCtxTotalBudgetBytes;
  const char* override_env = std::getenv("VT_DFLASH_CTX_MAX_TOKENS");
  int64_t cap_slots = 0;
  if (override_env != nullptr && override_env[0] != '\0') {
    const long long v = std::atoll(override_env);
    if (v > 0) {
      z.overridden = true;
      cap_slots = round_down(static_cast<int64_t>(v));
    }
  }
  if (!z.overridden)
    cap_slots = round_down(z.budget_bytes / (z.bytes_per_slot * z.max_num_reqs));
  // A store that cannot hold one page cannot hold one block, which is not a
  // smaller store but a broken one.
  if (cap_slots < kDflashPageSize) cap_slots = kDflashPageSize;
  z.budget_slots = cap_slots;
  // AFTER the floor, so the reported budget is the one that was actually
  // applied. Computing it from the pre-floor count let a sub-page override
  // report a zero-byte budget for a store that in fact holds a page. It is the
  // AGGREGATE that is reported, because that is the quantity the budget bounds
  // and the one the device pays.
  z.budget_bytes = z.budget_slots * z.bytes_per_slot * z.max_num_reqs;

  z.slots = std::min(z.want_slots, z.budget_slots);
  z.capped = z.slots < z.want_slots;
  z.bytes_per_request = z.slots * z.bytes_per_slot;
  z.bytes_total = z.bytes_per_request * z.max_num_reqs;
  return z;
}

std::shared_ptr<DflashDeviceKVStore> Qwen3DFlashModel::MakeDeviceKVStore(
    const HfConfig& config, vt::Queue& queue, int64_t max_ctx_slots) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t L = config.num_hidden_layers;
  VT_CHECK(max_ctx_slots > 0 && max_ctx_slots % kDflashPageSize == 0,
           "MakeDeviceKVStore: the context store's capacity must be a positive multiple "
           "of the page size; resolve it with Qwen3DFlashModel::ResolveCtxStoreSizing "
           "(SPEC-DFLASH2, #1919)");
  auto s = std::make_shared<DflashDeviceKVStore>();
  s->num_layers = L;
  s->block_size = kDflashPageSize;
  s->max_pages = max_ctx_slots / kDflashPageSize;
  s->kdim = Hkv * Dh;
  s->pool_k.reserve(static_cast<size_t>(L));
  s->pool_v.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    s->pool_k.emplace_back(d, DType::kBF16,
                           std::vector<int64_t>{s->max_pages, s->block_size, Hkv, Dh});
    s->pool_v.emplace_back(d, DType::kBF16,
                           std::vector<int64_t>{s->max_pages, s->block_size, Hkv, Dh});
  }
  // Identity block_table (logical page p -> physical page p) + zero seq_lens, uploaded
  // once; these persistent device buffers never move, so a captured graph reads the
  // growing context purely through the in-place seq_lens value.
  std::vector<int32_t> bt(static_cast<size_t>(s->max_pages));
  for (int64_t p = 0; p < s->max_pages; ++p) bt[static_cast<size_t>(p)] = static_cast<int32_t>(p);
  s->block_table = std::make_unique<DBuf>(d, DType::kI32,
                                          std::vector<int64_t>{1, s->max_pages}, bt.data());
  const int32_t zero = 0;
  s->seq_lens = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{1}, &zero);
  return s;
}

int64_t Qwen3DFlashModel::DeviceKVNumCtx(const DflashDeviceKVStore& store) {
  return store.num_ctx;
}

int64_t Qwen3DFlashModel::DeviceKVCapacity(const DflashDeviceKVStore& store) {
  return store.max_pages * store.block_size;
}

namespace {

// The shared append TAIL (SPEC-DFLASH2 W8, #1838): the capacity/contiguity
// refusals and the paged-slot IndexCopy scatter, factored out of
// AppendContextKVDevice so the device-fed AppendContextKVDeviceRows runs the
// IDENTICAL bytes-to-slots path rather than a second copy of it.
void ScatterProjectedContextRows(Dev d, DflashDeviceKVStore& store, const ContextKVDev& dev,
                                 const std::vector<int32_t>& new_positions,
                                 const HfConfig& config, vt::Queue& queue) {
  const int64_t L = config.num_hidden_layers;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  VT_CHECK(store.pool_k.size() == static_cast<size_t>(L) &&
               store.pool_v.size() == static_cast<size_t>(L),
           "AppendContextKVDevice: store layer count mismatch (call MakeDeviceKVStore)");
  const int64_t L0 = store.num_ctx;
  const int64_t max_slots = store.max_pages * store.block_size;
  // #1919: an INTERNAL INVARIANT, not a production refusal. The runner checks
  // the store's capacity before it appends and drops the request to the
  // non-speculative path when it no longer fits (`propose_drafts_block`), so
  // reaching this line means a caller appended without asking. It used to be the
  // only guard, it fired from inside an EngineCore step on any prompt above 4096
  // tokens, and it asked the operator to recompile a constant that no longer
  // exists.
  VT_CHECK(L0 + count <= max_slots,
           "AppendContextKVDevice: paged store capacity exceeded — the caller must "
           "check Qwen3DFlashModel::DeviceKVCapacity before appending, and fall back "
           "to the non-speculative path for a request that no longer fits "
           "(SPEC-DFLASH2, #1919)");
  // The runner appends only accepted-prefix rows in ascending order, so the new rows sit
  // at contiguous absolute positions [L0, L0+count) == identity paged slots [L0, L0+count).
  VT_CHECK(new_positions.front() == static_cast<int32_t>(L0) &&
               new_positions.back() == static_cast<int32_t>(L0 + count - 1),
           "AppendContextKVDevice: new_positions must be contiguous [num_ctx, num_ctx+count)");
  std::vector<int32_t> slot(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) slot[static_cast<size_t>(i)] = static_cast<int32_t>(L0 + i);
  DBuf slot_d(d, DType::kI32, {count}, slot.data());
  for (int64_t l = 0; l < L; ++l) {
    Tensor pk = Reshape(store.pool_k[static_cast<size_t>(l)].t(), {max_slots, store.kdim});
    Tensor pv = Reshape(store.pool_v[static_cast<size_t>(l)].t(), {max_slots, store.kdim});
    vt::IndexCopy(d.q, pk, dev.k[static_cast<size_t>(l)].t(), slot_d.t());
    vt::IndexCopy(d.q, pv, dev.v[static_cast<size_t>(l)].t(), slot_d.t());
  }
  store.num_ctx = L0 + count;
  // Update the persistent seq_lens (the paged kernel's context bound) in place.
  const int32_t nc = static_cast<int32_t>(store.num_ctx);
  d.b.Copy(queue, store.seq_lens->ptr(), &nc, sizeof(int32_t));
}

}  // namespace

void Qwen3DFlashModel::AppendContextKVDevice(DflashDeviceKVStore& store,
                                             const std::vector<float>& new_features,
                                             const std::vector<int32_t>& new_positions,
                                             const Qwen3DFlashWeights& weights,
                                             const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t count = static_cast<int64_t>(new_positions.size());
  VT_CHECK(static_cast<int64_t>(new_features.size()) == count * config.hidden_size,
           "AppendContextKVDevice: new_features must be [count, H]");
  if (count == 0) return;
  // Project the new rows on device (the EXACT op the D9/D11 store ran), then IndexCopy-
  // scatter each layer's [count,kdim] K/V into the fixed pools at slots [L0,L0+count).
  ContextKVDev dev = PrecomputeContextKVDevice(d, new_features.data(), new_positions.data(),
                                               count, weights, config);
  ScatterProjectedContextRows(d, store, dev, new_positions, config, queue);
}

void Qwen3DFlashModel::AppendContextKVDeviceRows(DflashDeviceKVStore& store,
                                                 const vt::Tensor& combined,
                                                 const std::vector<int32_t>& rows,
                                                 const std::vector<int32_t>& new_positions,
                                                 const Qwen3DFlashWeights& weights,
                                                 const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  VT_CHECK(static_cast<int64_t>(rows.size()) == count,
           "AppendContextKVDeviceRows: one source row per appended position");
  VT_CHECK(combined.dtype == DType::kBF16 && combined.rank == 2 && combined.shape[1] == H,
           "AppendContextKVDeviceRows: combined must be [T, H] bf16 "
           "(CombineAuxFeaturesDevice output)");
  if (count == 0) return;
  for (const int32_t r : rows)
    VT_CHECK(r >= 0 && r < static_cast<int32_t>(combined.shape[0]),
             "AppendContextKVDeviceRows: source row out of range");
  // Gather the accepted-prefix rows ON DEVICE (SPEC-DFLASH2 W8, #1838) — this
  // replaces the host float gather + f32 re-upload, and is bit-identical to it
  // because the f32 detour around these bf16 bits was an exact round trip.
  DBuf rows_d(d, DType::kI32, {count}, rows.data());
  DBuf gathered(d, DType::kBF16, {count, H});
  vt::IndexSelect(d.q, gathered.t(), combined, rows_d.t());
  DBuf cpos(d, DType::kI32, {count}, new_positions.data());
  ContextKVDev dev =
      PrecomputeContextKVDeviceBf16(d, gathered.t(), cpos.t(), count, weights, config);
  ScatterProjectedContextRows(d, store, dev, new_positions, config, queue);
}

// Whether the single-request DFlash block forward runs through the capture-safe PAGED
// kernel (Part B) reading the persistent paged store, vs the D11 materialized
// [context;block] path. Default ON (the D13 production path); VT_DFLASH_PAGED=0 selects
// the materialized fallback for a same-binary A/B.
static bool UsePagedDflashForward() {
  const char* e = std::getenv("VT_DFLASH_PAGED");
  return e == nullptr || e[0] != '0';
}

// Whether the single-request paged draft step is CUDA-graph captured + replayed (D13
// Part C). Default ON; VT_DFLASH_GRAPH=0 keeps the eager paged path (a same-binary A/B
// of exactly the capture lever, over the identical paged forward).
static bool UseDflashGraph() {
  const char* e = std::getenv("VT_DFLASH_GRAPH");
  return e == nullptr || e[0] != '0';
}

// SPEC-DFLASH2 W11 (#1890): the draft block attention's route, gathered once per
// forward from the store + config so the CALLER (which allocates the two extra
// persistent buffers) and the FORWARD (which issues the ops) read the same
// bytes. Every field is host-known; nothing here touches device memory.
static detail::DflashBlockAttnEligibility DflashBlockEligibility(
    const DflashDeviceKVStore& store, const HfConfig& config, int64_t tq) {
  detail::DflashBlockAttnEligibility e;
  e.num_reqs = 1;  // this body serves ONE request's (1+k) block, by construction
  e.tq = tq;
  e.ctx_len = store.num_ctx;
  e.max_pages = store.max_pages;
  e.block_size = store.block_size;
  e.head_dim = config.head_dim;
  e.hq = config.num_attention_heads;
  e.hkv = config.num_key_value_heads;
  e.block_table_col_stride =
      store.block_table != nullptr ? store.block_table->t().stride[1] : 0;
  // The block forward allocates q/k/v and the attention output as bf16
  // unconditionally (the DBuf constructions a few lines below), so those two
  // read true here by construction; the POOL dtype is the store's and is read.
  e.bf16_query = true;
  e.bf16_out = true;
  e.bf16_pool = !store.pool_k.empty() && !store.pool_v.empty() &&
                store.pool_k[0].t().dtype == DType::kBF16 &&
                store.pool_v[0].t().dtype == DType::kBF16;
  e.enabled = DflashBlockPagedRouteEnabled();
  return e;
}

// Capture/replay counters (proof the graph path RAN; printed when VT_DFLASH_GRAPH_STATS set).
static int64_t g_dflash_captures = 0;
static int64_t g_dflash_replays = 0;
static bool DflashGraphStats() {
  static const bool on = std::getenv("VT_DFLASH_GRAPH_STATS") != nullptr;
  return on;
}

// Capture-safe static-shape single-request draft block forward (D13 Part C). Runs the
// (1+k) block over the PAGED context store via vt::DFlashPagedBlockAttention (no
// [context;block] materialization, no function-local host uploads of ctx/blk index maps).
// The embedding is done OUTSIDE by the caller (device flag + sync stays out of any
// capture region); `hidden_in`/`dpos`/`cu_seqlens` are the caller-owned inputs (eager:
// per-call DBufs; capture: the graph slot's persistent buffers). Returns [Tq, vocab] f32
// logits ON DEVICE (the caller downloads + samples OUTSIDE the graph). Bit-identical to
// ForwardWithCtxKVDev over the same context (Part B == materialized DFlashBlockAttention).
static DBuf ForwardPagedBody(Dev d, DflashDeviceKVStore& store, const Tensor& hidden_in,
                             const Tensor& dpos, const Tensor& cu_seqlens,
                             const Tensor& slot_map, const Tensor& seq_ext,
                             const detail::DflashBlockPagedInputs& paged_host_inputs,
                             const Qwen3DFlashWeights& weights, const HfConfig& config,
                             std::optional<DBuf>* out_final_hidden = nullptr) {
  const int64_t Tq = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  // SPEC-DFLASH2 W2 (#1314): this body serves ONE request whose whole (1+k)
  // query block is rows [0, Tq), so the conv's alignment condition is just
  // Tq == conv_block_size.
  if (weights.IsDflash2()) CheckDflashConvBatch(weights, {0, static_cast<int32_t>(Tq)});
  // SPEC-DFLASH2 W11 (#1890): classified ONCE per forward — every layer of one
  // draft step shares the shape, the store and the switch, so a per-layer
  // re-classification could only differ by reading something it must not.
  const detail::DflashBlockAttnRoute route =
      detail::ClassifyDflashBlockAttn(DflashBlockEligibility(store, config, Tq));
  if (route == detail::DflashBlockAttnRoute::kPagedSeam) {
    VT_CHECK(slot_map.rank == 1 && slot_map.shape[0] == Tq &&
                 slot_map.dtype == DType::kI64 && seq_ext.rank == 1 &&
                 seq_ext.shape[0] == 1 && seq_ext.dtype == DType::kI32,
             "ForwardPagedBody(paged seam): the caller owes a [Tq] i64 slot map and "
             "a [1] i32 extended context length (SPEC-DFLASH2 W11, #1890)");
  }
  Tensor cur = hidden_in;
  std::vector<DBuf> keep;  // keep each layer's post-MLP `down` alive across iterations
  keep.reserve(static_cast<size_t>(config.num_hidden_layers));
  DBuf res(d, DType::kBF16, {Tq, H});
  res.Zero(d);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), cur, w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), cur, w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // SPEC-DFLASH2 W2 (#1314): attention_conv.prepare. This body is the one the
    // production decode path reaches (runner.cpp -> ForwardBlockLogitsWithDeviceKV)
    // and the one that is CUDA-graph captured; vt::DFlashGroupedConv is a plain
    // stream kernel with no host upload, so it captures like every other op here.
    DBuf attn_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      attn_coef = DflashConvPrepare(d, layer.attention_conv, weights, config, &dhn);

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    DBuf q, k, v;
    // #2202 put the MERGED QKV seam in this body, which had bypassed it: the
    // path production actually runs at c>1 kept re-reading the activation three
    // times and issuing three GEMMs where one does. MODEL-DFLASH2-EXL3 (#2495
    // item 7) then moved the whole choice into `ProjectDflashQkv`, so the three
    // bodies cannot drift apart again the way they did for a year.
    ProjectDflashQkv(d, weights, layer, dhn.t(), qdim, kdim, &q, &k, &v);
    Tensor q2 = Reshape(q.t(), {Tq * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {Tq * Hkv, Dh});
    Tensor q3 = Reshape(q.t(), {Tq, Hq, Dh});
    Tensor k3 = Reshape(k.t(), {Tq, Hkv, Dh});
    Tensor v3 = Reshape(v.t(), {Tq, Hkv, Dh});
    vt::RmsNorm(d.q, q2, q2, ResidentWeight(d, layer.q_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, ResidentWeight(d, layer.k_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RopeNeox(d.q, q3, k3, dpos, MakeRopeArgs(config));

    // Paged in-block attention over the persistent paged context store (Part B). The
    // output is the Tq block-query rows directly (no combined buffer, no IndexSelect).
    DBuf a3(d, DType::kBF16, {Tq, Hq, Dh});
    Tensor pool_k = store.pool_k[static_cast<size_t>(l)].t();
    Tensor pool_v = store.pool_v[static_cast<size_t>(l)].t();
    if (route == detail::DflashBlockAttnRoute::kPagedSeam) {
      // SPEC-DFLASH2 W11 (#1890) — THE SHARED SEAM. The block's own K/V become
      // RESIDENT at slots [C, C+Tq) and the whole thing is then one paged read
      // over [0, C+Tq). This is the presentation upstream uses for the same
      // work (`append_paged_kv_cache` then a paged attention), and it is the
      // ONE property that kept the draft off every split-KV lane: the bespoke
      // op below reads the block K/V out of contiguous per-layer tensors that
      // are in no cache, so no launcher that addresses K/V through a block
      // table could ever see them.
      //
      // The write is safe because slots [C, C+Tq) sit BEYOND the store's
      // `seq_lens`, so nothing reads them as context, and their only other
      // writer (`ScatterProjectedContextRows`) overwrites exactly that range
      // with the accepted rows before it advances `seq_lens`.
      //
      // The counter is moved INSIDE the branch on purpose. Recorded beside the
      // classification instead, it would count what the forward DECIDED rather
      // than what it RAN, and deleting this whole branch would leave the
      // production-runner gate green — a counter measuring a class, not a
      // capability (.agents/reachability.md). The W11 mutation pass found that
      // exact defect and this is the repair.
      detail::NoteDflashBlockRoute(detail::DflashBlockAttnRoute::kPagedSeam);
      // The write and the read are ONE call, and the mask translation lives with
      // them, so the byte-for-byte equivalence gate exercises exactly what runs
      // here rather than a transcription of it
      // (qwen3_dflash_internal.h::DflashBlockPagedAttention).
      Tensor a3t = a3.t();
      detail::DflashBlockPagedAttention(d.q, a3t, q3, k3, v3, pool_k, pool_v,
                                        store.block_table->t(), seq_ext, cu_seqlens, slot_map,
                                        paged_host_inputs, scale, layer.attn_mode.causal,
                                        layer.attn_mode.sliding_window, store.num_ctx);
    } else {
      detail::NoteDflashBlockRoute(detail::DflashBlockAttnRoute::kBlockKernel);
      vt::DFlashPagedBlockAttentionArgs pa;
      pa.scale = scale;
      pa.causal = layer.attn_mode.causal;
      pa.sliding_window = layer.attn_mode.sliding_window;
      pa.num_reqs = 1;
      pa.block_size = store.block_size;
      vt::DFlashPagedBlockAttention(d.q, a3.t(), q3, k3, v3, pool_k, pool_v, cu_seqlens,
                                    store.seq_lens->t(), store.block_table->t(), pa);
    }
    Tensor a = Reshape(a3.t(), {Tq, Hq * Dh});
    DBuf attn = DflashLinear(layer.o_proj, layer.o_proj_exl3, layer.o_proj_fp4)
                    ->Apply(d, a, DType::kBF16);

    // SPEC-DFLASH2 W2: attention_conv.finish.
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.attention_conv, weights, config, &attn, attn_coef);

    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
    const int64_t I = config.intermediate_size;
    DBuf mlp_coef(d, DType::kBF16, {0});
    if (weights.IsDflash2())
      mlp_coef = DflashConvPrepare(d, layer.mlp_conv, weights, config, &dh2);
    // #2202 put the SHARED gate-up MLP seam in this body: the Tier-A1 fold
    // (`18ed6f038`) took `ForwardBlockLogits` and left both hot bodies on the
    // hand-roll. Its closing sentence said the point was that "a quantized
    // gate-up arm can now reach these bodies at all"; MODEL-DFLASH2-EXL3
    // (#2495 item 7) is the arm that does, and the factory is what selects it.
    DBuf act = DflashMlpGateUp(layer.gate_up_proj, layer.gate_proj_exl3,
                                    layer.up_proj_exl3, layer.gate_proj_fp4,
                                    layer.up_proj_fp4, I)
                   ->Apply(d, dh2.t());
    DBuf down = DflashLinear(layer.down_proj, layer.down_proj_exl3, layer.down_proj_fp4)
                    ->Apply(d, act.t(), DType::kBF16);
    if (weights.IsDflash2())
      DflashConvFinish(d, layer.mlp_conv, weights, config, &down, mlp_coef);
    keep.push_back(std::move(down));
    cur = keep.back().t();
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {Tq, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), cur, w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), cur, w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  DBuf logits = DflashLogitsF32D(d, dnorm.t(), weights, vocab, H);
  // SPEC-DFLASH2 W8 (#1837): a DFlash2 caller keeps the post-final-norm hidden —
  // the candidate selector's projection input — as a device buffer off the SAME
  // forward. Moved out AFTER the logits GEMM read it; the move changes ownership,
  // not bytes, and a graph capture records the same kernels either way.
  if (out_final_hidden != nullptr) *out_final_hidden = std::move(dnorm);
  return logits;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
    const std::vector<DflashDeviceKVStore*>& stores, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out, DflashBlockDeviceOut* device_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t L = config.num_hidden_layers;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  const int P = static_cast<int>(stores.size());
  VT_CHECK(static_cast<int>(ctx_cu.size()) == P + 1 && ctx_cu.front() == 0,
           "ForwardBlockLogitsWithDeviceKV: ctx_cu must be [num_reqs+1]");
  VT_CHECK(device_out == nullptr || final_out == nullptr,
           "ForwardBlockLogitsWithDeviceKV: device_out and final_out are one hidden "
           "two ways — resident or downloaded — and no caller wants both "
           "(SPEC-DFLASH2 W8, #1837)");
  const int64_t C = ctx_cu.back();

  // D13 Part C — the production single-request path: run the (1+k) block through the
  // capture-safe PAGED kernel reading the persistent paged store directly (no combined
  // buffer, no function-local host index maps). The single-request propose (c1 speed gate
  // + the e2e gate, which drives one request at a time) is exactly this case.
  if (P == 1 && weights.mask_embedding.Empty() && per_layer_out == nullptr &&
      final_out == nullptr && UsePagedDflashForward()) {
    DflashDeviceKVStore& st = *stores[0];
    const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
    const int64_t H = config.hidden_size;
    const int64_t vocab = weights.draft_vocab_size;
    VT_CHECK(static_cast<int64_t>(block_positions.size()) == Tq,
             "ForwardBlockLogitsWithDeviceKV(paged): positions length mismatch");
    VT_CHECK(cu.size() == 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(Tq),
             "ForwardBlockLogitsWithDeviceKV(paged): single-request cu must be {0,Tq}");
    VT_CHECK(ctx_cu.back() == static_cast<int32_t>(st.num_ctx),
             "ForwardBlockLogitsWithDeviceKV(paged): ctx_cu.back() must equal store num_ctx");

    // `vt::GraphCaptureEnabled()` is the THIRD conjunct and it is not decoration
    // (#1352, found and fixed while landing #1335). Before W5 this driver's
    // capture was its own `BeginCapture` pair, so `VLLM_CPP_CUDAGRAPH` could not
    // reach it and the two conjuncts below were the whole predicate. The capture
    // is now the seam's, and the seam reads that switch itself — so without this
    // conjunct `VLLM_CPP_CUDAGRAPH=0` would still route into the CAPTURE lane,
    // run the eager warm pass, open an INERT scope, and run the whole
    // `ForwardPagedBody` a SECOND time inside it. Two full draft forwards per
    // propose, forever, because the driver would never reach `g_state == 2`.
    // Not wrong, just wasteful, which is exactly the kind of defect that
    // survives a token gate. Asking here makes the switch select this driver's
    // existing single-forward eager path, which is what it means everywhere else.
    // The TT captured arm of this family is not gated (#2812): explicit opt-in only.
    const bool graph_ok =
        UseDflashGraph() && vt::GraphCaptureEnabled() && d.b.SupportsGraphCapture() &&
        platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
        !platforms::GetPlatform(queue.device.type).static_graph_requires_opt_in();

    // --- Eager paged path (VT_DFLASH_GRAPH=0, VLLM_CPP_CUDAGRAPH=0, or capture
    //     unsupported) --------------------------------------------------------
    if (!graph_ok) {
      DBuf hidden(d, DType::kBF16, {Tq, H});
      {
        Tensor dtab = ResidentWeight(d, weights.EmbedTable(), {config.vocab_size, H});
        DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
        vt::Embedding(d.q, hidden.t(), dtab, dids.t());
      }
      DBuf dpos(d, DType::kI32, {Tq}, block_positions.data());
      const std::vector<int32_t> cus = {0, static_cast<int32_t>(Tq)};
      DBuf cu_d(d, DType::kI32, {2}, cus.data());
      // SPEC-DFLASH2 W11 (#1890): the paged-seam inputs, built ONLY on the
      // routed arm so `VT_FA2_DFLASH_BLOCK=0` issues exactly the pre-W11 work.
      // The host sources outlive the DBufs because the H2D copy is stream-
      // ordered (the same reason `cus` above is scoped here and not inline),
      // and `paged_in` travels DOWN as well so the routed attention can refuse
      // inputs that were not derived from the store's own context length.
      detail::DflashBlockPagedInputs paged_in;
      std::optional<DBuf> slot_d, sext_d;
      Tensor slot_t{}, sext_t{};
      if (detail::ClassifyDflashBlockAttn(DflashBlockEligibility(st, config, Tq)) ==
          detail::DflashBlockAttnRoute::kPagedSeam) {
        // ONE derivation, from the store's own context length; the routed
        // attention re-derives it and refuses a mismatch by name.
        paged_in = detail::DflashBlockPagedInputsOf(st.num_ctx, Tq);
        slot_d.emplace(d, DType::kI64, std::vector<int64_t>{Tq}, paged_in.slots.data());
        sext_d.emplace(d, DType::kI32, std::vector<int64_t>{1}, &paged_in.seq_ext);
        slot_t = slot_d->t();
        sext_t = sext_d->t();
      }
      std::optional<DBuf> hid;
      DBuf logits = ForwardPagedBody(d, st, hidden.t(), dpos.t(), cu_d.t(), slot_t, sext_t,
                                     paged_in, weights, config,
                                     device_out != nullptr ? &hid : nullptr);
      // SPEC-DFLASH2 W8 (#1837): the device hand-off — same buffers, released to
      // the caller instead of downloaded; the host return is deliberately empty.
      if (device_out != nullptr) {
        device_out->hidden = hid->t();
        device_out->keep_hidden = hid->ReleaseShared();
        device_out->logits = logits.t();
        device_out->keep_logits = logits.ReleaseShared();
        return {};
      }
      std::vector<float> out(static_cast<size_t>(Tq) * vocab);
      logits.Download(d, out.data());
      return out;
    }

    // --- CUDA-graph path (D13 Part C): cold (eager warm) -> warm (capture) -> replay.
    // (Re)allocate the persistent graph inputs when the block width changes (k is fixed
    // per config, so this fires once per request lifetime). A width change invalidates a
    // prior graph.
    if (st.g_tq != Tq) {
      // Reset() releases every segment through Backend::DestroyGraph and returns
      // the container to its as-constructed state, which is also what lets the
      // next capture open a scope on it: the scope REFUSES a container that
      // already holds one, because appending to it would leave
      // `break_count() == segment_count()` and Replay would drop the last break.
      st.g_graph.Reset();
      st.g_hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{Tq, H});
      st.g_dpos = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{Tq});
      const std::vector<int32_t> cus = {0, static_cast<int32_t>(Tq)};
      st.g_cu = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{2}, cus.data());
      st.g_logits.reset();
      st.g_final_hidden.reset();
      st.g_slot_map.reset();
      st.g_seq_ext.reset();
      st.g_tq = Tq;
      st.g_state = 0;
      st.g_route = -1;
    }
    // SPEC-DFLASH2 W11 (#1890): the ROUTE is part of the captured shape — a
    // capture bakes the kernel sequence, and the two arms issue different
    // kernels. The classification can legitimately move under a live store
    // (the capacity conjunct reads `num_ctx`, which grows), so a step that
    // classifies differently RESETS and recaptures instead of replaying the
    // other lane. Same handling `g_final_hidden` gets just below, for the same
    // reason.
    const detail::DflashBlockAttnRoute route =
        detail::ClassifyDflashBlockAttn(DflashBlockEligibility(st, config, Tq));
    if (st.g_state == 2 && st.g_route != static_cast<int>(route)) {
      st.g_graph.Reset();
      st.g_logits.reset();
      st.g_final_hidden.reset();
      st.g_state = 0;
    }
    if (route == detail::DflashBlockAttnRoute::kPagedSeam) {
      if (st.g_slot_map == nullptr)
        st.g_slot_map = std::make_unique<DBuf>(d, DType::kI64, std::vector<int64_t>{Tq});
      if (st.g_seq_ext == nullptr)
        st.g_seq_ext = std::make_unique<DBuf>(d, DType::kI32, std::vector<int64_t>{1});
    }
    // A store captured WITHOUT the hidden output cannot serve a device_out
    // replay (and the other way around): the graph's output set is part of the
    // captured shape. One draft keeps one calling convention for its lifetime,
    // so this fires only on a wiring defect — reset and recapture rather than
    // hand out a null or download a hidden the caller wanted resident.
    if (st.g_state == 2 &&
        (device_out != nullptr) != (st.g_final_hidden != nullptr)) {
      st.g_graph.Reset();
      st.g_logits.reset();
      st.g_final_hidden.reset();
      st.g_state = 0;
    }
    // Refresh the persistent graph inputs IN PLACE (fixed addresses; only contents move),
    // ALWAYS OUTSIDE the captured region: embed the block tokens into g_hidden and copy
    // this step's block positions into g_dpos. seq_lens/block_table already updated in the
    // store (append), and cu_seqlens is the constant {0,Tq}.
    {
      Tensor dtab = ResidentWeight(d, weights.EmbedTable(), {config.vocab_size, H});
      DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
      vt::Embedding(d.q, st.g_hidden->t(), dtab, dids.t());
    }
    d.b.Copy(queue, st.g_dpos->ptr(), block_positions.data(),
             static_cast<size_t>(Tq) * sizeof(int32_t));
    // SPEC-DFLASH2 W11 (#1890): the two paged-seam inputs, refreshed IN PLACE
    // here for the same reason g_dpos is — the addresses are baked into the
    // graph and only the CONTENTS move. `paged_in` stays alive to the end of
    // this scope because the copies are stream-ordered, and it is handed to
    // `ForwardPagedBody` as well so the routed attention can refuse a slot map
    // or a bound that was not derived from the store's own context length.
    detail::DflashBlockPagedInputs paged_in;
    Tensor slot_t{}, sext_t{};
    if (route == detail::DflashBlockAttnRoute::kPagedSeam) {
      // The SAME derivation the eager path uses, and the same refusal downstream:
      // what gets copied into the persistent buffers is what this produced.
      paged_in = detail::DflashBlockPagedInputsOf(st.num_ctx, Tq);
      d.b.Copy(queue, st.g_slot_map->ptr(), paged_in.slots.data(),
               static_cast<size_t>(Tq) * sizeof(int64_t));
      d.b.Copy(queue, st.g_seq_ext->ptr(), &paged_in.seq_ext, sizeof(int32_t));
      slot_t = st.g_slot_map->t();
      sext_t = st.g_seq_ext->t();
    }

    std::vector<float> out(static_cast<size_t>(Tq) * vocab);
    if (st.g_state == 2) {
      // Captured: relaunch the graph over the refreshed persistent inputs + grown context
      // (which enters purely via the in-place seq_lens value + paged store), then download.
      // Through the seam's container, never `Backend::ReplayGraph` directly: the
      // container replays its segments in order (one, here, because this capture
      // is kFull) and owns the G3 replay counter the reachability gate reads.
      st.g_graph.Replay(queue);
      if (DflashGraphStats()) {
        ++g_dflash_replays;
        if (g_dflash_replays % 32 == 0)
          std::fprintf(stderr, "[DFLASH-GRAPH] replays=%lld captures=%lld\n",
                       static_cast<long long>(g_dflash_replays),
                       static_cast<long long>(g_dflash_captures));
      }
      // SPEC-DFLASH2 W8 (#1837): a device_out caller reads the persistent graph
      // outputs in place — no download, empty keeps (the STORE owns these
      // buffers, they outlive the step, and the selector consumes them now).
      if (device_out != nullptr) {
        device_out->logits = st.g_logits->t();
        device_out->hidden = st.g_final_hidden->t();
        device_out->keep_logits = {};
        device_out->keep_hidden = {};
        return {};
      }
      st.g_logits->Download(d, out.data());
      return out;
    }

    // First propose step for this request (g_state 0): WARM-then-CAPTURE in this same step.
    // A full 27B target verify forward + KV append runs between draft steps and perturbs
    // the shared DevicePool, so an eager pass from a *previous* step does NOT guarantee the
    // free-list holds the draft's size classes at capture time (a Get miss -> cudaMalloc is
    // forbidden mid-capture). Running one eager ForwardPagedBody HERE, immediately before
    // BeginCapture with no intervening allocation, returns exactly the draft's peak-concurrent
    // blocks to the free-list (retire-don't-free, GB10 pool cap 0), so the capture's identical
    // allocation sequence is a pure pool hit. The eager pass also warms resident draft weights,
    // the RoPE cache and the cuBLASLt/workspace scratch. Its result IS this step's output.
    {
      DBuf warm_lg = ForwardPagedBody(d, st, st.g_hidden->t(), st.g_dpos->t(), st.g_cu->t(),
                                      slot_t, sext_t, paged_in, weights, config);
      // SPEC-DFLASH2 W8 (#1837): a device_out caller does not download the warm
      // result — and it must not KEEP these buffers either, because the capture
      // below relies on the free-list holding exactly what this pass returned to
      // it (a Get miss mid-capture is a forbidden cudaMalloc). Its output is
      // re-produced into the PERSISTENT graph buffers by the one replay after
      // the capture (wave spec D2), bit-identically: same kernels, same inputs.
      if (device_out == nullptr) warm_lg.Download(d, out.data());
    }  // warm_lg + all ForwardPagedBody scratch freed to the pool free-list here.
    // ENG-CUDAGRAPH-BREAK W5 (#1335): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair with its own
    // `try`/drain. The scope owns the segment, the handle, its release, the drain
    // a mid-capture throw needs, and the G3 counters.
    //
    // kFULL, INHERITED FROM W2 AND NOT RE-ARGUED. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin `5559679229`)
    // is documented at `:630-632` as a FULL graph for DECODE batches and a
    // piecewise one for prefill and mixed batches, and `decode_mode()` (`:65-66`)
    // returns the full half. This is the (1+k) DRAFT step of a speculative
    // decode, which is a decode batch, so its capture is ONE segment with the
    // attention calls INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every draft layer's attention
    // into an eager call between two graph replays, which is not vLLM's decode
    // behaviour and which nothing in this row's record supports.
    std::optional<DBuf> lg;
    std::optional<DBuf> lg_hid;
    {
      vt::GraphCaptureScope scope(d.b, queue, st.g_graph, vt::GraphCaptureMode::kFull);
      lg = ForwardPagedBody(d, st, st.g_hidden->t(), st.g_dpos->t(), st.g_cu->t(),
                            slot_t, sext_t, paged_in, weights, config,
                            device_out != nullptr ? &lg_hid : nullptr);
    }  // ~GraphCaptureScope closes the segment and files it on st.g_graph
    // NOT CAPTURED covers TWO states, and only one of them may continue.
    //
    //   * INERT (`capture_failed() == false`): unreachable here, because
    //     `graph_ok` above already required `SupportsGraphCapture()`; the
    //     remaining inert cause is `VLLM_CPP_CUDAGRAPH=0`, which the seam reads
    //     and this driver no longer does. The region ran EAGERLY, `*lg` is a real
    //     result, and the step falls back to the eager lane for good.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw.
    //     Under stream capture NOTHING between `BeginCapture` and the throw
    //     executed — every kernel was RECORDED — so `*lg` is pool-recycled memory
    //     and downloading it would hand this draft step uncomputed device memory
    //     as its logits. No fault, and a token gate cannot see it, because a
    //     draft the target rejects is indistinguishable from a bad draft.
    //
    // The pre-W5 driver rethrew after draining, and so does this.
    if (!st.g_graph.captured()) {
      if (st.g_graph.capture_failed()) {
        const std::exception_ptr err = st.g_graph.capture_error();
        st.g_graph.Reset();  // clear the failure with the graph it described
        std::fprintf(stderr, "[DFLASH-GRAPH] capture FAILED\n");
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "DFlash draft graph: the capture was ABANDONED and its logits were "
                 "never computed; refusing to return uncaptured device memory");
      }
      // INERT is now UNREACHABLE from here: `graph_ok` above required both
      // `vt::GraphCaptureEnabled()` and `SupportsGraphCapture()`, which are the
      // only two things that make a scope inert. Kept as a total branch rather
      // than an assertion, because a future inert cause added to the seam must
      // degrade to a correct eager step rather than to undefined behaviour —
      // the region DID run eagerly, so `*lg` holds real values.
      st.g_state = 0;  // stay eager, and re-warm rather than re-capture
      if (device_out != nullptr) {
        // The region ran EAGERLY, so these buffers hold real values — hand them
        // out with ownership, exactly like the eager paged lane.
        device_out->hidden = lg_hid->t();
        device_out->keep_hidden = lg_hid->ReleaseShared();
        device_out->logits = lg->t();
        device_out->keep_logits = lg->ReleaseShared();
        return {};
      }
      lg->Download(d, out.data());
      return out;
    }
    st.g_logits = std::make_unique<DBuf>(std::move(*lg));
    if (device_out != nullptr) st.g_final_hidden = std::make_unique<DBuf>(std::move(*lg_hid));
    if (DflashGraphStats()) {
      ++g_dflash_captures;
      std::fprintf(stderr, "[DFLASH-GRAPH] captured #%lld Tq=%lld C=%lld\n",
                   static_cast<long long>(g_dflash_captures), static_cast<long long>(Tq),
                   static_cast<long long>(st.num_ctx));
    }
    st.g_state = 2;  // subsequent steps replay
    // SPEC-DFLASH2 W11 (#1890): record WHICH attention route this graph baked,
    // so a later step that classifies differently recaptures instead of
    // replaying the wrong lane.
    st.g_route = static_cast<int>(route);
    // SPEC-DFLASH2 W8 (#1837): a device_out caller reads the persistent graph
    // outputs, and under real stream capture they hold NO computed values until a
    // replay — the capture RECORDED the kernels. One replay here, in the same
    // step, produces the exact bits the warm pass computed (same kernels over the
    // same persistent inputs), once per request lifetime (wave spec D2). On the
    // capture-capable CPU harness the "capture" executed eagerly, so the replay
    // recomputes nothing and the values are already the warm pass's.
    if (device_out != nullptr) {
      st.g_graph.Replay(queue);
      device_out->logits = st.g_logits->t();
      device_out->hidden = st.g_final_hidden->t();
      device_out->keep_logits = {};
      device_out->keep_hidden = {};
      return {};
    }
    return out;      // this step's output is the eager warm pass (bit-identical to the graph)
  }

  // Fallback (P>1, a separate mask embedding, a parity dump, or VT_DFLASH_PAGED=0):
  // materialize one combined per-layer [C, kdim] context by gathering each store's
  // [0,num_ctx) paged slots (ctx_cu order == ascending-position), then run the D11
  // materialized forward. Bit-identical to the paged path; not capture-targeted.
  ContextKVDev ckv;
  ckv.num_ctx = C;
  for (int64_t l = 0; l < L; ++l) {
    ckv.k.emplace_back(d, DType::kBF16, std::vector<int64_t>{C, kdim});
    ckv.v.emplace_back(d, DType::kBF16, std::vector<int64_t>{C, kdim});
  }
  if (C > 0) {
    int64_t off = 0;
    for (int r = 0; r < P; ++r) {
      const DflashDeviceKVStore& st = *stores[static_cast<size_t>(r)];
      VT_CHECK(static_cast<int64_t>(st.pool_k.size()) == L &&
                   static_cast<int64_t>(st.pool_v.size()) == L,
               "ForwardBlockLogitsWithDeviceKV: store layer count mismatch");
      const int64_t cr = st.num_ctx;
      if (cr == 0) continue;
      // #2202: BOTH index maps are the identity, so this is a contiguous copy.
      //
      // The gather was `gidx[i] = i` over `[0, cr)` and the scatter
      // `didx[i] = off + i`. A store's paged rows `[0, num_ctx)` are contiguous
      // by construction — its block table is the identity, slot `p` holds
      // position `p` — so selecting rows `0..cr-1` yields the first `cr * kdim`
      // elements of the pool, and writing rows `off..off+cr-1` fills a
      // contiguous span of the combined buffer. Neither map ever permuted
      // anything; they described a memcpy in index form.
      //
      // Four device ops per (request, layer) become one `Backend::Copy` each for
      // K and V, and the two `[cr, kdim]` temporaries disappear. At P=8, L=5
      // that is 160 index kernels plus 16 index-map uploads replaced by 80
      // device-to-device copies, and the bytes drop from 24 to 16 per element
      // (the staging read and write are gone).
      //
      // BYTE-FOR-BYTE IDENTICAL: same source elements, same destination
      // offsets, no arithmetic anywhere in the path.
      const size_t row_bytes = static_cast<size_t>(kdim) * vt::SizeOf(DType::kBF16);
      const size_t span_bytes = static_cast<size_t>(cr) * row_bytes;
      const size_t dst_off = static_cast<size_t>(off) * row_bytes;
      for (int64_t l = 0; l < L; ++l) {
        d.b.Copy(d.q, static_cast<char*>(ckv.k[static_cast<size_t>(l)].ptr()) + dst_off,
                 st.pool_k[static_cast<size_t>(l)].ptr(), span_bytes);
        d.b.Copy(d.q, static_cast<char*>(ckv.v[static_cast<size_t>(l)].ptr()) + dst_off,
                 st.pool_v[static_cast<size_t>(l)].ptr(), span_bytes);
      }
      off += cr;
    }
    VT_CHECK(off == C,
             "ForwardBlockLogitsWithDeviceKV: gathered ctx rows != ctx_cu.back()");
  }
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out, device_out);
}

}  // namespace vllm
