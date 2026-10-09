// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "vt/fp8_kv.h"
#include "vt/fused_recipe.h"
#include "vt/op_provider.h"
#include "vt/tensor.h"

namespace vt {

// Upstream-compatible vllm::ScalarType IDs without importing the Marlin-only
// scalar_type.hpp into the backend-neutral public surface. The bit packing is
// ported from csrc/core/scalar_type.hpp:80-151 @ e24d1b24. Storage DType and
// semantic type are deliberately separate: DType::kI8 never guesses whether
// its bytes contain int8, FP4, FP8, or a block scale.
using ScalarTypeId = int64_t;

namespace scalar_type {

enum class NanRepr : uint8_t { kNone = 0, kIeee754 = 1, kExtendedRangeMaxMin = 2 };

constexpr ScalarTypeId Make(uint8_t exponent, uint8_t mantissa, bool is_signed,
                            int32_t bias, bool finite_values_only, NanRepr nan_repr) {
  const uint64_t bias_bits = static_cast<uint32_t>(bias);
  return static_cast<ScalarTypeId>(
      static_cast<uint64_t>(exponent) |
      (static_cast<uint64_t>(mantissa) << 8) |
      (static_cast<uint64_t>(is_signed) << 16) |
      (bias_bits << 17) |
      (static_cast<uint64_t>(finite_values_only) << 49) |
      (static_cast<uint64_t>(nan_repr) << 50));
}

inline constexpr ScalarTypeId kF32 = Make(8, 23, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kF16 = Make(5, 10, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kBF16 = Make(8, 7, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI8 = Make(0, 7, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI32 = Make(0, 31, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI64 = Make(0, 63, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kI4 = Make(0, 3, true, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kU4 = Make(0, 4, false, 0, false, NanRepr::kIeee754);
inline constexpr ScalarTypeId kFE2M1f = Make(2, 1, true, 0, true, NanRepr::kNone);
inline constexpr ScalarTypeId kFE4M3fn =
    Make(4, 3, true, 0, true, NanRepr::kExtendedRangeMaxMin);
inline constexpr ScalarTypeId kFE8M0fnu =
    Make(8, 0, false, 0, true, NanRepr::kExtendedRangeMaxMin);

}  // namespace scalar_type

ScalarTypeId ToScalarType(DType dtype);

enum class KernelLayout : uint8_t {
  kStrided = 0,
  kPackedTwoFp4PerByte,
  kBlockScaleLinear,
  kBlockScaleSwizzled,
  kMarlinInterleaved,
};

// Explicit output layout for dynamic NVFP4 activation block scales. Keep this
// separate from KernelLayout: it is an op argument which selects how a producer
// addresses its output, not metadata inferred from a Tensor's shape. Aligned
// linear and CUTLASS-swizzled buffers can have the same dimensions.
enum class Fp4ScaleLayout : uint8_t {
  kLinear = 0,
  kCutlassSwizzled,
};

struct KernelTensorDesc {
  void* data = nullptr;
  DType storage_dtype = DType::kF32;
  ScalarTypeId scalar_type = vt::scalar_type::kF32;
  Device device;
  int rank = 0;
  int64_t shape[kMaxRank] = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};
  KernelLayout layout = KernelLayout::kStrided;
};

KernelTensorDesc Describe(const Tensor& tensor, ScalarTypeId scalar_type,
                          KernelLayout layout);

enum class OpId : uint8_t {
  kMatmul,
  kRmsNorm,
  kSiluAndMul,
  kRopeNeox,
  kEmbedding,
  kCausalConv1dFwd,
  kCausalConv1dUpdate,
  kCausalConv1dSpecUpdate,
  kL2Norm,
  kRmsNormGated,
  kGdnPrefill,
  kGdnDecode,
  kGdnSpecDecode,
  kGdnPackedDecode,
  kKdaGatedDeltaRule,
  kKdaChunkPrefill,
  kMoeRouterTopK,
  kMoeCombine,
  kAttention,
  kAttentionCross,
  kAttentionDenseFast,
  kAttentionDenseFlash,
  kAttentionDenseFa2,
  kDFlashBlockAttention,
  kDFlashPagedBlockAttention,
  kDFlashGroupedConv,
  kDflash2SelectorEdges,
  kDflash2PathWalk,
  kReshapeAndCache,
  kConcatAndCacheMla,
  // The fp8_ds_mla paged K cache (`KV-DSV4-MULTICACHE` W8, #2455). DeepSeek-V4's
  // K cache is a REGION-SPLIT byte page and NOT a rank-3
  // (num_blocks, block_size, width) tensor, so kConcatAndCacheMla — which
  // requires exactly that rank — cannot express it and these are separate ops
  // rather than a flag on it. Ported from
  // `vllm/models/deepseek_v4/common/ops/cache_utils.py` @ `5559679229`:
  // `quantize_and_insert_k_kernel` (`:36-159`) and
  // `_dequantize_and_gather_k_kernel` (`:228-341`).
  kConcatAndCacheDsMla,
  kDequantAndGatherDsMla,
  kMlaDecodeAttention,
  kMlaPrefillAttention,
  // The DSA "Lightning Indexer" selection pair (dots3-note W4b-3c, #699). The
  // vt::Tensor form of the host reference `vllm::deepseek_v4::DsaIndexerLogits`
  // / `DsaTopkSelect`, which stay as the gate's oracle. Ported from
  // `vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156` and
  // `vllm/model_executor/layers/sparse_attn_indexer.py:509`
  // (`ops.top_k_per_row_prefill`) @ `bc2d63e650`. Registered on kCPU
  // (src/vt/cpu/cpu_dsa_indexer.cpp), on kCUDA (src/vt/cuda/cuda_dsa_indexer.cu)
  // and, since MODEL-MM-QWEN4-EXP W5, on kROCM
  // (src/vt/rocm/rocm_dsa_indexer.hip). These two lines read "CPU + CUDA" until
  // 2026-09-12, by which date the kROCM arms had landed. NO OTHER DEVICE IS
  // REGISTERED, so the dispatcher still refuses those BY NAME. Additive:
  // only a model whose config carries `index_topk` dispatches them.
  kDsaIndexerLogits,
  kDsaTopkSelect,
  kGatherMlaCache,
  kMergeAttnStates,
  kPagedAttention,
  kApplyTemperature,
  kGreedyArgmax,
  kApplyTopKTopP,
  kTopKValuesIndices,
  kComputeProbs,
  kComputeLogprobs,
  kRandomSample,
  kApplyPenalties,
  kApplyMinP,
  kApplyLogitBias,
  kApplyTokenMask,
  kApplyAllowedTokenIds,
  kMatmulNvfp4,
  kScaledFp4Quant,
  kSiluMulFp4Quant,
  kSiluAndMulFp4Quant,
  kSigmoidGateFp4Quant,
  kMatmulNvfp4Fp4,
  kMatmulNvfp4Cutlass,
  kMatmulFp8Cutlass,
  kMatmulFp8CublasLt,
  kQuantFp8Static,
  kSwizzleBlockscale,
  kMoeGroupedGemmNvfp4,
  kMoeSiluMul,
  kCastBf16,
  kCastF32,
  kCastF16,
  kPermuteVHeads,
  kMulColVecF32,
  kAttnGateSplit,
  kSigmoidGateBf16,
  kGdnGBeta,
  kGdnConvSplit,
  kQkvSplit,
  kSharedExpertGate,
  kMoeCombineGate,
  kMoeGroupedGemmNvfp4Marlin,
  kGdnPostConv,
  kRopeCosSinCache,
  kAttnQkNormRopeGate,
  // Gate-FREE sibling: per-head standard RMSNorm(q)+RMSNorm(k)+partial NeoX RoPE,
  // the Qwen3-DENSE preamble. Backends may register it as kAttnQkNormRope's
  // fast_op; those that do not keep the byte-exact composite automatically.
  kAttnQkNormRope,
  kFusedChain,
  kRmsNormQuantFp8,
  kRmsNormGatedQuantFp8,
  kMatmulBT,
  // kMatmulBT with a BLOCK-QUANTIZED [N,K] weight kept in its native ggml
  // encoding — llama.cpp's `ggml_compute_forward_mul_mat`
  // (ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96).
  kMatmulBTQuant,
  // GROUPED keep-quant GEMM over an expert-index list: out[p,:] = act[p,:] .
  // weight[expert_ids[p]*N .. +N]. Collapses the DeepSeek-V4 MoE's 6 routed
  // experts x {gate,up,down} = 18 tiny T=1 kMatmulBTQuant matvecs/layer into 3
  // grouped kernels (fewer host launches + higher GPU occupancy). Same arithmetic
  // as the per-expert loop, so BYTE-IDENTICAL on the CPU provider (which loops the
  // kMatmulBTQuant kernel per group). Mirrors kMoeGroupedGemmBf16's expert-index
  // structure for the keep-quant (IQ2_XXS/Q2_K/Q8_0) tower.
  kMatmulBTQuantGrouped,
  // W0-only raw-signature probe for the shared drop-in adapter boundary. It is
  // not a production kernel-family migration.
  kDropinProbe,
  // In-place base/MRoPE rotation from a supplied dtype-specific global cache.
  kRopeFromCache,
  // Indexed persistent GDN cache boundary: one launch replaces the former
  // per-row copies plus a separate BF16<->F32 cast.
  kGdnStateGather,
  kGdnStateScatter,
  // Row gather/scatter over dim 0 (torch index_select / index_copy_). Additive
  // op powering the MIXED spec+non-spec GDN batch split/merge (SPEC-MTP): gather
  // the per-token conv inputs into compact spec / non-spec working buffers and
  // scatter the per-group core outputs back to their original row positions.
  kIndexSelect,
  kIndexCopy,
  // V-head re-indexing of a GDN projection vector. Additive op that lets the
  // qwen4exp GGUF loader keep its Gated DeltaNet projections QUANTIZED: the
  // converter's tiled V-head order is undone on the [T, N] activation instead
  // of on the weight, which a k-quant block stream cannot represent. Pure
  // re-indexing, no arithmetic. See `VHeadPermuteArgs`.
  //
  // BIT-IDENTICAL TO THE LOAD-TIME PERMUTATION FOR FOUR OF THE FIVE
  // PROJECTIONS, NOT ALL FIVE. A ROW permutation moves whole dot products, so
  // `attn_qkv`, `attn_gate`, `ssm_beta` and `ssm_alpha` are bit-identical by
  // construction. `ssm_out`'s is a COLUMN permutation: it reorders the
  // summation inside every dot product, and floating-point addition is not
  // associative, so there it is bit-identical only under exact arithmetic. That
  // it holds in this tree is a measurement at `value_dim` 96 on a bf16 CPU
  // `Matmul`, not a property of the released 6144-wide Q6_K path — see
  // `tests/vllm/models/test_gdn_v_head_permute.cpp` and
  // `qwen4_exp_weights.cpp`'s `LoadGdn`.
  //
  // THE COST IS O(T * N) PER LAYER PER FORWARD. It is net-negative per DECODE
  // step by ~2.646 GB and net-POSITIVE for one large prefill step in isolation
  // (breakeven T ~= 817; a T = 2048 prefill overpays ~4.0 GB and is repaid
  // after 1.5 decode steps). `LoadGdn` carries the full arithmetic, and neither
  // half is device-measured yet.
  kVHeadPermute,
  // BF16 grouped-MoE GEMM: the dtype-native analog of kMoeGroupedGemmNvfp4 (no
  // fp4 decode). Powers the Qwen3-Coder (Qwen3MoeForCausalLM) fast bf16 MoE path.
  kMoeGroupedGemmBf16,
  // Cross-family dense primitives introduced by the OPT (`OPTForCausalLM`)
  // bring-up — the pre-RMSNorm/pre-SwiGLU transformer vocabulary every
  // non-Qwen family needs. All three mirror torch/vLLM semantics exactly:
  //   kLayerNorm — `nn.LayerNorm` (mean+variance normalization with a BIAS
  //                term), as used by opt.py:146-148,164-166,248-251. Distinct
  //                from kRmsNorm, which subtracts no mean and has no bias.
  //   kRelu      — `get_act_fn("relu")` (opt.py:156), i.e. ReLU rather than the
  //                SwiGLU every Qwen model uses.
  //   kAdd       — elementwise add, plus the rank-1 row-BROADCAST form that
  //                applies a `nn.Linear` bias (opt.py:90-104,149-163: OPT's
  //                q/k/v/out/fc1/fc2 all carry `enable_bias` bias terms, which
  //                the bias-free Qwen projections never needed).
  kLayerNorm,
  kRelu,
  // Elementwise GELU activations (NEW for the Qwen3-VL vision tower). Both are
  // plain (non-gated) elementwise ops, may alias in-place, f32 compute →
  // out-dtype store.
  //   kGeluTanh — `gelu_pytorch_tanh` / F.gelu(approximate="tanh"): the vision
  //               MLP act_fn (qwen3_vl.py::Qwen3_VisionMLP, hidden_act
  //               "gelu_pytorch_tanh"). Same constant as kGeluAndMul's gate.
  //   kGeluErf  — exact erf GELU / nn.GELU(): the patch-merger act_fn
  //               (qwen3_vl.py::Qwen3_VisionPatchMerger, self.act_fn=nn.GELU()).
  kGeluTanh,
  kGeluErf,
  kAdd,
  // Batched dense GEMM (`torch.bmm`). The primitive MLA weight absorption is
  // expressed in — mla_attention.py:789 (q-side W_UK fold) and :1034 (W_UV
  // v-up-projection). See vt::BatchedMatmul.
  kBatchedMatmul,
  // MLA nope|rope head concatenation — upstream `concat_mla_q`
  // (csrc/libtorch_stable/concat_mla_q.cuh) and `_concat_k_nope_k_pe`
  // (mla_attention.py:2063-2092). See vt::ConcatMlaNopeRope.
  kConcatMlaNopeRope,
  // Gemma GeGLU activation: gelu_pytorch_tanh(gate) * up — the tanh-approx GELU
  // on the gate half elementwise-multiplied by the up half. The GeGLU analog of
  // kSiluAndMul, mirroring vLLM GeluAndMul(approximate="tanh") (activation.py).
  // NEW for the Gemma family (Gemma 1/2/3/4 MLP).
  kGeluAndMul,
  // Elementwise multiply by a runtime scalar: out[i] = x[i] * scalar (f32
  // compute, out-dtype store). The Gemma embedding normalizer
  // `embed_tokens(ids) * sqrt(hidden_size)` (gemma3.py:328-341). Additive; no
  // Qwen/Llama/OPT/GLM model sets it.
  kMulScalar,
  // Logit soft-cap: out[i] = cap * tanh(x[i] / cap) (f32 compute, out-dtype
  // store). The Gemma-2 final logit soft-cap (gemma2.py:344-345,
  // LogitsProcessor(soft_cap=final_logit_softcapping)) — a monotone squashing of
  // the logits. NEW for the Gemma-2 family. Additive; default-unused otherwise.
  kSoftCap,
  // Greedy speculative-decoding rejection sampling over the EXPANDED verify
  // logits `[Σ(1+k_i), vocab]` — the accept-iff-draft-equals-target-argmax rule
  // plus the bonus/replacement token. Mirrors the greedy branch of vLLM's
  // `_rejection_kernel` (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:
  // 564-585,628) + the greedy short-circuit of `_resample_kernel` (:846-861) and
  // `_insert_resampled_kernel` (:828-841). See vt::GreedyRejectionSample.
  // Additive: nothing on the non-speculative path calls it.
  kGreedyRejectionSample,
  // --- Collective transport ops (BACKEND-DISTRIBUTED-COMM W2) -----------------
  // The vt::Communicator collectives, deferred from W1 (a direct method was the
  // cleaner W1 gate; W2 routes them through OpProvider so a backend SUPPLIES the
  // transport). Two arms are registered, and only two: CPU in-process reduce
  // (src/vt/communicator.cpp:297-302) and NCCL on kCUDA
  // (src/vt/cuda/nccl_communicator.cu:152-158). This paragraph also read
  // "MLX-ring on kMETAL" until 2026-09-12; no kMETAL registration exists for any
  // of these four ids, so that transport is OWED and a Metal queue refuses all
  // four BY NAME. Each is dispatched on the queue's DeviceType and hands the
  // bound Communicator its device-specific data plane. Mirrors
  // DeviceCommunicatorBase.all_reduce:215 / all_gather:219 / send:321 /
  // recv:328 (base_device_communicator.py). See
  // vt::CommAllReduceFn (include/vt/communicator.h). Additive: nothing on the
  // world_size==1 single-GPU path dispatches them (the collective returns before
  // the lookup — parallel_state.py:638 bypass).
  kAllReduce,
  kAllGather,
  kSend,
  kRecv,
  // --- DeepSeek-V4-Flash device kernels (W7-device) --------------------------
  // The four NEW V4 op families' CUDA kernels (MHC Sinkhorn+pre/post+head, DSA
  // Lightning-Indexer+seams, Compressor pool+fp8_ds_mla KV, sqrtsoftplus/hash
  // router+clamped SwiGLU), registered through the OpProvider seam so
  // DeepseekV4Model::ForwardDevice can dispatch them. Each OpId's `fn` points at
  // a family kernels-struct (deepseek_v4_device.h) of typed device launchers,
  // each a 1:1 CUDA port of the landed portable HOST reference
  // (deepseek_v4_{mhc,dsa,compressor,moe}.{h,cpp}) it is unit-gated against. The
  // 512-wide MLA attention + expert grouped-GEMM REUSE the existing kernels
  // (kMlaDecodeAttention / kMoeGroupedGemmNvfp4) and are NOT re-ported. Additive:
  // nothing outside the V4 device forward dispatches them.
  kDeepseekV4Mhc,
  kDeepseekV4Dsa,
  kDeepseekV4Compressor,
  kDeepseekV4Moe,
  // fp8 KV-cache STORE (KV-FP8 W1). The fp8 sibling of kReshapeAndCache: the
  // paged K/V cache pages are 1-byte fp8-e4m3fn (DType::kI8 storage) and the
  // write quantizes each K/V element as Quantize(hp / k_scale|v_scale). Mirrors
  // reshape_and_cache_flash's fp8 branch (cache_kernels.cu:314-401,
  // CopyWithScaleOp :241-252). Kept a SEPARATE op so every float-cache caller
  // stays byte-identical. Appended before kCount so no existing op's id shifts.
  kReshapeAndCacheFp8,
  // SHARED fused routed-MoE gate+up+SwiGLU keep-quant epilogue. Promoted from the
  // DeepSeek-V4-private MoeDeviceKernels::moe_gate_up_swiglu seam
  // (deepseek_v4_device.h) into a first-class vt:: op so EVERY keep-quant MoE arch
  // inherits the tuned single-launch kernel — the contraction-tier sibling of the
  // grouped keep-quant GEMM kMatmulBTQuantGrouped. ONE launch computes, per
  // (expert-slot p, mid-row j): gate=gate_w[e,j]·xq, up=up_w[e,j]·xq (shared Q8_K
  // act, broadcast), then adown[p*N+j] = silu(min(gate,limit))·clamp(up,±limit) —
  // gate/up never touch HBM. BIT-IDENTICAL to {2× kMatmulBTQuantGrouped +
  // clamped-SwiGLU}; the CPU provider runs exactly that composite as the golden.
  // Appended before kCount so no existing op's id shifts.
  kMoeGateUpSwiGLUGrouped,
  // SHARED fused MLA norm-rope (Tier-A5 fold; ground: DeepSeek-V4-private
  // NormRopeRowsKernel, deepseek_v4.cpp Brick-7). ONE launch over the merged
  // kv_a projection row [T, off+rot] computes BOTH DeepSeek-MLA decoupled-rope
  // halves: latent_out[t,:off] = RmsNorm(x[t,:off]) with norm_weight (the
  // kv_a_layernorm over kv_lora_rank), and pe_out[t,:rot] = RopeFromCache-rotate
  // x[t,off:off+rot] (the UNNORMED, UNWEIGHTED decoupled k_pe). The two halves
  // are DISJOINT dims (latent normed, rope part not), so this is BIT-IDENTICAL
  // to {vt::RmsNorm(x[:,:off]); vt::RopeFromCache(x[:,off:])} — the CPU provider
  // runs exactly that composite as the golden. NOTE this reads the precomputed
  // cos|sin CACHE (like the MLA rope it replaces), NOT ds4's in-kernel analytic
  // recompute — DeepSeek-V2/kimi rope from a cache, so the cache path is what is
  // byte-exact here; ds4's own NormRopeRows stays its per-head analytic form.
  // Appended before kCount so no existing op's id shifts.
  kFusedNormRope,
  // SHARED fused BF16 grouped-MoE gate+up+SwiGLU (Tier-A4 fold). The bf16-native
  // arm of the routed-MoE gate+up+SwiGLU family (keep-quant arm is
  // kMoeGateUpSwiGLUGrouped): ONE vt entry replaces the {gate grouped GEMM; up
  // grouped GEMM; SiluAndMul} triplet the bf16 grouped-MoE archs (Qwen3-Coder,
  // DeepSeek-V2, kimi) ran. gate_w/up_w travel as the SAME per-expert bf16
  // device-pointer arrays [E] i64 kMoeGroupedGemmBf16 consumes (NOT a contiguous
  // [E*N,K] tensor — the bf16 experts are separate resident allocations), plus
  // the optional pair->token row_map. In the decode/non-WMMA regime the fused
  // kernels compute gate+up in ONE grouped launch (reusing the exact split-K
  // sequential-k accumulation) and reduce+SwiGLU in a second, dropping the two
  // f32 [P,I] HBM round-trips; in the WMMA regime it reuses LaunchGroupedBf16
  // twice + the byte-identical silu-mul. BIT-IDENTICAL to {2x kMoeGroupedGemmBf16
  // (f32 out) + kMoeSiluMul (bf16 out)} in every regime — that composite is the
  // golden the A/B unit test gates against. Registered on kCUDA
  // (src/vt/cuda/cuda_matmul_nvfp4.cu:2725) and on kROCM
  // (src/vt/rocm/rocm_ops.hip:295), like kMoeGroupedGemmBf16 (:2722 and :292).
  // Appended before kCount so no existing op's id shifts.
  kMoeGroupedGemmBf16GateUpSilu,
  // Laguna-S-2.1 device-resident-decode glue table (the 5 small host ops the
  // NVFP4/Marlin arm still ran on the host: sequential RMSNorm, partial-NeoX RoPE,
  // GQA T=1 decode attention, per-head softplus out-gate, sigmoid-noaux top-k).
  // Registered on kCUDA by cuda_laguna.cu; resolved via laguna::LagunaDevice().
  // BYTE-EXACT (sequential reductions) to the host Laguna forward. Additive: only
  // LagunaForwardResidentDecode dispatches it. Appended before kCount (no id shift).
  kLaguna,
  // DENSE Marlin W4A16 GEMM (lift of vLLM's own dense marlin.cu marlin_gemm; see
  // MarlinDenseGemm below). Byte-preserving replacement for the single-expert
  // MoE-marlin route the dense E=1 NVFP4/MXFP4 projections use today — direct-A,
  // tile-per-CTA, vLLM's own dense fp32-C_tmp reduce (no par regrouping ULP).
  // CUDA-only (Blackwell sm_12xa; vendored dense marlin TUs, VT_MARLIN_NVFP4).
  // Appended before kCount (no existing op's id shifts).
  kMarlinDenseGemm,
  // MiniMax-H3 DiT device-resident-forward glue table (brick H3-2b). Only the 3
  // small ops the shared vt:: surface does NOT already cover: the two indexed
  // AdaLN modulates and plain elementwise SiLU. Everything else in the DiT
  // forward reuses tuned shared ops (kMatmulBT, kRmsNorm, kQkvSplit, kSiluAndMul,
  // kAdd, kIndexSelect, kIndexCopy, kRopeFromCache, kDFlashBlockAttention), so
  // this table stays deliberately tiny. Registered on BOTH kCPU and kCUDA
  // (cpu_ops.cpp / cuda_minimax_h3.cu) so the device forward is exercised in CPU
  // CI too; resolved via minimax_h3::MiniMaxH3Device(). Additive: only
  // MiniMaxH3DitForwardDevice dispatches it. Appended before kCount (no id shift).
  kMiniMaxH3,
  // --- Conformer / FastConformer audio-encoder kernels (spike
  // .agents/specs/parakeet-conformer-encoder.md rows P1/P2/P3). Three primitives
  // the tree had no device op for, each mirroring a transformers 5.3.0
  // `transformers/models/parakeet/modeling_parakeet.py` module and, where the
  // structure is identical, vLLM's own native conformer
  // (`vllm/model_executor/models/conformer_encoder.py`). See vt::Conv2d,
  // vt::DepthwiseConv1d and vt::AttentionRelPos below for the exact contracts.
  // Additive: nothing outside the audio-encoder path dispatches them. Appended
  // before kCount so no existing op's id shifts.
  kConv2d,
  kDepthwiseConv1d,
  kAttentionRelPos,
  // PERF-FP8-ALPHA-FOLD (.agents/specs/perf-fp8-alpha-fold.md, #402 §3 "Lever
  // B"): the vector-alpha form of kMatmulFp8CublasLt — a per-output-column f32
  // alpha applied INSIDE the cuBLASLt epilogue instead of by a second
  // full-tensor pass. A distinct id, not a parameter of the existing op,
  // because the pointer mode lives on the matmul DESCRIPTOR and therefore
  // participates in plan/algo selection. Appended before kCount so no existing
  // op's id shifts.
  kMatmulFp8CublasLtAlphaVec,
  // --- Mamba2 / SSD selective-scan core (.agents/specs/mamba2-ssd.md W1,
  // #496). The three numerical objects the GDN arm of KERNEL-SSM-MAMBA never
  // built. SSD is NOT the gated delta rule: there is no `(I - beta*k*k^T)`
  // removal term, the decay is a diagonal `exp(A*dt)` driven by `A_log`/`dt`,
  // and `B`/`C` are shared across `n_groups` head groups. See
  // vt::Mamba2ChunkScan / vt::Mamba2StateUpdate / vt::RmsNormGatedGroup below
  // for the exact contracts and their upstream anchors. Additive: nothing
  // outside the Mamba2 path dispatches them. Appended before kCount so no
  // existing op's id shifts.
  kMamba2ChunkScan,
  kMamba2StateUpdate,
  kRmsNormGatedGroup,
  // LTX-2.5 DiT device-resident-forward glue table (phase L8). Only the seven
  // small ops the shared vt:: surface does NOT already cover: the AdaLN table
  // lookup, the AdaZero affine, the gated residual accumulate, the per-head
  // attention gate, LTX's split/interleaved RoPE, the output head's
  // table+embedded affine, and plain ungated SiLU. Everything else in the DiT
  // forward reuses tuned shared ops (kMatmulBT, kRmsNorm, kLayerNorm, kGeluTanh,
  // kAdd, kAttention, kAttentionCross), so this table stays deliberately small.
  // Registered on BOTH kCPU and kCUDA (cpu_ltx2.cpp / cuda_ltx2.cu) so the
  // device forward is exercised in CPU CI too; resolved via ltx2::Ltx2Device().
  // Additive: only Ltx2DitForwardDevice dispatches it. Appended before kCount
  // so no existing op's id shifts.
  kLtx2,
  // The NON-GATED MoE activation: out = relu(x)^2, the whole epilogue of a
  // NemotronH expert (models/nemotron_h.py:227 activation_without_mul("relu2")
  // -> MoEActivation.RELU2_NO_MUL). Sibling of kMoeSiluMul with ONE input
  // instead of two, because a non-gated expert has no gate half to multiply by
  // (nemotron_h.py:220 ckpt_names=("up_proj","down_proj","")). See vt::MoeRelu2.
  // Appended before kCount so no existing op's id shifts.
  kMoeRelu2,
  // --- BigVGAN / DAC vocoder 1-D convolutions (#672,
  // .agents/specs/minimax-music3.md §11.4). The GENERAL grouped `nn.Conv1d` and
  // the transposed `nn.ConvTranspose1d` that the three `vocoder1d` consumers
  // (MiniMax-Music3, MiniMax-H3's audio VAE, IndexTTS-2.5, plus LTX-2.5's audio
  // VAE) are built from. Two reasons these are new ids rather than parameters of
  // kDepthwiseConv1d:
  //   * `vt` had NO transposed 1-D convolution of any kind. kCausalConv1dFwd is
  //     causal/stateful/SiLU-folded and kDepthwiseConv1d is centre-padded and
  //     depthwise; neither can express a scatter that GROWS the time axis.
  //   * the bias seeding and the zero-skip differ and are part of the contract,
  //     not implementation detail — see vt::ConvTranspose1d clause (3), where
  //     the skip decides the SIGN of a zero output cell. The ACCUMULATOR WIDTH
  //     used to differ too and no longer does: kDepthwiseConv1d accumulates in
  //     f32 and since #1474 so do these, because f32 is what torch accumulates
  //     a float convolution in. Either one's width still moves a shipped
  //     model's numerics, so they remain SIBLINGS and kDepthwiseConv1d is
  //     untouched.
  // See vt::Conv1d / vt::ConvTranspose1d below for the exact contracts.
  // Appended before kCount so no existing op's id shifts.
  kConv1d,
  kConvTranspose1d,
  // --- Block-wise FP8 (VT-QUANT-FP8-GROUP, #1189 milestone M1). The DYNAMIC
  // per-token, per-group fp8 activation quant that a 128x128 block-scaled FP8
  // GEMM consumes. It is not a parameter of kQuantFp8Static: that op takes ONE
  // static per-tensor scale from the checkpoint and emits no scale tensor at
  // all, while this one derives a scale per (row, group) at run time and emits
  // an f32 [M, K/group_size] second output. The nearest existing shape is
  // kScaledFp4Quant, which is dynamic per-token with a 2-D scale as well.
  // See vt::QuantFp8Group below for the contract.
  // Appended before kCount so no existing op's id shifts.
  kQuantFp8Group,
  // --- Block-wise FP8 (VT-MATMUL-FP8-BLOCK-REF, #1189 milestone M2). The
  // 128x128 block-scaled fp8 GEMM that consumes kQuantFp8Group's output. It is
  // NOT a parameter of kMatmulFp8Cutlass and cannot be: that op folds ONE
  // scalar alpha into the epilogue, which has exactly one degree of freedom per
  // output element, while this scheme has cdiv(K, block_k) of them and applies
  // them in the MAINLOOP. See vt::MatmulFp8BlockScaled below for the contract.
  // Appended before kCount so no existing op's id shifts.
  kMatmulFp8BlockScaled,
  // --- General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007). The
  // primitive `vt` has never had on ANY device, and the reason the LTX-2.5
  // video VAE decode ran on the host while every oracle runs it GPU-resident.
  // It is NOT a parameter of kConv2d, and the difference is the ACCUMULATION
  // ORDER rather than the rank: kConv2d keeps one f32 accumulator over the whole
  // (ic, kh, kw) sweep, while this op keeps one f32 PARTIAL PER INPUT CHANNEL —
  // which is what torch's blocked-GEMM f32 convolution does, and what every
  // committed LTX-2.5 video VAE golden was taken through. Same sibling
  // relationship, and the same reason, as kConv1d vs kDepthwiseConv1d above.
  // See vt::Conv3d below for the contract. Appended before kCount so no
  // existing op's id shifts.
  kConv3d,
  // --- EXL3 (exllamav3 trellis) DEVICE kernels, MODEL-DSV4-EXL3 W2a/W2b. The
  // W1a entries below (`Exl3TileCodeword` ..  `Exl3DequantLinear`) are plain
  // host functions because they decode a CHECKPOINT FORMAT at load time. These
  // two are ops, because they run per FORWARD on whatever device the queue names
  // and are exactly what makes the trellis tower fast rather than merely
  // readable. See vt::Exl3HadR128 / vt::Exl3Gemm below for the contracts.
  // Appended before kCount so no existing op's id shifts.
  kExl3HadR128,
  kExl3Gemm,
  // MODEL-DSV4-EXL3 W2d. The FUSED mixture-of-experts MLP: one call covers every
  // routed expert of every token in a block, where `kExl3Gemm` alone costs one
  // call per (token, expert, projection). A separate id and not a widening of
  // `kExl3Gemm`, because upstream's `exl3_moe` is a separate entry point with a
  // separate argument list (`exl3_moe.cuh:8-46`). See vt::Exl3MoeMlp below.
  // Appended before kCount so no existing op's id shifts.
  kExl3MoeMlp,
  // QUANT-EXL3 W6. The reconstruct+cuBLAS GEMM path for M > 144, mirroring
  // exllamav3's AUTO_RECONSTRUCT_THRESHOLD dispatch (exl3.py:10,135). A separate
  // id and not a widening of `kExl3Gemm`, because the reconstruct path
  // materializes the dequantized weight and runs cuBLAS, while `kExl3Gemm` runs
  // the fused cooperative kernel. See vt::Exl3ReconstructGemm below.
  kExl3ReconstructGemm,
  // --- The LTX-2.5 CONV VIDEO VAE's device-resident glue table
  // (LTX25-VAE-DEVICE-RESIDENCY, #1451). The ten stages between the decode's
  // convolutions that the shared `vt::` surface does NOT express: pixel-norm,
  // GroupNorm over a channel-major volume, the ada-LN affine, the spatial-noise
  // broadcast, depth-to-space, a frame slice, a channel repeat, a channel-major
  // 1x1x1 linear, unpatchify, and the causal/spatial pad.
  //
  // It is LONGER than kLtx2's seven because a convolutional decoder is mostly
  // SHAPE MOVEMENT and this tree has no `vt::` permute, transpose,
  // depth-to-space or slice op at all, and no GroupNorm and no pixel-norm. What
  // the decode does reuse is `vt::Conv3d`, kLtx2's `silu` and `vt::Add`.
  //
  // Registered on BOTH kCPU (cpu_ltx2_vae.cpp) and kCUDA (cuda_ltx2_vae.cu), so
  // the RESIDENT STRUCTURE is exercised in CPU CI and a GPU is needed to gate
  // the KERNELS rather than the port; resolved via ltx2_vae::Ltx2VaeDevice().
  // Appended before kCount so no existing op's id shifts.
  kLtx2Vae,
  // MODEL-MM-QWEN4-EXP W5b-3 (#2156) — the PLE DILATED depthwise causal conv,
  // `Qwen4ExpTextPLELayer._short_conv` (transformers v5.16.0
  // modeling_qwen4_exp.py:1150-1167). kernel 4, dilation 3, so output t reads
  // lags {9, 6, 3, 0} and the persistent state is a 9-deep history read at
  // stride 3.
  //
  // WHY A NEW OpId AND NOT A `dilation` FIELD ON `CausalConv1dArgs`. That was
  // weighed and rejected on evidence, not on taste:
  //
  //   * `CausalConv1dArgs` is READ BY FIVE BACKENDS — cpu_ops.cpp,
  //     cuda_gdn.cu, rocm_gdn_conv.hip, vulkan_ops.cpp and
  //     tenstorrent_ops.cpp — across kCausalConv1dFwd, kCausalConv1dUpdate and
  //     kCausalConv1dSpecUpdate. A new field is silently IGNORED by every
  //     kernel that does not read it, and this is a CPU-only host: four of
  //     those five arms could not be gated here, so the field would ship as a
  //     live wrong-answer path on the Mamba/GDN/KDA/Kimi conv rather than as a
  //     refusal.
  //   * The state WIDTH is welded to `K - 1` where it is USED, and naming the
  //     wrong site would overstate this, so name both. The shared checker's
  //     width test (`CheckConvCommon`, src/vt/ops.cpp:1732) is
  //     `conv_state.shape[2] >= k - 1` — a LOWER bound, already widened once so
  //     the spec-decode cache's `(K-1)+num_spec` row fits. It is NOT the weld: a
  //     nine-column PLE state passes it (9 >= 3). The weld is
  //     `const int64_t max_query_len = state_len - (k - 1) + 1;`
  //     (src/vt/ops.cpp:1997), which reads the state width as `K - 1` plus
  //     spec-decode slack. Hand it a nine-column dilated state at K = 4 and it
  //     computes 9 - 3 + 1 = 7, a bound that means nothing, and then feeds that
  //     nonsense to the per-request query-length and accepted-token checks at
  //     :2007 and :2009. PLE needs `(K - 1) * dilation`, and there is no value
  //     of `max_query_len` that describes both geometries. Re-deriving it per
  //     mode weakens the guard for every existing caller, which is the "never
  //     make a red gate green by widening its scope" failure.
  //   * There is nothing to mirror. vLLM has no dilated conv anywhere
  //     (`git grep -in dilat` at origin/main 6a5e8f5979: zero hits in
  //     `layers/mamba/`, zero in `csrc/`), so the fused op is not a divergence
  //     from an upstream that merged the two — upstream hand-rolled this one
  //     too, saying so: "We cannot use the usual functions/kernels here for the
  //     short conv as the conv1d has dilation".
  //
  // Registered on kCPU (src/vt/cpu/cpu_qwen4_exp_ple.cpp), since W6-CUDA on
  // kCUDA (src/vt/cuda/cuda_qwen4_exp_ple.cu), and since MODEL-MM-QWEN4-EXP
  // W3-ROCm on kROCM (src/vt/rocm/rocm_qwen4_exp_ple.hip), which is a
  // transcription of the CUDA arm. BOTH device arms inherit this kernel's
  // DOUBLE four-tap accumulator rather than choosing a width of their own, read
  // `query_start_loc` and `conv_state_indices` on the DEVICE, and are gated
  // against a CPU oracle at BOTH admitted dtypes: the CUDA arm against the
  // lane-pinned transformers goldens at all three dilations
  // (tests/vllm/models/test_qwen4_exp_cuda.cpp), the ROCm arm against the
  // cross-device NMSE band (tests/vt/test_backend_cross_device.cpp), where the
  // accumulator WIDTH has its own case on each. NO OTHER DEVICE IS REGISTERED,
  // so the dispatcher still refuses those BY NAME.
  // Appended before kCount so no existing op's id shifts.
  kQwen4ExpPleConv,
  // MODEL-MM-QWEN4-EXP W5b (#2031) — the Qwen4-Exp 4-branch GATED-RESIDUAL
  // hyper-connection stream, the one structure the whole forward is threaded
  // through: `Qwen4ExpTextDecoderLayer` reads it twice per layer (attention and
  // MLP) over 48 layers and `Qwen4ExpTextModel` once more for the terminal
  // `use_combine=false` mixer, so 97 sites in the released config.
  //
  // WHY A FUSED FAMILY OP RATHER THAN A COMPOSITION of existing vt:: ops. The
  // shared surface cannot express it. There is no ungated per-group RMS norm
  // (`kRmsNormGated` has no group_size; `kRmsNormGatedGroup` requires a SILU
  // gate), no standalone `silu`/`sigmoid`, no elementwise binary multiply and no
  // axis reduction, so a composition would need five NEW general ops and would
  // still materialise the [T, hc, H] broadcast the write-back exists to avoid.
  // ONE OF THOSE FIVE NOW EXISTS AND THE RATIONALE IS UNCHANGED BY IT: W5d-1
  // (#2249 item 1) added `kRmsNormGroup` for the PLE block, so the first clause
  // above reads as the history it is — it is why that op was written — and the
  // other four are still absent, which is what makes this a fused family op.
  // `.agents/specs/qwen4-exp-flash-next.md` names this exact seam: "A device arm
  // reads `block_out` once per (j, h) tile and `injection_weights[j]` once per
  // row; it never allocates the broadcast." `kDeepseekV4Mhc` is the in-tree
  // precedent for an architecture's hyper-connection glue as one OpId.
  //
  // Registered on kCPU (cpu_qwen4_exp.cpp) and gated bit-comparably against the
  // lane-pinned transformers goldens. BOTH OPS NOW HAVE kCUDA
  // (src/vt/cuda/cuda_qwen4_exp.cu) and kROCM (src/vt/rocm/rocm_qwen4_exp.hip)
  // arms; this paragraph said `kQwen4ExpGatedResidual`'s CUDA arm was "still
  // OWED, not written" until 2026-09-12, and both had landed by then. What the
  // sentence was recording is still true of the DESIGN and is kept as that: the
  // two ops differ in that `kQwen4ExpGatedResidual` carries a grouped RMS norm
  // whose sum of squares this kernel accumulates in DOUBLE, so its device arms
  // had a reduction-width decision to make first — a measured 571x separation
  // from an f32 block reduction at group size 2560 — while
  // `kQwen4ExpGatedResidualWriteBack` has NO reduction at all and its arms are
  // BYTE-IDENTICAL to this one, `__fmul_rn`/`__fadd_rn` against the host's
  // `-ffp-contract=off`, so they meet these goldens by exactly the margin this
  // kernel does. NO OTHER DEVICE IS REGISTERED, so the dispatcher still refuses
  // those BY NAME.
  // Appended before kCount so no existing op's id shifts.
  kQwen4ExpGatedResidual,
  kQwen4ExpGatedResidualWriteBack,
  // MODEL-MM-QWEN4-EXP W5b-4 (#2167) — Qwen Sparse Attention: the pooled-key
  // COMPRESSOR that writes the indexer side cache, and the GATHER consumer that
  // reduces over only the selected raw rows.
  //
  // WHAT IS **NOT** HERE, AND THAT IS THE POINT. The MQA block SCORE and the
  // per-query top-k are NOT new ops, because this tree already has them:
  // `kDsaIndexerLogits` computes `sum_h fold[t,h] * ReLU(dot(q[t,h,:], k[s,:]))`
  // over a ONE-key-head MQA cache with a per-query `[win_start, win_end)` window,
  // which with `weights == 1`, `n_head_scale == 1` and
  // `softmax_scale == index_head_dim ** -0.5` IS the QSA block score; and
  // `kDsaTopkSelect` is the same all-select-below-k / ties-to-the-lower-index /
  // ASCENDING-emission top-k that `Qwen4ExpTextQSAIndexer` needs, over the block
  // axis instead of the token axis. Writing a QSA-private scoring kernel beside
  // them would be the parallel path AGENTS.md "Shared seams" forbids, so the
  // indexer is COMPOSED from those two and only the two things they cannot
  // express land as ops. `tests/vllm/models/test_qwen4_exp_qsa_device.cpp` gates
  // that composition against the same lane-pinned transformers goldens the W4
  // host reference answers to, so the reuse is measured rather than asserted.
  //
  // kQwen4ExpQsaCompress — mean-pool `compress_ratio` consecutive RAW index keys
  // into one, `k_layernorm` the pooled key, and RoPE it at the position of the
  // block's FIRST token (`compressed_pos = (position // CR) * CR`). Two of the
  // three stages exist as ops (`vt::RmsNorm` with `gemma=true` is the
  // `(1.0 + weight)` polarity, `vt::RopeFromCache` rotates a leading slice at
  // caller-supplied positions) and the POOL does not: this tree has no mean, no
  // pool, no axis reduction and no transpose, so the window cannot even be faked
  // as a strided depthwise conv. Fusing all three mirrors upstream, whose own
  // compressor is ONE kernel
  // (`_fused_kv_compress_norm_rope_insert_indexer_attn`), and `kFusedNormRope`
  // is the in-tree precedent for exactly that pair fused into one OpId.
  //
  // kQwen4ExpQsaGatherAttention — dense GQA over ONLY the gathered rows. Nothing
  // upstream supplies it: every DeepSeek-V4 sparse consumer attends the
  // COMPRESSED MLA KV (one state per four tokens) and MiniMax-M3's consumers
  // attend raw tokens at KV-PAGE granularity, while QSA attends RAW tokens
  // selected at ratio-4 granularity. It expands block `b` into tokens
  // [CR*b, CR*b + CR) as ADDRESSES rather than materialising an index buffer,
  // appends the always-attended ragged tail, and counts the key rows it actually
  // READ. That count is the wave's instrument: llama.cpp #27739 records that a
  // sparse MASK over a dense cache costs what dense attention costs under CUDA
  // flash attention, so a mask-shaped port passes every value comparison and
  // forfeits the long-context lever this row exists for.
  //
  // Registered on kCPU (src/vt/cpu/cpu_qwen4_exp_qsa.cpp:354,357), on kCUDA
  // (src/vt/cuda/cuda_qwen4_exp_qsa.cu:641,644) and on kROCM
  // (src/vt/rocm/rocm_qwen4_exp_qsa.hip — MODEL-MM-QWEN4-EXP W4), every one of
  // the three unconditional and at preprocessor depth 0. This paragraph read
  // "kCPU only. The CUDA arm is OWED, not written" until 2026-09-12, by which
  // date the CUDA arm had landed, and said the kROCM arms were owed until W4
  // landed them. It then read "THE ROCm FORWARD STILL REFUSES, and not at this
  // pair: the QSA block composes its indexer from kDsaIndexerLogits and
  // kDsaTopkSelect (qwen4_exp_qsa_block.cpp:391,403) and NEITHER has a ROCm
  // arm" until MODEL-MM-QWEN4-EXP W5 landed BOTH of those arms
  // (src/vt/rocm/rocm_dsa_indexer.hip), which removes the last op refusal this
  // row has MEASURED on that board.
  //
  // THAT IS NOT A CLAIM THAT THE MODEL RUNS. Nothing has loaded a `qwen4_exp`
  // checkpoint or completed a `ModelRegistry::Forward` on a ROCm device, so
  // whether the step completes is UNMEASURED and no claim either way follows
  // from a registration. W6 owns the first load and forward, and the spec's
  // `## Owed` owns the gap. The GLM-5.3-side note at rocm_ops.hip:370-375 still
  // says the pair is absent; that sentence is now false, is left deliberately
  // because `rocm_ops.hip` is a surface every ROCm row writes, and is owned by
  // BACKEND-ROCM issue #2715 — again the spec's `## Owed`.
  // Appended before kCount so no existing op's id shifts.
  kQwen4ExpQsaCompress,
  kQwen4ExpQsaGatherAttention,
  // MODEL-MM-QWEN4-EXP W5d-1 (#2249 item 1) — the UNGATED per-group RMS norm.
  // A SIBLING of kRmsNorm and of kRmsNormGatedGroup, and neither of those two
  // can stand in for it: `kRmsNorm` reduces over the WHOLE row and has no
  // group_size, `kRmsNormGated`/`kRmsNormGatedQuantFp8` fold a gate in, and
  // `kRmsNormGatedGroup` groups correctly but always multiplies by
  // `silu(gate)` first, so there is no way to ask any of them for a plain
  // grouped norm. The only grouped reduction this tree had was FUSED inside
  // `kQwen4ExpGatedResidual` and could not be called on its own, which is the
  // gap `include/vt/ops.h` states in its own words at the kQwen4ExpGatedResidual
  // comment above ("There is no ungated per-group RMS norm").
  //
  // Adding `group_size` to `RmsNormArgs` instead was REJECTED, and the reason is
  // the silent-wrong-answer shape this row keeps meeting. `kRmsNorm` is
  // registered on six backends (kCPU, kCUDA, kROCM, kVULKAN, kTENSTORRENT and
  // kMETAL); a new field on its args struct is ignored by every kernel that is
  // not taught to read it, so a CUDA or Metal caller would get a whole-row norm
  // back from a grouped request, with no crash and no
  // refusal. A separate OpId cannot do that: an unregistered device refuses BY
  // NAME. `kRmsNormGatedGroup` is the in-tree precedent for exactly this split
  // ("SIBLING of RmsNormGatedArgs, not a mode of it").
  //
  // Registered on kCPU (src/vt/cpu/cpu_ops.cpp), on kCUDA since W6-CUDA-B
  // (src/vt/cuda/cuda_rms_norm_group.cu, #2391) and on kROCM since
  // MODEL-MM-QWEN4-EXP W1-ROCm (src/vt/rocm/rocm_rms_norm_group.hip). This
  // paragraph read "kCPU only, the CUDA arm is OWED" until 2026-09-12, by which
  // time both device arms had landed; it is corrected rather than left, because
  // a knowingly false statement in a public header is worse than the edit. NO
  // OTHER DEVICE IS REGISTERED, so the dispatcher still refuses those BY NAME.
  // Appended before kCount so no existing op's id shifts.
  kRmsNormGroup,
  // MODEL-MM-QWEN4-EXP W5e-1 (#2336) — the Qwen4-Exp PLE GATE: the signed
  // square root of the hyper-connection score, and the sigmoid of it applied to
  // a per-token `value` row broadcast across the hc streams.
  // `Qwen4ExpTextPLELayer.forward`, transformers v5.16.0
  // `models/qwen4_exp/modeling_qwen4_exp.py:1181-1182` (+ the :1184 flatten):
  //
  //     gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
  //     gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
  //
  // WHAT IS **NOT** HERE, AND THAT IS THE POINT. The DOT that produces `gate`
  // (:1180) is NOT this op. `vt::BatchedMatmul` already computes it — `[T*hc, 1,
  // H] x [T*hc, H, 1]` over VIEWS of the two `[T, hc*H]` buffers, since only the
  // innermost dim must be unit-stride and `stride[0]`/`stride[1]` are free — so
  // a private scoring loop beside it would be the parallel path AGENTS.md
  // "Shared seams" forbids. `test_qwen4_exp_ple_gate.cpp` RUNS that composition
  // against the same lane-pinned golden this op is gated on, so the reuse is
  // measured rather than asserted. The `/ math.sqrt(hidden_size)` that :1180
  // ends with has no op and no home in `BatchedMatmul`, so it rides here as
  // `Qwen4ExpPleGateArgs::gate_divisor`.
  //
  // WHY A FUSED OP RATHER THAN A COMPOSITION for what is left. The shared
  // surface has no `abs`, no `clamp`, no `sqrt`, no `sign` and no standalone
  // `sigmoid` — the same absence `kQwen4ExpGatedResidual` above already
  // enumerates — so the middle line alone would need four new general ops. The
  // multiply then needs a fifth, and no existing one can serve, because BOTH of
  // its operands broadcast: `gate` is `[T, hc, 1]` and `value.unsqueeze(-2)` is
  // `[T, 1, H]`. `vt::SigmoidGateBf16` refuses it by count
  // ("sigmoid_gate_bf16: out/attn/gate must have the same element count") and
  // `vt::MulColVecF32` scales per output COLUMN, where this scales per (t, j)
  // ROW of the flattened `[T, hc*H]`. Composing it anyway means tiling `value`
  // to `[T*hc, H]` with `vt::IndexSelect` first, which materialises exactly the
  // broadcast a fused op exists to avoid.
  //
  // WHY A NEW OpId RATHER THAN A FIELD on an existing args struct. Same
  // arithmetic as `kRmsNormGroup` above: the candidate hosts are registered on
  // MORE THAN ONE backend, so a new field would be silently IGNORED by every
  // arm that does not learn it and the caller would get a wrong answer with no
  // crash and no refusal. `kSigmoidGateBf16` has FIVE registered arms — kCPU
  // (cpu_ops.cpp), kCUDA (cuda_glue.cu), kROCM (rocm_ops.hip), kVULKAN
  // (vulkan_ops.cpp) and kTENSTORRENT (tenstorrent_ops.cpp) — and
  // `kMulColVecF32` has TWO (kCPU, kCUDA). Four arms and one arm respectively
  // would answer a broadcast request with an elementwise product. An
  // unregistered device refuses BY NAME instead.
  //
  // THE CLAMP ORDER IS THE TRAP. `clamp_min` is applied BEFORE the square root,
  // so the floor on |output| is sqrt(1e-6) = 1e-3 and not 1e-6, and tiny scores
  // are AMPLIFIED. Exactly zero is the one exception and it is not a special
  // case in the code: `sign(0) == 0` kills the floor, so the origin maps to 0
  // and the function is genuinely discontinuous there. A fully masked row
  // reaches it.
  //
  // A NaN SCORE PROPAGATES, and that is a guard rather than a fall-through.
  // Upstream returns NaN here (`sign(NaN) == 0`, but `NaN * 0.0 == NaN`), while
  // every comparison in a naive signed-sqrt is FALSE for NaN, so the sign
  // branches miss and the zero arm returns 0 — turning poison into a plausible
  // `0.5 * value`. The kernel tests `isnan` first. `+/-inf` and `+/-0.0` need
  // no guard and get none; they already match the pin term for term.
  //
  // Registered on kCPU (src/vt/cpu/cpu_qwen4_exp_ple.cpp), since W6-CUDA on
  // kCUDA (src/vt/cuda/cuda_qwen4_exp_ple.cu), and since MODEL-MM-QWEN4-EXP
  // W3-ROCm on kROCM (src/vt/rocm/rocm_qwen4_exp_ple.hip). THE NaN OBLIGATION
  // ABOVE IS DISCHARGED ON ALL THREE ARMS, and each arm's own suite carries the
  // case that separates a NaN from the plausible `0.5 * value` a missing guard
  // returns: `tests/vllm/models/test_qwen4_exp_cuda.cpp` for kCUDA, and a NaN
  // in the score pool of the gate case in
  // `tests/vt/test_backend_cross_device.cpp` for kROCM, whose NaN outputs are
  // asserted AS NaN rather than averaged into a norm — a norm cannot see this
  // defect, because the missing guard returns a finite number. NO OTHER DEVICE
  // IS REGISTERED, so the dispatcher still refuses those BY NAME.
  // Appended before kCount so no existing op's id shifts.
  kQwen4ExpPleGate,
  // The DEQUANTIZING GATHER (KGATHER): `vt::Embedding` over a BLOCK-QUANTIZED
  // table, decoding one row per gathered id. It is a separate op id from
  // `kEmbedding` and not a dtype branch inside it, for one reason: a device's
  // ability to decode a block row is a CAPABILITY the GGUF residency policy has
  // to ask about BEFORE it decides where a tensor lives, and the only
  // device-agnostic way to ask is the provider table. `DeviceQuantGatherSupported`
  // (gguf_keep_quant.cpp) is `OpRegistered(kEmbeddingQuant, dev)` and names no
  // device, exactly as `GgufQuantComputeAvailable()` beside it asks about
  // `kMatmulBTQuant`; the alternative was a hand-kept per-device set in the
  // shared loader, which `scripts/check-device-leakage.py` refuses and which
  // could drift from the kernels it claims to describe.
  //
  // `vt::Embedding` routes here when `IsBlockQuant(table.dtype)`, so a device
  // that registers only `kEmbedding` refuses a block table BY NAME through the
  // ordinary GetOp message instead of silently gathering garbage. Registered on
  // kCPU (src/vt/cpu/cpu_ops.cpp, the same kernel — it already branches on the
  // table dtype), on kCUDA (src/vt/cuda/cuda_ops.cu, decoders in
  // src/vt/cuda/cuda_quant_dequant.cuh) and on kROCM
  // (src/vt/rocm/rocm_ops.hip:207). METAL, VULKAN and TENSTORRENT are NOT
  // registered: each of their gather kernels asserts a float table by name, and
  // their arms are owed.
  // Appended before kCount so no existing op's id shifts.
  kEmbeddingQuant,
  // MODEL-MM-GLM53-FLASH W9c-0 ([#2415]) — GLM-5.3-Flash's K-POOL DSA indexer,
  // the one primitive family on this model's critical path that nothing in this
  // tree implemented on a device.
  //
  // THIS IS NOT THE LIGHTNING INDEXER AND THE PAIR ABOVE CANNOT SERVE IT.
  // `kDsaIndexerLogits` / `kDsaTopkSelect` score RAW TOKENS and pick the top
  // `index_topk` of them. `Glm5NextTextIndexer` (transformers v5.16.1,
  // `models/glm5_next/modular_glm5_next.py`) compresses `index_kpool = 4`
  // consecutive VALID tokens into one candidate under a LEARNED per-channel
  // 4-way softmax, picks the top `index_topk / index_kpool = 512` POOLS, expands
  // each back into its member token indices, and then appends the ragged visible
  // tail raw and UNSCORED. Feeding raw candidates into a pooled top-k, or pooled
  // candidates into a consumer expecting raw ones, yields plausible indices
  // either way — which is why this is a separate op family and not a mode.
  //
  // Composition out of what already exists was checked and is not available:
  // this `OpId` inventory has no general softmax, no general reduction and no
  // pooled gather, and its only top-k entries are `kMoeRouterTopK` and the
  // sampler's `kTopKValuesIndices`, neither of which selects over pools or
  // expands a pool back to its members.
  //
  // kGlm5NextKpoolCompress — `get_pooled_states` (`:897-970`). The pool grid
  // starts at the first VALID token and not at slot 0, so a left-padded row
  // groups differently; a pool is a candidate only when ALL of its members are
  // valid; and `keep = pool_valid.any(0)` (`:968`) COMPACTS the pool axis. That
  // compaction is not cosmetic: `select_k = min(index_topk // index_kpool, P)`
  // reads the compacted width, and a `P` too large by one moves the ragged
  // tail's write offset and changes what the final truncation keeps. So the op
  // compacts ON THE DEVICE and publishes `P` as a `[1]` i32 DEVICE scalar. It
  // never synchronises and never returns anything to the host.
  //
  // kGlm5NextKpoolSelect — the selection half of `forward` (`:821-875`), with
  // `get_visible_tokens` (`:877-895`) folded in as a PREDICATE rather than
  // materialised (a `[B, S, kv_len]` visibility tensor is 2.7 GiB at 32k
  // context for something a thread evaluates in two instructions; upstream
  // materialises it because torch has no other way to gather under it) and
  // `append_visible_tail` (`:972-1022`) folded in as the tail write, because the
  // tail's write offset is `select_k * index_kpool` and only this op knows `P`.
  // Output width is `index_topk + index_kpool - 1` = **2051** on the published
  // checkpoint, not 2048, and it carries `-1` sentinels and duplicates, which
  // upstream absorbs downstream with `scatter_add_` + `ne(0)`.
  //
  // f32 THROUGHOUT, which is upstream's own arithmetic and not a choice:
  // `:823` scores in fp32 and `:960-964` takes the pool softmax in fp32. The
  // HOST reference `glm5_next_dsa.cpp` accumulates in `double`, a
  // host-reference widening the device deliberately does not copy — a fp64 pool
  // softmax would put the model path on the 1/64-rate pipe to be more precise
  // than the thing it mirrors.
  //
  // Registered on kCUDA (src/vt/cuda/cuda_glm5_next.cu:561,564) and on kROCM
  // (src/vt/rocm/rocm_ops.hip:345,348). There is deliberately NO CPU provider:
  // the CPU answer is `glm5_next_dsa.cpp`, which
  // is this family's ORACLE, and registering it a second time under these ids
  // would make the seam its own oracle. A CPU queue is therefore refused BY NAME
  // by the dispatcher.
  //
  // Additive, and NOT REACHED from any production entry point yet: only
  // `tests/vllm/models/test_glm5_next_kpool_device.cpp` and the availability
  // probe `vllm::glm5_next::KpoolDeviceOpsAvailable()` call them. W9c-3 owns the
  // wiring and #2410 tracks it; the spec's O36 says so in the specific.
  // Appended before kCount so no existing op's id shifts.
  kGlm5NextKpoolCompress,
  kGlm5NextKpoolSelect,
  // Packed k-quant block decode: packed {rows, blocks} (dtype carries the
  // encoding; bytes per row = blocks * BlockBytes) -> f32 {rows,
  // blocks * BlockElems}. Decode-only (BACKEND-TENSTORRENT-KEEPQUANT W1);
  // the dot provider and the keep-quant predicate arm ride W2.
  kKeepQuantDecode,
  // Explicit native BF16 MoE numerics. The legacy grouped signatures and
  // FP32 gate/up intermediates remain unchanged for existing callers.
  kMoeGroupedGemmBf16GateUpSiluNative,
  kMoeGroupedGemmBf16Weighted,
  kMoeCombinePreweighted,
  kResidualRmsNorm,
  // ClampedSwiGLU: silu(clamp(gate, max=limit)) * clamp(up, -limit, limit) at
  // alpha=1, beta=0 — DeepSeek-V4's activation (activation.py:197-201) and
  // GLM-5.3-Flash's ExpertGate (_apply_gate, modeling_glm5_next.py:137-142).
  // Takes concatenated gate_up [T, 2D] -> out [T, D], same layout as SiluAndMul
  // but with per-element clamping. Appended before kCount so no existing op's
  // id shifts.
  kClampedSwiGLU,
  kScaledRmsNorm,
  kSandwichRmsNorm,
  kCompiledGeluErfMul,
  kCount
};

enum class WorkspaceSlot : uint8_t {
  kWorkspace = 0,
  kOutput,
  kLse,
  kSemaphore,
  kDeviceScalar0,
  kDeviceScalar1,
};

enum class WorkspaceInit : uint8_t {
  kUninitialized = 0,
  kZeroOnFirstUse,
  kZeroEachUse,
};

struct WorkspaceKey {
  Device device;
  uint64_t queue_id = 0;
  uintptr_t native_handle = 0;
  OpId op = OpId::kMatmul;
  WorkspaceSlot slot = WorkspaceSlot::kWorkspace;

  friend bool operator==(const WorkspaceKey& a, const WorkspaceKey& b) {
    return a.device == b.device && a.queue_id == b.queue_id &&
           a.native_handle == b.native_handle && a.op == b.op && a.slot == b.slot;
  }
};

WorkspaceKey MakeWorkspaceKey(const Queue& q, OpId op, WorkspaceSlot slot);

struct DropinProbeArgs {
  ScalarTypeId scalar_type = vt::scalar_type::kF32;
  KernelLayout layout = KernelLayout::kStrided;
  size_t workspace_bytes = sizeof(uint32_t);
  size_t workspace_alignment = alignof(uint32_t);
  WorkspaceInit workspace_init = WorkspaceInit::kZeroEachUse;
  WorkspaceSlot workspace_slot = WorkspaceSlot::kWorkspace;
  WorkspaceSlot scalar_slot = WorkspaceSlot::kDeviceScalar0;
  float scalar = 0.0f;
};

struct RmsNormArgs {
  float eps = 1e-6f;
  bool gemma = false;  // weight applied as (1 + w), GemmaRMSNorm style
};

struct ResidualRmsNormArgs {
  float eps = 1e-6f;
  ResidualNormDesc descriptor{};
};

// Ungated GROUP RMS norm args (vt::RmsNormGroup). A SIBLING of RmsNormArgs, not
// a mode of it: see the kRmsNormGroup comment for why the group extent is not a
// field on that struct. `eps` and `gemma` keep RmsNormArgs's names and meanings
// exactly, so a caller migrating a whole-row norm to a grouped one changes the
// extent and nothing else.
//
// Algorithm oracle: transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py::Qwen4ExpTextRMSNorm` (:158-181),
// whose `group_size` argument this mirrors by NAME. `RmsNormGatedGroupArgs`
// spells the same axis as `n_groups` because ITS upstream
// (`Mixer2RMSNormGated`, mamba_mixer2.py:80) does; the two upstreams disagree
// on which half of the quotient is the parameter, and each op takes the one its
// own upstream takes rather than a normalized third form.
struct RmsNormGroupArgs {
  float eps = 1e-6f;
  // `weight` applied as `(1 + w)` rather than `w`. TRUE is the polarity every
  // `qwen4_exp` gamma needs: `Qwen4ExpTextRMSNorm.forward` is
  // `output * (1.0 + self.weight.float())` over a ZERO-initialised parameter
  // (:162, :177), the loader stores every gamma RAW, and every consumer adds
  // the 1 itself (#2218). It is a flag and not a constant because this op is a
  // general `vt::` primitive and `RmsNormArgs` already spells the same choice
  // the same way.
  bool gemma = false;
  // Elements per group. The reduction runs over `group_size` CONSECUTIVE
  // elements of the last dimension, so the norm of one group cannot see another
  // (`x.reshape(*x.shape[:-1], -1, group_size)` then `mean(-1)`, :168-171).
  //
  // ZERO IS REFUSED rather than meaning "the whole row", and that default is
  // deliberate. Whole-row is what `vt::RmsNorm` already does; letting a
  // forgotten field silently produce it would make the single most likely
  // caller mistake indistinguishable from success, and a full-row reduction
  // over a grouped stream is a plausible tensor rather than a crash. Upstream
  // refuses the analogous case at :164-165 (`hidden_size must be divisible by
  // group_size`) and so does the dispatcher.
  int64_t group_size = 0;
};

// ─── EXL3 device-kernel argument records — MODEL-DSV4-EXL3 W2 ────────────────
// Ported from exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT). See
// the vt::Exl3HadR128 / vt::Exl3Gemm contracts at the end of this header, and
// `.agents/specs/model-dsv4-exl3.md` `## W2 design` for the parity contract,
// the output dtype decision and the shape policy.

// `had_r_128(input, output, pre_scale, post_scale, scale)`
// (quant/hadamard.cuh:5-12). At most ONE of the two scale vectors may be set —
// upstream instantiates the kernel as <pre_scale, post_scale> and never with
// both true (hadamard.cu:112-172).
struct Exl3HadArgs {
  const Tensor* pre_scale = nullptr;   // fp16 [cols], multiplied BEFORE the transform
  const Tensor* post_scale = nullptr;  // fp16 [cols], multiplied AFTER
  float scale = 1.0f;                  // folded into r_scale = scale / sqrt(128)
};

struct Exl3GemmArgs {
  int bits = 0;             // K, bits per weight (1..8); 3 for the 3.0bpw artifact
  int codebook = 1;         // cb; 1 == mcg, the only codebook this row decodes
  // HALF-INTEGER rate (K+0.5), BACKEND-ROCM frac rates: `bits` carries the
  // integer KA (1..7) and `codebook` MUST be 2 (mul1 — the only codebook the
  // oracle defines the fractional rates for, `frac.cu`). Per TENSOR, set
  // alongside `bits`/`codebook` per call, never per model: integer and frac
  // tensors coexist in one mixed-rate checkpoint.
  bool half = false;
  int force_shape_idx = 0;  // 0 == let Exl3SelectGemmShape decide (the default)
  // MODEL-DSV4-EXL3 W2c. The m<=8 GEMV arm, mirroring upstream's own `force`
  // parameter (`exl3_gemv.cu:105`, and the direct entry point at `:171-241`
  // that sets it):
  //   -1  let the envelope + VT_EXL3_GEMV decide  (the DEFAULT, and production)
  //    0  never take it
  //    1  take it wherever the HARD constraints allow, bypassing the heuristic
  // Forcing bypasses the heuristic and never the hard constraints, exactly as
  // upstream's does. It exists so a device gate measures the arm it names
  // instead of whatever the occupancy query happened to choose.
  int force_gemv = -1;
};

// ─── The fused MoE MLP — MODEL-DSV4-EXL3 W2d ─────────────────────────────────
//
// Ported from `exl3_moe.cu:99-301` + `exl3_moe_kernel.cuh:17-283`. The contract
// is at vt::Exl3MoeMlp below and the design is in
// `.agents/specs/model-dsv4-exl3.md` `## W2cd design`.

// `act_function` (`exl3_moe_common.cuh:6-8`), plus ONE value that is ours.
enum class Exl3MoeAct : int {
  kSilu = 0,          // MOE_ACT_SILU
  kGelu = 1,          // MOE_ACT_GELU
  kRelu2NoGate = 2,   // MOE_ACT_RELU2_NOGATE: the gate GEMM and staging are skipped
  // vLLM's `SiluAndMulWithClamp` (`model_executor/layers/activation.py:197-201`),
  // which is what DeepSeek-V4 uses and is NOT any of upstream's three. The clamp
  // lands BEFORE the silu where upstream's lands after, and the limit is applied
  // UNCONDITIONALLY where upstream guards it with `act_limit != 0.0f`
  // (`hadamard_inner.cuh:110`) — a zero limit therefore means "clamp to zero"
  // here and "no clamp" there. AGENTS.md makes vLLM the authority wherever it
  // implements the behavior; exllamav3 is this format's oracle, not this model's.
  kSiluAndMulClamp = 3,
};

// `exl3_moe_common.cuh:10-12` and `exl3_devctx.cuh:13`, plus
// `block_sparse_mlp.py:19`'s TEMP_ROWS_FUSED. Named constants rather than
// literals because the launch geometry, the buffer shape and the "too many
// tokens for the fused kernel" cut all read them.
inline constexpr int kExl3MoeSmsPerExpert = 8;
inline constexpr int kExl3MoeMaxGroups = 64;
inline constexpr int kExl3MoeTempRowsFused = 128;

// `exl3_moe_max_concurrency` (`exl3_moe.cu:14-18`): how many expert groups a
// device can run at once, which is also how many temp-buffer sets the caller
// must allocate. Clamped to `kExl3MoeMaxGroups` (the scheduler's ticket array is
// that long) and never below one.
int Exl3MoeMaxConcurrency(int device_sms);

// The host-side batching, ported from `block_sparse_mlp.py:1079-1105`. PURE, and
// in `src/vt/exl3_policy.cpp` beside the shape table for the same reason: a
// bincount and a grouping that only a device run could check is arithmetic
// nobody checks.
//
//   topk_ids       [num_tokens * topk] the routed expert per assignment
//   topk_weights   [num_tokens * topk] the routing weight per assignment
//   expert_count   OUT [num_experts + 1]; the last slot is upstream's sentinel
//                  for an assignment outside this shard and a single-shard load
//                  never fills it
//   token_sorted   OUT [num_tokens * topk] the token index, grouped by expert
//   weight_sorted  OUT [num_tokens * topk] fp16, the same grouping
//
// Returns `num_active`: the number of experts with
// `0 < count <= max_tokens_per_expert`, which is what sizes the launch
// (`exl3_moe.cu:93-96`). An expert ABOVE that cut is skipped by the fused kernel
// (`exl3_moe_kernel.cuh:66`) and the caller runs it per-expert, exactly as
// upstream's caller does (`block_sparse_mlp.py:1141,1151-1156`).
//
// DEVIATION, recorded: a STABLE grouping where upstream calls the unstable
// `Tensor.argsort`. The two agree on `expert_count` and on the multiset of each
// segment; nothing downstream reads the within-segment order except which temp
// row a token occupies, and the epilogue scatters back by token index.
int Exl3MoeSortTokensByExpert(const int64_t* topk_ids, const float* topk_weights,
                              int64_t num_tokens, int64_t topk, int64_t num_experts,
                              int64_t max_tokens_per_expert, int64_t* expert_count,
                              int64_t* token_sorted, uint16_t* weight_sorted);

// The nine per-expert POINTER TABLES (`exl3_moe.cu:118-126`). Each is an i64
// tensor of `num_experts` entries whose values are the ADDRESSES of that
// expert's trellis / suh / svh buffers. Upstream passes exactly this, as tensors
// of `void*`; the i64 spelling keeps them inside the vt Tensor world so the
// device tagging and the contiguity checks apply to them like anything else.
struct Exl3MoeExpertTables {
  const Tensor* gate_trellis = nullptr;
  const Tensor* gate_suh = nullptr;
  const Tensor* gate_svh = nullptr;
  const Tensor* up_trellis = nullptr;
  const Tensor* up_suh = nullptr;
  const Tensor* up_svh = nullptr;
  const Tensor* down_trellis = nullptr;
  const Tensor* down_suh = nullptr;
  const Tensor* down_svh = nullptr;
};

// The batching triple `Exl3MoeSortTokensByExpert` produces (`exl3_moe.cu:103-105`).
struct Exl3MoeRouting {
  const Tensor* expert_count = nullptr;   // i64 [num_experts + 1]
  const Tensor* token_sorted = nullptr;   // i64 [num_tokens * topk]
  const Tensor* weight_sorted = nullptr;  // f16 [num_tokens * topk]
};

// The four staging buffers (`exl3_moe.cu:107-110`), each
// [concurrency, max_tokens_per_expert, dim] fp16. `concurrency` is how many
// expert groups run at once and `max_tokens_per_expert` is the cut above which
// the fused arm declines an expert; both are READ OFF these shapes rather than
// passed again, which is upstream's own arrangement (`:168-169`).
struct Exl3MoeTemps {
  Tensor* state_g = nullptr;
  Tensor* state_u = nullptr;
  Tensor* intermediate_g = nullptr;
  Tensor* intermediate_u = nullptr;
};

struct Exl3MoeArgs {
  int bits_gate = 0;  // K_gate / K_up / K_down; upstream keeps them separate and
  int bits_up = 0;    // switches K at run time when they differ
  int bits_down = 0;  // (exl3_moe_kernel.cuh:139-149)
  // ONE codebook for all three, because `exl3_moe.cu:182-184` refuses a call
  // whose gate/up/down disagree: three fields that must be equal are three
  // chances to disagree. 1 == mcg, the only one this row decodes.
  int codebook = 1;
  Exl3MoeAct act = Exl3MoeAct::kSilu;
  float act_limit = 0.0f;
  // The number of experts the fused arm will process, which sizes the launch
  // (`exl3_moe.cu:93-96,217-221`). -1 means unknown and takes the default
  // group width; 0 means there is nothing to do and the call returns.
  int num_active = -1;
};

// torch `nn.LayerNorm` arguments (opt.py:146-148,164-166,248-251 construct it
// with the default eps=1e-5 and `elementwise_affine=config.
// layer_norm_elementwise_affine`). Unlike RmsNormArgs there is no gemma variant:
// LayerNorm subtracts the mean and applies weight AND bias.
struct LayerNormArgs {
  float eps = 1e-5f;
};

struct RopeArgs {
  float base = 10000.0f;
  int rotary_dim = 0;  // <= head_dim; even
  bool is_neox_style = true;
  // Empty (all zero) for 1-D RoPE. For positions[3,T], the entries are the
  // temporal/height/width counts in the half-rotary frequency dimension.
  std::array<int32_t, 3> mrope_section = {0, 0, 0};
  bool mrope_interleaved = false;

  // RopeCosSinCache only: positive selects the primary's FP32 cache math,
  // including position / factor before the outer product. One is unscaled.
  // Zero preserves the legacy FP64 cache math and Llama-3 rescaling.
  float linear_scaling_factor = 0.0f;

  // Llama-3 rope frequency rescaling (rope_type=="llama3", e.g. Llama-3.2). When
  // llama3_scaling_factor <= 0 (the default) NO rescale is applied and the RoPE
  // is byte-identical to plain RoPE — so every existing caller (Qwen, the gate
  // models) that leaves these zero is UNCHANGED. When set, the base inv_freq
  // (base^(-2i/rotary_dim)) is rescaled per frequency by a piecewise low/high
  // wavelength interpolation, mirroring vLLM Llama3RotaryEmbedding._compute_inv_freq
  // (vllm/model_executor/layers/rotary_embedding/llama3_rope.py:33-54). Consumed
  // by RopeNeox + RopeCosSinCache (the cache feeds RopeFromCache, so no extra
  // field is needed there).
  float llama3_scaling_factor = 0.0f;    // rope_scaling "factor" (0 => disabled)
  float llama3_low_freq_factor = 0.0f;   // rope_scaling "low_freq_factor"
  float llama3_high_freq_factor = 0.0f;  // rope_scaling "high_freq_factor"
  float llama3_orig_max_position = 0.0f;  // "original_max_position_embeddings"
};

// GDN op args (.agents/specs/gdn-semantics.md is the formula reference; sections
// cited on each op below).
struct CausalConv1dArgs {
  // Upstream `activation` is "silu"/"swish" (→ silu) or None (→ identity);
  // Qwen GDN always uses silu (gdn-semantics.md §2).
  bool silu_activation = true;
  // Optional exact upstream prefill work descriptor, both i32 [num_programs]
  // on the queue device. Entry p owns sequence batch_ptr[p] and its
  // token_chunk_offset_ptr[p]-th 8-token chunk. CUDA consumes these when the
  // default CUDA path is selected; VT_CONV_EXACT_CHUNKS=0 restores the legacy mapping and
  // CPU keeps its scalar reference.
  const Tensor* batch_ptr = nullptr;
  const Tensor* token_chunk_offset_ptr = nullptr;
};

// Qwen4-Exp PLE short-conv args. SIBLING of CausalConv1dArgs, deliberately not
// a mode of it — see the kQwen4ExpPleConv comment for the five-backend reason.
// Algorithm oracle: transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py::Qwen4ExpTextPLELayer._short_conv`
// (:1150-1167), with `conv_dilation = config.ngram_size` (:1134) and
// `short_conv_state_len = (conv_kernel_size - 1) * conv_dilation` (:1135).
struct Qwen4ExpPleConvArgs {
  // `config.ngram_size`; 3 in the released `Qwen/Qwen3.8-Flash-Next` config.
  // Cross-checked against the state width rather than derived from it: a
  // `conv_state` whose last dim is not `(K - 1) * dilation` is a caller that
  // believes a different geometry, and deriving would make that agreement
  // unfalsifiable.
  int64_t dilation = 1;
};

// Qwen4-Exp PLE gate args (vt::Qwen4ExpPleGate). Algorithm oracle: transformers
// v5.16.0 `models/qwen4_exp/modeling_qwen4_exp.py::Qwen4ExpTextPLELayer.forward`
// (:1180-1182, flattened at :1184). Both numbers are held here rather than
// derived from a shape, for the reason Qwen4ExpPleConvArgs::dilation gives: a
// caller that believes a different geometry must be visible, not interpolated.
struct Qwen4ExpPleGateArgs {
  // `math.sqrt(self.hidden_size)`, the DIVISOR upstream ends :1180 with. Held as
  // the divisor and not as its reciprocal so the op performs the same operation
  // upstream does; multiplying by a pre-computed 1/sqrt(H) differs in the last
  // place. 1.0 is the identity, which is what a caller that has already scaled
  // its scores passes.
  float gate_divisor = 1.0f;
  // The `clamp_min(1e-6)` LITERAL at :1181 — not a config key, and upstream has
  // no knob for it. It is a field so the number appears ONCE for every arm
  // rather than being re-typed in each kernel, and so a caller that changes it
  // is visible in the call rather than invisible in a backend. Applied to
  // |gate| BEFORE the square root, so the floor it puts on the output magnitude
  // is its own square root; see the kQwen4ExpPleGate comment.
  float clamp_min = 1e-6f;
};

struct L2NormArgs {
  float eps = 1e-6f;  // upstream default (gdn-semantics.md §4)
};

// Geometry of one Gated DeltaNet V-head re-indexing. See `vt::VHeadPermute`.
struct VHeadPermuteArgs {
  int64_t prefix_elems = 0;   // leading columns copied through unpermuted
  int64_t num_key_heads = 0;  // K
  int64_t heads_per_key = 0;  // R = num_value_heads / K
  int64_t head_width = 1;     // elements per V head (value_head_dim, or 1)
  bool inverse = false;       // false: tiled -> grouped; true: grouped -> tiled
};

struct RmsNormGatedArgs {
  float eps = 1e-6f;
  // Gate activation: silu by default; "sigmoid" allowed by upstream
  // output_gate_type (gdn-semantics.md §5). norm_before_gate=True and
  // group_size=None (the only configuration Qwen GDN uses) are baked in.
  bool sigmoid_gate = false;
};

// Silu-gated GROUP RMS norm args (Mixer2RMSNormGated, mamba_mixer2.py:69-149).
// SIBLING of RmsNormGatedArgs, not a mode of it: the GDN/KDA gated norm reduces
// over the WHOLE row with an optional sigmoid gate, this one always uses SILU and
// reduces over `group_size = hidden / n_groups` slices (:136-141).
struct RmsNormGatedGroupArgs {
  float eps = 1e-6f;  // Mixer2RMSNormGated default (mamba_mixer2.py:76)
  // full_n_groups. group_size = hidden / n_groups must divide the last dim
  // (mamba_mixer2.py:80). n_groups == 1 degenerates to a whole-row RMS norm,
  // which is exactly upstream's `self.n_groups == 1` branch (:120-131).
  int64_t n_groups = 1;
  // W1 lands tp_world_size == 1 only. Any other value is REFUSED with a message
  // naming `extra_groups_for_head_shards` (mamba_utils.py:187) rather than
  // silently computing a wrong split (mamba2-ssd.md §7).
  int64_t tp_world_size = 1;
};

// Qwen4-Exp gated-residual (hyper-connection) geometry. Shared by the read op
// and the write-back op so a caller cannot describe the same stream two ways.
// Algorithm oracle: transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py::Qwen4ExpTextGatedResidual` (:941-969)
// and `::Qwen4ExpTextRMSNorm` (:158-181); op form for the grouped norm: vLLM
// `model_executor/layers/layernorm.py::RMSNormGated` (:172, group_size :187).
struct Qwen4ExpGatedResidualArgs {
  // `config.hc_count`, the number of residual branches. Upstream's
  // `__post_init__` rejects <= 1, and so does this op: the mean over hc and the
  // `/ hc_count` inside both activations are undefined at 0 and degenerate at 1.
  int64_t hc_count = 0;
  int64_t hidden_size = 0;  // `config.hidden_size`; ALSO the norm's group_size
  int64_t lowrank = 0;      // `config.hc_lowrank`, the mix down/up rank
  float eps = 1e-6f;        // `config.rms_norm_eps`, INSIDE the rsqrt
};

// Qwen Sparse Attention COMPRESSOR geometry (vt::Qwen4ExpQsaCompress).
// Algorithm oracle: transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py::Qwen4ExpTextQSAIndexer` (:677-688),
// with the scaffolding — boundary predicate, block-start RoPE, paged store —
// from the TRITON head_dim=128 variant of DeepSeek-V4's
// `_fused_kv_compress_norm_rope_insert_indexer_attn`
// (models/deepseek_v4/common/ops/fused_compress_quant_cache.py :677-830 @ vLLM
// origin/main 6a5e8f5979). Only the scaffolding: DSv4's pool is a LEARNED
// softmax over an OVERLAPPING window driven by a score channel this checkpoint
// does not have, and QSA's is an unweighted mean over a NON-overlapping one.
struct Qwen4ExpQsaCompressArgs {
  // `config.indexer_compress_ratio`; 4 in the released `Qwen/Qwen3.8-Flash-Next`
  // config, and the number of consecutive RAW keys that pool into one state.
  int64_t compress_ratio = 0;
  // `int(head_dim * partial_rotary_factor)` = 64 at the released config, which
  // upstream then requires to FIT `indexer_head_dim`
  // (configuration_qwen4_exp.py:225-231). The LEADING `rotary_dim` dims rotate
  // NeoX-style and the NoPE dims trail untouched — the halves are swapped end
  // for end against DeepSeek-V4's indexer, which ropes a TRAILING span.
  int64_t rotary_dim = 0;
  float eps = 1e-6f;  // `config.rms_norm_eps`, INSIDE the rsqrt
  // Mirrors the model dtype's intermediate rounding: upstream's pooled key goes
  // through `.to(raw_keys.dtype)` before `k_layernorm`, `Qwen4ExpTextRMSNorm`
  // closes on `.type_as(x)`, and a bf16 elementwise RoPE rounds per operation.
  // TRUE reproduces that op by op, which is what a bf16 model path does and what
  // the lane-pinned goldens were dumped under; FALSE is the f32 arm. The host
  // reference `vllm::qwen4_exp::QsaCompressNormRope` carries the same flag with
  // the same meaning, so the two arms answer to one oracle rather than to each
  // other. This is deliberately NOT the house "round once on the store"
  // contract: here the rounding is load-bearing (mutation M9 measures it), and
  // an op that could not express it could not be gated against the oracle at
  // all.
  bool round_intermediates_to_bf16 = false;
};

// Qwen Sparse Attention GATHER-CONSUMER arguments
// (vt::Qwen4ExpQsaGatherAttention).
struct Qwen4ExpQsaAttnArgs {
  // Softmax scale, `head_dim ** -0.5` on the MODEL's attention head (256 in the
  // released config), NOT the indexer's. Must be set explicitly (> 0), the
  // `AttentionArgs::scale` convention.
  float scale = 0.0f;
  // `config.indexer_compress_ratio`. Selected block `b` IS tokens
  // [CR*b, CR*b + CR); this op expands that as ADDRESSES and never materialises
  // the token index buffer, which is the difference between a gather and a mask.
  int64_t compress_ratio = 0;
  // OPTIONAL host-side instrument: the number of key ROWS the kernel actually
  // read, ACCUMULATED over every query token of the call and INCREMENTED AT THE
  // READ. It is never assigned from the selection, and that distinction is the
  // whole point of it. W4's first fresh review found a counter set to
  // `sel.size()` and compared against `SelectedCount(sel)` — the same quantity
  // by the same rule — under which a body that dot-products every one of the
  // `kv_len` cached rows still reported the sparse figure and passed 12 cases /
  // 7251 assertions, including the case named "the GATHER touches only the
  // selected rows". Counted at the read, an honest gather comes out at
  // `sum_t selected(t) * num_q_heads * 2` (two softmax passes) and a dense walk
  // at `sum_t kv_len(t) * num_q_heads * 2`, so the ratio between a gather and a
  // mask is MEASURED rather than asserted.
  //
  // WHAT IT CANNOT SEE, stated because it was MEASURED rather than feared. It
  // counts reads at THIS body's read site, so it says what an honest body cost;
  // it cannot convict a dishonest one. W5b-4 mutation M11 is a dense masked walk
  // that also reports `sel.size() * 2` per head, and it passed 10 cases and 4167
  // assertions — a mask-shaped port changes the loop and the counter together,
  // which is what a mask-shaped port IS, and no value comparison separates them
  // either because `exp(-inf - m)` is exactly +0. The gate that DOES separate
  // them is an observable of the walk: `test_qwen4_exp_qsa_device.cpp` runs this
  // op over a cache whose UNSELECTED rows are NaN, which a gather never
  // addresses and a mask multiplies by a zero weight into `0.0f * NaN`. It also
  // cannot see a body that iterates 0..kv_len and `continue`s past unselected
  // rows without touching them — correctly, because the loop counter is not the
  // cost llama.cpp #27739 measures, the key-row traffic is.
  // A host pointer, on the `GdnArgs::query_start_loc_host` precedent; a CUDA arm
  // owes a device-side counter and its copy-back.
  int64_t* keys_visited = nullptr;

  // ─── THE SOFTMAX-FOLD PROBE, W9's REPAIR ────────────────────────────────────
  //
  // OPTIONAL host-side instrument, on the `keys_visited` precedent and for the
  // same reason: the property it exposes cannot be asserted from the outputs.
  // The gather's denominator `sum_s w(s)` is accumulated SERIALLY ASCENDING over
  // the gathered rows, and THAT ORDER is what makes a sub-budget gather reduce
  // over exactly the dense sequence and so be BIT-identical to dense attention.
  // It is the property W9's spec calls the whole argument for choosing its
  // lever, and NOTHING MEASURED IT: a fresh review reassociated it two ways on
  // `thor:gpu0` -- a descending fold, and the warp-shuffle tree the CUDA
  // kernel's header explicitly declines -- and all three committed suites passed
  // unmodified, while the outputs demonstrably moved.
  //
  // A TIGHTER TOLERANCE CANNOT REPLACE THIS AND THAT IS MEASURED, not feared. At
  // the multi-tile shapes the reassociated denominator lands at 0.01-0.02% of
  // the derived arm-vs-arm bound, and the WORST ratio on a CORRECT kernel
  // (0.2041) is HIGHER than on a reassociated one (0.1136), so no threshold
  // separates them. `test_qwen4_exp_cuda_reductions.cpp` already records a
  // fitted `kUlpTol` that failed correct kernels by 112-290%; this is the
  // instrument chosen instead of a third attempt at a constant.
  //
  // WHAT IT HANDS BACK. `softmax_probe_weights` receives the per-row softmax
  // weights the arm ACTUALLY COMPUTED, at `pair * softmax_probe_stride + s` for
  // pair `t * num_q_heads + h` and gathered position `s` in visit order;
  // `softmax_probe_denom` receives that pair's finished denominator. A caller
  // sets all three or none. Entries at `s >= softmax_probe_stride` are dropped,
  // so a stride below `|sel|` measures a prefix and the caller owns the sizing.
  //
  // WHY BOTH, AND WHY THAT MAKES THE GATE ROBUST. The test refolds the weights
  // ASCENDING **on the host, from these exact device-produced floats**, and
  // requires BIT equality with the denominator. Both sides then consume the same
  // `expf` outputs, so a toolkit that moves `exp` cannot make it red -- the
  // fragility a stored device golden would have -- while a descending fold or a
  // tree reduction is red immediately, because those are the only things the
  // comparison can see.
  //
  // A ROW WHOSE SELECTION IS MALFORMED (the CUDA arm's NaN poison) writes no
  // denominator: its weights are not the weights of any well-formed softmax.
  // Honoured by the CPU and CUDA arms. The ROCm arm REFUSES the probe BY NAME
  // rather than returning an unwritten buffer.
  float* softmax_probe_weights = nullptr;
  float* softmax_probe_denom = nullptr;
  int64_t softmax_probe_stride = 0;

  // ─── THE PAGED ADDRESS MODE (row MODEL-MM-QWEN4-EXP W5d-3, #2249 item 2) ───
  //
  // WHY IT IS HERE AND NOT A SECOND OP. The engine allocates this model's QSA
  // K/V as a PAGED `FullAttentionSpec` group (`MakeQwen4ExpKVCache`), so the
  // contiguous `[max_kv, Hkv, Dh]` arm above could not serve from the cache the
  // runner actually hands a forward. What differs between the two is the
  // resolution of ONE address — the key/value row for logical position `p` —
  // and nothing else: the expansion, the visit ORDER, the two softmax passes and
  // the f32 accumulation are the same body, so a second op would be the parallel
  // path AGENTS.md "Shared seams" forbids.
  //
  // `nullptr` keeps the contiguous arm byte-for-byte. When set, `key`/`value`
  // are the rank-4 `[num_pages, kv_block_size, num_kv_heads, head_dim]` unbind
  // views of the runner's flash cache (`dense_attn::KvSlice`), STRIDED rather
  // than contiguous because K and V interleave at dim 1, and logical position
  // `p` resolves as vLLM's paged read does
  // (`vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward`,
  // mirrored in this tree's `vt::PagedAttention` contract):
  //
  //     page = kv_block_table[p / kv_block_size]
  //     row  = key[page, p % kv_block_size, kv_head, :]
  //
  // TWO THINGS ARE CALLED A "BLOCK" IN THIS OP AND THEY ARE NOT THE SAME
  // OBJECT. `block_ids` names QSA's COMPRESS blocks of `compress_ratio` tokens
  // (4 at the released config); `kv_block_table`/`kv_block_size` name the KV
  // CACHE PAGE (the engine's `block_size`, 16 or more). The `kv_` prefix is what
  // keeps them apart, and `MakeQwen4ExpKVCache` refuses a `block_size` the
  // compress ratio does not divide, so a compress block never straddles a page.
  //
  // ONE REQUEST. `kv_block_table` is `[1, max_pages]` i32: this op is called per
  // QSA layer for one sequence, exactly as the contiguous arm is, and a ragged
  // multi-request batch needs the per-request `query_start_loc` plumbing the
  // block does not carry yet. Recorded under the spec's `## Owed`.
  const Tensor* kv_block_table = nullptr;
  // Tokens per KV cache page. Must be > 0 exactly when `kv_block_table` is set.
  int64_t kv_block_size = 0;
};

// Mamba2 SSD args, shared by the chunked prefill scan and the decode state
// update (ssd_combined.py:27-235, mamba_ssm.py:497+).
struct Mamba2Args {
  // Physical chunk length. MUST be an integer power of two — upstream asserts it
  // (`is_int_pow_2`, ssd_combined.py:48). Ignored by Mamba2StateUpdate.
  int64_t chunk_size = 0;
  // Whether dt goes through softplus before the decay (`dt_softplus`). Upstream
  // guards it as `dt <= 20 ? softplus(dt) : dt` (ssd_chunk_state.py:94,
  // mamba_kernels.hpp:177). mamba_mixer2.py:1097 passes True.
  bool dt_softplus = false;
  // dt_limit, applied AFTER softplus (`tl.clamp(dt, dt_min, dt_max)`,
  // ssd_chunk_state.py:96). Upstream default is (0.0, +inf).
  // Ignored by Mamba2StateUpdate: selective_state_update has no dt_limit.
  float dt_min = 0.0f;
  float dt_max = std::numeric_limits<float>::infinity();
  // W1 lands tp_world_size == 1 only; see RmsNormGatedGroupArgs::tp_world_size.
  int64_t tp_world_size = 1;
};

// KERNEL-GDN-CHUNKED-MIRROR (.agents/specs/gdn-chunked-mirror.md D0/D3).
// THE algorithm predicate for GDN prefill, shared by every backend so that a
// `--device cpu` run and a `--device cuda` run cannot end up on different
// algorithms because only one of them read the flag.
//
// `GdnChunkedPrefillEnabled()` is the raw `VT_GDN_CHUNKED` read, lifted
// verbatim out of cuda_gdn.cu (the bespoke `e == nullptr || e[0] != '0'` parse
// is KEPT as-is rather than rewritten to EnvOnOr: the off-value spelling is
// recorded in four evidence files and six spec lines, so changing the accepted
// spellings would be a semantic change wearing a cleanup).
//
// `GdnUseChunkedPrefill(dtype)` adds D0's dtype term. vLLM's chunked kernels
// REFUSE f32 — the Triton wrapper asserts (`chunk.py:213-215`) and the CPU
// kernel type-checks (`csrc/cpu/sgl-kernels/fla.cpp:2205-2207`, bf16 only) — so
// at f32 the sequential recurrence IS the mirror, because it is the only gated
// delta rule upstream will execute at that dtype.
bool GdnChunkedPrefillEnabled();
bool GdnUseChunkedPrefill(DType q_dtype);

struct GdnArgs {
  // q scale, applied to q only after l2norm; upstream default Dk^-0.5
  // (gdn-semantics.md §1). Must be set explicitly (> 0).
  float scale = 0.0f;
  // OPTIONAL host-resident query_start_loc[N+1] (same values as the device
  // `query_start_loc` tensor). When set, the CUDA chunked-prefill path builds
  // its chunk layout from these host values + a device meta-kernel, avoiding
  // the per-layer D2H copy + cudaStreamSynchronize (the prefill host-tax — it
  // forced host↔GPU lockstep every GDN layer, ~67% GPU-idle). nullptr => the
  // path falls back to the D2H+sync (op tests / callers without host qsl).
  // Mirrors the decode StepDevInputs device-resident metadata pattern.
  const int32_t* query_start_loc_host = nullptr;
};

// Dense causal attention args (.agents/specs/qwen36-forward-notes.md §5 is the
// formula reference — Qwen3NextAttention's core scaled-dot-product).
struct AttentionArgs {
  // Softmax scale, applied to the qk dot product. Upstream sets it to
  // head_dim^-0.5 (Qwen3NextAttention.scaling). Must be set explicitly (> 0).
  float scale = 0.0f;
  // Causal masking: key position j attends only when j <= query position i.
  // Always true for the M0.9 decoder path (bidirectional is a M1.6+ concern).
  bool causal = true;
};

// Dense NON-CAUSAL CROSS attention args. vt::Attention requires key/value to
// carry the SAME token count as query (ops.cpp: "query/key/value token count must
// match"), which no cross-attention can satisfy: LTX-2.5's text cross-attention
// (transformer.py:113 `attn2`) and its audio<->video cross-attention
// (transformer.py:154 `audio_to_video_attn`, :166 `video_to_audio_attn`) each
// project queries from one stream and keys/values from another, so Tq != S by
// construction. This is that seam, kept SEPARATE from vt::Attention so every
// existing self-attention call stays byte-identical.
//
// query [Tq,Hq,D], key/value [S,Hkv,D], out [Tq,Hq,D]; Hq a multiple of Hkv
// (GQA broadcast, exactly as vt::Attention). Query i attends to EVERY key j in
// [0,S) — bidirectional, no causal mask; the only masking is the optional
// additive `bias` (see vt::AttentionCross). f32 softmax with max subtraction.
struct AttentionCrossArgs {
  // Softmax scale applied to the qk dot product. torch SDPA's default is
  // E**-0.5 with E = query.size(-1) = head_dim, which is what every LTX
  // attention gets (attention.py:98). Must be set explicitly (> 0).
  float scale = 0.0f;
};

// --- Conformer / FastConformer audio-encoder op args (spike
// .agents/specs/parakeet-conformer-encoder.md P1/P2/P3). ------------------------

// torch `nn.Conv2d` arguments. Mirrors the constructor keywords 1:1 so a reader
// of `ParakeetEncoderSubsamplingConv2D.__init__`
// (transformers 5.3.0 transformers/models/parakeet/modeling_parakeet.py:357-390)
// can map every field by name. `groups == in_channels == out_channels` is the
// DEPTHWISE form that module's inner layers use (:377-386); `groups == 1` with
// a 1x1 kernel is its pointwise layer (:388).
struct Conv2dArgs {
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  int64_t groups = 1;
};

// --- General 3-D convolution args (LTX25-DEVICE-RESIDENCY W5, #1007). --------

// torch `nn.Conv3d` arguments, in `(T, H, W)` axis order. Field names mirror the
// constructor keywords 1:1, exactly as Conv2dArgs mirrors `nn.Conv2d`'s, so a
// reader of `CausalConv3d.__init__` (Lightricks/LTX-2 @ fd4ded7f2,
// packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:267-302) can map
// every field by name.
//
// `padding` here is ZERO padding on both sides of an axis, and that is the whole
// of what this op does: a non-zero `padding_mode` — LTX's `reflect` and
// `replicate` — is realised by the CALLER materialising the padded volume and
// passing pad 0, which is what torch itself does (`nn.Conv3d` with a non-`zeros`
// `padding_mode` runs `F.pad` and then a zero-padded convolution). The LTX
// decoder's asymmetric CAUSAL temporal pad is materialised the same way, and for
// the same reason: upstream materialises it too, with a `torch.concatenate`
// (convolution.py:305-311).
struct Conv3dArgs {
  int64_t stride_t = 1;
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  int64_t pad_t = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t dilation_t = 1;
  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  int64_t groups = 1;
};

// torch `nn.Conv1d(C, C, K, stride, padding, groups=C)` arguments — the
// NON-CAUSAL depthwise conv1d of `ParakeetEncoderConvolutionModule`
// (modeling_parakeet.py:116, ctor :138-146; padding = (K-1)//2 at :136). Not to
// be confused with CausalConv1dArgs, which drives the Mamba/GDN CAUSAL conv and
// carries a persistent conv_state; this op is stateless and centre-padded.
struct DepthwiseConv1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;
  int64_t dilation = 1;
};

// --- BigVGAN / DAC vocoder 1-D convolution args (#672). --------------------

// torch `nn.Conv1d` arguments, the GENERAL grouped form. Field names mirror the
// constructor keywords 1:1. `groups == 1` is the dense conv the vocoder's
// conv_pre/conv_post/1x1 projections use; `groups == C_in == C_out` is the
// depthwise form the alias-free low-pass filter uses.
//
// NOT to be confused with DepthwiseConv1dArgs, which drives the conformer's
// f32-accumulate depthwise op — see OpId::kConv1d for why the two are siblings.
struct Conv1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;  // ZERO padding on BOTH sides; out-of-range taps skipped
  int64_t dilation = 1;
  int64_t groups = 1;
};

// torch `nn.ConvTranspose1d` arguments. Mirrors the constructor keywords 1:1.
// `padding` here CROPS (torch's "dilation * (kernel_size - 1) - padding"
// implicit zero-padding on both sides), and `output_padding` appends to the
// right only.
struct ConvTranspose1dArgs {
  int64_t stride = 1;
  int64_t padding = 0;
  int64_t output_padding = 0;
  int64_t dilation = 1;
  int64_t groups = 1;
};

// Transformer-XL relative-position self-attention args — the conformer
// attention of `ParakeetEncoderAttention` (modeling_parakeet.py:259, forward
// :302-347, `_rel_shift` :349-355) and of vLLM's own native
// `RelPosMultiHeadAttention` (conformer_encoder.py:170, forward :188-217,
// `_rel_shift` :179-186). See vt::AttentionRelPos for the full formula.
struct AttentionRelPosArgs {
  // Softmax scale. Upstream sets it to head_dim^-0.5 (ParakeetEncoderAttention
  // .scaling :271; RelPosMultiHeadAttention.scale :174).
  float scale = 0.0f;
  // WHERE the scale is applied, the one arithmetic difference between the two
  // upstreams. false (default) = HF's form: `matrix_bd *= scaling` (:322) and
  // `attn_weights = q@k^T * scaling + matrix_bd` (eager_attention_forward :247),
  // i.e. `s = ac*scale + bd*scale`. true = vLLM's native form:
  // `attn_scores = (matrix_ac + matrix_bd); attn_scores.mul_(self.scale)`
  // (conformer_encoder.py:212-213), i.e. `s = (ac + bd) * scale`. The two agree
  // in exact arithmetic and differ in f32 rounding, so the flag is exposed
  // rather than chosen, and each upstream gets its own byte-exact path.
  bool scale_after_sum = false;
};

// Arguments for vt::DFlashBlockAttention — the DFlash draft's IN-BLOCK attention
// (SPEC-DFLASH D2, DF-DRAFT-MODEL; the project's FIRST bidirectional/non-causal
// attention primitive). Ported from the semantics of DFlashQwen3Attention +
// _resolve_layer_attention (vllm/model_executor/models/qwen3_dflash.py:86-146,
// 149-263 @ 555967922) and grounded in flashinfer's non-causal attention path
// (the #48167 Blackwell non-causal kernel now in-pin). This op computes attention
// for the uniform (1+k)-token QUERY block of each request over the keys IN THAT
// SAME block only (the context K/V is pre-inserted separately by the D3
// context-KV precompute; D2 isolates the block forward). Kept a SEPARATE op from
// vt::Attention / vt::PagedAttention so every CAUSAL model stays byte-identical.
//
// Per-request block boundaries come from `cu_seqlens` (host, length num_reqs+1):
// request r owns query/key rows [cu_seqlens[r], cu_seqlens[r+1]). Each query i in
// the block attends to keys j in the same block subject to:
//   - full-attention layer  (causal=false): ALL j in the block (BIDIRECTIONAL);
//   - sliding-window layer   (causal=true):  j <= i AND j >= i-(window-1)
//     (window<=0 means plain causal). Positions are the intra-block offsets, which
//     matches DFlash's contiguous (1+k) block; the z-lab 27B window (2048) >> 17
//     so the SWA layer degenerates to plain causal over the block — the mask still
//     computes the true window bound for fidelity to other DFlash checkpoints.
// f32 softmax accumulation (max-subtracted), matching vLLM. GQA broadcast as in
// vt::Attention. query [Tq,Hq,D], key/value [T,Hkv,D], out [Tq,Hq,D], T = ΣblockLen
// and Tq = T unless `cu_seqlens_q` is set (see the field, SPEC-DFLASH2 W12 D1).
struct DFlashBlockAttentionArgs {
  float scale = 0.0f;              // head_dim^-0.5 (DFlashQwen3Attention.scaling)
  bool causal = false;            // per-layer: false=full(non-causal), true=SWA
  int64_t sliding_window = 0;     // SWA window (>0); 0 = full causal when causal
  const int32_t* cu_seqlens = nullptr;  // host, length num_reqs+1 (KEY/VALUE bounds)
  int num_reqs = 1;               // number of query blocks
  // SPEC-DFLASH2 W12 D1 (#2087) — the SEPARATE QUERY cu. When null (the default,
  // and every pre-W12 caller), query and key share `cu_seqlens` and this op is
  // exactly what it was: T query rows over T key rows, one square block per
  // request. When set (host, length num_reqs+1, spanning [0, query.shape[0]]),
  // request r owns query rows [cu_seqlens_q[r], cu_seqlens_q[r+1]) and key rows
  // [cu_seqlens[r], cu_seqlens[r+1]), and the query block is the BOTTOM-RIGHT
  // suffix of the key block — query offset `ii` sits at combined key offset
  // `klen_r - qlen_r + ii`, which is the [context ; block] layout the DFlash
  // draft materializes (qwen3_dflash.cpp `ForwardWithCtxKVDev`). The mask reads
  // the COMBINED offset, so causal/SWA sees the context as the past exactly as
  // vt::DFlashPagedBlockAttention does, and `cu_seqlens_q == cu_seqlens` is
  // arithmetically the null case (klen == qlen ⇒ the offset is 0).
  //
  // WHY IT EXISTS. Without it the caller must build an `Ncomb`-row query buffer
  // whose context rows are zero, compute attention for every one of them, and
  // discard the result: `sum_r (ctx_r + 1 + k)^2` pairs per layer per step
  // instead of `(1+k) x C`. That was the whole c>1 draft cost (#2087).
  const int32_t* cu_seqlens_q = nullptr;
};

// SPEC-DFLASH D12 Part B — CAPTURE-SAFE paged variant of DFlashBlockAttention.
// The (1+k) block queries attend over [persistent PAGED context ; their own (1+k)
// block] exactly as DFlashBlockAttention does over a materialized [context; block]
// combined buffer, but the growing context enters as DATA (a paged K/V cache +
// per-request block_table + seq_lens) instead of a variable-size combined buffer —
// so the launch grid is STATIC over the fixed (1+k)*num_reqs query rows and every
// metadata input is a persistent DEVICE tensor read in place (NO function-local
// host upload, the cudagraph-capture-bakes-stack-addresses UAF class the eager
// DFlashBlockAttention launcher had). Same f32 online-softmax recurrence + the D2
// in-block mask (full/non-causal or causal-SWA), applied over the COMBINED index
// (context rows [0,C_r) then block rows [C_r, C_r+blen_r)); the context is always
// position-ordered (ascending), so the combined mask matches the materialized one
// bit-for-bit. CUDA is capture-safe; CPU is the reference; a unit test cross-checks
// both against DFlashBlockAttention over an explicit combined buffer.
struct DFlashPagedBlockAttentionArgs {
  float scale = 0.0f;             // head_dim^-0.5
  bool causal = false;            // false=full(non-causal); true=causal-SWA
  int64_t sliding_window = 0;     // SWA window (>0); 0 = full causal when causal
  int num_reqs = 1;               // number of (1+k) query blocks
  int64_t block_size = 0;         // rows per paged context page (>0)
};

// Arguments for vt::DFlashGroupedConv — the DFlash2 draft's GROUPED DYNAMIC
// DEPTHWISE CONVOLUTION, wrapped around each attention and each MLP sublayer
// (SPEC-DFLASH2 W2, #1314).
//
// BEYOND-PIN. Ported from `_grouped_conv` and `DFlashGroupedConv`
// (vllm/model_executor/models/qwen3_dflash2.py @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`); the parity pin `555967922` does
// not carry the architecture at all and this op does NOT advance it.
//
// The math, in upstream's own terms:
//
//   out[i,c] = sum_{t=0..taps-1} (base[side,t,c] + delta[i,side,t,g(c)]) * x[i-t,c]
//
// with `g(c) = c / group_size` the channel's GROUP, and tap `t` contributing only
// where `(i mod block_size) >= t`. That mask is what makes the op a BLOCK
// convolution rather than a sequence one: a proposal position sees the positions
// before it inside its own (1+k) query block and NOTHING across the block
// boundary, which is how a DFlash2 draft gets causal structure without another
// backbone pass. `i` is the GLOBAL row index, exactly as upstream's
// `torch.arange(hidden_states.shape[0])` is, and it is the right one here for the
// same reason it is there: every request's block is contiguous and
// block_size-aligned, so `i mod block_size` IS the intra-block offset.
//
// Upstream computes that mask on a POWER-OF-TWO block as `position & (block-1)`
// and otherwise as `position % block`. Both arms are mirrored, and both are
// gated: the two published DFlash2 checkpoints ship block 8 and block 16 (both
// power-of-two, `z-lab/Qwen3.8-27B-DFlash2` and `z-lab/Muse-Glimmer-30B-DFlash2`)
// and upstream's own reference test parametrises 5 to reach the modulo arm.
//
// `block_size` is `1 + num_speculative_tokens`, NOT `dflash_config.block_size` —
// upstream sizes the conv by the QUERY block (the bonus token plus the mask
// tokens) rather than by the checkpoint key, which only supplies that value's
// default (`DFlash2Qwen3DecoderLayer.__init__` @ that head).
//
// SIDES. `base_kernel` is `[2, taps, hidden]` and dim 0 is the SIDE — 0 =
// `prepare` (before the sublayer), 1 = `finish` (after it) — NOT a tap. One
// projection of the sublayer input produces BOTH sides' deltas
// (`kernel_projection`: hidden -> 2*taps*num_groups), so `coefficients` is the
// same buffer for both calls and `side` selects the half. Passing the whole
// buffer rather than a slice mirrors upstream's `coefficients[:, side]` view
// without materializing a copy of a non-contiguous slice.
//
// ACCUMULATION. Every intermediate is rounded to the tensor dtype after each
// step, because upstream's chain materializes bf16 tensors at each one
// (`base + delta`, `coefficients * blocks`, `output += ...`). This is elementwise
// with no reduction-order freedom, so the CPU reference and the CUDA kernel are
// specified BIT-IDENTICAL rather than within an envelope.
//
// WHAT IS ACTUALLY GATED, because the two halves of that sentence are not
// equally proven. The per-step POLICY is pinned on CPU in bf16 by
// `tests/vt/test_ops_dflash2_grouped_conv.cpp` — one hand-computed case with
// literal expectations that differ from the rounded-once-at-the-end answer in
// six of eight outputs, plus three shapes asserted bit-exact against a reference
// that rounds where UPSTREAM materializes. On f32 this rounding is the identity
// by construction, so no f32 case can see it and none is claimed to. The CPU ==
// CUDA half is NOT proven: that case exists and is written to run, but it has
// never compiled on a host without `nvcc` and reports `no CUDA backend;
// skipping`. See `## Owed` O6 of `.agents/specs/dflash2-spec-decode.md`.
struct DFlashGroupedConvArgs {
  int64_t block_size = 0;   // 1 + num_speculative_tokens (the query block)
  int64_t taps = 0;         // dflash_config.conv_kernel_size
  int64_t num_groups = 0;   // hidden_size / conv_group_size
  int64_t group_size = 0;   // dflash_config.conv_group_size
  int64_t side = 0;         // 0 = prepare, 1 = finish (selects base/coefficient half)
};

// Arguments for vt::Dflash2SelectorEdges — the DFlash2 CANDIDATE SELECTOR's edge
// lattice (SPEC-DFLASH2 W3, #1314).
//
// BEYOND-PIN. Ported from `_score_edges` and `CandidateSelector.forward`
// (vllm/model_executor/models/qwen3_dflash2.py:208-276 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`); the parity pin `555967922`
// does not carry the architecture at all and this op does NOT advance it. The
// PR head MOVED from `19c9351904df4c63042671bc67a866ca48dc7d6f` on 2026-08-19
// (#1404); `_score_edges` and `CandidateSelector` are BYTE-IDENTICAL at the two
// heads, and only the enclosing model's `set_model_tag` and the LM-head guard
// changed, so this op cites the NEW head and its math is unaffected.
//
// The math, in upstream's own terms:
//
//   edge(b, l, p, c) = unary[b,l,c]
//                    + sum_r (pred_codebook[pid(b,l,p), r] * hidden[b,l,r])
//                            * succ_codebook[cand[b,l,c], r]
//
// where `pid(b,l,p)` is the PREDECESSOR token of slot `p` at step `l`: the
// request's verified ANCHOR token at `l == 0` (the same token for every `p`,
// upstream's `anchor_token_ids[:, None, None].expand(-1, 1, top_k)`), and
// `cand[b, l-1, p]` at every later step. So `p` indexes the previous step's K
// candidates and `c` this step's K candidates, which is exactly the transition
// lattice the W4 path walk consumes.
//
// `unary` is the candidate VALUE from `compute_candidates` — the target head's
// top-K logit AFTER `output_multiplier` and `final_logit_softcapping` — and it
// is f32 there and f32 here. Upstream broadcasts it over `p`
// (`unary_logits[:, :, None]`), so it is a per-CHILD bias and not a per-edge one.
//
// ACCUMULATION AND DTYPE. Upstream materializes two bf16 tensors inside the
// einsum chain and then promotes: `predecessors * hidden[:, :, None]` is a bf16
// tensor, the einsum's own output is a bf16 tensor, and `unary_logits + <bf16>`
// promotes to f32 by torch's own type-promotion rule. This op mirrors that
// placement exactly: the elementwise product rounds to the codebook dtype, the
// rank reduction accumulates in f32 and rounds ONCE to the codebook dtype, and
// only then is the f32 `unary` added. On f32 codebooks both roundings are the
// identity, so the policy is observable ONLY in bf16 — which is where it is
// gated, for the reason `## Owed` O6 records for the W2 convolution.
//
// The rank reduction IS a reduction, so unlike vt::DFlashGroupedConv this op is
// NOT specified bit-identical across backends: a CUDA mirror is free to reduce
// in a different order and is gated within an f32 envelope. That is a real
// difference from W2 and is stated here rather than inherited by analogy.
struct Dflash2SelectorEdgesArgs {
  int64_t top_k = 0;  // dflash_config.selector_top_k (16 on both published drafts)
};

// Arguments for vt::Dflash2PathWalk — the DFlash2 candidate selector's PATH WALK
// (SPEC-DFLASH2 W4, #1314).
//
// BEYOND-PIN. Ported from `_selector_walk_kernel`
// (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:16-79 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`).
//
// WHAT IT IS. The selector scores every (predecessor, child) transition of every
// step; the walk turns that lattice into k tokens. It starts at the verified
// anchor -- which the lattice already carries as EVERY predecessor slot of step
// 0, so the walk enters at slot 0 -- takes the best child, and reads the NEXT
// step's block at the predecessor row it just chose. The slot-to-slot dependency
// is the whole shape of the problem: step l cannot start until step l-1 has
// picked, so `L` steps are inherently sequential.
//
// ITS SHAPE IS UPSTREAM'S SHAPE, and that is a decision rather than a detail.
// Upstream runs ONE program per request: the K scores of a slot stay in
// registers and the step loop lives INSIDE the program, so `L` steps cost one
// launch rather than `L`. Spec `## Risks/decisions` D3 requires the walk to be
// on device from the first landing, because the identical sequential shape in
// DSpark shipped host-side and measured 28% of the 27B draft step
// (https://github.com/mudler/vllm.cpp/issues/436) before
// `SampleSequentialDevice` moved it. The CUDA arm here is one block per request
// with a shuffle argmax; the CPU arm is the authoritative reference.
//
// GREEDY IS THE ONLY ARM, and it is upstream's greedy arm exactly. At the moved
// head the walk's two hand-written branches collapse into one
// `gumbel_noised_argmax` call whose temperature argument is
// `temperature if SAMPLE_PROBABILISTIC else 0.0`, and `SAMPLE_PROBABILISTIC` is
// `self.draft_logits is not None`, which is set only for
// `draft_sample_method == "probabilistic"`. At temperature 0 that helper divides
// by nothing, adds no noise, and returns `tl.max(logits, axis=0,
// return_indices=True)`; the `USE_FP64` cast it may apply is order-preserving on
// fp32 inputs and cannot move a greedy answer. This engine refuses
// `draft_sample_method: "probabilistic"` BY NAME at
// `vllm::ParseSpeculativeConfigJson` (src/vllm/config/speculative.cpp) and
// verifies accept-iff-equal, so no configuration can reach the noised arm and
// nothing could consume the realized proposal distribution it exists to feed.
// Landing that arm would be landing code no entry point can reach; it is owed,
// with its layout, at `## Owed` of .agents/specs/dflash2-spec-decode.md.
//
// THE TIE-BREAK AND THE -inf ROW ARE PART OF THE CONTRACT. `tl.max(...,
// return_indices=True)` resolves a tie to the LOWEST index, and upstream loads
// masked lanes as -inf, so a row that is entirely -inf resolves to index 0
// rather than to "no index". Both are pinned by literal cases and both arms
// implement them identically: the reduction is seeded at -inf with slot index
// `top_k`, keeps a candidate only when it STRICTLY exceeds the running best (so
// a NaN never wins, on either arm), and collapses a `top_k` seed back to 0. A
// different tie rule would pick a different PREDECESSOR for the next step, so it
// moves the whole remaining path and not one token.
//
// UNLIKE vt::Dflash2SelectorEdges this op is specified BIT-EXACT across
// backends. It performs no arithmetic -- only comparisons and one gather -- so
// there is no reduction order for a backend to differ in.
//
// STRICTNESS IS THE WHOLE OF THAT CLAIM, and it was not free. W4's fresh review
// measured the two arms apart on a NaN-bearing row: the CUDA per-lane scan
// carried `|| (v == best && j < slot)` beside its strict `>`, a clause that is
// unreachable once a lane has claimed anything (`j` only ascends) and whose one
// effect was at the SEED -- a lane holding -inf compared equal to the -inf seed
// and claimed a slot the CPU arm refuses. `[NaN,-inf]` read cpu 0 / cuda 1,
// `[NaN,NaN,-inf]` read cpu 0 / cuda 2, and every NaN-free row agreed,
// including the all -inf row and the forced tie group, so no fixture without a
// NaN could see it. The clause is now DELETED rather than this claim narrowed,
// and the lower-slot tie rule stays where it is needed: the cross-lane
// butterfly, which combines lane winners out of slot order. BOTH HALVES OF
// THAT DELETION ARE NOW COVERED, and #1518 corrects an earlier sentence here
// saying it had never been compiled or run on a device. It COMPILES on every
// pull request: `src/vt/cuda/cuda_ops.cu` is in the CUDA source list
// (CMakeLists.txt, `target_sources(vllm PRIVATE ...)` under `if(VLLM_CPP_CUDA)`)
// that CI's `build-cuda-fat` job builds for ten architectures
// (`80;86;87;89;90a;100a;103a;110;120a;121a`). And it has RUN: the operator
// executed this suite on `dgx:gpu0` (GB10, sm_121a) at the W4 merge commit,
// reporting 83 assertions on device against 49 on CPU, `Status: SUCCESS!`, with
// zero `no CUDA backend; skipping` lines -- the increment includes the NaN row
// chained into the parity fixture, which is the row that measured the
// divergence. What remains owed at `## Owed` O11 of
// .agents/specs/dflash2-spec-decode.md is the rest of that entry, not this
// deletion; the AUTHORING HOST still has no `nvcc` and still skips the case
// locally. NO SHIPPED PATH FEEDS THIS OP A NaN:
// the lattice comes from vt::Dflash2SelectorEdges over a target LM head, so the
// row that measured the difference is synthetic, and it is a gap in the reach
// of the contract rather than in any draft a user can obtain.
struct Dflash2PathWalkArgs {
  int64_t top_k = 0;  // dflash_config.selector_top_k (16 on both published drafts)
};

// Arguments for vt::TopKValuesIndices — the vocabulary top-k that EMITS the
// surviving (id, value) pairs (SPEC-DFLASH2 W3 / D2, #1314).
//
// BEYOND-PIN. Ported from `_topk`
// (vllm/model_executor/models/qwen3_dflash2.py:60-64 @ vllm-project/vllm#52816
// head `66e5414c6d75a8529473d977f7458c140bbab8a0`), which is `torch.topk(scores,
// k, dim=-1)` off CUDA and FlashInfer's radix `top_k(..., sorted=True,
// deterministic=True)` on it.
//
// WHY NOT FLASHINFER'S RADIX KERNEL. Spec `## Risks/decisions` D2: `topk.cuh` is
// 3380 lines of general kernel (multi-CTA, deterministic mode, three tie-break
// modes, dynamic shared-memory sizing) for a shape that is fixed and small here
// — K = 16 over a 248320 vocabulary for `num_reqs * k` rows, about 224 at
// concurrency 32. The CUDA arm instead extends the sort-free block-cooperative
// pivot-bracket threshold search this repository already carries and gates
// (`src/vt/cuda/cuda_sample.cu::ApplyTopKTopPRowKernel`, ported from the SAME
// FlashInfer `TopK/TopPRenormProb` approach) so that it COMPACTS and ORDERS the
// survivors instead of masking below the k-th largest.
//
// TIE-BREAK IS PART OF THE CONTRACT, not an implementation detail. The threshold
// search finds an exact array VALUE, so it keeps whole tie groups atomically and
// can leave more than k survivors; something then has to choose among equals.
// This op returns exactly k pairs ordered by DESCENDING value with ties broken
// by ASCENDING index, which is `torch.topk`'s CPU order and what FlashInfer's
// `deterministic=True` exists to provide. A backend that broke ties differently
// would reorder the selector's candidate slots and move acceptance without
// raising anything, so the CPU reference pins it and the CUDA arm mirrors it.
//
// NaN IS THE ONE POINT WHERE THE TWO ARMS ARE NOT EQUAL, and this is stated here
// because a contract that claimed otherwise would be a claim no shipped backend
// delivers. The CPU arm orders NaN FIRST, which is `torch.topk(largest=True)`'s
// own answer, and it is stated rather than left to the sort: leaving it implicit
// made the CPU comparator an intransitive equivalence and therefore undefined
// behaviour, not merely an unusual result. The CUDA arm DOES NOT ORDER NaN
// FIRST and cannot as written -- `TopKValuesIndicesRowKernel`'s pivot bracket
// uses `fmaxf`/`fminf`, which return the non-NaN operand, and its survivor pass
// tests `r[j] > thr`, which is false for a NaN, so the search can never select
// one. Measured on a GB10 on 2026-08-20 rather than argued:
// [#1489](https://github.com/mudler/vllm.cpp/issues/1489), where the direct
// cross-arm comparison read `gpu.indices[0] == cpu.indices[0]` as `2 == 1`.
// Reconciling the kernel to NaN-first is owed to that issue; until it lands the
// device gate is scoped to the rows the kernel implements. NO SHIPPED PATH FEEDS
// THIS OP A NaN LOGIT -- the candidate values come from a target LM head -- so
// the row that pins the CPU order is synthetic in the same sense the padding row
// is, and the asymmetry is a gap in the contract's reach rather than in any
// output a user can obtain.
//
// `num_org_vocab_padding` mirrors upstream's
// `lm_head.shard_indices.num_org_vocab_padding`: that many columns at the END of
// each row are forced to -inf BEFORE the search, so a padded head can never
// contribute a candidate. It is 0 on every path this engine ships today (the
// DFlash lane's lm_head is the raw unpadded checkpoint tensor and there is no
// vocab-parallel sharding), and it is implemented and gated synthetically rather
// than claimed as checkpoint coverage — the same posture `## Upstream chain`
// records for the three output scalars. Upstream's companion
// `org_vocab_start_index` is applied by the CALLER
// (`vllm::Qwen3DFlash2Model::ComputeCandidates`), not here, because it is an
// id-space rebase of the result rather than a property of the search.
struct TopKValuesIndicesArgs {
  int64_t k = 0;                        // dflash_config.selector_top_k
  int64_t num_org_vocab_padding = 0;    // trailing columns forced to -inf first
};

// Backend-neutral local-attention window, matching FlashAttention's
// `window_size=(left, right)` convention. The bounds are inclusive distances
// from the bottom-right-aligned absolute query position: (W-1, 0) is a causal
// decoder window of W tokens and (W-1, W-1) is the symmetric encoder form.
// Full attention is represented by std::nullopt on PagedAttentionArgs, never by
// a backend-specific sentinel pair.
struct AttentionWindow {
  int32_t left = 0;
  int32_t right = 0;
};

// Paged attention args (M1.6). Same softmax convention as AttentionArgs — the
// paged op generalizes the dense M0.9 attention to the varlen/batched/paged
// case and MUST agree with it on the single-sequence contiguous read.
struct PagedAttentionArgs {
  // Softmax scale, applied to the qk dot product (upstream FlashAttentionImpl
  // self.scale = head_size^-0.5). Must be set explicitly (> 0).
  float scale = 0.0f;
  // Causal masking: a query token at absolute position p attends only to key
  // positions j <= p. True for the decoder path; non-causal carried for
  // fidelity (matches AttentionArgs.causal).
  bool causal = true;
  // OPTIONAL local-attention bounds. For an absolute query position p, visible
  // keys are intersected with [p-left, p+right] after the causal/full bound is
  // applied. Query positions use FlashAttention's bottom-right alignment:
  // p = seq_len - query_len + local_query_index. std::nullopt preserves the
  // existing full causal/non-causal behavior exactly.
  std::optional<AttentionWindow> window_size = std::nullopt;
  // OPTIONAL attention logit soft-cap (vLLM Attention(logits_soft_cap=...),
  // gemma2.py:202 attn_logit_softcapping). When > 0 each pre-softmax score S is
  // replaced by cap * tanh(S / cap) before the online softmax. 0.0 (default)
  // leaves the plain scaled-dot path byte-identical — every existing model uses
  // the default, so this is diff-inert for them. Gemma-2/4 set it (50.0).
  float logits_soft_cap = 0.0f;
  // OPTIONAL host-resident query_start_loc[num_reqs+1] (same values as the
  // device `query_start_loc` tensor). When set, the CUDA prefill flash/WMMA
  // launchers size the per-request query-tile grid from these host values and
  // build the device tile array with a device meta-kernel, avoiding the
  // per-layer D2H copy + cudaStreamSynchronize that drained the pipeline every
  // full-attention prefill layer (~10-12 syncs/step; prefill only 43.7%
  // GPU-busy). nullptr => the launcher falls back to the D2H+sync (op unit tests
  // / callers without a host qsl). Mirrors GdnArgs::query_start_loc_host and the
  // decode StepDevInputs device-resident metadata pattern.
  const int32_t* query_start_loc_host = nullptr;
  // OPTIONAL host-known max context length in this batch (max over the device
  // `seq_lens` values = CommonAttentionMetadata::max_seq_len; an upper bound is
  // safe — it only sizes grids/rounded dims, per-request geometry stays on the
  // device values). When > 0 the FA-2 prefill launcher sizes its grid without a
  // device read (companion to query_start_loc_host). 0 => that launcher falls
  // back to the D2H+sync.
  int32_t max_seq_len = 0;
  // OPTIONAL fp8 KV-cache read (KV-FP8 W1 CPU, W2 CUDA, W6 ROCm). kAuto
  // (default) => the cache holds the model float dtype and is read directly —
  // every existing caller is byte-identical. When != kAuto the K/V cache pages
  // are 1-byte fp8 (DType::kI8 storage) and each read is DEQUANTIZED as
  // Dequant(fp8) * k_scale|v_scale before entering the f32 softmax, mirroring
  // the fp8 attention read path (scaled_vec_conversion<float,uint8_t>,
  // quant_utils.cuh:302-308). k_scale / v_scale are the per-tensor scales from
  // BaseKVCacheMethod (kv_cache.py:108-191) — 1.0 is the uncalibrated default.
  // Per-head scales are a later brick. Implemented on CPU, CUDA, and ROCm.
  // kMETAL registers kPagedAttention for the FLOAT path only, and because
  // these fields are ADDITIVE the provider table cannot tell the two arms
  // apart, so src/vt/ops.cpp refuses Metal by name.
  Fp8KVCacheDataType kv_cache_dtype = Fp8KVCacheDataType::kAuto;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  // OPTIONAL spec-as-decode classification (SPEC-DFLASH2 W10, #1857): the
  // batch's uniform speculative verify length, classified by the caller
  // (`vllm::v1::SpecAsDecodeQueryLen` mirrors backend.py:718-736 @ b389ac2946,
  // the 1 + 2*K reorder threshold for a parallel-drafting speculator), or 0.
  // A ROUTING HINT, not a semantic change: the attention output is defined by
  // the tensors + scale/causal/window exactly as before, so a backend that
  // ignores the field (CPU, and every non-CUDA provider) is still correct —
  // which is why, unlike `kv_cache_dtype`, this field extends no provider
  // refusal. Only the CUDA kernel SELECTION reads it, to keep a uniform-qlen
  // verify (q <= 1 + 2K) on the split-KV DECODE lane instead of the
  // num_splits=1 prefill ladder (`include/vt/paged_attn_route.h`).
  int32_t uniform_spec_query_len = 0;
  // OPTIONAL attention-sink bias (gpt-oss / MiMoV2 SWA layers). A 1-D per-head
  // scalar tensor [num_q_heads] that is appended as an extra key column BEFORE
  // the softmax: the score for sink head h at position p is attn_sink[h]. The
  // softmax probability mass assigned to the sink column is DISCARDED after the
  // softmax (the output is renormalised over the real keys only). This mirrors
  // the gpt-oss sink mechanism. nullptr => no sink (every model before MiMoV2).
  const Tensor* attn_sink = nullptr;
};

// Arguments for vt::MlaDecodeAttention (MLA campaign W4). Mirrors the scalar
// arguments `TritonMLAImpl.forward_mqa` passes to `decode_attention_fwd`
// (vllm/v1/attention/backends/mla/triton_mla.py:242-259 @ e24d1b24).
struct MlaDecodeAttentionArgs {
  // `self.scale` — for DeepSeek this is head_dim^-0.5 TIMES the YaRN mscale^2
  // correction (mla_attention.py computes it once and hands it to the kernel as
  // a plain float; the kernel itself knows nothing about mscale). Must be > 0.
  float scale = 0.0f;
  // NUM_KV_SPLITS. 0 (the default) => the impl computes it exactly like
  // `_compute_num_kv_splits` (triton_mla.py:40-47):
  //     min(next_pow2(max(1, max_seq_len // 512)), sm_count * 2)
  // from `max_seq_len` below. 1 forces the single-split (batch-invariant)
  // reduction upstream uses under VLLM_BATCH_INVARIANT (triton_mla.py:212-213).
  int32_t num_kv_splits = 0;
  // Host-known max over `seq_lens` (CommonAttentionMetadata::max_seq_len). Only
  // used to derive `num_kv_splits` when that is 0; an upper bound is safe. When
  // both are 0 the impl falls back to 1 split.
  int32_t max_seq_len = 0;
  // ─── the SLIDING-WINDOW decode arm (dots3-note W4b-2, #699) ───────────────
  // OPTIONAL local-attention bounds, the same `AttentionWindow` convention
  // `PagedAttentionArgs::window_size` already uses: for the bottom-right
  // aligned absolute query position `p = seq_len - 1` (MLA decode is ONE query
  // per row), visible keys are `[p - left, p + right]` intersected with
  // `[0, seq_len)`. `std::nullopt` — every DeepSeek / MiniCPM3 / Kimi-Linear
  // registration — leaves the full-context loop byte-identical: the window is
  // not a mask applied afterwards, it is the loop's START BOUND, so an absent
  // window is a NOT-TAKEN branch rather than a no-op.
  //
  // UPSTREAM. `TritonMLAImpl` itself REJECTS a sliding window
  // (triton_mla.py:165-171); dots3-note SUBCLASSES it —
  // `Dots3NoteTritonMLAImpl.__init__` passes `sliding_window=None` to super and
  // keeps the value on itself (`vllm/models/dots3_note/nvidia/attention.py`
  // :439-468 @ `bc2d63e650`), then `_forward_swa_mqa` (`:470-563`) gathers a
  // window-bounded slice of the paged latent and masks the scores with
  // `kv_positions >= query_position - WINDOW_SIZE + 1` (`:152`) and
  // `kv_positions <= query_position` (`:151`). `WINDOW_SIZE` is
  // `sliding_window_size` = 513, so `left == sliding_window - 1`, matching the
  // `window_size=(sliding_window - 1, 0)` upstream hands FlashAttention on the
  // PREFILL half (`:300`). The gather is upstream's Triton WORKSPACE strategy;
  // the paged kernels here read the block table directly over the same key
  // range, which is the same function with no gather and no mask.
  //
  // `right` must be 0: an MLA decode query IS the last position of its own
  // sequence, so a positive right bound could only admit keys that do not
  // exist. Anything else is refused BY NAME in ops.cpp rather than ignored.
  std::optional<AttentionWindow> window_size = std::nullopt;

  // ─── the DSA SELECTED-SLOT decode arm (dots3-note W4b-3c, #699) ───────────
  // OPTIONAL sparse key selection: instead of walking `[j_start, seq_len)`, the
  // kernel walks a per-request LIST of token positions.
  //
  //   topk_indices  [batch, topk] i32 — TOKEN POSITIONS within the request's own
  //                 sequence (not cache slots), `-1` meaning "no token".
  //   valid_counts  [batch]       i32 — how many leading entries of that row are
  //                 live. Entries at or past the count are ignored.
  //
  // BOTH NULL — every DeepSeek / MiniCPM3 / Kimi-Linear caller, and every
  // dots3-note SLIDING layer — is a NOT-TAKEN BRANCH rather than a no-op: the
  // key loop keeps the contiguous bounds it had, the split partition is derived
  // from the same `seq_len` it always was, and the result is bit-identical.
  // Exactly ONE of the two present is refused BY NAME in ops.cpp.
  //
  // THE POSITIONS STAY POSITIONS. The kernel keeps its existing
  // `blk = j / block_size` block-table walk for a selected `j`, which is this
  // tree's equivalent of upstream's `triton_convert_req_index_to_global_index`
  // (`vllm/models/dots3_note/nvidia/attention.py:760-767` @ `bc2d63e650`) —
  // done INSIDE the kernel rather than ahead of it. Upstream needs the
  // conversion because FlashAttention consumes a flat `as_strided` view of the
  // cache (`:792-795`); a block-table walk needs neither the conversion nor the
  // view, and the selected key set is identical.
  //
  // A SELECTION LISTING EVERY CAUSAL KEY IS BIT-FOR-BIT the unselected call, on
  // both backends: the list is walked in ascending order, so the f32 online
  // softmax sees the identical summation order, and the CUDA split partition is
  // derived from `count` exactly as the dense one is from `seq_len - kv_start`.
  // The op gate asserts that identity byte for byte, because a mask applied
  // after the fact cannot pass it.
  //
  // A WINDOW AND A SELECTION TOGETHER are refused BY NAME. Upstream cannot
  // produce the pair: `Dots3NoteSlidingAttention` sets `self.indexer = None`
  // and `is_sparse = False` (`vllm/models/dots3_note/nvidia/model.py:432-434`),
  // so a windowed layer carries no indexer at all.
  const Tensor* topk_indices = nullptr;
  const Tensor* valid_counts = nullptr;
  // PER-HEAD ATTENTION SINK — `[num_heads]` f32, or null.
  //
  // A sink is one extra logit that contributes to the softmax DENOMINATOR ONLY:
  // it removes probability mass from the row without contributing a value, so a
  // row can attend to "nothing" (`vllm/models/deepseek_v4/attention.py:218-222`;
  // this tree's host reference is `SoftmaxWithSink`,
  // `src/vllm/model_executor/models/deepseek_v4_dsa.cpp:121-139`).
  //
  // NULL BY DEFAULT, so every existing caller -- Kimi-Linear, dots3-note,
  // MiniCPM3, DeepSeek-V2 -- is BIT-IDENTICAL: the online softmax seeds `m` at
  // `-inf` and `l` at 0 exactly as it did before. Present, it seeds `m` at the
  // head's sink and `l` at 1 (`exp(sink - sink)`), which is the same arithmetic
  // the host reference does in its two-pass form.
  //
  // WHERE A SPLIT KERNEL MUST PUT IT. A CUDA implementation that splits the KV
  // over `num_kv_splits` must add the sink in the FINAL reduction, not per
  // split, or it is counted once per split instead of once per row. At
  // `num_kv_splits == 1` both placements agree, so a gate that only ran the
  // batch-invariant path would pass a kernel that is wrong everywhere else.
  const Tensor* attn_sink = nullptr;
};

// Arguments for vt::DsaIndexerLogits (dots3-note W4b-3c, #699). The two
// GLOBAL scalars of upstream's weight fold
// (`weights * q_scale * softmax_scale * n_head_scale`,
//  vllm/model_executor/models/deepseek_v2.py:840 and the fused kernel's
//  `weight * q_scale * softmax_scale * head_scale`,
//  vllm/model_executor/layers/sparse_attn_indexer.py:203-207, both @
//  `bc2d63e650`), plus the OPTIONAL per-(token,head) fp8 `q_scale`.
struct DsaIndexerLogitsArgs {
  // `self.softmax_scale = self.head_dim**-0.5` (deepseek_v2.py:709).
  float softmax_scale = 0.0f;
  // `self.n_head_scale = self.n_head**-0.5` (deepseek_v2.py:742).
  float n_head_scale = 0.0f;
  // OPTIONAL per-token-per-head fp8 quantization scale, f32 [num_tokens,
  // index_n_heads] on the queue's device. `nullptr` — the UNQUANTIZED arm —
  // is upstream's `q_scale == 1` and is what both dots3-note arms run today.
  //
  // NOTE what a mutation of the two scalars above can and cannot show: they are
  // GLOBAL and POSITIVE, so dropping either rescales every logit of every row
  // by one positive constant and CANNOT move an argmax. Their mutations read
  // green DEFINITIONALLY. `q_scale` is per-(token,head) and is the only member
  // of the fold that can. The op gate therefore asserts the folded logits BY
  // VALUE rather than relying on the selection to notice.
  const Tensor* q_scale = nullptr;
};

// Arguments for vt::MlaPrefillAttention (MLA campaign W5). Mirrors the scalar
// arguments `FlashAttnPrefillBackend` passes to `flash_attn_varlen_func`
// (vllm/v1/attention/backends/mla/prefill/flash_attn.py:205-248 @ e24d1b24).
struct MlaPrefillAttentionArgs {
  // `self.scale` — head_dim^-0.5 TIMES the YaRN mscale^2 correction for
  // DeepSeek, handed to the kernel as a plain float (`flash_attn.py:222,245`).
  float scale = 0.0f;
  // `causal=True` for the NEW-TOKENS call (`flash_attn.py:223`), `causal=False`
  // for every CONTEXT-CHUNK call (`:246`, "Context is unmasked"). Causal here is
  // FlashAttention's BOTTOM-RIGHT alignment: query index i of a request whose
  // query length is Lq and key length is Lk sees keys j <= i + (Lk - Lq).
  bool causal = true;
  // Host-known max over the per-request query / key lengths
  // (`max_seqlen_q` / `max_seqlen_k`, `flash_attn.py:220-221,243-244`). Used for
  // GRID SIZING and the rounded dims only — the per-request geometry reads the
  // DEVICE cu_seqlens, so an UPPER BOUND is safe. 0 => the launcher falls back
  // to a small D2H + sync (op unit tests / callers without host lengths), the
  // same fallback the FA-2 paged prefill launcher uses.
  int32_t max_seqlen_q = 0;
  int32_t max_seqlen_k = 0;
  // ─── the SLIDING-WINDOW prefill arm (dots3-note W4b-2, #699) ──────────────
  // OPTIONAL local-attention bounds, the `AttentionWindow` convention. Query
  // `i` of a request whose query length is `Lq` and key length is `Lk` sits at
  // the bottom-right aligned position `p = i + (Lk - Lq)`; visible keys are
  // `[p - left, p + right]` intersected with `[0, Lk)`.
  //
  // UPSTREAM is literally this pair: `Dots3NoteFlashAttnPrefillBackend.
  // run_sliding_window` calls `_flash_attn_varlen_diff_headdims(..., causal=
  // True, window_size=(sliding_window - 1, 0))`
  // (`vllm/models/dots3_note/nvidia/attention.py:279-305` @ `bc2d63e650`, the
  // window at `:300`), so `left == sliding_window - 1` and `right == 0`.
  //
  // `causal` must be TRUE whenever this is set, and `right` must be 0. Both are
  // refused BY NAME in ops.cpp rather than approximated: FlashAttention's local
  // mask REPLACES the causal specialization (the adapter normalizes
  // `is_causal = causal && !is_local`, cuda_flash_attn_fa2.cu:472-476), so a
  // non-causal window would have to be spelled with an infinite right bound,
  // which this struct cannot say. Upstream never asks for one — every windowed
  // call it makes is the causal `(W-1, 0)` pair above.
  std::optional<AttentionWindow> window_size = std::nullopt;
};

// Router SCORING function. softmax over all E is the Qwen3.6 / DeepSeek-V2
// behavior; sigmoid (elementwise, NOT normalized across experts) is what
// DeepSeek-V3 / R1 use with `topk_method == "noaux_tc"`. Mirrors
// vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:110-117.
enum class MoeScoringFunc {
  kSoftmax,  // scores = softmax(gating_output, dim=-1)      (:111-112)
  kSigmoid,  // scores = gating_output.sigmoid()             (:113-114)
};

// MoE router top-k args (.agents/specs/moe-semantics.md §3 is the formula reference).
//
// The four fields below `renormalize` are the W3 GROUPED-TOPK (`noaux_tc`)
// extension, ported from grouped_topk_router.py:80-161 @ pin e24d1b24. They are
// ADDITIVE: every field defaults to the pre-W3 behavior, and `num_expert_group
// == 0` selects the original ungrouped softmax+top-k path VERBATIM (a separate
// kernel — the existing one is not touched), so the 27B / 35B / Coder / dense
// routers stay byte-identical.
struct MoeRouterTopKArgs {
  // Number of experts selected per token (top_k = num_experts_per_tok).
  int top_k = 0;
  // renormalize = norm_topk_prob (True for Qwen3.6, moe-semantics.md §1/§3):
  // divide the k selected softmax probs by their sum (denom>0 guard).
  bool renormalize = true;

  // --- grouped-topk (`noaux_tc`) extension ---------------------------------
  // scoring_func: softmax (V2 / Qwen) vs sigmoid (V3 / R1).
  MoeScoringFunc scoring_func = MoeScoringFunc::kSoftmax;
  // num_expert_group == config.n_group. 0 DISABLES grouping entirely (the
  // pre-W3 path). When > 0 it must divide num_experts exactly.
  int num_expert_group = 0;
  // topk_group == config.topk_group: how many expert GROUPS survive the
  // first-level mask. Must be in [1, num_expert_group] when grouping is on.
  int topk_group = 0;
  // routed_scaling_factor: a final multiply on the routing weights
  // (grouped_topk_router.py:159-160; deepseek_v2.py:288). 1.0 == no-op.
  float routed_scaling_factor = 1.0f;
};

// Kernel registration contract. Backends register one kernel per (OpId,
// DeviceType); the kernel's signature must match the alias for its op
// exactly. Register with a static_cast against the alias so signature drift
// is a compile error:
//   RegisterOp(OpId::kMatmul, DeviceType::kCPU,
//              reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
// The public op functions below validate arguments, then dispatch through
// these types. A kernel that does not support a validated dtype combination
// must throw loudly, never silently truncate.
using MatmulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MatmulNvfp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, float);
using ScaledFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, float, Fp4ScaleLayout);
using SiluMulFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, float,
             Fp4ScaleLayout);
using SiluAndMulFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, float, Fp4ScaleLayout);
using SigmoidGateFp4QuantFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, float,
             Fp4ScaleLayout);
using MatmulNvfp4Fp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float);
using MatmulNvfp4CutlassFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor*, float);
using MatmulFp8CutlassFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, float);
// The trailing bool is `claims_splitk1_premise` — see MatmulFp8CublasLt below.
using MatmulFp8CublasLtFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, float, bool);
// Same operands as MatmulFp8CublasLtFn, but the trailing scalar alpha becomes a
// device f32 [N] vector — one folded alpha per OUTPUT COLUMN.
using MatmulFp8CublasLtAlphaVecFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor& /*alpha_vec*/, bool);
using QuantFp8StaticFn = void (*)(Queue&, Tensor&, const Tensor&, float);
// Two outputs, because the scale is computed rather than supplied: the fp8 bytes
// and the f32 [M, K/group_size] per-group scale.
using QuantFp8GroupFn = void (*)(Queue&, Tensor& /*out_fp8*/, Tensor& /*out_scale*/,
                                 const Tensor& /*x*/, int /*group_size*/);
// Two scale streams and the block geometry, where the per-tensor fp8 GEMMs above
// carry one scalar alpha. The geometry is not a convenience: it is what selects
// which scale pair each K-block multiplies by.
using MatmulFp8BlockScaledFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*a_fp8*/,
                                        const Tensor& /*a_scale*/, const Tensor& /*b_fp8*/,
                                        const Tensor& /*b_scale*/, int /*block_n*/,
                                        int /*block_k*/);
using RmsNormQuantFp8Fn = void (*)(Queue&, Tensor& /*out_fp8*/, Tensor* /*out_bf16*/,
                                   const Tensor& /*x*/, const Tensor& /*weight*/,
                                   const RmsNormArgs&, Tensor* /*residual*/, float /*input_scale*/);
using RmsNormGatedQuantFp8Fn = void (*)(Queue&, Tensor& /*out_fp8*/, const Tensor& /*x*/,
                                        const Tensor& /*gate*/, const Tensor& /*weight*/,
                                        const RmsNormGatedArgs&, float /*input_scale*/);
using SwizzleBlockscaleFn = void (*)(Queue&, Tensor&, const Tensor&);
// vt::BatchedMatmul (`torch.bmm`) — same shape as MatmulFn but rank-3 and
// stride-driven; a distinct alias so registrations read unambiguously.
using BatchedMatmulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// vt::ConcatMlaNopeRope — out[..., :Dn] = nope, out[..., Dn:] = rope.
using ConcatMlaNopeRopeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using MoeGroupedGemmNvfp4Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&,
             const Tensor&, const Tensor&);
using MoeGroupedGemmBf16Fn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&);
// kMoeGroupedGemmBf16GateUpSilu: out[P,N] bf16 = silu(gate)*up, gate/up = grouped
// bf16 GEMM of act[T|P,K] against per-expert weight-pointer arrays gate_ptrs/
// up_ptrs [E] i64, expert_ids[P] i32, optional row_map[P] i32. Same convention as
// MoeGroupedGemmBf16 with a SECOND weight-pointer array + the fused SwiGLU epilogue.
using MoeGroupedGemmBf16GateUpSiluFn =
    void (*)(Queue&, Tensor& /*out*/, const Tensor& /*act*/, const Tensor& /*expert_ids*/,
             const Tensor* /*row_map*/, const Tensor& /*gate_ptrs*/, const Tensor& /*up_ptrs*/);
using MoeGroupedGemmBf16GateUpSiluNativeFn = MoeGroupedGemmBf16GateUpSiluFn;
using MoeGroupedGemmBf16WeightedFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&,
             const Tensor& /*route_weights*/);
using MoeCombinePreweightedFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor* /*shared*/, float /*routed_scale*/);
// kMatmulBTQuantGrouped: out[P,N], act[P,K] (f32/bf16), weight[E*N,K] block-quant,
// expert_ids[P] i32 — weight row for (p,n) is expert_ids[p]*N + n.
using MatmulBTQuantGroupedFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&);
// kMoeGateUpSwiGLUGrouped: out[P,N] f32 adown, act[Pa,K] (Pa==1 broadcast),
// gate_w/up_w[E*N,K] SAME block-quant dtype, expert_ids[P] i32, float limit. See
// vt::MoeGateUpSwiGLUGrouped / OpId::kMoeGateUpSwiGLUGrouped.
using MoeGateUpSwiGLUGroupedFn =
    void (*)(Queue&, Tensor& /*out*/, const Tensor& /*act*/, const Tensor& /*gate_w*/,
             const Tensor& /*up_w*/, const Tensor& /*expert_ids*/, float /*limit*/);
// Marlin NVFP4 W4A16 grouped-MoE GEMM (lift of vLLM moe_wna16_marlin_gemm; see
// MoeGroupedGemmNvfp4Marlin below). Scalar params travel in MoeMarlinArgs.
struct MoeMarlinArgs {
  int moe_block_size = 0;  // vLLM moe_align_block_size block (16..64, or 8)
  int top_k = 0;
  int size_m = 0;  // number of tokens (rows of `a`)
  int size_n = 0;  // output features
  int size_k = 0;  // input features (contraction; multiple of 16)
  bool mul_topk_weights = false;  // fold topk_weights into the output (down proj)
  // Block-scale format selector. Default = NVFP4 (fp8-e4m3 scales, group 16,
  // per-tensor global scale). group_size 32 + mxfp4=true selects the MXFP4 path
  // (E8M0/UE8M0 scales => s_type kFE8M0fnu, group_blocks 2, NO global scale;
  // the `global_scale` tensor is ignored). Mirrors vLLM's is_nvfp4 branch.
  int group_size = 16;
  bool mxfp4 = false;
};
using MoeGroupedGemmNvfp4MarlinFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
             const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MoeMarlinArgs&);
// DENSE Marlin W4A16 GEMM (lift of vLLM's own dense marlin_gemm; see
// MarlinDenseGemm below). Scalar params travel in MarlinDenseArgs. Unlike the
// MoE path there is NO moe_align gather (sorted_token_ids/expert_ids/top_k):
// `a` is a plain [size_m, size_k] contiguous activation (lda = size_k).
struct MarlinDenseArgs {
  int size_m = 0;  // number of tokens (rows of `a`)
  int size_n = 0;  // output features
  int size_k = 0;  // input features (contraction; multiple of 16)
  // Block-scale format selector, identical semantics to MoeMarlinArgs: default =
  // NVFP4 (fp8-e4m3 scales, group 16, per-tensor global scale). group_size 32 +
  // mxfp4=true selects the MXFP4 path (E8M0 scales, group_blocks 2, NO global
  // scale; the `global_scale` tensor is ignored). Mirrors vLLM's is_nvfp4 branch.
  int group_size = 16;
  bool mxfp4 = false;
};
using MarlinDenseGemmFn =
    void (*)(Queue&, Tensor& /*c*/, const Tensor& /*a*/, const Tensor& /*b_q_weight*/,
             const Tensor& /*b_scales*/, const Tensor& /*global_scale*/, Tensor& /*workspace*/,
             const MarlinDenseArgs&);
using MoeSiluMulFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// kMoeRelu2: out[i] = relu(x[i])^2 — the NON-GATED MoE activation (one input).
using MoeRelu2Fn = void (*)(Queue&, Tensor&, const Tensor&);
// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). These replace host-side
// loops so the decode step can run entirely on-device (CUDA-graph capture).
// All math in f32; dims are inferred from the tensor shapes (no args structs).
using CastBf16Fn = void (*)(Queue&, Tensor&, const Tensor&);
using PermuteVHeadsFn = void (*)(Queue&, Tensor&, const Tensor&, int64_t, int64_t,
                                  int64_t, int64_t);
using CastF32Fn = void (*)(Queue&, Tensor&, const Tensor&);
using CastF16Fn = void (*)(Queue&, Tensor&, const Tensor&);
using MulColVecF32Fn = void (*)(Queue&, Tensor&, const Tensor&);
using AttnGateSplitFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&);
using SigmoidGateBf16Fn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using GdnGBetaFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&);
using GdnConvSplitFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
using QkvSplitFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
// Fused GDN post-conv prep (mirror of fla fused_gdn_prefill_post_conv):
// conv-split + q/k l2norm + g/beta gating in ONE launch. eps travels in
// L2NormArgs (the q/k l2norm eps; softplus threshold 20 baked in as in GdnGBeta).
using GdnPostConvFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                               const L2NormArgs&);
// Per-step RoPE cos|sin cache fill (fused-attn-preamble prep): cos_sin[T,rot] f32
// from positions[T] (RopeArgs.base/rotary_dim). Cols [0,rot/2)=cos, [rot/2,rot)=sin.
using RopeCosSinCacheFn = void (*)(Queue&, Tensor&, const Tensor&, const RopeArgs&);
// Fused full-attention preamble (split q|gate + gemma qk-RMSNorm + partial NeoX
// RoPE-from-cache + gate passthrough) in ONE launch. See AttnQkNormRopeGate below.
using AttnQkNormRopeGateFn =
    void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, const RmsNormArgs&, const RopeArgs&);
// Gate-free fused preamble (kAttnQkNormRope): q3/k3 are normed and rotated IN
// PLACE, so the op takes no separate outputs. See the recipe in recipes.h.
using AttnQkNormRopeFn =
    void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const RmsNormArgs&, const RopeArgs&);
using SharedExpertGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// MODEL-DSV4-EXL3 W2a/W2b. Both take their optional/small parameters through an
// args struct rather than more Tensor arguments, because `pre_scale`/`post_scale`
// are genuinely OPTIONAL (upstream's `c10::optional<at::Tensor>`,
// hadamard.cuh:5-12) and `bits`/`codebook` are compile-time template parameters
// on the device side that the host launcher dispatches on.
using Exl3HadR128Fn = void (*)(Queue&, Tensor&, const Tensor&, const Exl3HadArgs&);
using Exl3GemmFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&, Tensor&, const Exl3GemmArgs&);
using Exl3MoeMlpFn = void (*)(Queue&, Tensor&, const Tensor&, const Exl3MoeExpertTables&,
                              const Exl3MoeRouting&, const Exl3MoeTemps&, const Exl3MoeArgs&);
// QUANT-EXL3 W6. Same signature as Exl3GemmFn plus two scratch buffers:
//   a_had      fp16 [M, K]  — input Hadamard output (unfused path)
//   w_scratch  fp16 [K, min(N, 32768)] — reconstructed weight (sliced if N > 32768)
// The unfused path writes the GEMM output directly to `c`, then applies the
// output Hadamard in-place (safe: each 128-element block is independent).
using Exl3ReconstructGemmFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, Tensor&, Tensor&,
                                       const Exl3GemmArgs&);
using RmsNormFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);
using ResidualRmsNormFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*a*/,
                                   const Tensor& /*base*/, const Tensor* /*delta*/,
                                   const Tensor& /*weight*/, const ResidualRmsNormArgs&,
                                   Tensor* /*residual_out*/);
// Ungated group RMS norm (vt::RmsNormGroup). Same operand order as RmsNormFn
// minus the residual, which this op does not carry because its upstream has no
// residual arm and a knob nobody can set is a divergence with extra steps.
using RmsNormGroupFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                const Tensor& /*weight*/, const RmsNormGroupArgs&);
using SiluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using ClampedSwiGLUFn = void (*)(Queue&, Tensor&, const Tensor&, float);
using GeluAndMulFn = void (*)(Queue&, Tensor&, const Tensor&);
using MulScalarFn = void (*)(Queue&, Tensor&, const Tensor&, double);
using SoftCapFn = void (*)(Queue&, Tensor&, const Tensor&, double);
using LayerNormFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor*, const Tensor*,
                             const LayerNormArgs&);
using ReluFn = void (*)(Queue&, Tensor&, const Tensor&);
using AddFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using EmbeddingFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using KeepQuantDecodeFn = void (*)(Queue&, Tensor& out, const Tensor& packed);
using RopeFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const RopeArgs&);
using RopeFromCacheFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&,
                                 const Tensor&, const RopeArgs&);
// Fused MLA norm-rope (kFusedNormRope): latent RmsNorm + decoupled-pe
// RopeFromCache over one merged kv_a row, in ONE launch. See vt::FusedNormRope.
using FusedNormRopeFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const RmsNormArgs&,
                                 const RopeArgs&);
using CausalConv1dFwdFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                                   Tensor&, const Tensor&, const Tensor&,
                                   const CausalConv1dArgs&);
using CausalConv1dUpdateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                      const Tensor*, Tensor&, const Tensor*,
                                      const CausalConv1dArgs&);
// Qwen4-Exp PLE dilated depthwise causal conv (vt::Qwen4ExpPleConv). The
// `conv_state_indices` pointer is the PER-SEQUENCE cache row, nullable, named
// for `CausalConv1dUpdate`'s parameter of the same axis; see the declaration
// for why upstream's own `state_idx` is a different axis entirely.
using Qwen4ExpPleConvFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                   const Tensor& /*weight*/, Tensor& /*conv_state*/,
                                   const Tensor& /*query_start_loc*/,
                                   const Tensor* /*conv_state_indices*/,
                                   const Qwen4ExpPleConvArgs&);
// Qwen4-Exp PLE signed-sqrt gate + broadcast sigmoid scale (vt::Qwen4ExpPleGate).
using Qwen4ExpPleGateFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*score*/,
                                   const Tensor& /*value*/, const Qwen4ExpPleGateArgs&);
using L2NormFn = void (*)(Queue&, Tensor&, const Tensor&, const L2NormArgs&);
using RmsNormGatedFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                const RmsNormGatedArgs&);
using GdnPrefillFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, const Tensor&,
                              const GdnArgs&);
using GdnDecodeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, Tensor&, const Tensor*,
                             const GdnArgs&);
using GdnSpecDecodeFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const GdnArgs&);
using CausalConv1dSpecUpdateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                          const Tensor*, Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const CausalConv1dArgs&);
using GdnPackedDecodeFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, Tensor&, const Tensor&,
             const GdnArgs&);
// Per-k-channel-decay gated-delta recurrence (KDA). Same shape as GdnPrefillFn;
// the ONLY difference is g is [T,Hv,Dk] (per-channel) not [T,Hv] (per-head).
using KdaGatedDeltaRuleFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                     const GdnArgs&);
// KDA CHUNK-PREFILL: the chunked (WY-representation) forward of the SAME
// per-K-channel gated-delta linear attention as KdaGatedDeltaRule, but processing
// the whole prompt in BT=64 chunks through the vendored FLA Triton-AOT cubins
// (vLLM's actual prefill kernels) instead of the token-sequential recurrence.
// Takes the RAW gate projection g_raw + a_log + dt_bias (the gate is fused
// on-device by kda_gate_cumsum), NOT a pre-gated per-channel decay.
using KdaChunkPrefillFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   Tensor&, const Tensor&, const GdnArgs&);
// Mamba2 SSD varlen chunked prefill scan (vt::Mamba2ChunkScan). NOT a shape of
// GdnPrefillFn: the SSD recurrence has no delta-removal term and carries the
// per-chunk decay/state-passing metadata upstream builds in mamba2_attn.py.
using Mamba2ChunkScanFn = void (*)(Queue&, Tensor& /*out*/, Tensor& /*final_states*/,
                                   const Tensor& /*x*/, const Tensor& /*dt*/,
                                   const Tensor& /*A*/, const Tensor& /*B*/,
                                   const Tensor& /*C*/, const Tensor* /*D*/,
                                   const Tensor* /*z*/, const Tensor* /*dt_bias*/,
                                   const Tensor* /*initial_states*/,
                                   const Tensor& /*cu_seqlens*/,
                                   const Tensor& /*cu_chunk_seqlens*/,
                                   const Tensor& /*last_chunk_indices*/,
                                   const Tensor& /*seq_idx*/, const Mamba2Args&);
// Mamba2 single-token selective state update (vt::Mamba2StateUpdate).
using Mamba2StateUpdateFn = void (*)(Queue&, Tensor& /*out*/, Tensor& /*state*/,
                                     const Tensor& /*x*/, const Tensor& /*dt*/,
                                     const Tensor& /*A*/, const Tensor& /*B*/,
                                     const Tensor& /*C*/, const Tensor* /*D*/,
                                     const Tensor* /*z*/, const Tensor* /*dt_bias*/,
                                     const Tensor* /*state_indices*/, const Mamba2Args&);
// Silu-gated group RMS norm (vt::RmsNormGatedGroup). `weight` is nullable:
// upstream skips the parameter entirely when use_rms_norm is False.
using RmsNormGatedGroupFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                     const Tensor& /*gate*/, const Tensor* /*weight*/,
                                     const RmsNormGatedGroupArgs&);
// Qwen4-Exp gated residual (vt::Qwen4ExpGatedResidual). `injection` and
// `block_inject` are BOTH nullable and must be null together: a null pair is
// upstream's `use_combine=False` early return (`block_inject_weight is None`,
// modeling_qwen4_exp.py:966-967), which is the model-level mixer.
using Qwen4ExpGatedResidualFn = void (*)(Queue&, Tensor& /*mixed*/,
                                         Tensor* /*injection*/, const Tensor& /*hyper*/,
                                         const Tensor& /*hc_norm_w*/,
                                         const Tensor& /*mix_down*/,
                                         const Tensor& /*mix_up*/,
                                         const Tensor* /*block_inject*/,
                                         const Qwen4ExpGatedResidualArgs&);
// The rank-1 write-back (vt::Qwen4ExpGatedResidualWriteBack), IN PLACE on the
// stream.
using Qwen4ExpGatedResidualWriteBackFn =
    void (*)(Queue&, Tensor& /*hyper*/, const Tensor& /*block_out*/,
             const Tensor& /*injection*/, const Qwen4ExpGatedResidualArgs&);
// Qwen4-Exp QSA pooled-key compressor (vt::Qwen4ExpQsaCompress): the side
// cache's contents, one state per `compress_ratio` complete RAW keys.
using Qwen4ExpQsaCompressFn = void (*)(Queue&, Tensor& /*block_keys*/,
                                       const Tensor& /*raw_keys*/,
                                       const Tensor& /*k_norm_weight*/,
                                       const Tensor& /*cos*/, const Tensor& /*sin*/,
                                       const Qwen4ExpQsaCompressArgs&);
// Qwen4-Exp QSA gather consumer (vt::Qwen4ExpQsaGatherAttention).
using Qwen4ExpQsaGatherAttentionFn = void (*)(Queue&, Tensor& /*out*/,
                                              const Tensor& /*query*/,
                                              const Tensor& /*key*/,
                                              const Tensor& /*value*/,
                                              const Tensor& /*block_ids*/,
                                              const Tensor& /*kv_lens*/,
                                              const Qwen4ExpQsaAttnArgs&);
// GLM-5.3-Flash k-pool DSA indexer (MODEL-MM-GLM53-FLASH W9c-0, #2415). The
// pooled-candidate selection `vt::DsaIndexerLogits` / `vt::DsaTopkSelect`
// cannot express, because those two score RAW TOKENS. See the OpId comments.
struct Glm5NextKpoolSelectArgs {
  // `config.index_topk` — 2048 on the published checkpoint. The pool budget is
  // `index_topk // index_kpool` and `validate_architecture` enforces that the
  // division is exact (`configuration_glm5_next.py:219-220`).
  int64_t index_topk = 0;
  // `cache_layer.get_seq_length()` (`modular_glm5_next.py:811`). It differs from
  // `kv_len` only on a STATIC cache padded to a maximum length; the query at
  // step `s` sits at `current_length - seq_len + s`, which is what lets a decode
  // step's single query see the whole cached prefix instead of only slot 0.
  int64_t current_length = 0;
  // `self.softmax_scale = self.head_dim ** -0.5` (`modeling_glm5_next.py:765`).
  // The INDEXER head dim, not the MLA one.
  float softmax_scale = 0.0f;
  // `config.index_kpool_always_select_tail`. True on the published checkpoint,
  // and it is what widens the output to `index_topk + index_kpool - 1`.
  bool always_select_tail = true;
};
using Glm5NextKpoolCompressFn = void (*)(Queue&, Tensor& /*pool_keys*/,
                                         Tensor& /*pool_indices*/, Tensor& /*pool_valid*/,
                                         Tensor& /*num_pools*/, const Tensor& /*packed*/,
                                         const Tensor& /*ape*/);
using Glm5NextKpoolSelectFn =
    void (*)(Queue&, Tensor& /*topk_indices*/, Tensor& /*index_scores*/,
             const Tensor& /*q_states*/, const Tensor& /*head_weights*/,
             const Tensor& /*pool_keys*/, const Tensor& /*pool_indices*/,
             const Tensor& /*pool_valid*/, const Tensor& /*num_pools*/,
             const Tensor& /*valid_keys*/, const Tensor& /*q_mask*/,
             const Glm5NextKpoolSelectArgs&);
using GdnStateGatherFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
using GdnStateScatterFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using IndexSelectFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using IndexCopyFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using VHeadPermuteFn =
    void (*)(Queue&, Tensor&, const Tensor&, const VHeadPermuteArgs&);
using MoeRouterTopKFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&,
                                 const MoeRouterTopKArgs&, const Tensor*);
// The trailing float is `routed_scale` — the routed_scaling_factor applied to
// the ROUTED sum before the shared term is added (see vt::MoeCombine).
using MoeCombineFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*, float);
using MoeCombineGateFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&);
using AttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const AttentionArgs&);
using AttentionCrossFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor* /*bias*/, const AttentionCrossArgs&);
// Conformer / FastConformer audio-encoder kernels (spike P1/P2/P3).
using Conv2dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv2dArgs&);
using DepthwiseConv1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                   const Tensor& /*weight*/, const Tensor* /*bias*/,
                                   const DepthwiseConv1dArgs&);
// General 3-D convolution (#1007).
using Conv3dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv3dArgs&);
// BigVGAN / DAC vocoder 1-D convolutions (#672).
using Conv1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/, const Tensor& /*weight*/,
                          const Tensor* /*bias*/, const Conv1dArgs&);
using ConvTranspose1dFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*x*/,
                                   const Tensor& /*weight*/, const Tensor* /*bias*/,
                                   const ConvTranspose1dArgs&);
using AttentionRelPosFn = void (*)(Queue&, Tensor& /*out*/, const Tensor& /*query*/,
                                   const Tensor& /*key*/, const Tensor& /*value*/,
                                   const Tensor& /*rel_key*/, const Tensor* /*bias_u*/,
                                   const Tensor* /*bias_v*/, const Tensor* /*key_mask*/,
                                   const AttentionRelPosArgs&);
using DFlashBlockAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const DFlashBlockAttentionArgs&);
using DFlashPagedBlockAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor&, const Tensor&,
                                             const DFlashPagedBlockAttentionArgs&);
using DFlashGroupedConvFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const DFlashGroupedConvArgs&);
using Dflash2SelectorEdgesFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Dflash2SelectorEdgesArgs&);
using Dflash2PathWalkFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                   const Dflash2PathWalkArgs&);
using TopKValuesIndicesFn = void (*)(Queue&, Tensor&, Tensor&, const Tensor&,
                                     const TopKValuesIndicesArgs&);
// --- The fp8_ds_mla paged K-cache GEOMETRY (KV-DSV4-MULTICACHE W8, #2455) ----
// Upstream's host wrapper `quantize_and_insert_k_cache` writes these four as
// literals in its own body (`cache_utils.py:180-190` @ `5559679229`) and hands
// them to the kernel as `tl.constexpr`s; its gather twin repeats them
// (`:353-358`). They are named ONCE here because the op wrapper validates the
// shapes they imply and the kernels build `Fp8DsMlaLayout` from them, and two
// copies of 448 would be two things to keep in step.
inline constexpr int64_t kFp8DsMlaNopeDim = 448;    // TOKEN_FP8_DIM   (`:180`)
inline constexpr int64_t kFp8DsMlaRopeDim = 64;     // TOKEN_BF16_DIM  (`:181`)
inline constexpr int64_t kFp8DsMlaScaleDim = 8;     // TOKEN_SCALE_DIM (`:182`), 7 real + 1 pad
inline constexpr int64_t kFp8DsMlaQuantBlock = 64;  // QUANT_BLOCK_SIZE(`:183`)
// `input_dim` 512 — the latent width upstream asserts on `k` (`:167-169`).
inline constexpr int64_t kFp8DsMlaInputDim = kFp8DsMlaNopeDim + kFp8DsMlaRopeDim;
// TOKEN_DATA_SIZE = 448 + 64*2 = 576 (`:190`): the rope half is stored bf16.
inline constexpr int64_t kFp8DsMlaTokenDataSize =
    kFp8DsMlaNopeDim + kFp8DsMlaRopeDim * 2;
// 584 = 576 data + 8 scale, the per-token page cost
// (`kv_cache_interface.py:401-403`, "448B NoPE + 128B RoPE + 8B fp8 scale").
inline constexpr int64_t kFp8DsMlaTokenBytes =
    kFp8DsMlaTokenDataSize + kFp8DsMlaScaleDim;

// The fp8_ds_mla paged K-cache READ (W8 slice 3). Mirrors the two non-tensor
// arguments of `dequantize_and_gather_k_cache_triton` (`cache_utils.py:339-389`).
struct DequantAndGatherDsMlaArgs {
  // `cache_block_size` (`:379`): STORAGE rows per block, i.e. the spec's
  // `storage_block_size()` = `block_size / compress_ratio`. It is an argument
  // and not a shape because the page is rank-2 bytes.
  int64_t block_size = 0;
  // `offset` (`:369`): the first output COLUMN this gather writes, so a caller
  // can concatenate several gathers into one scratch row.
  int64_t offset = 0;
};

using ReshapeAndCacheFn = void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                   const Tensor&);
// fp8 KV-cache store (KV-FP8 W1). k_cache/v_cache are 1-byte fp8 (DType::kI8);
// each element is stored as Quantize(hp / k_scale|v_scale). `kind` selects the
// fp8 interpretation (kFp8E4M3 landed; kFp8E5M2 is a later brick).
using ReshapeAndCacheFp8Fn = void (*)(Queue&, const Tensor& /*k*/, const Tensor& /*v*/,
                                      Tensor& /*k_cache*/, Tensor& /*v_cache*/,
                                      const Tensor& /*slot_mapping*/, Fp8KVCacheDataType /*kind*/,
                                      float /*k_scale*/, float /*v_scale*/);
using ConcatAndCacheMlaFn =
    void (*)(Queue&, const Tensor&, const Tensor&, Tensor&, const Tensor&);
// The fp8_ds_mla paged K-cache STORE (`KV-DSV4-MULTICACHE` W8 slice 2, #2455).
// `k` is the compressed latent [num_tokens, 512], `kv_cache` the rank-2 byte
// page [num_blocks, block_bytes] (DType::kI8), `slot_mapping` [num_slots] i64,
// and the trailing int64 is upstream's `block_size` argument — the STORAGE rows
// per block, which a rank-2 page cannot carry in its own shape.
using ConcatAndCacheDsMlaFn =
    void (*)(Queue&, const Tensor&, Tensor&, const Tensor&, int64_t);
using DequantAndGatherDsMlaFn = void (*)(Queue&, Tensor& /*out*/,
                                         const Tensor& /*kv_cache*/,
                                         const Tensor& /*seq_lens*/,
                                         const Tensor* /*gather_lens*/,
                                         const Tensor& /*block_table*/,
                                         const DequantAndGatherDsMlaArgs&);
using MlaDecodeAttentionFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                      const Tensor&, const Tensor&,
                                      const MlaDecodeAttentionArgs&);
using MlaPrefillAttentionFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, const Tensor&,
                                       const MlaPrefillAttentionArgs&);
using DsaIndexerLogitsFn = void (*)(Queue&, Tensor& /*logits*/, const Tensor& /*q*/,
                                    const Tensor& /*k*/, const Tensor& /*weights*/,
                                    const Tensor& /*win_start*/, const Tensor& /*win_end*/,
                                    const DsaIndexerLogitsArgs&);
using DsaTopkSelectFn = void (*)(Queue&, Tensor& /*indices*/, Tensor& /*counts*/,
                                 const Tensor& /*logits*/, const Tensor& /*win_start*/,
                                 const Tensor& /*win_end*/);
using GatherMlaCacheFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor*, int64_t);
using MergeAttnStatesFn = void (*)(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, int64_t);
using PagedAttentionFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&,
                                  const PagedAttentionArgs&);
// --- V1 sampling ops (M1.7 Task 2). See the sampling-op section at the bottom.
using ApplyTemperatureFn = void (*)(Queue&, Tensor&, const Tensor&, bool);
using GreedyArgmaxFn = void (*)(Queue&, Tensor&, const Tensor&);
using ApplyTopKTopPFn = void (*)(Queue&, Tensor&, const Tensor*, const Tensor*);
using ComputeProbsFn = void (*)(Queue&, Tensor&, const Tensor&);
using ComputeLogprobsFn = void (*)(Queue&, Tensor&, const Tensor&);
using RandomSampleFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3).
using GreedyRejectionSampleFn = void (*)(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&);
// --- V1 penalty / mask / builtin-proc ops (M1.7 Task 3). See the section at the
// bottom of this header for the full contracts.
using ApplyPenaltiesFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&);
using ApplyMinPFn = void (*)(Queue&, Tensor&, const Tensor&);
using ApplyLogitBiasFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&);
using ApplyTokenMaskFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&);
using ApplyAllowedTokenIdsFn = void (*)(Queue&, Tensor&, const Tensor&);
// --- Fused declarative recipe (TDR; the portable fusion framework). The
// registered per-backend op is the Tier-1 single-pass INTERPRETER, over the
// canonical (out, x, weight, residual) 4-operand shape it serves (kFusedAddRmsNorm
// and any future all-elementwise recipe). Tier-0 composite is device-agnostic and
// lives in ops.cpp (it walks the recipe dispatching each opcode to the standalone
// vt:: op, so it is byte-exact to the unfused sequence — see FusedChain below).
using FusedChainFn =
    void (*)(Queue&, Tensor&, const Tensor&, const Tensor&, Tensor*, const FusedRecipe&, float);
using DropinProbeFn = void (*)(Queue&, Tensor&, const Tensor&, const DropinProbeArgs&);

// `RegisterOp`, `GetOp`, `OpRegistered` and the acceleration-provider seam they
// now dispatch through are declared in vt/op_provider.h, included at the top of
// this header. Their signatures and semantics are unchanged: RegisterOp still
// installs one kernel for (OpId, DeviceType) — it is simply the priority-0
// "vt-native" provider now, and a SECOND provider (MLX on Metal, cuBLASLt on
// CUDA, llama.cpp on CPU/Vulkan) can coexist deterministically instead of
// silently overwriting it under unspecified static-init order.

template <typename Fn>
void RegisterTypedOp(OpId op, DeviceType device, Fn fn) {
  static_assert(std::is_pointer_v<Fn> && std::is_function_v<std::remove_pointer_t<Fn>>,
                "registered op must be a function pointer");
  RegisterOp(op, device, reinterpret_cast<void*>(fn));
}

template <typename Fn>
Fn GetTypedOp(OpId op, DeviceType device) {
  static_assert(std::is_pointer_v<Fn> && std::is_function_v<std::remove_pointer_t<Fn>>,
                "looked-up op must be a function pointer");
  return reinterpret_cast<Fn>(GetOp(op, device));
}

// Contract: out must not alias any input tensor (RopeNeox is in-place by design).

// out[M,N] = a[M,K] @ b[K,N]; a/b float dtypes (f32/f16/bf16), out f32 or
// bf16, f32 accumulation, all contiguous, same device.
// Optional B/table weight_value_dtype preserves F16 storage while applying the
// resolved BF16/F32 model values on the native ROCm ordinary GEMM/gather path.
// Activations, outputs, ID operands, other providers, and other storage reject it.
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// Test-only ABI probe: a tiny adapter binds Queue/Tensor metadata to a raw
// pointer/shape/stride/semantic-type/workspace/stream launcher. Production
// families migrate independently after this spine is gated.
void DropinProbe(Queue& q, Tensor& out, const Tensor& in,
                 const DropinProbeArgs& args);

// out[M,N] = a[M,K] @ b^T with b [N,K] row-major — the torch Linear weight
// orientation, K contiguous in BOTH operands (the "TN" GEMM). This is the
// layout vLLM's F.linear hits for its bf16 projections (GDN in_proj_qkvz /
// in_proj_ba, qwen3_next.py packed_modules_mapping @ e24d1b24): on GB10 it
// selects the fast `nvjet_sm121_tst_..._TNNN` cuBLASLt kernels (~1.3x the
// per-token rate of our row-major x row-major kMatmul, which cuBLASLt serves
// with slower `NNNN`/sm80-cutlass kernels — measured 2026-07-10, 27B prefill:
// in_proj site ours 2.29 vs vLLM 1.80 us/tok). Same f32 accumulation and
// dtype contract as Matmul: a/b bf16 (or f32), out f32 or bf16, contiguous,
// same device. NOT bit-identical to kMatmul on the transposed weight (the
// cuBLASLt algo — and so the K-reduction split — differs); token-exact gates
// decide call-site adoption.
void MatmulBT(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// --- Compute-in-quant GEMM (QUANT-GGUF-CIQ-GEMM) ----------------------------
// out[M,N] = a[M,K] @ b^T where the WEIGHT `b` is [N,K] row-major kept in its
// native ggml BLOCK encoding (a block `DType`), never expanded to bf16. The
// 1:1 counterpart of llama.cpp's `ggml_compute_forward_mul_mat`
// (ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96): GGUF's on-disk
// [out_features, in_features] row-major order IS ggml's src0 layout and IS
// this `[N,K]` orientation, so keep-quant needs no transpose (block rows
// cannot be transposed without requantizing).
//
// Because `b` is block-typed, its `Tensor.shape` is in ELEMENTS but its bytes
// are `RowSizeBytes(b.dtype, K)` per row; `b.stride` is not meaningful and
// `b` must be block-contiguous. K must be a whole number of blocks.
//
// Dispatch (mirrors ggml, and is why this is a separate OpId rather than a
// dtype branch inside MatmulBT): when the weight type has both a `vec_dot`
// kernel and its `vec_dot_type`'s activation quantizer, the activation is
// quantized once and each output is one integer block-dot. Until those land
// (work rows G2/G3) the CPU kernel runs the GENERIC COMPOSITE fallback —
// decode the weight row to f32 via the traits table's `to_float` and take the
// f32 dot — which is numerically the dequant-to-f32 reference the ported
// MUL_MAT tests measure the quantized path against.
void MatmulBTQuant(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// vt::MatmulBTQuantGrouped — grouped keep-quant GEMM over an expert-index list.
// out[P,N] where out[p,:] = act[p,:] . weight[expert_ids[p]*N .. +N] (the block
// row-slice of the stacked expert weight [E*N,K]). Collapses the DeepSeek-V4 MoE's
// per-expert kMatmulBTQuant matvecs into ONE launch. The CPU provider loops
// kMatmulBTQuant per group ⇒ BYTE-IDENTICAL to the per-expert path; the CUDA
// provider runs one warp-per-(p,n) grouped kernel (fewer launches, higher
// occupancy). act f32/bf16, weight Q8_K-family block-quant, expert_ids i32.
void MatmulBTQuantGrouped(Queue& q, Tensor& out, const Tensor& act,
                          const Tensor& weight, const Tensor& expert_ids);

// vt::MoeGateUpSwiGLUGrouped — SHARED fused routed-MoE gate+up+SwiGLU keep-quant
// epilogue (OpId::kMoeGateUpSwiGLUGrouped). out[P,N] f32 = per (p,j):
//   gate = min(FinalFactor·(gate_w[e,j]·xq),  limit)
//   up   = clamp(FinalFactor·(up_w[e,j]·xq), -limit, limit)
//   out[p,j] = gate·sigmoid(gate)·up          (clamped SwiGLU, α=1, β=0)
// where e = expert_ids[p] and xq is act[p] (or act[0] broadcast when Pa==1) quantized
// to Q8_K ONCE. gate_w/up_w are the stacked [E*N,K] expert towers in the SAME CUDA
// keep-quant dtype. The CUDA provider is one warp-per-(p,j) fused launch (gate/up stay
// in registers). The CPU provider runs the BYTE-EXACT composite: two
// vt::MatmulBTQuantGrouped (gate,up) + the clamped-SwiGLU elementwise — so it is the
// golden the fused kernel is gated against. limit=+inf reduces to plain silu(gate)·up
// (a standard SwiGLU MLP with no clamp). Promoted from DeepSeek-V4's private
// MoeDeviceKernels::moe_gate_up_swiglu so any keep-quant MoE arch inherits it.
void MoeGateUpSwiGLUGrouped(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                            const Tensor& up_w, const Tensor& expert_ids, float limit);

// --- Batched dense GEMM (MLA campaign W6) -----------------------------------
// out[G,M,N] = a[G,M,K] @ b[G,K,N] — one independent row-major GEMM per batch
// entry. The 1:1 counterpart of `torch.bmm`, which is the primitive MLA WEIGHT
// ABSORPTION is expressed in upstream:
//   * vllm/model_executor/layers/attention/mla_attention.py:789
//       torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
//     — (N,B,P) x (N,P,L) -> (N,B,L), folding W_UK into the decode QUERY so the
//     MQA runs directly against the 576-wide cached latent;
//   * mla_attention.py:1034 (`_v_up_proj`)
//       torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
//     — (N,B,L) x (N,L,V) -> (N,B,V), un-projecting the latent-space attention
//     output back to `v_head_dim`.
// Upstream's absorption is therefore a LOAD-TIME weight transform plus these two
// batched GEMMs — not a fused kernel (the spike's §2.2 finding), which is why a
// portable port needs exactly this one new primitive.
//
// WHAT ACTUALLY RUNS UPSTREAM, per the whole-chain rule: `torch.bmm` on a CUDA
// bf16 tensor dispatches to ATen's `baddbmm_out_cuda_impl`, which for a
// non-broadcast, equal-strided batch calls cuBLAS
// `gemmStridedBatchedEx` (CUDA_R_16BF operands, CUBLAS_COMPUTE_32F). Our CUDA
// impl is the cuBLASLt strided-batched form of exactly that GEMM, reusing the
// same handle/workspace as vt::Matmul; the CPU impl is the sequential-over-K f32
// reference. No flashinfer / cutlass / TRT-LLM variant participates: the
// ROCm-only aiter fp8/fp4 bmm branches (`:766-776`, `:1024-1032`) are the only
// alternatives upstream has and they are unreachable on CUDA.
//
// STRIDE-DRIVEN on every operand. Both upstream call sites pass TRANSPOSED VIEWS
// (`mqa_q_nope = q_nope.transpose(0,1)`, `out.transpose(0,1)`), i.e. tensors
// whose batch axis is NOT the outermost storage axis, so a contiguity-only op
// would force two extra copies per layer per step. Only the innermost dimension
// must be unit-stride; `stride[0]` (batch) and `stride[1]` (row) are free.
//
// a/b share f32 or bf16; out is f32 or bf16; accumulation is f32. G/M/N may be 0
// (no-op); K == 0 zero-fills, mirroring an empty contraction.
void BatchedMatmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// --- MLA nope|rope head concatenation (MLA campaign W6) ---------------------
//   out[t, h, 0 : Dn)       = nope[t, h, :]
//   out[t, h, Dn : Dn + Dr) = rope[t, (Dr broadcast head), :]
//
// MLA has TWO places that build a head by concatenating a "nope" part with the
// decoupled rope part, and upstream implements both as a pre-allocated output
// plus direct copies rather than a `torch.cat` over an expanded non-contiguous
// tensor (which is why this is an op and not a view):
//
//   1. DECODE q — `torch.cat([q_nope, q_pe], dim=-1)`
//      (vllm/v1/attention/backends/mla/triton_mla.py:200-201) building the
//      576-wide MQA query out of the ABSORBED `ql_nope` (512, the transposed
//      output of the W_UK bmm — NON-CONTIGUOUS) and `q_pe` (64). vLLM ships a
//      dedicated csrc kernel for exactly this, `concat_mla_q`
//      (csrc/libtorch_stable/concat_mla_q.cuh `ConcatMLAQKernel`, host wrapper
//      csrc/libtorch_stable/cache_kernels.cu:1555-1600, bound at
//      torch_bindings.cpp:841,905 and reached from _custom_ops.py:2696-2708);
//      it is stride-driven on the token and head axes precisely so the
//      transposed bmm output needs no `.contiguous()`.
//   2. PREFILL k — `_concat_k_nope_k_pe` (mla_attention.py:2063-2092),
//      concatenating the materialized per-head `k_nope` (qk_nope_head_dim) with
//      the SINGLE-head `k_pe` (qk_rope_head_dim) BROADCAST across all heads.
//      Its docstring states the reason verbatim: "avoids the performance penalty
//      of torch.cat with expanded non-contiguous tensors by pre-allocating the
//      output and using direct copies".
//
// This op is the single generalization of both: stride-driven on the token and
// head axes of every operand, the nope/rope widths taken from the SHAPES rather
// than a compile-time template, and `rope.shape[1] == 1` with `out.shape[1] > 1`
// meaning the head-BROADCAST form case 2 needs. Only the innermost dimension
// must be unit-stride, exactly as upstream asserts
// (cache_kernels.cu:1572-1577). All three tensors share one f32/bf16/f16 dtype.
//
// DEVIATION (recorded, same shape as W5's MergeAttnStates note): upstream's
// kernel is 128/256-bit vectorized and instantiated only for NOPE_DIM=512 /
// rope 64 (`concat_mla_q.cuh:13,21-24,50-53`). Ours is SCALAR and width-generic
// so it serves the prefill K concat (nope 128) as well; the arithmetic is a pure
// copy either way, so the results are identical. Vectorization is a W9 concern.
void ConcatMlaNopeRope(Queue& q, Tensor& out, const Tensor& nope, const Tensor& rope);

// out[M,N] = act[M,K] @ dequant(w).T  — the modelopt W4A16_NVFP4 dequant-GEMM
// (M2.2a). The NVFP4 weight is read DIRECTLY from device memory and dequantized
// on the fly in the kernel (no host bf16 weight materialization); it is the
// drop-in equivalent of Matmul(act, DequantNvfp4ToBf16(w).T) but with the fp4
// weight kept resident on-device.
//
// The weight is a torch Linear weight [N=out_features, K=in_features] in the
// modelopt W4A16_NVFP4 layout (identical decode to
// vllm::DequantNvfp4ToBf16 — the authoritative reference):
//   weight_packed [N, K/2]  i8 bytes: two 4-bit E2M1 (fp4) codes per byte,
//                           low nibble = input elem 2i, high nibble = 2i+1;
//                           nibble bit 3 is the sign, bits 0..2 index the
//                           E2M1 magnitude LUT {0,.5,1,1.5,2,3,4,6}.
//   weight_scale  [N, K/16] i8 bytes: one IEEE fp8-e4m3fn scale per 16-elem
//                           input group (LINEAR layout, multiply not reciprocal).
//   weight_scale_2          per-tensor f32 global scale (amax/2688), multiplied.
// Group scale = f32(weight_scale[n, k/16]) * weight_scale_2 (f32), then the
// dequanted weight is ROUNDED TO BF16 before the multiply — bit-for-bit the
// value DequantNvfp4ToBf16 stores — so the two paths differ only in K-reduction
// order (matmul tolerance), not in the per-element product. These are IEEE
// fp8-e4m3fn scales: the GGUF killgate fork's UE4M3 x0.5 LUT trap does NOT apply.
//
// act [M,K] f32/bf16, out [M,N] f32/bf16, f32 accumulation. K must be a
// multiple of 16. CUDA only (no CPU kernel registered — the CPU reference path
// is DequantNvfp4ToBf16 + Matmul).
void MatmulNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                 const Tensor& weight_scale, float weight_scale_2);

// --- TRUE W4A4 (fp4 activations x fp4 weights) — the 27B path (notes §7). Mirror
// of vllm's dynamic activation quant + cutlass_scaled_fp4_mm_sm120a.
//
// ScaledFp4Quant (mirror vllm scaled_fp4_quant): dynamically per-token, per-16-
// group quantizes a bf16/f32 activation [M,K] to fp4, emitting the two streams
// the fp4xfp4 GEMM consumes:
//   out_packed [M, K/2]  i8  low-nibble-first E2M1 (a_fp4)
//   out_scale  linear [M,K/16], or CUTLASS-swizzled
//              [round_up(M,128),round_up(K/16,4)] i8 fp8-e4m3fn block scales
//              (a_scale_fp8, RAW — the GEMM folds 1/input_global_scale into
//              `alpha`). The swizzled operation initializes every padded byte
//              to zero; backends may do that in the producer body or as a
//              capture-safe pre-zero immediately before it.
// `input_global_scale_inv` is the ON-DISK activation divisor (2688/amax_act) used
// DIRECTLY. K a multiple of 16. Math = vllm cvt_warp_fp16_to_fp4 (notes §7.2) /
// the CPU vllm::RefScaledFp4Quant. CPU + CUDA.
void ScaledFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                    float input_global_scale_inv,
                    Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SiluMulFp4Quant (mirror vllm ActivationQuantFusionPass / silu_and_mul_nvfp4_quant):
// FUSES silu(gate)*up with the NVFP4 activation quant into one kernel, removing the
// bf16 intermediate that the unfused MoeSiluMul(->bf16 [M,I]) + ScaledFp4Quant path
// writes+reads (a memory-traffic win on the prefill). gate/up are [M,I] (our
// two-input MoeSiluMul form). Outputs match ScaledFp4Quant's selected linear or
// CUTLASS-swizzled scale-layout contract:
//   out_packed [M, I/2] i8
// BIT-IDENTICAL to MoeSiluMul(gate,up -> bf16) then ScaledFp4Quant(bf16): the
// silu*up value is rounded through bf16 before quant. I a multiple of 16. CPU+CUDA
// (CPU fallback = the composite sequence).
void SiluMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                     const Tensor& up, float input_global_scale_inv,
                     Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SiluAndMulFp4Quant is the exact one-input vLLM custom-op form. `gate_up` is
// contiguous [M,2I], with gate then up along the inner dimension. It fuses the
// BF16 SiluAndMul rounding boundary with ScaledFp4Quant and emits the same
// packed/scale streams as SiluMulFp4Quant, without materializing [M,I]. CPU +
// CUDA; FP16 is tracked as the separate NVFP4 W4 breadth leaf.
void SiluAndMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                        const Tensor& gate_up, float input_global_scale_inv,
                        Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// SigmoidGateFp4Quant (mirror vllm Inductor triton_poi_fused_mul_scaled_fp4_quant
// _sigmoid_view): FUSES the full-attention sigmoid output gate — attn*sigmoid(gate)
// — with the NVFP4 activation quant of the o_proj into one kernel, removing the
// bf16 intermediate that the unfused SigmoidGateBf16(->bf16 [M,K]) + ScaledFp4Quant
// path writes+reads. attn is [M,K] f32 OR bf16 (the FA-2 prefill hands bf16; the
// upcast is exact), gate is [M,K] f32 (sigmoid input must not be rounded). Outputs
// match ScaledFp4Quant's selected linear or CUTLASS-swizzled scale layout:
//   out_packed [M, K/2] i8 ; out_scale linear [M, K/16] or swizzled.
// BIT-IDENTICAL to SigmoidGateBf16(attn,gate -> bf16) then ScaledFp4Quant(bf16):
// the attn*sigmoid(gate) value is rounded through bf16 before quant. K a multiple
// of 16. CPU+CUDA (CPU fallback = the composite sequence).
void SigmoidGateFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                         const Tensor& attn, const Tensor& gate,
                         float input_global_scale_inv,
                         Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// MatmulNvfp4Fp4 (mirror vllm cutlass_scaled_fp4_mm / ..._sm120a; notes §7.3):
//   out[m,n] = alpha * Σ_k ( a_fp4[m,k]·a_scale_fp8[m,k/16] )
//                            · ( b_fp4[n,k]·b_scale_fp8[n,k/16] )
// a_packed [M,K/2] / a_scale [M,K/16] are ScaledFp4Quant's outputs; b_packed
// [N,K/2] / b_scale [N,K/16] are the on-disk weight_packed / weight_scale (RAW
// fp8, LINEAR — no swizzle: we own producer and consumer). `alpha` folds both
// on-disk globals: alpha = (1/input_divisor)·(1/weight_divisor). f32 accumulation;
// out [M,N] f32/bf16. This is the exact vllm::RunNvfp4Emulation numeric result.
// K a multiple of 16. CPU + CUDA.
void MatmulNvfp4Fp4(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                    const Tensor& b_packed, const Tensor& b_scale, float alpha);

// SwizzleBlockscale (mirror vllm swizzle_blockscale, nvfp4_utils.py:13-53): pad a
// LINEAR fp8-e4m3 block-scale [rows, groups] to [round_up(rows,128),
// round_up(groups,4)] and block-interleave into the atom layout the cutlass
// sm120a fp4 GEMM reads (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}). Both
// tensors i8 (raw fp8 bytes). Used once per weight (B_sf, at load) and per step
// (A_sf, after ScaledFp4Quant). CUDA (+ CPU reference).
void SwizzleBlockscale(Queue& q, Tensor& out_swizzled, const Tensor& in_linear);

// MatmulNvfp4Cutlass (lift of vllm cutlass_scaled_fp4_mm_sm120a — the near-peak
// Blackwell block-scaled fp4xfp4 GEMM). Same numeric contract as MatmulNvfp4Fp4
// but the two fp8 block-scale streams MUST be pre-swizzled (SwizzleBlockscale):
//   a_sf_sw [round_up(M,128), round_up(K/16,4)], b_sf_sw [round_up(N,128), ...].
// a_packed [M,K/2], b_packed [N,K/2] raw fp4 (e2m1x2). alpha = (1/input_divisor)
// ·(1/weight_divisor). The tensor overload mirrors vLLM/FlashInfer: alpha is a
// resident contiguous CUDA f32 scalar and its pointer passes straight into the
// CUTLASS epilogue. The float overload retains host-scalar staging as an exact
// compatibility/diagnostic path. out [M,N] bf16. CUDA-only (sm120a). K,N % 32 == 0.
void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, const Tensor& alpha);
void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_sf_sw,
                        const Tensor& b_packed, const Tensor& b_sf_sw, float alpha);

// --- Per-tensor FP8 W8A8 (the 35B attn q/k/v/o + GDN in_proj_qkv/z/out_proj).
// Mirror of vLLM's static-scale fp8 linear: static per-tensor activation quant +
// cutlass_scaled_mm_sm120_fp8. The checkpoint stores weight F8_E4M3 + a f32
// per-tensor weight_scale + a f32 per-tensor input_scale (both applied directly:
// dequant(w)=f8(w)*weight_scale, dequant(a)=f8(a)*input_scale).

// QuantFp8Static (mirror vLLM static_scaled_fp8_quant):
//   inv = 1/input_scale;  out_fp8[i] = fp8_e4m3( clamp(x[i] * inv, -448, 448) )
// RNE convert. The scale is applied as a RECIPROCAL MULTIPLY, not a divide, and
// the reciprocal is formed ONCE outside the elementwise loop — that is what
// upstream ships: `x = val * scale` under is_scale_inverted=true
// (csrc/quantization/w8a8/fp8/common.cuh:62, clamp at :68) with the inverse formed
// by the caller (csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31,
// `1.0f / scale[...]`). DO NOT "correct" the kernels to a divide to match a
// prose formula: `x/s` and `x*(1/s)` differ by up to one f32 ulp, and near an
// e4m3 tie that ulp changes the emitted byte on a default-ON 35B path.
// Static per-tensor scale (NOT dynamic/per-token). x [M,K] f32/bf16, out [M,K]
// i8 (raw fp8-e4m3fn bytes). CUDA + CPU (the CPU arm is the portable reference
// that makes the fp8 seam testable without a GPU, #468).
void QuantFp8Static(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale);

// --- Block-wise FP8 (VT-QUANT-FP8-GROUP, #1189 M1,
// .agents/specs/vt-quant-fp8-group.md). QuantFp8Group is the DYNAMIC per-token,
// per-group sibling of QuantFp8Static: the scale is derived from the data, once
// per contiguous run of `group_size` elements inside a row, and written out.
//
//   amax = max(1e-10, max |x_f32| over the group)
//   y_s  = amax / 448.0f
//   out_fp8[i] = fp8_e4m3( min(max(x_f32[i] / y_s, -448.0f), 448.0f) )
//   out_scale[row, group] = y_s
//
// x [M,K] f32/bf16, out_fp8 [M,K] i8 (raw fp8-e4m3fn bytes), out_scale
// [M, K/group_size] F32 — f32 because upstream allocates it f32 and the GEMM
// that consumes it multiplies in f32 (fp8_utils.py:629-631). K must be a
// multiple of group_size; the op refuses any other K by name, as upstream
// asserts at fp8_utils.py:596-599. CPU + CUDA.
//
// WHICH UPSTREAM ARM THIS MIRRORS, because there are two and they DISAGREE.
// `per_token_group_quant_fp8` reads like a Triton kernel with a C++ fast path
// and it is the other way round: on a CUDA-alike platform with a contiguous
// input it calls the C++ custom op and RETURNS (fp8_utils.py:635-650), so the
// Triton kernel never executes there. The executing kernel is
// csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:
//   :47  float local_absmax = eps           — eps SEEDS the reduction, so an
//                                             all-zero group cannot divide by 0
//   :53  fmaxf(local_absmax, fabsf((float)src))
//   :68  float y_s = local_absmax / max_8bit                    — a DIVIDE
//   :85  fminf(fmaxf((float)src / y_s, min_8bit), max_8bit)     — a DIVIDE
//   :86  DST_DTYPE(q)                       — hardware e4m3 RNE, saturating
// DO NOT "correct" either divide into a hoisted reciprocal multiply to match
// QuantFp8Static's form. That form is right for QuantFp8Static because upstream
// ships it there (common.cuh:62 with the reciprocal formed by the caller); here
// upstream ships a divide, the scale changes per group so nothing is
// loop-invariant, and the Triton fallback's `_absmax * (1.0 / fp8_max)`
// (fp8_utils.py:145) carries an upstream comment naming the 1-ULP gap it opens.
// One f32 ulp before an e4m3 round changes the emitted byte near a tie, and
// upstream's own test cannot see it: it compares values at rtol=0.15 and the
// scale at rtol=1e-5 (test_block_fp8.py:112-115). Only a byte comparison can,
// which is what tests/vt/test_ops_quant_fp8_group_cpu.cpp G1 is.
//
// The column-major and TMA-aligned scale layouts (fp8_utils.py:610-628) and the
// `use_ue8m0` DeepGEMM scale rounding (per_token_group_quant.cu:69-71) are NOT
// implemented. Both are recorded under `## Owed` in the row's spec; no consumer
// in this tree can read either yet, and upstream excludes the target model from
// DeepGEMM on family 120 (vllm/utils/deep_gemm.py:27-46).
void QuantFp8Group(Queue& q, Tensor& out_fp8, Tensor& out_scale, const Tensor& x,
                   int group_size);

// MatmulFp8BlockScaled (VT-MATMUL-FP8-BLOCK-REF, #1189 M2,
// .agents/specs/vt-matmul-fp8-block-ref.md) — the 128x128 block-scaled fp8 GEMM,
// mirroring native_w8a8_block_matmul (tests/kernels/quant_utils.py:91-154):
//
//   for each (m, n):
//     acc = 0                                              f32
//     for kt in [0, cdiv(K, block_k)):
//       part = 0                                           f32, a SEPARATE register
//       for k in the k-tile:  part += f8(a[m,k]) * f8(b[n,k])
//       acc += part * ( a_scale[m, kt] * b_scale[n / block_n, kt] )
//     out[m, n] = acc                                      stored to out's dtype
//
// THE SCALES APPLY IN THE MAINLOOP, ONCE PER K-BLOCK, INTO AN F32 ACCUMULATOR —
// NOT IN THE EPILOGUE, and that is a correctness constraint rather than an
// optimisation choice. MatmulFp8Cutlass above folds one scalar alpha after the
// whole K reduction. An epilogue has exactly ONE degree of freedom per output
// element; this scheme has cdiv(K, block_k) of them. An epilogue-only
// application therefore cannot express a per-K-block scale AT ALL, which is why
// this is a separate op. DO NOT "simplify" `part` away into `acc`: that IS the
// epilogue form, and tests/vt/test_ops_matmul_fp8_block_cpu.cpp G4 is built so
// that no single-alpha implementation can pass it.
//
// WHICH UPSTREAM ARM THIS MIRRORS. The Triton kernel at fp8_utils.py:826-836 is
// not what executes on the target architecture; CUTLASS is
// (vllm/model_executor/kernels/linear/__init__.py:355-377 ranks it third,
// DeepGEMM is auto-disabled for qwen3_5_text on family 120 at
// vllm/utils/deep_gemm.py:27-46, and Marlin is excluded at cc >= 89). Unlike the
// QuantFp8Group case above, the two AGREE: csrc/.../c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56-58,218-235 hands both scale
// pointers to the MAINLOOP arguments over an `ElementAccumulator = float`, and
// cutlass 4.5.0's sm120_mma_tma_blockwise_scaling.hpp:714-717 is literally
// `accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i)`.
// CUTLASS associates the two scale multiplies left to right where the reference
// forms their product first (quant_utils.py:150-151); the difference is at most
// one f32 ULP per K-block and upstream's own gate admits it, comparing the two
// at rel_diff < 0.001 (test_block_fp8.py:194-200). We mirror the reference's
// association, because this op IS the reference port and the CUDA kernel that
// #1189 milestone M5 lands will be measured against it.
//
// SHAPES, with CEIL on every tiling, so a ragged final block is legal and must
// work (upstream asserts exactly this at fp8_utils.py:935-936):
//   a_fp8   [M,K]                              i8, raw fp8-e4m3fn bytes
//   a_scale [M, cdiv(K, block_k)]              F32
//   b_fp8   [N,K]                              i8, raw fp8-e4m3fn bytes
//   b_scale [cdiv(N, block_n), cdiv(K, block_k)] F32
//   out     [M,N]                              f32 or bf16
// The scales are f32 because upstream refuses any other dtype on this path
// (csrc/.../c3x/scaled_mm_helper.hpp:15-18) and the accumulator is f32.
// a_scale's K axis is a CEIL too, so this op accepts a K that vt::QuantFp8Group
// would refuse; that asymmetry is upstream's own (fp8_utils.py:930 uses cdiv
// where fp8_utils.py:596-599 demands divisibility).
//
// No bias: upstream refuses one outright on the blockwise path
// (scaled_mm_helper.hpp:54), so this mirrors a refusal rather than deferring a
// feature.
//
// CPU + CUDA, and the CUDA arm is NARROW. This line read "CPU only" until W9d
// (#2881) and had been stale since #1189 milestone M5 (`489a9a4c0`) landed
// `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu`. That TU is compiled only for
// the `cutlass-fp8` arch cell `12.0a,12.1a` (cmake/CudaArchFeatures.cmake), so
// its registrar runs only there and the op is genuinely UNREGISTERED on every
// other CUDA arch — sm_110 and sm_87 among them. `dense_fp8_block::
// BlockFp8Runnable` is the runtime question and `RefuseUnrunnableFp8BlockWeight`
// is the message; neither reads this comment, which is why the comment could
// drift for as long as it did.
//
// The CPU arm is A CORRECTNESS REFERENCE, NOT A PERFORMANCE PATH — it is the
// numerical oracle the CUTLASS kernel is measured against, and it makes no
// speed claim. That kernel has still never executed on hardware
// (.agents/specs/vt-matmul-fp8-block-cuda.md).
void MatmulFp8BlockScaled(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& a_scale,
                          const Tensor& b_fp8, const Tensor& b_scale, int block_n,
                          int block_k);

// RmsNormQuantFp8 (fused fp8 RMSNorm -> static per-tensor activation quant). One
// HBM pass mirrors vLLM's Inductor `fused_add_rms_norm_static_fp8_quant`
// (vllm/compilation/passes/fusion/rms_quant_fusion.py:124) — the RMSNorm producer
// that directly emits the fp8 activation so the standalone QuantFp8Static pass (and
// its bf16 round-trip) disappears. The activation is quantized ONCE and shared by
// every projection that reads it (the fp8 analog of the fp4 quantize-once), so
// callers feed the single fp8 to all shared GEMMs (see MatmulFp8CutlassPreQuantD).
//   res += x  (rounded to res dtype, as fused_add_rms_norm);
//   n = rmsnorm(res, weight)  (gemma -> weight applied as 1+w);
//   b = bf16(n);  out_fp8[i] = fp8_e4m3( b * (1/input_scale) )   // RNE hw cvt
// BIT-IDENTICAL to RmsNorm(bf16 out, x, weight, {eps,gemma}, res) followed by
// QuantFp8Static(out_fp8, that bf16, input_scale) — the bf16-intermediate form the
// current path already rounds through (the RMSNorm output IS bf16 before the quant).
// out_fp8 [T,H] i8 (raw fp8-e4m3fn bytes). out_bf16 optional [T,H] bf16 (the normed
// activation, emitted only when a bf16 consumer of it coexists at the site — e.g.
// the GDN in_proj_a/b; nullptr for full-attn q/k/v where nothing reads it). x [T,H]
// / weight [H] float; residual optional [T,H] f32/bf16 (in/out). CUDA + CPU.
void RmsNormQuantFp8(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                     const Tensor& weight, const RmsNormArgs& args, Tensor* residual,
                     float input_scale);

// RmsNormGatedQuantFp8 (fused gated-RMSNorm -> static per-tensor activation quant).
// The gated-norm analog of RmsNormQuantFp8: the GDN gated-RMSNorm producer emits the
// fp8 activation DIRECTLY, so the standalone QuantFp8Static pass (and the bf16
// round-trip that the gated norm otherwise writes then the quant re-reads) disappear.
// Mirrors vLLM's Inductor fusion of the gated-RMSNorm epilogue with the fp8 activation
// quant of the following RowParallelLinear (fla layernorm_guard.py RMSNormGated ->
// out_proj W8A8), the gated sibling of rms_quant_fusion.py's static-fp8 fusion.
//   var = mean(x^2 over last dim);  n = x * (1/sqrt(var+eps)) * w * act(z);
//   b = bf16(n);  out_fp8[i] = fp8_e4m3( b * (1/input_scale) )   // RNE hw cvt
// BIT-IDENTICAL to RmsNormGated(bf16 out, x, gate, weight, args) followed by
// QuantFp8Static(out_fp8, that bf16, input_scale): the fp8 is taken from the SAME
// bf16-rounded value the split path already rounds through, and the variance reduction
// ORDER is the shipped RmsNormGated order (fast d==128 path bit-identical to shipped).
// out_fp8 same shape as x (rank-2 [rows,D] or rank-3 [T,Hv,D]) i8 (raw fp8-e4m3fn
// bytes). x/weight float; gate may carry a padded outer (token) stride. CUDA + CPU.
void RmsNormGatedQuantFp8(Queue& q, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                          const Tensor& weight, const RmsNormGatedArgs& args, float input_scale);

// MatmulFp8Cutlass (lift of vLLM cutlass_scaled_mm_sm120_fp8 — the per-tensor
// W8A8 fp8 GEMM vLLM selects on sm120/GB10). Same math as vLLM's ScaledEpilogue
//   out[M,N] = scale_a * (scale_b * (A_fp8 @ B_fp8^T))
// but the two PER-TENSOR static scales are folded into one scalar
//   alpha = input_scale * weight_scale
// (identical for per-tensor scalars; a single fused f32 multiply vs vLLM's
// sequential scale_a·(scale_b·acc) — within fp8 tolerance, ported deviation).
// a_fp8 [M,K] (= QuantFp8Static output), b_fp8 [N,K] the on-disk raw fp8-e4m3fn
// weight (K contiguous). out [M,N] bf16 (cutlass epilogue) or f32 (via cast).
// K,N multiples of 16 (128-bit fp8 alignment). CUDA (sm120a) + a CPU CORRECTNESS
// REFERENCE (f32 accumulate, naive triple loop — no speed claim, and no
// production model routes through it; it exists so the fp8 seam resolves on a
// CPU queue and can be gated without a GPU, #468). The CPU arm is EXPECTED to
// agree with the CUDA kernel to fp8/bf16 tolerance and NOT byte-for-byte, because
// the CUDA arm reduces K in tensor-core order and rounds its epilogue through
// bf16 — but that agreement is DECLARED AND OWED, not measured. No committed run
// has compared the two arms: gate G2 of .agents/specs/vt-fp8-w8a8-cpu-arm.md is
// PENDING for want of a GPU. Treat the tolerance above as the claim to be tested,
// not as a result.
void MatmulFp8Cutlass(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                      float alpha);

// MatmulFp8CublasLt — cuBLASLt FP8 (e4m3) dense GEMM, the native equivalent of
// vLLM's cuBLASLt fp8 path (the `nvjet_sm121_qqtst_*` / `qq*` kernels that
// torch._scaled_mm / cublasLt select for the 35B's fp8 dense projections on
// GB10/sm_121a). Same host surface + same math as MatmulFp8Cutlass — both fold
// the two per-tensor static scales into one f32 alpha (= input_scale*weight_
// scale) applied to the fp32 accumulator:
//   out[M,N] = alpha * (A_fp8[M,K] @ B_fp8[N,K]^T)
// but routed through cublasLtMatmul (CUBLAS_COMPUTE_32F, e4m3 A/B, f32 scale)
// instead of the cutlass sm120 kernel. cuBLASLt fp8 requires the TN layout
// (contraction K contiguous for both operands) — a_fp8 [M,K] and b_fp8 [N,K]
// row-major already satisfy it. a_fp8 [M,K] (= QuantFp8Static output), b_fp8
// [N,K] the on-disk raw fp8-e4m3fn weight (K contiguous). out [M,N] bf16 or f32
// (cublasLt writes the requested output type directly). K,N multiples of 16.
// CUDA-only. Falls back to the cutlass fp8 GEMM if cublasLt has no fp8 heuristic
// for a given shape (keeps the correctness gate robust on odd M).
//
// `claims_splitk1_premise` (default FALSE) is how a caller opts INTO a stricter
// contract than the op otherwise offers. Pass it only when this GEMM's bf16 `out`
// is asserted to be byte-equivalent to the SAME GEMM's f32 `out` — a pure
// store-width narrowing over one ordered f32 reduction. That claim requires the
// selected cuBLASLt plan to run at splitK=1 (a split-K sums per-split partials
// in an order the f32 arm never used, and f32 addition is not associative), so
// the implementation verifies it and REFUSES otherwise.
//
// It is FALSE by default because that claim is unusual. An ordinary bf16 `out`
// — what every `o_proj` / `out_proj` fp8 projection asks for, on a default-ON
// path — is just an output dtype: split-K is correct for it, it is compared
// against nothing, and it is never checked. Do not set this flag merely because
// `out` is bf16; set it when you are asserting equivalence with an f32 arm.
void MatmulFp8CublasLt(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                       float alpha, bool claims_splitk1_premise = false);

// MatmulFp8CublasLtAlphaVec — the SAME fp8 GEMM with a per-output-COLUMN alpha:
//   out[m,n] = alpha_vec[n] * (A_fp8[M,K] @ B_fp8[N,K]^T)[m,n]
// `alpha_vec` is f32 [N], contiguous, on the queue device; `out` is f32 OR bf16
// [M,N]. This is the form an N-CONCATENATED operand needs when its shards carry
// different folded alphas (input_scale * that shard's weight_scale), which no
// single host scalar can express. Mirrors the tensor-alpha overload the NVFP4
// CUTLASS path already took (.agents/specs/nvfp4-device-alpha.md).
//
// The op is TOTAL: when VT_FP8_ALPHA_VEC_EPILOGUE=1 AND the heuristic returns an
// algo whose CUBLASLT_ALGO_CAP_POINTER_MODE_MASK advertises
// ALPHA_DEVICE_VECTOR_BETA_ZERO, the alpha is applied in the cuBLASLt epilogue
// (one launch). Otherwise it runs the GEMM at alpha=1 and applies the vector
// with vt::MulColVecF32 — the two-launch form this seam shipped with, byte for
// byte. Callers therefore never branch on the toggle or the driver's capability;
// they express the per-column alpha ONCE, here. CUDA-only.
//
// A bf16 `out` (PERF-FP8-ALPHA-FOLD / #417) is what vLLM emits for this
// projection (ModelOptFp8LinearMethod's out_dtype is the model dtype,
// modelopt.py:458 @ the pin). It halves the bytes the column pass moves — the
// dominant cost, since that pass is a full read-modify-write measured at 77% of
// the device's peak bandwidth. It is NOT value-neutral: the GEMM's f32
// accumulator is rounded to bf16 before the alpha multiply instead of after it,
// so callers opt in rather than defaulting to it.
//
// A bf16 `out` always takes the TWO-LAUNCH arm regardless of the toggle. At bf16
// the epilogue would round once where the fallback rounds twice, so admitting it
// would make VT_FP8_ALPHA_VEC_EPILOGUE change VALUES rather than just speed; the
// toggle is kept a pure performance A/B at every dtype.
//
// `claims_splitk1_premise` carries exactly the meaning it has on
// MatmulFp8CublasLt above, and reaches the same check: a bf16 `out` always takes
// the two-launch arm, whose GEMM IS MatmulFp8CublasLt.
void MatmulFp8CublasLtAlphaVec(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                               const Tensor& alpha_vec, bool claims_splitk1_premise = false);

// --- Fused MoE grouped NVFP4 GEMM (M2.4). One kernel launch computes the expert
// projection for ALL (token, activated-expert) pairs at once, instead of the
// per-expert loop of tiny MatmulNvfp4 launches (the launch-overhead-bound decode
// bottleneck). For each output row p (a (token,slot) pair):
//   out[p, :] = act[row_map ? row_map[p] : p, :] @ dequant(W[expert_ids[p]]).T
// where W[e] is the modelopt W4A16_NVFP4 weight [N=out, K=in] of expert e (same
// on-the-fly decode as vt::MatmulNvfp4, bit-for-bit vllm::DequantNvfp4ToBf16).
// The E per-expert weights are passed as DEVICE POINTER ARRAYS (the fp4-resident
// buffers, M2.2b) indexed by expert id — no weight gather/copy:
//   out          [P, N] f32/bf16 (P = num (token,expert) pairs)
//   act          [R, K] f32/bf16 (R rows; gate/up read the token hidden [T,H],
//                down reads the per-pair silu output [P,I])
//   expert_ids   [P] i32  (device; the router's top-k indices, [T,top_k] viewed
//                as [P] — pair p = token p/top_k, slot p%top_k)
//   row_map      [P] i32  (device) or nullptr: act row for output row p (nullptr
//                => identity p; gate/up pass token-of-pair = p/top_k)
//   packed_ptrs  [E] i64  (device) each entry = (uintptr_t) of expert e's packed
//                [N,K/2] i8 buffer
//   scale_ptrs   [E] i64  (device) each entry = (uintptr_t) of expert e's scale
//                [N,K/16] i8 buffer
//   scale2s      [E] f32  (device) per-expert weight_scale_2
// K = act.shape[1] (multiple of 16), N = out.shape[1], P = out.shape[0]. f32
// accumulation, per-row bit-identical to the per-expert MatmulNvfp4. CUDA only
// (the CPU MoE path keeps the per-expert dequant+matmul reference).
void MoeGroupedGemmNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                         const Tensor* row_map, const Tensor& packed_ptrs,
                         const Tensor& scale_ptrs, const Tensor& scale2s);

// BF16 grouped-MoE GEMM (the fast bf16 MoE path, Qwen3-Coder Qwen3MoeForCausalLM).
// out[p, n] = sum_k act[row(p), k] * W_e[k, n], where e = expert_ids[p], W_e =
// weight_ptrs[e] is a bf16 [K, N] (Matmul-B / loader-transposed) weight, and
// row(p) = row_map ? row_map[p] : p. Structurally identical scheduling to
// MoeGroupedGemmNvfp4 (naive one-thread-per-output for small/decode P; expert-
// counting-sort + bf16 WMMA tensor-core tiles for large/prefill P) — the fp4
// on-the-fly decode is replaced by a direct bf16 weight read. f32 accumulation,
// out f32 (gate/up, matching the reference MatmulF32) or bf16 (down).
// Registered on kCUDA (src/vt/cuda/cuda_matmul_nvfp4.cu:2722) and on kROCM
// (src/vt/rocm/rocm_ops.hip:292); the CPU/GGUF MoE path registers no arm here
// and keeps the per-expert MatmulBf16 reference. This line read "CUDA only"
// until 2026-09-12, by which date the kROCM arm had landed and the
// kMoeGroupedGemmBf16GateUpSilu paragraph in the OpId enum already cited it.
//   act          [*, K] bf16 (row(p) selects the source row)
//   expert_ids   [P] i32 (device) — per-pair expert id (router top-k, viewed [P])
//   row_map      [P] i32 (device) or nullptr — pair p -> source act row
//   weight_ptrs  [E] i64 (device) — each entry = (uintptr_t) of expert e's bf16
//                [K, N] weight buffer
// K = act.shape[1], N = out.shape[1], P = out.shape[0].
void MoeGroupedGemmBf16(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& weight_ptrs);

// vt::MoeGroupedGemmBf16GateUpSilu — SHARED fused BF16 grouped-MoE gate+up+SwiGLU
// (Tier-A4 fold, OpId::kMoeGroupedGemmBf16GateUpSilu). ONE vt entry replaces the
// {MoeGroupedGemmBf16(gate); MoeGroupedGemmBf16(up); MoeSiluMul} triplet the bf16
// grouped-MoE archs (Qwen3-Coder, DeepSeek-V2, kimi) ran:
//   out[p, j] = silu(gate[p, j]) * up[p, j]           (bf16 store)
// where gate[p, j] = sum_k act[row(p), k] * gate_W_e[k, j],
//       up[p, j]   = sum_k act[row(p), k] * up_W_e[k, j],
//       e = expert_ids[p], row(p) = row_map ? row_map[p] : p, and gate_W_e/up_W_e
// are gate_ptrs[e]/up_ptrs[e] (bf16 [K, N] Matmul-B weights). Same argument
// convention as MoeGroupedGemmBf16 with a SECOND weight-pointer array. BIT-IDENTICAL
// to that composite in every launch regime — the decode/non-WMMA path fuses the two
// GEMMs into one grouped launch + a reduce+SwiGLU launch (reusing the exact split-K
// sequential-k accumulation and the MoeSiluMul math); the WMMA path reuses the
// grouped-GEMM dispatch twice + the identical silu-mul. Registered on kCUDA
// (src/vt/cuda/cuda_matmul_nvfp4.cu:2725) and on kROCM
// (src/vt/rocm/rocm_ops.hip:295), exactly like MoeGroupedGemmBf16 above. This
// line read "CUDA only" until 2026-09-12, by which date the kROCM arm had
// landed.
//   out          [P, N] bf16 (the per-(token,slot) silu(gate)*up)
//   act          [*, K] bf16
//   expert_ids   [P] i32 (device)
//   row_map      [P] i32 (device) or nullptr
//   gate_ptrs    [E] i64 (device) — gate expert weight pointers (bf16 [K, N])
//   up_ptrs      [E] i64 (device) — up   expert weight pointers (bf16 [K, N])
void MoeGroupedGemmBf16GateUpSilu(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& gate_ptrs, const Tensor& up_ptrs);

// Native BF16 boundaries from vLLM e126687a9a, triton_moe.py:388-527 and
// activation_kernels.cu:44,165-177: BF16 gate/up, BF16 SiLU, BF16 product.
// Uses the same pointer-array layout and validation as the legacy sibling.
void MoeGroupedGemmBf16GateUpSiluNative(Queue& q, Tensor& out, const Tensor& act,
                                      const Tensor& expert_ids, const Tensor* row_map,
                                      const Tensor& gate_ptrs, const Tensor& up_ptrs);

// Grouped dot with one FP32 route weight per pair, multiplied BEFORE output
// conversion (fused_moe.py:593-610 at the same pin). Output is BF16 or FP32.
void MoeGroupedGemmBf16Weighted(Queue& q, Tensor& out, const Tensor& act,
                               const Tensor& expert_ids, const Tensor* row_map,
                               const Tensor& weight_ptrs, const Tensor& route_weights);

// Sum already weighted BF16 expert_out[T,top_k,H] in FP32, then narrow to the
// BF16/FP32 output. No second route-weight multiplication. Optional shared[T,H]
// is BF16/FP32 and is added after routed_scale, matching the legacy shared term.
void MoeCombinePreweighted(Queue& q, Tensor& out, const Tensor& expert_out,
                           const Tensor* shared = nullptr, float routed_scale = 1.0f);

// Select the complete native sequence through provider availability.
bool MoeGroupedBf16NativeAvailable(DeviceType device);

// MoeGroupedGemmNvfp4Marlin (lift of vLLM moe_wna16_marlin_gemm, ops.cu:543 —
// the Marlin W4A16 kernel vLLM selects for the 35B's NVFP4 MoE experts). One
// launch computes the grouped expert projection over all padded (token,expert)
// blocks. Inputs mirror the NVFP4 branch: b_type=kFE2M1f + s_type=kFE4M3fn,
// group size 16, bf16 activation/output.
//   c            [size_m*top_k, size_n] bf16 (out; the per-(token,slot) result)
//   a            [size_m, size_k]        bf16 (token hidden)
//   b_q_weight   [E, size_k/16, size_n*8/pack] i32 — Marlin-interleaved fp4
//                (from the load-time gptq_marlin_moe_repack)
//   b_scales     [E, size_k/16, size_n]  fp8-e4m3 (processed:
//                marlin_permute_scales + nvfp4_marlin_process_scales)
//   global_scale [E]                     f32 (nvfp4_marlin_process_global_scale)
//   workspace    [>= sms*4 or per-tile]  i32 (zeroed reduction locks)
//   sorted_token_ids / expert_ids / num_tokens_past_padded / topk_weights:
//                the moe_align_block_size outputs (int32 / int32 / int32 / f32).
// CUDA-only (Blackwell sm_12xa; needs the vendored Marlin TUs, VT_MARLIN_NVFP4).
void MoeGroupedGemmNvfp4Marlin(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                               const Tensor& b_scales, const Tensor& global_scale,
                               Tensor& workspace, const Tensor& sorted_token_ids,
                               const Tensor& expert_ids, const Tensor& num_tokens_past_padded,
                               const Tensor& topk_weights, const MoeMarlinArgs& args);

// MarlinDenseGemm (lift of vLLM's DENSE marlin_gemm, marlin.cu:545 -> marlin_mm
// at :326 — the byte-preserving dense W4A16 kernel vLLM itself ships for a16
// weight-only linears). One launch computes y = a @ dequant(b).T with vLLM's own
// direct-A, tile-per-CTA layout and dense fp32-C_tmp reduce — NOT the MoE
// single-expert route, whose par regrouping of the reduce costs one bf16 ULP.
//   c            [size_m, size_n]         bf16 (out)
//   a            [size_m, size_k]         bf16 (token hidden; contiguous, lda=size_k)
//   b_q_weight   [size_k/16, size_n*8/pack] i32 — Marlin-interleaved fp4 (SAME
//                repack as the MoE path: marlin_permute; a shim is added only if
//                a layout divergence is proven — see dense_nvfp4_gemm.h)
//   b_scales     [size_k/group_size, size_n] fp8 (processed marlin scales)
//   global_scale [1]                      f32 (nvfp4 only; ignored for mxfp4)
//   workspace    [>= sms]                 i32 (zeroed reduction locks)
// CUDA-only (Blackwell sm_12xa; needs the vendored dense Marlin TUs, VT_MARLIN_NVFP4).
void MarlinDenseGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                     const Tensor& b_scales, const Tensor& global_scale, Tensor& workspace,
                     const MarlinDenseArgs& args);

// out[R,I] = silu(gate[R,I]) * up[R,I]  (moe-semantics.md §4; the fused-MoE
// element-wise activation between the grouped gate/up and down GEMMs). gate/up
// f32 or bf16, out f32/bf16; silu/mul computed in f32, rounded on store. Unlike
// vt::SiluAndMul (single [T,2D] input), this takes the two separately-produced
// projections so no concat/copy is needed. CPU + CUDA.
void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up);

// out[R,I] = relu(x[R,I])^2 — the NON-GATED MoE activation, and the whole
// epilogue of a NemotronH expert. Mirror of vLLM's `ReLUSquaredActivation`
// (layers/activation.py:609-628, forward_native = torch.square(F.relu(x))) as
// reached through the fused-MoE path: `activation_without_mul("relu2")` ->
// `MoEActivation.RELU2_NO_MUL` -> `apply_moe_activation`'s
// `F.relu(input, inplace=True); torch.square(input, out=output)`
// (layers/fused_moe/activation.py:34 `RELU2_NO_MUL`, :98
// `activation_without_mul`, and the :184 RELU2_NO_MUL branch; `:33` is
// GELU_TANH_NO_MUL, the neighbouring enumerator).
//
// Why this is NOT a MergedGemmGroup epilogue: a NON-gated expert has no gate
// half to merge with (nemotron_h.py:220 `ckpt_names=("up_proj","down_proj","")`
// — the empty third entry IS the absent gate). There is exactly ONE projection,
// so the expert is the EXISTING grouped GEMM plus this activation, exactly as
// the gated bf16 archs are kMoeGroupedGemmBf16 + kMoeSiluMul. See
// merged_gemm.h's note on the non-gated family.
//
// DTYPE/ROUNDING ORDER is the mirrored part, not an implementation detail:
// upstream's kernel (csrc/libtorch_stable/activation_kernels.cu:673-678)
// widens to f32, clamps at zero in f32, squares in f32 and rounds ONCE on the
// store — so a bf16 input with an f32 output keeps the FULL f32 square. x f32
// or bf16, out f32/bf16. CPU + CUDA.
void MoeRelu2(Queue& q, Tensor& out, const Tensor& x);

// out[T,H] = x[T,H] / sqrt(mean(x^2) + eps) * w  (or *(1+w) when gemma);
// out f32 or bf16 (computed in f32, rounded on store).
// With residual != nullptr (f32 OR bf16 [T,H]): residual += x first (new residual
// stream), and that sum is what gets normalized (upstream fused_add_rms_norm).
// The variance/normalize accumulation is always f32; the residual load/store dtype
// follows the tensor. A bf16 residual mirrors vLLM's bf16 model dtype (the add is
// rounded to bf16 before the f32 variance, matching csrc fused_add_rms_norm); a f32
// residual keeps full precision across layers (the byte-identical previous path).
// Note: unlike upstream forward_native, the standard path keeps full f32 precision
// (no x.to(weight.dtype) rounding before the weight multiply); parity tests vs
// upstream bf16 need bf16-eps tolerance on the non-gemma path.
void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual = nullptr);

// Compiled BF16 expression: normalize a+base or delta+(a+base), in FP32,
// multiplying gamma before the single BF16 output conversion. An optional
// residual output stores the same unrounded expression as BF16. It is never
// reloaded for normalization. Inputs/output are rank-2 with unit inner stride
// and nonoverlapping rows. Gamma is contiguous BF16 [H]. Empty rows are legal.
// Only descriptor-permitted exact aliases can modify an input. No persistent
// FP32 activation is needed. Other activation/gamma dtypes are refused.
void ResidualRmsNorm(Queue& q, Tensor& out, const Tensor& a, const Tensor& base,
                     const Tensor* delta, const Tensor& weight,
                     const ResidualRmsNormArgs& args, Tensor* residual_out = nullptr);

// The same typed operation through the shared recipe/composite seam.
void FusedChain(Queue& q, Tensor& out, const Tensor& a, const Tensor& base,
                const Tensor* delta, const Tensor& weight,
                const ResidualRmsNormArgs& args, Tensor* residual_out = nullptr);

// UNGATED PER-GROUP RMS NORM — `Qwen4ExpTextRMSNorm` (transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py:158-181`), the `group_size is not
// None` arm, executed per row:
//
//   g       = i / group_size                       (:168-169, the reshape)
//   ms[g]   = mean_{i in g}( x[i]^2 )              (:170, `pow(2).mean(-1)`)
//   out[i]  = x[i] * rsqrt(ms[g] + eps) * (1 + w[i])   when gemma
//   out[i]  = x[i] * rsqrt(ms[g] + eps) * w[i]         otherwise
//
// A SIBLING of vt::RmsNorm, which is the same expression with one group per
// row, and of vt::RmsNormGatedGroup, which groups the same way but multiplies
// `silu(gate)` in before the reduction. Neither can express this one; see the
// kRmsNormGroup comment for why the extent is not a mode of RmsNormArgs.
//
// THREE PLACES EPS IS NOT. It is INSIDE the rsqrt, added to the MEAN SQUARE,
// and added ONCE PER GROUP. The plausible slips are adding it to the norm
// rather than to the mean square, and computing it against the whole row's mean
// square. `scripts/gen-qwen4-exp-hc-goldens.py` case D exists for the first
// (at `hyper_scale = 1.7` an eps of 1e-6 moves the answer ~5e-7 relative, which
// is BELOW the gate tolerance and therefore invisible; at `hyper_scale = 0.01`
// it is 1% of the mean square), and the grouped/full-row separation on those
// same goldens is 4.0e-1 to 1.2e+0 against a 1e-5 tolerance.
//
// POLARITY. `gemma = true` applies `(1 + w)`, and that is what every
// `qwen4_exp` gamma needs: the loader stores each one RAW as HuggingFace ships
// it, centred on 0, and EVERY consumer adds the 1 itself — `vt::RmsNorm` under
// `gemma = true`, `vt::Qwen4ExpQsaCompress`, `vt::Qwen4ExpGatedResidual` since
// #2218, and this op. `ssm_norm` is the single exception in the architecture.
// The failure this rule prevents is silent: a gamma centred on 0 multiplied
// without the `+1` scales the stream by ~0 and reads as a corrupt checkpoint
// rather than as a wiring bug.
//
// SHAPES. x [T, H] and out [T, H] rank-2 contiguous; weight [H] rank-1, applied
// per COLUMN of the flat row (upstream's `self.weight` is `[dim]` over the
// unflattened width, and :177 multiplies AFTER `out.flatten(-2)`, so the weight
// index is the flat one and not the in-group one). `args.group_size` must be
// >= 1 and divide H; upstream raises `ValueError` on the same condition
// (:164-165) and this op refuses by name.
//
// PRECISION, AND IT IS A MIRROR RATHER THAN A CHOICE. The reduction, the
// reciprocal square root and the weight multiply all run in f32 and the result
// is rounded ONCE on the store, which is `output = self._norm(x.float())`,
// `output * (1.0 + self.weight.float())`, `output.type_as(x)` (:174-178) in
// order. Upstream says so in a comment at :175-176 — "Llama does x.to(float16)
// * w whilst Qwen4ExpText is (x * w).to(float16)" — so rounding the normalized
// value BEFORE the weight multiply is a different model, not a tolerance. This
// is the same order `vt::RmsNorm` already keeps; `vt::RmsNormGatedGroup` keeps
// the OTHER one, because its own upstream (mamba_mixer2.py:149) does.
void RmsNormGroup(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                  const RmsNormGroupArgs& args);

// --- Fused declarative recipe (TDR; see include/vt/recipes.h and
// .agents/specs/portable-fusion-framework.md). A recipe (a backend-agnostic
// constexpr FusedRecipe) is realized in tiers selected by VT_FUSED_TIER:
//   Tier 0 (default, composite): a device-agnostic walker (ops.cpp) dispatches
//     each opcode to the already-registered standalone vt:: op — BYTE-EXACT to the
//     unfused sequence the model hand-calls (kRmsNorm->RmsNorm, kSiluMul->
//     MoeSiluMul, kQuantFp4->ScaledFp4Quant, ...). Correct on every backend that
//     registers the constituent ops, the fp8 quant terminal included: this
//     bullet said "the fp8 quant terminal is CUDA-only" from 1115648d6
//     (2026-07-20) until 2026-09-12, and kQuantFp8Static gained its kCPU arm in
//     f270b4b0a (2026-08-15, src/vt/cpu/cpu_ops.cpp:4241, #468/#842), so the
//     terminal the walker dispatches is kCPU + kCUDA. The clause is DELETED
//     rather than narrowed, because it named no arm that is still owed.
//   Tier 1 (interpreter): a single-pass backend kernel for Tier-1-able recipes
//     (all steps elementwise/kRmsNorm, e.g. kFusedAddRmsNorm); the perf tier.
//
// Runtime scalars travel in FusedParams (structural constants like gemma live in
// the recipe). Tensors are bound positionally to the recipe's indexed operand
// table via FusedBinding (op[i] is the tensor for operand slot i; nullptr for an
// absent optional slot). Intermediate slots (a bf16 norm/activation result the
// next step quantizes) are caller-bound scratch — exactly as the unfused sequence
// materializes them.

// Physical tensors bound to a recipe's indexed operand table (op[i] == slot i).
struct FusedBinding {
  Tensor* op[kMaxFusedOperands] = {};
  int n = 0;
};

// Runtime scalars a recipe's opcodes consume (structural constants stay in the
// recipe). eps: RMSNorm/gated epsilon. quant_scale: the fp8 input_scale OR the
// fp4 input_global_scale_inv. fp4_layout: ScaledFp4Quant scale layout. rope: the
// kRope / kAttnQkNormRopeGate RoPE args.
struct FusedParams {
  float eps = 1e-6f;
  float quant_scale = 1.0f;
  Fp4ScaleLayout fp4_layout = Fp4ScaleLayout::kLinear;
  RopeArgs rope{};
};

// General entry: realize `recipe` over the bound operands with `params`.
//
// Realization order (W2): (1) if the recipe carries a fast_op (a bespoke single-
// launch fused kernel, e.g. kSiluMulFp4Quant) AND that OpId is registered on the
// device, dispatch to it — the SAME fast kernel the model called directly before
// migration, so the migration is perf-neutral by construction; (2) else, for a
// Tier-1-able recipe with VT_FUSED_TIER=1, the interpreter kernel; (3) else the
// Tier-0 composite. Every tier is byte-exact to the others per the §5 discipline.
void FusedChain(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                const FusedParams& params);

// Force the Tier-0 composite realization (the standalone-op-sequence oracle),
// bypassing any fast_op / interpreter tier. This is the byte-exact golden the fast
// realization is validated against; the parity tests call it to assert
// fast == composite == the unfused sequence. Production code uses FusedChain.
void FusedChainComposite(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                         const FusedParams& params);

// Narrow overload for the canonical (out, x, weight, residual) 4-operand shape —
// the W0-adopted kFusedAddRmsNorm site. Binds [x, weight, residual, out] and
// forwards to the general entry; bit-identical to RmsNorm(out, x, weight,
// {eps, gemma=true}, residual) for kFusedAddRmsNorm.
void FusedChain(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, Tensor* residual,
                const FusedRecipe& recipe, float eps);

// --- W2 convenience overloads: keep each migrated model call site a SINGLE
// FusedChain call (the bespoke fused-op call → one declarative recipe dispatch).
// Each binds the recipe's indexed operand table positionally and forwards to the
// general entry, which dispatches to the recipe's fast_op realization. The Tier-0
// composite intermediate slot (tmp_bf16) is bound nullptr — these sites feed the
// fast realization (which never materializes it); FusedChainComposite validates
// the composite separately with caller-provided scratch.

// Fp4-activation-quant shape (kSiluMulFp4Quant, kSigmoidGateFp4Quant): two float
// inputs a,b -> out_packed[M,K/2] + out_scale. Binds [a, b, nullptr, packed, scale].
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_packed, Tensor& out_scale,
                const Tensor& a, const Tensor& b, float quant_scale,
                Fp4ScaleLayout scale_layout = Fp4ScaleLayout::kLinear);

// RmsNorm->fp8 shape (kRmsNormQuantFp8): residual-add + gemma-RMSNorm -> static
// fp8, with an optional bf16 normed output. Binds [x, weight, residual, out_bf16,
// out_fp8]; residual/out_bf16 may be nullptr (matching the bespoke op contract).
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, Tensor* out_bf16,
                const Tensor& x, const Tensor& weight, Tensor* residual, float eps,
                float input_scale);

// Gated-RmsNorm->fp8 shape (kRmsNormGatedQuantFp8): gated-RMSNorm -> static fp8.
// Binds [x, gate, weight, nullptr, out_fp8].
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, const Tensor& x,
                const Tensor& gate, const Tensor& weight, float eps, float input_scale);

// Fused-attention-preamble MACRO shape (kAttnQkNormRopeGate): binds the recipe's
// fixed 8-operand table [qgate, kf, q_norm, k_norm, cos_sin, q_out, k_out, gate_out]
// and forwards to the general entry. This recipe has NO fast_op — its Tier-0
// composite already dispatches the whole preamble to the single vt::AttnQkNormRopeGate
// kernel, so the migration is perf-neutral by construction (same one launch).
void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& q_out, Tensor& k_out,
                Tensor& gate_out, const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                const Tensor& k_norm, const Tensor& cos_sin, float eps, const RopeArgs& rope);

// out[T,D] = silu(x[:, :D]) * x[:, D:], x is [T, 2D]; out f32 or bf16.
// Note: computes in f32 (upstream forward_native computes in x's dtype); bf16 parity tests need bf16-eps tolerance.
void SiluAndMul(Queue& q, Tensor& out, const Tensor& x);

// out[T,D] = silu(clamp(gate_up[:, :D], max=limit)) * clamp(gate_up[:, D:], -limit, limit)
// at alpha=1, beta=0. gate_up is [T, 2D] (gate = [:, :D], up = [:, D:]), same layout as
// SiluAndMul. The gate clamp is MAX-ONLY; the up clamp is BOTH sides. Computed in f32,
// silu narrowed through input dtype (same polarity as SiluAndMul). DeepSeek-V4
// activation (activation.py:197-201), GLM-5.3-Flash ExpertGate
// (modeling_glm5_next.py:137-142). CPU + CUDA + ROCm.
void ClampedSwiGLU(Queue& q, Tensor& out, const Tensor& gate_up, float limit);

// out[T,D] = gelu_tanh(x[:, :D]) * x[:, D:], x is [T, 2D]; out f32 or bf16.
// gelu_tanh(a) = 0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3))) — the exact
// `gelu_pytorch_tanh` / F.gelu(approximate="tanh"), computed in f32 then stored.
// Mirrors vLLM GeluAndMul(approximate="tanh") (activation.py NewGELU math). The
// GeGLU analog of vt::SiluAndMul; the one genuinely-new Gemma compute kernel.
void GeluAndMul(Queue& q, Tensor& out, const Tensor& x);

// out[i] = x[i] * scalar, elementwise, computed in f32 and rounded to out's
// dtype (f32/bf16). `scalar` is a runtime double (pass the bf16-rounded value
// for a bf16-exact match of a torch bf16-scalar multiply). Powers the Gemma
// embedding normalizer `embed_tokens(ids) * sqrt(hidden_size)` (gemma3.py:341).
void MulScalar(Queue& q, Tensor& out, const Tensor& x, double scalar);

// out[i] = cap * tanh(x[i] / cap), elementwise, computed in f32 and rounded to
// out's dtype. The Gemma-2 final logit soft-cap
// (LogitsProcessor(soft_cap=final_logit_softcapping), gemma2.py:344-345): a
// monotone squashing that leaves greedy argmax invariant but is applied for
// faithfulness. `cap` must be > 0. Shapes must match (same rank/extent).
void SoftCap(Queue& q, Tensor& out, const Tensor& x, double cap);

// out[..,D] = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias — torch
// `nn.LayerNorm` over the LAST dim (ported from ATen `native_layer_norm` /
// `vectorized_layer_norm_kernel`, the kernel `nn.LayerNorm` dispatches to for a
// CUDA half/bfloat16 input). `weight`/`bias` are optional rank-1 [D] tensors:
// both null == `elementwise_affine=False`. `var` is the BIASED (1/N) variance,
// as torch uses. x/out f32 or bf16; the mean/variance accumulation, the
// normalization and the affine are all computed in f32 and rounded once on
// store — matching torch's `acc_type<bfloat16> == float` contract, so a bf16
// LayerNorm here rounds exactly where torch's does.
//
// This is the mean-subtracting, bias-carrying sibling of vt::RmsNorm. It is
// what every pre-Llama-era family (OPT, GPT-2, BLOOM, ...) normalizes with;
// vllm/model_executor/models/opt.py:146-148,164-166,248-251. CPU + CUDA.
void LayerNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
               const Tensor* bias, const LayerNormArgs& args);

// out[i] = max(x[i], 0) — torch `F.relu`, i.e. vLLM `get_act_fn("relu")`
// (vllm/model_executor/layers/activation.py) as selected by OPT's
// `config.activation_function` (opt.py:156). Computed in f32, rounded on store;
// `out` may alias `x` (in-place). x/out f32 or bf16. CPU + CUDA.
void Relu(Queue& q, Tensor& out, const Tensor& x);

// Elementwise GELU (NEW, Qwen3-VL vision tower). out[i] = gelu(x[i]); `out` may
// alias `x`. x/out f32 or bf16, computed in f32. GeluTanh is the tanh-approx
// (`gelu_pytorch_tanh`, vision MLP); GeluErf is exact erf (`nn.GELU()`, merger).
void GeluTanh(Queue& q, Tensor& out, const Tensor& x);
void GeluErf(Queue& q, Tensor& out, const Tensor& x);

// out = a + b, in two shapes:
//   ELEMENTWISE  — b has a's exact shape (OPT's `residual + hidden_states`
//                  residual joins, opt.py:178,191, and the
//                  `inputs_embeds + pos_embeds` embedding join, opt.py:279).
//   ROW-BROADCAST — b is rank-1 [D] matching a's LAST dim, applied to every row
//                  (a `nn.Linear` bias term: OPT's q/k/v/out_proj/fc1/fc2 all
//                  carry one under `config.enable_bias`, opt.py:90-104,149-163).
// Computed in f32, rounded on store; `out` may alias `a` (in-place). All of
// a/b/out f32 or bf16. CPU + CUDA.
void Add(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);

// out[T,H] = table[ids[t], :]; ids i32/i64, bounds-checked; out f32 or bf16.
// CUDA note (M0.6 decision): ids live on the device, so the CUDA kernel clamps
// bad ids for the gather (no OOB read), records the first bad id in a device
// flag, and the wrapper synchronizes the stream and throws before returning.
// CUDA Embedding is therefore synchronizing for now — correctness-grade;
// revisit for full async when the model runner needs it (M0.9/M2).
void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);

// In-place partial NeoX RoPE on q[T,Hq,D] and k[T,Hk,D], positions[T].
// q/k dtype f32 or bf16 (same dtype for both); rotation computed in f32,
// rounded back on store for bf16.
void RopeNeox(Queue& q, Tensor& q_states, Tensor& k_states, const Tensor& positions,
              const RopeArgs& args);

// In-place partial RoPE from a supplied global cos|sin cache. Mirrors pinned
// vLLM's rotary_embedding custom op (base.py:160-252; _custom_ops.py:200-225)
// and MRotaryEmbedding's 3-axis selection (mrope.py:14-187,263-375).
//
// q [T,Hq,D], optional k [T,Hk,D], and cache [P,rotary_dim] share f32 or bf16.
// positions is [T] for ordinary/text RoPE or [3,T] for MRoPE. For the latter,
// args.mrope_section must sum to rotary_dim/2; contiguous and interleaved T/H/W
// layouts use the exact pinned selection rules. args.is_neox_style selects
// half-split NeoX or adjacent-pair GPT-J rotation. Formula construction is not
// part of this op: the hot path only gathers cache values and rotates.
//
// STRIDE-DRIVEN q/k (relaxed at MLA campaign W6; positions/cache stay
// contiguous). Only the innermost dimension must be unit-stride. DeepSeek's
// DECOUPLED RoPE rotates the TRAILING qk_rope_head_dim slice of each query head
// (`q[..., qk_nope_head_dim:]`, mla.py:160-167) with `is_neox_style=False`, and
// its k_pe is the trailing column block of the single fused
// `kv_a_proj_with_mqa` output (deepseek_v2.py:511) — both are strided views, so
// a contiguity requirement would cost two copies per layer per step. For a
// CONTIGUOUS tensor the strided offsets are integer-identical to the pre-W6
// (token * heads + head) * head_dim formula, so every existing caller (the
// Qwen3 dense/MoE preamble) is bit-identical by construction.
void RopeFromCache(Queue& q, Tensor& q_states, Tensor* k_states,
                   const Tensor& positions, const Tensor& cos_sin_cache,
                   const RopeArgs& args);

// FusedNormRope (kFusedNormRope): the fused DeepSeek-MLA norm-rope — one launch
// replacing the {RmsNorm(kv_c latent) ; RopeFromCache(k_pe)} pair over the merged
// kv_a projection output. Grid is one block per token; each block RMS-reduces the
// leading latent slice x[t, 0:off) and writes latent_out, then rotates the trailing
// decoupled-rope slice x[t, off:off+rot) from the precomputed cos|sin cache and
// writes pe_out. The two halves address DISJOINT dims, so it is BIT-FOR-BIT equal
// to composing the two standalone ops:
//   x         [T, off+rot] f32/bf16 — merged kv_a output (contiguous; off = norm_weight length)
//   norm_weight [off]      f32/bf16 — kv_a_layernorm weight (over kv_lora_rank)
//   positions [T]          i32/i64  — rope positions
//   cos_sin_cache [P, rot] f32      — the rope cache (cols [0,rot/2)=cos, [rot/2,rot)=sin)
//   latent_out[T, off]     f32/bf16 — RmsNorm(x[:, :off]) (NOT roped)
//   pe_out    [T, rot]     f32/bf16 — RopeFromCache-rotated x[:, off:] (NOT normed, single vector)
// norm_args: eps + gemma (gemma=false for DeepSeek). rope_args: rotary_dim (=rot) +
// is_neox_style (DeepSeek-V2/V3 use the GPT-J adjacent-pair form, is_neox_style=false).
// CPU + CUDA. Additive: only the DeepSeek-V2/kimi MLA block dispatches it.
void FusedNormRope(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                   const Tensor& norm_weight, const Tensor& positions,
                   const Tensor& cos_sin_cache, const RmsNormArgs& norm_args,
                   const RopeArgs& rope_args);

// --- Fused full-attention preamble (default-OFF prefill lever; mirror of vLLM's
// fused_qk_rmsnorm_rope / fla fused_qk_norm_rope.py:95-102, which reads a
// precomputed cos_sin_cache with ZERO in-kernel transcendentals). Two ops:
//
// RopeCosSinCache: precompute the batch's cos|sin ONCE per step so the fused
// preamble kernel below does no per-element pow/cos/sin (the current RopeNeox
// recomputes them in DOUBLE per element, per head, per layer). Fills
// cos_sin[T, rotary_dim] f32 from positions[T]: for token t, pair i in
// [0, rotary_dim/2):
//   freq  = base^(-2i/rotary_dim)            (double, matching RopeNeox)
//   angle = positions[t] * freq             (double)
//   cos_sin[t, i]              = (f32) cos(angle)
//   cos_sin[t, rotary_dim/2+i] = (f32) sin(angle)
// The double-precision angle + f32 cast reproduce RopeNeox's per-element c/sn
// BIT-FOR-BIT, so the cached rotation is token-exact vs the inline path.
// positions i32/i64; cos_sin f32 [T, rotary_dim]. CPU + CUDA.
void RopeCosSinCache(Queue& q, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args);

// AttnQkNormRopeGate: the fused full-attention preamble — one launch replacing the
// AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox four-kernel chain, removing
// their f32 HBM intermediate round-trips. Grid (T, Hq+Hkv), one block per
// (token, head); the block reduces over Dh (gemma-RMSNorm), then applies partial
// NeoX RoPE reading the precomputed cos_sin cache. Bit-for-bit equal to composing
// the four ops when the outputs are f32 (the wired default): the gemma RMSNorm
// (f32 variance, weight as (1+w)) and the RoPE (x*c - y*sn / x*sn + y*c from the
// same f32 c/sn) are the identical f32 arithmetic; only launch-count and HBM
// traffic change. The output dtype is templated (f32 keeps the PagedAttention f32
// query contract; bf16 halves the writes but rounds q/k/gate — a GPU-gated A/B).
//   qgate   [T, Hq*2*Dh]  f32/bf16 — the fused q|gate projection (per head [q|gate])
//   kf      [T, Hkv*Dh]   f32/bf16 — the k projection
//   q_norm/k_norm [Dh]    f32      — the per-head gemma-RMSNorm weights
//   cos_sin [T, rot]      f32      — RopeCosSinCache output (rot = rope_args.rotary_dim)
//   q_out   [T, Hq, Dh]   f32/bf16 — gemma-RMSNorm'd + RoPE'd q
//   k_out   [T, Hkv, Dh]  f32/bf16 — gemma-RMSNorm'd + RoPE'd k
//   gate_out[T, Hq, Dh]   f32/bf16 — the raw gate half (passthrough, no norm/rope)
// norm_args: eps + gemma (must be gemma=true for Qwen). rope_args: rotary_dim (the
// partial rotation width; base is unused — the cache is precomputed). q_out/k_out
// share one dtype; gate_out matches it OR stays f32 while q/k are bf16 (the FA-2
// prefill combo: bf16 q/k feed FA-2 + the bf16 KV-cache write — each store is the
// RN round of the same f32 value, bit-identical to f32-out + CastBf16 — while
// sigmoid(gate) keeps the un-rounded f32 gate). qgate/kf share one dtype. CPU + CUDA.
void AttnQkNormRopeGate(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin,
                        const RmsNormArgs& norm_args, const RopeArgs& rope_args);

// Gate-free fused attention preamble (the kAttnQkNormRope recipe): per-head
// standard RMSNorm(q) + RMSNorm(k) + partial RoPE-from-cache, with q3/k3 normed
// and rotated IN PLACE. A backend that registers no kernel for it is a refusal;
// the recipe's Tier-0 composite is reached through vt::FusedChain, which is the
// only caller that falls back.
void AttnQkNormRope(Queue& q, Tensor& q3, Tensor& k3, const Tensor& q_norm,
                    const Tensor& k_norm, const Tensor& cos_sin, const Tensor& positions,
                    const RmsNormArgs& norm_args, const RopeArgs& rope_args);

// --- GDN (Gated DeltaNet) ops. Formula reference: .agents/specs/gdn-semantics.md.
// All GDN state tensors are caller-allocated f32 and updated IN PLACE
// (upstream computes states in f32 and rounds to the cache dtype on store —
// that rounding point is M0.9 layer assembly, gdn-semantics.md §1).

// Varlen causal conv over the token stream (gdn-semantics.md §2, upstream
// causal_conv1d_fn). x[T,C] token-major, weight[C,K], optional bias[C]
// (nullptr = no bias; Qwen GDN conv has bias=False), conv_state[N,C,K-1] f32
// in/out (per-sequence slices, gathered — cache_indices/NULL-block handling is
// M0.9), query_start_loc[N+1] i32 cumulative token offsets (seq s spans
// [qsl[s], qsl[s+1])), has_initial_state[N] i8/i32 (0/1; upstream bool).
//   out[c,t] = act(bias[c] + sum_j w[c,j] * window[j]), window[j] = x token
//   t-(K-1-j), falling back to conv_state (if has_initial_state) or 0 for
//   tokens before the sequence start. w[:,K-1] multiplies the current token.
// State write-back: last K-1 RAW x tokens (pre-activation), left-padded with
// zeros (no init state) or shifted old state when T < K-1.
// x [T,C] may be a padded-row (inner-contiguous, outer stride >= C) view — the
// merged qkvz projection feeds mixed_qkv = mixed_qkvz[:, :conv_dim] without a
// copy; out/weight/conv_state stay contiguous.
void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args);

// Single-token conv step (gdn-semantics.md §3, upstream causal_conv1d_update
// seqlen==1 path). x[B,C] one token per sequence, conv_state[B,C,K-1] f32
// in/out. Read-old-then-roll:
//   out[c] = act(bias[c] + sum_j w[c,j] * [conv_state[c,:], x[c]][j])
//   conv_state[c,:] <- [conv_state[c,1:], x[c]]   (raw x)
// conv_state_indices (optional; mirrors mamba causal_conv1d_update conv_state_indices /
// cache_indices): when non-null, row bt reads/writes the persistent cache slot
// conv_state_indices[bt] (conv_state is then the FULL [num_slots,C,K-1] cache), so the
// caller need not gather/scatter per-request rows. When null, conv_state is compact
// [batch,C,K-1] and row == bt. x [B,C] may be a padded-row (inner-contiguous)
// view of the merged qkvz output; out stays contiguous.
void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args,
                        const Tensor* conv_state_indices = nullptr);

// QWEN4-EXP PLE DILATED DEPTHWISE CAUSAL CONV, batched over sequences.
// `Qwen4ExpTextPLELayer._short_conv` (transformers v5.16.0
// modeling_qwen4_exp.py:1150-1167) end to end, including the state read and the
// state write-back that `cache_utils.py::LinearAttentionLayer.update_conv_state`
// (:1036-1075) performs around it. Per channel c and output position t:
//
//   out[t, c] = silu( sum_{k=0..K-1} weight[c, k] * hist[t - (K-1-k)*dilation] )
//
// with `hist` the concatenation of `conv_state` and this call's `x`, so
// `weight[c, K-1]` multiplies the CURRENT token and the op is causal by that
// tap. At the released config (K = 4, dilation = `ngram_size` = 3) the taps are
// lags {9, 6, 3, 0}: a span of ten tokens for four multiply-accumulates, and a
// genuine 9-deep history read at stride 3 that CANNOT be compressed to three
// columns even though any one step touches only three of them.
//
// SILU IS UNCONDITIONAL and there is no activation switch, because upstream has
// none: the line is `F.silu(self.conv1d(hidden_states))`, full stop. Mirroring
// that is the point; a knob nobody can set is a divergence with extra steps.
//
// THERE IS NO `has_initial_state`, AND THAT IS AN ARGUMENT RATHER THAN AN
// OMISSION. `CausalConv1dFwd` needs one because its state is undefined before
// the first chunk. Here upstream's first call takes the `has_previous_state`
// false branch and LEFT-ZERO-PADS (cache_utils.py:1053-1060), which is
// arithmetically identical to running the steady path against a zeroed state.
// A zero-initialised `conv_state` is therefore bit-identical to a first call,
// and a flag would only add a way to disagree with upstream.
//
// SHAPES. x [T, C] (the NORMED conv input — never the raw hidden state, see the
// fork at modeling_qwen4_exp.py:1184-1188); out [T, C]; weight [C, K] depthwise,
// K >= 2; conv_state [N, C, L] f32 READ-WRITE with L == (K - 1) * dilation;
// query_start_loc [B + 1] i32 cumulative token offsets, sequence s spanning
// [qsl[s], qsl[s+1]); conv_state_indices [B] i32 or nullptr.
//
// `conv_state_indices` IS THE PER-SEQUENCE CACHE ROW, and the name is
// deliberate: this op does NOT call it `state_idx`, because upstream already
// uses that name for a different axis and the collision would read as agreement.
// Upstream writes `update_conv_state(..., state_idx=1, ...)`, where
// 1 selects the PLE conv out of the three states a PLE layer owns — the GDN
// conv, this conv, and the n-gram token history. Those three cannot be planes of
// one tensor: `cache_utils.py` keeps `conv_states` as a LIST with a per-entry
// `conv_kernel_size[state_idx]`, and the three widths here are 4, 9 and 2, over
// different channel counts and, for the third, over integers rather than floats.
// Upstream's selector is therefore resolved by the CALLER passing this op the
// right tensor, which is what taking the state as an explicit operand buys.
// What is left is the axis a batched engine actually needs and upstream does not
// have: which ROW of a shared cache each sequence owns. That is exactly what
// `CausalConv1dUpdate`'s `conv_state_indices` names, so this op spells it the
// same way rather than reusing upstream's word for somebody else's axis. Null
// means row s for sequence s, again as that sibling does.
//
// PRECISION. f32 interior, taps accumulated in DOUBLE — the same house
// convention the W2 host reference (`qwen4_exp_ple.cpp`) uses, kept so the two
// arms answer to one oracle. Four terms is a short reduction, so this is cheap
// insurance rather than a load-bearing width; a CUDA arm may accumulate in f32
// and must be gated against the oracle to say so.
void Qwen4ExpPleConv(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor* conv_state_indices, const Qwen4ExpPleConvArgs& args);

// The Qwen4-Exp PLE GATE — `Qwen4ExpTextPLELayer.forward` :1181-1182, with the
// :1184 flatten folded into the output layout. Per (t, j, d):
//
//   g       = score[t, j] / args.gate_divisor            // the tail of :1180
//   gate    = sign(g) * sqrt(max(|g|, args.clamp_min))   // :1181
//   out[t, j*H + d] = sigmoid(gate) * value[t, d]        // :1182, flattened :1184
//
// with sign(0) == 0, so g == 0 gives gate == 0 and out == 0.5 * value — NOT the
// 1e-3 floor. That discontinuity is upstream's and is reachable: a fully masked
// row scores exactly zero.
//
//   out    [T, hc*H]  f32 or bf16, contiguous
//   score  [T, hc]    f32, contiguous — the raw per-(t, j) dot from
//                     vt::BatchedMatmul, BEFORE the sqrt(hidden_size) divide
//   value  [T, H]     float, contiguous
//
// `hc` and `H` are read off `score` and `value`; `out`'s second dim must be
// their product, which is the check that catches a caller that flattened the
// two the other way round.
//
// SCORE IS F32 AND ONLY F32, for the reason vt::SigmoidGateBf16 gives for its
// own gate operand: it is the argument of a sigmoid and a transcendental's
// input must not be rounded. `value` and `out` carry the model's storage width.
//
// PRECISION. f32 in, DOUBLE interior — the divide, the clamp, the square root,
// the sigmoid and the product are all evaluated in double, matching the W2 host
// reference (`qwen4_exp_ple.cpp::PleForward`) term for term — then ONE store.
// The same house convention vt::Qwen4ExpPleConv states, and for the same
// reason: it lets the two arms be bit-identical rather than merely close. A
// CUDA arm that evaluates in f32 will not inherit that and must be gated
// against the oracle directly; the spec's `## Owed` records it.
void Qwen4ExpPleGate(Queue& q, Tensor& out, const Tensor& score, const Tensor& value,
                     const Qwen4ExpPleGateArgs& args);

// SPECULATIVE multi-token conv step (SPEC-MTP I4). Ported from
// vllm/model_executor/layers/mamba/ops/causal_conv1d.py @ e24d1b24
// (_causal_conv1d_update_kernel IS_SPEC_DECODING + IS_VARLEN branches
// :835-1067, wrapper `state_len = width - 1 + (seqlen - 1)` :1181-1184), as
// called by qwen_gdn_linear_attn.py:1344-1356.
//
// A speculative request submits `1 + k` query tokens per step, but only
// `num_accepted_tokens` of the PREVIOUS step's tokens were actually kept. The
// conv window must therefore advance by exactly the accepted count, NOT by
// 1 + k. Upstream does that WITHOUT any rollback copy, by widening the conv
// state row to `state_len = (K-1) + k` taps and treating it as a sliding
// window: the read offset for this step is `num_accepted_tokens[i] - 1`.
//
//   x            [num_tokens, C] varlen token stream of the SPEC rows only,
//                request i occupying [cu_seqlens[i], cu_seqlens[i+1]).
//                May be a padded-row (inner-contiguous) view. Written in place
//                is NOT done here — `out` is a separate [num_tokens, C] buffer.
//   conv_state   [num_slots, C, state_len] FULL persistent cache, updated in
//                place. state_len must be (K-1) + max_query_len - 1.
//   conv_state_indices [num_reqs] — the slot of each spec row. This is
//                spec_state_indices_tensor COLUMN 0 only: unlike the SSM state,
//                the conv window needs no per-timestep slots because the whole
//                1+k history fits in the widened row (qwen_gdn_linear_attn.py:1350).
//                Local ABI: index < 0 ⇒ NULL row, skipped (see GdnDecode).
//   num_accepted_tokens [num_reqs] i32, each in [1, max_query_len].
//   cu_seqlens   [num_reqs + 1] i32 (upstream query_start_loc).
//   max_query_len = 1 + k (upstream `max_query_len`, = spec_state_indices.size(-1)).
//
// Semantics, per request i and channel c, with off = num_accepted_tokens[i]-1,
// S = conv_state[slot][c][off : off+K-1] ++ x[i-th row range][c]:
//   out[t][c] = act(bias[c] + sum_{j<K} w[c][j] * S[t + j])
//   conv_state[slot][c] <- S[1 : 1 + state_len]      (left-shift by ONE tap)
// The left-shift-by-one (not by seqlen) is what makes the NEXT step's
// `off = num_accepted - 1` select exactly the window ending at the last
// accepted token — i.e. the conv rollback.
void CausalConv1dSpecUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                            const Tensor* bias, Tensor& conv_state,
                            const Tensor& conv_state_indices,
                            const Tensor& num_accepted_tokens, const Tensor& cu_seqlens,
                            const CausalConv1dArgs& args);

// Rowwise l2 normalization over the LAST dim (gdn-semantics.md §4, upstream
// l2norm_fwd): y = x * rsqrt(sum(x^2) + eps). Plain SUM, not mean — this is
// not an rmsnorm. x/out rank 2 or 3 ([rows, D] or [T, H, D]); f32 math.
void L2Norm(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args);

// Gated rmsnorm (gdn-semantics.md §5, upstream RMSNormGated with
// norm_before_gate=True, group_size=None, no bias):
//   var = mean(x^2 over last dim);  out = x * rsqrt(var + eps) * w * act(z)
// x/gate/out rank-2 [rows,D] or rank-3 [T,Hv,D], weight [D]; normalization is
// over the LAST dim either way; act = silu (or sigmoid, args.sigmoid_gate).
// x/out stay contiguous; the gate may carry a padded outer (token) stride with
// contiguous inner dims — the merged qkvz projection's z = mixed_qkvz[:,
// conv_dim:] slice viewed as [T,Hv,Dv] (qwen_gdn_linear_attn.py:929-936).
void RmsNormGated(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                  const Tensor& weight, const RmsNormGatedArgs& args);

// Gated-delta-rule recurrence over varlen prefill sequences
// (gdn-semantics.md §7, upstream fused_recurrent_gated_delta_rule — the
// pinned sequential statement of chunk_gated_delta_rule). q_in/k[T,Hk,Dk]
// ALREADY l2-normalized (upstream prefill normalizes in fused_post_conv_prep,
// §4), v[T,Hv,Dv], g/beta[T,Hv] f32 (log-space decay / sigmoid(b), derived
// upstream per §6), state[N,Hv,Dv,Dk] f32 in/out (zeros for fresh sequences),
// query_start_loc[N+1] i32. GQA broadcast: v-head hv reads q/k head
// hv / (Hv/Hk); Hv must be a multiple of Hk. Per token:
//   q' = q * scale;  S *= exp(g[hv]);  v' = (v - S @ k) * beta[hv];
//   S += outer(v', k);  out = S @ q'
// k is NOT scaled. All arithmetic f32.
void GdnPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& query_start_loc, const GdnArgs& args);

// Kimi Delta Attention (KDA) gated-delta recurrence — the PER-K-CHANNEL-DECAY
// variant of GdnPrefill. Ported 1:1 from FLA's
// fused_recurrent_gated_delta_rule_fwd_kernel with IS_KDA=True (third_party/
// flash_linear_attention/ops/fused_recurrent.py:88-175 @ pin 555967922; the KDA
// wrapper is ops/kda.py:109-146 fused_recurrent_kda). The plain-GDN kernel
// (IS_KDA=False) applies a PER-HEAD scalar decay `b_h *= exp(b_g)`; KDA applies a
// PER-K-CHANNEL decay `b_h *= exp(b_gk[None, :])` — so `g` here is [T,Hv,Dk]
// (one log-decay per K channel of the value head's [Dv,Dk] state), broadcast
// across the Dv rows. Everything else is byte-for-byte GdnPrefill's recurrence
// (decay -> predict -> beta -> rank-1 update -> read-out, all in f32):
//   S[:,k] *= exp(g[hv,k]);  v' = (v - S @ k) * beta[hv];  S += outer(v',k);
//   out = S @ (q*scale)
// The shared GDN kernels are UNTOUCHED (Qwen3.6 27B/35B GDN gate byte-identical);
// KDA lands as this additive per-channel op. q_in/k [T,Hk,Dk] MUST be
// l2-normalized by the caller (as GdnPrefill; upstream fuses it via
// USE_QK_L2NORM_IN_KERNEL, exact per gdn-semantics.md §4). v/out [T,Hv,Dv],
// g [T,Hv,Dk] f32 per-channel log-decay, beta [T,Hv] f32 (per-head sigmoid(b)),
// state [N,Hv,Dv,Dk] f32 in/out (zeros for fresh sequences),
// query_start_loc [N+1] i32. For KDA Hk==Hv; GQA broadcast supported for reuse.
void KdaGatedDeltaRule(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                       const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                       const Tensor& query_start_loc, const GdnArgs& args);

// KDA CHUNK-PREFILL — the chunked forward of KdaGatedDeltaRule, computing the same
// per-K-channel gated-delta linear-attention output but in BT=64 chunks through the
// vendored FLA Triton-AOT cubins (kda_gate_cumsum -> kkt(inter+intra) -> solve_tril
// -> recompute_w_u -> chunk_delta_h -> chunk_gla_o), mirroring vLLM's prefill path
// (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate; decode stays the recurrent
// KdaGatedDeltaRule). Result differs from the recurrence only by chunked-vs-recurrent
// REDUCTION ORDER (not bit-exact). Fires only at the pinned Kimi KDA geometry
// (Hk==Hv==32, Dk==Dv==128) in a CUDA + VLLM_CPP_TRITON build; any other shape or a
// CPU queue transparently falls back to the recurrence. Inputs:
//   q_in/k [T,H,Dk] f32/bf16, L2-normalized by the caller (as KdaGatedDeltaRule);
//   v [T,H,Dv] f32/bf16; out [T,H,Dv] f32/bf16;
//   g_raw [T,H,Dk] f32 — the RAW gate projection (kda_gate_cumsum fuses the gate:
//     -exp(a_log)*softplus(g_raw+dt_bias), chunk-cumsum, RCP_LN2 fold, on device);
//   beta [T,H] f32 — the per-head gate (sigmoid(b), applied directly);
//   a_log [H] f32; dt_bias [H*Dk] f32 (or empty => no bias);
//   state [1,H,Dv,Dk] f32 (fresh zeros in, final written); query_start_loc [N+1] i32.
// args.scale == Dk^-0.5 (baked in the cubins; guarded).
void KdaChunkPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                     const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                     const Tensor& query_start_loc, const GdnArgs& args);

// ─── Mamba2 / SSD (.agents/specs/mamba2-ssd.md, issue #496) ──────────────────
//
// SSD IS NOT THE GATED DELTA RULE. GdnPrefill/KdaGatedDeltaRule above carry a
// delta REMOVAL term `(I - beta*k*k^T)` and a decay that is a per-head (GDN) or
// per-K-channel (KDA) scalar. Mamba2's SSD is a diagonally-decayed gated linear
// recurrence with NO removal term, whose decay `exp(A[h]*dt[t,h])` is driven by
// `A_log`/`dt`, with a `D` skip and `B`/`C` SHARED across `n_groups` head groups
// (head h reads group `h / (nheads/ngroups)`). These are sibling ops, never a
// parameterisation of the GDN kernels (mamba2-ssd.md §0, §7).

// MAMBA2 CHUNKED SSD PREFILL — the varlen entry
// `mamba_chunk_scan_combined_varlen` (ssd_combined.py:157-235) over the 5-stage
// pipeline `_mamba_chunk_scan_combined_fwd` (:27-156), in upstream order:
//   1. `_chunk_cumsum_fwd`  (ssd_chunk_state.py:300-346) — dt bias/softplus/clamp
//      then per-chunk `dA_cumsum[h,c,i] = sum_{j<=i} dt[h,c,j]*A[h]`. Positions
//      past a partial chunk's length hold dt = 0, so `dA_cumsum[h,c,cs-1]` is the
//      chunk's TOTAL decay whatever its length (:96-110).
//   2. `_chunk_state_fwd`   (ssd_chunk_state.py:349-407) — the chunk-local state
//      `sum_i x*B * exp(min(dA_last - dA_i, 0)) * dt`, accumulated in f32
//      (`states_in_fp32=True`, ssd_combined.py:100-102).
//   3. `_state_passing_fwd` (ssd_state_passing.py:99-146) — the inter-chunk
//      recurrence `S_c = exp(dA_last[c]) * S_{c-1} + states[c]`, run per SEQUENCE
//      over the chunk range `last_chunk_indices` derives, seeded from
//      `initial_states`. Note out[c] is the state AFTER chunk c (:90-97).
//   4. `_bmm_chunk_fwd`     (ssd_bmm.py:148-209) — `CB[c,g,i,j] = C_i . B_j`,
//      accumulated and RETURNED in f32 regardless of activation dtype
//      (`output_dtype=torch.float32`, ssd_combined.py:124).
//   5. `_chunk_scan_fwd`    (ssd_chunk_scan.py:216-525) — per token i of chunk c:
//        out = exp(dA_i) * (C_i . S_{c-1})                       // inter-chunk
//            + sum_{j<=i} CB[i,j] * exp(min(dA_i - dA_j,0)) * dt_j * x_j // intra
//            + D * x_i                                           // skip
//        and `out *= z*sigmoid(z)` when z is given. `S_{c-1}` is
//        `initial_states[seq_idx[c]]` when `seq_idx[c] != seq_idx[c-1]` and
//        initial states were supplied, ZEROS when they were not (:236-250,
//        :271-289) — that is what makes a sequence boundary inside a physical
//        chunk correct.
//
// Shapes (varlen, implicit batch 1 — ssd_combined.py:158-215):
//   x            [T,H,P] float          dt      [T,H] float
//   A            [H] f32 (negative)     B,C     [T,G,N] float, H % G == 0
//   D            optional [H] or [H,P] f32       z  optional [T,H,P] float
//   dt_bias      optional [H] f32
//   initial_states optional [S,H,P,N] f32/bf16 (:79, :194)
//   cu_seqlens   [S+1] i32              seq_idx [nchunks] i32 — PER CHUNK, not
//                                       per token (:60-61, :189)
//   cu_chunk_seqlens [nchunks+1] i32    last_chunk_indices [S] i32
//   out          [T,H,P] f32/bf16 (pre-allocated, written in place)
//   final_states [S,H,P,N] f32/bf16 — `varlen_states`, the state after each
//                sequence's LAST chunk (:154). Its dtype is the SEPARATE
//                `state_dtype` knob (:46,119,176), NOT the activation dtype.
// `args.chunk_size` must be a power of two (:48). All arithmetic is f32; the
// device tile-precision downcasts inside the Triton dots are a W2 concern.
void Mamba2ChunkScan(Queue& q, Tensor& out, Tensor& final_states, const Tensor& x,
                     const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                     const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                     const Tensor* initial_states, const Tensor& cu_seqlens,
                     const Tensor& cu_chunk_seqlens, const Tensor& last_chunk_indices,
                     const Tensor& seq_idx, const Mamba2Args& args);

// MAMBA2 SINGLE-TOKEN SELECTIVE STATE UPDATE — the decode path,
// `selective_state_update` (mamba_ssm.py:497+) as mamba_mixer2.py:1087 calls it.
// Per row b and head h (group g = h / (H/G)), with dt SCALAR PER HEAD — the
// `tie_hdim` shape Mamba2 always uses, and the only one upstream's own CPU
// kernel implements (csrc/cpu/mamba_kernels.hpp:104-250, and the
// `current_platform.is_cpu()` skips in tests/kernels/mamba/test_mamba_ssm.py):
//   dt = softplus(dt[b,h] + dt_bias[h])            // guarded dt<=20, :177
//   s  = s * exp(A[h]*dt) + B[b,g,:] * x[b,h,p] * dt
//   out[b,h,p] = sum_n s[n]*C[b,g,n] + D[h]*x[b,h,p]   (then *= silu(z))
//
//   state   [S,H,P,N] f32/bf16 — the FULL cache when state_indices is given
//           (mamba2_state_dtype's ssm_dtype is its own knob, mamba_utils.py:73-81),
//           else compact [Nb,H,P,N] with row b == b. Updated IN PLACE.
//   x [Nb,H,P] float, dt [Nb,H] float, A [H] f32, B/C [Nb,G,N] float,
//   D optional [H] f32, z optional [Nb,H,P] float, dt_bias optional [H] f32,
//   out [Nb,H,P] f32/bf16.
//   state_indices optional [Nb] i32 (upstream `state_batch_indices`). LOCAL ABI:
//           index < 0 is the NULL row — its cache slot is left untouched and its
//           output row is zeroed, exactly as GdnDecode/CausalConv1dSpecUpdate
//           already model it (ops.h GdnDecode, cpu_ops.cpp GdnDecodeKernel).
//           Upstream's sentinel is `NULL_BLOCK_ID = 0`
//           (v1/attention/backends/utils.py:46) and it leaves the padded output
//           rows UNDEFINED (`continue`, mamba_kernels.hpp:147); the caller maps
//           its padding onto the local negative sentinel, and zeroing is strictly
//           more defined than what upstream promises.
void Mamba2StateUpdate(Queue& q, Tensor& out, Tensor& state, const Tensor& x,
                       const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                       const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                       const Tensor* state_indices, const Mamba2Args& args);

// SILU-GATED GROUP RMS NORM — `Mixer2RMSNormGated.forward_native`
// (mamba_mixer2.py:100-149). A SIBLING of RmsNormGated, not a mode of it: that
// one is the GDN/KDA gate (sigmoid or silu) over the WHOLE row; this one always
// SILU-gates and reduces the variance over `group_size = hidden / n_groups`
// slices. Both the activation and the reduction extent differ (mamba2-ssd.md §0).
//   v      = x * silu(f32(gate))                                     (:114)
//   out    = weight * dtype(x)( v * rsqrt(mean(v^2 over its group) + eps) )
//            (:136-141 grouped, :127-131 the n_groups == 1 whole-row branch,
//             :149 the `self.weight * x.to(input_dtype)` cast point)
// x/gate/out are rank 2 [rows,Hd] or rank 3 [T,H,D]; the LAST dim is the hidden
// dim and every leading dim is a row (`*prefix_dims, hidden_dim`, :136).
// `weight` is [Hd] float, or NULLPTR for `use_rms_norm=False`, where upstream
// registers no parameter at all and returns just the gated value (:94-96,
// :115-116). args.n_groups must divide the last dim.
void RmsNormGatedGroup(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                       const Tensor* weight, const RmsNormGatedGroupArgs& args);

// QWEN4-EXP GATED RESIDUAL — one hyper-connection READ, batched over T tokens.
// `Qwen4ExpTextGatedResidual.forward` (modeling_qwen4_exp.py:952-969) fused end
// to end, with the grouped RMS norm of `Qwen4ExpTextRMSNorm` (:167-178) at the
// front. Per token:
//
//   normed[j*H+h] = hyper[j*H+h] * rsqrt(mean_h(hyper[j*H+.]^2) + eps)
//                                * (1 + hc_norm_w[j*H+h])    (group_size == H)
//   low[r]        = silu( (mix_down[r] . normed) / hc )       -- DIVIDE INSIDE
//   gate[p]       = sigmoid( mix_up[p] . low )                -- NO divide here
//   mixed[h]      = mean_j( gate[j*H+h] * normed[j*H+h] )     -- MEAN, not sum
//   inject[j]     = 2 * sigmoid( (block_inject[j] . normed) / hc )
//
// Three spellings in there are single-character defects that produce plausible
// output, and all three are gated (tests/vllm/models/test_qwen4_exp_hc_device.cpp):
// the `/ hc` sits INSIDE the SiLU (SiLU is not homogeneous), there is NO `/ hc`
// on the up projection, and the collapse is a MEAN over the branches while the
// product it collapses is against the NORMED stream and not the raw one.
//
// `hc_norm_w` IS THE RAW HUGGINGFACE GAMMA, `w_hf`, and THIS OP ADDS THE 1.
// That is upstream's parameterization verbatim — `Qwen4ExpTextRMSNorm.forward`
// is `output * (1.0 + self.weight.float())` over a ZERO-initialised weight
// (:173-178) — and it is the SAME polarity `vt::RmsNorm(gemma=true)` and
// `vt::Qwen4ExpQsaCompress` apply to this architecture's other gammas, which is
// the point: `Qwen4ExpWeights` holds every gamma raw (the `qwen4exp` loader
// inverts the `+1` ggml-org/llama.cpp#27742 bakes in at convert time, with
// `linear_attn.norm.weight` the one tensor the converter never folds), so ONE
// rule covers the whole model and no call site has to remember which of two
// forms this particular op wanted.
//
// IT READ THE OTHER WAY UNTIL #2218, and the correction is recorded here rather
// than only in the spec because the failure is silent. Under the old contract a
// forward that handed this op the loaded `hc_norm` multiplied every
// hyper-connection norm by a gamma centred on ZERO — a plausible tensor, never
// a crash, and unreachable by any single-sided gate, because the loader was
// right about its own output and the op was right about its own input.
// `vllm::qwen4_exp::HcNormWeightFromHf` remains the `w_hf -> 1 + w_hf` bridge
// the W3 HOST reference needs (`qwen4_exp_hc.h` `GroupedRmsNorm` is vLLM's
// `out * w` form and keeps it); it is NOT a step any caller of this op takes.
//
// SHAPES. hyper [T, hc*H]; hc_norm_w [hc*H]; mix_down [R, hc*H]; mix_up [hc*H, R]
// (both in PyTorch `nn.Linear(bias=False)` `(out_features, in_features)` order);
// block_inject [hc, hc*H] or nullptr; mixed [T, H]; injection [T, hc] or nullptr.
// `injection` and `block_inject` must be null TOGETHER — a null pair is
// upstream's `use_combine=False` mixer, which returns `mixed_input` alone.
// `hyper` is READ ONLY and is NOT normalized in place: upstream returns the RAW
// stream and it is the raw stream the write-back adds to.
//
// PRECISION. f32 interior, per-group sum of squares accumulated in DOUBLE — the
// house host-reference convention (`deepseek_v4_mhc.cpp`), kept because at the
// model's real group size of 2560 an f32 accumulator differs from it by 742x on
// magnitude-separated data. A CUDA arm therefore cannot simply f32-accumulate
// this reduction and meet the same bound; the spec records that choice as owed.
// Tensors may be f32 or bf16 (widen on load, round once on store), mirroring
// upstream's `_norm(x.float())` ... `.type_as(x)`.
//
// EXCEPT THE THREE PROJECTIONS, WHICH MAY KEEP THE FILE'S BLOCKS (W5p, #2031).
// `mix_down`, `mix_up` and `block_inject` also accept a block-quantized `[N,K]`
// dtype, and the kernel then routes that projection through `vt::MatmulBT`,
// which dispatches `kMatmulBTQuant`. The released
// `unsloth/Qwen3.8-Flash-Next-GGUF` stores all 194 of these mix weights as Q8_0
// and cannot prefill otherwise. `hyper`, `hc_norm_w`, `mixed` and `injection`
// stay float and a block-typed one is refused BY NAME: the gamma is an
// ELEMENTWISE multiplicand, which is the split llama.cpp makes for this same
// architecture -- six projections declared `GGML_OP_MUL_MAT`
// (`src/llama-arch.cpp:759,760,761,763,764,765` at `6c84c7d5d`) against
// `GGML_OP_MUL` for `hc_*_norm` (`:758`, `:762`), with an explicit f32 cast
// where a file-typed weight of this architecture meets an elementwise multiply
// (`src/models/qwen4exp.cpp:1198-1202`). A float weight is bit-identical to
// before: it keeps the same scalar accumulation in the same index order.
void Qwen4ExpGatedResidual(Queue& q, Tensor& mixed, Tensor* injection, const Tensor& hyper,
                           const Tensor& hc_norm_w, const Tensor& mix_down,
                           const Tensor& mix_up, const Tensor* block_inject,
                           const Qwen4ExpGatedResidualArgs& args);

// THE RANK-1 WRITE-BACK, in place, batched over T tokens. The two verbatim lines
// of `Qwen4ExpTextDecoderLayer.forward`:
//   injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
//   hidden_states = hyper_input + injection.flatten(-2)
// i.e. `hyper[t, j*H + h] += block_out[t, h] * injection[t, j]`.
//
// IT IS A RANK-1 UPDATE AND THIS OP IS WHY IT STAYS ONE. Both llama.cpp
// implementations of this architecture materialise it as `repeat_4d` + `mul` —
// a dense [H, hc, T] broadcast built and thrown away at every one of the 96
// per-layer sites — and the composition available from the shared vt:: surface
// would have to do the same, because there is no broadcasting elementwise
// multiply. This op reads `block_out[t, h]` once per (j, h) and
// `injection[t, j]` once per row.
//
// SHAPES. hyper [T, hc*H] (READ-WRITE); block_out [T, H]; injection [T, hc].
void Qwen4ExpGatedResidualWriteBack(Queue& q, Tensor& hyper, const Tensor& block_out,
                                    const Tensor& injection,
                                    const Qwen4ExpGatedResidualArgs& args);

// --- QWEN SPARSE ATTENTION (row MODEL-MM-QWEN4-EXP W5b-4, #2167) ------------
//
// THE INDEXER IS THREE STEPS AND ONLY THE FIRST IS A NEW OP. The other two are
// `vt::DsaIndexerLogits` and `vt::DsaTopkSelect`, called over the BLOCK axis:
//
//   Qwen4ExpQsaCompress(block_keys, raw_keys, k_norm_w, cos, sin, {CR, rot, eps})
//   DsaIndexerLogits(scores, q_index, block_keys, ones, win_start, win_end,
//                    {softmax_scale = index_head_dim ** -0.5, n_head_scale = 1})
//   DsaTopkSelect(block_ids, counts, scores, win_start, win_end)
//   Qwen4ExpQsaGatherAttention(out, q, k, v, block_ids, kv_lens, {scale, CR})
//
// with `win_start[t] = 0` and `win_end[t] = kv_len[t] / compress_ratio` — the
// COMPLETE blocks visible to query token t — and `block_ids` sized
// `token_budget / compress_ratio`. `DsaIndexerLogits`'s fold
// `weights[t,h] * q_scale[t,h] * softmax_scale * n_head_scale` collapses to the
// single constant `index_head_dim ** -0.5` when `weights` is ones and `q_scale`
// is null, and that IS QSA's scale: QSA has neither DeepSeek-V4's learned
// `weights_proj` nor its `n_head ** -0.5`. One reassociation survives and is
// named rather than hidden — upstream divides AFTER the head sum
// (`scores.sum(0) / sqrt(D)`, modeling_qwen4_exp.py:690-693) and the fold
// multiplies BEFORE it, which is `sum_h c*r_h` against `c*sum_h r_h`: equal in
// exact arithmetic, up to an ulp apart in f32, and top-k is invariant under a
// positive scalar, so no selection can move except through a tie created at that
// ulp. `test_qwen4_exp_qsa_device.cpp` measures the composed selection against
// the lane-pinned oracle's own selected sets rather than trusting that argument.
//
// THE POOLED-KEY COMPRESSOR. Per COMPLETE block b of `compress_ratio` raw index
// keys, `Qwen4ExpTextQSAIndexer` (transformers v5.16.0
// modeling_qwen4_exp.py:677-688):
//
//   pooled[b] = mean(raw_keys[CR*b : CR*b + CR])   rounded back to the cache dtype
//   block_keys[b] = rope( k_layernorm(pooled[b]), position = CR*b )
//
// THE MEAN IS UNWEIGHTED AND THE WINDOW DOES NOT OVERLAP, which is the one place
// DeepSeek-V4's compressor must NOT be copied: its pool is
// `score = tl.softmax(score, dim=0); sum(kv * score)` over a window of
// `(1 + OVERLAP) * COMPRESS_RATIO`, driven by a score channel this checkpoint
// has no tensor for, and the CuteDSL C4 variant refuses `overlap=False` at
// compile time. THE ROPE POSITION IS THE BLOCK'S FIRST TOKEN, not its last:
// upstream reads it as `group_starts = block_token_indices[:, 0]` and the Triton
// kernel computes the same value as `compressed_pos = (position // CR) * CR`.
// Taking the last position instead is a silent one-block phase error that no
// shape check can see.
//
// ONLY COMPLETE BLOCKS PRODUCE A STATE. The compressor early-exits unless
// `(position + 1) % compress_ratio == 0`, so `num_keys` must be a multiple of
// `compress_ratio` and the ragged tail costs no state at all — it is attended
// from the RAW KV cache instead, always, whatever the scores said. That floor is
// why the side cache is 64 B/token/layer at the released shape and a per-token
// index cache would be 256.
//
// SHAPES. raw_keys [num_keys, D] f32/bf16, UN-normed and UN-roped exactly as
// `Cache.update_indexer` stores them, `num_keys % compress_ratio == 0`;
// k_norm_weight [D] — the HuggingFace gamma, applied as `(1.0 + weight)`, which
// is upstream's zero-initialised polarity and NOT vLLM's `out * weight`;
// cos/sin [>= num_keys, rotary_dim] f32, the FULL-position tables, of which this
// op reads row `compress_ratio * b`; block_keys [num_keys / compress_ratio, D]
// f32/bf16 OUT. Registered on kCPU (src/vt/cpu/cpu_qwen4_exp_qsa.cpp:354) and
// on kCUDA (src/vt/cuda/cuda_qwen4_exp_qsa.cu:641 — unconditional, at
// preprocessor depth 0). This line read "CPU only; the CUDA arm is owed" until
// 2026-09-12, by which date the CUDA arm had landed; the kQwen4ExpQsaCompress
// paragraph in the OpId enum above already said so, and a header that
// contradicts itself about its own arms is worse than the edit. The kROCM arm
// IS owed and MODEL-MM-QWEN4-EXP W4 owns it; until it lands a ROCm queue
// refuses this op BY NAME through the ordinary GetOp message.
void Qwen4ExpQsaCompress(Queue& q, Tensor& block_keys, const Tensor& raw_keys,
                         const Tensor& k_norm_weight, const Tensor& cos,
                         const Tensor& sin, const Qwen4ExpQsaCompressArgs& args);

// THE GATHER CONSUMER — the point of the wave, and the one piece with no
// upstream counterpart at all. Dense GQA over ONLY the selected raw rows.
//
// For query token t with `kv_len[t]` cached tokens and
// `complete[t] = kv_len[t] / compress_ratio` complete blocks, the attended set is
//
//   { CR*b + i : b in block_ids[t], 0 <= i < CR }  U  [CR*complete[t], kv_len[t])
//
// — the selected blocks expanded to their four real tokens, plus the ALWAYS
// attended ragged tail. `block_ids` is ASCENDING and `-1`-terminated, which is
// exactly what `vt::DsaTopkSelect` emits, so the expansion is ascending too and
// the online softmax reduces over the same positions in the same order dense
// attention would. That is what makes a sub-budget selection BIT-IDENTICAL to
// dense attention rather than merely close: llama.cpp #27742 measures a max
// logit delta of 0.0 over all 2051 such rows, and this op reproduces that
// exactly, which is a free oracle needing no checkpoint.
//
// WHY THIS IS NOT A MASK, AND WHY THAT IS GATED RATHER THAN ASSERTED. A sparse
// mask over a dense cache is CORRECT — it agrees with this op value for value,
// because `exp(-inf - m)` is exactly +0 and adding an exact zero changes no
// accumulator — and it is not faster. llama.cpp #27739 names the mechanism:
// under CUDA flash attention `flash_attn_mask_to_KV_max` scans backwards and
// stops at the first tile that is not all `-inf`, so a mask that keeps any late
// key pays the whole dense prefix. A mask-shaped port therefore passes a token
// gate and forfeits the long-context lever this row exists for. The
// discriminator is `Qwen4ExpQsaAttnArgs::keys_visited`, counted at the key-row
// read; see that field for what it can and cannot see.
//
// WHY IT TAKES BLOCK IDS RATHER THAN TOKEN IDS. Expanding to a materialised
// `[T, token_budget + compress_ratio - 1]` token buffer is the mask-shaped
// intermediate in miniature — at the released config that is 2051 int32 per
// query token, 8 KiB a token, to say what four multiplications of a block id
// say. The expansion is address arithmetic and belongs inside the consumer.
//
// SHAPES. query [T, num_q_heads, head_dim] f32/bf16; key and value
// [max_kv, num_kv_heads, head_dim] f32/bf16, the raw KV cache — or, in the PAGED
// address mode, the rank-4 [num_pages, kv_block_size, num_kv_heads, head_dim]
// unbind views of the runner's flash cache, which are STRIDED (see
// `Qwen4ExpQsaAttnArgs::kv_block_table` for the resolution and for why the two
// arms are one op);
// block_ids [T, block_topk] i32, ascending, `-1` = no block;
// kv_lens [T] i32, the causal visible length per query token;
// out [T, num_q_heads, head_dim] f32/bf16. GQA: num_q_heads % num_kv_heads == 0.
// Registered on kCPU (src/vt/cpu/cpu_qwen4_exp_qsa.cpp:357) and on kCUDA
// (src/vt/cuda/cuda_qwen4_exp_qsa.cu:644 — unconditional, at preprocessor depth
// 0). This line read "CPU only; the CUDA arm is owed" until 2026-09-12, by
// which date the CUDA arm had landed. The kROCM arm IS owed and
// MODEL-MM-QWEN4-EXP W4 owns it; until it lands a ROCm queue refuses this op BY
// NAME through the ordinary GetOp message.
void Qwen4ExpQsaGatherAttention(Queue& q, Tensor& out, const Tensor& query,
                                const Tensor& key, const Tensor& value,
                                const Tensor& block_ids, const Tensor& kv_lens,
                                const Qwen4ExpQsaAttnArgs& args);

// Single-token gated-delta-rule step, one token per sequence
// (gdn-semantics.md §7 decode path). Same math as GdnPrefill with T == B and
// state[B,Hv,Dv,Dk] row b for token b. q_in/k must be l2-normalized by the
// caller (vt::L2Norm) — upstream fuses the l2norm and the §6 g/beta gating
// into the decode kernel; the decomposition is exact (gdn-semantics.md §4,
// §9). g/beta derivation from raw a/b/A_log/dt_bias is M0.9.
// state_idx (optional; mirrors fla fused_recurrent_gated_delta_rule ssm_state_indices):
// when non-null, row bt reads/writes the persistent cache slot state_idx[bt] (state is
// then the FULL [num_slots,Hv,Dv,Dk] cache), so the caller need not gather/scatter
// per-request state rows. When null, state is compact [batch,Hv,Dv,Dk] and row == bt.
void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args,
               const Tensor* state_idx = nullptr);

// SPECULATIVE gated-delta-rule step (SPEC-MTP I4) — the multi-token (T>1),
// slot-snapshotting variant of GdnDecode. Ported from
// vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py @ e24d1b24
// (fused_recurrent_gated_delta_rule_fwd_kernel, IS_VARLEN :66-72 +
// IS_SPEC_DECODING initial-state select :103-116 + INPLACE_FINAL_STATE
// per-timestep snapshot :156-166), as called by
// qwen_gdn_linear_attn.py:1455-1476.
//
// THE MECHANISM (why there is no rollback code anywhere). A speculative step
// runs 1+k query positions for a request, but a later verify may reject some of
// them. Instead of undoing the state, the kernel WRITES A SNAPSHOT PER
// TIMESTEP into a per-request array of k+1 state slots, and the NEXT step
// SELECTS its initial slot by the accepted count:
//
//   initial state  = state[ state_indices[i][ num_accepted_tokens[i] - 1 ] ]
//   after token t  → stored to state[ state_indices[i][t] ]      (t = 0..T-1)
//
// Rejecting at position j is therefore nothing but reading slot j next step.
// Because the recurrence is strictly sequential and slot j is written with the
// state after exactly j+1 tokens, the surviving state is BIT-IDENTICAL to
// running only the accepted prefix — no copy, no undo, no drift.
//
//   q_in/k       [num_tokens, Hk, Dk] ALREADY l2-normalized (like GdnDecode)
//   v            [num_tokens, Hv, Dv]
//   g/beta       [num_tokens, Hv] f32 (log-decay / sigmoid(b))
//   state        [num_slots, Hv, Dv, Dk] FULL persistent cache, in place
//   cu_seqlens   [num_reqs + 1] i32 — request i owns tokens
//                [cu_seqlens[i], cu_seqlens[i+1]) of the (already gathered)
//                spec token stream. Upstream spec_query_start_loc.
//   state_indices [num_reqs, num_cols] i32 ROW-MAJOR, num_cols == k+1.
//                Upstream spec_state_indices_tensor. Local ABI: an entry < 0 is
//                a NULL slot — a NULL INITIAL slot skips the request entirely
//                (output zeroed), a NULL timestep slot skips only that store.
//   num_accepted_tokens [num_reqs] i32, each in [1, num_cols].
void GdnSpecDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                   const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& cu_seqlens,
                   const Tensor& state_indices, const Tensor& num_accepted_tokens,
                   const GdnArgs& args);

// Pure non-spec packed decode, ported from vLLM v0.25.0
// vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-478 @ 702f4814.
// mixed_qkv [B, 2*Hk*Dk + Hv*Dv] and a/b [B,Hv] are last-dimension
// contiguous and may have padded outer row strides. A_log/dt_bias [Hv], out
// [B,Hv,Dv], state [slots,Hv,Dv,Dk], state_idx [B]. mixed_qkv/a/b/out share
// one FP16/BF16/F32 activation dtype. State has an independent floating cache
// dtype, and A_log/dt_bias may independently use any floating dtype; this is
// the real Qwen3.6 contract (BF16 activations/output, FP32 SSM state, FP32
// A_log, BF16 dt_bias). The op normalizes raw q/k in F32, rounds sigmoid(b)
// through b.dtype, applies args.scale to q, and updates state in place. Local
// cache ABI: state_idx < 0
// zeros/skips the row; slot 0 is valid. Live CUDA indices are engine metadata
// and must be unique and in range before upload; the kernel also bounds-checks
// every slot without adding a capture-breaking host synchronization. CPU calls
// validate both properties directly.
void GdnPackedDecode(Queue& q, Tensor& out, const Tensor& mixed_qkv,
                     const Tensor& a, const Tensor& b, const Tensor& a_log,
                     const Tensor& dt_bias, Tensor& state,
                     const Tensor& state_idx, const GdnArgs& args);

// Indexed persistent-state cache boundary used by GDN mixed prefill. `cache`
// is [num_slots,...] f32 or bf16, `state_idx` is i32 [N], and `working` is the
// compact f32 [N,...] state consumed by CausalConv1dFwd/GdnPrefill. Gather
// fuses cache indexing + BF16->F32 conversion in one launch; optional
// has_initial_state (i8 or i32 [N]) zeros fresh rows while gathering. Scatter
// performs the inverse indexed F32->cache-dtype store in one launch. Cache rows
// not named by state_idx are untouched.
void GdnStateGather(Queue& q, Tensor& working, const Tensor& cache,
                    const Tensor& state_idx,
                    const Tensor* has_initial_state = nullptr);
void GdnStateScatter(Queue& q, Tensor& cache, const Tensor& working,
                     const Tensor& state_idx);

// Row gather over dim 0 (torch `index_select(0, idx)`): out[i, ...] =
// in[idx[i], ...]. `idx` is i32 [M] on the same device; out is [M, D...]
// contiguous; in is [N, D...] and MAY carry an outer row stride (a padded /
// row-strided packed view), but its inner dims must be contiguous. in/out share
// dtype (any elementwise dtype). Powers the MIXED spec+non-spec GDN split
// (qwen_gdn_linear_attn.py:1334-1335,1407-1408).
void IndexSelect(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx);

// ── V-head re-indexing of a GDN projection vector ──────────────────────────
//
// WHY THIS EXISTS. The `qwen4exp` converter writes the Gated DeltaNet V heads
// TILED (`t = r * num_key_heads + k`) while every downstream GDN kernel and
// every HuggingFace reference indexes them GROUPED (`g = k * heads_per_key + r`,
// so value head g belongs to key head `g / heads_per_key` — the rule the
// recurrence hard-codes). The loader used to undo that on the WEIGHT, which
// forces a k-quant superblock to be dequantized because the permutation moves
// elements the superblock spans. For `out = W x` a ROW permutation satisfies
// `(P W) x = P (W x)` and a COLUMN permutation satisfies `(W P) x = W (P x)`,
// so the same re-indexing on the VECTOR leaves the weight verbatim. This op is
// that re-indexing.
//
// `out` and `in` are [T, N] with N == prefix_elems + num_key_heads *
// heads_per_key * head_width. Both are inner-contiguous and MAY carry an outer
// row stride on dim 0; they must not alias. Elements [0, prefix_elems) are
// copied straight through — that is `in_proj_qkv`'s leading q and k rows, which
// the converter does NOT tile. Any elementwise dtype; in/out share it.
//
// `inverse == false` maps TILED -> GROUPED: `out[g] = in[r * K + k]` for
// `g = k * R + r`. That is the four ROW-permuted projections (`attn_qkv`'s V
// rows, `attn_gate`, `ssm_beta`, `ssm_alpha`), permuted on their OUTPUT.
// `inverse == true` maps GROUPED -> TILED: `out[t] = in[k * R + r]` for
// `t = r * K + k`. That is `ssm_out`, whose permutation is on its COLUMNS and
// therefore on its INPUT.
void VHeadPermute(Queue& q, Tensor& out, const Tensor& in,
                  const VHeadPermuteArgs& args);

// Row scatter over dim 0 (torch `index_copy_(0, idx, src)`): out[idx[i], ...] =
// in[i, ...]. `idx` is i32 [M]; in is [M, D...] contiguous; out is [N, D...]
// contiguous. Rows of out not named by idx are untouched. Merges the per-group
// GDN core outputs back into their original positions
// (qwen_gdn_linear_attn.py:1570-1571).
void IndexCopy(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx);

// --- MoE (sparse mixture-of-experts) ops. Formula reference:
// .agents/specs/moe-semantics.md. The expert MLP itself is NOT an op — it is composed
// in the layer/runner from Matmul + SiluAndMul (§4). These two ops cover the
// pieces composition cannot express: the router top-k/normalize and the
// weighted scatter-combine.

// Router top-k (moe-semantics.md §3, upstream topk_softmax + fused_topk).
// logits [T,E] any float dtype; softmax computed in f32 over ALL E experts,
// greedy top-k (weights emitted in descending order per token), lowest expert
// index wins ties, then optional renormalize (divide the k probs by their sum,
// denom<=0 -> 1 guard). weights [T,top_k] f32, indices [T,top_k] i32.
//
// --- GROUPED-TOPK / `noaux_tc` (W3) ----------------------------------------
// When `args.num_expert_group > 0` this instead runs the DeepSeek grouped-topk
// router, a 1:1 port of
// vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:80-161 @
// pin e24d1b24 (the `native_impl`/`forward_native` path — the fully fused
// `ops.grouped_topk` CUDA path at `:28-70` is an OPTIMIZATION of the same
// formula, gated on `VLLM_USE_FUSED_MOE_GROUPED_TOPK` + sigmoid + a bias, and is
// not a different result):
//   1. scores = softmax(logits, -1) | sigmoid(logits)          (:110-117)
//   2. with a bias: original_scores = scores; scores += bias    (:120-124)
//        group_score[g] = SUM of the TOP-2 scores in group g    (:124-126)
//      without a bias:
//        group_score[g] = MAX score in group g                  (:128-131)
//   3. keep the top `topk_group` groups, mask the rest to -inf  (:133-145)
//   4. top-k over the masked scores; with a bias the WEIGHT is read from the
//      UNBIASED `original_scores` at the selected ids — the biased score selects,
//      the unbiased score weights                               (:147-150)
//   5. optional renormalize (:156-157), then routed_scaling_factor (:159-160).
//
// `e_score_correction_bias` [E] f32 is upstream's optional learned gate bias
// (deepseek_v2.py:313-318 — created ONLY for `topk_method == "noaux_tc"`, hence
// nullptr for DeepSeek-V2-Lite and for every Qwen model). Passing it with
// `num_expert_group == 0` is rejected: upstream only ever reaches the bias
// asymmetry through the grouped path.
//
// DETERMINISM DEVIATION (recorded): upstream uses `torch.topk`, whose tie order
// is unspecified (`sorted=` is only forced under VLLM_BATCH_INVARIANT). We keep
// our house convention — greedy selection with a strict `>` scan over ascending
// index, so the LOWEST expert index wins an exact tie — for BOTH the group
// selection and the expert selection. This is the same rule the ungrouped router
// already uses, and it is what makes CPU and CUDA agree bit-for-bit.
void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args,
                   const Tensor* e_score_correction_bias = nullptr);

// Weighted scatter-combine of the per-expert outputs (moe-semantics.md §4/§6).
//   out[t,:] = routed_scale * sum_j weights[t,j] * expert_out[t,j,:]  (f32 accum)
//              + shared[t,:]                            (when shared != nullptr)
// expert_out [T,K,H] any float dtype (the K per-slot expert MLP outputs for
// token t), weights [T,K] f32 (router weights, §3), optional shared [T,H] any
// float dtype (the shared-expert term, §5). out [T,H] f32 or bf16. The routed
// f32 sum is stored at out's dtype; the shared term is added in that same store
// (§6 combine order: shared_output + routed_output). The activation-dtype
// rounding of the routed sum before the shared add is carried by the caller
// materializing expert_out/shared in the activation dtype.
//
// `routed_scale` is upstream's `apply_routed_scale_to_output=True` arm
// (layers/fused_moe/runner/moe_runner.py:390-407, :402-406 `fused_output *=
// routed_scaling_factor`, then :722-725 `result = shared_output + fused_output`).
// It multiplies the ROUTED sum ONLY — the shared-expert term is added unscaled,
// which is the whole point of the flag and the error a token gate catches late.
// The DEFAULT 1.0f is the `apply_routed_scale_to_output=False` polarity every
// landed caller uses, where the factor is instead folded into the router weights
// by MoeRouterTopKArgs::routed_scaling_factor (layer.py:291-300 forces the
// router's factor to 1.0 exactly when this one is not).
void MoeCombine(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                const Tensor* shared = nullptr, float routed_scale = 1.0f);

// --- Fused MoE combine + shared-expert gate (MoE glue fusion). Equivalent to
// SharedExpertGate(shared=bf16(sigmoid(gl)*sd)) followed by MoeCombine(...,shared),
// but in ONE launch: the shared term is gated inline (sigmoid(gl[t])*sd[t,c],
// rounded through bf16 exactly as SharedExpertGate's store, then re-added in f32)
// and folded into the top-k weighted reduction. Removes the separate
// SharedExpertGate launch and the shared [T,H] global round-trip. Bit-identical
// to the two-kernel path. sd [T,H] f32, gl [T,1] f32.
void MoeCombineGate(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor& sd, const Tensor& gl);

// --- Dense causal attention (M0.9). Formula reference:
// .agents/specs/qwen36-forward-notes.md §5 (pinned Qwen3NextAttention core).
//
// Causal scaled-dot-product attention with GQA broadcast over a single packed
// sequence. query [T,Hq,D], key/value [T,Hk,D], out [T,Hq,D]; Hq a multiple of
// Hk (q-head h reads kv-head h / (Hq/Hk)). q/k arrive ALREADY qk-normed and
// RoPE'd (compose vt::RmsNorm + vt::RopeNeox upstream); v is raw. The output
// gate (sigmoid) is applied by the caller (it is elementwise on the projection
// split, not attention math). Per q-head h (kv-head g), query i:
//   s[j] = scale * (query[i,h] · key[j,g])   for j <= i (causal), else -inf
//   p    = softmax_j(s)                       (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// f32 or bf16 in, f32/bf16 out; all softmax/accumulation math in f32.
void Attention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
               const Tensor& value, const AttentionArgs& args);

// --- Dense non-causal CROSS attention (LTX-2.5 L2). See AttentionCrossArgs for
// why vt::Attention cannot serve it. Per q-head h (kv-head g = h/(Hq/Hkv)),
// query i in [0,Tq), keys j in [0,S):
//   s[j] = scale * (query[i,h] · key[j,g]) + bias[i or 0, j]
//   p    = softmax_j(s)                       (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// `bias` is an OPTIONAL rank-2 additive score bias [Tq, S] or [1, S] (the
// broadcast key-only form a padding mask produces), f32, on the same device;
// nullptr means no bias. It is additive in torch SDPA's sense — upstream builds
// it as `(mask - 1) * finfo.max` for the prompt mask (transformer_args.py:204)
// and as log-space attenuation for the self-attention strength mask (:232-237),
// so a fully masked key reaches the softmax as a large negative number, NOT as
// -inf, and an all-masked row degenerates to a uniform average exactly as torch's
// does. All softmax/accumulation math is f32.
//
// BACKENDS, recorded so it cannot be discovered later: this op ships with a CPU
// kernel (`AttentionCrossKernel`, src/vt/cpu/cpu_ops.cpp) AND — since phase L8,
// 2026-08-12 — a NATIVE CUDA kernel (src/vt/cuda/cuda_attention_cross.cu). An
// earlier revision of this paragraph recorded the CUDA one as OWED "alongside the
// LTX-2.5 device-resident forward, which is the first caller that would need it";
// that caller arrived and so did the kernel, in the same change, and a CUDA
// device now has a real provider rather than a refusal or a unified-memory
// fallback to the host.
//
// The CUDA kernel is a structural port of `AttentionDenseFlashKernel`
// (src/vt/cuda/cuda_ops.cu), generalized on the three axes AttentionCrossArgs
// exists for. It uses the online-softmax recurrence where the CPU kernel uses the
// explicit three-pass max/exp/normalize, so the two agree to f32 summation-order
// slack and are NOT bit-identical — the same relationship `AttentionDenseFast`
// already has with `AttentionKernel`.
//
// A device with NO provider — kXPU, say — still refuses through `GetOp`, naming
// this op via `vt::OpName` rather than falling back. Callers route on what a call
// MEANS, not on whether its numbers happen to be square, so that refusal is
// deterministic per call site instead of per prompt length — see Ltx2Attention
// (src/vllm/model_executor/models/ltx2.cpp).
void AttentionCross(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                    const Tensor& value, const Tensor* bias, const AttentionCrossArgs& args);

// --- Conformer / FastConformer audio-encoder kernels -------------------------
// Spike: .agents/specs/parakeet-conformer-encoder.md (rows P1/P2/P3). Upstream
// mirror: transformers 5.3.0 transformers/models/parakeet/modeling_parakeet.py,
// which is what vLLM itself runs (vllm/model_executor/models/parakeet.py:37,62
// delegate to `from transformers import ParakeetEncoder`); the structurally
// identical vLLM-NATIVE conformer at
// vllm/model_executor/models/conformer_encoder.py is cited alongside where it
// agrees, and its one arithmetic divergence is a flag (AttentionRelPosArgs).
//
// P1 — torch `nn.Conv2d`, the encoder front end.
//   out    [N, Cout, Hout, Wout]
//   x      [N, Cin,  Hin,  Win ]
//   weight [Cout, Cin/groups, KH, KW]   (torch's exact parameter layout)
//   bias   optional rank-1 [Cout]
// Hout = (Hin + 2*pad_h - dilation_h*(KH-1) - 1)/stride_h + 1, likewise Wout.
// `groups` must divide both Cin and Cout; output channel oc reads input group
// oc/(Cout/groups). Zero padding: taps outside [0,Hin)x[0,Win) are SKIPPED.
// Per output element the reduction is a SINGLE f32 accumulator walked strictly
// in (ic, kh, kw) order with the bias added LAST, so the result does not depend
// on the thread count and is byte-reproducible; the in-test scalar reference in
// tests/vt/test_ops_conv2d.cpp gates exactly that order.
//
// This is the op `ParakeetEncoderSubsamplingConv2D` (modeling_parakeet.py:357)
// is built from: a dense 1->C k/stride/pad conv (:369-371), then per extra
// stage a DEPTHWISE C->C conv (groups=C, :377-386) and a POINTWISE 1x1 conv
// (:388). vLLM's native `Conv2dSubsampling` (conformer_encoder.py:18) is the
// same stack. It also REPLACES the host `std::vector<float>` Conv2dK3S2P1 loop
// in src/vllm/model_executor/models/gemma4_audio.cpp:92, which stays as an
// independent correctness reference for that model's prefix.
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void Conv2d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv2dArgs& args);

// --- General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007) --------------
// Spec: .agents/specs/ltx25-device-residency.md `## W5`. Upstream mirror:
// torch `nn.Conv3d` as `CausalConv3d` instantiates it — Lightricks/LTX-2 @
// fd4ded7f2, packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:292-302,
// called once at :312, which is the WHOLE of the LTX-2.5 video VAE's arithmetic.
//
//   out    [Cout, Tout, Hout, Wout]
//   x      [Cin,  Tin,  Hin,  Win ]
//   weight [Cout * Cin/groups, KT, KH, KW]
//   bias   optional rank-1 [Cout]
// Tout = (Tin + 2*pad_t - dilation_t*(KT-1) - 1)/stride_t + 1, likewise H and W.
// `groups` must divide both Cin and Cout; output channel oc reads input group
// oc/(Cout/groups). Zero padding: taps outside the input extent are SKIPPED.
//
// BATCH 1, DELIBERATELY. torch's tensor is [N, Cin, T, H, W] and this one drops
// the N axis, because `vt::Tensor` caps rank at 4 (include/vt/tensor.h:12) and
// raising that cap changes the one struct every op in the tree passes. Batch 1
// is what the video VAE decodes. A caller that needs N > 1 gets a refusal by
// name from the wrapper, never a silently folded axis.
//
// THE WEIGHT'S TWO LEADING AXES ARE MERGED, for the same rank reason: torch's
// [Cout, Cin/groups, KT, KH, KW] is rank 5, and [Cout * Cin/groups, KT, KH, KW]
// is the same bytes in the same order. `Cout` comes from `out` and `Cin/groups`
// from `x` and `args.groups`, so nothing is lost and the wrapper CHECKS the
// product rather than trusting it.
//
// ACCUMULATION ORDER — CONTRACT, NOT DETAIL, AND NOT kConv2d's. Per output
// element: one f32 accumulator seeded with the bias, then ONE f32 PARTIAL PER
// INPUT CHANNEL walked strictly in (kt, kh, kw) order and added into it. That is
// what torch's f32 convolution does — it is a blocked GEMM and sums one partial
// per channel block — and it is what every committed LTX-2.5 video VAE golden
// was taken through. kConv2d's single flat accumulator is a DIFFERENT number:
// src/vllm/model_executor/models/ltx2_video_vae.cpp measures the flat order
// pushing the non-causal tiled golden to 5.00679e-06 against a 5e-06 tolerance.
// The two ops are siblings for exactly the reason kConv1d and kDepthwiseConv1d
// are, and neither may be re-pointed at the other.
//
// The order is why the CPU and CUDA arms are BYTE-IDENTICAL rather than close:
// the device kernel walks the same sweep with __fmul_rn/__fadd_rn, and the host
// side is already pinned against FMA contraction by -ffp-contract=off
// (CMakeLists.txt). tests/vt/test_ops_conv3d.cpp gates the order against an
// independent in-test scalar reference.
//
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void Conv3d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv3dArgs& args);

// P2 — NON-CAUSAL depthwise `nn.Conv1d(C, C, K, stride, padding, groups=C)`, the
// conformer convolution module's temporal mixer.
//   out    [N, C, Lout]
//   x      [N, C, Lin]
//   weight [C, 1, K] (torch's depthwise parameter shape) or [C, K]
//   bias   optional rank-1 [C]
// Lout = (Lin + 2*padding - dilation*(K-1) - 1)/stride + 1. Zero padding: taps
// outside [0,Lin) are SKIPPED. Per output element a SINGLE f32 accumulator over
// k in increasing order, bias added LAST — thread-count independent and
// byte-reproducible, gated in tests/vt/test_ops_conv1d_depthwise.cpp.
//
// Upstream: `ParakeetEncoderConvolutionModule.depthwise_conv`
// (modeling_parakeet.py:116, constructed :138-146 with padding=(K-1)//2 :136,
// applied :180); vLLM native `ConformerConvolution.depthwise_conv`
// (conformer_encoder.py:229). DELIBERATELY a sibling of, never a modification
// of, vt::CausalConv1dFwd (ops.h kCausalConv1dFwd): that op is causal, carries a
// persistent conv_state across steps and folds a SiLU; this one is centre-padded,
// stateless and activation-free, exactly as the conformer module wants.
// x/weight/bias f32/f16/bf16; out f32/f16/bf16; all math in f32.
void DepthwiseConv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const DepthwiseConv1dArgs& args);

// --- BigVGAN / DAC vocoder 1-D convolutions (#672, spec
// .agents/specs/minimax-music3.md §11.4). ------------------------------------
//
// These two are the whole convolution vocabulary of `vllm::vocoder1d`, the
// shared 1-D BigVGAN core that MiniMax-Music3's vocoder, MiniMax-H3's audio VAE,
// LTX-2.5's audio VAE and IndexTTS-2.5 all decode through
// (include/vllm/model_executor/models/vocoder1d.h). The transposed one is 88.5 %
// of MiniMax-Music3's acoustic-half profile and `vt` had no op of any kind that
// could express it, which is why the whole stage had no device kernel to route
// to.
//
// Upstream mirror: `torch.nn.functional.conv1d` / `conv_transpose1d` as the
// checkpoints instantiate them —
//   * MiniMax-Music3 `minimax_music3_vocoder.py:42,44,55,89,98`
//     (`nn.Conv1d` at :42/:44/:89/:98, `nn.ConvTranspose1d` at :55),
//   * LTX-2.5 `audio_vae/vocoder.py:104-184` (the alias-free resample pair) and
//     its BigVGAN conv_pre/conv_post,
//   * MiniMax-H3's DAC audio VAE (`dac_alias_free_resample.py`).
//
//   Conv1d              out    [N, Cout, Lout]
//                       x      [N, Cin,  Lin ]
//                       weight [Cout, Cin/groups, K]        (torch's layout)
//                       bias   optional rank-1 [Cout]
//     Lout = (Lin + 2*padding - dilation*(K-1) - 1)/stride + 1
//
//   ConvTranspose1d     out    [N, Cout, Lout]
//                       x      [N, Cin,  Lin ]
//                       weight [Cin, Cout/groups, K]        (torch's layout —
//                              note dim 0 is the INPUT channel here)
//                       bias   optional rank-1 [Cout]
//     Lout = (Lin-1)*stride - 2*padding + dilation*(K-1) + 1 + output_padding
//
// `groups` must divide both Cin and Cout. Zero padding is realised by SKIPPING
// out-of-range taps.
//
// THE NUMERIC CONTRACT, which is the load-bearing part.
//
// (1) Every output element owns ONE f32 accumulator. Not f64. That is what
//     torch accumulates a float convolution in, MEASURED rather than read: on a
//     27-tap `[+1e8, 0.1 x 25, -1e8]` probe, where an f32 accumulator lands on
//     exactly 0.0 in ANY summation order and an f64 one lands near 2.5,
//     `F.conv1d` returns 0.0 at f32 and at bf16 and `F.conv_transpose1d`
//     returns 0.0 at f32 (torch 2.11.0+cu130). vLLM owns neither op at the
//     parity pin `555967922` — it drops the vocoder it would otherwise own,
//     `qwen3_omni_moe_thinker.py:1975` — so torch is the reference here through
//     the per-consumer secondary oracles, and where vLLM DOES own a convolution
//     it states the same polarity itself (`csrc/cpu/mamba_kernels.hpp`,
//     "Accumulate in float32 for precision").
//
//     THIS WAS f64 UNTIL VT-CONV1D-F32-ACC (#1474), justified by a claim that
//     did not hold. The host reference these replaced did accumulate in double
//     (`src/vllm/model_executor/models/vocoder1d.cpp` @ 8fa405bb7), but the
//     goldens were never taken through it at that width: all four consumers'
//     generators run torch in f32 (`scripts/gen-bigvgan-goldens.py:48` builds
//     f64 and then calls `.float()`; `gen-ltx2-vae-goldens.py:223,234` and
//     `gen-minimax-music3-acoustic-goldens.py:81,134` cast with
//     `astype(np.float32)`), so this op was WIDER than the oracle its own
//     goldens came from. The clause also cited `tests/parity/goldens/`, a
//     directory that contains no vocoder, BigVGAN, LTX-2.5 VAE, FVQ or
//     general-conv1d golden at all — they are `.inc` headers beside their tests
//     (`tests/vllm/models/bigvgan_goldens.inc`, `ltx2_vae_goldens.inc`,
//     `minimax_music3_acoustic_goldens.inc`). An uncheckable citation is how
//     the first claim survived. Narrowing moved the port TOWARD its goldens:
//     over 194 arms, 182 unchanged, 10 improved, 2 one ULP worse and three or
//     more decimal orders inside their bounds.
//     .agents/specs/vt-conv1d-f32-accumulator.md.
//
// (2) THE VISIT ORDER IS PINNED, not merely the value.
//     Conv1d accumulates over (ic ascending, k ascending) with the bias seeded
//     FIRST — `acc = bias; for ic: for k: acc += x*w`.
//     ConvTranspose1d accumulates over (ic ascending, then input position t
//     ascending, taking the single tap k with `t*stride + k*dilation == p`) with
//     the bias added LAST. That is the sequence of additions the host scatter
//     performed into each destination cell, which is what lets the gather-form
//     kernels here be BIT-IDENTICAL to each other rather than merely close. The
//     ORDER is the host loop's; since #1474 the WIDTH is not, so the identity
//     holds between the providers here and not against that loop.
//
// (3) ConvTranspose1d SKIPS an input whose value compares equal to 0.0, exactly
//     as the host loop did. That is not an optimisation that may be dropped: it
//     decides the sign of a zero output cell, because (-0.0) + (+0.0) == +0.0
//     while (-0.0) alone is -0.0.
//
// Parallelism partitions OUTPUT elements only, so results do not depend on the
// thread count or the launch geometry. Gated in
// tests/vt/test_ops_conv1d_general.cpp and tests/vllm/models/test_host_parallel.cpp
// against in-test serial references that walk the declared order. Those
// references were verbatim copies of the pre-op host loop until #1474 narrowed
// them in lockstep with the kernels, so what they now assert is that the ORDER
// did not move, not that the arithmetic matches a historical loop. Clause (1)
// is what gates the WIDTH, and it is gated against torch's own answer rather
// than against a copy of ourselves — `tests/vllm/models/test_host_parallel.cpp`,
// `vocoder1d Conv1d / ConvTranspose1d accumulates in f32, which is what torch
// does`, entering through the production `vllm::vocoder1d::*` entry point.
//
// (4) THE CUDA PROVIDER IS BYTE-IDENTICAL TO THE CPU ONE, not merely close.
//     Both are one f32 accumulator per output element walked in the order
//     above; the host is compiled `-ffp-contract=off` (CMakeLists.txt:40-56)
//     and the device kernel uses `__fmul_rn`/`__fadd_rn`, so every operation on
//     both arms is an IEEE single multiply or add with round-to-nearest-even on
//     the same values in the same sequence. The gate asserts `memcmp` equality,
//     not a tolerance. **UNVERIFIED at the narrowed width**: #1474 had no CUDA
//     toolkit and no lease, so this clause rests on the construction and not on
//     a run. Owed, and named as owed:
//     .agents/specs/vt-conv1d-f32-accumulator.md §7.
//
// x/weight/bias/out are f32 only. f16/bf16 are REFUSED with a message naming
// the gap rather than silently widened — no consumer has them and no golden
// covers them (owed: .agents/specs/minimax-music3.md §11.4).
void Conv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv1dArgs& args);

void ConvTranspose1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const ConvTranspose1dArgs& args);

// Output length for the given input length and args — the single definition of
// torch's shape arithmetic, so a caller sizing an output buffer and the op
// validating it can never disagree. Returns <= 0 when the geometry is empty.
int64_t Conv1dOutLength(int64_t in_len, int64_t kernel, const Conv1dArgs& args);
int64_t ConvTranspose1dOutLength(int64_t in_len, int64_t kernel, const ConvTranspose1dArgs& args);

// P3 — Transformer-XL relative-position ENCODER self-attention. No KV cache, no
// paging, no RoPE, non-causal: every existing vt attention path is a decoder
// path and none of them can express this.
//   out      [T, Hq,  D]
//   query    [T, Hq,  D]   raw q projection (bias_u/bias_v added here)
//   key      [T, Hkv, D]   Hq a multiple of Hkv (GQA broadcast, as vt::Attention)
//   value    [T, Hkv, D]
//   rel_key  [P, Hq,  D]   P == 2*T-1, the projected relative-position keys
//   bias_u   optional [Hq, D]  global CONTENT bias   (term (c))
//   bias_v   optional [Hq, D]  global POSITION bias  (term (d))
//   key_mask optional rank-1 [T] i8/i32, 1 = valid key, 0 = masked to -inf
// Per q-head h (kv-head g = h/(Hq/Hkv)), query i, key j:
//   ac = Σ_d (query[i,h,d] + bias_u[h,d]) * key[j,g,d]              (a)+(c)
//   bd = Σ_d (query[i,h,d] + bias_v[h,d]) * rel_key[T-1-i+j, h, d]  (b)+(d)
//   s[j] = scale*ac + scale*bd     (or (ac+bd)*scale, see scale_after_sum)
//   p    = softmax_j(s)            (f32, max-subtracted)
//   out[i,h] = Σ_j p[j] * value[j,g]
// The `T-1-i+j` index IS the `_rel_shift` of both upstreams, closed-form. Proof:
// _rel_shift left-pads the [T, 2T-1] matrix_bd by one column, reinterprets it as
// [2T, T], drops the first row and reinterprets as [T, 2T-1]; element (i,p) then
// carries flat index i*(2T-1)+p+T, whose (row, col) in the [T, 2T] padded view is
// (i, T-i+p) because 1 <= T-i+p <= 2T-1 for all i,p in [0,T). Column c>=1 of the
// padded view is original column c-1, so shifted(i,p) = raw(i, T-1-i+p); the
// `[..., :T]` truncation then leaves p == j. HF (modeling_parakeet.py:349-355,
// truncated :320) and vLLM native (conformer_encoder.py:179-186, truncated to
// T2//2+1 == T) perform the identical map, so the closed form serves both.
//
// Deviation, recorded: upstream MATERIALIZES query+bias_u and query+bias_v as
// activation-dtype tensors (:295-300 / conformer_encoder.py:205-206); we add the
// bias inside the kernel in f32, which is at least as accurate but rounds
// differently for a bf16/f16 activation. Pass bias_u/bias_v == nullptr and a
// pre-biased query to reproduce upstream's rounding exactly.
//
// A query row whose every key is masked yields ZEROS rather than upstream's NaN
// (softmax over an all -inf row); such a row is itself padding and its output is
// discarded. Recorded deviation, asserted in tests/vt/test_ops_attn_relpos.cpp.
// Reductions are strictly sequential per output element => thread-count
// independent and byte-reproducible. f32/f16/bf16 in, f32/f16/bf16 out; all
// softmax and accumulation math in f32.
void AttentionRelPos(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const Tensor& rel_key, const Tensor* bias_u,
                     const Tensor* bias_v, const Tensor* key_mask,
                     const AttentionRelPosArgs& args);

// Same contract and math as Attention (dense full/causal, GQA broadcast, f32
// online-softmax), but the CUDA impl is WARP-scoped (one warp per (query,head),
// __shfl head_dim reduction, register accumulator, no __syncthreads) — the fast
// path for small head_dim dense attention (the Qwen3-VL vision tower). NOT
// bit-identical to Attention (different head_dim partial-sum grouping), same f32
// math; adopt per token-exact gate. On CPU it dispatches to the SAME kernel as
// Attention (byte-identical there). Separate op so kAttention stays untouched.
void AttentionDenseFast(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                        const Tensor& value, const AttentionArgs& args);

// Same contract and math as AttentionDenseFast, but the CUDA impl is FLASH-TILED:
// a block of query-warps SHARES each streamed K/V tile out of shared memory (classic
// FlashAttention K/V tiling, ported in structure from src/vt/cuda/flash_attn/src/
// flash_fwd_kernel.h), eliminating the O(t^2) redundant global K/V reads that make
// AttentionDenseFast memory-bound on long non-causal contexts (the Whisper audio
// encoder, 1500 frames × 32 layers). The per-warp online-softmax arithmetic and its
// order are UNCHANGED from AttentionDenseFast, so the CUDA output is BIT-IDENTICAL to
// it (K/V bytes merely sourced from shared memory) ⇒ token-identical by construction.
// On CPU it dispatches to the SAME kernel as Attention (byte-identical). Separate op
// so kAttention / kAttentionDenseFast stay untouched.
void AttentionDenseFlash(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args);

// The head_dim domain AttentionDenseFlash can actually LAUNCH, as arithmetic a
// host without a GPU can execute and a test can pin (#1544).
//
// The tiling is what bounds it: each CTA stages `kAttentionDenseFlashTileCols`
// columns of BOTH K and V in dynamic shared memory, so it asks the driver for
// `2 * cols * head_dim * sizeof(input element)` bytes. There is no
// `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
// anywhere in src/vt/cuda/, so that request is capped at CUDA's DEFAULT 48 KiB per
// block on every architecture — not at the 256 the kernel's register blocking
// allows, which is what this op used to advertise on its own. head_dim 256 in bf16
// wants 64 KiB and in f32 wants 128 KiB; both are refused by the driver, and the
// caller used to learn that from a bare launch error one `cudaGetLastError` later.
//
// This mirrors what vLLM makes every backend declare — `get_supported_head_sizes`
// / `supports_head_size`, vllm/v1/attention/backend.py:155-163 @ 555967922 — where
// the domain is consulted BEFORE dispatch rather than discovered by launching.
// AttentionDenseFast (kAttentionDenseFast) uses NO shared memory and is the rung
// that serves head_dim above the bound below.
inline constexpr int64_t kAttentionDenseFlashTileCols = 64;
// The register blocking: 8 elements per lane across 32 lanes.
inline constexpr int64_t kAttentionDenseMaxHeadDim = 256;
// CUDA's default per-block dynamic shared-memory cap, and it is INCLUSIVE. That
// direction matters: head_dim 192 in bf16 lands on exactly 49152 bytes and launches
// today, so an exclusive bound would refuse work that currently runs.
inline constexpr int64_t kCudaDefaultDynamicSmemBytes = 49152;

// Bytes of dynamic shared memory one CTA requests for `head_dim` at `elem_size`.
constexpr int64_t AttentionDenseFlashSmemBytes(int64_t head_dim, int64_t elem_size) {
  return 2 * kAttentionDenseFlashTileCols * head_dim * elem_size;
}

// The largest head_dim AttentionDenseFlash can launch for an input element size.
// bf16 -> 192, f32 -> 96.
constexpr int64_t AttentionDenseFlashMaxHeadDim(int64_t elem_size) {
  if (elem_size <= 0) return 0;  // no element size, no admissible head_dim
  const int64_t by_smem =
      kCudaDefaultDynamicSmemBytes / (2 * kAttentionDenseFlashTileCols * elem_size);
  return by_smem < kAttentionDenseMaxHeadDim ? by_smem : kAttentionDenseMaxHeadDim;
}

// Same contract as AttentionDenseFlash, but the CUDA impl runs the VENDORED
// FlashAttention-2 forward (src/vt/cuda/flash_attn/) on its tensor cores instead of a
// scalar per-warp recurrence — the kernel vLLM itself dispatches for dense non-causal
// encoder self-attention (vllm/model_executor/models/whisper.py
// WhisperEncoderAttention:255 -> forward:298-317 -> flash_attn_varlen_func).
//
// NOT bit-identical to AttentionDenseFast/Flash: `mma.sync` reassociates the QK^T and
// PV reductions, so results differ within the bf16 envelope and adoption is gated on
// the token-exact / ratified near-tie gate, never assumed. The FA-2 fast path applies
// only to bf16, head_dim 64 or 128, non-causal, MHA (h_k == h) on CUDA with the
// vendored kernels compiled — that pair is the set of compiled NON-SPLIT
// instantiations, `run_mha_fwd_<bfloat16_t, {64,128}, false>`, and the head dim is a
// template parameter rather than an argument, so the set does not widen by editing a
// comparison. Every other shape falls back to kAttentionDenseFlash, which is why this
// op is safe to call generically. On CPU it dispatches to the SAME kernel as
// Attention. Separate op so kAttention / kAttentionDenseFast / kAttentionDenseFlash
// stay untouched.
void AttentionDenseFa2(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                       const Tensor& value, const AttentionArgs& args);

// DFlash in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL) — the project's FIRST
// non-causal / bidirectional attention primitive. See DFlashBlockAttentionArgs for
// the full contract. Per request block [cu_seqlens[r], cu_seqlens[r+1]), each query
// attends to keys IN THAT BLOCK: BIDIRECTIONAL when args.causal is false (the
// DFlash full-attention layers, qwen3_dflash.py _resolve_layer_attention -> causal
// False), causal-within-window when args.causal is true (the SWA layers). f32
// online softmax, GQA broadcast. A SEPARATE op from vt::Attention/PagedAttention so
// the causal decoder path used by every other model is byte-identical. CUDA impl
// mirrors AttentionKernel's block-reduction recurrence; CPU is the reference.
void DFlashBlockAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const DFlashBlockAttentionArgs& args);

// DFlash PAGED in-block attention (SPEC-DFLASH D12 Part B) — the capture-safe form
// of DFlashBlockAttention. See DFlashPagedBlockAttentionArgs. Tensors:
//   out          [Nq, Hq, D]           block-query outputs (Nq = sum of block lens)
//   query        [Nq, Hq, D]           the (1+k)*num_reqs block queries
//   block_key    [Nq, Hkv, D]          the block K (same rows as query)
//   block_value  [Nq, Hkv, D]          the block V
//   ctx_key      [num_pages, block_size, Hkv, D]   PAGED context K (normed+RoPE'd)
//   ctx_value    [num_pages, block_size, Hkv, D]   PAGED context V (raw)
//   cu_seqlens   [num_reqs+1] i32       block-query row bounds (device)
//   seq_lens     [num_reqs]   i32       per-request context length C_r (device)
//   block_table  [num_reqs, max_pages] i32  logical-page -> physical-page (device)
// Every metadata tensor is read in place on the op's device (device for CUDA =
// capture-safe; host for CPU). GQA broadcast; f32 online softmax. Bit-identical to
// DFlashBlockAttention over the materialized [context; block] combined buffer.
void DFlashPagedBlockAttention(Queue& q, Tensor& out, const Tensor& query,
                               const Tensor& block_key, const Tensor& block_value,
                               const Tensor& ctx_key, const Tensor& ctx_value,
                               const Tensor& cu_seqlens, const Tensor& seq_lens,
                               const Tensor& block_table,
                               const DFlashPagedBlockAttentionArgs& args);

// DFlash2 grouped dynamic depthwise convolution (SPEC-DFLASH2 W2, #1314). See
// DFlashGroupedConvArgs for the contract and the upstream anchor. Tensors:
//   out           [T, H]                       the convolved sublayer stream
//   x             [T, H]                       the sublayer input/output stream
//   coefficients  [T, sides, taps, num_groups] the per-position kernel DELTAS
//   base          [sides, taps, H]             the checkpoint's base_kernel
// with H == num_groups * group_size and `args.side` in [0, sides). All four share
// one float dtype (bf16 on every published checkpoint). CPU is the authoritative
// reference; CUDA mirrors it bit-for-bit.
void DFlashGroupedConv(Queue& q, Tensor& out, const Tensor& x, const Tensor& coefficients,
                       const Tensor& base, const DFlashGroupedConvArgs& args);

// DFlash2 candidate-selector edge lattice (SPEC-DFLASH2 W3, #1314). See
// Dflash2SelectorEdgesArgs for the contract and the upstream anchor. Tensors:
//   scores          [B, L, K, K] f32   edge(b,l,predecessor,child)
//   pred_codebook   [V, R]             candidate_selector.predecessor_codebook
//   succ_codebook   [V, R]             candidate_selector.successor_codebook
//   candidate_ids   [B, L, K] i64      compute_candidates' ids (target vocab)
//   unary           [B, L, K] f32      compute_candidates' values
//   hidden          [B, L, R]          hidden_projection(final hidden states)
//   anchors         [B] i64            each request's verified anchor token
// The two codebooks and `hidden` share one float dtype (bf16 on every published
// checkpoint); `scores` and `unary` are always f32. CPU is the authoritative
// reference.
void Dflash2SelectorEdges(Queue& q, Tensor& scores, const Tensor& pred_codebook,
                          const Tensor& succ_codebook, const Tensor& candidate_ids,
                          const Tensor& unary, const Tensor& hidden, const Tensor& anchors,
                          const Dflash2SelectorEdgesArgs& args);

// DFlash2 candidate-selector PATH WALK (SPEC-DFLASH2 W4, #1314). See
// Dflash2PathWalkArgs for the contract, the tie-break and the upstream anchor.
// Tensors:
//   tokens        [B, L] i64        the drafted token per (request, step)
//   scores        [B, L, K, K] f32  vt::Dflash2SelectorEdges' output
//   candidate_ids [B, L, K] i64     compute_candidates' ids (target vocab)
// One request is one program: the step loop is INSIDE it, because step l reads
// the predecessor row step l-1 chose.
void Dflash2PathWalk(Queue& q, Tensor& tokens, const Tensor& scores,
                     const Tensor& candidate_ids, const Dflash2PathWalkArgs& args);

// Top-k that EMITS the surviving (id, value) pairs (SPEC-DFLASH2 W3 / D2,
// #1314). See TopKValuesIndicesArgs for the contract, the tie-break and the
// upstream anchor. Tensors:
//   values   [rows, k] f32   the k largest logits of each row, DESCENDING
//   indices  [rows, k] i64   their column indices, ties broken ASCENDING
//   logits   [rows, V] f32   read-only (unlike vt::ApplyTopKTopP, which masks
//                            its input in place)
// `args.k` must be in [1, V - args.num_org_vocab_padding].
void TopKValuesIndices(Queue& q, Tensor& values, Tensor& indices, const Tensor& logits,
                       const TopKValuesIndicesArgs& args);

// --- Paged KV-cache write (M1.6). Semantics ported from the FlashAttention
// path of vllm/csrc/.../cache_kernels.cu::reshape_and_cache_flash @ e24d1b24;
// the NHD cache layout is the one FlashAttentionBackend::get_kv_cache_shape
// allocates (num_blocks, 2, block_size, num_kv_heads, head_size) — NOT the HND
// cpu_attn layout.
//
// Writes each new per-token K/V into the paged cache at its slot id.
//   k / v          [num_tokens, num_kv_heads, head_size]   (source rows)
//   k_cache/v_cache[num_blocks, block_size, num_kv_heads, head_size]  (the two
//                  dim-1 slices of the flash cache; written in place)
//   slot_mapping   [num_slots] i64  (num_slots <= num_tokens; the tail of k/v is
//                  CUDA-graph padding and is ignored)
// For token t with slot s = slot_mapping[t]: block = s / block_size,
// offset = s % block_size, and k[t] is copied to k_cache[block, offset, :, :]
// (all kv-heads, all head_size). A slot s < 0 skips token t (padding). The copy
// is a raw element copy (the "auto" cache path: cache dtype == k/v dtype, no
// fp8 scaling — fp8 KV cache is out of T0 scope).
void ReshapeAndCache(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                     Tensor& v_cache, const Tensor& slot_mapping);

// --- fp8 KV-cache write (KV-FP8 W1). The fp8 sibling of ReshapeAndCache: the
// K/V cache pages are 1-byte fp8 (DType::kI8 storage, `kind` = the fp8
// interpretation) and each element is quantized as Quantize(hp / k_scale) /
// Quantize(hp / v_scale). Mirrors the fp8 branch of vLLM reshape_and_cache_flash
// (csrc/libtorch_stable/cache_kernels.cu:314-401 + CopyWithScaleOp :241-252 +
// the fp8::scaled_convert scale convention, quant_utils.cuh:296-308) @ pin
// 555967922. k_scale/v_scale are the per-tensor scales BaseKVCacheMethod loads
// from the checkpoint (kv_cache.py:108-191); both must be > 0. Same shape/stride
// contract as ReshapeAndCache; the ONLY difference is the fp8 store. Implemented
// on CPU (W1, src/vt/cpu/cpu_cache.cpp), CUDA (W2, src/vt/cuda/cuda_cache.cu,
// gated byte-for-byte against the CPU arm) and ROCm (W6,
// src/vt/rocm/rocm_ops.hip:253, landed 191f64608 / #2065); a backend that
// registers no provider refuses by name in GetOp. This paragraph named only CPU
// and CUDA until 2026-09-12, while PagedAttentionArgs::kv_cache_dtype above
// already recorded the ROCm arm of the matching READ path. kFp8E5M2 is a named later brick (spec W5).
void ReshapeAndCacheFp8(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                        Tensor& v_cache, const Tensor& slot_mapping, Fp8KVCacheDataType kind,
                        float k_scale, float v_scale);

// --- MLA paged KV-cache write (W3). The MLA counterpart of ReshapeAndCache.
// Ported 1:1 from vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
// (`concat_and_cache_mla_kernel`) + its host wrapper `:842-905`
// (`concat_and_cache_mla`) @ pin e24d1b24, reached from
// vllm/_custom_ops.py:2532 and called at vllm/v1/attention/backend.py:995,1075.
//
// WHAT ACTUALLY RUNS UPSTREAM (verified, not assumed — AGENTS.md "ground every
// check in the whole execution chain"): unlike the GEMM/attention families,
// this one is NOT delegated to a dependency. `concat_and_cache_mla` binds
// straight to `torch.ops._C_cache_ops.concat_and_cache_mla`
// (`_custom_ops.py:2540-2542`), registered from vLLM's OWN
// csrc/libtorch_stable/torch_bindings.cpp over the kernel above. There is no
// flashinfer / cutlass / TRT-LLM variant in the dense-bf16 path — the only
// alternative in that TU is `concat_and_cache_ds_mla_kernel` (`:445+`), which is
// the fp8_ds_mla 656-byte V3.2 layout, out of campaign scope. The one thing that
// can displace it is the COMPILE-TIME fusion pass
// vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:40, which folds
// RoPE into `concat_and_cache_mla_rope_fused` (`_custom_ops.py:2545`) — same
// math, one launch; our fusion-catalog analogue is deferred to W9, exactly as
// the campaign spec sequences it (unfused byte-exact first).
//
// MLA caches ONE row per token: the compressed latent CONCATENATED with the
// decoupled rope part, `kv_lora_rank + qk_rope_head_dim` wide (512 + 64 == 576
// for every DeepSeek variant and for Kimi Linear's MLA layers), with
// num_kv_heads == 1 and NO separate V — V is reconstructed from the same latent
// via W_UV at decode. That is why the cache is 3-D
// (num_blocks, block_size, head_size; mla_attention.py:1216-1224) and why
// ReshapeAndCache's (k, v, k_cache, v_cache) signature cannot express this write.
//
//   kv_c         [num_tokens, kv_lora_rank]        (post-`kv_a_layernorm` latent)
//   k_pe         [num_tokens, qk_rope_head_dim]    (the shared single-head rope part)
//   kv_cache     [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
//   slot_mapping [num_slots] i64  (num_slots <= num_tokens; the tail of kv_c/k_pe
//                is CUDA-graph padding and is ignored — upstream uses
//                slot_mapping.size(0) as the token count, `:855-863`)
// For token t with slot s = slot_mapping[t]: block = s / block_size,
// offset = s % block_size; kv_c[t] lands at columns [0, kv_lora_rank) of that
// entry and k_pe[t] at [kv_lora_rank, kv_lora_rank + pe_dim). A slot s < 0 skips
// token t (padding). Indexing is driven by the tensor STRIDES (upstream reads
// kv_cache.stride(0)/stride(1) and kv_c/k_pe.stride(0)), so a strided cache view
// or a split-projection source view is handled without a copy.
// The "auto" path only: cache dtype == kv_c dtype, no fp8 scaling — the
// `fp8_ds_mla` / int4 layouts are out of scope and are REFUSED loudly rather
// than silently mis-written.
void ConcatAndCacheMla(Queue& q, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                       const Tensor& slot_mapping);

// --- The fp8_ds_mla paged K cache (KV-DSV4-MULTICACHE W8, #2455) ------------
// DeepSeek-V4's K cache is NOT a rank-3 (num_blocks, block_size, width) tensor,
// and that is the whole reason these are separate ops. Its block is
// REGION-SPLIT (`cache_utils.py:59-66` @ `5559679229`, verbatim):
//
//     K Cache block layout (block_size=64 tokens):
//     - [0, 64*576):              Token data, each token has 448 fp8 + 128 bf16
//     - [64*576, 64*576 + 64*8):  Scales, each token has 8 uint8 scales
//     - [64*576 + 64*8, block_stride): Padding
//
// A token's scale bytes live in a DIFFERENT region of the block from its data
// bytes, so no (block, row, column) indexing reaches both. `vt::ConcatAndCacheMla`
// requires `rank == 3` and keeps its "auto path only, fp8_ds_mla refused"
// contract unchanged — that refusal is what stops an f32 write (2048 bytes at
// head_dim 512, 3.5x the 584 the spec declares) landing in a byte page.
//
// STORE — `quantize_and_insert_k_cache` / `quantize_and_insert_k_kernel`
// (`cache_utils.py:36-159` kernel, `:162-227` host wrapper).
//
//   k             [num_tokens, 512] float   the compressed latent: NoPE part in
//                 columns [0, 448), the already-rotated RoPE part in [448, 512).
//                 Upstream asserts bf16 (`:167-171`); we accept any float dtype
//                 because the encoder's FIRST act is upstream's own fp32->bf16
//                 round, so an f32 row and its bf16 rounding produce the SAME
//                 bytes.
//   kv_cache      [num_blocks, block_bytes] DType::kI8 — the byte page. Its
//                 block stride comes from `kv_cache.stride[0]`, exactly as
//                 upstream reads `k_cache.stride(0)` (`:189`).
//   slot_mapping  [num_slots] i64. `slot < 0` writes NOTHING — not a zero row,
//                 not a partial row (upstream `if slot_idx == -1: return`,
//                 `:77-78`). Upstream takes the token count from
//                 `slot_mapping.shape[0]` (`:186-188`), so trailing rows of `k`
//                 are padding and are ignored.
//   block_size    STORAGE rows per block (`cache_block_size`, `:216`). A scalar
//                 rather than a shape because the page is rank-2 bytes.
//
// For token t with slot s: block = s / block_size, pos = s % block_size; the
// data lands at `pos * 576` inside the block and the 8 scale bytes at
// `block_size * 576 + pos * 8`, with the 8th EXPLICITLY zeroed (`:148-149`).
// Nothing past `block_size * 584` is ever written: the tail is the alignment
// padding (`:63`).
void ConcatAndCacheDsMla(Queue& q, const Tensor& k, Tensor& kv_cache,
                         const Tensor& slot_mapping, int64_t block_size);

// READ — `dequantize_and_gather_k_cache_triton` /
// `_dequantize_and_gather_k_kernel` (`cache_utils.py:228-341` kernel,
// `:339-389` host wrapper).
//
// A GATHER into a float scratch, not a fused fp8 attention. Upstream splits the
// same way: its prefill dequant-gathers (`nvidia/flashmla.py:296`) while its
// decode hands the packed page to a vendor kernel (`:219-226`) we do not have.
// The scratch costs `seq_len * 512 * sizeof(out dtype)` per layer per step,
// which is why the native fp8 decode is a later wave.
//
//   out          [num_reqs, max_num_tokens, 512] f32 or bf16
//   kv_cache     [num_blocks, block_bytes] DType::kI8
//   seq_lens     [num_reqs] i32
//   gather_lens  [num_reqs] i32 or nullptr. nullptr is upstream's
//                `gather_lens_ptr is None` arm: gather the WHOLE sequence
//                (`:257-262`).
//   block_table  [num_reqs, max_blocks_per_seq] i32
//
// For request b it walks `pos` in `[seq_len - gather_len, seq_len)` and writes
// row `offset + i` of `out[b]` (`:263-296`).
//
// THE ASYMMETRY, and it is upstream's: the store writes `scale_dim == 8` scale
// bytes and this read consumes `n_quant_blocks == 7` (`:385`). The 8th byte is
// written and never read. Do not "tidy" the store's pad byte away — upstream
// stores it explicitly so the page holds 0 there rather than whatever it held
// before, and a later native-fp8 kernel reading 8 blocks would then read stale
// bytes instead of a zero.
//
// DTYPE. Upstream's `out` is bf16 (`:328`, `:337`) and a bf16 `out` here is
// bit-identical to it. An f32 `out` stores the dequantized value BEFORE that
// final bf16 rounding — annotated per AGENTS.md rather than left to be found:
// it is not a widened model buffer but a scratch whose dtype must equal the
// query dtype that `vt::MlaDecodeAttention` will consume it with, and our CPU
// MLA decode runs at f32.
void DequantAndGatherDsMla(Queue& q, Tensor& out, const Tensor& kv_cache,
                           const Tensor& seq_lens, const Tensor* gather_lens,
                           const Tensor& block_table,
                           const DequantAndGatherDsMlaArgs& args);

// --- MLA decode attention (MLA campaign W4) ---------------------------------
// The MQA decode half of Multi-head Latent Attention: every one of the Hq query
// heads attends to the SAME single-head compressed latent row in the 3-D MLA
// cache, and V is the leading `v_head_dim` slice of that same row — there is no
// V tensor. Upstream calls this `forward_mqa`
// (vllm/v1/attention/backends/mla/triton_mla.py:189-260 @ e24d1b24), which hands
// the SAME buffer to `decode_attention_fwd` twice
// (`kv_c_and_k_pe_cache` as K, `kv_c_and_k_pe_cache[..., :kv_lora_rank]` as V,
// `:236-244`) with `is_mla=True` and `layer._k_scale` used for BOTH k_scale and
// v_scale (`:256-257`). vt::PagedAttention's (k_cache, v_cache) signature cannot
// express that, which is why this is its own op.
//
//   out          [B, Hq, Dv]           (Dv == kv_lora_rank == 512 for DeepSeek)
//   lse          [B, Hq] or nullptr    (upstream `can_return_lse_for_decode`)
//   query        [B, Hq, D]            (D == kv_lora_rank + qk_rope_head_dim == 576)
//   kv_cache     [num_blocks, block_size, D]   — the 3-D MLA cache
//                (mla_attention.py:1216-1224), i.e. exactly what
//                vt::ConcatAndCacheMla writes
//   block_table  [B, max_blocks_per_seq] i32   (upstream `Req_to_tokens`)
//   seq_lens     [B] i32                       (upstream `B_Seqlen`)
//
// SEMANTICS (the numbers, independent of the split schedule) — for request b and
// head h, over key positions j in [0, seq_lens[b]) with
// entry = kv_cache[block_table[b, j / block_size], j % block_size, :]:
//     qk[j] = scale * dot(query[b,h,:], entry[:])          (the FULL D == 576)
//     p     = softmax_j(qk)                                (f32, online/streaming)
//     out[b,h,:]  = sum_j p[j] * entry[:Dv]                (V is the Dv PREFIX)
//     lse[b,h]    = log(sum_j exp(qk[j] - max)) + max
// There is NO causal mask: decode has exactly one query token per request whose
// position is seq_lens[b]-1, so the whole context is visible. `logit_cap` is 0
// on every reachable MLA path (TritonMLAImpl rejects `logits_soft_cap`,
// triton_mla.py:165-171) and is therefore not ported.
//
// TWO-STAGE SPLIT-KV (the CUDA impl; ported from the upstream Triton pair
// `_fwd_grouped_kernel_stage1` triton_decode_attention.py:278-458 and
// `_fwd_kernel_stage2` `:575-639`). Stage 1 partitions [0, seq_len) into
// `num_kv_splits` contiguous chunks of `cdiv(seq_len, num_kv_splits)` and emits
// one NORMALIZED partial `acc/e_sum` plus its `e_max + log(e_sum)` per split;
// stage 2 merges them with the same online-softmax rescale. A split whose
// [start, end) is empty is SKIPPED by both stages (upstream `:361`, `:610`), so
// the scratch row for it is never written and never read.
//
// DETERMINISM: stage 2 merges splits in a fixed ASCENDING split order (upstream
// `:607` `for split_kv_id in range(0, NUM_KV_SPLITS)`) — no atomicAdd anywhere,
// so the result is run-to-run bit-reproducible for a fixed `num_kv_splits`, our
// house convention. Changing `num_kv_splits` changes the f32 summation order and
// therefore may change the last bits; that is upstream's behavior too.
//
// dtypes: query/kv_cache/out share one float dtype (f32 or bf16 — upstream's
// `supported_dtypes` are fp16/bf16, `triton_mla.py:82`); all accumulation is f32.
// The fp8 KV-cache branch (`if k.dtype.is_fp8()`, `:390-391`) is out of scope and
// refused, exactly as the W3 cache write refuses `fp8_ds_mla`.
void MlaDecodeAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                        const Tensor& kv_cache, const Tensor& block_table,
                        const Tensor& seq_lens, const MlaDecodeAttentionArgs& args);

// --- the DSA "Lightning Indexer" selection pair (dots3-note W4b-3c, #699) ----
//
// These two ops are the `vt::Tensor` form of the host reference
// `vllm::deepseek_v4::DsaIndexerLogits` / `DsaTopkSelect`
// (include/vllm/model_executor/models/deepseek_v4_dsa.h), which W3 landed as
// `std::vector<float>` round-trips on a V4-private seam and which stay as this
// gate's ORACLE. Nothing about the maths changes; what changes is that a device
// path can now reach it.
//
// The MQA logit for query row `t` over key row `s`
// (`vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156` @ `bc2d63e650` —
//  `tl.dot(..., input_precision="ieee")`, `* kv_scales`, `tl.maximum(_, 0.0)`,
//  `* w_block`, `tl.sum(_, axis=0)`):
//
//     logit[t,s] = sum_h fold[t,h] * ReLU( dot(q[t,h,:], k[s,:]) )
//     fold[t,h]  = weights[t,h] * q_scale[t,h] * softmax_scale * n_head_scale
//
// The ReLU is the load-bearing nuance: a key whose every head dots negative
// scores EXACTLY 0.0, which is why exact ties are ordinary here rather than
// exotic, and why the selection's tie rule has to be pinned.
//
//   logits    [num_tokens, num_keys]                 f32, OUT
//   q         [num_tokens, index_n_heads, index_head_dim]  f32/bf16
//   k         [num_keys,   index_head_dim]                 f32/bf16 (MQA: 1 KV head)
//   weights   [num_tokens, index_n_heads]                  f32/bf16 — the RAW
//             `weights_proj` output; the fold happens HERE, once, in f32
//   win_start [num_tokens] i32, win_end [num_tokens] i32 — the per-query causal
//             candidate range `[start, end)` into `k`
//
// Keys outside `[win_start[t], win_end[t])` are written `-inf`, mirroring the
// kernel's masked store (`triton_fp8_mqa_logits.py:155-156`), so a plain top-k
// downstream needs no second mask. The dot and the sum accumulate in f32
// whatever the operand dtype, which is what `input_precision="ieee"` on a bf16
// `tl.dot` gives. Registered on kCPU (src/vt/cpu/cpu_dsa_indexer.cpp), kCUDA
// (src/vt/cuda/cuda_dsa_indexer.cu) and kROCM
// (src/vt/rocm/rocm_dsa_indexer.hip, MODEL-MM-QWEN4-EXP W5). This line read
// "CPU + CUDA" until 2026-09-12, by which date the kROCM arm had landed.
void DsaIndexerLogits(Queue& q, Tensor& logits, const Tensor& q_states, const Tensor& k,
                      const Tensor& weights, const Tensor& win_start,
                      const Tensor& win_end, const DsaIndexerLogitsArgs& args);

// Per-row sparse top-k selection — `ops.top_k_per_row_prefill`
// (`vllm/model_executor/layers/sparse_attn_indexer.py:509` @ `bc2d63e650`)
// plus the short-context all-select. For each query row `t`, with
// `n = clamp(win_end[t], 0, num_keys) - clamp(win_start[t], 0, num_keys)`:
//
//   * `n <= topk`  — EVERY candidate is selected, in ascending key order, and
//                    the remaining slots stay `-1`. This is the branch that
//                    makes a short context bit-identical to dense attention.
//   * `n >  topk`  — the `topk` keys with the largest logits, ties broken
//                    toward the SMALLER key index, emitted in ASCENDING key
//                    order.
//
//   indices [num_tokens, topk] i32, OUT — TOKEN POSITIONS, `-1` = no token.
//                                          Feeds `MlaDecodeAttentionArgs::topk_indices`.
//   counts  [num_tokens]       i32, OUT — `min(n, topk)`.
//                                          Feeds `MlaDecodeAttentionArgs::valid_counts`.
//   logits  [num_tokens, num_keys] f32 — from vt::DsaIndexerLogits.
//
// The ascending emission order is not cosmetic: it is what makes a full
// selection BIT-FOR-BIT equal to no selection at all in vt::MlaDecodeAttention,
// because the online softmax then sees the identical summation order.
// Registered on kCPU (src/vt/cpu/cpu_dsa_indexer.cpp), kCUDA
// (src/vt/cuda/cuda_dsa_indexer.cu) and kROCM
// (src/vt/rocm/rocm_dsa_indexer.hip, MODEL-MM-QWEN4-EXP W5). This line read
// "CPU + CUDA" until 2026-09-12, by which date the kROCM arm had landed.
void DsaTopkSelect(Queue& q, Tensor& indices, Tensor& counts, const Tensor& logits,
                   const Tensor& win_start, const Tensor& win_end);

// GLM-5.3-Flash's k-pool DSA indexer — the LEARNED pooled compression
// (`Glm5NextTextIndexer.get_pooled_states`, `modular_glm5_next.py:897-970` @
// transformers v5.16.1) and the pooled selection that consumes it
// (`forward` `:821-875`, with `get_visible_tokens` `:877-895` folded in as a
// predicate and `append_visible_tail` `:972-1022` as the tail write). vLLM
// implements `glm5_next` at no revision, so transformers is the reference here
// under AGENTS.md "When vLLM has no implementation". BOTH ops of this family are
// registered on kCUDA (src/vt/cuda/cuda_glm5_next.cu:561,564) and on kROCM
// (src/vt/rocm/rocm_ops.hip:345,348); there is deliberately NO CPU provider,
// because the CPU answer is `glm5_next_dsa.cpp` and registering the oracle under
// these ids would make the seam its own oracle — see the
// kGlm5NextKpoolCompress paragraph in the OpId enum. This line read "CUDA only"
// until 2026-09-12, by which date the kROCM arms had landed.
//
//   packed       f32 [batch, kv_len, 2 * head_dim + 1] — `concat[k, gate, valid]`
//                (`:798-801`), the FULL key history the pool grid is re-formed
//                over on every call
//   ape          f32 [index_kpool, head_dim] — `index_kpool_compress_ape`
//   pool_keys    f32 [batch, np, head_dim]        np = ceil(kv_len / index_kpool)
//   pool_indices i32 [batch, np, index_kpool]     -1 for an invalid member
//   pool_valid   i32 [batch, np]                  1 iff ALL members are valid
//   num_pools    i32 [1]                          `P` AFTER `keep` (`:968-970`)
//
// `np` is the STATIC upper bound and `P <= np` is the live width. Only
// `[0, P)` carries meaning; the slack is zeroed (and `pool_indices` filled with
// the -1 sentinel) so a downstream read of it is empty rather than undefined.
// `num_pools` stays on the DEVICE: the whole point of the family is that the
// eleven DSA layers stop paying a device-to-host round trip per step.
void Glm5NextKpoolCompress(Queue& q, Tensor& pool_keys, Tensor& pool_indices,
                           Tensor& pool_valid, Tensor& num_pools, const Tensor& packed,
                           const Tensor& ape);

// The selection over those pooled candidates.
//
//   q_states     f32 [batch, seq_len, index_n_heads, head_dim] — `wq_b(q_resid)` (`:795`)
//   head_weights f32 [batch, seq_len, index_n_heads] — `weights_proj(hidden)` BEFORE
//                the `n_heads ** -0.5` scale, which this op applies (`:827`)
//   valid_keys   i32 [batch, kv_len] — the packed row's validity channel (`:814`)
//   q_mask       i32 [batch, seq_len] — the query-side padding mask (`:873`)
//   topk_indices i32 [batch, seq_len, index_topk (+ index_kpool - 1)] — **2051**
//                wide on the published checkpoint, NOT 2048, carrying -1
//                sentinels and possible duplicates, which upstream absorbs with
//                `scatter_add_` + `ne(0)` (`:1119-1129`)
//   index_scores f32 [batch, seq_len, np] — the per-pool score BEFORE the
//                validity mask (`:828`). It is an output rather than an internal
//                because it is the only way a gate can show the selection is a
//                strict separation and not a coin flip: top-k error is BIMODAL,
//                so a tolerance on the selected values passes a wrong set whose
//                values happen to be close.
void Glm5NextKpoolSelect(Queue& q, Tensor& topk_indices, Tensor& index_scores,
                         const Tensor& q_states, const Tensor& head_weights,
                         const Tensor& pool_keys, const Tensor& pool_indices,
                         const Tensor& pool_valid, const Tensor& num_pools,
                         const Tensor& valid_keys, const Tensor& q_mask,
                         const Glm5NextKpoolSelectArgs& args);

// --- MLA prefill attention (MLA campaign W5) --------------------------------
// The MHA prefill half of Multi-head Latent Attention — upstream's
// "Compute Friendly Approach" (mla_attention.py:66-89): `kv_b_proj` has already
// MATERIALIZED per-head K `[total_k, H, qk_nope+qk_rope]` and V
// `[total_k, H, v_head_dim]` from the latent, so this is an ordinary varlen MHA
// with an ASYMMETRIC pair of head dims — QK 192 (128 nope + 64 rope) and V 128
// for every DeepSeek variant — over CONTIGUOUS (not paged) k/v.
//
// WHAT ACTUALLY RUNS UPSTREAM ON GB10 (OBSERVED at W0, not inferred: the oracle
// logs `Using FLASH_ATTN MLA prefill backend` on sm_121). The MLA prefill
// selector gives major != 10 the single entry `[FLASH_ATTN]`
// (vllm/v1/attention/backends/mla/prefill/selector.py:66-76) and hard-raises if
// it is unavailable (`:191-194`) — there is no fallback below FA on sm_121.
// That backend is
// vllm/v1/attention/backends/mla/prefill/flash_attn.py:40 FlashAttnPrefillBackend,
// whose two entry points are `:205 run_prefill_new_tokens` (causal, over the new
// tokens' own K/V) and `:229 run_prefill_context_chunk` (NON-causal, over one
// gathered context chunk), both funnelled through
// `:153 _flash_attn_varlen_diff_headdims` into `flash_attn_varlen_func`.
//
// V ZERO-PADDING IS UPSTREAM BEHAVIOUR, ported exactly. `requires_v_padding` is
// TRUE on GB10 (`flash_attn.py:88-99` clears it only for FA3-on-SM90 or FA4), so
// upstream pads V from v_head_dim to the QK head dim with ZEROS
// (`:164-168 torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)`)
// and slices the output back to v_head_dim afterwards (`:196-197`). We do the
// same, inside the op: the caller passes V at its true width and gets `out` at
// its true width; the padded staging buffer is an implementation detail. Zero
// padding is exact — the padded output columns are sum_j p[j]*0 == 0 — so the
// slice loses nothing.
//
//   out          [total_q, H, Dv]     (Dv == v_head_dim == 128)
//   lse          [H, total_q] f32 or nullptr — upstream's UNPADDED varlen LSE
//                layout (`Flash_fwd_params::unpadded_lse`), which is what
//                vt::MergeAttnStates consumes. `return_softmax_lse` is True for
//                every context chunk and for the new-tokens call WHEN there is
//                context to merge with (`mla_attention.py:2385`).
//   query        [total_q, H, Dqk]    (Dqk == qk_nope_head_dim + qk_rope_head_dim == 192)
//   key          [total_k, H, Dqk]
//   value        [total_k, H, Dv]
//   cu_seqlens_q [B+1] i32            (`_prefill_metadata.query_start_loc`)
//   cu_seqlens_k [B+1] i32            (the SAME tensor for the new-tokens call,
//                `chunked_context.cu_seq_lens[chunk_idx]` for a context chunk —
//                flash_attn.py:218-219 vs :241-242)
//
// H is the QUERY head count and is also the K/V head count: MLA prefill is
// genuinely multi-head on both sides (the latent has been up-projected), so
// there is no GQA grouping here — mla_attention.py:315-318 records the shape as
// K `[S, 128, 192]` / V `[S, 128, 128]` with 128 KV heads.
//
// dtypes: query/key/value/out share one float dtype. The CUDA path is bf16 only
// (the vendored FA-2 instantiations); the CPU reference also accepts f32/f16.
// The fp8-prefill branch (`use_fp8_prefill`, mla_attention.py:2360-2379) needs
// `is_device_capability_family(100)` (`mla_attention.py:1382-1385`) and is
// therefore UNREACHABLE on GB10 — not ported, refused by the dtype check.
void MlaPrefillAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                         const Tensor& key, const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_k, const MlaPrefillAttentionArgs& args);

// --- MLA chunked-context cache gather (MLA campaign W5) ---------------------
// Ported 1:1 from vllm/csrc/libtorch_stable/cache_kernels.cu:992-1064
// (`vllm::gather_and_maybe_dequant_cache`) + its host wrapper `:1099-1157`,
// reached from vllm/_custom_ops.py and called at
// mla_attention.py:2119-2129 (`ops.gather_and_maybe_dequant_cache`) — the FIRST
// step of every chunked-context iteration.
//
// WHAT ACTUALLY RUNS UPSTREAM (whole-chain rule): like the W3 cache write this
// is vLLM's OWN csrc kernel, not a dependency. The sibling `cp_gather_cache`
// (`:1237`) is the fp8/DCP path (`mla_attention.py:2132-2139`), out of scope.
//
// It gathers, for each prefill request b, the context rows
// [seq_starts[b], seq_starts[b] + chunk_seq_lens[b]) out of the PAGED 3-D MLA
// cache into a CONTIGUOUS varlen workspace laid out by cu_seq_lens:
//
//   dst          [>= num_tokens, D]                the chunk workspace
//   src_cache    [num_blocks, block_size, D]       the 3-D MLA cache
//   block_table  [B, max_blocks] i32
//   cu_seq_lens  [B+1] i32   cumulative per-request token counts IN THIS CHUNK
//   token_to_seq [>= num_tokens] i32  back-map token index -> request (`:1015`)
//   seq_starts   [B] i32 or nullptr — the chunk's start offset within each
//                request's context (`chunked_context.starts[i]`). Upstream
//                rounds `max_context_chunk` DOWN to a multiple of page_size
//                (mla_attention.py:1687-1690) precisely because this kernel
//                indexes `(seq_starts[b] + within_chunk) / block_size` into the
//                block table; we mirror the requirement and REFUSE a
//                non-page-aligned start rather than silently mis-gather.
//   num_tokens   the chunk's total token count (`chunk_total_token[i]`)
//
// Everything is STRIDE-driven exactly as upstream (`:1150-1153` reads
// block_table.stride(0), src_cache.stride(0)/stride(1), dst.stride(0)), so a
// per-layer cache slice and a workspace slice both work copy-free. The fp8
// dequant branch (`kv_dt != kAuto`) is out of scope and refused.
void GatherMlaCache(Queue& q, Tensor& dst, const Tensor& src_cache, const Tensor& block_table,
                    const Tensor& cu_seq_lens, const Tensor& token_to_seq,
                    const Tensor* seq_starts, int64_t num_tokens);

// --- Attention-state LSE merge (MLA campaign W5) ----------------------------
// Ported 1:1 from vllm/csrc/libtorch_stable/attention/merge_attn_states.cu:18-192
// (`vllm::merge_attn_states_kernel`), reached from
// vllm/v1/attention/ops/merge_attn_states.py:9 — which selects the CUDA kernel
// on CUDA for f32/f16/bf16 with a head_size divisible by the 128-bit pack
// (`:59-77`), i.e. exactly our case, and only falls back to the Triton
// transcription otherwise. Call sites: mla_attention.py:2188-2195 (merging
// consecutive CONTEXT CHUNKS) and `:2413-2420` (merging the whole context result
// with the new-tokens result).
//
// Implements §2.2 of https://www.arxiv.org/pdf/2501.01005: two partial softmax
// attention results over DISJOINT key sets are combined by their log-sum-exps.
//   m = max(p_lse, s_lse); ps = exp(p_lse-m); ss = exp(s_lse-m); t = ps+ss
//   out = prefix_out * (ps/t) + suffix_out * (ss/t)
//   out_lse = log(t) + m
// Arithmetic is f32 regardless of the tensor dtype (upstream converts through
// `to_float`/`from_float`, `:158-171`).
//
//   output       [num_tokens, H, Dv]
//   output_lse   [H, num_tokens] f32 or nullptr
//   prefix_output/suffix_output [num_tokens, H, Dv]
//   prefix_lse/suffix_lse       [H, num_tokens] f32
//   prefill_tokens_with_context — tokens at index >= this take the SUFFIX
//     output verbatim with no merge (`:66-89`); < 0 means "all tokens have
//     context", upstream's `prefill_tokens_with_context=None`.
//
// TWO EDGE CASES PORTED VERBATIM, both of which a naive merge gets wrong:
//   * `+inf` LSE is normalized to `-inf` first (`:97-98`);
//   * when BOTH are `-inf` (a chunk in which a request has no keys at all —
//     upstream's comment at `:100-106` describes exactly the chunked-prefill
//     situation that produces it) the merge would be 0/0 => NaN, so upstream
//     emits the PREFIX output and `-inf` LSE instead. We do the same.
// The fp8-output branch (`USE_FP8_OUTPUT`) is out of scope and not ported.
void MergeAttnStates(Queue& q, Tensor& output, Tensor* output_lse, const Tensor& prefix_output,
                     const Tensor& prefix_lse, const Tensor& suffix_output,
                     const Tensor& suffix_lse, int64_t prefill_tokens_with_context);

// --- Paged attention (M1.6). Correctness-grade varlen prefill + paged decode.
// Semantics ported from the FlashAttention path
// vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward @
// e24d1b24 (causal GQA softmax over the paged K/V; scale = self.scale). The
// cache read is the NHD layout FlashAttentionBackend::get_kv_cache_shape
// allocates (num_blocks, 2, block_size, num_kv_heads, head_size) — NOT cpu_attn's
// HND arithmetic — and is driven by k_cache/v_cache STRIDES (the two dim-1 unbind
// slices; block stride 2*bs*H*D). This GENERALIZES the dense M0.9 vt::Attention:
// on a single contiguous sequence the two ops agree.
//
// For each query token t (found via query_start_loc: token t belongs to request
// r with query_start_loc[r] <= t < query_start_loc[r+1]) at absolute position
// p = (seq_lens[r] - query_len_r) + (t - query_start_loc[r]) (query_len_r =
// query_start_loc[r+1] - query_start_loc[r]; seq_lens[r] is the total context
// INCLUDING this chunk), and q-head h (kv-head g = h / (Hq/Hk)):
//   for key position j in 0..p (causal), intersected with
//   [p-window.left,p+window.right] when args.window_size is present —
//   block = block_table[r, j / block_size],
//   offset = j % block_size, K = k_cache[block, offset, g, :], V likewise —
//   s[j] = scale * (query[t,h] · K);  out[t,h,:] = Σ_j softmax(s)_j * V.
// Softmax/accumulation in f32 (max-subtracted). The current step's K/V must
// already be in the cache (compose vt::ReshapeAndCache upstream), so no separate
// key/value args — the read is entirely from the paged cache.
//
//   query          [num_actual_tokens, num_q_heads, head_size]  (contiguous)
//   out            same shape/dtype as query (contiguous)
//   k_cache/v_cache[num_blocks, block_size, num_kv_heads, head_size]  (the two
//                  dim-1 slices of the flash cache; STRIDED, read-only)
//   block_table    [num_reqs, max_blocks] i32  (Task 1's block_table_tensor)
//   seq_lens       [num_reqs] i32              (context length incl. this step)
//   query_start_loc[num_reqs + 1] i32          (cumulative query offsets)
void PagedAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                    const Tensor& query_start_loc, const PagedAttentionArgs& args);

// SPEC-DFLASH2 W10 repair (#1865): process counter of spec-CLASSIFIED batches
// arriving at the PagedAttention dispatch wrapper — a batch whose
// `args.uniform_spec_query_len` passes `PagedAttnUniformSpecShape`. This is the
// seam the W10 review named dead: deleting the model's
// `pa.uniform_spec_query_len = meta.uniform_spec_query_len` threading redded
// nothing on a CPU box, because the CUDA lane it feeds is `HasCuda`-gated and
// the CPU kernel ignores the field. The counter makes the THREADING observable
// on every backend: the production fixture asserts one arrival per uniform
// verify step per full-attention layer. In-memory diagnostics, same shape as
// v1::GraphDispatchStats; reset+read from one test at a time.
uint64_t PagedAttnSpecClassifiedCount();
void ResetPagedAttnSpecClassifiedCount();

// --- V1 sampling ops (M1.7 Task 2). Ported from
// vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
// The Sampler pipeline (M1.7 Task 4) composes these over the model's final
// logits `[num_reqs, vocab_size]` (row-major f32). Every op is correctness-grade
// CPU + CUDA and indexes contiguous rows; greedy_argmax is the bit-exact parity
// primitive (matches torch.argmax's lowest-index tie-break).

// _SAMPLING_EPS (sampler.py:17). Greedy rows carry temperature < this; the
// temperature guard and the greedy/random where-merge both key off it.
constexpr float kSamplingEps = 1e-5f;

// apply_temperature (sampler.py::Sampler.apply_temperature). In-place per-row
// `logits[i] /= temp[i]`. When `!all_random`, greedy rows (temp < eps) would
// divide by ~0, so temp is first replaced per-row with `where(temp<eps, 1.0,
// temp)` (upstream comment: "Avoid division by zero if there are greedy
// requests"). logits [num_reqs, vocab] f32 in place; temp [num_reqs] f32.
void ApplyTemperature(Queue& q, Tensor& logits, const Tensor& temp, bool all_random);

// greedy_sample (sampler.py::Sampler.greedy_sample) = argmax(logits, dim=-1).
// LOWEST-INDEX tie-break: torch.argmax returns the FIRST occurrence of the max,
// so a strict `>` scan (keep the first max) is bit-exact vs torch on f32 logits.
// This is the M0-exit parity gate primitive. token_ids [num_reqs] i64 (torch
// argmax returns int64); logits [num_reqs, vocab] f32.
void GreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits);

// apply_top_k_top_p (topk_topp_sampler.py::apply_top_k_top_p_pytorch, the CPU
// allow_cpu_sync path). Masks non-top-k / non-top-p logits to -inf IN PLACE.
// k [num_reqs] i32 (nullptr => skip top-k), p [num_reqs] f32 (nullptr => skip
// top-p); per-row. When p is nullptr and k given, uses the no-sort
// apply_top_k_only fast path; otherwise the sort-based path (top-k threshold =
// the k-th largest, ties at the threshold kept; top-p = smallest tail whose
// cumulative prob >= p, at-least-one). logits [num_reqs, vocab] f32.
void ApplyTopKTopP(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p);

// probs = softmax(logits, dim=-1) in f32 (forward_native's
// `logits.softmax(dim=-1, dtype=torch.float32)`; numerically stable, row-max
// subtracted). probs/logits [num_reqs, vocab] f32.
void ComputeProbs(Queue& q, Tensor& probs, const Tensor& logits);

// compute_logprobs (sampler.py::Sampler.compute_logprobs) =
// log_softmax(logits, dim=-1) in f32 (row-max subtracted, log-sum-exp).
// logprobs/logits [num_reqs, vocab] f32.
void ComputeLogprobs(Queue& q, Tensor& logprobs, const Tensor& logits);

// random_sample (topk_topp_sampler.py::random_sample +
// sample_with_exponential_noise): exponential-noise gumbel-max. For each element
// draw q ~ Exponential(1); pick `argmax_j(probs[i,j] / q[i,j])`. Per-request
// seeded RNG: `seeds[i]` is the resolved per-row seed (the Sampler picks the
// per-request override from `SamplingMetadata.generators` or the batch default —
// M1.7 Task 4). token_ids [num_reqs] i64; probs [num_reqs, vocab] f32;
// seeds [num_reqs] i64.
//
// RNG DEVIATION (recorded, T1 carry): the Exponential(1) draws come from a
// deterministic splitmix64 hash of (seed, row, vocab_index) mapped through the
// inverse-CDF q = -log(U), U in (0,1). This is distribution-correct (the
// exponential race gives P(argmax) == softmax, validated at large N) and
// reproducible under a fixed seed, but is NOT bit-exact vs torch's Philox4x32
// `exponential_()` — exact random-sampling parity vs torch is the documented
// M1.7 deferral. Greedy stays bit-exact; random is validated by algorithm +
// determinism + distribution.
void RandomSample(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds);

// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3) --------------
// Ported from vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py @ e24d1b24:
// the `is_greedy` branch of `_rejection_kernel` (:564-585 accept + :628 store of
// the accepted length), plus the greedy short-circuits of `_resample_kernel`
// (:846-849) and `_insert_resampled_kernel` (:828-841 num_sampled += 1 and the
// bonus-token argmax insert). The per-row `target_argmax` this needs is upstream's
// `_compute_global_target_argmax` over `_compute_local_logits_stats_kernel`'s
// per-vocab-block partials (:923-946).
//
// THE GREEDY ACCEPT RULE (the whole correctness story — see the header note in
// include/vllm/v1/spec_decode/rejection_sampler.h):
//   For request r the expanded logit rows are [cu[r], cu[r+1]); k_r = cu[r+1] -
//   cu[r] - 1 draft positions plus one bonus position. Walking i = 0..k_r-1 while
//   still accepting:
//     * `target_argmax = argmax(logits[cu[r]+i])`
//     * `draft = draft_sampled[cu[r]+i+1]`   (NOTE the +1: draft token i is the
//       INPUT id at the NEXT expanded row, mirroring :534)
//     * accept iff `draft == target_argmax`; store `draft` when accepted, else
//       store `target_argmax` (the replacement) and STOP accepting.
//   If all k_r accepted, store `argmax(logits[cu[r]+k_r])` (the bonus).
//   num_sampled[r] = accepted_length + 1 (always ≥ 1: a rejection still emits the
//   target argmax at the mismatch position).
// A placeholder draft id of -1 can never equal an argmax (≥ 0), so it is rejected
// with no out-of-bounds read — mirroring the upstream `-1` padding contract.
//
//   logits        [num_logits, vocab] f32   the EXPANDED verify logits
//   draft_sampled [num_logits] i32          input ids gathered at logits_indices
//   cu_num_logits [num_reqs + 1] i32        per-request expanded-row offsets
//   sampled       [num_reqs, max_num_logits] i32  OUT; positions past
//                 num_sampled[r] are filled with -1 (PLACEHOLDER_TOKEN_ID,
//                 vllm/v1/sample/rejection_sampler.py:30). Upstream leaves them
//                 uninitialized (`new_empty`); we fill so the buffer is
//                 deterministic and the ported legacy-sampler assertions read
//                 directly. Recorded deviation.
//   num_sampled   [num_reqs] i32            OUT; accepted_length + 1
//   target_argmax [num_logits] i32          SCRATCH, written then read by this
//                 op: the per-expanded-row argmax of `logits`, upstream's
//                 `_compute_global_target_argmax` output (:923-946). It is a
//                 PARAMETER and not a private static for one reason
//                 (SPEC-DFLASH2 A2-2, #2802): the CUDA arm launches two kernels
//                 and returns while both are still queued, so the buffer between
//                 them has to be owned by whoever owns the in-flight window. A
//                 process-global grow-only scratch cannot be: a second caller
//                 with more rows frees it under the first caller's queued accept
//                 kernel. The caller allocates it, keeps it alive until it has
//                 waited, and frees it then —
//                 `vllm::v1::RejectionSamplerDeviceOutput` is that owner.
//
// Argmax tie-break is LOWEST INDEX (torch.argmax), identical to vt::GreedyArgmax,
// so a k=0 request reduces EXACTLY to the non-speculative greedy sampler.
void GreedyRejectionSample(Queue& q, Tensor& sampled, Tensor& num_sampled,
                           Tensor& target_argmax, const Tensor& logits,
                           const Tensor& draft_sampled, const Tensor& cu_num_logits);

// --- V1 penalty / mask / builtin-proc ops (M1.7 Task 3). Ported from
// vllm/model_executor/layers/utils.py (apply_penalties), vllm/_custom_ops.py
// (apply_repetition_penalties), vllm/v1/sample/ops/bad_words.py, and
// vllm/v1/sample/logits_processor/builtin.py @ e24d1b24. The Sampler pipeline
// (M1.7 Task 4) composes these over the model's final logits [num_reqs, vocab]
// (row-major f32). Every op is correctness-grade CPU + CUDA. The higher-level
// ported entry points (src/vllm/v1/sample/ops/{penalties,bad_words}.{h,cpp},
// src/vllm/v1/sample/logits_processor/builtin.{h,cpp}) build the per-request
// derived tensors from the host SamplingMetadata and call these ops.

// apply_penalties (utils.py::apply_penalties + _custom_ops.py::
// apply_repetition_penalties_torch). Fuses the repetition penalty and the
// frequency / presence subtractions into a single elementwise pass over the
// pre-computed masks/counts. Per element (row i, col j):
//   penalty = (prompt_mask[i,j] || output_mask[i,j]) ? rep[i] : 1.0
//   logits *= (logits > 0) ? 1/penalty : penalty            (repetition)
//   logits -= frequency_penalties[i] * output_bin_counts[i,j]
//   logits -= presence_penalties[i] * output_mask[i,j]
// prompt_mask / output_mask [num_reqs, vocab] i8 (0/1), output_bin_counts
// [num_reqs, vocab] i32, frequency/presence/repetition [num_reqs] f32. In place.
void ApplyPenalties(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                    const Tensor& output_bin_counts, const Tensor& output_mask,
                    const Tensor& frequency_penalties, const Tensor& presence_penalties,
                    const Tensor& repetition_penalties);

// apply (builtin.py::MinPLogitsProcessor.apply). Per row: probs = softmax(logits),
// pmax = max_j probs; mask probs < min_p[i] * pmax to -inf. Rows with min_p[i]==0
// are unaffected (threshold 0). IS argmax-invariant (the max-prob token survives).
// logits [num_reqs, vocab] f32 in place; min_p [num_reqs] f32.
void ApplyMinP(Queue& q, Tensor& logits, const Tensor& min_p);

// apply (builtin.py::LogitBiasLogitsProcessor.apply) — the sparse scatter-add
// `logits[(rows, cols)] += biases`. rows/cols [m] i32 (flattened (req, token)
// pairs), biases [m] f32. NOT argmax-invariant. In place.
void ApplyLogitBias(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                    const Tensor& biases);

// Sparse scatter of -inf at the (rows[k], cols[k]) positions — the primitive
// behind builtin.py::MinTokensLogitsProcessor.apply (index_put_ of -inf over the
// stop-token slice) AND bad_words.py::apply_bad_words (block the final n-gram
// token). rows/cols [m] i32. In place.
void ApplyTokenMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols);

// sampler.py:396-397 `logits.masked_fill_(allowed_token_ids_mask, -inf)`. Sets
// logits[i,j] = -inf wherever mask[i,j] != 0. The mask is TRUE for tokens to
// EXCLUDE (gpu_input_batch.py fills the row True then clears the allowed ids to
// False). mask [num_reqs, vocab] i8. In place.
void ApplyAllowedTokenIds(Queue& q, Tensor& logits, const Tensor& mask);

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). These fuse the small
// host-side reshape/split/activation loops that sit between the big ops in the
// decode forward, so the whole decode step stays on-device (CUDA-graph
// capturable). All arithmetic is f32; every dimension is inferred from the
// tensor shapes (no args structs). CPU + CUDA.

// out[i] = F32ToBF16(in[i]); out bf16, in f32, same element count. The plain
// f32 -> bf16 activation-dtype cast used before feeding a bf16-consuming op.
void CastBf16(Queue& q, Tensor& out, const Tensor& in);

// T25: Permute V-heads from grouped (k*rpk+r) to tiled (r*num_k+k) order.
// out[T, value_dim] = in[T, value_dim] with the last dim permuted:
//   out[t*dv + h] = in[g*dv + h]  where  t = r*num_k + k,  g = k*rpk + r
// Used before the K-quant GEMV when ssm_out is kept as Q5_K in tiled order.
// value_dim = num_k * rpk * dv.  T, in, out are bf16.
void PermuteVHeads(Queue& q, Tensor& out, const Tensor& in,
                   int64_t T, int64_t num_k, int64_t rpk, int64_t dv);

// out[i] = f32(in[i]); out f32, in bf16, same element count. The bf16 -> f32
// upcast used to expose a bf16-only GEMM (Marlin) as an f32 result, matching the
// value the bf16 output rounds to (mirror of the cutlass f32-output scratch cast).
void CastF32(Queue& q, Tensor& out, const Tensor& in);

// out[i] = F32ToF16(in[i]); out f16, in f32 or bf16, same element count. The
// third sibling of the two casts above, and the NARROWING one.
//
// WHY IT EXISTS (QUANT-EXL3 W1a, #2181). `Exl3Gemm` reads its activation as f16
// and nothing else — the CPU arm calls `HadRows(HadIo::kHalfHalf, ...)` on `a`
// (`cpu_exl3_kernels.cpp:205`) and the device arm stages `a_had` in fp16 —
// because exllamav3 runs the whole linear in fp16 (`exl3.py:183-214`). A model
// whose residual stream is bf16 or f32 therefore needs one narrowing cast on
// the way into an EXL3 linear. The direction is not EXL3-specific, so this is a
// general op rather than a scheme-private helper.
//
// A bf16 source is read through its f32 value, which is exact (bf16 -> f32
// widens), and then rounded once to f16 — NOT reinterpreted. An f16 SOURCE is
// refused rather than copied, mirroring the two casts above, each of which
// names exactly one source dtype: a cast that accepted anything would make a
// wrong-dtype activation invisible at the call site.
void CastF16(Queue& q, Tensor& out, const Tensor& in);

// In-place per-output-column scale: x[m,n] *= col[n], with x an F32 or BF16
// [M,N] (row-major, inner-contiguous rows; row stride may be padded) and col an
// F32 [N] contiguous broadcast vector. The load-time-free realization of a merged
// per-tensor-fp8 projection's per-shard dequant: one fp8 GEMM over the
// N-concatenated weight is run with alpha=1 (raw f32 accumulation), then this
// applies each output column's folded scalar (= input_scale * that shard's
// weight_scale) in f32 — byte-identical to the separate per-shard GEMMs when the
// GEMM's accumulation matches (the alpha multiply is the same IEEE f32 op cuBLASLt
// would fold). CPU + CUDA. (Mirrors the fp4 merge's per-column block-scale
// concatenation, qwen3_5.cpp ResidentNvfp4Qkv.)
//
// The MULTIPLY is f32 on both x dtypes, and `col` is always f32 — x's dtype is
// the STORE width only. A bf16 x (PERF-FP8-ALPHA-FOLD / #417) halves the bytes
// this read-modify-write moves, which is its whole cost: it is bandwidth-bound,
// measured at 209.5 GB/s = 77% of the GB10's peak over the merged FP8 GDN
// in_proj output. It also rounds the product, so the byte-identity above holds
// only for the f32 arm; a bf16 x is the vLLM-faithful width (its fp8 linear
// emits the model dtype) but is a real value change and is opt-in at the caller.
void MulColVecF32(Queue& q, Tensor& x, const Tensor& col);

// Splits the fused q/gate attention projection into its two halves. qgate is
// [T, Hq*2*Dh] contiguous, laid out per (t,hq) as [q(Dh) | gate(Dh)]; q_out and
// gate_out are [T,Hq,Dh]. For t in [0,T), hq in [0,Hq):
//   q_out[t,hq,:]    = qgate row t at offset (hq*2*Dh)      .. +Dh
//   gate_out[t,hq,:] = qgate row t at offset (hq*2*Dh + Dh) .. +2*Dh
// T/Hq/Dh are inferred from q_out's shape.
//
// DTYPES. `qgate` is f32 or bf16 (bf16 = a VT_BF16_GEMM_OUT q_proj). `q_out` is
// f32 or bf16: the split is a pure copy, so a bf16 `q_out` over a bf16 `qgate` is
// the IDENTITY on every element and costs half the bytes of an f32 one. That is
// the width vLLM carries — `torch.chunk` does not widen (qwen3_next.py:430) and
// `q_norm` promotes inside its own kernel, not into an allocation. `gate_out` is
// f32 on every arm, because `vt::SigmoidGateBf16` takes an f32 gate.
void AttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate);

// out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])), sigmoid(x)=1/(1+exp(-x)); out
// bf16, attn f32 OR bf16 (the FA-2 prefill path hands bf16 attention out; the
// upcast is exact so bf16-attn is bit-identical to f32-attn holding the same
// values), gate f32 (sigmoid input must not be rounded), same element count.
// The sigmoid output-gate applied to the attention result before the o_proj
// (elementwise on the projection split).
void SigmoidGateBf16(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate);

// Derives the GDN per-head decay g and gate beta from the raw projections
// (gdn-semantics.md §6). g_out/beta_out/araw/braw are [T,Hv]; a_log/dt_bias are
// [Hv]. For t in [0,T), hv in [0,Hv), idx=t*Hv+hv:
//   x  = araw[idx] + dt_bias[hv]
//   sp = softplus(x) = (x > 20) ? x : log1p(exp(x))    (threshold 20)
//   g_out[idx]    = -exp(a_log[hv]) * sp
//   beta_out[idx] = sigmoid(braw[idx])
// T/Hv inferred from g_out's shape. g/beta/a_log/dt_bias are f32. araw/braw
// share either f32 or bf16 dtype and may be inner-contiguous row-strided views
// (the merged `[b,a]` projection has row stride 2*Hv); kernels upcast on load.
void GdnGBeta(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw, const Tensor& braw,
              const Tensor& a_log, const Tensor& dt_bias);

// Splits the GDN mixed-qkv conv output into its q/k/v parts. conv is
// [T, conv_dim] contiguous, laid out per row as [q(key_dim) | k(key_dim) |
// v(value_dim)], conv_dim = 2*key_dim + value_dim. q_out/k_out are [T,key_dim],
// v_out is [T,value_dim]; for each row t:
//   q_out row = conv row [0, key_dim);  k_out row = conv row [key_dim, 2*key_dim);
//   v_out row = conv row [2*key_dim, 2*key_dim + value_dim)
// T = conv.shape[0]; key_dim = q_out.Numel()/T, value_dim = v_out.Numel()/T
// (rows treated row-major). All f32.
void GdnConvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv);

// Splits a merged QKVParallelLinear projection into its q/k/v parts with
// INDEPENDENT head dims (GQA: q_dim != k_dim, unlike the GDN GdnConvSplit which
// assumes q_dim == k_dim). qkv is [T, q_dim + k_dim + v_dim] contiguous, laid
// out per row as [q(q_dim) | k(k_dim) | v(v_dim)]; q_out [T,q_dim], k_out
// [T,k_dim], v_out [T,v_dim], all contiguous, same dtype as qkv (f32 or bf16).
// For each row t: q_out row = qkv row [0,q_dim); k_out row = qkv row
// [q_dim, q_dim+k_dim); v_out row = qkv row [q_dim+k_dim, q_dim+k_dim+v_dim).
// q_dim/k_dim/v_dim are inferred from q_out/k_out/v_out.Numel()/T. Mirrors
// vLLM's qkv_proj output split (qwen3.py Qwen3Attention: one qkv GEMM then
// .split([q_size, kv_size, kv_size], dim=-1)). CPU + CUDA.
void QkvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv);

// Fused GDN post-conv preparation (launch-count fusion; mirror of upstream
// vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
// _fused_post_conv_kernel — "split → l2norm*2 → gating" in a single kernel,
// grid (L, H+HV)). Replaces the GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta
// four-launch chain with one launch, bit-for-bit equal to composing them:
//   [q|k|v]     = split(conv[T, 2*key_dim+value_dim])   (GdnConvSplit)
//   q_out,k_out = l2norm(q), l2norm(k) over Dk           (L2Norm, args.eps)
//   v_out       = v                                      (copy)
//   g_out,beta_out from araw/braw + a_log/dt_bias        (GdnGBeta, §6)
// q_out/k_out [T,Hk,Dk] (l2-normed), v_out [T,Hv,Dv], g_out/beta_out [T,Hv];
// conv [T, 2*Hk*Dk + Hv*Dv]; araw/braw [T,Hv]; a_log/dt_bias [Hv]. araw/braw
// share f32 or bf16 dtype and may have a padded row stride; all gate math is f32.
void GdnPostConv(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                 Tensor& beta_out, const Tensor& conv, const Tensor& araw, const Tensor& braw,
                 const Tensor& a_log, const Tensor& dt_bias, const L2NormArgs& args);

// out[t,c] = F32ToBF16(sigmoid(gl[t]) * sd[t*H+c]); out bf16 [T,H], sd f32
// [T,H], gl f32 with T elements (shape [T] or [T,1]). The shared-expert
// sigmoid gate (moe-semantics.md §5), applied per token to the shared MLP
// output. T inferred from out.shape[0], H = out.Numel()/T.
void SharedExpertGate(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl);

// ─── EXL3 (exllamav3 trellis) reference dequant — CPU tier ───────────────────
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT); vLLM
// implements no EXL3 at the pin, so exllamav3 is the secondary oracle this
// mirrors (.agents/specs/model-dsv4-exl3.md). Row MODEL-DSV4-EXL3 W1a.
//
// An EXL3 linear stores FOUR tensors and NO scales (`exl3.py:38`):
//   trellis  int16 [k/16, n/16, 16*bits]  — 16x16 tiles of 256 codewords
//   suh      fp16  [k]                    — left  sign/scale vector
//   svh      fp16  [n]                    — right sign/scale vector
//   mcg      int32 scalar                 — codebook marker, never read at
//                                           inference (quantize.py:1414-1424)
// and the full-weight reference is `LinearEXL3.get_weight_tensor`
// (`exl3.py:227-237`): reconstruct -> blockwise Hadamard-128 on the left ->
// `* suh[:,None]` -> blockwise Hadamard-128 on the right -> `* svh[None,:]`
// (`exl3_lib/quantize.py:340-358`, both scaled by 1/sqrt(128); `had_k = had_n =
// 128` at `quantize.py:15`). The RUNTIME instead transforms activations
// (`had_r_128` in/out, `exl3.py:183-214`) — that arm is MODEL-DSV4-EXL3 W2.
//
// These are plain host functions, not OpProvider ops: they decode a CHECKPOINT
// FORMAT at load time on the host, exactly like the GGUF/NVFP4 block dequants,
// and W2's device kernels (`exl3_gemm`, `had_r_128`) are gated for byte parity
// AGAINST them rather than dispatched beside them.

// The 16-bit tail-biting codeword for weight `t` (0..255) of one packed tile.
// `tile` is the tile's 16*bits int16 words as stored. Mirrors `dq`
// (`exl3_dq.cuh:15-31`): the window ENDS at weight t, so t's own `bits` bits sit
// in the low positions and weights t-1, t-2 … wrap around the tile above them.
//
// WHY `const void*` AND NOT `const uint16_t*` HERE AND IN THE THREE BELOW. A
// trellis is a borrowed safetensors payload typed kI8 (`vt::Exl3Gemm`: "the
// trellis travels as opaque i8 BYTES"), so its base is 1-byte aligned and lands
// on an odd address in roughly half of all checkpoints. Forming a
// `const uint16_t*` over it is undefined and `-fsanitize=alignment` aborts on
// the load (#2558); these decoders read it through `vt::LoadUnaligned` off a
// byte cursor instead. Every caller still passes the int16 words as stored, and
// a `uint16_t*` converts implicitly, so no call site changed.
uint16_t Exl3TileCodeword(const void* tile, int bits, int t);

// The HALF-INTEGER rate twin of `Exl3TileCodeword` (BACKEND-ROCM frac rates).
// `bits` is the integer KA of `K = KA + 0.5`; the tile is `16*KA + 8` uint16
// words. Position `i`'s 16-bit window ENDS at
// `S(i) = KA*(i+1) + ((i&15)+1)/2 + 8*(i/16)` ring bits — odd positions carry
// the extra bit (mask 0xAAAA, period 16) — and the ring is TAIL-BITING mod
// `tile_bits = 256*KA + 128`, the same wrap the integer read gets from
// `+256*bits` (`exllamav3_ext/quant/frac.cu`, `exl3_dq.cuh:254-286`).
uint16_t Exl3FracTileCodeword(const void* tile, int ka, int t);

// The MCG codebook (cb == 1), three instructions (`codebook.cuh:67-75`):
// `x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` then the two fp16
// halves summed in fp16. Returns that fp16 value widened to f32.
float Exl3DecodeMcg(uint16_t codeword);

// The codebook decode, SELECTED rather than assumed (`codebook.cuh:56-90`).
//
//   cb 0  the original QTIP 3INST: `x *= 89226354; x += 64248484`
//   cb 1  MCG:                     `x *= 0xCBAC1FED`
//
// then, for those two, `x = (x & 0x8fff8fff) ^ 0x3b603b60` and the two fp16
// halves summed in fp16.
//
//   cb 2  `mul1`: `x *= 0x83DCD12D`, then a DIFFERENT SHAPE — the four unsigned
//         bytes of the product are summed into 0x6400 (`__dp4a`), the sum is
//         reinterpreted as an fp16 bit pattern, and an fp16 fused affine map
//         (`* 0x1eee + 0xc931`) yields the value. It is not the mask/xor/pair-sum
//         of the other two, so it cannot be reached by swapping the multiplier.
//         `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` marks every linear this way (#2495).
//
// Any other value REFUSES BY NAME. Upstream defines no other value either.
//
// WHICH ONE A CHECKPOINT USES IS DECIDED BY TENSOR PRESENCE, AND THE POLARITY
// IS THE OPPOSITE OF THE OBVIOUS GUESS. `LinearEXL3` sets `self.mcg =
// (self.mcg_tensor is not None)` (`exl3.py:74-77`) and passes those BOOLEANS to
// `ext.reconstruct` (`:197,223`), so a checkpoint that ships NO `mcg` tensor is
// NOT MCG -- it is cb 0. The SparkInfer DeepSeek-V4 artifact ships an `mcg`
// marker and is cb 1; every stock `turboderp/*-exl3` artifact ships neither and
// is cb 0. Reading absence as MCG decodes with the wrong multiplier, which
// yields a weight with the RIGHT DISTRIBUTION and no correlation to the true
// one -- measured on `turboderp/Llama-3.2-1B-Instruct-exl3` as RMS 0.0385
// against the reference's 0.0361 with a cosine of -0.0006, and as fluent
// nonsense out of the model.
float Exl3DecodeCodeword(uint16_t codeword, int codebook);

// Row-major position (0..255) inside the 16x16 tile that codeword `t` decodes
// into — upstream's `tensor_core_perm` (`exl3_lib/quantize.py:22-42`), which the
// quantizer applies to a row-major tile before encoding.
int Exl3TileRowMajorIndex(int t);

// Decode one packed tile into 256 f32 values in ROW-MAJOR 16x16 order.
// `half` selects the half-integer rate (`bits` = KA, mul1 codebook required by
// the caller's own validation).
void Exl3DecodeTile(const void* tile, int bits, int codebook, float* out256,
                    bool half = false);

// `LinearEXL3.get_inner_weight_tensor` (`exl3.py:222-225`): the pre-Hadamard
// reconstruct. `out` is f32 [k, n] row-major and holds exact fp16 codebook
// values. `k` and `n` must be multiples of 16.
void Exl3ReconstructInner(const void* trellis, int64_t k, int64_t n, int bits, int codebook,
                          float* out, bool half = false);

// `LinearEXL3.get_weight_tensor` (`exl3.py:227-237`): the full dequantized
// weight, f32 [k, n] row-major (k = in_features, both multiples of 128 because
// both sides were Hadamard-transformed at quantization time). `suh`/`svh` are
// the raw fp16 bit patterns as stored — they are NOT assumed to be signs; the
// real DeepSeek-V4 checkpoint folds a per-channel scale into them.
//
// DEVIATION, recorded: upstream runs each 128-term Hadamard as an f32 GEMM;
// this uses the fast Walsh-Hadamard butterfly, which is the same value in exact
// arithmetic but a different f32 summation order. The fp16 round upstream
// performs after each transform (`quantize.py:342-346,351-355` `.to(x_dtype)`)
// absorbs it for all but a fraction of entries, and MODEL-DSV4-EXL3 W2's device
// parity gate is stated against THIS function, not against torch.
void Exl3DequantLinear(const void* trellis, const void* suh,
                       const void* svh, int64_t k, int64_t n, int bits, int codebook,
                       float* out, bool half = false);

// ─── EXL3 device kernels — MODEL-DSV4-EXL3 W2a / W2b ─────────────────────────
//
// The W1a entries above decode a checkpoint FORMAT once, on the host, at load.
// The two ops below run per FORWARD, on whatever device the queue names, and
// are what makes the trellis tower fast rather than merely readable. The
// contracts, the parity tiers, the output-dtype decision and the shape policy
// are settled in `.agents/specs/model-dsv4-exl3.md` `## W2 design`; the summary
// that a reader of this header needs is repeated at each declaration.
//
// DTYPE, stated once. `A`, `A_had`, `suh`, `svh` and (by default) `C` are
// **fp16**, mirroring upstream's own memory format — `LinearEXL3
// .default_out_dtype = out_dtype or torch.half` (`exl3.py:72`) and
// `reconstruct_hgemm`'s `torch.empty(..., dtype = out_dtype or
// self.default_out_dtype)` (`exl3.py:167`). `A` in particular has no freedom:
// `ldmatrix.sync.aligned.m8n8.x4.shared.b16` +
// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (`ptx.cuh:52-74,203-212`)
// read fp16 fragments. `C` may be f32, which is upstream's own `c_fp32` arm
// (`exl3_gemm.cu:134` reads it off `C.dtype()`), kept because the MoE weighted
// reduction earns the width — it is NOT the default, and it is NOT inherited
// from `Exl3DequantLinear`'s `float* out`, which is a host decoder's carrier.
//
// The TRELLIS travels as `DType::kI8`, shape `[k/16, n/16, 32*bits]` BYTES —
// the same opaque-storage idiom the fp8 KV pages use (`DType::kI8` storage,
// see kFp8KvStore above). It is a packed bit stream, not an array of numbers,
// and 32*bits bytes is upstream's `16*K` uint16 words per 16x16 tile.

// `had_r_128` (`quant/hadamard.cu:88-110`):
//   out = (in.view(-1, 128) @ H128) * (args.scale / sqrt(128))
// with `args.pre_scale` multiplied in BEFORE the transform and
// `args.post_scale` AFTER; at most one may be set. `in`/`out` are 2-D, same
// dtype (f16 or f32), same shape, `cols % 128 == 0`, and `out` MAY alias `in`
// (upstream: "Works inplace if y == x", `hadamard.cu:86`). The scale vectors are
// fp16 with `cols` elements. H128 is the natural-order Sylvester matrix
// (`util/hadamard.py:34-42` recurses to `hadamard_1.txt` = "+").
//
// PARITY: the CPU and CUDA arms are BYTE-IDENTICAL, not merely close. Both run
// the same radix-2 FWHT in the same order — levels 1 and 2 on the four values a
// lane holds (`hadamard_inner.cuh:118-129`), levels 4..64 as five xor-shuffle
// steps (`shuffle_had_f4x32`, `:17-44`) — every operation in f32, one multiply
// by `r_scale` at the end, one round to the destination dtype. It is NOT
// bit-comparable to `Exl3DequantLinear`'s Hadamard, which rounds to fp16 after
// every 128-block because it transforms WEIGHTS at quantization time.
void Exl3HadR128(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args);

// `exl3_gemm` (`exl3_gemm.cu:110-309`): the fused EXL3 linear.
//   C = had_r_128( had_r_128(A, pre_scale=suh) @ reconstruct(trellis),
//                  post_scale=svh )
// which is algebraically `A @ Exl3DequantLinear(trellis, suh, svh)` — the two
// differ only in whether the Hadamards ride the activations or the weights
// (`exl3.py:183-214` vs `:227-237`).
//
//   c        f16 or f32 [m, n], contiguous, need NOT be zeroed
//   a        f16 [m, k], contiguous
//   trellis  i8  [k/16, n/16, 32*bits] (bytes), contiguous
//   suh      f16 [k]
//   svh      f16 [n]
//   a_had    f16 [m, k] scratch for the input transform; MAY alias `a`
//   k % 16 == 0, n % 128 == 0, both % 128 for the Hadamards to be defined
//
// PARITY: the trellis DECODE is bit-exact against `Exl3DecodeTile`; the GEMM is
// bounded, because `mma.m16n8k16`'s internal accumulation order is unspecified
// by PTX and a split-K threadblock reduction sits on top. The bound is RMS
// relative 1.0e-3 and 8 fp16 ulps of the output RMS elementwise, set by the fp16
// destination (one store round is already 4.9e-4) and not by summation order
// (f32 over k=4096 contributes 3.8e-6). It is stated here so no gate discovers
// it; see the spec for why a wrong codeword misses by four orders more.
void Exl3Gemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
              const Tensor& suh, const Tensor& svh, Tensor& a_had,
              const Exl3GemmArgs& args);

// ─── The kernel-shape policy, as PURE HOST functions ─────────────────────────
//
// Ported verbatim from `exl3_kernel_map.cu:23-91,153-160` and the shape macros
// at `exl3_kernel_map.cuh:53-60`. They are host functions with no CUDA in them
// ON PURPOSE: the selection table is the part of W2b that can be gated on a
// machine with no GPU, and a table that only a device run can check is a table
// nobody checks.

// Upstream's five-way compute-capability BUCKET (`exl3_devctx.cu:32-46`). Not a
// new classification: `major >= 10` is Blackwell, so GB10 (sm_121) is
// kBlackwell, and every threshold below is upstream's.
enum class Exl3Cc : int {
  kOld = 1,
  kAmpere = 2,
  kAda = 3,
  kHopper = 4,
  kBlackwell = 5,
};
Exl3Cc Exl3CcFromSm(int sm_major, int sm_minor);

// One row of the shape table (`exl3_kernel_map.cuh:53-60`). `shape_idx` is
// 1-based; index 0 is upstream's `nullptr` slot and has no geometry.
struct Exl3GemmShape {
  int tile_m = 0;
  int tile_k = 0;
  int tile_n = 0;
  int sh_stages = 0;
  int frag_stages = 0;
  int block_dim = 0;
};
int Exl3GemmNumShapes();
Exl3GemmShape Exl3GemmShapeParams(int shape_idx);

// `select_gemm_shape` (`exl3_kernel_map.cu:23-75`). Returns a 1-based shape
// index, or 0 when the bucket has no rule (upstream's fallthrough). `multi` is
// upstream's mgemm flag; this wave always passes false (the fused MoE mgemm is
// W2d), and the parameter is kept so the ported table stays readable against its
// anchor.
int Exl3SelectGemmShape(Exl3Cc cc, int size_m, int size_k, int size_n, int bits,
                        bool multi);

// `exl3_gemm_shape_compat` (`exl3_kernel_map.cu:86-91`). DEVIATION, recorded:
// upstream's signature also takes `size_m` and `K` and reads NEITHER, so they
// are dropped here rather than carried as unused parameters that `-Werror`
// would then have to be told about.
bool Exl3GemmShapeCompat(int shape_idx, int size_k, int size_n);

// The empty-block clamp (`exl3_kernel_map.cu:153-160`): a launch never gets more
// blocks than there are k x n tiles to give them, and never fewer than one.
int Exl3GemmNumSms(int shape_idx, int size_k, int size_n, int device_sms);

// ─── The m<=8 GEMV envelope, also PURE HOST — MODEL-DSV4-EXL3 W2c ────────────
//
// Ported from `exl3_gemv.cu:29-42,46-72,110-114`. Host functions for the same
// reason the shape table is: this is where the arm is CHOSEN, and a choice only
// a device run can inspect is a choice nobody inspects. It matters more here,
// because the sentence this arm is usually dismissed with — "Ada/Blackwell are
// memory-bound here and keep the regular kernel" — describes a guard that is
// COMMENTED OUT at `exl3_gemv.cu:53` (and again at `:119`). No
// compute-capability test is live; the shape envelope alone decides.

// `EXL3_GEMV_MAX_M` (`exl3_gemv_kernel.cuh:31`). Above it the kernel has no
// fragment rows to put the extra tokens in.
inline constexpr int kExl3GemvMaxM = 8;

// The HARD constraints (`exl3_gemv.cu:108-114`), free integer tests that run
// before any environment read or device query. `has_su_sv` is upstream's own
// precondition that the call carries suh, A_had and svh: the GEMV kernel does
// the input and output Hadamards itself and has no arm without them.
// `exl3_gemv.cu:110-114` plus ONE extension of ours, not upstream's: `half`
// marks a HALF-INTEGER rate (K+0.5, mul1). A frac tensor reported as bits=KA
// would pass every upstream test at e.g. (3, 2) and then decode through the
// INTEGER-3 GEMV arm — silently wrong — so `half` refuses here by name; the
// frac GEMV arm is owed (BACKEND-ROCM, spec `## Owed`).
bool Exl3GemvHardEligible(int size_m, int size_k, int size_n, int bits, int codebook,
                          bool has_su_sv, bool half = false);

// `exl3_gemv_cfg` (`exl3_gemv.cu:46-72`). Returns -1 (not eligible), 0 (the
// narrow config: 512 threads, one block per 32 output columns) or 1 (the wide
// config: 256 threads, 64 columns).
//
// `narrow_coresident` is `cudaOccupancyMaxActiveBlocksPerMultiprocessor(narrow,
// 512) * num_sms` (`exl3_gemv.cu:127-142`) — a DEVICE query, and a PARAMETER
// here rather than a call made inside, which is what makes the envelope
// gateable with no GPU. For this checkpoint it is also the only input that
// decides anything: w2 (k=2048, n=4096) is eligible without it, and w1/w3
// (k=4096, n=2048) rest entirely on whether it reaches 2048/32 = 64.
//
// `mode` is `VT_EXL3_GEMV` (see Exl3GemvParseMode).
int Exl3GemvSelectConfig(Exl3Cc cc, int size_m, int size_k, int size_n, int bits, int codebook,
                         int mode, int narrow_coresident);

// `VT_EXL3_GEMV`, mirroring upstream's `EXL3_GEMV` (`exl3_gemv.cu:29-34`):
// 0 off, 1 or unset the heuristic, 2 wherever the hard constraints allow,
// 3 force narrow, 4 force wide. The parse is separated from the getenv so it is
// unit-testable without touching the environment.
int Exl3GemvParseMode(const char* env_value);
int Exl3GemvMode();  // the process-cached read of VT_EXL3_GEMV

// `VT_EXL3_GEMV_SMEM`, mirroring `EXL3_GEMV_SMEM` (`exl3_gemv.cu:37-42`):
// -1 or unset the per-bits default, 0 force shuffle extraction, 1 force
// shared-memory staging.
int Exl3GemvParseSmemMode(const char* env_value);
int Exl3GemvSmemMode();

// ─── The fused MoE MLP — MODEL-DSV4-EXL3 W2d ─────────────────────────────────
//
// `exl3_moe` (`exl3_moe.cu:99-301`): ONE call runs gate, activation and down for
// every routed expert of every token in a block, where `Exl3Gemm` alone costs
// one call per (token, expert, projection). Per expert the kernel gathers that
// expert's tokens through `token_sorted`, applies the input Hadamard while
// gathering, runs the gate and up GEMMs into the intermediate buffers, fuses the
// output Hadamards with the activation and the down input Hadamard in one pass,
// runs the down GEMM, and scatter-adds the weighted result into `output_state`.
//
//   output_state    **f32** [bsz, hidden], ZERO-INITIALIZED, ACCUMULATED into
//   hidden_state    f16 [bsz, hidden]
//   tables          nine i64 [num_experts] pointer arrays (Exl3MoeExpertTables)
//   routing         expert_count / token_sorted / weight_sorted
//   temps           four f16 [concurrency, max_tokens_per_expert, dim] buffers
//   hidden % 128 == 0 and intermediate % 128 == 0 (both sides carry a blockwise
//   Hadamard-128, the same rule Exl3Gemm states)
//
// `output_state` IS f32 and that is upstream's own width
// (`TORCH_CHECK_DTYPE(output_state, kFloat)`, `exl3_moe.cu:151`), not a
// widening: the epilogue accumulates one contribution per (token, active
// expert) with an atomicAdd (`hadamard_inner.cuh:469-472`), and an fp16
// accumulator over six weighted contributions loses bits a token gate can see.
// Everything else stays fp16.
//
// PARITY: tiers 1 and 2 are unchanged, because the fused kernel calls the same
// tile loop and the same 128-point butterfly. Against the per-expert
// `Exl3Gemm` loop the bound is TIER 4 — 2.0e-2 relative RMS — and the reason is
// the ACTIVATION, not the GEMM: the loop arm rounds the intermediate to f32,
// activates in f32 and rounds back, while the fused arm keeps it fp16
// throughout as upstream does, and the down projection then sums `intermediate`
// of them. See the spec's `## W2cd design` W2d-7 for the arithmetic.
void Exl3MoeMlp(Queue& q, Tensor& output_state, const Tensor& hidden_state,
                const Exl3MoeExpertTables& tables, const Exl3MoeRouting& routing,
                const Exl3MoeTemps& temps, const Exl3MoeArgs& args);

// `reconstruct_hgemm` (exl3.py:161-218): the EXL3 prefill path for M > 144.
// Dequantizes the trellis to a full fp16 weight matrix, then runs cuBLAS fp16
// GEMM. Upstream dispatches here when `rows > AUTO_RECONSTRUCT_THRESHOLD` (144);
// `Exl3Gemm` (the fused cooperative kernel) handles M <= 144.
//
// The algebra is identical to `Exl3Gemm` — `C = had_r_128(had_r_128(A, suh) @
// reconstruct(trellis), svh)` — but the Hadamards ride the WEIGHTS, not the
// activations. Two sub-paths, selected at M >= 1024 when both dims are
// 128-divisible (exl3.py:184):
//   FUSED:   reconstruct_had_slice emits original-basis weights (both Hadamards
//            + sign vectors folded in), so the GEMM runs on raw `a` and no
//            standalone had_r_128 launches are needed.
//   UNFUSED: reconstruct + had_r_128 on input (suh) + had_r_128 on output (svh).
//
// For N > 32768 (MAX_RECONSTRUCT_SLICE_N), the weight is reconstructed and
// GEMMed in 32768-column slices to bound the temporary (exl3.py:199-211).
//
//   c        f16 or f32 [m, n], contiguous
//   a        f16 [m, k], contiguous
//   trellis  i8  [k/16, n/16, 32*bits] (bytes), contiguous
//   suh      f16 [k]
//   svh      f16 [n]
//   a_had    f16 [m, k] scratch for the input Hadamard (unfused path)
//   w_scratch f16 [k, min(n, 32768)] scratch for the reconstructed weight
//   args     bits and codebook (same as Exl3GemmArgs)
//
// PARITY: the trellis decode and Hadamard are the same code as Exl3Gemm's, so
// the bound is set by cuBLAS's fp16 GEMM accumulation order, which is
// non-deterministic in reduction order. The bound is RMS relative 1.0e-3 and
// 8 fp16 ulps, the same tier as Exl3Gemm.
//
// TEST-ONLY since QUANT-EXL3 W7. No production path calls this overload; a model
// forward uses the overload below. It stays as the reference spelling with a
// caller-owned scratch, which the W7 byte-identity case in
// tests/vt/test_exl3_matmul_dispatch.cpp and tests/vt/test_exl3_gemm.cpp call.
void Exl3ReconstructGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          Tensor& w_scratch, const Exl3GemmArgs& args);

// QUANT-EXL3 W7 (.agents/specs/quant-exl3-recon-scratch.md). The same op with no
// caller scratch: the backend supplies the reconstructed weight from ONE
// persistent, grow-only buffer per (device, queue) that it owns outside every
// scratch pool. This is the spelling a model forward uses. A per-call pool
// scratch is part of a CUDA-graph step's demand profile, so every captured
// graph pinned its own copy of each [k, min(n, 32768)] class; exllamav3's
// `torch.empty` in `reconstruct_hgemm` (exl3.py:161-217) is one shared block
// under torch's caching allocator instead.
//
// Growth is refused while the queue's stream is capturing, by name. A driver
// that runs one eager step at a shape before capturing it never trips that,
// because the buffer's size depends on the weight and not on m.
//
// Internally the registered kernel receives an EMPTY `w_scratch` (data ==
// nullptr, rank 0). A backend that registers `kExl3ReconstructGemm` must honour
// that; CUDA is the only one today.
void Exl3ReconstructGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          const Exl3GemmArgs& args);

}  // namespace vt
