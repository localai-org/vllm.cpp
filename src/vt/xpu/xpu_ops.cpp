#include "xpu_kernels.h"
#ifdef VLLM_CPP_XPU_ONEDNN
#include "xpu_vision_attention.h"
#endif
namespace vt::xpu {
namespace {
struct Registrar {
  Registrar() {
#define XPU_OP(id, type, fn) RegisterOp(OpId::id, DeviceType::kXPU, reinterpret_cast<void*>(static_cast<type>(&fn)))
    XPU_OP(kCopy, CopyFn, CopyKernel);
    XPU_OP(kCastBf16, CastBf16Fn, CopyKernel);
    XPU_OP(kCastF16, CastF16Fn, CopyKernel);
    XPU_OP(kCastF32, CastF32Fn, CopyKernel);
    XPU_OP(kAdd, AddFn, AddKernel);
    XPU_OP(kSiluAndMul, SiluAndMulFn, SiluAndMulKernel);
    XPU_OP(kMoeSiluMul, MoeSiluMulFn, MoeSiluMulKernel);
    XPU_OP(kSigmoidGateBf16, SigmoidGateBf16Fn, SigmoidGateKernel);
    XPU_OP(kIndexSelect, IndexSelectFn, IndexSelectKernel);
    XPU_OP(kIndexCopy, IndexCopyFn, IndexCopyKernel);
    XPU_OP(kEmbedding, EmbeddingFn, EmbeddingKernel);
    XPU_OP(kMatmul, MatmulFn, MatmulKernel);
    XPU_OP(kMatmulBT, MatmulFn, MatmulBTKernel);
    XPU_OP(kRmsNorm, RmsNormFn, RmsNormKernel);
    XPU_OP(kLayerNorm, LayerNormFn, LayerNormKernel);
    XPU_OP(kGeluTanh, ReluFn, GeluTanhKernel);
    XPU_OP(kGeluErf, ReluFn, GeluErfKernel);
    XPU_OP(kVisionPosEmbedInterpolate, VisionPosEmbedInterpolateFn, VisionPosEmbedInterpolateKernel);
    XPU_OP(kVisionRopeGrid, VisionRopeGridFn, VisionRopeGridKernel);
    XPU_OP(kVisionRopeApply, VisionRopeApplyFn, VisionRopeApplyKernel);
#ifdef VLLM_CPP_XPU_ONEDNN
    XPU_OP(kAttentionDenseFlash, AttentionFn, VisionAttentionDenseFlashKernel);
#endif
    XPU_OP(kQkvSplit, QkvSplitFn, QkvSplitKernel);
    XPU_OP(kGreedyArgmax, GreedyArgmaxFn, GreedyArgmaxKernel);
    XPU_OP(kMappedGreedyArgmax, MappedGreedyArgmaxFn, MappedGreedyArgmaxKernel);
    XPU_OP(kGreedyRejectionSample, GreedyRejectionSampleFn,
           GreedyRejectionSampleKernel);
    XPU_OP(kApplyTopKTopP, ApplyTopKTopPFn, ApplyTopKTopPKernel);
    XPU_OP(kRandomSample, RandomSampleFn, RandomSampleKernel);
    XPU_OP(kApplyTemperature, ApplyTemperatureFn, ApplyTemperatureKernel);
    XPU_OP(kComputeProbs, ComputeProbsFn, ComputeProbsKernel);
    XPU_OP(kComputeLogprobs, ComputeLogprobsFn, ComputeLogprobsKernel);
    XPU_OP(kApplyMinP, ApplyMinPFn, ApplyMinPKernel);
    XPU_OP(kApplyPenalties, ApplyPenaltiesFn, ApplyPenaltiesKernel);
    XPU_OP(kApplyLogitBias, ApplyLogitBiasFn, ApplyLogitBiasKernel);
    XPU_OP(kApplyTokenMask, ApplyTokenMaskFn, ApplyTokenMaskKernel);
    XPU_OP(kApplyAllowedTokenIds, ApplyAllowedTokenIdsFn, ApplyAllowedTokenIdsKernel);
    XPU_OP(kExl3HadR128, Exl3HadR128Fn, Exl3HadR128Kernel);
    XPU_OP(kExl3Gemm, Exl3GemmFn, Exl3GemmKernel);
    XPU_OP(kExl3GroupedLinear, Exl3GroupedLinearFn, Exl3GroupedLinearKernel);
    XPU_OP(kExl3GroupedW8A8, Exl3GroupedLinearFn, Exl3GroupedW8A8Kernel);
    XPU_OP(kCausalConv1dFwd, CausalConv1dFwdFn, CausalConv1dFwdKernel);
    XPU_OP(kCausalConv1dUpdate, CausalConv1dUpdateFn, CausalConv1dUpdateKernel);
    XPU_OP(kCausalConv1dSpecUpdate, CausalConv1dSpecUpdateFn, CausalConv1dSpecUpdateKernel);
    XPU_OP(kGdnPostConv, GdnPostConvFn, GdnPostConvKernel);
    XPU_OP(kGdnPrefill, GdnPrefillFn, GdnPrefillKernel);
#ifdef VLLM_CPP_XPU_XE2_GDN
    XPU_OP(kGdnPrefillRawGate, GdnPrefillRawGateFn, GdnPrefillRawGateKernel);
    XPU_OP(kGdnPackedDecode, GdnPackedDecodeFn, GdnPackedDecodeKernel);
#endif
    XPU_OP(kGdnDecode, GdnDecodeFn, GdnDecodeKernel);
    XPU_OP(kGdnSpecDecode, GdnSpecDecodeFn, GdnSpecDecodeKernel);
    XPU_OP(kRmsNormGated, RmsNormGatedFn, RmsNormGatedKernel);
    XPU_OP(kGdnStateGather, GdnStateGatherFn, GdnStateGatherKernel);
    XPU_OP(kGdnStateScatter, GdnStateScatterFn, GdnStateScatterKernel);
    XPU_OP(kAttnGateSplit, AttnGateSplitFn, AttnGateSplitKernel);
    XPU_OP(kAttnQkNormRopeGate, AttnQkNormRopeGateFn, AttnQkNormRopeGateKernel);
    XPU_OP(kRopeNeox, RopeFn, RopeNeoxKernel);
    XPU_OP(kRopeCosSinCache, RopeCosSinCacheFn, RopeCosSinCacheKernel);
    XPU_OP(kRopeFromCache, RopeFromCacheFn, RopeFromCacheKernel);
    XPU_OP(kReshapeAndCache, ReshapeAndCacheFn, ReshapeAndCacheKernel);
    XPU_OP(kReshapeAndCacheFp8, ReshapeAndCacheFp8Fn, ReshapeAndCacheFp8Kernel);
    XPU_OP(kAttention, AttentionFn, AttentionKernel);
    XPU_OP(kPagedAttention, PagedAttentionFn, PagedAttentionKernel);
#undef XPU_OP
  }
};
[[maybe_unused]] Registrar registrar;
}
}  // namespace vt::xpu
