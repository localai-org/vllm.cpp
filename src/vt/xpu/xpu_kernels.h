#pragma once
#include "vt/ops.h"
#include "vt/exl3_grouped.h"
namespace vt::xpu {
void CopyKernel(Queue&, Tensor&, const Tensor&);
void AddKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void SiluAndMulKernel(Queue&, Tensor&, const Tensor&);
void MoeSiluMulKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void SigmoidGateKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void IndexSelectKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void IndexCopyKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void EmbeddingKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void MatmulKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void MatmulBTKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void RmsNormKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);
void LayerNormKernel(Queue&, Tensor&, const Tensor&, const Tensor*, const Tensor*, const LayerNormArgs&);
void GeluTanhKernel(Queue&, Tensor&, const Tensor&);
void GeluErfKernel(Queue&, Tensor&, const Tensor&);
void VisionPosEmbedInterpolateKernel(Queue&, Tensor&, const Tensor&, const VisionPosEmbedArgs&);
void VisionRopeGridKernel(Queue&, Tensor&, const Tensor&, const VisionRopeGridArgs&);
void VisionRopeApplyKernel(Queue&, Tensor&, Tensor&, const Tensor&);
void QkvSplitKernel(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&);
void GreedyArgmaxKernel(Queue&, Tensor&, const Tensor&);
void MappedGreedyArgmaxKernel(Queue&, Tensor&, const Tensor&, const Tensor&, int64_t);
void GreedyRejectionSampleKernel(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&);
void ApplyTopKTopPKernel(Queue&, Tensor&, const Tensor*, const Tensor*);
void RandomSampleKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void ApplyTemperatureKernel(Queue&, Tensor&, const Tensor&, bool);
void ComputeProbsKernel(Queue&, Tensor&, const Tensor&);
void ComputeLogprobsKernel(Queue&, Tensor&, const Tensor&);
void ApplyMinPKernel(Queue&, Tensor&, const Tensor&);
void ApplyPenaltiesKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&);
void ApplyLogitBiasKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&);
void ApplyTokenMaskKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void ApplyAllowedTokenIdsKernel(Queue&, Tensor&, const Tensor&);
void Exl3HadR128Kernel(Queue&, Tensor&, const Tensor&, const Exl3HadArgs&);
void Exl3GemmKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    const Tensor&, Tensor&, const Exl3GemmArgs&);
void Exl3GroupedLinearKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, Tensor&, Tensor&, const Exl3GroupedLinearArgs&);
void Exl3GroupedW8A8Kernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, Tensor&, Tensor&, const Exl3GroupedLinearArgs&);
// Eager diagnostic only: capture the selected packed/reference leaf's F32
// intermediate in caller-owned storage. Refuses fused/prefill routes instead
// of changing dispatch. The normal registered operator never uses this hook.
void Exl3GemmReplayKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                          const Tensor&, Tensor&, const Exl3GemmArgs&, Tensor& raw);
bool Exl3PrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, int,
                       bool matrix = true, bool all_rows = false);
bool GdnChunkedPrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, Tensor&, const Tensor&, const GdnArgs&);
#ifdef VLLM_CPP_XPU_XE2_GDN
void GdnPackedDecodeKernel(Queue&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, Tensor&, const Tensor&, const GdnArgs&);
void GdnPrefillRawGateKernel(Queue&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    Tensor&, const Tensor&, const GdnArgs&);
bool GdnNativePrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&, const Tensor&, Tensor&, const Tensor&, const GdnArgs&);
#endif
void Exl3OutputHadPanel(Queue&, Tensor&, const Tensor&, const Tensor&, int64_t);
void CausalConv1dFwdKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                           Tensor&, const Tensor&, const Tensor&, const CausalConv1dArgs&);
void CausalConv1dUpdateKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                              Tensor&, const Tensor*, const CausalConv1dArgs&);
void CausalConv1dSpecUpdateKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                                  Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const CausalConv1dArgs&);
void GdnPostConvKernel(Queue&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const GdnPostConvArgs&);
void GdnPrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                       const Tensor&, Tensor&, const Tensor&, const GdnArgs&);
void GdnDecodeKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                      const Tensor&, Tensor&, const Tensor*, const GdnArgs&);
void GdnSpecDecodeKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, const GdnArgs&);
void RmsNormGatedKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const RmsNormGatedArgs&);
void GdnStateGatherKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
void GdnStateScatterKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void AttnGateSplitKernel(Queue&, Tensor&, Tensor&, const Tensor&);
void AttnQkNormRopeGateKernel(Queue&, Tensor&, Tensor&, Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, const RmsNormArgs&, const RopeArgs&);
void RopeNeoxKernel(Queue&, Tensor&, Tensor&, const Tensor&, const RopeArgs&);
void RopeCosSinCacheKernel(Queue&, Tensor&, const Tensor&, const RopeArgs&);
void RopeFromCacheKernel(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&, const RopeArgs&);
void ReshapeAndCacheKernel(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&, const Tensor&);
void ReshapeAndCacheFp8Kernel(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&, const Tensor&,
                              Fp8KVCacheDataType, float, float);
void AttentionKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                     const AttentionArgs&);
void PagedAttentionKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
bool PagedAttentionSplitKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
bool PagedAttentionPrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
bool PagedAttentionExl3OneDnnKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
#ifdef VLLM_CPP_XPU_XE2_PREFILL
bool PagedAttentionXe2PrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                    const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
#endif
#ifdef VLLM_CPP_XPU_XE2_VERIFY
bool PagedAttentionXe2DecodeKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
// C4 admission requires a uniform host offset hint, proved against device
// offsets by PagedAttentionKernel before any output/workspace write.
int64_t PagedAttentionXe2VerifyQueryLength(int64_t tokens, int64_t requests,
                                         const int32_t* host_offsets);
// Same admission for the donor and its packed-only device metadata contract.
bool CanUsePagedAttentionXe2Verify(Queue&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
bool PagedAttentionXe2VerifyKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
#endif
}  // namespace vt::xpu
