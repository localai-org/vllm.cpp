// ROCm op-table registration (BACKEND-ROCM).
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_embedding_quant.h"

namespace vt::rocm {

void RmsNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual);
void EmbeddingKernelRocm(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids);
void MulScalarKernelRocm(Queue& q, Tensor& out, const Tensor& x, double scalar);
void SoftCapKernelRocm(Queue& q, Tensor& out, const Tensor& x, double cap);
void AddKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void GeluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void SiluAndMulKernelRocm(Queue& q, Tensor& out, const Tensor& x);

// ClampedSwiGLU (DeepSeek-V4 activation): silu(clamp(gate, max=limit)) *
// clamp(up, -limit, limit) at alpha=1, beta=0. Same layout as SiluAndMul.
void ClampedSwiGLUKernelRocm(Queue& q, Tensor& out, const Tensor& gate_up, float limit);
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
// MLA/DSA campaign W1 (#2715). The batched GEMM lives beside its dense siblings
// in rocm_matmul_hipblaslt.hip; the three data-movement ops in rocm_mla_ops.hip.
void BatchedMatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void ConcatAndCacheMlaKernelRocm(Queue& q, const Tensor& kv_c, const Tensor& k_pe,
                                 Tensor& kv_cache, const Tensor& slot_mapping);
void ConcatMlaNopeRopeKernelRocm(Queue& q, Tensor& out, const Tensor& nope,
                                 const Tensor& rope);
void GatherMlaCacheKernelRocm(Queue& q, Tensor& dst, const Tensor& src_cache,
                              const Tensor& block_table, const Tensor& cu_seq_lens,
                              const Tensor& token_to_seq, const Tensor* seq_starts,
                              int64_t num_tokens);
// The two attention ops GLM-5.3 non-flash reaches and, before rocm_mla_attn.hip,
// REFUSED on gfx1151 (#2926): the reference tier is withdrawn on a part that
// reports PageableMemoryAccess = 0 (docs/ROCM.md:83-85), so an unregistered op
// is a refusal and not a slow path.
void MlaPrefillAttentionKernelRocm(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                                   const Tensor& key, const Tensor& value,
                                   const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k,
                                   const MlaPrefillAttentionArgs& args);
void MlaDecodeAttentionKernelRocm(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                                  const Tensor& kv_cache, const Tensor& block_table,
                                  const Tensor& seq_lens,
                                  const MlaDecodeAttentionArgs& args);
void RopeCosSinCacheKernelRocm(Queue& q, Tensor& cos_sin, const Tensor& pos,
                               const RopeArgs& args);
void GreedyArgmaxKernelRocm(Queue& q, Tensor& out, const Tensor& logits);
void CastBf16KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void CastF32KernelRocm(Queue& q, Tensor& out, const Tensor& in);
// The NARROWING third sibling (rocm_dense_basic.hip). BACKEND-ROCM-EXL3 / #2433:
// one of the exactly two ops an EXL3 checkpoint ran on the CPU reference tier.
void CastF16KernelRocm(Queue& q, Tensor& out, const Tensor& in);
void PermuteVHeadsKernelRocm(Queue& q, Tensor& out, const Tensor& in,
                             int64_t T, int64_t num_k, int64_t rpk, int64_t dv);
void QkvSplitKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                        const Tensor& qkv);
void LayerNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                         const Tensor* bias, const LayerNormArgs& args);
void ReluKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void RopeNeoxKernelRocm(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args);
void RopeFromCacheKernelRocm(Queue& q, Tensor& qs, Tensor* ks, const Tensor& positions,
                             const Tensor& cache, const RopeArgs& args);
void ReshapeAndCacheKernelRocm(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping);
void ReshapeAndCacheFp8KernelRocm(Queue& q, const Tensor& k, const Tensor& v,
                                  Tensor& k_cache, Tensor& v_cache,
                                  const Tensor& slot_mapping,
                                  Fp8KVCacheDataType kind, float k_scale,
                                  float v_scale);
void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args);
void GeluTanhKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void GeluErfKernelRocm(Queue& q, Tensor& out, const Tensor& x);
void MoeRouterTopKKernelRocm(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args, const Tensor* bias);
void ApplyAllowedTokenIdsKernelRocm(Queue& q, Tensor& logits, const Tensor& mask);
void ApplyTokenMaskKernelRocm(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols);
void ApplyTemperatureKernelRocm(Queue& q, Tensor& logits, const Tensor& temp, bool all_random);
void ApplyTopKTopPKernelRocm(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p);
void ComputeProbsKernelRocm(Queue& q, Tensor& probs, const Tensor& logits);
void ComputeLogprobsKernelRocm(Queue& q, Tensor& logprobs, const Tensor& logits);
void RandomSampleKernelRocm(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds);
void ApplyPenaltiesKernelRocm(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                              const Tensor& output_bin_counts, const Tensor& output_mask,
                              const Tensor& frequency_penalties, const Tensor& presence_penalties,
                              const Tensor& repetition_penalties);
void ApplyMinPKernelRocm(Queue& q, Tensor& logits, const Tensor& min_p);
void ApplyLogitBiasKernelRocm(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                              const Tensor& biases);
// Companion MoE-path op (same TU): elementwise silu(gate)*up.
void MoeSiluMulKernelRocm(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up);
void MoeGroupedGemmBf16KernelRocm(Queue&, Tensor&, const Tensor&, const Tensor&,
                                  const Tensor*, const Tensor&);
void MoeGroupedGemmBf16GateUpSiluKernelRocm(Queue&, Tensor&, const Tensor&, const Tensor&,
                                           const Tensor*, const Tensor&, const Tensor&);
void MoeGroupedGemmBf16GateUpSiluNativeKernelRocm(Queue&, Tensor&, const Tensor&, const Tensor&,
                                                 const Tensor*, const Tensor&, const Tensor&);
void MoeGroupedGemmBf16WeightedKernelRocm(Queue&, Tensor&, const Tensor&, const Tensor&,
                                          const Tensor*, const Tensor&, const Tensor&);
void MoeCombinePreweightedKernelRocm(Queue&, Tensor&, const Tensor&, const Tensor*, float);
// MoE-path combine/gate ops (rocm_moe_chain.hip): shared-expert gate and the
// weighted expert combinations.
void SharedExpertGateKernelRocm(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl);
void MoeCombineKernelRocm(Queue& q, Tensor& out, const Tensor& expert_out,
                          const Tensor& weights, const Tensor* shared, float routed_scale);
void MoeCombineGateKernelRocm(Queue& q, Tensor& out, const Tensor& expert_out,
                              const Tensor& weights, const Tensor& sd, const Tensor& gl);
// Grouped quant expert GEMM (rocm_grouped_gemm.hip): Q8_0/Q4_K/Q5_K/Q6_K native.
void MatmulBTQuantKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void MatmulBTQuantGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight, const Tensor& expert_ids);
// The fused MoE gate+up+clamped-SwiGLU grouped GEMM (rocm_moe_gate_up_swiglu.hip).
// GLM-5.3-Flash reads this and kMatmulBTQuantGrouped through `vt::OpRegistered`
// before it takes a device queue (glm5_next_forward.cpp:307-311, #2942).
void MoeGateUpSwiGLUGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                      const Tensor& gate_w, const Tensor& up_w,
                                      const Tensor& expert_ids, float limit);
// BACKEND-ROCM-GDN-KERNELS family 1 (rocm_gdn_state.hip): the indexed state I/O
// pair `IndexedGdnOpsNative()` requires (issue #41, spec rocm-gdn-kernels.md).
void GdnStateGatherKernelRocm(Queue& q, Tensor& working, const Tensor& cache,
                              const Tensor& state_idx, const Tensor* has_initial_state);
void GdnStateScatterKernelRocm(Queue& q, Tensor& cache, const Tensor& working,
                               const Tensor& state_idx);
// BACKEND-ROCM-GDN-KERNELS family 2 (rocm_gdn_conv.hip): causal conv1d prefill
// fwd + decode update (gdn-semantics §2/§3).
void CausalConv1dFwdKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                               const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                               const Tensor& his, const CausalConv1dArgs& args);
void CausalConv1dUpdateKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                                  const Tensor* bias, Tensor& conv_state,
                                  const Tensor* conv_state_indices,
                                  const CausalConv1dArgs& args);
// BACKEND-ROCM-GDN-KERNELS family 3 (rocm_gdn_postconv.hip): the fused
// post-conv glue (conv-split + q/k l2norm + g/beta), gdn-semantics §4/§6.
void GdnPostConvKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                           Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                           const Tensor& araw, const Tensor& braw, const Tensor& a_log,
                           const Tensor& dt_bias, const L2NormArgs& args);
// BACKEND-ROCM-GDN-KERNELS family 4 (rocm_gdn_scan.hip): the gated-delta
// recurrence over the portable scan (gdn-semantics §7/§8).
void GdnPrefillKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta,
                          Tensor& state, const Tensor& qsl, const GdnArgs& args);
void GdnDecodeKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor* state_idx, const GdnArgs& args);
// BACKEND-ROCM-KDA (#2942, rocm_kda_scan.hip): per-K-channel-decay gated-delta
// recurrence. Same shape as GdnPrefillFn; the only difference is g is [T,Hv,Dk]
// (per-channel) not [T,Hv] (per-head). Staged slice: unreached by the model
// forward until the CPU-only refusal at glm5_next_kda.cpp:322 lifts (W9c-2).
void KdaGatedDeltaRuleKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                                 const Tensor& v, const Tensor& g, const Tensor& beta,
                                 Tensor& state, const Tensor& qsl, const GdnArgs& args);
// BACKEND-ROCM-KPOOL (#2942, rocm_glm5_next_kpool.hip): the GLM-5.3-Flash
// k-pool DSA indexer pair. Staged slice: unreached by the model forward until
// the compose (W9c-0, #2415) that would call them exists. No CPU provider by
// design — the host reference glm5_next_dsa.cpp is the oracle, not a second
// registration under the same OpId.
void Glm5NextKpoolCompressRocm(Queue& q, Tensor& pool_keys, Tensor& pool_indices,
                               Tensor& pool_valid, Tensor& num_pools, const Tensor& packed,
                               const Tensor& ape);
void Glm5NextKpoolSelectRocm(Queue& q, Tensor& topk_indices, Tensor& index_scores,
                             const Tensor& q_states, const Tensor& head_weights,
                             const Tensor& pool_keys, const Tensor& pool_indices,
                             const Tensor& pool_valid, const Tensor& num_pools,
                             const Tensor& valid_keys, const Tensor& q_mask,
                             const Glm5NextKpoolSelectArgs& args);
// BACKEND-ROCM-GDN-KERNELS family 5 (rocm_gdn_fused.hip): gated RMSNorm,
// sigmoid gate, and the fused full-attention preamble.
void RmsNormGatedKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args);
void SigmoidGateBf16KernelRocm(Queue& q, Tensor& out, const Tensor& attn,
                               const Tensor& gate);
void AttnGateSplitKernelRocm(Queue& q, Tensor& query, Tensor& gate, const Tensor& packed);
// BACKEND-ROCM-EXL3 (rocm_exl3.hip): the fused EXL3 trellis linear, transcribed
// from the PORTABLE CPU reference (cpu_exl3_kernels.cpp / cpu_exl3_dequant.cpp)
// and not from the CUDA kernel, whose 90 KiB shared-memory budget, m16n8k16 mma,
// ldmatrix and cp.async have no gfx1151 form. See .agents/specs/backend-rocm-exl3.md.
void Exl3GemmKernelRocm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                        const Tensor& suh, const Tensor& svh, Tensor& a_had,
                        const Exl3GemmArgs& args);
// BACKEND-ROCM EXL3 slice B (rocm_exl3_recon.hip, spec rocm-exl3-gemv): the
// unfused reconstruct+hipBLAS arm of the M > 144 dispatch — registering it is
// what makes dense_attn_block.h's threshold reachable on ROCm at all.
void Exl3ReconstructGemmKernelRocm(Queue& q, Tensor& c, const Tensor& a,
                                   const Tensor& trellis, const Tensor& suh,
                                   const Tensor& svh, Tensor& a_had,
                                   Tensor& w_scratch, const Exl3GemmArgs& args);
// ROCM-FUSED-NORM-ROPE (rocm_mla_fused_norm_rope.hip, #2564): the MLA
// norm-rope fold. Registering it is what makes `mla_attention.cpp`'s `fused_nr`
// predicate true on this backend, and that predicate -- not the op call -- is
// what a keep-quant MLA checkpoint depends on.
void FusedNormRopeKernelRocm(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                             const Tensor& w, const Tensor& positions, const Tensor& cache,
                             const RmsNormArgs& norm_args, const RopeArgs& rope_args);
// The gate-free sibling (rocm_gdn_fused.hip): the Qwen3-dense qk-norm + partial
// RoPE preamble. Registered so the fused-chain recipe `kAttnQkNormRope` takes its
// fast realisation on ROCm instead of the Tier-0 composite.
void AttnQkNormRopeKernelRocm(Queue& q, Tensor& q3, Tensor& k3, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const Tensor& positions, const RmsNormArgs& na,
                              const RopeArgs& ra);
void AttnQkNormRopeGateKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                                  const Tensor& qgate, const Tensor& kf,
                                  const Tensor& q_norm, const Tensor& k_norm,
                                  const Tensor& cos_sin, const RmsNormArgs& na,
                                  const RopeArgs& ra);

namespace {

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelRocm)));
    RegisterOp(OpId::kEmbedding, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelRocm)));
    RegisterOp(OpId::kEmbeddingQuant, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingQuantKernelRocm)));
    RegisterOp(OpId::kMulScalar, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MulScalarFn>(&MulScalarKernelRocm)));
    RegisterOp(OpId::kSoftCap, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SoftCapFn>(&SoftCapKernelRocm)));
    RegisterOp(OpId::kAdd, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernelRocm)));
    RegisterOp(OpId::kGeluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&GeluAndMulKernelRocm)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelRocm)));
    RegisterOp(OpId::kClampedSwiGLU, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ClampedSwiGLUFn>(&ClampedSwiGLUKernelRocm)));
    RegisterOp(OpId::kMatmul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelRocm)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernelRocm)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernelRocm)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernelRocm)));
    RegisterOp(OpId::kCastBf16, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16KernelRocm)));
    RegisterOp(OpId::kCastF16, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastF16Fn>(&CastF16KernelRocm)));
    RegisterOp(OpId::kPermuteVHeads, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<PermuteVHeadsFn>(&PermuteVHeadsKernelRocm)));
    RegisterOp(OpId::kCastF32, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32KernelRocm)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernelRocm)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernelRocm)));
    RegisterOp(OpId::kRelu, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernelRocm)));
    RegisterOp(OpId::kGeluTanh, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluTanhKernelRocm)));
    RegisterOp(OpId::kGeluErf, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluErfKernelRocm)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelRocm)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RopeFromCacheFn>(&RopeFromCacheKernelRocm)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernelRocm)));
    RegisterOp(OpId::kReshapeAndCacheFp8, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFp8Fn>(
                   &ReshapeAndCacheFp8KernelRocm)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<PagedAttentionFn>(&PagedAttentionKernelRocm)));
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelRocm)));
    // Full V1 sampler surface for Hermes (temp/top-p/allowed ids/penalties/...).
    RegisterOp(OpId::kApplyTemperature, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTemperatureFn>(&ApplyTemperatureKernelRocm)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPKernelRocm)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsKernelRocm)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ComputeLogprobsFn>(&ComputeLogprobsKernelRocm)));
    RegisterOp(OpId::kRandomSample, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleKernelRocm)));
    RegisterOp(OpId::kApplyPenalties, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyPenaltiesFn>(&ApplyPenaltiesKernelRocm)));
    RegisterOp(OpId::kApplyMinP, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ApplyMinPFn>(&ApplyMinPKernelRocm)));
    RegisterOp(OpId::kApplyLogitBias, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyLogitBiasFn>(&ApplyLogitBiasKernelRocm)));
    RegisterOp(OpId::kApplyAllowedTokenIds, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyAllowedTokenIdsFn>(&ApplyAllowedTokenIdsKernelRocm)));
    RegisterOp(OpId::kApplyTokenMask, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ApplyTokenMaskFn>(&ApplyTokenMaskKernelRocm)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernelRocm)));
    RegisterOp(OpId::kMoeGroupedGemmBf16, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeGroupedGemmBf16Fn>(
                   &MoeGroupedGemmBf16KernelRocm)));
    RegisterOp(OpId::kMoeGroupedGemmBf16GateUpSilu, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeGroupedGemmBf16GateUpSiluFn>(
                   &MoeGroupedGemmBf16GateUpSiluKernelRocm)));
    RegisterOp(OpId::kMoeGroupedGemmBf16GateUpSiluNative, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeGroupedGemmBf16GateUpSiluNativeFn>(
                   &MoeGroupedGemmBf16GateUpSiluNativeKernelRocm)));
    RegisterOp(OpId::kMoeGroupedGemmBf16Weighted, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeGroupedGemmBf16WeightedFn>(
                   &MoeGroupedGemmBf16WeightedKernelRocm)));
    RegisterOp(OpId::kMoeCombinePreweighted, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeCombinePreweightedFn>(
                   &MoeCombinePreweightedKernelRocm)));
    RegisterOp(OpId::kSharedExpertGate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<SharedExpertGateFn>(&SharedExpertGateKernelRocm)));
    RegisterOp(OpId::kMoeCombine, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernelRocm)));
    RegisterOp(OpId::kMoeCombineGate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MoeCombineGateFn>(&MoeCombineGateKernelRocm)));
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernelRocm)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedKernelRocm)));
    RegisterOp(OpId::kMoeGateUpSwiGLUGrouped, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MoeGateUpSwiGLUGroupedFn>(
                   &MoeGateUpSwiGLUGroupedKernelRocm)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnStateGatherFn>(&GdnStateGatherKernelRocm)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnStateScatterFn>(&GdnStateScatterKernelRocm)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernelRocm)));
    RegisterOp(OpId::kCausalConv1dUpdate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernelRocm)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GdnPostConvFn>(&GdnPostConvKernelRocm)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernelRocm)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernelRocm)));
    RegisterOp(OpId::kKdaGatedDeltaRule, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<KdaGatedDeltaRuleFn>(
                   &KdaGatedDeltaRuleKernelRocm)));
    RegisterOp(OpId::kGlm5NextKpoolCompress, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<Glm5NextKpoolCompressFn>(
                   &Glm5NextKpoolCompressRocm)));
    RegisterOp(OpId::kGlm5NextKpoolSelect, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<Glm5NextKpoolSelectFn>(
                   &Glm5NextKpoolSelectRocm)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<RmsNormGatedFn>(&RmsNormGatedKernelRocm)));
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16KernelRocm)));
    RegisterOp(OpId::kAttnGateSplit, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<AttnGateSplitFn>(&AttnGateSplitKernelRocm)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelRocm)));
    RegisterOp(OpId::kExl3ReconstructGemm, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<Exl3ReconstructGemmFn>(&Exl3ReconstructGemmKernelRocm)));
    RegisterOp(
        OpId::kFusedNormRope, DeviceType::kROCM,
        reinterpret_cast<void*>(static_cast<FusedNormRopeFn>(&FusedNormRopeKernelRocm)));
    RegisterOp(OpId::kAttnQkNormRopeGate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernelRocm)));
    RegisterOp(OpId::kAttnQkNormRope, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeFn>(&AttnQkNormRopeKernelRocm)));
    // MLA/DSA campaign W1 (#2715). Before these lines each of these four
    // resolved to NO native ROCm kernel: the portable CPU reference tier where
    // it is eligible (which docs/ROCM.md:60-61 disqualifies a performance result
    // on) and a REFUSAL where #2511 withdrew it — see rocm_mla_ops.hip's header.
    // The DSA indexer pair (kDsaIndexerLogits, kDsaTopkSelect) is still absent,
    // so the GLM-5.3 speed axis stays VOID and a SPARSE step still refuses.
    RegisterOp(OpId::kBatchedMatmul, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<BatchedMatmulFn>(&BatchedMatmulKernelRocm)));
    RegisterOp(OpId::kConcatAndCacheMla, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ConcatAndCacheMlaFn>(&ConcatAndCacheMlaKernelRocm)));
    RegisterOp(OpId::kConcatMlaNopeRope, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<ConcatMlaNopeRopeFn>(&ConcatMlaNopeRopeKernelRocm)));
    RegisterOp(OpId::kGatherMlaCache, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<GatherMlaCacheFn>(&GatherMlaCacheKernelRocm)));
    // #2926. Before these two lines GLM-5.3 non-flash could not emit a token on
    // gfx1151 at all: `GetOp` refused `kMlaPrefillAttention` on the first
    // (prefill) forward and `kMlaDecodeAttention` on every step after it, and
    // the portable reference tier that used to absorb them is withdrawn on this
    // part. Spec `.agents/specs/rocm-mla-attention-nonflash.md`.
    RegisterOp(OpId::kMlaPrefillAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MlaPrefillAttentionFn>(&MlaPrefillAttentionKernelRocm)));
    RegisterOp(OpId::kMlaDecodeAttention, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<MlaDecodeAttentionFn>(&MlaDecodeAttentionKernelRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
