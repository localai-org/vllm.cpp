// Internal Qwen3.5/3.6 model-selection contracts. Not installed.
// Packed-decode selection mirrors vLLM v0.25.0
// qwen_gdn_linear_attn.py:1286-1298 @ 702f4814, with the first local consumer
// deliberately scoped to the 27B CUDA packed-BA/BF16-activation path. The SSM
// cache dtype is independent (FP32 in the gate checkpoint).
#pragma once

#include <cstdint>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/device.h"           // vt::DeviceType, for the ActDType resolver
#include "vt/paged_attn_route.h"  // W10 repair (#1865): the uniform-spec shape guard
#include "vt/tensor.h"

namespace vt {
struct Queue;
class Backend;
}  // namespace vt

namespace vllm {

struct GdnStateCache;
struct PagedKvCache;
struct HfConfig;
struct GdnLayerWeights;
struct OwnedTensor;

namespace v1 {
struct GDNAttentionMetadata;
struct CommonAttentionMetadata;
}  // namespace v1

// Test-only entry point (SPEC-MTP I5a): run one GDN layer's paged forward over a
// batched step (spec or non-spec, per `meta`) and return the [T*H] output on host
// (f32). Drives the real per-step upload (BuildStepDevInputs) + layer assembly
// (GdnBlockPaged), so the synthetic spec-branch test exercises the exact
// production routing. Defined in qwen3_5.cpp.
// Test-only entry point (PERF-27B-GDN-FP8-QKVZ): run ONE GDN layer's FP8 input
// projection over `h_host` ([T*H] f32, rounded to bf16 on upload exactly like
// the real forward's embed target) through EITHER the merged single-GEMM arm
// (`merged=true`) or the exact two legacy fp8 GEMMs, and return the
// concatenation [mixed_qkv | z] as [T*(conv_dim+value_dim)] f32. `z` is
// produced at BF16 when `z_bf16` (the `GdnOutDType` default, on every arm since
// #1168) and upcast on the way out, so the two arms are directly byte-comparable. The
// merged arm is byte-identical to the split arm by construction; this proves it
// on the real fp8 GEMM. CUDA-only (the fp8 W8A8 path is). Defined in
// qwen3_5.cpp.
std::vector<float> ProjectGdnFp8QkvzForTest(vt::Queue queue,
                                            const GdnLayerWeights& w,
                                            const std::vector<float>& h_host,
                                            int64_t T, int64_t conv_dim,
                                            int64_t value_dim, bool merged,
                                            bool z_bf16);

std::vector<float> GdnBlockPagedForTest(vt::Queue queue, const GdnLayerWeights& w,
                                        const HfConfig& cfg,
                                        const std::vector<float>& h_host,
                                        const v1::GDNAttentionMetadata& meta,
                                        std::vector<float>& ssm_host,
                                        std::vector<float>& conv_host,
                                        int64_t num_slots, int64_t conv_len,
                                        int64_t T);

// ─── FA-2 attention dtype/lane class (SPEC-DFLASH2 W10 repair, #1865) ────────
//
// WHY A SEAM. `FullAttnBlockPaged` (qwen3_5.cpp) selects the attention dtype —
// `attn_dt = fa2-eligible ? kBF16 : kF32` — and the CUDA dispatch
// (cuda_paged_attn.cu) then requires that bf16 query before ANY FA-2 lane,
// spec-as-decode included, can serve the batch. #1865 is what the coupling
// costs when the two sides disagree: the runner CLASSIFIED every q=9 verify
// and the model still handed the dispatch an f32 query, so the admission died
// on its dtype conjuncts, the verify fell to the CUDA-core prefill flash
// (`PagedFlashKernel`, nsys-proven), and no counter moved. The predicate was
// inline in a `.cu`-coupled model file where the CPU tier can neither compile
// the dispatch nor make `fa2_platform` true through the runner; this is the
// same extraction W10 made for the vt-side split (include/vt/paged_attn_route.h)
// so a test without a GPU pins the selection.
//
// The INPUTS are the call site's own reads, taken by value so the function is
// pure: topology, step shape, the runner's spec classification, and each lane
// toggle exactly as the model-side wrappers answer them (`Fa2PrefillOn()`,
// `Fa2Decode*On()`, `FuseAttnPreambleOn(...) && sdi.has_attn_cos_sin`,
// platform FA-2 support). A toggle wrapper already answers `false` when
// `VLLM_CPP_FLASH_ATTN` is not compiled, so a no-FA2 build classifies `kNone`
// everywhere and stays on the f32 lanes byte-identically.
struct DenseFa2Eligibility {
  // Topology (config): query heads, KV heads, head dim.
  int64_t num_q_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  // Step shape.
  int64_t num_tokens = 0;
  int64_t num_reqs = 0;
  // The runner's spec-as-decode classification (CommonAttentionMetadata /
  // PagedAttentionArgs field; 0 = not a classified verify).
  int64_t uniform_spec_query_len = 0;
  bool causal = true;
  // Call-site environment.
  bool kv_cache_bf16 = false;       // kv.dtype == DType::kBF16
  bool kv_block_multiple_16 = false;  // kv.block_size % 16 == 0
  bool preamble_with_cos_sin = false;  // FuseAttnPreambleOn(fp4) && sdi.has_attn_cos_sin
  bool fa2_platform = false;        // GetPlatform(...).supports_fa2_attention()
  // Lane toggles, one per env switch (each false when FA2 is not compiled).
  bool prefill_on = false;          // Fa2PrefillOn()      (VT_FA2_PREFILL)
  bool decode_r4_on = false;        // Fa2Decode4BOn()     (VT_FA2_DECODE_4B)
  bool decode_r6_on = false;        // Fa2DecodeOn()       (VT_FA2_DECODE)
  bool decode_r8_on = false;        // Fa2Decode35BOn()    (VT_FA2_DECODE_35B)
  bool spec_decode_on = false;      // Fa2SpecDecodeOn()   (VT_FA2_SPEC_DECODE)
};

// The class decides ONLY the model-side presentation (bf16 q/k/out vs f32);
// the CUDA dispatch keeps its own admission and every inadmissible batch
// falls through to a lane that serves the presented dtype (WMMA for bf16,
// CUDA-core flash for f32).
enum class DenseFa2Class {
  kNone,        // f32 presentation, the graph-captured fallback lanes
  kPrefill,     // FA-2 prefill varlen (num_tokens > num_reqs)
  kDecode,      // FA-2 split-KV pure decode (num_tokens == num_reqs)
  kSpecVerify,  // FA-2 split-KV spec-as-decode (classified uniform verify)
};

inline DenseFa2Class ClassifyDenseFa2(const DenseFa2Eligibility& e) {
  // Conjuncts shared by every FA-2 lane: the fused bf16 preamble must be able
  // to emit the query this class promises, the BINARY must carry SASS for the
  // device (fa2_platform reads the compiled-arch manifest, #1357), and the
  // KV store must be the bf16 flash_attn layout at head_dim 256.
  const bool base = e.preamble_with_cos_sin && e.fa2_platform &&
                    e.kv_cache_bf16 && e.head_dim == 256;
  if (!base) return DenseFa2Class::kNone;
  // The same three d256 decode topologies the CUDA admission gates on, each
  // behind its own rollback toggle (MUST mirror cuda_paged_attn.cu).
  const bool decode_topology =
      (e.num_q_heads == 16 && e.num_kv_heads == 4 && e.decode_r4_on) ||
      (e.num_q_heads == 24 && e.num_kv_heads == 4 && e.decode_r6_on) ||
      (e.num_q_heads == 16 && e.num_kv_heads == 2 && e.decode_r8_on);
  // W10 repair (#1865): a CLASSIFIED uniform verify selects bf16 through the
  // SPEC lane's OWN gates — `spec_decode_on` (VT_FA2_SPEC_DECODE) plus the
  // decode-arm conjuncts the CUDA admission composes — never through the
  // PREFILL lever. Before this arm existed the verify's bf16-ness rode
  // `prefill_on`, and any state with the prefill arm off and the spec lane on
  // handed the dispatch an f32 query: the admission then failed on its dtype
  // conjuncts and the classified verify silently fell to the CUDA-core prefill
  // flash (the #1865 nsys finding), while VT_FA2_SPEC_DECODE flips moved
  // nothing. Checked FIRST because `PagedAttnIsPrefill` gives an admitted
  // verify the DECODE class; the shape guard keeps a stale classification over
  // a rewritten batch out (S == q*S only at q == 1).
  if (vt::PagedAttnUniformSpecShape(e.num_tokens, e.num_reqs,
                                    e.uniform_spec_query_len) &&
      decode_topology && e.spec_decode_on && e.kv_block_multiple_16 &&
      e.causal) {
    return DenseFa2Class::kSpecVerify;
  }
  if (decode_topology && e.kv_block_multiple_16 && e.causal &&
      e.num_tokens == e.num_reqs) {
    return DenseFa2Class::kDecode;
  }
  if (e.prefill_on && e.num_tokens > e.num_reqs) return DenseFa2Class::kPrefill;
  return DenseFa2Class::kNone;
}

}  // namespace vllm

namespace vllm::detail {

// Borrow a validated model-owned contiguous dim0 interval, preserving the
// producer row stride. Other token maps/layouts retain the checked gather.
std::optional<vt::Tensor> GdnContiguousTokenRowsView(
    const vt::Tensor& source, const std::vector<int32_t>& indices);

// Two writable views must exactly partition a contiguous model-owned output.
// Any gap, overlap, permutation or unsupported layout keeps checked scatter.
std::optional<std::array<vt::Tensor, 2>> GdnContiguousTokenOutputViews(
    const vt::Tensor& output, const std::vector<int32_t>& spec_indices,
    const std::vector<int32_t>& prefill_indices);

struct GdnPackedDecodeEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool has_packed_ba = false;
  bool merged_ba_enabled = false;
  bool dtype_compatible = false;
  bool has_state_indices = false;
  int64_t num_prefills = 0;
  int64_t num_prefill_tokens = 0;
  int64_t num_decodes = 0;
  int64_t num_decode_tokens = 0;
  int64_t num_spec_decodes = 0;
  int64_t num_spec_decode_tokens = 0;
  int64_t num_actual_tokens = 0;
};

bool ShouldUsePackedGdnDecode(const GdnPackedDecodeEligibility& eligibility);

// Process-level ENV resolution of the packed pure-decode arm for the real 27B
// dense CUDA gate (owner resident, pure non-spec one-token decode assumed).
// Mirrors, field for field, the exact process-cached env couplings the model
// wires into GdnPackedDecodeEligibility (qwen3_5.cpp GdnBlockPaged):
// PackedGdnDecodeRuntimeEnabled (VT_GDN_PACKED_DECODE), MergedGdnBaEnabled
// (master VT_GDN_MERGED_PROJ + leaf VT_GDN_MERGED_BA), and dtype_compatible —
// GdnInDType (VT_GDN_IN_BF16) == BF16, GdnOutDType (VT_GDN_OUT_BF16) == BF16 —
// its default on every arm since #1168 — MergedGdnBaOutputDType(packed)
// (VT_GDN_BA_OUT_BF16) == BF16. Splitting BA (master OR leaf) or reverting any
// coupled dtype therefore deselects packed and runs the decomposed recurrence:
// the gate's dispatch-count contract must expect ZERO packed launches on those
// arms, exactly like VT_GDN_PACKED_DECODE=0. Fields carry the raw getenv
// values (nullptr = unset) so the CPU tier can pin the truth table.
struct GdnPackedDecodeEnvConfig {
  const char* packed_decode = nullptr;  // VT_GDN_PACKED_DECODE
  const char* merged_proj = nullptr;    // VT_GDN_MERGED_PROJ (master)
  const char* merged_ba = nullptr;      // VT_GDN_MERGED_BA (leaf)
  const char* in_bf16 = nullptr;        // VT_GDN_IN_BF16
  const char* out_bf16 = nullptr;       // VT_GDN_OUT_BF16
  const char* ba_out_bf16 = nullptr;    // VT_GDN_BA_OUT_BF16
};

bool PackedGdnDecodeEnvSelected(const GdnPackedDecodeEnvConfig& env);

// PERF-27B-GDN-PACKED-REACHABLE (#365) — what `dtype_compatible` above is
// ACTUALLY about.
//
// Packed-decode selection happens BEFORE `ProjectGdnQkvz` runs, because the
// decision changes how BA is projected (`ProjectGdnBA(..., packed_decode)`), so
// the eligibility cannot read `mixed.dtype`; it has to PREDICT it. This mirrors
// `ProjectGdnQkvz`'s branch order exactly — the merged BF16 `in_proj_qkvz`
// owner is checked first, then the native-FP8 owner, then the split BF16 owner
// — and `fp8_out_dtype` is sourced from the same single helper the projection
// itself reads, so the prediction cannot drift from what the GEMM emits.
struct GdnMixedQkvDTypeInputs {
  bool has_bf16_qkvz_owner = false;  // w.in_proj_qkvz populated (27B bf16, 4B)
  bool has_fp8_qkv_owner = false;    // w.in_proj_qkv_fp8 populated (modelopt)
  // PERF-GDN-PACKED-BRIDGE (#365). ShouldUseMergedGdnFp8Qkvz held for this
  // layer, i.e. ProjectGdnQkvz takes the MERGED fp8 arm rather than the split
  // one. Required because `VT_GDN_FP8_IN_BF16` narrows the merged arm ONLY --
  // `fp8_indt` reaches MergedFp8QkvzD and nothing else, while the split arm
  // still hardcodes F32. A prediction that ignored the arm would claim BF16 on
  // a split-arm checkpoint, which is the UNSAFE direction: vt::GdnPackedDecode
  // would then be handed an f32 mixed_qkv against bf16 a/b/out and THROW.
  bool fp8_merged_arm = false;
  vt::DType in_dtype = vt::DType::kF32;       // GdnInDType()
  vt::DType fp8_out_dtype = vt::DType::kF32;  // the fp8 in_proj epilogue's dtype
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4). `w.in_proj_qkv_exl3` populated, i.e.
  // ProjectGdnQkvz takes the trellis arm.
  //
  // EXTENDED here rather than left to the fallthrough, because the fallthrough
  // is a DEFAULT and this is a statement. The predictor names every arm the
  // projection has, so the two branch orders can be read against each other; an
  // arm resolved by "whatever is left" is how the deleted
  // `in_proj_qkv_fp8.Empty()` proxy above came to stand for a dtype question it
  // could not answer.
  //
  // LAST, and defaulted, so the existing aggregate initializers keep their
  // meaning. At today's dtypes this answers `in_dtype`, exactly as the trailing
  // default does, so no test can separate the two; that is recorded rather than
  // hidden. The branch earns its place when the trellis arm's epilogue dtype
  // moves, which is the same event that moves `ProjectGdnQkvz`'s EXL3 rung.
  bool has_exl3_qkv_owner = false;
};

vt::DType GdnProjectedMixedQkvDType(const GdnMixedQkvDTypeInputs& in);

// PERF-GDN-PACKED-BRIDGE (#365) -- the SINGLE decision for the dtype the
// native-FP8 MERGED GDN in_proj emits, shared by the two parties that must
// agree about it: the PRODUCER (`ProjectGdnQkvz`'s `fp8_indt`, handed to
// `MergedFp8QkvzD`, which allocates the buffer) and the PREDICTOR
// (`GdnFp8MixedQkvDType`, read by the packed-decode eligibility BEFORE the
// projection has run, because the decision feeds `ProjectGdnBA`). They cannot
// drift because they are the same call.
//
// Terms are PERF-FP8-ALPHA-FOLD's own, verbatim: the opt-in toggle
// (`VT_GDN_FP8_IN_BF16`, default OFF), `indt == BF16` so `VT_GDN_IN_BF16`'s
// documented rollback still restores f32 on this arm too, and `outdt == BF16`,
// which keeps the chain dtype-uniform. That third term used to be described as
// what "confines the narrowing to the dense 27B, since the 35B is MoE and
// `GdnOutDType` is f32 there"; #1168 made `outdt` BF16 on both arms, so the
// DEFAULT-OFF toggle is now the only thing keeping this inert on the 35B.
vt::DType GdnFp8MergedMixedQkvDType(bool fp8_in_bf16_enabled, vt::DType in_dtype,
                                    vt::DType out_dtype);

// The packed-decode dtype rule, keyed on the ACTIVATION dtypes the op needs and
// on NOTHING about how the weights are stored. `vt::GdnPackedDecode`'s own
// contract (src/vt/ops.cpp, "gdn_packed_decode: mixed_qkv/a/b/out must share
// FP16/BF16/F32 dtype") is a uniformity rule over the four activation tensors;
// the SSM state is explicitly INDEPENDENT of them ("state must use an
// independent FP16/BF16/F32 dtype"), matching upstream FLA, which requires only
// last-dim contiguity per tensor and casts each operand to f32 on load
// (fla/ops/fused_recurrent.py:256-336 @ 702f4814). The model's packed leg pins
// the shared dtype to BF16 — the dtype both the hand `GdnPackedDecodeKernel`
// and the vendored FLA AOT cubin (`gdn_decode_h48`) are exercised at.
//
// This deliberately replaces the previous `in_proj_qkv_fp8.Empty() &&
// in_proj_z_fp8.Empty()` term. That term was a PROXY: it excluded an fp8 GDN
// tower because the fp8 GEMM emits f32 while a/b/out are bf16, which is a
// statement about the ACTIVATION dtype, not about the weight format. Upstream
// conditions the packed decode on nothing about the quantization method.
struct GdnPackedDecodeDTypes {
  vt::DType mixed_qkv = vt::DType::kF32;  // post-conv activation (== mixed.dtype)
  vt::DType ba_out = vt::DType::kF32;     // MergedGdnBaOutputDType(packed)
  vt::DType core_out = vt::DType::kF32;   // dcore, GdnOutDType()
  vt::DType ssm_state = vt::DType::kF32;  // the recurrent cache, independent
};

bool GdnPackedDecodeDTypesCompatible(const GdnPackedDecodeDTypes& dtypes);

// VT_GDN_PACKED_DECODE_FP8_TOWER, DEFAULT OFF — the same-binary rollback of the
// rule above. OFF reproduces the legacy exclusion exactly (an fp8 GDN tower is
// never packed-decode eligible, whatever dtypes it produces), so the production
// default is byte-identical to the pre-#365 selection on every checkpoint. ON
// drops the weight-format term and lets `GdnPackedDecodeDTypesCompatible` decide
// alone. Parsed here rather than at the getenv so the CPU tier can pin it;
// '1'-leading is ON and everything else (including unset) is OFF, the house
// default-OFF convention (vt::cuda::GdnPackedRegTileFlagIsOn).
//
// NOTE this can only ever PERMIT, never DESELECT: the call-site clause is
// `!fp8_tower || allowed`, so on a checkpoint with no fp8 GDN shards it is
// unconditionally true and `PackedGdnDecodeEnvSelected`'s truth table above is
// unaffected.
bool PackedGdnDecodeFp8TowerFlagIsOn(const char* env_value);

// GDN-MOE-BF16-OUT (#1168). VT_GDN_OUT_BF16, DEFAULT ON — the whole decision
// behind `GdnOutDType()`, which resolves the dtype of the GDN recurrence output
// `dcore` [T,Hv,Dv], the `z` output gate, and the gated-RMSNorm weight that has
// to match them. Parsed here rather than at the getenv so the CPU tier can pin
// the truth table, the same way PackedGdnDecodeFp8TowerFlagIsOn is; the resolver
// caches its getenv in a function-local static, so a process observes exactly
// one value of it and the table cannot be asked of the resolver itself.
//
// Note the POLARITY differs from the flag above: this one is default ON and only
// a leading '0' turns it off, matching VT_GDN_IN_BF16 and the `out_bf16` term
// PackedGdnDecodeEnvSelected already mirrors. There is no model-shape argument.
// `GdnOutDType` used to take a `bool dense_model` and default to it, which made
// every MoE checkpoint f32 while vLLM keeps both tensors at the one model dtype
// it resolves for every layer (qwen_gdn_linear_attn.py:870-873, :843, :459-465 @
// 5559679).
bool GdnOutBf16FlagIsOn(const char* env_value);

// GDN-MOE-BF16-OUT (#1168), fresh-review repair. The RESOLVER the model actually
// calls, declared here so a gate can observe what it answers and not only what
// its parser answers. The two are separable, and severing them is silent: a
// `GdnOutDType()` hardwired to BF16 keeps every default-environment gate green
// while `VT_GDN_OUT_BF16=0` stops rolling anything back — and that variable is
// the denominator of this row's whole same-binary A/B, so a disconnected lever
// would invalidate the measurement that decides the row rather than merely lose
// coverage.
//
// It caches its getenv in a function-local static, so ONE process observes ONE
// value. The gate is therefore the shape tests/CMakeLists.txt already uses for a
// read-once lever (`_glue_fuse_off`, `test_dense_gateup_fused_marlin_off_*`): the
// SAME binary is registered a second time with `VT_GDN_OUT_BF16=0`, and the case
// asserts against the environment as it reads it directly.
vt::DType GdnOutDType();

// #2534. The qwen35 trunk's ONE resolved activation dtype, and the parser that
// feeds it, declared here for exactly the reason `GdnOutDType` is: a gate that
// can only reach the parser cannot see a resolver that has been severed from it,
// and this lever is the denominator of the Q4_K_M arm's same-binary A/B.
//
// `ActF32FlagIsOn` is opt-IN (only a leading '1' turns it on), the OPPOSITE
// polarity to `GdnOutBf16FlagIsOn`, because turning it on changes numerics on a
// shipped path and the default must be the behaviour every recorded measurement
// was taken under. `ActDType` answers BF16 for every device type when the flag
// is off, and F32 only for a CPU device type when it is on -- so a test that
// asserts the CUDA answer is unchanged is a real assertion and not a tautology.
// The getenv is cached in a function-local static, so one process observes one
// value; the second registration of the same binary under `VT_ACT_F32=1` is what
// exercises the other arm, as tests/CMakeLists.txt already does for
// `VT_GDN_OUT_BF16`.
// Values baked by the dense XPU attention graph. Device lengths/table entries
// remain staged inputs; route, layout and host partition policy require retirement.
struct XpuAttentionGraphPolicy {
  std::array<int64_t, 2> boundaries{};
  std::array<std::string, 9> settings{};
  std::vector<std::array<uint64_t, 13>> kv;
  bool causal = false;
  bool operator==(const XpuAttentionGraphPolicy&) const = default;
};
XpuAttentionGraphPolicy BuildXpuAttentionGraphPolicy(
    int64_t requests, int64_t query_rows, int32_t max_seq_len, bool causal,
    const std::vector<PagedKvCache>& kv);
int32_t XpuBatchedVerifyContextBound(const PagedKvCache& kv, bool fp16,
    int64_t query_heads, int64_t tokens, const v1::CommonAttentionMetadata& meta);

bool ActF32FlagIsOn(const char* env_value);
vt::DType ActDType(vt::DeviceType dev_type);

// W2 merged-qkvz dispatch. vLLM always issues one in_proj_qkvz GEMM
// (qwen_gdn_linear_attn.py:923-936 @ 702f4814); locally the single GEMM is
// selected only on CUDA with the packed 27B owner resident, the runtime
// toggles on (VT_GDN_MERGED_PROJ master, VT_GDN_MERGED_QKVZ leaf) and one
// uniform output dtype (mixed_qkv and z leave one GEMM, so GdnInDType must
// equal GdnOutDType — BF16/BF16 by default on every arm since #1168, so this
// term no longer excludes a MoE checkpoint; `has_packed_qkvz` still does).
// Every other combination
// issues the exact two split GEMMs sliced from the same resident owner.
struct GdnMergedQkvzEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool has_packed_qkvz = false;
  bool uniform_dtype = false;
};

bool ShouldUseMergedGdnQkvz(const GdnMergedQkvzEligibility& eligibility);

// PERF-27B-GDN-FP8-QKVZ — the FP8 (W8A8) leaf of the same merge. The BF16 leaf
// above owns a merged `in_proj_qkvz` BF16 parameter; a ModelOpt FP8 tower
// (`nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed`, and the 35B shares the
// tower) keeps the two shards NATIVE and so cannot use that owner. vLLM still
// runs ONE merged qkvz GEMM per GDN layer (qwen_gdn_linear_attn.py:923-936 @
// 702f4814; MergedColumnParallelLinear packs qkv+z along N), which is what this
// leaf mirrors: the two RAW fp8 [N,K] shards are N-concatenated ONCE into one
// resident operand and one fp8 GEMM replaces the two.
//
// The merge only reproduces the split arithmetic when the two shards agree on
// the ACTIVATION scale, because the merged GEMM quantizes the activation once:
// ModelOpt fp8 here is per-tensor, so `in_proj_qkv.input_scale` must equal
// `in_proj_z.input_scale` BITWISE. That is exactly the predicate
// `GdnFp8SharedInputScale` (the same one `Fp8SharedInputScale` already applies
// to this pair for the fused RmsNorm+quant), evaluated ONCE when the resident is
// built, never per step. The per-shard WEIGHT scales need no agreement: each
// shard's folded alpha is applied per output column (folded into the GEMM when
// both are equal, else through the resident alpha vector), exactly as the
// merged FP8 QKV path does.
struct GdnMergedFp8QkvzEligibility {
  bool runtime_enabled = false;  // VT_GDN_MERGED_QKVZ_FP8 (+ the BF16 leaf's
                                 // master VT_GDN_MERGED_PROJ / leaf
                                 // VT_GDN_MERGED_QKVZ rollbacks)
  bool fp8_platform = false;     // supports_fp8() + the fp8 GEMM registered
  bool has_fp8_shards = false;   // both in_proj_qkv_fp8 and in_proj_z_fp8 live
  bool shared_k = false;         // both shards read the same [M,K] activation
  bool shared_input_scale = false;  // bitwise-equal per-tensor input_scale
  bool shard_widths_match = false;  // shard N == conv_dim / value_dim
};

bool ShouldUseMergedGdnFp8Qkvz(const GdnMergedFp8QkvzEligibility& eligibility);

// Process-level ENV resolution of the FP8 leaf, mirroring
// PackedGdnDecodeEnvSelected: fields carry the raw getenv values (nullptr =
// unset) so the CPU tier can pin the truth table that `MergedGdnFp8QkvzEnabled`
// caches. VT_GDN_MERGED_QKVZ_FP8 is the leaf switch and defaults ON; the BF16
// leaf's master (VT_GDN_MERGED_PROJ) and leaf (VT_GDN_MERGED_QKVZ) rollbacks
// also disable it, so one switch retires the whole merged-input-projection
// topology.
struct GdnMergedFp8QkvzEnvConfig {
  const char* merged_proj = nullptr;      // VT_GDN_MERGED_PROJ (master)
  const char* merged_qkvz = nullptr;      // VT_GDN_MERGED_QKVZ (BF16 leaf)
  const char* merged_qkvz_fp8 = nullptr;  // VT_GDN_MERGED_QKVZ_FP8 (FP8 leaf)
};

bool MergedGdnFp8QkvzEnvSelected(const GdnMergedFp8QkvzEnvConfig& env);

// The load-time scale-compatibility predicate itself, exposed so the CPU tier
// can pin it: true (and *scale filled) only when BOTH fp8 GDN input shards are
// populated and carry the SAME per-tensor activation scale, compared with exact
// float equality. `Fp8SharedInputScale`'s linear-attention branch delegates
// here, so the fused-quant guard and the merge guard can never drift apart.
bool GdnFp8SharedInputScale(const GdnLayerWeights& gdn, float* scale);

// Model-selection instrumentation for the FP8 GDN input projections, mirroring
// vt::cuda::testing::GdnPackedDecodeDebugStats: counts the host GEMM dispatches
// (`merged` = one per GDN layer, `split` = two per GDN layer) so a real-model
// step can assert the structural 96 -> 48 collapse without a profiler. Graph
// replay has no host dispatch, so this is read on an eager step.
struct GdnFp8InProjDebugStats {
  uint64_t merged_launches = 0;
  uint64_t split_launches = 0;
  uint64_t Total() const { return merged_launches + split_launches; }
};

void ResetGdnFp8InProjDebugStats();
GdnFp8InProjDebugStats GetGdnFp8InProjDebugStats();
void DisableGdnFp8InProjDebugStats();

// GDN-MOE-BF16-OUT (#1168) — the dtypes the GDN block's two OUTPUT-side
// activations ACTUALLY carried on the last NON-MIXED-SPEC paged GDN layer this
// process ran.
//
// `GdnOutDType()` is a predicate, and a test that calls it proves only that the
// predicate answers. This records what `GdnBlockPaged` allocated and projected:
// `core_out` is `dcore`'s own `[T,Hv,Dv]` tensor dtype and `z_gate` is the dtype
// `ProjectGdnQkvz` actually emitted for the gate, both read off the tensors
// after they exist. So a gate can enter through `ModelRegistry::Forward` on a
// real MoE config and ask what the model ran, which is the question
// .agents/reachability.md says a unit test over the predicate cannot answer.
//
// `observed` is false until such a layer has run since the last reset. It is the
// term that separates "the forward produced f32" from "the forward never reached
// that path", which are the same silence to a dtype comparison.
//
// FOUR LIMITS, stated because the sentence above is narrower than it first
// reads and a reader who takes `observed` for a complete answer gets a wrong one
// (fresh-review finding).
//
//  1. It does NOT cover `GdnBlockPagedMixedSpec`. That is also a paged GDN
//     layer, at the same `outdt`, and it records nothing: `GdnBlockPaged`
//     returns into it before reaching the one recording site. On a step whose
//     batch mixes speculative and non-speculative requests `observed` therefore
//     stays false, or worse stays TRUE from an earlier step — which is the
//     ambiguity `observed` exists to remove, reappearing one level up. Read it
//     only from a gate that controls its own batch shape, as
//     test_qwen35_paged_forward does. The recording is not extended to the mixed
//     path because nothing gates that path's dtypes today, and a second call
//     site no test enters is the shape .agents/reachability.md warns about.
//  2. It is UNCONDITIONAL, unlike `GdnFp8InProjDebugStats` above, whose shape it
//     otherwise borrows. That sibling is default-off behind an enable flag and
//     counts only after `ResetGdnFp8InProjDebugStats`. This one stores three
//     atomics on every non-mixed-spec paged GDN layer of every forward of every
//     model, whether or not anybody reads it. Relaxed stores off the critical
//     path, so nothing observable changes — but do not read the resemblance as
//     "off by default", because it is not.
//  3. It records at CUDA-graph CAPTURE, not at replay. Replay does no host
//     dispatch, so under graphed decode the values are whatever the capturing
//     step wrote. Same caveat as the fp8 sibling's "read on an eager step".
//  4. It does not cover the NON-PAGED `GdnBlock` either (qwen3_5.cpp), which
//     allocates at the same `outdt` and records nothing. The word "paged" above
//     already excludes it, but a list a reader takes for complete has to say so:
//     a forward that runs the non-paged block leaves `observed` exactly as
//     limit 1 does, and this list is complete only with both uncovered sites on
//     it.
struct GdnOutActivationDTypes {
  bool observed = false;
  vt::DType core_out = vt::DType::kF32;  // dcore [T,Hv,Dv]
  vt::DType z_gate = vt::DType::kF32;    // the output gate handed to the norm
};

void ResetGdnOutActivationDTypes();
GdnOutActivationDTypes LastGdnOutActivationDTypes();

// Validate the exact prefix that will be uploaded. Negative rows are inert
// padding; every live slot must be unique and in range. This runs on host
// metadata before the device buffer is constructed, keeping CUDA capture free
// of a validation synchronization. Uniqueness is an O(n) seen-set pass (a live
// slot is drawn from a free-list of distinct slots by construction, so a single
// pass fails closed on any duplicate/out-of-range/negative slot); it is bounded
// by state_slots (== max_num_reqs). `force_full_uniqueness` (driven globally by
// VT_GDN_VALIDATE=1) additionally runs the exhaustive O(n^2) pairwise
// cross-verification — a redundant paranoid check, never needed for
// correctness.
void ValidateGdnStateIndices(const std::vector<int32_t>& indices,
                             int64_t required, int64_t state_slots,
                             bool force_full_uniqueness = false);

// Validate the complete eager/graph metadata contract before any state index
// is uploaded or consumed. Prefill-only vectors must be exact suffix/rebased
// views of the full non-spec vectors. Only CUDA-graph padding may contain the
// inert -1 state-slot sentinel.
void ValidateGdnAttentionMetadata(
    const v1::GDNAttentionMetadata& metadata, int64_t state_slots,
    bool allow_inert_padding);

// Row-copy state I/O cannot consume inert padded rows. Exact-size graphs are
// safe under either state-I/O mode; larger padded graph sizes require indexed
// state I/O, whose kernels define the -1 sentinel.
bool CanUseGdnDecodeGraphSize(int64_t real_batch, int64_t capture_batch,
                              bool indexed_state_io);

// Validate that every per-layer GDN cache exposes the same slot domain. Eager
// and graph paths both upload one shared state-index vector, so accepting a
// smaller later-layer cache would turn an index valid for layer zero into an
// out-of-bounds access. Returns the common slot count (zero for no caches).
int64_t ValidateGdnStateCacheLayout(
    const std::vector<GdnStateCache>& state_caches);

// Upstream MambaStateDtypeCalculator::_mamba_state_dtype. The temporal/SSM
// cache dtype is independent from the convolution cache dtype and accepts the
// exact torch dtype aliases used by raw HF configs.
vt::DType ResolveMambaSsmCacheDType(const HfConfig& config,
                                    vt::DType conv_dtype);

// Host preflight run at the entrance of every CUDA-graph Step, before padding,
// refresh, capture, or replay. It validates the real live request prefix
// against both state tensors in every GDN layer; the lower-level upload check
// remains defense in depth for eager/mixed paths.
void ValidateGdnDecodeGraphState(
    const v1::GDNAttentionMetadata& metadata,
    const std::vector<GdnStateCache>& state_caches, int64_t real_batch);

// ─── ENG-ASYNC-SCHED W4: device-resident input ids for the embed ─────────────
//
// `ModelForwardInput::device_token_ids` says "the input ids for this step are
// ALREADY on the device; the host vector is stale for decode rows". Only the
// embed at the very top of the forward cares, and it sits under five layers of
// entry points (eager / gathered / tap / multi-tap / decode-graph replay), each
// of which takes `const std::vector<int32_t>& token_ids` and passes it down.
//
// Rather than add a defaulted pointer parameter to every one of those and to the
// decode-graph class, the two Qwen3.5 registry forwards establish this SCOPED
// override for the duration of one forward and the embed consults it. The
// trade-off is deliberate and bounded: it is thread-local (a forward runs on one
// host thread), strictly RAII so it cannot leak past the call that set it, and
// set ONLY from the registry entry points that receive the ModelForwardInput —
// so its lifetime is exactly the forward's, not process state. It is null on
// every path except the discrete-CUDA async runner.
// The COUNT travels with the pointer so the embed can prove the buffer is the
// one meant for it. A forward can reach a second, unrelated embed over different
// ids (the multimodal generate helper embeds a prompt and then single tokens);
// an override that matched on "non-null" alone would silently feed that embed
// the wrong row count. Length disagreement means "not mine" and falls back to
// the host upload, which is always correct.
struct DeviceTokenIds {
  const int32_t* ids = nullptr;
  int64_t count = 0;
};

DeviceTokenIds& DeviceTokenIdsOverride();

struct DeviceTokenIdsScope {
  DeviceTokenIdsScope(const int32_t* ids, int64_t count)
      : prev(DeviceTokenIdsOverride()) {
    DeviceTokenIdsOverride() = DeviceTokenIds{ids, count};
  }
  ~DeviceTokenIdsScope() { DeviceTokenIdsOverride() = prev; }
  DeviceTokenIdsScope(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIdsScope& operator=(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIds prev;
};

// ─── THE CONSUMER SIDE, ONCE (#1305) ────────────────────────────────────────
//
// The scope above publishes. Reading it back is four lines of take-and-clear
// plus five of bounds-checked copy, and until #1305 every model that consumed it
// wrote its own pair: `qwen3.cpp`, `qwen3_5.cpp`, and then — in a row whose
// stated purpose is deleting hand-rolled copies — `qwen3_moe.cpp` and
// `deepseek_v2.cpp` as a third and fourth. A fresh review named that, and this
// is the answer: one declaration here beside the publisher, one definition in
// `qwen3_5.cpp` beside `DeviceTokenIdsOverride()`, four call sites.

// TAKE the published override and CLEAR it, so the FIRST embed in a forward is
// the one that gets it. A forward can reach a second, unrelated embed — the
// multimodal generate helper embeds a prompt and then single tokens — and
// consuming on first use means those cannot be handed a row count that was never
// meant for them. Returns a null `ids` when no override is live, which is every
// path except the asynchronous CUDA runner.
DeviceTokenIds TakeDeviceTokenIds();

// TAKE the override and SPLICE it over an embed's device identifier buffer.
// `dst` holds `dst_count` int32 identifiers that a host upload has already
// filled; the override replaces its first `ov.count` rows. That is right for the
// PADDED graph case, where only the real prefix is patched and the inert tail
// must keep the host vector's values, and it degenerates to "replace everything"
// on the eager path where `ov.count == dst_count`.
//
// The copy is enqueued on `queue`, so it is ordered AFTER the runner's combine
// that produced the source rather than racing it — which is the whole point, and
// the reason a host read of `ModelForwardInput::token_ids` cannot substitute.
//
// An override LONGER than `dst_count` can only mean the runner and the model
// disagree about this step's shape, so it throws with `what` naming the caller
// rather than embedding past the end. Returns true when an override was applied,
// false when none was live — in which case nothing is written and the caller is
// byte-identical to its pre-#1305 self.
bool ApplyDeviceTokenIds(vt::Backend& backend, vt::Queue& queue, void* dst,
                         int64_t dst_count, const char* what);

// THE SAME SPLICE, over identifiers the caller was handed EXPLICITLY rather than
// through the thread-local override.
//
// ENG-MM-EMBED-DEVICE-IDS (#2730). `ModelRegistry::EmbedMm` runs BEFORE the
// forward, from `GPUModelRunner::execute_model`, so no `DeviceTokenIdsScope` is
// live when a multimodal `embed_mm` hook reaches its identifier buffer -- the
// scope is documented as set ONLY from the registry entry points that receive a
// `ModelForwardInput`, and the runner's channel to that hook is
// `MmEmbedInputs`. The publisher differs; the splice must not.
//
// So the seam is EXTENDED rather than copied: this is the body, and the entry
// point above is one line that passes `TakeDeviceTokenIds()` into it. One bounds
// check, one `Copy`, one queue argument -- which is what makes a mutation of any
// of the three turn BOTH the text gates and the multimodal gate red instead of
// one of them.
bool ApplyDeviceTokenIds(vt::Backend& backend, vt::Queue& queue, void* dst,
                         int64_t dst_count, DeviceTokenIds ids,
                         const char* what);

// ─── ENG-EXPERT-STREAM (#912): the streamed-expert lane, seen from outside ───
//
// The lane lives in the anonymous namespace of qwen3_5.cpp because nothing
// outside the forward may construct it. Two things still have to reach it.
//
// A BENCHMARK has to be able to prove the lane stayed live. The row's published
// "streaming ON: no decode gain" number was measured on a cache that had
// switched itself off partway through the third token, and nothing in the run
// could have said so: the process printed one line at startup and none
// afterwards. `steps` and `exhausted` are the two fields that make that state
// visible — steps==0 means the step clock never advanced, exhausted>0 means
// slices were refused and served from the mapping instead.
//
// A GATE has to be able to prove decode still REACHES the lane. Counters that
// stay at zero when the production call site is deleted are what turns "the
// class works" into "the capability is wired", which is the distinction this
// row previously failed.
struct ExpertStreamStats {
  // A slot store exists. False when streaming was never requested, and also
  // when it was requested but no expert slice was ever taken — which is itself
  // the reachability failure worth catching.
  bool active = false;
  int64_t steps = 0;
  int64_t hits = 0;
  int64_t misses = 0;
  int64_t evictions = 0;
  int64_t fills = 0;
  int64_t bytes_filled = 0;
  // Slices the cache could not serve, which fell back to the mapping. Nonzero
  // means the budget is smaller than one step's working set, OR that the step
  // boundary is not being called at all.
  int64_t exhausted = 0;
  // Slices refused because a GATE asked for the unstreamed arm through
  // `ExpertStreamSetForceFallback`, which no production path calls. It is
  // separate from `exhausted` because that number is an operator-facing budget
  // diagnosis, and a test switch inflating it says "raise
  // VT_MOE_EXPERT_STREAM_SLOTS" about a budget that was never the reason.
  int64_t forced = 0;
  // madvise(MADV_WILLNEED) calls the kernel ACCEPTED. Zero while slices are
  // being filled from a mapping means the hint is being rejected, which is what
  // an unaligned address does silently.
  int64_t advised = 0;
};

ExpertStreamStats ExpertStreamSnapshot();

// Force every slice to take the cache-exhaustion fallback, i.e. the resident
// tower view. A real production state (a budget below one step's working set
// reaches it), exposed so one process can compare the streamed and unstreamed
// arms and prove they produce identical bytes.
void ExpertStreamSetForceFallback(bool on);

// ENG-EXPERT-STREAM-DEVICE W0c (issue #1124). The expert-slice seam and the
// weight-staging helper, callable from a gate.
//
// WHY THEY ARE EXPOSED, given that a unit test which constructs a type by hand
// proves nothing about reachability. Reachability of `KqExpertSlice` itself is
// NOT what these are for and is not what they claim: it is already gated through
// the production forward by `test_expert_stream_wiring`, which asserts the lane
// filled slots during `Qwen3_5Model::Forward`. What these two reach is the
// PLATFORM BRANCH inside that seam, which only a weight-staging,
// host-addressable device takes — a device this project owns exactly one of, and
// which no CPU test tier can register a real one of. Without them the branch
// that decides whether a 369.96 GiB checkpoint loads at all would be provable
// only on `dgx:gpu0`, which is the untestable-device shape this row has been
// bitten by before.
//
// `StageWeightForTest` is the same shape as `ExpertStreamSetForceFallback`
// above: no production path calls it, and it exists so a gate can observe a
// guarantee from inside a running process. Here the guarantee is the refusal
// that fires when a streamed tower reaches device staging — a tripwire whose
// whole value is that it never fires in production.
vt::Tensor ExpertSliceForTest(vt::Queue& q, const OwnedTensor& w, int64_t N,
                              int64_t K, int64_t row_off, int64_t expert);
vt::Tensor StageWeightForTest(vt::Queue& q, const OwnedTensor& w);

// End one decode step for the streamed-expert cache. The Qwen3.5 MoE forward
// runs this from its own layer driver; a SECOND full-attention MoE model
// (qwen3_moe.cpp) composes the same block from another translation unit and
// calls it from its layer driver for the same reason.
void EndExpertStreamStep();

// Print the streamed-expert statistics line NOW, once, whatever the run did.
//
// THIS FUNCTION HAS ZERO PRODUCTION CALLERS, and a grep-and-quote reader should
// get that before anything else: it exists for the gate. The only production
// path to the LINE is `~Qwen35ExpertStream`, which does NOT route through here —
// the store is a function-local static, so it is destroyed on the normal exit
// path and prints what the run ended up doing directly. The two share the
// once-flag rather than a call, so exactly one of them prints; an earlier
// revision headed this comment "~Qwen35ExpertStream IS THE ONLY PRODUCTION
// CALLER", which reads as a call that is not there (#1106).
//
// There is no second hook either: nothing registers an `atexit` handler when
// streaming is merely REQUESTED, and none was landed — that shape is recorded
// under the spec's `## Owed` with its reason. An even earlier revision of this
// comment claimed the hook existed, while the change that wrote it was fixing
// exactly this class of overclaim one file away (#1091). Two revisions, two
// overstatements of the same four lines, which is why they now name the
// mechanism rather than summarise it.
//
// So the guarantee carries the same two qualifiers `docs/USAGE.md` does, and it
// is one line per process under both: a store must have been BUILT, and the
// process must RUN its static destructors. No store means no line — and no
// `[expert-stream] ON ...` banner either, which is how the absent pair is read
// — and a crash, a signal or `_exit` prints nothing. Under those it holds even
// on a run with zero steps and with the periodic report silenced, which is what
// makes `steps == 0` readable at all.
//
// This entry exists because a static destructor fires after main returns and
// nothing inside the process can assert on it. Calling it TAKES the once-flag,
// so it suppresses the teardown line for the rest of the process and the caller
// becomes the one place the line appears. A second call prints nothing.
void ExpertStreamFlushStats();

// ONE decode step, as a scope, for a gate that needs to hold the boundary
// itself rather than reach it through a forward.
//
// It exists for one question: the step guard REFUSES TO NEST, and no legitimate
// call graph in the tree can ask it to. Every forward that takes expert slices
// is a complete forward that no other one contains, so the refusal was asserted
// in three places and pinned in none — deleting its `VT_CHECK` left both
// focused binaries fully green (#1091 review of #1100). A gate cannot reach it
// through production code, and a gate that re-implemented the flag would prove
// its own copy, so the guard's boundary is exposed here and this scope forwards
// to it.
//
// Constructing a second scope, or entering a forward while one is held, throws
// `std::runtime_error`. That is armed on the DEFAULT path and not only on the
// streaming lane, on purpose: see the note on `Qwen35ExpertStreamStep`.
class ExpertStreamStepScope {
 public:
  ExpertStreamStepScope();
  ~ExpertStreamStepScope();
  ExpertStreamStepScope(const ExpertStreamStepScope&) = delete;
  ExpertStreamStepScope& operator=(const ExpertStreamStepScope&) = delete;
};

}  // namespace vllm::detail
