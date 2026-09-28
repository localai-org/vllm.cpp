#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>  // ::sysconf(_SC_PAGESIZE) in the readahead hint below
#endif
// vllm.cpp original; see qwen3_5.h. Forward math mirrored 1:1 from the pinned
// upstream (qwen3_next.py::Qwen3NextDecoderLayer / Qwen3NextModel.forward,
// qwen_gdn_linear_attn.py, qwen3_next.py::Qwen3NextAttention /
// Qwen3NextSparseMoeBlock @ e24d1b24). References:
// .agents/specs/qwen36-forward-notes.md (assembly, §2 mRoPE->NeoX, §5 attention),
// .agents/specs/gdn-semantics.md (§1 layout, §6 g/beta prep, §7 recurrence),
// .agents/specs/moe-semantics.md (§1-§6 MoE block + activated-expert gather).
#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/host_expert_slot_store.h"
#include "vllm/model_executor/expert_streamer.h"
#include "vllm/model_executor/expert_slot_cache.h"
#include "vllm/model_executor/expert_stream_seam.h"  // MODEL-TEXT-GLM-MOE-DSA W3 (#2214): the lifted lane
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) slot-key + dispatch counters
#include "vllm/model_executor/models/qwen3_5.h"

#include "vllm/model_executor/models/decode_graph_sizes.h"
#include "vllm/model_executor/models/kv_cache_route.h"  // KV-FP8 W3 store/read route
#include "vllm/model_executor/models/dense_exl3_linear.h"  // MODEL-QWEN35-EXL3 (#2495): the EXL3 linear seam
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"  // MODEL-FP8-BLOCK-LINEAR (#1189 M4)
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/device_pool.h"  // DevicePool/Pool/AuxPool/ActivePool (shared)
#include "vt/tenstorrent/tenstorrent_device.h"  // DebugDeviceReadbackF32 (TT-only debug seam)

#include "vllm/model_executor/models/act_dump.h"  // ROCM-TIER-DIVERGENCE (#2590)
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/moe_placement_seam.h"
#include "vllm/model_executor/models/qwen3_5_gdn_block.h"  // RunGdnBlockPaged (W5b seam, #2110)
#include "vllm/model_executor/models/qwen3_5_moe_block.h"  // RunMoeBlock (SEAM GAP #2 exposure)
#include "vllm/model_executor/models/qwen3_5_mrope.h"  // BuildMropeCosSinHost (W5d-2 seam, #2249)
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"  // M3-b: Qwen3VLGetRopeIndex (MRoPE positions)
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) per-tensor residency seam

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>  // tt-27b-region-capture: std::accumulate over the region census
#include <string>
#include <string_view>  // tt-27b-region-capture: VLLM_CPP_REGION_CAPTURE parse
#include <unordered_map>
#include <utility>
#include <optional>
#include <vector>

#include "vllm/model_executor/models/dense_fp8_gemm.h"   // dense_fp8:: FP8 W8A8 seam (#940)
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // dense_nvfp4::MarlinDenseEnabled
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W4: the shared capture seam
#include "vt/persistent_step_input.h"  // ENG-CUDAGRAPH-BREAK W4: the persistent step inputs
#ifdef VT_BENCH_PROFILE_CONTROL
#include "vt/cuda/cuda_profiler_control.h"
#endif
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"
#include "vt/unaligned.h"
#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {

namespace {
// Telemetry: how many times the MIXED spec+non-spec GDN batch path
// (GdnBlockPagedMixedSpec) ran this process. Incremented per GDN layer per
// mixed step; a nonzero count proves the concurrency split/merge was actually
// exercised (a c>1 spec run that never mixed would leave it at 0). Used by the
// c>1 identity gate to distinguish "mixed batch handled" from "pure-spec only".
std::atomic<int64_t> g_mixed_spec_invocations{0};
}  // namespace

int64_t Qwen3_5MixedSpecInvocations() {
  return g_mixed_spec_invocations.load(std::memory_order_relaxed);
}
void ResetQwen3_5MixedSpecInvocations() {
  g_mixed_spec_invocations.store(0, std::memory_order_relaxed);
}

// GDN-MOE-BF16-OUT (#1168) Edit 2 dropped the `e.dense_model` term. It entered at
// f344decf4 ("dispatch exact packed decode") as one of that change's "real-model
// safety gates", was never revisited, and neither reference has an equivalent:
// VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE defaults True with no shape term
// (vllm/envs.py:124 @ 5559679) and SGLang keys supports_packed_decode on the
// platform alone (gdn_triton.py:43 @ f63458b5be). It became REDUNDANT once
// GdnOutDType stopped branching on model shape: `core_out` is `outdt`, and
// GdnPackedDecodeDTypesCompatible below already pins it to BF16, so an f32
// recurrence output is what deselects packed decode on either arm. Removing it
// BEFORE the dtype change would have removed a term that the dtype rule did not
// yet subsume, which is why the two edits are one change and in this order.
//
// When that removal landed it was not observable in production on its own
// (fresh-review finding): `has_packed_ba` needs `in_proj_ba`, and the owner was
// then written at exactly one site in the tree, the dense loader, so on a MoE
// checkpoint the eligibility was false before the shape term was ever read.
// GDN-MOE-PACKED-BA (#1169) closed that: the MoE safetensors loader now builds
// the same merged owner (`LoadGdn`, qwen3_5_weights.cpp), so `has_packed_ba` is
// true on the 35B and this predicate is what selects the packed leg there. The
// GGUF MoE loader still keeps the shards split (#1793, owed).
bool detail::ShouldUsePackedGdnDecode(
    const GdnPackedDecodeEligibility& e) {
  return e.runtime_enabled && e.cuda && e.has_packed_ba &&
         e.merged_ba_enabled && e.dtype_compatible && e.has_state_indices &&
         e.num_prefills == 0 && e.num_prefill_tokens == 0 &&
         e.num_spec_decodes == 0 && e.num_spec_decode_tokens == 0 &&
         e.num_decodes > 0 && e.num_decode_tokens == e.num_decodes &&
         e.num_actual_tokens == e.num_decode_tokens;
}

// PERF-27B-GDN-PACKED-REACHABLE (#365). Mirrors ProjectGdnQkvz's branch order:
// the merged BF16 owner wins, then the native-FP8 owner, then the split BF16
// owner. Order matters — a checkpoint that carries both owners projects through
// the BF16 one, so predicting the FP8 dtype there would be wrong.
vt::DType detail::GdnProjectedMixedQkvDType(const GdnMixedQkvDTypeInputs& in) {
  if (in.has_bf16_qkvz_owner) return in.in_dtype;
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): the trellis arm emits `indt` for
  // `mixed_qkv`, which is what `ProjectGdnQkvz`'s EXL3 rung allocates. Placed
  // in ProjectGdnQkvz's own branch order, ahead of the fp8 terms, so the two
  // orders read the same way down the page.
  if (in.has_exl3_qkv_owner) return in.in_dtype;
  // PERF-GDN-PACKED-BRIDGE (#365): only the MERGED fp8 arm can carry the
  // narrowed epilogue dtype; the split arm hardcodes F32 (ProjectGdnQkvz).
  if (in.has_fp8_qkv_owner)
    return in.fp8_merged_arm ? in.fp8_out_dtype : vt::DType::kF32;
  return in.in_dtype;
}

// PERF-GDN-PACKED-BRIDGE (#365). PERF-FP8-ALPHA-FOLD's three terms, verbatim,
// in the ONE place both the producer and the predictor read. See the header for
// why each term is required; the short version is that the toggle is the opt-in,
// `indt` keeps VT_GDN_IN_BF16's rollback honest on this arm too, and `outdt`
// keeps the chain dtype-uniform. `outdt` used to be described as what "confines
// the narrowing to the dense 27B"; GDN-MOE-BF16-OUT (#1168) removed the
// model-shape argument from `GdnOutDType`, so `outdt` is BF16 on BOTH arms at
// the default and confines nothing. The DEFAULT-OFF `VT_GDN_FP8_IN_BF16` toggle
// (`GdnFp8InBf16Enabled`, which requires a leading '1') is now the only term
// keeping this inert on the 35B.
vt::DType detail::GdnFp8MergedMixedQkvDType(bool fp8_in_bf16_enabled,
                                           vt::DType in_dtype,
                                           vt::DType out_dtype) {
  return (fp8_in_bf16_enabled && in_dtype == vt::DType::kBF16 &&
          out_dtype == vt::DType::kBF16)
             ? vt::DType::kBF16
             : vt::DType::kF32;
}

// The dtype rule vt::GdnPackedDecode actually imposes (ops.cpp gdn_packed_decode
// dtype checks), plus the model leg's BF16 pin. No term keys on weight storage.
bool detail::GdnPackedDecodeDTypesCompatible(const GdnPackedDecodeDTypes& d) {
  // Uniformity over the four activation tensors, pinned to BF16.
  if (d.mixed_qkv != vt::DType::kBF16) return false;
  if (d.ba_out != d.mixed_qkv || d.core_out != d.mixed_qkv) return false;
  // The recurrent state is independent: any of the dtypes the caches can hold.
  return d.ssm_state == vt::DType::kF32 || d.ssm_state == vt::DType::kF16 ||
         d.ssm_state == vt::DType::kBF16;
}

// Default OFF: ON only for a '1'-leading value (vt::cuda::GdnPackedRegTileFlagIsOn).
bool detail::PackedGdnDecodeFp8TowerFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

// GDN-MOE-BF16-OUT (#1168) — VT_GDN_OUT_BF16, default ON, parsed here so the CPU
// tier can pin the truth table that GdnOutDType() caches. There is no model-shape
// term: the environment is the whole decision, on the dense and the MoE arms
// alike, exactly as upstream resolves one model dtype for every layer.
bool detail::GdnOutBf16FlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

// ...and the RESOLVER that consumes it, here rather than in the anonymous
// namespace below so the CPU tier can call the thing the model calls. Pinning
// the parser alone does not pin that anything reads it: a `GdnOutDType()`
// hardwired to BF16 keeps every default-environment gate green, and the
// documented `VT_GDN_OUT_BF16=0` rollback then silently stops rolling back. See
// the header for why that matters more here than coverage — the variable is the
// denominator of this row's same-binary A/B. The full derivation of WHAT this
// dtype is stays at the `using` declaration below, next to the call sites.
vt::DType detail::GdnOutDType() {
  static const bool bf16 =
      detail::GdnOutBf16FlagIsOn(std::getenv("VT_GDN_OUT_BF16"));
  return bf16 ? vt::DType::kBF16 : vt::DType::kF32;
}

// --- The model's ONE resolved activation dtype (#2534) -----------------------
//
// vLLM resolves one model dtype and every layer inherits it; this is that value
// for the qwen35 trunk. It answers BF16 everywhere by default, which is what the
// device tiers measured and ship, and it answers F32 on the CPU tier -- and only
// there -- when `VT_ACT_F32=1`.
//
// WHY THE CPU TIER CAN HAVE A DIFFERENT ANSWER AT ALL. On a device tier the BF16
// activation is the format the GEMM consumes, so the narrow store is what the
// hardware wants. On the CPU tier nothing consumes it: `LoadActF32`
// (cpu_quant_gemm.cpp:41) widens every activation element straight back to f32
// before the row is quantized, and `WidenRowToF32` (cpu_ops.cpp:577) does the
// same inside RMSNorm. There the narrow store buys memory traffic and costs up
// to 2^-9 = 1.95e-3 relative per store -- twice per layer on the residual alone,
// which is added in f32, rounded, and then RE-READ (cpu_ops.cpp:576-583).
//
// WHY IT EXISTS. The Q4_K_M arm's declared oracle is llama.cpp b10451, whose CPU
// graph is f32 with an f16 KV cache (llama-context.cpp:3538-3539), so the token
// gate has been comparing two engines at different precisions. That difference
// is about two hundred times ggml's own arch-versus-generic k-quant spread,
// measured at the pin at 1.4e-05 to 9.8e-05 rms relative. This is the
// same-binary A/B that separates the two. See
// .agents/specs/qwen38-27b-q4km-token-exactness.md and issue #2534.
//
// Default OFF, so no shipped default and no recorded device measurement moves
// until that A/B exists. The polarity is opt-IN, unlike `VT_GDN_OUT_BF16`: a
// leading '1' is the only thing that turns it on.
bool detail::ActF32FlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

vt::DType detail::ActDType(vt::DeviceType dev_type) {
  static const bool f32 = detail::ActF32FlagIsOn(std::getenv("VT_ACT_F32"));
  if (!f32) return vt::DType::kBF16;
  // `VT_ACT_F32=1` is REFUSED, and refusing is the point. The f32 conversion is
  // INCOMPLETE, so honouring the flag does not produce an f32 engine: it
  // produces a broken one. Measured on thor:gpu0, rc job 18fc60f0 -- the default
  // arm stays 33/33 and 780/780 while the f32 arm fails 9 cases and 293
  // assertions, all of them CROSS-PATH consistency assertions (the indexed
  // versus fallback GDN pools, the tap versus plain routes, one
  // CHECK(2.69807 < 0.001)), plus a SIGABRT in the NVFP4 lm_head case. Paired
  // paths required to agree bitwise no longer do, because one side is retyped
  // and its reference is not.
  //
  // AGENTS.md: refuse an unimplemented arm with a message that NAMES the missing
  // part, and never leave the missing path to be discovered later. A SIGABRT in
  // the ninth test case is exactly "discovered later", so the flag aborts here,
  // once, by name, before any buffer is allocated.
  //
  // The resolver itself stays, and it is not dead: every qwen35 activation
  // buffer reads its answer on every forward, which is the one-model-dtype shape
  // vLLM resolves and every layer inherits. What is missing is only the f32
  // ANSWER, and this names what that would take.
  VT_CHECK(false,
           "VT_ACT_F32=1 is REFUSED: the qwen35 f32 activation conversion is "
           "incomplete and would run a numerically inconsistent engine. Missing: "
           "retype every PAIRED path together (the indexed and fallback GDN "
           "pools, the tap and plain routes) so the cross-path equality gates "
           "still hold, and resolve the two bf16 dtype CONTRACTS the trunk feeds "
           "-- vt::SigmoidGateBf16 (ops.cpp:5105) and the NVFP4 lm_head. Owed in "
           ".agents/specs/qwen38-27b-q4km-token-exactness.md; issue #2534. Unset "
           "VT_ACT_F32 to run the shipped bf16 path.");
  (void)dev_type;
  return vt::DType::kBF16;  // unreachable; VT_CHECK(false) always throws
}

bool detail::ShouldUseMergedGdnQkvz(const GdnMergedQkvzEligibility& e) {
  return e.runtime_enabled && e.cuda && e.has_packed_qkvz && e.uniform_dtype;
}

// PERF-27B-GDN-FP8-QKVZ. Every term is required: dropping any one of them must
// leave the exact two legacy fp8 GEMMs. `shared_input_scale` is the load-time
// scale-compatibility guard (the merged GEMM quantizes the activation ONCE, so
// the two shards must agree bitwise on the per-tensor activation scale) and is
// the term that keeps a checkpoint whose scales differ on the split path.
bool detail::ShouldUseMergedGdnFp8Qkvz(const GdnMergedFp8QkvzEligibility& e) {
  return e.runtime_enabled && e.fp8_platform && e.has_fp8_shards &&
         e.shared_k && e.shared_input_scale && e.shard_widths_match;
}

// True (and *scale filled) only when both fp8 GDN input shards are populated and
// carry the SAME per-tensor activation scale, by exact float equality. This is
// the single definition; `Fp8SharedInputScale`'s linear-attention branch calls
// it, so the fused RmsNorm+quant guard and the merge guard cannot drift.
bool detail::GdnFp8SharedInputScale(const GdnLayerWeights& gdn, float* scale) {
  if (gdn.in_proj_qkv_fp8.Empty() || gdn.in_proj_z_fp8.Empty()) return false;
  if (gdn.in_proj_qkv_fp8.input_scale != gdn.in_proj_z_fp8.input_scale)
    return false;
  if (scale != nullptr) *scale = gdn.in_proj_qkv_fp8.input_scale;
  return true;
}

bool detail::MergedGdnFp8QkvzEnvSelected(const GdnMergedFp8QkvzEnvConfig& env) {
  if (env.merged_proj != nullptr && env.merged_proj[0] == '0') return false;
  if (env.merged_qkvz != nullptr && env.merged_qkvz[0] == '0') return false;
  return env.merged_qkvz_fp8 == nullptr || env.merged_qkvz_fp8[0] != '0';
}

namespace {
std::atomic<bool> g_gdn_fp8_inproj_debug_enabled{false};
std::atomic<uint64_t> g_gdn_fp8_inproj_merged{0};
std::atomic<uint64_t> g_gdn_fp8_inproj_split{0};
}  // namespace

void detail::ResetGdnFp8InProjDebugStats() {
  g_gdn_fp8_inproj_merged.store(0, std::memory_order_relaxed);
  g_gdn_fp8_inproj_split.store(0, std::memory_order_relaxed);
  g_gdn_fp8_inproj_debug_enabled.store(true, std::memory_order_release);
}

detail::GdnFp8InProjDebugStats detail::GetGdnFp8InProjDebugStats() {
  GdnFp8InProjDebugStats out;
  out.merged_launches = g_gdn_fp8_inproj_merged.load(std::memory_order_relaxed);
  out.split_launches = g_gdn_fp8_inproj_split.load(std::memory_order_relaxed);
  return out;
}

void detail::DisableGdnFp8InProjDebugStats() {
  g_gdn_fp8_inproj_debug_enabled.store(false, std::memory_order_release);
}

namespace {
// GDN-MOE-BF16-OUT (#1168). What the last NON-MIXED-SPEC paged GDN layer actually
// allocated and projected, recorded off the tensors themselves rather than off
// the predicate. `GdnBlockPagedMixedSpec` is a paged GDN layer too and records
// nothing, the stores are unconditional (the fp8 sibling above is default-off),
// and they happen at graph CAPTURE and not at replay. The header states all
// three; none of them is what the shape of this code suggests.
std::atomic<bool> g_gdn_out_dtypes_observed{false};
std::atomic<int> g_gdn_out_core_dtype{static_cast<int>(vt::DType::kF32)};
std::atomic<int> g_gdn_out_z_dtype{static_cast<int>(vt::DType::kF32)};

void RecordGdnOutActivationDTypes(vt::DType core_out, vt::DType z_gate) {
  g_gdn_out_core_dtype.store(static_cast<int>(core_out), std::memory_order_relaxed);
  g_gdn_out_z_dtype.store(static_cast<int>(z_gate), std::memory_order_relaxed);
  g_gdn_out_dtypes_observed.store(true, std::memory_order_release);
}
}  // namespace

void detail::ResetGdnOutActivationDTypes() {
  g_gdn_out_dtypes_observed.store(false, std::memory_order_release);
  g_gdn_out_core_dtype.store(static_cast<int>(vt::DType::kF32), std::memory_order_relaxed);
  g_gdn_out_z_dtype.store(static_cast<int>(vt::DType::kF32), std::memory_order_relaxed);
}

detail::GdnOutActivationDTypes detail::LastGdnOutActivationDTypes() {
  GdnOutActivationDTypes out;
  out.observed = g_gdn_out_dtypes_observed.load(std::memory_order_acquire);
  out.core_out =
      static_cast<vt::DType>(g_gdn_out_core_dtype.load(std::memory_order_relaxed));
  out.z_gate =
      static_cast<vt::DType>(g_gdn_out_z_dtype.load(std::memory_order_relaxed));
  return out;
}

bool detail::PackedGdnDecodeEnvSelected(const GdnPackedDecodeEnvConfig& env) {
  // Mirror PackedGdnDecodeRuntimeEnabled: enabled unless first char is '0'.
  const bool runtime_enabled =
      env.packed_decode == nullptr || env.packed_decode[0] != '0';
  // Mirror MergedGdnBaEnabled's env core: master '0' wins; leaf default-on.
  const bool merged_ba =
      !(env.merged_proj != nullptr && env.merged_proj[0] == '0') &&
      (env.merged_ba == nullptr || env.merged_ba[0] != '0');
  // Mirror the dtype_compatible expression on the real 27B dense gate:
  // GdnInDType (default BF16), GdnOutDType (default BF16 on every arm since
  // #1168; override '0' -> F32), MergedGdnBaOutputDType(packed) (default BF16 under packed;
  // override '0' -> F32). The SSM cache dtype term is always a float dtype.
  const bool in_bf16 = env.in_bf16 == nullptr || env.in_bf16[0] != '0';
  const bool out_bf16 = env.out_bf16 == nullptr || env.out_bf16[0] != '0';
  const bool ba_out_bf16 =
      env.ba_out_bf16 == nullptr || env.ba_out_bf16[0] != '0';
  return runtime_enabled && merged_ba && in_bf16 && out_bf16 && ba_out_bf16;
}

// VT_GDN_VALIDATE=1 (read ONCE) forces the exhaustive O(n^2) pairwise
// duplicate cross-verification on top of the default O(n) seen-set pass. The
// default already fails closed on duplicate/out-of-range/negative live slots;
// this env is a redundant paranoid verifier for debugging, never required for
// correctness.
static bool GdnForceFullValidationEnv() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_VALIDATE");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

void detail::ValidateGdnStateIndices(const std::vector<int32_t>& indices,
                                     int64_t required,
                                     int64_t state_slots,
                                     bool force_full_uniqueness) {
  VT_CHECK(required >= 0 &&
               required <= static_cast<int64_t>(indices.size()),
           "qwen3_5: GDN state index metadata is too short");
  VT_CHECK(state_slots >= 0,
           "qwen3_5: GDN state cache has invalid slot count");
  // O(n) uniqueness: a live slot is in [0, state_slots) and drawn from a
  // free-list of distinct slots by construction, so a single seen-set pass
  // fails closed on any duplicate/out-of-range/negative slot without the former
  // O(n^2) inner scan. `seen` is bounded by state_slots (== max_num_reqs,
  // small) and lives only for this call; -1 is the inert padding sentinel.
  std::vector<uint8_t> seen(static_cast<size_t>(state_slots), 0);
  for (int64_t i = 0; i < required; ++i) {
    const int32_t slot = indices[static_cast<size_t>(i)];
    if (slot < 0) {
      VT_CHECK(slot == -1,
               "qwen3_5: invalid negative GDN state index");
      continue;
    }
    VT_CHECK(slot < state_slots,
             "qwen3_5: GDN state index out of range");
    VT_CHECK(seen[static_cast<size_t>(slot)] == 0,
             "qwen3_5: duplicate live GDN state index");
    seen[static_cast<size_t>(slot)] = 1;
  }
  // Paranoid exhaustive re-verification (VT_GDN_VALIDATE=1 or an explicit
  // caller request). Identical verdict to the seen-set pass above.
  if (force_full_uniqueness || GdnForceFullValidationEnv()) {
    for (int64_t i = 0; i < required; ++i) {
      const int32_t slot = indices[static_cast<size_t>(i)];
      if (slot < 0) continue;
      for (int64_t j = 0; j < i; ++j) {
        VT_CHECK(indices[static_cast<size_t>(j)] != slot,
                 "qwen3_5: duplicate live GDN state index");
      }
    }
  }
}

void detail::ValidateGdnAttentionMetadata(
    const v1::GDNAttentionMetadata& metadata, int64_t state_slots,
    bool allow_inert_padding) {
  const int64_t nd = metadata.num_decodes;
  const int64_t np = metadata.num_prefills;
  const int64_t nd_tok = metadata.num_decode_tokens;
  const int64_t np_tok = metadata.num_prefill_tokens;
  // Spec-decode segmentation (SPEC-MTP I5a). ns == 0 on every production step —
  // the default path validated below is byte-identical to pre-I5a.
  const int64_t ns = metadata.num_spec_decodes;
  const int64_t ns_tok = metadata.num_spec_decode_tokens;
  const int64_t nreq = nd + np;

  VT_CHECK(nd >= 0 && np >= 0 && nd_tok >= 0 && np_tok >= 0 &&
               ns >= 0 && ns_tok >= 0 && metadata.num_actual_tokens >= 0,
           "qwen3_5: negative GDN metadata count");
  VT_CHECK(nd_tok == nd,
           "qwen3_5: non-spec decode requires one token per request");
  VT_CHECK(metadata.num_actual_tokens == nd_tok + np_tok + ns_tok,
           "qwen3_5: GDN decode+prefill+spec tokens must equal actual tokens");

  // ── Spec metadata validation (SPEC-MTP I5a; mirrors the gdn_attn.py build()
  // spec contract, src/vllm/v1/attention/backends/gdn_attn.cpp:181-276). Runs
  // only when the step actually carries drafts; on the default path (ns == 0)
  // none of it executes. ──
  if (ns > 0) {
    const int64_t num_cols = metadata.spec_state_indices_num_cols;
    VT_CHECK(num_cols >= 1,
             "qwen3_5: spec GDN state slot column count must be >= 1");
    VT_CHECK(metadata.spec_state_indices_tensor.has_value() &&
                 metadata.spec_query_start_loc.has_value() &&
                 metadata.spec_sequence_masks.has_value() &&
                 metadata.spec_token_indx.has_value() &&
                 metadata.num_accepted_tokens.has_value(),
             "qwen3_5: incomplete spec GDN metadata");
    const std::vector<int32_t>& ssi = *metadata.spec_state_indices_tensor;
    const std::vector<int32_t>& sqsl = *metadata.spec_query_start_loc;
    const std::vector<int32_t>& nat = *metadata.num_accepted_tokens;
    const std::vector<int32_t>& stx = *metadata.spec_token_indx;
    VT_CHECK(static_cast<int64_t>(ssi.size()) == ns * num_cols,
             "qwen3_5: spec_state_indices_tensor must be [num_spec_decodes, "
             "num_spec+1]");
    VT_CHECK(static_cast<int64_t>(sqsl.size()) == ns + 1,
             "qwen3_5: spec_query_start_loc must be [num_spec_decodes + 1]");
    VT_CHECK(static_cast<int64_t>(nat.size()) == ns,
             "qwen3_5: num_accepted_tokens must be [num_spec_decodes]");
    VT_CHECK(static_cast<int64_t>(stx.size()) == ns_tok,
             "qwen3_5: spec_token_indx must be [num_spec_decode_tokens]");
    VT_CHECK(sqsl.front() == 0 && sqsl.back() == ns_tok,
             "qwen3_5: spec query offsets must span the spec tokens");
    for (int64_t i = 0; i < ns; ++i) {
      VT_CHECK(sqsl[static_cast<size_t>(i + 1)] > sqsl[static_cast<size_t>(i)],
               "qwen3_5: spec query offsets must be strictly increasing");
      const int32_t acc = nat[static_cast<size_t>(i)];
      VT_CHECK(acc >= 1 && acc <= num_cols,
               "qwen3_5: num_accepted_tokens must be in [1, num_spec + 1]");
      // The INITIAL-state slot (column num_accepted-1) must be a live slot in
      // range; per-timestep snapshot slots may be null (< 0) only under padding.
      const int32_t init_slot =
          ssi[static_cast<size_t>(i * num_cols + (acc - 1))];
      VT_CHECK(init_slot >= 0 || allow_inert_padding,
               "qwen3_5: spec initial GDN state slot must be live");
      VT_CHECK(init_slot < state_slots,
               "qwen3_5: spec GDN state slot out of range");
    }
    for (int32_t slot : ssi)
      VT_CHECK(slot < state_slots,
               "qwen3_5: spec GDN state slot out of range");
    for (int32_t t : stx)
      VT_CHECK(t >= 0 && t < metadata.num_actual_tokens,
               "qwen3_5: spec_token_indx entry out of range");
    if (metadata.non_spec_token_indx.has_value()) {
      VT_CHECK(static_cast<int64_t>(metadata.non_spec_token_indx->size()) ==
                   nd_tok + np_tok,
               "qwen3_5: non_spec_token_indx must cover the non-spec tokens");
    }
  }

  if (nreq == 0 && ns == 0) {
    VT_CHECK(metadata.num_actual_tokens == 0,
             "qwen3_5: GDN tokens require state metadata");
    return;
  }
  // Pure spec batch (no non-spec rows): the non-spec segmentation is nullopt by
  // construction (gdn_attn.cpp:202-217), so skip the non-spec validation below.
  if (nreq == 0) return;

  VT_CHECK(metadata.non_spec_state_indices_tensor.has_value(),
           "qwen3_5: missing non-spec GDN state indices");
  const std::vector<int32_t>& indices =
      *metadata.non_spec_state_indices_tensor;
  VT_CHECK(static_cast<int64_t>(indices.size()) == nreq,
           "qwen3_5: non-spec GDN state index count must equal request count");
  detail::ValidateGdnStateIndices(indices, nreq, state_slots);
  if (!allow_inert_padding) {
    for (int32_t slot : indices) {
      VT_CHECK(slot >= 0,
               "qwen3_5: live GDN state index must be non-negative");
    }
  }

  if (np == 0) return;

  VT_CHECK(metadata.non_spec_query_start_loc.has_value() &&
               metadata.has_initial_state.has_value() &&
               metadata.prefill_state_indices.has_value() &&
               metadata.prefill_query_start_loc.has_value() &&
               metadata.prefill_has_initial_state.has_value(),
           "qwen3_5: incomplete GDN prefill metadata");
  const std::vector<int32_t>& full_qsl =
      *metadata.non_spec_query_start_loc;
  const std::vector<uint8_t>& full_initial = *metadata.has_initial_state;
  const std::vector<int32_t>& prefill_indices =
      *metadata.prefill_state_indices;
  const std::vector<int32_t>& prefill_qsl =
      *metadata.prefill_query_start_loc;
  const std::vector<uint8_t>& prefill_initial =
      *metadata.prefill_has_initial_state;

  VT_CHECK(static_cast<int64_t>(full_qsl.size()) == nreq + 1 &&
               static_cast<int64_t>(full_initial.size()) == nreq,
           "qwen3_5: non-spec GDN prefill metadata has invalid shape");
  VT_CHECK(static_cast<int64_t>(prefill_indices.size()) == np &&
               static_cast<int64_t>(prefill_qsl.size()) == np + 1 &&
               static_cast<int64_t>(prefill_initial.size()) == np,
           "qwen3_5: prefill-only GDN metadata has invalid shape");
  // The non-spec cu_seqlens spans the NON-SPEC tokens only. On the default path
  // (ns == 0) that equals num_actual_tokens; in a MIXED spec batch the spec
  // tokens (ns_tok) are carried by the separate spec segmentation, so the
  // non-spec span is nd_tok + np_tok < num_actual_tokens (gdn_attn.cpp:240-253).
  VT_CHECK(full_qsl.front() == 0 && full_qsl.back() == nd_tok + np_tok,
           "qwen3_5: non-spec GDN query offsets must span the non-spec tokens");
  for (int64_t i = 0; i < nreq; ++i) {
    VT_CHECK(full_qsl[static_cast<size_t>(i + 1)] >
                 full_qsl[static_cast<size_t>(i)],
             "qwen3_5: non-spec GDN query offsets must be strictly increasing");
  }
  VT_CHECK(full_qsl[static_cast<size_t>(nd)] == nd_tok,
           "qwen3_5: non-spec GDN decode prefix does not match token count");

  for (int64_t i = 0; i < np; ++i) {
    const size_t prefill_row = static_cast<size_t>(i);
    const size_t full_row = static_cast<size_t>(nd + i);
    VT_CHECK(prefill_indices[prefill_row] == indices[full_row],
             "qwen3_5: prefill state indices must match non-spec suffix");
    VT_CHECK(prefill_indices[prefill_row] >= 0,
             "qwen3_5: prefill GDN state index must be non-negative");
    VT_CHECK(prefill_qsl[prefill_row] ==
                 full_qsl[full_row] - nd_tok,
             "qwen3_5: prefill query offsets must match rebased non-spec suffix");
    VT_CHECK(prefill_initial[prefill_row] == full_initial[full_row],
             "qwen3_5: prefill initial-state mask must match non-spec suffix");
  }
  VT_CHECK(prefill_qsl.back() == np_tok,
           "qwen3_5: prefill query offsets must span prefill tokens");
  VT_CHECK(metadata.batch_ptr.has_value() &&
               metadata.token_chunk_offset_ptr.has_value(),
           "qwen3_5: missing exact causal-conv chunk metadata");
  const v1::CausalConv1dMetadata expected_conv =
      v1::ComputeCausalConv1dMetadata(full_qsl);
  VT_CHECK(*metadata.batch_ptr == expected_conv.batch_ptr &&
               *metadata.token_chunk_offset_ptr ==
                   expected_conv.token_chunk_offset_ptr,
           "qwen3_5: causal-conv chunk metadata does not exactly cover query offsets");
}

bool detail::CanUseGdnDecodeGraphSize(int64_t real_batch,
                                      int64_t capture_batch,
                                      bool indexed_state_io) {
  return real_batch > 0 && capture_batch >= real_batch &&
         (capture_batch == real_batch || indexed_state_io);
}

int64_t detail::ValidateGdnStateCacheLayout(
    const std::vector<GdnStateCache>& state_caches) {
  if (state_caches.empty()) return 0;
  int64_t state_slots = -1;
  for (const GdnStateCache& cache : state_caches) {
    VT_CHECK(cache.ssm_state.rank == 4 && cache.conv_state.rank == 3,
             "qwen3_5: GDN SSM/conv state ranks must be 4/3");
    VT_CHECK(cache.ssm_state.shape[0] == cache.conv_state.shape[0],
             "qwen3_5: GDN conv/SSM state slot counts must match");
    if (state_slots < 0) {
      state_slots = cache.ssm_state.shape[0];
    } else {
      VT_CHECK(cache.ssm_state.shape[0] == state_slots,
               "qwen3_5: all GDN layers must use the same state slot count");
    }
  }
  return state_slots;
}

// ENG-ASYNC-SCHED W4 (see qwen3_5_internal.h for why this is a scoped override
// rather than a parameter on five entry points). Thread-local: one host thread
// drives a forward, and a serving process may drive independent engines from
// different threads, so a process-global would let one engine's device ids leak
// into another's embed.
detail::DeviceTokenIds& detail::DeviceTokenIdsOverride() {
  thread_local DeviceTokenIds ids;
  return ids;
}

// #1305 — THE CONSUMER SIDE, once, beside the publisher it reads. See
// `qwen3_5_internal.h` for what each argument means and why the copy goes on the
// queue. Four models call these: this one, `qwen3.cpp`, `qwen3_moe.cpp` and
// `deepseek_v2.cpp`, each of which used to carry its own copy of both bodies.
detail::DeviceTokenIds detail::TakeDeviceTokenIds() {
  const DeviceTokenIds ov = DeviceTokenIdsOverride();
  if (ov.ids != nullptr) DeviceTokenIdsOverride() = DeviceTokenIds{};
  return ov;
}

bool detail::ApplyDeviceTokenIds(vt::Backend& backend, vt::Queue& queue,
                                 void* dst, int64_t dst_count, const char* what) {
  return ApplyDeviceTokenIds(backend, queue, dst, dst_count,
                             TakeDeviceTokenIds(), what);
}

// #2730 — the body, once, for both publishers. The override arm above takes and
// clears the thread-local; the multimodal arm is handed
// `MmEmbedInputs::device_token_ids` directly, because `ModelRegistry::EmbedMm`
// runs before any forward has established a scope. Everything after the source
// differs by nothing at all, and that is the point: the bounds check that refuses
// a shape disagreement, and the `Copy` that goes ON `queue` so it is ordered
// AFTER the runner's combine rather than racing it, are one expression.
bool detail::ApplyDeviceTokenIds(vt::Backend& backend, vt::Queue& queue,
                                 void* dst, int64_t dst_count,
                                 DeviceTokenIds ov, const char* what) {
  if (ov.ids == nullptr) return false;
  VT_CHECK(ov.count <= dst_count,
           std::string(what) + ": device input ids longer than the embed input");
  backend.Copy(queue, dst, ov.ids,
               static_cast<size_t>(ov.count) * sizeof(int32_t));
  return true;
}

vt::DType detail::ResolveMambaSsmCacheDType(const HfConfig& config,
                                            vt::DType conv_dtype) {
  const std::string& dtype = config.mamba_ssm_dtype;
  if (dtype.empty() || dtype == "auto") return conv_dtype;
  if (dtype == "float32" || dtype == "float") return vt::DType::kF32;
  if (dtype == "float16" || dtype == "half") return vt::DType::kF16;
  if (dtype == "bfloat16") return vt::DType::kBF16;
  throw std::runtime_error(
      "qwen3_5: unsupported mamba_ssm_dtype '" + dtype +
      "' (expected float16/half, bfloat16, float32/float, or auto)");
}

void detail::ValidateGdnDecodeGraphState(
    const v1::GDNAttentionMetadata& metadata,
    const std::vector<GdnStateCache>& state_caches, int64_t real_batch) {
  VT_CHECK(real_batch > 0,
           "qwen3_5 decode graph: real batch must be positive");
  // SPEC-DSPARK W8 (#442): a UNIFORM SPEC batch is capturable too. vLLM's
  // captured decode length is `1 + num_speculative_tokens`
  // (cudagraph_dispatcher.py:37), so its T=1+k verify is graphed by construction
  // while ours was refused right here. The assertion is RE-EXPRESSED for that
  // shape, never dropped: a spec batch is still required to be EXACT and pure
  // (no prefill mixed in), with every token a spec token.
  const bool spec_batch = metadata.num_spec_decodes > 0;
  if (spec_batch) {
    VT_CHECK(metadata.num_prefills == 0 && metadata.num_prefill_tokens == 0 &&
                 metadata.num_decodes == 0 && metadata.num_decode_tokens == 0 &&
                 metadata.num_spec_decode_tokens == real_batch &&
                 metadata.num_actual_tokens == real_batch,
             "qwen3_5 decode graph: a spec batch must be an exact PURE spec "
             "decode (every token a spec token, no prefill)");
    // Uniformity is what makes the shape a graph key: every request contributes
    // the same 1+k query span, so real_batch == num_spec_decodes * (1+k).
    VT_CHECK(metadata.num_spec_decodes > 0 &&
                 real_batch % metadata.num_spec_decodes == 0,
             "qwen3_5 decode graph: spec batch must be uniform across requests");
    VT_CHECK(metadata.spec_state_indices_tensor.has_value(),
             "qwen3_5 decode graph: missing GDN spec state indices");
    VT_CHECK(!state_caches.empty(),
             "qwen3_5 decode graph: missing GDN state caches");
    return;
  }
  VT_CHECK(metadata.num_prefills == 0 && metadata.num_prefill_tokens == 0 &&
               metadata.num_spec_decodes == 0 &&
               metadata.num_spec_decode_tokens == 0 &&
               metadata.num_decodes == real_batch &&
               metadata.num_decode_tokens == real_batch &&
               metadata.num_actual_tokens == real_batch,
           "qwen3_5 decode graph: metadata must describe exact pure non-spec "
           "decode");
  VT_CHECK(metadata.non_spec_state_indices_tensor.has_value(),
           "qwen3_5 decode graph: missing GDN state indices");
  const std::vector<int32_t>& indices =
      *metadata.non_spec_state_indices_tensor;
  VT_CHECK(static_cast<int64_t>(indices.size()) == real_batch,
           "qwen3_5 decode graph: state index count must equal the real decode "
           "batch");
  VT_CHECK(!state_caches.empty(),
           "qwen3_5 decode graph: missing GDN state caches");

  const int64_t state_slots =
      detail::ValidateGdnStateCacheLayout(state_caches);
  for (int64_t i = 0; i < real_batch; ++i) {
    VT_CHECK(indices[static_cast<size_t>(i)] >= 0,
             "qwen3_5 decode graph: live GDN state index must be non-negative");
  }
  ValidateGdnStateIndices(indices, real_batch, state_slots);
}

DenseGateUpGlobals MergeDenseGateUpGlobals(const Nvfp4Weight& gate,
                                           const Nvfp4Weight& up) {
  VT_CHECK(gate.weight_global_scale_inv > 0.0F &&
               up.weight_global_scale_inv > 0.0F,
           "qwen3_5 dense merged gate_up: missing CT weight divisor");
  VT_CHECK(gate.input_global_scale_inv > 0.0F &&
               up.input_global_scale_inv > 0.0F,
           "qwen3_5 dense merged gate_up: missing CT input divisor");
  DenseGateUpGlobals globals;
  globals.input_global_scale_inv =
      std::max(gate.input_global_scale_inv, up.input_global_scale_inv);
  const float weight_global_scale_inv =
      std::max(gate.weight_global_scale_inv, up.weight_global_scale_inv);
  // Preserve vLLM/PyTorch's operation order: reciprocal each selected maximum,
  // then multiply. Do not derive the maximum divisor back from scale2.
  const float input_global_scale = 1.0F / globals.input_global_scale_inv;
  globals.weight_global_scale = 1.0F / weight_global_scale_inv;
  globals.alpha = input_global_scale * globals.weight_global_scale;
  return globals;
}

FullAttnQkvGlobals MergeFullAttnQkvGlobals(const Nvfp4Weight& q,
                                           const Nvfp4Weight& k,
                                           const Nvfp4Weight& v) {
  VT_CHECK(q.weight_global_scale_inv > 0.0F &&
               k.weight_global_scale_inv > 0.0F &&
               v.weight_global_scale_inv > 0.0F,
           "qwen3_5 packed QKV: missing CT weight divisor");
  VT_CHECK(q.input_global_scale_inv > 0.0F &&
               k.input_global_scale_inv > 0.0F &&
               v.input_global_scale_inv > 0.0F,
           "qwen3_5 packed QKV: missing CT input divisor");
  FullAttnQkvGlobals globals;
  globals.input_global_scale_inv =
      std::max({q.input_global_scale_inv, k.input_global_scale_inv,
                v.input_global_scale_inv});
  const float weight_global_scale_inv =
      std::max({q.weight_global_scale_inv, k.weight_global_scale_inv,
                v.weight_global_scale_inv});
  const float input_global_scale = 1.0F / globals.input_global_scale_inv;
  globals.weight_global_scale = 1.0F / weight_global_scale_inv;
  globals.alpha = input_global_scale * globals.weight_global_scale;
  return globals;
}

namespace {

using vt::Backend;
using vt::Device;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using v1::GDNAttentionMetadata;

// Backend + queue bundle threaded through every helper.
// ENG-QWEN35-SHARED-GLUE: `Dev`, `DBuf`, `MakeTensor`, `Reshape` and the
// device-pool policy resolver were PRIVATE COPIES here and are now the shared
// ones from `dense_device_glue.h`. This file kept its own set, which is the
// off-framework divergence its `ResidentWeight` comment records — a repair
// landed in the shared glue for 25 model files and never reached this one.
// Keeping a private `Dev`/`DBuf` also gave them INTERNAL LINKAGE, so nothing
// this file returned could be declared in a header, which is what forced the
// MoE placement seam to carry two spellings.
//
// The two definitions were compared line by line before this change. They
// differed in exactly one behaviour, and the shared one is the safer: it
// guards `bytes_ > 0` before a host copy, where the private one issued a
// zero-byte `Copy`. Everything else was comments and ordering.
//
// `ResidentWeight` and `ResidentWeightF32` stay private ON PURPOSE. They carry
// behaviour the shared ones do not (the i8mm repack marker, the elementwise
// transpose marker, keep-quant residency and the host-alias report), so
// migrating them is a separate change with its own gate.
using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using dense_attn::ResolveDevicePoolPolicy;



// Contiguous reinterpret of a device tensor's buffer at a new shape (same numel,
// same dtype/device). Used to view [T,H,D] as [T*H,D] etc. for rank-2 ops.

// DevicePool / Pool() / AuxPool() / ActivePool() / ActivePoolScope now live in
// the shared header include/vllm/model_executor/models/device_pool.h (extracted
// VERBATIM so the dense Qwen3 forward reuses the same pooled scratch; behavior
// here is byte-for-byte unchanged).

// The device-scratch residency policy (BACKEND-PLATFORM item 2), resolved from
// the running device's platform (per-object: keyed on the DBuf's own
// device.type, NOT the process-global CurrentPlatform). The DevicePool soft cap
// is now platform data, not an inline constant — a discrete GPU sets a bound and
// this file is unchanged. Memoized PER DEVICE TYPE because DBuf is a per-op hot
// path; platforms are fixed at static registration, so the value never changes
// afterward. It used to be ONE function-local static, which cached whichever
// device asked first and applied that cap to every later one — the same
// ambient-device assumption #516 fixed one layer down (dense_device_glue.h
// carries the identical repair). A backend whose platform was never REGISTERED
// therefore throws out of GetPlatform rather than inheriting the first device's
// cap — a cap read off another platform is a wrong number, not a default.

// --- Fused-MoE per-layer resident constants (M2.5 Phase 2, CUDA-graph unblock) -
// MoeBlockFusedCuda used to rebuild + re-upload, EVERY forward step, a set of
// per-layer CONSTANT device buffers: the E fp4-expert device-pointer/scale
// arrays (gate/up/down packed+scale ptrs, scale2) and the pair->token row map
// (tok_map, a function of T only). Those uploads copy from HOST STACK temporaries
// — illegal to have inside a CUDA-graph capture region (their host addresses
// dangle on replay). They are also pure per-step waste (the values never change).
// This process-lifetime cache (keyed by the layer's MoeBlockWeights address)
// uploads them ONCE, during the pre-warm forward, so the captured region only
// READS resident device buffers — no host-sourced copy, nothing to dangle. The
// device buffers leak at process exit (like the cublasLt workspace / the resident
// weights); they are bounded by (num_layers * (9*E + one tok_map per distinct T)).
struct MoeFusedResident {
  void* gp = nullptr;  // i64 [E] device: expert gate packed ptrs
  void* gs = nullptr;  // i64 [E] device: expert gate scale ptrs
  void* up = nullptr;  // i64 [E] device: expert up packed ptrs
  void* us = nullptr;  // i64 [E] device: expert up scale ptrs
  void* dp = nullptr;  // i64 [E] device: expert down packed ptrs
  void* ds = nullptr;  // i64 [E] device: expert down scale ptrs
  void* g2 = nullptr;  // f32 [E] device: expert gate scale2
  void* u2 = nullptr;  // f32 [E] device: expert up scale2
  void* d2 = nullptr;  // f32 [E] device: expert down scale2
  std::unordered_map<int64_t, void*> tok_map;  // T -> i32 [T*top_k] device
  bool ready = false;
};

// Fetch (building on first use) the resident state a weight owns. Replaces the
// process-lifetime `static std::unordered_map<const W*, R>` these accessors used
// to be: keying on the weight's ADDRESS let a second engine inherit a freed
// engine's device pointers (issue #237). See ResidentSlot in qwen3_5_weights.h.
//
// The lock is the one the map accessors already took on every call, kept rather
// than narrowed: this fix is about lifetime, and quietly changing the
// synchronisation of a hot path at the same time would make any regression
// ambiguous between the two.
template <typename R>
R& ResidentIn(const ResidentSlot& slot) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!slot.state) slot.state = std::make_shared<R>();
  return *static_cast<R*>(slot.state.get());
}

MoeFusedResident& MoeResidentFor(const MoeBlockWeights* w) {
  return ResidentIn<MoeFusedResident>(w->resident_fused);
}

// --- BF16 fast-MoE per-layer resident constants (Qwen3-Coder Qwen3MoeForCausalLM,
// W5). The bf16 analog of MoeFusedResident: the E per-expert bf16 [K,N] weight
// DEVICE pointers (gate/up/down) + the pair->token row map, uploaded ONCE during
// the pre-warm forward so the captured decode region only reads resident device
// buffers (no host-sourced copy to dangle on graph replay). The device pointers
// are the stable ResidentWeight uploads (each OwnedTensor's d_dev owns the copy
// for process lifetime); we capture them once. Leaked at process exit like the
// fp4 resident arrays / cublasLt workspace.
struct MoeBf16Resident {
  void* gate = nullptr;  // i64 [E] device: per-expert gate weight ptrs ([H,I] bf16)
  void* up = nullptr;    // i64 [E] device: per-expert up weight ptrs   ([H,I] bf16)
  void* down = nullptr;  // i64 [E] device: per-expert down weight ptrs  ([I,H] bf16)
  std::unordered_map<int64_t, void*> tok_map;  // T -> i32 [T*top_k] device
  bool ready = false;
};

MoeBf16Resident& MoeBf16ResidentFor(const MoeBlockWeights* w) {
  return ResidentIn<MoeBf16Resident>(w->resident_bf16);
}

// Fast BF16 grouped-MoE path (Qwen3-Coder). DEFAULT ON per the parity-enablers-
// ship-as-defaults policy: the lever the every-axis speed parity depends on ships
// default-ON, gated, BEFORE the binding speed run. VT_MOE_BF16_FAST=0 restores the
// per-expert host-gather reference loop (the correctness oracle) for same-binary
// A/B. Only consulted for bf16 experts on CUDA (fp4 experts always take the fused
// Marlin/wmma path; CPU/GGUF keeps the reference loop regardless).
bool MoeBf16FastEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_BF16_FAST");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 rolls back
  }();
  return on;
}

// LAYOUT PRECONDITION for the fast bf16 grouped-MoE path. The grouped kernel reads
// each expert weight as a bf16 [K,N] Matmul-B buffer (element (k,n) at k*N+n) and
// the router gate through the plain `vt::Matmul` (B = [H,E]) — i.e. the
// `LoadBf16Transposed` orientation (`nk == false`) produced by the Qwen3-Coder
// safetensors loader (qwen3_moe_weights.cpp:77-85) and the GGUF loader
// (qwen3_5_gguf_weights.cpp:403-422, `LoadExpertsT` transposes to [in,out]).
// It CANNOT read the raw torch-Linear orientation (`nk == true`, [N,K]), which the
// 35B MTP loader produces (`LoadBf16RawNK`/`CopyRawNK`, qwen3_5_mtp.cpp:109-133) —
// the reference loop below handles that via MatmulF32/MatmulBf16's `w.nk` branch
// (qwen3_5.cpp:721-737). So the fast path is taken ONLY when every weight it reads
// is in the layout it supports; anything else falls through to the reference loop
// (correct on every layout). This keeps the new path provably inert for the 35B
// (fp4 experts), the MTP module (nk=true), and any future nk=true producer instead
// of silently transposing their math.
bool MoeBf16FastLayoutOk(const MoeBlockWeights& w, const HfConfig& cfg) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t E = cfg.num_experts;
  if (w.router_gate.nk || w.router_gate.rank != 2 || w.router_gate.shape[0] != H ||
      w.router_gate.shape[1] != cfg.num_experts)
    return false;
  if (w.expert_gate.size() != static_cast<size_t>(E) ||
      w.expert_up.size() != static_cast<size_t>(E) ||
      w.expert_down.size() != static_cast<size_t>(E))
    return false;
  auto ok = [](const OwnedTensor& t, int64_t k, int64_t n) {
    return !t.nk && t.rank == 2 && t.shape[0] == k && t.shape[1] == n &&
           t.dtype == vt::DType::kBF16;
  };
  for (int64_t e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    if (!ok(w.expert_gate[se], H, I) || !ok(w.expert_up[se], H, I) ||
        !ok(w.expert_down[se], I, H))
      return false;
  }
  return true;
}

#ifdef VT_MARLIN_NVFP4
// --- Marlin NVFP4 W4A16 MoE: per-layer resident repacked weights (M0.8 drop-in).
// When VT_NVFP4_MARLIN=1, the routed experts are repacked ONCE at first touch
// into Marlin's interleaved layout (gptq_marlin_moe_repack + S0E5M3 scales +
// processed global scales, all bit-exact to vLLM — tools/marlin/repack_*), and
// the wmma-fp4 fused path is replaced by moe_wna16_marlin_gemm. The original
// per-expert fp4 device copies are freed after repack (the wmma path is unused
// when the gate is on) so peak weight memory stays flat. Buffers leak at exit
// (like the wmma resident / cublasLt workspace).
struct MoeMarlinResident {
  void* w_gate = nullptr;  // i32 [E, K/16, N*2]  (gate: size_k=K, size_n=N)
  void* w_up = nullptr;    // i32 [E, K/16, N*2]
  void* w_down = nullptr;  // i32 [E, N/16, K*2]  (down: size_k=N, size_n=K)
  void* s_gate = nullptr;  // fp8 [E, K/16, N]
  void* s_up = nullptr;
  void* s_down = nullptr;  // fp8 [E, N/16, K]
  void* g_gate = nullptr;  // f32 [E]
  void* g_up = nullptr;
  void* g_down = nullptr;
  // Fused w13 layout (VT_MOE_FUSED_W13): gate+up CONCATENATED along N per expert
  // — rows [0,N) = gate (vLLM w1), rows [N,2N) = up (vLLM w3) — repacked as ONE
  // Marlin B operand of size_n=2N, mirroring vLLM's stacked w13_weight
  // (marlin_utils_fp4.py prepare_nvfp4_moe_layer_for_marlin:374-401 repacks the
  // stacked [E, 2N, K/2] per expert with size_n = num_shards*N). Populated
  // INSTEAD of w_gate/w_up (same total bytes) when fused_w13 is true.
  void* w_gu = nullptr;       // i32 [E, K/16, (2N)*2]
  void* s_gu = nullptr;       // fp8 [E, K/16, 2N]
  void* g_gu = nullptr;       // f32 [E]  (gate scale2 — vLLM w13_weight_scale_2[:, 0])
  bool fused_w13 = false;
  void* workspace = nullptr;  // i32 [sms]
  int sms = 0;
  bool ready = false;
};

MoeMarlinResident& MoeMarlinResidentFor(const MoeBlockWeights* w) {
  return ResidentIn<MoeMarlinResident>(w->resident_marlin);
}

bool MarlinMoeEnabled() {
  // Default ON: the vendored Marlin NVFP4 W4A16 GEMM is the validated 35B path
  // (measured gate +22%, decode-heavy +80%, 16/16 token-for-token vs the pinned
  // oracle — see parity-ledger). Only an explicit VT_NVFP4_MARLIN=0 opts back out
  // to the naive redundant-dequant / cublas bf16 GEMM (kept as an escape hatch).
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_MARLIN");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Fused w13 grouped GEMM (VT_MOE_FUSED_W13, DEFAULT ON): run the routed
// experts' gate+up as ONE Marlin grouped GEMM over the N-concatenated w13
// weights (size_n=2I, output [P,2I]) + one SiluAndMul over the halves, instead
// of TWO grouped GEMMs (+2 workspace memsets, 2 schedule passes) + MoeSiluMul;
// same fusion for the shared-expert gate_up (SharedGateUpFusedMarlinD). This is
// exactly vLLM's marlin_moe.py shape: ONE moe_wna16_marlin_gemm with
// size_n = w13_num_shards * N into intermediate_cache1 [M*topk, 2N]
// (fused_moe/experts/marlin_moe.py:133-160), then silu_and_mul on the [:N]/[N:]
// halves (:162-170). At the 35B decode shape (I=512, top_k=8, many tiny
// latency-bound tiles) the second GEMM's fixed costs are pure overhead.
// GATED ON (GB10, 2026-07-10): fused-vs-split BIT-EXACT (test_ops_moe_grouped
// probe), 35B greedy 16/16-vs-oracle BOTH arms, and a clean same-binary A/B win
// (in1024/out128 np200 conc-64, 3+3 interleaved reps: 3166.85 -> 3262.28 tok/s
// = +3.01%, TPOT -3.1%, TTFT -1.4%; MoE-expert fusion alone +0.53%, the rest
// from the shared-expert gate_up fusion). VT_MOE_FUSED_W13=0 opts back out to
// the split two-GEMM layout for A/B.
// The layout choice is made at LOAD (BuildMoeMarlinResident builds either the
// fused or the split resident), so A/B = two runs of the same binary.
bool MoeFusedW13Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_FUSED_W13");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}
#endif  // VT_MARLIN_NVFP4

// Owned device allocation + tensor view. On CPU the backend's Alloc/Copy are
// malloc/memcpy; on CUDA they are cudaMalloc / h2d-d2h on the queue's stream.
// Allocation is routed through the DevicePool so the buffer's storage is reused
// rather than freed to the driver (avoiding the cudaMalloc/cudaFree sync).

float SizeF(int64_t n) { return static_cast<float>(n); }
float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// Upcast a bf16 owned weight to an f32 host buffer (lossless). The CUDA norm /
// conv kernels require the weight dtype to match the activation dtype; where
// activations are f32 (GDN conv/gated-norm, attention qk-norm, final-norm
// replay), the bf16 weight must be presented as f32.
//
// `w.bytes` MAY BE A BORROWED MAPPING AT AN ODD ADDRESS, so the bytes are read
// through `vt::LoadUnaligned` off a byte cursor rather than a `const uint16_t*`.
// A safetensors payload starts at `8 + <JSON header length>`
// (`safetensors_reader.cpp:78`), a header length is arbitrary, and
// `BorrowStTensorBytes` hands those bytes over verbatim. This site aborted
// `test_qwen35_exl3` under `-fsanitize=alignment`, reached from
// `ModelRegistry::Forward` through `FullAttnBlockPaged`, and it was hidden
// behind the EXL3 finding because that lane stops at the first report (#2578,
// the sixth recurrence of the class in
// `.agents/specs/unaligned-safetensors-consumers.md`).
std::vector<float> WeightF32(const OwnedTensor& w) {
  const auto* src = w.bytes.data();
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    out[static_cast<size_t>(i)] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(src + i * 2));
  return out;
}

// Device-resident raw-dtype view over an owned weight, uploaded ONCE (lazily)
// and reused across every forward step (mirrors ResidentNvfp4). The forward's
// bf16/f32 weights (embed table, layernorms, attention/GDN projections, router)
// were re-uploaded per op — the ~600MB embed table alone re-copied every step —
// dominating the measured 67.5%-of-wall cudaMemcpyAsync. Caching kills the
// re-upload: the shared_ptr in the (const) weight owns the device buffer for the
// model's lifetime. On CPU the bytes are already host-resident, so a direct view
// avoids the copy. The weight is a read-only matmul-B / norm / embed operand, so
// the const_cast is safe. `shape` defaults to the owned shape.
// Print what the W0f aliasing branch has actually done, every 4 GiB of weight it
// has seen, on the same `VT_LOAD_STATS` switch the loader's byte counters use.
//
// WHY PERIODIC AND NOT AT EXIT. The `[vt load] bytes@exit` line is registered
// with `std::atexit`, and the run this instruments is one a memory guard
// SIGKILLs — no exit handler runs, so the one number that would have explained
// the run is the one number the run cannot print. W0f's first device attempt was
// read from an RSS curve for exactly that reason, and an RSS curve cannot tell
// "declined and staged" from "re-homed and the pages did not come back".
void ReportHostAliasResidency() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LOAD_STATS");
    return e != nullptr && e[0] != '0';
  }();
  if (!on) return;
  const vllm::HostAliasStats s = vllm::HostAliasSnapshot();
  const uint64_t total = s.aliased_in_place_bytes + s.rehomed_bytes +
                         s.declined_borrow_bytes + s.declined_other_bytes;
  static uint64_t last = 0;
  constexpr uint64_t kStep = 4ULL << 30;
  if (total < last + kStep && last != 0) return;
  last = total;
  // BOUNDED, because the counter this trips on is CUMULATIVE OVER CALLS and
  // never stops growing. `ResidentWeight` re-enters the alias branch about 1,361
  // times per decode step, roughly 70 GiB of counted bytes, so a 4 GiB step
  // prints about 17 lines EVERY step for the life of the process. The first
  // forward is what this instrument exists to explain — it is where the aliasing
  // set is established and where the recorded 60.793 GiB was read — and that
  // fits inside the cap with room to spare. Everything after it is the same
  // weights being counted again.
  static int lines = 0;
  constexpr int kMaxLines = 24;
  if (lines >= kMaxLines) return;
  ++lines;
  const double gib = 1024.0 * 1024.0 * 1024.0;
  // "per call", spelled out in the line itself. These are BYTES SEEN, not bytes
  // resident: a weight aliased on every step is counted on every step, so the
  // figures are traffic and become a residency measurement only when read at a
  // stated point (see HostAliasStats in qwen3_5_weights.h).
  std::fprintf(stderr,
               "[vt load] w0f-alias per-call totals: calls=%llu "
               "aliased_in_place=%.3f GiB rehomed=%.3f GiB "
               "declined_borrow=%.3f GiB declined_other=%.3f GiB\n",
               static_cast<unsigned long long>(s.calls),
               static_cast<double>(s.aliased_in_place_bytes) / gib,
               static_cast<double>(s.rehomed_bytes) / gib,
               static_cast<double>(s.declined_borrow_bytes) / gib,
               static_cast<double>(s.declined_other_bytes) / gib);
  if (lines == kMaxLines)
    std::fprintf(stderr,
                 "[vt load] w0f-alias: %d lines printed; further lines are "
                 "suppressed (the counters keep running)\n",
                 kMaxLines);
}

Tensor ResidentWeight(Dev d, const OwnedTensor& w, std::vector<int64_t> shape = {}) {
  if (shape.empty()) shape.assign(w.shape, w.shape + w.rank);
  // HOST-POINTER ALIASING IS A CPU PROPERTY, NOT A "NOT-CUDA" PROPERTY (issue
  // #125). This read `!needs_weight_staging()`, which is true ONLY on CUDA
  // (platforms/cuda.cpp; the base default is false and neither Vulkan, Metal nor
  // XPU overrides it) -- so every DEVICE backend except CUDA aliased the host
  // weight bytes into a tensor tagged with a device and handed a HOST pointer to
  // a DEVICE kernel. On Vulkan that surfaces as "embedding: table points outside
  // every Vulkan allocation" on the first native kernel of the forward.
  //
  // The correct predicate is `is_cpu()`: alias when the "device" IS the host,
  // upload otherwise. It leaves CPU and CUDA on exactly the branches they already
  // took (CPU: is_cpu true / staging false -> alias; CUDA: is_cpu false / staging
  // true -> upload), so this cannot change either.
  //
  // include/vllm/model_executor/models/dense_attn_block.h carries the SAME helper
  // already fixed this way, and 25 model files inherit it. This file is not one of
  // them -- it kept a private copy, so the fix never reached it. That is the
  // off-framework-model hazard the decode-framework-routing audit names.
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu()) {
    Tensor t = w.ViewOn(const_cast<uint8_t*>(w.bytes.data()), d.q.device, shape);
    // This arm owns repacked/elem_kn_repacked and NOT q8_0_aligned, exactly as
    // it did before ViewOn: MakeTensor dropped all three and this arm restored
    // these two. The second resident helper below returns the view unchanged,
    // which also matches main, where it returns MakeTensor with no markers.
    t.repacked = w.repacked;
    t.elem_kn_repacked = w.elem_kn_repacked;
    return t;
  }
  // AUDIT GUARD (KERNEL-GEMM-CPU-TILED lever 2). Only the CPU MatmulBTKernel
  // honours elem_kn_repacked, and the staging path below uploads bytes verbatim
  // and returns a tensor WITHOUT the marker, so a repacked weight reaching a
  // staged device would be read as plain [N,K] and produce garbage silently.
  // VT_CPU_ELEM_KN_REPACK is CPU-only and the loader policy cannot see the
  // device, so this is where the invariant is enforced: fail loudly at load
  // rather than corrupt tokens at inference.
  VT_CHECK(!w.elem_kn_repacked,
           "qwen3_5: an elem_kn_repacked ([K,N]) weight reached device staging; "
           "VT_CPU_ELEM_KN_REPACK is a CPU-only load transform");
  // ENG-EXPERT-STREAM-DEVICE W0c (issues #1123, #1124). THE ALLOCATION THIS
  // ROW EXISTS TO PREVENT, guarded at the one line that makes it.
  //
  // `nb` below is `w.bytes.size()` — the WHOLE stacked `[E*N,K]` tower, which on
  // `Qwen3.8-2.4T-A95B UD-Q1_0` is 1,275,068,416 bytes, 512 experts at once. The
  // streamed lane exists so that exactly one 2,490,368-byte slice of it is
  // resident at a time. A tower that has been claimed by the lane and then
  // reaches this branch has therefore defeated the lane completely, and it does
  // so INVISIBLY: the first 48 towers fit, and the load dies partway through
  // layer 16 of 93 with `cudaMalloc: out of memory` (issue #1123). Failing here,
  // by name, turns that into one legible error at the first tower.
  //
  // It is a tripwire and not a reachable production path — see the field's
  // comment in qwen3_5_weights.h for the two routes that would otherwise arrive
  // here and why neither can today.
  VT_CHECK(!w.expert_streamed,
           "qwen3_5: a STREAMED expert tower reached device staging; the "
           "expert-stream lane serves its slices from host slot storage and the "
           "whole tower must never be uploaded (ENG-EXPERT-STREAM-DEVICE W0c, "
           "issues #1123 and #1124)");
  // The SAME invariant as the elem_kn_repacked guard above, for the i8mm
  // interleave, and it was missing until now (issue #1320). The CUDA
  // quant dot reads `block_q8_0`; `VT_CPU_QUANT_REPACK` rewrites the buffer to
  // `block_q8_0x4` at load and only the CPU MatmulBTKernel understands that.
  // THE POLICY NOW AGREES WITH THIS GUARD (#2406). `quant_repack` used to ride
  // `QuantRepackActive()` alone — a HOST-CPU i8mm probe with no device term —
  // so an aarch64 box doing `--device cuda` repacked a Q8_0 weight and uploaded
  // it verbatim to a kernel that misreads it. `GgufLoadPolicy::FromEnv` now
  // resolves it through `QuantRepackForDevice(..., dev)`, gated `dev == kCPU`
  // exactly as its sibling `elem_kn_repack` always was.
  //
  // TWO CORRECTIONS TO WHAT THIS COMMENT USED TO SAY. It called the gap
  // "measured harmless on the target checkpoint (one Q8_0 tensor, 0.01% of
  // parameters, and the instrumented load recorded `quant_repack = 0`)". The
  // released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores **194** Q8_0
  // hyper-connection mix weights (docs/USAGE.md), so the population was never
  // one tensor; and a repacked `block_q8_0x4` weight DID reach device residency
  // and refuse here by name on `thor:gpu0`, so the reading of `quant_repack = 0`
  // did not generalise past the run that took it. The guard stays as belt to
  // the policy's braces — a runtime refusal is what catches a future caller that
  // builds a policy by hand — but it is no longer the only thing standing
  // between an i8mm host and a device upload. `VT_CPU_QUANT_REPACK=0` remains
  // the operator's same-binary way past the transform entirely.
  VT_CHECK(!w.repacked,
           "qwen3_5: an i8mm-repacked (block_q8_0x4) weight reached device "
           "residency; VT_CPU_QUANT_REPACK is a CPU-only load transform and the "
           "device quant kernels read plain block_q8_0");
  // ENG-EXPERT-STREAM-DEVICE W0f (issue #1299). THE SECOND COPY THIS ROW EXISTS
  // TO PREVENT, at the one line that makes it.
  //
  // Everything below this branch is a VERBATIM byte copy: `Alloc(w.bytes.size())`,
  // `Copy`, then a tensor with the same dtype, the same shape and the same
  // (dropped) marker set as the source. Nothing about the bytes changes, which is
  // exactly why a token gate cannot see the cost — and the cost is a second full
  // resident copy of every dense weight. On a discrete device that copy is the
  // whole point: the kernel cannot follow a host pointer. On a part whose kernels
  // CAN, it buys nothing and comes out of the same RAM the first copy did.
  //
  // MEASURED (#1299, `dgx:gpu0`, seven runs). `Qwen3.8-2.4T-A95B UD-Q1_0` loads
  // on `--device cuda` at 61.20 GiB resident and then exhausts a 119.631 GiB box
  // inside the FIRST forward, zero decode steps, every time. A 0.15 GiB slot
  // arena died exactly where an 18.55 GiB one did, so the arena is not the cost;
  // growth was anonymous while file-backed stayed flat, so the mapping is not
  // pinned. About 39 GiB of that 61.20 is `attn_qkv` (21.56) and `ssm_out`
  // (17.25), which the GDN V-head reorder makes `kTransformedWeight` and
  // therefore expands to bf16 in OWNED host buffers — the split is measured in
  // `.agents/specs/expert-streaming.md`, not derived here. The CPU arm pays that
  // once and serves. This branch is what stops the CUDA arm paying it twice.
  //
  // WHY THE SAME PREDICATE AS W0c AND NOT A NEW ONE. `KqExpertSlice` already
  // hands this platform a host pointer for every expert slice it serves; a dense
  // weight is the same question about a different tensor. `is_cpu()` is what the
  // early return above answers, `needs_weight_staging()` is true on CUDA
  // everywhere and would gate nothing, and `is_unified_memory()` answers the
  // opposite question — GB10 reports unified while a `cudaMalloc` pointer is
  // still not host-dereferenceable (vt/backend.h). A DISCRETE device answers
  // false here, falls through, and gets byte-for-byte what it gets today.
  if (vllm::platforms::GetPlatform(d.q.device.type)
          .host_memory_is_device_addressable()) {
    // A weight with NEITHER host bytes NOR a device copy cannot be served at
    // all, and the staging branch below would not notice: it would `Alloc(0)`,
    // copy nothing, and hand out a pointer to nothing. That is the precondition
    // this states.
    //
    // THE `w.d_dev` HALF IS NOT DEFENSIVE, AND THIS CHANGE IS WHAT CREATED THE
    // POPULATION IT SERVES (found by a fresh review of #1299). A weight whose
    // host mirror is gone but whose `d_dev` is populated has ALWAYS been served,
    // by the memo below, and it returned the device tensor without complaint.
    // W0f put this check ABOVE that memo, so the same weight began to throw. The
    // justification written here first — "`ReleaseHost()` is not reachable for
    // the dense weights this branch serves" — is true of the dense weights and
    // FALSE of the expert weights the same function serves at the `gp/up/dp`
    // capture below, whose misaligned GGUF borrows decline the alias, stage, get
    // a `d_dev`, and are then released by the guarded loop beside that capture.
    // So the condition is "nothing to serve", not "no host bytes".
    VT_CHECK(!w.bytes.empty() || w.d_dev != nullptr,
             "qwen3_5: a weight reaching device residency has no host bytes and "
             "no device copy; its host mirror was released and there is nothing "
             "to alias or upload");
    // ...and with no host bytes there is nothing to alias, so skip the attempt
    // rather than charging a `kDeclinedEmpty` to the residency instrument for a
    // weight that is already device-resident.
    // PERF: prefer a TRUE DEVICE COPY when the box can hold one. The retag
    // below is a measured decode tax on GB10 -- +22.5% throughput when staged
    // instead, interleaved A/B on one boot (see `DeviceStagingFits`) -- and it
    // exists only because #1299's 2.4T checkpoint cannot afford to pay for its
    // weights twice. `DeviceStagingFits` asks that question of the BOX, so a
    // model that fits stages and a model that does not keeps the retag.
    // Falling through skips the alias attempt and reaches the staging branch.
    const bool stage_instead = !w.bytes.empty() && StageOwnedWeightsToDevice();
    if (!w.bytes.empty() && !stage_instead) {
      const bool aliased = MakeHostBytesDeviceAliasable(w);
      ReportHostAliasResidency();
      if (aliased) {
        // NOT `load_stats::AddDeviceUpload`: nothing was uploaded. Issue #150's
        // counter measures bytes moved host->device, and this branch moves none.
        return w.ViewOn(const_cast<uint8_t*>(w.bytes.data()), d.q.device, shape);
      }
    }
    // A MISALIGNED BORROW, or the `VT_QWEN35_ALIAS_HOST_WEIGHTS=0` A/B, reaches
    // here. A borrow's pages are clean and file-backed, so staging copies them
    // without adding anonymous residency. Falling through is deliberate and is
    // not a failure; `ReportHostAliasResidency` above says how often it happens
    // and for how many bytes.
  }
  if (!w.d_dev) {
    const size_t nb = w.bytes.size();
    void* p = d.b.Alloc(nb);
    // Issue #150 accounting: the host->device weight upload for this family.
    // When `w.bytes` borrows the safetensors mapping (ENG-LOAD-DIRECT-UPLOAD)
    // the source of this copy IS the file mapping, so the byte moved once.
    vllm::load_stats::AddDeviceUpload(nb);
    d.b.Copy(d.q, p, w.bytes.data(), nb);
    Backend* bk = &d.b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    // THE SOURCE PAGES ARE SPENT, AND ON A STAGING DEVICE NOTHING WAS DROPPING
    // THEM (.agents/specs/rocm-host-residency-after-upload.md). The copy above
    // is the only read of this weight's host bytes that will ever happen there,
    // and for a GGUF keep-quant load those bytes are the whole model -- 65.488
    // GiB on `Qwen3.8-Flash-Next UD-IQ1_S`, faulted in by the load-time prefault
    // and then held resident for the process lifetime, which is what wedges a
    // 31 GiB gfx1151 host inside `svm_range_set_attr`. llama.cpp
    // `unmap_fragment`s the equivalent range after its offloaded tensors are set
    // (llama-model-loader.cpp:1683-1694). `AdoptDeviceBytesAsHost` below cannot
    // do this job: it returns immediately for a GGUF borrow, by design, because
    // the borrow must NOT be re-pointed at a device allocation the kernels are
    // the only readers of.
    //
    // INSIDE THE `d_dev` MEMO ON PURPOSE, which is the whole of #1299's lesson:
    // this function runs about 1,361 times per forward step, and a release that
    // re-tested its condition each time would madvise away the pages the GPU is
    // about to read, every step. The helper states its other two preconditions
    // and synchronizes the queue before it touches anything.
    vllm::MaybeReleaseStagedBorrowSource(
        d.b, d.q, w,
        vllm::platforms::GetPlatform(d.q.device.type)
            .host_memory_is_device_addressable());
    // Same adoption as the dense block's ResidentWeight: on a host-addressable
    // device the uploaded buffer IS the host buffer, so keeping the mirror
    // costs a second full copy of the model out of the same unified RAM.
    AdoptDeviceBytesAsHost(d.b, w);
  }
  return w.ViewOn(w.d_dev.get(), d.q.device, shape);
}

}  // namespace (closed so the bridge below has EXTERNAL linkage; the unnamed
   // namespace reopens immediately after, and its names stay visible)

// See qwen3_5.h. One line of body on purpose: this is a NAME for the bridge the
// forward already used, not a new path. Making it nameable is what lets the
// reachability gate enter through the same call the forward makes instead of
// hand-building the operand one step later.
Tensor Qwen3_5EmbeddingTable(vt::Backend& backend, vt::Queue& queue,
                             const OwnedTensor& embed_tokens, int64_t vocab,
                             int64_t hidden) {
  Dev d{backend, queue};
  return ResidentWeight(d, embed_tokens, {vocab, hidden});
}

namespace {

// ROCM-TIER-DIVERGENCE (#2590): the device-side half of the activation dump.
// The keying, the manifest and every refusal live in
// `vllm/model_executor/models/act_dump.h`, apart from these two so that a test
// can drive the writer with no backend at all. See that header for why the
// three predecessor dumps could not answer this question.
// Download a device tensor and write it. `rows`/`cols` are the logical [T,H] or
// [T,N] view; the manifest carries them so the comparator never guesses.
void ActDumpTensor(Dev d, const char* knob, const std::string& dir,
                   const char* stage, const Tensor& t, int64_t rows, int64_t cols) {
  const actdump::Where w = actdump::Current();
  const int64_t n = rows * cols;
  std::vector<uint8_t> raw(static_cast<size_t>(n) * vt::SizeOf(t.dtype));
  DBuf tmp(d, t.dtype, {n});
  d.b.Copy(d.q, tmp.ptr(), t.data, raw.size());
  tmp.Download(d, raw.data());  // Copy + Synchronize
  actdump::WriteBlob(knob, dir, w.step, w.layer, stage, t.dtype, rows, cols,
                     raw.data(), raw.size());
}


// Both halves of the residual stream at one position. `layer` is -1 for the
// post-embedding snapshot and the loop index otherwise. Inert unless
// `VT_DUMP_ACT` names a directory.
void ActDumpStream(Dev d, int64_t step, int64_t layer, DBuf& hidden, DBuf& res,
                   int64_t T, int64_t H) {
  const char* dir = actdump::StreamDir();
  if (dir == nullptr || step < 0) return;
  const actdump::LayerScope here(step, layer);
  ActDumpTensor(d, "VT_DUMP_ACT", dir, "hidden", hidden.t(), T, H);
  ActDumpTensor(d, "VT_DUMP_ACT", dir, "res", res.t(), T, H);
  // Counted apart from every other stage: see `g_stream_blobs_step`. A gate
  // keyed on the total cannot tell this call site from the GDN stage probes.
  actdump::g_stream_blobs_step.fetch_add(2, std::memory_order_relaxed);
}

// Device-resident f32 upcast of a bf16 owned weight, uploaded ONCE. Matches the
// CUDA norm/conv kernels' requirement that the weight dtype equal the (f32)
// activation dtype (GDN conv1d / gated-norm, attention qk-norm). `shape` is the
// logical view (e.g. {conv_dim, Kw}).
Tensor ResidentWeightF32(Dev d, const OwnedTensor& w,
                         const std::vector<int64_t>& shape) {
  // THIS HELPER STAYS PRIVATE (see :790) BUT ITS UPLOAD DOES NOT. The install
  // is `dense_attn::InstallResidentF32` (dense_device_glue.h, included at :24),
  // the same one `dense_attn_block.h` uses. What used to be private was a
  // verbatim second copy of the upload, which is why #2711 -- an asynchronous
  // `Copy` out of a function-local `std::vector<float>` that died at this
  // closing brace -- had to be found and repaired twice. The behaviour this
  // file's version carries beyond the shared one is in `ResidentWeight`, not
  // here: the two f32 bodies were identical.
  if (!w.d_dev_f32) dense_attn::InstallResidentF32(d, w, WeightF32(w));
  return MakeTensor(w.d_dev_f32.get(), DType::kF32, d.q.device, shape);
}

// y[M,N] f32 = x[M,K] bf16 @ w[K,N] (w owns [K,N] bf16). f32 output keeps the
// GEMM's f32 accumulation for the f32 glue that consumes it.
std::vector<float> MatmulF32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                             int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// y[M,N] bf16 = x[M,K] bf16 @ w[K,N] bf16 (bf16 output mirrors the model's bf16
// hidden states where the result feeds the residual stream / next matmul).
std::vector<uint16_t> MatmulBf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                 int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}


// --- NVFP4 fp4-resident weight helpers (M2.2b) ------------------------------

// Device-resident views over an Nvfp4Weight's packed + scale buffers. Valid for
// the lifetime of the weight (the buffers are owned by the weight's shared_ptr).
struct Nvfp4Dev {
  Tensor packed;
  Tensor scale;
};

// Upload packed + scale to the device ONCE (lazily, on first use) and keep them
// resident: the shared_ptr in the (const) weight owns the device buffer across
// every forward step, so subsequent calls reuse the resident copy — no per-op
// weight staging. CUDA path only; the deleter frees through the vt Backend.
Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    // ENG-LOAD-DIRECT-UPLOAD (issue #150): the 27B `LoadCtNvfp4Raw` weights
    // BORROW packed/scale from the safetensors mmap, so this is their one
    // host->device move. Account it and run the same post-upload residency step
    // every other qualifying weight gets, exactly as dense_nvfp4_gemm.h's
    // shared ResidentNvfp4 does. Publishing the allocation on the OwnedTensor
    // is what lets AdoptDeviceBytesAsHost run (it keys on `d_dev`); the two
    // handles share one control block, so the buffer is freed exactly once.
    vllm::load_stats::AddDeviceUpload(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    w.packed.d_dev = w.d_packed;
    AdoptDeviceBytesAsHost(d.b, w.packed);
  }
  if (!w.d_scale) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    vllm::load_stats::AddDeviceUpload(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.d_scale = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    w.scale.d_dev = w.d_scale;
    AdoptDeviceBytesAsHost(d.b, w.scale);
  }
  Nvfp4Dev r;
  r.packed = MakeTensor(w.d_packed.get(), DType::kI8, d.q.device, {w.n, w.k / 2});
  r.scale = MakeTensor(w.d_scale.get(), DType::kI8, d.q.device, {w.n, w.k / 16});
  return r;
}

std::vector<int64_t> CutlassFp4ScaleShape(int64_t rows, int64_t k) {
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return {round_up(rows, 128), round_up(k / 16, 4)};
}

#ifdef VT_CUTLASS_NVFP4
// Resident SWIZZLED weight block scale for the cutlass fp4 GEMM. Computed ONCE
// (lazily) from the resident linear d_scale via vt::SwizzleBlockscale and kept
// on the weight's shared_ptr. Shape [round_up(n,128), round_up(k/16,4)].
Tensor ResidentNvfp4ScaleSwizzled(Dev d, const Nvfp4Weight& w) {
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  const int64_t Np = round_up(w.n, 128), Kp = round_up(w.k / 16, 4);
  if (!w.d_scale_sw) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);  // ensures d_scale (linear device copy)
    void* p = d.b.Alloc(static_cast<size_t>(Np * Kp));
    Backend* bk = &d.b;
    w.d_scale_sw = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    Tensor sw = MakeTensor(p, DType::kI8, d.q.device, {Np, Kp});
    vt::SwizzleBlockscale(d.q, sw, dw.scale);
  }
  return MakeTensor(w.d_scale_sw.get(), DType::kI8, d.q.device, {Np, Kp});
}

// Device-resident packed operand for the dense CT gate_up_proj. Pinned vLLM
// represents gate/up as one MergedColumnParallelLinear (`qwen2_moe.py:75-115`)
// and loads both checkpoint shards into one N-concatenated parameter
// (`linear.py:580-695`). We retain the split host weights for the diagnostic
// arm, but production uploads one [2I,H/2] packed buffer and swizzles the one
// concatenated [2I,H/16] block-scale buffer exactly once.
struct Nvfp4GateUpDev {
  Tensor packed;
  Tensor scale_sw;
  DenseGateUpGlobals globals;
};

Nvfp4GateUpDev ResidentNvfp4GateUp(Dev d, const DenseMlpWeights& w) {
  const Nvfp4Weight& gate = w.gate_proj_fp4;
  const Nvfp4Weight& up = w.up_proj_fp4;
  VT_CHECK(!gate.Empty() && !up.Empty(),
           "qwen3_5 dense merged gate_up: empty logical shard");
  VT_CHECK(gate.n == up.n && gate.k == up.k,
           "qwen3_5 dense merged gate_up: logical shard shape mismatch");
  const int64_t n = gate.n;
  const int64_t k = gate.k;
  const size_t packed_shard_bytes = gate.packed.bytes.size();
  const size_t scale_shard_bytes = gate.scale.bytes.size();
  VT_CHECK(up.packed.bytes.size() == packed_shard_bytes &&
               up.scale.bytes.size() == scale_shard_bytes,
           "qwen3_5 dense merged gate_up: logical shard byte mismatch");

  if (!w.d_gate_up_packed || !w.d_gate_up_scale_sw) {
    VT_CHECK(!w.d_gate_up_packed && !w.d_gate_up_scale_sw,
             "qwen3_5 dense merged gate_up: partial resident state");
    Backend* backend = &d.b;
    void* packed_data = d.b.Alloc(2 * packed_shard_bytes);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* packed_bytes = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, packed_bytes, gate.packed.bytes.data(), packed_shard_bytes);
    d.b.Copy(d.q, packed_bytes + packed_shard_bytes, up.packed.bytes.data(),
             packed_shard_bytes);

    // Linear scale staging is pool-backed and can return immediately after the
    // swizzle launch: all reuse is on this queue and therefore stream-ordered.
    DBuf scale_linear(d, DType::kI8, {2 * n, k / 16});
    auto* scale_bytes = static_cast<uint8_t*>(scale_linear.ptr());
    d.b.Copy(d.q, scale_bytes, gate.scale.bytes.data(), scale_shard_bytes);
    d.b.Copy(d.q, scale_bytes + scale_shard_bytes, up.scale.bytes.data(),
             scale_shard_bytes);

    const auto round_up = [](int64_t value, int64_t multiple) {
      return (value + multiple - 1) / multiple * multiple;
    };
    const int64_t np = round_up(2 * n, 128);
    const int64_t kp = round_up(k / 16, 4);
    void* scale_sw_data = d.b.Alloc(static_cast<size_t>(np * kp));
    std::shared_ptr<void> scale_sw_owner(
        scale_sw_data, [backend](void* pointer) { backend->Free(pointer); });
    Tensor scale_sw =
        MakeTensor(scale_sw_data, DType::kI8, d.q.device, {np, kp});
    vt::SwizzleBlockscale(d.q, scale_sw, scale_linear.t());

    w.d_gate_up_packed = std::move(packed_owner);
    w.d_gate_up_scale_sw = std::move(scale_sw_owner);
  }

  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return Nvfp4GateUpDev{
      MakeTensor(w.d_gate_up_packed.get(), DType::kI8, d.q.device,
                 {2 * n, k / 2}),
      MakeTensor(w.d_gate_up_scale_sw.get(), DType::kI8, d.q.device,
                 {round_up(2 * n, 128), round_up(k / 16, 4)}),
      MergeDenseGateUpGlobals(gate, up)};
}

// Device-resident packed operand for Qwen3NextAttention's QKVParallelLinear.
// vLLM loads the three logical checkpoint shards into one N-concatenated
// parameter (`qwen3_5.py:279-288`, `qwen3_next.py:252-270`,
// `linear.py:942-1050`) and performs one GEMM before splitting the output.
struct Nvfp4QkvDev {
  Tensor packed;
  Tensor scale_sw;
  FullAttnQkvGlobals globals;
  int64_t qn = 0;
  int64_t kn = 0;
  int64_t vn = 0;
};

Nvfp4QkvDev ResidentNvfp4Qkv(Dev d, const FullAttnLayerWeights& w) {
  const Nvfp4Weight& q = w.q_proj_fp4;
  const Nvfp4Weight& k = w.k_proj_fp4;
  const Nvfp4Weight& v = w.v_proj_fp4;
  VT_CHECK(!q.Empty() && !k.Empty() && !v.Empty(),
           "qwen3_5 packed QKV: empty logical shard");
  VT_CHECK(q.k == k.k && q.k == v.k,
           "qwen3_5 packed QKV: logical shard K mismatch");
  const int64_t total_n = q.n + k.n + v.n;
  const int64_t inner_k = q.k;
  const size_t q_packed_bytes = q.packed.bytes.size();
  const size_t k_packed_bytes = k.packed.bytes.size();
  const size_t v_packed_bytes = v.packed.bytes.size();
  const size_t q_scale_bytes = q.scale.bytes.size();
  const size_t k_scale_bytes = k.scale.bytes.size();
  const size_t v_scale_bytes = v.scale.bytes.size();
  VT_CHECK(q_packed_bytes == static_cast<size_t>(q.n * inner_k / 2) &&
               k_packed_bytes == static_cast<size_t>(k.n * inner_k / 2) &&
               v_packed_bytes == static_cast<size_t>(v.n * inner_k / 2),
           "qwen3_5 packed QKV: packed shard byte mismatch");
  VT_CHECK(q_scale_bytes == static_cast<size_t>(q.n * inner_k / 16) &&
               k_scale_bytes == static_cast<size_t>(k.n * inner_k / 16) &&
               v_scale_bytes == static_cast<size_t>(v.n * inner_k / 16),
           "qwen3_5 packed QKV: scale shard byte mismatch");

  if (!w.d_qkv_packed || !w.d_qkv_scale_sw) {
    VT_CHECK(!w.d_qkv_packed && !w.d_qkv_scale_sw,
             "qwen3_5 packed QKV: partial resident state");
    Backend* backend = &d.b;
    const size_t packed_bytes =
        q_packed_bytes + k_packed_bytes + v_packed_bytes;
    void* packed_data = d.b.Alloc(packed_bytes);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* packed_dst = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, packed_dst, q.packed.bytes.data(), q_packed_bytes);
    d.b.Copy(d.q, packed_dst + q_packed_bytes, k.packed.bytes.data(),
             k_packed_bytes);
    d.b.Copy(d.q, packed_dst + q_packed_bytes + k_packed_bytes,
             v.packed.bytes.data(), v_packed_bytes);

    DBuf scale_linear(d, DType::kI8, {total_n, inner_k / 16});
    auto* scale_dst = static_cast<uint8_t*>(scale_linear.ptr());
    d.b.Copy(d.q, scale_dst, q.scale.bytes.data(), q_scale_bytes);
    d.b.Copy(d.q, scale_dst + q_scale_bytes, k.scale.bytes.data(),
             k_scale_bytes);
    d.b.Copy(d.q, scale_dst + q_scale_bytes + k_scale_bytes,
             v.scale.bytes.data(), v_scale_bytes);

    const auto round_up = [](int64_t value, int64_t multiple) {
      return (value + multiple - 1) / multiple * multiple;
    };
    const int64_t np = round_up(total_n, 128);
    const int64_t kp = round_up(inner_k / 16, 4);
    void* scale_sw_data = d.b.Alloc(static_cast<size_t>(np * kp));
    std::shared_ptr<void> scale_sw_owner(
        scale_sw_data, [backend](void* pointer) { backend->Free(pointer); });
    Tensor scale_sw =
        MakeTensor(scale_sw_data, DType::kI8, d.q.device, {np, kp});
    vt::SwizzleBlockscale(d.q, scale_sw, scale_linear.t());

    w.d_qkv_packed = std::move(packed_owner);
    w.d_qkv_scale_sw = std::move(scale_sw_owner);
  }

  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return Nvfp4QkvDev{
      MakeTensor(w.d_qkv_packed.get(), DType::kI8, d.q.device,
                 {total_n, inner_k / 2}),
      MakeTensor(w.d_qkv_scale_sw.get(), DType::kI8, d.q.device,
                 {round_up(total_n, 128), round_up(inner_k / 16, 4)}),
      MergeFullAttnQkvGlobals(q, k, v), q.n, k.n, v.n};
}
#endif  // VT_CUTLASS_NVFP4

// Host reference dequant of an fp4 weight to bf16 [K=in, N=out] (Matmul-B
// layout) — the CPU fallback for the fp4 path (no CPU MatmulNvfp4 kernel). Only
// exercised when a real fp4 checkpoint is run on the host device; the CUDA path
// never dequants. Bit-for-bit vllm::DequantNvfp4ToBf16 + transpose.
std::vector<uint16_t> DequantNvfp4ToBLayout(const Nvfp4Weight& w) {
  const int64_t out_dim = w.n, in_dim = w.k;
  std::vector<uint16_t> oi(static_cast<size_t>(out_dim) * in_dim);
  DequantNvfp4ToBf16(reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
                     reinterpret_cast<const uint8_t*>(w.scale.bytes.data()),
                     w.scale2, out_dim, in_dim, oi.data());
  std::vector<uint16_t> io(static_cast<size_t>(in_dim) * out_dim);
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      io[static_cast<size_t>(c) * out_dim + r] =
          oi[static_cast<size_t>(r) * in_dim + c];
  return io;
}

// The SAME bf16 [K=in, N=out] operand, uploaded ONCE and kept resident on the
// weight (mirror of ResidentNvfp4, same Backend deleter). OPT-IN per weight
// (`keep_dequant_b`, qwen3_5_weights.h): only the output head is worth a lifetime
// bf16 expansion of ~4x its packed bytes.
Tensor ResidentNvfp4DequantB(Dev d, const Nvfp4Weight& w) {
  VT_CHECK(w.keep_dequant_b, "nvfp4: dequant-B residency is opt-in per weight");
  if (!w.d_dequant_b) {
    const std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    const size_t nb = wb.size() * sizeof(uint16_t);
    void* p = d.b.Alloc(nb);
    d.b.Copy(d.q, p, wb.data(), nb);
    Backend* bk = &d.b;
    w.d_dequant_b = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_dequant_b.get(), DType::kBF16, d.q.device, {w.k, w.n});
}

// out[M,N] = x[M,K] @ dequant(w).T — the fallback both device dispatchers take on
// a backend with NO fp4 GEMM (CPU registers only kMatmulNvfp4Fp4; Vulkan/Metal
// neither kMatmulNvfp4 nor the Marlin grouped GEMM). A weight that did not opt in
// keeps the PER-CALL temporary it has always had; caching the whole NVFP4 tower
// would quadruple its steady-state bytes on exactly those backends.
void MatmulNvfp4DequantB(Dev d, Tensor& out, const Tensor& x,
                         const Nvfp4Weight& w) {
  if (w.keep_dequant_b) {
    vt::Matmul(d.q, out, x, ResidentNvfp4DequantB(d, w));
    return;
  }
  const std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
  DBuf dwb(d, DType::kBF16, {w.k, w.n}, wb.data());
  vt::Matmul(d.q, out, x, dwb.t());
}

// y[M,N] f32 = x[M,K] bf16 @ dequant(w).T, w fp4-resident [N=out, K=in]. Drops
// in for MatmulF32 where the weight is NVFP4 (experts/shared/lm_head).
std::vector<float> MatmulNvfp4F32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                  int64_t K, const Nvfp4Weight& w) {
  const int64_t N = w.n;
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kF32, {M, N});
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), dx.t(), dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), dx.t(), dwb.t());
  }
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// y[M,N] bf16 = x[M,K] bf16 @ dequant(w).T, w fp4-resident [N=out, K=in]. Drops
// in for MatmulBf16 (expert down projection).
std::vector<uint16_t> MatmulNvfp4Bf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                      int64_t K, const Nvfp4Weight& w) {
  const int64_t N = w.n;
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kBF16, {M, N});
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), dx.t(), dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), dx.t(), dwb.t());
  }
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// --- Device-resident matmul helpers (M2.5 Phase 1) --------------------------
// Same GEMMs as MatmulF32/MatmulBf16/MatmulNvfp4* but device-in / device-out:
// the input activation is already a device tensor and the result STAYS on the
// device (returned as a DBuf) — NO host Download/Synchronize. This is what lets
// the whole decode step run async-on-stream (the prerequisite for graph
// capture). x is [M,K] bf16 (device); the returned DBuf owns the [M,N] output.

// Both helpers route on the weight's orientation flag: nk=true (raw torch
// Linear [N,K], LoadBf16RawNK) -> vt::MatmulBT, the cuBLASLt TN fast path;
// nk=false (loader-transposed [K,N]) -> row-major vt::Matmul, unchanged.

// --- The model's ONE resolved activation dtype (#2534) -----------------------
//
// The RESOLVER lives in `detail::` (qwen3_5_internal.h) beside `GdnOutDType`
// and for the same reason: a gate that only reads the parser proves nothing
// about what the model calls, and this lever is the denominator of the arm's
// same-binary A/B. This file keeps only the `Dev` adapter.
DType ActDType(Dev d) { return detail::ActDType(d.q.device.type); }

DBuf MatmulF32D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.nk ? w.shape[0] : w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), x, dw);
  else
    vt::Matmul(d.q, dout.t(), x, dw);
  return dout;
}

DBuf MatmulBf16D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.nk ? w.shape[0] : w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, ActDType(d), {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), x, dw);
  else
    vt::Matmul(d.q, dout.t(), x, dw);
  return dout;
}

// T25: When out_proj is kept as K-quant in tiled order (out_proj_tiled), permute
// the gated-norm output from grouped→tiled before the K-quant GEMV. The `nk`
// flag alone is insufficient: gdn_expand_nk also sets nk=true for the bf16
// expanded weight, but that weight has ReorderVCols applied and needs NO
// input permutation. Only the T25 tiled Q5_K path (out_proj_tiled=true) does.
static DBuf GdnOutProjMatmul(Dev d, const GdnLayerWeights& w,
                              const DBuf& gated_bf16,
                              int64_t T, int64_t Hk, int64_t Hv, int64_t Dv) {
  if (w.out_proj_tiled) {
    const int64_t value_dim = Hv * Dv;
    const int64_t rpk = Hk > 0 ? Hv / Hk : 1;
    DBuf permuted(d, DType::kBF16, {T, value_dim});
    vt::PermuteVHeads(d.q, permuted.t(), gated_bf16.t(), T, Hk, rpk, Dv);
    return MatmulBf16D(d, permuted.t(), w.out_proj);
  }
  return MatmulBf16D(d, gated_bf16.t(), w.out_proj);
}

// A tied BF16 lm_head follows torch Linear's model-dtype output, then the
// engine exposes f32 logits to the sampler. Explicit 27B heads retain the
// existing f32-output MatmulF32D path.
DBuf MatmulBf16LogitsF32D(Dev d, const Tensor& x, const OwnedTensor& w) {
  DBuf bf16 = MatmulBf16D(d, x, w);
  DBuf f32(d, DType::kF32, {bf16.t().shape[0], bf16.t().shape[1]});
  vt::CastF32(d.q, f32.t(), bf16.t());
  return f32;
}

const OwnedTensor& DenseLmHead(const Qwen3_5DenseWeights& weights) {
  return weights.tied_lm_head ? weights.embed_tokens : weights.lm_head;
}

// --- FP8 W8A8 dense seam (issue #940) ---------------------------------------
// `ResidentFp8`, `DenseCublasLtFp8Enabled`, `MatmulFp8CutlassD` and
// `MatmulFp8CutlassPreQuantD` USED to be defined right here, in this anonymous
// namespace, which meant a second model could not reach them at all — the exact
// "hand-roll a parallel path" trap AGENTS.md §"Shared seams" forbids. Their ONE
// definition now lives in dense_fp8_gemm.h (the FP8 sibling of
// dense_nvfp4_gemm.h) and the three names below are pure type adapters: this
// file keeps its own anonymous-namespace `Dev`/`DBuf` (the KNOWN DUPLICATION
// dense_nvfp4_gemm.h records), so it instantiates the shared template with THOSE
// types and the generated code is what it was. No logic lives here; a change to
// the seam changes this model's arithmetic, which is what makes the extraction
// provable rather than decorative.
using dense_fp8::DenseCublasLtFp8Enabled;

DBuf MatmulFp8CutlassD(Dev d, const Tensor& x, const Fp8Weight& w, DType out_dtype) {
  return dense_fp8::MatmulFp8CutlassD<DBuf>(d, x, w, out_dtype);
}

DBuf MatmulFp8CutlassPreQuantD(Dev d, const Tensor& a_fp8, const Fp8Weight& w,
                               DType out_dtype) {
  return dense_fp8::MatmulFp8CutlassPreQuantD<DBuf>(d, a_fp8, w, out_dtype);
}

// --- Merged FP8 QKVParallelLinear (VT_FP8_MERGED_QKV, opt-in). The FP8 (W8A8)
// analog of the fp4 MergedQkvCutlassD/ResidentNvfp4Qkv pair above: vLLM loads the
// three logical Q/K/V shards into ONE QKVParallelLinear parameter and runs one
// GEMM before splitting (`qwen3_5.py:279-288`, `linear.py:942-1050`). Our fp8
// path stored per-tensor scales and ran three SEPARATE per-shard GEMMs; here we
// N-concatenate the RAW fp8 bytes into one [Nq+Nk+Nv,K] operand and run ONE fp8
// GEMM. Unlike fp4 (per-block scales that concatenate losslessly), fp8 here is
// PER-TENSOR scaled — each shard has its own folded alpha (= shared input_scale *
// that shard's weight_scale). We therefore run the merged GEMM with alpha=1 (raw
// f32 accumulation) and apply each output column's shard alpha via the resident
// per-column vector (vt::MulColVecF32). Because the three shards share one
// input_scale (guarded), the activation quant is identical across them, so this
// is byte-identical to the three separate MatmulFp8CutlassD/PreQuantD GEMMs when
// the GEMM's per-column accumulation matches (the alpha multiply is the same IEEE
// f32 op the folded-alpha GEMM would apply). Fewer launches + a larger-N GEMM
// fills the SMs better at M=1 (the 35B c1/c2 decode residual).
struct Fp8QkvDev {
  Tensor packed;     // i8 [Nq+Nk+Nv, K] raw e4m3fn (K contiguous)
  Tensor alpha_vec;  // f32 [Nq+Nk+Nv]  per-column folded alpha (input_scale*wscale)
  int64_t qn = 0, kn = 0, vn = 0;
};

// Build (lazily, once) the resident N-concatenated fp8 QKV operand + per-column
// alpha vector. Mirrors ResidentNvfp4Qkv's byte-concatenation (no swizzle: the
// raw fp8 weight is read in [N,K] orientation directly by the fp8 GEMM).
Fp8QkvDev ResidentFp8Qkv(Dev d, const FullAttnLayerWeights& w) {
  const Fp8Weight& q = w.q_proj_fp8;
  const Fp8Weight& k = w.k_proj_fp8;
  const Fp8Weight& v = w.v_proj_fp8;
  VT_CHECK(!q.Empty() && !k.Empty() && !v.Empty(),
           "qwen3_5 packed FP8 QKV: empty logical shard");
  VT_CHECK(q.k == k.k && q.k == v.k,
           "qwen3_5 packed FP8 QKV: logical shard K mismatch");
  const int64_t inner_k = q.k;
  const int64_t total_n = q.n + k.n + v.n;
  const size_t qpb = q.packed.bytes.size();
  const size_t kpb = k.packed.bytes.size();
  const size_t vpb = v.packed.bytes.size();
  VT_CHECK(qpb == static_cast<size_t>(q.n * inner_k) &&
               kpb == static_cast<size_t>(k.n * inner_k) &&
               vpb == static_cast<size_t>(v.n * inner_k),
           "qwen3_5 packed FP8 QKV: packed shard byte mismatch");

  if (!w.d_qkv_fp8_packed || !w.d_qkv_fp8_alpha) {
    VT_CHECK(!w.d_qkv_fp8_packed && !w.d_qkv_fp8_alpha,
             "qwen3_5 packed FP8 QKV: partial resident state");
    Backend* backend = &d.b;
    void* packed_data = d.b.Alloc(qpb + kpb + vpb);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* dst = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, dst, q.packed.bytes.data(), qpb);
    d.b.Copy(d.q, dst + qpb, k.packed.bytes.data(), kpb);
    d.b.Copy(d.q, dst + qpb + kpb, v.packed.bytes.data(), vpb);

    std::vector<float> alpha_host(static_cast<size_t>(total_n));
    std::fill(alpha_host.begin(), alpha_host.begin() + q.n, q.alpha);
    std::fill(alpha_host.begin() + q.n, alpha_host.begin() + q.n + k.n, k.alpha);
    std::fill(alpha_host.begin() + q.n + k.n, alpha_host.end(), v.alpha);
    void* alpha_data = d.b.Alloc(static_cast<size_t>(total_n) * sizeof(float));
    std::shared_ptr<void> alpha_owner(
        alpha_data, [backend](void* pointer) { backend->Free(pointer); });
    d.b.Copy(d.q, alpha_data, alpha_host.data(),
             alpha_host.size() * sizeof(float));

    w.d_qkv_fp8_packed = std::move(packed_owner);
    w.d_qkv_fp8_alpha = std::move(alpha_owner);
  }

  return Fp8QkvDev{
      MakeTensor(w.d_qkv_fp8_packed.get(), DType::kI8, d.q.device,
                 {total_n, inner_k}),
      MakeTensor(w.d_qkv_fp8_alpha.get(), DType::kF32, d.q.device, {total_n}),
      q.n, k.n, v.n};
}

bool MergedFp8QkvEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP8_MERGED_QKV");
    return e != nullptr && e[0] != '0';  // DEFAULT OFF (opt-in) until gated
  }();
  return on;
}

bool MergedFp8QkvEligible(const FullAttnLayerWeights& w, Dev d,
                          bool packed_consumers) {
  const Fp8Weight& q = w.q_proj_fp8;
  const Fp8Weight& k = w.k_proj_fp8;
  const Fp8Weight& v = w.v_proj_fp8;
  // packed_consumers: the merged output exposes row-strided Q/K/V views, which
  // only the fused attn preamble consumes correctly (the split AttnGateSplit path
  // assumes contiguous rows). Shared per-tensor input_scale is required: the
  // merged GEMM quantizes the activation ONCE, so all three shards must read the
  // same activation scale (the real 35B q/k/v do; QKVParallelLinear is one input).
  return packed_consumers && MergedFp8QkvEnabled() &&
         vllm::platforms::GetPlatform(d.q.device.type).supports_fp8() &&
         !q.Empty() && !k.Empty() &&
         !v.Empty() && q.k == k.k && q.k == v.k &&
         q.input_scale == k.input_scale && q.input_scale == v.input_scale;
}

// ONE fp8 GEMM over the N-concatenated QKV operand -> f32 [M,Nq+Nk+Nv]; the
// per-column alpha applies each shard's folded scalar. h_fp8 is the shared
// pre-quantized activation (quantize-once) when supplied, else quantize here
// with the shared input_scale. Same GEMM backend as MatmulFp8CutlassD.
DBuf MergedFp8QkvD(Dev d, const Tensor& x, const Tensor* h_fp8,
                   const FullAttnLayerWeights& w) {
  Fp8QkvDev qkv = ResidentFp8Qkv(d, w);
  const int64_t total_n = qkv.qn + qkv.kn + qkv.vn;
  const int64_t M = x.shape[0];
  DBuf out(d, DType::kF32, {M, total_n});
  const float one = 1.0F;
  const Tensor* a_fp8_p = h_fp8;
  std::optional<DBuf> a_fp8_owner;
  if (a_fp8_p == nullptr) {
    const int64_t K = x.shape[1];
    a_fp8_owner.emplace(d, DType::kI8, std::vector<int64_t>{M, K});
    vt::QuantFp8Static(d.q, a_fp8_owner->t(), x, w.q_proj_fp8.input_scale);
    a_fp8_p = &a_fp8_owner->t();
  }
  // PERF-FP8-ALPHA-FOLD: same seam as the GDN qkvz merge above — the per-column
  // alpha is expressed once and applied in the cuBLASLt epilogue when the arm is
  // enabled and supported, else by the unchanged two-launch form. This site is
  // default OFF (MergedFp8QkvEnabled), so nothing here moves a gate today; it is
  // routed anyway so the seam has ONE per-column-alpha path rather than two, and
  // so enabling the merge later cannot re-add 16 full-tensor passes per step
  // (#402 §3 "D", which is sequenced after this row for exactly that reason).
  if (DenseCublasLtFp8Enabled()) {
    vt::MatmulFp8CublasLtAlphaVec(d.q, out.t(), *a_fp8_p, qkv.packed, qkv.alpha_vec);
  } else {
    vt::MatmulFp8Cutlass(d.q, out.t(), *a_fp8_p, qkv.packed, one);
    vt::MulColVecF32(d.q, out.t(), qkv.alpha_vec);
  }
  return out;
}

// FUSE fp8 RMSNorm -> static quant + quantize-once (35B W8A8): fold the input-
// layernorm (residual-add + gemma RMSNorm) and the shared activation's fp8 quant
// into ONE pass (vt::RmsNormQuantFp8, mirror vLLM Inductor
// fused_add_rms_norm_static_fp8_quant, rms_quant_fusion.py:124), feeding the SINGLE
// fp8 activation to every projection that reads it (attn q/k/v; GDN in_proj_qkv/z)
// via MatmulFp8CutlassPreQuantD — removing the standalone QuantFp8Static pass + its
// bf16 round-trip AND the redundant per-projection re-quant of the same [T,H].
// Bit-identical (bf16-intermediate form); only fires when the shared projections
// carry ONE input_scale (guarded at the RunLayer site — the real 35B q/k/v and
// in_proj_qkv/z do). DEFAULT ON: token-exact (op-level byte-identical + 35B greedy
// 16/16) and a clean same-binary win at the gate shape (in1024/out128 conc32 np192:
// +0.85% total & prefill tok/s, every ON run > every OFF run; nsys: the input-fed
// QuantFp8Static instances drop 130->40 per step, folded into RmsNormQuantFp8). The
// 27B (no fp8 weights) never fires it. VT_FUSE_RMSNORM_FP8QUANT=0 opts out for an A/B.
bool FuseRmsNormFp8QuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_RMSNORM_FP8QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// KERNEL-FUSION-FRAMEWORK W0 — route the plain add+residual+RMSNorm site through
// the declared vt::FusedChain(kFusedAddRmsNorm) recipe seam instead of the direct
// vt::RmsNorm(residual) hand-call. Behavior-preserving by construction: the recipe
// encodes the exact op order (res += hidden; out = gemma-RMSNorm(res)) and the
// default Tier-0 composite dispatches to the same vt::RmsNorm(residual) primitive,
// so the FusedChain path is BIT-IDENTICAL to the prior hand-call (proven byte-exact
// in tests/vt/test_ops_fused_chain.cpp). Default ON to exercise the production seam;
// VT_FUSED_CHAIN_ADOPT=0 restores the exact prior hand-call as a same-binary
// rollback. This is structural plumbing (the fusion framework's first real adoption),
// NOT a perf lever — perf-neutral by construction.
bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// TRUE W4A4 toggle (A/B; default ON — mirrors vLLM, which runs this checkpoint as
// use_a16=False true-W4A4, notes §7.1). Set VT_W4A4_TRUE=0 to fall back to the
// W4A16 6a fast path (bf16 activations) for a throughput A/B. Only affects
// weights that carry the activation-quant globals (27B; alpha>0) — the 35B
// (alpha==0) is untouched regardless.
bool TrueW4A4Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_W4A4_TRUE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// FUSE silu-mul with the down_proj activation quant (mirror vllm
// ActivationQuantFusionPass / silu_and_mul_nvfp4_quant) — one kernel, no bf16
// intermediate. Default ON (opt-out VT_FUSE_SILU_QUANT=0): MEASURED +2.4% prefill-
// heavy, token-exact (the earlier "-2%" was the stale software quant ladder, not the
// fusion). Only the 27B true-W4A4 dense MLP is affected (down_proj quantizes its act).
bool FuseSiluQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_SILU_QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// FUSE the full-attention sigmoid output gate with the o_proj activation quant
// (mirror vLLM's Inductor triton_poi_fused_mul_scaled_fp4_quant_sigmoid) — one
// kernel over attn*sigmoid(gate), no bf16 intermediate. Only the 27B true-W4A4
// full-attention o_proj quantizes its activation, so only it is affected; the 35B
// W4A16-Marlin (and any fp8/bf16 o_proj) reads bf16 activations and keeps the
// standalone SigmoidGateBf16. sigmoid·mul is elementwise/non-reducing → the fused
// producer is bit-identical to SigmoidGateBf16 -> ScaledFp4Quant (byte-exact op
// test + 27B 235/235 both arms prove it). DEFAULT OFF (opt-in
// VT_FUSE_SIGMOID_QUANT=1): the in-situ 27B c1/c2 TTFT A/B measured NEUTRAL (the
// o_proj is a small slice of 27B prefill, dominated by GDN + MoE + the QKV/gate/up
// GEMMs) — token-exact but not measurably faster, so it ships opt-in per the "flip
// only on a measured win" rule (mirror of the perf-neutral fp8-merged-QKV
// launch-fusion disposition).
// [[maybe_unused]]: the sole caller (SigmoidGateOProjD) is under #ifdef
// VT_CUTLASS_NVFP4, so a CPU-only build compiles this out and would otherwise
// trip -Werror=unused-function on this anonymous-namespace function.
[[maybe_unused]] bool FuseSigmoidGateQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_SIGMOID_QUANT");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// FUSE the full-attention preamble (prefill-gap-scan dominant lever): the 4-5
// separate f32 kernels before the attention kernel (AttnGateSplit + q-RMSNorm +
// k-RMSNorm + partial NeoX RoPE, the last recomputing cos/sin transcendentals in
// DOUBLE per element/head/layer) collapse into ONE launch that reads a precomputed
// cos_sin cache — mirror of vLLM's fused_qk_rmsnorm_rope (fla
// fused_qk_norm_rope.py:95-102, zero in-kernel transcendentals). Measured 27B
// site (2026-07-10): unfused preamble+rope 1.36 vs vLLM 0.27 us/tok/layer.
//
// DEFAULT ON, ALL ARCHES (2026-07-18, MIRROR vLLM). vLLM dispatches the fused
// qk-norm-rope-gate preamble for EVERY Qwen3.5 full-attn layer (attn_output_gate
// && neox && cuda && text_only — true for BOTH the 27B and the 35B; the fp4/fp8
// split was OUR heuristic, not vLLM's). On the 35B (fp8 attn) this pairs with the
// FA-2 prefill (bf16 q feeds flash_fwd_splitkv) — the token-exact gate
// test_qwen36_paged_engine holds 315/315 with the full default set (measured
// 2026-07-18, CLAIM-35B-FA2-FLIP-1), and the 27B (fp4) stays 235/235. The old
// "35B diverges within 16 tokens" (2026-07-09) is STALE (grounded in spec
// qwen36-35b-fa2-prefill-oracle-2026-07-18: the stored oracle IS the graphed
// vLLM production stream and FA2 bf16-q is the MOST vLLM-faithful path).
// NOTE: the spec's "round normed q/k to bf16 before RoPE" tighten
// (fused_qk_norm_rope.py:67) is op-level bit-identical but flips the 27B tok6
// near-tie in combination (RMSNorm-saga), so the preamble ships UNTIGHTENED —
// both arches are token-exact on their graphed oracles either way (see
// AttnQkNormRopeGateKernel NOTE in cuda_ops.cu). Unset env => ON;
// VT_FUSE_ATTN_PREAMBLE=1/0 force-overrides (same-binary A/B rollback).
bool FuseAttnPreambleOn(bool /*fp4_attn*/) {
  static const char* e = std::getenv("VT_FUSE_ATTN_PREAMBLE");
  if (e != nullptr) return e[0] == '1';
  return true;
}

// FA-2 PREFILL (the last measured 27B prefill gap): route the full-attn PREFILL
// segment through the vendored FlashAttention-2 flash_fwd_splitkv kernel (the
// exact kernel vLLM runs on GB10; vllm-project/flash-attention @ 2c839c33) by
// making the fused preamble emit bf16 q/k and the attention output bf16 — the
// natively-bf16 combo the FA-2 dispatch gate (cuda_paged_attn.cu) requires, with
// ZERO cast kernels (the earlier f32-glued wiring measured negative; see
// cuda_flash_attn_fa2.cu header). Measured 27B site (2026-07-10): our WMMA
// prefill attention 1.37 vs vLLM's FA-2 0.25 us/tok/layer (~18 us/tok e2e).
// bf16-q/out is NOT bit-identical to the f32-q WMMA path (FA-2 rounds q to bf16
// and accumulates in its own order) but IS vLLM-faithful (vLLM's whole attn
// path is bf16) — validated by the token-exact greedy gate (2026-07-10: 27B
// engine gate PASS ON and OFF, same tie branch; chunked==one-shot holds).
// MEASURED (2026-07-10, GB10, same binary): kernel 3.68x faster (475.3ms ->
// 129.2ms per np16xin1024 profile; ~1.81 -> 0.49 us/tok/layer), 27B e2e
// conc16/np96 752.8 -> 761.6 (+1.2%), conc32/np192 1045.6 -> 1050.4 (+0.5%),
// TTFT -3.4/-3.6%, putting the 27B at/above fresh graphed-vLLM denominators —
// hence DEFAULT ON when compiled (VLLM_CPP_FLASH_ATTN); VT_FA2_PREFILL=0
// restores the WMMA prefill for a same-binary A/B. Decode has its own bounded
// FA2 selection below and remains independent of this toggle.
// Without VLLM_CPP_FLASH_ATTN the env is ignored and the WMMA path runs.
bool Fa2PrefillOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_PREFILL");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// FA2 PURE DECODE (W3-G): vLLM v0.25 and its pinned FA2 dependency reinterpret
// GQA groups as query rows and run the paged split-KV main+combine path. The
// first default-on slice is exactly the Qwen3.6-27B Hq/Hkv=24/4 BF16/D256
// topology. VT_FA2_DECODE=0 restores the existing F32 paged kernel in the same
// binary. Read fresh because component/tests flip the arm in-process.
bool Fa2DecodeOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_DECODE");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// SPEC-DFLASH2 W10 repair (#1865): the model-side half of the spec-as-decode
// toggle. MUST match cuda_paged_attn.cu Fa2SpecDecodeEnabled() — the CUDA
// admission requires a bf16 query, so the model side must select bf16 for a
// classified verify under exactly the switch the admission reads, or the lane
// dies at the dtype conjunct with every counter green (the #1865 failure:
// VT_FA2_SPEC_DECODE flips were a no-op because the verify's dtype rode
// VT_FA2_PREFILL instead). Read fresh so in-process tests can flip it.
bool Fa2SpecDecodeOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_SPEC_DECODE");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// 35B ratio-8 (Hq/Hkv=16/2) hd-256 FA2 split-KV decode arm
// (CLAIM-35B-FA2-DECODE-1). The old ratio-8 decode ran the fused GQA kernel with
// grid=(num_reqs,num_kv_heads) = 2 blocks/step at c1 (near-zero GB10 occupancy);
// the vendored split-KV kernel splits the KV dimension so the grid fills the
// machine. Independent of VT_FA2_DECODE (the 27B ratio-6 arm) so the 35B default
// and same-binary fallback are controllable on their own. MUST mirror
// cuda_paged_attn.cu Fa2Decode35BEnabled().
bool Fa2Decode35BOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_DECODE_35B");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// Qwen3.5-4B ratio-4 (Hq/Hkv=16/4), hd-256 arm of the same FA2 split-KV
// launcher. Kept independently controllable so the established 27B/35B
// dispatch and their rollback flags remain unchanged.
bool Fa2Decode4BOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_DECODE_4B");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// QUANTIZE-ONCE: q/k/v (and gate/up) share their input activation AND their on-disk
// input_global_scale (verified: 27B layer-3 q/k/v all 28.75; gate/up 812/476), so we
// can ScaledFp4Quant the shared activation ONCE and feed each projection's fp4xfp4
// GEMM the same packed/scale — removing the 2-3x redundant per-projection quant of
// [T,H] (mirrors vLLM's fused qkv/gate_up MergedColumnParallelLinear = one quant).
// Bit-identical only when the shared input_global_scale_inv match (guarded at the
// call site). DEFAULT ON (bit-identical + measured +0.3-0.5% on the 27B prefill,
// mirrors vLLM's fused qkv/gate_up MergedColumnParallelLinear); VT_FUSE_QUANT_ONCE=0
// restores per-projection quant for an A/B.
bool FuseQuantOnceEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_QUANT_ONCE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// BF16 GEMM OUTPUTS (prefill-lever-scan rank 1-2): the fp4 gate/up (and later q/k/v)
// projections currently output f32 (MatmulNvfp4F32D); vLLM keeps them bf16 (model
// dtype). Emitting bf16 halves the GEMM output write + the downstream glue read
// traffic (MoeSiluMul) on the memory-bound prefill. NOT bit-identical (gate/up are
// bf16-rounded before silu) but MORE faithful to vLLM's bf16 model dtype — validated
// by the token-exact gate. DEFAULT ON (measured 27B +5.4% standard / +12.5% prefill-
// heavy, gate 9/9; matches vLLM bf16 dtype); VT_BF16_GEMM_OUT=0 restores f32 for an A/B.
bool Bf16GemmOutEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_BF16_GEMM_OUT");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

#ifdef VT_CUTLASS_NVFP4
// cutlass sm120a fp4xfp4 GEMM path toggle (DEFAULT ON when compiled with
// VT_CUTLASS_NVFP4 — mirrors how the validated 35B fp8/Marlin defaults were
// flipped on: absence of the env selects cutlass; VT_NVFP4_CUTLASS=0 is the
// opt-out to the emulation-grade path). Routes the 27B true-W4A4 projections
// through vt::MatmulNvfp4Cutlass (the lifted vLLM near-peak kernel) instead of
// our emulation-grade MatmulNvfp4Fp4. Mirrors vLLM, which auto-selects the
// cutlass/flashinfer sm120a fp4×fp4 kernel for CT-W4A4
// (compressed_tensors_w4a4_nvfp4.py; notes §7.1). MEASURED same-binary A/B
// 37.21→124.33 tok/s (3.34×), gap vs vLLM 11.2×→3.19× (parity-ledger row 66).
//
// 27B-ONLY BY CONSTRUCTION: NvfpCutlassEnabled() is reached ONLY from
// MatmulNvfp4Fp4D, itself guarded by `w.IsTrueW4A4()` (alpha>0 — the 27B W4A4
// alone). The 35B is W4A16 (alpha==0, IsTrueW4A4()==false) and never enters this
// path; its dense/MoE NVFP4 run the Marlin W4A16 branch. So this default-flip
// cannot affect the 35B.
//
// THROUGHPUT lever, near-emulation correctness. Swapping the GEMM to real cutlass
// does NOT recover vLLM's native flashinfer stream (198) — the 27B still yields
// tok6=271; tok6 is a razor near-tie tipped by the aggregate non-fp4 forward
// numerics, not the fp4 GEMM (measured 2026-07-05). MEASURED 2026-07-06: cutlass
// is ~0.19% off the emulation-grade MatmulNvfp4Fp4 (test_ops_nvfp4_fp4, NOT
// bit-exact), so on the near-tie-dense 27B greedy gate it reproduces emulation on
// the semantic tokens (0-7) then DETERMINISTICALLY flips a later whitespace
// near-tie at tok8 (271 "\n\n" -> 198 "\n"). Output stays coherent; the token-exact
// correctness gate therefore pins the emulation-grade reference (VT_NVFP4_CUTLASS=0)
// while this default carries the throughput win — see
// tests/parity/test_qwen27_paged_engine.cpp. Only meaningful when the cutlass TU
// was compiled (VT_CUTLASS_NVFP4).
bool NvfpCutlassEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_CUTLASS");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// vLLM v0.25's normal and fused NVFP4 quant producers write activation block
// scales directly in the CUTLASS tensor-core atom layout. Default to that
// topology only on the compiled CUDA CUTLASS W4A4 path; the opt-out preserves
// the exact former linear producer -> standalone SwizzleBlockscale sequence for
// same-binary A/B and rollback.
bool DirectFp4ScaleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_DIRECT_SF");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

bool DirectFp4ScaleEligible(Dev d) {
  return DirectFp4ScaleEnabled() && NvfpCutlassEnabled() &&
         vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported();
}

// vLLM's CompressedTensorsW4A4Nvfp4 retains alpha as a non-trainable device
// parameter and FlashInfer passes that pointer directly into CUTLASS. Default to
// the same ownership; the opt-out is the exact former per-GEMM host-scalar
// staging path for same-binary component measurement and rollback.
bool DeviceFp4AlphaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_DEVICE_ALPHA");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

#ifdef VT_CUTLASS_NVFP4
Tensor ResidentDeviceAlpha(Dev d, const float* host_alpha,
                           std::shared_ptr<void>& owner,
                           const char* context) {
  VT_CHECK(host_alpha != nullptr && *host_alpha > 0.0F, context);
  if (!owner) {
    Backend* backend = &d.b;
    void* data = d.b.Alloc(sizeof(float));
    std::shared_ptr<void> candidate(
        data, [backend](void* pointer) { backend->Free(pointer); });
    d.b.Copy(d.q, data, host_alpha, sizeof(float));
    owner = std::move(candidate);
  }
  return MakeTensor(owner.get(), DType::kF32, d.q.device, {1});
}

Tensor ResidentNvfp4Alpha(Dev d, const Nvfp4Weight& w) {
  VT_CHECK(w.IsTrueW4A4(), "qwen3_5 NVFP4 device alpha: true-W4A4 required");
  VT_CHECK(w.d_packed && w.d_scale && w.d_scale_sw,
           "qwen3_5 NVFP4 device alpha: incomplete weight resident state");
  return ResidentDeviceAlpha(d, &w.alpha, w.d_alpha,
                             "qwen3_5 NVFP4 device alpha: invalid scalar");
}

Tensor ResidentNvfp4GateUpAlpha(Dev d, const DenseMlpWeights& w,
                                float alpha) {
  VT_CHECK(w.d_gate_up_packed && w.d_gate_up_scale_sw,
           "qwen3_5 dense merged gate_up alpha: incomplete resident state");
  if (!w.d_gate_up_alpha) {
    w.gate_up_alpha = alpha;
  } else {
    VT_CHECK(w.gate_up_alpha == alpha,
             "qwen3_5 dense merged gate_up alpha changed after upload");
  }
  return ResidentDeviceAlpha(
      d, &w.gate_up_alpha, w.d_gate_up_alpha,
      "qwen3_5 dense merged gate_up alpha: invalid scalar");
}

Tensor ResidentNvfp4QkvAlpha(Dev d, const FullAttnLayerWeights& w,
                             float alpha) {
  VT_CHECK(w.d_qkv_packed && w.d_qkv_scale_sw,
           "qwen3_5 packed QKV alpha: incomplete resident state");
  if (!w.d_qkv_alpha) {
    w.qkv_alpha = alpha;
  } else {
    VT_CHECK(w.qkv_alpha == alpha,
             "qwen3_5 packed QKV alpha changed after upload");
  }
  return ResidentDeviceAlpha(d, &w.qkv_alpha, w.d_qkv_alpha,
                             "qwen3_5 packed QKV alpha: invalid scalar");
}

void MatmulNvfp4CutlassModel(Dev d, Tensor& out,
                             const Tensor& a_packed,
                             const Tensor& a_sf_sw,
                             const Tensor& b_packed,
                             const Tensor& b_sf_sw, float alpha_host,
                             const Tensor& alpha_device) {
  if (alpha_device.data != nullptr) {
    vt::MatmulNvfp4Cutlass(d.q, out, a_packed, a_sf_sw, b_packed,
                           b_sf_sw, alpha_device);
  } else {
    vt::MatmulNvfp4Cutlass(d.q, out, a_packed, a_sf_sw, b_packed,
                           b_sf_sw, alpha_host);
  }
}
#endif

// SWIZZLE-ONCE dedup (prefill-gap-scan quant-hw-swizzle lever). The fused qkv/
// gate-up projections share ONE ScaledFp4Quant activation + its LINEAR fp8 block
// scale, but each MatmulNvfp4Fp4DirectD re-runs the identical internal
// SwizzleBlockscale on that SAME shared scale — 3x for a fused qkv, 2x for gate/up
// (the nsys "2,856 SwizzleBlockscaleKernel launches" per short prefill). When ON we
// swizzle the shared activation SF exactly ONCE per fuse-site and pass the already-
// swizzled SF into each projection GEMM (skipping its internal re-swizzle), removing
// the redundant per-projection re-swizzles. BIT-IDENTICAL: SwizzleBlockscale is a
// pure deterministic reorder of the linear SF, so one swizzled buffer reused across
// the projections equals each projection swizzling independently. Default OFF until
// GPU-gated (token-exact vs OFF by construction); VT_SWIZZLE_IN_QUANT=1 enables.
bool SwizzleInQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_SWIZZLE_IN_QUANT");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Dense gate/up topology mirror. The full-tactic fallback must remain the
// immutable W1 arm, so VT_FP4_FULL_TACTICS=0 also disables this W2 model-side
// adaptation. VT_FP4_MERGED_GATE_UP=0 isolates the split W2 arm while retaining
// the same 32 raw tactics.
bool MergedGateUpEnabled() {
  static const bool on = [] {
    const char* full_tactics = std::getenv("VT_FP4_FULL_TACTICS");
    if (full_tactics != nullptr && full_tactics[0] == '0') return false;
    const char* merged = std::getenv("VT_FP4_MERGED_GATE_UP");
    return merged == nullptr || merged[0] != '0';
  }();
  return on;
}

// vLLM's QKVParallelLinear is one physical projection even at TP=1. Keep the
// W1 tactic fallback independent and expose a dedicated same-binary W3-D arm.
bool MergedQkvEnabled() {
  static const bool on = [] {
    const char* full_tactics = std::getenv("VT_FP4_FULL_TACTICS");
    if (full_tactics != nullptr && full_tactics[0] == '0') return false;
    const char* merged = std::getenv("VT_FP4_MERGED_QKV");
    return merged == nullptr || merged[0] != '0';
  }();
  return on;
}

// vLLM's ActivationQuantFusionPass consumes the merged BF16 [M,2I] gate_up
// result directly and emits the down-projection NVFP4 operands in one kernel.
// Keep a dedicated same-binary fallback so this W2 sub-iteration can be timed
// independently from the already-gated merged topology and tactic family.
bool MergedSiluQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_MERGED_SILU_QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

bool MergedGateUpEligible(const DenseMlpWeights& w, Dev d) {
  const Nvfp4Weight& gate = w.gate_proj_fp4;
  const Nvfp4Weight& up = w.up_proj_fp4;
  return MergedGateUpEnabled() && NvfpCutlassEnabled() &&
         Bf16GemmOutEnabled() && vt::OpRegistered(vt::OpId::kMatmulNvfp4Cutlass, d.q.device.type) &&
         !gate.Empty() && !up.Empty() && gate.IsTrueW4A4() &&
         up.IsTrueW4A4() && TrueW4A4Enabled() && gate.n == up.n &&
         gate.k == up.k && gate.weight_global_scale_inv > 0.0F &&
         up.weight_global_scale_inv > 0.0F;
}

bool MergedQkvEligible(const FullAttnLayerWeights& w, Dev d,
                       bool packed_consumers) {
  const Nvfp4Weight& q = w.q_proj_fp4;
  const Nvfp4Weight& k = w.k_proj_fp4;
  const Nvfp4Weight& v = w.v_proj_fp4;
  // Packed output views are row-strided. The fused preamble consumes Q/K with
  // their real row strides, while Attention/ReshapeAndCache consume the V view.
  // If the preamble is explicitly disabled, retain the fully contiguous split
  // reference rather than materializing split-copy kernels.
  return packed_consumers && MergedQkvEnabled() &&
         NvfpCutlassEnabled() && Bf16GemmOutEnabled() &&
         vt::OpRegistered(vt::OpId::kMatmulNvfp4Cutlass, d.q.device.type) && !q.Empty() && !k.Empty() &&
         !v.Empty() && q.IsTrueW4A4() && k.IsTrueW4A4() &&
         v.IsTrueW4A4() && TrueW4A4Enabled() && q.k == k.k && q.k == v.k &&
         q.weight_global_scale_inv > 0.0F &&
         k.weight_global_scale_inv > 0.0F &&
         v.weight_global_scale_inv > 0.0F;
}

// One CT-W4A4 gate_up projection, matching Qwen2MoeMLP.forward and the fused
// scale processing in CompressedTensorsW4A4Fp4. This is deliberately a BF16
// result: vLLM's model dtype is BF16 and SiluAndMul consumes that one [M,2I]
// buffer. The split diagnostic remains in DenseMlpBlock below.
DBuf MergedGateUpCutlassD(Dev d, const Tensor& x, const DenseMlpWeights& w) {
  const int64_t m = x.shape[0];
  const int64_t k = x.shape[1];
  const int64_t n = w.gate_proj_fp4.n;
  Nvfp4GateUpDev gate_up = ResidentNvfp4GateUp(d, w);

  DBuf a_packed(d, DType::kI8, {m, k / 2});
  const bool direct_scale = DirectFp4ScaleEligible(d);
  DBuf a_scale(d, DType::kI8,
               direct_scale ? CutlassFp4ScaleShape(m, k)
                            : std::vector<int64_t>{m, k / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                     gate_up.globals.input_global_scale_inv,
                     direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                                  : vt::Fp4ScaleLayout::kLinear);
  std::optional<DBuf> composed_scale;
  const Tensor* scale_for_gemm = &a_scale.t();
  if (!direct_scale) {
    composed_scale.emplace(d, DType::kI8, CutlassFp4ScaleShape(m, k));
    vt::SwizzleBlockscale(d.q, composed_scale->t(), a_scale.t());
    scale_for_gemm = &composed_scale->t();
  }

  DBuf out(d, DType::kBF16, {m, 2 * n});
  Tensor alpha_device;
  if (DeviceFp4AlphaEnabled()) {
    alpha_device =
        ResidentNvfp4GateUpAlpha(d, w, gate_up.globals.alpha);
  }
  MatmulNvfp4CutlassModel(d, out.t(), a_packed.t(), *scale_for_gemm,
                          gate_up.packed, gate_up.scale_sw,
                          gate_up.globals.alpha, alpha_device);
  return out;
}

// One CT-W4A4 QKVParallelLinear. The result is the standard row-major
// [M,Q+K+V] tensor; consumers use row-strided, inner-contiguous logical views
// exactly like torch.split on vLLM's packed result.
DBuf MergedQkvCutlassD(Dev d, const Tensor& x,
                       const FullAttnLayerWeights& w) {
  const int64_t m = x.shape[0];
  const int64_t inner_k = x.shape[1];
  Nvfp4QkvDev qkv = ResidentNvfp4Qkv(d, w);
  const int64_t total_n = qkv.qn + qkv.kn + qkv.vn;

  DBuf a_packed(d, DType::kI8, {m, inner_k / 2});
  const bool direct_scale = DirectFp4ScaleEligible(d);
  DBuf a_scale(d, DType::kI8,
               direct_scale ? CutlassFp4ScaleShape(m, inner_k)
                            : std::vector<int64_t>{m, inner_k / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                     qkv.globals.input_global_scale_inv,
                     direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                                  : vt::Fp4ScaleLayout::kLinear);
  std::optional<DBuf> composed_scale;
  const Tensor* scale_for_gemm = &a_scale.t();
  if (!direct_scale) {
    composed_scale.emplace(d, DType::kI8,
                           CutlassFp4ScaleShape(m, inner_k));
    vt::SwizzleBlockscale(d.q, composed_scale->t(), a_scale.t());
    scale_for_gemm = &composed_scale->t();
  }

  DBuf out(d, DType::kBF16, {m, total_n});
  Tensor alpha_device;
  if (DeviceFp4AlphaEnabled()) {
    alpha_device = ResidentNvfp4QkvAlpha(d, w, qkv.globals.alpha);
  }
  MatmulNvfp4CutlassModel(d, out.t(), a_packed.t(), *scale_for_gemm,
                          qkv.packed, qkv.scale_sw, qkv.globals.alpha,
                          alpha_device);
  return out;
}
#endif

// TRUE W4A4 (fp4 activations x fp4 weights) device GEMM — the 27B path (notes §7).
// ScaledFp4Quant(x) -> per-token fp4 activations + fp8 block scales, then the
// fp4xfp4 GEMM with the folded alpha (= vllm cutlass_scaled_fp4_mm_sm120a). CUDA
// only; the CPU fp4 path uses the bf16-dequant fallback in the callers. out_dtype
// f32 or bf16.
// The fp4xfp4 GEMM from PRE-QUANTIZED activations (a_packed [M,K/2] + a_scale
// [M,K/16], the ScaledFp4Quant outputs). Factored out of MatmulNvfp4Fp4D so a
// fused producer (SiluMulFp4Quant / RmsNormResFp4Quant) can feed the GEMM directly
// without the bf16 intermediate. out_dtype f32 or bf16.
// `a_sf_sw_pre` may be produced directly by the v0.25-style quant kernel or by
// the older VT_SWIZZLE_IN_QUANT dedup. In either case it is already in CUTLASS
// layout and bypasses the standalone swizzle. When null, `a_scale` is linear
// and the former composition remains byte-identical.
DBuf MatmulNvfp4Fp4DirectD(Dev d, const Tensor& a_packed, const Tensor& a_scale,
                           const Nvfp4Weight& w, DType out_dtype,
                           [[maybe_unused]] const Tensor* a_sf_sw_pre = nullptr) {
  const int64_t M = a_packed.shape[0], N = w.n;
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  DBuf dout(d, out_dtype, {M, N});
#ifdef VT_CUTLASS_NVFP4
  if (NvfpCutlassEnabled()) {
    // Swizzle the per-token activation scale into the cutlass atom layout (the
    // weight scale is swizzled once, cached). Then the lifted sm120a fp4xfp4 GEMM.
    const int64_t K = a_packed.shape[1] * 2;
    auto round_up = [](int64_t v, int64_t y) { return (v + y - 1) / y * y; };
    const int64_t Mp = round_up(M, 128), Kp = round_up(K / 16, 4);
    Tensor b_sf_sw = ResidentNvfp4ScaleSwizzled(d, w);
    Tensor alpha_device;
    if (DeviceFp4AlphaEnabled()) {
      alpha_device = ResidentNvfp4Alpha(d, w);
    }
    // SWIZZLE-ONCE dedup: the shared activation SF was already swizzled ONCE by the
    // fused caller — reuse it (bit-identical to re-swizzling here) and skip our
    // internal SwizzleBlockscale for this projection.
    if (a_sf_sw_pre != nullptr) {
      MatmulNvfp4CutlassModel(d, dout.t(), a_packed, *a_sf_sw_pre,
                              dw.packed, b_sf_sw, w.alpha, alpha_device);
      return dout;
    }
    DBuf a_sf_sw(d, DType::kI8, {Mp, Kp});  // SwizzleBlockscale zero-fills padding
    vt::SwizzleBlockscale(d.q, a_sf_sw.t(), a_scale);
    MatmulNvfp4CutlassModel(d, dout.t(), a_packed, a_sf_sw.t(),
                            dw.packed, b_sf_sw, w.alpha, alpha_device);
    return dout;
  }
#endif
  vt::MatmulNvfp4Fp4(d.q, dout.t(), a_packed, a_scale, dw.packed, dw.scale, w.alpha);
  return dout;
}

#ifdef VT_CUTLASS_NVFP4
// SWIZZLE-ONCE dedup helper (VT_SWIZZLE_IN_QUANT). Swizzle the shared fused-site
// activation block scale [M, groups] into the cutlass atom layout [round_up(M,128),
// round_up(groups,4)] exactly ONCE; the returned buffer is fed to each projection
// GEMM via MatmulNvfp4Fp4DirectD's a_sf_sw_pre. Shape/bytes are identical to the
// buffer MatmulNvfp4Fp4DirectD would build internally (M=a_packed.shape[0],
// groups=K/16), so reuse is bit-identical to per-projection swizzling.
DBuf SwizzleActScaleOnce(Dev d, const Tensor& a_scale) {
  const int64_t M = a_scale.shape[0], groups = a_scale.shape[1];
  auto round_up = [](int64_t v, int64_t y) { return (v + y - 1) / y * y; };
  DBuf a_sf_sw(d, DType::kI8, {round_up(M, 128), round_up(groups, 4)});
  vt::SwizzleBlockscale(d.q, a_sf_sw.t(), a_scale);
  return a_sf_sw;
}
#endif

DBuf MatmulNvfp4Fp4D(Dev d, const Tensor& x, const Nvfp4Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1];
  DBuf a_packed(d, DType::kI8, {M, K / 2});
#ifdef VT_CUTLASS_NVFP4
  if (DirectFp4ScaleEligible(d)) {
    DBuf a_scale(d, DType::kI8, CutlassFp4ScaleShape(M, K));
    vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                       w.input_global_scale_inv,
                       vt::Fp4ScaleLayout::kCutlassSwizzled);
    return MatmulNvfp4Fp4DirectD(d, a_packed.t(), a_scale.t(), w,
                                 out_dtype, &a_scale.t());
  }
#endif
  DBuf a_scale(d, DType::kI8, {M, K / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x, w.input_global_scale_inv);
  return MatmulNvfp4Fp4DirectD(d, a_packed.t(), a_scale.t(), w, out_dtype);
}

// Defined after the W4A16 Marlin selection helpers below; the split QKV
// diagnostic still uses their normal model-side dispatch.
DBuf MatmulNvfp4F32D(Dev d, const Tensor& x, const Nvfp4Weight& w);
DBuf MatmulNvfp4Bf16D(Dev d, const Tensor& x, const Nvfp4Weight& w);

bool FuseSigmoidGateQuantEnabled();

// Full-attention sigmoid output gate (attn*sigmoid(gate)) folded into the o_proj.
// attn2d/gate2d are [T, Hq*Dh] (attn f32 or bf16; gate f32). When the o_proj is
// true-W4A4 fp4 (the 27B path) on CUDA, FUSE the gate into the fp4 activation
// quant — one kernel, no bf16 gated intermediate — mirroring vLLM's Inductor
// triton_poi_fused_mul_scaled_fp4_quant_sigmoid and our SiluMulFp4Quant precedent
// (bit-identical to SigmoidGateBf16 -> MatmulNvfp4Fp4D). Every other o_proj dtype
// (35B W4A16-Marlin fp4, fp8, bf16) reads a bf16 activation, so keep the standalone
// SigmoidGateBf16 + the three-way GEMM there.
DBuf SigmoidGateOProjD(Dev d, const Tensor& attn2d, const Tensor& gate2d,
                       const FullAttnLayerWeights& w, bool fp4) {
  const int64_t T = attn2d.shape[0], K = attn2d.shape[1];
#ifdef VT_CUTLASS_NVFP4
  if (fp4 && w.o_proj_fp8.Empty() && FuseSigmoidGateQuantEnabled() &&
      vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() && !w.o_proj_fp4.Empty() &&
      w.o_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
    DBuf ap(d, DType::kI8, {T, K / 2});
    const bool direct_scale = DirectFp4ScaleEligible(d);
    DBuf as(d, DType::kI8,
            direct_scale ? CutlassFp4ScaleShape(T, K)
                         : std::vector<int64_t>{T, K / 16});
    const vt::Fp4ScaleLayout lay =
        direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled : vt::Fp4ScaleLayout::kLinear;
    // KERNEL-FUSION-FRAMEWORK W2 — route attn·sigmoid(gate) + NVFP4 quant through
    // vt::FusedChain(kSigmoidGateFp4Quant); its fast_op binds the SAME bespoke
    // SigmoidGateFp4Quant kernel (byte-identical + perf-neutral). VT_FUSED_CHAIN_ADOPT=0
    // restores the direct hand-call.
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, vt::kSigmoidGateFp4Quant, ap.t(), as.t(), attn2d, gate2d,
                     w.o_proj_fp4.input_global_scale_inv, lay);
    } else {
      vt::SigmoidGateFp4Quant(d.q, ap.t(), as.t(), attn2d, gate2d,
                              w.o_proj_fp4.input_global_scale_inv, lay);
    }
    return MatmulNvfp4Fp4DirectD(d, ap.t(), as.t(), w.o_proj_fp4, DType::kBF16,
                                 direct_scale ? &as.t() : nullptr);
  }
#endif
  // BF16 here is `vt::SigmoidGateBf16`'s own contract ("out must be bf16",
  // ops.cpp:5105), NOT the trunk's activation dtype, so it does NOT follow
  // `ActDType`. Every other consumer this row retyped takes `IsOutFloat`, which
  // admits f32; this one does not, and routing an f32 buffer into it aborted 9
  // cases of test_qwen27_paged_forward under VT_ACT_F32=1. Widening the trunk
  // therefore leaves ONE bf16 rounding on the gated-attention output, which the
  // spec records as owed: closing it needs an f32-capable gate kernel, not a
  // dtype change here.
  DBuf gated(d, DType::kBF16, {T, K});
  vt::SigmoidGateBf16(d.q, gated.t(), attn2d, gate2d);
  // MODEL-QWEN35-EXL3 (#2495 item 3): o_proj is a single projection in every
  // arm, so the EXL3 form differs from the bf16 one only in the kernel the
  // shared linear seam binds. The gated bf16 activation IS the input the
  // trellis GEMM stages to fp16.
  if (w.IsExl3()) {
    return dense_exl3::Linear(d, gated.t(), w.o_proj, w.o_proj_exl3,
                              DType::kBF16);
  }
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4): exclusive and first, out_dtype bf16 --
  // the same dtype every other arm of this o_proj returns.
  if (!w.o_proj_fp8_block.Empty()) {
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, gated.t(),
                                                        w.o_proj_fp8_block,
                                                        DType::kBF16);
  }
  return !w.o_proj_fp8.Empty() ? MatmulFp8CutlassD(d, gated.t(), w.o_proj_fp8, DType::kBF16)
         : fp4 ? MatmulNvfp4Bf16D(d, gated.t(), w.o_proj_fp4)
               : MatmulBf16D(d, gated.t(), w.o_proj);  // [T,H]
}

// Owners plus logical views for full-attention Q/K/V. The packed path owns one
// [T,Q+K+V] allocation and exposes row-strided inner-contiguous views; the
// diagnostic/non-FP4 paths own three ordinary contiguous allocations.
struct FullAttnQkvOutput {
  bool fp4 = false;
  std::optional<DBuf> packed_owner;
  std::optional<DBuf> q_owner;
  std::optional<DBuf> k_owner;
  std::optional<DBuf> v_owner;
  Tensor qgate;
  Tensor key;
  Tensor value;
};

FullAttnQkvOutput ProjectFullAttnQkv(Dev d, const FullAttnLayerWeights& w,
                                     const Tensor& h, int64_t t,
                                     const Tensor* h_fp8,
                                     bool packed_consumers) {
  FullAttnQkvOutput out;
  out.fp4 = !w.q_proj_fp4.Empty();
  const bool fp8 = !w.q_proj_fp8.Empty();

  // MODEL-QWEN35-EXL3 (#2495 item 3). FIRST and exclusive: the loader fills the
  // trellis fields and leaves the bf16, fp4, per-tensor fp8 and block fp8 ones
  // EMPTY, so a non-empty EXL3 weight IS the scheme -- the same rung order the
  // loader uses, read from the forward's end.
  //
  // THREE projections, never the merged operand the branches below build. A
  // trellis is `[k/16, n/16, 32*bits]`, so joining on the output dim
  // interleaves per input tile rather than row-stacking; that merge is valid
  // for this family and worth doing, and it is owed its own gate (`## Owed` in
  // `specs/quant-exl3-shared.md`). `packed_consumers` is therefore never
  // consulted here, and no `QkvSplit` runs.
  //
  // `out_dtype` is bf16 -- upstream's `out_dtype` at this site, the MODEL dtype
  // every layer inherits -- and NOT the f32 the plain-bf16 arm below happens to
  // emit. A token gate cannot see a dtype that is too wide.
  if (w.IsExl3()) {
    out.q_owner.emplace(dense_exl3::Linear(d, h, w.q_proj, w.q_proj_exl3, DType::kBF16));
    out.k_owner.emplace(dense_exl3::Linear(d, h, w.k_proj, w.k_proj_exl3, DType::kBF16));
    out.v_owner.emplace(dense_exl3::Linear(d, h, w.v_proj, w.v_proj_exl3, DType::kBF16));
    out.qgate = out.q_owner->t();
    out.key = out.k_owner->t();
    out.value = out.v_owner->t();
    return out;
  }

  // MODEL-FP8-BLOCK-MERGED (#1189 M6): vLLM's QKVParallelLinear, as ONE
  // block-scaled GEMM over the N-concatenated q|k|v operand
  // (`linear.py:1247-1260` @ `5559679229`). FIRST, before the fp4 and
  // per-tensor fp8 branches, mirroring the loader's own rung order: a
  // block-wise weight is `F8_E4M3` on disk and the per-tensor probe would
  // otherwise claim it (#1166 seen from the forward's end).
  //
  // `packed_consumers` is the site's own precondition and predates this row:
  // the merged output exposes ROW-STRIDED q/k/v views, which only the fused
  // attn preamble reads correctly, so the fp4 and per-tensor fp8 merged arms
  // carry the same gate. It is a property of the consumer, not of the scales.
  if (!w.q_proj_fp8_block.Empty() && packed_consumers) {
    const dense_fp8_block::Fp8BlockShard qkv_shards[3] = {
        {&w.q_proj_fp8_block, "q_proj"},
        {&w.k_proj_fp8_block, "k_proj"},
        {&w.v_proj_fp8_block, "v_proj"}};
    const dense_fp8_block::Fp8BlockMergedView qkv =
        dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockQkv, "qkv_proj", qkv_shards, 3,
            w.qkv_fp8_block_merged);
    // bf16, which is upstream's `out_dtype` at this site and what the split
    // block arm below already emits, so the merge moves no dtype.
    out.packed_owner.emplace(dense_fp8_block::MatmulFp8BlockMergedD<DBuf>(
        d, h, qkv, DType::kBF16, vt::kFp8BlockQkv));
    Tensor all = out.packed_owner->t();
    const int64_t qn = w.q_proj_fp8_block.n;
    const int64_t kn = w.k_proj_fp8_block.n;
    const int64_t vn = w.v_proj_fp8_block.n;
    VT_CHECK(all.shape[1] == qn + kn + vn,
             "qwen3_5 packed block-wise FP8 QKV: output shape mismatch");
    out.qgate = all.Slice(1, 0, qn);
    out.key = all.Slice(1, qn, qn + kn);
    out.value = all.Slice(1, qn + kn, qn + kn + vn);
    return out;
  }
#ifdef VT_CUTLASS_NVFP4
  if (out.fp4 && MergedQkvEligible(w, d, packed_consumers)) {
    out.packed_owner.emplace(MergedQkvCutlassD(d, h, w));
    Tensor all = out.packed_owner->t();
    const int64_t qn = w.q_proj_fp4.n;
    const int64_t kn = w.k_proj_fp4.n;
    const int64_t vn = w.v_proj_fp4.n;
    VT_CHECK(all.shape[1] == qn + kn + vn,
             "qwen3_5 packed QKV: output shape mismatch");
    out.qgate = all.Slice(1, 0, qn);
    out.key = all.Slice(1, qn, qn + kn);
    Tensor value2 = all.Slice(1, qn + kn, qn + kn + vn);
    out.value = value2;
    return out;
  }
#endif

  // Merged FP8 QKVParallelLinear (opt-in): ONE fp8 GEMM over the N-concatenated
  // operand + per-column alpha, then row-strided views (mirror the fp4 branch).
  // The F32 output matches the split fp8 path's F32 q/k/v (kF32 below).
  if (fp8 && MergedFp8QkvEligible(w, d, packed_consumers)) {
    out.packed_owner.emplace(MergedFp8QkvD(d, h, h_fp8, w));
    Tensor all = out.packed_owner->t();
    const int64_t qn = w.q_proj_fp8.n;
    const int64_t kn = w.k_proj_fp8.n;
    const int64_t vn = w.v_proj_fp8.n;
    VT_CHECK(all.shape[1] == qn + kn + vn,
             "qwen3_5 packed FP8 QKV: output shape mismatch");
    out.qgate = all.Slice(1, 0, qn);
    out.key = all.Slice(1, qn, qn + kn);
    out.value = all.Slice(1, qn + kn, qn + kn + vn);
    return out;
  }

  // Split reference: quantize the shared activation once when the three input
  // divisors are equal, then retain one independently-scaled GEMM per shard.
  const bool fuse_qkv =
      out.fp4 && FuseQuantOnceEnabled() &&
      vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() &&
      w.q_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled() &&
      w.q_proj_fp4.input_global_scale_inv ==
          w.k_proj_fp4.input_global_scale_inv &&
      w.q_proj_fp4.input_global_scale_inv ==
          w.v_proj_fp4.input_global_scale_inv;
  std::optional<DBuf> qkv_ap;
  std::optional<DBuf> qkv_as;
#ifdef VT_CUTLASS_NVFP4
  const bool qkv_direct_scale =
      fuse_qkv && DirectFp4ScaleEligible(d);
#else
  const bool qkv_direct_scale = false;
#endif
  if (fuse_qkv) {
    const int64_t hidden = h.shape[1];
    qkv_ap.emplace(d, DType::kI8,
                   std::vector<int64_t>{t, hidden / 2});
    qkv_as.emplace(d, DType::kI8,
                   qkv_direct_scale
                       ? CutlassFp4ScaleShape(t, hidden)
                       : std::vector<int64_t>{t, hidden / 16});
    vt::ScaledFp4Quant(d.q, qkv_ap->t(), qkv_as->t(), h,
                       w.q_proj_fp4.input_global_scale_inv,
                       qkv_direct_scale
                           ? vt::Fp4ScaleLayout::kCutlassSwizzled
                           : vt::Fp4ScaleLayout::kLinear);
  }
  const Tensor* qkv_sf_sw_p = nullptr;
#ifdef VT_CUTLASS_NVFP4
  std::optional<DBuf> qkv_sf_sw;
  if (qkv_direct_scale) {
    qkv_sf_sw_p = &qkv_as->t();
  } else if (fuse_qkv && SwizzleInQuantEnabled() &&
             NvfpCutlassEnabled()) {
    qkv_sf_sw.emplace(SwizzleActScaleOnce(d, qkv_as->t()));
    qkv_sf_sw_p = &qkv_sf_sw->t();
  }
#endif
  const DType q_out_dt =
      (Bf16GemmOutEnabled() && out.fp4) ? DType::kBF16 : DType::kF32;
  const auto project = [&](const Nvfp4Weight& fp4_weight,
                           const Fp8Weight& fp8_weight,
                           const Fp8BlockWeight& block_weight,
                           const OwnedTensor& plain_weight) -> DBuf {
    // MODEL-FP8-BLOCK-LINEAR (#1189 M4). FIRST, and exclusive: M3's loader
    // fills the block field and leaves the bf16, per-tensor fp8 and fp4 ones
    // EMPTY (qwen3_5_dense_weights.cpp), so a non-empty block weight IS the
    // scheme. `out_dtype` is bf16 because that is upstream's `out_dtype` here
    // -- the MODEL dtype (fp8.py:284) -- not the f32 the per-tensor arm below
    // happens to use; a token gate cannot see a dtype that is too wide.
    if (!block_weight.Empty()) {
      return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, h, block_weight,
                                                          DType::kBF16);
    }
    if (fuse_qkv) {
      return MatmulNvfp4Fp4DirectD(d, qkv_ap->t(), qkv_as->t(),
                                    fp4_weight, q_out_dt, qkv_sf_sw_p);
    }
    if (fp8) {
      return h_fp8 != nullptr
                 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, fp8_weight,
                                              DType::kF32)
                 : MatmulFp8CutlassD(d, h, fp8_weight, DType::kF32);
    }
    if (out.fp4) {
      return Bf16GemmOutEnabled()
                 ? MatmulNvfp4Bf16D(d, h, fp4_weight)
                 : MatmulNvfp4F32D(d, h, fp4_weight);
    }
    // Ordinary Qwen3.5 safetensors retain raw torch BF16 weights and therefore
    // torch Linear's BF16 output. Existing transposed GGUF/synthetic paths keep
    // their established f32 output.
    return plain_weight.nk ? MatmulBf16D(d, h, plain_weight)
                           : MatmulF32D(d, h, plain_weight);
  };
  out.q_owner.emplace(
      project(w.q_proj_fp4, w.q_proj_fp8, w.q_proj_fp8_block, w.q_proj));
  out.k_owner.emplace(
      project(w.k_proj_fp4, w.k_proj_fp8, w.k_proj_fp8_block, w.k_proj));
  out.v_owner.emplace(
      project(w.v_proj_fp4, w.v_proj_fp8, w.v_proj_fp8_block, w.v_proj));
  out.qgate = out.q_owner->t();
  out.key = out.k_owner->t();
  out.value = out.v_owner->t();
  return out;
}

#ifdef VT_MARLIN_NVFP4
// --- Dense NVFP4 W4A16 Marlin (M0.8 dense sibling of the MoE grouped GEMM).
// The 35B's dense NVFP4 projections (shared-expert gate/up/down + lm_head) run
// at decode (m=8) through MatmulNvfp4's naive one-thread-per-output kernel, which
// re-dequants the whole weight column PER activation row (8x redundant dequant,
// ~19% of decode GPU per the nsys profile). Route them through the SAME vendored,
// bit-exact Marlin W4A16 GEMM as the MoE experts, run as a SINGLE-expert grouped
// GEMM (E=1, top_k=1): y[M,N] = x[M,K] @ dequant(W[N,K]).T. Repack is load-time
// (BuildMarlinDenseResident); the align inputs are trivial (all tokens -> expert
// 0) and cached per token count. Gated by VT_NVFP4_MARLIN (MarlinMoeEnabled()).
struct MarlinDenseResident {
  void* w = nullptr;  // i32 [K/16, N*2]  Marlin-interleaved weight
  void* s = nullptr;  // fp8 [K/16, N]    processed S0E5M3 scales
  void* g = nullptr;  // f32 [1]          processed global scale
  int64_t n = 0, k = 0;
  bool ready = false;
};

MarlinDenseResident& MarlinDenseResidentFor(const Nvfp4Weight* w) {
  return ResidentIn<MarlinDenseResident>(w->resident_marlin);
}

// Repack one dense NVFP4 weight into the resident Marlin layout, then free the
// fp4 device originals (Marlin is the only consumer once the gate is on) so peak
// weight memory stays flat. Same repack primitives as BuildMoeMarlinResident;
// the single weight is its own "expert" (E=1), so combined_scale_factor is taken
// over just this scale buffer.
void BuildMarlinDenseResident(Dev d, const Nvfp4Weight& w, MarlinDenseResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(w.k);
  const int N = static_cast<int>(w.n);
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(N) * 2);
  const size_t s_b = static_cast<size_t>(K / 16) * N;
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = w.n;
  mr.k = w.k;
  std::vector<const uint8_t*> bufs{reinterpret_cast<const uint8_t*>(w.scale.bytes.data())};
  std::vector<size_t> lens{w.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, static_cast<uint32_t*>(mr.w),
                                     static_cast<const uint8_t*>(dw.packed.data), K, N);
  vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dw.scale.data),
                                      static_cast<uint8_t*>(mr.s), K, N, sf);
  const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
  d.b.Copy(d.q, mr.g, &g, sizeof(float));
  d.b.Synchronize(d.q);  // repack done -> safe to free the fp4 originals
  w.d_packed.reset();
  w.d_scale.reset();
  mr.ready = true;
}

// Trivial single-expert moe_align inputs (all M tokens -> expert 0), cached per
// token count M (decode M is constant, so this runs once). Avoids a per-GEMM
// moe_align launch + allocations for the ~120 dense Marlin GEMMs of a step.
struct DenseAlignCache {
  void* sorted = nullptr;  // i32 [max_tok]
  void* expert = nullptr;  // i32 [max_blk] (all 0)
  void* numpad = nullptr;  // i32 [1]
  void* topkw = nullptr;   // f32 [M] (ones; unused since mul_topk_weights=false)
  int block = 0, max_tok = 0, max_blk = 0;
};

DenseAlignCache& DenseAlignFor(Dev d, int M) {
  static std::mutex mu;
  static std::unordered_map<int, DenseAlignCache> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(M);
  if (it != cache.end()) return it->second;
  DenseAlignCache c;
  c.block = vt::cuda::MarlinMoeAlignBlockSizeSelect(M, 1, 1);
  vt::cuda::MarlinMoeAlignSizes(M, 1, 1, c.block, &c.max_tok, &c.max_blk);
  c.sorted = d.b.Alloc(static_cast<size_t>(c.max_tok) * sizeof(int32_t));
  c.expert = d.b.Alloc(static_cast<size_t>(c.max_blk) * sizeof(int32_t));
  c.numpad = d.b.Alloc(sizeof(int32_t));
  c.topkw = d.b.Alloc(static_cast<size_t>(M) * sizeof(float));
  void* tid = d.b.Alloc(static_cast<size_t>(M) * sizeof(int32_t));
  d.b.Memset(d.q, tid, 0, static_cast<size_t>(M) * sizeof(int32_t));  // topk_ids = 0 -> expert 0
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(tid), M, 1, 1, c.block,
                                    static_cast<int32_t*>(c.sorted),
                                    static_cast<int32_t*>(c.expert),
                                    static_cast<int32_t*>(c.numpad));
  std::vector<float> ones(static_cast<size_t>(M), 1.0F);
  d.b.Copy(d.q, c.topkw, ones.data(), ones.size() * sizeof(float));
  d.b.Synchronize(d.q);
  d.b.Free(tid);
  return cache.emplace(M, c).first->second;
}

// Shared zeroed reduction workspace for the dense Marlin GEMMs (sms*4 i32 locks,
// mirror marlin_make_workspace_new). Memset to zero before each launch.
void* DenseMarlinWorkspace(Dev d, int* out_sms) {
  static std::mutex mu;
  static void* ws = nullptr;
  static int sms = 0;
  std::lock_guard<std::mutex> lk(mu);
  if (!ws) {
    sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
    ws = d.b.Alloc(static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  }
  *out_sms = sms;
  return ws;
}

// y[M,N] = x[M,K] bf16 @ dequant(w).T via the single-expert Marlin W4A16 GEMM.
DBuf MatmulNvfp4MarlinD(Dev d, const Tensor& x, const Nvfp4Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  MarlinDenseResident& mr = MarlinDenseResidentFor(&w);
  if (!mr.ready) BuildMarlinDenseResident(d, w, mr);
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));

  // VT_MARLIN_DENSE (default ON): the framework-wide dense NVFP4 route this TU
  // was forked off. E=1 dense projections go through vLLM's OWN dense marlin
  // GEMM instead of the single-expert MoE-marlin route. Same resident
  // (mr.w/mr.s/mr.g) and workspace; rank-2 operand views (the dense launcher
  // wants [K/16, N*2] / [K/gs, N], not the MoE rank-3 [1, ...]) and direct-A,
  // so no moe_align cache. Mirrors dense_nvfp4_gemm.h's MatmulNvfp4MarlinD,
  // which qwen3 / olmo2 / deepseek_v2 / minimax_h3 already take.
  if (dense_nvfp4::MarlinDenseEnabled() &&
      vt::OpRegistered(vt::OpId::kMarlinDenseGemm, d.q.device.type)) {
    DBuf outd(d, DType::kBF16, {M, N});
    Tensor wqd = MakeTensor(mr.w, DType::kI32, d.q.device, {K / 16, N * 2});
    Tensor scd = MakeTensor(mr.s, DType::kI8, d.q.device, {K / 16, N});
    Tensor ggd = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
    Tensor wstd = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
    vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(N),
                              static_cast<int>(K)};
    dargs.group_size = 16;
    dargs.mxfp4 = false;
    vt::MarlinDenseGemm(d.q, outd.t(), x, wqd, scd, ggd, wstd, dargs);
    if (out_dtype == DType::kBF16) return outd;
    DBuf outf(d, DType::kF32, {M, N});
    vt::CastF32(d.q, outf.t(), outd.t());
    return outf;
  }

  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));

  // Marlin's output is bf16 (c_type=kBFloat16); an f32 result is the bf16 output
  // upcast (same value it rounds to — mirror of the cutlass f32-scratch cast).
  DBuf outbf(d, DType::kBF16, {M, N});
  Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, N * 2});
  Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / 16, N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
  Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
  Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
  Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
  vt::MoeGroupedGemmNvfp4Marlin(
      d.q, outbf.t(), x, wq, sc, gg, wst, sorted, expert, numpad, topkw,
      vt::MoeMarlinArgs{ac.block, 1, static_cast<int>(M), static_cast<int>(N),
                        static_cast<int>(K), false});
  if (out_dtype == DType::kBF16) return outbf;
  DBuf out(d, DType::kF32, {M, N});
  vt::CastF32(d.q, out.t(), outbf.t());
  return out;
}

// --- Fused shared-expert gate_up Marlin resident (VT_MOE_FUSED_W13, the dense
// sibling of the MoE fused w13). The shared expert's gate/up dense NVFP4
// projections are TWO separate checkpoint weights here, but in vLLM they are
// ONE merged parameter (MergedColumnParallelLinear gate_up_proj — w1 rows
// first, w3 rows second) repacked WHOLE as a single Marlin operand
// (marlin_utils_fp4.py prepare_fp4_layer_for_marlin). Mirroring that: the pair
// is N-concatenated (gate rows [0,Is), up rows [Is,2Is)) and repacked ONCE
// with size_n=2Is + ONE combined_scale_factor over both shards, so the forward
// runs ONE Marlin GEMM [T,2Is] + SiluAndMul instead of two GEMMs (+2 workspace
// memsets, +2 CastF32) + MoeSiluMul. Requires gate.scale2 == up.scale2 (single
// per-GEMM global scale — same rule as the MoE fused w13); the caller guards.
struct MarlinDensePairResident {
  void* w = nullptr;   // i32 [K/16, (2N)*2]
  void* s = nullptr;   // fp8 [K/16, 2N]
  void* g = nullptr;   // f32 [1]
  int64_t n = 0, k = 0;  // n = per-shard N (Is); operand size_n = 2n
  bool ready = false;
};

MarlinDensePairResident& MarlinDensePairResidentFor(const Nvfp4Weight* gate) {
  // Held on the GATE weight: it was the pair's map key, and it is the member of
  // the pair whose lifetime the fused repack must not outlive.
  return ResidentIn<MarlinDensePairResident>(gate->resident_marlin_pair);
}

void BuildMarlinDensePairResident(Dev d, const Nvfp4Weight& gw, const Nvfp4Weight& uw,
                                  MarlinDensePairResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(gw.k);
  const int N = static_cast<int>(gw.n);
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(2 * N) * 2);
  const size_t s_b = static_cast<size_t>(K / 16) * (2 * N);
  const size_t pk_b = static_cast<size_t>(N) * (K / 2);   // one shard's packed bytes
  const size_t sc_b = static_cast<size_t>(N) * (K / 16);  // one shard's scale bytes
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = gw.n;
  mr.k = gw.k;
  // combined_scale_factor over BOTH shards (vLLM computes it over the merged
  // gate_up scale tensor).
  std::vector<const uint8_t*> bufs{reinterpret_cast<const uint8_t*>(gw.scale.bytes.data()),
                                   reinterpret_cast<const uint8_t*>(uw.scale.bytes.data())};
  std::vector<size_t> lens{gw.scale.bytes.size(), uw.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dg = ResidentNvfp4(d, gw);
  Nvfp4Dev du = ResidentNvfp4(d, uw);
  // Flat row-stack concat (packed [N,K/2] u8 / scales [N,K/16] fp8 are
  // row-major over N; gate rows first — the vLLM merged shard order).
  auto* tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
  auto* tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sc_b));
  d.b.Copy(d.q, tmp_w, dg.packed.data, pk_b);
  d.b.Copy(d.q, tmp_w + pk_b, du.packed.data, pk_b);
  d.b.Copy(d.q, tmp_s, dg.scale.data, sc_b);
  d.b.Copy(d.q, tmp_s + sc_b, du.scale.data, sc_b);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, static_cast<uint32_t*>(mr.w),
                                     tmp_w, K, 2 * N);
  vt::cuda::MarlinProcessExpertScales(stream, tmp_s, static_cast<uint8_t*>(mr.s), K, 2 * N, sf);
  // Single global scale for both shards (gate's; equality guarded by caller —
  // the vLLM merged parameter has exactly one weight_global_scale).
  const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(gw.scale2, sf);
  d.b.Copy(d.q, mr.g, &g, sizeof(float));
  d.b.Synchronize(d.q);  // repack done -> safe to free staging + fp4 originals
  d.b.Free(tmp_w);
  d.b.Free(tmp_s);
  gw.d_packed.reset();
  gw.d_scale.reset();
  uw.d_packed.reset();
  uw.d_scale.reset();
  mr.ready = true;
}

// True when the shared-expert gate/up pair takes the fused Marlin gate_up path
// (one GEMM [T,2Is] + SiluAndMul). Must be checked IDENTICALLY at load
// (PrepareMarlinResident) and forward so exactly one resident layout is built.
bool SharedGateUpFusedEligible(const Nvfp4Weight& gw, const Nvfp4Weight& uw) {
  return MoeFusedW13Enabled() && !gw.Empty() && !uw.Empty() && !gw.IsTrueW4A4() &&
         !uw.IsTrueW4A4() && gw.n == uw.n && gw.k == uw.k && gw.scale2 == uw.scale2;
}

// silu(x@gate.T) * (x@up.T) -> bf16 [M,Is] via ONE fused Marlin gate_up GEMM.
DBuf SharedGateUpFusedMarlinD(Dev d, const Tensor& x, const Nvfp4Weight& gw,
                              const Nvfp4Weight& uw) {
  const int64_t M = x.shape[0], K = x.shape[1], N = gw.n;
  MarlinDensePairResident& mr = MarlinDensePairResidentFor(&gw);
  if (!mr.ready) BuildMarlinDensePairResident(d, gw, uw, mr);
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));

  DBuf gu(d, DType::kBF16, {M, 2 * N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  // VT_MARLIN_DENSE_PAIR (default ON): the single-projection sink already takes vLLM's OWN
  // dense marlin GEMM (MatmulNvfp4MarlinD, VT_MARLIN_DENSE). This fused
  // shared-expert gate_up sink did NOT, so it still ran the single-expert
  // MoE-marlin route: measured at c8 that is 20320 launches (one per layer per
  // step) of <128,1,8,4,m_block_size_8=false> = 5.4% of GPU time, a kernel
  // configuration the pinned vLLM never launches. Same resident (mr.w/mr.s/
  // mr.g) and workspace; rank-2 operand views and direct-A, so no moe_align
  // cache and no row padding.
  if (dense_nvfp4::MarlinDensePairEnabled() &&
      vt::OpRegistered(vt::OpId::kMarlinDenseGemm, d.q.device.type)) {
    Tensor wqd = MakeTensor(mr.w, DType::kI32, d.q.device, {K / 16, 2 * N * 2});
    Tensor scd = MakeTensor(mr.s, DType::kI8, d.q.device, {K / 16, 2 * N});
    vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(2 * N),
                              static_cast<int>(K)};
    dargs.group_size = 16;
    dargs.mxfp4 = false;
    vt::MarlinDenseGemm(d.q, gu.t(), x, wqd, scd, gg, wst, dargs);
  } else {
    DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
    Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, 2 * N * 2});
    Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / 16, 2 * N});
    Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
    Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
    Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
    Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
    vt::MoeGroupedGemmNvfp4Marlin(
        d.q, gu.t(), x, wq, sc, gg, wst, sorted, expert, numpad, topkw,
        vt::MoeMarlinArgs{ac.block, 1, static_cast<int>(M), static_cast<int>(2 * N),
                          static_cast<int>(K), false});
  }
  DBuf act(d, ActDType(d), {M, N});
  vt::SiluAndMul(d.q, act.t(), gu.t());
  return act;
}

// PERF-27B-DENSE-MARLIN-GATEUP (issue #365). The DENSE MLP's W4A16 gate/up pair
// through the SAME fused Marlin gate_up GEMM the shared expert already takes:
// ONE [T,H]x[2I,H] GEMM + SiluAndMul per layer instead of two GEMMs + a
// MoeSiluMul. `nullopt` means "not this configuration" and the caller keeps the
// split path unchanged.
//
// It is the SAME call, the SAME MarlinDensePairResident (held on the gate
// weight's `resident_marlin_pair` slot, so a dense gate weight keys it exactly
// like a shared-expert gate weight does) and the SAME guard terms — there is no
// second fused implementation and no second resident cache.
//
// The extra `Bf16GemmOutEnabled()` term is what keeps the SINK equivalent: the
// split path only produces bf16 gate/up operands for its MoeSiluMul when that
// lever is on, and the fused sink's SiluAndMul reads bf16 halves of one [T,2I]
// buffer. Both epilogues compute `g/(1+expf(-g))*u` in f32 and round on store
// (cuda_ops.cu SiluAndMulKernel / cuda_moe.cu MoeSiluMulKernel), so on bf16
// inputs the two activations agree bit for bit.
//
// The GEMM itself does NOT: Marlin's fp32 split-K reduce groups the K-slices
// differently for a [2N,K] operand than for two [N,K] operands, which is one
// bf16 ULP on ~0.1% of elements — MEASURED and recorded for this same fused
// entry point at tests/.../test_linear_method.cpp:202 (99.9% bit-exact vs
// split). So the bar for flipping this on is token-exactness against the pinned
// ORACLE, exactly as it was for the shared-expert pair, and NOT bit-equality
// with our own split path.
//
// Like BuildDenseHeadMarlinResident below, it lives INSIDE the region that
// already owns this kernel family rather than behind a second
// `#ifdef VT_MARLIN_NVFP4` at the DenseMlpBlock call site, for the DSR-ratchet
// reason spelled out there.
std::optional<DBuf> DenseGateUpFusedMarlinD(Dev d, const Tensor& x,
                                            const DenseMlpWeights& w) {
  if (x.dtype != DType::kBF16 || !Bf16GemmOutEnabled() ||
      !dense_nvfp4::DenseMlpGateUpFusedMarlinEligible(w.gate_proj_fp4, w.up_proj_fp4,
                                                      d.q.device.type)) {
    return std::nullopt;
  }
  return SharedGateUpFusedMarlinD(d, x, w.gate_proj_fp4, w.up_proj_fp4);
}

// PERF-27B-LMHEAD-FP4 (issue #213). Build the dense head's Marlin resident when
// THIS configuration will actually take the Marlin logits GEMM, and report
// whether it did. The predicate is EXACTLY the one MatmulNvfp4F32D selects with,
// so a configuration that will not take that path never builds for it.
//
// It lives here, inside the region that already owns this kernel family, rather
// than as a second `#ifdef VT_MARLIN_NVFP4` at the call site in
// PrepareLmHeadResident: the DSR ratchet (scripts/check-device-leakage.py)
// counts every build-time kernel-feature gate in the device-agnostic shared
// layer and fails on any increase. A new guard at the call site was one, and the
// honest repair is to keep the gate in the one place that already has it.
bool BuildDenseHeadMarlinResident(Dev d, const Nvfp4Weight& w) {
  if (w.IsTrueW4A4() || !MarlinMoeEnabled() ||
      !vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type)) {
    return false;
  }
  BuildMarlinDenseResident(d, w, MarlinDenseResidentFor(&w));
  d.b.Synchronize(d.q);
  return true;
}
#else
// No Marlin NVFP4 in this build: there is no dense-head Marlin resident to
// build, so PrepareLmHeadResident falls straight through to the arm this
// backend's logits GEMM will actually take, and the dense MLP keeps its split
// gate/up pair (there is no fused Marlin GEMM to substitute).
bool BuildDenseHeadMarlinResident(Dev, const Nvfp4Weight&) { return false; }
std::optional<DBuf> DenseGateUpFusedMarlinD(Dev, const Tensor&, const DenseMlpWeights&) {
  return std::nullopt;
}
#endif  // VT_MARLIN_NVFP4

DBuf MatmulNvfp4F32D(Dev d, const Tensor& x, const Nvfp4Weight& w) {
  const int64_t M = x.shape[0], N = w.n;
  if (vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() && w.IsTrueW4A4() && TrueW4A4Enabled())
    return MatmulNvfp4Fp4D(d, x, w, DType::kF32);
#ifdef VT_MARLIN_NVFP4
  // NVFP4 W4A16 dense (shared expert / lm_head): the load-time-repacked Marlin
  // GEMM replaces the naive redundant-dequant kernel when VT_NVFP4_MARLIN=1.
  // Marlin requires a bf16 activation (the 35B dense NVFP4 sinks all are).
  if (vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) && !w.IsTrueW4A4() && MarlinMoeEnabled() &&
      x.dtype == DType::kBF16)
    return MatmulNvfp4MarlinD(d, x, w, DType::kF32);
#endif
  DBuf dout(d, DType::kF32, {M, N});
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    MatmulNvfp4DequantB(d, dout.t(), x, w);
  }
  return dout;
}

// The ONE dense-gate logits GEMM: y[M,vocab] f32 = x[M,H] @ lm_head.
// PERF-27B-LMHEAD-FP4 (issue #213). A ModelOpt NVFP4 head stays PACKED, so the
// GEMM reads K*N/2 + K*N/16 bytes per step instead of the 2*K*N of a dequantized
// bf16 operand (~0.715 GB vs ~2.543 GB at the real 248320x5120), and keeps its
// on-disk [N,K] orientation instead of forcing the row-major NN GEMM that has no
// nvjet_sm121 kernel. Mirrors logits_processor._apply_head ->
// lm_head.quant_method.apply (logits_processor.py:98-133). Every dense consumer
// (eager ForwardDense, the gathered and non-gathered paged arms) routes here, so
// exactly one head layout is selected; the bf16 arm keeps both of its shapes.
DBuf DenseLogitsF32D(Dev d, const Tensor& x, const Qwen3_5DenseWeights& weights) {
  // MODEL-QWEN35-EXL3 (#2495 item 5): a trellis head, kept quantized and read
  // through the shared linear seam. f32 out, because that is the dtype every
  // other arm of this function returns and the sampler's contract; the trellis
  // GEMM writes f32 directly, so nothing is widened on the way.
  if (!weights.lm_head_exl3.Empty()) {
    return dense_exl3::Linear(d, x, weights.lm_head, weights.lm_head_exl3,
                              DType::kF32);
  }
  if (!weights.lm_head_fp4.Empty())
    return MatmulNvfp4F32D(d, x, weights.lm_head_fp4);
  const OwnedTensor& lm_head = DenseLmHead(weights);
  // #2534: a GGUF KEEP-QUANT head takes the f32-output GEMM, joining the EXL3
  // and NVFP4 heads above rather than the bf16 helper below.
  //
  // It used to take the bf16 helper, and only because `nk` is a LAYOUT flag:
  // `qwen3_5_gguf_weights.cpp::OwnGgufQuantBlocks` sets it true because "GGUF
  // disk order [out, in] IS the MatmulBT [N, K] orientation", which says nothing
  // about numerics. It therefore inherited a rule authored for the TIED BF16
  // torch-Linear head the helper's own comment names. Measured cost of that
  // inheritance: 288 of 288 of the Q4_K_M arm's top-1 logits landed exactly on
  // the bf16 grid, whose ULP is 0.125 at the magnitude 16-32 these logits carry,
  // while the six near-ties the llama.cpp token gate convicts us on are gaps of
  // 0.027 to 0.178 -- five of six below our own resolution, and six steps EXACT
  // TIES (#2534, docs/bench-evidence/qwen38-27b-q4km-logit-delta-20260902.md).
  //
  // This DIVERGES from vLLM's default, deliberately, and the divergence is
  // argued in .agents/specs/qwen38-27b-q4km-logits-f32.md rather than assumed.
  // Upstream keeps the head in the model dtype (`logits_processor.py:99-136`
  // `_apply_head`; `config/model.py:2187-2208` `_get_head_dtype` defaults a
  // GENERATION model to the model dtype) and widens afterwards in the sampler
  // (`v1/sample/sampler.py:96`), exactly as `MatmulBf16LogitsF32D` does. But
  // vLLM has no in-tree GGUF reader at the pin, so it resolves no model dtype
  // for this checkpoint at all (#979); and where upstream DOES select an f32
  // head it accumulates straight into f32 rather than casting operands
  // (`logits_processor.py:127-131`, `torch.mm(..., out_dtype=...)`), which is
  // what this arm now does -- the keep-quant kernels already accumulate in f32
  // and only round at the store, so no weight copy and no cast pass is added.
  // Upstream's refusal of an f32 head on a QUANTIZED head
  // (`logits_processor.py:111-115`) guards its `.to(f32)` weight-cast fallback,
  // a mechanism we do not use.
  //
  // Elementwise bf16/f16 heads are untouched below, so no safetensors default
  // and no recorded device measurement on those arms moves.
  if (vt::IsBlockQuant(lm_head.dtype)) return MatmulF32D(d, x, lm_head);
  return lm_head.nk ? MatmulBf16LogitsF32D(d, x, lm_head)
                    : MatmulF32D(d, x, lm_head);
}

// Same as MatmulNvfp4F32D but bf16 output (the down/o/out_proj sinks that feed
// the residual add). CUDA: fp4-resident vt::MatmulNvfp4 (bf16 out). CPU: the
// DequantNvfp4ToBLayout fallback (no CPU MatmulNvfp4 kernel).
DBuf MatmulNvfp4Bf16D(Dev d, const Tensor& x, const Nvfp4Weight& w) {
  const int64_t M = x.shape[0], N = w.n;
  if (vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() && w.IsTrueW4A4() && TrueW4A4Enabled())
    return MatmulNvfp4Fp4D(d, x, w, DType::kBF16);
#ifdef VT_MARLIN_NVFP4
  if (vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) && !w.IsTrueW4A4() && MarlinMoeEnabled() &&
      x.dtype == DType::kBF16)
    return MatmulNvfp4MarlinD(d, x, w, DType::kBF16);
#endif
  DBuf dout(d, DType::kBF16, {M, N});
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    MatmulNvfp4DequantB(d, dout.t(), x, w);
  }
  return dout;
}

// --- Paged-path helpers (M1.8 Task 3) --------------------------------------

// Non-owning strided view over dim0 [row_offset, row_offset+rows) of a
// contiguous device tensor (all trailing dims kept). Used to hand GdnPrefill/
// GdnDecode the decode vs prefill token sub-slices without copying.
Tensor SubView(const Tensor& src, int64_t row_offset, int64_t rows) {
  Tensor t = src;
  int64_t row_elems = 1;
  for (int i = 1; i < src.rank; ++i) row_elems *= src.shape[i];
  t.data = static_cast<char*>(src.data) +
           static_cast<size_t>(row_offset * row_elems) * vt::SizeOf(src.dtype);
  t.shape[0] = rows;
  return t;
}

// One unbind(1) slice of a (num_blocks, 2, block_size, H, D) FlashAttention KV
// buffer: which=0 -> K, which=1 -> V. The result is the rank-4 STRIDED view
// backends read (block stride 2*bs*H*D — the "2" is NOT collapsed), matching the
// M1.6 Task-2 contract (cpu_cache.cpp / cpu_paged_attn.cpp).
Tensor KvSlice(const PagedKvCache& kv, Device dev, int which) {
  const int64_t bs = kv.block_size, h = kv.num_kv_heads, dd = kv.head_size;
  Tensor t;
  t.data = static_cast<char*>(kv.data) +
           static_cast<size_t>(which) * static_cast<size_t>(bs * h * dd) *
               vt::SizeOf(kv.dtype);
  t.dtype = kv.dtype;
  t.device = dev;
  t.rank = 4;
  t.shape[0] = kv.num_blocks;
  t.shape[1] = bs;
  t.shape[2] = h;
  t.shape[3] = dd;
  t.stride[0] = 2 * bs * h * dd;
  t.stride[1] = h * dd;
  t.stride[2] = dd;
  t.stride[3] = 1;
  return t;
}

// Gather the `idx`-indexed rows of a persistent state cache `src`
// [Nblk, row_elems] into contiguous `dst` (device-to-device via Backend::Copy;
// on CPU a memcpy, on CUDA a same-device copy). Row-major per row.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// Inverse of GatherRows: write the contiguous per-request rows back to their
// persistent cache slots. `dst` is a non-owning view whose buffer is mutable
// through .data even when the view is const.
void ScatterRows(Dev d, const Tensor& dst, const void* src,
                 const std::vector<int32_t>& idx, int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(dst.dtype);
  auto* dp = static_cast<char*>(dst.data);
  const auto* sp = static_cast<const char*>(src);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + static_cast<size_t>(idx[s]) * rb, sp + s * rb, rb);
}

// Strided GDN state row copy (CPU/f32 reference path).
//
// The persistent conv row is WIDENED to (K-1)+num_spec under speculative
// decoding so a rejected step can roll back, but the prefill working copy is
// legitimately narrow: prefill only produces the K-1 leading taps. GatherRows /
// ScatterRows assume ONE row size for both the slot stride and the row contents,
// which silently mis-addresses every channel past the first once those differ
// (and mis-strides the slot itself). Copy per CHANNEL instead: `taps` elements
// out of a `cache_taps`-strided cache row into a `taps`-strided working row.
//
// `gather` selects direction. Byte-identical to the contiguous helpers whenever
// cache_taps == taps, which is every non-speculative path.
void CopyStateRowsStrided(Dev d, void* work, const Tensor& cache,
                          const std::vector<int32_t>& idx, int64_t channels,
                          int64_t taps, int64_t cache_taps, bool gather) {
  const size_t esz = vt::SizeOf(cache.dtype);
  const size_t cache_slot = static_cast<size_t>(channels * cache_taps) * esz;
  const size_t work_slot = static_cast<size_t>(channels * taps) * esz;
  const size_t chunk = static_cast<size_t>(taps) * esz;
  auto* wp = static_cast<char*>(work);
  auto* cp = static_cast<char*>(cache.data);
  for (size_t s = 0; s < idx.size(); ++s) {
    char* crow = cp + static_cast<size_t>(idx[s]) * cache_slot;
    char* wrow = wp + s * work_slot;
    for (int64_t c = 0; c < channels; ++c) {
      char* csrc = crow + static_cast<size_t>(c * cache_taps) * esz;
      char* wdst = wrow + static_cast<size_t>(c * taps) * esz;
      if (gather) {
        d.b.Copy(d.q, wdst, csrc, chunk);
      } else {
        d.b.Copy(d.q, csrc, wdst, chunk);
      }
    }
  }
}

// Gather idx-indexed rows of a persistent GDN state cache into a fresh f32
// working buffer. Compressed cache dtypes round-trip at the cache boundary;
// the temporal state may independently be fp16/bf16/fp32 while the conv cache
// follows mamba_cache_dtype. The f32 buffer is what GdnPrefill consumes.
DBuf GatherStateF32(Dev d, const Tensor& cache, const std::vector<int32_t>& idx,
                    int64_t row_elems, const std::vector<int64_t>& shape) {
  VT_CHECK(!shape.empty() && shape[0] == static_cast<int64_t>(idx.size()),
           "qwen3_5: gathered state row count " +
               std::to_string(shape.empty() ? -1 : shape[0]) +
               " must match index count " + std::to_string(idx.size()));
  DBuf f32buf(d, DType::kF32, shape);
  if (cache.dtype == DType::kF16 || cache.dtype == DType::kBF16) {
    DBuf didx(d, DType::kI32,
              {static_cast<int64_t>(idx.size())}, idx.data());
    vt::GdnStateGather(d.q, f32buf.t(), cache, didx.t());
  } else if (cache.rank == 3 && shape.size() == 3 && cache.shape[2] != shape[2]) {
    // Widened speculative conv row vs narrow prefill working row.
    CopyStateRowsStrided(d, f32buf.ptr(), cache, idx, shape[1], shape[2],
                         cache.shape[2], /*gather=*/true);
  } else {
    GatherRows(d, f32buf.ptr(), cache, idx, row_elems);
  }
  return f32buf;
}

// Inverse of GatherStateF32: downcast to the independent cache dtype and
// scatter back to the indexed slots.
void ScatterStateF32(Dev d, const Tensor& cache, DBuf& f32buf,
                     const std::vector<int32_t>& idx, int64_t row_elems) {
  VT_CHECK(f32buf.t().rank > 0 &&
               f32buf.t().shape[0] == static_cast<int64_t>(idx.size()),
           "qwen3_5: scattered state row count " +
               std::to_string(f32buf.t().rank > 0 ? f32buf.t().shape[0] : -1) +
               " must match index count " + std::to_string(idx.size()));
  if (cache.dtype == DType::kF16 || cache.dtype == DType::kBF16) {
    DBuf didx(d, DType::kI32,
              {static_cast<int64_t>(idx.size())}, idx.data());
    Tensor mutable_cache = cache;
    vt::GdnStateScatter(d.q, mutable_cache, f32buf.t(), didx.t());
  } else if (cache.rank == 3 && f32buf.t().rank == 3 &&
             cache.shape[2] != f32buf.t().shape[2]) {
    CopyStateRowsStrided(d, f32buf.ptr(), cache, idx, f32buf.t().shape[1],
                         f32buf.t().shape[2], cache.shape[2],
                         /*gather=*/false);
  } else {
    ScatterRows(d, cache, f32buf.ptr(), idx, row_elems);
  }
}

// Are the four operators the indexed state-I/O path needs realized NATIVELY for
// this device? `OpRegistered` deliberately excludes the portable reference tier
// (op_provider.cpp), so this answers "can the device do the indexing on its own
// hardware", not "will the call succeed" — a device that would fall back to the
// host tier for these gains nothing from the switch and must keep the row-copy
// reference. Decode needs the two *Update/Decode index arguments; a mixed step
// with prefills additionally needs the fused gather/scatter.
bool IndexedGdnOpsNative(Device device) {
  return vt::OpRegistered(vt::OpId::kCausalConv1dUpdate, device.type) &&
         vt::OpRegistered(vt::OpId::kGdnDecode, device.type) &&
         vt::OpRegistered(vt::OpId::kGdnStateGather, device.type) &&
         vt::OpRegistered(vt::OpId::kGdnStateScatter, device.type);
}

// W1 indexed state-I/O dispatch. CUDA + device-resident W0 storage defaults to
// the fused indexed gather/scatter operators. Either diagnostic opt-out restores
// the exact row-copy + cast baseline on the same binary. CPU keeps that baseline
// as its reference implementation.
//
// BACKEND-VULKAN-DEVICE-RESIDENT. The old non-staging arm keyed the default on
// `needs_weight_staging()`, which is a statement about WEIGHT residency on a
// discrete device, not about whether the device can index its own state cache.
// The consequence on Vulkan was measured, not guessed: the row-copy arm issues
// four `Backend::Copy` calls per GDN layer (gather+scatter for conv and for ssm),
// and the 27B has 48 linear-attention layers, so a single decode token drove ~192
// host memcpys over device memory. Each one that intersects the open command
// batch forces a submit-plus-blocking-fence drain, which is where the measured 98
// copy-dst/copy-src flushes per token came from. The indexed arm passes the state
// slot indices to the kernels instead and issues NONE of those copies.
//
// The switch is therefore keyed on the real question — does this device have the
// indexed kernels natively — with CPU still pinned to the row-copy reference so
// its golden path is untouched, and with `needs_weight_staging()` devices (CUDA)
// evaluated by exactly the branch they took before.
bool IndexedGdnStateIoEnabled(Device device) {
  const char* indexed = std::getenv("VT_GDN_INDEXED_STATE_IO");
  const vllm::platforms::Platform& plat =
      vllm::platforms::GetPlatform(device.type);
  if (!plat.needs_weight_staging()) {
    // An explicit setting wins on every non-staging device: =1 is the CPU test
    // hook that drives the model integration through the reference kernels, =0
    // is the same-binary A/B that restores the row-copy baseline on Vulkan.
    if (indexed != nullptr) return indexed[0] == '1';
    // CPU keeps the row-copy reference by default.
    if (plat.is_cpu()) return false;
    return IndexedGdnOpsNative(device);
  }
  const char* cache = std::getenv("VT_DEVICE_KV_CACHE");
  if (cache != nullptr && cache[0] == '0') return false;
  return indexed == nullptr || indexed[0] != '0';
}

// Prefill launch-gap fusion (perf/glue-fuse): fold the GDN post-conv glue chain
// GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta (4 launches) into ONE
// vt::GdnPostConv launch, and the gated-RMSNorm + CastBf16 pair (2 launches)
// into a single RmsNormGated writing bf16 directly (layernorm_guard.py:57
// `out.to(dtype)`). Bit-for-bit vs the unfused chain. VT_GLUE_FUSE=0 restores
// the per-op path for A/B measurement (default ON).
bool GlueFuseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GLUE_FUSE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Prefill glue-fusion (35B GDN out_proj): fold the W8A8 out_proj's static fp8
// activation quant INTO the gated-RMSNorm output store (vt::RmsNormGatedQuantFp8),
// removing the standalone QuantFp8Static pass and the bf16 gated-norm output that it
// would otherwise write then re-read. Byte-identical to the split
// RmsNormGated(bf16)+QuantFp8Static path (the fp8 is quantized from the SAME bf16
// value; the gated-norm variance reduction order is unchanged — the fused kernel
// reproduces RmsNormGatedRowFastKernel's exact reduction). Only the 35B GDN out_proj
// is W8A8 fp8 (27B's out_proj is W4A4 fp4 and is untouched), so this lever affects
// the 35B alone — no 27B greedy-razor exposure. DEFAULT ON per the parity-enabler
// policy (byte-exact + measured-faster: 35B 315/315 both arms, memcheck 0, isolated
// −28.7% on the gated-norm+out_proj-quant chain — the fused fp8 1-byte store is even
// cheaper than the unfused gated norm's bf16 store — and in-situ TTFT c1 −1.4% /
// c2 −1.3% / c8 −0.4%, prefill tput +1%, zero regression). `VT_GDN_OUT_FP8_FUSE=0`
// rolls back to the split RmsNormGated + QuantFp8Static path. The fused path also
// requires VT_GLUE_FUSE on (bf16 gated store) — the production default.
bool GdnOutFp8FuseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_OUT_FP8_FUSE");
    return e == nullptr || e[0] != '0';  // default ON; =0 rolls back
  }();
  return on;
}

// Coupled GDN bf16 activations (the measured #1 prefill lever). Default ON. When
// on, the GDN chunk-scan matmul-INPUT activations (q/k/v out of the post-conv
// prep, hence the derived u/w/v_new/hstate scratch — those follow the input
// dtype in cuda_gdn.cu's LaunchChunkedPrefill) are bf16, so the WMMA chunk trio
// (WU Gram / DeltaH / ChunkO) runs on native bf16 tensor-core fragments (2× vs
// TF32) AND moves half the activation/scratch bytes. The recurrent ssm_state,
// the g log-decay + its cumsum, beta, and the WMMA f32 accumulators stay f32 —
// mirroring vLLM FLA's dtype split (chunk_delta_h.py final_state=torch.float32
// while v_new/w are k.dtype=bf16; wy_fast.py u/w = k.new_empty bf16; chunk_o.py
// b_o/b_A f32 accumulators, o stored bf16). The op-level bf16 WMMA path is the
// same one test_ops_gdn's bf16 chunked-vs-sequential case exercises (3e-2 tol).
// A/B: VT_GDN_BF16=0 restores the f32/TF32 activations in the same binary.
DType GdnActDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_GDN_BF16");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}

// GDN INPUT-side bf16 (VT_GDN_IN_BF16, DEFAULT ON — gated). Mirrors vLLM FLA
// carrying the *pre-chunk* activations in bf16, not just the chunk fragments:
// when ON, the GDN in_proj mixed_qkv GEMM emits bf16 (MatmulBf16D instead of
// MatmulF32D), the causal conv1d then runs bf16 in/out (bf16 conv weight,
// f32-accumulated internally, f32 conv_state unchanged), and the post-conv
// split/l2norm reads bf16 conv — halving the traffic of the two big [T,conv_dim]
// activation buffers (mixed write+conv read, conv write+post-conv read) that
// stayed f32 while the chunk trio was already bf16 (GdnActDType). The
// l2norm/softplus math is f32-accumulated regardless (Load() upcasts); g/beta,
// ssm_state, and the a/b GEMMs stay f32 (FLA's split). Gated on the bf16-weight
// in_proj branch (the 27B's in_proj_qkv is a plain bf16 weight); the fp8 in_proj
// branch reaches it only under GdnFp8InBf16Enabled() below (#417), which is
// default OFF, so an fp8 tower keeps its f32 output unless opted in.
// MEASURED (27B, GB10, same-binary A/B): conv kernel -31.5%, post-conv -17.6%
// (chunk trio FLAT — already bf16); e2e +0.68% (conc16, non-overlapping) / +0.83%
// (conc32), TTFT -1.5%; token-exact (27B greedy paged-engine + 35B 16/16). The
// GDN-vs-vLLM gap only 2.07×→~1.9× (the ~1.9× residual is the chunk's codegen
// gap, not dtype). A/B: VT_GDN_IN_BF16=0 restores the byte-identical f32 path.
DType GdnInDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_GDN_IN_BF16");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}

// PERF-FP8-ALPHA-FOLD / issue #417 — make GdnInDType() REACHABLE on an fp8-tower
// checkpoint (VT_GDN_FP8_IN_BF16, DEFAULT OFF).
//
// VT_GDN_IN_BF16 above is default ON with recorded conv -31.5% / post-conv
// -17.6%, but it cannot fire on the ModelOpt FP8 GDN tower (the `nvidia` 27B is
// `modelopt_mixed`, and the 35B shares the tower) because the merged fp8 in_proj
// leaf hardcoded DType::kF32 for its output. `convdt = mixed.dtype` then carries
// that f32 into the conv weight choice, the conv in/out buffer and the post-conv
// read — the two largest GDN activation buffers, at 2x the bytes vLLM moves.
//
// That f32 is also what makes the per-column alpha pass cost what it does. The
// merged [qkv;z] operand's shards carry DIFFERENT folded alphas on this
// checkpoint (ResidentFp8Qkvz's `folded` is FALSE), so the alpha is applied by
// vt::MulColVecF32 in a second full-tensor read-modify-write: MEASURED 122.99
// ms/request over 48 calls at T=4096 prefill = 43.6% of the whole 27B prefill
// deficit, at 209.5 GB/s = 77% of the GB10's 273.1 GB/s peak. It is
// bandwidth-saturated, so its cost IS its width and a bf16 buffer halves it
// (~61 ms/req, ~21.8% of the gap). All THREE cuBLASLt routes to ELIMINATING that
// pass are measured unavailable on this hardware
// (.agents/specs/perf-fp8-alpha-fold.md §Outcome); halving it depends on no
// vendor capability whatsoever.
//
// vLLM emits BF16 here: ModelOptFp8LinearMethod applies with
// out_dtype = torch.get_default_dtype() (vllm/model_executor/layers/quantization/
// modelopt.py:458 @ the pin), and qwen_gdn_linear_attn.py:1285-1295 runs the
// causal conv on that bf16 tensor. So this narrows TOWARD the oracle; our f32 is
// the deviation, and it is the "too WIDE" kind a token-exactness gate cannot see
// (.agents/porting.md, "Mirror the memory format, not just the math").
//
// It is NOT free, and that is why it is opt-in. Narrowing the GEMM's D rounds the
// f32 accumulator to bf16 BEFORE the alpha multiply instead of after it, so
// tokens CAN move. Both SACRED engine gates decide it, at their exact case AND
// assertion counts; a lost token is NEEDS_DECISION, never a re-cut golden.
//
// f32-ONLY consumers are untouched: conv_state, g/beta, a_log/dt_bias, ssm_state
// and the a/b GEMMs (FLA's split, ops.cpp gdn_g_beta). Only the STORE dtype
// changes anywhere on this path — every kernel reads through Load(), which
// upcasts to f32 before any arithmetic.
bool GdnFp8InBf16Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_FP8_IN_BF16");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// GDN recurrence-OUTPUT + z-gate in bf16, on EVERY arm (default ON;
// VT_GDN_OUT_BF16=0 is the same-binary f32 rollback).
// vLLM keeps core_attn_out and the z gate bf16 (the gated-RMSNorm consumes them):
// FLA chunk_o.py stores o bf16, and Qwen3NextGatedRMSNorm reads bf16 core/gate,
// upcasting to f32 only for the variance reduction (layernorm_guard.py). Our
// `dcore` (recurrence out) + `z` were f32 (a more-precise deviation that doubled
// the [T,Hv,Dv] core traffic in/out of GdnDecode/GdnPrefill AND the gated-norm
// read). When ON, `dcore` is bf16 (GdnDecode/GdnPrefill already dispatch a Tout=bf16
// path so they store bf16 directly), `z` is bf16 (MatmulBf16D), and the gated-norm
// weight is loaded native bf16 (RmsNormGated requires gate/weight dtype == x dtype)
// — the norm's variance/normalize math stays f32-accumulated regardless, so only
// the core/z I/O dtype changes. ssm_state, g (+cumsum), and beta stay f32 (FLA's
// split). Distinct from the earlier VT_BF16_GDN (in_proj/conv/z-gate, neutral):
// this lever is the f32 `dcore` recurrence output that attempt left untouched.
// This is correctness-significant for the 27B: with the repaired full NVFP4
// tactic stack, f32 core/z takes the alternate whitespace near-tie branch while
// bf16 reproduces native vLLM 16/16.
//
// GDN-MOE-BF16-OUT (#1168) removed the `bool dense_model` parameter this used to
// default to. It resolved bf16 for a dense checkpoint and f32 for a MoE one, and
// all three call sites passed `cfg.num_experts == 0`, so every MoE checkpoint
// carried `dcore`, `z` and the gated-norm weight at double width. Upstream does
// not branch on model shape anywhere on this path — `Qwen3_5ForCausalLMBase`
// (vllm/model_executor/models/qwen3_5.py:280-297 @ 5559679) is the shared base of
// the dense and MoE causal-LM arms — and the deferral quoted above ("keep every
// unmeasured 35B arm on its prior f32 default") named its own successor campaign,
// which this is. The parameter is gone rather than defaulted because a signature
// that accepts a model shape makes the default unreadable at the definition and
// lets a new call site reintroduce the split silently. VT_GDN_OUT_BF16=0 is now
// the f32 rollback for BOTH arms rather than for the dense one alone.
//
// Fresh-review repair: the definition moved up beside `GdnOutBf16FlagIsOn` and
// into `detail::`, so that a gate can observe the RESOLVER and not only its
// parser. Nothing about the resolution changed. The call sites below are
// unqualified and keep reading it through this declaration.
using detail::GdnOutDType;

// bf16 residual stream (default ON). vLLM runs the 35B in bf16
// (model_config.dtype=bfloat16): qwen3_next.py keeps `residual` as the bf16 hidden
// dtype across all 48 layers, and Qwen3NextRMSNorm==GemmaRMSNorm's fused_add_rms_norm
// adds x into the residual and stores it bf16 while accumulating the variance in f32
// (csrc/layernorm_kernels.cu). OUR residual was f32 — a more-precise accepted
// deviation that doubled the residual memory traffic (read+write per RmsNorm, 2×
// per layer). Making `res` bf16 mirrors vLLM exactly (the RmsNorm kernel keeps its
// f32 variance/normalize accumulation regardless — only the residual load/store
// dtype changes) and halves that traffic. A/B: VT_BF16_RESIDUAL=0 restores the f32
// residual in the same binary.
// DISPOSITION, measured 2026-09-02 (#2534), so nobody re-runs this experiment:
// the bf16 default STAYS, and the `VT_BF16_RESIDUAL=0` rollback was tried
// against the Qwen3.8-27B Q4_K_M token gate and DID NOT HELP. Same binary, same
// artifact and recipe, one lever: the divergence rate held at 5 of 6 and two
// prompts diverged EARLIER (first difference 34 -> 21 and 20 -> 4), so the
// agreeing-prefix total fell from 155 tokens to 126. The lever is live -- the
// outputs changed -- it simply does not buy correctness on this arm. So this
// knob remains what its comment above says it is: a MEMORY-TRAFFIC choice that
// mirrors vLLM's bf16 residual, plus a diagnostic A/B. It is not a correctness
// lever and must not be reached for as one.
//
// #2534 also wired the trunk in: when `ActDType` resolves f32 the residual
// inherits it, because a residual narrower than the stream it accumulates is the
// same lost mantissa twice per layer. That arm is currently REFUSED at the
// resolver (see ActDType), so this term is bf16-unless-`VT_BF16_RESIDUAL=0`.
DType ResidualDType(Dev d) {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_BF16_RESIDUAL");
    return e == nullptr || e[0] != '0';
  }();
  if (ActDType(d) == DType::kF32) return DType::kF32;
  return bf16 ? DType::kBF16 : DType::kF32;
}

// vLLM's Qwen3.5/3.6 GDN owns one physical `in_proj_ba` and invokes it once,
// then exposes logical [b,a] views. W1 enabled that topology for the real 27B
// loader; GDN-MOE-PACKED-BA (#1169) made the MoE safetensors loader populate the
// same owner, so both dense and MoE safetensors checkpoints reach it (the GGUF
// MoE loader still keeps the split pair, #1793). The resident owner is shared by
// both arms: fallback slices its output rows and issues the two legacy F32
// GEMMs, never retaining duplicate split weights.
// The decomposed fallback emits F32 by default, preserving the already-gated
// token-correct stream. vLLM emits BF16 from torch.nn.functional.linear; W1D2
// couples that exact dtype to the packed pure-decode branch. Packed activations
// and output are BF16 while temporal state, A_log, and dt_bias retain their
// independent upstream dtypes. VT_GDN_BA_OUT_BF16 remains an explicit
// diagnostic override. With
// VT_GDN_PACKED_DECODE=0 and no dtype override, rollback is therefore the
// legacy F32 BA + decomposed consumer from the same resident owner.
bool MergedGdnBaEnabled(Dev d) {
  static const bool enabled = [] {
    const char* master = std::getenv("VT_GDN_MERGED_PROJ");
    if (master != nullptr && master[0] == '0') return false;
    const char* leaf = std::getenv("VT_GDN_MERGED_BA");
    return leaf == nullptr || leaf[0] != '0';
  }();
  return enabled &&
         vllm::platforms::GetPlatform(d.q.device.type).needs_weight_staging();
}

DType MergedGdnBaOutputDType(bool packed_decode) {
  static const int override = [] {
    const char* e = std::getenv("VT_GDN_BA_OUT_BF16");
    if (e == nullptr) return -1;
    return e[0] == '0' ? 0 : 1;
  }();
  const bool bf16 = override >= 0 ? override != 0 : packed_decode;
  return bf16 ? DType::kBF16 : DType::kF32;
}

// Mirrors vLLM's VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE default-on process
// configuration (envs.py:1123-1125 @ 702f4814). Cached once: a process is one
// production or rollback arm, never a per-request mixture.
bool PackedGdnDecodeRuntimeEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_GDN_PACKED_DECODE");
    return e == nullptr || e[0] != '0';
  }();
  return enabled;
}

// PERF-27B-GDN-PACKED-REACHABLE (#365), DEFAULT OFF. Process-cached like every
// sibling GDN toggle: a process is one arm, never a per-request mixture.
bool PackedGdnDecodeFp8TowerEnabled() {
  static const bool enabled = detail::PackedGdnDecodeFp8TowerFlagIsOn(
      std::getenv("VT_GDN_PACKED_DECODE_FP8_TOWER"));
  return enabled;
}

// PERF-27B-GDN-PACKED-REACHABLE (#365) — the SINGLE source for the dtype the
// native-FP8 GDN input projection writes for `mixed_qkv`. `ProjectGdnQkvz` reads
// it instead of hardcoding a literal, and the packed-decode eligibility reads it
// to predict `mixed.dtype` before the projection has run, so the two cannot
// drift. Today the fp8 cutlass epilogue always emits F32 (the merged arm's
// `MergedFp8QkvzD` allocates F32 as well, VT_CHECKed at its call site). Making
// it BF16 is PERF-27B-BF16-FP8-OUT / #339 (VT_BF16_GEMM_OUT_FP8); when that
// lands, THIS function is the one line the two rows compose through, and the
// packed path becomes eligible on the fp8 27B with no further change here.
DType GdnFp8MergedInProjDType(DType indt, DType outdt) {
  return detail::GdnFp8MergedMixedQkvDType(GdnFp8InBf16Enabled(), indt, outdt);
}

DBuf MatmulBTRawD(Dev d, const Tensor& x, const Tensor& weight,
                  DType out_dtype) {
  VT_CHECK(x.rank == 2 && weight.rank == 2,
           "qwen3_5 merged GDN proj: input/weight must be rank-2");
  VT_CHECK(weight.shape[1] == x.shape[1],
           "qwen3_5 merged GDN proj: input/weight K mismatch");
  DBuf out(d, out_dtype, {x.shape[0], weight.shape[0]});
  if (const char* qdir = std::getenv("VT_DUMP_QKVZ")) {
    static std::atomic<int> mmseq{0};
    const int call = mmseq.fetch_add(1, std::memory_order_relaxed);
    auto cap = [&](const char* tag, const void* bytes, size_t n) {
      std::FILE* f = std::fopen(
          (std::string(qdir) + "/mm" + std::to_string(call) + "_" + tag + ".bin")
              .c_str(), "wb");
      if (f) { std::fwrite(bytes, 1, n, f); std::fclose(f); }
    };
    // In-process arbitration: hash-and-capture x BEFORE, run the REAL GEMM,
    // capture out, then run a SHADOW GEMM from the same tensor and capture
    // its output, then re-read x. Answers, without cross-process
    // assumptions: did the real GEMM consume these bytes, is consumption
    // deterministic, and does x change across the op?
    std::vector<uint8_t> xpre(static_cast<size_t>(x.Numel()) * vt::SizeOf(x.dtype));
    d.b.Synchronize(d.q);
    d.b.Copy(d.q, xpre.data(), x.data, xpre.size());
    vt::MatmulBT(d.q, out.t(), x, weight);
    std::vector<uint8_t> ore(static_cast<size_t>(out.t().Numel()) *
                             vt::SizeOf(out_dtype));
    d.b.Synchronize(d.q);
    d.b.Copy(d.q, ore.data(), out.t().data, ore.size());
    std::vector<uint8_t> xpost(static_cast<size_t>(x.Numel()) * vt::SizeOf(x.dtype));
    d.b.Copy(d.q, xpost.data(), x.data, xpost.size());
    DBuf shadow(d, out_dtype, {x.shape[0], weight.shape[0]});
    vt::MatmulBT(d.q, shadow.t(), x, weight);
    std::vector<uint8_t> osh(static_cast<size_t>(shadow.t().Numel()) *
                             vt::SizeOf(out_dtype));
    d.b.Synchronize(d.q);
    d.b.Copy(d.q, osh.data(), shadow.t().data, osh.size());
    cap("xpre", xpre.data(), xpre.size());
    cap("xpost", xpost.data(), xpost.size());
    cap("out_real", ore.data(), ore.size());
    cap("out_shadow", osh.data(), osh.size());
    return out;
  }
  vt::MatmulBT(d.q, out.t(), x, weight);
  return out;
}

// ── DEFERRED V-HEAD PERMUTATION (MODEL-MM-QWEN4-EXP,
//    ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM) ─────────────────────────────────
//
// INERT unless `w.v_head_perm_key_heads` is non-zero, which only the `qwen4exp`
// GGUF loader sets. See `GdnLayerWeights::v_head_perm_key_heads` for why it
// exists; in one line, undoing the converter's tiled V-head order on the WEIGHT
// forces a k-quant superblock to be dequantized at load, and the identity
// `(P W) x = P (W x)` lets the same re-indexing happen on the VECTOR instead.
//
// Returns an OWNING permuted copy, or nullopt when nothing is deferred (the
// caller then keeps the tensor it had, byte for byte). The copy is out of
// place because `vt::VHeadPermute` is a gather and the map is not an involution
// at the released 16-vs-48 ratio.
std::optional<DBuf> PermuteVHeadsIfDeferred(Dev d, const GdnLayerWeights& w,
                                            const Tensor& src,
                                            int64_t value_heads,
                                            int64_t prefix_elems,
                                            bool inverse) {
  if (w.v_head_perm_key_heads == 0) return std::nullopt;
  const int64_t kk = w.v_head_perm_key_heads;
  VT_CHECK(value_heads > 0 && kk > 0 && value_heads % kk == 0,
           "gdn deferred V-head permutation: value_heads is not a multiple of "
           "the key-head count");
  VT_CHECK(src.rank == 2, "gdn deferred V-head permutation: expected [T, N]");
  const int64_t n = src.shape[1];
  const int64_t body = n - prefix_elems;
  VT_CHECK(prefix_elems >= 0 && body > 0 && body % value_heads == 0,
           "gdn deferred V-head permutation: [T, N] does not split into a "
           "prefix plus whole V heads");
  DBuf out(d, src.dtype, {src.shape[0], n});
  Tensor o = out.t();
  vt::VHeadPermute(d.q, o, src,
                   vt::VHeadPermuteArgs{prefix_elems, kk, value_heads / kk,
                                        body / value_heads, inverse});
  return out;
}

struct GdnBaOutput {
  std::optional<DBuf> packed_owner;
  std::optional<DBuf> b_owner;
  std::optional<DBuf> a_owner;
  // Owners of the deferred V-head permutation's output (ProjectGdnBA). Empty on
  // every checkpoint but the `qwen4exp` GGUF one. The owners above are kept
  // alongside them, never replaced.
  std::optional<DBuf> b_perm_owner;
  std::optional<DBuf> a_perm_owner;
  Tensor b;
  Tensor a;
};

GdnBaOutput ProjectGdnBARaw(Dev d, const GdnLayerWeights& weights,
                            const Tensor& hidden, int64_t value_heads,
                            bool packed_decode) {
  GdnBaOutput out;
  if (!weights.in_proj_ba.Empty()) {
    VT_CHECK(weights.in_proj_ba.nk && weights.in_proj_ba.rank == 2 &&
                 weights.in_proj_ba.shape[0] == 2 * value_heads &&
                 weights.in_proj_ba.shape[1] == hidden.shape[1],
             "qwen3_5 merged GDN BA: invalid packed owner");
    Tensor packed_weight = ResidentWeight(d, weights.in_proj_ba);
    if (const char* qdir = std::getenv("VT_DUMP_QKVZ")) {
      static std::atomic<int> baseq{0};
      const int call = baseq.fetch_add(1, std::memory_order_relaxed);
      if (call == 0) {
        std::vector<uint8_t> raw(static_cast<size_t>(packed_weight.Numel()) *
                                 vt::SizeOf(packed_weight.dtype));
        DBuf tmp(d, packed_weight.dtype, {packed_weight.Numel()}, packed_weight.data);
        d.b.Copy(d.q, tmp.ptr(), packed_weight.data, raw.size());
        tmp.Download(d, raw.data());
        std::FILE* f = std::fopen((std::string(qdir) + "/w_ba.bin").c_str(), "wb");
        if (f) { std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f); }
      }
    }
    if (MergedGdnBaEnabled(d)) {
      out.packed_owner.emplace(
          MatmulBTRawD(d, hidden, packed_weight,
                       MergedGdnBaOutputDType(packed_decode)));
      Tensor packed = out.packed_owner->t();
      out.b = packed.Slice(1, 0, value_heads);
      out.a = packed.Slice(1, value_heads, 2 * value_heads);
      if (const char* td = std::getenv("VT_DUMP_TRUST")) {
        static std::atomic<int> ba_seq{0};
        if (ba_seq.fetch_add(1, std::memory_order_relaxed) == 0) {
          // Device truth of the packed matmul result and of the interior
          // a-window, captured through the same verified instrument the
          // kernel-side probes use.
          vt::tenstorrent::TrustDump(d.q, td, "ba_packed_dev", packed);
          vt::tenstorrent::TrustDump(d.q, td, "ba_a_win_dev", out.a);
          vt::tenstorrent::TrustDump(d.q, td, "ba_b_win_dev", out.b);
        }
      }
      return out;
    }

    Tensor b_weight = packed_weight.Slice(0, 0, value_heads);
    Tensor a_weight = packed_weight.Slice(0, value_heads, 2 * value_heads);
    // HF transformers' nn.Linear outputs the model dtype (BF16) for a/b.
    // The merged-CUDA arm keeps F32 (FLA's split), but the split arm is
    // CPU-only and must round to match the reference.
    const DType ba_dt = GdnInDType();
    out.b_owner.emplace(MatmulBTRawD(d, hidden, b_weight, ba_dt));
    out.a_owner.emplace(MatmulBTRawD(d, hidden, a_weight, ba_dt));
  } else {
    VT_CHECK(!weights.in_proj_b.Empty() && !weights.in_proj_a.Empty(),
             "qwen3_5 GDN BA: packed or both split weights are required");
    // HF transformers' nn.Linear outputs BF16 for a/b in BF16 mode.
    const DType ba_dt = GdnInDType();
    out.b_owner.emplace(ba_dt == DType::kBF16
                            ? MatmulBf16D(d, hidden, weights.in_proj_b)
                            : MatmulF32D(d, hidden, weights.in_proj_b));
    out.a_owner.emplace(ba_dt == DType::kBF16
                            ? MatmulBf16D(d, hidden, weights.in_proj_a)
                            : MatmulF32D(d, hidden, weights.in_proj_a));
  }
  out.b = out.b_owner->t();
  out.a = out.a_owner->t();
  if (const char* td = std::getenv("VT_DUMP_TRUST")) {
    static std::atomic<int> bas_seq{0};
    if (bas_seq.fetch_add(1, std::memory_order_relaxed) == 0) {
      vt::tenstorrent::TrustDump(d.q, td, "sba_a_dev", out.a);
      vt::tenstorrent::TrustDump(d.q, td, "sba_b_dev", out.b);
      vt::tenstorrent::TrustDump(d.q, td, "sba_h_dev", hidden);
      Tensor sba_wa = ResidentWeight(d, weights.in_proj_a);
      vt::tenstorrent::TrustDump(d.q, td, "sba_wa_dev", sba_wa);
    }
  }
  return out;
}

// The production entry point. `ProjectGdnBARaw` above is every existing arm,
// unchanged; this wrapper adds the deferred V-head permutation and NOTHING else
// — with `v_head_perm_key_heads == 0` (every loader but `qwen4exp`'s GGUF one)
// `PermuteVHeadsIfDeferred` returns nullopt and the views below are the raw
// ones, byte for byte.
//
// `b` and `a` are [T, Hv]: one element per V head, so the head width is 1 and
// there is no prefix. Their consumers — `vt::GdnGBeta`, `vt::GdnPostConv` and
// `vt::GdnPackedDecode` — pair them ELEMENTWISE with `a_log` and `dt_bias`,
// which the loader still permutes at load, so both sides are grouped here.
GdnBaOutput ProjectGdnBA(Dev d, const GdnLayerWeights& weights,
                         const Tensor& hidden, int64_t value_heads,
                         bool packed_decode = false) {
  GdnBaOutput out =
      ProjectGdnBARaw(d, weights, hidden, value_heads, packed_decode);
  if (weights.v_head_perm_key_heads == 0) return out;
  out.b_perm_owner = PermuteVHeadsIfDeferred(d, weights, out.b, value_heads,
                                             /*prefix_elems=*/0,
                                             /*inverse=*/false);
  out.a_perm_owner = PermuteVHeadsIfDeferred(d, weights, out.a, value_heads,
                                             /*prefix_elems=*/0,
                                             /*inverse=*/false);
  out.b = out.b_perm_owner->t();
  out.a = out.a_perm_owner->t();
  // The raw owners stay alive in `out` on purpose: the permutation above is
  // queued, not complete, so returning their pool blocks here would hand a
  // later allocation the memory this gather still reads.
  return out;
}


// --- PERF-27B-GDN-FP8-QKVZ: the FP8 leaf of the merged GDN input projection.
// The BF16 leaf below owns a merged `in_proj_qkvz` parameter; a ModelOpt FP8
// tower (`nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed`; the 35B shares the
// tower) keeps the two shards native, so the loader leaves that owner empty and
// this arm merges the RAW fp8 bytes instead. Same upstream behavior being
// mirrored: MergedColumnParallelLinear packs qkv+z along N and ONE GEMM runs
// per GDN layer (qwen_gdn_linear_attn.py:923-936, linear.py:580-636 @ 702f4814).

// VT_GDN_MERGED_QKVZ_FP8, DEFAULT ON. Also honors the BF16 leaf's rollbacks so a
// single switch turns the whole merged-input-projection topology off: master
// VT_GDN_MERGED_PROJ=0 or leaf VT_GDN_MERGED_QKVZ=0 disables this arm too.
// Process-cached, resolved outside the hot loop.
bool MergedGdnFp8QkvzEnabled() {
  static const bool enabled = [] {
    return detail::MergedGdnFp8QkvzEnvSelected(
        detail::GdnMergedFp8QkvzEnvConfig{
            std::getenv("VT_GDN_MERGED_PROJ"),
            std::getenv("VT_GDN_MERGED_QKVZ"),
            std::getenv("VT_GDN_MERGED_QKVZ_FP8")});
  }();
  return enabled;
}

detail::GdnMergedFp8QkvzEligibility GdnMergedFp8QkvzEligibilityFor(
    Dev d, const GdnLayerWeights& w, int64_t conv_dim, int64_t value_dim) {
  detail::GdnMergedFp8QkvzEligibility e;
  e.runtime_enabled = MergedGdnFp8QkvzEnabled();
  e.fp8_platform =
      vllm::platforms::GetPlatform(d.q.device.type).supports_fp8() &&
      vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, d.q.device.type);
  e.has_fp8_shards = !w.in_proj_qkv_fp8.Empty() && !w.in_proj_z_fp8.Empty();
  e.shared_k = e.has_fp8_shards && w.in_proj_qkv_fp8.k == w.in_proj_z_fp8.k;
  e.shared_input_scale = detail::GdnFp8SharedInputScale(w, nullptr);
  e.shard_widths_match = e.has_fp8_shards &&
                         w.in_proj_qkv_fp8.n == conv_dim &&
                         w.in_proj_z_fp8.n == value_dim;
  return e;
}

// The resident N-concatenated [qkv;z] fp8 operand + its column-alpha policy.
// Mirrors ResidentFp8Qkv (the attention QKV sibling) byte for byte in structure.
struct Fp8QkvzDev {
  Tensor packed;     // i8 [conv_dim+value_dim, K] raw e4m3fn (K contiguous)
  Tensor alpha_vec;  // f32 [conv_dim+value_dim]; valid only when !folded
  float alpha = 1.0F;  // the GEMM scalar (the shared folded alpha, or 1)
  bool folded = false;
};

// Build (lazily, ONCE — and eagerly pre-capture via PrepareGdnFp8Resident) the
// merged operand. The two shards' packed rows are byte-concatenated: fp8 e4m3 is
// a raw byte encoding read in [N,K] orientation, so concatenating along N is
// lossless and needs no repack. The per-tensor input_scale must already be
// shared (the caller's eligibility guarantees it, re-checked here). Each shard's
// folded alpha (= shared input_scale * that shard's weight_scale) is applied per
// OUTPUT COLUMN: folded into the GEMM scalar when both shards fold the same
// alpha — the byte-exact case, no extra launch — else through the resident
// per-column vector, exactly as MergedFp8QkvD does.
Fp8QkvzDev ResidentFp8Qkvz(Dev d, const GdnLayerWeights& w) {
  const Fp8Weight& qkv = w.in_proj_qkv_fp8;
  const Fp8Weight& z = w.in_proj_z_fp8;
  VT_CHECK(!qkv.Empty() && !z.Empty(),
           "qwen3_5 merged FP8 GDN qkvz: empty logical shard");
  VT_CHECK(qkv.k == z.k, "qwen3_5 merged FP8 GDN qkvz: logical shard K mismatch");
  VT_CHECK(qkv.input_scale == z.input_scale,
           "qwen3_5 merged FP8 GDN qkvz: shards do not share one input_scale");
  const int64_t inner_k = qkv.k;
  const int64_t total_n = qkv.n + z.n;
  const size_t qpb = qkv.packed.bytes.size();
  const size_t zpb = z.packed.bytes.size();
  VT_CHECK(qpb == static_cast<size_t>(qkv.n * inner_k) &&
               zpb == static_cast<size_t>(z.n * inner_k),
           "qwen3_5 merged FP8 GDN qkvz: packed shard byte mismatch");
  const bool folded = qkv.alpha == z.alpha;

  if (!w.d_qkvz_fp8_packed) {
    Backend* backend = &d.b;
    void* packed_data = d.b.Alloc(qpb + zpb);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* dst = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, dst, qkv.packed.bytes.data(), qpb);
    d.b.Copy(d.q, dst + qpb, z.packed.bytes.data(), zpb);
    if (!folded) {
      std::vector<float> alpha_host(static_cast<size_t>(total_n));
      std::fill(alpha_host.begin(), alpha_host.begin() + qkv.n, qkv.alpha);
      std::fill(alpha_host.begin() + qkv.n, alpha_host.end(), z.alpha);
      void* alpha_data = d.b.Alloc(static_cast<size_t>(total_n) * sizeof(float));
      std::shared_ptr<void> alpha_owner(
          alpha_data, [backend](void* pointer) { backend->Free(pointer); });
      d.b.Copy(d.q, alpha_data, alpha_host.data(),
               alpha_host.size() * sizeof(float));
      w.d_qkvz_fp8_alpha = std::move(alpha_owner);
    }
    w.d_qkvz_fp8_packed = std::move(packed_owner);
  }

  Fp8QkvzDev out;
  out.packed = MakeTensor(w.d_qkvz_fp8_packed.get(), DType::kI8, d.q.device,
                          {total_n, inner_k});
  out.folded = folded;
  out.alpha = folded ? qkv.alpha : 1.0F;
  if (!folded) {
    VT_CHECK(static_cast<bool>(w.d_qkvz_fp8_alpha),
             "qwen3_5 merged FP8 GDN qkvz: partial resident state");
    out.alpha_vec = MakeTensor(w.d_qkvz_fp8_alpha.get(), DType::kF32,
                               d.q.device, {total_n});
  }
  return out;
}

// ONE fp8 GEMM over the N-concatenated [qkv;z] operand -> `want` [M, conv_dim +
// value_dim]. `h_fp8` is the shared pre-quantized activation (quantize-once)
// when supplied, else the activation is quantized here with the shared
// input_scale — the SAME activation bytes both split GEMMs would have read,
// which is why one shared input_scale is a hard precondition.
//
// `want` is f32 (the shipped default) or bf16. At f32 each column's alpha is
// applied by the same IEEE f32 multiply the folded-alpha GEMM would apply, so
// the merged result is identical to the concatenation of the two split f32 GEMM
// outputs — the #213 equivalence this leaf shipped with, unchanged.
//
// PERF-FP8-ALPHA-FOLD / #417: `want == kBF16` narrows the GEMM's D, which is the
// ONLY remaining lever on the unfolded arm's per-column alpha pass — that pass is
// a bandwidth-saturated full read-modify-write, so halving its width halves its
// cost, and all three cuBLASLt routes to removing it outright are measured
// unavailable on this hardware. It applies to BOTH arms uniformly and needs no
// `folded` condition, because vt::MulColVecF32 now carries a bf16 store arm; the
// alpha is still the same f32 multiply against the same f32 resident vector.
// It is NOT byte-preserving — the accumulator is rounded to bf16 before the alpha
// rather than after — so it is reached only through GdnFp8InBf16Enabled(), which
// is DEFAULT OFF. Callers must read the returned buffer's dtype, never assume f32.
DBuf MergedFp8QkvzD(Dev d, const Tensor& x, const Tensor* h_fp8,
                    const GdnLayerWeights& w, DType want = DType::kF32) {
  VT_CHECK(want == DType::kF32 || want == DType::kBF16,
           "qwen3_5 merged FP8 GDN qkvz: output dtype must be f32 or bf16");
  Fp8QkvzDev qkvz = ResidentFp8Qkvz(d, w);
  const int64_t M = h_fp8 != nullptr ? h_fp8->shape[0] : x.shape[0];
  const int64_t total_n = qkvz.packed.shape[0];
  DBuf out(d, want, {M, total_n});
  const Tensor* a_fp8_p = h_fp8;
  std::optional<DBuf> a_fp8_owner;
  if (a_fp8_p == nullptr) {
    const int64_t K = x.shape[1];
    a_fp8_owner.emplace(d, DType::kI8, std::vector<int64_t>{M, K});
    vt::QuantFp8Static(d.q, a_fp8_owner->t(), x, w.in_proj_qkv_fp8.input_scale);
    a_fp8_p = &a_fp8_owner->t();
  }
  // THIS is the call site that claims the splitK=1 premise, and the only one in
  // the tree (#339, review finding F-A). `want == kBF16` here is reachable ONLY
  // through GdnFp8InBf16Enabled() — the leaf above resolves `fp8_indt` to bf16
  // under that toggle and nothing else — and that lever's whole argument is that
  // its bf16 D is the f32 D narrowed at the STORE, nothing more. A split-K plan
  // would silently make it a reduction-order change instead, which a bf16 store
  // is very good at hiding from a token gate, so the op verifies it and refuses.
  //
  // The f32 arm passes false: it is the baseline the claim is made AGAINST, so
  // it claims nothing. Neither does any other bf16-D fp8 GEMM in this file —
  // o_proj_fp8 / out_proj_fp8 go through MatmulFp8Cutlass{,PreQuant}D at
  // DType::kBF16 on a DEFAULT-ON path and simply want a bf16 output; split-K is
  // correct for them and they are never checked.
  const bool claims_splitk1 = want == DType::kBF16;
  if (qkvz.folded) {
    // One shared alpha: already a single GEMM scalar, nothing to apply after.
    if (DenseCublasLtFp8Enabled())
      vt::MatmulFp8CublasLt(d.q, out.t(), *a_fp8_p, qkvz.packed, qkvz.alpha, claims_splitk1);
    else
      vt::MatmulFp8Cutlass(d.q, out.t(), *a_fp8_p, qkvz.packed, qkvz.alpha);
  } else if (DenseCublasLtFp8Enabled()) {
    // PERF-FP8-ALPHA-FOLD: express the per-column alpha ONCE, at the seam. The
    // op applies it in the cuBLASLt epilogue when VT_FP8_ALPHA_VEC_EPILOGUE=1
    // and the selected algo supports the pointer mode, and otherwise runs the
    // GEMM at alpha=1 plus vt::MulColVecF32 — byte for byte what this line pair
    // did before. The model no longer owns that choice.
    vt::MatmulFp8CublasLtAlphaVec(d.q, out.t(), *a_fp8_p, qkvz.packed, qkvz.alpha_vec,
                                  claims_splitk1);
  } else {
    // CUTLASS arm (VT_DENSE_CUBLASLT_FP8=0): no epilogue alpha vector, so the
    // two-launch form stays here verbatim. Both launches follow `out`'s dtype,
    // so this arm narrows with `want` exactly as the cuBLASLt one does — and bf16
    // is CUTLASS's NATIVE fp8 epilogue, so it also drops the bf16 scratch and the
    // CastBf16ToF32 pass an f32 `out` costs here.
    vt::MatmulFp8Cutlass(d.q, out.t(), *a_fp8_p, qkvz.packed, qkvz.alpha);
    vt::MulColVecF32(d.q, out.t(), qkvz.alpha_vec);
  }
  return out;
}

// vLLM's Qwen3.5/3.6 GDN owns one physical `in_proj_qkvz` and invokes it once,
// then exposes logical [mixed_qkv, z] last-dim views
// (qwen_gdn_linear_attn.py:923-936 @ 702f4814). W2 enables that topology only
// for the real 27B loader — the only path that populates `in_proj_qkvz`. The
// resident owner is shared by both arms: the fallback slices its output ROWS
// (dim-0 raw-NK slices stay contiguous) and issues the two legacy GEMMs at
// their independent dtypes, never retaining duplicate split weights. The
// merged arm requires one uniform output dtype (GdnInDType == GdnOutDType, and
// since GDN-MOE-BF16-OUT (#1168) that is BF16/BF16 on every arm, matching vLLM's
// model-dtype projection). The uniformity term therefore no longer excludes a
// MoE checkpoint; `has_packed_qkvz` still does, because no MoE or GGUF loader
// builds the merged `in_proj_qkvz` owner either — the same gap as #1169's.
// VT_GDN_MERGED_QKVZ=0 (or master VT_GDN_MERGED_PROJ=0) restores the split
// GEMMs from the same binary and the same resident owner.
bool MergedGdnQkvzEnabled(Dev d) {
  static const bool enabled = [] {
    const char* master = std::getenv("VT_GDN_MERGED_PROJ");
    if (master != nullptr && master[0] == '0') return false;
    const char* leaf = std::getenv("VT_GDN_MERGED_QKVZ");
    return leaf == nullptr || leaf[0] != '0';
  }();
  return enabled &&
         vllm::platforms::GetPlatform(d.q.device.type).needs_weight_staging();
}

struct GdnQkvzOutput {
  std::optional<DBuf> packed_owner;
  std::optional<DBuf> mixed_owner;
  std::optional<DBuf> z_owner;
  // Owners of the deferred V-head permutation's output (ProjectGdnQkvz). Empty
  // on every checkpoint but the `qwen4exp` GGUF one. The owners above are kept
  // alongside them, never replaced.
  std::optional<DBuf> mixed_perm_owner;
  std::optional<DBuf> z_perm_owner;
  Tensor mixed;  // [T, conv_dim]; inner-contiguous, row stride may be padded
  Tensor z;      // [T, value_dim]; inner-contiguous, row stride may be padded
};

// Input projections (mixed_qkv | z), shared by the dense and paged GDN blocks.
// Packed 27B owner: ONE BF16 GEMM + logical views when merged is selected,
// else two split GEMMs sliced from the same owner. Legacy owners: W8A8 cutlass
// fp8 (35B) when populated — qkv/z read the shared pre-quantized fp8
// activation (h_fp8, quantize-once) when supplied — else bf16 (GGUF/synthetic;
// mixed at GdnInDType, z at GdnOutDType).
GdnQkvzOutput ProjectGdnQkvzRaw(Dev d, const GdnLayerWeights& w,
                                const Tensor& h, int64_t conv_dim,
                                int64_t value_dim, DType indt,
                                DType outdt, const Tensor* h_fp8) {
  GdnQkvzOutput out;
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4). FIRST and EXCLUSIVE, mirroring the
  // loader rung: an EXL3 load populates NO other in-projection field, so every
  // branch below would fall through to an empty owner and refuse by name.
  //
  // TWO GEMMs, because the artifact ships two independently quantized
  // trellises (see `GdnLayerWeights::in_proj_qkv_exl3`). This is not a
  // regression against vLLM's single merged `in_proj_qkvz` GEMM: there is no
  // merged trellis operand to issue, and the split native-FP8 arm the 35B runs
  // already takes this same shape when the merged owner is empty.
  //
  // `indt` for mixed and `outdt` for z are the dtypes the SPLIT BF16 arm below
  // emits, so `VT_GDN_IN_BF16`'s and `VT_GDN_OUT_BF16`'s documented rollbacks
  // stay honest on this arm too. `dense_exl3::Linear` forwards to
  // `layers::MakeLinearMethod` -> `Exl3LinearMethod` -> the ONE
  // `dense_attn::Exl3MatmulD`; no second EXL3 matmul exists.
  if (!w.in_proj_qkv_exl3.Empty()) {
    VT_CHECK(w.in_proj_qkv_exl3.OutFeatures() == conv_dim &&
                 w.in_proj_z_exl3.OutFeatures() == value_dim &&
                 w.in_proj_qkv_exl3.InFeatures() == h.shape[1],
             "qwen3_5 EXL3 GDN qkvz: the trellis geometry and the layer's "
             "conv_dim/value_dim disagree");
    out.mixed_owner.emplace(
        dense_exl3::Linear(d, h, w.in_proj_qkv, w.in_proj_qkv_exl3, indt));
    out.z_owner.emplace(
        dense_exl3::Linear(d, h, w.in_proj_z, w.in_proj_z_exl3, outdt));
    // The same anti-drift guard the merged fp8 arm carries, read off the
    // ALLOCATED buffer rather than off a literal, so it fails whichever side
    // moves. The packed-decode eligibility runs before this function and must
    // have predicted exactly this dtype.
    VT_CHECK(out.mixed_owner->t().dtype ==
                 detail::GdnProjectedMixedQkvDType(
                     detail::GdnMixedQkvDTypeInputs{
                         !w.in_proj_qkvz.Empty(), !w.in_proj_qkv_fp8.Empty(),
                         /*fp8_merged_arm=*/false, indt,
                         GdnFp8MergedInProjDType(indt, outdt),
                         /*has_exl3_qkv_owner=*/true}),
             "qwen3_5 EXL3 GDN qkvz: the allocated mixed_qkv dtype and the "
             "packed-decode prediction disagree (GdnProjectedMixedQkvDType)");
    out.mixed = out.mixed_owner->t();
    out.z = out.z_owner->t();
    return out;
  }
  if (!w.in_proj_qkvz.Empty()) {
    VT_CHECK(w.in_proj_qkvz.nk && w.in_proj_qkvz.rank == 2 &&
                 w.in_proj_qkvz.shape[0] == conv_dim + value_dim &&
                 w.in_proj_qkvz.shape[1] == h.shape[1],
             "qwen3_5 merged GDN qkvz: invalid packed owner");
    Tensor packed_weight = ResidentWeight(d, w.in_proj_qkvz);
    if (const char* qdir = std::getenv("VT_DUMP_QKVZ")) {
      // Per-invocation replay capture (the engine WARMS UP with a dummy
      // forward, so the FIRST call is not the real step): h per invocation,
      // the resident merged weight once (device-independent).
      static std::atomic<int> qseq{0};
      const int call = qseq.fetch_add(1, std::memory_order_relaxed);
      const std::string dir = qdir;
      if (call == 0) {
        const Tensor& wt = packed_weight;
        // HOST master bytes...
        std::vector<uint8_t> raw(static_cast<size_t>(wt.Numel()) * vt::SizeOf(wt.dtype));
        DBuf tmp(d, wt.dtype, {wt.Numel()}, wt.data);
        d.b.Copy(d.q, tmp.ptr(), wt.data, raw.size());
        tmp.Download(d, raw.data());
        std::FILE* f = std::fopen((dir + "/w_host.bin").c_str(), "wb");
        if (f) { std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f); }
        // ...and the DEVICE-STAGED copy the GEMM will actually consume,
        // read back through ttnn via the ops seam.
        {
          std::vector<float> vec =
              vt::tenstorrent::DebugDeviceReadbackF32(d.q, wt);
          std::FILE* f2 = std::fopen((dir + "/w_device.bin").c_str(), "wb");
          if (f2) {
            // issue #2021: `vec` may be empty, in which case `data()` may
            // return `nullptr` -- well-defined for a size-0 `fwrite` by the
            // C standard, but `fwrite`'s `nonnull` attribute makes GCC 15
            // flag the call itself under -Werror=nonnull regardless of the
            // runtime size. Guard rather than silence: there is nothing to
            // write for an empty readback either way.
            if (!vec.empty()) std::fwrite(vec.data(), 4, vec.size(), f2);
            std::fclose(f2);
          }
        }
      }
      {
        if (call == 0)
          std::fprintf(stderr, "[TT-DUMP] qkvz-h0 ptr=%p rows=%lld\n",
                       static_cast<const void*>(h.data), (long long)h.shape[0]);
        // NO DBuf here: a pool-backed tmp aliases live blocks (see the
        // capture-reliability finding); read straight into a host vector.
        std::vector<uint8_t> raw(static_cast<size_t>(h.Numel()) * vt::SizeOf(h.dtype));
        d.b.Synchronize(d.q);
        d.b.Copy(d.q, raw.data(), h.data, raw.size());
        std::FILE* f = std::fopen(
            (dir + "/h" + std::to_string(call) + ".bin").c_str(), "wb");
        if (f) { std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f); }
      }
    }
    if (detail::ShouldUseMergedGdnQkvz(detail::GdnMergedQkvzEligibility{
            MergedGdnQkvzEnabled(d),
            vllm::platforms::GetPlatform(d.q.device.type).needs_weight_staging(),
            true, indt == outdt})) {
      out.packed_owner.emplace(MatmulBTRawD(d, h, packed_weight, indt));
      Tensor packed = out.packed_owner->t();
      out.mixed = packed.Slice(1, 0, conv_dim);
      out.z = packed.Slice(1, conv_dim, conv_dim + value_dim);
      return out;
    if (const char* td = std::getenv("VT_DUMP_TRUST"))
      vt::tenstorrent::TrustDump(d.q, td, "packed", packed);
    }
    Tensor qkv_weight = packed_weight.Slice(0, 0, conv_dim);
    Tensor z_weight = packed_weight.Slice(0, conv_dim, conv_dim + value_dim);
    out.mixed_owner.emplace(MatmulBTRawD(d, h, qkv_weight, indt));
    out.z_owner.emplace(MatmulBTRawD(d, h, z_weight, outdt));
    out.mixed = out.mixed_owner->t();
    out.z = out.z_owner->t();
    return out;
    if (const char* td = std::getenv("VT_DUMP_TRUST")) {
      vt::tenstorrent::TrustDump(d.q, td, "mixed", out.mixed);
      vt::tenstorrent::TrustDump(d.q, td, "z", out.z);
    }
  }
  // PERF-FP8-ALPHA-FOLD / #417 — the output dtype of the merged fp8 in_proj leaf.
  // vLLM's ModelOpt fp8 linear emits the model dtype (bf16); ours hardcoded f32,
  // which is both what made GdnInDType() unreachable on an fp8 tower AND what
  // makes the unfolded arm's per-column alpha pass cost twice what it needs to
  // (see GdnFp8InBf16Enabled). All THREE conditions are required:
  //   1. the opt-in toggle (default OFF; no measurement exists yet, and this
  //      narrowing is not value-neutral),
  //   2. indt == BF16, i.e. VT_GDN_IN_BF16 is on (its default) — this IS the
  //      lever being unblocked, so honouring its rollback is mandatory,
  //   3. outdt == BF16, which keeps the whole chain dtype-uniform — the
  //      downstream contracts need that: vt::RmsNormGatedQuantFp8 requires
  //      gate.dtype == x.dtype (ops.cpp), and `z` below is exactly that gate.
  //      This term USED to read "confines the change to the 27B, because the
  //      35B is MoE and GdnOutDType(dense_model=false) is f32 there". #521
  //      asked for that correction and GDN-MOE-BF16-OUT (#1168) is what makes
  //      it wrong: outdt is BF16 on every arm now. What still keeps this leaf
  //      inert on the 35B is condition 1, the DEFAULT-OFF VT_GDN_FP8_IN_BF16
  //      toggle — a toggle term, not a model-shape one. Nothing moves by
  //      default in either merge order, but whoever turns that toggle on owns
  //      measuring the 35B too (#417).
  // PERF-GDN-PACKED-BRIDGE (#365): PERF-FP8-ALPHA-FOLD's three-term decision,
  // moved into the shared `GdnFp8MergedInProjDType` so the PRODUCER (this line,
  // which reaches MergedFp8QkvzD and allocates the buffer) and the PREDICTOR
  // (the packed-decode eligibility, which must answer BEFORE this runs because
  // the decision feeds ProjectGdnBA) are literally the same call and cannot
  // drift. The terms themselves are unchanged.
  const DType fp8_indt = GdnFp8MergedInProjDType(indt, outdt);
  // PERF-27B-GDN-FP8-QKVZ — the native-FP8 owner's merged arm. ONE fp8 GEMM
  // over the N-concatenated [qkv;z] operand replaces the two below; `mixed_qkv`
  // and `z` become last-dim views of its output, exactly as in the BF16 leaf.
  // At the f32 default the merged output is f32 — the dtype the split
  // `mixed_qkv` GEMM already emits — so `mixed_qkv` is byte-identical. `z`'s
  // split GEMM emits `outdt`; when that is not f32 the f32 view is cast, which
  // rounds the SAME f32 product the split GEMM's epilogue would have rounded.
  // Nothing about the split arithmetic changes, so this leaf is a pure
  // launch/shape change.
  if (!w.in_proj_qkv_fp8.Empty() &&
      detail::ShouldUseMergedGdnFp8Qkvz(
          GdnMergedFp8QkvzEligibilityFor(d, w, conv_dim, value_dim))) {
    if (g_gdn_fp8_inproj_debug_enabled.load(std::memory_order_acquire))
      g_gdn_fp8_inproj_merged.fetch_add(1, std::memory_order_relaxed);
    out.packed_owner.emplace(MergedFp8QkvzD(d, h, h_fp8, w, fp8_indt));
    Tensor packed = out.packed_owner->t();
    // PERF-GDN-PACKED-BRIDGE (#365) -- the replacement anti-drift guard.
    // The deleted one read `GdnFp8MixedQkvDType() == DType::kF32`: a property of
    // the PREDICTOR against a literal, which cannot observe the producer and so
    // passed unchanged once the merged arm began emitting bf16. This asserts the
    // invariant that actually matters -- the dtype the eligibility PREDICTED
    // equals the dtype this GEMM ALLOCATED -- read off the allocated buffer, so
    // it fails whichever side moves.
    VT_CHECK(packed.dtype == detail::GdnProjectedMixedQkvDType(
                                 detail::GdnMixedQkvDTypeInputs{
                                     !w.in_proj_qkvz.Empty(),
                                     !w.in_proj_qkv_fp8.Empty(),
                                     /*fp8_merged_arm=*/true, indt, fp8_indt}),
             "qwen3_5 merged FP8 GDN qkvz: the allocated mixed_qkv dtype and the "
             "packed-decode prediction disagree (GdnProjectedMixedQkvDType)");
    out.mixed = packed.Slice(1, 0, conv_dim);
    Tensor z_raw = packed.Slice(1, conv_dim, conv_dim + value_dim);
    // When the merged GEMM already emitted the z gate's dtype, the CastBf16 pass
    // disappears with it (#417): the strided view IS the bf16 gate, and the
    // gated-RMSNorm consumes a padded-row gate view natively (GdnGateView3, the
    // z_strided path the f32 arm and the BF16-weight leaf already take). The
    // f32-buffer arm below is the pre-existing behavior, unchanged.
    if (z_raw.dtype == outdt) {
      out.z = z_raw;
    } else {
      VT_CHECK(outdt == DType::kBF16 && z_raw.dtype == DType::kF32,
               "qwen3_5 merged FP8 GDN qkvz: unsupported z output dtype");
      out.z_owner.emplace(d, DType::kBF16,
                          std::vector<int64_t>{packed.shape[0], value_dim});
      vt::CastBf16(d.q, out.z_owner->t(), z_raw);
      out.z = out.z_owner->t();
    }
    return out;
  }
  if (!w.in_proj_qkv_fp8.Empty() &&
      g_gdn_fp8_inproj_debug_enabled.load(std::memory_order_acquire))
    g_gdn_fp8_inproj_split.fetch_add(2, std::memory_order_relaxed);
  // PERF-GDN-PACKED-BRIDGE (#365): the SPLIT fp8 arm stays F32. VT_GDN_FP8_IN_BF16
  // narrows the MERGED arm only (PERF-FP8-ALPHA-FOLD `fp8_indt`), so this literal
  // is correct rather than a drift risk -- and the eligibility predicts exactly
  // this, via GdnMixedQkvDTypeInputs::fp8_merged_arm == false. The guard below
  // asserts the two agree, so if either side ever moves the build fails loudly.
  const DType fp8_mixed_dt = DType::kF32;
  // Same invariant on the other arm, and in the direction that matters here: a
  // predictor that ever claimed BF16 for a split-arm layer would hand
  // vt::GdnPackedDecode an f32 mixed_qkv against bf16 a/b/out and make it throw.
  if (!w.in_proj_qkv_fp8.Empty())
    VT_CHECK(fp8_mixed_dt == detail::GdnProjectedMixedQkvDType(
                                 detail::GdnMixedQkvDTypeInputs{
                                     !w.in_proj_qkvz.Empty(), true,
                                     /*fp8_merged_arm=*/false, indt,
                                     GdnFp8MergedInProjDType(indt, outdt)}),
             "qwen3_5 split FP8 GDN qkv: the allocated mixed_qkv dtype and the "
             "packed-decode prediction disagree (GdnProjectedMixedQkvDType)");
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4): exclusive and first on both projections.
  // `indt`/`outdt` rather than a literal bf16 here, because the packed-decode
  // predictor above derives the SAME dtype for this arm and a disagreement
  // makes vt::GdnPackedDecode throw.
  if (!w.in_proj_qkv_fp8_block.Empty()) {
    out.mixed_owner.emplace(dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(
        d, h, w.in_proj_qkv_fp8_block, indt));
    out.z_owner.emplace(dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(
        d, h, w.in_proj_z_fp8_block, outdt));
    out.mixed = out.mixed_owner->t();
    out.z = out.z_owner->t();
    return out;
  }
  // W4d W4 (tenstorrent-27b-gdn-keepquant.md): block-encoded GDN
  // projections — the packed row-reorder keeps attn_qkv (q6_K) and
  // attn_gate (q4_K) verbatim through the V-head reorder — decode through
  // their keep-quant words via the same MatmulBTRawD the merged arm uses,
  // per tensor (the merged arm needs ONE encoding; this checkpoint mixes
  // q6_K qkv with a q4_K gate). bf16 output: the same dtype the default
  // bf16 in_proj arm emits, which the conv/post-conv consumers expect.
  if (!w.in_proj_qkv.Empty() && vt::IsBlockQuant(w.in_proj_qkv.dtype)) {
    VT_CHECK(!w.in_proj_z.Empty() && vt::IsBlockQuant(w.in_proj_z.dtype),
             "qwen3_5 GDN: a block-encoded in_proj_qkv must pair with a "
             "block-encoded in_proj_z");
    out.mixed_owner.emplace(MatmulBTRawD(
        d, h, ResidentWeight(d, w.in_proj_qkv), DType::kBF16));
    out.z_owner.emplace(
        MatmulBTRawD(d, h, ResidentWeight(d, w.in_proj_z), DType::kBF16));
    out.mixed = out.mixed_owner->t();
    out.z = out.z_owner->t();
    return out;
  }
  out.mixed_owner.emplace(
      !w.in_proj_qkv_fp8.Empty()
          ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_qkv_fp8,
                                               fp8_mixed_dt)
                   : MatmulFp8CutlassD(d, h, w.in_proj_qkv_fp8, fp8_mixed_dt))
      : indt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_qkv)
                             : MatmulF32D(d, h, w.in_proj_qkv));
  out.z_owner.emplace(
      !w.in_proj_z_fp8.Empty()
          ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_z_fp8, outdt)
                   : MatmulFp8CutlassD(d, h, w.in_proj_z_fp8, outdt))
      : outdt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_z)
                              : MatmulF32D(d, h, w.in_proj_z));
  out.mixed = out.mixed_owner->t();
  out.z = out.z_owner->t();
  return out;
}

// The production entry point. `ProjectGdnQkvzRaw` above is every existing arm,
// unchanged; this wrapper adds the deferred V-head permutation and NOTHING else
// — with `v_head_perm_key_heads == 0` (every loader but `qwen4exp`'s GGUF one)
// `PermuteVHeadsIfDeferred` returns nullopt and the views below are the raw
// ones, byte for byte.
//
// `mixed` is [T, conv_dim] and ONLY its trailing V rows are tiled: the leading
// `2 * key_dim` q and k rows are what the converter left alone, so they are the
// permutation's PREFIX and are copied straight through. That keeps `mixed` in
// exactly the channel order `conv1d_weight` (still permuted at load) and
// `vt::GdnConvSplit` expect. `z` is [T, value_dim], all V heads, no prefix; its
// consumer is the gated RMSNorm, which reads it as [T, Hv, Dv].
GdnQkvzOutput ProjectGdnQkvz(Dev d, const GdnLayerWeights& w, const Tensor& h,
                             int64_t conv_dim, int64_t value_dim, DType indt,
                             DType outdt, const Tensor* h_fp8,
                             int64_t value_heads) {
  GdnQkvzOutput out =
      ProjectGdnQkvzRaw(d, w, h, conv_dim, value_dim, indt, outdt, h_fp8);
  if (w.v_head_perm_key_heads == 0) return out;
  out.mixed_perm_owner =
      PermuteVHeadsIfDeferred(d, w, out.mixed, value_heads,
                              /*prefix_elems=*/conv_dim - value_dim,
                              /*inverse=*/false);
  out.z_perm_owner = PermuteVHeadsIfDeferred(d, w, out.z, value_heads,
                                             /*prefix_elems=*/0,
                                             /*inverse=*/false);
  out.mixed = out.mixed_perm_owner->t();
  out.z = out.z_perm_owner->t();
  // The raw owners stay alive in `out`; see ProjectGdnBA.
  return out;
}


// Rank-3 [T,Hv,Dv] gate view over a (possibly padded-row) [T, value_dim] z
// slice for the gated RMSNorm — the inner [Hv,Dv] block is contiguous while
// the token stride follows the packed parent (upstream z.reshape(T,-1,Dv) on
// the mixed_qkvz slice, qwen_gdn_linear_attn.py:934).
// Resolved GDN output-gate activation. Upstream hands RMSNormGated the string
// (`activation=output_gate_type`, qwen_gdn_linear_attn.py:452-464 @555967922);
// vt::RmsNormGatedArgs models the same silu/sigmoid split as a bool. HfConfig
// canonicalizes the key at LOAD time (only "silu"/"sigmoid" survive, "swish"
// already collapsed, anything else refused), so this is a lookup and never a
// second normalization -- exactly one place decides what the gate is.
bool GdnSigmoidGate(const HfConfig& cfg) {
  return cfg.output_gate_type == "sigmoid";
}

Tensor GdnGateView3(const Tensor& z, int64_t T, int64_t hv, int64_t dv) {
  Tensor g = z;
  g.rank = 3;
  g.shape[0] = T;
  g.shape[1] = hv;
  g.shape[2] = dv;
  g.stride[0] = z.stride[0];
  g.stride[1] = dv;
  g.stride[2] = 1;
  return g;
}

// --- GDN (linear_attention) block. gdn-semantics.md §1 (layout), §6 (g/beta),
// §7 (recurrence); qwen_gdn_linear_attn.py forward. Device-resident (M2.5
// Phase 1): h [T,H] bf16 (device) -> DBuf [T,H] bf16 (device); no host round-
// trips (the g/beta prep + conv split are device ops, not host loops).
DBuf GdnBlock(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
              const Tensor& h, int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Kw = cfg.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // Input projections (mixed_qkvz | ba). The 27B loader mirrors vLLM's TWO
  // physical BF16 projections (one qkvz + one ba, qwen_gdn_linear_attn.py:
  // 923-936); 35B/GGUF/synthetic paths retain the split legacy owners. W8A8
  // cutlass fp8 (35B) when populated, else bf16 (default / GGUF). qkv/z read
  // the shared pre-quantized fp8 activation (h_fp8, quantize-once) when
  // supplied; a/b stay bf16 GEMMs on h (so h_fp8's producer also emits bf16 h
  // for them). mixed_qkv: bf16 output under VT_GDN_IN_BF16 (bf16-weight branch),
  // and on the merged fp8 branch too under VT_GDN_FP8_IN_BF16 (default OFF, see
  // GdnFp8InBf16Enabled). See GdnInDType().
  const DType indt = GdnInDType();
  const DType outdt = GdnOutDType();
  GdnQkvzOutput qkvz =
      ProjectGdnQkvz(d, w, h, conv_dim, value_dim, indt, outdt, h_fp8, Hv);
  Tensor mixed = qkvz.mixed;  // [T,conv_dim], contiguous or row-strided view
  Tensor z = qkvz.z;          // [T,value_dim], contiguous or row-strided view
  GdnBaOutput ba = ProjectGdnBA(d, w, h, Hv);
  Tensor braw = ba.b;  // [T,Hv], F32 contiguous or row-strided merged view
  Tensor araw = ba.a;

  // Causal conv1d over the token stream (silu activation), fresh zero state. conv
  // in/out dtype follows the in_proj output (bf16 under VT_GDN_IN_BF16); f32
  // conv_state + f32-accumulated math unchanged. The conv reads the merged
  // mixed_qkv view's padded row stride directly — no materialization.
  const DType convdt = mixed.dtype;
  Tensor dcw = convdt == DType::kBF16 ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
                                      : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dstate(d, DType::kF32, {1, conv_dim, Kw - 1});
  dstate.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  DBuf dconv(d, convdt, {T, conv_dim});
  vt::CausalConv1dFwd(d.q, dconv.t(), mixed, dcw, nullptr, dstate.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});

  // Post-conv prep (gdn-semantics.md §1 layout, §4 l2norm, §6 g/beta): split the
  // conv output into q|k|v, l2-normalize q/k over Dk, derive g/beta. Fused into a
  // single vt::GdnPostConv launch (perf/glue-fuse; mirror fla
  // fused_gdn_prefill_post_conv), or the four per-op launches when disabled.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  // Coupled bf16 (VT_GDN_BF16, default ON): the matmul-input activations q/k/v
  // feed the WMMA chunk trio as native bf16 (halved traffic + bf16 fragments);
  // g/beta/state stay f32 (FLA's split). VT_GDN_BF16=0 keeps f32/TF32.
  const DType actdt = GdnActDType();
  DBuf vf(d, actdt, {T, Hv, Dv});
  DBuf g(d, DType::kF32, {T, Hv});
  DBuf beta(d, DType::kF32, {T, Hv});
  DBuf dql2(d, actdt, {T, Hk, Dk});
  DBuf dkl2(d, actdt, {T, Hk, Dk});
  if (GlueFuseEnabled()) {
    vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), g.t(), beta.t(), dconv.t(), araw,
                    braw, a_log_dev, dt_bias_dev, vt::L2NormArgs{1e-6F});
  } else {
    DBuf qf(d, actdt, {T, Hk, Dk});
    DBuf kf(d, actdt, {T, Hk, Dk});
    Tensor q2 = Reshape(qf.t(), {T, key_dim});
    Tensor k2 = Reshape(kf.t(), {T, key_dim});
    Tensor v2 = Reshape(vf.t(), {T, value_dim});
    vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());
    vt::GdnGBeta(d.q, g.t(), beta.t(), araw, braw, a_log_dev, dt_bias_dev);
    vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
    vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
  }
  // scale = Dk^-0.5, applied to q only inside the gated-delta-rule recurrence.
  DBuf dssm(d, DType::kF32, {1, Hv, Dv, Dk});
  dssm.Zero(d);
  DBuf dcore(d, outdt, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  vt::GdnPrefill(d.q, dcore.t(), dql2.t(), dkl2.t(), vf.t(), g.t(), beta.t(),
                 dssm.t(), dqsl.t(), vt::GdnArgs{scale});

  // Gated RMSNorm over Dv with the z gate (gdn-semantics.md §5): the split arm
  // keeps the byte-identical rank-2 [T*Hv, Dv] views; the merged arm passes
  // rank-3 [T,Hv,Dv] so the z gate's padded packed-row stride is representable
  // (same kernel and grid — only the gate addressing changes). Cast to bf16,
  // flatten heads, out-project.
  Tensor dnw = outdt == DType::kBF16 ? ResidentWeight(d, w.norm_weight, {Dv})
                                     : ResidentWeightF32(d, w.norm_weight, {Dv});
  const bool sigmoid_gate = GdnSigmoidGate(cfg);
  // A deferred V-head permutation MUST reach the out-projection input, and the
  // arms that quantize straight into the GEMM (`out_proj_fp8`'s fused store,
  // fp4, block-fp8, EXL3) never materialize `gated_bf16` or never take a bf16
  // activation. `qwen4exp` GGUF, the only loader that defers, populates none of
  // them — so this refuses a combination that does not exist today rather than
  // letting a future one produce a silently wrong V-head order.
  VT_CHECK(w.v_head_perm_key_heads == 0 ||
               (w.out_proj_fp8.Empty() && w.out_proj_fp4.Empty() &&
                w.out_proj_fp8_block.Empty() && w.out_proj_exl3.Empty()),
           "gdn: a deferred V-head permutation is implemented only for the "
           "bf16/block-quant out_proj arm; this layer carries a quantized "
           "out_proj owner that bypasses the permuted activation");
  const bool z_strided = z.stride[0] != value_dim;
  Tensor core2 = z_strided ? dcore.t() : Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = z_strided ? GdnGateView3(z, T, Hv, Dv) : Reshape(z, {T * Hv, Dv});
  // Gated RMSNorm writes bf16 directly (perf/glue-fuse: fold the CastBf16 into
  // the op store, mirror layernorm_guard.py:57 `out.to(dtype)`); VT_GLUE_FUSE=0
  // keeps the f32 RmsNormGated + separate CastBf16 pair.
  // Prefill glue-fusion (VT_GDN_OUT_FP8_FUSE): when the out_proj is W8A8 fp8 (35B),
  // fold its static fp8 activation quant INTO the gated-RMSNorm store — one launch
  // emits the fp8 activation directly, removing the standalone QuantFp8Static pass
  // and the bf16 gated-norm output it would write then re-read. Byte-identical to the
  // split RmsNormGated(bf16)+QuantFp8Static path (the fp8 is quantized from the SAME
  // bf16 value; the variance reduction order is unchanged). 27B's out_proj is fp4, so
  // it never takes this branch.
  if (!w.out_proj_fp8.Empty() && GdnOutFp8FuseEnabled() && GlueFuseEnabled() &&
      vllm::platforms::GetPlatform(d.q.device.type).supports_fp8()) {
    DBuf a_fp8(d, DType::kI8, {T, value_dim});
    Tensor a_fp8_v = z_strided ? Reshape(a_fp8.t(), {T, Hv, Dv})
                               : Reshape(a_fp8.t(), {T * Hv, Dv});
    // KERNEL-FUSION-FRAMEWORK W2 — route gated-RMSNorm + static fp8 quant through
    // vt::FusedChain(kRmsNormGatedQuantFp8); its fast_op binds the SAME bespoke
    // RmsNormGatedQuantFp8 kernel (byte-identical + perf-neutral). VT_FUSED_CHAIN_ADOPT=0
    // restores the direct hand-call.
    if (FusedChainAdoptEnabled()) {
      // sigmoid_gate is a STRUCTURAL recipe flag (fused_recipe.h FStep), not a
      // call scalar, so the sigmoid arm binds a COPY of the recipe with that
      // step flag set. On the silu arm the copy is byte-identical to
      // vt::kRmsNormGatedQuantFp8, so this stays perf- and bit-neutral.
      vt::FusedRecipe gated_fp8 = vt::kRmsNormGatedQuantFp8;
      gated_fp8.steps[0].sigmoid_gate = sigmoid_gate;
      vt::FusedChain(d.q, gated_fp8, a_fp8_v, core2, z2, dnw, eps,
                     w.out_proj_fp8.input_scale);
    } else {
      vt::RmsNormGatedQuantFp8(d.q, a_fp8_v, core2, z2, dnw,
                               vt::RmsNormGatedArgs{eps, sigmoid_gate},
                               w.out_proj_fp8.input_scale);
    }
    return MatmulFp8CutlassPreQuantD(d, a_fp8.t(), w.out_proj_fp8, DType::kBF16);
  }
  DBuf gated_bf16(d, ActDType(d), {T, value_dim});
  if (GlueFuseEnabled()) {
    Tensor gated2 = z_strided ? Reshape(gated_bf16.t(), {T, Hv, Dv})
                              : Reshape(gated_bf16.t(), {T * Hv, Dv});
    vt::RmsNormGated(d.q, gated2, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
  } else {
    DBuf dgated(d, DType::kF32, {T * Hv, Dv});
    Tensor gated_f32 = z_strided ? Reshape(dgated.t(), {T, Hv, Dv}) : dgated.t();
    vt::RmsNormGated(d.q, gated_f32, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
    vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  }

  // DEFERRED V-HEAD PERMUTATION on the OUT-PROJECTION INPUT (MODEL-MM-QWEN4-EXP).
  // `out_proj`'s permutation is on its COLUMNS, so the identity that moves it is
  // `(W P) x = W (P x)` — the INPUT is permuted, and in the INVERSE direction
  // (grouped -> tiled) because everything from the conv to this point ran in
  // grouped order. Inert when nothing is deferred.
  std::optional<DBuf> gated_perm = PermuteVHeadsIfDeferred(
      d, w, gated_bf16.t(), Hv, /*prefix_elems=*/0, /*inverse=*/true);
  const Tensor gated_in = gated_perm ? gated_perm->t() : gated_bf16.t();
  // W8A8 cutlass fp8 (35B) when populated, else fp4-resident W4A4 (27B, notes
  // §3.6), else bf16 (default / GGUF).
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): exclusive and FIRST, for the reason
  // every other arm here is exclusive. An EXL3 load populates none of the
  // fields below, so each would fall through to an empty owner and refuse by
  // name. bf16 out, as every other arm of this out_proj returns.
  //
  // The gated-RMSNorm above already wrote `gated_bf16`, so the fp8 fusion
  // branch earlier in this function cannot have fired: it requires
  // `out_proj_fp8`, which an EXL3 load leaves empty.
  if (!w.out_proj_exl3.Empty()) {
    return dense_exl3::Linear(d, gated_in, w.out_proj, w.out_proj_exl3,
                              DType::kBF16);
  }
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4): exclusive and first; bf16 out, as every
  // other arm of this out_proj returns.
  if (!w.out_proj_fp8_block.Empty()) {
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, gated_in,
                                                        w.out_proj_fp8_block,
                                                        DType::kBF16);
  }
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_in, w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : GdnOutProjMatmul(d, w, gated_bf16, T, Hk, Hv, Dv);  // [T,H]
}

// PERSISTENT per-step input device buffers (decode host-tax #2): the flattened
// positions + the full-attn metadata (slot_mapping/block_table/seq_lens/
// query_start_loc) + all GDN non-spec/prefill state metadata, uploaded ONCE
// per step and read by EVERY layer — mirrors vLLM's persistent input buffers in
// gpu_model_runner.py (self.input_batch.{positions,slot_mapping,block_table,
// seq_lens,query_start_loc} device tensors, refreshed once per step). Collapses
// the previous per-full-attn-layer (×10) and per-GDN-layer (×30) H2D re-uploads —
// each a BLOCKING pageable cudaMemcpyAsync that serialized the decode stream
// (nsys: ~110 blocking copies/step ≈ 5.1s host-stall in an 8s decode window) — to
// ONE upload per input, per step. The buffers live for the whole layer loop (held
// by the caller). Bit-exact (same bytes, uploaded once instead of per layer).
// Graph-safe: on the num_reqs==1 decode-graph the single upload is captured from
// the persistent host metadata address (same as the per-layer copies were) and
// replayed.
struct StepDevInputs {
  DBuf positions;        // i32 [T]
  DBuf slot_mapping;     // i64 [T]
  DBuf block_table;      // i32 [num_reqs, cols]
  DBuf seq_lens;         // i32 [num_reqs]
  DBuf query_start_loc;  // i32 [num_reqs+1]
  DBuf gdn_state_idx;    // i32 [num_reqs] full non-spec state slots
  bool has_gdn_idx = false;
  DBuf gdn_non_spec_qsl;       // i32 [num_reqs+1]
  DBuf gdn_has_initial;        // i8 [num_reqs], upstream bool mask
  DBuf gdn_prefill_state_idx;  // i32 [num_prefills]
  DBuf gdn_prefill_qsl;        // i32 [num_prefills+1]
  DBuf gdn_prefill_has_initial;  // i8 [num_prefills]
  bool has_gdn_prefill_meta = false;
  DBuf gdn_conv_batch_ptr;  // i32 [num exact conv programs]
  DBuf gdn_conv_token_chunk_offsets;  // i32 [num exact conv programs]
  bool has_gdn_conv_chunks = false;
  bool indexed_gdn_state_io = false;
  // ── Spec-decode device tensors (SPEC-MTP I5a). Uploaded ONCE per step (shared
  // by every GDN layer's spec branch), only when the step carries drafts
  // (num_spec_decodes > 0). Every one is a stub of size 1 on the default path,
  // so the upload is byte-identical to pre-I5a there. Mirror the six host arrays
  // I4's GDN builder emits (gdn_attn.cpp:181-287). ──
  DBuf gdn_spec_state_idx;   // i32 [num_spec_decodes * num_cols] (row-major)
  DBuf gdn_spec_qsl;         // i32 [num_spec_decodes + 1]
  DBuf gdn_spec_token_indx;  // i32 [num_spec_decode_tokens]
  DBuf gdn_non_spec_token_indx;  // i32 [num_decode_tokens + num_prefill_tokens]
  DBuf gdn_spec_seq_masks;   // i8  [num_reqs], upstream bool mask
  DBuf gdn_num_accepted;     // i32 [num_spec_decodes]
  DBuf gdn_spec_conv_state_idx;  // i32 [num_spec_decodes] = spec_state_idx col 0
  bool has_gdn_spec = false;
  int64_t gdn_spec_num_cols = 0;  // == num_spec + 1 (spec_state_indices .size(-1))
  // f32 [T, rotary_dim] cos|sin cache for the fused full-attn preamble, built ONCE
  // per step (VT_FUSE_ATTN_PREAMBLE) and reused by every full-attn layer; a 1-elem
  // stub when the toggle is off (has_attn_cos_sin=false).
  DBuf attn_cos_sin;
  bool has_attn_cos_sin = false;
};

StepDevInputs BuildStepDevInputs(Dev d, const std::vector<int32_t>& positions,
                                 const CommonAttentionMetadata& am,
                                 const GDNAttentionMetadata& gm,
                                 int64_t gdn_state_slots) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const bool indexed_state_io = IndexedGdnStateIoEnabled(d.q.device);
  VT_CHECK(gm.num_actual_tokens == T,
           "qwen3_5: GDN metadata token count must match step input");
  // Internal graph calls may append -1 sentinel rows; eager entry points have
  // already rejected them. Every other shape/suffix/range invariant is checked
  // here immediately before upload so no state-I/O branch can bypass it.
  detail::ValidateGdnAttentionMetadata(
      gm, gdn_state_slots, /*allow_inert_padding=*/true);
  // Engine-owned state indices are validated on their host representation
  // before any DBuf upload. Packed CUDA then needs no capture-breaking D2H flag
  // or stream synchronization; its kernel retains an independent bounds guard.
  if (gm.non_spec_state_indices_tensor.has_value() &&
      (indexed_state_io || gm.num_decodes > 0)) {
    const int64_t index_count =
        indexed_state_io
            ? static_cast<int64_t>(gm.non_spec_state_indices_tensor->size())
            : static_cast<int64_t>(gm.num_decodes);
    detail::ValidateGdnStateIndices(*gm.non_spec_state_indices_tensor,
                                    index_count, gdn_state_slots);
  }
  StepDevInputs s{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kI32, {1}),  // non-spec index stub
      false,
      DBuf(d, DType::kI32, {1}),  // non-spec qsl stub
      DBuf(d, DType::kI8, {1}),   // full has-initial stub
      DBuf(d, DType::kI32, {1}),  // prefill index stub
      DBuf(d, DType::kI32, {1}),  // prefill qsl stub
      DBuf(d, DType::kI8, {1}),   // prefill has-initial stub
      false,
      DBuf(d, DType::kI32, {1}),  // exact conv batch-ptr stub
      DBuf(d, DType::kI32, {1}),  // exact conv chunk-offset stub
      false,
      indexed_state_io,
      DBuf(d, DType::kI32, {1}),  // spec state-idx stub
      DBuf(d, DType::kI32, {1}),  // spec qsl stub
      DBuf(d, DType::kI32, {1}),  // spec token-indx stub
      DBuf(d, DType::kI32, {1}),  // non-spec token-indx stub
      DBuf(d, DType::kI8, {1}),   // spec seq-masks stub
      DBuf(d, DType::kI32, {1}),  // num-accepted stub
      DBuf(d, DType::kI32, {1}),  // spec conv-state-idx (col0) stub
      false,                       // has_gdn_spec
      0,                           // gdn_spec_num_cols
      DBuf(d, DType::kF32, {1}),  // attn cos|sin stub (filled by MaybeBuildAttnCosSin)
      false,
  };
  // Full non-spec state indices are shared by decode and mixed-prefill paths.
  // Decode consumes their leading num_decodes rows; indexed W1 gather/scatter
  // consumes the whole vector. One upload replaces every per-layer row copy.
  if (gm.non_spec_state_indices_tensor.has_value() &&
      (indexed_state_io || gm.num_decodes > 0)) {
    const int64_t index_count =
        indexed_state_io
            ? static_cast<int64_t>(gm.non_spec_state_indices_tensor->size())
            : static_cast<int64_t>(gm.num_decodes);
    s.gdn_state_idx = DBuf(d, DType::kI32,
                           {index_count},
                           gm.non_spec_state_indices_tensor->data());
    s.has_gdn_idx = true;
  }
  if (indexed_state_io && gm.num_prefills > 0 &&
      gm.non_spec_query_start_loc.has_value() &&
      gm.has_initial_state.has_value() &&
      gm.prefill_state_indices.has_value() &&
      gm.prefill_query_start_loc.has_value() &&
      gm.prefill_has_initial_state.has_value()) {
    s.gdn_non_spec_qsl = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.non_spec_query_start_loc->size())},
        gm.non_spec_query_start_loc->data());
    s.gdn_has_initial = DBuf(
        d, DType::kI8,
        {static_cast<int64_t>(gm.has_initial_state->size())},
        gm.has_initial_state->data());
    s.gdn_prefill_state_idx = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.prefill_state_indices->size())},
        gm.prefill_state_indices->data());
    s.gdn_prefill_qsl = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.prefill_query_start_loc->size())},
        gm.prefill_query_start_loc->data());
    s.gdn_prefill_has_initial = DBuf(
        d, DType::kI8,
        {static_cast<int64_t>(gm.prefill_has_initial_state->size())},
        gm.prefill_has_initial_state->data());
    s.has_gdn_prefill_meta = true;
  }
  if (gm.num_prefills > 0 && gm.batch_ptr.has_value() &&
      gm.token_chunk_offset_ptr.has_value()) {
    s.gdn_conv_batch_ptr = DBuf(
        d, DType::kI32, {static_cast<int64_t>(gm.batch_ptr->size())},
        gm.batch_ptr->data());
    s.gdn_conv_token_chunk_offsets = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.token_chunk_offset_ptr->size())},
        gm.token_chunk_offset_ptr->data());
    s.has_gdn_conv_chunks = true;
  }
  // ── Spec-decode tensor upload (SPEC-MTP I5a). The six device tensors the GDN
  // spec branch of GdnBlockPaged reads (mirror qwen_gdn_linear_attn.py:
  // 1344-1476). Uploaded once per step from I4's builder output; NONE of this
  // runs on the default path (gm.num_spec_decodes == 0), so the per-step upload
  // is byte-identical to pre-I5a there. ValidateGdnAttentionMetadata above has
  // already verified every shape/range invariant of these host arrays. ──
  if (gm.num_spec_decodes > 0) {
    s.gdn_spec_state_idx = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.spec_state_indices_tensor->size())},
        gm.spec_state_indices_tensor->data());
    s.gdn_spec_qsl = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.spec_query_start_loc->size())},
        gm.spec_query_start_loc->data());
    s.gdn_spec_token_indx = DBuf(
        d, DType::kI32, {static_cast<int64_t>(gm.spec_token_indx->size())},
        gm.spec_token_indx->data());
    s.gdn_spec_seq_masks = DBuf(
        d, DType::kI8, {static_cast<int64_t>(gm.spec_sequence_masks->size())},
        gm.spec_sequence_masks->data());
    s.gdn_num_accepted = DBuf(
        d, DType::kI32, {static_cast<int64_t>(gm.num_accepted_tokens->size())},
        gm.num_accepted_tokens->data());
    // Column 0 of spec_state_indices per request — the conv slot (the conv
    // window needs no per-timestep slots; qwen_gdn_linear_attn.py:1350). The
    // CausalConv1dSpecUpdate op reads it as a contiguous [num_spec_decodes]
    // buffer, so materialize the strided column here rather than per layer.
    const int64_t spec_cols = gm.spec_state_indices_num_cols;
    const int64_t nspec = gm.num_spec_decodes;
    std::vector<int32_t> conv_col0(static_cast<size_t>(nspec));
    for (int64_t i = 0; i < nspec; ++i)
      conv_col0[static_cast<size_t>(i)] =
          (*gm.spec_state_indices_tensor)[static_cast<size_t>(i * spec_cols)];
    s.gdn_spec_conv_state_idx =
        DBuf(d, DType::kI32, {nspec}, conv_col0.data());
    // non_spec_token_indx is present only for a MIXED spec batch (nullopt/empty
    // for a pure spec batch, gdn_attn.cpp:211); upload it only when non-empty.
    if (gm.non_spec_token_indx.has_value() &&
        !gm.non_spec_token_indx->empty()) {
      s.gdn_non_spec_token_indx = DBuf(
          d, DType::kI32,
          {static_cast<int64_t>(gm.non_spec_token_indx->size())},
          gm.non_spec_token_indx->data());
    }
    s.gdn_spec_num_cols = gm.spec_state_indices_num_cols;
    s.has_gdn_spec = true;
  }
  return s;
}

// Build the per-step fused-preamble cos|sin cache into `sdi` (once per step,
// reused by every full-attn layer's fused preamble) when VT_FUSE_ATTN_PREAMBLE is
// on. No-op otherwise, so the default forward path is byte-identical. Uses the
// PERSISTENT sdi.positions device buffer (same source the RopeNeox path reads) so
// the fill is a single device kernel — eager and graph-replay identical.
void MaybeBuildAttnCosSin(Dev d, StepDevInputs& sdi, const HfConfig& cfg, int64_t T,
                          bool fp4_attn = false) {
  if (!FuseAttnPreambleOn(fp4_attn)) return;
  const int rot = static_cast<int>(cfg.rotary_dim);
  if (rot <= 0) return;
  sdi.attn_cos_sin = DBuf(d, DType::kF32, {T, rot});
  vt::RopeCosSinCache(d.q, sdi.attn_cos_sin.t(), sdi.positions.t(),
                      vt::RopeArgs{static_cast<float>(cfg.rope_theta), rot});
  sdi.has_attn_cos_sin = true;
}

// VT_ASYNC_EXECUTOR (Option A): re-fill an ALREADY-ALLOCATED cos|sin cache in place
// from the current sdi.positions. The alloc happened pre-capture (MaybeBuildAttnCosSin
// on the persistent StepDevInputs); this fill runs INSIDE the captured region so each
// replay re-derives rope from the freshly-staged positions. No allocation, capture-
// safe. Only reached when sdi.has_attn_cos_sin (the fused-preamble arch, e.g. 27B W4A4).
void FillAttnCosSin(Dev d, StepDevInputs& sdi, const HfConfig& cfg) {
  const int rot = static_cast<int>(cfg.rotary_dim);
  if (rot <= 0) return;
  vt::RopeCosSinCache(d.q, sdi.attn_cos_sin.t(), sdi.positions.t(),
                      vt::RopeArgs{static_cast<float>(cfg.rope_theta), rot});
}

// --- Batched PAGED GDN block (M1.8 Task 3). Same conv1d + l2norm + q/k/v/g/beta
// prep + gated-norm + out_proj as GdnBlock, but driven by the batched
// GDNAttentionMetadata segmentation over the PERSISTENT ssm_state/conv_state:
// leading num_decode_tokens are decode (vt::GdnDecode + causal_conv1d_update),
// the rest prefill (vt::GdnPrefill + causal_conv1d_fn). Mirrors
// qwen_gdn_linear_attn.py::_forward_core @ e24d1b24 (conv split L1360-1388;
// recurrence split L1480-1559; the ssm gather+ZERO L1513-1514, scatter L1532).
// h [T*H] bf16 -> [T*H] bf16.
// MIXED spec+non-spec GDN batch (SPEC-MTP concurrency). A speculative-decode
// request shares a step with an ordinary prefill request; upstream reclassifies
// any non-spec DECODE to a prefill whenever a spec row exists (gdn_attn.py:
// 243-251), so the non-spec side is pure prefill (num_decodes == 0). Split
// mixed_qkv / a / b by spec_token_indx / non_spec_token_indx into compact
// per-group buffers (index_select), run the I5a spec recurrence + the prefill
// recurrence independently over the SHARED persistent state, merge the two core
// outputs back to their original row positions (index_copy), then finish with
// the shared gated-RMSNorm + out_proj. 1:1 mirror of
// qwen_gdn_linear_attn.py::_forward_core:1329-1576 @ e24d1b24. Reached ONLY under
// an active speculator at concurrency > 1; the default forward never builds it.
DBuf GdnBlockPagedMixedSpec(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
                            const Tensor& mixed, const Tensor& z,
                            const Tensor& araw, const Tensor& braw,
                            const GdnStateCache& state, const StepDevInputs& sdi,
                            const GDNAttentionMetadata& meta, int64_t T) {
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Kw = cfg.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const DType convdt = mixed.dtype;
  const DType outdt = GdnOutDType();
  const DType actdt = GdnActDType();
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  g_mixed_spec_invocations.fetch_add(1, std::memory_order_relaxed);

  // The mixed split reuses the WIDENED-cache-aware indexed conv gather/scatter
  // (I5e) for the non-spec prefill, so indexed state IO + the uploaded spec and
  // non-spec prefill metadata are required. This holds on the CUDA production
  // path (indexed state IO defaults ON); refuse loudly otherwise.
  VT_CHECK(sdi.indexed_gdn_state_io,
           "gdn paged mixed spec: requires indexed GDN state IO (widened conv "
           "cache); enable VT_GDN_INDEXED_STATE_IO");
  VT_CHECK(sdi.has_gdn_spec && sdi.has_gdn_prefill_meta && sdi.has_gdn_idx,
           "gdn paged mixed spec: spec + non-spec prefill metadata must be uploaded");
  VT_CHECK(meta.non_spec_token_indx.has_value() &&
               meta.prefill_query_start_loc.has_value(),
           "gdn paged mixed spec: builder must emit non_spec_token_indx + prefill qsl");

  const int64_t ns = meta.num_spec_decodes;
  const int64_t ns_tok = meta.num_spec_decode_tokens;
  const int64_t nns_tok = static_cast<int64_t>(meta.non_spec_token_indx->size());
  const int64_t np = meta.num_prefills;
  VT_CHECK(ns_tok + nns_tok == T, "gdn paged mixed spec: spec+non-spec != T");

  Tensor spec_tok = sdi.gdn_spec_token_indx.t();
  Tensor ns_tok_idx = sdi.gdn_non_spec_token_indx.t();

  // ── 1. Split mixed_qkv / a / b by token group (index_select; :1334-1335,
  // :1407-1408). araw/braw may be inner-contiguous row-strided merged views;
  // IndexSelect follows the outer stride and packs each group contiguously. ──
  DBuf mixed_spec(d, convdt, {ns_tok, conv_dim});
  DBuf mixed_ns(d, convdt, {nns_tok, conv_dim});
  vt::IndexSelect(d.q, mixed_spec.t(), mixed, spec_tok);
  vt::IndexSelect(d.q, mixed_ns.t(), mixed, ns_tok_idx);
  DBuf a_spec(d, araw.dtype, {ns_tok, Hv});
  DBuf b_spec(d, braw.dtype, {ns_tok, Hv});
  DBuf a_ns(d, araw.dtype, {nns_tok, Hv});
  DBuf b_ns(d, braw.dtype, {nns_tok, Hv});
  vt::IndexSelect(d.q, a_spec.t(), araw, spec_tok);
  vt::IndexSelect(d.q, b_spec.t(), braw, spec_tok);
  vt::IndexSelect(d.q, a_ns.t(), araw, ns_tok_idx);
  vt::IndexSelect(d.q, b_ns.t(), braw, ns_tok_idx);

  Tensor dcw = convdt == DType::kBF16
                   ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
                   : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});

  // ── 2. Conv per group. spec: multi-query causal_conv1d_update over the
  // WIDENED conv_state (:1344-1357). non-spec: gather narrow conv rows, causal
  // conv1d_fn, scatter back (:1365-1375). ──
  DBuf dconv_spec(d, convdt, {ns_tok, conv_dim});
  {
    Tensor conv_cache = state.conv_state;
    vt::CausalConv1dSpecUpdate(d.q, dconv_spec.t(), mixed_spec.t(), dcw,
                               /*bias=*/nullptr, conv_cache,
                               sdi.gdn_spec_conv_state_idx.t(),
                               sdi.gdn_num_accepted.t(), sdi.gdn_spec_qsl.t(),
                               vt::CausalConv1dArgs{true});
  }
  DBuf dconv_ns(d, convdt, {nns_tok, conv_dim});
  {
    VT_CHECK(sdi.has_gdn_conv_chunks,
             "gdn paged mixed spec: exact causal-conv chunks must be uploaded");
    Tensor conv_batch_ptr = sdi.gdn_conv_batch_ptr.t();
    Tensor conv_chunk_offsets = sdi.gdn_conv_token_chunk_offsets.t();
    vt::CausalConv1dArgs conv_args{true, &conv_batch_ptr,
                                   &conv_chunk_offsets};
    DBuf dcs(d, DType::kF32, {np, conv_dim, Kw - 1});
    vt::GdnStateGather(d.q, dcs.t(), state.conv_state, sdi.gdn_state_idx.t());
    vt::CausalConv1dFwd(d.q, dconv_ns.t(), mixed_ns.t(), dcw, nullptr, dcs.t(),
                        sdi.gdn_non_spec_qsl.t(), sdi.gdn_has_initial.t(),
                        conv_args);
    Tensor conv_cache = state.conv_state;
    vt::GdnStateScatter(d.q, conv_cache, dcs.t(), sdi.gdn_state_idx.t());
  }

  // ── 3. post-conv prep (split q|k|v, l2-normalize q/k, derive g/beta) per
  // group; identical op set to the pure paths (fused GdnPostConv or 4-op). ──
  auto post_conv = [&](Tensor dconv_g, Tensor a_g, Tensor b_g, int64_t n_tok,
                       DBuf& dql2, DBuf& dkl2, DBuf& vf, DBuf& dg, DBuf& dbeta) {
    if (GlueFuseEnabled()) {
      vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), dg.t(), dbeta.t(), dconv_g,
                      a_g, b_g, a_log_dev, dt_bias_dev, vt::L2NormArgs{1e-6F});
    } else {
      DBuf qf(d, actdt, {n_tok, Hk, Dk});
      DBuf kf(d, actdt, {n_tok, Hk, Dk});
      Tensor q2 = Reshape(qf.t(), {n_tok, key_dim});
      Tensor k2 = Reshape(kf.t(), {n_tok, key_dim});
      Tensor v2 = Reshape(vf.t(), {n_tok, value_dim});
      vt::GdnConvSplit(d.q, q2, k2, v2, dconv_g);
      vt::GdnGBeta(d.q, dg.t(), dbeta.t(), a_g, b_g, a_log_dev, dt_bias_dev);
      vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
      vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
    }
  };

  DBuf dcore(d, outdt, {T, Hv, Dv});

  // ── 4.1 spec recurrence (fused_sigmoid_gating_delta_rule_update, :1455-1475):
  // k+1 state slots per request, initial state selected by num_accepted. ──
  {
    DBuf dql2(d, actdt, {ns_tok, Hk, Dk});
    DBuf dkl2(d, actdt, {ns_tok, Hk, Dk});
    DBuf vf(d, actdt, {ns_tok, Hv, Dv});
    DBuf dg(d, DType::kF32, {ns_tok, Hv});
    DBuf dbeta(d, DType::kF32, {ns_tok, Hv});
    post_conv(dconv_spec.t(), a_spec.t(), b_spec.t(), ns_tok, dql2, dkl2, vf, dg,
              dbeta);
    DBuf dcore_spec(d, outdt, {ns_tok, Hv, Dv});
    Tensor ssm_cache = state.ssm_state;
    Tensor spec_idx_2d =
        Reshape(sdi.gdn_spec_state_idx.t(), {ns, sdi.gdn_spec_num_cols});
    vt::GdnSpecDecode(d.q, dcore_spec.t(), dql2.t(), dkl2.t(), vf.t(), dg.t(),
                      dbeta.t(), ssm_cache, sdi.gdn_spec_qsl.t(), spec_idx_2d,
                      sdi.gdn_num_accepted.t(), vt::GdnArgs{scale});
    vt::IndexCopy(d.q, dcore.t(), dcore_spec.t(), spec_tok);  // :1570
  }

  // ── 4.2 non-spec prefill recurrence (chunk_gated_delta_rule, :1504-1532):
  // gather initial state (zeroing fresh rows), chunked prefill, scatter final. ──
  {
    DBuf dql2(d, actdt, {nns_tok, Hk, Dk});
    DBuf dkl2(d, actdt, {nns_tok, Hk, Dk});
    DBuf vf(d, actdt, {nns_tok, Hv, Dv});
    DBuf dg(d, DType::kF32, {nns_tok, Hv});
    DBuf dbeta(d, DType::kF32, {nns_tok, Hv});
    post_conv(dconv_ns.t(), a_ns.t(), b_ns.t(), nns_tok, dql2, dkl2, vf, dg,
              dbeta);
    DBuf dcore_ns(d, outdt, {nns_tok, Hv, Dv});
    DBuf dss(d, DType::kF32, {np, Hv, Dv, Dk});
    vt::GdnStateGather(d.q, dss.t(), state.ssm_state,
                       sdi.gdn_prefill_state_idx.t(),
                       &sdi.gdn_prefill_has_initial.t());
    vt::GdnArgs gdn_args{scale};
    gdn_args.query_start_loc_host = meta.prefill_query_start_loc->data();
    vt::GdnPrefill(d.q, dcore_ns.t(), dql2.t(), dkl2.t(), vf.t(), dg.t(),
                   dbeta.t(), dss.t(), sdi.gdn_prefill_qsl.t(), gdn_args);
    Tensor ssm_cache = state.ssm_state;
    vt::GdnStateScatter(d.q, ssm_cache, dss.t(), sdi.gdn_prefill_state_idx.t());
    vt::IndexCopy(d.q, dcore.t(), dcore_ns.t(), ns_tok_idx);  // :1571
  }

  // ── 5. shared gated-RMSNorm(z) + out_proj over the merged [T,Hv,Dv] core.
  // Byte-identical op sequence to GdnBlockPaged's tail. ──
  Tensor dnw = outdt == DType::kBF16 ? ResidentWeight(d, w.norm_weight, {Dv})
                                     : ResidentWeightF32(d, w.norm_weight, {Dv});
  const bool sigmoid_gate = GdnSigmoidGate(cfg);
  // A deferred V-head permutation MUST reach the out-projection input, and the
  // arms that quantize straight into the GEMM (`out_proj_fp8`'s fused store,
  // fp4, block-fp8, EXL3) never materialize `gated_bf16` or never take a bf16
  // activation. `qwen4exp` GGUF, the only loader that defers, populates none of
  // them — so this refuses a combination that does not exist today rather than
  // letting a future one produce a silently wrong V-head order.
  VT_CHECK(w.v_head_perm_key_heads == 0 ||
               (w.out_proj_fp8.Empty() && w.out_proj_fp4.Empty() &&
                w.out_proj_fp8_block.Empty() && w.out_proj_exl3.Empty()),
           "gdn: a deferred V-head permutation is implemented only for the "
           "bf16/block-quant out_proj arm; this layer carries a quantized "
           "out_proj owner that bypasses the permuted activation");
  const bool z_strided = z.stride[0] != value_dim;
  Tensor core2 = z_strided ? dcore.t() : Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = z_strided ? GdnGateView3(z, T, Hv, Dv) : Reshape(z, {T * Hv, Dv});
  if (!w.out_proj_fp8.Empty() && GdnOutFp8FuseEnabled() && GlueFuseEnabled() &&
      vllm::platforms::GetPlatform(d.q.device.type).supports_fp8()) {
    DBuf a_fp8(d, DType::kI8, {T, value_dim});
    Tensor a_fp8_v = z_strided ? Reshape(a_fp8.t(), {T, Hv, Dv})
                               : Reshape(a_fp8.t(), {T * Hv, Dv});
    if (FusedChainAdoptEnabled()) {
      // sigmoid_gate is a STRUCTURAL recipe flag (fused_recipe.h FStep), not a
      // call scalar, so the sigmoid arm binds a COPY of the recipe with that
      // step flag set. On the silu arm the copy is byte-identical to
      // vt::kRmsNormGatedQuantFp8, so this stays perf- and bit-neutral.
      vt::FusedRecipe gated_fp8 = vt::kRmsNormGatedQuantFp8;
      gated_fp8.steps[0].sigmoid_gate = sigmoid_gate;
      vt::FusedChain(d.q, gated_fp8, a_fp8_v, core2, z2, dnw, eps,
                     w.out_proj_fp8.input_scale);
    } else {
      vt::RmsNormGatedQuantFp8(d.q, a_fp8_v, core2, z2, dnw,
                               vt::RmsNormGatedArgs{eps, sigmoid_gate},
                               w.out_proj_fp8.input_scale);
    }
    return MatmulFp8CutlassPreQuantD(d, a_fp8.t(), w.out_proj_fp8, DType::kBF16);
  }
  DBuf gated_bf16(d, ActDType(d), {T, value_dim});
  if (GlueFuseEnabled()) {
    Tensor gated2 = z_strided ? Reshape(gated_bf16.t(), {T, Hv, Dv})
                              : Reshape(gated_bf16.t(), {T * Hv, Dv});
    vt::RmsNormGated(d.q, gated2, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
  } else {
    DBuf dgated(d, DType::kF32, {T * Hv, Dv});
    Tensor gated_f32 = z_strided ? Reshape(dgated.t(), {T, Hv, Dv}) : dgated.t();
    vt::RmsNormGated(d.q, gated_f32, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
    vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  }

  // DEFERRED V-HEAD PERMUTATION on the OUT-PROJECTION INPUT (MODEL-MM-QWEN4-EXP).
  // `out_proj`'s permutation is on its COLUMNS, so the identity that moves it is
  // `(W P) x = W (P x)` — the INPUT is permuted, and in the INVERSE direction
  // (grouped -> tiled) because everything from the conv to this point ran in
  // grouped order. Inert when nothing is deferred.
  std::optional<DBuf> gated_perm = PermuteVHeadsIfDeferred(
      d, w, gated_bf16.t(), Hv, /*prefix_elems=*/0, /*inverse=*/true);
  const Tensor gated_in = gated_perm ? gated_perm->t() : gated_bf16.t();
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): exclusive and FIRST, for the reason
  // every other arm here is exclusive. An EXL3 load populates none of the
  // fields below, so each would fall through to an empty owner and refuse by
  // name. bf16 out, as every other arm of this out_proj returns.
  //
  // The gated-RMSNorm above already wrote `gated_bf16`, so the fp8 fusion
  // branch earlier in this function cannot have fired: it requires
  // `out_proj_fp8`, which an EXL3 load leaves empty.
  if (!w.out_proj_exl3.Empty()) {
    return dense_exl3::Linear(d, gated_in, w.out_proj, w.out_proj_exl3,
                              DType::kBF16);
  }
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4): exclusive and first; bf16 out, as every
  // other arm of this out_proj returns.
  if (!w.out_proj_fp8_block.Empty()) {
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, gated_in,
                                                        w.out_proj_fp8_block,
                                                        DType::kBF16);
  }
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_in, w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : GdnOutProjMatmul(d, w, gated_bf16, T, Hk, Hv, Dv);  // [T,H]
}

// VT_DUMP_ACT stage probe (GDN): dump named intermediates so a layer-level
// divergence can be pinned to one kernel. Debug-only; Download syncs, never set
// on capture paths.
//
// ROCM-TIER-DIVERGENCE (#2590): keyed on (step, layer) through `actdump` instead
// of on a process-global call counter. The counter named an invocation ordinal,
// which is a different quantity on two runs whose forward-call counts differ by
// one and cannot be joined across two tiers at all.
void DumpGdnStage(Dev d, const char* stage, const Tensor& t) {
  const char* dir = actdump::StreamDir();
  if (dir == nullptr) return;
  ActDumpTensor(d, "VT_DUMP_ACT", dir, (std::string("gdn_") + stage).c_str(),
                      t, 1, t.Numel());
}

DBuf GdnBlockPaged(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
                   const Tensor& h, const StepDevInputs& sdi,
                   const GDNAttentionMetadata& meta,
                   const GdnStateCache& state, int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Kw = cfg.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  const int64_t nd = meta.num_decodes;
  const int64_t np = meta.num_prefills;
  const int64_t nd_tok = meta.num_decode_tokens;
  const int64_t np_tok = meta.num_prefill_tokens;
  // Spec-decode segmentation (SPEC-MTP I5a). `spec` is false on every production
  // step, so the whole branch below is dead code on the default path.
  const int64_t ns = meta.num_spec_decodes;
  const int64_t ns_tok = meta.num_spec_decode_tokens;
  const bool spec = ns > 0;
  VT_CHECK(meta.num_actual_tokens == T, "gdn paged: num_actual_tokens != T");
  VT_CHECK(nd_tok + np_tok + ns_tok == T,
           "gdn paged: decode+prefill+spec tokens != T");
  // A spec batch is either PURE (num_prefills == 0 && num_decodes == 0 — the
  // steady-state k=1 decode, the I5a fast path below) or MIXED (spec rows share
  // the step with reclassified non-spec prefill rows, under concurrency > 1).
  // Upstream reclassifies any non-spec DECODE to a prefill when a spec row
  // exists (gdn_attn.py:243-251), so a spec batch never carries num_decodes > 0;
  // the MIXED case is handled by the index_select split/merge helper.
  VT_CHECK(!spec || nd == 0,
           "gdn paged: a spec batch must have num_decodes == 0 (upstream "
           "reclassifies non-spec decodes to prefill, gdn_attn.py:243-251)");
  const bool mixed_spec = spec && np > 0;

  const DType indt = GdnInDType();
  const DType outdt = GdnOutDType();
  // PERF-27B-GDN-PACKED-REACHABLE (#365). `dtype_compatible` is decided by the
  // ACTIVATION dtypes vt::GdnPackedDecode requires, not by how the GDN weights
  // are stored. `mixed_qkv` has to be PREDICTED because this decision runs
  // before ProjectGdnQkvz (it feeds ProjectGdnBA's output dtype below).
  const bool gdn_fp8_tower =
      !w.in_proj_qkv_fp8.Empty() || !w.in_proj_z_fp8.Empty();
  // PERF-GDN-PACKED-BRIDGE (#365): the arm term is the SAME predicate over the
  // SAME inputs that ProjectGdnQkvz uses to choose the merged fp8 leaf, so the
  // prediction cannot select a different arm than the projection does.
  const bool gdn_fp8_merged_arm =
      !w.in_proj_qkv_fp8.Empty() &&
      detail::ShouldUseMergedGdnFp8Qkvz(
          GdnMergedFp8QkvzEligibilityFor(d, w, conv_dim, value_dim));
  const DType mixed_dt = detail::GdnProjectedMixedQkvDType(
      detail::GdnMixedQkvDTypeInputs{!w.in_proj_qkvz.Empty(),
                                     !w.in_proj_qkv_fp8.Empty(),
                                     gdn_fp8_merged_arm, indt,
                                     GdnFp8MergedInProjDType(indt, outdt),
                                     // MODEL-QWEN35-GDN-EXL3 (#2495 item 4).
                                     !w.in_proj_qkv_exl3.Empty()});
  const bool packed_decode = detail::ShouldUsePackedGdnDecode(
      detail::GdnPackedDecodeEligibility{
          PackedGdnDecodeRuntimeEnabled(),
          vllm::platforms::GetPlatform(d.q.device.type).needs_weight_staging(),
          !w.in_proj_ba.Empty(),
          MergedGdnBaEnabled(d),
          detail::GdnPackedDecodeDTypesCompatible(
              detail::GdnPackedDecodeDTypes{mixed_dt,
                                            MergedGdnBaOutputDType(true), outdt,
                                            state.ssm_state.dtype}) &&
              // VT_GDN_PACKED_DECODE_FP8_TOWER, DEFAULT OFF: the same-binary
              // rollback of the relaxation. OFF reproduces the legacy blanket
              // exclusion of an fp8 GDN tower exactly; ON lets the dtype rule
              // above decide alone. Never deselects a non-fp8 checkpoint.
              (!gdn_fp8_tower || PackedGdnDecodeFp8TowerEnabled()),
          sdi.has_gdn_idx,
          np,
          np_tok,
          nd,
          nd_tok,
          meta.num_spec_decodes,
          meta.num_spec_decode_tokens,
          T});

  // GDN packed-decode geometry diagnostic (default OFF; read ONCE). Gated on the
  // same VT_GDN_DIAG_STEP_LOG toggle as the runner's step log so a c16 packed
  // reproduction can attribute the decision that selects the packed leg.
  static const bool gdn_diag_step_log = [] {
    const char* e = std::getenv("VT_GDN_DIAG_STEP_LOG");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  if (gdn_diag_step_log) {
    std::cerr << "[VT_GDN_DIAG] packed_decode=" << (packed_decode ? 1 : 0)
              << " nd=" << nd << " np=" << np << " nd_tok=" << nd_tok
              << " np_tok=" << np_tok << " T=" << T << "\n";
  }

  // Input projections (mixed_qkvz | ba). The 27B loader mirrors vLLM's TWO
  // physical BF16 projections (one qkvz + one ba, qwen_gdn_linear_attn.py:
  // 923-936); 35B/GGUF/synthetic paths retain the split legacy owners. W8A8
  // cutlass fp8 (35B) when populated, else bf16 (default / GGUF). qkv/z read
  // the shared pre-quantized fp8 activation (h_fp8, quantize-once) when
  // supplied; a/b stay bf16 GEMMs on h (so h_fp8's producer also emits bf16 h
  // for them). mixed_qkv: bf16 output under VT_GDN_IN_BF16 (bf16-weight branch,
  // halves the conv-input traffic); the merged fp8 branch reaches the same bf16
  // output under VT_GDN_FP8_IN_BF16 (default OFF, GdnFp8InBf16Enabled) and keeps
  // f32 otherwise. The z gate follows the recurrence-output dtype
  // (VT_GDN_OUT_BF16): the gated-RMSNorm requires gate.dtype == core.dtype.
  GdnQkvzOutput qkvz =
      ProjectGdnQkvz(d, w, h, conv_dim, value_dim, indt, outdt, h_fp8, Hv);
  Tensor mixed = qkvz.mixed;  // [T,conv_dim], contiguous or row-strided view
  Tensor z = qkvz.z;          // [T,value_dim], contiguous or row-strided view
  GdnBaOutput ba = ProjectGdnBA(d, w, h, Hv, packed_decode);
  Tensor braw = ba.b;  // [T,Hv], F32 contiguous or row-strided merged view
  Tensor araw = ba.a;

  // MIXED spec+non-spec batch: split the per-token projections into compact spec
  // / non-spec groups, run each recurrence path, merge back (qwen_gdn_linear_attn
  // .py:1329-1576). Reached only under an active speculator at concurrency > 1.
  if (mixed_spec) {
    return GdnBlockPagedMixedSpec(d, w, cfg, mixed, z, araw, braw, state, sdi,
                                  meta, T);
  }

  // Causal conv1d over the token stream, PERSISTENT conv_state (gathered by the
  // per-request state indices, updated in place, scattered back). conv in/out
  // dtype follows the in_proj output (bf16 under VT_GDN_IN_BF16 → bf16 weight +
  // bf16 dconv halve the conv read/write); the f32 conv_state and the
  // f32-accumulated conv math are unchanged. The post-conv split reads dconv's
  // dtype (GdnPostConv/GdnConvSplit are templated on it). The conv reads the
  // merged mixed_qkv view's padded row stride directly — no materialization.
  const DType convdt = mixed.dtype;
  Tensor dcw = convdt == DType::kBF16 ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
                                      : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dconv(d, convdt, {T, conv_dim});
  const int64_t conv_row_elems = conv_dim * (Kw - 1);
  const bool indexed_state_io = sdi.indexed_gdn_state_io;
  std::vector<int32_t> row_copy_decode_indices;
  if (!indexed_state_io && nd > 0) {
    const auto& all_indices = *meta.non_spec_state_indices_tensor;
    VT_CHECK(static_cast<int64_t>(all_indices.size()) >= nd,
             "row-copy GDN decode state index metadata is too short");
    row_copy_decode_indices.assign(all_indices.begin(),
                                   all_indices.begin() + nd);
  }
  if (spec) {
    // ── PURE spec conv (SPEC-MTP I5a; qwen_gdn_linear_attn.py:1344-1357). Every
    // token is a spec token (pure batch: np == nd == 0), so mixed_qkv_spec is
    // the whole stream and the multi-query conv runs over it in place on the
    // widened conv_state rows ((K-1)+num_spec taps). The slot is column 0 of
    // spec_state_indices; the window advances by num_accepted, not 1+k, which is
    // the conv rollback. Persistent per-step device tensors (uploaded once). ──
    VT_CHECK(sdi.has_gdn_spec,
             "gdn paged: spec batch requires uploaded spec metadata");
    Tensor conv_cache = state.conv_state;  // widened [slots, conv_dim, (K-1)+k]
    vt::CausalConv1dSpecUpdate(d.q, dconv.t(), mixed, dcw, /*bias=*/nullptr,
                               conv_cache, sdi.gdn_spec_conv_state_idx.t(),
                               sdi.gdn_num_accepted.t(), sdi.gdn_spec_qsl.t(),
                               vt::CausalConv1dArgs{true});
  } else if (np > 0) {
    // Any prefill: conv over the WHOLE non-spec stream (decodes lead, each with
    // has_initial_state=1). qwen_gdn_linear_attn.py:1360-1375.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    const int64_t nreq = static_cast<int64_t>(sidx.size());
    // Gather the persistent conv_state rows into an f32 working buffer (bf16
    // cache on CUDA → upcast; f32 cache on CPU → direct), run the f32
    // CausalConv1dFwd, then downcast + scatter back to the cache.
    const std::vector<int64_t> cs_shape = {nreq, conv_dim, Kw - 1};
    VT_CHECK(sdi.has_gdn_conv_chunks,
             "gdn paged: exact causal-conv chunks must be uploaded");
    Tensor conv_batch_ptr = sdi.gdn_conv_batch_ptr.t();
    Tensor conv_chunk_offsets = sdi.gdn_conv_token_chunk_offsets.t();
    vt::CausalConv1dArgs conv_args{true, &conv_batch_ptr,
                                   &conv_chunk_offsets};
    if (indexed_state_io) {
      VT_CHECK(sdi.has_gdn_idx && sdi.has_gdn_prefill_meta,
               "indexed GDN conv requires persistent non-spec metadata");
      DBuf dcs(d, DType::kF32, cs_shape);
      vt::GdnStateGather(d.q, dcs.t(), state.conv_state,
                         sdi.gdn_state_idx.t());
      vt::CausalConv1dFwd(d.q, dconv.t(), mixed, dcw, nullptr,
                          dcs.t(), sdi.gdn_non_spec_qsl.t(),
                          sdi.gdn_has_initial.t(),
                          conv_args);
      if (const char* td = std::getenv("VT_DUMP_TRUST")) vt::tenstorrent::TrustDump(d.q, td, "conv", dconv.t());
      DumpGdnStage(d, "conv", dconv.t());
      Tensor conv_cache = state.conv_state;
      vt::GdnStateScatter(d.q, conv_cache, dcs.t(),
                          sdi.gdn_state_idx.t());
    } else {
      const auto& qsl_full = *meta.non_spec_query_start_loc;
      const auto& his_u8 = *meta.has_initial_state;
      DBuf dcs =
          GatherStateF32(d, state.conv_state, sidx, conv_row_elems, cs_shape);
      std::vector<int32_t> his(his_u8.begin(), his_u8.end());
      DBuf dqsl(d, DType::kI32, {nreq + 1}, qsl_full.data());
      DBuf dhis(d, DType::kI32, {nreq}, his.data());
      vt::CausalConv1dFwd(d.q, dconv.t(), mixed, dcw, nullptr,
                          dcs.t(), dqsl.t(), dhis.t(),
                          conv_args);
      ScatterStateF32(d, state.conv_state, dcs, sidx, conv_row_elems);
      if (const char* td = std::getenv("VT_DUMP_TRUST")) vt::tenstorrent::TrustDump(d.q, td, "conv", dconv.t());
      DumpGdnStage(d, "conv2", dconv.t());
    }
  } else {
    // Pure decode: single-token conv step per sequence, IN PLACE on the persistent
    // conv_state at each sequence's slot (mirrors mamba causal_conv1d_update
    // conv_state_indices, qwen_gdn_linear_attn.py:1376-1388). Passing the state
    // indices to the op eliminates the per-request gather+scatter — the two
    // host<->device (unified-cache) copies per sequence per layer that dominate
    // the decode memcpy tax (they scale with concurrency, flattening throughput).
    // Upload the decode state-slot indices straight from the PERSISTENT metadata
    // vector (its leading nd entries are the decode slots; decodes lead the batch)
    // — NOT a stack-local copy, so the decode-CUDA-graph replay (num_reqs==1) can
    // re-read this H2D copy from a fixed host address across replays.
    // State slot indices from the PERSISTENT per-step buffer (uploaded once,
    // shared by conv-update + the ssm recurrence across all GDN layers).
    if (indexed_state_io) {
      Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
      Tensor conv_cache = state.conv_state;  // mutable view over the shared buffer
      vt::CausalConv1dUpdate(d.q, dconv.t(), mixed, dcw, nullptr,
                             conv_cache, vt::CausalConv1dArgs{true}, &gidx);
    } else {
      VT_CHECK(nd_tok == nd,
               "row-copy GDN conv decode requires one token per request");
      const auto& sidx = row_copy_decode_indices;
      const std::vector<int64_t> cs_shape = {nd, conv_dim, Kw - 1};
      DBuf dcs = GatherStateF32(d, state.conv_state, sidx, conv_row_elems,
                                cs_shape);
      vt::CausalConv1dUpdate(d.q, dconv.t(), mixed, dcw, nullptr,
                             dcs.t(), vt::CausalConv1dArgs{true});
      ScatterStateF32(d, state.conv_state, dcs, sidx, conv_row_elems);
    }
  }

  // Post-conv prep (§1 layout, §4 l2norm, §6 g/beta): split q|k|v, l2-normalize
  // q/k over Dk, derive g/beta. Fused into one vt::GdnPostConv launch
  // (perf/glue-fuse; mirror fla fused_gdn_prefill_post_conv), or four per-op
  // launches when disabled. g/beta uniform over all tokens; recurrence segments.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  // dcore is common to both branches. The packed branch consumes raw dconv and
  // row-strided a/b directly, selecting before any normalized q/k/v/g/beta
  // intermediates are allocated. This mirrors vLLM v0.25.0
  // _forward_core_decode_non_spec:1644-1695.
  DBuf dcore(d, outdt, {T, Hv, Dv});
  // GDN-MOE-BF16-OUT (#1168): read off the tensors, not off GdnOutDType, so a
  // gate entering through ModelRegistry::Forward observes what this layer RAN.
  // This is the ONLY recording site. The mixed spec+non-spec batch returns into
  // GdnBlockPagedMixedSpec above and never reaches it, so a mixed step leaves
  // the record untouched rather than stale-free — see the header's limits.
  RecordGdnOutActivationDTypes(dcore.t().dtype, z.dtype);
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  if (packed_decode) {
    Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
    Tensor ssm_cache = state.ssm_state;
    vt::GdnPackedDecode(d.q, dcore.t(), dconv.t(), araw, braw,
                        a_log_dev, dt_bias_dev, ssm_cache, gidx,
                        vt::GdnArgs{scale});
  } else {
    // Coupled bf16 (VT_GDN_BF16, default ON): matmul-input activations q/k/v
    // are bf16 (native WMMA fragments + halved traffic); g/beta and recurrence
    // arithmetic stay f32.
    const DType actdt = GdnActDType();
    DBuf vf(d, actdt, {T, Hv, Dv});
    DBuf dg(d, DType::kF32, {T, Hv});
    DBuf dbeta(d, DType::kF32, {T, Hv});
    DBuf dql2(d, actdt, {T, Hk, Dk});
    DBuf dkl2(d, actdt, {T, Hk, Dk});
    if (GlueFuseEnabled()) {
      vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), dg.t(), dbeta.t(),
                      dconv.t(), araw, braw, a_log_dev, dt_bias_dev,
                      vt::L2NormArgs{1e-6F});
      DumpGdnStage(d, "mixed", mixed);
      DumpGdnStage(d, "postconv_q", dql2.t());
      DumpGdnStage(d, "postconv_v", vf.t());
      if (const char* td = std::getenv("VT_DUMP_TRUST")) {
        vt::tenstorrent::TrustDump(d.q, td, "pc_q", dql2.t());
        vt::tenstorrent::TrustDump(d.q, td, "pc_v", vf.t());
      }
    } else {
      DBuf qf(d, actdt, {T, Hk, Dk});
      DBuf kf(d, actdt, {T, Hk, Dk});
      Tensor q2 = Reshape(qf.t(), {T, key_dim});
      Tensor k2 = Reshape(kf.t(), {T, key_dim});
      Tensor v2 = Reshape(vf.t(), {T, value_dim});
      vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());
      vt::GdnGBeta(d.q, dg.t(), dbeta.t(), araw, braw, a_log_dev,
                   dt_bias_dev);
      vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
      vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
    }
    // scale = Dk^-0.5 (q only, inside the recurrence). dcore (recurrence
    // output / core_attn_out) is bf16 under VT_GDN_OUT_BF16 —
    // GdnDecode/GdnPrefill store Tout directly; otherwise use the f32 arm.
    const int64_t ssm_row_elems = Hv * Dv * Dk;

    if (spec) {
      // ── PURE spec recurrence (SPEC-MTP I5a; qwen_gdn_linear_attn.py:
      // 1455-1475). Every token is a spec token, in batch order (identity gather
      // in a pure batch), so the whole [T,...] post-conv output feeds one
      // GdnSpecDecode over the per-request k+1 state slots. The kernel selects
      // each request's initial state from column num_accepted-1 and snapshots
      // the post-token state of timestep t into column t — the SSM rollback.
      // Persistent per-step device tensors (uploaded once, shared by all GDN
      // layers). state_indices is the flat [num_spec_decodes*num_cols] upload
      // viewed row-major as [num_spec_decodes, num_cols]. ──
      Tensor ssm_cache = state.ssm_state;  // full [slots, Hv, Dv, Dk] cache
      Tensor spec_idx_2d =
          Reshape(sdi.gdn_spec_state_idx.t(), {ns, sdi.gdn_spec_num_cols});
      vt::GdnSpecDecode(d.q, dcore.t(), dql2.t(), dkl2.t(), vf.t(), dg.t(),
                        dbeta.t(), ssm_cache, sdi.gdn_spec_qsl.t(), spec_idx_2d,
                        sdi.gdn_num_accepted.t(), vt::GdnArgs{scale});
    } else {
    // Recurrence — decode segment first (leading nd_tok tokens), then prefill.
    if (nd > 0) {
      // Decode recurrence IN PLACE on the persistent ssm_state at each sequence's
      // slot (mirrors fla fused_recurrent ssm_state_indices) — no per-request
      // gather+scatter (the other two host<->device copies per sequence per layer).
      // Persistent metadata source (see the conv branch) for graph-replay safety.
      // The persistent W1 buffer covers every non-spec request. A turnover step
      // may have a leading decode subset followed by prefills, so both decode
      // consumers must narrow the view to exactly `nd` state slots.
      Tensor q_dec = SubView(dql2.t(), 0, nd_tok);
      Tensor k_dec = SubView(dkl2.t(), 0, nd_tok);
      Tensor v_dec = SubView(vf.t(), 0, nd_tok);
      Tensor g_dec = SubView(dg.t(), 0, nd_tok);
      Tensor b_dec = SubView(dbeta.t(), 0, nd_tok);
      Tensor o_dec = SubView(dcore.t(), 0, nd_tok);
      if (indexed_state_io) {
        Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
        Tensor ssm_cache = state.ssm_state;  // mutable view over the shared buffer
        vt::GdnDecode(d.q, o_dec, q_dec, k_dec, v_dec, g_dec, b_dec,
                      ssm_cache, vt::GdnArgs{scale}, &gidx);
      } else {
        VT_CHECK(nd_tok == nd,
                 "row-copy GDN recurrence decode requires one token per request");
        const auto& sidx = row_copy_decode_indices;
        const std::vector<int64_t> ss_shape = {nd, Hv, Dv, Dk};
        DBuf dss = GatherStateF32(d, state.ssm_state, sidx, ssm_row_elems,
                                  ss_shape);
        vt::GdnDecode(d.q, o_dec, q_dec, k_dec, v_dec, g_dec, b_dec,
                      dss.t(), vt::GdnArgs{scale});
        ScatterStateF32(d, state.ssm_state, dss, sidx, ssm_row_elems);
      }
    }
    if (np > 0) {
      const auto& pidx = *meta.prefill_state_indices;
      const auto& p_qsl = *meta.prefill_query_start_loc;
      // Gather the persistent ssm_state rows into an f32 working buffer
      // (fp16/bf16 cache on CUDA → upcast; f32 cache → direct), run the f32
      // chunked GdnPrefill, then downcast + scatter to the configured cache.
      const std::vector<int64_t> ss_shape = {np, Hv, Dv, Dk};
      DBuf dss(d, DType::kF32, ss_shape);
      if (indexed_state_io) {
        VT_CHECK(sdi.has_gdn_prefill_meta,
                 "indexed GDN prefill requires persistent prefill metadata");
        // Fuses indexing, compressed->F32, and fresh-request zeroing.
        vt::GdnStateGather(d.q, dss.t(), state.ssm_state,
                           sdi.gdn_prefill_state_idx.t(),
                           &sdi.gdn_prefill_has_initial.t());
      } else {
        dss = GatherStateF32(d, state.ssm_state, pidx, ssm_row_elems,
                             ss_shape);
        const auto& p_his = *meta.prefill_has_initial_state;
        const size_t rb = static_cast<size_t>(ssm_row_elems) * sizeof(float);
        for (size_t s = 0; s < p_his.size(); ++s)
          if (p_his[s] == 0)
            d.b.Memset(d.q, static_cast<char*>(dss.ptr()) + s * rb, 0, rb);
      }
      Tensor q_pre = SubView(dql2.t(), nd_tok, np_tok);
      Tensor k_pre = SubView(dkl2.t(), nd_tok, np_tok);
      Tensor v_pre = SubView(vf.t(), nd_tok, np_tok);
      Tensor g_pre = SubView(dg.t(), nd_tok, np_tok);
      Tensor b_pre = SubView(dbeta.t(), nd_tok, np_tok);
      Tensor o_pre = SubView(dcore.t(), nd_tok, np_tok);
      // Hand the CUDA chunked-prefill path the HOST query_start_loc (p_qsl,
      // already materialized by the GDN metadata build) so it skips the
      // per-layer D2H copy + synchronization. p_qsl outlives this call.
      vt::GdnArgs gdn_args{scale};
      gdn_args.query_start_loc_host = p_qsl.data();
      if (indexed_state_io) {
        vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre,
                       dss.t(), sdi.gdn_prefill_qsl.t(), gdn_args);
        Tensor ssm_cache = state.ssm_state;
        DumpGdnStage(d, "core", o_pre);
        if (const char* td = std::getenv("VT_DUMP_TRUST")) vt::tenstorrent::TrustDump(d.q, td, "core", o_pre);
        vt::GdnStateScatter(d.q, ssm_cache, dss.t(),
                            sdi.gdn_prefill_state_idx.t());
      } else {
        DBuf dpqsl(d, DType::kI32, {np + 1}, p_qsl.data());
        vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre,
                       dss.t(), dpqsl.t(), gdn_args);
        ScatterStateF32(d, state.ssm_state, dss, pidx, ssm_row_elems);
        if (const char* td = std::getenv("VT_DUMP_TRUST")) vt::tenstorrent::TrustDump(d.q, td, "core", o_pre);
      }
    }
    }  // end non-spec recurrence
  }

  // Gated RMSNorm over Dv with the z gate, cast bf16, flatten heads, out-project.
  // Weight follows core/z dtype (RmsNormGated requires w.dtype == x.dtype): native
  // bf16 under VT_GDN_OUT_BF16 (the norm still accumulates variance in f32), else
  // the f32 upcast. Mirrors the q_norm/k_norm resident-weight dtype gate.
  // The split arm keeps the byte-identical rank-2 [T*Hv, Dv] views; the merged
  // arm passes rank-3 [T,Hv,Dv] so the z gate's padded packed-row stride is
  // representable (same kernel and grid — only the gate addressing changes).
  Tensor dnw = outdt == DType::kBF16 ? ResidentWeight(d, w.norm_weight, {Dv})
                                     : ResidentWeightF32(d, w.norm_weight, {Dv});
  const bool sigmoid_gate = GdnSigmoidGate(cfg);
  // A deferred V-head permutation MUST reach the out-projection input, and the
  // arms that quantize straight into the GEMM (`out_proj_fp8`'s fused store,
  // fp4, block-fp8, EXL3) never materialize `gated_bf16` or never take a bf16
  // activation. `qwen4exp` GGUF, the only loader that defers, populates none of
  // them — so this refuses a combination that does not exist today rather than
  // letting a future one produce a silently wrong V-head order.
  VT_CHECK(w.v_head_perm_key_heads == 0 ||
               (w.out_proj_fp8.Empty() && w.out_proj_fp4.Empty() &&
                w.out_proj_fp8_block.Empty() && w.out_proj_exl3.Empty()),
           "gdn: a deferred V-head permutation is implemented only for the "
           "bf16/block-quant out_proj arm; this layer carries a quantized "
           "out_proj owner that bypasses the permuted activation");
  const bool z_strided = z.stride[0] != value_dim;
  Tensor core2 = z_strided ? dcore.t() : Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = z_strided ? GdnGateView3(z, T, Hv, Dv) : Reshape(z, {T * Hv, Dv});
  // Gated RMSNorm writes bf16 directly (perf/glue-fuse: fold the CastBf16 into
  // the op store, mirror layernorm_guard.py:57 `out.to(dtype)`); VT_GLUE_FUSE=0
  // keeps the f32 RmsNormGated + separate CastBf16 pair.
  // Prefill glue-fusion (VT_GDN_OUT_FP8_FUSE): when the out_proj is W8A8 fp8 (35B),
  // fold its static fp8 activation quant INTO the gated-RMSNorm store — one launch
  // emits the fp8 activation directly, removing the standalone QuantFp8Static pass
  // and the bf16 gated-norm output it would write then re-read. Byte-identical to the
  // split RmsNormGated(bf16)+QuantFp8Static path (the fp8 is quantized from the SAME
  // bf16 value; the variance reduction order is unchanged). 27B's out_proj is fp4, so
  // it never takes this branch.
  if (!w.out_proj_fp8.Empty() && GdnOutFp8FuseEnabled() && GlueFuseEnabled() &&
      vllm::platforms::GetPlatform(d.q.device.type).supports_fp8()) {
    DBuf a_fp8(d, DType::kI8, {T, value_dim});
    Tensor a_fp8_v = z_strided ? Reshape(a_fp8.t(), {T, Hv, Dv})
                               : Reshape(a_fp8.t(), {T * Hv, Dv});
    // KERNEL-FUSION-FRAMEWORK W2 — route gated-RMSNorm + static fp8 quant through
    // vt::FusedChain(kRmsNormGatedQuantFp8); its fast_op binds the SAME bespoke
    // RmsNormGatedQuantFp8 kernel (byte-identical + perf-neutral). VT_FUSED_CHAIN_ADOPT=0
    // restores the direct hand-call.
    if (FusedChainAdoptEnabled()) {
      // sigmoid_gate is a STRUCTURAL recipe flag (fused_recipe.h FStep), not a
      // call scalar, so the sigmoid arm binds a COPY of the recipe with that
      // step flag set. On the silu arm the copy is byte-identical to
      // vt::kRmsNormGatedQuantFp8, so this stays perf- and bit-neutral.
      vt::FusedRecipe gated_fp8 = vt::kRmsNormGatedQuantFp8;
      gated_fp8.steps[0].sigmoid_gate = sigmoid_gate;
      vt::FusedChain(d.q, gated_fp8, a_fp8_v, core2, z2, dnw, eps,
                     w.out_proj_fp8.input_scale);
    } else {
      vt::RmsNormGatedQuantFp8(d.q, a_fp8_v, core2, z2, dnw,
                               vt::RmsNormGatedArgs{eps, sigmoid_gate},
                               w.out_proj_fp8.input_scale);
    }
    return MatmulFp8CutlassPreQuantD(d, a_fp8.t(), w.out_proj_fp8, DType::kBF16);
  }
  DBuf gated_bf16(d, ActDType(d), {T, value_dim});
  if (GlueFuseEnabled()) {
    Tensor gated2 = z_strided ? Reshape(gated_bf16.t(), {T, Hv, Dv})
                              : Reshape(gated_bf16.t(), {T * Hv, Dv});
    vt::RmsNormGated(d.q, gated2, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
    DumpGdnStage(d, "gated", gated_bf16.t());
    if (const char* td = std::getenv("VT_DUMP_TRUST")) vt::tenstorrent::TrustDump(d.q, td, "gated", gated_bf16.t());
  } else {
    DBuf dgated(d, DType::kF32, {T * Hv, Dv});
    Tensor gated_f32 = z_strided ? Reshape(dgated.t(), {T, Hv, Dv}) : dgated.t();
    vt::RmsNormGated(d.q, gated_f32, core2, z2, dnw,
                     vt::RmsNormGatedArgs{eps, sigmoid_gate});
    vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  }

  // DEFERRED V-HEAD PERMUTATION on the OUT-PROJECTION INPUT (MODEL-MM-QWEN4-EXP).
  // `out_proj`'s permutation is on its COLUMNS, so the identity that moves it is
  // `(W P) x = W (P x)` — the INPUT is permuted, and in the INVERSE direction
  // (grouped -> tiled) because everything from the conv to this point ran in
  // grouped order. Inert when nothing is deferred.
  std::optional<DBuf> gated_perm = PermuteVHeadsIfDeferred(
      d, w, gated_bf16.t(), Hv, /*prefix_elems=*/0, /*inverse=*/true);
  const Tensor gated_in = gated_perm ? gated_perm->t() : gated_bf16.t();
  // W8A8 cutlass fp8 (35B) when populated, else fp4-resident W4A4 (27B, notes
  // §3.6), else bf16 (default / GGUF).
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): exclusive and FIRST, for the reason
  // every other arm here is exclusive. An EXL3 load populates none of the
  // fields below, so each would fall through to an empty owner and refuse by
  // name. bf16 out, as every other arm of this out_proj returns.
  //
  // The gated-RMSNorm above already wrote `gated_bf16`, so the fp8 fusion
  // branch earlier in this function cannot have fired: it requires
  // `out_proj_fp8`, which an EXL3 load leaves empty.
  if (!w.out_proj_exl3.Empty()) {
    return dense_exl3::Linear(d, gated_in, w.out_proj, w.out_proj_exl3,
                              DType::kBF16);
  }
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4): exclusive and first; bf16 out, as every
  // other arm of this out_proj returns.
  if (!w.out_proj_fp8_block.Empty()) {
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, gated_in,
                                                        w.out_proj_fp8_block,
                                                        DType::kBF16);
  }
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_in, w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : GdnOutProjMatmul(d, w, gated_bf16, T, Hk, Hv, Dv);  // [T,H]
}

// --- Dense full_attention block. qwen36-forward-notes.md §5; pinned
// Qwen3NextAttention. h [T*H] bf16 -> [T*H] bf16.
DBuf FullAttnBlock(Dev d, const FullAttnLayerWeights& w, const HfConfig& cfg,
                   const Tensor& h, const std::vector<int32_t>& positions,
                   int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  const bool fp4_attn = !w.q_proj_fp4.Empty();
  const bool packed_consumers =
      FuseAttnPreambleOn(fp4_attn) && rot > 0 &&
      vt::OpRegistered(vt::OpId::kAttnQkNormRopeGate, d.q.device.type);
  FullAttnQkvOutput qkv =
      ProjectFullAttnQkv(d, w, h, T, h_fp8, packed_consumers);
  const bool fp4 = qkv.fp4;
  Tensor qgate = qkv.qgate;  // [T,2*Hq*Dh], possibly row-strided packed view
  Tensor kf = qkv.key;       // [T,Hkv*Dh], possibly row-strided packed view
  Tensor vf = qkv.value;     // [T,Hkv*Dh], possibly row-strided packed view

  // Split q|gate + per-head gemma-RMSNorm(q,k) + partial NeoX RoPE + gate
  // passthrough, producing q[T,Hq,Dh]/k[T,Hkv,Dh] (normed+RoPE'd) and the raw
  // gate[T,Hq,Dh]. VT_FUSE_ATTN_PREAMBLE collapses the four ops into ONE launch
  // reading a precomputed cos_sin cache; it emits f32 q/k/gate — byte-identical
  // intermediates to the four-op path (value-exact). OFF keeps the exact original
  // AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox sequence.
  DBuf dq3(d, DType::kF32, {T, Hq, Dh});
  DBuf dk3(d, DType::kF32, {T, Hkv, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  if (FuseAttnPreambleOn(fp4) && rot > 0 && vt::OpRegistered(vt::OpId::kAttnQkNormRopeGate, d.q.device.type)) {
    DBuf dpos(d, DType::kI32, {T}, positions.data());
    DBuf cos_sin(d, DType::kF32, {T, rot});
    vt::RopeCosSinCache(d.q, cos_sin.t(), dpos.t(), vt::RopeArgs{base, rot});
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
    // KERNEL-FUSION-FRAMEWORK W2 — route the fused attn preamble through
    // vt::FusedChain(kAttnQkNormRopeGate). The recipe has no fast_op; its composite
    // MACRO dispatches to the SAME single vt::AttnQkNormRopeGate launch, so this is
    // perf-neutral + byte-identical. VT_FUSED_CHAIN_ADOPT=0 restores the hand-call.
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, vt::kAttnQkNormRopeGate, dq3.t(), dk3.t(), gatef.t(), qgate, kf, dqw,
                     dkw, cos_sin.t(), eps, vt::RopeArgs{base, rot});
    } else {
      vt::AttnQkNormRopeGate(d.q, dq3.t(), dk3.t(), gatef.t(), qgate, kf, dqw, dkw, cos_sin.t(),
                             vt::RmsNormArgs{eps, true}, vt::RopeArgs{base, rot});
    }
  } else {
    DBuf qf(d, DType::kF32, {T, Hq, Dh});
    vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate);
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dqn2d = Reshape(dq3.t(), {T * Hq, Dh});
    vt::RmsNorm(d.q, dqn2d, Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
    // `RmsNorm` NO LONGER requires `w.dtype == x.dtype`; #2477/#2493 gave the CUDA
    // kernel its own `Tw`, and #2492 did the same for the ROCm and fp8 twins. This
    // selection is therefore a WORK-AROUND FOR A CONSTRAINT THAT NO LONGER EXISTS,
    // and it is left in place rather than removed because it is behaviourally
    // neutral (the f32 upcast of a bf16 on-disk gamma is exact, so both arms feed
    // the kernel the same values) and removing it changes a shipped model's code
    // for no measured gain. When kf is bf16 (VT_BF16_GEMM_OUT on the fp4 path) use
    // the raw bf16 on-disk k_norm; otherwise (fp8/35B or toggle off) keep the f32
    // upcast. bf16 kf · bf16 dkw -> f32.
    Tensor dkw = (kf.dtype == DType::kBF16) ? ResidentWeight(d, w.k_norm, {Dh})
                                           : ResidentWeightF32(d, w.k_norm, {Dh});
    Tensor dkn2d = Reshape(dk3.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, dkn2d, Reshape(kf, {T * Hkv, Dh}), dkw,
                vt::RmsNormArgs{eps, true});
    DBuf dpos(d, DType::kI32, {T}, positions.data());
    // A resolved zero rotary width preserves the normalized query/key bytes.
    if (rot != 0) vt::RopeNeox(d.q, dq3.t(), dk3.t(), dpos.t(), vt::RopeArgs{base, rot});
  }
  Tensor qn3 = dq3.t();
  Tensor kn3 = dk3.t();

  // Causal GQA scaled-dot-product attention, scale = Dh^-0.5.
  Tensor v3 = vf;
  v3.rank = 3;
  v3.shape[0] = T;
  v3.shape[1] = Hkv;
  v3.shape[2] = Dh;
  v3.stride[1] = Dh;
  v3.stride[2] = 1;
  // vt::Attention requires q/k/v the same float dtype; qn3/kn3 are f32 after
  // norm+rope, so upcast a bf16 V (VT_BF16_GEMM_OUT fp4 path) back to f32. This is
  // the reference (non-paged) path — not perf-critical, so the small cast is fine.
  std::optional<DBuf> v3f32;
  if (v3.dtype == DType::kBF16) {
    v3f32.emplace(d, DType::kF32, std::vector<int64_t>{T, Hkv, Dh});
    vt::CastF32(d.q, v3f32->t(), v3);
    v3 = v3f32->t();
  } else if (!v3.IsContiguous()) {
    // Merged-QKV packed F32 value view: the token stride spans Q+K+V, but
    // vt::Attention requires contiguous q/k/v. Materialize a dense [T,Hkv,Dh]
    // copy (the reference non-paged path; the paged path instead round-trips
    // through the KV cache). The inner [Hkv,Dh] block is already contiguous, so
    // this is T per-row device copies — not perf-critical for the reference path.
    v3f32.emplace(d, DType::kF32, std::vector<int64_t>{T, Hkv, Dh});
    const int64_t inner = Hkv * Dh;
    for (int64_t tok = 0; tok < T; ++tok) {
      d.b.Copy(d.q, v3f32->t().Ptr<float>() + tok * inner,
               vf.Ptr<float>() + tok * vf.stride[0],
               static_cast<size_t>(inner) * sizeof(float));
    }
    v3 = v3f32->t();
  }
  DBuf dattn(d, DType::kF32, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  // VT-ATTN-NAIVE: the REFERENCE (non-paged) dense arm, as the comment on the V
  // upcast above already says. Production decode runs `FullAttnBlockPaged`, which
  // replaces this call with vt::ReshapeAndCache + vt::PagedAttention; this arm is
  // what that path is compared against, so a rung change here moves the golden
  // rather than the shipping kernel (#1544).
  vt::Attention(d.q, dattn.t(), qn3, kn3, v3, vt::AttentionArgs{scale, true});

  // Sigmoid output gate on the raw gate split, folded into the o_proj activation
  // quant on the true-W4A4 path (§5) — see SigmoidGateOProjD.
  return SigmoidGateOProjD(d, Reshape(dattn.t(), {T, Hq * Dh}),
                           Reshape(gatef.t(), {T, Hq * Dh}), w, fp4);  // [T,H]
}

// --- Batched PAGED full_attention block (M1.8 Task 3). Identical q/k/v prep to
// FullAttnBlock (gemma qk-RMSNorm + partial NeoX RoPE + GQA + output gate), but
// replaces vt::Attention with vt::ReshapeAndCache (write new K/V into the paged
// NHD cache at slot_mapping) + vt::PagedAttention (read causal K/V from the
// cache via block_table/seq_lens/query_start_loc). Mirrors
// qwen3_next.py::Qwen3NextAttention.forward @ e24d1b24 (self.attn(q,k,v) is the
// reshape_and_cache + paged read). h [T*H] bf16 -> [T*H] bf16.
DBuf FullAttnBlockPaged(Dev d, const FullAttnLayerWeights& w, const HfConfig& cfg,
                        const Tensor& h, const StepDevInputs& sdi,
                        const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                        int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  // KV-FP8 W3: a third storage dtype joins the two float ones — 1-byte fp8
  // (`vt::DType::kI8`), which `dense_attn::IsFp8KvCache` admits only together
  // with a matching fp8 interpretation, so a bare `kI8` view still fails here.
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32 ||
               dense_attn::IsFp8KvCache(kv),
           "full-attn paged: KV cache must be bf16, f32, or 1-byte fp8 "
           "(--kv-cache-dtype fp8)");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "full-attn paged: KV cache head dims mismatch config");

  const bool fp4_attn = !w.q_proj_fp4.Empty();
  const bool packed_consumers =
      FuseAttnPreambleOn(fp4_attn) && sdi.has_attn_cos_sin &&
      vt::OpRegistered(vt::OpId::kAttnQkNormRopeGate, d.q.device.type);
  FullAttnQkvOutput qkv_out =
      ProjectFullAttnQkv(d, w, h, T, h_fp8, packed_consumers);
  const bool fp4 = qkv_out.fp4;
  Tensor qgate = qkv_out.qgate;
  Tensor kf = qkv_out.key;
  Tensor vf = qkv_out.value;

  // Split q|gate + per-head gemma-RMSNorm(q,k) + partial NeoX RoPE + gate
  // passthrough. VT_FUSE_ATTN_PREAMBLE (default OFF) collapses the four ops into
  // ONE launch reading the per-step cos_sin cache (sdi.attn_cos_sin, built once by
  // MaybeBuildAttnCosSin and reused by all 11 full-attn layers) — mirror of vLLM's
  // fused_qk_rmsnorm_rope (fla fused_qk_norm_rope.py:95-102, zero in-kernel
  // transcendentals). Emits f32 q/k/gate: byte-identical intermediates to the
  // four-op path (the query stays f32 for PagedAttention), so ON is token-exact.
  // positions/slot_mapping/... are the PERSISTENT per-step device buffers; sdi.*
  // Tensors are const views over the shared DBufs (no per-layer H2D re-upload).
  //
  // FA-2 (VT_FA2_PREFILL / VT_FA2_DECODE): on an eligible prefill or bounded
  // ratio-6 pure-decode step the preamble emits bf16 q/k (gate stays f32) and
  // the attention output is bf16 — the natively-bf16 combo the FA2 dispatch
  // gate requires, with zero cast kernels. bf16 k feeds the cache directly
  // (skipping the CastBf16 below — bit-identical, both are the RN round of the
  // same f32 value); bf16 attention out feeds SigmoidGateBf16 (exact upcast).
  // Eligibility MUST mirror cuda_paged_attn.cu so every bf16 query is consumed
  // by FA2. The prefill FA2 dispatch (cuda_paged_attn.cu:2494) admits ANY GQA
  // ratio at head_dim 256, so BOTH the 27B (ratio-6) and the 35B (ratio-8) take
  // FA2 prefill (2026-07-18 flip, default-ON via FuseAttnPreambleOn). Pure
  // decode FA2 now covers BOTH gate topologies through the same vendored
  // split-KV kernel: 4B ratio-4 (Hq/Hkv=16/4, VT_FA2_DECODE_4B),
  // 27B ratio-6 (Hq/Hkv=24/4, VT_FA2_DECODE), and 35B ratio-8
  // (Hq/Hkv=16/2, VT_FA2_DECODE_35B — CLAIM-35B-FA2-DECODE-1). All windows /
  // non-causal / non-256 / other-ratio shapes stay f32 on the graph-captured
  // fallback.
  const bool fa2_platform =
      vllm::platforms::GetPlatform(d.q.device.type).supports_fa2_attention();
  // W10 repair (#1865): the FA-2 dtype/lane class is a host-testable seam
  // (ClassifyDenseFa2, qwen3_5_internal.h) instead of an inline predicate the
  // CPU tier could never red. Same inputs, same conjuncts; the spec-as-decode
  // arm selects bf16 through the SPEC lane's own toggles so the verify cannot
  // be starved to f32 by the PREFILL lever (the #1865 dead link).
  const DenseFa2Eligibility fa2_elig{
      /*num_q_heads=*/Hq,
      /*num_kv_heads=*/Hkv,
      /*head_dim=*/Dh,
      /*num_tokens=*/T,
      /*num_reqs=*/meta.num_reqs,
      /*uniform_spec_query_len=*/meta.uniform_spec_query_len,
      /*causal=*/meta.causal,
      /*kv_cache_bf16=*/kv.dtype == DType::kBF16,
      /*kv_block_multiple_16=*/kv.block_size % 16 == 0,
      /*preamble_with_cos_sin=*/FuseAttnPreambleOn(fp4) && sdi.has_attn_cos_sin,
      /*fa2_platform=*/fa2_platform,
      /*prefill_on=*/Fa2PrefillOn(),
      /*decode_r4_on=*/Fa2Decode4BOn(),
      /*decode_r6_on=*/Fa2DecodeOn(),
      /*decode_r8_on=*/Fa2Decode35BOn(),
      /*spec_decode_on=*/Fa2SpecDecodeOn()};
  const bool fa2_attention = ClassifyDenseFa2(fa2_elig) != DenseFa2Class::kNone;
  const DType attn_dt = fa2_attention ? DType::kBF16 : DType::kF32;
  DBuf dq3(d, attn_dt, {T, Hq, Dh});
  DBuf dk3(d, attn_dt, {T, Hkv, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  if (FuseAttnPreambleOn(fp4) && sdi.has_attn_cos_sin) {
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
    // KERNEL-FUSION-FRAMEWORK W2 — route the fused attn preamble through
    // vt::FusedChain(kAttnQkNormRopeGate); composite MACRO = the SAME single
    // vt::AttnQkNormRopeGate launch (perf-neutral + byte-identical). ADOPT=0 rolls back.
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, vt::kAttnQkNormRopeGate, dq3.t(), dk3.t(), gatef.t(), qgate, kf, dqw,
                     dkw, sdi.attn_cos_sin.t(), eps, vt::RopeArgs{base, rot});
    } else {
      vt::AttnQkNormRopeGate(d.q, dq3.t(), dk3.t(), gatef.t(), qgate, kf, dqw, dkw,
                             sdi.attn_cos_sin.t(), vt::RmsNormArgs{eps, true},
                             vt::RopeArgs{base, rot});
    }
  } else {
    DBuf qf(d, DType::kF32, {T, Hq, Dh});
    vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate);
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dqn2d = Reshape(dq3.t(), {T * Hq, Dh});
    vt::RmsNorm(d.q, dqn2d, Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
    // k-norm weight dtype must equal kf's (RmsNorm requires w.dtype == x.dtype). When
    // kf is bf16 (VT_BF16_GEMM_OUT on the fp4 path) use the raw bf16 on-disk k_norm;
    // otherwise (fp8/35B or toggle off) keep the f32 upcast. bf16 kf · bf16 dkw -> f32.
    Tensor dkw = (kf.dtype == DType::kBF16) ? ResidentWeight(d, w.k_norm, {Dh})
                                           : ResidentWeightF32(d, w.k_norm, {Dh});
    Tensor dkn2d = Reshape(dk3.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, dkn2d, Reshape(kf, {T * Hkv, Dh}), dkw,
                vt::RmsNormArgs{eps, true});
    // Keep negative-width validation in the shared primitive. Only zero is no work.
    if (rot != 0) vt::RopeNeox(d.q, dq3.t(), dk3.t(), sdi.positions.t(), vt::RopeArgs{base, rot});
  }
  Tensor qn3 = dq3.t();
  Tensor kn3 = dk3.t();

  Tensor v3 = vf;
  v3.rank = 3;
  v3.shape[0] = T;
  v3.shape[1] = Hkv;
  v3.shape[2] = Dh;
  v3.stride[1] = Dh;
  v3.stride[2] = 1;

  // KV-cache dtype: the production runner allocates a bf16 cache (mirrors vLLM's
  // bf16 flash_attn KV store, halves KV memory); the paged==dense unit anchors
  // allocate an f32 cache to stay bit-exact. The "auto" ReshapeAndCache copy
  // requires cache dtype == k/v dtype, so down-cast the rope'd f32 K and the f32
  // V to bf16 only when the cache is bf16. The query stays f32 either way
  // (Phase 1: f32 query · <cache-dtype> cache, f32-accumulate softmax — the
  // attention kernel converts bf16 cache reads to f32).
  //
  // KV-FP8 W3 (#1593): the fp8 cache takes the SAME bf16 normalisation, and it
  // has to. `vt::ReshapeAndCacheFp8` quantizes from ONE source dtype
  // (`k.dtype == v.dtype`, ops.cpp), and K and V do not arrive in the same one:
  // K is `attn_dt`, which is f32 for every fp8 cache because `kv.dtype ==
  // DType::kBF16` is a term of both FA2 eligibility tests above, while V is
  // whatever the v_proj GEMM emitted — bf16 on the block-wise fp8 arm
  // (`MatmulFp8BlockScaledD`), bf16 on the NVFP4 arm under the default
  // `VT_BF16_GEMM_OUT`, and bf16 on ordinary torch safetensors
  // (`MatmulBf16D`). Only the per-tensor fp8 arm and the transposed
  // GGUF/synthetic path pair f32 with f32, which is why a CPU gate over
  // synthetic weights could not see this. bf16 rather than f32 because bf16 is
  // the dtype upstream quantizes from: its model IS bf16 where
  // `reshape_and_cache_flash` takes key/value (`cache_kernels.cu:314-401`).
  Tensor kw = kn3;
  Tensor vw = v3;
  DBuf kbf(d, DType::kBF16, {T, Hkv, Dh});
  DBuf vbf(d, DType::kBF16, {T, Hkv, Dh});
  if (kv.dtype == DType::kBF16 || dense_attn::IsFp8KvCache(kv)) {
    // K may already be bf16 (an FA2 preamble emits bf16 k directly —
    // the RN round of the same f32 value this CastBf16 would produce); only
    // down-cast when the preamble/fallback produced f32 K.
    if (kn3.dtype == DType::kBF16) {
      kw = kn3;
    } else {
      vt::CastBf16(d.q, kbf.t(), kn3);
      kw = kbf.t();
    }
    // V may already be bf16 (VT_BF16_GEMM_OUT: the fp4 v_proj GEMM emits bf16
    // directly, removing the cutlass CastBf16ToF32 + this CastBf16 round-trip);
    // only down-cast when the GEMM produced f32 V.
    if (v3.dtype == DType::kBF16) {
      vw = v3;
    } else {
      vt::CastBf16(d.q, vbf.t(), v3);
      vw = vbf.t();
    }
  }

  // Write the new K/V into the paged cache, then read K/V from the cache.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  Tensor dslot = sdi.slot_mapping.t();
  Tensor dblk = sdi.block_table.t();
  Tensor dsl = sdi.seq_lens.t();
  Tensor dqsl = sdi.query_start_loc.t();
  // KV-FP8 W3: routes to `vt::ReshapeAndCacheFp8` when this layer's cache is
  // 1-byte fp8, and the cast block above has already put K and V into the ONE
  // model dtype that store quantizes from.
  dense_attn::WriteKvCache(d.q, kv, kw, vw, k_cache, v_cache, dslot);

  // bf16 attention out on an FA2 path (FA2 writes bf16; the sigmoid
  // gate upcast is exact) — f32 everywhere else, byte-identical to today.
  DBuf dattn(d, attn_dt, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  // Hand the prefill flash/WMMA launchers the HOST query_start_loc (already
  // materialized per step by the attention metadata build) so they size the
  // query-tile grid from these host values + a device meta-kernel, skipping the
  // per-layer D2H copy + cudaStreamSynchronize that drained the pipeline every
  // full-attention prefill layer (~10-12 syncs/step; prefill only 43.7%
  // GPU-busy). meta.query_start_loc outlives this call. Device-resident metadata,
  // mirroring the GDN GdnArgs::query_start_loc_host / decode StepDevInputs fix.
  // max_seq_len is the FA-2 launcher's host grid bound (same pattern).
  vt::PagedAttentionArgs pa_args{scale, meta.causal};
  pa_args.query_start_loc_host = meta.query_start_loc.data();
  pa_args.max_seq_len = meta.max_seq_len;
  // SPEC-DFLASH2 W10 (#1857): the runner's spec-as-decode classification — a
  // uniform-qlen verify stays on the FA-2 split-KV DECODE lane instead of the
  // num_splits=1 prefill ladder. 0 on every non-verify step (routing unchanged).
  pa_args.uniform_spec_query_len = meta.uniform_spec_query_len;
  dense_attn::ApplyKvCacheQuant(pa_args, kv);
  vt::PagedAttention(d.q, dattn.t(), qn3, k_cache, v_cache, dblk, dsl, dqsl, pa_args);

  // VT_DUMP_ATTN (issue #41, 0.8B ROCm divergence spike W1/W2): dump the
  // full-attn block's internals per full-attn-layer call index (0-based), as
  // raw little-endian dumps under $VT_DUMP_ATTN/fa<k>_{qkv,q,attn,gate}.bin.
  // Inert when unset; the Downloads sync, so never set on a graph path.
  static thread_local int64_t dump_fa_idx = -1;
  dump_fa_idx++;
  if (std::getenv("VT_DUMP_ATTN") != nullptr) {
    auto DumpT = [&](const char* stage, const Tensor& t) {
      int64_t n = 1;
      for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
      const size_t es = vt::SizeOf(t.dtype);
      // contiguous check: innermost stride 1 and packed
      std::vector<uint8_t> raw(static_cast<size_t>(n) * es);
      DBuf tmp(d, t.dtype, {n});
      d.b.Copy(d.q, tmp.ptr(), t.data, raw.size());
      tmp.Download(d, raw.data());
      const std::string path = std::string(std::getenv("VT_DUMP_ATTN")) +
                               "/fa" + std::to_string(dump_fa_idx) + "_" +
                               stage + ".bin";
      std::FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) { std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f); }
    };
    DumpT("qkv", qgate);
    DumpT("q", qn3);
    DumpT("attn", dattn.t());
    DumpT("gate", gatef.t());
  }

  // Sigmoid output gate, folded into the o_proj activation quant on the true-W4A4
  // path (§5) — see SigmoidGateOProjD.
  return SigmoidGateOProjD(d, Reshape(dattn.t(), {T, Hq * Dh}),
                           Reshape(gatef.t(), {T, Hq * Dh}), w, fp4);  // [T,H]
}

// Per-expert silu-mul MLP over the gathered token rows `x` [n, H] bf16 ->
// [n, H] bf16 (moe-semantics.md §4; gate/up/down kept separate at TP=1).
std::vector<uint16_t> ExpertMlp(Dev d, const OwnedTensor& gate,
                                const OwnedTensor& up, const OwnedTensor& down,
                                const std::vector<uint16_t>& x, int64_t n,
                                int64_t H, int64_t I) {
  std::vector<float> hg = MatmulF32(d, x, n, H, gate);  // [n,I]
  std::vector<float> hu = MatmulF32(d, x, n, H, up);    // [n,I]
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(Silu(hg[i]) * hu[i]);
  return MatmulBf16(d, act, n, I, down);  // [n,H]
}

// A3 W3a: a MatmulBT over a ROW-SLICE [row_off, row_off+N) of a stacked keep-quant
// expert tower `w` [E*out, in]. Byte-IDENTICAL to MatmulF32/MatmulBf16 on a standalone
// per-expert OwnedTensor (same kMatmulBTQuant core, same slice bytes). CRITICAL: the
// whole tower is made DEVICE-RESIDENT via ResidentWeight (CUDA `needs_weight_staging`
// is TRUE), staged ONCE (cached on `w.d_dev`) and REUSED per expert; the slice offsets
// the RESIDENT ptr. The tower is nk=true ([N,K]) so MatmulBT applies (row = whole
// blocks, row_off*row_bytes never cuts a block).
//
// THIS COMMENT USED TO SAY "the kernel CANNOT read host bytes; a raw-host-ptr view
// produces garbage", FULL STOP, AND THAT IS NO LONGER TRUE OF EVERY STAGING PLATFORM
// (ENG-EXPERT-STREAM-DEVICE W0c, issue #1124). It is still true of a DISCRETE one, and
// that is why this function exists and why a discrete device keeps taking it. But
// `expert_stream::HostSliceView` (expert_stream_seam.h, lifted out of this file
// by MODEL-TEXT-GLM-MOE-DSA W3) deliberately builds exactly the raw-host-ptr
// view the old sentence forbade, on a platform whose probed
// `host_memory_is_device_addressable()` says its kernels can follow a host pointer —
// a GB10, where `PageableMemoryAccess` and `Integrated` are both 1 (W0a, measured).
// The predicate is what separates the two cases; "CUDA stages, therefore host bytes
// are unreadable" was a conflation of two different device properties, and it is the
// same conflation `interface.h` warns about where it says `is_unified_memory()` and
// `is_integrated_gpu()` are the WRONG predicate for staging.
Tensor KqResidentSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                       int64_t row_off) {
  const Tensor whole = ResidentWeight(d, w);  // stage/view the WHOLE tower (cached)
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  Tensor wt = whole;  // inherit resident data ptr + device + dtype + repacked
  wt.data = static_cast<void*>(static_cast<uint8_t*>(whole.data) +
                               static_cast<size_t>(row_off) * row_bytes);
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;
  wt.stride[1] = 1;
  return wt;
}

// ENG-EXPERT-STREAM W4 / MODEL-TEXT-GLM-MOE-DSA W3 (#2214): the streamed-expert
// lane now lives in `expert_stream_seam.{h,cpp}` so a second model's translation
// unit can reach it (spec .agents/specs/glm-dsa-latest-deepseek.md §3.7). Until
// that lift this file was the ONLY one that constructed a `HostExpertSlotStore`,
// which is why a new architecture could include every streaming header and still
// not stream. Qwen3.5 is the seam's first client, and these four declarations are
// the whole of what that costs: the lane, the step guard, the request predicate
// and the slice seam are the SAME objects this file used to define, moved rather
// than reimplemented.
//
// `KqExpertSlice` passes THIS file's `KqResidentSlice` in as the fallback, and
// that is load-bearing rather than tidy. `KqResidentSlice` calls the
// `ResidentWeight` defined in this translation unit, which SHADOWS
// `dense_attn_block.h`'s and carries the `expert_streamed`, `elem_kn_repacked`
// and `repacked` staging refusals plus the host-alias arm that the header's
// version does not have. A seam that called `ResidentWeight` itself would bind
// to the header's definition and quietly drop all of them, which is a behaviour
// change to a gated model. Injecting the function keeps the streamed lane
// byte-identical to what it was before the lift, by construction rather than by
// inspection; the seam header carries the full argument.
using Qwen35ExpertStream = ::vllm::expert_stream::ExpertStreamLane;
using Qwen35ExpertStreamStep = ::vllm::expert_stream::ExpertStreamStepGuard;

inline bool Qwen35ExpertStreamRequested() {
  return ::vllm::expert_stream::StreamRequested();
}

Tensor KqExpertSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                     int64_t row_off, int64_t expert) {
  return ::vllm::expert_stream::ExpertSlice(d, w, N, K, row_off, expert,
                                            &KqResidentSlice);
}

std::vector<float> MatmulF32Slice(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                  int64_t N, int64_t K, const OwnedTensor& w,
                                  int64_t row_off, int64_t expert) {
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kF32, {M, N});
  // `expert` is a routed-expert index and is never negative: every call comes
  // from ExpertMlpKq's `for (e = 0; e < E; ++e)`. An earlier revision branched
  // to KqResidentSlice on `expert < 0`, which no caller could reach; the check
  // now states the precondition instead of pretending to handle its negation,
  // because an unreachable fallback is a place for a defect to hide rather than
  // a safety net. KqExpertSlice itself falls back to the tower view whenever
  // streaming is off or the cache cannot serve the slice.
  VT_CHECK(expert >= 0, "qwen3_5: MatmulF32Slice needs a routed expert index");
  Tensor dw = KqExpertSlice(d, w, N, K, row_off, expert);
  vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

std::vector<uint16_t> MatmulBf16Slice(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                      int64_t N, int64_t K, const OwnedTensor& w,
                                      int64_t row_off, int64_t expert) {
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kBF16, {M, N});
  // See MatmulF32Slice: `expert` is always a real routed index here.
  VT_CHECK(expert >= 0, "qwen3_5: MatmulBf16Slice needs a routed expert index");
  Tensor dw = KqExpertSlice(d, w, N, K, row_off, expert);
  vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// A3 W3a: ExpertMlp for expert `e` over the STACKED keep-quant towers (gate/up [E*I,H]
// sliced at e*I; down [E*H,I] sliced at e*H). Byte-IDENTICAL to ExpertMlp on the
// per-expert copies — the ONLY difference is the weight is a slice-view of the whole
// tower, not a standalone OwnedTensor (same bytes, same compute). This makes the
// stacked-tower loader (A3 W2) a byte-exact memory-layout refactor; grouping (W3b) is
// an additive optimization on top.
std::vector<uint16_t> ExpertMlpKq(Dev d, const OwnedTensor& gate_kq,
                                  const OwnedTensor& up_kq, const OwnedTensor& down_kq,
                                  const std::vector<uint16_t>& x, int64_t e, int64_t n,
                                  int64_t H, int64_t I) {
  // Declare the LARGEST of the three slices before taking any of them, so the
  // slot store is sized once and correctly. gate/up and down differ in size
  // whenever a UD (dynamic) quant keeps down_proj at a higher precision, and
  // sizing from whichever slice arrived first would then refuse the first down
  // slice mid-decode. Inert unless streaming was asked for.
  Qwen35ExpertStream::Reserve(
      std::max({static_cast<size_t>(I) * vt::RowSizeBytes(gate_kq.dtype, H),
                static_cast<size_t>(I) * vt::RowSizeBytes(up_kq.dtype, H),
                static_cast<size_t>(H) * vt::RowSizeBytes(down_kq.dtype, I)}));
  std::vector<float> hg = MatmulF32Slice(d, x, n, I, H, gate_kq, e * I, e);  // [n,I]
  std::vector<float> hu = MatmulF32Slice(d, x, n, I, H, up_kq, e * I, e);    // [n,I]
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(Silu(hg[i]) * hu[i]);
  return MatmulBf16Slice(d, act, n, H, I, down_kq, e * H, e);  // [n,H]
}

// W3b: keep-quant grouped MoE — default-ON, VT_QWEN35_GROUPED_MOE=0 restores the
// byte-exact per-expert ExpertMlpKq loop in the same binary.
inline bool Qwen35GroupedMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN35_GROUPED_MOE");
    const bool grouped = (e == nullptr || std::string(e) != "0");
    // The grouped path stages the WHOLE tower through ResidentWeight, so it
    // bypasses the expert slot cache entirely. Leaving both on would silently
    // do no streaming at all while the operator believes it is streaming, which
    // is the invisible-fallback shape this tree refuses elsewhere. Streaming is
    // explicit and opt-in, so it wins, and it says so once.
    if (grouped && Qwen35ExpertStreamRequested()) {
      std::fprintf(stderr,
                   "[expert-stream] VT_MOE_EXPERT_STREAM=1 disables the grouped "
                   "MoE path (it stages the whole tower and cannot stream); set "
                   "VT_MOE_EXPERT_STREAM=0 to keep grouping\n");
      return false;
    }
    return grouped;
  }();
  return on;
}

// W3b: ONE grouped keep-quant GEMM over the stacked [E*N,K] tower (ResidentWeight-
// staged, cached) for ALL P rows — out[p,:] = act[p,:] · tower[eids[p]*N .. +N].
// Replaces P per-expert MatmulF32Slice matvecs with ONE vt::MatmulBTQuantGrouped launch.
// BYTE-IDENTICAL to the per-expert slice (same kMatmulBTQuant integer-dot core + eids
// slice; ds4-gated byte-exact). act bf16, out f32.
std::vector<float> KqGrouped(Dev d, const std::vector<uint16_t>& act, int64_t P,
                             int64_t N, int64_t K, const OwnedTensor& w_kq,
                             const std::vector<int32_t>& eids) {
  DBuf da(d, DType::kBF16, {P, K}, act.data());
  std::vector<int32_t> ids = eids;  // stable buffer backing the device tensor
  DBuf dids(d, DType::kI32, {P}, ids.data());
  DBuf dout(d, DType::kF32, {P, N});
  Tensor w = ResidentWeight(d, w_kq);  // stage the WHOLE [E*N,K] tower (device, cached)
  vt::MatmulBTQuantGrouped(d.q, dout.t(), da.t(), w, dids.t());
  std::vector<float> out(static_cast<size_t>(P) * N);
  dout.Download(d, out.data());  // drains before ids/out leave scope
  return out;
}

// fp4-resident per-expert silu-mul MLP (M2.2b): identical to ExpertMlp but the
// gate/up/down NVFP4 weights are read on-device via vt::MatmulNvfp4.
std::vector<uint16_t> ExpertMlpNvfp4(Dev d, const Nvfp4Weight& gate,
                                     const Nvfp4Weight& up, const Nvfp4Weight& down,
                                     const std::vector<uint16_t>& x, int64_t n,
                                     int64_t H, int64_t I) {
  std::vector<float> hg = MatmulNvfp4F32(d, x, n, H, gate);  // [n,I]
  std::vector<float> hu = MatmulNvfp4F32(d, x, n, H, up);    // [n,I]
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(Silu(hg[i]) * hu[i]);
  return MatmulNvfp4Bf16(d, act, n, I, down);  // [n,H]
}

// Shared-expert MLP (moe-semantics.md §5): silu-mul MLP then sigmoid(x@Wseg) *
// out. Device-resident (M2.5 Phase 1): h [T,H] bf16 (device) -> DBuf [T,H] bf16
// (device); the silu-mul + sigmoid-gate are device ops (not host loops), so the
// shared expert adds no host round-trip to the captured decode step.
// MoE glue-kernel fusion (VT_MOE_GLUE_FUSE, default ON): fold the shared-expert
// sigmoid gate into the weighted MoeCombine (one launch, no shared [T,H]
// round-trip) instead of a separate SharedExpertGate kernel. Bit-identical to
// the two-kernel path (see MoeCombineGate). VT_MOE_GLUE_FUSE=0 restores the
// unfused SharedExpertGate + MoeCombine sequence for A/B.
bool MoeGlueFuseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_GLUE_FUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// --- MoE shared-expert aux-stream overlap (ENG-MOE-SHARED-AUX) ----------------
// Mirror of vLLM's decode overlap: run the shared-expert MLP on a SECOND CUDA
// stream concurrent with the routed-expert router/align/grouped-GEMMs on the main
// stream, then join before the combine so the output is BYTE-IDENTICAL to serial
// (overlap changes WHEN the independent shared path runs, never WHAT it computes).
// vLLM `fused_moe/runner/shared_experts.py:99-104,129-142` gates the aux stream on
// `hidden_states.shape[0] <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (default
// 256) AND cuda; the fork/join primitive is `maybe_execute_in_parallel`
// (`utils/multi_stream_utils.py:20-58`, a TRT-LLM port): event0.record() on main,
// aux waits event0, aux runs fn1 + records event1, main waits event1.
//
// VT_MOE_SHARED_AUX_STREAM gates the whole feature. DEFAULT ON, flipped per the
// parity-enablers-ship-as-defaults policy after the in-situ 35B A/B: token-exact
// (overlap ON == OFF byte-identical, 315/315) AND faster at EVERY tested decode
// point with zero regression — same-binary interleaved TPOT (drop cold rep1)
// c1 −5.6% / c2 −2.7% / c4 −3.7% / c8 −3.4% / c16 −1.6% / c32 −1.8% (GB10's 48 SMs
// leave spare occupancy for the shared MLP to overlap the routed GEMMs across the
// whole low-concurrency band). `VT_MOE_SHARED_AUX_STREAM=0` is the same-binary
// rollback. VT_MOE_SHARED_AUX_THRESHOLD caps the token gate — vLLM's 256 is for
// large-SM GPUs; 128 is the GB10 calibration: it covers every measured decode
// point (T=C<=32 all win) while keeping large prefill/mixed steps (T>128, where
// the routed GEMMs already saturate the GPU) on the serial path. 0 disables the
// gate entirely (never overlap).
#ifdef VT_MARLIN_NVFP4  // overlap is only wired into the Marlin MoE decode path
bool MoeSharedAuxStreamEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_SHARED_AUX_STREAM");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 rolls back
  }();
  return on;
}
int MoeSharedAuxThreshold() {
  static const int t = [] {
    const char* e = std::getenv("VT_MOE_SHARED_AUX_THRESHOLD");
    return (e != nullptr) ? std::atoi(e) : 128;  // GB10 48-SM calibration
  }();
  return t;
}

// Persistent per-device aux stream + two reusable cross-stream events for the
// overlap fork/join. Created lazily on first use (during the eager pre-warm decode
// step, so the queue/events exist BEFORE any CUDA-graph capture) and leaked at
// process exit like the resident weights / cublasLt workspace. ONE global aux
// stream per device mirrors vLLM's single `aux_stream()`
// (utils/torch_utils.py:736-756) — not one per layer — to avoid a stream
// explosion and keep profiling legible.
struct MoeAuxStream {
  Queue q;
  vt::Event fork;  // recorded on main; aux waits it (shared input dh is ready)
  vt::Event done;  // recorded on aux after the shared MLP; main waits it (join)
  bool ready = false;
};
MoeAuxStream& MoeAuxStreamFor(Dev d) {
  static std::mutex mu;
  static std::unordered_map<int, MoeAuxStream> cache;  // key: device index
  std::lock_guard<std::mutex> lk(mu);
  MoeAuxStream& a = cache[d.q.device.index];
  if (!a.ready) {
    a.q = d.b.CreateQueue();
    a.fork = d.b.CreateEvent();
    a.done = d.b.CreateEvent();
    a.ready = true;
  }
  return a;
}
#endif  // VT_MARLIN_NVFP4

// Shared-expert pre-gate parts: sd [T,H] f32 (down projection) and gl [T,1] f32
// (gate logit), before the sigmoid gate + bf16 round. The unfused SharedExpert
// applies SharedExpertGate to these; the fused MoeBlock passes them straight to
// MoeCombineGate.
struct SharedExpertParts {
  DBuf sd;
  DBuf gl;
};

SharedExpertParts SharedExpertUngated(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                                      const Tensor& h, int64_t T, bool fp4) {
  const int64_t Is = cfg.shared_expert_intermediate_size;
#ifdef VT_MARLIN_NVFP4
  // Fused gate_up (VT_MOE_FUSED_W13, dense sibling of the MoE fused w13): ONE
  // Marlin GEMM [T,2Is] + SiluAndMul — vLLM's merged gate_up_proj layout. The
  // silu input values match the unfused path bit-for-bit given equal GEMM
  // outputs: unfused MatmulNvfp4F32D is the SAME Marlin bf16 GEMM upcast to f32
  // (value-preserving), and MoeSiluMul/SiluAndMul share the f32 silu math.
  if (fp4 && vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) && MarlinMoeEnabled() &&
      h.dtype == DType::kBF16 &&
      SharedGateUpFusedEligible(w.shared_gate_proj_fp4, w.shared_up_proj_fp4)) {
    DBuf sact = SharedGateUpFusedMarlinD(d, h, w.shared_gate_proj_fp4, w.shared_up_proj_fp4);
    // bf16 down-proj out (VT_SHARED_DOWN_BF16, default ON): the Marlin GEMM
    // already produces bf16 and BOTH consumers re-round through bf16, so the
    // f32 form wrote and re-read a whole [T,H] buffer for a value it had.
    // Bit-identical; drops one CastF32 launch per layer per step (CastF32 was
    // measured at 3.1% of the 35B decode step).
    DBuf sd = dense_nvfp4::SharedDownBf16Enabled()
                  ? MatmulNvfp4Bf16D(d, sact.t(), w.shared_down_proj_fp4)   // [T,H] bf16
                  : MatmulNvfp4F32D(d, sact.t(), w.shared_down_proj_fp4);   // [T,H] f32
    DBuf gl = MatmulF32D(d, h, w.shared_gate);                       // [T,1] f32
    return {std::move(sd), std::move(gl)};
  }
#endif
  DBuf sg = fp4 ? MatmulNvfp4F32D(d, h, w.shared_gate_proj_fp4)
                : MatmulF32D(d, h, w.shared_gate_proj);  // [T,Is]
  DBuf su = fp4 ? MatmulNvfp4F32D(d, h, w.shared_up_proj_fp4)
                : MatmulF32D(d, h, w.shared_up_proj);    // [T,Is]
  DBuf sact(d, DType::kBF16, {T, Is});
  vt::MoeSiluMul(d.q, sact.t(), sg.t(), su.t());  // silu(sg) * su -> bf16
  DBuf sd = fp4 ? MatmulNvfp4F32D(d, sact.t(), w.shared_down_proj_fp4)
                : MatmulF32D(d, sact.t(), w.shared_down_proj);  // [T,H] f32
  DBuf gl = MatmulF32D(d, h, w.shared_gate);                    // [T,1] f32
  return {std::move(sd), std::move(gl)};
}

DBuf SharedExpert(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                  const Tensor& h, int64_t T, bool fp4) {
  const int64_t H = cfg.hidden_size;
  SharedExpertParts p = SharedExpertUngated(d, w, cfg, h, T, fp4);
  DBuf shared(d, DType::kBF16, {T, H});
  vt::SharedExpertGate(d.q, shared.t(), p.sd.t(), p.gl.t());  // sigmoid(gl)*sd -> bf16
  return shared;
}

// The fused MoE blocks' router GEMM, in EITHER weight orientation.
//
// The router gate reaches the two fp4 fused blocks below in one of two layouts,
// and which one is not a property of the model but of the LOADER:
//   * the safetensors loader transposes it to Matmul-B [K=H, N=E] (`nk == false`,
//     LoadBf16Transposed, qwen3_5_weights.cpp:362);
//   * the GGUF loader keeps the file's own [N=E, K=H] under `expand_nk`
//     (`nk == true`, OwnMatmulWeight -> qwen3_5_gguf_weights.cpp:1195), which is
//     the DEFAULT wherever the running device can execute the quantized GEMM.
// The reference MoE loop has always branched on that flag (MatmulBf16, :5082) and
// the bf16 fast path REFUSES the nk=true layout outright (MoeBf16FastLayoutOk,
// :576) - but the fp4 fused blocks hardcoded `vt::Matmul` and so assumed the
// safetensors layout. That assumption held only because a GGUF load never
// produced fp4-resident experts; `QUANT-GGUF-NVFP4` column C's MoE arm made it
// reachable, and the first 35B NVFP4 GGUF forward threw
// "vt: matmul: inner dims mismatch" here (a [T,2048] activation against a
// [256,2048] gate). Falling through to the reference loop is not an option for
// this block - its bf16 expert fields are EMPTY by construction - so the fused
// path HANDLES the orientation instead, exactly as MatmulBf16D does. Inert for
// the safetensors path: `nk == false` still issues the identical vt::Matmul.
void MoeRouterLogits(Dev d, Tensor& out, const Tensor& dh,
                     const OwnedTensor& router_gate) {
  Tensor drg = ResidentWeight(d, router_gate);
  if (router_gate.nk)
    vt::MatmulBT(d.q, out, dh, drg);  // gate [N=E, K=H]
  else
    vt::Matmul(d.q, out, dh, drg);    // gate [K=H, N=E]
}

// --- Fused MoE block (M2.4). CUDA + fp4-resident only. Replaces the per-expert
// loop of tiny MatmulNvfp4 launches (each with a host round-trip Download) with
// ~3 GROUPED NVFP4 GEMM launches over ALL (token, activated-expert) pairs, kept
// entirely on-device (no host round-trip in the expert compute). The router
// top-k indices [T,top_k] ARE the per-pair expert ids (viewed as [P=T*top_k]);
// gate/up read the token hidden via a row-map (pair p -> token p/top_k), down
// reads the per-pair silu output. The E fp4-resident expert weights are indexed
// by device pointer arrays. Result is per-pair bit-identical to ExpertMlpNvfp4
// (same on-the-fly NVFP4 decode + f32 accumulation); the combine + shared expert
// match MoeBlock exactly. h [T*H] bf16 -> [T*H] bf16.
DBuf MoeBlockFusedCuda(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                       const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;

  // Hidden states already device-resident (dh [T,H] bf16); router logits +
  // top-k stay on device (the ids [T,top_k] are the pair expert ids, the
  // weights [T,top_k] feed the combine).
  DBuf dlog(d, DType::kBF16, {T, E});
  MoeRouterLogits(d, dlog.t(), dh, w.router_gate);        // logits [T,E]
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  Tensor eids = Reshape(dtid.t(), {P});                   // [P] i32 expert ids

  // Per-layer RESIDENT fp4-expert device pointer/scale arrays + pair->token map
  // (MoeResidentFor cache): uploaded ONCE (here on first touch, during the
  // pre-warm forward), then read on every step — no per-step host-sourced upload,
  // so nothing dangles inside a captured graph. residency of the expert weights
  // themselves is still lazy through ResidentNvfp4 (their shared_ptr owns the
  // device copy); we capture the resulting stable device pointers once.
  MoeFusedResident& mr = MoeResidentFor(&w);
  if (!mr.ready) {
    std::vector<int64_t> gp(E), gs(E), up(E), us(E), dp(E), ds(E);
    std::vector<float> g2(E), u2(E), d2(E);
    for (int64_t e = 0; e < E; ++e) {
      const size_t se = static_cast<size_t>(e);
      Nvfp4Dev g = ResidentNvfp4(d, w.expert_gate_fp4[se]);
      Nvfp4Dev u = ResidentNvfp4(d, w.expert_up_fp4[se]);
      Nvfp4Dev dn = ResidentNvfp4(d, w.expert_down_fp4[se]);
      gp[se] = reinterpret_cast<int64_t>(g.packed.data);
      gs[se] = reinterpret_cast<int64_t>(g.scale.data);
      up[se] = reinterpret_cast<int64_t>(u.packed.data);
      us[se] = reinterpret_cast<int64_t>(u.scale.data);
      dp[se] = reinterpret_cast<int64_t>(dn.packed.data);
      ds[se] = reinterpret_cast<int64_t>(dn.scale.data);
      g2[se] = w.expert_gate_fp4[se].scale2;
      u2[se] = w.expert_up_fp4[se].scale2;
      d2[se] = w.expert_down_fp4[se].scale2;
    }
    const size_t eb = static_cast<size_t>(E) * sizeof(int64_t);
    const size_t fb = static_cast<size_t>(E) * sizeof(float);
    auto up_i64 = [&](const std::vector<int64_t>& h) {
      void* p = d.b.Alloc(eb);
      d.b.Copy(d.q, p, h.data(), eb);
      return p;
    };
    auto up_f32 = [&](const std::vector<float>& h) {
      void* p = d.b.Alloc(fb);
      d.b.Copy(d.q, p, h.data(), fb);
      return p;
    };
    mr.gp = up_i64(gp); mr.gs = up_i64(gs); mr.up = up_i64(up);
    mr.us = up_i64(us); mr.dp = up_i64(dp); mr.ds = up_i64(ds);
    mr.g2 = up_f32(g2); mr.u2 = up_f32(u2); mr.d2 = up_f32(d2);
    mr.ready = true;
  }
  // Resident pair->token row map for this T (function of T + top_k only).
  auto tok_it = mr.tok_map.find(T);
  if (tok_it == mr.tok_map.end()) {
    std::vector<int32_t> tok_map(static_cast<size_t>(P));
    for (int64_t p = 0; p < P; ++p)
      tok_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    const size_t tb = static_cast<size_t>(P) * sizeof(int32_t);
    void* p = d.b.Alloc(tb);
    d.b.Copy(d.q, p, tok_map.data(), tb);
    tok_it = mr.tok_map.emplace(T, p).first;
  }
  Tensor dgp = MakeTensor(mr.gp, DType::kI64, d.q.device, {E});
  Tensor dgs = MakeTensor(mr.gs, DType::kI64, d.q.device, {E});
  Tensor dup = MakeTensor(mr.up, DType::kI64, d.q.device, {E});
  Tensor dus = MakeTensor(mr.us, DType::kI64, d.q.device, {E});
  Tensor ddp = MakeTensor(mr.dp, DType::kI64, d.q.device, {E});
  Tensor dds = MakeTensor(mr.ds, DType::kI64, d.q.device, {E});
  Tensor dg2 = MakeTensor(mr.g2, DType::kF32, d.q.device, {E});
  Tensor du2 = MakeTensor(mr.u2, DType::kF32, d.q.device, {E});
  Tensor dd2 = MakeTensor(mr.d2, DType::kF32, d.q.device, {E});
  Tensor dtok = MakeTensor(tok_it->second, DType::kI32, d.q.device, {P});

  // Grouped gate/up GEMM over all pairs (one launch each), silu-mul, grouped
  // down GEMM (act = per-pair silu output, identity row-map). expert_out lands
  // as [T,top_k,H] contiguous — exactly what MoeCombine consumes.
  DBuf dgate(d, DType::kF32, {P, I});
  DBuf dup_out(d, DType::kF32, {P, I});
  vt::MoeGroupedGemmNvfp4(d.q, dgate.t(), dh, eids, &dtok, dgp, dgs, dg2);
  vt::MoeGroupedGemmNvfp4(d.q, dup_out.t(), dh, eids, &dtok, dup, dus, du2);
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());
  DBuf ddown(d, DType::kBF16, {P, H});
  vt::MoeGroupedGemmNvfp4(d.q, ddown.t(), dact.t(), eids, nullptr, ddp, dds, dd2);
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  // Shared expert + weighted combine (out = shared + sum_j w_j * expert_out_j),
  // all device-resident (no host round-trip). MoE glue fusion folds the shared
  // sigmoid gate into the combine (one launch, no shared [T,H] round-trip).
  DBuf dout(d, DType::kBF16, {T, H});
  if (MoeGlueFuseEnabled()) {
    SharedExpertParts sp = SharedExpertUngated(d, w, cfg, dh, T, true);
    vt::MoeCombineGate(d.q, dout.t(), expert_out, dtw.t(), sp.sd.t(), sp.gl.t());
  } else {
    DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
  }
  return dout;
}

// --- Sparse-MoE block (moe-semantics.md §1-§6). Router top-k over ALL experts,
// then the ACTIVATED-EXPERT token-gather loop (not O(E)-dense), shared expert
// with sigmoid gate, and the weighted combine. h [T*H] bf16 -> [T*H] bf16.
#ifdef VT_MARLIN_NVFP4
// Build (once) the per-layer resident Marlin repacked experts. Repacks all E
// routed experts' gate/up/down (weight interleave + S0E5M3 scales + processed
// global scales) from the resident fp4 buffers, then frees the fp4 originals.
void BuildMoeMarlinResident(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                            MoeMarlinResident& mr) {
  const int E = static_cast<int>(cfg.num_experts);
  const int K = static_cast<int>(cfg.hidden_size);
  const int N = static_cast<int>(cfg.moe_intermediate_size);
  void* stream = d.q.handle;

  const int sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
  mr.sms = sms;

  const size_t wg_i32 = static_cast<size_t>(K / 16) * (N * 2);  // gate/up weight elems
  const size_t wd_i32 = static_cast<size_t>(N / 16) * (K * 2);  // down weight elems
  const size_t sg_b = static_cast<size_t>(K / 16) * N;          // gate/up scale bytes
  const size_t sd_b = static_cast<size_t>(N / 16) * K;          // down scale bytes

  // Fused w13 (VT_MOE_FUSED_W13): one Marlin B operand per expert with gate+up
  // concatenated along N — needs ONE per-expert global scale for both halves
  // (the grouped GEMM takes global_scale[e], a scalar per expert). Mirror vLLM,
  // which checks allclose(w13_weight_scale_2[:, 0], w13_weight_scale_2[:, 1])
  // and then uses [:, 0] (modelopt.py:1556-1564, "Use a single gscale for
  // w13"). vLLM merely WARNS on mismatch ("Accuracy may be affected"); our
  // token-exact gate forbids that, so on any gate-vs-up scale2 mismatch we fall
  // back to the split two-GEMM layout (and say so) instead of degrading.
  bool fuse = MoeFusedW13Enabled();
  for (int e = 0; fuse && e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    if (w.expert_gate_fp4[se].scale2 != w.expert_up_fp4[se].scale2) {
      std::fprintf(stderr,
                   "vllm.cpp: VT_MOE_FUSED_W13: expert %d gate/up scale2 differ "
                   "(%g vs %g) — falling back to the split w13 layout\n",
                   e, static_cast<double>(w.expert_gate_fp4[se].scale2),
                   static_cast<double>(w.expert_up_fp4[se].scale2));
      fuse = false;
    }
  }
  mr.fused_w13 = fuse;

  if (fuse) {
    // Same total bytes as the split w_gate+w_up / s_gate+s_up pair.
    mr.w_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * wg_i32 * 4);
    mr.s_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * sg_b);
    mr.g_gu = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  } else {
    mr.w_gate = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.s_gate = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.s_up = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.g_gate = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
    mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  }
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  // marlin_make_workspace_new(device, max_blocks_per_sm=4): sms*4 int32 locks.
  mr.workspace = d.b.Alloc(static_cast<size_t>(sms) * 4 * sizeof(int32_t));

  // combined_scale_factor: w13 = gate+up jointly (mirror vLLM), w2 = down alone.
  std::vector<const uint8_t*> gu_bufs;
  std::vector<size_t> gu_lens;
  std::vector<const uint8_t*> dn_bufs;
  std::vector<size_t> dn_lens;
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_gate_fp4[se].scale.bytes.data()));
    gu_lens.push_back(w.expert_gate_fp4[se].scale.bytes.size());
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_up_fp4[se].scale.bytes.data()));
    gu_lens.push_back(w.expert_up_fp4[se].scale.bytes.size());
    dn_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_down_fp4[se].scale.bytes.data()));
    dn_lens.push_back(w.expert_down_fp4[se].scale.bytes.size());
  }
  const float sf_gu = vt::cuda::MarlinNvfp4CombinedScaleFactor(gu_bufs, gu_lens);
  const float sf_dn = vt::cuda::MarlinNvfp4CombinedScaleFactor(dn_bufs, dn_lens);

  // Fused-w13 concat staging (device, reused across experts — all copies and
  // repack kernels are issued on the SAME stream, so each expert's repack reads
  // its staging bytes before the next expert's copy overwrites them). The fp4
  // source layouts are row-major over N (packed [N, K/2] u8, scales [N, K/16]
  // fp8), so the vLLM w13 stack — w1 (gate) rows first, then w3 (up)
  // (fused_moe layer weight_loader shard order; silu_and_mul reads [:N] as
  // gate) — is a flat back-to-back device copy.
  const size_t pk_b = static_cast<size_t>(N) * (K / 2);  // one shard's packed bytes
  uint8_t* tmp_w = nullptr;
  uint8_t* tmp_s = nullptr;
  if (fuse) {
    tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
    tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sg_b));
  }

  std::vector<float> gg(E), gu(E), gd(E);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    Nvfp4Dev g = ResidentNvfp4(d, w.expert_gate_fp4[se]);
    Nvfp4Dev u = ResidentNvfp4(d, w.expert_up_fp4[se]);
    Nvfp4Dev dn = ResidentNvfp4(d, w.expert_down_fp4[se]);
    auto* wd = static_cast<uint32_t*>(mr.w_down) + se * wd_i32;
    auto* sdp = static_cast<uint8_t*>(mr.s_down) + se * sd_b;
    const auto* pg = static_cast<const uint8_t*>(g.packed.data);
    const auto* pu = static_cast<const uint8_t*>(u.packed.data);
    const auto* pd = static_cast<const uint8_t*>(dn.packed.data);
    if (fuse) {
      // ONE repack + ONE scale-process over the N-concatenated gate|up, with
      // size_n = 2N — the per-expert body of vLLM's repack_weight/
      // permute_scales over the stacked w13 (marlin_utils_fp4.py:388-398,
      // :423-434; size_n = num_shards * N at :375-378 / :413-415).
      auto* wgu = static_cast<uint32_t*>(mr.w_gu) + se * 2 * wg_i32;
      auto* sgup = static_cast<uint8_t*>(mr.s_gu) + se * 2 * sg_b;
      d.b.Copy(d.q, tmp_w, pg, pk_b);
      d.b.Copy(d.q, tmp_w + pk_b, pu, pk_b);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wgu, tmp_w, K, 2 * N);
      d.b.Copy(d.q, tmp_s, g.scale.data, sg_b);
      d.b.Copy(d.q, tmp_s + sg_b, u.scale.data, sg_b);
      vt::cuda::MarlinProcessExpertScales(stream, tmp_s, sgup, K, 2 * N, sf_gu);
      // vLLM w13_weight_scale_2[:, 0] (the gate/w1 scale; equality with up/w3
      // was verified above).
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_gate_fp4[se].scale2, sf_gu);
    } else {
      auto* wg = static_cast<uint32_t*>(mr.w_gate) + se * wg_i32;
      auto* wu = static_cast<uint32_t*>(mr.w_up) + se * wg_i32;
      auto* sgp = static_cast<uint8_t*>(mr.s_gate) + se * sg_b;
      auto* sup = static_cast<uint8_t*>(mr.s_up) + se * sg_b;
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wg, pg, K, N);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wu, pu, K, N);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(g.scale.data), sgp,
                                          K, N, sf_gu);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(u.scale.data), sup,
                                          K, N, sf_gu);
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_gate_fp4[se].scale2, sf_gu);
      gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_up_fp4[se].scale2, sf_gu);
    }
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wd, pd, N, K);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dn.scale.data), sdp, N,
                                        K, sf_dn);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_down_fp4[se].scale2, sf_dn);
  }
  if (fuse) {
    d.b.Copy(d.q, mr.g_gu, gg.data(), gg.size() * sizeof(float));
  } else {
    d.b.Copy(d.q, mr.g_gate, gg.data(), gg.size() * sizeof(float));
    d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  }
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // repack done → safe to free fp4 originals
  if (fuse) {
    d.b.Free(tmp_w);
    d.b.Free(tmp_s);
  }

  // The device Marlin resident (mr.w_*/s_*/g_*) is now the committed compute path
  // for EVERY forward step. Free the per-expert fp4 device transients used only
  // for the repack, AND — the decisive 35B host-memory lever — the ~16.9 GiB
  // HOST mirror of the routed experts' packed fp4 codes + fp8 group scales
  // (LoadNvfp4Raw's MakeOwned+memcpy, retained in expert_*_fp4[e].{packed,scale}
  // .bytes). On GB10 the host .bytes and the device d_packed are DISTINCT
  // allocations (ResidentNvfp4 Alloc+Copy), so once repacked into the Marlin
  // resident the host copies are dead weight (they inflated peak PSS ~21GB vs
  // vLLM 13.3GB). Returning them realizes
  // residency_policy().release_host_weights_after_upload (BACKEND-PLATFORM
  // item 2) for the dominant host consumer; see
  // .agents/specs/moe-marlin-host-free.md.
  //
  // POLICY vs KERNEL-PATH split (BACKEND-PLATFORM item 2): the two questions are
  // ORTHOGONAL. (a) WHETHER the platform frees host weight bytes after the device
  // upload is the residency POLICY — read from
  // GetPlatform(d.q.device.type).residency_policy().release_host_weights_after_upload
  // (true on GB10/CUDA today, so identical behavior; a discrete GPU or a future
  // retain-host platform flips it with NO edit here). (b) WHETHER Marlin is the
  // committed compute path is the KERNEL gate MarlinMoeEnabled(): the
  // VT_NVFP4_MARLIN=0 wmma fallback (MoeBlockFusedCuda) re-reads these host bytes
  // via ResidentNvfp4, so freeing is safe ONLY when that path can never run.
  // MarlinMoeEnabled() is a process-static const and TRUE here by construction
  // (BuildMoeMarlinResident is reached only from MoeBlockFusedMarlinCuda, itself
  // gated on MarlinMoeEnabled()); the explicit re-check enforces the invariant.
  // Only the routed experts' bytes are freed (never the shared-expert bytes, which
  // the forward still reads every step).
  // Same-binary A/B rollback (default ON): VT_MOE_HOST_FREE=0 retains the host
  // copies on the Marlin compute path, isolating exactly the host-free effect
  // for a peak-PSS A/B without changing the device compute (unlike flipping
  // VT_NVFP4_MARLIN, which also swaps the GEMM path). Process-static like the
  // Marlin gate; consumed as the `host_free_env` override to the platform policy.
  static const bool host_free_on = [] {
    const char* e = std::getenv("VT_MOE_HOST_FREE");
    return !(e != nullptr && e[0] == '0');
  }();
  const bool release_host = vllm::platforms::ShouldReleaseHostWeights(
      vllm::platforms::GetPlatform(d.q.device.type).residency_policy(),
      /*marlin_committed=*/MarlinMoeEnabled(), /*host_free_env=*/host_free_on);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    w.expert_gate_fp4[se].d_packed.reset();
    w.expert_gate_fp4[se].d_scale.reset();
    w.expert_up_fp4[se].d_packed.reset();
    w.expert_up_fp4[se].d_scale.reset();
    w.expert_down_fp4[se].d_packed.reset();
    w.expert_down_fp4[se].d_scale.reset();
    if (release_host) {
      w.expert_gate_fp4[se].packed.ReleaseHost();
      w.expert_gate_fp4[se].scale.ReleaseHost();
      w.expert_up_fp4[se].packed.ReleaseHost();
      w.expert_up_fp4[se].scale.ReleaseHost();
      w.expert_down_fp4[se].packed.ReleaseHost();
      w.expert_down_fp4[se].scale.ReleaseHost();
    }
  }
  mr.ready = true;
}

// Marlin fused MoE block: same router/silu/combine as MoeBlockFusedCuda, but the
// grouped GEMMs are moe_wna16_marlin_gemm over the resident repacked experts —
// 3 of them (gate, up, down), or 2 when VT_MOE_FUSED_W13 built the concatenated
// w13 operand (ONE gate+up GEMM with size_n=2I, vLLM marlin_moe.py:133-170).
// Marlin's per-pair output layout (row = t*top_k+k) matches dgate/dup/ddown, so
// MoeSiluMul/SiluAndMul + MoeCombine are unchanged. Per-pair equivalent to the
// wmma path (same weight-only fp4 dequant), so token-for-token identical.
DBuf MoeBlockFusedMarlinCuda(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                             const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;
  void* stream = d.q.handle;

  MoeMarlinResident& mr = MoeMarlinResidentFor(&w);
  if (!mr.ready) BuildMoeMarlinResident(d, w, cfg, mr);

  // --- Shared-expert overlap FORK (ENG-MOE-SHARED-AUX) -----------------------
  // When gated ON (VT_MOE_SHARED_AUX_STREAM, T <= threshold), issue the shared-
  // expert MLP on the aux stream NOW so it runs concurrently with the routed
  // router/align/grouped-GEMMs issued on the main stream below. dh (the block
  // input) is already resident on the main stream from the preceding norm, so the
  // aux stream only needs to wait the main stream's fork point before reading it.
  // The shared path and the routed path are INDEPENDENT (they read the same dh,
  // write disjoint buffers) and both complete before the combine at the bottom,
  // so the output is byte-identical to the serial order. Mirrors
  // maybe_execute_in_parallel (multi_stream_utils.py:47-54): event0.record() on
  // main, aux waits event0, aux runs fn1, aux records event1.
  const bool aux_overlap = MoeSharedAuxStreamEnabled() &&
                           d.b.SupportsAuxStream() &&
                           T <= static_cast<int64_t>(MoeSharedAuxThreshold());
  std::optional<SharedExpertParts> sp_aux;
  MoeAuxStream* ax = nullptr;
  if (aux_overlap) {
    ax = &MoeAuxStreamFor(d);
    d.b.RecordEvent(ax->fork, d.q);       // event0.record() on the main stream
    d.b.QueueWaitEvent(ax->q, ax->fork);  // aux stream waits event0 (dh ready)
    Dev auxd{d.b, ax->q};
    // Draw the shared path's scratch from AuxPool so the concurrent main-stream
    // routed allocations never share a live block with it (see AuxPool()).
    ActivePoolScope guard(&AuxPool(d.b));
    sp_aux.emplace(SharedExpertUngated(auxd, w, cfg, dh, T, true));
    d.b.RecordEvent(ax->done, ax->q);     // event1.record() on the aux stream
  }

  // Router (identical to MoeBlockFusedCuda), in either weight orientation.
  DBuf dlog(d, DType::kBF16, {T, E});
  MoeRouterLogits(d, dlog.t(), dh, w.router_gate);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});

  // moe_align_block_size over the router top-k ids.
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(T),
                                                            static_cast<int>(top_k),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(static_cast<int>(T), static_cast<int>(top_k), static_cast<int>(E),
                                block, &max_tok, &max_blk);
  DBuf sorted_ids(d, DType::kI32, {max_tok});
  DBuf expert_ids(d, DType::kI32, {max_blk});
  DBuf num_pad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(
      stream, static_cast<const int32_t*>(dtid.t().data), static_cast<int>(T),
      static_cast<int>(top_k), static_cast<int>(E), block,
      static_cast<int32_t*>(sorted_ids.t().data), static_cast<int32_t*>(expert_ids.t().data),
      static_cast<int32_t*>(num_pad.t().data));

  // SPEC-DSPARK #442 diagnostic (VT_MOE_PAD_STATS=1, DEFAULT OFF, adds a D2H
  // sync so it must never be on for a timing run). The Marlin MoE kernel loops
  // div_ceil(num_tokens_past_padded, block) blocks PER LAUNCH, so two engines can
  // launch it the same number of times and still do different amounts of work if
  // their tokens route to different numbers of experts. Ours and upstream's token
  // streams DIVERGE by a near-tie, so this is what decides whether the measured
  // 8.2% kernel gap is an implementation defect or just a different token path.
  static const bool pad_stats = [] {
    const char* v = std::getenv("VT_MOE_PAD_STATS");
    return v != nullptr && v[0] == '1';
  }();
  if (pad_stats) {
    int32_t npad = 0;
    d.b.Copy(d.q, &npad, num_pad.t().data, sizeof(int32_t));
    d.b.Synchronize(d.q);
    static int64_t calls = 0, sum_pad = 0, sum_blocks = 0;
    ++calls;
    sum_pad += npad;
    sum_blocks += (npad + block - 1) / block;
    if (calls % 500 == 0)
      std::fprintf(stderr,
                   "[MOE-PAD] calls=%lld padded_tokens_total=%lld blocks_total=%lld "
                   "avg_pad=%.1f avg_blocks=%.1f (block=%d T=%d topk=%d E=%d)\n",
                   static_cast<long long>(calls), static_cast<long long>(sum_pad),
                   static_cast<long long>(sum_blocks),
                   static_cast<double>(sum_pad) / calls,
                   static_cast<double>(sum_blocks) / calls, block, static_cast<int>(T),
                   static_cast<int>(top_k), static_cast<int>(E));
  }

  Tensor wd = MakeTensor(mr.w_down, DType::kI32, d.q.device, {E, I / 16, H * 2});
  Tensor sd = MakeTensor(mr.s_down, DType::kI8, d.q.device, {E, I / 16, H});
  Tensor gd = MakeTensor(mr.g_down, DType::kF32, d.q.device, {E});
  Tensor ws = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  const int bi = block, tki = static_cast<int>(top_k);
  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int Pi = static_cast<int>(P);

  DBuf dact(d, DType::kBF16, {P, I});
  if (mr.fused_w13) {
    // ONE grouped GEMM over the N-concatenated w13 (size_n=2I, output [P,2I])
    // + one SiluAndMul on the halves — vLLM's exact marlin_moe.py shape (ONE
    // moe_wna16_marlin_gemm with size_n = w13_num_shards*N into
    // intermediate_cache1 [M*topk, 2N], fused_moe/experts/marlin_moe.py:133-160,
    // then silu_and_mul at :162-170). Removes the second GEMM's workspace
    // memset, schedule pass, and launch tail.
    Tensor wgu = MakeTensor(mr.w_gu, DType::kI32, d.q.device, {E, H / 16, 2 * I * 2});
    Tensor sgu = MakeTensor(mr.s_gu, DType::kI8, d.q.device, {E, H / 16, 2 * I});
    Tensor ggu = MakeTensor(mr.g_gu, DType::kF32, d.q.device, {E});
    DBuf dgu(d, DType::kBF16, {P, 2 * I});
    // No workspace memset: Marlin self-resets its reduction locks to 0 at
    // completion (marlin_template.h:200-205 barrier_release reset=true) and the
    // buffer is zeroed once at build (BuildMoeMarlinResident), so it re-enters
    // every GEMM all-zero. memcheck-verified redundant (L4).
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dgu.t(), dh, wgu, sgu, ggu, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, 2 * Ii, Hi, false});
    // SiluAndMul reads gate = dgu[:, :I], up = dgu[:, I:] (same row) — identical
    // f32 silu math + bf16 store as MoeSiluMul, so per-element it matches the
    // split path bit-for-bit given equal GEMM outputs.
    vt::SiluAndMul(d.q, dact.t(), dgu.t());
  } else {
    Tensor wg = MakeTensor(mr.w_gate, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor wu = MakeTensor(mr.w_up, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor sg = MakeTensor(mr.s_gate, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor su = MakeTensor(mr.s_up, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor gg = MakeTensor(mr.g_gate, DType::kF32, d.q.device, {E});
    Tensor gu = MakeTensor(mr.g_up, DType::kF32, d.q.device, {E});
    DBuf dgate(d, DType::kBF16, {P, I});
    DBuf dup_out(d, DType::kBF16, {P, I});
    // No workspace memset (L4): Marlin self-resets locks + build-time zero-init
    // leave the buffer all-zero at every GEMM entry (see fused branch above).
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dgate.t(), dh, wg, sg, gg, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dup_out.t(), dh, wu, su, gu, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
    vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());
  }

  DBuf ddown(d, DType::kBF16, {P, H});
  // No workspace memset (L4): see the fused/split branches above — Marlin
  // self-resets locks + build-time zero-init keep the buffer all-zero.
  vt::MoeGroupedGemmNvfp4Marlin(d.q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted_ids.t(),
                                expert_ids.t(), num_pad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, 1, Pi, Hi, Ii, false});
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  // --- Shared-expert overlap JOIN (ENG-MOE-SHARED-AUX) -----------------------
  // Make the main stream wait for the aux shared MLP (event1.wait) so the combine
  // below reads a fully-computed sp.sd/sp.gl. Both the routed path (main) and the
  // shared path (aux) are now complete → the combine result is byte-identical to
  // the serial order. When overlap is OFF, the shared expert is computed inline on
  // the main stream exactly as before (no fork/join, no aux pool).
  if (aux_overlap) d.b.QueueWaitEvent(d.q, ax->done);  // event1.wait() on main

  DBuf dout(d, DType::kBF16, {T, H});
  if (MoeGlueFuseEnabled()) {
    // Fuse shared-expert gate into the combine (one launch, no shared round-trip).
    SharedExpertParts sp =
        aux_overlap ? std::move(*sp_aux) : SharedExpertUngated(d, w, cfg, dh, T, true);
    vt::MoeCombineGate(d.q, dout.t(), expert_out, dtw.t(), sp.sd.t(), sp.gl.t());
  } else if (aux_overlap) {
    // Unfused A/B path with overlap: the aux stream produced the ungated parts;
    // apply the sigmoid gate on the main stream (post-join) then combine.
    DBuf shared(d, DType::kBF16, {T, H});
    vt::SharedExpertGate(d.q, shared.t(), sp_aux->sd.t(), sp_aux->gl.t());
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
  } else {
    DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
  }
  return dout;
}
#endif  // VT_MARLIN_NVFP4

// --- Fast BF16 grouped MoE block (Qwen3-Coder Qwen3MoeForCausalLM, W5). The
// bf16-native analog of MoeBlockFusedCuda: replaces the per-expert host-gather
// loop of MoeBlock's reference branch (download hidden -> host router gather ->
// E serialized cuBLASLt ExpertMlp launches) with ~3 GROUPED bf16 GEMM launches
// kept entirely on-device (no host round-trip in the expert compute). The router
// top-k ids [T,top_k] ARE the per-pair expert ids (viewed [P=T*top_k]); gate/up
// read the token hidden via the resident pair->token row map, down reads the
// per-pair silu output. Structurally mirrors MoeBlockFusedCuda; the fp4 on-the-fly
// decode is replaced by a direct bf16 weight read (vt::MoeGroupedGemmBf16). f32
// gate/up outputs + f32 silu match the reference ExpertMlp (MatmulF32 then
// F32ToBF16(silu*up)), so each output row stays within the near-tie band of the
// reference path it replaces. Shared expert GUARDED on
// shared_expert_intermediate_size>0 (Coder has none; SEAM GAP #3). CUDA only.
DBuf MoeBlockBf16Cuda(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                      const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;

  // Router: logits = dh @ gate (bf16), softmax/top-k/renormalize — all on device.
  Tensor drg = ResidentWeight(d, w.router_gate);  // [H,E] bf16
  DBuf dlog(d, DType::kBF16, {T, E});
  vt::Matmul(d.q, dlog.t(), dh, drg);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  Tensor eids = Reshape(dtid.t(), {P});  // [P] i32 expert ids

  // Per-layer RESIDENT expert device-pointer arrays + pair->token row map,
  // uploaded ONCE (first touch, during the pre-warm forward) — nothing dangles
  // inside a captured graph.
  MoeBf16Resident& mr = MoeBf16ResidentFor(&w);
  if (!mr.ready) {
    std::vector<int64_t> gp(static_cast<size_t>(E)), up(static_cast<size_t>(E)),
        dp(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
      const size_t se = static_cast<size_t>(e);
      gp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_gate[se]).data);
      up[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_up[se]).data);
      dp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_down[se]).data);
    }
    const size_t eb = static_cast<size_t>(E) * sizeof(int64_t);
    auto up_i64 = [&](const std::vector<int64_t>& h) {
      void* p = d.b.Alloc(eb);
      d.b.Copy(d.q, p, h.data(), eb);
      return p;
    };
    mr.gate = up_i64(gp);
    mr.up = up_i64(up);
    mr.down = up_i64(dp);

    // HOST-MIRROR RELEASE (the decisive Qwen3-Coder host-memory lever; the bf16
    // analog of the 35B Marlin release above, same mechanism/policy). The routed
    // experts ARE essentially the whole model: 128 experts x 3 x [2048,768] bf16 x
    // 48 layers = ~57 GiB, i.e. ~94% of the checkpoint. On GB10 the host `.bytes`
    // and the device `d_dev` (ResidentWeight Alloc+Copy) are DISTINCT allocations
    // out of ONE 119 GiB unified pool, so retaining both made the model cost ~114
    // GiB and left nothing for KV cache/activations — MEASURED: the box ran at
    // free=1 GiB and c>=2 collapsed into thrash (c2 output throughput BELOW c1;
    // c4 never completed). Once the device copy exists it is authoritative and
    // nothing reads the host bytes again: every consumer of an expert weight goes
    // through ResidentWeight, which returns `d_dev` when populated.
    //
    // ORDERING: d.b.Copy is stream-async, so Synchronize FIRST — freeing the host
    // source under an in-flight H2D copy would corrupt the device weights.
    //
    // POLICY vs KERNEL-PATH split (BACKEND-PLATFORM item 2), identical to Marlin:
    // (a) WHETHER to free is the platform residency policy
    // (release_host_weights_after_upload); (b) WHETHER the host bytes can ever be
    // re-read is the KERNEL gate — here MoeBf16FastEnabled(), a process-static
    // const that is TRUE by construction (this function is reached only from the
    // MoeBlock branch gated on it), so the reference loop that would re-read them
    // can never run in this process. VT_MOE_HOST_FREE=0 retains the host copies
    // for a same-binary peak-memory A/B without changing the device compute.
    static const bool host_free_on = [] {
      const char* e = std::getenv("VT_MOE_HOST_FREE");
      return !(e != nullptr && e[0] == '0');
    }();
    if (vllm::platforms::ShouldReleaseHostWeights(
            vllm::platforms::GetPlatform(d.q.device.type).residency_policy(),
            /*committed_compute_path=*/MoeBf16FastEnabled(),
            /*host_free_env=*/host_free_on)) {
      d.b.Synchronize(d.q);  // all E x 3 H2D uploads complete before any free
      // W0f (#1299) FALSIFIED THIS BLOCK'S PREMISE, AND THIS IS THE REPAIR.
      //
      // The paragraph above says "once the device copy exists it is
      // authoritative and nothing reads the host bytes again", and it was true
      // while `ResidentWeight` had exactly two behaviours. It has three now: on
      // a platform whose kernels can dereference host storage the function
      // ALIASES, `d_dev` is never populated, and the pointers captured into
      // `gp/up/dp` above ARE `w.bytes.data()`. Releasing the host mirror then
      // frees the memory the resident device pointer table points at, and the
      // grouped GEMM keeps reading it for the model's lifetime — including from
      // inside a captured graph. A fresh review caught it with a scratch case
      // that replays this exact sequence and takes SIGSEGV.
      //
      // The condition is therefore not "did we upload" but "IS THERE A DEVICE
      // COPY TO BE AUTHORITATIVE", asked per weight, which is what `d_dev`
      // already answers. It is `nullptr` on precisely the arm that aliases, and
      // non-null on every arm that staged, so the discrete behaviour this
      // paragraph was written for is unchanged.
      for (int64_t e = 0; e < E; ++e) {
        const size_t se = static_cast<size_t>(e);
        if (vllm::HostMirrorIsRedundant(w.expert_gate[se]))
          w.expert_gate[se].ReleaseHost();
        if (vllm::HostMirrorIsRedundant(w.expert_up[se]))
          w.expert_up[se].ReleaseHost();
        if (vllm::HostMirrorIsRedundant(w.expert_down[se]))
          w.expert_down[se].ReleaseHost();
      }
    }
    mr.ready = true;
  }
  auto tok_it = mr.tok_map.find(T);
  if (tok_it == mr.tok_map.end()) {
    std::vector<int32_t> tok_map(static_cast<size_t>(P));
    for (int64_t p = 0; p < P; ++p)
      tok_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    const size_t tb = static_cast<size_t>(P) * sizeof(int32_t);
    void* p = d.b.Alloc(tb);
    d.b.Copy(d.q, p, tok_map.data(), tb);
    tok_it = mr.tok_map.emplace(T, p).first;
  }
  Tensor dgate_ptrs = MakeTensor(mr.gate, DType::kI64, d.q.device, {E});
  Tensor dup_ptrs = MakeTensor(mr.up, DType::kI64, d.q.device, {E});
  Tensor ddown_ptrs = MakeTensor(mr.down, DType::kI64, d.q.device, {E});
  Tensor dtok = MakeTensor(tok_it->second, DType::kI32, d.q.device, {P});

  // Fused grouped gate+up GEMM + SwiGLU over all pairs (Tier-A4 fold): ONE vt op
  // replaces the {gate GEMM (f32); up GEMM (f32); MoeSiluMul} triplet — decode
  // fuses to a partials launch + reduce+SwiGLU (drops the two f32 [P,I] round-
  // trips), prefill reuses the tuned grouped GEMM twice + the identical silu-mul.
  // BIT-IDENTICAL to the old sequence. Then the grouped down GEMM (act = per-pair
  // silu output, identity row-map). expert_out lands as [T,top_k,H] contiguous —
  // exactly what MoeCombine consumes. Native BF16 providers round gate/up and
  // SiLU before multiplication, then apply route weights before down narrowing.
  // Existing callers keep the legacy FP32 intermediate compatibility contract.
  const bool native_bf16 = vt::MoeGroupedBf16NativeAvailable(d.q.device.type);
  DBuf dact(d, DType::kBF16, {P, I});
  if (native_bf16)
    vt::MoeGroupedGemmBf16GateUpSiluNative(d.q, dact.t(), dh, eids, &dtok, dgate_ptrs, dup_ptrs);
  else
    vt::MoeGroupedGemmBf16GateUpSilu(d.q, dact.t(), dh, eids, &dtok, dgate_ptrs, dup_ptrs);
  DBuf ddown(d, DType::kBF16, {P, H});
  if (native_bf16) {
    const Tensor weights = Reshape(dtw.t(), {P});
    vt::MoeGroupedGemmBf16Weighted(d.q, ddown.t(), dact.t(), eids, nullptr, ddown_ptrs, weights);
  } else {
    vt::MoeGroupedGemmBf16(d.q, ddown.t(), dact.t(), eids, nullptr, ddown_ptrs);
  }
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  // Shared expert (SEAM GAP #3): Coder has none (shared_expert_intermediate_size
  // == 0) -> null shared term to MoeCombine (pure routed sum). A bf16 full-attn
  // MoE with a shared expert would run it here (bf16 path, fp4=false).
  const bool has_shared = cfg.shared_expert_intermediate_size > 0;
  std::optional<DBuf> shared;
  if (has_shared) shared.emplace(SharedExpert(d, w, cfg, dh, T, false));
  DBuf dout(d, DType::kBF16, {T, H});
  if (native_bf16)
    vt::MoeCombinePreweighted(d.q, dout.t(), expert_out, has_shared ? &shared->t() : nullptr);
  else
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), has_shared ? &shared->t() : nullptr);
  return dout;
}

// ─── VT_MOE_SEL_FP — the SELECTED-EXPERT-ID tap (MOEDIV, #2552) ──────────────
//
// `VT_Q4EXP_LAYER_FP` (#2547) taps VALUES, and a value tap cannot separate an
// expert-selection FLIP from re-association inside the expert GEMM. A discrete
// selection has BIMODAL error and not a tolerance: a token's top-k set either
// changes — and that token's MoE output moves by an O(1) amount — or it does not
// change at all and the residue is rounding. One `rel(sum|x|)` averages the two
// together and destroys exactly the bit that says which happened.
//
// So this prints the SELECTION. Per token: the selected ids SORTED, because the
// assertion between two arms is SET equality and the selection ORDER is not part
// of it (`vt::MoeCombine` sums over the k slots and is order-invariant given the
// weights). Per call: `sel`, an FNV-1a hash over every token's sorted list, so
// comparing two arms' selections at one layer is ONE string comparison.
//
// THE MARGIN IS IN LOGIT SPACE AND ITS UNIT IS bf16 ULPS. Probability space is
// the wrong space: the softmax denominator is a device-order f32 reduction and
// differs between the arms, while the bf16 logits are what the selection is
// actually a function of. And a probability difference reads as "small" for a
// gap of one representable step and for a gap of fifty. `ulps` is the number of
// representable bf16 values between the largest REJECTED logit and the smallest
// SELECTED one under the sign-magnitude total order; `ulps == 0` means the two
// are the SAME bf16 value and the boundary was decided by the lowest-index
// tie-break, which one ulp of re-association anywhere upstream will flip.
//
// `lines=` IS THE COUNTED PROPERTY. An instrument that never ran and two arms
// whose taps agreed look identical in a diff. The measuring job asserts
// `lines == calls * T` and refuses to report a comparison otherwise.
//
// IT IS ON THE REFERENCE ARM ONLY, which is the arm a stacked keep-quant
// checkpoint takes and the only arm any `qwen4_exp` checkpoint reaches. The
// three fused CUDA arms keep their ids on device; adding a readback to a
// capturable path, to instrument a model that cannot enter it, would be dead
// code by construction. See the spec's "Wave MOEDIV" section.
int64_t MoeSelFpCalls() {
  static const int64_t n = [] {
    const char* e = std::getenv("VT_MOE_SEL_FP");
    if (e == nullptr || e[0] == '\0') return static_cast<int64_t>(0);
    const long long parsed = std::atoll(e);
    return parsed > 0 ? static_cast<int64_t>(parsed) : static_cast<int64_t>(0);
  }();
  return n;
}
int64_t& MoeSelFpCall() {
  static int64_t call = 0;
  return call;
}
int64_t& MoeSelFpLines() {
  static int64_t lines = 0;
  return lines;
}

// bf16 under the sign-magnitude TOTAL ORDER, as a signed key. Monotone in the
// value it encodes, so the difference of two keys counts representable steps.
// -0 and +0 both map to 0, which is what makes an exact tie read `ulps=0`.
int32_t Bf16Ordered(uint16_t b) {
  return (b & 0x8000u) != 0 ? -static_cast<int32_t>(b & 0x7fffu)
                            : static_cast<int32_t>(b & 0x7fffu);
}

double SumAbsBf16(const std::vector<uint16_t>& v) {
  double s = 0.0;
  for (uint16_t b : v) {
    const float f = vt::BF16ToF32(b);
    if (std::isfinite(f)) s += std::fabs(static_cast<double>(f));
  }
  return s;
}

// One tapped MoE block invocation. Every buffer here is one the reference path
// had ALREADY downloaded, so the tap moves no extra bytes off the device except
// the one guarded shared-expert read its caller passes in.
void MoeSelFp(int dev_type, int64_t T, int64_t E, int64_t top_k,
              const std::vector<uint16_t>& h, const std::vector<uint16_t>& logits,
              const std::vector<int32_t>& ids,
              const std::vector<uint16_t>& expert_out,
              const std::vector<uint16_t>* shared) {
  const int64_t call = MoeSelFpCall();
  if (call >= MoeSelFpCalls()) return;
  uint64_t sel = 1469598103934665603ull;  // FNV-1a offset basis
  int64_t min_ulps = -1;
  int64_t min_tok = -1;
  std::vector<int32_t> sorted(static_cast<size_t>(top_k));
  std::vector<char> picked(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < top_k; ++j)
      sorted[static_cast<size_t>(j)] = ids[static_cast<size_t>(t * top_k + j)];
    std::sort(sorted.begin(), sorted.end());
    std::string idstr;
    for (int64_t j = 0; j < top_k; ++j) {
      if (j != 0) idstr += ",";
      idstr += std::to_string(sorted[static_cast<size_t>(j)]);
      // The hash is over the SORTED list, so it is a set identity and not a
      // selection-order identity.
      const uint32_t v = static_cast<uint32_t>(sorted[static_cast<size_t>(j)]);
      for (int b = 0; b < 4; ++b) {
        sel ^= static_cast<uint64_t>((v >> (8 * b)) & 0xffu);
        sel *= 1099511628211ull;
      }
    }
    // The boundary: the smallest SELECTED logit against the largest REJECTED
    // one, read from the bf16 the router GEMM actually stored.
    for (int64_t e = 0; e < E; ++e) picked[static_cast<size_t>(e)] = 0;
    for (int64_t j = 0; j < top_k; ++j)
      picked[static_cast<size_t>(sorted[static_cast<size_t>(j)])] = 1;
    int64_t lo_e = -1, hi_e = -1;
    for (int64_t e = 0; e < E; ++e) {
      const uint16_t b = logits[static_cast<size_t>(t * E + e)];
      if (picked[static_cast<size_t>(e)] != 0) {
        if (lo_e < 0 || Bf16Ordered(b) < Bf16Ordered(logits[static_cast<size_t>(t * E + lo_e)]))
          lo_e = e;
      } else {
        if (hi_e < 0 || Bf16Ordered(b) > Bf16Ordered(logits[static_cast<size_t>(t * E + hi_e)]))
          hi_e = e;
      }
    }
    const uint16_t lo_b = lo_e >= 0 ? logits[static_cast<size_t>(t * E + lo_e)] : 0;
    const uint16_t hi_b = hi_e >= 0 ? logits[static_cast<size_t>(t * E + hi_e)] : 0;
    // top_k == E leaves nothing rejected; the boundary is then undefined and is
    // reported as such rather than as a zero margin.
    const int64_t ulps = hi_e >= 0 && lo_e >= 0
                             ? static_cast<int64_t>(Bf16Ordered(lo_b)) -
                                   static_cast<int64_t>(Bf16Ordered(hi_b))
                             : -1;
    if (ulps >= 0 && (min_ulps < 0 || ulps < min_ulps)) {
      min_ulps = ulps;
      min_tok = t;
    }
    ++MoeSelFpLines();
    std::fprintf(stderr,
                 "moesel call=%lld dev=%d tok=%lld ids=%s lo_e=%lld lo=%.9g "
                 "lo_raw=0x%04x hi_e=%lld hi=%.9g hi_raw=0x%04x margin=%.9g "
                 "ulps=%lld\n",
                 static_cast<long long>(call), dev_type, static_cast<long long>(t),
                 idstr.c_str(), static_cast<long long>(lo_e),
                 static_cast<double>(vt::BF16ToF32(lo_b)), static_cast<unsigned>(lo_b),
                 static_cast<long long>(hi_e), static_cast<double>(vt::BF16ToF32(hi_b)),
                 static_cast<unsigned>(hi_b),
                 static_cast<double>(vt::BF16ToF32(lo_b)) -
                     static_cast<double>(vt::BF16ToF32(hi_b)),
                 static_cast<long long>(ulps));
  }
  // The digest. `x`/`logit`/`exp`/`shr` decompose the block's own output, so
  // with the selection sets equal these four say WHICH GEMM carries the residue.
  std::fprintf(stderr,
               "moesel call=%lld dev=%d T=%lld E=%lld k=%lld sel=%016llx "
               "minulps=%lld mintok=%lld x=%.9g logit=%.9g exp=%.9g shr=%.9g "
               "lines=%lld END\n",
               static_cast<long long>(call), dev_type, static_cast<long long>(T),
               static_cast<long long>(E), static_cast<long long>(top_k),
               static_cast<unsigned long long>(sel), static_cast<long long>(min_ulps),
               static_cast<long long>(min_tok), SumAbsBf16(h), SumAbsBf16(logits),
               SumAbsBf16(expert_out), shared != nullptr ? SumAbsBf16(*shared) : 0.0,
               static_cast<long long>(MoeSelFpLines()));
  ++MoeSelFpCall();
}

// VK4 keep-quant fast MoE path for Vulkan + TQ-quantized experts. Mirrors
// MoeBlockBf16Cuda but uses the TQ-quantized ops: Matmul for the router,
// MoeGateUpSwiGLUGrouped for the fused gate+up+SwiGLU, MatmulBTQuantGrouped
// for the down GEMM, and MoeCombine. ALL ops run on-device with NO host
// round-trip — the _dev shaders quantize Q8_K inside the kernel, so no
// FlushBatch is needed.
DBuf MoeBlockVulkanTQ(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                      const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;

  // Router: logits = dh @ gate (bf16 out). Handles both weight orientations
  // (nk=true for GGUF [N,E], nk=false for safetensors [H,E]) via the shared
  // MoeRouterLogits helper. For TQ-quantized router weights, MatmulBT redirects
  // to kMatmulBTQuant which dispatches the _dev shader (on-device Q8_K quantize).
  DBuf dlog(d, DType::kBF16, {T, E});
  MoeRouterLogits(d, dlog.t(), dh, w.router_gate);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  Tensor eids = Reshape(dtid.t(), {P});

  // Fused gate+up+SwiGLU: ONE dispatch replaces {gate GEMM; up GEMM; SiluAndMul}.
  // The shader reads bf16 activation from the device buffer, quantizes Q8_K
  // on-device, and reads TQ-quantized gate/up weights — no host round-trip.
  // limit=+inf reduces to plain silu(gate)*up, matching the reference path's
  // ExpertMlpKq (Silu(hg)*hu, no clamp).
  Tensor gate_w = ResidentWeight(d, w.expert_gate_kq);
  Tensor up_w = ResidentWeight(d, w.expert_up_kq);
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeGateUpSwiGLUGrouped(d.q, dact.t(), dh, gate_w, up_w, eids, 1e30f);

  // Down GEMM: act [P,I] bf16 @ down_w [E*N,K] TQ -> [P,H] bf16.
  // MatmulBTQuantGrouped dispatches the _dev shader (on-device Q8_K quantize).
  Tensor down_w = ResidentWeight(d, w.expert_down_kq);
  DBuf ddown(d, DType::kBF16, {P, H});
  vt::MatmulBTQuantGrouped(d.q, ddown.t(), dact.t(), down_w, eids);
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  // Shared expert: Qwen3-Coder has none (shared_expert_intermediate_size==0).
  const bool has_shared = cfg.shared_expert_intermediate_size > 0;
  std::optional<DBuf> shared;
  if (has_shared) shared.emplace(SharedExpert(d, w, cfg, dh, T, false));
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(),
                 has_shared ? &shared->t() : nullptr);
  return dout;
}

// ─── W8: the DEVICE-RESIDENT keep-quant grouped MoE arm (QUANT-CUDA-GATES) ───
//
// Owned by `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`;
// scoped by `.agents/specs/qwen4-exp-flash-next.md` §"W8 scope".
//
// WHAT THIS REPLACES, and it is a STRUCTURE rather than a kernel. The reference
// arm below is the only arm a GGUF k-quant checkpoint can take, because it has
// neither fp4 nor bf16 experts. That arm copies the hidden to host and
// synchronizes, downloads the router top-k, and then makes three `KqGrouped`
// calls that each upload an activation, launch ONE grouped GEMM and `Download`
// an f32 result — a blocking drain — with the SwiGLU in a host loop between
// them and the routed output assembled in a host `std::vector`. Nothing in that
// sequence needs the host: `vt::MatmulBTQuantGrouped`, `vt::MoeSiluMul`,
// `vt::CastBf16`, `vt::MoeRouterTopK` and `vt::MoeCombine` are all registered
// device ops. What was missing is an arm that keeps the intermediates on device
// BETWEEN them, and this is that arm. NO new kernel is added.
//
// THE BAR IS PER-PAIR BIT IDENTITY WITH THE REFERENCE ARM, EXCEPT AT THE
// SwiGLU, where it is one bf16 ULP. Not a tolerance anywhere else, and the
// whole shape of this function is chosen to make that reachable rather than
// hoped for. Each step below reaches the SAME op on the SAME bytes as the
// reference step it replaces, and the one that does not is named as such:
//
//   * router GEMM — `vt::Matmul`/`vt::MatmulBT` over the SAME `ResidentWeight`,
//     branched on `w.router_gate.nk` exactly as `MatmulBf16` branches. The
//     reference downloads the bf16 logits and re-uploads them into `dlog`; that
//     round trip moves no bits, so the tensor `MoeRouterTopK` reads is the same
//     one.
//   * top-k — the same `vt::MoeRouterTopK` on the same logits.
//   * gate/up — the same `vt::MatmulBTQuantGrouped` over the same
//     `ResidentWeight`-staged tower with the same `eids`. The reference
//     materialises `act[P,H]` by copying `h[t]` into every routed pair's row;
//     at `T == 1` every row is the same row, which is exactly the broadcast the
//     grouped kernel already implements (`Pa == 1 && P > 1`,
//     `cuda_quant_dot.cu`). The kernel SELECTION does not depend on `bcast` —
//     it is passed as a row-index flag — and the activation quantizer produces
//     the same Q8_K/Q8_0 block from the same bytes, so broadcasting is
//     bit-identical to P copies rather than merely close.
//   * SwiGLU — `vt::MoeSiluMul`, whose CUDA kernel computes
//     `g/(1+expf(-g)) * u` in f32 and stores through `__float2bfloat16`
//     (round-to-nearest-even, the same rounding as the host `vt::F32ToBF16`).
//     THIS IS THE ONE STEP THAT IS NOT BIT-IDENTICAL BY CONSTRUCTION: the
//     reference computes `Silu()` with `std::exp` and this computes it with
//     `expf`, and the two are not required to agree to the last f32 ULP. The
//     op's own comment (`cuda_moe.cu:823-827`) calls that an ACCEPTED
//     deviation, so this arm cannot claim identity for it.
//
//     IT IS GATED AT THE OP RATHER THAN THROUGH THIS BLOCK, because the block
//     comparison has almost no power over that class: a 1-f32-ULP move in
//     `silu(g)` changes `bf16(silu(g)*u)` in 1.44e-5 of N(0,3) samples
//     (288/20,000,000, measured), so a fixture's handful of SwiGLU elements
//     would miss a kernel that disagreed everywhere. The
//     `SwiGLU expf deviation, bounded` cases in
//     `tests/vllm/models/test_qwen35_moe_kq_device.cpp` sweep 2^22 pairs
//     against this arm's own host formula and hold the result to ONE bf16 ULP,
//     printing the observed disagreement count on the green run. The CPU case
//     of that sweep must be EXACTLY equal, because `cpu_ops.cpp:733` is the
//     same `std::exp` formula the reference arm runs.
//   * down GEMM — the same grouped op, `Pa == P` on both arms.
//   * the bf16 narrowing of the routed output — `vt::CastBf16` is
//     `__float2bfloat16`, which is `vt::F32ToBF16`.
//   * shared expert + combine — the same calls the reference makes, on the same
//     `dh` and the same `dtw`.
//
// T == 1 ONLY, and that is a scope statement rather than an oversight. The
// broadcast is what removes the host gather, and it exists only for one hidden
// row. Prefill (`T > 1`) needs a device row-gather that `MatmulBTQuantGrouped`
// has no row-map argument for, so it keeps the reference loop; decode is the
// capture target and decode is `T == 1`.
//
// This arm does NOT gate on the weight dtype. An encoding the CUDA grouped
// kernel cannot read falls back to the CPU provider INSIDE `vt` (with its own
// drain) on both arms identically, so refusing it here would change nothing
// except which of two identical answers is computed.
int64_t& MoeKqDeviceCallCount() {
  static int64_t n = 0;
  return n;
}

// The ship gate. DEFAULT ON, `VT_MOE_KQ_FAST=0` restores the reference loop in
// the same binary — the polarity `MoeBf16FastEnabled()` set.
//
// IT IS READ ON EVERY CALL AND NOT CACHED IN A PROCESS-STATIC, which is where
// this deliberately departs from `MoeBf16FastEnabled()`. The bar this arm has
// to clear is BYTE EQUALITY WITH THE OTHER ARM, and a process-static makes that
// unmeasurable: one process can then only ever run one arm, so the comparison
// would have to cross two processes through a golden file and the assertion
// would live in neither run. One `getenv` per MoE layer — 48 per decode token,
// beside three grouped GEMMs — is not a cost worth trading that for.
bool MoeKqFastEnabled() {
  const char* e = std::getenv("VT_MOE_KQ_FAST");
  return !(e != nullptr && e[0] == '0');  // default ON; =0 rolls back
}

DBuf MoeBlockKqDevice(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                    const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;
  ++MoeKqDeviceCallCount();

  // Router: logits = dh @ gate, then softmax/top-k/renormalize — both on device.
  // The `nk` branch mirrors `MatmulBf16`, which is what the reference arm calls.
  Tensor drg = ResidentWeight(d, w.router_gate);
  DBuf dlog(d, DType::kBF16, {T, E});
  if (w.router_gate.nk)
    vt::MatmulBT(d.q, dlog.t(), dh, drg);
  else
    vt::Matmul(d.q, dlog.t(), dh, drg);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  Tensor eids = Reshape(dtid.t(), {P});  // [P] i32, pair p = t*top_k + j

  // gate/up over the BROADCAST hidden. `dh` is already the [1,H] row the
  // grouped kernel wants for `Pa == 1`; nothing is gathered and nothing is
  // copied.
  Tensor wg = ResidentWeight(d, w.expert_gate_kq);
  Tensor wu = ResidentWeight(d, w.expert_up_kq);
  Tensor wd = ResidentWeight(d, w.expert_down_kq);
  DBuf dg(d, DType::kF32, {P, I});
  DBuf du(d, DType::kF32, {P, I});
  vt::MatmulBTQuantGrouped(d.q, dg.t(), dh, wg, eids);
  vt::MatmulBTQuantGrouped(d.q, du.t(), dh, wu, eids);

  // SwiGLU, on device: eact = bf16(silu(g) * u) — the host loop's exact shape.
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeSiluMul(d.q, dact.t(), dg.t(), du.t());

  // down, then the f32 -> bf16 narrowing the host loop did with F32ToBF16.
  DBuf ddn(d, DType::kF32, {P, H});
  vt::MatmulBTQuantGrouped(d.q, ddn.t(), dact.t(), wd, eids);
  DBuf deo(d, DType::kBF16, {P, H});
  vt::CastBf16(d.q, deo.t(), ddn.t());
  Tensor expert_out = Reshape(deo.t(), {T, top_k, H});

  // Shared expert + combine: the reference arm's own calls, unchanged.
  const bool has_shared = cfg.shared_expert_intermediate_size > 0;
  std::optional<DBuf> shared;
  if (has_shared) shared.emplace(SharedExpert(d, w, cfg, dh, T, /*fp4=*/false));
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(),
                 has_shared ? &shared->t() : nullptr);
  return dout;
}

DBuf MoeBlock(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
              const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  // fp4-resident NVFP4 experts/shared (M2.2b real-ckpt CUDA load) vs bf16
  // (synthetic / GGUF). Exactly one set is populated (see qwen3_5_weights.h).
  const bool fp4 = !w.expert_gate_fp4.empty();

  // M2.4/M2.5 fused MoE: CUDA + fp4-resident does the expert compute in ~3
  // grouped GEMM launches fully on-device (no host round-trip — capturable).
  // The bf16 / CPU / GGUF reference below keeps the per-expert token-gather
  // path on host (not the capture target); the fused output is per-pair
  // bit-identical to it (same NVFP4 decode).
  if (fp4 && vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4, d.q.device.type)) {
#ifdef VT_MARLIN_NVFP4
    if (MarlinMoeEnabled()) return MoeBlockFusedMarlinCuda(d, w, cfg, dh, T);
#endif
    return MoeBlockFusedCuda(d, w, cfg, dh, T);
  }

  // Fast BF16 grouped-MoE (Qwen3-Coder): CUDA + bf16 experts -> ~3 grouped bf16
  // GEMM launches fully on-device (W5). DEFAULT ON (VT_MOE_BF16_FAST); =0 falls
  // through to the reference loop below (same-binary A/B / correctness oracle).
  // LAYOUT-GUARDED (MoeBf16FastLayoutOk): only the [K,N] Matmul-B (`nk == false`)
  // orientation the grouped kernel can read; nk=true producers (35B MTP) fall
  // through to the reference loop.
  if (!fp4 && vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16, d.q.device.type) &&
      vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16GateUpSilu, d.q.device.type) && MoeBf16FastEnabled() &&
      !w.expert_gate.empty() && MoeBf16FastLayoutOk(w, cfg))
    return MoeBlockBf16Cuda(d, w, cfg, dh, T);

  // W8 (QUANT-CUDA-GATES): device-resident keep-quant grouped MoE — the GGUF
  // k-quant analog of the bf16 arm above, selected at the same site and under
  // the same shape of condition. DEFAULT ON (VT_MOE_KQ_FAST); =0 falls through
  // to the reference loop below, which stays intact as BOTH the fallback and
  // the correctness oracle the arm is gated byte-for-byte against.
  //
  // Three of the five conditions are the arm's preconditions: a keep-quant tower
  // set, every op it calls registered on THIS queue's device, and T == 1 for the
  // broadcast. It asks the op table rather than naming CUDA — the same question
  // the bf16 arm asks, and the one `check-device-leakage.py` requires of the
  // device-agnostic layer. Every op it calls has a CPU provider too, and the CPU
  // grouped kernel implements the same `act.shape[0] == 1` broadcast
  // (`cpu_quant_gemm.cpp:244`), so the arm is correct wherever it is registered
  // and the suite exercises it on CPU, CUDA and ROCm.
  //
  // THAT MAKES IT DEFAULT-ON ON TENSTORRENT TOO, which registers all three ops
  // (`tenstorrent_ops.cpp:8298, 8307, 8309`) and which no suite here runs a MoE
  // block on. The bf16 arm above does NOT reach that backend — Tenstorrent
  // registers no `kMoeGroupedGemmBf16` — so this gap is this arm's alone and is
  // tracked by
  // `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2D7X94RTPJ453QGHYKQZPY8.md`. The other two conditions are
  // refusals of an INVISIBLE FALLBACK rather than of a defect.
  // `Qwen35GroupedMoeEnabled()` is the existing grouped-vs-per-expert route
  // switch, and it is already false when expert streaming was asked for; taking
  // this arm anyway would silently do neither. `MoeSelFpCalls()` is the
  // VT_MOE_SEL_FP selection tap, which reads the host `h`/`logits`/`ids`/
  // `expert_out` the reference arm downloads and this arm never materialises —
  // so an operator who turned the tap on gets the reference arm and a real tap,
  // not this arm and a silent nothing.
  if (!fp4 && !w.expert_gate_kq.Empty() && T == 1 &&
      vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, d.q.device.type) &&
      vt::OpRegistered(vt::OpId::kMoeSiluMul, d.q.device.type) &&
      vt::OpRegistered(vt::OpId::kCastBf16, d.q.device.type) &&
      Qwen35GroupedMoeEnabled() && MoeSelFpCalls() == 0 && MoeKqFastEnabled())
    return MoeBlockKqDevice(d, w, cfg, dh, T);
  // VK4 keep-quant fast MoE path: Vulkan + TQ-quantized experts -> fully
  // on-device (no host round-trip). The router GEMM, MoeRouterTopK, fused
  // gate+up+SwiGLU, down GEMM, and MoeCombine all run as native Vulkan ops
  // with NO FlushBatch — the _dev shaders quantize Q8_K on-device and the
  // fused MoE kernel reads bf16 activations straight from the device buffer.
  // Eliminates the ~30 FlushBatch calls/layer of the reference path.
  if (!fp4 && d.q.device.type == vt::DeviceType::kVULKAN &&
      !w.expert_gate_kq.Empty() &&
      vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped, d.q.device.type) &&
      vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, d.q.device.type))
    return MoeBlockVulkanTQ(d, w, cfg, dh, T);

  // Reference path: download the hidden once, then gather + per-expert MLP.
  std::vector<uint16_t> h(static_cast<size_t>(T) * H);
  d.b.Copy(d.q, h.data(), dh.data, h.size() * sizeof(uint16_t));
  d.b.Synchronize(d.q);

  // Router: logits = x @ gate.T (bf16, §2), softmax/top-k/renormalize (§3).
  std::vector<uint16_t> logits = MatmulBf16(d, h, T, H, w.router_gate);  // [T,E]
  DBuf dlog(d, DType::kBF16, {T, E}, logits.data());
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  std::vector<float> weights(static_cast<size_t>(T) * top_k);
  std::vector<int32_t> ids(static_cast<size_t>(T) * top_k);
  dtw.Download(d, weights.data());
  dtid.Download(d, ids.data());

  // Activated-expert gather: per expert, the (token, slot) pairs routed to it.
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
      static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < top_k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t) * top_k + j])]
          .push_back({t, j});

  std::vector<uint16_t> expert_out(static_cast<size_t>(T) * top_k * H, 0);
  if (!w.expert_gate_kq.Empty() && Qwen35GroupedMoeEnabled()) {
    // W3b: keep-quant grouped MoE — the per-expert {gate,up,down} matvecs collapse to
    // 3 grouped vt::MatmulBTQuantGrouped launches over the stacked expert_*_kq towers.
    // Pair p = t*top_k+j: expert = ids[p], act = h[t]; expert_out is built directly in
    // [T,top_k,H] order, so it is BYTE-IDENTICAL to the per-expert ExpertMlpKq scatter
    // below (same eids slice + kMatmulBTQuant core + F32ToBF16 SwiGLU + down cast).
    const int64_t P = T * top_k;
    std::vector<int32_t> eids(static_cast<size_t>(P));
    std::vector<uint16_t> act(static_cast<size_t>(P) * H);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < top_k; ++j) {
        const int64_t p = t * top_k + j;
        eids[static_cast<size_t>(p)] = ids[static_cast<size_t>(t * top_k + j)];
        std::memcpy(act.data() + static_cast<size_t>(p) * H,
                    h.data() + static_cast<size_t>(t) * H,
                    static_cast<size_t>(H) * sizeof(uint16_t));
      }
    const std::vector<float> g = KqGrouped(d, act, P, I, H, w.expert_gate_kq, eids);
    const std::vector<float> u = KqGrouped(d, act, P, I, H, w.expert_up_kq, eids);
    std::vector<uint16_t> eact(static_cast<size_t>(P) * I);
    for (size_t i = 0; i < eact.size(); ++i)
      eact[i] = vt::F32ToBF16(Silu(g[i]) * u[i]);
    const std::vector<float> dn = KqGrouped(d, eact, P, H, I, w.expert_down_kq, eids);
    for (size_t i = 0; i < expert_out.size(); ++i)
      expert_out[i] = vt::F32ToBF16(dn[i]);
  } else
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    std::vector<uint16_t> xg(static_cast<size_t>(n) * H);
    for (int64_t r = 0; r < n; ++r)
      std::memcpy(xg.data() + static_cast<size_t>(r) * H,
                  h.data() + static_cast<size_t>(list[r].first) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    std::vector<uint16_t> y =
        fp4 ? ExpertMlpNvfp4(d, w.expert_gate_fp4[static_cast<size_t>(e)],
                             w.expert_up_fp4[static_cast<size_t>(e)],
                             w.expert_down_fp4[static_cast<size_t>(e)], xg, n, H, I)
        : !w.expert_gate_kq.Empty()
            // A3 keep-quant: gate/up/down are slices of the stacked tower (byte-exact
            // with the pre-A3 per-expert ExpertMlp; W3b grouped path above).
            ? ExpertMlpKq(d, w.expert_gate_kq, w.expert_up_kq, w.expert_down_kq, xg, e,
                          n, H, I)
            : ExpertMlp(d, w.expert_gate[static_cast<size_t>(e)],
                        w.expert_up[static_cast<size_t>(e)],
                        w.expert_down[static_cast<size_t>(e)], xg, n, H, I);
    for (int64_t r = 0; r < n; ++r) {
      const int64_t t = list[r].first, j = list[r].second;
      std::memcpy(expert_out.data() + static_cast<size_t>(t * top_k + j) * H,
                  y.data() + static_cast<size_t>(r) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
  }

  // Shared expert (moe-semantics.md §5): device-resident (takes dh). GUARDED on
  // shared_expert_intermediate_size>0 — the 35B has a gated shared expert (runs
  // exactly as before, byte-identical), but a full-attention MoE with NO shared
  // expert (Qwen3-Coder `Qwen3MoeForCausalLM`, shared_expert_intermediate_size==0,
  // no shared weights) SKIPS it and passes a nullptr shared term to MoeCombine
  // (which treats a null shared as the pure routed sum). Mirrors vLLM
  // Qwen3MoeSparseMoeBlock: `shared_expert = None` when the size is 0
  // (qwen3_moe.py:180-202). SEAM GAP #3, sweep-qwen3-coder-30b.md §3b.
  const bool has_shared = cfg.shared_expert_intermediate_size > 0;
  std::optional<DBuf> shared;
  if (has_shared) shared.emplace(SharedExpert(d, w, cfg, dh, T, fp4));

  // MOEDIV (#2552): the selected-expert-id tap. Inert unless VT_MOE_SEL_FP is
  // set; the outer guard is what keeps the shared-expert readback off the
  // default path. Everything else it reads was already downloaded above.
  if (MoeSelFpCalls() > 0) {
    std::vector<uint16_t> shr;
    if (has_shared) {
      shr.resize(static_cast<size_t>(T) * H);
      shared->Download(d, shr.data());
    }
    MoeSelFp(static_cast<int>(dh.device.type), T, E, top_k, h, logits, ids,
             expert_out, has_shared ? &shr : nullptr);
  }

  // Combine (moe-semantics.md §6): out = shared + sum_j w_j * expert_out_j.
  DBuf deo(d, DType::kBF16, {T, top_k, H}, expert_out.data());
  DBuf dwt(d, DType::kF32, {T, top_k}, weights.data());
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), deo.t(), dwt.t(),
                 has_shared ? &shared->t() : nullptr);
  return dout;
}

// Returns true (+ fills *scale) when the layer's input-fed fp8 projections share
// ONE static input_scale, so a single RmsNormQuantFp8 can quantize the shared
// activation once and feed them all (attn q/k/v; GDN in_proj_qkv/z). The fp8
// analog of the fp4 fuse_qkv input_global_scale guard; exact float equality (only
// fuse when the checkpoint scales are truly identical).
bool Fp8SharedInputScale(bool is_linear_attention, const GdnLayerWeights& g,
                         const FullAttnLayerWeights& a, float* scale) {
  // ONE definition of the GDN pair's scale-compatibility rule, shared with the
  // PERF-27B-GDN-FP8-QKVZ merge guard (detail::GdnFp8SharedInputScale) so the
  // two can never drift.
  if (is_linear_attention) return detail::GdnFp8SharedInputScale(g, scale);
  if (a.q_proj_fp8.Empty() || a.k_proj_fp8.Empty() || a.v_proj_fp8.Empty()) return false;
  if (a.q_proj_fp8.input_scale != a.k_proj_fp8.input_scale ||
      a.q_proj_fp8.input_scale != a.v_proj_fp8.input_scale)
    return false;
  *scale = a.q_proj_fp8.input_scale;
  return true;
}

// Input-layernorm producer shared by RunLayer / RunLayerPaged: residual-add +
// gemma RMSNorm -> bf16 `dhn`. With VT_FUSE_RMSNORM_FP8QUANT (35B, shared fp8
// input_scale) it ALSO emits the shared static-quant fp8 activation in the SAME
// pass (vt::RmsNormQuantFp8, mirror vLLM Inductor fused_add_rms_norm_static_fp8_
// quant), returned so the block feeds it to q/k/v (or in_proj_qkv/z) once. GDN's
// in_proj_a/b still read bf16 `dhn`, so it is emitted there; full-attn reads only
// the fp8, so bf16 `dhn` is skipped. std::nullopt (no fp8) = the byte-identical
// plain RmsNorm path.
std::optional<DBuf> InputLayernormFp8(Dev d, const Qwen3_5MoeLayerWeights& layer,
                                      const HfConfig& cfg, DBuf& hidden, DBuf& res, DBuf& dhn,
                                      int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  float fp8_scale = 0.0F;
  const bool fuse = FuseRmsNormFp8QuantEnabled() && vllm::platforms::GetPlatform(d.q.device.type).supports_fp8() &&
                    Fp8SharedInputScale(layer.is_linear_attention, layer.gdn, layer.attn,
                                        &fp8_scale);
  if (fuse) {
    std::optional<DBuf> dhn_fp8;
    dhn_fp8.emplace(d, DType::kI8, std::vector<int64_t>{T, H});
    Tensor* out_bf16 = layer.is_linear_attention ? &dhn.t() : nullptr;
    // KERNEL-FUSION-FRAMEWORK W2 — route residual-add + gemma-RMSNorm + static fp8
    // quant through vt::FusedChain(kRmsNormQuantFp8); its fast_op binds the SAME
    // bespoke RmsNormQuantFp8 kernel (byte-identical + perf-neutral; the optional
    // bf16 normed output is preserved). VT_FUSED_CHAIN_ADOPT=0 restores the hand-call.
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, vt::kRmsNormQuantFp8, dhn_fp8->t(), out_bf16, hidden.t(), dw_in,
                     &res.t(), eps, fp8_scale);
    } else {
      vt::RmsNormQuantFp8(d.q, dhn_fp8->t(), out_bf16, hidden.t(), dw_in,
                          vt::RmsNormArgs{eps, true}, &res.t(), fp8_scale);
    }
    return dhn_fp8;
  }
  // Qwen3NextRMSNorm == GemmaRMSNorm (weight applied as 1+w). res += hidden.
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());
  return std::nullopt;
}

// One decoder layer over the fused residual stream. `hidden` (bf16 [T*H]) is
// the previous block's output (the delta); `res` (f32 [T,H], device) is the
// accumulator. Mirrors qwen3_next.py::Qwen3NextDecoderLayer.forward:
//   h  = input_layernorm(hidden, res)          # res += hidden; h = norm(res)
//   a  = attn/gdn(h)
//   h2 = post_attention_layernorm(a, res)      # res += a; h2 = norm(res)
//   hidden = mlp(h2)                            # MoE block; returned as delta
void RunLayer(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const std::vector<int32_t>& positions,
              int64_t T, int64_t layer_index) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  DBuf dhn(d, ActDType(d), {T, H});
  std::optional<DBuf> dhn_fp8 = InputLayernormFp8(d, layer, cfg, hidden, res, dhn, T);
  const Tensor* h_fp8 = dhn_fp8 ? &dhn_fp8->t() : nullptr;

  DBuf attn = layer.is_linear_attention
                  ? GdnBlock(d, layer.gdn, cfg, dhn.t(), T, h_fp8)
                  : FullAttnBlock(d, layer.attn, cfg, dhn.t(), positions, T, h_fp8);

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  // ENG-HYBRID-PLACEMENT W3d: through the shared seam. Inert by construction when
  // this layer is not placed — it resolves to the same `MoeBlock` call.
  hidden = RunMoePlaced(d, layer_index, dh2.t(), T, H,
                        [&](Dev p, const Tensor& h) {
                          return MoeBlock(p, layer.moe, cfg, h, T);
                        },
                        // fp4-resident experts are built on the device at LOAD,
                        // so placing them would upload every expert and then
                        // compute across the bus. Refuse instead.
                        /*placeable=*/layer.moe.expert_gate_fp4.empty(),
                        "the routed experts are fp4-resident and their device "
                        "residents are built at load");
}

// --- Dense SwiGLU MLP block (the 27B's replacement for the MoE block; notes
// §2). down( silu(gate(x)) * up(x) ), intermediate = cfg.intermediate_size.
// Mirrors the shared-expert silu-mul MLP (no router, no expert gather, no output
// gate). h [T,H] bf16 (device) -> DBuf [T,H] bf16 (device). Reused by the dense
// forward below; the gate/up/down weights are W4A4-materialized-to-bf16 at load.
DBuf DenseMlpBlock(Dev d, const DenseMlpWeights& w, const HfConfig& cfg,
                   const Tensor& dh, int64_t T) {
  const int64_t I = cfg.intermediate_size;
  // MODEL-QWEN35-EXL3 (#2495 item 3). FIRST and exclusive. Routed through the
  // SHARED `layers::MlpGateUpMethodBase` seam AGENTS.md names, exactly as
  // `qwen3.cpp:136-145` does for the Llama/Qwen3 dense MLP -- the model calls
  // one method and never asks which scheme it bound.
  if (w.IsExl3()) {
    DBuf act = dense_exl3::GateUp(d, dh, w.gate_up_proj, w.gate_proj_exl3,
                                  w.up_proj_exl3, I);
    (void)T;
    return dense_exl3::Linear(d, act.t(), w.down_proj, w.down_proj_exl3,
                              DType::kBF16);
  }
  // fp4-resident W4A4 path (real 27B, notes §5 step-6a) when populated; else the
  // bf16 path (synthetic CPU tests). Exactly one representation is filled.
  const bool fp4 = !w.gate_proj_fp4.Empty();
#ifdef VT_CUTLASS_NVFP4
  // vLLM's production topology is one MergedColumnParallelLinear gate_up_proj,
  // not two independently-scaled linears. Its CT loader takes max(input
  // divisor) and max(weight divisor) across the logical shards, computes one
  // alpha, quantizes once and launches one [T,H]x[2I,H] GEMM. This branch is W2
  // default; VT_FP4_MERGED_GATE_UP=0 restores the split W2 diagnostic and
  // VT_FP4_FULL_TACTICS=0 restores W1 including its split model topology.
  if (fp4 && MergedGateUpEligible(w, d)) {
    DBuf gate_up = MergedGateUpCutlassD(d, dh, w);  // bf16 [T,2I]
    if (FuseSiluQuantEnabled() &&
        w.down_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
      DBuf ap(d, DType::kI8, {T, I / 2});
      const bool direct_scale = DirectFp4ScaleEligible(d);
      DBuf as(d, DType::kI8,
              direct_scale ? CutlassFp4ScaleShape(T, I)
                           : std::vector<int64_t>{T, I / 16});
      const vt::Fp4ScaleLayout scale_layout =
          direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                       : vt::Fp4ScaleLayout::kLinear;
      if (MergedSiluQuantEnabled()) {
        vt::SiluAndMulFp4Quant(d.q, ap.t(), as.t(), gate_up.t(),
                               w.down_proj_fp4.input_global_scale_inv,
                               scale_layout);
      } else {
        DBuf act(d, ActDType(d), {T, I});
        vt::SiluAndMul(d.q, act.t(), gate_up.t());
        vt::ScaledFp4Quant(d.q, ap.t(), as.t(), act.t(),
                           w.down_proj_fp4.input_global_scale_inv,
                           scale_layout);
      }
      return MatmulNvfp4Fp4DirectD(d, ap.t(), as.t(), w.down_proj_fp4,
                                   DType::kBF16,
                                   direct_scale ? &as.t() : nullptr);
    }
    DBuf act(d, ActDType(d), {T, I});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());
    return MatmulNvfp4Bf16D(d, act.t(), w.down_proj_fp4);
  }
#endif
  // PERF-27B-DENSE-MARLIN-GATEUP (issue #365) — the W4A16 sibling of the CUTLASS
  // W4A4 merged branch above. The 27B gate checkpoint (modelopt_mixed) is W4A16,
  // so `MergedGateUpEligible` is false for it and the split gate+up Marlin GEMMs
  // below are the FALLBACK; vLLM runs ONE merged gate_up_proj.
  // VT_DENSE_MARLIN_GATEUP (default ON, opt out with =0 — the same-binary A/B
  // measured +2.12% at c1 and +1.70% at c8 on the 27B, complete separation at
  // both, tokens identical) substitutes the ALREADY-EXISTING fused Marlin pair —
  // one GEMM into [T,2I] plus the same silu/mul sink — feeding the identical
  // down projection. `nullopt` = not this configuration; nothing below changes.
  if (std::optional<DBuf> gu_act = DenseGateUpFusedMarlinD(d, dh, w))
    return MatmulNvfp4Bf16D(d, gu_act->t(), w.down_proj_fp4);
  if (!fp4 && !w.gate_up_proj.Empty()) {
    // Qwen3.5's MergedColumnParallelLinear: one raw-NK [2I,H] projection,
    // followed by SiluAndMul and the raw-NK down projection.
    DBuf gate_up = MatmulBf16D(d, dh, w.gate_up_proj);
    DBuf act(d, ActDType(d), {T, I});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());
    return MatmulBf16D(d, act.t(), w.down_proj);
  }
  // QUANTIZE-ONCE for gate/up: shared activation dh + shared input_global_scale ->
  // one ScaledFp4Quant feeding both fp4 GEMMs (removes 1 redundant [T,H] quant).
  const bool fuse_gu =
      fp4 && FuseQuantOnceEnabled() && vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() &&
      w.gate_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled() &&
      w.gate_proj_fp4.input_global_scale_inv == w.up_proj_fp4.input_global_scale_inv;
  std::optional<DBuf> gu_ap, gu_as;
#ifdef VT_CUTLASS_NVFP4
  const bool gu_direct_scale = fuse_gu && DirectFp4ScaleEligible(d);
#else
  const bool gu_direct_scale = false;
#endif
  if (fuse_gu) {
    const int64_t H = dh.shape[1];
    gu_ap.emplace(d, DType::kI8, std::vector<int64_t>{T, H / 2});
    gu_as.emplace(d, DType::kI8,
                  gu_direct_scale
                      ? CutlassFp4ScaleShape(T, H)
                      : std::vector<int64_t>{T, H / 16});
    vt::ScaledFp4Quant(
        d.q, gu_ap->t(), gu_as->t(), dh,
        w.gate_proj_fp4.input_global_scale_inv,
        gu_direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                        : vt::Fp4ScaleLayout::kLinear);
  }
  // SWIZZLE-ONCE (VT_SWIZZLE_IN_QUANT): swizzle the shared gate/up activation SF
  // ONCE and feed the already-swizzled SF to both GEMMs (skipping each one's
  // internal SwizzleBlockscale). nullptr (OFF / non-cutlass) = per-projection
  // swizzle, byte-identical to the current path.
  const Tensor* gu_sf_sw_p = nullptr;
#ifdef VT_CUTLASS_NVFP4
  std::optional<DBuf> gu_sf_sw;
  if (gu_direct_scale) {
    gu_sf_sw_p = &gu_as->t();
  } else if (fuse_gu && SwizzleInQuantEnabled() && NvfpCutlassEnabled()) {
    gu_sf_sw.emplace(SwizzleActScaleOnce(d, gu_as->t()));
    gu_sf_sw_p = &gu_sf_sw->t();
  }
#endif
  // gate/up output bf16 (VT_BF16_GEMM_OUT, rank-1 lever) — matches vLLM bf16 dtype,
  // halves the GEMM write + MoeSiluMul read; else f32 (current). MoeSiluMul is
  // templated on the input dtype so both work.
  const DType gu_out = Bf16GemmOutEnabled() ? DType::kBF16 : DType::kF32;
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4). The dense SwiGLU MLP's three
  // projections, exclusive and first. bf16 out on all three -- upstream's
  // `out_dtype` is the model dtype (fp8.py:284) and `vt::MoeSiluMul` is
  // templated on its input dtype, so gate/up feed it unchanged; down_proj's
  // bf16 is what every other arm of this return already produces.
  if (!w.gate_proj_fp8_block.Empty()) {
    // MODEL-FP8-BLOCK-MERGED (#1189 M6). ONE gate_up GEMM, which is vLLM's own
    // topology (`MergedColumnParallelLinear`, named for this model at
    // `models/qwen3_5.py:288-298`, loaded at `layers/linear.py:660`). Unlike the
    // per-tensor fp8 merge two screens up, this needs no alpha vector and no
    // shared-scale guard: block scales concatenate losslessly along N. The
    // merged GEMM is byte-identical to the two split ones, so the ONLY
    // instrument that can see the merge is the dispatch counter, which
    // `tests/vllm/model_executor/models/test_fp8_block_merged.cpp` reads.
    const dense_fp8_block::Fp8BlockShard gate_up_shards[2] = {
        {&w.gate_proj_fp8_block, "gate_proj"},
        {&w.up_proj_fp8_block, "up_proj"}};
    const dense_fp8_block::Fp8BlockMergedView gate_up =
        dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", gate_up_shards, 2,
            w.gate_up_fp8_block_merged);
    VT_CHECK(gate_up.n_total == 2 * I,
             "qwen3_5 dense MLP: the merged block-wise FP8 gate_up operand's N "
             "does not match twice the intermediate size");
    DBuf bact = dense_fp8_block::Fp8BlockGateUpSwiGLUD<DBuf>(d, dh, gate_up,
                                                            DType::kBF16);
    return dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(
        d, bact.t(), w.down_proj_fp8_block, DType::kBF16);
  }
  DBuf gate = fuse_gu ? MatmulNvfp4Fp4DirectD(d, gu_ap->t(), gu_as->t(), w.gate_proj_fp4, gu_out,
                                              gu_sf_sw_p)
              : fp4 ? (Bf16GemmOutEnabled() ? MatmulNvfp4Bf16D(d, dh, w.gate_proj_fp4)
                                            : MatmulNvfp4F32D(d, dh, w.gate_proj_fp4))
                    : MatmulF32D(d, dh, w.gate_proj);  // [T,I]
  DBuf up = fuse_gu ? MatmulNvfp4Fp4DirectD(d, gu_ap->t(), gu_as->t(), w.up_proj_fp4, gu_out,
                                            gu_sf_sw_p)
            : fp4 ? (Bf16GemmOutEnabled() ? MatmulNvfp4Bf16D(d, dh, w.up_proj_fp4)
                                          : MatmulNvfp4F32D(d, dh, w.up_proj_fp4))
                  : MatmulF32D(d, dh, w.up_proj);      // [T,I]
  // FUSED silu-mul + fp4-quant → down GEMM (no bf16 intermediate). Only when the
  // down_proj would have quantized its activation anyway (true-W4A4, CUDA) — same
  // guard as MatmulNvfp4Bf16D's MatmulNvfp4Fp4D route. Bit-identical to the else.
  if (fp4 && FuseSiluQuantEnabled() && vllm::platforms::GetPlatform(d.q.device.type).cutlass_fp4_supported() &&
      w.down_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
    DBuf ap(d, DType::kI8, {T, I / 2});
#ifdef VT_CUTLASS_NVFP4
    const bool direct_scale = DirectFp4ScaleEligible(d);
#else
    const bool direct_scale = false;
#endif
    DBuf as(d, DType::kI8,
            direct_scale ? CutlassFp4ScaleShape(T, I)
                         : std::vector<int64_t>{T, I / 16});
    const vt::Fp4ScaleLayout lay =
        direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled : vt::Fp4ScaleLayout::kLinear;
    // KERNEL-FUSION-FRAMEWORK W2 — route silu·up + NVFP4 quant through the declared
    // vt::FusedChain(kSiluMulFp4Quant) recipe. The recipe's fast_op binds the SAME
    // bespoke SiluMulFp4Quant kernel, so dispatch is byte-identical AND perf-neutral
    // by construction (no extra kernel). VT_FUSED_CHAIN_ADOPT=0 restores the direct
    // hand-call (same-binary rollback).
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, vt::kSiluMulFp4Quant, ap.t(), as.t(), gate.t(), up.t(),
                     w.down_proj_fp4.input_global_scale_inv, lay);
    } else {
      vt::SiluMulFp4Quant(d.q, ap.t(), as.t(), gate.t(), up.t(),
                          w.down_proj_fp4.input_global_scale_inv, lay);
    }
    return MatmulNvfp4Fp4DirectD(
        d, ap.t(), as.t(), w.down_proj_fp4, DType::kBF16,
        direct_scale ? &as.t() : nullptr);
  }
  DBuf act(d, ActDType(d), {T, I});
  vt::MoeSiluMul(d.q, act.t(), gate.t(), up.t());  // silu(gate)*up -> bf16
  return fp4 ? MatmulNvfp4Bf16D(d, act.t(), w.down_proj_fp4)
             : MatmulBf16D(d, act.t(), w.down_proj);  // [T,H] bf16
}

// One dense decoder layer (notes §2). Same residual/norm thread as RunLayer, but
// the MoE block is swapped for the dense SwiGLU MLP; the GDN / full-attention
// blocks are the 35B helpers reused verbatim. `hidden` (bf16 [T,H]) is the delta;
// `res` (f32 [T,H]) the accumulator.
void RunDenseLayer(Dev d, const Qwen3_5DenseLayerWeights& layer,
                   const HfConfig& cfg, DBuf& hidden, DBuf& res,
                   const std::vector<int32_t>& positions, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf attn = layer.is_linear_attention
                  ? GdnBlock(d, layer.gdn, cfg, dhn.t(), T)
                  : FullAttnBlock(d, layer.attn, cfg, dhn.t(), positions, T);

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  hidden = DenseMlpBlock(d, layer.mlp, cfg, dh2.t(), T);
}

// Batched PAGED decoder layer (M1.8 Task 3). Same residual/norm/MoE thread as
// RunLayer, but the attention block reads/writes the paged KV cache
// (full-attn: attn_kv) or the persistent GDN mamba state (GDN: gdn_state).
// Exactly one of {attn_kv, gdn_state} is non-null (per layer type).
void RunLayerPaged(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
                   DBuf& hidden, DBuf& res, const StepDevInputs& sdi,
                   const CommonAttentionMetadata& attn_meta,
                   const GDNAttentionMetadata& gdn_meta,
                   const PagedKvCache* attn_kv, const GdnStateCache* gdn_state,
                   int64_t T, int64_t layer_index) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // ROCM-TIER-DIVERGENCE (#2590): carry (step, layer) to every dump nested in
  // this layer's mixer, so the GDN stage probes stop keying on a counter of
  // their own. Inert when no dump knob is set.
  const actdump::LayerScope dump_here(actdump::Current().step, layer_index);

  DBuf dhn(d, ActDType(d), {T, H});
  std::optional<DBuf> dhn_fp8 = InputLayernormFp8(d, layer, cfg, hidden, res, dhn, T);
  const Tensor* h_fp8 = dhn_fp8 ? &dhn_fp8->t() : nullptr;

  DBuf attn = [&] {
    if (layer.is_linear_attention) {
      VT_CHECK(gdn_state != nullptr, "paged layer: GDN layer needs a GdnStateCache");
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), sdi, gdn_meta, *gdn_state, T, h_fp8);
    }
    VT_CHECK(attn_kv != nullptr, "paged layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), sdi, attn_meta,
                              *attn_kv, T, h_fp8);
  }();

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, ActDType(d), {T, H});
  // KERNEL-FUSION-FRAMEWORK W0 — the first production adoption of the declared
  // fusion seam. post_attention_layernorm is a plain add+residual+gemma-RMSNorm
  // (res += attn; dh2 = norm(res)), the exact chain kFusedAddRmsNorm transcribes.
  // The FusedChain path is bit-identical to the vt::RmsNorm(residual) hand-call
  // (Tier-0 composite dispatches to the same primitive; byte-exact in
  // test_ops_fused_chain.cpp). VT_FUSED_CHAIN_ADOPT=0 restores the hand-call.
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), dw_post, &res.t(), vt::kFusedAddRmsNorm, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());
  }

  // ENG-HYBRID-PLACEMENT W3d: through the shared seam. Inert by construction when
  // this layer is not placed — it resolves to the same `MoeBlock` call.
  hidden = RunMoePlaced(d, layer_index, dh2.t(), T, H,
                        [&](Dev p, const Tensor& h) {
                          return MoeBlock(p, layer.moe, cfg, h, T);
                        },
                        // fp4-resident experts are built on the device at LOAD,
                        // so placing them would upload every expert and then
                        // compute across the bus. Refuse instead.
                        /*placeable=*/layer.moe.expert_gate_fp4.empty(),
                        "the routed experts are fp4-resident and their device "
                        "residents are built at load");
}

// Batched PAGED dense decoder layer (27B; notes §5). Identical residual/norm
// thread + paged attention wiring as RunLayerPaged, but the MoE block is swapped
// for the dense SwiGLU MLP (DenseMlpBlock) and the layer carries dense weights.
// The GDN / full-attn paged blocks are the 35B helpers reused VERBATIM. Exactly
// one of {attn_kv, gdn_state} is non-null (per layer type).
void RunDenseLayerPaged(Dev d, const Qwen3_5DenseLayerWeights& layer,
                        const HfConfig& cfg, DBuf& hidden, DBuf& res,
                        const StepDevInputs& sdi,
                        const CommonAttentionMetadata& attn_meta,
                        const GDNAttentionMetadata& gdn_meta,
                        const PagedKvCache* attn_kv,
                        const GdnStateCache* gdn_state, int64_t T,
                        int64_t layer_index) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // ROCM-TIER-DIVERGENCE (#2590): the sub-stage dump is keyed on (step, layer),
  // taken from the caller the way the MoE sibling `RunLayerPaged` already takes
  // `layer_index`. It used to be keyed on a `static thread_local` counter that
  // was never reset, so a file name encoded an invocation ordinal rather than a
  // position in the model, and one extra forward call on either side of a
  // comparison silently shifted the whole join. The scope also carries the key
  // to every dump NESTED inside the mixer, so the GDN stage probes no longer
  // keep a counter of their own.
  const actdump::LayerScope dump_here(actdump::Current().step, layer_index);
  const char* dump_sub_dir = actdump::StageDir();
  auto DumpStage = [&](const char* stage, DBuf& buf) {
    if (dump_sub_dir == nullptr) return;
    ActDumpTensor(d, "VT_DUMP_ACT_SUB", dump_sub_dir, stage, buf.t(), T, H);
  };

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());
  DumpStage("post_input_norm", dhn);
  if (dump_sub_dir != nullptr) {
    // Same-instant second read via the DIRECT pattern (no DBuf/pool): if these
    // two disagree, the download patterns themselves diverge. Kept because it
    // is the control that tells a download artefact from a kernel difference,
    // and a cross-tier comparison needs that told apart before it can report a
    // layer.
    std::vector<uint8_t> rawD(static_cast<size_t>(T) * static_cast<size_t>(H) *
                              vt::SizeOf(dhn.t().dtype));
    d.b.Synchronize(d.q);
    d.b.Copy(d.q, rawD.data(), dhn.t().data, rawD.size());
    actdump::WriteBlob("VT_DUMP_ACT_SUB", dump_sub_dir, actdump::Current().step,
                       layer_index, "post_input_norm_DIRECT", dhn.t().dtype, T, H,
                       rawD.data(), rawD.size());
  }

  DBuf attn = [&] {
    if (layer.is_linear_attention) {
      VT_CHECK(gdn_state != nullptr,
               "paged dense layer: GDN layer needs a GdnStateCache");
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), sdi, gdn_meta, *gdn_state, T);
    }
    VT_CHECK(attn_kv != nullptr,
             "paged dense layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), sdi, attn_meta,
                              *attn_kv, T);
  }();
  DumpStage("block_out", attn);

  if (dump_sub_dir != nullptr) {
    // Triple-read probe: re-download the INPUT-NORM buffer after the mixer ran.
    // If it differs from the pre-mixer dump, something between them writes it;
    // if equal, the earlier disagreement is a download artifact.
    ActDumpTensor(d, "VT_DUMP_ACT_SUB", dump_sub_dir,
                        "post_input_norm_RECHECK", dhn.t(), T, H);
  }
  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());
  DumpStage("post_attn_norm", dh2);

  hidden = DenseMlpBlock(d, layer.mlp, cfg, dh2.t(), T);
  DumpStage("mlp_out", hidden);
}

// ── Qwen3.5/3.6 MTP head shared preamble (SPEC-MTP I5c). ────────────────────
// The fc-cat-norm head from qwen3_5_mtp.py:129-140, shared by the standalone
// (Qwen3_5MTPModel::Forward) and paged (ForwardPaged) drafts so a single copy of
// the head math feeds both. Produces `h = fc(cat[pre_fc_norm_embedding(embed),
// pre_fc_norm_hidden(target_hidden)])` as a [T,H] bf16 device buffer; the caller
// runs the (dense or MoE) decoder layer + final norm over it. `embed_tokens` is
// the shared target embedding; `target_hidden_states` is the target model's
// post-final-norm bf16 [T,H] output (the drafter's hidden-state tap).
DBuf MtpHeadHidden(Dev device, const Qwen3_5MTPWeights& weights,
                   const HfConfig& config, const OwnedTensor& embed_tokens,
                   const std::vector<int32_t>& input_ids,
                   const Tensor& target_hidden_states, int64_t tokens) {
  const int64_t hidden_size = config.hidden_size;
  const int64_t vocab_size = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  Tensor embedding_table = Qwen3_5EmbeddingTable(device.b, device.q, embed_tokens,
                                                vocab_size, hidden_size);
  DBuf device_ids(device, DType::kI32, {tokens}, input_ids.data());
  DBuf embedding(device, DType::kBF16, {tokens, hidden_size});
  vt::Embedding(device.q, embedding.t(), embedding_table, device_ids.t());

  Tensor embedding_norm_weight =
      ResidentWeight(device, weights.pre_fc_norm_embedding, {hidden_size});
  Tensor hidden_norm_weight =
      ResidentWeight(device, weights.pre_fc_norm_hidden, {hidden_size});
  DBuf embedding_norm(device, DType::kBF16, {tokens, hidden_size});
  DBuf target_norm(device, DType::kBF16, {tokens, hidden_size});
  vt::RmsNorm(device.q, embedding_norm.t(), embedding.t(), embedding_norm_weight,
              vt::RmsNormArgs{eps, true});
  vt::RmsNorm(device.q, target_norm.t(), target_hidden_states, hidden_norm_weight,
              vt::RmsNormArgs{eps, true});

  // torch.cat([embedding_norm, target_norm], -1), row by row (portable, exact).
  DBuf concatenated(device, DType::kBF16, {tokens, 2 * hidden_size});
  const size_t row_bytes =
      static_cast<size_t>(hidden_size) * vt::SizeOf(DType::kBF16);
  auto* cat = static_cast<uint8_t*>(concatenated.ptr());
  const auto* embed = static_cast<const uint8_t*>(embedding_norm.t().data);
  const auto* target = static_cast<const uint8_t*>(target_norm.t().data);
  for (int64_t token = 0; token < tokens; ++token) {
    const size_t source_offset = static_cast<size_t>(token) * row_bytes;
    const size_t target_offset = static_cast<size_t>(token) * 2 * row_bytes;
    device.b.Copy(device.q, cat + target_offset, embed + source_offset, row_bytes);
    device.b.Copy(device.q, cat + target_offset + row_bytes, target + source_offset,
                  row_bytes);
  }
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): the fc-cat projection through the ONE
  // EXL3 linear seam when the checkpoint quantized it, bf16 out because that is
  // what `MatmulBf16D` returns and what the decoder layer below consumes. There
  // is no second matmul: `dense_exl3::Linear` forwards to
  // `layers::Exl3LinearMethod`, which calls `dense_attn::Exl3MatmulD`.
  if (weights.IsExl3()) {
    return dense_exl3::Linear(device, concatenated.t(), weights.fc,
                              weights.fc_exl3, DType::kBF16);
  }
  return MatmulBf16D(device, concatenated.t(), weights.fc);
}

template <class Weights>
bool HasGdnConsumers(const Weights& weights) {
  // vLLM qwen3_5.py:144-160 @ e126687a9a constructs GDN only for these layers.
  // Cache absence cannot establish this: a hybrid model can have missing caches.
  return std::any_of(weights.layers.begin(), weights.layers.end(),
                     [](const auto& layer) { return layer.is_linear_attention; });
}

void ValidateFullAttnStepMetadata(int64_t tokens, const CommonAttentionMetadata& am) {
  VT_CHECK(am.num_actual_tokens == tokens,
           "qwen3_5 full-attn: attn metadata token count must match positions");
  VT_CHECK(static_cast<int64_t>(am.slot_mapping.size()) == tokens,
           "qwen3_5 full-attn: slot_mapping must cover every token");
  VT_CHECK(am.num_reqs >= 0 &&
               static_cast<int64_t>(am.seq_lens.size()) == am.num_reqs &&
               static_cast<int64_t>(am.query_start_loc.size()) == am.num_reqs + 1,
           "qwen3_5 full-attn: malformed full-attn metadata shapes");
  VT_CHECK(am.block_table_num_cols >= 0 &&
               static_cast<int64_t>(am.block_table_tensor.size()) ==
                   static_cast<int64_t>(am.num_reqs) * am.block_table_num_cols,
           "qwen3_5 full-attn: malformed block table");
  VT_CHECK(am.query_start_loc.front() == 0 && am.query_start_loc.back() == tokens &&
               std::is_sorted(am.query_start_loc.begin(), am.query_start_loc.end()),
           "qwen3_5 full-attn: query offsets must span tokens in order");
}

// ── Full-attention-only per-step device inputs (SPEC-MTP I5c). ──────────────
// BuildStepDevInputs sibling for a step with NO GDN layers (the MTP draft head
// is a single layer_type="full_attention" decoder — qwen3_5_mtp.py:105-112). It
// uploads exactly the full-attn tensors FullAttnBlockPaged reads (positions +
// slot_mapping / block_table / seq_lens / query_start_loc) and leaves every GDN
// field a size-1 stub, so it needs no GDN state slots or GDN metadata (which do
// not exist for the draft layer). MaybeBuildAttnCosSin fills attn_cos_sin.
StepDevInputs BuildFullAttnStepDevInputs(Dev d,
                                         const std::vector<int32_t>& positions,
                                         const CommonAttentionMetadata& am) {
  const int64_t T = static_cast<int64_t>(positions.size());
  ValidateFullAttnStepMetadata(T, am);
  return StepDevInputs{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kI32, {1}),  // gdn_state_idx stub
      false,
      DBuf(d, DType::kI32, {1}),  // gdn_non_spec_qsl stub
      DBuf(d, DType::kI8, {1}),   // gdn_has_initial stub
      DBuf(d, DType::kI32, {1}),  // gdn_prefill_state_idx stub
      DBuf(d, DType::kI32, {1}),  // gdn_prefill_qsl stub
      DBuf(d, DType::kI8, {1}),   // gdn_prefill_has_initial stub
      false,                       // has_gdn_prefill_meta
      DBuf(d, DType::kI32, {1}),  // gdn_conv_batch_ptr stub
      DBuf(d, DType::kI32, {1}),  // gdn_conv_token_chunk_offsets stub
      false,                       // has_gdn_conv_chunks
      false,                       // indexed_gdn_state_io (no GDN layers)
      DBuf(d, DType::kI32, {1}),  // gdn_spec_state_idx stub
      DBuf(d, DType::kI32, {1}),  // gdn_spec_qsl stub
      DBuf(d, DType::kI32, {1}),  // gdn_spec_token_indx stub
      DBuf(d, DType::kI32, {1}),  // gdn_non_spec_token_indx stub
      DBuf(d, DType::kI8, {1}),   // gdn_spec_seq_masks stub
      DBuf(d, DType::kI32, {1}),  // gdn_num_accepted stub
      DBuf(d, DType::kI32, {1}),  // gdn_spec_conv_state_idx stub
      false,                       // has_gdn_spec
      0,                           // gdn_spec_num_cols
      DBuf(d, DType::kF32, {1}),  // attn cos|sin stub (MaybeBuildAttnCosSin fills)
      false,
  };
}

// ── MTP final-norm + owning return (SPEC-MTP I5c), shared by Forward/ForwardPaged.
// mtp.norm over the fused residual stream (qwen3_5_mtp.py:163-165), then move the
// [T,H] bf16 result into a pool-backed owning carrier (the WrapDeviceLogits idiom).
Qwen3_5MTPHiddenStates MtpFinalize(Dev device, const Qwen3_5MTPWeights& weights,
                                   const HfConfig& config, DBuf& hidden,
                                   DBuf& residual, int64_t tokens) {
  const int64_t hidden_size = config.hidden_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  Tensor final_norm_weight =
      ResidentWeight(device, weights.final_norm, {hidden_size});
  DBuf normalized(device, DType::kBF16, {tokens, hidden_size});
  vt::RmsNorm(device.q, normalized.t(), hidden.t(), final_norm_weight,
              vt::RmsNormArgs{eps, true}, &residual.t());
  Qwen3_5MTPHiddenStates out;
  out.tensor = normalized.t();
  out.storage = normalized.ReleaseShared();
  return out;
}

}  // namespace

// Test-only exposed wrapper over the anon-ns GdnBlockPaged + BuildStepDevInputs
// (SPEC-MTP I5a). Runs ONE GDN layer's paged forward over one batched step,
// driving the exact production per-step upload (BuildStepDevInputs) and layer
// assembly (GdnBlockPaged) — including the spec branch when `meta` carries
// drafts — and returns the [T*H] output on host (f32). It stages the SSM/conv
// state onto the queue's device, runs, and downloads the mutated state back into
// `ssm_host`/`conv_host` (both f32, updated in place), so the caller works purely
// in host vectors on either backend. The synthetic spec-branch test uses this to
// compare the spec pass bit-for-bit against a token-sequential non-spec decode
// chain over the same weights and inputs; a mis-wired split / slot select / merge
// diverges. h_host is the f32 hidden [T*H] (rounded to bf16 on upload, exactly
// like the real forward's embed target). conv_len is (K-1) for the non-spec
// decode reference and (K-1)+num_spec for the widened spec state.
// PERF-27B-GDN-FP8-QKVZ numerical harness. Both arms are driven from ONE
// process (the env toggle is process-cached, so an in-process A/B has to select
// the arm explicitly), over the same uploaded activation and the same resident
// bytes, so a bitwise comparison of the two results is exactly the spec's
// "merged output byte-identical to the concatenation of the two legacy GEMM
// outputs".
std::vector<float> ProjectGdnFp8QkvzForTest(vt::Queue queue,
                                            const GdnLayerWeights& w,
                                            const std::vector<float>& h_host,
                                            int64_t T, int64_t conv_dim,
                                            int64_t value_dim, bool merged,
                                            bool z_bf16) {
  Backend& b = vt::GetBackend(queue.device.type);
  Dev d{b, queue};
  VT_CHECK(!w.in_proj_qkv_fp8.Empty() && !w.in_proj_z_fp8.Empty(),
           "ProjectGdnFp8QkvzForTest: fp8 GDN shards required");
  const int64_t H = w.in_proj_qkv_fp8.k;
  VT_CHECK(static_cast<int64_t>(h_host.size()) == T * H,
           "ProjectGdnFp8QkvzForTest: h_host must be [T*H]");
  const DType outdt = z_bf16 ? DType::kBF16 : DType::kF32;
  DBuf hf(d, DType::kF32, {T, H}, h_host.data());
  DBuf h(d, DType::kBF16, {T, H});
  vt::CastBf16(d.q, h.t(), hf.t());

  GdnQkvzOutput out;
  if (merged) {
    out.packed_owner.emplace(MergedFp8QkvzD(d, h.t(), nullptr, w));
    Tensor packed = out.packed_owner->t();
    out.mixed = packed.Slice(1, 0, conv_dim);
    Tensor z_f32 = packed.Slice(1, conv_dim, conv_dim + value_dim);
    if (outdt == DType::kF32) {
      out.z = z_f32;
    } else {
      out.z_owner.emplace(d, DType::kBF16, std::vector<int64_t>{T, value_dim});
      vt::CastBf16(d.q, out.z_owner->t(), z_f32);
      out.z = out.z_owner->t();
    }
  } else {
    out.mixed_owner.emplace(
        MatmulFp8CutlassD(d, h.t(), w.in_proj_qkv_fp8, DType::kF32));
    out.z_owner.emplace(MatmulFp8CutlassD(d, h.t(), w.in_proj_z_fp8, outdt));
    out.mixed = out.mixed_owner->t();
    out.z = out.z_owner->t();
  }

  // Assemble [mixed_qkv | z] on the HOST, so no device op has to consume the
  // merged arm's strided views (which is the point of them).
  const int64_t total = conv_dim + value_dim;
  std::vector<float> host(static_cast<size_t>(T * total), 0.0F);
  if (merged) {
    // packed_owner is contiguous f32 [T, conv+value]; mixed (and, when z stays
    // f32, z) are exactly its column ranges.
    std::vector<float> packed(static_cast<size_t>(T * total));
    out.packed_owner->Download(d, packed.data());
    for (int64_t t = 0; t < T; ++t)
      for (int64_t i = 0; i < conv_dim; ++i)
        host[static_cast<size_t>(t * total + i)] =
            packed[static_cast<size_t>(t * total + i)];
    if (!z_bf16) {
      for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < value_dim; ++i)
          host[static_cast<size_t>(t * total + conv_dim + i)] =
              packed[static_cast<size_t>(t * total + conv_dim + i)];
    }
  } else {
    std::vector<float> mixed(static_cast<size_t>(T * conv_dim));
    out.mixed_owner->Download(d, mixed.data());
    for (int64_t t = 0; t < T; ++t)
      for (int64_t i = 0; i < conv_dim; ++i)
        host[static_cast<size_t>(t * total + i)] =
            mixed[static_cast<size_t>(t * conv_dim + i)];
    if (!z_bf16) {
      std::vector<float> zf(static_cast<size_t>(T * value_dim));
      out.z_owner->Download(d, zf.data());
      for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < value_dim; ++i)
          host[static_cast<size_t>(t * total + conv_dim + i)] =
              zf[static_cast<size_t>(t * value_dim + i)];
    }
  }
  if (z_bf16) {
    // Both arms own a contiguous bf16 z here; upcast losslessly so the caller
    // compares the exact stored bf16 bit patterns as floats.
    std::vector<uint16_t> zb(static_cast<size_t>(T * value_dim));
    out.z_owner->Download(d, zb.data());
    for (int64_t t = 0; t < T; ++t)
      for (int64_t i = 0; i < value_dim; ++i)
        host[static_cast<size_t>(t * total + conv_dim + i)] =
            vt::BF16ToF32(zb[static_cast<size_t>(t * value_dim + i)]);
  }
  return host;
}

std::vector<float> GdnBlockPagedForTest(vt::Queue queue, const GdnLayerWeights& w,
                                        const HfConfig& cfg,
                                        const std::vector<float>& h_host,
                                        const v1::GDNAttentionMetadata& meta,
                                        std::vector<float>& ssm_host,
                                        std::vector<float>& conv_host,
                                        int64_t num_slots, int64_t conv_len,
                                        int64_t T) {
  Backend& b = vt::GetBackend(queue.device.type);
  Dev d{b, queue};
  const int64_t H = cfg.hidden_size;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t conv_dim = 2 * Hk * Dk + Hv * Dv;
  VT_CHECK(static_cast<int64_t>(h_host.size()) == T * H,
           "GdnBlockPagedForTest: h_host must be [T*H]");
  VT_CHECK(static_cast<int64_t>(ssm_host.size()) == num_slots * Hv * Dv * Dk,
           "GdnBlockPagedForTest: ssm_host must be [slots*Hv*Dv*Dk]");
  VT_CHECK(static_cast<int64_t>(conv_host.size()) == num_slots * conv_dim * conv_len,
           "GdnBlockPagedForTest: conv_host must be [slots*conv_dim*conv_len]");
  DBuf hf(d, DType::kF32, {T, H}, h_host.data());
  DBuf h(d, DType::kBF16, {T, H});
  vt::CastBf16(d.q, h.t(), hf.t());
  // Device-staged f32 SSM/conv caches (mutated in place by GdnBlockPaged).
  DBuf ssm(d, DType::kF32, {num_slots, Hv, Dv, Dk}, ssm_host.data());
  DBuf conv(d, DType::kF32, {num_slots, conv_dim, conv_len}, conv_host.data());
  GdnStateCache state;
  state.ssm_state = ssm.t();
  state.conv_state = conv.t();

  // Minimal CommonAttentionMetadata — GdnBlockPaged never reads it, but
  // BuildStepDevInputs uploads its positions/slot_mapping/block_table/seq_lens/
  // query_start_loc. One request per non-spec segment or one spec request; the
  // exact non-GDN fields do not affect the GDN layer output.
  const int num_reqs = meta.num_decodes + meta.num_prefills + meta.num_spec_decodes;
  v1::CommonAttentionMetadata am;
  am.num_reqs = num_reqs;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = std::max<int>(1, meta.spec_state_indices_num_cols);
  am.block_table_tensor.assign(
      static_cast<size_t>(num_reqs) * static_cast<size_t>(am.block_table_num_cols), 0);
  am.seq_lens.assign(static_cast<size_t>(num_reqs), static_cast<int32_t>(T));
  am.query_start_loc.assign(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r)
    am.query_start_loc[static_cast<size_t>(r) + 1] =
        am.query_start_loc[static_cast<size_t>(r)] +
        static_cast<int32_t>(T / std::max(1, num_reqs));
  am.query_start_loc.back() = static_cast<int32_t>(T);
  am.slot_mapping.assign(static_cast<size_t>(T), 0);
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  const int64_t state_slots = state.ssm_state.shape[0];
  StepDevInputs sdi = BuildStepDevInputs(d, positions, am, meta, state_slots);
  DBuf out = GdnBlockPaged(d, w, cfg, h.t(), sdi, meta, state, T);

  DBuf of(d, DType::kF32, {T, H});
  vt::CastF32(d.q, of.t(), out.t());
  std::vector<float> host(static_cast<size_t>(T * H));
  of.Download(d, host.data());
  // Return the mutated SSM/conv state to the caller (rollback slots, evolved
  // decode state) so a token-sequential reference chain can carry it forward.
  ssm.Download(d, ssm_host.data());
  conv.Download(d, conv_host.data());
  return host;
}

// Exposed wrapper over the anon-ns `MoeBlock` (SEAM GAP #2) so a full-attention
// MoE in another TU (qwen3_moe.cpp, W3) can reuse the exact same sparse-MoE block.
// Builds the internal Dev, runs MoeBlock, and releases the combined DBuf into an
// owning MoeBlockOutput whose deleter returns the pool block to the DevicePool
// (the WrapDeviceLogits release pattern). The 35B path never calls this — it is a
// pure ADD, so Qwen3.6-35B remains byte-identical.
MoeBlockOutput RunMoeBlock(vt::Queue& queue, const MoeBlockWeights& weights,
                           const HfConfig& config, const vt::Tensor& dh, int64_t T) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf out = MoeBlock(d, weights, config, dh, T);
  MoeBlockOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

// How many times the W8 device-resident keep-quant arm (`MoeBlockKqDevice`) has
// run in this process. The ARM-SELECTION probe, and the only thing that can see
// the defect this row closes: the two arms agree on every byte they produce, so
// an output comparison alone cannot say WHICH one ran, and a gate that cannot
// say that would have stayed green through the whole history of this bug. The
// `qwen3_5.cpp`-internal `MoeBlock` has internal linkage, so this is the same
// cross-TU shape `RunMoeBlock` above already uses to expose it.
int64_t MoeKqDeviceCalls() { return MoeKqDeviceCallCount(); }


// Exposed wrapper over the anon-ns `GdnBlockPaged` (row MODEL-MM-QWEN4-EXP W5b,
// issue #2110) so a hybrid architecture in another TU — Qwen4-Exp, whose
// `kLinearAttention` layers ARE this block — reuses the exact same GDN layer.
// Mirrors `RunMoeBlock` above, for the identical reason and in the identical
// shape: build the internal Dev, run the block, release the pooled DBuf into an
// owning GdnBlockOutput. The Qwen3.5/3.6 forward never calls this — it is a pure
// ADD, so that path stays byte-identical. See qwen3_5_gdn_block.h.
GdnStepInputs BuildGdnStepInputs(vt::Queue& queue,
                                 const std::vector<int32_t>& positions,
                                 const CommonAttentionMetadata& attn_meta,
                                 const GDNAttentionMetadata& gdn_meta,
                                 int64_t gdn_state_slots) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  GdnStepInputs s;
  s.impl = std::make_shared<StepDevInputs>(
      BuildStepDevInputs(d, positions, attn_meta, gdn_meta, gdn_state_slots));
  return s;
}

GdnBlockOutput RunGdnBlockPaged(vt::Queue& queue, const GdnLayerWeights& weights,
                                const HfConfig& config, const vt::Tensor& dh,
                                const GdnStepInputs& step,
                                const GDNAttentionMetadata& gdn_meta,
                                const GdnStateCache& state, int64_t T,
                                const vt::Tensor* dh_fp8) {
  VT_CHECK(step.impl != nullptr,
           "RunGdnBlockPaged: the step inputs are empty; build them once per "
           "step with BuildGdnStepInputs and keep the handle alive across the "
           "layer loop");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const auto& sdi = *static_cast<const StepDevInputs*>(step.impl.get());
  DBuf out = GdnBlockPaged(d, weights, config, dh, sdi, gdn_meta, state, T, dh_fp8);
  GdnBlockOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

// ENG-EXPERT-STREAM (#912): the streamed-expert lane seen from outside this TU.
// See qwen3_5_internal.h for why a benchmark and a gate both need to reach it.
detail::ExpertStreamStats detail::ExpertStreamSnapshot() {
  ExpertStreamStats s;
  const Qwen35ExpertStream* st = Qwen35ExpertStream::Existing();
  if (st == nullptr) return s;  // never requested, or requested and never used
  s.active = true;
  s.steps = st->cache().steps();
  s.hits = st->cache().hits();
  s.misses = st->cache().misses();
  s.evictions = st->cache().evictions();
  s.fills = st->streamer().fills();
  s.bytes_filled = st->streamer().bytes_filled();
  s.exhausted = st->exhausted();
  s.forced = st->forced();
  s.advised = st->advised();
  return s;
}

void detail::ExpertStreamSetForceFallback(bool on) {
  Qwen35ExpertStream::SetForceFallback(on);
}

// ENG-EXPERT-STREAM-DEVICE W0c (issue #1124). See the declarations in
// qwen3_5_internal.h for what these reach and what they deliberately do not
// claim to prove. Both are one-line delegations to the production helpers, so a
// gate observes exactly the code the forward runs and not a copy of it.
vt::Tensor detail::ExpertSliceForTest(vt::Queue& q, const OwnedTensor& w,
                                      int64_t N, int64_t K, int64_t row_off,
                                      int64_t expert) {
  Dev d{vt::GetBackend(q.device.type), q};
  return KqExpertSlice(d, w, N, K, row_off, expert);
}

vt::Tensor detail::StageWeightForTest(vt::Queue& q, const OwnedTensor& w) {
  Dev d{vt::GetBackend(q.device.type), q};
  return ResidentWeight(d, w);
}

void detail::EndExpertStreamStep() { Qwen35ExpertStream::EndStepIfActive(); }

void detail::ExpertStreamFlushStats() { Qwen35ExpertStream::FlushFinalStats(); }

// The step guard as a scope a gate can hold. These forward to the SAME
// `Begin`/`End` the production guard's constructor and destructor call, so the
// nesting refusal a gate observes here is the one every forward in this file is
// protected by, not a re-statement of it.
detail::ExpertStreamStepScope::ExpertStreamStepScope() {
  Qwen35ExpertStreamStep::Begin();
}
detail::ExpertStreamStepScope::~ExpertStreamStepScope() {
  Qwen35ExpertStreamStep::End();
}

// ENG-ASYNC-SCHED W4: overwrite the REAL prefix of a freshly uploaded input-id
// buffer with the device-resident ids the async runner's combine produced.
//
// Why patch a prefix instead of embedding straight from the runner's buffer: the
// decode-graph path does not embed `token_ids` as given — it embeds a version
// PADDED up to the captured batch size, whose first B rows are the real requests
// and whose tail is inert. So the correct operation is "replace the first
// ov.count rows", which is exactly right for the padded case AND degenerates to
// "replace everything" on the eager path where ov.count == T. The stale host
// upload that precedes it is a handful of int32s and its real rows are
// immediately overwritten.
//
// The override is CONSUMED here. A forward can reach a second, unrelated embed
// (the multimodal helper embeds a prompt and then single tokens); consuming on
// first use means those cannot be handed ids that were never meant for them.
// The first embed in a registry forward is always the step's own.
// #1305: the take-and-clear and the bounds-checked copy this used to spell out
// are `detail::ApplyDeviceTokenIds` (defined above, declared in
// `qwen3_5_internal.h`), one body for the four models that consume the scope.
static void ApplyDeviceTokenIdsOverride(Dev d, DBuf& dids, int64_t T) {
  detail::ApplyDeviceTokenIds(d.b, d.q, dids.ptr(), T, "qwen3_5 embed");
}

// Embed: hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident table).
// KEPT OUTSIDE THE CUDA-GRAPH (M2.5 Phase 2): the CUDA Embedding op allocates a
// device bounds-check flag (cudaMalloc/cudaFree) and syncs the stream, all of
// which are illegal inside a capture region — and it consumes host token_ids.
// The graph driver runs this per step into its PERSISTENT hidden buffer, then
// captures/replays ForwardLayers over that fixed hidden address.
static void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
                      const Qwen3_5MoeWeights& weights, const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
    if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-FWD] EmbedInto T=%lld H=%lld\n",
                 static_cast<long long>(T), static_cast<long long>(H));
const int64_t vocab = config.vocab_size;
  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, vocab, H);
  // ENG-ASYNC-SCHED W4: when the async runner has already placed this step's
  // input ids on the device (and spliced each decode row's sampled token into
  // them there), embed straight from that buffer. `token_ids` is stale for
  // decode rows in that case BY DESIGN — materializing it on the host is the
  // synchronize W4 removes — so it must not be uploaded here. Its SIZE is still
  // authoritative: the runner sized the device buffer from the same step.
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  ApplyDeviceTokenIdsOverride(d, dids, T);
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// DFlash DF-AUX-TAPS (SPEC-DFLASH D1) — capture the residual-stream value at a
// decoder-layer boundary into the aux-taps buffer. Called after each RunLayerPaged
// with the JUST-RUN layer index `l`; if `l` is in `aux_layer_ids`, it writes
// (hidden + res) — the exact value vLLM appends at eagle3 aux key l+1
// (interfaces.py:1382 `hidden_states + residual`) — as bf16 into the k-th [T,H]
// column block of `aux_out` = [T, H×taps], where k is l's index in aux_layer_ids
// (so column order == ascending layer_ids == cat(aux, dim=-1)). INERT when
// `aux_out` is null: no shipped forward passes a buffer, so the object code on the
// default/MTP-spec path is byte-identical (identical discipline to the single tap).
// The strided per-tap write is a T-row copy loop; DFlash captures on small query
// blocks and this path is unused in production until D4, so a fused column scatter
// is deferred (D-later perf, not correctness).
// DFlash DF-AUX-TAPS layer-id contract: the target_layer_ids must be non-empty,
// STRICTLY ASCENDING (the capture-order == concat-order invariant, matching vLLM's
// increasing-layer_idx aux append), and each in [0, num_hidden_layers). A wrong or
// out-of-order id list is a construction bug, not a runtime near-tie — fail loud.
static void ValidateAuxTapLayerIds(const std::vector<int32_t>& layer_ids,
                                   int64_t num_hidden_layers) {
  VT_CHECK(!layer_ids.empty(), "qwen3_5 aux tap: layer_ids must be non-empty");
  for (size_t k = 0; k < layer_ids.size(); ++k) {
    VT_CHECK(layer_ids[k] >= 0 &&
                 static_cast<int64_t>(layer_ids[k]) < num_hidden_layers,
             "qwen3_5 aux tap: target_layer_id out of [0, num_hidden_layers)");
    VT_CHECK(k == 0 || layer_ids[k] > layer_ids[k - 1],
             "qwen3_5 aux tap: target_layer_ids must be strictly ascending "
             "(capture order == concat order)");
  }
}

static void MaybeCaptureAuxTap(Dev d, int64_t l,
                               const std::vector<int32_t>* aux_layer_ids,
                               const Tensor* aux_out, const Tensor& hidden,
                               const Tensor& res, int64_t T, int64_t H) {
  if (aux_out == nullptr) return;
  for (size_t k = 0; k < aux_layer_ids->size(); ++k) {
    if (static_cast<int64_t>((*aux_layer_ids)[k]) != l) continue;
    DBuf tmp(d, DType::kBF16, {T, H});
    Tensor tt = tmp.t();
    vt::Add(d.q, tt, hidden, res);  // bf16 (hidden+res); matches eagle3 aux value
    const size_t row_bytes = static_cast<size_t>(H) * vt::SizeOf(DType::kBF16);
    const int64_t taps = aux_out->shape[1] / H;
    const size_t dst_pitch = static_cast<size_t>(taps) * row_bytes;
    char* dst0 = static_cast<char*>(aux_out->data) + static_cast<size_t>(k) * row_bytes;
    const char* src0 = static_cast<const char*>(tt.data);
    for (int64_t t = 0; t < T; ++t)
      d.b.Copy(d.q, dst0 + static_cast<size_t>(t) * dst_pitch,
               src0 + static_cast<size_t>(t) * row_bytes, row_bytes);
  }
}

// The CAPTURABLE paged forward region: everything AFTER the embedding — the
// residual stream (res=0), the N paged decoder layers, the final RMSNorm and the
// lm_head — returning the [T,vocab] f32 logits as a device DBuf (NO host
// Download; the caller Downloads, or the graph keeps it resident as the graph's
// output). `hidden_in` is the embedded input (a view over the graph's persistent
// hidden buffer on the replay path); it is COPIED into a working buffer so the
// layers' in-place residual thread never disturbs the persistent embedding.
// Split out of Forward (M2.5 Phase 2) so the exact op sequence is what the graph
// captures/replays; every per-step-varying input is read from a HOST vector
// argument (positions / the metadata vectors), whose host->device copies are
// capturable on GB10 (pageable memory access) and which the graph driver keeps
// persistent + mutates in place so replays pick up each new token's inputs.
static DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                          const std::vector<int32_t>& positions,
                          const CommonAttentionMetadata& attn_meta,
                          const GDNAttentionMetadata& gdn_meta,
                          const std::vector<PagedKvCache>& attn_kv,
                          const std::vector<GdnStateCache>& gdn_state,
                          const Qwen3_5MoeWeights& weights, const HfConfig& config,
                          const std::vector<int32_t>& logits_indices = {},
                          const Tensor* hidden_tap = nullptr,
                          const std::vector<float>* mrope_cos_sin = nullptr,
                          const std::vector<int32_t>* aux_layer_ids = nullptr,
                          const Tensor* aux_out = nullptr,
                          StepDevInputs* persistent_sdi = nullptr) {
  // ONE decode step, for the PAGED forwards: ForwardBody, the VL path and the
  // graph driver's eager fallback all funnel through here exactly once per
  // forward. This is the step boundary the expert slot cache is defined against,
  // and it is neither once per layer nor once per expert.
  //
  // IT IS NOT THE ONLY ENTRY POINT, and the comment this replaces said it was
  // (#1091 finding 3). Four more forwards reach `ExpertMlpKq -> KqExpertSlice`
  // without passing through here — `Qwen3_5Model::ForwardDense`,
  // `Qwen3_5MTPModel::Forward`, `Qwen3_5MTPModel::ForwardPaged` and
  // `Qwen3_5ReplayLayer` — and each now carries its own guard.
  //
  // ONE OF THOSE FOUR HAS A PRODUCTION CALLER, not all of them, and an earlier
  // revision of this comment said "the MTP pair" (#1106 finding 2, #1108). It is
  // `Qwen3_5MTPModel::ForwardPaged`, the spec-decode DRAFT forward, reached from
  // `runner.cpp:2183` through `spec_decode/mtp/speculator.cpp:107,262` — so the
  // shape that was actually running is draft forwards that pin every slot they
  // touch across the following target forward. That caller is itself
  // "UNREACHABLE unless a speculator is configured" (`runner.cpp:2120`), so a
  // DEFAULT-configuration run reaches none of these four guards; one of them has
  // a production caller, which is not the same claim. `Qwen3_5MTPModel::Forward`,
  // `Qwen3_5Model::ForwardDense` and `Qwen3_5ReplayLayer` are parity entry
  // points whose every caller is under `tests/`, and per `.agents/reachability.md`
  // a call site inside a test is not reach: their guards land UNREACHED, which
  // the spec's `## Owed` records as a staged slice rather than claiming.
  //
  // `RunMoeBlock` is the deliberate exception: it is one block, not a forward,
  // and qwen3_moe.cpp owns the boundary for the model that composes it
  // (qwen3_moe.cpp:150).
  //
  // Inert unless a store exists, which needs both VT_MOE_EXPERT_STREAM and a
  // slice taken.
  const Qwen35ExpertStreamStep expert_stream_step;
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy of the embedded hidden (device->device; captured). RunLayerPaged
  // reassigns `hidden` per layer, so this must NOT alias the persistent buffer.
  DBuf hidden(d, ActDType(d), {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  // Upload the per-step inputs ONCE (positions + full-attn metadata + GDN decode
  // state indices) into persistent device buffers all layers read — replaces the
  // per-layer H2D re-uploads (the decode host-stall root; see StepDevInputs).
  // VT_ASYNC_EXECUTOR (Option A): a non-null persistent_sdi points at the SizeSlot's
  // persistent device buffers, which the graph driver stages the H2D into OUTSIDE
  // this captured region — so here we do NOT allocate or upload; we only re-derive
  // the on-device cos|sin cache from the (re-staged) persistent positions so replays
  // pick up each new token's rope. Eager / capture-of-Option-B path (nullptr) builds
  // and uploads the inputs inline as before (byte-identical).
  const int64_t gdn_state_slots =
      gdn_state.empty() ? 0 : gdn_state.front().ssm_state.shape[0];
  const bool fp4_attn = [&] {
    for (const auto& l : weights.layers)
      if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
    return false;
  }();
  std::optional<StepDevInputs> local_sdi;
  if (persistent_sdi == nullptr)
    local_sdi.emplace(HasGdnConsumers(weights)
                          ? BuildStepDevInputs(d, positions, attn_meta, gdn_meta,
                                               gdn_state_slots)
                          : BuildFullAttnStepDevInputs(d, positions, attn_meta));
  StepDevInputs& sdi = persistent_sdi != nullptr ? *persistent_sdi : *local_sdi;
  // Build the fused-preamble cos|sin cache ONCE; fp4_attn keys the per-arch
  // default (fp8/bf16 attn — the 35B — stays OFF; VT_FUSE_ATTN_PREAMBLE overrides).
  //
  // MOE VL path (issue #891): a prebuilt MRoPE cos|sin cache [T, rotary_dim]
  // (host f32, interleaved 3-section selection already baked in) is injected
  // verbatim into the fused full-attn preamble, replacing the 1-D RoPE cache
  // MaybeBuildAttnCosSin would build — the SAME injection point the dense arm's
  // VL forward uses (DenseForwardLayers below). mrope_cos_sin == nullptr (every
  // text caller, and every graph-captured caller) ⇒ byte-identical to before.
  if (persistent_sdi != nullptr) {
    // Persistent buffer already allocated (pre-capture); re-fill only, captured so
    // every replay re-derives rope from the freshly-staged positions.
    if (sdi.has_attn_cos_sin) FillAttnCosSin(d, sdi, config);
  } else if (mrope_cos_sin != nullptr) {
    const int rot = static_cast<int>(config.rotary_dim);
    VT_CHECK(rot > 0 && static_cast<int64_t>(mrope_cos_sin->size()) == T * rot,
             "qwen3_5 moe VL: MRoPE cos|sin cache must be [T, rotary_dim]");
    sdi.attn_cos_sin = DBuf(d, DType::kF32, {T, rot}, mrope_cos_sin->data());
    sdi.has_attn_cos_sin = true;
  } else {
    MaybeBuildAttnCosSin(d, sdi, config, T, fp4_attn);
  }

  // ROCM-TIER-DIVERGENCE (#2590): the MoE arm is keyed the same way as the dense
  // one. It used to write ONE hand-rolled pre-layer snapshot with the silent
  // `if (f != nullptr)` shape, no per-layer rows, and a name (`layer_-1_*`) that
  // no step index could distinguish — so a second forward overwrote the first.
  // It now shares the writer, the (step, layer) key and the manifest, and its
  // nested GDN stage probes get a valid key from the same scope.
  const int64_t dump_step = actdump::BeginStep();
  if (dump_step >= 0) {
    actdump::NarrateOnce(d.q.device.type, "Qwen3_5Model",
                         config.num_hidden_layers, T, H);
  }
  ActDumpStream(d, dump_step, -1, hidden, res, T, H);

  int64_t fa_idx = 0, gdn_idx = 0;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3_5MoeLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    const PagedKvCache* kv =
        layer.is_linear_attention ? nullptr : &attn_kv[static_cast<size_t>(fa_idx++)];
    const GdnStateCache* gs =
        layer.is_linear_attention ? &gdn_state[static_cast<size_t>(gdn_idx++)] : nullptr;
    RunLayerPaged(d, layer, config, hidden, res, sdi, attn_meta, gdn_meta,
                  kv, gs, T, /*layer_index=*/l);
    // DFlash DF-AUX-TAPS: capture (hidden+res) at configured boundaries. Inert
    // (no-op) when aux_out is null — every non-DFlash caller.
    MaybeCaptureAuxTap(d, l, aux_layer_ids, aux_out, hidden.t(), res.t(), T, H);
    ActDumpStream(d, dump_step, l, hidden, res, T, H);
  }
  // The MoE loop carries no sub-stage probes, so a VT_DUMP_ACT_SUB-only run here
  // legitimately writes nothing and `any_due` follows the stream knob.
  actdump::EndStep(dump_step,
                   actdump::StreamDir() == nullptr
                       ? 0
                       : 2 * (config.num_hidden_layers + 1),
                   /*any_due=*/actdump::StreamDir() == nullptr ? 0 : 1);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // Hidden-state tap (SPEC-MTP I5c): the full [T,H] post-final-norm hidden the
  // MTP drafter consumes (qwen3_5_mtp.py:129-140 `hidden_states` input; the
  // tap tensor is exactly upstream's target-model forward output). Captured
  // BEFORE the logits gather so the drafter sees every token's hidden, not just
  // the sampled rows. INERT on the production path: hidden_tap is nullptr for
  // every existing caller (only I5c's ForwardDeviceTap passes a buffer), so no
  // shipped forward writes it and the object code is byte-identical there.
  if (hidden_tap != nullptr) {
    VT_CHECK(hidden_tap->shape[0] == T && hidden_tap->shape[1] == H &&
                 hidden_tap->dtype == DType::kBF16,
             "qwen3_5: hidden tap buffer must be bf16 [T,H]");
    d.b.Copy(d.q, hidden_tap->data, dnorm.t().data,
             static_cast<size_t>(T) * static_cast<size_t>(H) *
                 vt::SizeOf(DType::kBF16));
  }

  // Logits gather (perf): mirror vLLM's gather-BEFORE-lm_head. On a prefill/mixed
  // step (len(logits_indices) < T) gather only the per-request last-token hidden
  // rows [num_reqs,H] and run lm_head on those → [num_reqs,vocab]. This is the
  // whole win: lm_head over ~num_reqs rows instead of T, and only that tiny
  // logits tensor is materialized/Downloaded. Pure-decode / graph replay pass
  // empty indices (identity) so the full [T,vocab] path is unchanged.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  if (do_gather) {
    const int64_t n_out = static_cast<int64_t>(logits_indices.size());
    DBuf dgather(d, DType::kBF16, {n_out, H});
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    return weights.lm_head_fp4.Empty()
               ? MatmulF32D(d, dgather.t(), weights.lm_head)
               : MatmulNvfp4F32D(d, dgather.t(), weights.lm_head_fp4);
  }

  // lm_head: fp4-resident (M2.2b) when populated, else the bf16/GGUF path.
  return weights.lm_head_fp4.Empty()
             ? MatmulF32D(d, dnorm.t(), weights.lm_head)
             : MatmulNvfp4F32D(d, dnorm.t(), weights.lm_head_fp4);
}

// Full eager paged forward body: embed (host token_ids) then the capturable
// layer region. Used by Qwen3_5Model::Forward and the graph driver's eager
// fallback / cold-shape pre-warm step (one contiguous stream, no capture).
static DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                        const std::vector<int32_t>& positions,
                        const CommonAttentionMetadata& attn_meta,
                        const GDNAttentionMetadata& gdn_meta,
                        const std::vector<PagedKvCache>& attn_kv,
                        const std::vector<GdnStateCache>& gdn_state,
                        const Qwen3_5MoeWeights& weights, const HfConfig& config,
                        const std::vector<int32_t>& logits_indices = {},
                        const Tensor* hidden_tap = nullptr,
                        const std::vector<int32_t>* aux_layer_ids = nullptr,
                        const Tensor* aux_out = nullptr) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-FWD] ForwardBody T=%lld H=%lld num_reqs=%lld "
                 "logits_indices=%zu\n",
                 static_cast<long long>(T), static_cast<long long>(H),
                 static_cast<long long>(attn_meta.num_reqs),
                 logits_indices.size());
  DBuf hidden(d, ActDType(d), {T, H});
  EmbedInto(d, hidden, token_ids, weights, config);
  return ForwardLayers(d, hidden.t(), positions, attn_meta, gdn_meta, attn_kv,
                       gdn_state, weights, config, logits_indices, hidden_tap,
                       /*mrope_cos_sin=*/nullptr, aux_layer_ids, aux_out);
}

// Shared shape/count validation for the paged forward entry points.
static void CheckPagedForward(const std::vector<int32_t>& token_ids,
                              const std::vector<int32_t>& positions,
                              const CommonAttentionMetadata& attn_meta,
                              const GDNAttentionMetadata& gdn_meta,
                              const std::vector<PagedKvCache>& attn_kv,
                              const std::vector<GdnStateCache>& gdn_state,
                              const Qwen3_5MoeWeights& weights,
                              const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "qwen3_5 paged forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 paged forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 paged forward: weights.layers size must equal num_hidden_layers");
  VT_CHECK(attn_meta.num_actual_tokens == T,
           "qwen3_5 paged forward: attn_meta.num_actual_tokens must equal T");
  int64_t n_full = 0, n_gdn = 0;
  for (const auto& l : weights.layers)
    (l.is_linear_attention ? n_gdn : n_full) += 1;
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == n_full,
           "qwen3_5 paged forward: attn_kv count must equal full-attn layer count");
  VT_CHECK(static_cast<int64_t>(gdn_state.size()) == n_gdn,
           "qwen3_5 paged forward: gdn_state count must equal GDN layer count");
  const int64_t state_slots =
      detail::ValidateGdnStateCacheLayout(gdn_state);
  if (n_gdn > 0) {
    VT_CHECK(gdn_meta.num_actual_tokens == T,
             "qwen3_5 paged forward: gdn_meta.num_actual_tokens must equal T");
    detail::ValidateGdnAttentionMetadata(
        gdn_meta, state_slots, /*allow_inert_padding=*/false);
  }
}

// Transfer a freshly-produced [rows, vocab] device logits DBuf into an OWNING
// ForwardLogits (the sampler-on-device return). The pool block's lifetime moves
// into a shared_ptr whose deleter returns it to the DevicePool — so there is NO
// per-step cudaMalloc/cudaFree and the buffer safely outlives sampling (the
// runner holds the ForwardLogits across execute_model -> sample_tokens).
static ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = dlogits.t().shape[0];
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();  // view (raw data ptr survives Release)
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

// Wrap the first `rows` rows of an EXTERNALLY-owned (persistent) device logits
// buffer as a NON-owning ForwardLogits view (the decode-graph slot keeps the
// storage alive across steps; each replay overwrites it, so the sampler may
// mutate the view in place). Rows are contiguous (row-major), so the first `rows`
// rows are a plain prefix view over `base`.
static ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                                      int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  // Non-owning: keep on_device() true without taking ownership of `base`.
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  fl.non_owning_view = true;  // releasing this carrier frees nothing (runner drain-skip signal)
  return fl;
}

// Materialize every deferred layer's routed-expert host copies at once (the
// non-interleaved fallback: CPU / wmma / VT_NVFP4_MARLIN=0 / no-VT_MARLIN_NVFP4
// build — paths whose forward reads the host bytes and that never reach the
// per-layer device build+free). No-op when experts were loaded eagerly.
static void MaterializeAllDeferredExperts(const Qwen3_5MoeWeights& weights) {
  if (!weights.load_layer_experts) return;
  for (size_t l = 0; l < weights.layers.size(); ++l) {
    weights.load_layer_experts(
        static_cast<int64_t>(l),
        const_cast<MoeBlockWeights&>(weights.layers[l].moe));
  }
  weights.load_layer_experts = nullptr;  // drop the shard keepalive
}

void Qwen3_5Model::PrepareMarlinResident(const Qwen3_5MoeWeights& weights,
                                         const HfConfig& config, vt::Queue& queue) {
#ifdef VT_MARLIN_NVFP4
  // POLICY vs KERNEL-PATH split (BACKEND-PLATFORM item 2): the per-layer load
  // interleave + host release runs when (a) the platform's residency POLICY frees
  // host weights after upload (release_host_weights_after_upload — read per-device
  // from the queue's platform; true on GB10/CUDA today, false on a retain-host /
  // CPU platform) AND (b) Marlin is the committed KERNEL path (MarlinMoeEnabled()
  // — there is a per-layer device build to interleave with and the host-reading
  // wmma fallback can never run). ShouldInterleaveLoadStream reproduces the old
  // `queue.device.type != kCUDA || !MarlinMoeEnabled()` gate exactly (CPU/unified
  // retains ⇒ policy false ⇒ materialize-all), now driven by platform data.
  if (!vllm::platforms::ShouldInterleaveLoadStream(
          vllm::platforms::GetPlatform(queue.device.type).residency_policy(),
          /*marlin_committed=*/MarlinMoeEnabled())) {
    // Not the committed Marlin path — no per-layer device build to interleave
    // with, so materialize any deferred experts to host for the wmma/CPU forward.
    MaterializeAllDeferredExperts(weights);
    return;
  }
  // The routed-expert host copies dominate 35B load-phase peak PSS. When the
  // loader deferred them (`load_layer_experts` set — the disk safetensors path),
  // materialize EACH layer's experts here, immediately before that layer's device
  // Marlin build + host free (BuildMoeMarlinResident frees the host bytes via
  // ReleaseHost once the repack Synchronizes). At most ONE layer's ~256 experts
  // coexist on the host, dropping whole-window peak host residency from all N
  // layers toward one. Byte-identical device residents: the source bytes and the
  // per-layer build order are unchanged — only the host-copy lifetime differs.
  const bool interleave = static_cast<bool>(weights.load_layer_experts);
  Dev d{vt::GetBackend(queue.device.type), queue};
  for (size_t li = 0; li < weights.layers.size(); ++li) {
    const MoeBlockWeights& moe = weights.layers[li].moe;
    if (interleave)
      weights.load_layer_experts(static_cast<int64_t>(li),
                                 const_cast<MoeBlockWeights&>(moe));
    if (!moe.expert_gate_fp4.empty())
      BuildMoeMarlinResident(d, moe, config, MoeMarlinResidentFor(&moe));
    if (SharedGateUpFusedEligible(moe.shared_gate_proj_fp4, moe.shared_up_proj_fp4)) {
      // Fused gate_up pair resident INSTEAD of the two singles (same total
      // bytes; the forward takes the fused path under the identical guard).
      BuildMarlinDensePairResident(d, moe.shared_gate_proj_fp4, moe.shared_up_proj_fp4,
                                   MarlinDensePairResidentFor(&moe.shared_gate_proj_fp4));
    } else {
      if (!moe.shared_gate_proj_fp4.Empty())
        BuildMarlinDenseResident(d, moe.shared_gate_proj_fp4,
                                 MarlinDenseResidentFor(&moe.shared_gate_proj_fp4));
      if (!moe.shared_up_proj_fp4.Empty())
        BuildMarlinDenseResident(d, moe.shared_up_proj_fp4,
                                 MarlinDenseResidentFor(&moe.shared_up_proj_fp4));
    }
    if (!moe.shared_down_proj_fp4.Empty())
      BuildMarlinDenseResident(d, moe.shared_down_proj_fp4,
                               MarlinDenseResidentFor(&moe.shared_down_proj_fp4));
  }
  if (!weights.lm_head_fp4.Empty())
    BuildMarlinDenseResident(d, weights.lm_head_fp4, MarlinDenseResidentFor(&weights.lm_head_fp4));
  // Every layer materialized + built + freed; drop the closure (returns the
  // mmap'd shards it kept alive) so nothing else can re-stream the experts.
  weights.load_layer_experts = nullptr;
  d.b.Synchronize(d.q);
#else
  (void)config;
  (void)queue;
  // No Marlin build in this configuration — the forward reads host expert bytes.
  MaterializeAllDeferredExperts(weights);
#endif
}

// PERF-27B-LMHEAD-FP4 (issue #213). Build whatever resident form of the PACKED
// dense head THIS backend's logits GEMM will actually consume, once, at prepare
// time. Inert on every BF16/FP8/GGUF/tied head (`lm_head_fp4` empty).
//
// CUDA/Marlin: prepare time is strictly BEFORE any decode-graph capture, and that
// matters — BuildMarlinDenseResident Allocs, launches the repack, and Copies a
// host float whose source is a function-local temporary. Same arm as
// Qwen3_5Model::PrepareMarlinResident's lm_head build above. A backend with NO
// fp4 GEMM builds the dequantized bf16 [K,N] operand here instead, so the head —
// the one weight that opted into it — never pays it on the forward path.
void Qwen3_5DenseModel::PrepareLmHeadResident(const Qwen3_5DenseWeights& weights,
                                              vt::Queue& queue) {
  if (weights.lm_head_fp4.Empty()) return;
  Dev d{vt::GetBackend(queue.device.type), queue};
  // Build under EXACTLY the guard MatmulNvfp4F32D uses to select the Marlin
  // GEMM, so a configuration that will not take that path never builds for it.
  // The build-time gate itself lives with the kernel family (see
  // BuildDenseHeadMarlinResident), not here.
  if (BuildDenseHeadMarlinResident(d, weights.lm_head_fp4)) return;
  // Same selection order as MatmulNvfp4F32D: the fp4-activation and packed-GEMM
  // arms stage the packed bytes lazily and NOT per call, so only the
  // dequantizing fallback needs eager work here.
  if (vllm::platforms::GetPlatform(queue.device.type).cutlass_fp4_supported() &&
      weights.lm_head_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
    return;
  }
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, queue.device.type)) return;
  (void)ResidentNvfp4DequantB(d, weights.lm_head_fp4);
  d.b.Synchronize(d.q);
}

// PERF-27B-GDN-FP8-QKVZ — build the merged FP8 [qkv;z] operand PRE-CAPTURE.
// Registered on the dense prepare hook, so it runs at model load, strictly
// before the first forward and therefore before any decode-graph capture: the
// alloc + two H2D copies can never land inside a stream capture. Skipping this
// leaves the forward's lazy build to run at first use, which is correct only
// because an eager warm step precedes capture — this makes it unconditional.
void Qwen3_5DenseModel::PrepareGdnFp8Resident(
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  if (!platforms::GetPlatform(queue.device.type).needs_weight_staging()) return;
  const int64_t key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
  const int64_t value_dim =
      config.linear_num_value_heads * config.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  Dev d{vt::GetBackend(queue.device.type), queue};
  bool built = false;
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    if (!layer.is_linear_attention) continue;
    if (!detail::ShouldUseMergedGdnFp8Qkvz(
            GdnMergedFp8QkvzEligibilityFor(d, layer.gdn, conv_dim, value_dim)))
      continue;
    (void)ResidentFp8Qkvz(d, layer.gdn);
    built = true;
  }
  if (built) d.b.Synchronize(d.q);
}

void Qwen3_5DenseModel::PrepareBf16Resident(
    const Qwen3_5DenseWeights& weights, vt::Queue& queue) {
  VT_CHECK(platforms::GetPlatform(queue.device.type).needs_weight_staging(),
           "PrepareBf16Resident: a weight-staging (device-resident) queue required");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const auto raw = [&d](const OwnedTensor& tensor) {
    if (!tensor.Empty()) {
      Tensor t = ResidentWeight(d, tensor);
      // W4d W6: stage the keep-quant word shadow HERE, at load, while the
      // banks are unfragmented — a lazy first-matmul stage at 27B asked the
      // head's 1 GiB words against a 100 MB largest free block and died.
      // No-op for non-block dtypes; the warm's later call hits the cache.
      vt::tenstorrent::StageKeepQuantWordsFor(t);
    }
  };
  const auto f32 = [&d](const OwnedTensor& tensor) {
    if (!tensor.Empty()) {
      const std::vector<int64_t> shape(tensor.shape,
                                       tensor.shape + tensor.rank);
      (void)ResidentWeightF32(d, tensor, shape);
    }
  };

  raw(weights.embed_tokens);
  raw(weights.final_norm);
  // PERF-27B-LMHEAD-FP4: already a no-op for a PACKED head (empty bf16 owner, and
  // IsPlainBf16Qwen3_5Dense is false whenever the head is packed); its resident is
  // built by PrepareLmHeadResident.
  raw(DenseLmHead(weights));
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    raw(layer.input_layernorm);
    raw(layer.post_attention_layernorm);
    if (layer.is_linear_attention) {
      const GdnLayerWeights& gdn = layer.gdn;
      raw(gdn.in_proj_qkv);
      raw(gdn.in_proj_z);
      raw(gdn.in_proj_qkvz);
      raw(gdn.in_proj_b);
      raw(gdn.in_proj_a);
      raw(gdn.in_proj_ba);
      raw(gdn.conv1d_weight);
      f32(gdn.conv1d_weight);
      raw(gdn.a_log);
      raw(gdn.dt_bias);
      raw(gdn.norm_weight);
      f32(gdn.norm_weight);
      raw(gdn.out_proj);
    } else {
      const FullAttnLayerWeights& attn = layer.attn;
      raw(attn.q_proj);
      raw(attn.k_proj);
      raw(attn.v_proj);
      raw(attn.o_proj);
      raw(attn.q_norm);
      raw(attn.k_norm);
      f32(attn.q_norm);
      f32(attn.k_norm);
    }
    raw(layer.mlp.gate_proj);
    raw(layer.mlp.up_proj);
    raw(layer.mlp.gate_up_proj);
    raw(layer.mlp.down_proj);
  }
}

std::vector<float> Qwen3_5Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                    gdn_state, weights, config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices);
  // dlogits is [n_out, vocab]: n_out == num_reqs when gathered (prefill/mixed),
  // else T. Download exactly the produced rows (the ONE host Download).
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3_5Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                    gdn_state, weights, config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices);
  // Keep the [n_out, vocab] logits ON DEVICE (no Download) — the sampler reads
  // them directly.
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

ForwardLogits Qwen3_5Model::ForwardDeviceTap(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, Qwen3_5MTPHiddenStates* hidden_out,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                    gdn_state, weights, config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  // Owning [T,H] bf16 buffer that the layer body fills via the inert tap hook;
  // moved into hidden_out so the drafter (I5d) can consume it with no re-forward.
  DBuf tap(d, DType::kBF16, {T, H});
  const Tensor tap_view = tap.t();
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices, &tap_view);
  if (hidden_out != nullptr) {
    hidden_out->tensor = tap.t();
    hidden_out->storage = tap.ReleaseShared();
  }
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

ForwardLogits Qwen3_5Model::ForwardDeviceMultiTap(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, Qwen3_5AuxTaps* aux_out,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                    gdn_state, weights, config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  if (aux_out == nullptr) {
    return ForwardDevice(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                         gdn_state, weights, config, queue, logits_indices);
  }
  ValidateAuxTapLayerIds(aux_out->layer_ids, config.num_hidden_layers);
  const int64_t taps = static_cast<int64_t>(aux_out->layer_ids.size());
  // Owning [T, H×taps] bf16 buffer, filled column-block per tap by the inert
  // aux-tap hook (MaybeCaptureAuxTap); moved into aux_out for the DFlash drafter.
  DBuf aux(d, DType::kBF16, {T, H * taps});
  const Tensor aux_view = aux.t();
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices,
                             /*hidden_tap=*/nullptr, &aux_out->layer_ids, &aux_view);
  aux_out->tensor = aux.t();
  aux_out->storage = aux.ReleaseShared();
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

std::vector<float> Qwen3_5Model::ForwardDense(const std::vector<int32_t>& token_ids,
                                              const std::vector<int32_t>& positions,
                                              const Qwen3_5MoeWeights& weights,
                                              const HfConfig& config,
                                              vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 forward: weights.layers size must equal num_hidden_layers");
  // ONE decode step. This forward does NOT go through ForwardLayers — it runs
  // its own unpaged layer loop — so it needs its own boundary, and every MoE
  // layer it runs takes expert slices (#1091 finding 3).
  const Qwen35ExpertStreamStep expert_stream_step;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, vocab, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, ActDType(d), {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
             positions, T, /*layer_index=*/l);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  // Final norm is GemmaRMSNorm too (weight applied as 1+w).
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head (the one host Download): fp4-resident (M2.2b) or bf16/GGUF path.
  DBuf dlogits = weights.lm_head_fp4.Empty()
                     ? MatmulF32D(d, dnorm.t(), weights.lm_head)
                     : MatmulNvfp4F32D(d, dnorm.t(), weights.lm_head_fp4);
  std::vector<float> logits(static_cast<size_t>(T) * vocab);
  dlogits.Download(d, logits.data());
  return logits;
}

std::vector<float> Qwen3_5Model::ForwardMoeHidden(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& positions,
    const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue) {
  // MODEL-XOR Phase 1: mirrors ForwardDense (MoE) through the embed + layer
  // stack + final GemmaRMSNorm, then downloads [T, H] f32 hidden states with
  // NO lm_head. Same as ForwardDenseHidden (dense) but calls RunLayer (MoE)
  // instead of RunDenseLayer.
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  VT_CHECK(T > 0, "qwen3_5 moe forward hidden: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 moe forward hidden: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 moe forward hidden: weights.layers size must equal num_hidden_layers");
  const Qwen35ExpertStreamStep expert_stream_step;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, config.vocab_size, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, ActDType(d), {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
             positions, T, /*layer_index=*/l);

  // Final RMSNorm over the fused stream, then download hidden as f32 (no lm_head).
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf dhidden_f32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, dhidden_f32.t(), dnorm.t());
  std::vector<float> hidden_f32(static_cast<size_t>(T) * static_cast<size_t>(H));
  dhidden_f32.Download(d, hidden_f32.data());
  return hidden_f32;
}

std::vector<float> Qwen3_5DenseModel::ForwardDense(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 dense forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense forward: weights.layers size must equal num_hidden_layers");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  // For a TEXT-only step the three mRoPE position streams are identical, so the
  // partial NeoX RoPE in FullAttnBlock degenerates to 1-D RoPE over `positions`
  // (notes §2). The vision tower / image-video merger are DEFERRED.
  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, vocab, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, ActDType(d), {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunDenseLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
                  positions, T);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head (the one host Download): PACKED NVFP4 (PERF-27B-LMHEAD-FP4) when the
  // checkpoint ships a ModelOpt/CT NVFP4 head, else the bf16/tied owner.
  DBuf dlogits = DenseLogitsF32D(d, dnorm.t(), weights);
  std::vector<float> logits(static_cast<size_t>(T) * vocab);
  dlogits.Download(d, logits.data());
  return logits;
}

std::vector<float> Qwen3_5DenseModel::ForwardDenseHidden(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  // MODEL-KEV Phase 5a: mirrors ForwardDense through the embed + layer stack +
  // final GemmaRMSNorm, then downloads [T, H] f32 hidden states with NO lm_head.
  // The kev PointerHead readout is host-side, so the one Download is f32 here.
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  VT_CHECK(T > 0, "qwen3_5 dense forward hidden: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense forward hidden: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense forward hidden: weights.layers size must equal num_hidden_layers");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, config.vocab_size, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, ActDType(d), {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    RunDenseLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
                  positions, T);
    if (std::getenv("VT_KEV_LAYER_DUMP")) {
      DBuf dh32(d, DType::kF32, {T, H});
      vt::CastF32(d.q, dh32.t(), hidden.t());
      std::vector<float> hf(static_cast<size_t>(T) * static_cast<size_t>(H));
      dh32.Download(d, hf.data());
      DBuf dr32(d, DType::kF32, {T, H});
      vt::CastF32(d.q, dr32.t(), res.t());
      std::vector<float> rf(static_cast<size_t>(T) * static_cast<size_t>(H));
      dr32.Download(d, rf.data());
      int dump_pos[] = {0, 3, static_cast<int>(T) - 1};
      for (int dp : dump_pos) {
        if (dp < 0 || dp >= T) continue;
        std::fprintf(stderr, "LAYER %2lld pos %3d:", (long long)l, dp);
        for (int i = 0; i < 10 && i < H; ++i) {
          size_t idx = static_cast<size_t>(dp) * static_cast<size_t>(H) +
                        static_cast<size_t>(i);
          std::fprintf(stderr, " %.6f", hf[idx] + rf[idx]);
        }
        std::fprintf(stderr, "\n");
      }
    }
  }

  // Final RMSNorm over the fused stream, then download hidden as f32 (no lm_head).
  // Mirrors the return_hidden branch in DenseForwardLayers (Phase 3).
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf dhidden_f32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, dhidden_f32.t(), dnorm.t());
  std::vector<float> hidden_f32(static_cast<size_t>(T) * static_cast<size_t>(H));
  dhidden_f32.Download(d, hidden_f32.data());
  return hidden_f32;
}

std::vector<float> Qwen3_5DenseModel::ForwardDenseLastLogits(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 dense forward last logits: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense forward last logits: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense forward last logits: weights.layers size must equal "
           "num_hidden_layers");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  Tensor dtab = Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, vocab, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, ActDType(d), {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunDenseLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
                  positions, T);

  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf dlast(d, ActDType(d), {1, H});
  GatherRows(d, dlast.ptr(), dnorm.t(), {static_cast<int32_t>(T - 1)}, H);
  DBuf dlogits = DenseLogitsF32D(d, dlast.t(), weights);
  std::vector<float> logits(static_cast<size_t>(vocab));
  dlogits.Download(d, logits.data());
  return logits;
}

Qwen3_5MTPModel::Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                                 const Qwen3_5DenseWeights& target,
                                 const HfConfig& config)
    : weights_(&weights),
      config_(&config),
      embed_tokens_(&target.embed_tokens),
      lm_head_(&DenseLmHead(target)),
      // PERF-27B-LMHEAD-FP4: the drafter shares the TARGET's head, so it must see
      // the packed one too. Empty on every BF16/FP8/tied dense target.
      lm_head_fp4_(&target.lm_head_fp4),
      // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): and the TRELLIS one, for the same
      // reason. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships no `lm_head.weight` at
      // all, so on that target `DenseLmHead(target)` above is EMPTY and this is
      // the only head there is. Empty on every other dense target.
      lm_head_exl3_(&target.lm_head_exl3) {
  VT_CHECK(weights.kind == Qwen3_5MTPKind::kDense,
           "qwen3_5 MTP: dense target requires dense MTP weights");
}

Qwen3_5MTPModel::Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                                 const Qwen3_5MoeWeights& target,
                                 const HfConfig& config)
    : weights_(&weights),
      config_(&config),
      embed_tokens_(&target.embed_tokens),
      lm_head_(&target.lm_head),
      lm_head_fp4_(&target.lm_head_fp4),
      // `Qwen3_5MoeWeights` has NO EXL3 head owner, so there is nothing to point
      // at. Said out loud rather than left to a default, and rather than adding a
      // field to the MoE container that no loader fills: no EXL3 MoE checkpoint
      // is in scope (.agents/specs/model-qwen35-exl3-head.md `## Owed`).
      lm_head_exl3_(nullptr) {
  VT_CHECK(weights.kind == Qwen3_5MTPKind::kMoe,
           "qwen3_5 MTP: MoE target requires MoE MTP weights");
}

Qwen3_5MTPHiddenStates Qwen3_5MTPModel::Forward(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions,
    const vt::Tensor& target_hidden_states, vt::Queue& queue,
    int64_t spec_step_idx) const {
  const int64_t tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = config_->hidden_size;
  const int64_t vocab_size = config_->vocab_size;
  const int64_t num_layers = weights_->NumLayers();
  VT_CHECK(tokens > 0, "qwen3_5 MTP forward: empty input_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == tokens,
           "qwen3_5 MTP forward: positions length must equal token count");
  VT_CHECK(spec_step_idx >= 0 && num_layers > 0,
           "qwen3_5 MTP forward: invalid spec step/layer count");
  VT_CHECK(target_hidden_states.rank == 2 &&
               target_hidden_states.shape[0] == tokens &&
               target_hidden_states.shape[1] == hidden_size &&
               target_hidden_states.dtype == DType::kBF16 &&
               target_hidden_states.IsContiguous() &&
               target_hidden_states.device == queue.device,
           "qwen3_5 MTP forward: target hidden states must be contiguous "
           "bf16 [T,H] on the queue device");
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): ONE precondition, two containers.
  // The trellis stores [K=2H, N=H] where the torch Linear stores [N=H, K=2H],
  // so the same projection is asserted through the orientation its own owner
  // uses. `IsExl3()` is the loader's answer, not a second derivation.
  VT_CHECK(weights_->IsExl3()
               ? (weights_->fc_exl3.InFeatures() == 2 * hidden_size &&
                  weights_->fc_exl3.OutFeatures() == hidden_size)
               : (weights_->fc.rank == 2 && weights_->fc.nk &&
                  weights_->fc.shape[0] == hidden_size &&
                  weights_->fc.shape[1] == 2 * hidden_size),
           "qwen3_5 MTP forward: fc must be raw bf16 [H,2H] or an EXL3 trellis "
           "[K=2H, N=H]");

  (void)vocab_size;
  // ONE decode step. A DRAFT forward is a complete forward with its own working
  // set: its slices are finished with when it returns, and leaving them pinned
  // across the target's forward would shrink the evictable set for the whole run
  // — F1 at draft scale (#1091 finding 3). A spec-decode iteration therefore
  // advances the clock once per draft plus once for the target, which is what
  // "one step is one forward" means for a draft+target pair.
  //
  // THAT PAIR IS RUN BY `ForwardPaged`, NOT BY THIS OVERLOAD. This one is
  // reached only through `ForwardLogitsHost`, a standalone parity convenience
  // (qwen3_5_mtp.h:135) whose every caller is under `tests/`, so the guard here
  // lands unreached and is recorded as a staged slice (#1108). It is kept
  // because the reasoning above is what makes it correct the moment this
  // overload gains a caller, and adding the guard later with the caller is how
  // this row lost its step boundary the first time.
  const Qwen35ExpertStreamStep expert_stream_step;
  Dev device{vt::GetBackend(queue.device.type), queue};

  // Qwen3_5MultiTokenPredictor.forward head: shared embedding + independent Gemma
  // RMSNorms + cat + fc (extracted into MtpHeadHidden, shared with ForwardPaged).
  DBuf hidden = MtpHeadHidden(device, *weights_, *config_, *embed_tokens_,
                              input_ids, target_hidden_states, tokens);
  DBuf residual(device, ResidualDType(device), {tokens, hidden_size});
  residual.Zero(device);
  const size_t layer_index =
      static_cast<size_t>(spec_step_idx % num_layers);
  if (weights_->kind == Qwen3_5MTPKind::kDense) {
    RunDenseLayer(device, weights_->dense_layers[layer_index], *config_, hidden,
                  residual, positions, tokens);
  } else {
    RunLayer(device, weights_->moe_layers[layer_index], *config_, hidden,
             residual, positions, tokens, layer_index);
  }
  return MtpFinalize(device, *weights_, *config_, hidden, residual, tokens);
}

Qwen3_5MTPHiddenStates Qwen3_5MTPModel::ForwardPaged(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions,
    const vt::Tensor& target_hidden_states,
    const v1::CommonAttentionMetadata& attn_meta, PagedKvCache& draft_kv,
    vt::Queue& queue, int64_t spec_step_idx) const {
  const int64_t tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = config_->hidden_size;
  const int64_t num_layers = weights_->NumLayers();
  VT_CHECK(tokens > 0, "qwen3_5 MTP paged forward: empty input_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == tokens,
           "qwen3_5 MTP paged forward: positions length must equal token count");
  VT_CHECK(spec_step_idx >= 0 && num_layers > 0,
           "qwen3_5 MTP paged forward: invalid spec step/layer count");
  VT_CHECK(target_hidden_states.rank == 2 &&
               target_hidden_states.shape[0] == tokens &&
               target_hidden_states.shape[1] == hidden_size &&
               target_hidden_states.dtype == DType::kBF16 &&
               target_hidden_states.IsContiguous() &&
               target_hidden_states.device == queue.device,
           "qwen3_5 MTP paged forward: target hidden states must be contiguous "
           "bf16 [T,H] on the queue device");
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): ONE precondition, two containers.
  // The trellis stores [K=2H, N=H] where the torch Linear stores [N=H, K=2H],
  // so the same projection is asserted through the orientation its own owner
  // uses. `IsExl3()` is the loader's answer, not a second derivation.
  VT_CHECK(weights_->IsExl3()
               ? (weights_->fc_exl3.InFeatures() == 2 * hidden_size &&
                  weights_->fc_exl3.OutFeatures() == hidden_size)
               : (weights_->fc.rank == 2 && weights_->fc.nk &&
                  weights_->fc.shape[0] == hidden_size &&
                  weights_->fc.shape[1] == 2 * hidden_size),
           "qwen3_5 MTP paged forward: fc must be raw bf16 [H,2H] or an EXL3 trellis "
           "[K=2H, N=H]");
  VT_CHECK(attn_meta.num_actual_tokens == tokens,
           "qwen3_5 MTP paged forward: attn metadata token count must equal T");
  VT_CHECK(draft_kv.num_kv_heads == config_->num_key_value_heads &&
               draft_kv.head_size == config_->head_dim,
           "qwen3_5 MTP paged forward: draft KV cache dims mismatch config");

  // ONE decode step, for the same reason as the unpaged draft forward above.
  const Qwen35ExpertStreamStep expert_stream_step;
  Dev device{vt::GetBackend(queue.device.type), queue};

  // Same head math as Forward; the difference is the DECODER LAYER, which runs
  // paged (writes/reads the draft KV layer via slot_mapping/block_table).
  DBuf hidden = MtpHeadHidden(device, *weights_, *config_, *embed_tokens_,
                              input_ids, target_hidden_states, tokens);
  DBuf residual(device, ResidualDType(device), {tokens, hidden_size});
  residual.Zero(device);

  // Per-step device inputs for the single full-attention layer (no GDN state) +
  // the fused-preamble cos|sin cache (a stub unless VT_FUSE_ATTN_PREAMBLE). The
  // MTP head is bf16-unquantized, so the fp4 attn preamble default is OFF.
  StepDevInputs sdi = BuildFullAttnStepDevInputs(device, positions, attn_meta);
  MaybeBuildAttnCosSin(device, sdi, *config_, tokens, /*fp4_attn=*/false);

  // The MTP layer is always layer_type="full_attention": gdn_meta/gdn_state are
  // never touched by RunDense/RunLayerPaged for a non-linear-attention layer.
  const v1::GDNAttentionMetadata gdn_meta_unused;
  const size_t layer_index = static_cast<size_t>(spec_step_idx % num_layers);
  if (weights_->kind == Qwen3_5MTPKind::kDense) {
    RunDenseLayerPaged(device, weights_->dense_layers[layer_index], *config_,
                       hidden, residual, sdi, attn_meta, gdn_meta_unused,
                       &draft_kv, /*gdn_state=*/nullptr, tokens,
                       static_cast<int64_t>(layer_index));
  } else {
    RunLayerPaged(device, weights_->moe_layers[layer_index], *config_, hidden,
                  residual, sdi, attn_meta, gdn_meta_unused, &draft_kv,
                  /*gdn_state=*/nullptr, tokens, layer_index);
  }
  return MtpFinalize(device, *weights_, *config_, hidden, residual, tokens);
}

// SPEC-MTP-K-GT-1 (#81): the device row gather the multi-step draft needs
// between its prefill and its first decode step — the mirror of
// `self.hidden_states[:num_reqs] = hidden_states[last_token_indices]`
// (autoregressive/speculator.py:367-371). Row-by-row device-to-device copies,
// the same idiom MtpHeadHidden uses for torch.cat, so the whole hidden carry
// stays on the queue's device and never round-trips the host.
Qwen3_5MTPHiddenStates Qwen3_5MTPModel::GatherHiddenRows(
    const vt::Tensor& hidden_states, const std::vector<int64_t>& rows,
    vt::Queue& queue) const {
  const int64_t hidden_size = config_->hidden_size;
  const int64_t num_rows = static_cast<int64_t>(rows.size());
  VT_CHECK(num_rows > 0, "qwen3_5 MTP gather: empty row list");
  VT_CHECK(hidden_states.rank == 2 && hidden_states.shape[1] == hidden_size &&
               hidden_states.dtype == DType::kBF16 &&
               hidden_states.IsContiguous() &&
               hidden_states.device == queue.device,
           "qwen3_5 MTP gather: hidden states must be contiguous bf16 [T,H] on "
           "the queue device");
  const int64_t tokens = hidden_states.shape[0];

  Dev device{vt::GetBackend(queue.device.type), queue};
  DBuf out(device, DType::kBF16, {num_rows, hidden_size});
  const size_t row_bytes =
      static_cast<size_t>(hidden_size) * vt::SizeOf(DType::kBF16);
  auto* dst = static_cast<uint8_t*>(out.ptr());
  const auto* src = static_cast<const uint8_t*>(hidden_states.data);
  for (int64_t i = 0; i < num_rows; ++i) {
    const int64_t row = rows[static_cast<size_t>(i)];
    VT_CHECK(row >= 0 && row < tokens,
             "qwen3_5 MTP gather: row index out of range");
    device.b.Copy(device.q, dst + static_cast<size_t>(i) * row_bytes,
                  src + static_cast<size_t>(row) * row_bytes, row_bytes);
  }
  Qwen3_5MTPHiddenStates gathered;
  gathered.tensor = out.t();
  gathered.storage = out.ReleaseShared();
  return gathered;
}

ForwardLogits Qwen3_5MTPModel::ComputeLogits(
    const vt::Tensor& hidden_states, vt::Queue& queue) const {
  const int64_t hidden_size = config_->hidden_size;
  VT_CHECK(hidden_states.rank == 2 &&
               hidden_states.shape[1] == hidden_size &&
               hidden_states.dtype == DType::kBF16 &&
               hidden_states.IsContiguous() &&
               hidden_states.device == queue.device,
           "qwen3_5 MTP logits: hidden states must be contiguous bf16 [T,H] "
           "on the queue device");
  Dev device{vt::GetBackend(queue.device.type), queue};
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5). The arms are ordered
  // `exl3 -> fp4 -> bf16`, which is `DenseLogitsF32D`'s order and
  // `DflashLogitsF32D`'s, so the three readers of ONE target head cannot
  // disagree about precedence. The trellis head is COMPUTED WITH, never widened:
  // a dequantized copy of the real 248320x5120 head is 2.543 GB, and upstream
  // reaches the packed path without a branch because `_apply_head` calls
  // `lm_head.quant_method.apply` (logits_processor.py:132-142).
  DBuf logits =
      (lm_head_exl3_ != nullptr && !lm_head_exl3_->Empty())
          ? dense_exl3::Linear(device, hidden_states, *lm_head_, *lm_head_exl3_,
                               DType::kF32)
          : ((lm_head_fp4_ != nullptr && !lm_head_fp4_->Empty())
                 ? MatmulNvfp4F32D(device, hidden_states, *lm_head_fp4_)
                 : MatmulF32D(device, hidden_states, *lm_head_));
  return WrapDeviceLogits(device, std::move(logits), config_->vocab_size);
}

std::vector<float> Qwen3_5MTPModel::ForwardLogitsHost(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions,
    const vt::Tensor& target_hidden_states, vt::Queue& queue,
    int64_t spec_step_idx) const {
  Qwen3_5MTPHiddenStates hidden =
      Forward(input_ids, positions, target_hidden_states, queue, spec_step_idx);
  ForwardLogits logits = ComputeLogits(hidden.tensor, queue);
  std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
  Backend& backend = vt::GetBackend(queue.device.type);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);
  return host;
}

// Shared shape/count validation for the dense paged forward entry points (the
// 27B analogue of CheckPagedForward). Same contract, over the dense weights.
static void CheckDensePagedForward(const std::vector<int32_t>& token_ids,
                                   const std::vector<int32_t>& positions,
                                   const CommonAttentionMetadata& attn_meta,
                                   const GDNAttentionMetadata& gdn_meta,
                                   const std::vector<PagedKvCache>& attn_kv,
                                   const std::vector<GdnStateCache>& gdn_state,
                                   const Qwen3_5DenseWeights& weights,
                                   const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "qwen3_5 dense paged forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense paged forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense paged forward: weights.layers size must equal "
           "num_hidden_layers");
  VT_CHECK(attn_meta.num_actual_tokens == T,
           "qwen3_5 dense paged forward: attn_meta.num_actual_tokens must equal T");
  int64_t n_full = 0, n_gdn = 0;
  for (const auto& l : weights.layers) (l.is_linear_attention ? n_gdn : n_full) += 1;
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == n_full,
           "qwen3_5 dense paged forward: attn_kv count must equal full-attn layers");
  VT_CHECK(static_cast<int64_t>(gdn_state.size()) == n_gdn,
           "qwen3_5 dense paged forward: gdn_state count must equal GDN layers");
  const int64_t state_slots =
      detail::ValidateGdnStateCacheLayout(gdn_state);
  if (n_gdn > 0) {
    VT_CHECK(gdn_meta.num_actual_tokens == T,
             "qwen3_5 dense paged forward: gdn_meta.num_actual_tokens must equal T");
    detail::ValidateGdnAttentionMetadata(
        gdn_meta, state_slots, /*allow_inert_padding=*/false);
  }
}

// Dense embed (27B): hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident
// table). The 27B analogue of EmbedInto — KEPT OUTSIDE THE CUDA-GRAPH for the
// same reasons (the Embedding op allocs a bounds-check flag + syncs, and it
// consumes host token_ids). The dense-graph driver runs this per step into its
// PERSISTENT hidden buffer, then captures/replays DenseForwardLayers over that
// fixed address. Text-only: the three mRoPE streams coincide so the partial NeoX
// RoPE degenerates to 1-D RoPE over `positions` (notes §2); vision tower DEFERRED.
static void DenseEmbedInto(Dev d, DBuf& hidden,
                           const std::vector<int32_t>& token_ids,
                           const Qwen3_5DenseWeights& weights,
                           const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  Tensor dtab =
      Qwen3_5EmbeddingTable(d.b, d.q, weights.embed_tokens, vocab, H);
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  ApplyDeviceTokenIdsOverride(d, dids, T);
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): the dense captured step's host
// phase clock (.agents/specs/tenstorrent-27b-step-decompose.md). READ-ONLY —
// it brackets existing calls and prints one stderr line per step; with the
// knob unset every check below is a getenv compare and nothing else runs.
// The line decomposes the step the driver can see: warmup refreshes (does
// each step re-warm, and how many warm passes does it make), embed, the
// capture pass (begin/body/end, the tt-metal trace build), the replay launch
// — plus the two things around it: `gap_prev` (the runner/sampler window
// between the previous step's return and this entry, which the seam-side
// `sync` lines further split into the blocking completion read and host
// residue) and `dram_free` (the per-step retention read-out, the
// ~950 MB/request staircase tt-metal#57970 names). `boundary=1` marks a
// request's first decode step (the seq-continuation predicate), which is the
// pressure-axis join: request 1's steps vs request 4's.
namespace {
using StepPhaseClk = std::chrono::steady_clock;
inline double StepPhaseMsOf(StepPhaseClk::time_point a, StepPhaseClk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
struct StepPhaseTrace {
  bool on = false;
  StepPhaseClk::time_point t0{};  // Step() entry
  double warms_ms = 0, embed_ms = 0, pregrow_ms = 0, cap_begin_ms = 0,
         cap_body_ms = 0, cap_end_ms = 0, pin_ms = 0, replay_ms = 0,
         body_ms = 0;
  int64_t warm_calls = 0;
  bool boundary = false;
  int64_t B = 0, S = 0;
  int64_t dram_free = -1;
};
// One line per step, emitted at every return the captured arm takes. The
// `gap_prev` half of the wall lives in the CALLER's window (sampler + runner
// between this step's return and the next entry), so the sum check closes as
// step wall + gap + the seam's sync (gap + wait) lines with no hidden residue.
void StepPhaseEmit(const StepPhaseTrace& p, const char* kind) {
  static int64_t n = 0;
  static StepPhaseClk::time_point prev_return{};
  const auto now = StepPhaseClk::now();
  double gap_prev_ms = -1.0;
  if (prev_return.time_since_epoch().count() != 0)
    gap_prev_ms = StepPhaseMsOf(prev_return, p.t0);
  prev_return = now;
  ++n;
  std::fprintf(stderr,
               "[TT-STEP-PHASE] step=%lld kind=%s B=%lld S=%lld boundary=%d "
               "warm_calls=%lld dram_free_mib=%lld wall_ms=%.1f warms_ms=%.1f "
               "embed_ms=%.1f pregrow_ms=%.1f cap_begin_ms=%.1f "
               "cap_body_ms=%.1f cap_end_ms=%.1f pin_ms=%.1f replay_ms=%.1f "
               "body_ms=%.1f gap_prev_ms=%.1f\n",
               static_cast<long long>(n), kind, static_cast<long long>(p.B),
               static_cast<long long>(p.S), p.boundary ? 1 : 0,
               static_cast<long long>(p.warm_calls),
               static_cast<long long>(p.dram_free),
               StepPhaseMsOf(p.t0, now), p.warms_ms, p.embed_ms, p.pregrow_ms,
               p.cap_begin_ms, p.cap_body_ms, p.cap_end_ms, p.pin_ms,
               p.replay_ms, p.body_ms, gap_prev_ms);
}
}  // namespace

// The CAPTURABLE dense paged forward region (27B): everything AFTER the embedding
// — the residual stream (res=0), the N paged dense decoder layers, the final
// RMSNorm and the bf16 lm_head — returning the [n_out,vocab] f32 logits as a
// device DBuf (NO host Download). The 27B analogue of ForwardLayers; split out so
// the exact op sequence is what the graph captures/replays. `hidden_in` is the
// embedded input (a view over the graph's persistent hidden buffer on the replay
// path); it is COPIED into a working buffer so the layers' in-place residual
// thread never disturbs the persistent embedding. All per-step-varying inputs are
// read from HOST vector args (positions / metadata), whose host->device copies
// are capturable on GB10; the driver keeps them persistent + mutates in place.
// Every per-call scratch is pool-backed (DevicePool) or resident/StreamScratch-
// pooled (the cutlass/emulation fp4 GEMMs, cublas lm_head) so a cold pre-warm at
// this size makes the capture region do ZERO cudaMalloc.
static DBuf DenseForwardLayers(Dev d, const Tensor& hidden_in,
                               const std::vector<int32_t>& positions,
                               const CommonAttentionMetadata& attn_meta,
                               const GDNAttentionMetadata& gdn_meta,
                               const std::vector<PagedKvCache>& attn_kv,
                               const std::vector<GdnStateCache>& gdn_state,
                               const Qwen3_5DenseWeights& weights,
                               const HfConfig& config,
                               const std::vector<int32_t>& logits_indices = {},
                               const Tensor* hidden_tap = nullptr,
                               const std::vector<float>* mrope_cos_sin = nullptr,
                               const std::vector<int32_t>* aux_layer_ids = nullptr,
                               const Tensor* aux_out = nullptr,
                               bool return_hidden = false,
                               StepDevInputs* persistent_sdi = nullptr) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-FWD] DenseForwardLayers T=%lld n_idx=%zu\n",
                 static_cast<long long>(T), logits_indices.size());
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy of the embedded hidden (device->device; captured). RunDenseLayer
  // Paged reassigns `hidden` per layer, so this must NOT alias the persistent buf.
  DBuf hidden(d, ActDType(d), {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, ResidualDType(d), {T, H});
  res.Zero(d);

  // Per-step inputs uploaded ONCE (see StepDevInputs) — no per-layer re-upload.
  // VT_ASYNC_EXECUTOR (Option A): a non-null persistent_sdi points at the SizeSlot's
  // persistent device buffers the graph driver stages the H2D into OUTSIDE this
  // captured region (see ForwardLayers for the full note); here we only re-derive
  // cos|sin from the re-staged persistent positions.
  const int64_t gdn_state_slots =
      gdn_state.empty() ? 0 : gdn_state.front().ssm_state.shape[0];
  std::optional<StepDevInputs> local_sdi;
  if (persistent_sdi == nullptr)
    local_sdi.emplace(HasGdnConsumers(weights)
                          ? BuildStepDevInputs(d, positions, attn_meta, gdn_meta,
                                               gdn_state_slots)
                          : BuildFullAttnStepDevInputs(d, positions, attn_meta));
  StepDevInputs& sdi = persistent_sdi != nullptr ? *persistent_sdi : *local_sdi;
  // Build the fused-preamble cos|sin cache ONCE; fp4_attn keys the per-arch
  // default (the real 27B W4A4 => ON; bf16/GGUF dense => OFF; env overrides).
  const bool fp4_attn = [&] {
    for (const auto& l : weights.layers)
      if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
    return false;
  }();
  // M3-b VL path: a prebuilt MRoPE cos|sin cache [T, rotary_dim] (host f32,
  // interleaved 3-section selection already baked in) is injected verbatim into
  // the fused full-attn preamble, replacing the 1-D RoPE cache MaybeBuildAttnCosSin
  // would build. FuseAttnPreamble is default-ON (VT_FUSE_ATTN_PREAMBLE) so
  // AttnQkNormRopeGate reads this per-token cache; sdi.positions is unused for rope
  // then. mrope_cos_sin==nullptr (every text caller) ⇒ byte-identical to before.
  if (persistent_sdi != nullptr) {
    // Option A: the persistent cos|sin buffer was allocated pre-capture; re-fill only
    // (captured) so every replay re-derives rope from the freshly-staged positions.
    // The MRoPE VL path never drives the decode graph (persistent_sdi is nullptr
    // there), so text-decode's 1-D RoPE re-fill is the only persistent case.
    if (sdi.has_attn_cos_sin) FillAttnCosSin(d, sdi, config);
  } else if (mrope_cos_sin != nullptr) {
    const int rot = static_cast<int>(config.rotary_dim);
    VT_CHECK(rot > 0 && static_cast<int64_t>(mrope_cos_sin->size()) == T * rot,
             "qwen3_5 VL: MRoPE cos|sin cache must be [T, rotary_dim]");
    sdi.attn_cos_sin = DBuf(d, DType::kF32, {T, rot}, mrope_cos_sin->data());
    sdi.has_attn_cos_sin = true;
  } else {
    MaybeBuildAttnCosSin(d, sdi, config, T, fp4_attn);
  }

  // N paged decoder layers: full-attn layers read/write attn_kv[fa_idx], GDN
  // layers the persistent gdn_state[gdn_idx] (same layer-order indexing as the
  // 35B paged forward).
  // ROCM-TIER-DIVERGENCE (#2590). One forward-call ordinal for every dump this
  // step writes, so two runs are joined on a POSITION IN THE MODEL rather than
  // on an invocation counter. -1, and every dump below inert, unless a knob is
  // set.
  const int64_t dump_step = actdump::BeginStep();
  if (dump_step >= 0) {
    actdump::NarrateOnce(d.q.device.type, "Qwen3_5DenseModel",
                         config.num_hidden_layers, T, H);
  }
  // The POST-EMBEDDING snapshot, layer -1. The dense path never had one, so a
  // cross-tier profile could not tell "the two tiers embed the token
  // differently" from "layer 0 computes differently" -- and those two are a live
  // pair here, because `token_embd` keeps its Q4_K blocks on the CPU tier and
  // expands to bf16 on ROCm (gguf_keep_quant.cpp `DeviceQuantGatherSupported`).
  // The MoE sibling has carried this snapshot since the Tenstorrent bisect.
  ActDumpStream(d, dump_step, -1, hidden, res, T, H);

  // TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): the per-layer sampling probe
  // over the 64 layers — host record/enqueue cost per layer always, and with
  // VT_TT_STEP_PHASES=sync a blocking queue drain (StepPhaseSyncProbe) after
  // each layer so the line carries that layer's true DEVICE time. The drain
  // is INERT while a trace capture is open (a readback inside the captured
  // region is prohibited; the capture pass then reports record cost only)
  // and on every backend the knob does not enable. Zero cost when unset.
  const bool sph_layer = vt::tenstorrent::StepPhasesEnabled();
  const bool sph_layer_sync =
      sph_layer && d.q.device.type == vt::DeviceType::kTENSTORRENT &&
      std::strcmp(vt::tenstorrent::StepPhaseMode(), "sync") == 0;
  int64_t fa_idx = 0, gdn_idx = 0;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3_5DenseLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    const PagedKvCache* kv =
        layer.is_linear_attention ? nullptr : &attn_kv[static_cast<size_t>(fa_idx++)];
    const GdnStateCache* gs =
        layer.is_linear_attention ? &gdn_state[static_cast<size_t>(gdn_idx++)] : nullptr;
    const StepPhaseClk::time_point sph_l0 =
        sph_layer ? StepPhaseClk::now() : StepPhaseClk::time_point{};
#ifdef VLLM_CPP_TENSTORRENT
    // W4d (#3042) attribution: the chunk-loop ledger brackets only the
    // keep-quant webs; 13 GB of the 27B first pass lands in the spans this
    // pair now brackets (attention, GDN, norms, rolls).
    char tt_lbl[48];
    if (vt::tenstorrent::DeviceAvailable()) {
      std::snprintf(tt_lbl, sizeof tt_lbl, "block/%lld/pre",
                    static_cast<long long>(l));
      vt::tenstorrent::AllocTraceSnapshot(vt::tenstorrent::SharedMeshDevice(),
                                          tt_lbl);
    }
#endif
    RunDenseLayerPaged(d, layer, config, hidden, res, sdi, attn_meta,
                       gdn_meta, kv, gs, T, l);
    if (sph_layer) {
      const double host_ms = StepPhaseMsOf(sph_l0, StepPhaseClk::now());
      double dev_ms = -1.0;
      if (sph_layer_sync && !vt::tenstorrent::TraceCaptureActive()) {
        const StepPhaseClk::time_point sph_s0 = StepPhaseClk::now();
        vt::tenstorrent::StepPhaseSyncProbe();
        dev_ms = StepPhaseMsOf(sph_s0, StepPhaseClk::now());
      }
      std::fprintf(stderr,
                   "[TT-STEP-PHASE] layer=%lld type=%s host_ms=%.3f dev_ms=%.3f\n",
                   static_cast<long long>(l),
                   layer.is_linear_attention ? "gdn" : "fa", host_ms, dev_ms);
    }
#ifdef VLLM_CPP_TENSTORRENT
    if (vt::tenstorrent::DeviceAvailable()) {
      std::snprintf(tt_lbl, sizeof tt_lbl, "block/%lld/post",
                    static_cast<long long>(l));
      vt::tenstorrent::AllocTraceSnapshot(vt::tenstorrent::SharedMeshDevice(),
                                          tt_lbl);
      if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)  // W4d: census spam only on trace
        vt::tenstorrent::DumpSlotCensus(tt_lbl);
    }
#endif
    // DFlash DF-AUX-TAPS: capture (hidden+res) at configured boundaries. Inert
    // (no-op) when aux_out is null — every non-DFlash caller.
    MaybeCaptureAuxTap(d, l, aux_layer_ids, aux_out, hidden.t(), res.t(), T, H);
    // tt-27b-region-capture: ONE REGION PER LAYER. The bare break splits the
    // kPiecewise scope into a new segment with NO eager call and NO
    // destination — the region boundary is a pure capture split, and the
    // handoff is the in-place one: hidden/res are pool-backed buffers whose
    // captured addresses the #2274 pinning holds for the graph's life, and
    // the GDN ssm/conv + KV state slots are persistent shadows committed IN
    // PLACE (tenstorrent_gdn.cpp's W3 discipline). Inert (a counter tick)
    // in every kFull scope and every eager call — byte-identical to today.
    vt::GraphBreak();
    // VT_DUMP_ACT (issue #41, ROCm 0.8B forward-divergence fix spike W1; keyed
    // and completed for #2590): dump the residual stream after each layer.
    //
    // BOTH HALVES, because in this model neither one is the hidden state. The
    // stream is carried as a pair -- `res` accumulates and `hidden` holds the
    // layer's MLP delta -- and they are summed only at the final norm, which is
    // also what `MaybeCaptureAuxTap` above computes. The predecessor hook wrote
    // `hidden` alone and called it "the residual stream"; it was the MLP delta,
    // and half a stream cannot be compared against another engine or another
    // tier.
    //
    // A debug hook, never the hot path: the Download forces a sync, so
    // capture-graph paths must not set the env.
    ActDumpStream(d, dump_step, l, hidden, res, T, H);
  }
  // Two stream halves per layer, plus the pre-layer snapshot. Zero due when the
  // stream knob is not the one that is set. This path also carries the
  // sub-stage probes, so SOMETHING is always due here whichever knob it was.
  actdump::EndStep(dump_step,
                   actdump::StreamDir() == nullptr
                       ? 0
                       : 2 * (config.num_hidden_layers + 1),
                   /*any_due=*/1);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, ActDType(d), {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // Hidden-state tap (SPEC-MTP I5c): the full [T,H] post-final-norm hidden the MTP
  // drafter consumes; captured before the gather. INERT (nullptr on every shipped
  // caller — only ForwardDeviceTap passes a buffer). See the 35B ForwardLayers tap.
  if (hidden_tap != nullptr) {
    VT_CHECK(hidden_tap->shape[0] == T && hidden_tap->shape[1] == H &&
                 hidden_tap->dtype == DType::kBF16,
             "qwen3_5 dense: hidden tap buffer must be bf16 [T,H]");
    d.b.Copy(d.q, hidden_tap->data, dnorm.t().data,
             static_cast<size_t>(T) * static_cast<size_t>(H) *
                 vt::SizeOf(DType::kBF16));
  }

  // MODEL-KEV Phase 3: the pooling tail — stop after the final GemmaRMSNorm
  // (+ optional gather) and return the [n_out, H] post-final-norm hidden rows
  // upcast to f32, with NO lm_head. Mirrors Qwen3's return_hidden branch
  // (qwen3.cpp:347-351). Never taken by any text caller (defaults false).
  if (return_hidden) {
    const bool do_gather = !logits_indices.empty() &&
                           static_cast<int64_t>(logits_indices.size()) < T;
    Tensor src = dnorm.t();
    if (do_gather) {
      const int64_t n_out = static_cast<int64_t>(logits_indices.size());
      DBuf dgather(d, DType::kBF16, {n_out, H});
      GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
      src = dgather.t();
    }
    const int64_t n_out = src.shape[0];
    DBuf dhid(d, DType::kF32, {n_out, H});
    vt::CastF32(d.q, dhid.t(), src);
    return dhid;
  }

  // Logits gather-before-lm_head (prefill/mixed): same semantics as the 35B path.
  // Both arms route through DenseLogitsF32D (PERF-27B-LMHEAD-FP4). Pure-decode /
  // graph replay pass empty indices (identity) → the full [T,vocab] path.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  if (do_gather) {
    const int64_t n_out = static_cast<int64_t>(logits_indices.size());
    DBuf dgather(d, DType::kBF16, {n_out, H});
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    return DenseLogitsF32D(d, dgather.t(), weights);
  }
  return DenseLogitsF32D(d, dnorm.t(), weights);
}

// W4d (tenstorrent-27b-decode-shape-capture.md): per-sequence LAST-TOKEN
// logits indices for a mixed prefill step, the vLLM logits_processor
// semantics — generation samples only the last scheduled token per sequence
// (prepare_inputs.cpp:210-221: `logits_indices[i] = query_start_loc[i+1]-1`).
// Empty (identity, full [T,vocab] head) when the metadata cannot express the
// gather or it would be a no-op: a pure-decode step has size == T, and the
// graph-replay/capture contract passes {} by design. The driver's eager
// fallback feeds these to the forward so the gather-before-lm_head fires and
// the head allocates [num_seqs, vocab], never [T, vocab].
static std::vector<int32_t> LastTokenLogitsIndices(
    const v1::CommonAttentionMetadata& am, int64_t T) {
  const int64_t n =
      static_cast<int64_t>(am.query_start_loc.size()) - 1;
  if (n <= 0 || n >= T) return {};
  std::vector<int32_t> indices(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    indices[static_cast<size_t>(i)] = am.query_start_loc[static_cast<size_t>(i) + 1] - 1;
  }
  return indices;
}

// Full eager dense paged forward body: embed (host token_ids) then the capturable
// dense layer region. Used by Qwen3_5DenseModel::Forward/ForwardDevice and the
// dense-graph driver's eager fallback / cold-shape pre-warm step (one contiguous
// stream, no capture). Returns [n_out,vocab] f32 (n_out == num_reqs when gathered,
// else T). Shared op sequence with the graph so eager output == replay output.
static DBuf DenseForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                             const std::vector<int32_t>& positions,
                             const CommonAttentionMetadata& attn_meta,
                             const GDNAttentionMetadata& gdn_meta,
                             const std::vector<PagedKvCache>& attn_kv,
                             const std::vector<GdnStateCache>& gdn_state,
                             const Qwen3_5DenseWeights& weights,
                             const HfConfig& config,
                             const std::vector<int32_t>& logits_indices,
                             const Tensor* hidden_tap = nullptr,
                             const std::vector<int32_t>* aux_layer_ids = nullptr,
                             const Tensor* aux_out = nullptr,
                             bool return_hidden = false) {
  CheckDensePagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                         gdn_state, weights, config);
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-FWD] DenseForwardBody T=%lld H=%lld num_reqs=%lld "
                 "logits_indices=%zu\n",
                 static_cast<long long>(T), static_cast<long long>(H),
                 static_cast<long long>(attn_meta.num_reqs),
                 logits_indices.size());
  DBuf hidden(d, ActDType(d), {T, H});
  DenseEmbedInto(d, hidden, token_ids, weights, config);
  return DenseForwardLayers(d, hidden.t(), positions, attn_meta, gdn_meta, attn_kv,
                            gdn_state, weights, config, logits_indices, hidden_tap,
                            /*mrope_cos_sin=*/nullptr, aux_layer_ids, aux_out,
                            /*return_hidden=*/return_hidden);
}

std::vector<float> Qwen3_5DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3_5DenseModel::ForwardHidden(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  // MODEL-KEV Phase 3: the pooling forward — the same embed + layer stack as
  // Forward, stopping after the final GemmaRMSNorm (+ gather) with NO lm_head,
  // mirroring Qwen3DenseModel::ForwardHidden (qwen3.cpp:570-592). The
  // [n_out, H] f32 rows are downloaded to the host carrier for the kev
  // PointerHead readout, whose ops are host-side.
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dhidden = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices, /*hidden_tap=*/nullptr,
                                  /*aux_layer_ids=*/nullptr, /*aux_out=*/nullptr,
                                  /*return_hidden=*/true);
  const int64_t n_out = dhidden.t().shape[0];
  const int64_t H = config.hidden_size;
  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = H;
  fl.host.resize(static_cast<size_t>(n_out) * static_cast<size_t>(H));
  dhidden.Download(d, fl.host.data());
  return fl;
}

// ── M3-b: the Qwen3.6-27B GDN-hybrid VL image->text greedy driver. ──────────
// Builds the per-token MRoPE cos|sin cache [T, rotary_dim] (host f32) from the
// 3-D positions [3,T] + config.mrope_section, mirroring RopeCosSinCacheKernel's
// angle math (cpu_ops.cpp:812) and MropeAxisForPair's interleaved 3-section axis
// selection (cpu_ops.cpp:731) exactly, so the fused AttnQkNormRopeGate applies
// true MRoPE by reading this cache row-per-token (the text path bakes 1-D RoPE
// into the same cache). No Llama3 freq scaling (mrope rope_type ⇒ identity).
// EXTERNAL LINKAGE (W5d-2, #2249): Qwen4-Exp's QSA block builds the SAME
// tables from another TU. Declared in qwen3_5_mrope.h; body unchanged.
std::vector<float> BuildMropeCosSinHost(
    const std::vector<int32_t>& positions3, int64_t T, const HfConfig& config) {
  const int rot = static_cast<int>(config.rotary_dim);
  VT_CHECK(rot > 0, "qwen3_5 VL: rotary_dim must be > 0");
  const int64_t half = rot / 2;
  const double base = config.rope_theta;
  const bool interleaved = config.rope_parameters.mrope_interleaved;
  const std::vector<int64_t>& sec = config.rope_parameters.mrope_section;
  VT_CHECK(sec.size() == 3 && sec[0] + sec[1] + sec[2] == half,
           "qwen3_5 VL: mrope_section must be 3 entries summing to rotary_dim/2");
  std::vector<float> cache(static_cast<size_t>(T * rot));
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t pair = 0; pair < half; ++pair) {
      int axis;
      if (interleaved) {
        if (pair % 3 == 1 && pair <= 3LL * sec[1]) {
          axis = 1;
        } else if (pair % 3 == 2 && pair <= 3LL * sec[2]) {
          axis = 2;
        } else {
          axis = 0;
        }
      } else {
        if (pair < sec[0]) {
          axis = 0;
        } else if (pair < sec[0] + sec[1]) {
          axis = 1;
        } else {
          axis = 2;
        }
      }
      const int32_t p = positions3[static_cast<size_t>(axis) * T + i];
      const double freq = std::pow(
          base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      const double angle = static_cast<double>(p) * freq;
      cache[static_cast<size_t>(i * rot + pair)] =
          static_cast<float>(std::cos(angle));
      cache[static_cast<size_t>(i * rot + half + pair)] =
          static_cast<float>(std::sin(angle));
    }
  }
  return cache;
}

namespace {
// On-GPU greedy argmax: reduce the [1,vocab] f32 logits ON DEVICE and download
// ONLY the winning int64 token id, instead of the full-vocab (~993 KiB) f32
// D2H + a host argmax scan. Mirrors vLLM's greedy sampler device path
// (src/vllm/v1/sample/sampler.cpp Sampler::sample -> vt::GreedyArgmax; upstream
// vllm/v1/sample/ops greedy). vt::GreedyArgmax uses the LOWEST-index tie-break
// (torch.argmax), byte-for-byte the same winner the old host scan produced, so
// the greedy token stream is unchanged. The host argmax is GONE — this is the
// ONLY greedy path on the mm decode loop (multimodal-speed.md §5 #2).
int32_t VLArgMaxDevice(Dev d, DBuf& dlogits) {
  DBuf ids(d, DType::kI64, {1});
  vt::GreedyArgmax(d.q, ids.t(), dlogits.t());
  int64_t id = 0;
  ids.Download(d, &id);
  return static_cast<int32_t>(id);
}
}  // namespace

// ── M3-b image + M3d video shared GDN-hybrid VL greedy core. ────────────────
// Given the merge `mask` (true at each visual-token row of `prompt_ids`), the
// tower merger rows `mm_main` [N,H] (N == mask true-count), and the prebuilt
// MRoPE 3-D prefill positions `pos3_prefill` [3,T0] + decode continuation
// `delta`, this: embeds the prompt ids, scatters `mm_main` (bf16-rounded) into
// the masked rows to form inputs_embeds, runs the GDN-hybrid prefill/decode over
// its own paged KV + GDN recurrent state with the [T,rotary_dim] MRoPE cos|sin
// cache (built by BuildMropeCosSinHost), and greedy-decodes up to
// max_new_tokens (stops on eos_token_id). The IMAGE and VIDEO drivers differ
// ONLY in how `mask`/`pos3_prefill`/`delta` are built (image_token mask +
// Qwen3VLGetRopeIndex vs video_token mask + Qwen3VLGetRopeIndexVideo); the
// forward, KV/GDN state, MRoPE application, and decode continuation here are
// IDENTICAL, so the image path is byte-identical across the M3d refactor.
//
// MOE ARM (issue #891, .agents/specs/moe-vision-tower.md): the core is templated
// on the weights arm rather than copied. Upstream composes the SAME
// `Qwen3_VisionTransformer` on `Qwen3_5MoeForConditionalGeneration` and
// `Qwen3_5ForConditionalGeneration` (pinned `qwen3_5.py`) over two text
// backbones, so the tower, the processor, the MRoPE index math and this driver
// are shared and the ONLY difference is which backbone consumes the merger
// output. Two implementations of one tower would drift; the dense one is gated
// at image 32/32 + video 32/32. The dense instantiation calls exactly the
// functions it called before, in the same order.
//
// `graph_decode_supported` mirrors what the arm's PRODUCTION forward routes to a
// captured decode graph: the dense arm always (qwen3_5_dense.h), the MoE arm
// only on the fp4 CUDA path (`ForwardQwen3_5Moe`, qwen3_5_moe.cpp) — a bf16 MoE
// checkpoint decodes eagerly in production and so decodes eagerly here.
template <class W>
struct VLDecodeGraphFor;
template <>
struct VLDecodeGraphFor<Qwen3_5DenseWeights> {
  using type = Qwen3_5DenseDecodeGraph;
};
template <>
struct VLDecodeGraphFor<Qwen3_5MoeWeights> {
  using type = Qwen3_5DecodeGraph;
};

static void VLEmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& ids,
                        const Qwen3_5DenseWeights& w, const HfConfig& c) {
  DenseEmbedInto(d, hidden, ids, w, c);
}
static void VLEmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& ids,
                        const Qwen3_5MoeWeights& w, const HfConfig& c) {
  EmbedInto(d, hidden, ids, w, c);
}
static DBuf VLForwardLayersFor(Dev d, const Tensor& hidden_in,
                               const std::vector<int32_t>& positions,
                               const CommonAttentionMetadata& am,
                               const GDNAttentionMetadata& gm,
                               const std::vector<PagedKvCache>& attn_kv,
                               const std::vector<GdnStateCache>& gdn_state,
                               const Qwen3_5DenseWeights& w, const HfConfig& c,
                               const std::vector<int32_t>& logits_indices,
                               const std::vector<float>* mrope_cos_sin) {
  return DenseForwardLayers(d, hidden_in, positions, am, gm, attn_kv, gdn_state,
                            w, c, logits_indices, /*hidden_tap=*/nullptr,
                            mrope_cos_sin);
}
static DBuf VLForwardLayersFor(Dev d, const Tensor& hidden_in,
                               const std::vector<int32_t>& positions,
                               const CommonAttentionMetadata& am,
                               const GDNAttentionMetadata& gm,
                               const std::vector<PagedKvCache>& attn_kv,
                               const std::vector<GdnStateCache>& gdn_state,
                               const Qwen3_5MoeWeights& w, const HfConfig& c,
                               const std::vector<int32_t>& logits_indices,
                               const std::vector<float>* mrope_cos_sin) {
  return ForwardLayers(d, hidden_in, positions, am, gm, attn_kv, gdn_state, w, c,
                       logits_indices, /*hidden_tap=*/nullptr, mrope_cos_sin);
}

template <class W>
static std::vector<int32_t> VLGenerateCoreGdn(
    Dev d, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_main, const std::vector<bool>& mask,
    const std::vector<int32_t>& pos3_prefill, int64_t delta,
    int32_t eos_token_id, const W& weights,
    const HfConfig& config, int max_new_tokens,
    bool graph_decode_supported = true) {
  Backend& backend = d.b;
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // GDN recurrent-state dims (mirror the runner / test CachePool).
  const int64_t Hk = config.linear_num_key_heads;
  const int64_t Hv = config.linear_num_value_heads;
  const int64_t Dk = config.linear_key_head_dim;
  const int64_t Dv = config.linear_value_head_dim;
  const int64_t Kw = config.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = Kw - 1;

  // Persistent per-layer caches: bf16 paged KV for full-attn layers, F32 SSM +
  // BF16 conv recurrent state (one slot) for GDN layers. One big KV block.
  const int64_t block_size = T0 + max_new_tokens + 8;
  std::vector<std::shared_ptr<void>> storage;
  auto alloc0 = [&](size_t bytes) -> void* {
    void* p = backend.Alloc(bytes);
    backend.Memset(d.q, p, 0, bytes);
    storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
    return p;
  };
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const bool is_gdn =
        config.layer_types[static_cast<size_t>(l)] == "linear_attention";
    if (is_gdn) {
      void* ssm =
          alloc0(static_cast<size_t>(1 * Hv * Dv * Dk) * vt::SizeOf(DType::kF32));
      void* conv = alloc0(static_cast<size_t>(1 * conv_dim * conv_len) *
                          vt::SizeOf(DType::kBF16));
      GdnStateCache gs;
      gs.ssm_state =
          vt::Tensor::Contiguous(ssm, DType::kF32, d.q.device, {1, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(conv, DType::kBF16, d.q.device,
                                             {1, conv_dim, conv_len});
      gdn_state.push_back(gs);
    } else {
      const size_t kv_bytes = static_cast<size_t>(2 * block_size * Hkv * Dh) *
                              vt::SizeOf(DType::kBF16);
      PagedKvCache kv;
      kv.data = alloc0(kv_bytes);
      kv.dtype = DType::kBF16;
      kv.num_blocks = 1;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }

  // Metadata builders for one single-sequence step in block 0.
  auto attn_meta = [&](int64_t qlen, int64_t context) {
    CommonAttentionMetadata m;
    m.num_reqs = 1;
    m.num_actual_tokens = static_cast<int>(qlen);
    m.query_start_loc = {0, static_cast<int32_t>(qlen)};
    m.query_start_loc_cpu = m.query_start_loc;
    m.seq_lens = {static_cast<int32_t>(context + qlen)};
    m.seq_lens_cpu = m.seq_lens;
    m.max_query_len = static_cast<int>(qlen);
    m.max_seq_len = static_cast<int>(context + qlen);
    m.block_table_num_cols = 1;
    m.block_table_tensor = {0};
    for (int64_t t = 0; t < qlen; ++t) m.slot_mapping.push_back(context + t);
    m.causal = true;
    return m;
  };
  auto gdn_prefill_meta = [&](int64_t qlen) {
    GDNAttentionMetadata g;
    g.num_prefills = 1;
    g.num_prefill_tokens = static_cast<int>(qlen);
    g.num_decodes = 0;
    g.num_decode_tokens = 0;
    g.num_actual_tokens = static_cast<int>(qlen);
    g.has_initial_state = std::vector<uint8_t>{0};
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(qlen)};
    g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(qlen)};
    g.prefill_state_indices = std::vector<int32_t>{0};
    g.prefill_has_initial_state = std::vector<uint8_t>{0};
    const v1::CausalConv1dMetadata conv =
        v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
    g.batch_ptr = conv.batch_ptr;
    g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    return g;
  };
  auto gdn_decode_meta = [&]() {
    GDNAttentionMetadata g;
    g.num_prefills = 0;
    g.num_prefill_tokens = 0;
    g.num_decodes = 1;
    g.num_decode_tokens = 1;
    g.num_actual_tokens = 1;
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
    return g;
  };

  // Embed prompt ids, then scatter the (bf16-rounded) tower merger rows into the
  // image-token rows to form inputs_embeds (masked-scatter; NO deepstack on 27B).
  std::vector<uint16_t> emb_bits(static_cast<size_t>(T0 * H));
  {
    DBuf hemb(d, DType::kBF16, {T0, H});
    VLEmbedInto(d, hemb, prompt_ids, weights, config);
    hemb.Download(d, emb_bits.data());
  }
  {
    int64_t k = 0;
    for (int64_t t = 0; t < T0; ++t) {
      if (!mask[static_cast<size_t>(t)]) continue;
      for (int64_t h = 0; h < H; ++h)
        emb_bits[static_cast<size_t>(t * H + h)] =
            vt::F32ToBF16(mm_main[static_cast<size_t>(k * H + h)]);
      ++k;
    }
  }

  // 1-D positions (unused for rope under the fused MRoPE path; kept valid [T]).
  std::vector<int32_t> pos1d_prefill(static_cast<size_t>(T0));
  for (int64_t t = 0; t < T0; ++t)
    pos1d_prefill[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  const std::vector<float> mrope_prefill =
      BuildMropeCosSinHost(pos3_prefill, T0, config);

  // ---- PREFILL: gather only the last token's logits. ----
  std::vector<int32_t> generated;
  {
    DBuf merged(d, DType::kBF16, {T0, H}, emb_bits.data());
    const CommonAttentionMetadata am = attn_meta(T0, 0);
    const GDNAttentionMetadata gm = gdn_prefill_meta(T0);
    const std::vector<int32_t> last_idx = {static_cast<int32_t>(T0 - 1)};
    DBuf dlogits =
        VLForwardLayersFor(d, merged.t(), pos1d_prefill, am, gm, attn_kv,
                           gdn_state, weights, config, last_idx, &mrope_prefill);
    generated.push_back(VLArgMaxDevice(d, dlogits));
  }

  // ---- DECODE: one token/step; MRoPE positions equal on all 3 axes (text). ----
  //
  // Lever #3 (multimodal-speed.md §5.3): route the pure-DECODE steps through the
  // production graphed decode driver Qwen3_5DenseDecodeGraph (the SAME captured
  // cold->warm->replay step the text runner uses; qwen3_5_dense.h:314), instead
  // of the eager per-step DenseForwardLayers loop. The mm decode is single-seq
  // (B=1) so the padded capture size is S=1 == B — the "bit-identical rebuild of
  // the eager inputs" case (BuildPaddedDecode, qwen3_5.cpp:7151): at S==B the
  // captured graph output equals the eager Forward exactly. This makes the mm
  // decode step graph-capturable (closes the §3 "un-graphed eager loop" structural
  // gap) and drops the per-step host launch tax that dominates the audio-decode
  // 1.52x gap.
  //
  // ROPE EQUIVALENCE: during decode every position is a text token with the MRoPE
  // 3-axis positions equal on all axes (pos3_dec = {p,p,p}), so MRoPE degenerates
  // to standard 1-D RoPE at position p. The graph's forward applies 1-D device
  // RoPE from the `positions` vector, so passing positions = {p} (p = abs_idx +
  // delta, the MRoPE-adjusted decode position) reproduces the eager mrope decode's
  // rope angle. The KV physical slot stays abs_idx (attn_meta), unaffected. The
  // eager mrope path stays reachable via VT_MM_DECODE_EAGER=1 for the A/B.
  const bool decode_eager =
      (std::getenv("VT_MM_DECODE_EAGER") != nullptr &&
       std::string(std::getenv("VT_MM_DECODE_EAGER")) == "1");
  using DecodeGraph = typename VLDecodeGraphFor<W>::type;
  std::unique_ptr<DecodeGraph> dgraph;
  if (!decode_eager && graph_decode_supported)
    dgraph = std::make_unique<DecodeGraph>(weights, config, d.q,
                                           /*max_num_reqs=*/1);
  for (int step = 1; step < max_new_tokens; ++step) {
    if (generated.back() == eos_token_id) break;
    const int64_t abs_idx = T0 + (step - 1);  // sequence index of the fed token
    const int32_t p = static_cast<int32_t>(abs_idx + delta);
    const std::vector<int32_t> one_id = {generated.back()};
    const CommonAttentionMetadata am = attn_meta(1, abs_idx);
    const GDNAttentionMetadata gm = gdn_decode_meta();
    if (dgraph) {
      // Graphed pure-decode step (embed happens ON DEVICE inside Step). The 1-D
      // position p drives the device RoPE (== degenerate MRoPE {p,p,p}); the
      // returned logits stay on device and feed GreedyArgmax with no full-vocab
      // D2H.
      const std::vector<int32_t> pos_dec = {p};
      ForwardLogits fl =
          dgraph->Step(one_id, pos_dec, am, gm, attn_kv, gdn_state);
      DBuf ids(d, DType::kI64, {1});
      vt::GreedyArgmax(d.q, ids.t(), fl.device_tensor);
      int64_t id = 0;
      ids.Download(d, &id);
      generated.push_back(static_cast<int32_t>(id));
    } else {
      // Eager fallback (VT_MM_DECODE_EAGER=1): the original per-step mrope forward.
      const std::vector<int32_t> pos3_dec = {p, p, p};
      const std::vector<int32_t> pos1d_dec = {static_cast<int32_t>(abs_idx)};
      const std::vector<float> mrope_dec = BuildMropeCosSinHost(pos3_dec, 1, config);
      DBuf tok(d, DType::kBF16, {1, H});
      VLEmbedInto(d, tok, one_id, weights, config);
      DBuf dlogits = VLForwardLayersFor(d, tok.t(), pos1d_dec, am, gm, attn_kv,
                                        gdn_state, weights, config, {}, &mrope_dec);
      generated.push_back(VLArgMaxDevice(d, dlogits));
    }
  }
  return generated;
}

// ── Shared image / video PREFILL PLAN (both arms). ──────────────────────────
// The merge mask, the mm_main row-count check and the MRoPE 3-D prefill
// positions + decode continuation delta. NONE of this depends on the text
// backbone — the tower, the processor and the rope-index math are shared between
// `Qwen3_5ForConditionalGeneration` (dense 27B) and
// `Qwen3_5MoeForConditionalGeneration` (MoE 35B) upstream — so it is computed
// once here rather than copied per arm (issue #891).
namespace {
struct VLPrefillPlan {
  std::vector<bool> mask;
  std::vector<int32_t> pos3_prefill;
  int64_t delta = 0;
};

// spatial_merge_size is 2 for the whole Qwen3-VL lineage (not a text-config
// field; matches the 4B, the 27B and the 35B `vision_config`).
constexpr int64_t kVLSpatialMergeSize = 2;

VLPrefillPlan VLPlanImage(const std::vector<int32_t>& prompt_ids,
                          const std::vector<float>& mm_main, int64_t H,
                          const std::array<int64_t, 3>& grid_thw,
                          int32_t image_token_id) {
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  VLPrefillPlan plan;
  plan.mask.assign(static_cast<size_t>(T0), false);
  int64_t offset = -1, n_img = 0;
  for (int64_t t = 0; t < T0; ++t) {
    if (prompt_ids[static_cast<size_t>(t)] == image_token_id) {
      if (offset < 0) offset = t;
      plan.mask[static_cast<size_t>(t)] = true;
      ++n_img;
    }
  }
  VT_CHECK(offset >= 0, "qwen3_5 VL: no image token in prompt");
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_img, "qwen3_5 VL: mm_main rows != image-token count");
  const std::vector<multimodal::MmImageSpan> spans = {{offset, grid_thw}};
  plan.pos3_prefill = multimodal::Qwen3VLGetRopeIndex(
      prompt_ids, spans, kVLSpatialMergeSize, &plan.delta);
  return plan;
}

VLPrefillPlan VLPlanVideo(const std::vector<int32_t>& prompt_ids,
                          const std::vector<float>& mm_main, int64_t H,
                          const std::array<int64_t, 3>& grid_thw,
                          int32_t video_token_id, int32_t vision_start_token_id,
                          int32_t vision_end_token_id) {
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  VLPrefillPlan plan;
  plan.mask.assign(static_cast<size_t>(T0), false);
  int64_t n_vid = 0;
  for (int64_t t = 0; t < T0; ++t) {
    if (prompt_ids[static_cast<size_t>(t)] == video_token_id) {
      plan.mask[static_cast<size_t>(t)] = true;
      ++n_vid;
    }
  }
  VT_CHECK(n_vid > 0, "qwen3_5 VL: no video token in prompt");
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_vid, "qwen3_5 VL: mm_main rows != video-token count");
  plan.pos3_prefill = multimodal::Qwen3VLGetRopeIndexVideo(
      prompt_ids, grid_thw, kVLSpatialMergeSize, vision_start_token_id,
      video_token_id, vision_end_token_id, &plan.delta);
  return plan;
}
}  // namespace

std::vector<int32_t> Qwen3_5VLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3_5DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const VLPrefillPlan plan = VLPlanImage(prompt_ids, mm_main, config.hidden_size,
                                         grid_thw, image_token_id);
  return VLGenerateCoreGdn(d, prompt_ids, mm_main, plan.mask, plan.pos3_prefill,
                           plan.delta, eos_token_id, weights, config,
                           max_new_tokens);
}

// ── M3d: the Qwen3.6-27B GDN-hybrid VL video->text greedy driver. ───────────
// Mirrors the image driver above through the shared VLGenerateCoreGdn; the two
// video differences (identical to the 4B M3c split) are (a) the merge mask is on
// video_token_id across ALL frames (not image_token_id) and (b) the MRoPE prefill
// positions come from Qwen3VLGetRopeIndexVideo, which scans the timestamp-
// interleaved, per-frame placeholder structure for grid_t frames. NO DeepStack
// (empty deepstack_visual_indexes on the 27B). The tower + video processor +
// video MRoPE were already proven bit/near-exact by M3c on the 4B path; the tower
// per-frame windowing is config-shared, so this driver is the only new wiring.
std::vector<int32_t> Qwen3_5VLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3_5DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const VLPrefillPlan plan =
      VLPlanVideo(prompt_ids, mm_main, config.hidden_size, grid_thw,
                  video_token_id, vision_start_token_id, vision_end_token_id);
  return VLGenerateCoreGdn(d, prompt_ids, mm_main, plan.mask, plan.pos3_prefill,
                           plan.delta, eos_token_id, weights, config,
                           max_new_tokens);
}

// ── issue #891: the Qwen3.6-35B-A3B MoE GDN-hybrid VL image / video drivers. ──
//
// The MoE arm of the SAME model family. Upstream's
// `Qwen3_5MoeForConditionalGeneration` composes the SAME `Qwen3_VisionTransformer`
// as `Qwen3_5ForConditionalGeneration` over the MoE text backbone (pinned vLLM
// `qwen3_5.py`), and `deepstack_visual_indexes: []` compiles the deepstack path
// out for the whole family (`qwen3_vl.py:1709-1716`). So these are the dense
// drivers above with `Qwen3_5MoeWeights` substituted: the SAME shared tower
// (`LoadQwen3_5MoeVision` -> `LoadQwen3VLVisionWeights` ->
// `Qwen3VLVisionForward`), the SAME processor, the SAME MRoPE index math
// (`VLPlanImage` / `VLPlanVideo`) and the SAME greedy core
// (`VLGenerateCoreGdn<Qwen3_5MoeWeights>`).
//
// GATED ON MM INPUT: these are separate entry points, reached only when a request
// carries an image or a video. A text-only request never enters here — it goes
// through the registered `ForwardQwen3_5Moe`, which is untouched, so the text
// path executes the identical instruction sequence it did before.
//
// DECODE GRAPH: `graph_decode_supported` mirrors `ForwardQwen3_5Moe`'s own
// routing — the MoE decode graph is taken only on the fp4 CUDA path, so a bf16
// checkpoint (`Qwen/Qwen3.6-35B-A3B`) decodes eagerly here exactly as it does in
// production text decode.
namespace {
bool MoeDecodeGraphSupported(const Qwen3_5MoeWeights& weights) {
  return !weights.layers.empty() &&
         !weights.layers.front().moe.expert_gate_fp4.empty();
}
}  // namespace

std::vector<int32_t> Qwen3_5MoeVLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const VLPrefillPlan plan = VLPlanImage(prompt_ids, mm_main, config.hidden_size,
                                         grid_thw, image_token_id);
  return VLGenerateCoreGdn(d, prompt_ids, mm_main, plan.mask, plan.pos3_prefill,
                           plan.delta, eos_token_id, weights, config,
                           max_new_tokens, MoeDecodeGraphSupported(weights));
}

std::vector<int32_t> Qwen3_5MoeVLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const VLPrefillPlan plan =
      VLPlanVideo(prompt_ids, mm_main, config.hidden_size, grid_thw,
                  video_token_id, vision_start_token_id, vision_end_token_id);
  return VLGenerateCoreGdn(d, prompt_ids, mm_main, plan.mask, plan.pos3_prefill,
                           plan.delta, eos_token_id, weights, config,
                           max_new_tokens, MoeDecodeGraphSupported(weights));
}

ForwardLogits Qwen3_5DenseModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices);
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

ForwardLogits Qwen3_5DenseModel::ForwardDeviceTap(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    Qwen3_5MTPHiddenStates* hidden_out,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  DBuf tap(d, DType::kBF16, {T, H});
  const Tensor tap_view = tap.t();
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices, &tap_view);
  if (hidden_out != nullptr) {
    hidden_out->tensor = tap.t();
    hidden_out->storage = tap.ReleaseShared();
  }
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

ForwardLogits Qwen3_5DenseModel::ForwardDeviceMultiTap(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    Qwen3_5AuxTaps* aux_out, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  if (aux_out == nullptr) {
    return ForwardDevice(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                         gdn_state, weights, config, queue, logits_indices);
  }
  ValidateAuxTapLayerIds(aux_out->layer_ids, config.num_hidden_layers);
  const int64_t taps = static_cast<int64_t>(aux_out->layer_ids.size());
  DBuf aux(d, DType::kBF16, {T, H * taps});
  const Tensor aux_view = aux.t();
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices, /*hidden_tap=*/nullptr,
                                  &aux_out->layer_ids, &aux_view);
  aux_out->tensor = aux.t();
  aux_out->storage = aux.ReleaseShared();
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

std::vector<float> Qwen3_5ReplayLayer(const Qwen3_5MoeLayerWeights& layer,
                                      const HfConfig& config,
                                      const std::vector<float>& hidden_in,
                                      const std::vector<int32_t>& positions,
                                      int64_t seqlen, vt::Queue& queue) {
  const int64_t T = seqlen;
  const int64_t H = config.hidden_size;
  VT_CHECK(static_cast<int64_t>(hidden_in.size()) == T * H,
           "qwen3_5 replay: hidden_in must be [T*H]");
  // ONE decode step. This replays a single layer as a self-contained unit of
  // work, so the slices it takes are finished with when it returns; without a
  // boundary they stay pinned for the life of the process (#1091 finding 3).
  const Qwen35ExpertStreamStep expert_stream_step;
  Dev d{vt::GetBackend(queue.device.type), queue};

  // Seed the fused stream with the combined residual input: res = hidden_in,
  // hidden delta = 0. The layer's input_layernorm then normalizes hidden_in.
  DBuf res(d, DType::kF32, {T, H}, hidden_in.data());
  DBuf hidden(d, ActDType(d), {T, H});
  hidden.Zero(d);
  // `Qwen3_5ReplayLayer` replays ONE layer in isolation and is not told which,
  // so it cannot consult a per-layer placement. Passing -1 makes the plan answer
  // the engine device, which is the inert path — a replay must reproduce the
  // layer's arithmetic, and silently placing it would change which device the
  // replay measured.
  RunLayer(d, layer, config, hidden, res, positions, T, /*layer_index=*/-1);

  // Combined stream out = residual + hidden (f32), directly comparable to the
  // layer golden's `out`.
  std::vector<float> res_host(static_cast<size_t>(T) * H);
  res.Download(d, res_host.data());
  std::vector<uint16_t> hidden_host(static_cast<size_t>(T) * H);
  hidden.Download(d, hidden_host.data());
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = res_host[i] + vt::BF16ToF32(hidden_host[i]);
  return out;
}

std::vector<float> Qwen3_5ReplayDenseLayer(
    const Qwen3_5DenseLayerWeights& layer, const HfConfig& config,
    const std::vector<float>& hidden_in, const std::vector<int32_t>& positions,
    int64_t seqlen, vt::Queue& queue) {
  const int64_t T = seqlen;
  const int64_t H = config.hidden_size;
  VT_CHECK(static_cast<int64_t>(hidden_in.size()) == T * H,
           "qwen3_5 dense replay: hidden_in must be [T*H]");
  Dev d{vt::GetBackend(queue.device.type), queue};

  // Match Qwen3_5ReplayLayer's fused residual contract: the captured layer
  // input is the combined stream, so seed it as `res` and start the bf16 delta
  // at zero before running the real dense attention + SwiGLU layer.
  DBuf res(d, DType::kF32, {T, H}, hidden_in.data());
  DBuf hidden(d, ActDType(d), {T, H});
  hidden.Zero(d);
  RunDenseLayer(d, layer, config, hidden, res, positions, T);

  std::vector<float> res_host(static_cast<size_t>(T) * H);
  res.Download(d, res_host.data());
  std::vector<uint16_t> hidden_host(static_cast<size_t>(T) * H);
  hidden.Download(d, hidden_host.data());
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = res_host[i] + vt::BF16ToF32(hidden_host[i]);
  }
  return out;
}

// ─── Qwen3_5DecodeGraph (decode CUDA-graph driver) ──────────────────────────
namespace {

// Overwrite dst's CONTENTS from src WITHOUT changing dst.data() when the sizes
// already match (preserves the fixed address a captured host->device copy reads
// from); reallocate only when the shape actually changed.
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}
template <typename T>
void CopyInPlace(std::optional<std::vector<T>>& dst,
                 const std::optional<std::vector<T>>& src) {
  if (!src.has_value()) {
    dst.reset();
    return;
  }
  if (!dst.has_value()) dst.emplace();
  CopyInPlace(*dst, *src);
}

// ENG-CUDAGRAPH-BREAK W6 (#1374): THE DECODE-GRAPH SLOT KEY, closing the half of
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020) that is not the
// predicate.
//
// The ring used to be keyed on the padded token count S ALONE, and `Refresh`'s
// own comment already recorded the residual: "the slot ring is keyed on S alone,
// and two different uniform query lengths can now reach one key". That was
// harmless only while the predicate admitted exactly one query length per
// engine, which is the guarantee W6 removes.
//
// IT WAS NOT ONLY A FUTURE HAZARD. `S = spec_step ? B : PadToCaptureSize(B)`, so
// a SPEC step of 4 requests at 1+1 tokens and a NON-SPEC padded step of 8
// requests both land on S == 8 TODAY. The two carry different metadata --
// `num_spec_decodes`, `spec_query_start_loc`, `num_accepted_tokens` -- and the
// second would replay a graph captured against the first. `Refresh` copies IN
// PLACE only while the sizes match; a size change reassigns the vector, so the
// step that follows a shape swap also moves the host addresses a capture baked
// (spec `## Risks/decisions` D2). Silently wrong logits, invisible to a token
// gate.
//
// `spec` is in the key and `query_len` alone is not enough for the same reason:
// a spec step whose requests each carry ONE token is uniform at q == 1 and still
// carries spec segmentation, so it must not share a ring with a padded pure
// decode of the same width.
struct DecodeGraphSlotKey {
  int64_t size = 0;       // S, the captured token count
  int64_t query_len = 1;  // the uniform query length the graph was captured for
  bool spec = false;      // the step carried spec-decode segmentation
};
inline bool operator<(const DecodeGraphSlotKey& a, const DecodeGraphSlotKey& b) {
  if (a.size != b.size) return a.size < b.size;
  if (a.query_len != b.query_len) return a.query_len < b.query_len;
  return static_cast<int>(a.spec) < static_cast<int>(b.spec);
}

// The bound on how many DISTINCT speculative query lengths one driver captures.
//
// WHY A BOUND EXISTS AT ALL. Before W6 the predicate admitted one query length
// per engine, so the spec shape count was bounded by `max_num_seqs`. Reading the
// step's ACTUAL length multiplies that ceiling by `1 + k`, and every shape
// retains an [S, vocab] f32 logits block plus an [S, H] hidden -- at a 151k
// vocabulary the logits alone are ~0.6 MB per token of S, times two ring slots.
// An unbounded widening is a memory regression nobody measured, so the widening
// is bounded, tunable and COUNTED rather than open.
//
// The default of 2 is the smallest value that admits anything new: one steady-
// state length (the configured `1 + k`, which every unclamped verify takes) plus
// one clamped length. `VT_SPEC_GRAPH_MAX_QLENS=0` removes the bound; a larger
// value widens it. A step past the bound falls to the eager arm, which is
// exactly what it did before W6, and `qlen_cap_declines` reports it.
inline int64_t SpecQueryLenCap() {
  static const int64_t cap = [] {
    const char* v = std::getenv("VT_SPEC_GRAPH_MAX_QLENS");
    if (v == nullptr || v[0] == '\0') return static_cast<int64_t>(2);
    return static_cast<int64_t>(std::strtoll(v, nullptr, 10));
  }();
  return cap;
}

// True when capturing `key` would push this driver past `SpecQueryLenCap()`
// distinct speculative query lengths. A key already in the map is never capped:
// declining a shape the driver already holds a graph for would strand it.
template <class MapT>
bool DecodeGraphQueryLenCapped(const MapT& slots, const DecodeGraphSlotKey& key) {
  if (!key.spec || key.query_len <= 1) return false;
  const int64_t cap = SpecQueryLenCap();
  if (cap <= 0) return false;
  if (slots.find(key) != slots.end()) return false;
  // The map is ordered by (size, query_len, spec), so equal query lengths are
  // NOT adjacent and a run-length count would over-report. Collect them; the
  // set is bounded by the cap plus one and this runs once per step.
  std::vector<int64_t> seen;
  for (const auto& kv : slots) {
    if (!kv.first.spec || kv.first.query_len <= 1) continue;
    if (kv.first.query_len == key.query_len) return false;
    if (std::find(seen.begin(), seen.end(), kv.first.query_len) == seen.end()) {
      seen.push_back(kv.first.query_len);
    }
  }
  return static_cast<int64_t>(seen.size()) >= cap;
}

// The decode-graph capture set + pad-to-capture selector live in
// vllm/model_executor/models/decode_graph_sizes.h (DecodeGraphSizes /
// PadToCaptureSize), derived from max_num_seqs to mirror vLLM's
// _set_cudagraph_sizes reduced to the full-decode-cudagraph regime. A real
// decode batch of B requests is PADDED up to PadToCaptureSize(B, max_num_seqs)
// and that size's graph is replayed; the padded rows are inert (see
// BuildPaddedDecode). The cap at max_num_seqs (the GDN state-cache slot count)
// keeps a decode batch from padding beyond the conv/ssm state cache, tripping
// the causal_conv1d_update `conv_state.shape[0] >= x.shape[0]` guard.

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// decode forward is ROW-INDEPENDENT (paged attn is per-request causal, GDN
// recurrence is per-sequence, MoE/router/norm/lm_head are per-token — no cross-
// row reduction), so appending S-B INERT rows cannot perturb the real rows'
// logits. The padding rows are made inert exactly as vLLM's cudagraph padding:
//   * token id / position 0 (embed row is discarded);
//   * slot_mapping = -1  → ReshapeAndCache skips the KV write (cuda_cache.cu:50);
//   * gdn state index = -1 → causal_conv1d_update / GdnDecode skip the in-place
//     mamba/conv update (cuda_gdn.cu:153,471), so no real state slot is touched;
//   * seq_lens = 1 + block_table row 0 → PagedAttention does a valid in-bounds
//     read of block 0 whose output row is discarded (never returned to the caller).
// The real prefix [0,B) is copied verbatim, so at S==B this is a bit-identical
// rebuild of the eager inputs (the S==B graph output equals Forward exactly).
void BuildPaddedDecode(int64_t S, const std::vector<int32_t>& tok,
                       const std::vector<int32_t>& pos,
                       const v1::CommonAttentionMetadata& am,
                       const v1::GDNAttentionMetadata& gm,
                       std::vector<int32_t>& tok_out,
                       std::vector<int32_t>& pos_out,
                       v1::CommonAttentionMetadata& am_out,
                       v1::GDNAttentionMetadata& gm_out) {
  const int64_t B = static_cast<int64_t>(tok.size());
  const int64_t cols = am.block_table_num_cols;

  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  // W10 (#1857): a pure-decode rewrite is never spec-classified. Belt on the
  // vt shape guard's braces (S == q*S only at q == 1).
  am_out.uniform_spec_query_len = 0;
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  gm_out = gm;
  gm_out.num_prefills = 0;
  gm_out.num_prefill_tokens = 0;
  gm_out.num_decodes = static_cast<int>(S);
  gm_out.num_decode_tokens = static_cast<int>(S);
  gm_out.num_actual_tokens = static_cast<int>(S);
  {
    std::vector<int32_t> si(static_cast<size_t>(S), -1);  // inert padding slots
    if (gm.non_spec_state_indices_tensor.has_value())
      std::copy(gm.non_spec_state_indices_tensor->begin(),
                gm.non_spec_state_indices_tensor->end(), si.begin());
    gm_out.non_spec_state_indices_tensor = std::move(si);
  }
  {
    std::vector<int32_t> q(static_cast<size_t>(S + 1));
    for (int64_t i = 0; i <= S; ++i) q[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    gm_out.non_spec_query_start_loc = std::move(q);
  }
  // Pure decode (num_prefills==0): the prefill-only fields are unused.
  gm_out.has_initial_state.reset();
  gm_out.prefill_query_start_loc.reset();
  gm_out.prefill_state_indices.reset();
  gm_out.prefill_has_initial_state.reset();
  gm_out.batch_ptr.reset();
  gm_out.token_chunk_offset_ptr.reset();
  (void)B;
}

}  // namespace

// VT_ASYNC_EXECUTOR — Option A: the decode-graph per-step input H2D staged OUT of
// the captured replay (the faithful vLLM structure, the real c8-c32 overlap unlock).
// vLLM stages each step's inputs on the main stream OUTSIDE its graph into PERSISTENT
// DEVICE buffers the graph reads (gpu_model_runner.py _prepare_input_ids), guarded by
// a blocking input-prep event recorded right after the staging H2D (states.py:64,
// synchronize_input_prep) — so the next step's host prep waits only that tiny copy,
// never the forward. Option A ports exactly that: each SizeSlot gets persistent device
// input buffers (a StepDevInputs, s.dev) and PINNED host staging buffers (s.pin); the
// graph is captured reading s.dev; per step the H2D is enqueued on the main queue
// BEFORE ReplayGraph and an input-staged event is recorded right after it. This
// REPLACES Option B (#36), where the H2D was baked inside the captured replay so the
// runner's depth-2 drain had to wait the whole GPU tail (c16 ~7%).
//
// The 2-slot parity ring is RETAINED: the depth-2 loop enqueues sample(i-1) AFTER
// forward(i) (core.cpp step_with_batch_queue), so the persistent logits MUST double-
// buffer or forward(i+1) overwrites logits(i) before sample(i) reads them; the ring
// also gives each slot's pinned inputs a 2-step reuse window. DEFAULT OFF:
// unset/anything-but-"1" ⇒ single-slot (slot[0]), no staging, no events — byte-
// identical to the pre-lever baked-H2D driver. VT_ASYNC_EXECUTOR_NO_DBUF=1 is a TEST-
// ONLY escape hatch that forces the ring OFF while the runner still skips the drain,
// so the gate SEES the logits/input hazard in a RED arm. Production never sets it.
static bool DecodeGraphDoubleBufferEnabled() {
  const char* v = std::getenv("VT_ASYNC_EXECUTOR");
  if (v == nullptr || v[0] != '1' || v[1] != '\0') return false;
  const char* nod = std::getenv("VT_ASYNC_EXECUTOR_NO_DBUF");
  if (nod != nullptr && nod[0] == '1' && nod[1] == '\0') return false;
  return true;
}

// Pinned host staging for the per-step-varying decode-graph inputs (positions,
// slot_mapping, block_table, seq_lens, query_start_loc, gdn_state_idx, token_ids).
// cudaHostAlloc'd so the H2D into the persistent device buffers is a TRUE async DMA:
// a pageable std::vector H2D on GB10 is effectively host-synchronous (which is
// precisely why #36's poison could never race the copy and its RED was
// unreproducible). Sized once at capture for the padded size S / block-table cols /
// gdn index count. Non-owning of the SizeSlot's std::vectors — StageStepInputs copies
// them in each step, then enqueues the async device copy.
//
// ENG-CUDAGRAPH-BREAK W4 (#1307): each field is now a `vt::PersistentStepInput`
// — the capture-stable device destination BOUND together with its pinned host
// staging block — instead of a bare pinned pointer this struct allocated and a
// `Backend::Copy` the stager issued by hand beside it. The two halves were
// separated by ninety lines, so nothing checked that the staged byte count and
// the destination's size agreed; the seam refuses a refresh longer than the
// bound capacity, which is the address-stability rule the captured graph depends
// on.
//
// THE PINNED `token_ids` BLOCK IS GONE, and its absence is a finding rather than
// a tidy-up. It was allocated here, filled by `StageStepInputs` and zeroed by
// the poison hook, and it was NEVER uploaded and never read: `StepDevInputs`
// has no token-id member, because the embed runs OUTSIDE the captured region
// from the HOST vector (`EmbedInto`), so the decode graph carries no token ids
// to the device at all. That is also why making `StepDevInputs` a seam
// capability does not by itself close the `qwen3.cpp` async decline — the fix
// that decline names needs a DEVICE token-id destination no driver has. See the
// spec's `## Owed`.
struct PinnedStepInputs {
  vt::PersistentStepInput positions;
  vt::PersistentStepInput slot_mapping;
  vt::PersistentStepInput block_table;
  vt::PersistentStepInput seq_lens;
  vt::PersistentStepInput qsl;
  vt::PersistentStepInput gdn_state_idx;
  int64_t S = 0, R = 0, cols = 0, idx = 0;  // S = tokens, R = requests
  bool has_idx = false;
  bool ready = false;

  PinnedStepInputs() = default;
  PinnedStepInputs(const PinnedStepInputs&) = delete;
  PinnedStepInputs& operator=(const PinnedStepInputs&) = delete;
  void Free() {
    positions.Unbind();
    slot_mapping.Unbind();
    block_table.Unbind();
    seq_lens.Unbind();
    qsl.Unbind();
    gdn_state_idx.Unbind();
    S = 0; R = 0; cols = 0; idx = 0;
    has_idx = false;
    ready = false;
  }
  // SPEC-DSPARK W8 (#442): `S` is TOKENS and `R` is REQUESTS. They are equal for
  // pure decode, which is why one count sufficed until a speculative verify
  // arrived with S = R * (1+k). Sizing the per-request arrays by S overran both
  // the host block table and the device buffer (`cudaMemcpyAsync: invalid
  // argument`). Upstream keeps both for the same reason: its graph key is
  // BatchDescriptor(num_tokens, num_reqs, uniform).
  // BINDS the destinations `dev` already owns; it does not allocate them. The
  // device side is drawn from a DEDICATED pool so a retained input never pops a
  // block the captured forward's own scratch then needs, and the seam
  // deliberately does not take that decision over (see
  // `include/vt/persistent_step_input.h`).
  void Alloc(Backend& bk, StepDevInputs& dev, int64_t S_, int64_t R_, int64_t cols_,
             int64_t idx_, bool has_idx_) {
    Free();
    S = S_; R = R_ > 0 ? R_ : S_; cols = cols_; idx = idx_; has_idx = has_idx_;
    positions.Bind(bk, dev.positions.ptr(), sizeof(int32_t) * S);
    slot_mapping.Bind(bk, dev.slot_mapping.ptr(), sizeof(int64_t) * S);
    block_table.Bind(bk, dev.block_table.ptr(), sizeof(int32_t) * R * cols);
    seq_lens.Bind(bk, dev.seq_lens.ptr(), sizeof(int32_t) * R);
    qsl.Bind(bk, dev.query_start_loc.ptr(), sizeof(int32_t) * (R + 1));
    if (has_idx) gdn_state_idx.Bind(bk, dev.gdn_state_idx.ptr(), sizeof(int32_t) * idx);
    ready = true;
  }
};

// Dedicated pool for the Option A PERSISTENT decode-graph device inputs (s.dev) and
// the persistent cos|sin cache. These are RETAINED across steps, so drawing them
// from the main scratch Pool() would pop-and-hold a block out of a size-class the
// captured forward's own scratch then needs — an empty free list there is a
// cudaMalloc mid-capture (aborts capture). Isolating them in their own pool leaves
// the main Pool() exactly as the eager pre-warm step left it, so every allocation
// the captured region makes is a pool HIT. DBuf remembers its owning pool, so these
// buffers return here on slot reset.
//
// PER DEVICE, like every other pool (#516): the isolation this wants is from the
// main scratch pool of the SAME device, and a process-wide instance would hand
// one device's retained decode inputs to another's captured forward.
static DevicePool& PersistentDecodeInputPool(vt::Backend& b) {
  static detail::PoolTable table;
  return table.For(b);
}

// Option A per-step input staging: copy the slot's refreshed host inputs into its
// pinned buffers (host memcpy), then enqueue the true-async H2D pinned->persistent
// device buffers on the main queue. This is the H2D moved OUT of the captured replay.
// The caller records the input-staged event immediately after, so the next same-slot
// Refresh waits only this tiny copy, not the GPU tail. Templated because the two
// decode-graph drivers nest their own SizeSlot with identical field names.
// SPEC-DSPARK W8 (#442): refill the PERSISTENT spec device tensors in place from
// this slot's host metadata, OUTSIDE the captured region.
//
// This is the piece whose absence made the first spec capture produce garbage:
// StageStepInputs refills only the pure-decode fields, so a replay re-read the
// PREVIOUS step's `num_accepted` -- the very value GdnSpecDecode uses to select
// each request's initial state (column num_accepted-1) and the conv window
// advance. Stale there means the recurrence rolls back to the wrong state, which
// measured as incoherent tokens AND 5x slower (26 vs 136 tok/s), because zero
// acceptance multiplies the step count.
//
// Copies are issued on the main queue before the replay, so the graph reads the
// addresses it was captured with, holding this step's values.
template <class Slot>
static void StageSpecStepInputs(Dev d, Slot& s) {
  if (s.dev == nullptr) return;
  StepDevInputs& dev = *s.dev;
  if (!dev.has_gdn_spec) return;
  const v1::GDNAttentionMetadata& gm = s.gdn_meta;
  const auto cp32 = [&](DBuf& dst, const std::optional<std::vector<int32_t>>& src) {
    if (src.has_value() && !src->empty())
      d.b.Copy(d.q, dst.ptr(), src->data(), sizeof(int32_t) * src->size());
  };
  cp32(dev.gdn_spec_state_idx, gm.spec_state_indices_tensor);
  cp32(dev.gdn_spec_qsl, gm.spec_query_start_loc);
  cp32(dev.gdn_num_accepted, gm.num_accepted_tokens);
  cp32(dev.gdn_spec_token_indx, gm.spec_token_indx);
  cp32(dev.gdn_non_spec_token_indx, gm.non_spec_token_indx);
  if (gm.spec_sequence_masks.has_value() && !gm.spec_sequence_masks->empty())
    d.b.Copy(d.q, dev.gdn_spec_seq_masks.ptr(), gm.spec_sequence_masks->data(),
             gm.spec_sequence_masks->size());
  // conv state slot == column 0 of each request's spec_state_indices row
  // (BuildStepDevInputs derives it the same way).
  if (gm.spec_state_indices_tensor.has_value() && gm.spec_state_indices_num_cols > 0) {
    const int64_t ns = gm.num_spec_decodes;
    const int64_t nc = gm.spec_state_indices_num_cols;
    const std::vector<int32_t>& ssi = *gm.spec_state_indices_tensor;
    if (ns > 0 && static_cast<int64_t>(ssi.size()) >= ns * nc) {
      std::vector<int32_t> col0(static_cast<size_t>(ns));
      for (int64_t i = 0; i < ns; ++i)
        col0[static_cast<size_t>(i)] = ssi[static_cast<size_t>(i * nc)];
      d.b.Copy(d.q, dev.gdn_spec_conv_state_idx.ptr(), col0.data(),
               sizeof(int32_t) * static_cast<size_t>(ns));
    }
  }
}

template <class Slot>
static void StageStepInputs(Dev d, Slot& s) {
  PinnedStepInputs& pin = s.pin;
  StepDevInputs& dev = *s.dev;
  const int64_t S = pin.S, R = pin.R, cols = pin.cols;  // tokens, requests
  // ENG-CUDAGRAPH-BREAK W4 (#1307): each line is the memcpy-into-pinned PLUS the
  // asynchronous H2D into the SAME device address, which the seam holds together
  // so the pair cannot drift apart. The destination address is unchanged by a
  // refresh BY CONSTRUCTION, which is what the captured graph baked.
  pin.positions.RefreshFromHost(d.q, s.positions.data(), sizeof(int32_t) * S);
  pin.slot_mapping.RefreshFromHost(d.q, s.attn_meta.slot_mapping.data(),
                                   sizeof(int64_t) * S);
  pin.block_table.RefreshFromHost(d.q, s.attn_meta.block_table_tensor.data(),
                                  sizeof(int32_t) * R * cols);
  pin.seq_lens.RefreshFromHost(d.q, s.attn_meta.seq_lens.data(), sizeof(int32_t) * R);
  pin.qsl.RefreshFromHost(d.q, s.attn_meta.query_start_loc.data(),
                          sizeof(int32_t) * (R + 1));
  // Skipping the refresh and re-uploading the previous step's staged bytes are
  // the same device state, because the staging block still holds them; the skip
  // is the one that does not move bytes to say so.
  if (pin.has_idx && dev.has_gdn_idx &&
      s.gdn_meta.non_spec_state_indices_tensor.has_value())
    pin.gdn_state_idx.RefreshFromHost(
        d.q, s.gdn_meta.non_spec_state_indices_tensor->data(),
        sizeof(int32_t) * pin.idx);
}

// Test-only (VT_ASYNC_EXECUTOR_POISON): immediately AFTER StageStepInputs enqueued the
// async H2D and the input-staged event was recorded, overwrite the PINNED source
// WITHOUT syncing. A true-async DMA reads the corrupted source, so s.dev — and the
// replay that reads it — see garbage: the DETERMINISTIC RED proving the input-staged
// event is load-bearing (a Refresh that ran ahead of that event would do exactly
// this). Pageable staging is host-synchronous and could not race, which is why the
// #36 poison never reproduced. Never set in production (gated by an Impl `poison`
// member read once at construction, so the default hot path pays no getenv).
template <class Slot>
static void MaybePoisonStagedInputs(bool poison, Slot& s) {
  if (!poison || !s.pin.ready) return;
  PinnedStepInputs& pin = s.pin;
  // The STAGING block is what a true asynchronous DMA is still reading, so
  // corrupting it is what races the copy. `token_ids` is not in this list any
  // more because its pinned block is gone: it was never uploaded, so zeroing it
  // could never have poisoned anything (see PinnedStepInputs).
  //
  // EACH CELL IS ZEROED OVER ITS OWN CAPACITY, and that closes #1319. This hook
  // used to fill all four blocks over `pin.S` — TOKENS — while `seq_lens` is
  // allocated with `pin.R`, REQUESTS. The two are equal on a pure-decode step
  // and NOT on a speculative one, where `S = R * (1 + k)`, so the fill ran
  // `(S - R)` int32s past the end of a `cudaHostAlloc`'d block, corrupting
  // pinned memory inside the very arm that exists to fail loudly. Asking each
  // cell for its own capacity makes the count and the allocation the same
  // fact.
  const auto zero = [](vt::PersistentStepInput& c) {
    if (c.staging() != nullptr) std::memset(c.staging(), 0, c.capacity());
  };
  zero(pin.positions);
  zero(pin.slot_mapping);
  zero(pin.seq_lens);
}

struct Qwen3_5DecodeGraph::Impl {
  Impl(const Qwen3_5MoeWeights& w, const HfConfig& c, vt::Queue q,
       int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the FRAMEWORK-WIDE switch is the SEAM's,
    // not this driver's. `vt::GraphCaptureEnabled()` reads `VLLM_CPP_CUDAGRAPH`
    // once per process into a function-local static. These were the LAST TWO of
    // the six batched-driver reads the spec's `## Our baseline` item 1 counted.
    Backend& b = vt::GetBackend(queue.device.type);
    // #2812/#1625 gate: the captured arm joins the evidence families through
    // the arch-scoped overload, so ambient captures and VT_TT_DECODE_CAPTURE=0
    // opts out — the same shape as the qwen3 driver's gate.
    enabled = vt::GraphCaptureEnabled() &&
              vllm::platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !vllm::platforms::GetPlatform(queue.device.type)
                   .static_graph_requires_opt_in(config.architectures) &&
              b.SupportsGraphCapture();
    dbuf = enabled && DecodeGraphDoubleBufferEnabled();
    poison = enabled && std::getenv("VT_ASYNC_EXECUTOR_POISON") != nullptr;
  }
  ~Impl() {
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      for (auto& s : kv.second.slot) {
        // No DestroyGraph: every segment handle belongs to the slot's
        // `vt::BreakableGraph`, whose destructor releases it through
        // `Backend::DestroyGraph`. That routing is what lets
        // ENG-CUDAGRAPH-DEDUP (#1162) interpose at the backend later without
        // editing this driver (spec `## Risks/decisions` D4).
        // #2274: the graph dies with this slot, so its baked scratch stops being
        // baked. Give the pinned blocks back before the pool outlives us.
        Pool(b).UnpinForGraph(b, s.pinned);
        s.pinned.clear();
        if (s.reuse_event.handle != nullptr) b.DestroyEvent(s.reuse_event);
      }
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses, so each size needs
  // its own fixed-address buffers), its persistent embed target + logits output,
  // and its instantiated graph. The state machine per slot mirrors the original
  // single-shape driver: cold (eager pre-warm) → warm (capture+replay) → replay.
  struct SizeSlot {
    std::vector<int32_t> token_ids;   // [S]
    std::vector<int32_t> positions;   // [S]
    v1::CommonAttentionMetadata attn_meta;
    v1::GDNAttentionMetadata gdn_meta;
    std::unique_ptr<DBuf> hidden;     // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;     // [S,vocab] f32 held graph output
    // SPEC-DSPARK W8 (#442): the DFlash/DSpark verify must ALSO emit the
    // [S, H*taps] aux hidden capture the drafter conditions on. That is exactly
    // why the verify never reached this graph: ForwardDeviceMultiTap returns
    // BEFORE the decode-graph gate. Held PERSISTENTLY per slot (like `logits`) so
    // the captured region writes a fixed address across replays -- the eager path
    // allocates a fresh buffer per call, which a captured graph cannot do.
    std::unique_ptr<DBuf> aux;        // [S, H*taps] bf16, null when no aux tap
    int64_t aux_taps = 0;             // tap count this buffer + graph were built for
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the instantiated graph, its handle
    // ownership, its release and its `captured()` state now live in the shared
    // seam instead of a raw `void*` plus a `bool` this driver maintained by
    // hand. `vt::BreakableGraph` is non-copyable and is constructed in place.
    vt::BreakableGraph graph;
    int fa_cols = -1;                 // captured block-table column count
    bool warm = false;
    int64_t replays = 0;
    // #1380: what the EAGER step at THIS shape demanded of the scratch pool, per
    // size class. Taken at the end of the cold step and handed back to
    // `DevicePool::PreGrowForCapture` immediately before this slot's capture, so
    // every allocation the captured forward makes is a pool HIT. Per SLOT rather
    // than per pool because two shapes interleave through one pool and the pool's
    // own last-step state would answer for whichever step ran most recently.
    DevicePool::StepDemand demand;
    // #2274: the pool blocks this slot's CAPTURE baked, held OUT of the free
    // list for as long as the graph that baked them lives. Taken right after the
    // capture succeeds and given back at every site that drops the graph. The
    // counterpart to the `PreGrowForCapture` above: pre-grow guarantees the
    // blocks exist before the capture, this keeps them from being handed to
    // anyone else after it.
    std::vector<std::pair<size_t, void*>> pinned;
    // VT_ASYNC_EXECUTOR (Option A) input-staged event: recorded on the main queue
    // right AFTER StageStepInputs enqueues the async H2D (NOT after the replay), and
    // host-waited before this slot's next Refresh — so the wait costs only the tiny
    // staging copy, never the GPU tail (the c16/c32 unlock). Blocking-sync flavor;
    // null-handle (unused) unless the double-buffer is on.
    vt::Event reuse_event{};
    // Option A persistent per-step device inputs the captured graph reads (positions/
    // slot_mapping/block_table/seq_lens/qsl/gdn_state_idx + stubs + cos|sin). Built
    // once at capture; the H2D is staged into these OUTSIDE the captured region each
    // step. Null (unused) unless the double-buffer is on.
    std::unique_ptr<StepDevInputs> dev;
    // Option A pinned host staging (the true-async H2D source). Sized once at capture.
    PinnedStepInputs pin;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const v1::CommonAttentionMetadata& am,
                 const v1::GDNAttentionMetadata& gm) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
      // W10 (#1857): the spec-as-decode classification must survive the slot
      // copy, or the captured verify silently re-routes onto the prefill lane.
      attn_meta.uniform_spec_query_len = am.uniform_spec_query_len;
      CopyInPlace(gdn_meta.non_spec_state_indices_tensor,
                  gm.non_spec_state_indices_tensor);
      CopyInPlace(gdn_meta.non_spec_query_start_loc, gm.non_spec_query_start_loc);
      CopyInPlace(gdn_meta.has_initial_state, gm.has_initial_state);
      CopyInPlace(gdn_meta.prefill_query_start_loc, gm.prefill_query_start_loc);
      CopyInPlace(gdn_meta.prefill_state_indices, gm.prefill_state_indices);
      CopyInPlace(gdn_meta.prefill_has_initial_state, gm.prefill_has_initial_state);
      CopyInPlace(gdn_meta.batch_ptr, gm.batch_ptr);
      CopyInPlace(gdn_meta.token_chunk_offset_ptr,
                  gm.token_chunk_offset_ptr);
      gdn_meta.num_prefills = gm.num_prefills;
      gdn_meta.num_prefill_tokens = gm.num_prefill_tokens;
      gdn_meta.num_decodes = gm.num_decodes;
      gdn_meta.num_decode_tokens = gm.num_decode_tokens;
      gdn_meta.num_spec_decodes = gm.num_spec_decodes;
      gdn_meta.num_spec_decode_tokens = gm.num_spec_decode_tokens;
      // Spec-decode segmentation (SPEC-MTP I5a). This comment used to say "spec
      // never captures -- ValidateGdnDecodeGraphState rejects a spec batch", and
      // that stopped being true when SPEC-DSPARK W8 (#442) RE-EXPRESSED that
      // assertion to admit a uniform pure spec batch (`:469-497`). These copies
      // are therefore INERT ONLY on the pure-decode path, where the optionals are
      // empty and CopyInPlace is a no-op. On a captured SPEC step they carry the
      // step's real segmentation and are load-bearing. Corrected while auditing
      // the graph layer for MTP depth (#81), and the residual ambiguity that
      // audit found is [#1020]: the slot ring is keyed on S alone, and two
      // different uniform query lengths can now reach one key.
      CopyInPlace(gdn_meta.spec_state_indices_tensor, gm.spec_state_indices_tensor);
      CopyInPlace(gdn_meta.spec_query_start_loc, gm.spec_query_start_loc);
      CopyInPlace(gdn_meta.spec_sequence_masks, gm.spec_sequence_masks);
      CopyInPlace(gdn_meta.spec_token_indx, gm.spec_token_indx);
      CopyInPlace(gdn_meta.non_spec_token_indx, gm.non_spec_token_indx);
      CopyInPlace(gdn_meta.num_accepted_tokens, gm.num_accepted_tokens);
      gdn_meta.spec_state_indices_num_cols = gm.spec_state_indices_num_cols;
      gdn_meta.num_actual_tokens = gm.num_actual_tokens;
    }
  };

  const Qwen3_5MoeWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;
  bool dbuf = false;         // VT_ASYNC_EXECUTOR parity ring (2 slots/size + events)
  bool poison = false;       // VT_ASYNC_EXECUTOR_POISON: deterministic hazard-C proof

  // The parity ring: two independent SizeSlots per padded size, alternated per
  // step (`next`). OFF (dbuf==false) always uses slot[0] with no event — the
  // second slot stays default-constructed (no graph, no buffers) and the driver is
  // byte-identical to the single-slot original.
  struct SlotRing {
    SizeSlot slot[2];
    int next = 0;
  };
  std::map<DecodeGraphSlotKey, SlotRing> slots;  // (S, q, spec) -> parity ring
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3_5DecodeGraph::Qwen3_5DecodeGraph(const Qwen3_5MoeWeights& weights,
                                       const HfConfig& config, vt::Queue queue,
                                       int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3_5DecodeGraph::~Qwen3_5DecodeGraph() = default;

bool Qwen3_5DecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3_5DecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3_5DecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, Qwen3_5AuxTaps* aux_out) {
  CheckPagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                    gdn_state, impl_->weights, impl_->config);
  const int64_t B = static_cast<int64_t>(token_ids.size());
  const bool has_gdn = HasGdnConsumers(impl_->weights);
  if (has_gdn) detail::ValidateGdnDecodeGraphState(gdn_meta, gdn_state, B);
  else ValidateFullAttnStepMetadata(B, attn_meta);
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  // #1380: open a fresh demand measurement for this step. `PreGrowForCapture`
  // reads the profile the COLD step at this shape recorded, and a profile is a
  // property of one step, so the boundary is here and not inside a branch.
  Pool(b).MarkStepBoundary();
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // The step returns the [B, vocab] real-row logits ON DEVICE — NO full-logits
  // D2H / Synchronize here (removed: the per-step drain). The captured/warm paths
  // return a NON-owning view over the slot's persistent [S,vocab] logits (the
  // first B rows are the real requests; the padded rows follow). Stream ordering
  // guarantees the sampler's later reads see the replay's writes; the next
  // same-size replay overwrites the buffer, so in-place sampler mutation is safe.
  // SPEC-DSPARK W8 (#442): a uniform SPEC batch carries B = num_reqs * (1+k)
  // tokens, which exceeds max_num_reqs and would make PadToCaptureSize return -1
  // (eager). Capture its EXACT shape instead of padding: upstream pads only in
  // whole multiples of the uniform query length (cudagraph_dispatcher.py:145
  // asserts `num_tokens_padded % uniform_decode_query_len == 0`), and taking the
  // exact shape trivially satisfies that while keeping the padded-row inertness
  // question out of the spec path. The shape count stays bounded by max_num_seqs
  // because num_reqs is.
  const bool spec_step = has_gdn && gdn_meta.num_spec_decodes > 0;
  const int64_t S = spec_step ? B : PadToCaptureSize(B, impl_->max_num_reqs);
  // ENG-CUDAGRAPH-BREAK W6 (#1374): this step's uniform query length, and the
  // ring key built from it. `Q == 0` means the batch does not divide evenly into
  // its requests, which no captured decode shape describes.
  const int64_t Q = (attn_meta.num_reqs > 0 && B % attn_meta.num_reqs == 0)
                        ? B / attn_meta.num_reqs
                        : 0;
  const DecodeGraphSlotKey key{S, Q, spec_step};
  // A q > 1 batch that carries NO spec segmentation is refused rather than
  // captured, and this is a TIGHTENING that arrived with the widening. `S` for a
  // non-spec step is `PadToCaptureSize(B)` and `BuildPaddedDecode` rewrites the
  // metadata as a pure decode of S single-token requests, which is not what a
  // multi-token-per-request batch is. The predicate never sent one here, but it
  // was the predicate saying so and nothing in the driver.
  const bool servable_shape = Q >= 1 && (Q == 1 || spec_step);
  const bool qlen_capped = DecodeGraphQueryLenCapped(impl_->slots, key);
  if (qlen_capped) v1::NoteDecodeGraphQueryLenDecline();
  if (!impl_->enabled || S < 0 || !servable_shape || qlen_capped ||
      (has_gdn && !detail::CanUseGdnDecodeGraphSize(
          B, S, IndexedGdnStateIoEnabled(impl_->queue.device)))) {
    if (aux_out != nullptr && !aux_out->layer_ids.empty()) {
      // The graph cannot serve this batch (disabled / unsupported size), so fall
      // back to the EAGER multi-tap forward, which fills aux_out itself. Without
      // this the drafter sees no taps at all.
      return Qwen3_5Model::ForwardDeviceMultiTap(
          token_ids, positions, attn_meta, gdn_meta, attn_kv, gdn_state,
          impl_->weights, impl_->config, impl_->queue, aux_out, {});
    }
    // W4d (tenstorrent-27b-decode-shape-capture.md): a MIXED step the graph
    // declines (it captures only the decode shape, Q == 1) runs eagerly here --
    // pass the per-sequence last-token indices so the head gathers
    // [num_reqs, vocab] instead of asking for the [T, vocab] plane the 27B
    // banks cannot hold.
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                          gdn_state, impl_->weights, impl_->config,
                          LastTokenLogitsIndices(attn_meta, B));
    // ForwardBody returns [B,vocab] (owned pool block; hand ownership out).
    return WrapDeviceLogits(d, std::move(lg), vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then
  // refresh THIS size's persistent host buffers in place. VT_ASYNC_EXECUTOR: pick
  // this step's slot from the size's parity ring (alternating), and host-wait its
  // previous replay before Refresh touches its persistent host inputs (hazard-C).
  // OFF: always slot[0], no wait — byte-identical to the single-slot driver.
  const bool new_shape = impl_->slots.find(key) == impl_->slots.end();
  Impl::SlotRing& ring = impl_->slots[key];
  // W6: a distinct captured shape is the observable the widening is measured by.
  // Without it a two-shape driver and a one-shape driver are indistinguishable
  // from outside, which is how #1020 stayed silent.
  if (new_shape) v1::NoteDecodeGraphShape();
  // SPEC-DSPARK W8 (#442): a captured SPEC graph must read PERSISTENT step
  // inputs. The per-call StepDevInputs is a pool block freed when the call
  // returns, so a graph captured against it replays over recycled memory.
  const bool dbuf = impl_->dbuf || spec_step;
  Impl::SizeSlot& s = ring.slot[dbuf ? ring.next : 0];
  if (dbuf) {
    ring.next ^= 1;
    // Wait this slot's input-staged event (2 steps back) before Refresh overwrites its
    // host inputs / StageStepInputs re-fills its pinned buffers — the staged copy that
    // read them must be done. It is a tiny copy at the FRONT of a 2-steps-back replay,
    // so this is effectively free (never the GPU tail; that is the Option A unlock).
    if (s.reuse_event.handle != nullptr) b.SynchronizeEvent(s.reuse_event);
  }
  // Record this slot's input-staged event on the main queue right AFTER the H2D stage
  // (Option A), so the next same-slot Refresh waits only that tiny copy. No-op unless
  // the double-buffer is on.
  const auto record_staged = [&] {
    if (!dbuf) return;
    if (s.reuse_event.handle == nullptr)
      s.reuse_event = b.CreateEvent(/*blocking=*/true);
    b.RecordEvent(s.reuse_event, impl_->queue);
  };

  // SPEC-DSPARK W8 (#442): size this slot's PERSISTENT aux buffer and invalidate a
  // graph captured for a different tap count. Only the EXACT-shape case (S == B,
  // which is what a spec batch takes) is served: the aux contract is [T, H*taps]
  // and padded rows would change T.
  const int64_t aux_taps =
      aux_out != nullptr ? static_cast<int64_t>(aux_out->layer_ids.size()) : 0;
  if (s.aux_taps != aux_taps) {
    // Reset() releases every segment through Backend::DestroyGraph and returns
    // the container to its as-constructed state, which is also what lets the
    // next capture open a scope on it (the scope refuses a container that
    // already holds one).
    s.graph.Reset();
    Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
    s.pinned.clear();
    s.warm = false;
    s.aux.reset();
    s.aux_taps = aux_taps;
  }
  if (aux_taps > 0 && (s.aux == nullptr || s.aux->t().shape[0] != S)) {
    s.aux = std::make_unique<DBuf>(d, DType::kBF16,
                                   std::vector<int64_t>{S, H * aux_taps});
  }
  Tensor aux_view{};
  const std::vector<int32_t>* aux_ids_arg = nullptr;
  const Tensor* aux_out_arg = nullptr;
  if (aux_taps > 0) {
    ValidateAuxTapLayerIds(aux_out->layer_ids, impl_->config.num_hidden_layers);
    aux_view = s.aux->t();
    aux_ids_arg = &aux_out->layer_ids;
    aux_out_arg = &aux_view;
  }
  // The captured region writes into the slot's buffer, so the drafter reads a
  // NON-owning view valid until this slot's next replay -- the same lifetime the
  // returned logits already carry.
  const auto publish_aux = [&] {
    if (aux_taps > 0) {
      aux_out->storage.reset();
      aux_out->tensor = s.aux->t();
    }
  };
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  v1::CommonAttentionMetadata pam;
  v1::GDNAttentionMetadata pgm;
  if (spec_step) {
    // S == B for a spec batch, so there is nothing to pad, and BuildPaddedDecode
    // would REWRITE the metadata as pure non-spec decode (num_decodes = S,
    // spec fields dropped). Carry the real metadata through untouched.
    ptok = token_ids;
    ppos = positions;
    pam = attn_meta;
    pgm = gdn_meta;
  } else {
    // The padding helper copies GDN indices into S entries. Without consumers,
    // the caller's unused metadata has no validated bound and must stay inert.
    BuildPaddedDecode(S, token_ids, positions, attn_meta,
                      has_gdn ? gdn_meta : GDNAttentionMetadata{}, ptok, ppos,
                      pam, pgm);
  }

  // A block-table column-count change reallocates the persistent block_table (the
  // staged/baked H2D dest shape moves) → invalidate this slot's graph and persistent
  // device inputs, and re-warm/re-capture.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam, pgm);
  s.fa_cols = cols;
  if (cols_changed && s.graph.captured()) {
    s.graph.Reset();
    Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
    s.pinned.clear();
    s.warm = false;
    // UNBIND BEFORE THE DESTINATION DIES. `s.pin`'s cells name device pointers
    // that `s.dev` owns, so dropping `s.dev` first leaves them naming freed
    // pool blocks. `Unbind()` does not dereference the destination today, so
    // the old order was correct by the implementation's silence rather than by
    // anything stated — which is one edit away from a use-after-free.
    s.pin.Free();
    s.dev.reset();
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, stage this step's inputs into the persistent device
  // buffers (Option A: H2D out of the captured region), then relaunch the graph.
  if (s.graph.captured()) {
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    if (dbuf) {
      StageStepInputs(d, s);
      StageSpecStepInputs(d, s);
      record_staged();
      MaybePoisonStagedInputs(impl_->poison, s);
    }
    // Through the seam's container, never `Backend::ReplayGraph` directly: the
    // container replays its segments in order (one, here, because a decode
    // capture is kFull) and owns the G3 replay counter the gate reads.
    s.graph.Replay(impl_->queue);
    ++s.replays;
    ++impl_->replays;
    publish_aux();
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + residency were warmed for this size by the previous (eager)
  // step. CAPTURE the layer region once, instantiate the graph, then launch it.
  if (s.warm) {
    // #1380: THE POOL MUST BE ABLE TO SERVE THE WHOLE CAPTURED FORWARD, not one
    // block of one tensor. This used to alloc-and-free a single [S, vocab] f32
    // block, on the reasoning that the capture RETAINS its logits while the
    // other ring slot still holds its own, so the free list is one short. The
    // retention half is right and the "one block of that shape" half is not:
    // `DevicePool` is keyed by SIZE CLASS, so the block the capture then misses
    // on need not be the logits. Measured on `thor:gpu0` (sm_110), the second
    // ring slot's capture died in `dconv`, the GDN causal-conv output
    // (`GdnBlockPaged`, this file), whose [T, conv_dim] lands in the SAME class
    // as the retained [S, vocab] logits at the gate's shape -- and a cudaMalloc
    // inside a captured region aborts the capture. An alloc-and-free grows the
    // pool only when that class's free list is EMPTY, so with one block free
    // and TWO needed live at once it grew nothing at all.
    //
    // So the driver stops naming a tensor and asks the pool for the demand the
    // EAGER step at this shape actually made (`s.demand`, taken at the end of
    // the cold step). Working scratch is still freed at ForwardLayers return and
    // still SAFELY shared between the two graphs -- they replay sequentially on
    // one stream. This runs OUTSIDE the capture, which is where a cudaMalloc is
    // legal, and it is a no-op once the free list is deep enough.
    //
    // #2029: IT RUNS BEFORE EVERY CAPTURE, and until this line moved it ran
    // before almost none of them. It sat inside the `if (dbuf)` below, and
    // `dbuf` is `impl_->dbuf || spec_step` -- the `VT_ASYNC_EXECUTOR` parity
    // ring, or a speculative step. The DEFAULT server sets neither, so the lane
    // a user gets by omitting `--speculative-config` opened its capture over a
    // pool nobody had prepared, and #1380's own failure came back at c=8 as
    // `cudaMalloc: operation not permitted when stream is capturing` while the
    // SAME binary with a speculative config served. The guard was never about
    // the pre-grow: that block exists for the parity ring's drain and its
    // persistent step inputs, and #1393 replaced the pre-grow in place without
    // revisiting what it sat under.
    //
    // The `!dbuf` arm needs this at least as much as the `dbuf` arm. It passes
    // `persistent_sdi == nullptr` to ForwardLayers, so `BuildStepDevInputs`
    // and `MaybeBuildAttnCosSin` run from the MAIN pool INSIDE the captured
    // region -- and that is also what makes `s.demand` exact for it, because the
    // cold step takes the identical `nullptr` branch. On the `dbuf` arm the
    // captured region's main-pool demand is a strict SUBSET of the cold step's,
    // which is the containment #1393 recorded.
    //
    // Ordering: this now precedes the `b.Synchronize` below rather than
    // following it. A pre-grow is `Backend::Alloc` and nothing else -- it
    // enqueues no work and reads no in-flight buffer -- and the drain still
    // happens before `BeginCapture`, which is the property it was added for.
    Pool(b).PreGrowForCapture(b, s.demand);
    // dbuf: the runner may have skipped the depth-2 drain (the previous step
    // returned a slot view), so a prior replay can still be in flight. Capture must
    // begin on an idle stream — drain once here. One-time (≤2 captures per size);
    // steady-state replay never captures, so this never touches the overlap path.
    if (dbuf) {
      b.Synchronize(impl_->queue);
      // Option A: build this slot's PERSISTENT device inputs + pinned staging OUTSIDE
      // the capture. BuildStepDevInputs fills s.dev from the refreshed host vectors so
      // the capture step reads correct inputs; the persistent cos|sin (fused-preamble
      // arch) is allocated + filled here (pre-capture) and only RE-FILLED inside the
      // captured region (FillAttnCosSin), so nothing allocates during capture. These
      // RETAINED buffers draw from a DEDICATED pool (not the main scratch Pool()) so
      // they never pop-and-hold a block the captured forward's scratch then needs.
      const int64_t gdn_state_slots =
          gdn_state.empty() ? 0 : gdn_state.front().ssm_state.shape[0];
      const bool fp4_attn = [&] {
        for (const auto& l : impl_->weights.layers)
          if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
        return false;
      }();
      // UNBIND BEFORE THE DESTINATION DIES, the same rule the shape-change path
      // above states, applied to the other site that replaces `s.dev`. The
      // assignment below destroys the PREVIOUS `StepDevInputs`, returning its
      // pool blocks to the free list while `s.pin`'s cells still name them, and
      // the rebind is four statements later at `s.pin.Alloc`. Reachable: the
      // `aux_taps` branch resets the graph and clears `warm` WITHOUT calling
      // `s.pin.Free()`, so the next warm step arrives here with the cells bound.
      // Harmless today only because `Unbind()` does not dereference — which is
      // exactly the argument the shape-change site refused to rely on. Free costs
      // nothing extra: `Alloc` calls `Free()` first regardless.
      s.pin.Free();
      {
        ActivePoolScope persistent_scope(&PersistentDecodeInputPool(d.b));
        s.dev = std::make_unique<StepDevInputs>(
            has_gdn ? BuildStepDevInputs(d, s.positions, s.attn_meta, s.gdn_meta,
                                         gdn_state_slots)
                    : BuildFullAttnStepDevInputs(d, s.positions, s.attn_meta));
        MaybeBuildAttnCosSin(d, *s.dev, impl_->config, S, fp4_attn);
      }
      const bool has_idx = s.dev->has_gdn_idx &&
                           s.gdn_meta.non_spec_state_indices_tensor.has_value();
      const int64_t idx =
          has_idx ? static_cast<int64_t>(
                        s.gdn_meta.non_spec_state_indices_tensor->size())
                  : 0;
      s.pin.Alloc(b, *s.dev, S, s.attn_meta.num_reqs, cols, idx, has_idx);
    }
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope owns
    // the segment, the handle, its release and the drain a mid-capture throw
    // needs — the drain three drivers each hand-rolled as the same
    // `try { EndCaptureGraph(); } catch (...) {}`, and which the scope now
    // performs by comparing `std::uncaught_exceptions()` against the depth it
    // recorded at entry. A refusal raised inside the forward (the fp8 splitK
    // premise check is one such, and not the only one) still reaches the caller
    // with the ORIGINAL error, because nothing here swallows it.
    //
    // kFULL, and the mode is the whole argument. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63`) is documented at
    // `:630-632` as a FULL graph for DECODE batches and a piecewise one for
    // prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the full
    // half. This is a decode driver, so its capture is ONE segment with the
    // attention calls INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every layer's attention into an
    // eager call between two graph replays, which is not vLLM's decode behaviour
    // and which nothing in this row's record supports.
    std::optional<DBuf> lg;
    {
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                         s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                         impl_->config, {}, nullptr, /*mrope_cos_sin=*/nullptr,
                         aux_ids_arg, aux_out_arg,
                         dbuf ? s.dev.get() : nullptr);
    }  // ~GraphCaptureScope closes the segment and files it on s.graph
    // NOT CAPTURED covers TWO states, and returning the buffer is correct for
    // exactly one of them. `~GraphCaptureScope` must swallow a throwing
    // `EndCaptureGraph` — a destructor that propagates terminates — so a FAILED
    // capture leaves the container reporting what an INERT scope reports.
    //
    //   * INERT (`capture_failed() == false`): the backend cannot capture, or
    //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, the layer
    //     region above ran EAGERLY, and the buffer is a real result. Return it
    //     and go back to cold so the next same-size step re-warms.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw.
    //     Under stream capture NOTHING between `BeginCapture` and the throw
    //     executed: every kernel was RECORDED, so the buffer is pool-recycled
    //     memory. Returning it would hand this step uncomputed device memory as
    //     its logits — silently wrong tokens, no fault, and a token gate cannot
    //     see it.
    //
    // So the failure PROPAGATES, carrying the runtime's own exception. This is
    // exactly what the pre-W4 driver did (`s.graph = b.EndCaptureGraph(...)` was
    // unguarded), and it is not a recovery this stage can justify inventing: a
    // stream whose capture was INVALIDATED has not told us it is usable.
    if (!s.graph.captured()) {
      s.warm = false;  // either way this slot goes back to cold
      if (s.graph.capture_failed()) {
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();  // clear the failure with the graph it described
        Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
        s.pinned.clear();
        // The runtime's OWN diagnosis where the seam holds it. It is empty only
        // on the arm where an exception was already propagating THROUGH the
        // scope, which cannot reach this line; the refusal below makes that
        // unreachability an assertion rather than a claim.
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "Qwen3.5 decode graph: the capture was ABANDONED and its logits were "
                 "never computed; refusing to return uncaptured device memory");
      }
      publish_aux();
      record_staged();
      ForwardLogits drained = WrapDeviceLogits(d, std::move(*lg), vocab);
      if (drained.rows != B) {
        drained.rows = B;
        drained.device_tensor = MakeTensor(drained.device_storage.get(), DType::kF32,
                                           d.q.device, {B, vocab});
      }
      return drained;
    }
    // #2274 THE FIX. The capture succeeded, so from here the graph's replays
    // write through the device pointers it just baked. `ForwardLayers` has
    // already RETURNED its working scratch to the pool's free list, and the
    // comment at `PreGrowForCapture` above argues that is safe because the two
    // graphs "replay sequentially on one stream". That argument covers the two
    // GRAPHS. It does not cover a THIRD party: a DFlash2 draft store created
    // later takes a pool block for its block table, is handed one of these
    // blocks, and the next replay writes f32 activations over the table -- which
    // is why a live block table reads `bt[0] = 1060730955` (0.727f) and
    // `vt::PagedAttention` indexes page 1.06e9 in a 513-page pool.
    //
    // So take the capture's demand back OUT of the free list. `PreGrowForCapture`
    // guaranteed `s.demand` free blocks per class BEFORE the capture and nothing
    // else allocated in between, so the free list is LIFO-ordered with exactly
    // those blocks on top; this pops the same count off the same end.
    //
    // BEFORE the `s.logits` assignment below, and the order is load-bearing on
    // the RE-capture path (a column-count change resets this slot's graph, which
    // is what a new draft store's wider block table causes). That assignment
    // destroys the previous `s.logits` and frees ITS block to the same free list,
    // where it would be popped first and leave one of the graph's own blocks
    // reachable. Freeing the old logits is correct -- the graph it belonged to
    // was Reset -- it just must not happen while we are counting.
    if (s.pinned.empty()) s.pinned = Pool(b).PinForGraph(b, s.demand);
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    s.graph.Replay(impl_->queue);
    record_staged();
    s.replays = 1;
    ++impl_->replays;
    publish_aux();
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool + resident weights /
  // fused-MoE constants for this size) and defer capture to the next same-size
  // step. This is a real decode step (its padded output's real rows are used;
  // nothing is wasted). (Re)allocate the persistent hidden buffer to this size.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, s.gdn_meta,
                          attn_kv, gdn_state, impl_->weights, impl_->config, {},
                          nullptr, /*mrope_cos_sin=*/nullptr, aux_ids_arg,
                          aux_out_arg);
  s.warm = true;
  // #1380: the cold step is the ONE eager run of this exact forward at this exact
  // shape, so it is where the capture's allocation demand is measurable. Recorded
  // per SLOT and consumed by `PreGrowForCapture` at this slot's capture. Taken
  // AFTER the forward and BEFORE the logits leave: the profile is a per-class
  // PEAK, so it already holds every block this step had live at once.
  s.demand = Pool(b).StepDemandProfile();
  publish_aux();
  record_staged();
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), vocab);
  if (fl.rows != B) {
    fl.rows = B;
    fl.device_tensor =
        MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  }
  return fl;
}

// ─── tt-27b-region-capture: the region-scoped decode-capture arm ─────────────
// The 27B decode graph does not fit ONE whole-graph trace: end_trace_capture
// asks for one ~3.15 GB staging buffer against ~298 MB free (the spec's
// `## Scope` census, 1,037 recorded commands), and the whole-graph arm serves
// nothing. Region scope splits the same command stream into ONE REGION PER
// LAYER: the decode driver opens its capture kPiecewise and DenseForwardLayers
// emits an in-place boundary after each layer (`vt::GraphBreak()`, the bare
// form — no eager call, no destination; the layer outputs flow device-side
// through the SAME persistent buffers a whole-graph capture bakes, which is
// exactly the in-place handoff discipline the #3327 class demands — no region
// boundary installs, frees, or re-shadows a state tensor). The replay is the
// container's host loop: segment, (no-op break), segment, ... — the per-region
// runtime-arg re-patch the RAC per-user mechanism already serves, because the
// RAC/rope hooks read the SAME persistent device inputs every segment bakes.
// Sizing: 1,037 commands / 64 layers ≈ 16.2 commands per layer ≈ 48.6 MiB at
// the measured 3.04 MB per command — the GDN precedent's 50 MiB region budget
// (tenstorrent_capture.cpp:90), asserted per region from the probe-fed census
// (`BreakableGraph::region_bytes()`), with an over-cap region declining the
// capture BY NAME (below).
// OFF by default in this slice (`VLLM_CPP_REGION_CAPTURE=1` opts in): the fit
// predicate's automatic model-by-model wiring (`vt::WholeGraphTraceFits`) is
// the next wave — wiring it now would re-route the 9B whole-graph arm the
// census cannot yet price per model.
static bool RegionCaptureRequested() {
  static const bool v = [] {
    const char* e = std::getenv("VLLM_CPP_REGION_CAPTURE");
    return e != nullptr && e[0] != '\0' && std::string_view(e) != "0";
  }();
  return v;
}
// The GDN precedent's fit number (tenstorrent_capture.cpp:90).
constexpr int64_t kRegionCaptureBudgetBytes = 50 * 1024 * 1024;

// ─── Qwen3_5DenseDecodeGraph (27B dense decode CUDA-graph driver) ────────────
// The 27B DENSE sibling of Qwen3_5DecodeGraph. Same cold→warm→replay state
// machine, same padded-batch capture set (kDecodeGraphSizes), same persistent
// fixed-address host inputs and persistent embed/logits buffers — but it drives
// the dense forward (DenseForwardLayers over DenseEmbedInto) instead of the MoE
// forward. It reuses the weight-agnostic PadToCaptureSize / BuildPaddedDecode
// verbatim. The dense W4A4 projections' per-call scratch is already persistent
// (the cutlass StreamScratch pool / the emulation path's pool-backed DBufs / the
// cublas lm_head), so a cold pre-warm at each size makes the capture region do
// ZERO cudaMalloc — mirroring the MoE path's EnsureMoeScratch/EnsureCtmp/pool
// discipline. The 35B MoE graph is UNTOUCHED.


struct Qwen3_5DenseDecodeGraph::Impl {
  Impl(const Qwen3_5DenseWeights& w, const HfConfig& c, vt::Queue q,
       int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the FRAMEWORK-WIDE switch is the SEAM's,
    // not this driver's. `vt::GraphCaptureEnabled()` reads `VLLM_CPP_CUDAGRAPH`
    // once per process into a function-local static. These were the LAST TWO of
    // the six batched-driver reads the spec's `## Our baseline` item 1 counted.
    Backend& b = vt::GetBackend(queue.device.type);
    // #2812/#1625 gate: the captured arm joins the evidence families through
    // the arch-scoped overload, so ambient captures and VT_TT_DECODE_CAPTURE=0
    // opts out — the same shape as the qwen3 driver's gate.
    enabled = vt::GraphCaptureEnabled() &&
              vllm::platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !vllm::platforms::GetPlatform(queue.device.type)
                   .static_graph_requires_opt_in(config.architectures) &&
              b.SupportsGraphCapture();
    dbuf = enabled && DecodeGraphDoubleBufferEnabled();
    poison = enabled && std::getenv("VT_ASYNC_EXECUTOR_POISON") != nullptr;
  }
  ~Impl() {
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr, "[DenseDecodeGraph] Qwen3.5 dense decode graph: %lld total "
                           "replays across %zu captured size(s)\n",
                   static_cast<long long>(replays), slots.size());
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      for (auto& s : kv.second.slot) {
        // No DestroyGraph: every segment handle belongs to the slot's
        // `vt::BreakableGraph`, whose destructor releases it through
        // `Backend::DestroyGraph`. That routing is what lets
        // ENG-CUDAGRAPH-DEDUP (#1162) interpose at the backend later without
        // editing this driver (spec `## Risks/decisions` D4).
        // #2274: the graph dies with this slot, so its baked scratch stops being
        // baked. Give the pinned blocks back before the pool outlives us.
        Pool(b).UnpinForGraph(b, s.pinned);
        s.pinned.clear();
        if (s.reuse_event.handle != nullptr) b.DestroyEvent(s.reuse_event);
      }
  }

  // One captured padded batch size (mirror of Qwen3_5DenseDecodeGraph SizeSlot).
  struct SizeSlot {
    std::vector<int32_t> token_ids;   // [S]
    std::vector<int32_t> positions;   // [S]
    v1::CommonAttentionMetadata attn_meta;
    v1::GDNAttentionMetadata gdn_meta;
    std::unique_ptr<DBuf> hidden;     // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;     // [S,vocab] f32 held graph output
    // SPEC-DSPARK W8 (#442): the DFlash/DSpark verify must ALSO emit the
    // [S, H*taps] aux hidden capture the drafter conditions on. That is exactly
    // why the verify never reached this graph: ForwardDeviceMultiTap returns
    // BEFORE the decode-graph gate. Held PERSISTENTLY per slot (like `logits`) so
    // the captured region writes a fixed address across replays -- the eager path
    // allocates a fresh buffer per call, which a captured graph cannot do.
    std::unique_ptr<DBuf> aux;        // [S, H*taps] bf16, null when no aux tap
    int64_t aux_taps = 0;             // tap count this buffer + graph were built for
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the instantiated graph, its handle
    // ownership, its release and its `captured()` state now live in the shared
    // seam instead of a raw `void*` plus a `bool` this driver maintained by
    // hand. `vt::BreakableGraph` is non-copyable and is constructed in place.
    vt::BreakableGraph graph;
    int fa_cols = -1;                 // captured block-table column count
    bool warm = false;
    // tt-27b-region-capture: a named per-region over-budget DECLINE (see the
    // census below) is sticky for this size — an over-budget layer's command
    // stream does not shrink between steps, so re-capturing every step would
    // be the boundary storm the spec's risk names. The slot serves EAGER.
    bool region_declined = false;
    int64_t replays = 0;
    // R2: the cur_pos the device held after this slot's last seeding step or
    // replay (WarmDecodePos continuation predicate, qwen3.cpp #2469).
    int64_t expected_cur_pos = -1;
    // #1380: what the EAGER step at THIS shape demanded of the scratch pool, per
    // size class. Taken at the end of the cold step and handed back to
    // `DevicePool::PreGrowForCapture` immediately before this slot's capture, so
    // every allocation the captured forward makes is a pool HIT. Per SLOT rather
    // than per pool because two shapes interleave through one pool and the pool's
    // own last-step state would answer for whichever step ran most recently.
    DevicePool::StepDemand demand;
    // #2274: the pool blocks this slot's CAPTURE baked, held OUT of the free
    // list for as long as the graph that baked them lives. Taken right after the
    // capture succeeds and given back at every site that drops the graph. The
    // counterpart to the `PreGrowForCapture` above: pre-grow guarantees the
    // blocks exist before the capture, this keeps them from being handed to
    // anyone else after it.
    std::vector<std::pair<size_t, void*>> pinned;
    // VT_ASYNC_EXECUTOR (Option A) input-staged event: recorded on the main queue
    // right AFTER StageStepInputs enqueues the async H2D (NOT after the replay), and
    // host-waited before this slot's next Refresh — so the wait costs only the tiny
    // staging copy, never the GPU tail (the c16/c32 unlock). Blocking-sync flavor;
    // null-handle (unused) unless the double-buffer is on.
    vt::Event reuse_event{};
    // Option A persistent per-step device inputs the captured graph reads (positions/
    // slot_mapping/block_table/seq_lens/qsl/gdn_state_idx + stubs + cos|sin). Built
    // once at capture; the H2D is staged into these OUTSIDE the captured region each
    // step. Null (unused) unless the double-buffer is on.
    std::unique_ptr<StepDevInputs> dev;
    // Option A pinned host staging (the true-async H2D source). Sized once at capture.
    PinnedStepInputs pin;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const v1::CommonAttentionMetadata& am,
                 const v1::GDNAttentionMetadata& gm) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
      // W10 (#1857): the spec-as-decode classification must survive the slot
      // copy, or the captured verify silently re-routes onto the prefill lane.
      attn_meta.uniform_spec_query_len = am.uniform_spec_query_len;
      CopyInPlace(gdn_meta.non_spec_state_indices_tensor,
                  gm.non_spec_state_indices_tensor);
      CopyInPlace(gdn_meta.non_spec_query_start_loc, gm.non_spec_query_start_loc);
      CopyInPlace(gdn_meta.has_initial_state, gm.has_initial_state);
      CopyInPlace(gdn_meta.prefill_query_start_loc, gm.prefill_query_start_loc);
      CopyInPlace(gdn_meta.prefill_state_indices, gm.prefill_state_indices);
      CopyInPlace(gdn_meta.prefill_has_initial_state, gm.prefill_has_initial_state);
      CopyInPlace(gdn_meta.batch_ptr, gm.batch_ptr);
      CopyInPlace(gdn_meta.token_chunk_offset_ptr,
                  gm.token_chunk_offset_ptr);
      gdn_meta.num_prefills = gm.num_prefills;
      gdn_meta.num_prefill_tokens = gm.num_prefill_tokens;
      gdn_meta.num_decodes = gm.num_decodes;
      gdn_meta.num_decode_tokens = gm.num_decode_tokens;
      gdn_meta.num_spec_decodes = gm.num_spec_decodes;
      gdn_meta.num_spec_decode_tokens = gm.num_spec_decode_tokens;
      // Spec-decode segmentation (SPEC-MTP I5a). This comment used to say "spec
      // never captures -- ValidateGdnDecodeGraphState rejects a spec batch", and
      // that stopped being true when SPEC-DSPARK W8 (#442) RE-EXPRESSED that
      // assertion to admit a uniform pure spec batch (`:469-497`). These copies
      // are therefore INERT ONLY on the pure-decode path, where the optionals are
      // empty and CopyInPlace is a no-op. On a captured SPEC step they carry the
      // step's real segmentation and are load-bearing. Corrected while auditing
      // the graph layer for MTP depth (#81), and the residual ambiguity that
      // audit found is [#1020]: the slot ring is keyed on S alone, and two
      // different uniform query lengths can now reach one key.
      CopyInPlace(gdn_meta.spec_state_indices_tensor, gm.spec_state_indices_tensor);
      CopyInPlace(gdn_meta.spec_query_start_loc, gm.spec_query_start_loc);
      CopyInPlace(gdn_meta.spec_sequence_masks, gm.spec_sequence_masks);
      CopyInPlace(gdn_meta.spec_token_indx, gm.spec_token_indx);
      CopyInPlace(gdn_meta.non_spec_token_indx, gm.non_spec_token_indx);
      CopyInPlace(gdn_meta.num_accepted_tokens, gm.num_accepted_tokens);
      gdn_meta.spec_state_indices_num_cols = gm.spec_state_indices_num_cols;
      gdn_meta.num_actual_tokens = gm.num_actual_tokens;
    }
  };

  const Qwen3_5DenseWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;
  bool dbuf = false;         // VT_ASYNC_EXECUTOR parity ring (2 slots/size + events)
  bool poison = false;       // VT_ASYNC_EXECUTOR_POISON: deterministic hazard-C proof

  // The parity ring: two independent SizeSlots per padded size, alternated per
  // step (`next`). OFF (dbuf==false) always uses slot[0] with no event — the
  // second slot stays default-constructed (no graph, no buffers) and the driver is
  // byte-identical to the single-slot original.
  struct SlotRing {
    SizeSlot slot[2];
    int next = 0;
  };
  std::map<DecodeGraphSlotKey, SlotRing> slots;  // (S, q, spec) -> parity ring
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3_5DenseDecodeGraph::Qwen3_5DenseDecodeGraph(const Qwen3_5DenseWeights& weights,
                                                 const HfConfig& config,
                                                 vt::Queue queue,
                                                 int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3_5DenseDecodeGraph::~Qwen3_5DenseDecodeGraph() = default;

bool Qwen3_5DenseDecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3_5DenseDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3_5DenseDecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, Qwen3_5AuxTaps* aux_out) {
  CheckDensePagedForward(token_ids, positions, attn_meta, gdn_meta, attn_kv,
                         gdn_state, impl_->weights, impl_->config);
  const int64_t B = static_cast<int64_t>(token_ids.size());
  const bool has_gdn = HasGdnConsumers(impl_->weights);
  if (has_gdn) detail::ValidateGdnDecodeGraphState(gdn_meta, gdn_state, B);
  else ValidateFullAttnStepMetadata(B, attn_meta);
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  // TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): the step phase clock. `on` is
  // a getenv compare and every bracket below is `sph.on ? now : t0` — zero
  // cost with the knob unset, on every backend.
  StepPhaseTrace sph;
  sph.on = vt::tenstorrent::StepPhasesEnabled();
  sph.t0 = sph.on ? StepPhaseClk::now() : StepPhaseClk::time_point{};
  sph.B = B;
  if (sph.on && d.q.device.type == vt::DeviceType::kTENSTORRENT)
    sph.dram_free = vt::tenstorrent::DeviceDramFreeBytes() >> 20;
  // #1380: open a fresh demand measurement for this step. `PreGrowForCapture`
  // reads the profile the COLD step at this shape recorded, and a profile is a
  // property of one step, so the boundary is here and not inside a branch.
  Pool(b).MarkStepBoundary();
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // TT-GDN-REGION-REPLAY attribution instrument (VT_TT_GDN_REPLAY_TAP): when
  // set, print — after every decode-graph step completes, between steps and
  // never during capture — a state-slot checksum per GDN cache (vt::tenstorrent
  //::GdnShadowProbe, the VT_TT_GDN_STATE_PROBE pattern extended to the replay
  // step) and the step's logits argmax/zero-count/first-five. The per-layer
  // residual-stream taps were attempted through the SPEC-DSPARK W8 aux buffer
  // and are NOT usable on this lane: the TT Backend::Copy D2D arm re-shadows
  // the aux slot with each tap's [1,H] clone instead of writing the [S,H*taps]
  // buffer (CopyDeviceDeviceIfCapture, tenstorrent_residency.cpp:1260-1280),
  // so the columns read back are pool residue — a pre-existing defect of the
  // aux seam on TT (no DFlash drafter runs here), recorded in the row's
  // evidence, not printed as if it were data. The state slots + logits carry
  // the attribution: slots whose step-N checksum was sane while the step-N+1
  // replay wrote all-zero logits were not read — the trace's baked state-read
  // address was freed by the out-of-place commit that produced the step-N
  // state. Inert (one getenv) when unset.
  const char* tap_knob = std::getenv("VT_TT_GDN_REPLAY_TAP");
  const bool tap_on = tap_knob != nullptr && tap_knob[0] != '\0';
  // The probe print, called at every step's return. `logits_ptr` is the step's
  // [S,vocab] f32 logits buffer (null on arms that never materialize one
  // here). All reads are D2H between steps, never during capture.
  const auto tap_emit = [&](const char* kind, const void* logits_ptr) {
    if (!tap_on) return;
    static std::atomic<int64_t> tap_steps{0};
    const int64_t step = tap_steps.fetch_add(1, std::memory_order_relaxed);
    for (size_t e = 0; e < gdn_state.size(); ++e) {
      char what[48];
      std::snprintf(what, sizeof what, "step=%lld ssm[%zu]",
                    static_cast<long long>(step), e);
      vt::tenstorrent::GdnShadowProbe(gdn_state[e].ssm_state.data, what);
      std::snprintf(what, sizeof what, "step=%lld conv[%zu]",
                    static_cast<long long>(step), e);
      vt::tenstorrent::GdnShadowProbe(gdn_state[e].conv_state.data, what);
    }
    if (logits_ptr != nullptr) {
      std::vector<float> lg(static_cast<size_t>(vocab));
      d.b.Copy(d.q, lg.data(), logits_ptr,
               static_cast<size_t>(vocab) * sizeof(float));
      d.b.Synchronize(d.q);
      int64_t am = 0;
      int64_t zeros = 0;
      for (int64_t i = 0; i < vocab; ++i) {
        if (lg[static_cast<size_t>(i)] == 0.0f) ++zeros;
        if (lg[static_cast<size_t>(i)] > lg[static_cast<size_t>(am)]) am = i;
      }
      // The top-2 margin: a discrete selection's error is bimodal (the
      // VT_GLM5_DIAG precedent). A near-tie (margin ~0) is the alternate-greedy
      // path class; a different distribution (the top-2 far apart and neither
      // near the reference) is a stale binding. This is what adjudicates the
      // 9B's single-token coherent divergence.
      int64_t am2 = am == 0 ? 1 : 0;
      for (int64_t i = 0; i < vocab; ++i)
        if (i != am && lg[static_cast<size_t>(i)] > lg[static_cast<size_t>(am2)])
          am2 = i;
      std::fprintf(stderr,
                   "[GDN-TAP] kind=%s step=%lld LOGITS argmax=%lld(%a) top2=%lld(%a) "
                   "margin=%a zeros=%lld/%lld first5=[%a,%a,%a,%a,%a]\n",
                   kind, static_cast<long long>(step),
                   static_cast<long long>(am), lg[static_cast<size_t>(am)],
                   static_cast<long long>(am2), lg[static_cast<size_t>(am2)],
                   lg[static_cast<size_t>(am)] - lg[static_cast<size_t>(am2)],
                   static_cast<long long>(zeros), static_cast<long long>(vocab),
                   lg[0], lg[1], lg[2], lg[3], lg[4]);
      std::fflush(stderr);
    }
  };

  // Returns the [B,vocab] real-row logits ON DEVICE (no D2H). The captured/warm
  // paths return a NON-owning view over the slot's persistent [S,vocab] logits
  // (first B rows are the real requests). Stream ordering guarantees the sampler
  // sees the replay's writes; the next same-size replay overwrites the buffer.
    if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
      std::fprintf(stderr, "[TT-FWD] DenseGraph::Step B=%lld num_reqs=%lld\n",
                   static_cast<long long>(token_ids.size()),
                   static_cast<long long>(attn_meta.num_reqs));
  // SPEC-DSPARK W8 (#442): a uniform SPEC batch carries B = num_reqs * (1+k)
  // tokens, which exceeds max_num_reqs and would make PadToCaptureSize return -1
  // (eager). Capture its EXACT shape instead of padding: upstream pads only in
  // whole multiples of the uniform query length (cudagraph_dispatcher.py:145
  // asserts `num_tokens_padded % uniform_decode_query_len == 0`), and taking the
  // exact shape trivially satisfies that while keeping the padded-row inertness
  // question out of the spec path. The shape count stays bounded by max_num_seqs
  // because num_reqs is.
  const bool spec_step = has_gdn && gdn_meta.num_spec_decodes > 0;
  const int64_t S = spec_step ? B : PadToCaptureSize(B, impl_->max_num_reqs);
  if (sph.on) sph.S = S;
  // ENG-CUDAGRAPH-BREAK W6 (#1374): this step's uniform query length, and the
  // ring key built from it. `Q == 0` means the batch does not divide evenly into
  // its requests, which no captured decode shape describes.
  const int64_t Q = (attn_meta.num_reqs > 0 && B % attn_meta.num_reqs == 0)
                        ? B / attn_meta.num_reqs
                        : 0;
  const DecodeGraphSlotKey key{S, Q, spec_step};
  // A q > 1 batch that carries NO spec segmentation is refused rather than
  // captured, and this is a TIGHTENING that arrived with the widening. `S` for a
  // non-spec step is `PadToCaptureSize(B)` and `BuildPaddedDecode` rewrites the
  // metadata as a pure decode of S single-token requests, which is not what a
  // multi-token-per-request batch is. The predicate never sent one here, but it
  // was the predicate saying so and nothing in the driver.
  const bool servable_shape = Q >= 1 && (Q == 1 || spec_step);
  const bool qlen_capped = DecodeGraphQueryLenCapped(impl_->slots, key);
  if (qlen_capped) v1::NoteDecodeGraphQueryLenDecline();
  if (!impl_->enabled || S < 0 || !servable_shape || qlen_capped ||
      (has_gdn && !detail::CanUseGdnDecodeGraphSize(
          B, S, IndexedGdnStateIoEnabled(impl_->queue.device)))) {
    if (aux_out != nullptr && !aux_out->layer_ids.empty()) {
      // The graph cannot serve this batch (disabled / unsupported size), so fall
      // back to the EAGER multi-tap forward, which fills aux_out itself. Without
      // this the drafter sees no taps at all.
      StepPhaseEmit(sph, "eager-mtap");
      ForwardLogits fl = Qwen3_5DenseModel::ForwardDeviceMultiTap(
          token_ids, positions, attn_meta, gdn_meta, attn_kv, gdn_state,
          impl_->weights, impl_->config, impl_->queue, aux_out, {});
      tap_emit("eager-mtap",
               fl.device_tensor.data == nullptr ? nullptr : fl.device_tensor.data);
      return fl;
    }
    const StepPhaseClk::time_point sph_tb0 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    DBuf lg = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                               attn_kv, gdn_state, impl_->weights, impl_->config,
                               LastTokenLogitsIndices(attn_meta, B));
    if (sph.on) {
      sph.body_ms = StepPhaseMsOf(sph_tb0, StepPhaseClk::now());
      // The eager forward's last enqueue: the seam's first blocking read after
      // this stamp reports the step's completion wait.
      vt::tenstorrent::StepPhaseNoteLaunch("eager");
      StepPhaseEmit(sph, "eager-fallback");
    }
    return WrapDeviceLogits(d, std::move(lg), vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then
  // refresh THIS size's persistent host buffers in place. VT_ASYNC_EXECUTOR: pick
  // this step's slot from the size's parity ring (alternating), and host-wait its
  // previous replay before Refresh touches its persistent host inputs (hazard-C).
  // OFF: always slot[0], no wait — byte-identical to the single-slot driver.
  const bool new_shape = impl_->slots.find(key) == impl_->slots.end();
  Impl::SlotRing& ring = impl_->slots[key];
  // W6: a distinct captured shape is the observable the widening is measured by.
  // Without it a two-shape driver and a one-shape driver are indistinguishable
  // from outside, which is how #1020 stayed silent.
  if (new_shape) v1::NoteDecodeGraphShape();
  // SPEC-DSPARK W8 (#442): a captured SPEC graph must read PERSISTENT step
  // inputs. The per-call StepDevInputs is a pool block freed when the call
  // returns, so a graph captured against it replays over recycled memory.
  const bool dbuf = impl_->dbuf || spec_step;
  Impl::SizeSlot& s = ring.slot[dbuf ? ring.next : 0];
  if (dbuf) {
    ring.next ^= 1;
    // Wait this slot's input-staged event (2 steps back) before Refresh/StageStepInputs
    // touch its host + pinned inputs — a tiny copy at the front of a 2-steps-back
    // replay, never the GPU tail (the Option A c16/c32 unlock).
    if (s.reuse_event.handle != nullptr) b.SynchronizeEvent(s.reuse_event);
  }
  // Record this slot's input-staged event on the main queue right AFTER the H2D stage
  // (Option A), so the next same-slot Refresh waits only that tiny copy. No-op unless
  // the double-buffer is on.
  const auto record_staged = [&] {
    if (!dbuf) return;
    if (s.reuse_event.handle == nullptr)
      s.reuse_event = b.CreateEvent(/*blocking=*/true);
    b.RecordEvent(s.reuse_event, impl_->queue);
  };

  // SPEC-DSPARK W8 (#442): size this slot's PERSISTENT aux buffer and invalidate a
  // graph captured for a different tap count. Only the EXACT-shape case (S == B,
  // which is what a spec batch takes) is served: the aux contract is [T, H*taps]
  // and padded rows would change T.
  const int64_t aux_taps =
      aux_out != nullptr ? static_cast<int64_t>(aux_out->layer_ids.size()) : 0;
  if (s.aux_taps != aux_taps) {
    // Reset() releases every segment through Backend::DestroyGraph and returns
    // the container to its as-constructed state, which is also what lets the
    // next capture open a scope on it (the scope refuses a container that
    // already holds one).
    s.graph.Reset();
    Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
    s.pinned.clear();
    s.warm = false;
    s.aux.reset();
    s.aux_taps = aux_taps;
  }
  if (aux_taps > 0 && (s.aux == nullptr || s.aux->t().shape[0] != S)) {
    s.aux = std::make_unique<DBuf>(d, DType::kBF16,
                                   std::vector<int64_t>{S, H * aux_taps});
  }
  Tensor aux_view{};
  const std::vector<int32_t>* aux_ids_arg = nullptr;
  const Tensor* aux_out_arg = nullptr;
  if (aux_taps > 0) {
    ValidateAuxTapLayerIds(aux_out->layer_ids, impl_->config.num_hidden_layers);
    aux_view = s.aux->t();
    aux_ids_arg = &aux_out->layer_ids;
    aux_out_arg = &aux_view;
  }
  // The captured region writes into the slot's buffer, so the drafter reads a
  // NON-owning view valid until this slot's next replay -- the same lifetime the
  // returned logits already carry.
  const auto publish_aux = [&] {
    if (aux_taps > 0) {
      aux_out->storage.reset();
      aux_out->tensor = s.aux->t();
    }
  };
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  v1::CommonAttentionMetadata pam;
  v1::GDNAttentionMetadata pgm;
  if (spec_step) {
    // S == B for a spec batch, so there is nothing to pad, and BuildPaddedDecode
    // would REWRITE the metadata as pure non-spec decode (num_decodes = S,
    // spec fields dropped). Carry the real metadata through untouched.
    ptok = token_ids;
    ppos = positions;
    pam = attn_meta;
    pgm = gdn_meta;
  } else {
    // The padding helper copies GDN indices into S entries. Without consumers,
    // the caller's unused metadata has no validated bound and must stay inert.
    BuildPaddedDecode(S, token_ids, positions, attn_meta,
                      has_gdn ? gdn_meta : GDNAttentionMetadata{}, ptok, ppos,
                      pam, pgm);
  }

  // A block-table column-count change reallocates the persistent block_table (the
  // staged/baked H2D dest shape moves) → invalidate this slot's graph + device inputs.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam, pgm);
  s.fa_cols = cols;
  bool seq_continuation = true;  // no seq_lens -> no boundary to detect
  // TT-27B-STEP-DECOMPOSE: the per-step TT refresh block is the "warmup
  // passes" phase — every Warm* call the captured arm makes per step is
  // inside this bracket, and `warm_calls` counts them (cos|sin + one paged-KV
  // shadow per full-attn layer + cur_pos + RAC idx + PA meta).
  const StepPhaseClk::time_point sph_tw0 =
      sph.on ? StepPhaseClk::now() : sph.t0;
  // TT captured arm (the WarmRopeCosSin populate-outside / content-HIT-inside
  // pattern; qwen3.cpp:827-900 ported to the dense driver): refresh every
  // persistent device input the captured region reads, OUTSIDE capture, on
  // every step (warm, capture, replay). No-op off the TT host-free decode lane.
  if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
    // Fused-preamble cos|sin table: refreshed so the captured
    // kAttnQkNormRopeGate serves a table matching the in-region
    // RopeCosSinCacheKernel refill of these same positions.
    vt::tenstorrent::WarmAttnCosSin(
        s.positions.data(), S, impl_->config.rotary_dim,
        static_cast<double>(static_cast<float>(impl_->config.rope_theta)));
    // Paged-KV device shadows for EVERY layer (must precede WarmRacIdx, which
    // builds its persistent sharded input from the shadows).
    for (const auto& kv : attn_kv) {
      const int64_t max_slot = pam.slot_mapping.empty() ? 0
          : *std::max_element(pam.slot_mapping.begin(), pam.slot_mapping.end());
      const int64_t used = (max_slot < 0) ? 1
          : std::max<int64_t>(1, max_slot / kv.block_size + 1);
      const size_t half = static_cast<size_t>(kv.block_size * kv.num_kv_heads *
                                              kv.head_size) * vt::SizeOf(kv.dtype);
      char* base = static_cast<char*>(kv.data);
      vt::tenstorrent::WarmPagedKvShadow(
          base, base + half, kv.num_blocks, kv.block_size,
          kv.num_kv_heads, kv.head_size, used);
    }
    // R2 cur_pos: seed outside capture; replay steps leave it to the captured
    // plus_one (the #1476/#2469 continuation predicate).
    if (!pam.seq_lens.empty()) {
      const bool continuation =
          (static_cast<int64_t>(pam.seq_lens[0]) - 1) == s.expected_cur_pos;
      seq_continuation = continuation;
      vt::tenstorrent::WarmDecodePos(
          pam.seq_lens.data(), static_cast<int64_t>(pam.num_reqs),
          /*replay_regime=*/s.graph.captured() && continuation);
      s.expected_cur_pos = pam.seq_lens[0] - 1;
    }
    vt::tenstorrent::WarmRacIdx(
        pam.slot_mapping.data(), pam.slot_mapping.data(),
        static_cast<int64_t>(pam.slot_mapping.size()),
        attn_kv.empty() ? 32 : attn_kv[0].block_size,
        pam.block_table_tensor.data(),
        static_cast<int64_t>(pam.block_table_num_cols),
        pam.seq_lens.data());
    if (!pam.block_table_tensor.empty() && !pam.seq_lens.empty()) {
      vt::tenstorrent::WarmPaMeta(
          pam.block_table_tensor.data(),
          static_cast<int64_t>(pam.num_reqs),
          static_cast<int64_t>(pam.block_table_num_cols),
          static_cast<int64_t>(pam.block_table_num_cols), 1,
          pam.seq_lens.data());
    }
  }
  if (sph.on) {
    sph.warms_ms = StepPhaseMsOf(sph_tw0, StepPhaseClk::now());
    sph.warm_calls = 1 + static_cast<int64_t>(attn_kv.size()) +
                     (pam.seq_lens.empty() ? 0 : 1) + 1 +
                     (pam.block_table_tensor.empty() || pam.seq_lens.empty()
                          ? 0
                          : 1);
    sph.boundary = !seq_continuation;
  }
  // #2469 (qwen3.cpp:903-921 port): a same-size request boundary must
  // re-capture, not seed-and-replay. The boundary seed alone leaves the first
  // post-boundary replays running a trace whose baked state the per-step
  // warms do not refresh; the qwen3 driver routes the boundary through the
  // proven cols-change lane instead: destroy the trace, run THIS step eagerly
  // against the freshly seeded cur_pos, and re-capture on the next step. The
  // dense captured arm was seed-only, and every request diverged from the
  // valid eager path at its 4th decode token.
  //
  // CONV-SHADOW GATE (TT captured arm): a prefill-bearing step transitions
  // the same conv_state buffer to the ssm/cache role (GdnStateGather, the
  // #2201 dual-role lane), which clears the transposed shadow AND REPLACES
  // the slot's device tensor. A replay would keep reading the trace's baked
  // stale pointer — the persistent post-boundary corruption this row chased —
  // and a capture-time rebuild is the refused read that aborted the boundary
  // probe (EnsureHost before its refusal CHECK). Query every layer OUTSIDE
  // capture: a miss takes the same lane — eager step rebuilds the shadow,
  // and the capture on the next step bakes the fresh pointers. Converges:
  // once the joining request's prefill chunks are done, one eager decode
  // step restores serveability and the next step captures.
  bool conv_shadow_stale = false;
  if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
    for (const auto& st : gdn_state) {
      if (!vt::tenstorrent::ConvShadowServeable(
              st.conv_state.data, st.conv_state.shape[0],
              st.conv_state.shape[1], st.conv_state.shape[2])) {
        conv_shadow_stale = true;
        break;
      }
    }
  }
  const bool tt_boundary =
      d.q.device.type == vt::DeviceType::kTENSTORRENT && !seq_continuation;
  if (((cols_changed || tt_boundary) && s.graph.captured()) ||
      (conv_shadow_stale && (s.graph.captured() || s.warm))) {
    s.graph.Reset();
    Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
    s.pinned.clear();
    s.warm = false;
    // UNBIND BEFORE THE DESTINATION DIES. `s.pin`'s cells name device pointers
    // that `s.dev` owns, so dropping `s.dev` first leaves them naming freed
    // pool blocks. `Unbind()` does not dereference the destination today, so
    // the old order was correct by the implementation's silence rather than by
    // anything stated — which is one edit away from a use-after-free.
    s.pin.Free();
    s.dev.reset();
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, stage this step's inputs into the persistent device
  // buffers (Option A: H2D out of the captured region), then relaunch the graph.
  //
  // TT RECAPTURE WORKAROUND (qwen3.cpp:928-950 port): on this tt-metal pin,
  // interleaved `copy_to_device` refreshes into trace-input buffers between
  // replays perturb the dispatch `expected_num_workers_completed` accounting,
  // and after ~38-50 perturbations the next replay's DISPATCH_WAIT never
  // completes — the main thread futex-waits forever (spec:
  // .agents/specs/tt-metal-trace-replay-write-desync.md). The dense captured
  // arm refreshes the SAME toxic class every step (WarmRacIdx + WarmPaMeta
  // INT32 idx/page-table copies), and the 0.8B captured run deadlocked at
  // ~30 replays exactly as documented. Destroy + re-capture the graph every
  // VT_TT_RECAPTURE_EVERY replays; the qwen3 battery runs the identical
  // setting (RC=8) green across 16-prompt cases.
  bool do_replay = s.graph.captured();
  if (do_replay && d.q.device.type == vt::DeviceType::kTENSTORRENT) {
    const char* rc_env = std::getenv("VT_TT_RECAPTURE_EVERY");
    const int rc_n = rc_env != nullptr ? std::atoi(rc_env) : 0;
    if (rc_n > 0 && s.replays >= rc_n) {
      s.graph.Reset();
      Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
      s.pinned.clear();
      s.warm = false;
      do_replay = false;
    }
  }
  if (do_replay) {
    const StepPhaseClk::time_point sph_te0 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
      // TT-DENSE-EMBED-IN-REGION (the qwen3.cpp:953-955 lane): the capture
      // scope covers DenseEmbedInto's device work, so the trace EMBEDS — this
      // step only REFRESHES the persistent ids tensor. Allocation-free
      // (WarmDecodeIds stages into the same device buffer it allocated at the
      // first capture); a miss fatals loudly inside EmbedDeviceIdsInto. NO
      // eager embedding here: the fresh dids alloc + EmbeddingKernel's own
      // device alloc + an eager ttnn::embedding program while the trace is
      // live is the documented replay-corrupting class
      // (tenstorrent_paged.cpp:2361-2364, #2469) — the defect #3323's
      // un-gating exposed.
      vt::tenstorrent::WarmDecodeIds(
          s.token_ids.data(), static_cast<int64_t>(s.token_ids.size()));
    } else {
      DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    }
    if (sph.on) sph.embed_ms = StepPhaseMsOf(sph_te0, StepPhaseClk::now());
    if (dbuf) {
      StageStepInputs(d, s);
      StageSpecStepInputs(d, s);
      record_staged();
      MaybePoisonStagedInputs(impl_->poison, s);
    }
#ifdef VT_BENCH_PROFILE_CONTROL
    vt::cuda::MarkCudaGraphReplayProfilerEligible(
        s.graph.segment(0), static_cast<uint32_t>(B), static_cast<uint32_t>(S),
        static_cast<uint64_t>(s.replays));
#endif
    // Through the seam's container, never `Backend::ReplayGraph` directly: the
    // container replays its segments in order (one, here, because a decode
    // capture is kFull) and owns the G3 replay counter the gate reads.
    const StepPhaseClk::time_point sph_tr0 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    s.graph.Replay(impl_->queue);
    if (sph.on) {
      sph.replay_ms = StepPhaseMsOf(sph_tr0, StepPhaseClk::now());
      StepPhaseEmit(sph, "replay");
    }
    ++s.replays;
    ++s.expected_cur_pos;  // the replay's in-trace plus_one advanced cur_pos
    ++impl_->replays;
    publish_aux();
    tap_emit("replay", s.logits->ptr());
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + residency were warmed for this size by the previous (eager)
  // step. CAPTURE the dense layer region once, instantiate the graph, launch it.
  if (s.warm && !s.region_declined) {
    // #1380: THE POOL MUST BE ABLE TO SERVE THE WHOLE CAPTURED FORWARD, not one
    // block of one tensor. This used to alloc-and-free a single [S, vocab] f32
    // block, on the reasoning that the capture RETAINS its logits while the
    // other ring slot still holds its own, so the free list is one short. The
    // retention half is right and the "one block of that shape" half is not:
    // `DevicePool` is keyed by SIZE CLASS, so the block the capture then misses
    // on need not be the logits. Measured on `thor:gpu0` (sm_110), the second
    // ring slot's capture died in `dconv`, the GDN causal-conv output
    // (`GdnBlockPaged`, this file), whose [T, conv_dim] lands in the SAME class
    // as the retained [S, vocab] logits at the gate's shape -- and a cudaMalloc
    // inside a captured region aborts the capture. An alloc-and-free grows the
    // pool only when that class's free list is EMPTY, so with one block free
    // and TWO needed live at once it grew nothing at all.
    //
    // So the driver stops naming a tensor and asks the pool for the demand the
    // EAGER step at this shape actually made (`s.demand`, taken at the end of
    // the cold step). Working scratch is still freed at DenseForwardLayers return and
    // still SAFELY shared between the two graphs -- they replay sequentially on
    // one stream. This runs OUTSIDE the capture, which is where a cudaMalloc is
    // legal, and it is a no-op once the free list is deep enough.
    //
    // #2029: IT RUNS BEFORE EVERY CAPTURE, and until this line moved it ran
    // before almost none of them. It sat inside the `if (dbuf)` below, and
    // `dbuf` is `impl_->dbuf || spec_step` -- the `VT_ASYNC_EXECUTOR` parity
    // ring, or a speculative step. The DEFAULT server sets neither, so the lane
    // a user gets by omitting `--speculative-config` opened its capture over a
    // pool nobody had prepared, and #1380's own failure came back at c=8 as
    // `cudaMalloc: operation not permitted when stream is capturing` while the
    // SAME binary with a speculative config served. The guard was never about
    // the pre-grow: that block exists for the parity ring's drain and its
    // persistent step inputs, and #1393 replaced the pre-grow in place without
    // revisiting what it sat under.
    //
    // The `!dbuf` arm needs this at least as much as the `dbuf` arm. It passes
    // `persistent_sdi == nullptr` to DenseForwardLayers, so `BuildStepDevInputs`
    // and `MaybeBuildAttnCosSin` run from the MAIN pool INSIDE the captured
    // region -- and that is also what makes `s.demand` exact for it, because the
    // cold step takes the identical `nullptr` branch. On the `dbuf` arm the
    // captured region's main-pool demand is a strict SUBSET of the cold step's,
    // which is the containment #1393 recorded.
    //
    // Ordering: this now precedes the `b.Synchronize` below rather than
    // following it. A pre-grow is `Backend::Alloc` and nothing else -- it
    // enqueues no work and reads no in-flight buffer -- and the drain still
    // happens before `BeginCapture`, which is the property it was added for.
    const StepPhaseClk::time_point sph_tp0 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    Pool(b).PreGrowForCapture(b, s.demand);
    if (sph.on) sph.pregrow_ms = StepPhaseMsOf(sph_tp0, StepPhaseClk::now());
    // dbuf: the runner may have skipped the depth-2 drain (the previous step
    // returned a slot view), so a prior replay can still be in flight. Capture must
    // begin on an idle stream — drain once here. One-time (≤2 captures per size);
    // steady-state replay never captures, so this never touches the overlap path.
    if (dbuf) {
      b.Synchronize(impl_->queue);
      // Option A: build this slot's PERSISTENT device inputs + pinned staging OUTSIDE
      // the capture (see the 35B driver). The dense fused-preamble arch (27B W4A4)
      // allocates + fills the persistent cos|sin here (pre-capture); the captured
      // region only RE-FILLS it (FillAttnCosSin), so nothing allocates during capture.
      // The RETAINED buffers draw from a DEDICATED pool (not the main scratch Pool())
      // so they never pop-and-hold a block the captured forward's scratch then needs.
      const int64_t gdn_state_slots =
          gdn_state.empty() ? 0 : gdn_state.front().ssm_state.shape[0];
      const bool fp4_attn = [&] {
        for (const auto& l : impl_->weights.layers)
          if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
        return false;
      }();
      // UNBIND BEFORE THE DESTINATION DIES, the same rule the shape-change path
      // above states, applied to the other site that replaces `s.dev`. The
      // assignment below destroys the PREVIOUS `StepDevInputs`, returning its
      // pool blocks to the free list while `s.pin`'s cells still name them, and
      // the rebind is four statements later at `s.pin.Alloc`. Reachable: the
      // `aux_taps` branch resets the graph and clears `warm` WITHOUT calling
      // `s.pin.Free()`, so the next warm step arrives here with the cells bound.
      // Harmless today only because `Unbind()` does not dereference — which is
      // exactly the argument the shape-change site refused to rely on. Free costs
      // nothing extra: `Alloc` calls `Free()` first regardless.
      s.pin.Free();
      {
        ActivePoolScope persistent_scope(&PersistentDecodeInputPool(d.b));
        s.dev = std::make_unique<StepDevInputs>(
            has_gdn ? BuildStepDevInputs(d, s.positions, s.attn_meta, s.gdn_meta,
                                         gdn_state_slots)
                    : BuildFullAttnStepDevInputs(d, s.positions, s.attn_meta));
        MaybeBuildAttnCosSin(d, *s.dev, impl_->config, S, fp4_attn);
      }
      const bool has_idx = s.dev->has_gdn_idx &&
                           s.gdn_meta.non_spec_state_indices_tensor.has_value();
      const int64_t idx =
          has_idx ? static_cast<int64_t>(
                        s.gdn_meta.non_spec_state_indices_tensor->size())
                  : 0;
      s.pin.Alloc(b, *s.dev, S, s.attn_meta.num_reqs, cols, idx, has_idx);
    }
    const StepPhaseClk::time_point sph_te1 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
      // TT-DENSE-EMBED-IN-REGION: stage ids for the captured embedding
      // (outside capture) — the persistent WarmDecodeIds tensor the captured
      // embedding reads (qwen3.cpp:998-1000).
      vt::tenstorrent::WarmDecodeIds(
          s.token_ids.data(), static_cast<int64_t>(s.token_ids.size()));
      // R4 dummy-run mirror (#1105, qwen3.cpp:1001-1016): the embedding and
      // the hidden-shadow copy run INSIDE the capture below, and both programs
      // are cold at first capture — tt-metal refuses to load new binaries
      // mid-trace (TT_FATAL mesh_workload.cpp:153 !is_capturing_trace). Run
      // the exact captured embed segment once OUTSIDE the scope; the captured
      // pass then hits the program cache. Safe to run twice: the hold in
      // EmbedDeviceIdsInto replaces the previous out tensor, and nothing
      // references the dummy run's output.
      Tensor dtab = Qwen3_5EmbeddingTable(
          d.b, d.q, impl_->weights.embed_tokens, vocab, H);
      vt::tenstorrent::EmbedDeviceIdsInto(
          s.hidden->ptr(), S, H, dtab.data, vocab, H,
          static_cast<int64_t>(s.token_ids.size()));
    } else {
      DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    }
    if (sph.on) sph.embed_ms = StepPhaseMsOf(sph_te1, StepPhaseClk::now());
    // ENG-CUDAGRAPH-BREAK W4 (#1307): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope owns
    // the segment, the handle, its release and the drain a mid-capture throw
    // needs — the drain three drivers each hand-rolled as the same
    // `try { EndCaptureGraph(); } catch (...) {}`, and which the scope now
    // performs by comparing `std::uncaught_exceptions()` against the depth it
    // recorded at entry. A refusal raised inside the forward (the fp8 splitK
    // premise check is one such, and not the only one) still reaches the caller
    // with the ORIGINAL error, because nothing here swallows it.
    //
    // kFULL, and the mode is the whole argument. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63`) is documented at
    // `:630-632` as a FULL graph for DECODE batches and a piecewise one for
    // prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the full
    // half. This is a decode driver, so its capture is ONE segment with the
    // attention calls INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every layer's attention into an
    // eager call between two graph replays, which is not vLLM's decode behaviour
    // and which nothing in this row's record supports.
    //
    // This is the 27B DENSE driver, and the bf16-D fp8 lever this region guards
    // (VT_GDN_FP8_IN_BF16) is off by DEFAULT (GdnFp8InBf16Enabled), which is the
    // only thing keeping it inert on the 35B — GDN-MOE-BF16-OUT (#1168) made
    // outdt BF16 on both arms, so the bound is a toggle now, not a model shape.
    std::optional<DBuf> lg;
    // TT-27B-STEP-DECOMPOSE: cap_begin brackets the scope CONSTRUCTION
    // (BeginCapture: ttnn's begin_trace_capture), cap_body the recorded
    // forward (host-side record cost per op — the capture body enqueues
    // nothing), cap_end the scope DESTRUCTION (EndCaptureGraph: the tt-metal
    // trace finalize + trace-buffer build — the capture-invocation cost).
    StepPhaseClk::time_point sph_tc1{};
    // tt-27b-region-capture: the mode IS the fit decision. Region scope
    // (env opt-in this slice; the automatic `vt::WholeGraphTraceFits`
    // wiring is the next wave) splits the same command stream one layer per
    // region — the whole-graph staging demand (~3.15 GB for 1,037 recorded
    // commands) never accrues, because each region's trace buffer lands
    // inside the 50 MiB budget the census below asserts. Every GraphBreak
    // in the forward is INERT in the kFull arm, so the default shape is
    // byte-identical to the one the comment above records.
    const bool region_scope = RegionCaptureRequested();
    {
      const StepPhaseClk::time_point sph_tc0 =
          sph.on ? StepPhaseClk::now() : sph.t0;
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph,
          region_scope ? vt::GraphCaptureMode::kPiecewise
                       : vt::GraphCaptureMode::kFull);
      if (sph.on) sph.cap_begin_ms = StepPhaseMsOf(sph_tc0, StepPhaseClk::now());
      sph_tc1 = StepPhaseClk::now();
      if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
        // TT-DENSE-EMBED-IN-REGION: capture-safe embedding over the persistent
        // ids tensor, writing the persistent hidden shadow the layer region
        // reads — the capture scope now COVERS DenseEmbedInto's device work
        // (the qwen3.cpp:1041-1051 lane; the #3321 W3 in-region discipline).
        // Replay steps refresh the ids outside (WarmDecodeIds) and re-run
        // this embedding inside the trace, so no replay step performs an
        // eager device alloc while the trace is live.
        Tensor dtab = Qwen3_5EmbeddingTable(
            d.b, d.q, impl_->weights.embed_tokens, vocab, H);
        vt::tenstorrent::EmbedDeviceIdsInto(
            s.hidden->ptr(), S, H, dtab.data, vocab, H,
            static_cast<int64_t>(s.token_ids.size()));
      }
      lg = DenseForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                              s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                              impl_->config, {}, nullptr, nullptr,
                              aux_ids_arg, aux_out_arg, /*return_hidden=*/false,
                              dbuf ? s.dev.get() : nullptr);
      // R2 (the qwen3.cpp:1054-1061 port): advance cur_pos on-device
      // (plus_one) INSIDE the captured trace, at the END of the body — after
      // every cur_pos read in the layers' RAC/PA. cur_pos/update_idxs are
      // ALIASED to this one device buffer (WarmRacIdx/WarmPaMeta), so without
      // the in-trace advance a replayed trace re-reads the seeded position
      // forever: RAC rewrites one frozen slot, PA attends a frozen window,
      // and the captured arm emits deterministic garbage that no token gate
      // can distinguish from a correct run by shape alone. The NEXT replay
      // then sees cur_pos+1, which is what the hooks' replay regime expects.
      if (d.q.device.type == vt::DeviceType::kTENSTORRENT &&
          !pam.seq_lens.empty()) {
        vt::tenstorrent::CaptureDecodePosAdvance(
            static_cast<int64_t>(pam.num_reqs));
      }
      if (sph.on) {
        sph.cap_body_ms = StepPhaseMsOf(sph_tc1, StepPhaseClk::now());
        // Re-stamp so the same variable brackets the scope DESTRUCTOR next.
        sph_tc1 = StepPhaseClk::now();
      }
    }  // ~GraphCaptureScope closes the segment and files it on s.graph
    if (sph.on) sph.cap_end_ms = StepPhaseMsOf(sph_tc1, StepPhaseClk::now());
    // NOT CAPTURED covers TWO states, and returning the buffer is correct for
    // exactly one of them. `~GraphCaptureScope` must swallow a throwing
    // `EndCaptureGraph` — a destructor that propagates terminates — so a FAILED
    // capture leaves the container reporting what an INERT scope reports.
    //
    //   * INERT (`capture_failed() == false`): the backend cannot capture, or
    //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, the layer
    //     region above ran EAGERLY, and the buffer is a real result. Return it
    //     and go back to cold so the next same-size step re-warms.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw.
    //     Under stream capture NOTHING between `BeginCapture` and the throw
    //     executed: every kernel was RECORDED, so the buffer is pool-recycled
    //     memory. Returning it would hand this step uncomputed device memory as
    //     its logits — silently wrong tokens, no fault, and a token gate cannot
    //     see it.
    //
    // So the failure PROPAGATES, carrying the runtime's own exception. This is
    // exactly what the pre-W4 driver did (`s.graph = b.EndCaptureGraph(...)` was
    // unguarded), and it is not a recovery this stage can justify inventing: a
    // stream whose capture was INVALIDATED has not told us it is usable.
    if (!s.graph.captured()) {
      s.warm = false;  // either way this slot goes back to cold
      if (s.graph.capture_failed()) {
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();  // clear the failure with the graph it described
        Pool(b).UnpinForGraph(b, s.pinned);  // #2274: no graph, nothing baked
        s.pinned.clear();
        // The runtime's OWN diagnosis where the seam holds it. It is empty only
        // on the arm where an exception was already propagating THROUGH the
        // scope, which cannot reach this line; the refusal below makes that
        // unreachability an assertion rather than a claim.
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "Qwen3.5 decode graph: the capture was ABANDONED and its logits were "
                 "never computed; refusing to return uncaptured device memory");
      }
      publish_aux();
      record_staged();
      if (sph.on) {
        // INERT scope: the region ran EAGERLY, so the drain read pays its own
        // device tail (the launch stamp below attributes it).
        vt::tenstorrent::StepPhaseNoteLaunch("eager");
        StepPhaseEmit(sph, "capture-eager");
      }
      ForwardLogits drained = WrapDeviceLogits(d, std::move(*lg), vocab);
      if (drained.rows != B) {
        drained.rows = B;
        drained.device_tensor = MakeTensor(drained.device_storage.get(), DType::kF32,
                                           d.q.device, {B, vocab});
      }
      tap_emit("capture-eager",
               drained.device_tensor.data == nullptr
                   ? nullptr
                   : drained.device_tensor.data);
      return drained;
    }
    // #2274 THE FIX. The capture succeeded, so from here the graph's replays
    // write through the device pointers it just baked. `ForwardLayers` has
    // already RETURNED its working scratch to the pool's free list, and the
    // comment at `PreGrowForCapture` above argues that is safe because the two
    // graphs "replay sequentially on one stream". That argument covers the two
    // GRAPHS. It does not cover a THIRD party: a DFlash2 draft store created
    // later takes a pool block for its block table, is handed one of these
    // blocks, and the next replay writes f32 activations over the table -- which
    // is why a live block table reads `bt[0] = 1060730955` (0.727f) and
    // `vt::PagedAttention` indexes page 1.06e9 in a 513-page pool.
    //
    // So take the capture's demand back OUT of the free list. `PreGrowForCapture`
    // guaranteed `s.demand` free blocks per class BEFORE the capture and nothing
    // else allocated in between, so the free list is LIFO-ordered with exactly
    // those blocks on top; this pops the same count off the same end.
    //
    // BEFORE the `s.logits` assignment below, and the order is load-bearing on
    // the RE-capture path (a column-count change resets this slot's graph, which
    // is what a new draft store's wider block table causes). That assignment
    // destroys the previous `s.logits` and frees ITS block to the same free list,
    // where it would be popped first and leave one of the graph's own blocks
    // reachable. Freeing the old logits is correct -- the graph it belonged to
    // was Reset -- it just must not happen while we are counting.
    const StepPhaseClk::time_point sph_tpin0 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    if (s.pinned.empty()) s.pinned = Pool(b).PinForGraph(b, s.demand);
    if (sph.on) sph.pin_ms = StepPhaseMsOf(sph_tpin0, StepPhaseClk::now());
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr, "[DenseDecodeGraph] captured Qwen3.5 dense decode graph "
                           "for padded size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    const StepPhaseClk::time_point sph_tr1 =
        sph.on ? StepPhaseClk::now() : sph.t0;
    s.graph.Replay(impl_->queue);
    if (sph.on) {
      sph.replay_ms = StepPhaseMsOf(sph_tr1, StepPhaseClk::now());
      StepPhaseEmit(sph, "capture");
    }
    record_staged();
    s.replays = 1;
    ++s.expected_cur_pos;  // the capture step's launch ran the trace once
    ++impl_->replays;
    publish_aux();
    tap_emit("capture", s.logits->ptr());
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool + resident weights /
  // fp4-GEMM StreamScratch pools for this size) and defer capture to the next
  // same-size step. This is a real decode step (its padded output's real rows are
  // used). (Re)allocate the persistent hidden buffer to this size.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  const StepPhaseClk::time_point sph_te2 =
      sph.on ? StepPhaseClk::now() : sph.t0;
  DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  if (sph.on) sph.embed_ms = StepPhaseMsOf(sph_te2, StepPhaseClk::now());
  // W2 (capture-warmup redesign): NO snapshot/restore around the warmup.
  // The restore 2ed5e912e4 added traded semantic correctness for program-
  // cache stability the commit already provides: the warmup's
  // GdnStateScatter commit lands at the SAME split geometry
  // ([slots*F, blk] TILE f32, dev_rows/dev_cols = slots/cache_row) that
  // `EnsureGdnCacheDevice` serves in either pass, so the capture's first
  // ssm read fast-serves the warmup's commit with the identical spec.
  // Rolling the slot back instead DISCARDED the warmup's state update —
  // every later step recomputed from the pre-warmup state, the frozen GDN
  // recurrence (period-3 279/6511/314; ISSUE-
  // LOCAL-01M3918KQ580Z3NHVRNXVF15FZ). Serving the commit directly is the
  // value-preserving form: step N+1 reads exactly the state step N
  // committed. (W1 already removed the conv slot for the serveability
  // reason; this removes the ssm slot for the same value-preservation
  // one, and the whole snapshot/restore with it.)
  const StepPhaseClk::time_point sph_tb1 =
      sph.on ? StepPhaseClk::now() : sph.t0;
  DBuf lg = DenseForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                               s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                               impl_->config, {}, nullptr, nullptr, aux_ids_arg,
                               aux_out_arg);
  if (sph.on) {
    sph.body_ms = StepPhaseMsOf(sph_tb1, StepPhaseClk::now());
    // The cold step's eager forward: the first blocking read after this stamp
    // (its logits drain) reports the step's completion wait.
    vt::tenstorrent::StepPhaseNoteLaunch("eager");
  }
  s.warm = true;
  // #1380: the cold step is the ONE eager run of this exact forward at this exact
  // shape, so it is where the capture's allocation demand is measurable. Recorded
  // per SLOT and consumed by `PreGrowForCapture` at this slot's capture. Taken
  // AFTER the forward and BEFORE the logits leave: the profile is a per-class
  // PEAK, so it already holds every block this step had live at once.
  s.demand = Pool(b).StepDemandProfile();
  publish_aux();
  record_staged();
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), vocab);
  if (fl.rows != B) {
    fl.rows = B;
    fl.device_tensor =
        MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  }
  if (sph.on) StepPhaseEmit(sph, "cold");
  tap_emit("cold", fl.device_tensor.data == nullptr ? nullptr
                                                    : fl.device_tensor.data);
  return fl;
}

}  // namespace vllm
