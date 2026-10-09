// Shared dense self-attention block + device glue for full-attention models.
//
// Extracted VERBATIM (behavior-preserving) from the anonymous namespace of
// src/vllm/model_executor/models/qwen3.cpp so a second full-attention arch
// (Qwen3-Coder `Qwen3MoeForCausalLM`, the first full-attention MoE bring-up —
// qwen3_moe.cpp, W3) reuses the EXACT same attention preamble (merged/3-shard
// QKV GEMM, per-head q/k RMSNorm, NeoX RoPE, paged FA2, o_proj) and the pooled
// device-scratch glue (Dev/DBuf/ResidentWeight/StepInputs) rather than
// re-deriving them. This mirrors how dense_weight_loaders.h + device_pool.h were
// extracted for reuse (.agents/specs/sweep-qwen3-coder-30b.md §3b SEAM GAP #1).
//
// It is a PURE RELOCATION: the env-flag readers, Dev/DBuf/pool-policy glue, the
// resident-weight uploaders, KvSlice, StepInputs/BuildStepInputs and AttnBlock
// are byte-for-byte the qwen3.cpp definitions (now `inline` in namespace
// `vllm::dense_attn`), so the Qwen3-dense (0.6B/4B) forward that includes this
// header and `using namespace dense_attn;`-imports it is BYTE-IDENTICAL — the
// same vt:: op sequence runs in the same order. qwen3.cpp keeps the dense-only
// MLP/decoder-layer/forward-body machinery that composes these pieces.
//
// Numeric contract (mirrors the qwen3_5 full-attention FALLBACK path — the
// token-exact paged==dense anchor): the residual stream is the model dtype (bf16,
// matching vLLM's fused_add_rms_norm residual); the qkv GEMM emits f32 q/k/v so
// the per-head q/k RMSNorm + RoPE run in f32 on the f32 A/B path (default bf16
// mirrors vLLM's per-op bf16 stores); the paged KV cache is written bf16 while
// the query stays f32 into vt::PagedAttention; o_proj flows bf16.
#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // NVFP4 W4A16 dispatch
#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/kv_cache_route.h"    // KV-FP8 W3 store/read route
#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseAttnWeights, PagedKvCache
#include "vllm/model_executor/models/tensor_parallel.h"   // TensorParallel/TpAllReduceSum (W2)
#include "vllm/platforms/interface.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vt/backend.h"
#include "vt/exl3_grouped.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace dense_attn {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// VT_FUSED_CHAIN_ADOPT (default ON, consistent with the framework): route the
// norm and qk-norm-rope preambles through the declared fusion recipes
// (kFusedAddRmsNormStd / kAttnQkNormRope) via vt::FusedChain. The Tier-0
// composite dispatches to the SAME standalone vt:: ops, so it is byte-identical
// to the hand-call fallback (=0). Read once (getenv is cheap; process-stable).
inline bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// VT_QWEN3_ROPE_CACHE (DEFAULT ON) — route the bf16 attention preamble's RoPE
// through the per-step cos/sin cache (RopeFromCache) instead of RopeNeox's
// per-element fp64 pow/cos/sin recompute. MEASURED the dominant dense-TTFT lever
// (Qwen3-4B c1 median TTFT 209->135 ms = median parity vs graphed vLLM; c8 316->
// 144 ms), and vLLM-faithful (bf16 cos/sin cache == vLLM RotaryEmbedding).
//
// FLIPPED default-ON 2026-07-20 after the opt-in rationale was GROUNDED AND
// DISPROVEN on GB10. RopeNeox and RopeFromCache are distinct CUDA __global__s so
// nvcc contracts `x*c - y*sn` to FMA differently — a deterministic 1-ULP shift
// that moves two genuine bf16 near-tie tokens (0.6B p0 tok5, 4B p1 tok6) and so
// requires regenerating the near-tie goldens (done: our_ids + neartie_gap). The
// claimed second reason — that the shift landed on a run-to-run "FA2 split-KV
// combine non-determinism" making the SACRED gate flaky — was NOT reproducible:
// the paged engine is byte-DETERMINISTIC run-to-run on the gate battery (K=4
// RoPE-off + K=3 RoPE-on, identical token ids every run), and for the short gate
// contexts num_splits==1 so the split-KV combine kernel is never even launched.
// With the goldens regenerated the near-tie gate PASSES 16/16 on both sizes
// (0.6B max gap 0.125 nats, 4B 0.250 nats — all divergences are genuine bf16
// near-ties, none over the 0.5-nat band). Opt out with VT_QWEN3_ROPE_CACHE=0 for
// a same-binary A/B against the RopeNeox recompute. Read once (process-stable).
inline bool RopeCacheEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN3_ROPE_CACHE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Build the RopeArgs for a dense model from its config. base + rotary_dim always;
// the llama3 frequency-rescale fields (factor/low/high/orig_max) are filled ONLY
// when the checkpoint declares rope_type=="llama3" (e.g. Llama-3.2). For every
// other model (rope_type=="default": Qwen, the gate models) the llama3 fields stay
// zero, so RopeArgs is byte-identical to the previous `RopeArgs{base, rot}` form
// and the RoPE kernels take their no-op branch — the Qwen3-dense forward is
// UNCHANGED. Mirrors vLLM get_rope()'s llama3 dispatch
// (rotary_embedding/__init__.py:155-171 -> Llama3RotaryEmbedding).
inline vt::RopeArgs MakeRopeArgs(const HfConfig& cfg) {
  vt::RopeArgs a;
  a.base = static_cast<float>(cfg.rope_theta);
  a.rotary_dim = static_cast<int>(cfg.rotary_dim);
  if (cfg.rope_parameters.rope_type == "llama3") {
    a.llama3_scaling_factor =
        static_cast<float>(cfg.rope_parameters.factor.value_or(0.0));
    a.llama3_low_freq_factor =
        static_cast<float>(cfg.rope_parameters.low_freq_factor.value_or(0.0));
    a.llama3_high_freq_factor =
        static_cast<float>(cfg.rope_parameters.high_freq_factor.value_or(0.0));
    a.llama3_orig_max_position = static_cast<float>(
        cfg.rope_parameters.original_max_position_embeddings.value_or(0));
  }
  return a;
}

// Merged QKVParallelLinear: issue ONE [q|k|v] GEMM over the merged qkv weight
// (mirror vLLM qwen3.py Qwen3Attention.qkv_proj, one F.linear then split via
// QkvSplit), replacing the three per-shard MatmulBT GEMMs. It folds the two tiny
// GQA k/v GEMMs (N=Hkv*Dh) into one wide, tensor-core-efficient GEMM and cuts 2
// GEMM launches/layer.
//
// MEASURED NEUTRAL (2026-07-21, Qwen3-4B GB10, merge ON vs OFF same binary):
// c1 tput 187.1 vs 185.7 (+0.8%), TPOT 46.8 vs 47.2; c8 tput 1314 vs 1308
// (+0.4%), TPOT 50.0 vs 50.4, P99 ITL 266 vs 258 — all within run-to-run noise.
// c8 decode is 93% GPU-busy (compute-bound); merging does not cut FLOPs and the
// launch saving is negligible against ~82k decode launches, so there is no win.
// It is also byte-AFFECTING (cuBLASLt picks a different K-reduction for the wider
// merged N, flipping ONE Qwen3-0.6B near-tie token — Qwen3-4B stays 16/16 exact),
// which required regenerating the SACRED 0.6B near-tie golden.
//
// D1 (fold plan Tier D1, 2026-07-31) FLIPPED THE DEFAULT ON and made this the
// SHARED merged-QKV gate for the whole bf16 dense/coder/dflash/Gemma family — the
// same OLMo-2/Granite/StableLM exemplar form (one MatmulBT over the merged
// [Nq+Nk+Nv,H] owner + a contiguous QkvSplit). Every consumer already packs the
// merged qkv_proj owner (no loader concat). The merge is bit-exact GEMM math
// (wider N; identical per-output-row reduction — see tests/vt/test_ops_qkv_merge.cpp,
// CPU byte-identity); the ONLY byte effect is the downstream FA2/attention 1-ULP on
// the 0.6B genuine bf16 near-tie, so the whole family was batched behind ONE 0.6B
// near-tie golden regen. Qwen3-4B stays STRICT (measured neutral). NOT interleaved
// with any RopeNeox->RopeFromCache swap (that would compound a second 1-ULP flip);
// rope handling is UNCHANGED. 27B/35B are UNAFFECTED (resident nvfp4/fp8 QKV, not
// this bf16 path). Opt out with VT_QWEN3_QKV_MERGE=0 for the byte-identical 3-shard
// A/B baseline (same binary, restores the pre-D1 sequence everywhere). Read once.
inline bool Qwen3QkvMergeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN3_QKV_MERGE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Readable alias for the shared merged-QKV gate at the non-Qwen (Gemma/dflash) D1
// call sites — the same env (VT_QWEN3_QKV_MERGE, default ON) and process-stable
// value, so the whole D1 family flips together behind one switch.
inline bool MergedQkvEnabled() { return Qwen3QkvMergeEnabled(); }

// Dev / MakeTensor / Reshape / DevicePoolPolicy / DBuf were RELOCATED VERBATIM
// to dense_device_glue.h (included above) so dense_nvfp4_gemm.h can layer
// beneath AttnBlock without an include cycle. They remain in namespace
// `vllm::dense_attn`, so every consumer resolves them exactly as before.

inline std::vector<float> WeightF32(const OwnedTensor& w) {
  VT_CHECK(w.dtype == DType::kBF16 || w.dtype == DType::kF16,
           "dense attention norm upcast requires BF16 or FP16 storage");
  const auto* src = w.bytes.data();
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    uint16_t bits;
    std::memcpy(&bits, src + i * 2, sizeof(bits));
    out[static_cast<size_t>(i)] = w.dtype == DType::kF16
        ? vt::F16ToF32(bits) : vt::BF16ToF32(bits);
  }
  return out;
}

// Device-resident raw-dtype view over an owned weight, uploaded ONCE (lazily) and
// reused across every forward step (mirrors qwen3_5.cpp ResidentWeight).
// A generated cache can initialize its device storage with backend operations.
// The callback runs only on the first device allocation. CPU uses host bytes.
inline Tensor ResidentWeight(Dev d, const OwnedTensor& w, std::vector<int64_t> shape = {},
                             const std::function<void(Tensor&)>& initialize = {}) {
  if (initialize)
    VT_CHECK(w.rank > 0 && w.rank <= vt::kMaxRank,
             "resident generated tensor requires a valid rank");
  if (shape.empty()) shape.assign(w.shape, w.shape + w.rank);
  // HOST-POINTER ALIASING IS A CPU PROPERTY, NOT A "NOT-CUDA" PROPERTY.
  // This read `!is_cuda()`, which is true for kMETAL, kVULKAN and kXPU as well
  // as kCPU — so any DEVICE backend other than CUDA aliased the host weight
  // bytes straight into a tensor and handed a HOST pointer to a DEVICE kernel.
  // Latent only because no model runs on a non-CUDA device backend yet; a hard
  // blocker for the Metal/Vulkan model bring-up
  // (.agents/specs/metal-mlx-reuse-study.md §3.3 item 2). The correct predicate
  // is `is_cpu()`: alias when the "device" IS the host, upload otherwise.
  // #1953 (found by #1946's review): AN EMPTY WEIGHT CANNOT BE MADE RESIDENT, AND
  // NOTHING DOWNSTREAM CAN TELL. Every op validates rank, shape, dtype, contiguity and
  // device — `vt::Embedding` (src/vt/ops.cpp) is the one this row cares about and
  // it is typical — and the SHAPE here comes from the CALLER, not from the bytes.
  // So a tensor with no bytes passes all of them: the alias arm below hands a
  // kernel a null host pointer (SIGSEGV) and the staging arm `Alloc(0)`s and then
  // views [vocab, H] over a zero-byte allocation (out-of-bounds device reads,
  // silently wrong values). Both are worse than a refusal and neither names the
  // weight. Refuse here, where the name is still in hand.
  //
  // `bytes.empty()` and NOT `Empty()`: a weight whose host buffer was reclaimed
  // after upload (`host_released`) IS populated, and the `d_dev` branch below is
  // what serves it. Each arm therefore asserts the precondition IT needs, which
  // is also why the staging assert sits inside `if (!w.d_dev)` — a resident
  // weight re-read on the hot path pays nothing for this.
  const vllm::platforms::Platform& plat =
      vllm::platforms::GetPlatform(d.q.device.type);
  if (plat.is_cpu()) {
    VT_CHECK(!w.bytes.empty(),
             std::string("resident weight: EMPTY tensor has no host bytes to "
                         "alias (host-alias arm, dtype ") +
                 vt::Name(w.dtype) + ", rank " + std::to_string(w.rank) + ")");
    vt::Tensor t = w.ViewOn(const_cast<uint8_t*>(w.bytes.data()), d.q.device, shape);
    // The CPU-alias arm owns these two markers and ONLY these two, exactly as it
    // did before ViewOn existed. q8_0_aligned is deliberately absent: setting it
    // on an aliased host buffer changes which kernel the CPU lane selects.
    t.repacked = w.repacked;
    t.elem_kn_repacked = w.elem_kn_repacked;
    return t;
  }
  // AUDIT GUARD, mirroring `qwen3_5.cpp` (:1070-1072). Only the CPU
  // `MatmulBTKernel` honours `elem_kn_repacked`, and the staging path below
  // uploads bytes verbatim and returns a tensor WITHOUT the marker, so a
  // repacked weight reaching a staged device would be read as plain [N,K] and
  // produce garbage silently. `VT_CPU_ELEM_KN_REPACK` is CPU-only and the
  // loader policy cannot see the device, so this is where the invariant is
  // enforced: fail loudly at load rather than corrupt tokens at inference.
  VT_CHECK(!w.elem_kn_repacked,
           "resident weight: an elem_kn_repacked ([K,N]) weight reached device "
           "staging; VT_CPU_ELEM_KN_REPACK is a CPU-only load transform");
  if (!w.d_dev) {
    VT_CHECK(!w.bytes.empty() || initialize,
             std::string("resident weight: EMPTY tensor has no host bytes to "
                         "upload (device-staging arm, dtype ") +
                 vt::Name(w.dtype) + ", rank " + std::to_string(w.rank) + ")");
    size_t nb = w.bytes.size();
    if (initialize) {
      VT_CHECK(!shape.empty() && shape.size() <= vt::kMaxRank,
               "resident generated tensor requires a valid rank");
      nb = vt::SizeOf(w.dtype);
      for (const int64_t dim : shape) {
        VT_CHECK(dim > 0 && nb > 0 &&
                     static_cast<uint64_t>(dim) <= static_cast<uint64_t>(INT64_MAX) / nb,
                 "resident generated tensor size overflow");
        nb *= static_cast<size_t>(dim);
      }
    }
    void* p = d.b.Alloc(nb);
    // Issue #150 accounting: this is the ONE host->device weight upload. When
    // `w.bytes` borrows the safetensors mapping (ENG-LOAD-DIRECT-UPLOAD) the
    // source of this copy IS the file mapping, so the load moved the bytes once
    // rather than twice.
    Backend* bk = &d.b;
    auto owner = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    if (initialize) {
      Tensor generated = w.ViewOn(p, d.q.device, shape);
      initialize(generated);
    } else {
      vllm::load_stats::AddDeviceUpload(nb);
      d.b.Copy(d.q, p, w.bytes.data(), nb);
    }
    w.d_dev = std::move(owner);
    if (initialize && w.bytes.empty()) w.host_released = true;
    // THE SOURCE PAGES ARE SPENT, AND THIS IS THE ARM QWEN4-EXP ACTUALLY TAKES.
    // The identical release landed first in `qwen3_5.cpp`'s own `ResidentWeight`
    // (the TU-local one, which shadows this function inside that file), and that
    // covered the Qwen3.5 dense weights and — through `KqResidentSlice` and
    // `KqGrouped` — the shared MoE seam's keep-quant expert towers. It did NOT
    // cover Qwen4-Exp's attention, norm, hyper-connection, PLE and lm_head
    // weights, because every one of those stages HERE: 12 call sites in
    // `qwen4_exp_forward.cpp`, 10 in `qwen4_exp_qsa_block.cpp`, 7 in
    // `qwen4_exp_ple_block.cpp` and 1 in `qwen4_exp_registry.cpp`, and not one
    // reference to the `qwen3_5.cpp` function anywhere in that model. Measured
    // with the release in the other function only: host `RssFile` on
    // `strix:gpu0` still climbed to 21.08 GB during the forward and stayed
    // there (.agents/specs/rocm-host-residency-after-upload.md §4).
    //
    // INSIDE THE `d_dev` MEMO ON PURPOSE, which is #1299's lesson and the same
    // placement the other arm was reviewed on: this function runs about 1,361
    // times per forward step on the target checkpoint, so a release that
    // re-tested its condition on every call would `MADV_DONTNEED` the pages the
    // GPU is about to read, every step. The helper states its other two
    // preconditions (a BORROWED span, and `mmap_fd >= 0` so the pages are
    // re-faultable from a file rather than anonymous) and synchronizes the
    // queue before it touches anything.
    //
    // The platform answer is the one COMPUTED ABOVE, passed in. It is
    // `host_memory_is_device_addressable()` and NOT the backend's
    // `DeviceMemoryIsHostAddressable()`: those answer different questions and
    // GB10 answers them differently, which is what the guard repair on this
    // branch existed for. The helper asks the backend's separately.
    vllm::MaybeReleaseStagedBorrowSource(d.b, d.q, w,
                                         plat.host_memory_is_device_addressable());
    // The host mirror is now redundant wherever device memory is host-
    // addressable (Vulkan). See AdoptDeviceBytesAsHost — this is what keeps a
    // unified-memory box from holding the whole model twice.
    AdoptDeviceBytesAsHost(d.b, w);
  }
  return w.ViewOn(w.d_dev.get(), d.q.device, shape);
}

// ── EXL3 (exllamav3 trellis) linear — QUANT-EXL3 W1b (#2181) ────────────────
//
// THIS IS THE ONE IMPLEMENTATION, and `layers::Exl3LinearMethod::Apply`
// delegates to it rather than repeating it, so the shared LinearMethod seam and
// this forward compute through the same code.
//
// It lives HERE, beside `ResidentWeight`, and that placement is forced rather
// than chosen. It needs `ResidentWeight`, whose body carries the #1953/#1946
// reasoning about empty weights and about host-pointer aliasing being a CPU
// property rather than a not-CUDA one. A second copy of that in a scheme header
// is exactly the hand-written parallel path AGENTS.md forbids, and a scheme
// header cannot include this one back: `linear.h` already includes it, so
// `dense_attn_block.h -> linear.h` would close a cycle.

// y[M,N] = x[M,K] @ dequant(w), with `out_dtype` the CALLER's choice.
//
// The activation is staged to fp16 because `vt::Exl3Gemm` reads it as fp16 and
// nothing else — the CPU arm calls `HadRows(HadIo::kHalfHalf, ...)` on `a`
// (`cpu_exl3_kernels.cpp:205`) and the device arm stages `a_had` in fp16, since
// exllamav3 runs the linear in fp16. An already-fp16 caller pays no copy.
//
// The OUTPUT dtype is never inherited from the kernel: `Exl3Gemm` writes f16 or
// f32, so an f16 or f32 request is written straight and a bf16 request is
// written f32 and cast ONCE. That is the polarity AGENTS.md §"Inherit vLLM
// defaults" requires, and the one a token gate cannot check for you. Native XPU
// folds these casts into the Hadamard kernels while retaining both FP16 input
// rounding steps and the single final BF16 store. Different suh projections
// still receive their own transformed activation.
// Model-owned packed group. Materialize/upload once, retaining neither an
// additional whole-model host copy nor separate source GPU residents. The
// first-upload wait protects staging lifetime; steady-state calls reuse all
// four resident tensors (the VT metadata check still has its own wait).
inline DBuf Exl3GroupedMatmulD(Dev d, const vt::Tensor& x,
                               const Exl3GroupedWeight& w) {
  const int64_t M = x.shape[0], K = w.suh.shape[1], N = w.svh.shape[0];
  VT_CHECK(d.q.device.type == vt::DeviceType::kXPU &&
               d.activation_dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 &&
               x.rank == 2 && x.shape[1] == K && w.codebook == 2,
           "exl3 grouped model: requires scoped XPU FP16 and matching K/mul1");
  const int bits = static_cast<int>(w.trellis.shape[2] / 32);
  // Validate arithmetic/extent before uploading weights. Large prefills use
  // the independent signed INT8 producer route, without dense reconstruction.
  const int panel_columns = M > 128 ? vt::Exl3W8A8ModelPanelColumns() : 128;
  if (M > 128) (void)vt::PlanExl3W8A8(M, K, N, w.suh.shape[0], bits, panel_columns);
  else (void)vt::PlanExl3SmallM(M, K, N, bits);
  const bool upload = !w.trellis.d_dev || !w.suh.d_dev || !w.svh.d_dev || !w.source_map.d_dev;
  auto trellis = ResidentWeight(d, w.trellis);
  auto suh = ResidentWeight(d, w.suh);
  auto svh = ResidentWeight(d, w.svh);
  auto map = ResidentWeight(d, w.source_map);
  if (upload) {
    d.b.Synchronize(d.q);
    w.trellis.ReleaseHost();
    w.suh.ReleaseHost();
    w.svh.ReleaseHost();
    w.source_map.ReleaseHost();
  }
  DBuf out(d, vt::DType::kF16, {M, N});
  if (M > 128) {
    vt::detail::Exl3GroupedW8A8Model(d.q, out.t(), x, trellis, suh, svh, map,
        {bits, w.codebook, w.name.c_str(), panel_columns}, w.source_map.d_dev, w.w8a8_model_map);
    // One backend-owned completion lease replaces per-projection DBuf scratch.
    return out;
  }
  const auto plan = vt::PlanExl3SmallM(M, K, N, bits);
  DBuf had(d, vt::DType::kF16, {w.suh.shape[0], K / 16, plan.padded_rows, 16});
  DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
  vt::detail::Exl3GroupedLinearModel(d.q, out.t(), x, trellis, suh, svh, map,
      had.t(), parts.t(), {bits, w.codebook, w.name.c_str()},
      w.source_map.d_dev, w.w8a8_model_map);
  return out;
}

inline DBuf Exl3MatmulD(Dev d, const vt::Tensor& x, const Exl3Weight& w,
                        vt::DType out_dtype) {
  const int64_t M = x.shape[0];
  const int64_t K = w.InFeatures();
  const int64_t N = w.OutFeatures();
  VT_CHECK(x.rank == 2 && x.shape[1] == K,
           "exl3 linear: activation is [" + std::to_string(x.shape[0]) + "," +
               std::to_string(x.rank == 2 ? x.shape[1] : -1) +
               "] but the weight needs K=" + std::to_string(K));
  VT_CHECK(out_dtype == vt::DType::kF32 || out_dtype == vt::DType::kBF16 ||
               out_dtype == vt::DType::kF16,
           "exl3 linear: out_dtype must be f32, bf16 or f16");

  // Scoped XPU FP16 model projections use the pinned producer's arithmetic.
  // A single checkpoint projection is one source group. Grouped QKV/QKVZ/
  // gate-up storage will use the same VT seam; legacy callers remain below.
  if (d.q.device.type == vt::DeviceType::kXPU &&
      d.activation_dtype == vt::DType::kF16 && x.dtype == vt::DType::kF16 &&
      out_dtype == vt::DType::kF16 && w.codebook == 2 &&
      (w.Bits() == 4 || w.Bits() == 6)) {
    const int panel_columns = M > 128 ? vt::Exl3W8A8ModelPanelColumns() : 128;
    if (M > 128) (void)vt::PlanExl3W8A8(M, K, N, 1, w.Bits(), panel_columns);
    else (void)vt::PlanExl3SmallM(M, K, N, w.Bits());
    auto trellis = ResidentWeight(d, w.trellis);
    auto suh = Reshape(ResidentWeight(d, w.suh), {1, K});
    auto svh = ResidentWeight(d, w.svh);
    if (w.single_source_map.rank == 0) {
      w.single_source_map.dtype = vt::DType::kI32;
      w.single_source_map.rank = 1;
      w.single_source_map.shape[0] = N / 128;
    }
    Tensor shard = ResidentWeight(d, w.single_source_map, {}, [&](Tensor& t) {
      d.b.Memset(d.q, t.data, 0, t.Bytes());
    });
    DBuf out(d, vt::DType::kF16, {M, N});
    if (M > 128) {
      vt::detail::Exl3GroupedW8A8Model(d.q, out.t(), x, trellis, suh, svh, shard,
          {w.Bits(), w.codebook, w.name.c_str(), panel_columns},
          w.single_source_map.d_dev, w.w8a8_model_map);
      return out;
    }
    const auto plan = vt::PlanExl3SmallM(M, K, N, w.Bits());
    DBuf had(d, vt::DType::kF16, {1, K / 16, plan.padded_rows, 16});
    DBuf parts(d, vt::DType::kF32, {plan.splits, M, N});
    vt::detail::Exl3GroupedLinearModel(d.q, out.t(), x, trellis, suh, svh, shard,
        had.t(), parts.t(), {w.Bits(), w.codebook, w.name.c_str()},
        w.single_source_map.d_dev, w.w8a8_model_map);
    return out;
  }

  static const bool xpu_cast_fusion = [] {
    const char* setting = std::getenv("VT_XPU_EXL3_CAST_FUSION");
    return setting == nullptr || std::string(setting) != "0";
  }();
  const bool fuse_casts = d.q.device.type == vt::DeviceType::kXPU && xpu_cast_fusion;
  DBuf a_owned;
  vt::Tensor a = x;
  if (x.dtype != vt::DType::kF16 && !fuse_casts) {
    a_owned = DBuf(d, vt::DType::kF16, {M, K});
    vt::CastF16(d.q, a_owned.t(), x);
    a = a_owned.t();
  }
  DBuf a_had(d, vt::DType::kF16, {M, K});

  // exl3.py:10,135. Upstream dispatches to `reconstruct_hgemm` (dequantize +
  // cuBLAS fp16 GEMM) when M > AUTO_RECONSTRUCT_THRESHOLD (144), and to the
  // fused cooperative kernel (`exl3_gemm`) otherwise. The cooperative kernel
  // is faster for small M because it avoids materializing the full weight;
  // cuBLAS wins for large M because the persistent kernel's occupancy drops.
  //
  // Upstream's reconstruct path is CUDA-only, and so is our registration of
  // `kExl3ReconstructGemm`. A backend with no native reconstruct kernel (CPU,
  // ROCm, Vulkan today) keeps `Exl3Gemm` at every M, as it did before #3150;
  // a device-blind threshold refused M > 144 there
  // (ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2). `OpRegistered` excludes the CPU
  // reference tier, which has no reconstruct kernel to lend anyway.
  //
  // No divisibility guard is needed here: `Exl3ReconstructGemm` requires k and
  // n to be multiples of 128, and `Exl3Gemm` already refuses the same shapes
  // (src/vt/ops.cpp), so the dispatch cannot turn a served shape into a refusal.
  constexpr int64_t kReconstructThreshold = 144;
  const bool use_reconstruct =
      M > kReconstructThreshold &&
      vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, d.q.device.type);
  // THE WEIGHT SCRATCH IS NOT A POOL BLOCK (QUANT-EXL3 W7,
  // .agents/specs/quant-exl3-recon-scratch.md). The fp16 [K, min(N, 32768)]
  // reconstructed weight comes from the backend's ONE persistent buffer per
  // (device, queue). As a per-call `DBuf` it was part of every CUDA-graph step's
  // demand profile, so each captured graph pinned its own block of every class,
  // about 786 MiB per slot on Qwen3.8-27B EXL3, and GB10 ran out of host memory
  // at c = 32 (ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ).

  vt::Tensor trellis = ResidentWeight(d, w.trellis);
  vt::Tensor suh = ResidentWeight(d, w.suh);
  vt::Tensor svh = ResidentWeight(d, w.svh);

  vt::Exl3GemmArgs args;
  args.bits = w.Bits();
  args.codebook = w.codebook;
  args.fuse_casts = fuse_casts;
  args.debug_name = w.name.c_str();

  auto run_gemm = [&](vt::Tensor& out) {
    if (use_reconstruct) {
      vt::Exl3ReconstructGemm(d.q, out, a, trellis, suh, svh, a_had.t(), args);
    } else {
      vt::Exl3Gemm(d.q, out, a, trellis, suh, svh, a_had.t(), args);
    }
  };

  if (out_dtype == vt::DType::kF16 || fuse_casts) {
    DBuf c(d, out_dtype, {M, N});
    run_gemm(c.t());
    return c;
  }
  DBuf c32(d, vt::DType::kF32, {M, N});
  run_gemm(c32.t());
  if (out_dtype == vt::DType::kF32) return c32;
  DBuf cbf(d, vt::DType::kBF16, {M, N});
  vt::CastBf16(d.q, cbf.t(), c32.t());
  return cbf;
}

// Device-resident f32 upcast of a bf16 owned weight (per-head q/k norm weights,
// consumed by the f32 RMSNorm), uploaded ONCE.
inline Tensor ResidentWeightF32(Dev d, const OwnedTensor& w, const std::vector<int64_t>& shape) {
  // ONE BODY, in dense_device_glue.h, shared with the private twin in
  // qwen3_5.cpp. It used to be spelled out here and again there, which is how
  // #2711 came to be one defect in two places: the upload handed an
  // asynchronous `Copy` a function-local `std::vector<float>` and destroyed it
  // at this closing brace. The alias-vs-upload branch (#1946), the upload and
  // the drain that closes #2711 all live in `InstallResidentF32` now.
  if (!w.d_dev_f32) InstallResidentF32(d, w, WeightF32(w));
  return MakeTensor(w.d_dev_f32.get(), DType::kF32, d.q.device, shape);
}

// The two dim-1 unbind slices of the flash KV cache (num_blocks, 2, block_size,
// Hkv, Dh): a rank-4 strided view (block stride 2*bs*Hkv*Dh). Mirrors
// qwen3_5.cpp KvSlice.
//
// MiMoV2: when `head_size_v` is set and non-zero, the V slice (`which == 1`)
// uses `head_size_v` for the head dimension and its strides. The K slice
// (`which == 0`) always uses `head_size`. The page layout is
// [K: bs*Hkv*Dh_k | V: bs*Hkv*Dh_v] per block, so the V offset already
// accounts for the K region size.
inline Tensor KvSlice(const PagedKvCache& kv, vt::Device dev, int which) {
  const int64_t bs = kv.block_size, h = kv.num_kv_heads;
  const int64_t dd_k = kv.head_size;
  const int64_t dd_v = (kv.head_size_v > 0) ? kv.head_size_v : kv.head_size;
  const int64_t dd = (which == 1) ? dd_v : dd_k;
  Tensor t;
  t.data = static_cast<char*>(kv.data) +
           static_cast<size_t>(which) * static_cast<size_t>(bs * h * dd_k) *
               vt::SizeOf(kv.dtype);
  t.dtype = kv.dtype;
  t.device = dev;
  t.rank = 4;
  t.shape[0] = kv.num_blocks;
  t.shape[1] = bs;
  t.shape[2] = h;
  t.shape[3] = dd;
  t.stride[0] = bs * h * dd_k + bs * h * dd_v;  // block stride = K region + V region
  t.stride[1] = h * dd;  // within-block stride uses THIS slice's head dim
  t.stride[2] = dd;
  t.stride[3] = 1;
  return t;
}

// Per-step device inputs, uploaded ONCE (positions/attention metadata + the
// per-step cos|sin cache), reused by every one of the N full-attn layers.
struct StepInputs {
  DBuf positions;        // i32 [T]
  DBuf slot_mapping;     // i64 [T]
  DBuf block_table;      // i32 [num_reqs, cols]
  DBuf seq_lens;         // i32 [num_reqs]
  DBuf query_start_loc;  // i32 [num_reqs+1]
  DBuf cos_sin;          // f32 [T, rotary_dim]   (per-step cos|sin, f32 A/B path)
  DBuf cos_sin_bf16;     // bf16 [T, rotary_dim]  (vLLM-dtype cache, bf16 rope)
  DBuf rope_row_idx;     // i32 [T] = 0..T-1 (token-index lookup into the cache)
};

inline StepInputs BuildStepInputs(Dev d, const std::vector<int32_t>& positions,
                                  const CommonAttentionMetadata& am, const HfConfig& cfg) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const int rot = static_cast<int>(cfg.rotary_dim);
  // Identity row index 0..T-1: the per-step cos|sin cache is TOKEN-indexed (row t
  // already holds f(positions[t])), so RopeFromCache — which looks the cache up by
  // its `positions` argument — must be fed the identity so it reads cache[t], the
  // correct angle for token t at ANY real position (prefill AND decode). Feeding
  // the real positions would double-apply the position map (and skip decode rows
  // whose position >= T, the cache row count). Mirrors the token-indexed fused
  // AttnQkNormRope(Gate) consumer used by the 27B/35B forward.
  // CUDA-GRAPH SAFETY (W7): the identity row index is uploaded by a host->device
  // copy that a decode CUDA graph CAPTURES as a memcpy node, baking this host
  // SOURCE ADDRESS and re-reading it on every replay. A stack-local vector here
  // (the pre-W7 form) is destroyed the moment BuildStepInputs returns, so every
  // replay would read freed stack memory — MEASURED as a wrong RoPE and a wrong
  // token on the first captured decode step of Qwen3-Coder. It is instead served
  // from a process-persistent per-T table: the contents are a pure function of T
  // (row_idx[i] == i), the storage is created once per distinct T and NEVER
  // resized or moved (a std::map node's vector never reallocates), so a captured
  // pointer stays valid for the process lifetime. Byte-identical contents to the
  // stack-local form, so the eager (non-graph) dense path is unchanged.
  //
  // Same discipline as the graph-safe persistent scratch on the device side
  // (EnsureMoeScratch / EnsureMoePartials, cuda_matmul_nvfp4.cu:767,986: RETIRE,
  // never free/move, because the pointer is baked into a captured graph).
  static std::mutex row_idx_mu;
  static std::map<int64_t, std::vector<int32_t>> row_idx_by_t;
  const std::vector<int32_t>* row_idx = nullptr;
  {
    std::lock_guard<std::mutex> lk(row_idx_mu);
    auto it = row_idx_by_t.find(T);
    if (it == row_idx_by_t.end()) {
      std::vector<int32_t> v(static_cast<size_t>(T));
      for (int64_t i = 0; i < T; ++i) v[static_cast<size_t>(i)] = static_cast<int32_t>(i);
      it = row_idx_by_t.emplace(T, std::move(v)).first;
    }
    row_idx = &it->second;
  }
  StepInputs s{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kF32, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kBF16, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kI32, {T}, row_idx->data()),
  };
  if (rot > 0) {
    // Build the per-step cos|sin cache ONCE (RopeCosSinCache does the fp64
    // pow/cos/sin over T*rot/2 elements, keyed by the REAL positions), so the
    // per-layer attention preamble reads it via RopeFromCache instead of
    // recomputing the fp64 transcendentals N times (the old RopeNeox). The cache
    // is f32; RopeCosSinCacheKernel uses the SAME double-angle math as
    // RopeNeoxKernel, so an f32-precision RoPE off this cache is BIT-IDENTICAL to
    // RopeNeox (cuda_ops.cu:720).
    vt::RopeCosSinCache(d.q, s.cos_sin.t(), s.positions.t(), MakeRopeArgs(cfg));
    // The bf16 model-dtype cache is only consumed by the opt-in RopeFromCache
    // path; skip the cast on the default (RopeNeox) path so default-OFF stays
    // perf-neutral vs baseline.
    if (RopeCacheEnabled()) vt::CastBf16(d.q, s.cos_sin_bf16.t(), s.cos_sin.t());
  }
  return s;
}

// One Qwen3 dense self-attention block (qwen3.py::Qwen3Attention.forward). `dhn`
// is the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
inline DBuf AttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
                      const Tensor& dhn, const StepInputs& si,
                      const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T,
                      const TensorParallel* tp = nullptr) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  // RoPE base/scaling now come from MakeRopeArgs(cfg) (llama3-aware); no local base.
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(w.qkv_bias.Empty(),
           "qwen3 dense forward: attention_bias not supported yet (Qwen3-0.6B has none)");
  // KV-FP8 W3: a third storage dtype joins the two float ones — 1-byte fp8
  // (`vt::DType::kI8`), which `IsFp8KvCache` admits only together with a
  // matching fp8 interpretation, so a bare `kI8` view still fails here. This
  // guard and the routing at the store/read below are ONE decision: leaving it
  // at the two float dtypes made `fp8_kv` provably false at every call and the
  // fp8 arms of `WriteKvCache`/`ApplyKvCacheQuant` unreachable, which is the
  // shape G12 exists to keep out (`qwen3_5.cpp:5313` is the same widening on
  // the other routed family).
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32 ||
               IsFp8KvCache(kv),
           "qwen3 dense: KV cache must be bf16, f32, or 1-byte fp8 "
           "(--kv-cache-dtype fp8)");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "qwen3 dense: KV cache head dims mismatch config");

  // vLLM runs Qwen3-0.6B in BF16: the qkv GEMM, per-head q/k RMSNorm (variance in
  // f32, result rounded to bf16), RoPE, and flash attention all flow bf16. Token-
  // exactness with the bf16 oracle requires mirroring that rounding exactly, so
  // the DEFAULT path keeps q/k/v/query/attn in bf16 (matching vLLM's per-op bf16
  // stores). VT_QWEN3_ATTN_F32=1 selects the f32 A/B path (q/k RMSNorm+RoPE in
  // f32, exercising the kAttnQkNormRope catalog recipe via cos/sin cache) — a
  // more-precise deviation, kept for diagnostics.
  const bool attn_f32 = [] {
    const char* e = std::getenv("VT_QWEN3_ATTN_F32");
    return e != nullptr && e[0] == '1';
  }();
  const DType adt = attn_f32 ? DType::kF32 : DType::kBF16;

  // Merged QKVParallelLinear. The weight owner is ONE raw-NK [qdim+2kdim, H]
  // tensor (vLLM's stacked qkv_proj). Two dispatch paths:
  //  - MERGE (default): one MatmulBT over the whole owner -> [T, qdim+2kdim],
  //    then QkvSplit into the contiguous q/k/v shards. Mirrors vLLM's single
  //    qkv GEMM + .split([q,kv,kv]) and folds the tiny GQA k/v GEMMs into one
  //    wide, tensor-core-efficient GEMM. Byte-affecting (near-tie-gated).
  //  - 3-SHARD (VT_QWEN3_QKV_MERGE=0): slice the owner's output rows and project
  //    each shard separately (the byte-identical baseline for the A/B).
  // The EXL3 arm PRODUCES q/k/v rather than filling them, so it declares them
  // empty and move-assigns; every other arm needs them allocated up front.
  // Allocating unconditionally cost three pooled blocks per attention block per
  // step that were never read.
  DBuf q, k, v;
  if (!w.IsExl3()) {
    q = DBuf(d, adt, {T, qdim});
    k = DBuf(d, adt, {T, kdim});
    v = DBuf(d, adt, {T, kdim});
  }
  if (w.IsExl3()) {
    // QUANT-EXL3 (#2181). THREE projections, not one merged operand, because
    // that is how an EXL3 checkpoint stores them: the trellis is
    // `[k/16, n/16, 32*bits]`, so joining on the output dim interleaves per
    // input tile rather than row-stacking. That merge is valid for this family
    // (`had_r_128` blocks the output in 128s and q=2048, k=v=512 are each a
    // multiple of 128, so no block straddles two matrices) and is worth doing,
    // but it is owed its own gate — `## Owed` in `specs/quant-exl3-shared.md`.
    // Producing q/k/v directly also means no `QkvSplit`.
    VT_CHECK(adt == DType::kBF16,
             "qwen3 dense: the EXL3 arm requires a bf16 activation "
             "(VT_QWEN3_ATTN_F32=1 is not supported on the quantized path)");
    q = dense_attn::Exl3MatmulD(d, dhn, w.q_proj_exl3, adt);
    k = dense_attn::Exl3MatmulD(d, dhn, w.k_proj_exl3, adt);
    v = dense_attn::Exl3MatmulD(d, dhn, w.v_proj_exl3, adt);
  } else if (w.IsNvfp4()) {
    // NVFP4 W4A16 qkv — ALWAYS the merged form: vLLM owns exactly one merged
    // `qkv_proj` parameter and repacks it WHOLE into one Marlin operand
    // (marlin_utils_fp4.py:221-306), so there is no 3-shard analog to A/B here
    // (slicing a Marlin-interleaved operand is not row-addressable). The a16
    // activation must be bf16, which is the default path's dtype.
    VT_CHECK(adt == DType::kBF16,
             "qwen3 dense: NVFP4 W4A16 qkv requires a bf16 activation "
             "(VT_QWEN3_ATTN_F32=1 is not supported on the quantized path)");
    DBuf qkv = dense_nvfp4::MatmulNvfp4W4A16D(d, dhn, w.qkv_proj_fp4, adt);
    vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
  } else {
    Tensor wqkv = ResidentWeight(d, w.qkv_proj);
    if (Qwen3QkvMergeEnabled()) {
      DBuf qkv(d, adt, {T, qdim + 2 * kdim});
      vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
      vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
    } else {
      Tensor wq = wqkv.Slice(0, 0, qdim);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      vt::MatmulBT(d.q, q.t(), dhn, wq);
      vt::MatmulBT(d.q, k.t(), dhn, wk);
      vt::MatmulBT(d.q, v.t(), dhn, wv);
    }
  }

  // Per-head q/k RMSNorm (RMSNorm(head_dim), non-gemma) BEFORE partial NeoX RoPE.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  // qk-norm is OPTIONAL: Qwen3 applies per-head q/k RMSNorm before RoPE
  // (w.q_norm/w.k_norm loaded); Llama (LlamaForCausalLM) has NO qk-norm, so its
  // loader leaves both EMPTY and the norm step is SKIPPED — RoPE runs directly on
  // the raw q/k. Byte-preserving for Qwen3 (norm present => same code path); the
  // fused-chain qk-norm-rope recipe requires the norm weights, so a qk-norm-absent
  // model always takes the standalone `else` branch below.
  const bool has_qk_norm = !w.q_norm.Empty();
  // The fused preamble applies to the BF16 path too, but ONLY when the rope cache
  // is on. The recipe's third step is kRope == RopeFromCache; the bf16 branch's
  // DEFAULT rope is RopeNeox. Routing bf16 through the recipe without this guard
  // would silently swap one RoPE implementation for another, and the near-tie
  // goldens would move for a reason that has nothing to do with fusion.
  //
  // With the guard, the recipe's Tier-0 composite is EXACTLY the standalone
  // sequence the else-branch runs — RmsNorm(q), RmsNorm(k), RopeFromCache — for
  // either dtype, so adoption is behaviour-preserving by construction and the
  // only new thing is the backend's fused realisation.
  const bool fused_preamble =
      FusedChainAdoptEnabled() && rot > 0 && has_qk_norm && (attn_f32 || RopeCacheEnabled());
  if (fused_preamble) {
    // f32 A/B ADOPT: the whole preamble through vt::FusedChain(kAttnQkNormRope) —
    // the Tier-0 composite = RmsNorm(q,false) + RmsNorm(k,false) + RopeFromCache,
    // byte-identical to the hand-call (test_ops_fused_chain.cpp). Qwen3's reuse of
    // the fusion catalog's non-gated qk-norm-rope recipe (f32 cos/sin cache).
    // Weight dtype follows q's (RmsNorm requires w.dtype == x.dtype), and the
    // rope operands follow the branch that would otherwise have run: the f32 path
    // rotates against the f32 cache indexed by real positions, the bf16 path
    // against the bf16 cache indexed by the identity row (the position map is
    // already baked into the cache rows).
    Tensor wqn = attn_f32 ? ResidentWeightF32(d, w.q_norm, {Dh})
                          : ResidentWeight(d, w.q_norm, {Dh});
    Tensor wkn = attn_f32 ? ResidentWeightF32(d, w.k_norm, {Dh})
                          : ResidentWeight(d, w.k_norm, {Dh});
    Tensor rope_cache = attn_f32 ? si.cos_sin.t() : si.cos_sin_bf16.t();
    Tensor rope_idx = attn_f32 ? si.positions.t() : si.rope_row_idx.t();
    vt::FusedBinding b;
    b.op[0] = &q2;
    b.op[1] = &wqn;
    b.op[2] = &k2;
    b.op[3] = &wkn;
    b.op[4] = &q3;
    b.op[5] = &k3;
    b.op[6] = &rope_cache;
    b.op[7] = &rope_idx;
    b.n = 8;
    vt::FusedParams p;
    p.eps = eps;
    p.rope = MakeRopeArgs(cfg);
    vt::FusedChain(d.q, vt::kAttnQkNormRope, b, p);
  } else {
    // BF16 attention preamble: standalone per-head RMSNorm (weight dtype == q
    // dtype) then partial NeoX RoPE. The norm weight follows q's dtype so bf16 q ·
    // bf16 q_norm, matching vLLM's per-op bf16 stores.
    //
    // RoPE has two paths (RopeCacheEnabled / VT_QWEN3_ROPE_CACHE):
    //  - DEFAULT (OFF): in-place RopeNeox — byte-identical + deterministic (the
    //    committed near-tie goldens + the stable SACRED gate).
    //  - OPT-IN (ON): RopeFromCache off the per-step cos|sin cache — MEASURED the
    //    dominant dense-TTFT lever (recomputing fp64 pow/cos/sin per element per
    //    layer, RopeNeox, is 36.5% of prefill GPU-busy; fp64 ≈ 1/64 fp32 on GB10).
    //    The cache is cast to BF16 so bf16 q/k rotate against a bf16 cos|sin —
    //    EXACTLY vLLM's RotaryEmbedding (base.cpp initialize_cache()), the
    //    vLLM-FAITHFUL rope. See RopeCacheEnabled() for why it is opt-in (CUDA
    //    FMA non-byte-identity + engine near-tie nondeterminism → flaky gate).
    //
    // Token-indexed lookup (opt-in path): cache row t already encodes positions[t], so
    // RopeFromCache is fed the identity row index (si.rope_row_idx) — NOT the
    // real positions — so cache[t] is read for token t at ANY position (the
    // position map is already applied when the cache is built), correct for
    // prefill AND decode.
    // Per-head q/k RMSNorm — Qwen3 only (SKIPPED when the model has no qk-norm,
    // e.g. Llama, which leaves w.q_norm/w.k_norm empty). RoPE runs either way.
    // The hand-call takes the registered fused op when the device has one, so
    // the ADOPT path (FusedChain -> the recipe's fast realisation) and this
    // fallback agree on every backend, which is the recipe's byte-exact
    // composite contract. Where no fast op is registered (CPU) both realizations
    // stay the standalone three-op sequence below.
    const bool fused_preamble_op =
        has_qk_norm && rot > 0 && RopeCacheEnabled() && !attn_f32 &&
        vt::OpRegistered(vt::OpId::kAttnQkNormRope, d.q.device.type);
    if (fused_preamble_op) {
      Tensor wqn = ResidentWeight(d, w.q_norm, {Dh});
      Tensor wkn = ResidentWeight(d, w.k_norm, {Dh});
      vt::AttnQkNormRope(d.q, q3, k3, wqn, wkn, si.cos_sin_bf16.t(), si.rope_row_idx.t(),
                         vt::RmsNormArgs{eps, false}, MakeRopeArgs(cfg));
    } else {
      if (has_qk_norm) {
        Tensor wqn = attn_f32 ? ResidentWeightF32(d, w.q_norm, {Dh})
                              : ResidentWeight(d, w.q_norm, {Dh});
        Tensor wkn = attn_f32 ? ResidentWeightF32(d, w.k_norm, {Dh})
                              : ResidentWeight(d, w.k_norm, {Dh});
        vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
        vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
      }
      if (RopeCacheEnabled() && rot > 0) {
        Tensor k3v = k3;
        vt::RopeFromCache(d.q, q3, &k3v, si.rope_row_idx.t(), si.cos_sin_bf16.t(),
                          MakeRopeArgs(cfg));
      } else {
        // DEFAULT (byte-identical, deterministic): in-place bf16 NeoX RoPE with
        // per-element fp64 cos/sin, mirroring vLLM's rotary_emb bf16 rounding.
        vt::RopeNeox(d.q, q3, k3, si.positions.t(), MakeRopeArgs(cfg));
      }
    }
  }

  // v [T,Hkv,Dh] view.
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});

  // Write the rope'd K + V into the paged cache, then run causal GQA paged
  // attention over the cache. On the bf16 default the K/V/query are ALREADY bf16
  // (matching the bf16 cache directly, no cast); the f32 A/B down-casts K/V for a
  // bf16 cache.
  // The "auto" ReshapeAndCache copy requires the K/V dtype == cache dtype. Cast
  // K/V to the cache dtype only when they differ (bf16 default + bf16 cache =
  // no-op, the common production case).
  Tensor kw = k3;
  Tensor vw = v3;
  // KV-FP8 W3: on an fp8 cache there is NO cast to do. `vt::ReshapeAndCacheFp8`
  // takes the model-dtype K/V and performs `Quantize(hp / k_scale|v_scale)`
  // itself (`quant_utils.cuh:296-300`), so the buffers below are allocated at
  // the SOURCE dtype and the branch is skipped — casting to `kI8` would be
  // meaningless, and allocating a `kI8` scratch here would silently halve it.
  const bool fp8_kv = IsFp8KvCache(kv);
  const DType cast_dt = fp8_kv ? adt : kv.dtype;
  DBuf kcast(d, cast_dt, {T, Hkv, Dh});
  DBuf vcast(d, cast_dt, {T, Hkv, Dh});
  if (!fp8_kv && kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    }
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  WriteKvCache(d.q, kv, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, adt, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // SPEC-DFLASH2 W10 (#1857): forward the runner's spec-as-decode
  // classification through the shared seam. Today only the d256 CUDA decode
  // lane consumes it (the d128 arms keep their q==1 gates), so every model on
  // this block routes exactly as before unless a d256 verify arrives.
  pa.uniform_spec_query_len = meta.uniform_spec_query_len;
  ApplyKvCacheQuant(pa, kv);
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16. The attention
  // output is already bf16 on the default path (matching vLLM's bf16 flash-attn
  // output); the f32 A/B down-casts it so MatmulBT's inputs share dtype.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  DBuf attn_bf(d, DType::kBF16, {T, Hq * Dh});
  if (adt != DType::kBF16) {
    vt::CastBf16(d.q, attn_bf.t(), Reshape(attn.t(), {T, Hq * Dh}));
    o_in = attn_bf.t();
  }
  if (w.IsExl3()) {
    // QUANT-EXL3 (#2181): o_proj is a single projection in every arm, so the
    // EXL3 form differs from the bf16 one only in the kernel it dispatches.
    DBuf o = dense_attn::Exl3MatmulD(d, o_in, w.o_proj_exl3, DType::kBF16);
    (void)rot;
    Tensor ot = o.t();
    TpAllReduceSum(tp, d.q, ot);
    return o;
  }
  if (w.IsNvfp4()) {
    // NVFP4 W4A16 o_proj — the bf16 attention output IS the a16 activation.
    DBuf o = dense_nvfp4::MatmulNvfp4W4A16D(d, o_in, w.o_proj_fp4, DType::kBF16);
    (void)rot;
    // RowParallelLinear all-reduce (linear.py:1766). No-op at tp_size==1.
    Tensor ot = o.t();
    TpAllReduceSum(tp, d.q, ot);
    return o;
  }
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  (void)rot;
  // o_proj is a RowParallelLinear: sum the per-rank partial products so every
  // rank holds the full [T,H] output (linear.py:1766). No-op — nothing enqueued —
  // at tp_size==1, so the single-GPU forward is byte-identical.
  Tensor ot = o.t();
  TpAllReduceSum(tp, d.q, ot);
  return o;
}

}  // namespace dense_attn
}  // namespace vllm
