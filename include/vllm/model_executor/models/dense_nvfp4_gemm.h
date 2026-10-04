// Shared NVFP4 **W4A16** (4-bit weights, BF16 activations) dense GEMM glue for
// full-attention models — the QUANT-SCHEME additivity seam.
//
// Extracted (behavior-preserving) from the anonymous namespace of
// src/vllm/model_executor/models/qwen3_5.cpp so a dense model whose forward is
// already token-exact in BF16 (Qwen3 `Qwen3ForCausalLM`, qwen3.cpp) can swap a
// BF16 projection for an NVFP4 one WITHOUT re-deriving the resident-repack,
// align-cache, workspace and dispatch machinery. This mirrors how
// dense_weight_loaders.h, device_pool.h and dense_attn_block.h were extracted
// for reuse (.agents/specs/sweep-qwen3-32b-nvfp4a16.md §Port map SEAM #1).
//
// SCOPE — W4A16 ONLY. This header deliberately does NOT carry the true-W4A4
// (fp4-ACTIVATION) path (`MatmulNvfp4Fp4D`, ScaledFp4Quant, the cutlass
// swizzled-blockscale/alpha residents). W4A4 stays private to qwen3_5.cpp with
// the 27B it serves. Selection is by the SAME predicate vLLM uses: a scheme
// whose `input_activations` is null is W4A16 (`Nvfp4Weight::IsTrueW4A4()` ==
// false, i.e. alpha == 0), and vLLM then FORCES the Marlin kernel —
// vllm/model_executor/kernels/linear/__init__.py:879-881
//   `elif linear_backend == "auto" and use_a16: force_kernel = MarlinNvFp4LinearKernel`
//   ("Force a16 (Marlin) when running weight-only quantization")
// bypassing the capability-based kernel registry entirely. So on sm_121 the
// dispatch below is Marlin, unconditionally, exactly like the oracle's.
//
// UPSTREAM CHAIN (ported FROM, cited per the ground-every-impl rule):
//   * scheme            vllm/model_executor/layers/quantization/compressed_tensors/
//                         schemes/compressed_tensors_w4a4_nvfp4.py:29-32,95-141
//                         (`CompressedTensorsW4A4Fp4(use_a16=True)`; note vLLM
//                         has NO separate W4A16 class — the a16 flag reuses this)
//   * kernel selection  vllm/model_executor/kernels/linear/__init__.py:842,879-892
//   * marlin wrapper    vllm/model_executor/kernels/linear/nvfp4/marlin.py:21-57
//   * repack + scales   vllm/model_executor/layers/quantization/utils/
//                         marlin_utils_fp4.py:61-122 (nvfp4_marlin_process_scales),
//                         :142-154 (nvfp4_marlin_process_global_scale),
//                         :221-306 (prepare_fp4_layer_for_marlin),
//                         :157-218 (apply_fp4_marlin_linear)
//   * CUDA GEMM         csrc/libtorch_stable/quantization/marlin/marlin.cu:545,600-611
//   * global-scale epi  csrc/libtorch_stable/quantization/marlin/marlin_template.h:1655-1657
// Our vendored, bit-exact lift of those primitives is include/vt/cuda/marlin_repack.h
// (MarlinRepackExpertWeight / MarlinProcessExpertScales /
// MarlinNvfp4CombinedScaleFactor / MarlinNvfp4ProcessGlobalScale) driving
// vt::MoeGroupedGemmNvfp4Marlin with num_experts=1 — the SINGLE-EXPERT grouped
// GEMM is how a dense [M,K]x[N,K]^T W4A16 linear runs on the MoE Marlin entry
// point (vLLM reaches the same csrc kernel through `ops.marlin_gemm`).
//
// KNOWN DUPLICATION (recorded, not accidental): qwen3_5.cpp retains its own
// copies of these definitions, exactly as it retains its own Dev/DBuf after the
// dense_attn_block.h extraction (see that header's preamble). Unifying the two
// device-glue families is a separate, gate-model-touching refactor; this header
// deliberately leaves the 27B/35B hot path BYTE-UNTOUCHED so the 235/235 +
// 315/315 regressions cannot move. Tracked in the spike's §Risks/decisions.
#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"  // DequantMxfp4ToBf16
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // DequantNvfp4ToBf16
#include "vllm/model_executor/models/dense_device_glue.h"    // Dev/DBuf/MakeTensor
#include "vllm/model_executor/models/qwen3_5_weights.h"      // Nvfp4Weight
#include "vt/backend.h"
#include "vt/dtype.h"  // VT_CHECK
#include "vt/ops.h"

#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {
namespace dense_nvfp4 {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::Backend;
using vt::DType;
using vt::Tensor;

// ── A2-Q2b (#810): the CALLER-OWNED resident, so a consumer can opt OUT of the
// address-keyed cache below without hand-rolling a parallel Marlin path.
//
// `MarlinDenseResidentFor` (below) keys its `static` map on the ADDRESS of the
// `Nvfp4Weight`. That is issue #984: destroy one engine and build another in
// the same process and the allocator can hand the new weights the old address,
// so the new weights inherit an entry marked `ready` whose device pointers
// belong to the previous engine. `Nvfp4Weight` already carries the
// `resident_marlin` `ResidentSlot` that #237 added to fix exactly this, and
// this header does not read it.
//
// A2-Q2b does not fix #984 — that needs the two-engine red-before #984 asks
// for, and it is out of this row's scope. It declines to INHERIT it: NemotronH
// owns its lm_head resident in a `ResidentSlot` (the same shape A2-Q2a used for
// the MoE arena) and hands it in here. Threading the resident through is
// additive — every existing caller passes nothing and keeps the address-keyed
// default byte-for-byte — so this EXTENDS the seam rather than forking it,
// which is what AGENTS.md `## Shared seams` asks for when the seam cannot
// represent the behaviour a consumer needs.
//
// Declared unconditionally (defined only under VT_MARLIN_NVFP4) so the
// dispatcher's signature does not change with the build flag; a pointer to an
// incomplete type is all the non-Marlin build ever needs.
struct MarlinDenseResident;

// VT_NVFP4_MARLIN (default ON): the vendored Marlin NVFP4 W4A16 GEMM is the
// validated path (35B gate +22%, token-for-token vs the pinned oracle). Only an
// explicit VT_NVFP4_MARLIN=0 opts back out to the naive redundant-dequant kernel
// (kept as an A/B escape hatch). Mirrors qwen3_5.cpp::MarlinMoeEnabled.
inline bool MarlinW4A16Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_MARLIN");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// VT_MOE_FUSED_W13 (default ON): run a gate+up PAIR as ONE Marlin GEMM over the
// N-concatenated operand (size_n = 2N) + one SiluAndMul, instead of two GEMMs.
// This is exactly vLLM's merged `gate_up_proj` MergedColumnParallelLinear, which
// `prepare_fp4_layer_for_marlin` repacks WHOLE as a single Marlin operand
// (marlin_utils_fp4.py:221-306) — so the FUSED layout is the vLLM-faithful one,
// and the split layout is our A/B fallback.
inline bool FusedGateUpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_FUSED_W13");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// VT_MARLIN_DENSE (default ON; VT_MARLIN_DENSE=0 opts back out to the MoE route):
// route the E=1 dense NVFP4/MXFP4 projections through vLLM's OWN dense marlin GEMM
// (vt::MarlinDenseGemm) instead of the single-expert MoE-marlin route. The dense
// kernel is direct-A + tile-per-CTA with vLLM's dense fp32-C_tmp reduce, so at M<=8
// it naturally runs the 48-CTA (sms-wide) grid the MoE path only reaches with the
// VT_MARLIN_E1_PAR1 clamp — WITHOUT that clamp's par regrouping, which costs one bf16
// ULP vs the oracle and flips a strict 32B token (row QUANT-CT-MXFP4-MARLIN-STRUCT /
// #50 / #54). Same resident weights + workspace; the repack permute is vLLM's shared
// marlin_permute for both dense and MoE. FLIPPED ON (row KERNEL-MARLIN-DENSE-EXEC):
// the dense reduce IS vLLM's own numerics — the teacher-forced near-tie razor on the
// 32B-NVFP4A16 SACRED gate scores max gap 0.000 nats (every dense token == vLLM's
// teacher-forced argmax, TIGHTER than the MoE route's 62 mnats), and the c8 decode
// marlin runs the 48-CTA grid at ~86us/call vs the MoE route's 128-CTA ~118us/call.
// The MoE route's greedy anchor (our_ids) shifts at two exact bf16 ties, so the 32B
// goldens were regenerated under dense-ON per the ratified-tie regen rule.
// Fused shared-expert gate_up dense route (VT_MARLIN_DENSE_PAIR, default ON,
// opt out with =0) — the sibling of MarlinDenseEnabled() for the ONE fused
// gate_up sink, which was still taking the single-expert MoE-marlin route.
// Shared-expert down-proj emits bf16 instead of f32 (VT_SHARED_DOWN_BF16,
// default ON, opt out with =0). BIT-IDENTICAL: both consumers (SharedExpertGate
// and MoeCombineGate) widen bf16 in-kernel, which is exact, and both re-round
// through bf16 on store. Drops one CastF32 launch per layer per step.
inline bool SharedDownBf16Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_SHARED_DOWN_BF16");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

inline bool MarlinDensePairEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MARLIN_DENSE_PAIR");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

inline bool MarlinDenseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MARLIN_DENSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// --- PERF-27B-DENSE-MARLIN-GATEUP (issue #365) -----------------------------
// The fused Marlin gate_up pair exists and is default-ON, but its only callers
// are the MoE shared expert. The DENSE MLP's W4A16 gate/up pair still launches
// TWO Marlin GEMMs, which is the measured 193-vs-129 Marlin calls/step against
// the pinned oracle on the identical batch-1 27B decode. vLLM's dense Qwen3.6
// MLP is ONE `MergedColumnParallelLinear` `gate_up_proj`
// (vllm/model_executor/layers/linear.py), repacked WHOLE as a single Marlin
// operand (marlin_utils_fp4.py:221-306), so one GEMM is the MIRRORED topology
// and the split pair is our divergence.
//
// VT_DENSE_MARLIN_GATEUP is DEFAULT ON (opt out with =0). It shipped OFF while
// the row held, because a lever's default moves on a measured same-binary A/B
// and never on an expectation. That A/B has now run: interleaved 4 reps per arm
// on nvidia/Qwen3.6-27B-NVFP4@0893e160 (GB10), toggle the only variable, caches
// dropped between arms — fused +2.12% at c1 (12.0823 vs 11.8313 tok/s) and
// +1.70% at c8 (83.6186 vs 82.2217), with COMPLETE SEPARATION at both
// concurrencies (c1 min fused 12.0203 > max split 11.8729; c8 min fused 83.2511
// > max split 82.4959), 4/4 paired, effect well outside each arm's spread.
//
// The correctness bar for the flip was token-exactness against the pinned
// ORACLE, NOT bit-equality with our own split path — the fused and split Marlin
// GEMMs differ by one bf16 ULP on ~0.1% of elements (the fp32 split-K reduce
// regroups the K-slices for a [2N,K] operand), MEASURED for this exact entry
// point in tests/vllm/model_executor/layers/test_linear_method.cpp:202. A
// 64-token greedy continuation captured on BOTH arms diffed IDENTICAL.
//
// The opt-out spelling mirrors the nearest parity levers, the sibling fused
// gate_up toggles above: MarlinDensePairEnabled (VT_MARLIN_DENSE_PAIR) and
// FusedGateUpEnabled (VT_MOE_FUSED_W13).
//
// Parsed by a PURE function so the truth table is testable without process
// -global state (the cached reader below can only ever observe one value).
inline bool DenseMarlinGateUpEnabledFor(const char* env) {
  return !(env != nullptr && env[0] == '0');
}

inline bool DenseMarlinGateUpEnabled() {
  static const bool on =
      DenseMarlinGateUpEnabledFor(std::getenv("VT_DENSE_MARLIN_GATEUP"));
  return on;
}

// The shape/scale precondition that makes N-concatenating a gate/up pair into
// ONE Marlin operand LEGAL — factored OUT of GateUpFusedEligible below so the
// dense MLP composes the SAME terms instead of restating them, and so a CPU
// test can pin the truth table (the composed guards below reach into a CUDA
// -only kernel family and answer false everywhere else).
//
// `scale2` equality is the load-bearing term. The merged resident emits ONE
// per-GEMM global scale over both shards, mirroring vLLM's merged parameter,
// which has exactly one `weight_global_scale`
// (compressed_tensors_w4a4_nvfp4.py:111-114). Two shards with DIFFERENT scale2
// cannot share it, so relaxing this equality would silently change numerics
// rather than fuse them.
inline bool GateUpPairFusableShape(const Nvfp4Weight& gw, const Nvfp4Weight& uw) {
  return !gw.Empty() && !uw.Empty() && !gw.IsTrueW4A4() && !uw.IsTrueW4A4() &&
         gw.is_mxfp4 == uw.is_mxfp4 && gw.group_size == uw.group_size &&
         gw.n == uw.n && gw.k == uw.k && gw.scale2 == uw.scale2;
}

// True when a DENSE MLP's gate/up pair takes the fused Marlin gate_up path,
// with the ONE thing a CPU test cannot observe — whether the vendored Marlin
// NVFP4 grouped GEMM is realized for the target device — INJECTED rather than
// probed. Every other term is a lever read or a shape/format term, so this
// overload is exactly the part of the guard a host build can put under test;
// `DenseMlpGateUpFusedMarlinEligible` below is this function applied to the real
// op registry and adds nothing else.
//
// Splitting it this way is what makes each term load-bearing under test: with
// the device term hard-false on CPU the composed predicate answered false no
// matter what the lever terms said, so deleting any one of them stayed green
// (fresh-review findings F1/F2). The composition is otherwise UNCHANGED — same
// terms, same order, same short-circuit — apart from the MXFP4 refusal below.
//
// MXFP4 IS REFUSED HERE, and that is not redundant with GateUpPairFusableShape.
// That shape term only requires the two shards to AGREE on the format, so a pair
// that is MXFP4 on both halves passes it — correctly, because the fused entry
// point this header owns (`GateUpFusedMarlinD`) handles E8M0/group-32 properly.
// The DENSE MLP call site does NOT reach that one: it reaches qwen3_5.cpp's
// `DenseGateUpFusedMarlinD` -> `SharedGateUpFusedMarlinD` ->
// `BuildMarlinDensePairResident`, a private NVFP4-ONLY copy that sizes the merged
// scale buffer at K/16 rows, runs the NVFP4 combined-scale-factor + global-scale
// processing, and pins `group_size = 16` / `mxfp4 = false` in the GEMM args. That
// is verbatim the defect recorded RED-first for the OTHER implementation at
// tests/vllm/model_executor/layers/test_linear_method.cpp:185-201 — group-32 E8M0
// scales misread as group-16 fp8-e4m3, "GROSSLY wrong". No dense loader sets
// `is_mxfp4` today (`LoadNvfp4AnyNaming`, qwen3_5_dense_weights.cpp:359-396, only
// ever produces NVFP4), so the defect is latent, not live — one loader line away.
// Refusing it here costs the currently-reached W4A16 NVFP4 config nothing
// (`is_mxfp4` is false on both halves) and keeps the split pair, which DOES
// handle MXFP4, as the answer for any dense MXFP4 checkpoint.
inline bool DenseMlpGateUpFusedMarlinEligibleWhen(const Nvfp4Weight& gw,
                                                  const Nvfp4Weight& uw,
                                                  bool marlin_nvfp4_op_available) {
  return DenseMarlinGateUpEnabled() && MarlinW4A16Enabled() &&
         FusedGateUpEnabled() && !gw.is_mxfp4 && !uw.is_mxfp4 &&
         GateUpPairFusableShape(gw, uw) && marlin_nvfp4_op_available;
}

// `vt::OpRegistered` is what makes a build WITHOUT the vendored Marlin NVFP4
// kernel (no VT_MARLIN_NVFP4) — and every non-CUDA device — answer false, so
// this needs no build-time gate of its own; the DSR ratchet
// (scripts/check-device-leakage.py) counts those, and a runtime probe of the
// op registry is the device-agnostic way to ask the same question.
inline bool DenseMlpGateUpFusedMarlinEligible(const Nvfp4Weight& gw,
                                              const Nvfp4Weight& uw,
                                              vt::DeviceType dev) {
  return DenseMlpGateUpFusedMarlinEligibleWhen(
      gw, uw, vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, dev));
}

// --- Execution counters (the "this path actually RAN" positive signal) ------
// A passing correctness gate does NOT prove a new code path was exercised — a
// mis-wired dispatch that silently fell back to the BF16 arm would also pass if
// the weights happened to be BF16. These counters make the W4A16 path
// OBSERVABLE, so the gate can assert on them (mirrors the ArchTacticStats
// pattern in cuda_arch_tactics.h, whose tests assert selections/fallbacks).
struct Nvfp4W4A16Stats {
  uint64_t marlin_gemms = 0;      // MatmulNvfp4MarlinD launches
  uint64_t fused_gate_up = 0;     // GateUpFusedMarlinD launches (one per MLP)
  uint64_t fallback_gemms = 0;    // naive vt::MatmulNvfp4 / CPU dequant launches
  uint64_t dense_gemms = 0;       // vt::MarlinDenseGemm launches (VT_MARLIN_DENSE route)
};

inline Nvfp4W4A16Stats& MutableW4A16Stats() {
  static Nvfp4W4A16Stats s;
  return s;
}

// Snapshot of the counters (test/diagnostic entry point).
inline Nvfp4W4A16Stats GetW4A16Stats() { return MutableW4A16Stats(); }

inline void ResetW4A16Stats() { MutableW4A16Stats() = Nvfp4W4A16Stats{}; }

// Device-resident views over an Nvfp4Weight's packed + scale buffers, uploaded
// ONCE (lazily) and owned by the (const) weight's shared_ptr for the model's
// lifetime. CUDA path only.
struct Nvfp4Dev {
  Tensor packed;
  Tensor scale;
};

inline Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
  if (!w.packed.d_dev) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    // ENG-LOAD-DIRECT-UPLOAD (issue #150). `LoadCtNvfp4W4A16`/`LoadCtMxfp4W4A16`
    // /`LoadCtNvfp4Raw` BORROW `packed` and `scale` from the safetensors mmap,
    // so this is the one host->device move of those bytes and it must be
    // accounted and followed by the same post-upload residency step every other
    // qualifying weight gets. Publishing on `d_dev` is what lets
    // `AdoptDeviceBytesAsHost` run (it keys on that slot), and it is the only
    // owner (Nvfp4Weight::ReleaseResident).
    vllm::load_stats::AddDeviceUpload(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.packed.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    AdoptDeviceBytesAsHost(d.b, w.packed);
  }
  if (!w.scale.d_dev) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    vllm::load_stats::AddDeviceUpload(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.scale.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    AdoptDeviceBytesAsHost(d.b, w.scale);
  }
  Nvfp4Dev r;
  r.packed = MakeTensor(w.packed.d_dev.get(), DType::kI8, d.q.device, {w.n, w.k / 2});
  // Scale grid is [N, K/group_size]: K/16 for NVFP4, K/32 for MXFP4.
  r.scale = MakeTensor(w.scale.d_dev.get(), DType::kI8, d.q.device, {w.n, w.k / w.group_size});
  return r;
}

// Host dequant of an fp4 weight to bf16 [K=in, N=out] (Matmul-B layout) — the
// CPU fallback (there is no CPU MatmulNvfp4 kernel). Bit-for-bit
// vllm::DequantNvfp4ToBf16 + transpose.
inline std::vector<uint16_t> DequantNvfp4ToBLayout(const Nvfp4Weight& w) {
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

// MXFP4 analog: host dequant of an E8M0/group-32 fp4 weight to bf16 [K=in, N=out]
// (Matmul-B layout) — the CPU / Marlin-disabled fallback. Bit-for-bit
// vllm::DequantMxfp4ToBf16 + transpose. Independent of the Marlin kernel's own
// E8M0 dequant, so a gate comparing the two paths is a real cross-check.
inline std::vector<uint16_t> DequantMxfp4ToBLayout(const Nvfp4Weight& w) {
  const int64_t out_dim = w.n, in_dim = w.k;
  std::vector<uint16_t> oi(static_cast<size_t>(out_dim) * in_dim);
  DequantMxfp4ToBf16(reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
                     reinterpret_cast<const uint8_t*>(w.scale.bytes.data()),
                     out_dim, in_dim, oi.data());
  std::vector<uint16_t> io(static_cast<size_t>(in_dim) * out_dim);
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      io[static_cast<size_t>(c) * out_dim + r] =
          oi[static_cast<size_t>(r) * in_dim + c];
  return io;
}

#ifdef VT_MARLIN_NVFP4

// --- Resident Marlin operands (repacked ONCE at first use) ------------------
// vLLM does this repack in `process_weights_after_loading`
// (marlin_utils_fp4.py:221-306); we do it lazily on first forward and then FREE
// the fp4 originals, so peak weight memory stays flat.
struct MarlinDenseResident {
  void* w = nullptr;  // i32 [K/16, N*2]  Marlin-interleaved weight
  void* s = nullptr;  // fp8 [K/16, N]    processed S0E5M3 scales
  void* g = nullptr;  // f32 [1]          processed global scale
  int64_t n = 0, k = 0;
  bool ready = false;
};

inline MarlinDenseResident& MarlinDenseResidentFor(const Nvfp4Weight* w) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDenseResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

inline void BuildMarlinDenseResident(Dev d, const Nvfp4Weight& w,
                                     MarlinDenseResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(w.k);
  const int N = static_cast<int>(w.n);
  const int gs = static_cast<int>(w.group_size);  // 16 (nvfp4) or 32 (mxfp4)
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(N) * 2);
  const size_t s_b = static_cast<size_t>(K / gs) * N;  // K/16 nvfp4, K/32 mxfp4
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = w.n;
  mr.k = w.k;
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index,
                                     static_cast<uint32_t*>(mr.w),
                                     static_cast<const uint8_t*>(dw.packed.data), K, N);
  if (w.is_mxfp4) {
    // MXFP4: E8M0 passthrough permute (no combined factor, no global scale).
    vt::cuda::MarlinProcessExpertScalesMxfp4(
        stream, static_cast<const uint8_t*>(dw.scale.data),
        static_cast<uint8_t*>(mr.s), K, N);
    const float g = 1.0F;  // unused (kernel skips global for E8M0)
    d.b.Copy(d.q, mr.g, &g, sizeof(float));
  } else {
    std::vector<const uint8_t*> bufs{
        reinterpret_cast<const uint8_t*>(w.scale.bytes.data())};
    std::vector<size_t> lens{w.scale.bytes.size()};
    const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
    vt::cuda::MarlinProcessExpertScales(stream,
                                        static_cast<const uint8_t*>(dw.scale.data),
                                        static_cast<uint8_t*>(mr.s), K, N, sf);
    const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
    d.b.Copy(d.q, mr.g, &g, sizeof(float));
  }
  d.b.Synchronize(d.q);  // repack done -> safe to free the fp4 originals
  w.ReleaseResident();
  mr.ready = true;
}

// Trivial single-expert moe_align inputs (all M tokens -> expert 0), cached per
// token count M. Avoids a per-GEMM moe_align launch + allocations.
struct DenseAlignCache {
  void* sorted = nullptr;  // i32 [max_tok]
  void* expert = nullptr;  // i32 [max_blk] (all 0)
  void* numpad = nullptr;  // i32 [1]
  void* topkw = nullptr;   // f32 [M] (ones; unused, mul_topk_weights=false)
  int block = 0, max_tok = 0, max_blk = 0;
};

inline DenseAlignCache& DenseAlignFor(Dev d, int M) {
  static std::mutex mu;
  static std::unordered_map<int, DenseAlignCache> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(M);
  if (it != cache.end()) return it->second;
  DenseAlignCache c;
  // c8 sliver (#46/#50): vLLM's DENSE marlin uses an 8-row tile for the a16
  // path at prob_m<=8 (`m_block_size_8 = prob_m<=8 && a16`,
  // csrc/libtorch_stable/quantization/marlin/marlin.cu:438) — NO padding. Our
  // grouped single-expert MoE-align picks block_size_m=16 at M=8
  // (MarlinMoeAlignBlockSizeSelect: 8*1/1/8 == 1.0 fails the `< 0.9` test at
  // cuda_marlin_repack.cu:362), padding 8 dummy rows into a 16-row tile
  // (m_block_size_8=false) and wasting ~half the tile — the reproducible
  // ~0.33ms/step at c8. The m_block_size_8=true 8-row kernels are vendored
  // (kernel_selector.h:3-8 nvfp4, :33-38 mxfp4) and the fp32 C_tmp reduce is
  // handled (marlin_mm_moe.cu:363-364 map block=8 -> thread_m_blocks=1 +
  // m_block_size_8=true). Force block=8 for the single-expert dense case at
  // M<=8 to match vLLM's dense tile exactly; M>8 is unchanged (already matches
  // vLLM, which drops m_block_size_8 above 8).
  c.block = (M <= 8) ? 8 : vt::cuda::MarlinMoeAlignBlockSizeSelect(M, 1, 1);
  vt::cuda::MarlinMoeAlignSizes(M, 1, 1, c.block, &c.max_tok, &c.max_blk);
  c.sorted = d.b.Alloc(static_cast<size_t>(c.max_tok) * sizeof(int32_t));
  c.expert = d.b.Alloc(static_cast<size_t>(c.max_blk) * sizeof(int32_t));
  c.numpad = d.b.Alloc(sizeof(int32_t));
  c.topkw = d.b.Alloc(static_cast<size_t>(M) * sizeof(float));
  void* tid = d.b.Alloc(static_cast<size_t>(M) * sizeof(int32_t));
  d.b.Memset(d.q, tid, 0, static_cast<size_t>(M) * sizeof(int32_t));  // -> expert 0
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(tid), M, 1, 1,
                                    c.block, static_cast<int32_t*>(c.sorted),
                                    static_cast<int32_t*>(c.expert),
                                    static_cast<int32_t*>(c.numpad));
  std::vector<float> ones(static_cast<size_t>(M), 1.0F);
  d.b.Copy(d.q, c.topkw, ones.data(), ones.size() * sizeof(float));
  d.b.Synchronize(d.q);
  d.b.Free(tid);
  return cache.emplace(M, c).first->second;
}

// Shared reduction workspace for the dense Marlin GEMMs (sms*4 i32 locks, mirror
// marlin_make_workspace_new). Zeroed ONCE at allocation; NOT re-zeroed per call
// (see the self-reset invariant below), exactly as vLLM allocates it with
// `torch.zeros` (marlin_utils.py:399-407) and reuses it across every call.
inline void* DenseMarlinWorkspace(Dev d, int* out_sms) {
  static std::mutex mu;
  static void* ws = nullptr;
  static int sms = 0;
  std::lock_guard<std::mutex> lk(mu);
  if (!ws) {
    sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
    ws = d.b.Alloc(static_cast<size_t>(sms) * 4 * sizeof(int32_t));
    // Zero ONCE. The kernel self-resets its barrier locks: our launch pins
    // use_atomic_add=false / use_fp32_reduce=true (cuda_moe_marlin.cu:141-142),
    // so the ONLY reachable cross-CTA reduce is the fp32 barrier, whose LAST
    // slice-block release re-zeroes the lock (marlin_template.h:2170
    // `barrier_release(&locks[locks_off], last)` -> `lock[0]=0` at :204); the
    // slice_count==1 case never touches locks at all (:2162). So every completed
    // GEMM leaves the workspace back at 0 and re-zeroing before each of the ~120
    // dense GEMMs/step is redundant host/launch work. (The non-self-clearing
    // atomic-add path at :614 is unreachable under this pinned config; if that
    // config ever flips, restore the per-call zero.)
    d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  }
  *out_sms = sms;
  return ws;
}

// y[M,N] = x[M,K] bf16 @ dequant(w).T via the single-expert Marlin W4A16 GEMM.
inline DBuf MatmulNvfp4MarlinD(Dev d, const Tensor& x, const Nvfp4Weight& w,
                               DType out_dtype, MarlinDenseResident* resident = nullptr) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  // A2-Q2b: a caller that owns the resident's lifetime supplies it; everyone
  // else keeps the address-keyed cache, unchanged (#984).
  MarlinDenseResident& mr = resident != nullptr ? *resident : MarlinDenseResidentFor(&w);
  if (!mr.ready) BuildMarlinDenseResident(d, w, mr);
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);  // zeroed once; kernel self-resets

  // VT_MARLIN_DENSE (default OFF): route through vLLM's OWN dense marlin GEMM.
  // Same resident (mr.w/mr.s/mr.g) + workspace; rank-2 operand views (the dense
  // launcher wants [K/16, N*2] / [K/gs, N], not the MoE rank-3 [1, ...]); NO
  // moe_align cache (direct-A). Byte-preserving vs the oracle (its own dense
  // fp32-C_tmp reduce). Only when the op is realized for this device.
  if (MarlinDenseEnabled() &&
      vt::OpRegistered(vt::OpId::kMarlinDenseGemm, d.q.device.type)) {
    ++MutableW4A16Stats().dense_gemms;
    DBuf outbf(d, DType::kBF16, {M, N});
    Tensor wqd = MakeTensor(mr.w, DType::kI32, d.q.device, {K / 16, N * 2});
    Tensor scd = MakeTensor(mr.s, DType::kI8, d.q.device, {K / w.group_size, N});
    Tensor ggd = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
    Tensor wstd = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
    vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)};
    dargs.group_size = static_cast<int>(w.group_size);
    dargs.mxfp4 = w.is_mxfp4;
    vt::MarlinDenseGemm(d.q, outbf.t(), x, wqd, scd, ggd, wstd, dargs);
    if (out_dtype == DType::kBF16) return outbf;
    DBuf out(d, DType::kF32, {M, N});
    vt::CastF32(d.q, out.t(), outbf.t());
    return out;
  }

  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  ++MutableW4A16Stats().marlin_gemms;

  // Marlin's output is bf16 (c_type=kBFloat16); an f32 result is the bf16 output
  // upcast (the same value it rounds to).
  DBuf outbf(d, DType::kBF16, {M, N});
  Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, N * 2});
  // Scale grid rows = K/group_size (K/16 nvfp4, K/32 mxfp4).
  Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / w.group_size, N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
  Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
  Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
  Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
  vt::MoeMarlinArgs margs{ac.block, 1, static_cast<int>(M), static_cast<int>(N),
                          static_cast<int>(K), false};
  margs.group_size = static_cast<int>(w.group_size);
  margs.mxfp4 = w.is_mxfp4;
  vt::MoeGroupedGemmNvfp4Marlin(d.q, outbf.t(), x, wq, sc, gg, wst, sorted, expert,
                                numpad, topkw, margs);
  if (out_dtype == DType::kBF16) return outbf;
  DBuf out(d, DType::kF32, {M, N});
  vt::CastF32(d.q, out.t(), outbf.t());
  return out;
}

// --- Fused gate_up (one Marlin GEMM over the N-concatenated pair) -----------
struct MarlinDensePairResident {
  void* w = nullptr;     // i32 [K/16, (2N)*2]
  void* s = nullptr;     // fp8 [K/16, 2N]
  void* g = nullptr;     // f32 [1]
  int64_t n = 0, k = 0;  // n = per-shard N; operand size_n = 2n
  bool ready = false;
};

inline MarlinDensePairResident& MarlinDensePairResidentFor(const Nvfp4Weight* gate) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDensePairResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[gate];
}

inline void BuildMarlinDensePairResident(Dev d, const Nvfp4Weight& gw,
                                         const Nvfp4Weight& uw,
                                         MarlinDensePairResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(gw.k);
  const int N = static_cast<int>(gw.n);
  const int gs = static_cast<int>(gw.group_size);  // 16 (nvfp4) or 32 (mxfp4)
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(2 * N) * 2);
  const size_t s_b = static_cast<size_t>(K / gs) * (2 * N);  // K/16 nvfp4, K/32 mxfp4
  const size_t pk_b = static_cast<size_t>(N) * (K / 2);   // one shard's packed bytes
  const size_t sc_b = static_cast<size_t>(N) * (K / gs);  // one shard's scale bytes
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = gw.n;
  mr.k = gw.k;
  Nvfp4Dev dg = ResidentNvfp4(d, gw);
  Nvfp4Dev du = ResidentNvfp4(d, uw);
  // Flat row-stack concat (packed [N,K/2] u8 / scales [N,K/gs] are row-major over
  // N; gate rows FIRST — vLLM's merged shard order, qwen3.py:271-274
  // `gate_up_proj: [gate_proj, up_proj]`).
  auto* tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
  auto* tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sc_b));
  d.b.Copy(d.q, tmp_w, dg.packed.data, pk_b);
  d.b.Copy(d.q, tmp_w + pk_b, du.packed.data, pk_b);
  d.b.Copy(d.q, tmp_s, dg.scale.data, sc_b);
  d.b.Copy(d.q, tmp_s + sc_b, du.scale.data, sc_b);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index,
                                     static_cast<uint32_t*>(mr.w), tmp_w, K, 2 * N);
  if (gw.is_mxfp4) {
    // MXFP4: E8M0 passthrough permute over the MERGED 2N scales (no combined
    // factor, no global — the kernel skips global for E8M0). Byte-identical PER
    // SHARD to the split-path single-expert resident, because the E8M0 permute is
    // row-local (each output column's scales depend only on its own group bytes),
    // so [gate;up] stacked == the two residents concatenated.
    vt::cuda::MarlinProcessExpertScalesMxfp4(stream, tmp_s,
                                             static_cast<uint8_t*>(mr.s), K, 2 * N);
    const float g = 1.0F;  // unused (kernel skips global for E8M0)
    d.b.Copy(d.q, mr.g, &g, sizeof(float));
  } else {
    // combined_scale_factor over BOTH shards (vLLM computes it over the MERGED
    // gate_up scale tensor — marlin_utils_fp4.py:281-284 operates on the whole
    // parameter, which for gate_up_proj is already the concatenation).
    std::vector<const uint8_t*> bufs{
        reinterpret_cast<const uint8_t*>(gw.scale.bytes.data()),
        reinterpret_cast<const uint8_t*>(uw.scale.bytes.data())};
    std::vector<size_t> lens{gw.scale.bytes.size(), uw.scale.bytes.size()};
    const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
    vt::cuda::MarlinProcessExpertScales(stream, tmp_s, static_cast<uint8_t*>(mr.s),
                                        K, 2 * N, sf);
    // ONE global scale for both shards (vLLM's merged parameter has exactly one
    // weight_global_scale — it takes `.max()` across the shards at
    // compressed_tensors_w4a4_nvfp4.py:111-114; equality is guarded by the caller).
    const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(gw.scale2, sf);
    d.b.Copy(d.q, mr.g, &g, sizeof(float));
  }
  d.b.Synchronize(d.q);  // repack done -> safe to free staging + fp4 originals
  d.b.Free(tmp_w);
  d.b.Free(tmp_s);
  gw.ReleaseResident();
  uw.ReleaseResident();
  mr.ready = true;
}

// True when a gate/up pair takes the fused Marlin gate_up path. Must be checked
// IDENTICALLY at every call site so exactly ONE resident layout is ever built.
inline bool GateUpFusedEligible(const Nvfp4Weight& gw, const Nvfp4Weight& uw) {
  // Both NVFP4 (group 16, combined scale + per-tensor global) or both MXFP4
  // (group 32, E8M0 passthrough, NO global). The fused merged gate_up resident
  // row-stacks the two shards ([gate;up] -> one [2N,K] operand). MXFP4 has NO
  // cross-shard scale interaction — each group's E8M0 byte is passed through
  // independently (MarlinProcessExpertScalesMxfp4), with no combined_scale_factor
  // and no global — so the fused MXFP4 GEMM is byte-identical to the two split
  // single-expert GEMMs (strictly SAFER than the NVFP4 case, which additionally
  // needs scale2 equality because its combined factor spans both shards). The
  // format/group must match (a MLP's gate and up always share both) and, for the
  // NVFP4 arm, scale2 must be equal (trivially true for MXFP4: both 0).
  //
  // Those shape/scale terms are GateUpPairFusableShape above — factored out, not
  // restated, so the dense MLP's guard cannot drift from this one.
  return FusedGateUpEnabled() && GateUpPairFusableShape(gw, uw);
}

// silu(x@gate.T) * (x@up.T) -> bf16 [M,N] via ONE fused Marlin gate_up GEMM.
inline DBuf GateUpFusedMarlinD(Dev d, const Tensor& x, const Nvfp4Weight& gw,
                               const Nvfp4Weight& uw) {
  const int64_t M = x.shape[0], K = x.shape[1], N = gw.n;
  MarlinDensePairResident& mr = MarlinDensePairResidentFor(&gw);
  if (!mr.ready) BuildMarlinDensePairResident(d, gw, uw, mr);
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);  // zeroed once; kernel self-resets

  // VT_MARLIN_DENSE (default OFF): fused gate_up over the 2N-concatenated operand
  // via vLLM's OWN dense marlin GEMM. Same merged resident (mr.w/mr.s/mr.g), rank-2
  // views, no moe_align; byte-preserving reduce. Only when the op is realized here.
  if (MarlinDenseEnabled() &&
      vt::OpRegistered(vt::OpId::kMarlinDenseGemm, d.q.device.type)) {
    ++MutableW4A16Stats().dense_gemms;
    DBuf gud(d, DType::kBF16, {M, 2 * N});
    Tensor wqd = MakeTensor(mr.w, DType::kI32, d.q.device, {K / 16, 2 * N * 2});
    Tensor scd = MakeTensor(mr.s, DType::kI8, d.q.device, {K / gw.group_size, 2 * N});
    Tensor ggd = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
    Tensor wstd = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
    vt::MarlinDenseArgs dargs{static_cast<int>(M), static_cast<int>(2 * N),
                              static_cast<int>(K)};
    dargs.group_size = static_cast<int>(gw.group_size);
    dargs.mxfp4 = gw.is_mxfp4;
    vt::MarlinDenseGemm(d.q, gud.t(), x, wqd, scd, ggd, wstd, dargs);
    DBuf actd(d, DType::kBF16, {M, N});
    vt::SiluAndMul(d.q, actd.t(), gud.t());
    return actd;
  }

  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  ++MutableW4A16Stats().fused_gate_up;

  DBuf gu(d, DType::kBF16, {M, 2 * N});
  // Weight is always K/16-tiled (marlin interleave is group-independent); the
  // SCALE grid rows are K/group_size (K/16 nvfp4, K/32 mxfp4).
  Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, 2 * N * 2});
  Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / gw.group_size, 2 * N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
  Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
  Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
  Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
  vt::MoeMarlinArgs margs{ac.block, 1, static_cast<int>(M), static_cast<int>(2 * N),
                          static_cast<int>(K), false};
  margs.group_size = static_cast<int>(gw.group_size);
  margs.mxfp4 = gw.is_mxfp4;
  vt::MoeGroupedGemmNvfp4Marlin(d.q, gu.t(), x, wq, sc, gg, wst, sorted, expert,
                                numpad, topkw, margs);
  DBuf act(d, DType::kBF16, {M, N});
  vt::SiluAndMul(d.q, act.t(), gu.t());
  return act;
}

#endif  // VT_MARLIN_NVFP4

// --- The W4A16 dispatcher ---------------------------------------------------

// ── A2-Q2b (#810): the dispatcher's OWN gate, as ONE predicate ───────────────
//
// `MatmulNvfp4W4A16D` below selects Marlin on THREE clauses. A caller that has
// to know the answer BEFORE it calls — because the fallback is not merely
// slower for it but pathological — must ask this function rather than restate
// the clauses, and this exists because a restatement is exactly what went
// wrong. A2-Q2b's first submission restated two of the three in NemotronH's
// `DeviceLmHeadEligible` and dropped `MarlinW4A16Enabled()`, so under an
// explicit `VT_NVFP4_MARLIN=0` the model said "eligible", the dispatcher fell
// to the naive redundant-dequant arm, and because the caller hands in a
// TRANSIENT `Nvfp4Weight` view the `ResidentNvfp4` cache below could never hit
// — re-uploading the whole `[vocab, hidden]` operand on every decode step.
//
// A token gate cannot see any of that: the fallback computes the SAME value.
// One expression with two call sites is what makes the two unable to drift.
inline bool MarlinW4A16Selects(Dev d, DType act_dtype) {
  // NO `#ifdef VT_MARLIN_NVFP4` HERE, and the absence is the design. This site
  // only SELECTS a path, and every term it reads exists in every build:
  // `MarlinW4A16Enabled()` is declared above the guarded region, and
  // `vt::OpRegistered` IS the op/provider table's own answer to "is the Marlin
  // arm realized for this device".
  //
  // The build flag and the registration are the SAME condition, so the guard
  // would decide nothing the query does not already decide. `CMakeLists.txt`'s
  // one `if(VLLM_CPP_MARLIN)` block adds `src/vt/cuda/cuda_moe_marlin.cu` —
  // whose file-scope `Registrar` holds the tree's only
  // `RegisterOp(OpId::kMoeGroupedGemmNvfp4Marlin, …)` — and defines
  // `VT_MARLIN_NVFP4=1`, in that same block. A build without the macro
  // therefore registers nothing, and this resolves false on exactly the builds
  // a `#ifdef` would have excluded. Asking the table rather than the
  // preprocessor is what `scripts/check-device-leakage.py` asks of a selection,
  // and it is the call `nemotron_h_device.cpp`'s `moe_on_device` already made.
  //
  // The device clause is likewise an OP-AVAILABILITY question, not a
  // `== kCUDA` one (registered only for kCUDA today, so this is byte-identical
  // on the production build — accelerator-seam audit class A, work row S4).
  return vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) &&
         MarlinW4A16Enabled() && act_dtype == DType::kBF16;
}

// y[M,N] = x[M,K] @ dequant(w).T for an NVFP4 W4A16 weight. Mirrors vLLM's
// forced-Marlin selection for `use_a16` (__init__.py:879-881): on CUDA with a
// BF16 activation take Marlin; otherwise fall back to the naive
// redundant-dequant vt::MatmulNvfp4 (CUDA) or a host dequant + bf16 Matmul (CPU
// reference). `w` MUST be W4A16 (alpha == 0) — a true-W4A4 weight belongs to
// qwen3_5.cpp's private fp4-activation path and is rejected here.
inline DBuf MatmulNvfp4W4A16D(Dev d, const Tensor& x, const Nvfp4Weight& w,
                              DType out_dtype, MarlinDenseResident* resident = nullptr) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(!w.IsTrueW4A4(),
           "dense_nvfp4: true-W4A4 weight routed into the W4A16 dispatcher");
#ifdef VT_MARLIN_NVFP4
  // Marlin requires a bf16 activation (vLLM's a16 path is bf16/fp16 too). The
  // three clauses live in `MarlinW4A16Selects` above so a caller can ask the
  // SAME question in advance instead of restating them (A2-Q2b).
  if (MarlinW4A16Selects(d, x.dtype))
    return MatmulNvfp4MarlinD(d, x, w, out_dtype, resident);
#else
  (void)resident;
#endif
  ++MutableW4A16Stats().fallback_gemms;
  DBuf dout(d, out_dtype, {M, N});
  // MXFP4 has no naive redundant-dequant device op (that op is NVFP4-only); the
  // fallback is always the host dequant + bf16 Matmul reference.
  if (w.is_mxfp4) {
    std::vector<uint16_t> wb = DequantMxfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), x, dwb.t());
    return dout;
  }
  // Same class-A conversion: the naive redundant-dequant NVFP4 GEMM exists only
  // where the op table realizes it (kCUDA); elsewhere fall to the host dequant +
  // bf16 Matmul reference. Byte-identical to the old `device == kCUDA` test.
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), x, dwb.t());
  }
  return dout;
}

// Named MXFP4 W4A16 entry point (mirrors MatmulNvfp4W4A16D). y[M,N] = x[M,K] @
// dequant_mxfp4(w).T for a compressed-tensors MXFP4 weight-only weight (E8M0,
// group 32, no global). Routes through the SAME shared Marlin/CPU dispatcher via
// w.is_mxfp4; the assert documents the contract.
inline DBuf MatmulMxfp4W4A16D(Dev d, const Tensor& x, const Nvfp4Weight& w,
                              DType out_dtype) {
  VT_CHECK(w.is_mxfp4, "dense_mxfp4: non-MXFP4 weight routed into MatmulMxfp4W4A16D");
  return MatmulNvfp4W4A16D(d, x, w, out_dtype);
}

}  // namespace dense_nvfp4
}  // namespace vllm
