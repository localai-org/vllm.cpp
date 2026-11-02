// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// cuBLASLt matmul for the CUDA backend (M0.6, correctness-grade).
// Supported dtype combos: (bf16,bf16)->f32, (bf16,bf16)->bf16, (f32,f32)->f32.
// `matmul_bt` additionally serves (f32 activation, bf16 weight) -- see the
// block comment above MixedF32ActBf16WeightBT for why that pair is not a
// widened activation but the shape the CPU arm and this device's own
// kMatmulBTQuant already answer.
// Any other combo the public-op validation admits (e.g. f16 inputs) throws
// here, naming the combo — never silently truncates. Compute type is
// CUBLAS_COMPUTE_32F with f32 scale; all layouts are CUBLASLT_ORDER_ROW so no
// host-side transposes are needed.
// Upstream counterpart: torch.matmul/cublas path (no csrc kernel); cuBLASLt is our native equivalent.
//
// Also hosts MatmulFp8CublasLt (see the block comment near the bottom): the
// cuBLASLt FP8 (e4m3) dense GEMM. It is a vt-runtime ORIGINAL, not a mirror —
// see the correction beside its definition below; vLLM runs no cuBLASLt for
// this GEMM at the pin. It reuses this same handle + workspace. Its
// matmul descriptor + 3 layouts + heuristic algo can be cached per device on the
// full shape/config key (fp8_plan_cache.h), mirroring vLLM's in-graph plan reuse
// so the per-call heuristic + descriptor/layout rebuild is paid once per shape.
// DEFAULT ON / VT_FP8_PLAN_CACHE=0 rollback (#1843): on CUDA 13.3 the heuristic
// FAILS inside CUDA-graph capture, and the uncached lane queried it per call, so
// the cached (eagerly warmed before capture) plan is the only safe default —
// correctness decides the polarity, not the wall-clock (measured NEUTRAL on
// GB10). Bit-identical either way: the cuBLASLt algo selection is
// process-deterministic per shape.
//
// Env-gated diagnostic: VT_GEMM_ALGO_LOG=1 emits one std::cerr line per unique
// (shape, dtype-combo, epilogue) cuBLASLt algo selection (see MaybeLogGemmAlgo).
// Default OFF, zero hot-path cost when unset. It exists to compare arm-wise algo
// latching on the packed GDN BF16-BA GEMM per the 2026-07-15 forensic record;
// the portable flag/uniqueness plumbing lives in gemm_algo_log.h (CPU-tested).
//
// The three bf16/f32 lanes below route their cublasLtMatmulAlgoGetHeuristic call
// through GetOrQueryGemmHeuristic, a per-full-call-key heuristic-result cache
// (gemm_plan_cache.h, DEFAULT ON): on CUDA 13.3 the query itself fails inside
// CUDA graph capture (issue #1732), and the eager warm step before every capture
// populates the cache so the in-capture call is a pure map hit.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cuda/cublas_lt_hgemm.h"
#include "cuda_device_caps.h"
#include "cuda_legacy_matmul.cuh"
#include "vt/cuda/fp8_plan_cache.h"
#include "vt/cuda/gemm_algo_log.h"
#include "vt/cuda/gemm_plan_cache.h"
#include "vt/ops.h"
#include "vt/cuda/cublas_lt_hgemm.h"
#include "vt/op_provider.h"

namespace vt::cuda {
namespace {

constexpr size_t kWorkspaceBytes = 32ull << 20;  // 32 MB, per the M0.6 plan

// requestedAlgoCount for EVERY cuBLASLt heuristic query below: 1 == take the single
// best heuristic, no algo search. Named (not a bare literal at each call site) so any
// future algo-policy change is one conscious, greppable edit — the invocation-parity
// bug class this guards is that the caller's choices (output dtype, entry point, algo
// policy) SELECT the resolved kernel template (a bf16 M=1 GEMV with an f32-out layout
// buys the slower gemvx<bf16,FLOAT> template than vLLM's gemvx<bf16,bf16>). Enforced by
// scripts/check-gemv-invocation-consistency.py.
constexpr int kGemvHeuristicAlgos = 1;

// requestedAlgoCount for the DIAGNOSTIC-ONLY bf16/f32 candidate dump, which runs
// on its own heuristic query and only when VT_GEMM_ALGO_LOG=1. It does NOT
// change what executes: the single-best query above still selects the kernel.
//
// WHY IT EXISTS. An nsys profile of the 27B decode (2026-08-30, GB10 sm_121a)
// puts 87.4% of GPU time in two variants of ONE kernel --
// `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8` at
// 73.2% over 11,679 calls and its `128x2` sibling at 14.2% -- an AMPERE-
// generation WMMA kernel with a 16x16 tile, on a Blackwell part, for a decode
// whose M is 8. "cuBLASLt ranked a Blackwell-native algo second" and "cuBLASLt
// enumerates no sm_120/121 bf16 algo for this descriptor at all" are different
// findings with different fixes, and a list of length one cannot tell them
// apart. This is the same question #1866 asked of the fp8 tower, which resolves
// to `sm89_xmma ... tilesize32x64x64` on the same part (#1857).
constexpr int kDenseAlgoLogCandidates = 8;

// requestedAlgoCount for the DIAGNOSTIC-ONLY fp8 candidate dump below, which
// runs on its own heuristic query and only when VT_GEMM_ALGO_LOG=1. It exists
// because #1866 asks a question the single-best query cannot answer: our fp8
// tower resolves to `sm89_xmma ... tilesize32x64x64` on sm_121a (#1857), and
// "cuBLASLt ranked an nvjet_sm121 algo second" and "cuBLASLt enumerates no
// sm121 fp8 algo for this descriptor at all" are different findings with
// different fixes, indistinguishable from a list of length one. Deliberately
// SEPARATE from the production query so the shipped call site keeps
// kGemvHeuristicAlgos byte-for-byte and the selected algo cannot change: a
// diagnostic that perturbs what it observes is not a diagnostic.
constexpr int kFp8AlgoLogCandidates = 8;

void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// Local status-name table: cublasGetStatusName lives in libcublas, which we
// do not link (only libcublasLt).
const char* StatusName(cublasStatus_t st) {
  switch (st) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_<unknown>";
  }
}

void CheckLt(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("vt cuda: matmul: ") + what + ": cublas status " +
                             std::to_string(static_cast<int>(st)) + " (" + StatusName(st) +
                             ")");
  }
}

// cublasLt handle + 32 MB device workspace, cached per device index. Created
// lazily under a mutex on first matmul — never at static-init time (the
// pre-main discipline forbids CUDA API calls during registration). Both live
// for the whole process and are deliberately never destroyed: freeing at exit
// would race CUDA driver teardown, and one handle + workspace per device is a
// bounded leak (documented in the M0.6 plan). Note the current backend
// registers device 0 only and never switches devices; creation happens on the
// caller's current device, so true multi-device use would additionally need
// cudaSetDevice here.
struct LtContext {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
};

// Defined beside the fp8 candidate dump it mirrors; declared here because the BT
// lane below calls it and `LtContext` has to be complete first.
void MaybeLogDenseAlgoCandidates(const LtContext& ctx, int64_t m, int64_t n, int64_t k,
                                 cudaDataType_t ab_type, cudaDataType_t out_type,
                                 const char* lane, cublasLtMatmulDesc_t desc,
                                 cublasLtMatrixLayout_t la, cublasLtMatrixLayout_t lb,
                                 cublasLtMatrixLayout_t lc, cublasLtMatmulPreference_t pref);

LtContext GetContext(int device) {
  static std::mutex mu;
  static std::unordered_map<int, LtContext> ctxs;  // values leak by design (see above)
  std::lock_guard<std::mutex> lock(mu);
  auto it = ctxs.find(device);
  if (it != ctxs.end()) return it->second;
  LtContext ctx;
  CheckLt(cublasLtCreate(&ctx.handle), "cublasLtCreate");
  cudaError_t err = cudaMalloc(&ctx.workspace, kWorkspaceBytes);
  if (err != cudaSuccess) {
    cublasLtDestroy(ctx.handle);  // creation failed part-way: don't cache a half-made context
    CheckCuda(err, "workspace cudaMalloc");
  }
  ctxs.emplace(device, ctx);
  return ctx;
}

// RAII for the per-call cublasLt descriptor objects (throw-safe cleanup).
struct LayoutGuard {
  cublasLtMatrixLayout_t v = nullptr;
  ~LayoutGuard() {
    if (v != nullptr) cublasLtMatrixLayoutDestroy(v);
  }
};
struct DescGuard {
  cublasLtMatmulDesc_t v = nullptr;
  ~DescGuard() {
    if (v != nullptr) cublasLtMatmulDescDestroy(v);
  }
};
struct PrefGuard {
  cublasLtMatmulPreference_t v = nullptr;
  ~PrefGuard() {
    if (v != nullptr) cublasLtMatmulPreferenceDestroy(v);
  }
};

// Overload with an EXPLICIT leading dimension, for strided views whose row
// stride exceeds `cols` (e.g. a column slice of a wider output tensor).
// Mirrors MakeRowMajorBatched's ld parameter (line 202). For a contiguous
// tensor, ld == cols and this is identical to the 4-arg overload below.
void MakeRowMajor(LayoutGuard& l, cudaDataType_t t, int64_t rows, int64_t cols,
                 int64_t ld) {
  CheckLt(cublasLtMatrixLayoutCreate(&l.v, t, static_cast<uint64_t>(rows),
                                     static_cast<uint64_t>(cols), ld),
          "cublasLtMatrixLayoutCreate");
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "set CUBLASLT_MATRIX_LAYOUT_ORDER");
}

void MakeRowMajor(LayoutGuard& l, cudaDataType_t t, int64_t rows, int64_t cols) {
  MakeRowMajor(l, t, rows, cols, cols);
}

// Row-major layout with an EXPLICIT leading dimension (the tensor's row stride,
// which may exceed `cols` for a strided view) plus strided-batch metadata. The
// batch stride is in ELEMENTS, matching vt::Tensor::stride.
void MakeRowMajorBatched(LayoutGuard& l, cudaDataType_t t, int64_t rows, int64_t cols,
                         int64_t ld, int32_t batch, int64_t batch_stride) {
  CheckLt(cublasLtMatrixLayoutCreate(&l.v, t, static_cast<uint64_t>(rows),
                                     static_cast<uint64_t>(cols), ld),
          "cublasLtMatrixLayoutCreate (batched)");
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "set CUBLASLT_MATRIX_LAYOUT_ORDER (batched)");
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                           sizeof(batch)),
          "set CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT");
  CheckLt(cublasLtMatrixLayoutSetAttribute(
              l.v, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
              sizeof(batch_stride)),
          "set CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET");
}

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
}

// ---- Env-gated cuBLASLt algo-selection diagnostic (VT_GEMM_ALGO_LOG) --------
// Short tag for a cuBLASLt operand data type (the BF16-vs-F32 output type is the
// packed-arm variable of interest).
const char* CudaTypeTag(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F: return "f32";
    case CUDA_R_16BF: return "bf16";
    case CUDA_R_16F: return "f16";
    case CUDA_R_8F_E4M3: return "e4m3";
    default: return "other";
  }
}

// When VT_GEMM_ALGO_LOG=1, emit ONE std::cerr line per unique (shape,
// dtype-combo, epilogue) selection naming the cuBLASLt-chosen algo config
// (id/tile/stages/splitK) and its heuristic workspace. OUR diagnostic — upstream
// logs the same selection under CUBLASLT_LOG_LEVEL / torch `_scaled_mm` verbose;
// we have no torch, so we mirror it under our own flag to compare arm-wise algo
// LATCHING on the packed GDN BF16-BA GEMM vs the F32-BA/decomposed arm, per the
// 2026-07-15 forensic record (a constant ~0.2% packed steady per-token tax whose
// one un-instrumented per-process variable is the BF16 GEMM algo selection; see
// .agents/state.md and .agents/specs/gdn-packed-decode.md). Zero cost when the
// flag is unset: GemmAlgoLogEnabled() is a cached bool and the body is skipped.
// The reads are best-effort (a driver that does not expose an attribute leaves
// its sentinel) and never throw — this is diagnostics, not a correctness path.
void MaybeLogGemmAlgo(const cublasLtMatmulHeuristicResult_t& heur, int64_t m, int64_t n,
                      int64_t k, cudaDataType_t a_t, cudaDataType_t b_t, cudaDataType_t c_t,
                      const char* epilogue) {
  if (!GemmAlgoLogEnabled()) return;  // cached bool; default OFF pays nothing here
  int32_t algo_id = -1, split_k = -1;
  uint32_t tile = 0, stages = 0;
  size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id,
                                       sizeof(algo_id), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                       sizeof(tile), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                                       sizeof(stages), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k,
                                       sizeof(split_k), &written);
  // One line per unique key across every cuBLASLt GEMM site (the key embeds the
  // backend/epilogue tag, so the three paths never collide).
  static LogOncePerKey once;
  std::string key = std::string("cublasLt|m=") + std::to_string(m) + " n=" + std::to_string(n) +
                    " k=" + std::to_string(k) + "|a=" + CudaTypeTag(a_t) + " b=" +
                    CudaTypeTag(b_t) + " c=" + CudaTypeTag(c_t) + "|" + epilogue;
  if (!once.ShouldLog(key)) return;
  std::cerr << "[VT_GEMM_ALGO] backend=cublasLt m=" << m << " n=" << n << " k=" << k
            << " a=" << CudaTypeTag(a_t) << " b=" << CudaTypeTag(b_t) << " c=" << CudaTypeTag(c_t)
            << " epilogue=" << epilogue << " algoId=" << algo_id << " tile=" << tile
            << " stages=" << stages << " splitK=" << split_k << " wsSize=" << heur.workspaceSize
            << std::endl;
}

// ---- Heuristic-result cache for the bf16/f32 lanes (issue #1732) ------------
// cublasLtMatmulAlgoGetHeuristic returns CUBLAS_STATUS_INTERNAL_ERROR (status
// 14; cuBLASLt's own trace: "Could not obtain green context information") when
// it runs while a CREATED stream is in CUDA graph capture, on CUDA 13.3 +
// sm_80. The three lanes below used to query it inline on every call, so the
// default (graphs-on) decode died at the first in-capture GEMM. The engine's
// eager warm step runs every decode slot's padded shapes eagerly BEFORE
// capture (s.warm, qwen3_5.cpp ~GraphCaptureScope), so caching the result per
// full call key makes the in-capture call a map hit: cuBLASLt is never queried
// under capture. Upstream vLLM is immune the same way — torch caches the
// selected cuBLASLt algo per shape and vLLM runs eager warmup before capture.
//
// The cached value is the cublasLtMatmulHeuristicResult_t ALONE (spec design):
// descriptor, layout and preference creation are host-side and legal under
// capture, and cublasLtMatmul still needs the desc/layouts, so the call sites
// keep building them exactly as before — every CheckLt name below the call
// sites is byte-identical to the pre-cache code. The heuristic query is the
// only step that reads green-context state and fails, so it is the only step
// the cache removes. Bit-exact: cuBLASLt selection is process-deterministic
// per shape, so the cached result is what a fresh query would return.
//
// Takes the call site's already-built desc/layouts/pref plus its existing
// CheckLt `what` so each lane's diagnostic surface does not move. `fresh`
// (may be null) reports whether THIS call ran a real query, so the call site
// fires MaybeLogGemmAlgo once per actual query only — a cache hit neither
// queries nor logs. Returns false iff the query returned zero algos; the call
// site then throws its pre-existing "no cublasLt heuristic" text unchanged.
bool GetOrQueryGemmHeuristic(const LtContext& ctx, const GemmPlanKey& key,
                             cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t la,
                             cublasLtMatrixLayout_t lb, cublasLtMatrixLayout_t lc,
                             cublasLtMatmulPreference_t pref, const char* what,
                             cublasLtMatmulHeuristicResult_t* out, bool* fresh) {
  static std::mutex mu;
  // Values leak by design, mirroring GetOrBuildCachedFp8Plan / GetContext: the
  // entries are plain structs (no handles to destroy), bounded by the finite
  // set of GEMM shapes a model runs, and freeing at exit would buy nothing.
  static std::unordered_map<GemmPlanKey, cublasLtMatmulHeuristicResult_t, GemmPlanKeyHash>
      heurs;
  if (GemmPlanCacheEnabled()) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = heurs.find(key);
    if (it != heurs.end()) {
      *out = it->second;  // cache hit: no cuBLASLt query, so none under capture
      if (fresh != nullptr) *fresh = false;
      return true;
    }
  }
  // Miss, or the VT_GEMM_PLAN_CACHE=0 rollback arm (escape hatch / A/B
  // measurement, not a supported configuration): today's exact query, run
  // OUTSIDE the lock so the disabled flag is byte-for-byte today's behavior
  // with no new serialization. A concurrent same-key miss may query twice;
  // selection is process-deterministic, so both results are equal and the
  // emplace below is idempotent. The lock is never held across a query or the
  // matmul itself.
  cublasLtMatmulHeuristicResult_t h{};
  int returned = 0;
  CheckLt(cublasLtMatmulAlgoGetHeuristic(ctx.handle, desc, la, lb, lc, lc, pref,
                                         /*requestedAlgoCount=*/kGemvHeuristicAlgos, &h,
                                         &returned),
          what);
  if (fresh != nullptr) *fresh = true;
  if (returned == 0) return false;  // caller keeps its existing throw text
  if (GemmPlanCacheEnabled()) {
    std::lock_guard<std::mutex> lock(mu);
    heurs.emplace(key, h);  // process-lifetime; values leak by design
  }
  *out = h;
  return true;
}

void MatmulKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16 &&
                       (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  const bool f32_all =
      a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32;
  const bool f16_in = a.dtype == DType::kF16 && b.dtype == DType::kF16 &&
                      (out.dtype == DType::kF16 || out.dtype == DType::kF32);
  if (!bf16_in && !f32_all && !f16_in) {
    throw std::runtime_error("vt cuda: matmul: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (f16,f16)->f16|f32, (bf16,bf16)->f32|bf16, (f32,f32)->f32");
  }
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {  // empty reduction: out = 0 (f32 and bf16 zero are all-zero bytes)
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "k=0 memset");
    return;
  }

  if (f16_in) {
    CublasLtHgemm(q, out, a, b);
    return;
  }
  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = a.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
  LayoutGuard la, lb, lc;
  MakeRowMajor(la, ab_type, m, k);
  MakeRowMajor(lb, ab_type, k, n);
  MakeRowMajor(lc, out_type, m, n);

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  // Key: every value that reached the layouts + descriptor above (#1732).
  // lda/ldb/ldc are the `cols` MakeRowMajor passed (its ld); the dense lane
  // sets no batch attributes, so batch/strides stay 0.
  GemmPlanKey key;
  key.device = q.device.index;
  key.op = kNn;
  key.m = m;
  key.n = n;
  key.k = k;
  key.lda = k;  // MakeRowMajor(la, ab_type, m, k): ld = cols = k
  key.ldb = n;  // MakeRowMajor(lb, ab_type, k, n): ld = n
  key.ldc = n;  // MakeRowMajor(lc, out_type, m, n): ld = n
  key.ab_type = static_cast<int>(ab_type);
  key.out_type = static_cast<int>(out_type);

  cublasLtMatmulHeuristicResult_t heur{};
  bool fresh = false;
  if (!GetOrQueryGemmHeuristic(ctx, key, desc.v, la.v, lb.v, lc.v, pref.v,
                               "cublasLtMatmulAlgoGetHeuristic", &heur, &fresh)) {
    throw std::runtime_error("vt cuda: matmul: no cublasLt heuristic for [" +
                             std::to_string(m) + "," + std::to_string(k) + "]x[" +
                             std::to_string(k) + "," + std::to_string(n) + "] " +
                             ComboName(a, b, out));
  }
  if (fresh) MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "rowmajor-NN");

  // out = 1.0 * a @ b + 0.0 * out; C and D share the same buffer and layout.
  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, a.data, la.v, b.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "cublasLtMatmul");
}

// ---- (f32 activation, bf16 weight) for the BT lane -------------------------
//
// WHY THIS PAIR EXISTS AT ALL, AND WHY THE ANSWER IS NOT "NARROW THE ACTIVATION".
// `vt::MatmulBT`'s contract is `IsFloat(a) && IsFloat(b)`, and THREE of the four
// implementations behind it already answer an f32 activation against a typed
// weight: the CPU elementwise kernel (`cpu_ops.cpp` MatmulChunked<true>, which
// reads both operands through a dtype-generic getter), the CPU block-quant
// kernel, and THIS DEVICE'S OWN `kMatmulBTQuant` (`vt::MatmulBTQuant` admits any
// `IsFloat(a.dtype)`). The cuBLASLt elementwise lane was the only one that did
// not, and `cuda_deepseek_v4.cu`'s `RouterGateKernel` says so in its own words --
// it hand-wrote a GEMV because "the CUDA elementwise MatmulBT lacks" this pair.
// A seam gap that makes callers write private kernels beside it is the shape
// AGENTS.md "Shared seams" names.
//
// THE ACTIVATION IS NOT AN ACCIDENTAL WIDENING, AND THAT WAS CHECKED RATHER THAN
// ASSUMED. The production caller is `Qwen4ExpGatedResidualKernelCuda`
// (`cuda_qwen4_exp.cu:414,418,428`), whose three projections read `t_normed`,
// the grouped-RMSNorm scratch. Upstream narrows there --
// `Qwen4ExpTextRMSNorm.forward` ends in `output.type_as(x)`
// (transformers 5.16.0, `models/qwen4_exp/modeling_qwen4_exp.py:173-178`, this
// row's lane oracle) -- and this tree deliberately does NOT: widen on load,
// compute in f32, round once on the store, "which is this tree's house contract
// and what `vt::RmsNorm` says of itself in the same terms". That divergence is
// RATIFIED and GATED, not incidental: it is recorded under `## Owed` in
// `.agents/specs/qwen4-exp-flash-next.md`, and reproducing upstream's rounding is
// mutation **M13**, which is RED in both the host and the device suite (1 of 9
// cases, 8 assertions). Narrowing the activation here would BE that mutation.
// On the RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` the same three calls carry
// Q8_0 mix weights and already run on this device through `kMatmulBTQuant`; the
// float weight is the dequantized/synthetic arm of the identical call.
//
// HOW, AND WHAT IT COSTS. cuBLASLt takes ONE data type for A and B, so a mixed
// pair has to be made uniform. Rounding the activation to bf16 is M13, so the
// WEIGHT is upcast instead: bf16 -> f32 is EXACT (`bits << 16`, the same upcast
// `RouterGateKernel` and the CPU `LoadF32` perform), so the f32 lane below then
// computes the CPU arm's values, with only the GEMM's own K-reassociation
// between them -- the divergence this file already owns for every other lane.
// The cost is honest and is stated here rather than discovered later: one
// [N,K] f32 scratch per call, so the weight is touched at 10 bytes/element
// instead of 2. Today's only caller projects the hyper-connection mix weights,
// which are small; a large-weight caller wants a native mixed kernel instead,
// and that is a performance row, not a correctness one.
//
// `cudaMallocAsync`/`cudaFreeAsync` rather than `cudaMalloc`/`cudaFree`: the
// latter pair is ILLEGAL during CUDA-graph capture and implicitly synchronizes
// the device, and this is a shared GEMM seam that capture paths reach.
__global__ void Bf16ToF32UpcastKernel(float* __restrict__ dst,
                                      const uint16_t* __restrict__ src, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    dst[i] = __uint_as_float(static_cast<uint32_t>(src[i]) << 16);
  }
}

// ---- cuBLASLt bf16/f32 "BT" dense GEMM (b = Linear weight [N,K]) -----------
// out[M,N] = a[M,K] @ b^T with b [N,K] row-major — K contiguous in BOTH
// operands, the TN layout vLLM's F.linear hits for its bf16 projections. On
// GB10 cuBLASLt serves this with the fast `nvjet_sm121_tst_..._TNNN` kernels
// (measured 27B GDN in_proj: 1.80 us/tok vs 2.29 us/tok for our row-major x
// row-major kMatmul, which falls to `NNNN` nvjet / sm80-cutlass kernels).
// Same column-major TN formulation as MatmulFp8CublasLtKernelCuda below (see
// its derivation comment); only the A/B dtypes differ (bf16/f32, alpha=1).
void MatmulBTKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32_in = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  const bool f32_act_bf16_w = a.dtype == DType::kF32 && b.dtype == DType::kBF16;
  const bool f16_in = a.dtype == DType::kF16 && b.dtype == DType::kF16 &&
                      (out.dtype == DType::kF16 || out.dtype == DType::kF32);
  if ((!bf16_in && !f32_in && !f32_act_bf16_w && !f16_in) ||
      (out.dtype == DType::kF16 && !f16_in)) {
    throw std::runtime_error(
        "vt cuda: matmul_bt: unsupported dtype combo " + ComboName(a, b, out) +
        "; supported: (f16,f16)->f16|f32, (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16, (f32,bf16)->f32|bf16");
  }
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "bt k=0 memset");
    return;
  }

  // The mixed pair, made uniform on the WEIGHT and re-entered on the f32 lane.
  // See the block comment above Bf16ToF32UpcastKernel: the activation is the
  // ratified f32 scratch of the hyper-connection mixer and rounding it here
  // would be that row's red mutation M13.
  if (f32_act_bf16_w) {
    const int64_t elems = b.Numel();
    float* w_f32 = nullptr;
    CheckCuda(cudaMallocAsync(&w_f32, static_cast<size_t>(elems) * sizeof(float), s),
              "bt mixed (f32,bf16): weight upcast alloc");
    constexpr unsigned kUpcastBlock = 256;
    int64_t blocks = (elems + kUpcastBlock - 1) / kUpcastBlock;
    if (blocks > 65535) blocks = 65535;
    Bf16ToF32UpcastKernel<<<static_cast<unsigned>(blocks), kUpcastBlock, 0, s>>>(
        w_f32, static_cast<const uint16_t*>(b.data), elems);
    CheckCuda(cudaGetLastError(), "bt mixed (f32,bf16): weight upcast launch");
    Tensor w = Tensor::Contiguous(w_f32, DType::kF32, b.device, {n, k});
    MatmulBTKernelCuda(q, out, a, w);
    CheckCuda(cudaFreeAsync(w_f32, s), "bt mixed (f32,bf16): weight upcast free");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = f32_in ? CUDA_R_32F : (f16_in ? CUDA_R_16F : CUDA_R_16BF);
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : (f16_in ? CUDA_R_16F : CUDA_R_16BF);

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "bt cublasLtMatmulDescCreate");
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)),
          "bt set TRANSA=T");
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
          "bt set TRANSB=N");

  // Column-major TN layouts: A=weight col[K,N] ld=K (TRANSA=T => [N,K] row),
  // B=act col[K,M] ld=K (= [M,K] row), C=D=out col[N,M] ld=N (= [M,N] row).
  LayoutGuard la, lb, lc;
  CheckLt(cublasLtMatrixLayoutCreate(&la.v, ab_type, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(n), k),
          "bt Adesc (weight)");
  // ld = the activation's ROW stride (== k for a contiguous activation, so
  // contiguous callers hand cuBLASLt byte-identical layouts and get the same
  // algo). A wider stride is the MLA chunked-prefill column slice, W6.
  CheckLt(cublasLtMatrixLayoutCreate(&lb.v, ab_type, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(m), a.stride[0]),
          "bt Bdesc (act)");
  CheckLt(cublasLtMatrixLayoutCreate(&lc.v, out_type, static_cast<uint64_t>(n),
                                     static_cast<uint64_t>(m), n),
          "bt Cdesc (out)");

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "bt cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "bt set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  // Key: every value that reached the layouts + descriptor above (#1732). This
  // TN lane is where the capture failure was observed; ldb is the activation's
  // row stride, which a chunked-prefill column slice can make wider than k, so
  // it cannot be derived from k. The transpose config is fixed by op=kTn.
  GemmPlanKey key;
  key.device = q.device.index;
  key.op = kTn;
  key.m = m;
  key.n = n;
  key.k = k;
  key.lda = k;           // A=[k,n] col-major, ld = k
  key.ldb = a.stride[0];  // B=[k,m] col-major, ld = the activation row stride
  key.ldc = n;           // C=D=[n,m] col-major, ld = n
  key.ab_type = static_cast<int>(ab_type);
  key.out_type = static_cast<int>(out_type);

  cublasLtMatmulHeuristicResult_t heur{};
  bool fresh = false;
  if (!GetOrQueryGemmHeuristic(ctx, key, desc.v, la.v, lb.v, lc.v, pref.v,
                               "bt cublasLtMatmulAlgoGetHeuristic", &heur, &fresh)) {
    throw std::runtime_error("vt cuda: matmul_bt: no cublasLt heuristic for [" +
                             std::to_string(m) + "," + std::to_string(k) + "]x[" +
                             std::to_string(n) + "," + std::to_string(k) + "]^T " +
                             ComboName(a, b, out));
  }
  // The 27B GDN in_proj_ba runs through this TN path; its BF16-vs-F32 output type
  // is the algo-latching variable the forensic record flagged.
  if (fresh) MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "TN-bt");
  if (fresh)
    MaybeLogDenseAlgoCandidates(ctx, m, n, k, ab_type, out_type, "TN-bt", desc.v, la.v, lb.v,
                                lc.v, pref.v);

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, b.data, la.v, a.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "bt cublasLtMatmul");
}

// ---- cuBLASLt strided-batched bf16/f32 GEMM (vt::BatchedMatmul) ------------
// out[G,M,N] = a[G,M,K] @ b[G,K,N] — the 1:1 counterpart of `torch.bmm`, the
// primitive MLA weight absorption is expressed in upstream
// (mla_attention.py:789 folds W_UK into the decode query, :1034 un-projects the
// latent output with W_UV). torch.bmm on CUDA bf16 resolves to cuBLAS
// `gemmStridedBatchedEx` (CUDA_R_16BF, CUBLAS_COMPUTE_32F); this is the
// cuBLASLt strided-batched form of exactly that GEMM, reusing the same handle
// and 32 MB workspace as the dense paths above.
//
// Row-major NN layouts with an EXPLICIT leading dimension, so a transposed view
// (both upstream call sites pass one) is consumed with no copy: the batch axis
// need not be the outermost storage axis, only the innermost dim must be
// unit-stride (enforced in ops.cpp).
void BatchedMatmulKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16 &&
                       (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  const bool f32_in = a.dtype == DType::kF32 && b.dtype == DType::kF32 &&
                      (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  if (!bf16_in && !f32_in) {
    throw std::runtime_error("vt cuda: batched_matmul: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16");
  }
  const int64_t g = out.shape[0], m = out.shape[1], n = out.shape[2], k = a.shape[2];
  if (g == 0 || m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {
    // Empty contraction: out = 0. Only safe to memset wholesale when `out` owns
    // a dense span; a strided view is zeroed row by row.
    if (out.IsContiguous()) {
      CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "batched k=0 memset");
    } else {
      const size_t esz = SizeOf(out.dtype);
      for (int64_t bi = 0; bi < g; ++bi) {
        for (int64_t i = 0; i < m; ++i) {
          char* row = static_cast<char*>(out.data) +
                      static_cast<size_t>(bi * out.stride[0] + i * out.stride[1]) * esz;
          CheckCuda(cudaMemsetAsync(row, 0, static_cast<size_t>(n) * esz, s),
                    "batched k=0 row memset");
        }
      }
    }
    return;
  }

  const auto& caps = GetDeviceCaps(q.device.index);
  if (caps.valid && caps.sm_major < 8 && f32_in && out.dtype == DType::kBF16) {
    CheckCuda(legacy::LaunchBatchedF32Bf16(q, out, a, b), "legacy batched f32 to bf16");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = a.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "batched cublasLtMatmulDescCreate");
  LayoutGuard la, lb, lc;
  const int32_t batch = static_cast<int32_t>(g);
  MakeRowMajorBatched(la, ab_type, m, k, a.stride[1], batch, a.stride[0]);
  MakeRowMajorBatched(lb, ab_type, k, n, b.stride[1], batch, b.stride[0]);
  MakeRowMajorBatched(lc, out_type, m, n, out.stride[1], batch, out.stride[0]);

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "batched cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "batched set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  // Key: every value that reached the layouts + descriptor above (#1732). The
  // batched lane's layouts carry per-operand ld (= the inner row stride) AND
  // batch count + batch stride (= the outer stride), all four per operand.
  GemmPlanKey key;
  key.device = q.device.index;
  key.op = kBatchedNn;
  key.m = m;
  key.n = n;
  key.k = k;
  key.lda = a.stride[1];    // MakeRowMajorBatched(la, .., m, k, a.stride[1], ..)
  key.ldb = b.stride[1];    // MakeRowMajorBatched(lb, .., k, n, b.stride[1], ..)
  key.ldc = out.stride[1];  // MakeRowMajorBatched(lc, .., m, n, out.stride[1], ..)
  key.batch = g;
  key.stride_a = a.stride[0];
  key.stride_b = b.stride[0];
  key.stride_c = out.stride[0];
  key.ab_type = static_cast<int>(ab_type);
  key.out_type = static_cast<int>(out_type);

  cublasLtMatmulHeuristicResult_t heur{};
  bool fresh = false;
  if (!GetOrQueryGemmHeuristic(ctx, key, desc.v, la.v, lb.v, lc.v, pref.v,
                               "batched cublasLtMatmulAlgoGetHeuristic", &heur, &fresh)) {
    throw std::runtime_error("vt cuda: batched_matmul: no cublasLt heuristic for g=" +
                             std::to_string(g) + " [" + std::to_string(m) + "," +
                             std::to_string(k) + "]x[" + std::to_string(k) + "," +
                             std::to_string(n) + "] " + ComboName(a, b, out));
  }
  if (fresh) MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "rowmajor-NN-batched");

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, a.data, la.v, b.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "batched cublasLtMatmul");
}

// ---- cuBLASLt FP8 (e4m3) dense GEMM ---------------------------------------
// A vt-runtime ORIGINAL. Reuses the SAME cublasLt handle + 32 MB workspace as
// the bf16 dense GEMM above; only the matmul descriptor changes to the fp8
// config: CUBLAS_COMPUTE_32F, e4m3 A/B, f32 scale.
//
// THIS LANE IS NOT A MIRROR OF vLLM, and the two claims that said it was are
// CORRECTED here (PERF-FP8-SMALL-M-DISPATCH, #1866). It used to read "the
// native equivalent of vLLM's cuBLASLt fp8 dense path (the
// `nvjet_sm121_qqtst_*` / `qq*` kernels torch._scaled_mm / cublasLt select)".
// Both halves are wrong at the pin `5559679229`:
//
//   * vLLM never reaches cuBLASLt for a per-tensor static fp8 linear. Its CUDA
//     fp8 backend order is Marlin -> FlashInfer -> Cutlass -> PerTensorTorch
//     (`vllm/model_executor/kernels/linear/__init__.py:325-334`), a
//     Cutlass-capable device takes `ops.cutlass_scaled_mm`
//     (`.../scaled_mm/cutlass.py:265`). Stated so that it survives CASING,
//     because the case-sensitive grep this used to cite does not. Three
//     readings at the pin, none of which turns on letter case: `git grep -i
//     -E "cublaslt|algogetheuristic" -- csrc` is EMPTY, so no compiled kernel
//     of upstream's touches cuBLASLt at all; `cublasLtMatmulAlgoGetHeuristic`
//     has ZERO hits repo-wide in any casing, so upstream issues no heuristic
//     query anywhere; and the ten `-i` hits under `vllm/` are both off this
//     path — `model_executor/layers/batch_invariant.py:916,927,960` is a
//     batch-invariance BLAS-preference knob (`CUBLASLT_WORKSPACE_SIZE=1` plus
//     `torch.backends.cuda.preferred_blas_library(backend="cublaslt")`,
//     reached only in batch-invariant mode) and the seven in
//     `utils/deep_gemm.py` are a lazy passthrough re-export of DeepGEMM's
//     `cublaslt_gemm_nt`, which has NO caller outside that file.
//   * We do not get the nvjet kernels either. #1857's artifact-verified GB10
//     profile (build7, FA2 manifest `[121a]`) measured this lane resolving to
//     `sm89_xmma_gemm_e4m3f32_e4m3f32_f32_tn_n_tilesize32x64x64` (26.92
//     ms/step) and `..._e4m3bf16_...` (12.91 ms/step) — the sm89 family, on
//     BOTH the f32-D and the bf16-D arm, with not one nvjet kernel in the
//     top-30.
//
// The mirror of vLLM's fp8 tower is `MatmulFp8Cutlass`
// (`cuda_matmul_fp8_cutlass.cu`), whose sm120 M ladder is now complete. Which
// of the two arms is FASTER at decode is unmeasured since that ladder landed,
// and is `## Owed` in .agents/specs/perf-fp8-small-m-dispatch.md; the arm is
// selected by `VT_DENSE_CUBLASLT_FP8`, still ON by default.
//
// cuBLASLt fp8 requires the "TN" layout — the contraction dim K must be the
// contiguous (leading) dim of BOTH operands. Our activation a_fp8 [M,K] and
// weight b_fp8 [N,K] are row-major (K contiguous), so they already satisfy it
// with no host-side transpose. We compute the row-major out[M,N] as its
// column-major transpose out^T[N,M] = op(weight,T)[N,K] @ op(act,N)[K,M]:
//   A = weight  : col-major [K,N] (rows=K,cols=N,ld=K), TRANSA=OP_T  (= [N,K] row-major)
//   B = act     : col-major [K,M] (rows=K,cols=M,ld=K), TRANSB=OP_N  (= [M,K] row-major)
//   C = D = out : col-major [N,M] (rows=N,cols=M,ld=N)               (= [M,N] row-major)
// The two per-tensor static scales are folded into the host alpha (=
// input_scale*weight_scale) applied to the fp32 accumulator — identical math to
// MatmulFp8Cutlass (per-tensor scalars: dequant = fp8 * scale, and
// alpha*(A_fp8@B_fp8) reproduces sum dequant(a)*dequant(w)). If cublasLt has no
// fp8 heuristic for a given shape (e.g. tiny M on some drivers), we fall back to
// the already-16/16-validated cutlass fp8 GEMM so the correctness gate holds.
// A built fp8 GEMM plan: the matmul descriptor, the three col-major TN layouts,
// and the heuristic-selected algo. Handles are opaque cuBLASLt pointers; when
// this plan lives in the per-device cache its handles are process-lifetime (never
// destroyed — same rationale as the LtContext above: freeing at exit races the
// CUDA driver teardown; the cache is bounded by the finite set of fp8 shapes).
// When built fresh (VT_FP8_PLAN_CACHE=0 rollback), the caller destroys it after
// the matmul via Fp8PlanGuard, mirroring the pre-cache per-call behavior.
struct Fp8Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr;  // A = weight, col-major [K,N] ld=K
  cublasLtMatrixLayout_t lb = nullptr;  // B = act,    col-major [K,M] ld=K
  cublasLtMatrixLayout_t lc = nullptr;  // C = D = out, col-major [N,M] ld=N
  cublasLtMatmulHeuristicResult_t heur{};
  // The selected algo's CUBLASLT_ALGO_CONFIG_SPLITK_NUM, OBSERVED ONCE here at
  // plan-build time rather than per GEMM (review finding F-C). It is a property
  // of `heur.algo`, so it is fixed the moment the algo is, and re-reading it on
  // every call bought nothing: a driver round-trip per layer per step, on a path
  // whose plan cache is DEFAULT OFF, so "equally cheap on a hit" described a
  // configuration almost nobody runs. Read only for a bf16 D — it is the only
  // arm any caller can claim a premise about (see Fp8Bf16DSplitKRefuses).
  bool split_k_read_ok = false;
  int32_t split_k = -1;
};

// RAII teardown for a FRESHLY-built plan (used only on the VT_FP8_PLAN_CACHE=0
// rollback path). Cached plans are deliberately NOT guarded — they leak by design.
struct Fp8PlanGuard {
  Fp8Plan* p = nullptr;
  ~Fp8PlanGuard() {
    if (p == nullptr) return;
    if (p->lc != nullptr) cublasLtMatrixLayoutDestroy(p->lc);
    if (p->lb != nullptr) cublasLtMatrixLayoutDestroy(p->lb);
    if (p->la != nullptr) cublasLtMatrixLayoutDestroy(p->la);
    if (p->desc != nullptr) cublasLtMatmulDescDestroy(p->desc);
  }
};

// Build the fp8 TN descriptor + three layouts + heuristic algo for `key` on
// `ctx.handle`. On success fills *out with raw handles (caller owns lifetime) and
// returns true. Returns false iff cuBLASLt reports no fp8 heuristic for the shape,
// or (vector-alpha keys only) the selected algo does not advertise the pointer
// mode we set — in either case any partially-created handles are destroyed and
// nothing is cached, and the caller falls back. This is the exact
// descriptor/layout/heuristic sequence the pre-cache code ran inline on every
// call; the key fields are the only inputs.
// Emit the NAMED cause of a refused plan build under VT_GEMM_ALGO_LOG=1, once
// per (shape, cause). Default OFF costs a cached bool, exactly like
// MaybeLogGemmAlgo. `cap_mask` is the mask actually read from the selected algo
// and is 0 when there was no algo to read one from.
void MaybeLogFp8PlanRefusal(const Fp8PlanKey& key, Fp8PlanRefusal refusal, uint32_t cap_mask) {
  if (!GemmAlgoLogEnabled()) return;
  static LogOncePerKey once;
  const char* tag = Fp8PlanRefusalTag(refusal);
  std::string log_key = std::string("cublasLt-fp8-refusal|m=") + std::to_string(key.m) +
                        " n=" + std::to_string(key.n) + " k=" + std::to_string(key.k) +
                        "|scale_mode=" + std::to_string(key.scale_mode) + "|" + tag;
  if (!once.ShouldLog(log_key)) return;
  std::cerr << "[VT_GEMM_ALGO] backend=cublasLt REFUSED m=" << key.m << " n=" << key.n
            << " k=" << key.k << " scale_mode=" << key.scale_mode << " reason=" << tag
            << " pointerModeCapMask=" << cap_mask << std::endl;
}

// DIAGNOSTIC ONLY (VT_GEMM_ALGO_LOG=1): dump the cuBLASLt heuristic's whole
// candidate LIST for one fp8 plan, ranked, one line per candidate, once per
// shape. Runs its own query on the already-built descriptor and layouts and
// touches neither `p.heur` nor anything the matmul reads, so with the flag
// unset (the default) this function is a cached-bool load and the plan build is
// byte-identical to what it was.
//
// Read it against an nsys kernel name: if a `nvjet`-class algo appears in this
// list below rank 0, the lever is a measured SELECTION (SGLang's fp8_gemm
// sweep is that lever). If the list is all one family, the lever is the
// DESCRIPTOR or the driver, and no sweep can help — which is a grounded
// negative rather than a no-op ship. See
// .agents/specs/perf-fp8-small-m-dispatch.md `## Owed`.
void MaybeLogDenseAlgoCandidates(const LtContext& ctx, int64_t m, int64_t n, int64_t k,
                                 cudaDataType_t ab_type, cudaDataType_t out_type,
                                 const char* lane, cublasLtMatmulDesc_t desc,
                                 cublasLtMatrixLayout_t la, cublasLtMatrixLayout_t lb,
                                 cublasLtMatrixLayout_t lc,
                                 cublasLtMatmulPreference_t pref) {
  if (!GemmAlgoLogEnabled()) return;  // cached bool; default OFF pays nothing here
  static LogOncePerKey once;
  const std::string log_key = std::string("cublasLt-dense-candidates|") + lane + "|m=" +
                              std::to_string(m) + " n=" + std::to_string(n) +
                              " k=" + std::to_string(k) + "|ab=" + std::to_string(ab_type) +
                              "|out=" + std::to_string(out_type);
  if (!once.ShouldLog(log_key)) return;
  cublasLtMatmulHeuristicResult_t results[kDenseAlgoLogCandidates] = {};
  int returned = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      ctx.handle, desc, la, lb, lc, lc, pref, /*requestedAlgoCount=*/kDenseAlgoLogCandidates,
      results, &returned);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[VT_GEMM_ALGO] backend=cublasLt DENSE-CANDIDATES lane=" << lane << " m=" << m
              << " n=" << n << " k=" << k << " query failed: " << StatusName(st) << std::endl;
    return;
  }
  for (int i = 0; i < returned; ++i) {
    int32_t algo_id = -1, split_k = -1;
    uint32_t tile = 0, stages = 0;
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id,
                                         sizeof(algo_id), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                         sizeof(tile), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &stages, sizeof(stages), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &split_k, sizeof(split_k), &written);
    std::cerr << "[VT_GEMM_ALGO] backend=cublasLt DENSE-CANDIDATE lane=" << lane
              << " rank=" << i << "/" << returned << " m=" << m << " n=" << n << " k=" << k
              << " algoId=" << algo_id << " tile=" << tile << " stages=" << stages
              << " splitK=" << split_k << " wsSize=" << results[i].workspaceSize
              << " waves=" << results[i].wavesCount << std::endl;
  }
}

void MaybeLogFp8AlgoCandidates(const LtContext& ctx, const Fp8PlanKey& key,
                               cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t la,
                               cublasLtMatrixLayout_t lb, cublasLtMatrixLayout_t lc,
                               cublasLtMatmulPreference_t pref) {
  if (!GemmAlgoLogEnabled()) return;  // cached bool; default OFF pays nothing here
  static LogOncePerKey once;
  const std::string log_key = std::string("cublasLt-fp8-candidates|m=") + std::to_string(key.m) +
                              " n=" + std::to_string(key.n) + " k=" + std::to_string(key.k) +
                              "|out=" + std::to_string(key.out_type) +
                              "|scale_mode=" + std::to_string(key.scale_mode);
  if (!once.ShouldLog(log_key)) return;
  cublasLtMatmulHeuristicResult_t results[kFp8AlgoLogCandidates] = {};
  int returned = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      ctx.handle, desc, la, lb, lc, lc, pref, /*requestedAlgoCount=*/kFp8AlgoLogCandidates,
      results, &returned);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[VT_GEMM_ALGO] backend=cublasLt CANDIDATES m=" << key.m << " n=" << key.n
              << " k=" << key.k << " scale_mode=" << key.scale_mode
              << " query failed: " << StatusName(st) << std::endl;
    return;
  }
  for (int i = 0; i < returned; ++i) {
    int32_t algo_id = -1, split_k = -1;
    uint32_t tile = 0, stages = 0;
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id,
                                         sizeof(algo_id), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                         sizeof(tile), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &stages, sizeof(stages), &written);
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &split_k, sizeof(split_k), &written);
    std::cerr << "[VT_GEMM_ALGO] backend=cublasLt CANDIDATE rank=" << i << "/" << returned
              << " m=" << key.m << " n=" << key.n << " k=" << key.k
              << " scale_mode=" << key.scale_mode << " algoId=" << algo_id << " tile=" << tile
              << " stages=" << stages << " splitK=" << split_k
              << " wsSize=" << results[i].workspaceSize
              << " waves=" << results[i].wavesCount << std::endl;
  }
}

bool BuildFp8Plan(const LtContext& ctx, const Fp8PlanKey& key, Fp8Plan* out) {
  Fp8Plan p;
  const cudaDataType_t out_type = static_cast<cudaDataType_t>(key.out_type);
  CheckLt(cublasLtMatmulDescCreate(&p.desc, static_cast<cublasComputeType_t>(key.compute_type),
                                   static_cast<cudaDataType_t>(key.scale_type)),
          "fp8 cublasLtMatmulDescCreate");
  const cublasOperation_t op_a = static_cast<cublasOperation_t>(key.trans_a);
  const cublasOperation_t op_b = static_cast<cublasOperation_t>(key.trans_b);
  CheckLt(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
          "fp8 set TRANSA=T");
  CheckLt(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
          "fp8 set TRANSB=N");
  // PERF-FP8-ALPHA-FOLD: the per-column alpha arm. The pointer mode goes on the
  // descriptor BEFORE the heuristic so cuBLASLt selects an algo that implements
  // it; that is exactly why scale_mode is in the plan key. `alpha` then points
  // at a device f32 array whose length must match the number of D's ROWS —
  // and D is our column-major [N,M], so its rows are our output's N columns.
  if (key.scale_mode == kFp8ScaleModeAlphaDeviceVec) {
    const int32_t mode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    CheckLt(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode,
                                           sizeof(mode)),
            "fp8 set POINTER_MODE=ALPHA_DEVICE_VECTOR_BETA_ZERO");
  }

  // Column-major TN layouts (default order — NOT ORDER_ROW; fp8 needs the native
  // col-major TN form). See the derivation in the block comment above.
  const cudaDataType_t a_type = static_cast<cudaDataType_t>(key.a_type);
  CheckLt(cublasLtMatrixLayoutCreate(&p.la, a_type, static_cast<uint64_t>(key.k),
                                     static_cast<uint64_t>(key.n), key.k),
          "fp8 Adesc (weight)");
  CheckLt(cublasLtMatrixLayoutCreate(&p.lb, a_type, static_cast<uint64_t>(key.k),
                                     static_cast<uint64_t>(key.m), key.k),
          "fp8 Bdesc (act)");
  CheckLt(cublasLtMatrixLayoutCreate(&p.lc, out_type, static_cast<uint64_t>(key.n),
                                     static_cast<uint64_t>(key.m), key.n),
          "fp8 Cdesc (out)");

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "fp8 cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "fp8 set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  int returned = 0;
  const cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
      ctx.handle, p.desc, p.la, p.lb, p.lc, p.lc, pref.v, /*requestedAlgoCount=*/kGemvHeuristicAlgos, &p.heur,
      &returned);
  // The vector-alpha arm additionally REQUIRES that the selected algo advertise
  // the mode we set. The heuristic is documented to honor the descriptor, but a
  // driver that returns an algo without the capability would silently apply
  // something other than the per-column alpha, so the bit is verified rather
  // than assumed; a refusal is not an error, it selects the two-launch fallback.
  const bool heuristic_ok = hst == CUBLAS_STATUS_SUCCESS && returned > 0;
  bool pointer_mode_ok = true;
  uint32_t cap_mask = 0;
  if (heuristic_ok && key.scale_mode == kFp8ScaleModeAlphaDeviceVec) {
    static_assert(kFp8PointerModeMaskAlphaDeviceVectorBetaZero ==
                      static_cast<unsigned int>(
                          CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO),
                  "fp8_plan_cache.h pointer-mode mask drifted from cublasLt.h");
    size_t written = 0;
    const cublasStatus_t cst = cublasLtMatmulAlgoCapGetAttribute(
        &p.heur.algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &cap_mask, sizeof(cap_mask), &written);
    pointer_mode_ok = cst == CUBLAS_STATUS_SUCCESS && written == sizeof(cap_mask) &&
                      Fp8AlphaVecCapSupported(cap_mask);
  }
  // The candidate dump runs AFTER the production query, on its own results
  // array, so nothing it does can precede or perturb the algo this plan
  // latches — an ordering choice, not a determinism argument (#1866). It runs
  // on the refusal path too, because a shape cuBLASLt has no fp8 heuristic for
  // is exactly a shape whose candidate list a reader wants to see.
  MaybeLogFp8AlgoCandidates(ctx, key, p.desc, p.la, p.lb, p.lc, pref.v);

  if (!heuristic_ok || !pointer_mode_ok) {
    // A refusal emits NOTHING from MaybeLogGemmAlgo (there is no plan to
    // describe), so on the GB10 run of 2026-08-11 the vector-alpha arm was
    // indistinguishable from "never called" — see fp8_plan_cache.h. Under the
    // existing diagnostic flag, name the cause and the mask that was actually
    // read, so the next run reports a reason instead of an absence.
    MaybeLogFp8PlanRefusal(key, Fp8PlanRefusalFor(heuristic_ok, pointer_mode_ok), cap_mask);
    if (p.lc != nullptr) cublasLtMatrixLayoutDestroy(p.lc);
    if (p.lb != nullptr) cublasLtMatrixLayoutDestroy(p.lb);
    if (p.la != nullptr) cublasLtMatrixLayoutDestroy(p.la);
    if (p.desc != nullptr) cublasLtMatmulDescDestroy(p.desc);
    return false;  // caller falls back to cutlass; nothing cached
  }
  // OBSERVE the selected algo's split-K, once, here — the decision that consumes
  // it is Fp8Bf16DSplitKRefuses at the call site, and only a caller that CLAIMED
  // the splitK=1 premise is refused by it. Reading it at build time (rather than
  // per GEMM, as the first repair did) keeps the observation on the cold path
  // beside the ~0.8 ms heuristic it belongs to, and leaves the hot path a plain
  // struct read. A short read counts as UNREAD: `split_k` would keep the -1
  // sentinel, and reporting that as a bad VALUE would name the wrong cause.
  if (out_type == CUDA_R_16BF) {
    size_t written = 0;
    const cublasStatus_t sst = cublasLtMatmulAlgoConfigGetAttribute(
        &p.heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &p.split_k, sizeof(p.split_k), &written);
    p.split_k_read_ok = sst == CUBLAS_STATUS_SUCCESS && written == sizeof(p.split_k);
  }
  *out = p;
  return true;
}

// Per-device fp8 plan cache. On a hit, returns the cached {desc, layouts, algo}
// by value (the handles are stable pointers into process-lifetime cached objects
// — std::unordered_map keeps element addresses stable across rehash, and the
// values are never destroyed), skipping the descriptor/layout creation + the
// ~0.8 ms heuristic entirely. On a miss, builds under the lock (a one-time cost
// per unique shape) and inserts. Returns false iff BuildFp8Plan found no fp8
// heuristic (not cached — a rare shape that falls back to cutlass). The lock is
// held only across the map access + the (cold) build, never across the matmul.
bool GetOrBuildCachedFp8Plan(const LtContext& ctx, const Fp8PlanKey& key, Fp8Plan* out) {
  static std::mutex mu;
  static std::unordered_map<Fp8PlanKey, Fp8Plan, Fp8PlanKeyHash> plans;
  std::lock_guard<std::mutex> lock(mu);
  auto it = plans.find(key);
  if (it != plans.end()) {
    *out = it->second;  // cache hit: skip desc/layout creation + heuristic
    return true;
  }
  Fp8Plan p;
  if (!BuildFp8Plan(ctx, key, &p)) return false;  // no heuristic: uncached fallback
  plans.emplace(key, p);  // process-lifetime; handles never destroyed (by design)
  *out = p;
  return true;
}

// ---- ENFORCE the splitK precondition, ON THE CALLER THAT CLAIMED IT ---------
// A caller passes `claims_splitk1_premise` when its bf16 D is asserted to be
// byte-equivalent to the f32 D of the same call site — i.e. when it claims the
// narrowing changed only the STORE WIDTH. That claim holds only while the
// accumulation behind the store is the same single ordered f32 reduction the
// f32-D arm used, i.e. while `splitK == 1` (measured 2026-08-12: algoId 67,
// splitK=1, every gate shape, every M; .agents/specs/perf-fp8-alpha-fold.md
// §Attempt 4). `out_type` is part of Fp8PlanKey's == and its hash, so the bf16 D
// deliberately selects a DIFFERENT plan from the f32 D — which is exactly the
// freedom cuBLASLt needs to pick a different split-K. Had it taken it, the delta
// would be a REDUCTION-ORDER change wearing a store-width change's clothes, and
// a bf16 store is very good at hiding those from a token gate.
//
// It is a hard refusal, not a tolerance — a "<1% of elements differ" allowance
// would only be measuring how well bf16 conceals the defect — and not a silent
// fallback to cutlass, which would substitute a third reduction order without
// saying so.
//
// WHAT THIS FUNCTION DOES NOT DO, and the first repair did (review finding F-A):
// it does not key off the DTYPE. bf16-D fp8 cuBLASLt GEMMs are a pre-existing,
// DEFAULT-ON capability — every `o_proj_fp8` / `out_proj_fp8` in qwen3_5.cpp
// arrives here at bf16 through MatmulFp8Cutlass{,PreQuant}D under
// DenseCublasLtFp8Enabled() — and NONE of those call sites ever claimed
// byte-equivalence with an f32-D arm. Split-K is perfectly correct for them.
// Guarding on `out_type == CUDA_R_16BF` alone therefore put a new throw on a
// default path and falsified the row's claim that with both toggles unset,
// behavior is byte-identical to before it. The premise travels WITH the caller
// that makes it.
//
// The whole decision is Fp8Bf16DSplitKRefuses, on the CPU tier, where it is
// unit-tested and mutation-proved. Nothing is decided here: this host has no
// nvcc, so a branch written HERE is a branch nothing on it can compile, let
// alone execute — which is how F-A survived the first repair's gate.
void RequireBf16DSplitKOne(const Fp8Plan& plan, bool claims_splitk1_premise,
                           cudaDataType_t out_type, int64_t m, int64_t n, int64_t k,
                           const char* site) {
  const bool out_is_bf16 = out_type == CUDA_R_16BF;
  if (!Fp8Bf16DSplitKRefuses(claims_splitk1_premise, out_is_bf16, plan.split_k_read_ok,
                             plan.split_k))
    return;
  const Fp8Bf16DSplitK verdict =
      Fp8Bf16DSplitKVerdict(out_is_bf16, plan.split_k_read_ok, plan.split_k);
  throw std::runtime_error(
      std::string("vt cuda: ") + site +
      ": this call site asked for a bf16 D on the express claim that it is byte-equivalent to "
      "the same GEMM's f32 D — a store-width narrowing over one ordered f32 reduction — which "
      "requires the selected cuBLASLt plan to report splitK=1. Got " +
      Fp8Bf16DSplitKTag(verdict) +
      " (splitK=" + (plan.split_k_read_ok ? std::to_string(plan.split_k) : "unread") +
      ") at m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k) +
      ". Re-measure .agents/specs/perf-fp8-alpha-fold.md §Attempt 4's premise on this driver "
      "before using that arm. Only a caller that CLAIMS the premise can reach this refusal; the "
      "one that does today is the merged GDN fp8 in_proj under VT_GDN_FP8_IN_BF16 (default OFF) "
      "— unset it to take the f32 D. An ordinary bf16-D fp8 GEMM (o_proj/out_proj) makes no such "
      "claim and is never checked.");
}

void MatmulFp8CublasLtKernelCuda(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                                 float alpha, bool claims_splitk1_premise) {
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {  // empty reduction: out = 0 (f32 and bf16 zero are all-zero bytes)
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "fp8 k=0 memset");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  // The FULL key that determines the descriptor + selected algo (see
  // fp8_plan_cache.h for the per-field rationale). alpha is applied per-call and
  // does NOT affect the descriptor/algo, so it is deliberately absent from the key.
  Fp8PlanKey key;
  key.device = q.device.index;
  key.m = m;
  key.n = n;
  key.k = k;
  key.out_type = static_cast<int>(out_type);
  key.a_type = static_cast<int>(CUDA_R_8F_E4M3);
  key.compute_type = static_cast<int>(CUBLAS_COMPUTE_32F);
  key.scale_type = static_cast<int>(CUDA_R_32F);
  key.trans_a = static_cast<int>(CUBLAS_OP_T);
  key.trans_b = static_cast<int>(CUBLAS_OP_N);
  key.epilogue = static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT);
  key.scale_mode = 0;  // per-tensor scale folded into host alpha; no device scale ptrs

  // Cache ON (DEFAULT; #1843): reuse the per-device plan, skipping the per-call
  // descriptor/layout creation + heuristic — on CUDA 13.3 that heuristic fails
  // inside CUDA-graph capture, so the warmed-before-capture cached plan is the
  // shipped production path. Cache OFF (VT_FP8_PLAN_CACHE=0, the rollback / A/B
  // arm): build a fresh plan and destroy it after the matmul (Fp8PlanGuard),
  // exactly the pre-cache per-call behavior — it reproduces the capture failure.
  const bool cache_on = Fp8PlanCacheEnabled();
  Fp8Plan plan;
  Fp8PlanGuard guard;  // engaged only on the rollback path
  bool have_plan;
  if (cache_on) {
    have_plan = GetOrBuildCachedFp8Plan(ctx, key, &plan);
  } else {
    have_plan = BuildFp8Plan(ctx, key, &plan);
    if (have_plan) guard.p = &plan;  // fresh plan: destroy after the matmul
  }
  if (!have_plan) {
    // No cublasLt fp8 kernel for this shape/config -> keep the gate robust by
    // routing to the already-validated cutlass fp8 GEMM (same fp8 math).
    ::vt::MatmulFp8Cutlass(q, out, a_fp8, b_fp8, alpha);
    return;
  }
  MaybeLogGemmAlgo(plan.heur, m, n, k, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, out_type, "TN-fp8");
  // The splitK precondition, on the plan actually selected (cached or fresh —
  // same algo either way, and the observation was taken when it was chosen). A
  // no-op for every caller that did not claim it, which is every caller but the
  // bf16-D lever: no driver round-trip, no branch beyond one bool.
  RequireBf16DSplitKOne(plan, claims_splitk1_premise, out_type, m, n, k, "matmul_fp8_cublaslt");

  // out = alpha * op(weight) @ op(act) + 0 * C; C and D share out's buffer/layout.
  const float beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, plan.desc, &alpha, b_fp8.data, plan.la, a_fp8.data, plan.lb,
                         &beta, out.data, plan.lc, out.data, plan.lc, &plan.heur.algo,
                         ctx.workspace, kWorkspaceBytes, s),
          "fp8 cublasLtMatmul");
}

// ---- cuBLASLt FP8 with a per-output-COLUMN alpha ---------------------------
// PERF-FP8-ALPHA-FOLD (.agents/specs/perf-fp8-alpha-fold.md, #402 §3 "Lever B").
//
// An N-concatenated fp8 operand whose shards carry DIFFERENT folded alphas
// cannot express its scale as one host scalar, so this seam shipped applying the
// per-column vector in a SECOND full-tensor pass (vt::MulColVecF32). At T=4096
// prefill that pass is a read-modify-write of a [T,16384] f32 tensor per GDN
// layer running at 77% of the device's peak bandwidth — 43.6% of the measured
// 27B prefill deficit. cuBLASLt can apply the same vector in the epilogue at no
// cost via CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO, whose vector
// length must equal the number of D's ROWS; our D is column-major [N,M], so its
// rows are the output's N columns and the caller's f32 [N] resident vector is
// already exactly right (no repack, no new allocation).
//
// Numerically the two forms are the same single IEEE f32 multiply on the same
// f32 accumulator: today is `accum -> x1.0 -> store f32 -> load f32 -> xalpha ->
// store f32`, this is `accum -> xalpha -> store f32`, at the same
// scale_type=CUDA_R_32F and the same store dtype. What CAN differ is the ALGO:
// the pointer mode is on the descriptor the heuristic reads, so it may select a
// different split-K, and f32 addition is not associative. That is why the mode
// is part of the plan key, why the arm is DEFAULT OFF, and why the CUDA-tier
// test asserts BITWISE equality at the real gate shapes (the same
// shape-conditional method #213 established for the merge itself).
//
// Falls back to the two-launch form — byte for byte what this file did before —
// when the toggle is off, when cuBLASLt has no plan, or when the selected algo
// does not advertise the pointer mode.
void MatmulFp8CublasLtAlphaVecKernelCuda(Queue& q, Tensor& out, const Tensor& a_fp8,
                                         const Tensor& b_fp8, const Tensor& alpha_vec,
                                         bool claims_splitk1_premise) {
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {  // empty reduction: out = 0, and alpha * 0 == 0 (as in the fallback)
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "fp8 alpha-vec k=0 memset");
    return;
  }

  // The shipped two-launch form: GEMM at alpha=1, then the column vector. Also
  // the rollback arm, reached in the same binary with the toggle unset.
  auto two_launch = [&]() {
    // The caller's premise travels WITH the GEMM it is a premise about.
    MatmulFp8CublasLtKernelCuda(q, out, a_fp8, b_fp8, 1.0F, claims_splitk1_premise);
    ::vt::MulColVecF32(q, out, alpha_vec);
  };
  if (!Fp8AlphaVecEpilogueEnabled()) {
    two_launch();
    return;
  }
  // A bf16 D takes the two-launch arm UNCONDITIONALLY, and that is a correctness
  // gate, not an oversight (PERF-FP8-ALPHA-FOLD / #417).
  //
  // This op's contract — and the whole byte-exactness argument in
  // .agents/specs/perf-fp8-alpha-fold.md §Byte-exactness — is that the epilogue
  // arm and the fallback arm compute the SAME thing, so VT_FP8_ALPHA_VEC_EPILOGUE
  // is a pure performance A/B that can never move a token. At an f32 D that holds:
  // `acc -> x1.0 -> store f32 -> load f32 -> xalpha -> store f32` and
  // `acc -> xalpha -> store f32` are the same single IEEE f32 multiply, because
  // x1.0 and the f32 round-trip are both exact.
  //
  // At a bf16 D it does NOT hold. The fallback rounds TWICE (store bf16, then
  // multiply and store bf16 again) while the epilogue rounds ONCE (multiply, then
  // store bf16), so the two arms would disagree by up to a ulp on a large
  // fraction of words — the identical double-rounding defect that got the z-slice
  // fallback REJECTED in this row's spec. Letting the toggle silently change
  // values would turn a performance switch into a numerics switch.
  //
  // Nothing is lost today: all three cuBLASLt vector-alpha mechanisms are MEASURED
  // unavailable on GB10/sm_121a (spec §Outcome), so the epilogue arm never
  // executes at any dtype here. Note for whoever gets hardware that offers it —
  // the single-rounding epilogue form is the MORE vLLM-faithful one (vLLM computes
  // bf16(acc * alpha) with its requantized single scalar, modelopt.py:458), so
  // lifting this gate is a deliberate, gated decision worth making, not a bug to
  // fix in passing.
  //
  // A caller's splitK premise is therefore enforced for this op too, just one
  // frame down: two_launch() forwards `claims_splitk1_premise` to
  // MatmulFp8CublasLtKernelCuda, which runs RequireBf16DSplitKOne on the plan it
  // selects. Repeating the check here would be unreachable code, since every
  // bf16 D leaves through this branch — and the epilogue arm below is f32-D
  // only, where no premise about a bf16 store can apply.
  if (out.dtype != DType::kF32) {
    two_launch();
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  // f32 D by the gate directly above; the derivation matches the scalar-alpha
  // GEMM's and is carried into key.out_type below. `out_type` is part of
  // Fp8PlanKey's == AND its hash, so a bf16-D plan could never be handed back for
  // an f32-D matmul in any case.
  const cudaDataType_t out_type = CUDA_R_32F;
  Fp8PlanKey key;
  key.device = q.device.index;
  key.m = m;
  key.n = n;
  key.k = k;
  key.out_type = static_cast<int>(out_type);
  key.a_type = static_cast<int>(CUDA_R_8F_E4M3);
  key.compute_type = static_cast<int>(CUBLAS_COMPUTE_32F);
  key.scale_type = static_cast<int>(CUDA_R_32F);
  key.trans_a = static_cast<int>(CUBLAS_OP_T);
  key.trans_b = static_cast<int>(CUBLAS_OP_N);
  key.epilogue = static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT);
  // The ONLY field that separates this plan from the scalar-alpha plan for the
  // identical shape. Without it the cache would hand back an algo selected under
  // a different pointer mode.
  key.scale_mode = Fp8ScaleModeFor(/*alpha_device_vector=*/true);

  Fp8Plan plan;
  // Same polarity as the scalar-alpha lane above (#1843): the cache is DEFAULT
  // ON so the heuristic is never queried under CUDA-graph capture;
  // VT_FP8_PLAN_CACHE=0 is the fresh-plan rollback arm.
  Fp8PlanGuard guard;  // engaged only on the VT_FP8_PLAN_CACHE=0 rollback
  bool have_plan;
  if (Fp8PlanCacheEnabled()) {
    have_plan = GetOrBuildCachedFp8Plan(ctx, key, &plan);
  } else {
    have_plan = BuildFp8Plan(ctx, key, &plan);
    if (have_plan) guard.p = &plan;
  }
  if (!have_plan) {  // no fp8 plan, or the algo refuses the pointer mode
    two_launch();
    return;
  }
  MaybeLogGemmAlgo(plan.heur, m, n, k, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, out_type,
                   "TN-fp8-alphavec");

  // alpha is a DEVICE pointer here (one f32 per output column); beta is zero by
  // the pointer mode's definition, and is passed as a host zero so the argument
  // is never an uninitialized read whatever the driver does with it.
  const float beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, plan.desc, alpha_vec.data, b_fp8.data, plan.la, a_fp8.data,
                         plan.lb, &beta, out.data, plan.lc, out.data, plan.lc, &plan.heur.algo,
                         ctx.workspace, kWorkspaceBytes, s),
          "fp8 cublasLtMatmul (device alpha vector)");
}

// Registers the CUDA matmul during static init (table fill only, no CUDA
// calls — see cuda_ops.cu for the rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelCuda)));
    RegisterOpScalarType(OpId::kMatmul, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(OpId::kMatmulBT, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernelCuda)));
    RegisterOpScalarType(OpId::kMatmulBT, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(OpId::kBatchedMatmul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<BatchedMatmulFn>(&BatchedMatmulKernelCuda)));
    RegisterOp(OpId::kMatmulFp8CublasLt, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8CublasLtFn>(&MatmulFp8CublasLtKernelCuda)));
    RegisterOp(OpId::kMatmulFp8CublasLtAlphaVec, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulFp8CublasLtAlphaVecFn>(&MatmulFp8CublasLtAlphaVecKernelCuda)));
  }
} registrar;

}  // namespace

// QUANT-EXL3 W6. fp16 @ fp16 -> fp16|f32 row-major GEMM for the EXL3
// reconstruct+cuBLAS prefill path. The reference (exllamavav3_ext/hgemm.cu)
// uses cublasGemmEx; this tree links cublasLt only, so the GEMM goes through
// cublasLtMatmul with the same heuristic cache as the bf16/f32 lanes.
//
// a is fp16 [M, K] row-major, b is fp16 [K, N] row-major, c is [M, N]
// row-major (fp16 or f32). Compute type CUBLAS_COMPUTE_32F with f32 alpha/beta,
// matching the reference's CUBLAS_COMPUTE_32F / CUBLAS_GEMM_DEFAULT_TENSOR_OP.
void CublasLtHgemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& b) {
  if (a.dtype != DType::kF16 || b.dtype != DType::kF16 ||
      (c.dtype != DType::kF16 && c.dtype != DType::kF32)) {
    throw std::runtime_error(std::string("vt cuda: CublasLtHgemm: unsupported dtype combo (") +
                             Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(c.dtype) +
                             "; expected (f16,f16)->f16|f32");
  }
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = CUDA_R_16F;
  const cudaDataType_t out_type = c.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16F;

  // ldc = c.stride[0]: the output's actual row stride. For a column slice of a
  // wider output (EXL3 N>32768 slicing), c.stride[0] > n, and using ldc = n
  // packs rows too tightly so they overwrite each other. The reference
  // (exllamav3 hgemm.cu:49,72) does the same: ldc = c.stride(-2).
  // ldb = n (= b.shape[1]): the reconstruct kernel writes B contiguously with
  // row stride n_slice even when the scratch buffer is wider, so cuBLAS must
  // read at that same stride, not at b.stride[0].
  const int64_t lda = a.stride[0];
  const int64_t ldb = n;
  const int64_t ldc = c.stride[0];

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate (hgemm)");
  LayoutGuard la, lb, lc;
  MakeRowMajor(la, ab_type, m, k, lda);
  MakeRowMajor(lb, ab_type, k, n, ldb);
  MakeRowMajor(lc, out_type, m, n, ldc);

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "cublasLtMatmulPreferenceCreate (hgemm)");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES (hgemm)");

  GemmPlanKey key;
  key.device = q.device.index;
  key.op = kFp16Nn;
  key.m = m;
  key.n = n;
  key.k = k;
  key.lda = lda;
  key.ldb = ldb;
  key.ldc = ldc;
  key.ab_type = static_cast<int>(ab_type);
  key.out_type = static_cast<int>(out_type);

  cublasLtMatmulHeuristicResult_t heur{};
  bool fresh = false;
  if (!GetOrQueryGemmHeuristic(ctx, key, desc.v, la.v, lb.v, lc.v, pref.v,
                               "hgemm cublasLtMatmulAlgoGetHeuristic", &heur, &fresh)) {
    throw std::runtime_error("vt cuda: CublasLtHgemm: no cublasLt heuristic for [" +
                             std::to_string(m) + "," + std::to_string(k) + "]x[" +
                             std::to_string(k) + "," + std::to_string(n) + "] (f16,f16)->" +
                             Name(c.dtype));
  }
  if (fresh) MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "hgemm-fp16-NN");

  if (k == 0) {
    CheckCuda(cudaMemsetAsync(c.data, 0, c.Bytes(), s), "hgemm k=0 memset");
    return;
  }

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, a.data, la.v, b.data, lb.v, &beta,
                         c.data, lc.v, c.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "cublasLtMatmul (hgemm)");
}

}  // namespace vt::cuda
