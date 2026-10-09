// EXL3 reconstruct + GEMM — the ROCm arm of OpId::kExl3ReconstructGemm, the
// op dense_attn_block.h::Exl3MatmulD dispatches at M > 144 (upstream's
// AUTO_RECONSTRUCT_THRESHOLD, exl3.py:10,135). Spec
// .agents/specs/rocm-exl3-gemv.md (Slice B), row BACKEND-ROCM.
//
// THE SHAPE IS THE CUDA ARM'S UNFUSED PATH, NOT ITS FUSED ONE.
// cuda_exl3.cu's Exl3ReconstructGemmKernelCuda picks `reconstruct_had_slice`
// — one kernel that decodes the trellis AND applies both Hadamard transforms
// to the reconstructed weight — when M >= 1024, and the plain
// `reconstruct_slice` + cuBLAS + separate had_r_128 calls otherwise. The fused
// kernel's geometry (had_fh_r_128_xxx_dual, 128x64 LDS tiles, xor-shuffled
// fragments) has no small port to LDS; the unfused chain is already fully
// served by this backend's pieces:
//
//   1. had_r_128(A, suh)             via Exl3LaunchHadRocm (rocm_exl3.hip —
//                                    the same kernel the fused chain runs)
//   2. W = reconstruct(trellis)      ReconstructDecodeK below, ONE thread per
//                                    output element over the inverse fragment
//                                    map (rocm_exl3_decode.h), so writes are
//                                    coalesced and every codeword is decoded
//                                    exactly once
//   3. C = A_had @ W                 hipblasGemmEx, (f16,f16)->f16|f32
//   4. had_r_128(C, svh), in place   Exl3LaunchHadRocm again
//
// M >= 1024 therefore takes the UNFUSED path on this backend: correct at every
// M the op serves, where the CUDA arm's M>=1024 fused kernel is owed
// separately. The N > 32768 slice loop mirrors upstream's own structure.
#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#ifdef VLLM_CPP_HAS_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_exl3_decode.h"
#include "vt/rocm/rocm_getblas_product.h"

namespace vt::rocm {
namespace {

inline void CheckR(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3 recon: ") + what + ": " +
                             hipGetErrorString(err));
}
inline void CheckBlas(hipblasStatus_t st, const char* what) {

  if (st != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("vt rocm exl3 recon: ") + what +
                             " failed with status " + std::to_string(static_cast<int>(st)));
}

bool StreamCapturing(hipStream_t s);  // defined below, same anonymous namespace
inline hipStream_t AsStreamR(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

constexpr float kInvSqrt128R = 0.088388347648f;
// Upstream's slice bound (reconstruct.cu / cuda_exl3.cu:2774): the fp16
// scratch never holds more than this many reconstructed columns at once.
// It is a memory/throughput trade-off, not a correctness bound — the decode
// kernel and the GEMM slice loop honour any positive width — so the 24 GiB
// gfx1100 serves that cannot hold a full 32768-column scratch beside the KV
// pool get an escape: VT_EXL3_RECON_SLICE_N overrides the width. Values below
// one decode tile (16) would leave `n_packed` arithmetic pointing past the
// slice, and anything over int range is not a slice; both refuse by name.
int64_t ReconSliceN() {
  static const int64_t n = [] {
    const int v = EnvIntStrict("VT_EXL3_RECON_SLICE_N", 32768);
    VT_CHECK(v >= 16,
             "vt rocm exl3 recon: VT_EXL3_RECON_SLICE_N=" +
                 std::to_string(v) +
                 " is below one 16-column decode tile; the smallest legal "
                 "slice is 16");
    return static_cast<int64_t>(v);
  }();
  return n;
}
// ─── step 2: trellis -> dense fp16 weight ────────────────────────────────────
//
// One thread per (row, col) output element. The fragment layout means codeword
// t's decoded weight lands at (r, c) = Exl3TileRowMajorIndex(t), so the thread
// OWNING (r, c) inverts the map (Exl3TileRowMajorIndexInv, rocm_exl3_decode.h)
// and decodes exactly its own codeword — no LDS, no barriers, every codeword
// decoded once, and the 16-wide x-tile keeps writes coalesced. `n_slice` is
// w's row stride and `n_packed` the packed trellis's own n: the two differ
// under the 32768-column slicing, and conflating them is the reconstruct-class
// defect this family files.
__global__ void ReconstructDecodeSliceK(half* __restrict__ w, const uint16_t* __restrict__ trellis,
                                        int64_t k, int64_t n_slice, int64_t n_packed,
                                        int bits, int codebook, int64_t packed_n_off) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= k || c >= n_slice) return;
  const int64_t tiles_n = n_packed / 16;
  const int64_t tile_words = 16 * static_cast<int64_t>(bits);
  const uint16_t* tp =
      trellis + (((r >> 4) * tiles_n) + packed_n_off + (c >> 4)) * tile_words;
  const int t = Exl3TileRowMajorIndexInv(static_cast<int>(r & 15), static_cast<int>(c & 15));
  w[r * n_slice + c] =
      __float2half_rn(Exl3DecodeCodeword(Exl3TileCodeword(tp, bits, t), codebook));
}


// ─── step 3, fast arm: the same GEMM through hipBLASLt ───────────────────────
//
// hipblasGemmEx + HIPBLAS_COMPUTE_32F + HIPBLAS_GEMM_DEFAULT resolves to a
// non-tensor-core assembly kernel on gfx1101 (rocprofv3 on Qwen3.5-9B-EXL3:
// Cijk...HSS at ~16.4 ms/call, ~2 TFLOP/s — 6% of the fp16 MMA peak for a
// m=2022 n=4096 k=4096 call). hipBLASLt's heuristic returns the MMA kernels
// for the same shape. VT_EXL3_RECON_NO_LT=1 pins the old arm for A/B. The Lt
// call honours `ldc` exactly as GemmEx does — Lt matrix layouts take a stride.
#ifdef VLLM_CPP_HAS_HIPBLASLT
struct ReconLtHandle {
  hipblasLtHandle_t h = nullptr;
  ReconLtHandle() {
    if (hipblasLtCreate(&h) != HIPBLAS_STATUS_SUCCESS) h = nullptr;
  }
  ~ReconLtHandle() {
    if (h) (void)hipblasLtDestroy(h);
  }
};

static bool ReconLtDisabled() {
  static const bool off = EnvBoolStrict("VT_EXL3_RECON_NO_LT", false);
  return off;
}

static bool HgemmLtRowMajor(hipStream_t s, void* C, const void* A, const void* W,
                            int m, int n, int k, int64_t ldc, bool c_f32) {
  static thread_local ReconLtHandle lt;
  if (!lt.h) return false;

  // Layout key: shapes + output dtype + ldc. Algorithms are cached; the
  // heuristic call is per new shape only.
  const hipDataType c_dt = c_f32 ? HIP_R_32F : HIP_R_16F;
  int dev = -1;
  if (hipGetDevice(&dev) != hipSuccess) dev = -1;
  struct LtKey {
    int m, n, k;
    int64_t ldc;
    hipDataType c;
    int dev;
    bool operator==(const LtKey& o) const {
      return m == o.m && n == o.n && k == o.k && ldc == o.ldc && c == o.c &&
             dev == o.dev;
    }
  };
  struct LtKeyHash {
    size_t operator()(const LtKey& x) const {
      return (static_cast<size_t>(x.m) * 1315423911u) ^
             (static_cast<size_t>(x.n) * 2654435761u) ^
             (static_cast<size_t>(x.k) * 97531u) ^
             (static_cast<size_t>(x.ldc) * 40503u) ^ static_cast<size_t>(x.c) ^
             (static_cast<size_t>(x.dev) * 31u);
    }
  };
  static thread_local std::unordered_map<LtKey, hipblasLtMatmulAlgo_t, LtKeyHash>
      cache;
  static thread_local std::unordered_map<LtKey, size_t, LtKeyHash> ws_sizes;
  const LtKey key{m, n, k, ldc, c_dt, dev};

  // Column-major views, matching HgemmRowMajor's OP_N/OP_N contraction:
  // C_cm(N,M) = W_cm(N,K) @ A_cm(K,M). ldc is C's column stride (M in a
  // column-major view), which is where the parent's row stride lands for a
  // sliced column panel — same identity as the GemmEx arm.
  hipblasLtMatrixLayout_t lw = nullptr, la = nullptr, lc = nullptr;
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  auto cleanup = [&]() {
    if (lw) (void)hipblasLtMatrixLayoutDestroy(lw);
    if (la) (void)hipblasLtMatrixLayoutDestroy(la);
    if (lc) (void)hipblasLtMatrixLayoutDestroy(lc);
    if (desc) (void)hipblasLtMatmulDescDestroy(desc);
    if (pref) (void)hipblasLtMatmulPreferenceDestroy(pref);
  };
  bool ok = hipblasLtMatrixLayoutCreate(&lw, HIP_R_16F, n, k, n) ==
                HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k) ==
                HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatrixLayoutCreate(&lc, c_dt, n, m, ldc) ==
                HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) ==
                HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS;
  if (!ok) {
    cleanup();
    return false;
  }
  const uint64_t max_ws = 32ull << 20;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

  auto it = cache.find(key);
  if (it == cache.end()) {
    // The timing sweep below runs hipMalloc + hipStreamSynchronize +
    // hipEvent* — all illegal inside stream capture. A new (shape,ldc,dtype)
    // key first seen during capture declines to the GemmEx arm instead of
    // poisoning the capture; the first warm call outside capture caches it.
    if (StreamCapturing(s)) {
      cleanup();
      return false;
    }
    // Rank-0 is not trustworthy on gfx11: for (m=2022,n=4096,k=4096,f16->f32)
    // the heuristic's first pick measured 15.7 ms while algo #4 of the same
    // request ran 1.5 ms (44.6 TFLOP/s, 10.8x). Ask for several candidates and
    // time each on the REAL buffers once — the sweep runs on first call only,
    // per (m,n,k,ldc,dtype) key, and its cost (~30 ms) amortizes inside the
    // first prefill it saves.
    hipblasLtMatmulHeuristicResult_t heur[16];
    int returned = 0;
    const hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
        lt.h, desc, lw, la, lc, lc, pref, 16, heur, &returned);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
      cleanup();
      return false;
    }
    const float walpha = 1.0f, wbeta = 0.0f;
    double best_ms = std::numeric_limits<double>::max();
    int best = -1;
    size_t best_ws = 0;
    for (int i = 0; i < returned; ++i) {
      if (heur[i].state != HIPBLAS_STATUS_SUCCESS) continue;
      void* tws = nullptr;
      if (heur[i].workspaceSize &&
          hipMalloc(&tws, heur[i].workspaceSize) != hipSuccess)
        continue;
      // Warm-up then time 3 back-to-back launches on the caller's buffers.
      // These writes land in c anyway — the GEMM is the caller's own output.
      // A failing matmul must not win 'best': an algo that fails at launch
      // records ~0 ms and would be cached for the production call below.
      hipblasStatus_t ms0 = hipblasLtMatmul(lt.h, desc, &walpha, W, lw, A, la,
                                            &wbeta, C, lc, C, lc, &heur[i].algo,
                                            tws, heur[i].workspaceSize, s);
      if (ms0 != HIPBLAS_STATUS_SUCCESS) {
        if (tws) (void)hipFree(tws);
        continue;
      }
      (void)hipStreamSynchronize(s);
      hipEvent_t ev0, ev1;
      (void)hipEventCreate(&ev0);
      (void)hipEventCreate(&ev1);
      (void)hipEventRecord(ev0, s);
      hipblasStatus_t timed_st = HIPBLAS_STATUS_SUCCESS;
      for (int r = 0; r < 3; ++r) {
        const hipblasStatus_t r_ =
            hipblasLtMatmul(lt.h, desc, &walpha, W, lw, A, la, &wbeta, C, lc,
                            C, lc, &heur[i].algo, tws, heur[i].workspaceSize, s);
        if (r_ != HIPBLAS_STATUS_SUCCESS) timed_st = r_;
      }
      (void)hipEventRecord(ev1, s);
      (void)hipStreamSynchronize(s);
      float ms = 0.f;
      (void)hipEventElapsedTime(&ms, ev0, ev1);
      ms /= 3.0f;
      (void)hipEventDestroy(ev0);
      (void)hipEventDestroy(ev1);
      if (tws) (void)hipFree(tws);
      if (timed_st != HIPBLAS_STATUS_SUCCESS) continue;
      if (ms < best_ms) {
        best_ms = ms;
        best = i;
        best_ws = heur[i].workspaceSize;
      }
    }
    if (best < 0) {
      cleanup();
      return false;
    }
    it = cache.emplace(key, heur[best].algo).first;
    ws_sizes[key] = best_ws;  // overwrite: a stale entry must not pin the
                            // freshly-swept algo to a too-small workspace
    std::fprintf(stderr,
                 "vt rocm exl3 recon: Lt algo %d of %d for m=%d n=%d k=%d "
                 "(%.3f ms)\n",
                 best, returned, m, n, k, best_ms);
  }

  // Grow-only Lt workspace per stream — the same capture-safety contract as
  // the recon scratch above.
  struct LtWs {
    void* p = nullptr;
    size_t bytes = 0;
  };
  static std::mutex ws_mu;
  static std::unordered_map<hipStream_t, LtWs> ws_pool;
  const size_t need = ws_sizes.at(key);
  void* ws = nullptr;
  if (need) {
    std::lock_guard<std::mutex> lk(ws_mu);
    LtWs& e = ws_pool[s];
    if (need > e.bytes) {
      if (StreamCapturing(s)) {
        cleanup();
        return false;  // refusing growth mid-capture; GemmEx arm picks it up
      }
      void* p = nullptr;
      if (hipMalloc(&p, need) != hipSuccess) {
        cleanup();
        return false;
      }
      e.p = p;
      e.bytes = need;
    }
    ws = e.p;
  }

  const float alpha = 1.0f, beta = 0.0f;
  const hipblasStatus_t st = hipblasLtMatmul(lt.h, desc, &alpha, W, lw, A, la,
                                           &beta, C, lc, C, lc, &it->second, ws,
                                           need, s);
  cleanup();
  if (st != HIPBLAS_STATUS_SUCCESS) {
    cache.erase(key);  // a failing cached algo is poison for the next call
    ws_sizes.erase(key);  // paired: a stale size would pin the re-swept algo
    return false;
  }
  return true;
}
#endif  // VLLM_CPP_HAS_HIPBLASLT

// ─── step 3: f16 @ f16 -> f16|f32 GEMM through hipblasGemmEx ─────────────────
//
// Row-major C[M,N] = A[M,K] @ W[K,N] is BLAS's column-major
// C_cm(N,M) = W_cm(N,K) @ A_cm(K,M): both inputs are already column-major in
// memory (a row-major [X,Y] buffer IS a column-major [Y,X]), so OP_N/OP_N.
// `ldc` honours a column slice's parent stride exactly as the CUDA arm's
// CublasLtHgemm honours c.stride[0] (cuda_matmul.cu:1336-1382).
void HgemmRowMajor(int device, hipStream_t s, void* C, const void* A, const void* W,
                   int64_t m, int64_t n, int64_t k, int64_t ldc, bool c_f32) {
  hipblasHandle_t h = ProductGetBlasHandle(device, s);
  if (h == nullptr) throw std::runtime_error("vt rocm exl3 recon: no hipBLAS handle");
  const float alpha = 1.0f, beta = 0.0f;
  const hipDataType c_type = c_f32 ? HIP_R_32F : HIP_R_16F;
  CheckBlas(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(n),
                          static_cast<int>(m), static_cast<int>(k), &alpha, W, HIP_R_16F,
                          static_cast<int>(n), A, HIP_R_16F, static_cast<int>(k), &beta,
                          C, c_type, static_cast<int>(ldc), HIPBLAS_COMPUTE_32F,
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx hgemm");
}

// ─── the persistent reconstruct scratch (QUANT-EXL3 W7's discipline) ─────────
//
// Grow-only, one buffer per stream, hipMallocAsync, never freed — the same
// contract rocm_exl3.hip's EnsureRawScratch carries and for the same reason:
// hipMalloc/hipFree/hipStreamSynchronize per call are illegal under hipGraph
// capture. Growth while the stream is capturing is refused BY NAME, mirroring
// the CUDA arm's Exl3PersistentReconScratch (cuda_exl3.cu:2824).
struct ReconScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

ReconScratch& ReconScratchFor(hipStream_t s) {
  static std::mutex mu;
  static std::unordered_map<hipStream_t, ReconScratch> pools;
  std::lock_guard<std::mutex> lk(mu);
  return pools[s];
}

bool StreamCapturing(hipStream_t s) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &status) != hipSuccess) return false;
  return status == hipStreamCaptureStatusActive;
}

half* EnsureReconScratch(size_t bytes, hipStream_t s) {
  ReconScratch& sc = ReconScratchFor(s);
  if (bytes > sc.bytes) {
    if (StreamCapturing(s)) {
      throw std::runtime_error(
          "vt rocm exl3 recon: persistent weight scratch must grow to " +
          std::to_string(bytes) +
          " bytes but the queue's stream is capturing; growth during capture would "
          "bake a pointer the pool then re-points. Run one eager step at this weight's "
          "shape first — the buffer's size depends on the weight, not on m.");
    }
    CheckR(hipMallocAsync(&sc.buf, bytes, s), "recon scratch grow");
    sc.bytes = bytes;
  }
  return static_cast<half*>(sc.buf);
}

}  // namespace

// The had_r_128 stages shared with rocm_exl3.hip.
void Exl3LaunchHadRocm(hipStream_t st, void* out, const void* in, const uint16_t* pre,
                       const uint16_t* post, float r_scale, int64_t rows, int64_t cols,
                       int in_dt, int out_dt);

void Exl3ReconstructGemmKernelRocm(Queue& q, Tensor& c, const Tensor& a,
                                   const Tensor& trellis, const Tensor& suh,
                                   const Tensor& svh, Tensor& a_had,
                                   Tensor& w_scratch, const Exl3GemmArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt rocm exl3 recon: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt rocm exl3 recon: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");
  // HALF-INTEGER rates (K+0.5, mul1) have no reconstruct arm (BACKEND-ROCM,
  // spec `## Owed`): ReconstructDecodeSliceK would decode the frac bitstream
  // through the integer-KA window arithmetic — silently wrong — so the frac
  // tensor refuses here by name and routes to the dot arm / Exl3GemmK.
  VT_CHECK(!args.half,
           "vt rocm exl3 recon: half-integer rates (K+0.5, mul1) have no reconstruct "
           "arm; this is an unimplemented arm, not an invalid tensor "
           "(BACKEND-ROCM frac rates, ## Owed).");
  // f16 and bf16 are the same width: a bf16 tensor here would be silently
  // reinterpreted by the f16 Hadamards and the f16 hipBLAS GEMM.
  VT_CHECK(a.dtype == DType::kF16,
           "vt rocm exl3 recon: a must be f16; got dtype code " +
               std::to_string(static_cast<int>(a.dtype)));
  VT_CHECK(c.dtype == DType::kF16 || c.dtype == DType::kF32,
           "vt rocm exl3 recon: c must be f16 or f32; got dtype code " +
               std::to_string(static_cast<int>(c.dtype)));
  // This backend's scalar decoder decodes EVERY width upstream defines — the
  // CUDA arm's {(3,0),(3,1),(6,0),(3,2),(4,2),(5,2),(6,2)} table is its own
  // instantiation limit, not the format's, so there is nothing to refuse here.
  const int64_t size_m = a.shape[0];
  const int64_t size_k = a.shape[1];
  const int64_t size_n = c.shape[1];
  if (size_m == 0 || size_k == 0 || size_n == 0) return;

  hipStream_t st = AsStreamR(q);

  // exl3.py:186-190. Input Hadamard — this arm is ALWAYS the unfused path (the
  // file header names why), so the transform always runs, into the caller's
  // a_had scratch.
  Exl3LaunchHadRocm(st, a_had.data, a.data, suh.Ptr<uint16_t>(), nullptr, kInvSqrt128R,
                    size_m, size_k, /*in_dt=*/1, /*out_dt=*/1);

  // QUANT-EXL3 W7: an EMPTY w_scratch selects this stream's persistent buffer,
  // resolved before any launch so a refused growth enqueues nothing.
  half* scratch_w = nullptr;
  const int64_t slice_n = ReconSliceN();
  int64_t slice_cols = size_n < slice_n ? size_n : slice_n;
  Tensor persistent_scratch;
  const Tensor* scratch = &w_scratch;
  if (w_scratch.data == nullptr) {
    persistent_scratch = vt::Tensor::Contiguous(
        EnsureReconScratch(static_cast<size_t>(size_k * slice_cols) * sizeof(half), st),
        DType::kF16, q.device, {size_k, slice_cols});
    scratch = &persistent_scratch;
  }
  scratch_w = scratch->Ptr<half>();

  const int64_t tiles_n_packed = trellis.shape[1];  // n/16
  const int64_t n_packed = tiles_n_packed * 16;

  // exl3.py:192-211. Reconstruct + GEMM, in ReconSliceN()-column slices when
  // the weight is wider than the scratch bound.
  for (int64_t n_start = 0; n_start < size_n; n_start += slice_n) {
    const int64_t n_slice =
        (size_n - n_start) < slice_n ? (size_n - n_start) : slice_n;

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n_slice + 15) / 16),
              static_cast<unsigned>((size_k + 15) / 16));
    ReconstructDecodeSliceK<<<grid, block, 0, st>>>(
        scratch_w, trellis.Ptr<uint16_t>(), size_k, n_slice, n_packed, args.bits,
        args.codebook, n_start / 16);
    CheckR(hipGetLastError(), "reconstruct decode");

    // GEMM into the matching column slice of c. A slice views the parent's row
    // stride, so ldc is c.stride[0], not n_slice.
    void* c_slice = static_cast<char*>(c.data) +
                    static_cast<size_t>(n_start) * (c.dtype == DType::kF32 ? 4 : 2);
#ifdef VLLM_CPP_HAS_HIPBLASLT
    // The Lt arm is an A/B lever, not a dispatch override: a decline (no
    // handle, heuristic miss, capture-time workspace growth) falls through to
    // the same GemmEx call the arm replaces.
    if (!ReconLtDisabled() &&
        HgemmLtRowMajor(st, c_slice, a_had.data, scratch_w,
                        static_cast<int>(size_m), static_cast<int>(n_slice),
                        static_cast<int>(size_k), c.stride[0],
                        c.dtype == DType::kF32))
      continue;
#endif
    HgemmRowMajor(q.device.index, st, c_slice, a_had.data, scratch_w, size_m, n_slice,
                  size_k, c.stride[0], c.dtype == DType::kF32);
  }

  // exl3.py:213-214. Output Hadamard, in place on c — safe because each
  // 128-block is independent (HadK documents the aliasing).
  const int c_dt = c.dtype == DType::kF32 ? 0 : 1;
  Exl3LaunchHadRocm(st, c.data, c.data, nullptr, svh.Ptr<uint16_t>(), kInvSqrt128R,
                    size_m, size_n, c_dt, c_dt);
}

}  // namespace vt::rocm
