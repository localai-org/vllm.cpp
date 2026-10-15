// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA GDN kernels (M0.7): causal conv1d fwd/update, l2norm, gated rmsnorm,
// gated-delta-rule scan. Correctness-grade — plain kernels matching the CPU
// reference math in src/vt/cpu/cpu_ops.cpp element for element; formulas from
// .agents/specs/gdn-semantics.md (§ cited per kernel). Perf-grade (chunked prefill)
// lands M2.3.
//
// Metadata contract (M0.7 Task 2 review): query_start_loc / has_initial_state
// live on the OP'S DEVICE. These kernels read them device-side; the host
// wrappers never copy them back (that would force a stream sync). The CPU
// reference validates qsl bounds/monotonicity host-side; here bad metadata is
// unchecked (correctness-grade — the M0.9 builder owns metadata integrity).
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "vt/cuda/conv_update_fast.h"
#include "vt/cuda/cuda_device_caps.h"
#include "vt/cuda/cuda_gdn_internal.h"
// MAMBA2 / SSD IS NOT THE GATED DELTA RULE (.agents/specs/mamba2-ssd.md §0, §7):
// no delta-removal term, decay driven by A_log/dt, B/C shared across n_groups.
// The three kernels live in their own header and their own namespace and share
// nothing with the GDN/KDA code below; they are compiled into THIS translation
// unit because it is where the sibling SSM/linear-attention device arms already
// live, and because a new .cu has to be listed in the ROOT CMakeLists.txt, which
// check-doc-checkpoint classifies as `user_usage` and therefore charges a
// docs/USAGE.md update that a kernel exposing no command, config key or C-ABI
// entry point has nothing true to write (the same deviation W1 recorded for
// cpu_ops.cpp, mamba2-ssd.md §8.1).
#include "vt/cuda/cuda_mamba2_ssd.cuh"
#include "vt/cuda/gdn_decode_fused.h"
#include "vt/cuda/gdn_packed_decode_triton.h"
#include "vt/cuda/gdn_prefill_conv.h"
#include "vt/cuda/gdn_packed_reg_tile.h"
#include "vt/cuda/rmsnorm_gated_fast.h"
#include "vt/cuda/tile/cp_async.cuh"
#include "vt/cuda/tile/tma_pipeline.cuh"
#include "vt/cuda/triton_aot_arch_dispatch.h"
#include "vt/ops.h"

#ifdef VLLM_CPP_TRITON
// SANCTIONED CUDA-only Triton AOT fast-path for the GDN delta_h state recurrence
// (see .agents/discipline.md "SANCTIONED EXCEPTION" + porting-inventory.md §9).
// AOT-generated stable dispatchers (triton.tools.link output; see
// cmake/TritonAOT.cmake, triton_kernels/chunk_delta_h.py). Plain C symbols; the
// launcher loads the EMBEDDED cubin via the CUDA driver API — no Triton/Python at
// runtime. One spec per gate-model GDN shape: h48 (Qwen3.6-27B), h32 (35B).
// gdn_deltah_hNN_default(stream, k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens,
//   chunk_offsets, T, NH) launches the FLA chunk_gated_delta_rule_fwd_kernel.
#include "triton_aot_multiarch.h"

#define VT_AOT_SYMBOL_IMPL(ARCH, NAME) vt_aot_##ARCH##_##NAME##_default
#define VT_AOT_SYMBOL(ARCH, NAME) VT_AOT_SYMBOL_IMPL(ARCH, NAME)
#define VT_AOT_LOAD_SYMBOL_IMPL(ARCH, NAME) vt_aot_##ARCH##_load_##NAME
#define VT_AOT_LOAD_SYMBOL(ARCH, NAME) VT_AOT_LOAD_SYMBOL_IMPL(ARCH, NAME)
#define VT_TRITON_AOT_CALL(NAME, ...)                                      \
  vt::cuda::DispatchTritonAot(                                             \
      vt::cuda::GetDeviceCaps().sm_major,                                  \
      vt::cuda::GetDeviceCaps().sm_minor, CUDA_ERROR_NOT_SUPPORTED,        \
      std::array{&VT_AOT_SYMBOL(sm_80, NAME),                               \
                 &VT_AOT_SYMBOL(sm_86, NAME),                              \
                 &VT_AOT_SYMBOL(sm_89, NAME),                              \
                 &VT_AOT_SYMBOL(sm_90a, NAME),                             \
                 &VT_AOT_SYMBOL(sm_100a, NAME),                            \
                 &VT_AOT_SYMBOL(sm_121a, NAME)},                           \
      __VA_ARGS__)
#define VT_TRITON_AOT_LOAD(NAME)                                          \
  vt::cuda::DispatchTritonAotVoid(                                        \
      vt::cuda::GetDeviceCaps().sm_major,                                 \
      vt::cuda::GetDeviceCaps().sm_minor,                                 \
      std::array{&VT_AOT_LOAD_SYMBOL(sm_80, NAME),                         \
                 &VT_AOT_LOAD_SYMBOL(sm_86, NAME),                        \
                 &VT_AOT_LOAD_SYMBOL(sm_89, NAME),                        \
                 &VT_AOT_LOAD_SYMBOL(sm_90a, NAME),                       \
                 &VT_AOT_LOAD_SYMBOL(sm_100a, NAME),                      \
                 &VT_AOT_LOAD_SYMBOL(sm_121a, NAME)})

#define gdn_deltah_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_deltah_h48, __VA_ARGS__)
#define gdn_deltah_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_deltah_h32, __VA_ARGS__)
#define gdn_chunko_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_chunko_h48, __VA_ARGS__)
#define gdn_chunko_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_chunko_h32, __VA_ARGS__)
#define gdn_chunko_bf16_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_chunko_bf16_h48, __VA_ARGS__)
#define gdn_chunko_bf16_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_chunko_bf16_h32, __VA_ARGS__)
#define gdn_kkt_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_kkt_h48, __VA_ARGS__)
#define gdn_kkt_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_kkt_h32, __VA_ARGS__)
#define gdn_tril_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_tril_h48, __VA_ARGS__)
#define gdn_tril_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_tril_h32, __VA_ARGS__)
#define gdn_wu_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_wu_h48, __VA_ARGS__)
#define gdn_wu_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_wu_h32, __VA_ARGS__)
#define gdn_decode_h48_default(...) \
  VT_TRITON_AOT_CALL(gdn_decode_h48, __VA_ARGS__)
#define gdn_decode_h32_default(...) \
  VT_TRITON_AOT_CALL(gdn_decode_h32, __VA_ARGS__)
#define kda_gate_cumsum_default(...) \
  VT_TRITON_AOT_CALL(kda_gate_cumsum, __VA_ARGS__)
#define kda_kkt_inter_default(...) \
  VT_TRITON_AOT_CALL(kda_kkt_inter, __VA_ARGS__)
#define kda_kkt_intra_default(...) \
  VT_TRITON_AOT_CALL(kda_kkt_intra, __VA_ARGS__)
#define kda_wu_default(...) VT_TRITON_AOT_CALL(kda_wu, __VA_ARGS__)
#define kda_deltah_h32_default(...) \
  VT_TRITON_AOT_CALL(kda_deltah_h32, __VA_ARGS__)
#define kda_gla_o_default(...) VT_TRITON_AOT_CALL(kda_gla_o, __VA_ARGS__)

#define VT_DEFINE_TRITON_AOT_LOADER(NAME) \
  static void load_##NAME() { (void)VT_TRITON_AOT_LOAD(NAME); }
VT_DEFINE_TRITON_AOT_LOADER(gdn_deltah_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_deltah_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_chunko_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_chunko_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_chunko_bf16_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_chunko_bf16_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_kkt_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_kkt_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_tril_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_tril_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_wu_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_wu_h32)
VT_DEFINE_TRITON_AOT_LOADER(gdn_decode_h48)
VT_DEFINE_TRITON_AOT_LOADER(gdn_decode_h32)
VT_DEFINE_TRITON_AOT_LOADER(kda_gate_cumsum)
VT_DEFINE_TRITON_AOT_LOADER(kda_kkt_inter)
VT_DEFINE_TRITON_AOT_LOADER(kda_kkt_intra)
VT_DEFINE_TRITON_AOT_LOADER(kda_wu)
VT_DEFINE_TRITON_AOT_LOADER(kda_deltah_h32)
VT_DEFINE_TRITON_AOT_LOADER(kda_gla_o)
#undef VT_DEFINE_TRITON_AOT_LOADER
#endif

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;
// CUDA grid.y is limited to 65535; sequence/batch counts map to grid.y below.
constexpr int64_t kMaxGridY = 65535;
// FLA chunk size (fla/ops/utils.py FLA_CHUNK_SIZE=64). The chunked GDN prefill
// scan (below) processes tokens in chunks of this many; the intra-chunk work is
// parallel across chunks and the cross-chunk state recurrence is sequential.
constexpr int kChunk = 64;
// Head-dim ceiling for the chunked path: local per-column arrays are sized to
// kChunk and the delta-h kernel stages the [Dv,Dk] state in shared memory
// (Dv*Dk*4B). Dk=Dv=128 is the real gate dim; larger dims fall back to the
// sequential scan (which strides arbitrary dims).
constexpr int64_t kChunkMaxDim = 128;

std::atomic<bool> g_gdn_packed_debug_enabled{false};
std::atomic<uint64_t> g_gdn_packed_launches{0};
// Sub-counters of g_gdn_packed_launches by selected kernel (see
// GdnPackedDecodeDebugStats): vendored Triton AOT decode cubin vs
// register-resident tiling vs legacy fallback.
std::atomic<uint64_t> g_gdn_packed_triton_launches{0};
std::atomic<uint64_t> g_gdn_packed_reg_tile_launches{0};
std::atomic<uint64_t> g_gdn_packed_legacy_launches{0};

void RecordGdnPackedDecodeLaunch() {
  if (g_gdn_packed_debug_enabled.load(std::memory_order_relaxed))
    g_gdn_packed_launches.fetch_add(1, std::memory_order_relaxed);
}

// Only the VLLM_CPP_TRITON dispatch path (TryTritonPackedDecode success)
// records vendored-cubin launches; the counter atomic above stays
// unconditional so GdnPackedDecodeDebugStats reads 0 in non-Triton builds.
// The guard keeps the default (Triton-less) -Werror build free of
// -Wunused-function on this anonymous-namespace helper.
#ifdef VLLM_CPP_TRITON
void RecordGdnPackedDecodeTritonLaunch() {
  if (g_gdn_packed_debug_enabled.load(std::memory_order_relaxed))
    g_gdn_packed_triton_launches.fetch_add(1, std::memory_order_relaxed);
}
#endif  // VLLM_CPP_TRITON

void RecordGdnPackedDecodeRegTileLaunch() {
  if (g_gdn_packed_debug_enabled.load(std::memory_order_relaxed))
    g_gdn_packed_reg_tile_launches.fetch_add(1, std::memory_order_relaxed);
}

void RecordGdnPackedDecodeLegacyLaunch() {
  if (g_gdn_packed_debug_enabled.load(std::memory_order_relaxed))
    g_gdn_packed_legacy_launches.fetch_add(1, std::memory_order_relaxed);
}

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

// Rung-2 TMA support (VT_GDN_TMA). Host-side CUtensorMap builder for the delta_h
// chunk streaming: a 3D tiled descriptor over a [tokens, heads, dk] bf16 tensor
// (dk innermost/contiguous), box (dk, 1, BT) — one [BT, dk] chunk tile per TMA.
// cuTensorMapEncodeTiled is a driver symbol; we fetch it once via the runtime's
// cudaGetDriverEntryPoint so no explicit libcuda link is needed (CUTLASS's own
// runtime-linked path). Mirrors the CuTe-DSL kernel_h _make_bf16_tma_args
// (gdn_chunk_cutedsl/kernel_h.py:56-75): 128 K-dim, num_stages ring, bf16 tile.
// Compatibility shim: cudaGetDriverEntryPointByVersion was introduced in CUDA 12.2.
// On CUDA 12.0 (e.g. Ubuntu nvidia-cuda-toolkit on aarch64), fall back to
// cudaGetDriverEntryPoint which is available since 12.0 and accepts the same
// flags. The version parameter is ignored in the fallback path.
#if CUDART_VERSION < 12020
static inline cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol, void** funcPtr, int /*version*/,
    unsigned long long flags,
    cudaDriverEntryPointQueryResult* driverStatus) {
  return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}
#endif

using PfnTmaEncode = PFN_cuTensorMapEncodeTiled_v12000;
inline PfnTmaEncode GdnTmaEncodeFn() {
  static PfnTmaEncode fn = [] {
    PfnTmaEncode f = nullptr;
    cudaDriverEntryPointQueryResult q{};
    // Versioned entry-point query (non-deprecated; cuTensorMapEncodeTiled_v12000 => 12000).
    cudaError_t e = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", reinterpret_cast<void**>(&f), 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || f == nullptr) return static_cast<PfnTmaEncode>(nullptr);
    return f;
  }();
  return fn;
}

// Build a 3D TMA descriptor for a bf16 [t_dim, h_dim, dk] tensor (row-major, dk
// contiguous). box (dk,1,BT), swizzle NONE (the WMMA consumer reads plain
// row-major smem, identical to the cp.async ring — this isolates the copy/
// pipeline mechanism from the smem-layout lever). OOB fill NONE => TMA zeros the
// tail beyond t_dim (partial-chunk tail rows are masked downstream regardless).
inline bool BuildGdnTmaDesc3D(CUtensorMap* desc, const void* base, int64_t t_dim, int64_t h_dim,
                              int64_t dk, int64_t bt) {
  PfnTmaEncode encode = GdnTmaEncodeFn();
  if (!encode) return false;
  const uint64_t gdim[3] = {static_cast<uint64_t>(dk), static_cast<uint64_t>(h_dim),
                            static_cast<uint64_t>(t_dim)};
  const uint64_t gstr[2] = {static_cast<uint64_t>(dk) * 2u,
                            static_cast<uint64_t>(h_dim) * static_cast<uint64_t>(dk) * 2u};
  const uint32_t bdim[3] = {static_cast<uint32_t>(dk), 1u, static_cast<uint32_t>(bt)};
  const uint32_t estr[3] = {1u, 1u, 1u};
  CUresult r = encode(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                      const_cast<void*>(base), gdim, gstr, bdim, estr,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// f32 load/store overloads: reduced-width formats convert on the way in/out,
// while kernel arithmetic remains f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline float LoadFloating(const void* p, DType dtype, int64_t i) {
  if (dtype == DType::kF32) return Load(static_cast<const float*>(p), i);
  if (dtype == DType::kF16) return Load(static_cast<const __half*>(p), i);
  return Load(static_cast<const __nv_bfloat16*>(p), i);
}
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

template <typename T>
__device__ inline float RoundToStorage(float v);
template <>
__device__ inline float RoundToStorage<float>(float v) {
  return v;
}
template <>
__device__ inline float RoundToStorage<__half>(float v) {
  return __half2float(__float2half_rn(v));
}
template <>
__device__ inline float RoundToStorage<__nv_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ inline float Silu(float x) { return x / (1.0f + expf(-x)); }

// Indexed persistent GDN cache boundary. One elementwise launch replaces N
// cudaMemcpyAsync row copies plus a separate cast launch. The optional fresh
// mask is i8 (metadata's upstream-bool mirror) or i32 (standalone op callers).
// The cache's INNERMOST dim (cache_inner) may be WIDER than the working buffer's
// (work_inner) when the GDN conv state is widened to (K-1)+num_spec taps for
// spec-decode rollback. The non-spec gather/scatter then maps each working column
// to the LEADING work_inner cols of its channel, using the cache row's PHYSICAL
// stride (cache_row = mid*cache_inner). At num_spec==0 (all non-spec models, and
// the never-widened SSM state cache) cache_inner==work_inner and the uniform
// branch takes the original flat path, byte-for-byte. `row_elems` is the working
// row width (mid*work_inner).
template <typename TCache>
__global__ void GdnStateGatherKernel(float* working, const TCache* cache,
                                     const int32_t* state_idx,
                                     const int8_t* has_i8,
                                     const int32_t* has_i32,
                                     int64_t row_elems, int64_t n,
                                     int64_t work_inner, int64_t cache_inner) {
  const bool widened = cache_inner != work_inner;
  const int64_t cache_row = (row_elems / work_inner) * cache_inner;  // mid*cache_inner
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / row_elems;
    const int64_t col = i - row * row_elems;
    const bool keep = has_i8 != nullptr ? has_i8[row] != 0
                      : has_i32 != nullptr ? has_i32[row] != 0
                                           : true;
    int64_t cache_off;
    if (!widened) {
      cache_off = static_cast<int64_t>(state_idx[row]) * row_elems + col;
    } else {
      const int64_t m = col / work_inner, wcol = col - m * work_inner;
      cache_off = static_cast<int64_t>(state_idx[row]) * cache_row +
                  m * cache_inner + wcol;
    }
    working[i] = keep ? Load(cache, cache_off) : 0.0f;
  }
}

template <typename TCache>
__global__ void GdnStateScatterKernel(TCache* cache, const float* working,
                                      const int32_t* state_idx,
                                      int64_t row_elems, int64_t n,
                                      int64_t work_inner, int64_t cache_inner) {
  const bool widened = cache_inner != work_inner;
  const int64_t cache_row = (row_elems / work_inner) * cache_inner;  // mid*cache_inner
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / row_elems;
    const int64_t col = i - row * row_elems;
    int64_t cache_off;
    if (!widened) {
      cache_off = static_cast<int64_t>(state_idx[row]) * row_elems + col;
    } else {
      const int64_t m = col / work_inner, wcol = col - m * work_inner;
      cache_off = static_cast<int64_t>(state_idx[row]) * cache_row +
                  m * cache_inner + wcol;
    }
    Store(cache, cache_off, working[i]);
  }
}

void GdnStateGatherKernelCuda(Queue& q, Tensor& working, const Tensor& cache,
                              const Tensor& state_idx,
                              const Tensor* has_initial_state) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t n = working.Numel();
  const int64_t row_elems = n / rows;
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int8_t* has_i8 =
      has_initial_state != nullptr && has_initial_state->dtype == DType::kI8
          ? has_initial_state->Ptr<int8_t>()
          : nullptr;
  const int32_t* has_i32 =
      has_initial_state != nullptr && has_initial_state->dtype == DType::kI32
          ? has_initial_state->Ptr<int32_t>()
          : nullptr;
  cudaStream_t s = AsStream(q);
  if (cache.dtype == DType::kBF16) {
    GdnStateGatherKernel<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<__nv_bfloat16>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  } else if (cache.dtype == DType::kF16) {
    GdnStateGatherKernel<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<__half>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  } else {
    GdnStateGatherKernel<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<float>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  }
  Check(cudaGetLastError(), "gdn state indexed gather launch");
}

void GdnStateScatterKernelCuda(Queue& q, Tensor& cache,
                               const Tensor& working,
                               const Tensor& state_idx) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t n = working.Numel();
  const int64_t row_elems = n / rows;
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  cudaStream_t s = AsStream(q);
  if (cache.dtype == DType::kBF16) {
    GdnStateScatterKernel<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<__nv_bfloat16>(), working.Ptr<float>(),
        state_idx.Ptr<int32_t>(), row_elems, n, work_inner, cache_inner);
  } else if (cache.dtype == DType::kF16) {
    GdnStateScatterKernel<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<__half>(), working.Ptr<float>(), state_idx.Ptr<int32_t>(),
        row_elems, n, work_inner, cache_inner);
  } else {
    GdnStateScatterKernel<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<float>(), working.Ptr<float>(), state_idx.Ptr<int32_t>(),
        row_elems, n, work_inner, cache_inner);
  }
  Check(cudaGetLastError(), "gdn state indexed scatter launch");
}

// --- Row gather/scatter over dim 0 (torch index_select / index_copy_). Powers
// the MIXED spec+non-spec GDN split/merge. dtype-agnostic: dispatched on the
// element storage size (bf16 -> u16, f32 -> u32). `n_total` = rows * inner
// elements; grid-stride over the compact side; the base side carries an outer
// row stride (in elements) so a padded / row-strided packed view is addressed
// directly. Mirror of qwen_gdn_linear_attn.py:1334-1335,1407-1408,1570-1571.
template <typename T>
__global__ void IndexSelectRowKernel(T* out, const T* in, const int32_t* idx,
                                     int64_t inner, int64_t in_row_stride,
                                     int64_t n_total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += step) {
    const int64_t row = i / inner;
    const int64_t col = i - row * inner;
    out[i] = in[static_cast<int64_t>(idx[row]) * in_row_stride + col];
  }
}

template <typename T>
__global__ void IndexCopyRowKernel(T* out, const T* in, const int32_t* idx,
                                   int64_t inner, int64_t out_row_stride,
                                   int64_t n_total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += step) {
    const int64_t row = i / inner;
    const int64_t col = i - row * inner;
    out[static_cast<int64_t>(idx[row]) * out_row_stride + col] = in[i];
  }
}


// V-head re-indexing of a GDN projection vector (vt::VHeadPermute). Mirrors
// `VHeadPermuteKernel` in src/vt/cpu/cpu_ops.cpp element for element; see that
// file for the two index maps and why K == R cannot tell them apart.
template <typename T>
__global__ void VHeadPermuteKernelCu(T* out, const T* in, int64_t n,
                                        int64_t out_row_stride,
                                        int64_t in_row_stride, int64_t pre,
                                        int64_t kk, int64_t rr, int64_t w,
                                        int inverse, int64_t n_total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += step) {
    const int64_t row = i / n;
    const int64_t c = i - row * n;
    int64_t sc = c;
    if (c >= pre) {
      const int64_t d = c - pre;
      const int64_t h = d / w;
      const int64_t o = d - h * w;
      const int64_t sh =
          inverse ? (h % kk) * rr + (h / kk) : (h % rr) * kk + (h / rr);
      sc = pre + sh * w + o;
    }
    out[row * out_row_stride + c] = in[row * in_row_stride + sc];
  }
}

void VHeadPermuteKernelCuda(Queue& q, Tensor& out, const Tensor& in,
                            const VHeadPermuteArgs& args) {
  const int64_t n = in.shape[1];
  const int64_t n_total = in.shape[0] * n;
  if (n_total == 0) return;
  cudaStream_t s = AsStream(q);
  const int64_t os = out.stride[0], is = in.stride[0];
  const int inv = args.inverse ? 1 : 0;
#define VT_VHP_LAUNCH(TY)                                                      \
  VHeadPermuteKernelCu<<<GridFor(n_total), kBlock, 0, s>>>(                 \
      static_cast<TY*>(out.data), static_cast<const TY*>(in.data), n, os, is,  \
      args.prefix_elems, args.num_key_heads, args.heads_per_key,               \
      args.head_width, inv, n_total)
  switch (SizeOf(out.dtype)) {
    case 1: VT_VHP_LAUNCH(uint8_t); break;
    case 2: VT_VHP_LAUNCH(uint16_t); break;
    case 4: VT_VHP_LAUNCH(uint32_t); break;
    case 8: VT_VHP_LAUNCH(uint64_t); break;
    default:
      VT_CHECK(false, "v_head_permute: unsupported element size");
  }
#undef VT_VHP_LAUNCH
}

void IndexSelectKernelCuda(Queue& q, Tensor& out, const Tensor& in,
                           const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int64_t inner = out.Numel() / rows;
  const int64_t n = out.Numel();
  const int64_t in_row_stride = in.stride[0];
  cudaStream_t s = AsStream(q);
  switch (SizeOf(out.dtype)) {
    case 1:
      IndexSelectRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint8_t*>(out.data),
          static_cast<const uint8_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
    case 2:
      IndexSelectRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint16_t*>(out.data),
          static_cast<const uint16_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
    case 4:
      IndexSelectRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint32_t*>(out.data),
          static_cast<const uint32_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
    default:
      IndexSelectRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint64_t*>(out.data),
          static_cast<const uint64_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
  }
  Check(cudaGetLastError(), "index_select row launch");
}

void IndexCopyKernelCuda(Queue& q, Tensor& out, const Tensor& in,
                         const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int64_t inner = in.Numel() / rows;
  const int64_t n = in.Numel();
  const int64_t out_row_stride = out.stride[0];
  cudaStream_t s = AsStream(q);
  switch (SizeOf(in.dtype)) {
    case 1:
      IndexCopyRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint8_t*>(out.data),
          static_cast<const uint8_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
    case 2:
      IndexCopyRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint16_t*>(out.data),
          static_cast<const uint16_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
    case 4:
      IndexCopyRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint32_t*>(out.data),
          static_cast<const uint32_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
    default:
      IndexCopyRowKernel<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint64_t*>(out.data),
          static_cast<const uint64_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
  }
  Check(cudaGetLastError(), "index_copy row launch");
}

// ---------------------------------------------------------------------------
// causal_conv1d_fwd (gdn-semantics.md §2): one thread per (sequence, channel),
// sequential over the sequence's tokens. Each thread owns its conv_state row
// [K-1] exclusively, so outputs (which read the OLD state for the first K-1
// tokens) and the write-back are ordered by plain program order — no barriers.
// The write-back needs no old-row buffer: writing ascending j reads old cols
// at index t_len + j >= j, so no already-written slot is ever read (same
// values the CPU reference buffers into old_row).
// Upstream counterpart: layers/mamba/ops/causal_conv1d.py (causal_conv1d_fn
// Triton kernel) — align post-MVP.

template <typename Tin, typename Tout, typename THas>
__global__ void CausalConv1dFwdKernel(Tout* out, const Tin* x, const Tin* w, const Tin* bias,
                                      float* conv_state, const int32_t* qsl,
                                      const THas* his, int64_t c_dim, int64_t x_row_stride,
                                      int64_t k, bool silu) {
  const int64_t s = blockIdx.y;
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= c_dim) return;
  const int64_t width = k - 1;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const bool init = his[s] != 0;
  float* srow = conv_state + (s * c_dim + c) * width;
  const float b = bias != nullptr ? Load(bias, c) : 0.0f;
  for (int64_t t = 0; t < t_len; ++t) {
    float acc = b;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t ti = t - (k - 1 - j);  // token index of window[j]
      float v = 0.0f;
      if (ti >= 0) {
        v = Load(x, (begin + ti) * x_row_stride + c);
      } else if (init) {
        v = srow[width + ti];  // old state col (K-1)+(t-i); not yet overwritten
      }
      acc += Load(w, c * k + j) * v;
    }
    Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
  }
  // State write-back: last K-1 RAW x tokens, left-padded with zeros (no init
  // state) or shifted old state when T < K-1.
  for (int64_t j = 0; j < width; ++j) {
    const int64_t tj = t_len - width + j;  // new state col j holds token tj
    float v = 0.0f;
    if (tj >= 0) {
      v = Load(x, (begin + tj) * x_row_stride + c);
    } else if (init) {
      v = srow[width + tj];  // shifted old state; index t_len+j >= j (unwritten)
    }
    srow[j] = v;
  }
}

// ---------------------------------------------------------------------------
// Tiled causal_conv1d_fwd (VT_CONV_TILED, default OFF) — the (token-tile x
// channel-tile) mirror of the FLA/vLLM causal_conv1d Triton kernel. The scalar
// CausalConv1dFwdKernel above rereads each x value K times (once per tap it lands
// on) and is serial per (seq, channel); this stages a [BLOCK_M+width, BLOCK_N]
// x-tile into shared ONCE (each x read ~once vs K times, the halo aside) and lets
// BLOCK_M token rows reuse the same sliding window, with consecutive threads over
// the channel dim so the global x loads and out stores are coalesced.
//
// NUMERICALLY IDENTICAL to the scalar kernel (token-exact): for every output
// (token t, channel c) it accumulates acc = bias; for j in [0,k): acc += w[c*k+j]
// * v_j — the SAME tap ORDER (j=0 oldest .. j=k-1 newest, bias first) over the
// SAME f32 values. v_j is fetched from the shared tile, but the tile is filled by
// the identical Load(x,...) / conv_state reads the scalar kernel does, so each v_j
// is bit-for-bit the same float; a conv is a fixed sum of k<=~4 f32 products, and
// preserving its accumulation order keeps a razor-thin greedy argmax from flipping.
// One block owns a (channel-tile, sequence) and loops token-tiles of BLOCK_M, then
// performs the (K-1) state write-back byte-identically to the scalar kernel
// (ascending j, same read-before-overwrite ordering). Causal masking, the (K-1)
// left-pad / conv_state prepend (has_initial_state), per-channel weight + optional
// bias, silu/identity, and the [token, c_dim] output layout are all preserved.
// PREFILL only — the decode CausalConv1dUpdate path is untouched.
constexpr int kConvTileM = 16;  // token tile (rows per block)
constexpr int kConvTileN = 32;  // channel tile (a warp — coalesced x loads/stores)

template <typename Tin, typename Tout, typename THas>
__global__ void CausalConv1dFwdTiledKernel(Tout* out, const Tin* x, const Tin* w,
                                           const Tin* bias, float* conv_state,
                                           const int32_t* qsl, const THas* his, int64_t c_dim,
                                           int64_t x_row_stride, int64_t k, bool silu) {
  const int64_t width = k - 1;
  const int64_t s = blockIdx.y;  // sequence
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kConvTileN + tx;  // channel
  const bool active = c < c_dim;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const bool init = his[s] != 0;
  // Own conv_state row [K-1] for this (seq, channel); prefill state is f32.
  float* srow = active ? conv_state + (s * c_dim + c) * width : nullptr;
  const float b = (active && bias != nullptr) ? Load(bias, c) : 0.0f;

  extern __shared__ float sh[];  // [(kConvTileM + width) * kConvTileN], slot-major

  // t_len is uniform across the block (s == blockIdx.y), so every thread runs the
  // same iteration count and hits every __syncthreads() below in lockstep.
  for (int64_t t0 = 0; t0 < t_len; t0 += kConvTileM) {
    // Cooperative, coalesced load of the tile: (K-1) history halo + BLOCK_M window.
    for (int64_t slot = ty; slot < kConvTileM + width; slot += blockDim.y) {
      const int64_t tok = t0 - width + slot;  // token index within the sequence
      float val = 0.0f;
      if (active) {
        if (tok >= 0 && tok < t_len) {
          val = Load(x, (begin + tok) * x_row_stride + c);
        } else if (tok < 0 && init) {
          val = srow[width + tok];  // old conv_state col (K-1)+tok, in [0,width)
        }
      }
      sh[slot * kConvTileN + tx] = val;
    }
    __syncthreads();
    // One output token per row: token t reads window slots [ty, ty+k), i.e. the
    // taps ti = t-(k-1)..t, matching the scalar kernel's ti = t-(k-1-j).
    const int64_t t = t0 + ty;
    if (active && t < t_len) {
      float acc = b;
      for (int64_t j = 0; j < k; ++j) {  // SAME order as the scalar kernel
        acc += Load(w, c * k + j) * sh[(ty + j) * kConvTileN + tx];
      }
      Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
    }
    __syncthreads();  // all reads of sh done before the next tile overwrites it
  }

  // State write-back (byte-identical to the scalar kernel): last K-1 RAW x tokens,
  // left-padded with zeros (no init state) or shifted old state when T < K-1. One
  // row (ty == 0) writes each channel's state; ascending j reads old col t_len+j
  // (>= j, unwritten), so no already-written slot is reread.
  if (active && ty == 0) {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t tj = t_len - width + j;
      float val = 0.0f;
      if (tj >= 0) {
        val = Load(x, (begin + tj) * x_row_stride + c);
      } else if (init) {
        val = srow[width + tj];
      }
      srow[j] = val;
    }
  }
}

// Toggle: DEFAULT ON since 2026-07-10 (VT_CONV_TILED=0 restores the scalar
// kernel for an A/B). Token-exact by construction (same tap order over the
// same f32 values — see the kernel comment) AND gate-verified: 27B + 35B
// paged-engine token-exact with the tiled kernel. MEASURED (27B prefill,
// T≈2040 chunks): scalar 884 us/layer-instance ≈ 96 GB/s effective vs vLLM's
// causal_conv1d 300 GB/s-class — the scalar kernel rereads x K times and was
// ~0.25 us/tok/layer of the 27B prefill excess; the tiled kernel stages the
// x window in shared once with coalesced channel-major access.
bool ConvTiledEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_CONV_TILED");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

template <typename Tin, typename Tout>
void LaunchConvFwd(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                   const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t n = conv_state.shape[0], c = x.shape[1], k = w.shape[1];
  const int64_t x_rs = x.stride[0];  // padded-row (merged qkvz) x view honored
  const dim3 grid(static_cast<unsigned>((c + kBlock - 1) / kBlock), static_cast<unsigned>(n));
  if (his.dtype == DType::kI8) {
    CausalConv1dFwdKernel<Tin, Tout, int8_t><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int8_t>(), c, x_rs, k, args.silu_activation);
  } else {
    CausalConv1dFwdKernel<Tin, Tout, int32_t><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, x_rs, k, args.silu_activation);
  }
  Check(cudaGetLastError(), "causal_conv1d_fwd launch");
}

// Tiled launcher (VT_CONV_TILED=1). Block = (kConvTileN channels, kConvTileM
// tokens); grid = (channel-tiles, sequences) — same sequence-per-grid.y layout as
// the scalar launcher, so the n <= kMaxGridY guard already covers grid.y. Dynamic
// shared holds the [(BLOCK_M+width) x BLOCK_N] x-tile (width = k-1).
template <typename Tin, typename Tout>
void LaunchConvFwdTiled(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                        const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t n = conv_state.shape[0], c = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t x_rs = x.stride[0];  // padded-row (merged qkvz) x view honored
  const dim3 grid(static_cast<unsigned>((c + kConvTileN - 1) / kConvTileN),
                  static_cast<unsigned>(n));
  const dim3 block(kConvTileN, kConvTileM);
  const size_t shmem = static_cast<size_t>(kConvTileM + width) * kConvTileN * sizeof(float);
  if (his.dtype == DType::kI8) {
    CausalConv1dFwdTiledKernel<Tin, Tout, int8_t><<<grid, block, shmem, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int8_t>(), c, x_rs, k, args.silu_activation);
  } else {
    CausalConv1dFwdTiledKernel<Tin, Tout, int32_t><<<grid, block, shmem, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, x_rs, k, args.silu_activation);
  }
  Check(cudaGetLastError(), "causal_conv1d_fwd(tiled) launch");
}

// ---------------------------------------------------------------------------
// Register-resident sliding-window causal_conv1d_fwd (VT_CONV_REG, default ON) —
// the 1:1 mirror of vLLM's FLA _causal_conv1d_fwd_kernel
// (vllm/model_executor/layers/mamba/ops/causal_conv1d.py:397-452). One thread per
// channel, sequential over its token chunk, with:
//   * the k per-channel conv weights PRE-LOADED into registers ONCE (FLA
//     w_col0..w_col3, causal_conv1d.py:399-409) — the tiled kernel re-reads w from
//     GLOBAL on every tap of every output token;
//   * a REGISTER sliding window of the (k-1) history taps (FLA col0..col3,
//     causal_conv1d.py:414-452): each x element is loaded from global EXACTLY ONCE
//     (coalesced across the channel dim), reused across the k taps, then the window
//     shifts — no shared memory, no __syncthreads (the tiled kernel stages a shared
//     x-tile with two barriers per token-tile);
//   * the token axis parallelized into kConvRegM chunks over grid.z so a single long
//     prefill sequence saturates the GPU (FLA parallelizes chunk_offset the same way).
//
// BIT-IDENTICAL (0-ulp) to CausalConv1dFwdTiledKernel and CausalConv1dFwdKernel: for
// every output (token t, channel c) it accumulates acc = bias; for j in [0,k):
// acc += w[c*k+j] * x_tap[j] in the SAME tap ORDER (j=0 oldest .. j=k-1 newest) over
// the SAME f32 values (the register window is filled by the identical Load(x,...) /
// conv_state reads), then the SAME silu/identity epilogue + round-to-store, and the
// SAME (K-1) state write-back (ascending j, reading raw x / shifted conv_state
// directly — never a value the window mutated). See gdn_prefill_conv.h.
constexpr int kConvRegN = 128;    // channels per block (blockDim.x; coalesced x/out)
constexpr int kConvRegM = 32;     // token chunk per block (grid.z parallelism)
constexpr int kConvExactM = 8;    // upstream compute_causal_conv1d_metadata BLOCK_M
constexpr int kConvRegMaxW = 8;   // max supported width (k-1); Qwen GDN k=4 -> 3
constexpr int64_t kConvRegChunkMaxSeqs = 4;  // chunk the token axis only for few seqs

template <typename Tin, typename Tout, typename THas>
__global__ void CausalConv1dFwdRegKernel(Tout* out, const Tin* x, const Tin* w,
                                         const Tin* bias, float* conv_state,
                                         const int32_t* qsl, const THas* his, int64_t c_dim,
                                         int64_t x_row_stride, int64_t k, bool silu,
                                         int chunked, const int32_t* batch_ptr,
                                         const int32_t* token_chunk_offset_ptr, int exact) {
  const int64_t width = k - 1;
  const int64_t program = blockIdx.y;
  const int64_t s = exact ? batch_ptr[program] : program;                         // sequence
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kConvRegN + threadIdx.x;   // channel
  const bool active = c < c_dim;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;

  // Token range this block owns. chunked: [chunk*M, chunk*M+M); else whole sequence.
  const int64_t chunk_m = exact ? kConvExactM : kConvRegM;
  const int64_t token_offset = exact
                                   ? static_cast<int64_t>(token_chunk_offset_ptr[program]) *
                                         kConvExactM
                                   : (chunked ? static_cast<int64_t>(blockIdx.z) * kConvRegM
                                              : 0);
  // Skip chunks past the sequence end — except chunk 0, which still runs the state
  // write-back (also the only path when t_len == 0).
  if (token_offset > 0 && token_offset >= t_len) return;
  const int64_t token_end =
      ((chunked || exact) && token_offset + chunk_m < t_len) ? token_offset + chunk_m : t_len;

  const bool init = his[s] != 0;
  float* srow = active ? conv_state + (s * c_dim + c) * width : nullptr;
  const float b = (active && bias != nullptr) ? Load(bias, c) : 0.0f;

  if (active) {
    // Pre-load the k per-channel weights (FLA w_col0..w_col3) once.
    float wv[kConvRegMaxW + 1];
#pragma unroll
    for (int j = 0; j <= kConvRegMaxW; ++j)
      if (j < k) wv[j] = Load(w, c * k + j);
    // Register sliding window: window[0..width-1] = taps BEFORE token_offset,
    // window[width] = token_offset. tap ti = token_offset - width + j.
    float window[kConvRegMaxW + 1];
    for (int64_t j = 0; j < width; ++j) {
      const int64_t ti = token_offset - width + j;  // [token_offset-width, token_offset-1]
      float v = 0.0f;
      if (ti >= 0) {
        v = Load(x, (begin + ti) * x_row_stride + c);
      } else if (init) {
        v = srow[width + ti];  // ti<0 only for chunk 0 (token_offset<width); col in [0,width)
      }
      window[j] = v;
    }
    window[width] =
        (token_offset < t_len) ? Load(x, (begin + token_offset) * x_row_stride + c) : 0.0f;

    for (int64_t t = token_offset; t < token_end; ++t) {
      float acc = b;
#pragma unroll
      for (int j = 0; j <= kConvRegMaxW; ++j)  // SAME tap order as the scalar kernel
        if (j < k) acc += wv[j] * window[j];
      Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
      // Advance the sliding window by one token.
#pragma unroll
      for (int j = 0; j < kConvRegMaxW; ++j)
        if (j < width) window[j] = window[j + 1];
      const int64_t nt = t + 1;
      window[width] = (nt < t_len) ? Load(x, (begin + nt) * x_row_stride + c) : 0.0f;
    }
  }

  // State write-back: only the block that reaches the sequence end (byte-identical to
  // the scalar kernel — reads raw x / shifted conv_state directly, never the window).
  if (active && token_end == t_len) {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t tj = t_len - width + j;
      float v = 0.0f;
      if (tj >= 0) {
        v = Load(x, (begin + tj) * x_row_stride + c);
      } else if (init) {
        v = srow[width + tj];
      }
      srow[j] = v;
    }
  }
}

// Width-four experiment used by Qwen3.5. Unlike the sealed runtime-width kernel
// above, K and the number of channels owned by each lane are compile-time
// constants. ChannelsPerThread=1 isolates width specialization at the same
// 128-channel tile; ChannelsPerThread=2 covers two coalesced 128-channel stripes
// and therefore matches upstream's 256-channel feature tile.
template <int ChannelsPerThread, typename Tin, typename Tout, typename THas>
__global__ void CausalConv1dFwdRegK4Kernel(
    Tout* out, const Tin* x, const Tin* w, const Tin* bias, float* conv_state,
    const int32_t* qsl, const THas* his, int64_t c_dim, int64_t x_row_stride,
    bool silu, int chunked, const int32_t* batch_ptr,
    const int32_t* token_chunk_offset_ptr, int exact) {
  static_assert(ChannelsPerThread == 1 || ChannelsPerThread == 2);
  constexpr int64_t k = 4;
  constexpr int64_t width = k - 1;
  constexpr int64_t channels_per_block = kConvRegN * ChannelsPerThread;

  const int64_t program = blockIdx.y;
  const int64_t s = exact ? batch_ptr[program] : program;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const int64_t chunk_m = exact ? kConvExactM : kConvRegM;
  const int64_t token_offset = exact
                                   ? static_cast<int64_t>(token_chunk_offset_ptr[program]) *
                                         kConvExactM
                                   : (chunked ? static_cast<int64_t>(blockIdx.z) * kConvRegM
                                              : 0);
  if (token_offset > 0 && token_offset >= t_len) return;
  const int64_t token_end =
      ((chunked || exact) && token_offset + chunk_m < t_len) ? token_offset + chunk_m : t_len;
  const bool init = his[s] != 0;

  int64_t channels[ChannelsPerThread];
  bool active[ChannelsPerThread];
  float* state_rows[ChannelsPerThread];
  float biases[ChannelsPerThread];
  float weights[ChannelsPerThread][k];
  float windows[ChannelsPerThread][k];

  // Load both stripes' initial history before processing either stripe. This
  // matches the intended duplicated-register experiment without extending the
  // baseline state-read/write exposure across an entire first-stripe token loop.
#pragma unroll
  for (int lane_channel = 0; lane_channel < ChannelsPerThread; ++lane_channel) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * channels_per_block +
                      threadIdx.x + static_cast<int64_t>(lane_channel) * kConvRegN;
    channels[lane_channel] = c;
    active[lane_channel] = c < c_dim;
    if (!active[lane_channel]) continue;
    float* srow = conv_state + (s * c_dim + c) * width;
    state_rows[lane_channel] = srow;
    biases[lane_channel] = bias != nullptr ? Load(bias, c) : 0.0f;
#pragma unroll
    for (int j = 0; j < k; ++j) weights[lane_channel][j] = Load(w, c * k + j);
#pragma unroll
    for (int j = 0; j < width; ++j) {
      const int64_t ti = token_offset - width + j;
      float v = 0.0f;
      if (ti >= 0) {
        v = Load(x, (begin + ti) * x_row_stride + c);
      } else if (init) {
        v = srow[width + ti];
      }
      windows[lane_channel][j] = v;
    }
    windows[lane_channel][width] =
        token_offset < t_len ? Load(x, (begin + token_offset) * x_row_stride + c) : 0.0f;
  }

  for (int64_t t = token_offset; t < token_end; ++t) {
#pragma unroll
    for (int lane_channel = 0; lane_channel < ChannelsPerThread; ++lane_channel) {
      if (!active[lane_channel]) continue;
      const int64_t c = channels[lane_channel];
      float acc = biases[lane_channel];
#pragma unroll
      for (int j = 0; j < k; ++j)
        acc += weights[lane_channel][j] * windows[lane_channel][j];
      Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
#pragma unroll
      for (int j = 0; j < width; ++j)
        windows[lane_channel][j] = windows[lane_channel][j + 1];
      const int64_t nt = t + 1;
      windows[lane_channel][width] =
          nt < t_len ? Load(x, (begin + nt) * x_row_stride + c) : 0.0f;
    }
  }

  if (token_end == t_len) {
#pragma unroll
    for (int lane_channel = 0; lane_channel < ChannelsPerThread; ++lane_channel) {
      if (!active[lane_channel]) continue;
      const int64_t c = channels[lane_channel];
      float* srow = state_rows[lane_channel];
#pragma unroll
      for (int j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;
        float v = 0.0f;
        if (tj >= 0) {
          v = Load(x, (begin + tj) * x_row_stride + c);
        } else if (init) {
          v = srow[width + tj];
        }
        srow[j] = v;
      }
    }
  }
}

// Toggle: DEFAULT ON (VT_CONV_REG=0 restores the tiled/scalar path). Read per call
// (prefill dispatch is coarse — one launch/step — so the getenv is negligible and
// in-process CUDA tests can flip the selection). Predicate factored to
// gdn_prefill_conv.h for CPU-tier unit coverage.
bool ConvRegEnabled() {
  return ConvRegFlagIsOn(std::getenv("VT_CONV_REG"));
}

bool ConvExactChunksEnabled() {
  return ConvExactChunksFlagIsOn(std::getenv("VT_CONV_EXACT_CHUNKS"));
}

template <int ChannelsPerThread, typename Tin, typename Tout>
void LaunchConvFwdRegK4(cudaStream_t s, Tensor& out, const Tensor& x,
                        const Tensor& w, const Tensor* bias, Tensor& conv_state,
                        const Tensor& qsl, const Tensor& his,
                        const CausalConv1dArgs& args, int64_t chan_tiles) {
  static_assert(ChannelsPerThread == 1 || ChannelsPerThread == 2);
  const int64_t n = conv_state.shape[0], c = x.shape[1];
  const int64_t total_tokens = x.shape[0];
  const int64_t x_rs = x.stride[0];
  int64_t grid_z = 1;
  int chunked = 0;
  const bool exact = ConvExactChunksEnabled() && args.batch_ptr != nullptr;
  int64_t grid_y = n;
  if (exact) {
    grid_y = args.batch_ptr->shape[0];
    VT_CHECK(grid_y <= kMaxGridY,
             "cuda causal_conv1d_fwd(reg-k4): too many exact chunk programs");
  } else if (n <= kConvRegChunkMaxSeqs) {
    const int64_t z = (total_tokens + kConvRegM - 1) / kConvRegM;
    if (z >= 1 && z <= kMaxGridY) {
      grid_z = z;
      chunked = 1;
    }
  }
  const dim3 grid(static_cast<unsigned>(chan_tiles), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(kConvRegN);
  const int32_t* batch_ptr = exact ? args.batch_ptr->Ptr<int32_t>() : nullptr;
  const int32_t* token_chunk_offset_ptr =
      exact ? args.token_chunk_offset_ptr->Ptr<int32_t>() : nullptr;
  if (his.dtype == DType::kI8) {
    CausalConv1dFwdRegK4Kernel<ChannelsPerThread, Tin, Tout, int8_t>
        <<<grid, block, 0, s>>>(
            out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
            bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
            qsl.Ptr<int32_t>(), his.Ptr<int8_t>(), c, x_rs,
            args.silu_activation, chunked, batch_ptr, token_chunk_offset_ptr,
            exact ? 1 : 0);
  } else {
    CausalConv1dFwdRegK4Kernel<ChannelsPerThread, Tin, Tout, int32_t>
        <<<grid, block, 0, s>>>(
            out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
            bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
            qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, x_rs,
            args.silu_activation, chunked, batch_ptr, token_chunk_offset_ptr,
            exact ? 1 : 0);
  }
  Check(cudaGetLastError(), "causal_conv1d_fwd(reg-k4) launch");
}

// Register-window launcher (VT_CONV_REG=1). The default exact descriptor maps
// grid.y to a flattened list of (sequence, 8-token chunk) programs, mirroring
// upstream and launching neither rectangular padding nor sequence-serial work.
// VT_CONV_EXACT_CHUNKS=0 restores the legacy grid=(channel tiles, sequences,
// chunks): it chunks grid.z only for <=4 sequences, and serially streams each
// whole sequence for larger batches.
template <typename Tin, typename Tout>
void LaunchConvFwdRegRuntime(cudaStream_t s, Tensor& out, const Tensor& x,
                             const Tensor& w, const Tensor* bias,
                             Tensor& conv_state, const Tensor& qsl,
                             const Tensor& his, const CausalConv1dArgs& args,
                             int64_t chan_tiles) {
  const int64_t n = conv_state.shape[0], c = x.shape[1], k = w.shape[1];
  const int64_t total_tokens = x.shape[0];
  const int64_t x_rs = x.stride[0];  // padded-row (merged qkvz) x view honored
  VT_CHECK(k - 1 <= kConvRegMaxW, "cuda causal_conv1d_fwd(reg): conv width exceeds kConvRegMaxW");
  int64_t gridZ = 1;
  int chunked = 0;
  const bool exact = ConvExactChunksEnabled() && args.batch_ptr != nullptr;
  int64_t gridY = n;
  if (exact) {
    gridY = args.batch_ptr->shape[0];
    VT_CHECK(gridY <= kMaxGridY,
             "cuda causal_conv1d_fwd(reg): too many exact chunk programs");
  } else if (n <= kConvRegChunkMaxSeqs) {
    const int64_t z = (total_tokens + kConvRegM - 1) / kConvRegM;
    if (z >= 1 && z <= kMaxGridY) {
      gridZ = z;
      chunked = 1;
    }
  }
  const dim3 grid(static_cast<unsigned>(chan_tiles), static_cast<unsigned>(gridY),
                  static_cast<unsigned>(gridZ));
  const dim3 block(kConvRegN);
  const int32_t* batch_ptr = exact ? args.batch_ptr->Ptr<int32_t>() : nullptr;
  const int32_t* token_chunk_offset_ptr =
      exact ? args.token_chunk_offset_ptr->Ptr<int32_t>() : nullptr;
  if (his.dtype == DType::kI8) {
    CausalConv1dFwdRegKernel<Tin, Tout, int8_t><<<grid, block, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int8_t>(), c, x_rs, k, args.silu_activation, chunked,
        batch_ptr, token_chunk_offset_ptr, exact ? 1 : 0);
  } else {
    CausalConv1dFwdRegKernel<Tin, Tout, int32_t><<<grid, block, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, x_rs, k, args.silu_activation, chunked,
        batch_ptr, token_chunk_offset_ptr, exact ? 1 : 0);
  }
  Check(cudaGetLastError(), "causal_conv1d_fwd(reg) launch");
}

template <typename Tin, typename Tout>
void LaunchConvFwdReg(cudaStream_t s, Tensor& out, const Tensor& x,
                      const Tensor& w, const Tensor* bias, Tensor& conv_state,
                      const Tensor& qsl, const Tensor& his,
                      const CausalConv1dArgs& args) {
  const int64_t c = x.shape[1], k = w.shape[1];
  DispatchConvChannelTileLaunch(
      std::getenv("VT_CONV_CHANNEL_TILE"), c, k,
      [&](const ConvChannelTileLaunchContract& contract) {
        LaunchConvFwdRegRuntime<Tin, Tout>(s, out, x, w, bias, conv_state, qsl,
                                           his, args, contract.feature_blocks);
      },
      [&](const ConvChannelTileLaunchContract& contract) {
        LaunchConvFwdRegK4<1, Tin, Tout>(s, out, x, w, bias, conv_state, qsl,
                                        his, args, contract.feature_blocks);
      },
      [&](const ConvChannelTileLaunchContract& contract) {
        LaunchConvFwdRegK4<2, Tin, Tout>(s, out, x, w, bias, conv_state, qsl,
                                        his, args, contract.feature_blocks);
      });
}

// Dispatch on the toggles. reg (VT_CONV_REG, default ON) wins; else tiled
// (VT_CONV_TILED); else the EXACT scalar launcher (byte-identical grid/block/kernel).
template <typename Tin, typename Tout>
void DispatchConvFwd(bool reg, bool tiled, cudaStream_t s, Tensor& out, const Tensor& x,
                     const Tensor& w, const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                     const Tensor& his, const CausalConv1dArgs& args) {
  if (reg) {
    LaunchConvFwdReg<Tin, Tout>(s, out, x, w, bias, conv_state, qsl, his, args);
  } else if (tiled) {
    LaunchConvFwdTiled<Tin, Tout>(s, out, x, w, bias, conv_state, qsl, his, args);
  } else {
    LaunchConvFwd<Tin, Tout>(s, out, x, w, bias, conv_state, qsl, his, args);
  }
}

// ---------------------------------------------------------------------------
// causal_conv1d_update (gdn-semantics.md §3): grid-stride, one thread per
// (token, channel). Read-old-then-roll on the thread's own state row.
// Upstream counterpart: layers/mamba/ops/causal_conv1d.py
// (causal_conv1d_update Triton kernel, seqlen==1 path) — align post-MVP.

// cache_idx (optional; mirrors mamba causal_conv1d_update conv_state_indices): when
// non-null, token bt's state row is the persistent-cache slot cache_idx[bt] (idx < 0 ==
// NULL block → skip, leaving out untouched), so the caller need not gather/scatter.
template <typename Tin, typename Tout, typename TState>
__global__ void CausalConv1dUpdateKernel(Tout* out, const Tin* x, const Tin* w,
                                         const Tin* bias, TState* conv_state,
                                         const int32_t* cache_idx, int64_t n, int64_t c_dim,
                                         int64_t x_row_stride, int64_t k, int64_t state_len,
                                         bool silu) {
  const int64_t width = k - 1;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t bt = idx / c_dim;
    const int64_t c = idx % c_dim;
    int64_t srow_off = idx;  // compact: row == bt (idx = bt*c_dim + c)
    if (cache_idx != nullptr) {
      const int32_t slot = cache_idx[bt];
      if (slot < 0) continue;  // NULL block
      srow_off = static_cast<int64_t>(slot) * c_dim + c;
    }
    // Persistent conv_state cache is bf16 (vLLM default) or f32 (unit test);
    // Load()/Store() upcast/downcast, the convolution accumulates in f32. x may
    // be a padded-row (merged qkvz) view; out stays contiguous at idx. The conv
    // row physical width is state_len ((K-1) normally, or (K-1)+num_spec when the
    // spec-decode rollback taps are allocated); we operate on the LEADING (K-1).
    TState* srow = conv_state + srow_off * state_len;  // row [state_len], op [0,K-1)
    const float xt = Load(x, bt * x_row_stride + c);
    float acc = bias != nullptr ? Load(bias, c) : 0.0f;
    for (int64_t j = 0; j < width; ++j) acc += Load(w, c * k + j) * Load(srow, j);
    acc += Load(w, c * k + width) * xt;
    Store(out, idx, silu ? Silu(acc) : acc);
    for (int64_t j = 0; j + 1 < width; ++j) Store(srow, j, Load(srow, j + 1));  // roll left
    if (width > 0) Store(srow, width - 1, xt);                                  // raw x
  }
}

// Decode-fast causal_conv1d_update variant (VT_CONV_UPDATE_FAST). BIT-IDENTICAL
// (0-ulp) to CausalConv1dUpdateKernel above — same bias init, same left-to-right
// conv accumulation, same `w[width]*x` tail, same silu/identity epilogue, same
// round-to-store, same rolled conv_state bytes — differing ONLY in numerics-neutral
// mechanics (see conv_update_fast.h): a 2D grid (blockIdx.y == token, x-dim ==
// channel) that eliminates the shipped kernel's two int64 div/mod per thread, and
// a register-cached state row (WIDTH == k-1 known at compile time so it stays
// register-resident, mirroring the fla per-KERNEL_WIDTH col0..col3 specialization)
// reused for both the accumulation and the roll-left instead of re-reading it from
// global memory. Templated on WIDTH so the two short loops fully unroll and the
// state array lives in registers.
template <typename Tin, typename Tout, typename TState, int WIDTH>
__global__ void CausalConv1dUpdateFastKernel(Tout* out, const Tin* x, const Tin* w,
                                             const Tin* bias, TState* conv_state,
                                             const int32_t* cache_idx, int64_t c_dim,
                                             int64_t x_row_stride, int64_t state_len,
                                             bool silu) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= c_dim) return;
  const int64_t bt = blockIdx.y;             // token — no idx/c_dim, idx%c_dim
  const int64_t idx = bt * c_dim + c;        // contiguous out slot
  const int k = WIDTH + 1;
  int64_t srow_off = idx;                    // compact: row == bt
  if (cache_idx != nullptr) {
    const int32_t slot = cache_idx[bt];
    if (slot < 0) return;                     // NULL block → leave out untouched
    srow_off = static_cast<int64_t>(slot) * c_dim + c;
  }
  // Conv row physical width is state_len ((K-1), or (K-1)+num_spec when spec
  // rollback taps are allocated); operate on the LEADING WIDTH=(K-1) cols.
  TState* srow = conv_state + srow_off * state_len;  // row [state_len]
  // Load the state row ONCE into registers (shipped re-reads srow[j+1] in the roll).
  float sreg[WIDTH];
#pragma unroll
  for (int j = 0; j < WIDTH; ++j) sreg[j] = Load(srow, j);
  const float xt = Load(x, bt * x_row_stride + c);
  float acc = bias != nullptr ? Load(bias, c) : 0.0f;
#pragma unroll
  for (int j = 0; j < WIDTH; ++j) acc += Load(w, c * k + j) * sreg[j];  // shipped order
  acc += Load(w, c * k + WIDTH) * xt;
  Store(out, idx, silu ? Silu(acc) : acc);
#pragma unroll
  for (int j = 0; j + 1 < WIDTH; ++j) Store(srow, j, sreg[j + 1]);  // roll left (cached bits)
  Store(srow, WIDTH - 1, xt);                                       // raw x (WIDTH>=1)
}

// Runtime predicate + width-specialized launch for the decode conv-update fast
// path. Returns true iff it ran (VT_CONV_UPDATE_FAST on, state width in the
// specialized set, and num_tokens within the grid.y limit); false keeps the shipped
// grid-stride CausalConv1dUpdateKernel. Bit-identical to shipped by construction.
template <typename Tin, typename Tout, typename TState>
bool TryLaunchConvUpdateFast(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                             const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                             const CausalConv1dArgs& args) {
  if (!ConvUpdateFastFlagIsOn(std::getenv("VT_CONV_UPDATE_FAST"))) return false;
  const int64_t nt = x.shape[0], c = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  if (nt <= 0 || nt > kMaxGridY) return false;  // 2D grid.y == num_tokens
  const int64_t x_rs = x.stride[0];             // padded-row (merged qkvz) x view honored
  const int64_t state_len = conv_state.shape[2];  // physical row width ((K-1)+num_spec)
  const Tin* wp = w.Ptr<Tin>();
  const Tin* bp = bias != nullptr ? bias->Ptr<Tin>() : nullptr;
  const dim3 grid(static_cast<unsigned>((c + kBlock - 1) / kBlock), static_cast<unsigned>(nt));
  auto launch = [&](auto width_tag) {
    constexpr int W = decltype(width_tag)::value;
    CausalConv1dUpdateFastKernel<Tin, Tout, TState, W><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), wp, bp, conv_state.Ptr<TState>(), cache_idx, c, x_rs,
        state_len, args.silu_activation);
  };
  switch (width) {  // specialize the production widths (Qwen GDN k=4 → width 3)
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    default: return false;  // uncommon width → shipped kernel handles it
  }
  Check(cudaGetLastError(), "causal_conv1d_update fast launch");
  return true;
}

template <typename Tin, typename Tout, typename TState>
void LaunchConvUpdate(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                      const CausalConv1dArgs& args) {
  // Decode conv-update fast path (VT_CONV_UPDATE_FAST): bit-identical to the shipped
  // kernel; only engages for the specialized production widths within the grid.y
  // limit, else falls through to CausalConv1dUpdateKernel.
  if (TryLaunchConvUpdateFast<Tin, Tout, TState>(s, out, x, w, bias, conv_state, cache_idx, args))
    return;
  const int64_t n = x.shape[0] * x.shape[1], c = x.shape[1], k = w.shape[1];
  const int64_t x_rs = x.stride[0];  // padded-row (merged qkvz) x view honored
  const int64_t state_len = conv_state.shape[2];  // physical row width ((K-1)+num_spec)
  CausalConv1dUpdateKernel<Tin, Tout, TState><<<GridFor(n), kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<TState>(), cache_idx, n, c, x_rs, k, state_len, args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_update launch");
}

// Persistent conv_state dtype (bf16 cache = vLLM default; f32 = unit test).
template <typename Tin, typename Tout>
void LaunchConvUpdateS(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                       const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                       const CausalConv1dArgs& args) {
  if (conv_state.dtype == DType::kBF16)
    LaunchConvUpdate<Tin, Tout, __nv_bfloat16>(s, out, x, w, bias, conv_state, cache_idx, args);
  else
    LaunchConvUpdate<Tin, Tout, float>(s, out, x, w, bias, conv_state, cache_idx, args);
}

void ConvFwdKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                       const Tensor& his, const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_fwd: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_fwd: weight/bias dtype must match x");
  const int64_t n = conv_state.shape[0];
  if (n == 0 || x.shape[1] == 0) return;
  VT_CHECK(n <= kMaxGridY, "cuda causal_conv1d_fwd: too many sequences (grid.y limit)");
  cudaStream_t s = AsStream(q);
  const bool reg = ConvRegEnabled();      // VT_CONV_REG (default ON) — FLA register window
  const bool tiled = ConvTiledEnabled();  // VT_CONV_TILED (default ON) — shared-mem tile
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      DispatchConvFwd<float, float>(reg, tiled, s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      DispatchConvFwd<float, __nv_bfloat16>(reg, tiled, s, out, x, w, bias, conv_state, qsl, his,
                                            args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      DispatchConvFwd<__nv_bfloat16, float>(reg, tiled, s, out, x, w, bias, conv_state, qsl, his,
                                            args);
    } else {
      DispatchConvFwd<__nv_bfloat16, __nv_bfloat16>(reg, tiled, s, out, x, w, bias, conv_state,
                                                    qsl, his, args);
    }
  }
}

void ConvUpdateKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                          const Tensor* bias, Tensor& conv_state,
                          const Tensor* conv_state_indices, const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_update: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_update: weight/bias dtype must match x");
  if (x.shape[0] * x.shape[1] == 0) return;
  const int32_t* ci =
      conv_state_indices != nullptr ? conv_state_indices->Ptr<int32_t>() : nullptr;
  cudaStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdateS<float, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdateS<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  }
}

// ---------------------------------------------------------------------------
// l2norm (gdn-semantics.md §4): one block per row, shared-memory f32 tree
// reduction (M0.6 rmsnorm pattern). Plain SUM of squares — not a mean.
// Upstream counterpart: layers/fla/ops/l2norm.py (l2norm_fwd_kernel2) —
// align post-MVP.

template <typename Tin, typename Tout>
__global__ void L2NormRowKernel(Tout* out, const Tin* x, int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  Tout* orow = out + row * d;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] + eps);
  for (int64_t j = threadIdx.x; j < d; j += kBlock) Store(orow, j, Load(xrow, j) * inv);
}

void L2NormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda l2norm: unsupported input dtype (f32/bf16 only)");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(rows);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      L2NormRowKernel<float, float>
          <<<grid, kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<float>(), d, args.eps);
    } else {
      L2NormRowKernel<float, __nv_bfloat16>
          <<<grid, kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<float>(), d, args.eps);
    }
  } else {
    if (out.dtype == DType::kF32) {
      L2NormRowKernel<__nv_bfloat16, float>
          <<<grid, kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<__nv_bfloat16>(), d, args.eps);
    } else {
      L2NormRowKernel<__nv_bfloat16, __nv_bfloat16>
          <<<grid, kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(), d,
                                   args.eps);
    }
  }
  Check(cudaGetLastError(), "l2norm launch");
}

// ---------------------------------------------------------------------------
// gdn_post_conv: fused split + q/k l2norm + g/beta gating (mirror of fla
// fused_gdn_prefill_post_conv _fused_post_conv_kernel, grid (L, H+HV)). One
// launch replaces GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta. Bit-for-bit
// equal: the q/k blocks (i_head < Hk) reuse the EXACT L2NormRowKernel shared-mem
// tree reduction; the trailing (i_head == Hk) block does the v copy + the §6
// g/beta gating (softplus threshold 20), identical to GdnConvSplit/GdnGBeta.
// Templated on the q/k/v output dtype (Tqkv): f32 by default, or bf16 for the
// coupled GDN bf16 path (VT_GDN_BF16) — mirrors FLA keeping the matmul-input
// activations (q/k/v) in bf16 while g/beta stay f32 (they gate the recurrence,
// FLA holds them f32). The l2norm reduction math is f32; only the store rounds
// to Tqkv. g_out/beta_out are always f32. Tgate is f32 for the legacy split
// projections or bf16 for vLLM's packed BA output; each row may be padded by
// the sibling logical view and is upcast on load. Tconv = the conv-output activation
// dtype: f32 (default) or bf16 under the input-side bf16 GDN path (VT_GDN_IN_BF16),
// which mirrors FLA carrying the post-conv activations in bf16 (halved conv-read
// traffic); the reads upcast to f32 via Load() so the norm math is unchanged.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvKernel(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                                  float* beta_out, const Tconv* conv, const Tgate* araw,
                                  const Tgate* braw, const float* a_log,
                                  const float* dt_bias, int64_t hk, int64_t dk,
                                  int64_t hv, int64_t dv, int64_t a_row_stride,
                                  int64_t b_row_stride, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t row = tok * conv_dim;
  if (head < hk) {
    __shared__ float partial[kBlock];
    const Tconv* qin = conv + row + head * dk;
    const Tconv* kin = conv + row + key_dim + head * dk;
    Tqkv* qo = q_out + (tok * hk + head) * dk;
    Tqkv* ko = k_out + (tok * hk + head) * dk;
    // ---- q l2norm (plain SUM of squares over Dk, §4) ----
    float acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(qin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float qinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(qo, j, Load(qin, j) * qinv);
    __syncthreads();
    // ---- k l2norm ----
    acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(kin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float kinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(ko, j, Load(kin, j) * kinv);
  } else {
    // v copy + §6 g/beta for this token.
    const Tconv* vin = conv + row + 2 * key_dim;
    Tqkv* vo = v_out + tok * value_dim;
    for (int64_t j = threadIdx.x; j < value_dim; j += kBlock) Store(vo, j, Load(vin, j));
    for (int64_t h = threadIdx.x; h < hv; h += kBlock) {
      const int64_t idx = tok * hv + h;
      const float av = Load(araw, tok * a_row_stride + h);
      const float bv = Load(braw, tok * b_row_stride + h);
      const float x = av + dt_bias[h];
      const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
      g_out[idx] = -expf(a_log[h]) * sp;
      beta_out[idx] = 1.0f / (1.0f + expf(-bv));
    }
  }
}

// 128-bit-staged copy of a contiguous [n] slice from conv (Tconv) to out (Tqkv),
// striped in 8-element groups across the block's `bd` threads (tid in [0,bd)).
// BYTE-IDENTICAL to the scalar `for j: Store(out,j,Load(conv,j))` loop: a raw int4
// copy when the dtypes match, else the SAME per-element __bfloat162float /
// __float2bfloat16 converts the scalar Load/Store use. Requires 16B alignment of
// both base pointers and n % 8 == 0 (caller guarantees for the Dk=Dv=128 gate dims;
// value_dim = Hv*128 is a multiple of 8). No arithmetic is reordered.
template <typename Tqkv, typename Tconv>
__device__ inline void GdnVecCopy8(Tqkv* out, const Tconv* in, int64_t n, int tid, int bd) {
  const int64_t ng = n >> 3;  // number of 8-element groups
  for (int64_t g = tid; g < ng; g += bd) {
    if constexpr (std::is_same<Tqkv, Tconv>::value) {
      if constexpr (sizeof(Tconv) == 2) {  // bf16/half -> same: one 16B transaction
        reinterpret_cast<int4*>(out)[g] = reinterpret_cast<const int4*>(in)[g];
      } else {  // f32 -> f32: 8 elems = 32B = two 16B transactions
        reinterpret_cast<int4*>(out)[2 * g] = reinterpret_cast<const int4*>(in)[2 * g];
        reinterpret_cast<int4*>(out)[2 * g + 1] = reinterpret_cast<const int4*>(in)[2 * g + 1];
      }
    } else {
      float f[8];
      if constexpr (sizeof(Tconv) == 4) {  // f32 in: two float4 loads
        const float4 a = reinterpret_cast<const float4*>(in)[2 * g];
        const float4 b = reinterpret_cast<const float4*>(in)[2 * g + 1];
        f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
        f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
      } else {  // bf16/half in: one int4 load, unpack via the scalar Load convert
        const int64_t base = g << 3;
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = Load(in, base + e);
      }
      if constexpr (sizeof(Tqkv) == 4) {  // f32 out: two float4 stores
        float4 a, b;
        a.x = f[0]; a.y = f[1]; a.z = f[2]; a.w = f[3];
        b.x = f[4]; b.y = f[5]; b.z = f[6]; b.w = f[7];
        reinterpret_cast<float4*>(out)[2 * g] = a;
        reinterpret_cast<float4*>(out)[2 * g + 1] = b;
      } else {  // bf16 out: pack 8 via the SAME __float2bfloat16 as scalar Store
        __align__(16) Tqkv o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) Store(o, e, f[e]);
        reinterpret_cast<int4*>(out)[g] = *reinterpret_cast<const int4*>(o);
      }
    }
  }
}

// Fast post-conv (VT_GDN_POSTCONV_FAST): the shipped single-megablock grid
// (T, Hk+1) but 128 threads/block + 128-bit-staged V copy. BYTE-IDENTICAL to
// GdnPostConvKernel for the Dk==Dv==128 gate dims (see GdnPostConvFastFlagIsOn):
// every lane owns one q/k element so the 128-wide reduction tree is the shipped
// 256-wide tree minus its leading +0 step, and the V copy reorders nothing. Gated
// to Dk==Dv==128 (16B alignment + value_dim%8==0); other shapes use the megablock.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvFastKernel(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                                      float* beta_out, const Tconv* conv, const Tgate* araw,
                                      const Tgate* braw, const float* a_log, const float* dt_bias,
                                      int64_t hk, int64_t dk, int64_t hv, int64_t dv,
                                      int64_t a_row_stride, int64_t b_row_stride, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t row = tok * conv_dim;
  const int bd = blockDim.x;  // 128 (one lane per Dk=Dv=128 element)
  if (head < hk) {
    // q/k L2-norm: identical tree over the identical squared values (bd == Dk here,
    // so the accumulation is one element per lane, matching the 256-thread path minus
    // its leading partial[t]+=partial[t+128] over the always-zero upper half).
    __shared__ float partial[128];  // launched with exactly 128 threads (bd)
    const Tconv* qin = conv + row + head * dk;
    const Tconv* kin = conv + row + key_dim + head * dk;
    Tqkv* qo = q_out + (tok * hk + head) * dk;
    Tqkv* ko = k_out + (tok * hk + head) * dk;
    float acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += bd) {
      const float v = Load(qin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = bd / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float qinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += bd) Store(qo, j, Load(qin, j) * qinv);
    __syncthreads();
    acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += bd) {
      const float v = Load(kin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = bd / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float kinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += bd) Store(ko, j, Load(kin, j) * kinv);
  } else {
    // V copy (128-bit staged) + §6 g/beta gating for this token.
    const Tconv* vin = conv + row + 2 * key_dim;
    Tqkv* vo = v_out + tok * value_dim;
    GdnVecCopy8<Tqkv, Tconv>(vo, vin, value_dim, static_cast<int>(threadIdx.x), bd);
    for (int64_t h = threadIdx.x; h < hv; h += bd) {
      const int64_t idx = tok * hv + h;
      const float av = Load(araw, tok * a_row_stride + h);
      const float bv = Load(braw, tok * b_row_stride + h);
      const float x = av + dt_bias[h];
      const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
      g_out[idx] = -expf(a_log[h]) * sp;
      beta_out[idx] = 1.0f / (1.0f + expf(-bv));
    }
  }
}

// Experimental 1:1 CUDA spelling of vLLM/FLA's post-conv launch schedule:
// grid=(ceil(T,16), Hk+Hv), BLOCK_T=16, four warps. Each warp owns four tokens
// in the tile. For Q/K, each lane keeps four feature values in registers across
// the float32 norm reduction and normalized store, eliminating the fast kernel's
// second conv read and all shared-memory barriers. For V, the same mapping gives
// four coalesced 32-element copy waves per token and lane zero computes gating.
// Arithmetic and strides otherwise retain the local GdnPostConv contract.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvTokenTileKernel(
    Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out, float* beta_out,
    const Tconv* conv, const Tgate* araw, const Tgate* braw, const float* a_log,
    const float* dt_bias, int64_t t, int64_t hk, int64_t dk, int64_t hv, int64_t dv,
    int64_t a_row_stride, int64_t b_row_stride, float eps) {
  constexpr int kWarps = 4;
  constexpr int kFeaturesPerLane = 4;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int64_t tile_start = static_cast<int64_t>(blockIdx.x) * kGdnPostConvTokenTileTokens;
  const int64_t head = blockIdx.y;
  const int64_t key_dim = hk * dk;
  const int64_t value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;

  if (head < hk) {
#pragma unroll
    for (int tile_token = warp; tile_token < kGdnPostConvTokenTileTokens;
         tile_token += kWarps) {
      const int64_t tok = tile_start + tile_token;
      if (tok >= t) continue;
      const int64_t row = tok * conv_dim;
      const Tconv* qin = conv + row + head * dk;
      const Tconv* kin = conv + row + key_dim + head * dk;
      Tqkv* qo = q_out + (tok * hk + head) * dk;
      Tqkv* ko = k_out + (tok * hk + head) * dk;

      float qv[kFeaturesPerLane];
      float kv[kFeaturesPerLane];
#pragma unroll
      for (int item = 0; item < kFeaturesPerLane; ++item) {
        const int feature = lane + item * 32;
        qv[item] = Load(qin, feature);
        kv[item] = Load(kin, feature);
      }
      // Reproduce the fast kernel's 128-lane shared reduction exactly. Its first
      // two levels are (i+i+64), then (i+i+32); those four terms are resident in
      // this lane. The remaining 32-lane tree is the same 16,8,4,2,1 order.
      float qsum = (qv[0] * qv[0] + qv[2] * qv[2]) +
                   (qv[1] * qv[1] + qv[3] * qv[3]);
      float ksum = (kv[0] * kv[0] + kv[2] * kv[2]) +
                   (kv[1] * kv[1] + kv[3] * kv[3]);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffffu, qsum, offset);
        ksum += __shfl_down_sync(0xffffffffu, ksum, offset);
      }
      qsum = __shfl_sync(0xffffffffu, qsum, 0);
      ksum = __shfl_sync(0xffffffffu, ksum, 0);
      const float qinv = 1.0f / sqrtf(qsum + eps);
      const float kinv = 1.0f / sqrtf(ksum + eps);
#pragma unroll
      for (int item = 0; item < kFeaturesPerLane; ++item) {
        const int feature = lane + item * 32;
        Store(qo, feature, qv[item] * qinv);
        Store(ko, feature, kv[item] * kinv);
      }
    }
  } else {
    const int64_t value_head = head - hk;
#pragma unroll
    for (int tile_token = warp; tile_token < kGdnPostConvTokenTileTokens;
         tile_token += kWarps) {
      const int64_t tok = tile_start + tile_token;
      if (tok >= t) continue;
      const int64_t row = tok * conv_dim;
      const Tconv* vin = conv + row + 2 * key_dim + value_head * dv;
      Tqkv* vo = v_out + tok * value_dim + value_head * dv;
#pragma unroll
      for (int item = 0; item < kFeaturesPerLane; ++item) {
        const int feature = lane + item * 32;
        Store(vo, feature, Load(vin, feature));
      }
      if (lane == 0) {
        const int64_t idx = tok * hv + value_head;
        const float av = Load(araw, tok * a_row_stride + value_head);
        const float bv = Load(braw, tok * b_row_stride + value_head);
        const float x = av + dt_bias[value_head];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        g_out[idx] = -expf(a_log[value_head]) * sp;
        beta_out[idx] = 1.0f / (1.0f + expf(-bv));
      }
    }
  }
}

// Split post-conv (VT_GDN_POSTCONV_SPLIT, opt-in) — grid (T, Hk+Hv), mirroring
// vLLM's grid (ceil(L,BLOCK_T), H+HV) in _fused_post_conv_kernel:57-149 where each V
// head is its own program. The shipped GdnPostConvKernel packs the ENTIRE
// value_dim = Hv*Dv copy + all Hv gating scalars into ONE grid.y block per token
// (head == Hk), serializing ~Hv× more work behind 1/Hv the blocks; this gives each
// (token, v-head) its own block. BIT-IDENTICAL (0-ulp): the head<Hk q/k L2-norm
// branch is byte-for-byte the shipped code; the V copy is the same per-element
// Load/Store; the gating computes the same -exp(A_log)*softplus(a+dt_bias) and
// sigmoid(b) per head over the same f32 inputs. See gdn_prefill_conv.h.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvSplitKernel(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                                       float* beta_out, const Tconv* conv, const Tgate* araw,
                                       const Tgate* braw, const float* a_log,
                                       const float* dt_bias, int64_t hk, int64_t dk,
                                       int64_t hv, int64_t dv, int64_t a_row_stride,
                                       int64_t b_row_stride, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t row = tok * conv_dim;
  if (head < hk) {
    // ---- q/k L2-norm (byte-for-byte the shipped GdnPostConvKernel head<hk path) ----
    __shared__ float partial[kBlock];
    const Tconv* qin = conv + row + head * dk;
    const Tconv* kin = conv + row + key_dim + head * dk;
    Tqkv* qo = q_out + (tok * hk + head) * dk;
    Tqkv* ko = k_out + (tok * hk + head) * dk;
    float acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(qin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float qinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(qo, j, Load(qin, j) * qinv);
    __syncthreads();
    acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(kin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float kinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(ko, j, Load(kin, j) * kinv);
  } else {
    // ---- single v-head copy + §6 g/beta gating for head i_hv = head - hk ----
    const int64_t i_hv = head - hk;
    const Tconv* vin = conv + row + 2 * key_dim + i_hv * dv;
    Tqkv* vo = v_out + tok * value_dim + i_hv * dv;
    for (int64_t j = threadIdx.x; j < dv; j += kBlock) Store(vo, j, Load(vin, j));
    if (threadIdx.x == 0) {
      const int64_t idx = tok * hv + i_hv;
      const float av = Load(araw, tok * a_row_stride + i_hv);
      const float bv = Load(braw, tok * b_row_stride + i_hv);
      const float x = av + dt_bias[i_hv];
      const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
      g_out[idx] = -expf(a_log[i_hv]) * sp;
      beta_out[idx] = 1.0f / (1.0f + expf(-bv));
    }
  }
}

// Toggle: DEFAULT OFF; a present non-'0'-leading value selects the split kernel.
// Read per call (post-conv dispatch is coarse — one launch/step).
bool GdnPostConvSplitEnabled() {
  return GdnPostConvSplitFlagIsOn(std::getenv("VT_GDN_POSTCONV_SPLIT"));
}

void GdnPostConvKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                           Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                           const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                           const GdnPostConvArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0) return;
  VT_CHECK(q_out.dtype == k_out.dtype && q_out.dtype == v_out.dtype,
           "cuda gdn_post_conv: q/k/v out dtypes must match");
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "cuda gdn_post_conv: q/k/v out must be f32 or bf16");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "cuda gdn_post_conv: conv must be f32 or bf16");
  // VT_GDN_POSTCONV_SPLIT (opt-in): grid (T, Hk+Hv) — each V head its own block
  // (mirrors vLLM). =0 (default) uses the single-megablock grid (T, Hk+1).
  const bool split = GdnPostConvSplitEnabled();
  // VT_GDN_POSTCONV_TOKEN_TILE (opt-in): the full upstream 16-token,
  // per-head/four-warp schedule. The explicit split control keeps priority when
  // both experimental flags are set. Only the production 128-wide heads route.
  const bool token_tile = GdnPostConvTokenTileEligible(
      split, std::getenv("VT_GDN_POSTCONV_TOKEN_TILE"), dk, dv);
  // VT_GDN_POSTCONV_FAST: byte-identical megablock at 128 threads + 128-bit V copy.
  // Only for the Dk==Dv==128 gate dims (16B alignment + value_dim%8==0); mutually
  // exclusive with the split (both target the same megablock). See predicate.
  const bool fast = !split && !token_tile &&
                    GdnPostConvFastFlagIsOn(std::getenv("VT_GDN_POSTCONV_FAST")) && dk == 128 &&
                    dv == 128;
  const unsigned grid_x = static_cast<unsigned>(token_tile ? GdnPostConvTokenTileGridX(t) : t);
  const unsigned grid_y = static_cast<unsigned>((split || token_tile) ? hk + hv : hk + 1);
  dim3 grid(grid_x, grid_y);
  cudaStream_t s = AsStream(q);
  // Dispatch over (q/k/v out dtype) x (conv-in dtype). conv is bf16 under the
  // input-side bf16 GDN path (VT_GDN_IN_BF16); the conv read upcasts to f32.
  auto launch = [&](auto qkv_tag, auto conv_tag, auto gate_tag) {
    using Tqkv = decltype(qkv_tag);
    using Tconv = decltype(conv_tag);
    using Tgate = decltype(gate_tag);
    if (token_tile) {
      GdnPostConvTokenTileKernel<Tqkv, Tconv, Tgate><<<grid, 128, 0, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
          a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hk, dk, hv, dv, araw.stride[0],
          braw.stride[0], args.eps);
    } else if (fast) {
      GdnPostConvFastKernel<Tqkv, Tconv, Tgate><<<grid, 128, 0, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
          a_log.Ptr<float>(), dt_bias.Ptr<float>(), hk, dk, hv, dv, araw.stride[0], braw.stride[0],
          args.eps);
    } else if (split) {
      GdnPostConvSplitKernel<Tqkv, Tconv, Tgate><<<grid, kBlock, 0, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(),
          braw.Ptr<Tgate>(), a_log.Ptr<float>(), dt_bias.Ptr<float>(), hk, dk,
          hv, dv, araw.stride[0], braw.stride[0], args.eps);
    } else {
      GdnPostConvKernel<Tqkv, Tconv, Tgate><<<grid, kBlock, 0, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(),
          braw.Ptr<Tgate>(), a_log.Ptr<float>(), dt_bias.Ptr<float>(), hk, dk,
          hv, dv, araw.stride[0], braw.stride[0], args.eps);
    }
  };
  auto launch_gate = [&](auto qkv_tag, auto conv_tag) {
    if (araw.dtype == DType::kBF16)
      launch(qkv_tag, conv_tag, __nv_bfloat16{});
    else
      launch(qkv_tag, conv_tag, float{});
  };
  const bool qkv_f32 = q_out.dtype == DType::kF32;
  const bool conv_f32 = conv.dtype == DType::kF32;
  if (qkv_f32 && conv_f32) {
    launch_gate(float{}, float{});
  } else if (qkv_f32) {
    launch_gate(float{}, __nv_bfloat16{});
  } else if (conv_f32) {
    launch_gate(__nv_bfloat16{}, float{});
  } else {
    launch_gate(__nv_bfloat16{}, __nv_bfloat16{});
  }
  Check(cudaGetLastError(), "gdn_post_conv launch");
}

// ---------------------------------------------------------------------------
// rmsnorm_gated (gdn-semantics.md §5): one block per row, same reduction; var
// is a MEAN (unlike §4), norm first, then act(gate) (norm_before_gate=True,
// group_size=None — the only configuration Qwen GDN uses, baked in).
// Upstream counterpart: layers/fla/ops/layernorm_guard.py
// (layer_norm_fwd_kernel via RMSNormGated) — align post-MVP.

// x/out are contiguous ([rows, D] flat). The gate may be a padded-row rank-3
// [T,Hv,D] view of the merged qkvz z slice: super-row `row` maps to
// (token=row/gate_group, head=row%gate_group) with gate_outer the token stride.
// Contiguous rank-2 gate uses gate_group=1, gate_outer=D (zrow = gate + row*D).
template <typename Tin, typename Tout>
__global__ void RmsNormGatedRowKernel(Tout* out, const Tin* x, const Tin* gate, const Tin* w,
                                      int64_t d, int64_t gate_group, int64_t gate_outer,
                                      float eps, bool sigmoid_gate) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  const Tin* zrow = gate + (row / gate_group) * gate_outer + (row % gate_group) * d;
  Tout* orow = out + row * d;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float z = Load(zrow, j);
    const float act = sigmoid_gate ? 1.0f / (1.0f + expf(-z)) : Silu(z);
    Store(orow, j, Load(xrow, j) * inv * Load(w, j) * act);
  }
}

// ---------------------------------------------------------------------------
// Fast gated-RMSNorm variant (VT_RMSNORM_GATED_FAST). BIT-IDENTICAL (0-ulp) to the
// shipped RmsNormGatedRowKernel above for the d==128 gated-norm path — covering BOTH
// the bf16 core/z/weight both arms carry by default AND the f32 core/z/weight the
// VT_GDN_OUT_BF16=0 rollback restores (before #1168 that f32 arm was the MoE 35B's
// DEFAULT), mirroring the RMSNorm decode-fast bit-safety technique (cuda_ops.cu
// RmsNormRowFastKernel, 348d12d). Templated over (Tin,Tout) via the shared Load/Store
// helpers so a single kernel is bit-exact for every dispatched dtype pair.
//
// WHY BIT-IDENTICAL (not <=1-ulp): the gated norm feeds the SAME 27B forward as the
// fast plain-RMSNorm + GDN decode cubin, all converging on the greedy RAZOR near-tie
// at token 6 (198 prod vs 271 emu). Per the a875397 lesson a <=1-ulp-close kernel can
// accumulate over the layers and tip that tie, failing the production stream. Making
// the fast gated output the SAME BITS as shipped removes the perturbation by
// construction: {fast-RMSNorm + fast-gated + cubin} logits == the passing baseline.
//
// The shape is [T*Hv, Dv] with Dv==128 (27B Hv=48, 35B Hv=32; z gate is a padded-row
// rank-3 [T,Hv,Dv] view or a contiguous rank-2 view — same gate_group/gate_outer
// addressing as shipped). Unlike the plain RMSNorm decode-fast (block-starved, ~16
// rows => 1024 threads to hide latency), the gated norm launches T*Hv (~512-768 at
// c16) blocks and is NOT block-starved: the shipped kernel's inefficiency is that it
// launches kBlock=256 threads for a 128-element row, so the upper half of EVERY block
// is idle in both passes, and it reloads x in the normalize pass. This kernel fixes
// both: 128 threads (one per element, no idle half => 2x the occupancy headroom) and
// x cached in a register (loaded once).
//
// BIT-IDENTITY, term by term vs shipped RmsNormGatedRowKernel:
//   (1) variance ORDER. shipped: 256 threads, per-thread acc = Sum_m sq[tid+256m];
//       for d==128 that is a SINGLE term sq[tid] for tid<128 and 0 for tid>=128,
//       then the shared-tree `for s=128; s>0; s>>=1`. Here 128 threads set
//       partial[tid]=sq[tid] and the tree runs from s=64. shipped's extra s=128 step
//       only does partial[i] += partial[i+128] where partial[128..255] are provably
//       0, so omitting it is byte-for-byte identical; the remaining s=64..1 pairing
//       is the same. f32(bf16 x)^2 is the same squared value on both sides.
//   (2) inv. shipped: `1.0f / sqrtf(partial[0]/d + eps)` (correctly-rounded sqrt+div),
//       reproduced verbatim (NOT rsqrtf).
//   (3) normalize + act. shipped: `Load(xrow,j)*inv*Load(w,j)*act` == ((x*inv)*w)*act,
//       act = sigmoid_gate ? 1/(1+exp(-z)) : Silu(z)=z/(1+exp(-z)); reproduced with
//       the same left-to-right multiply order, same act, same __float2bfloat16 store.
// Only the launch geometry (128 vs 256 threads) and the single register-cached x load
// differ; the arithmetic is identical.
//
// Scope: d==128 (both gate models' Dv), core/gate/weight one shared dtype in {f32,bf16},
// out in {f32,bf16}. Templated over (Tin,Tout) so ONE bit-identical kernel covers BOTH
// the default bf16 core/z/weight path AND the f32 one: GdnOutDType returns bf16 on
// every arm since #1168 and f32 under the VT_GDN_OUT_BF16=0 rollback, and before #1168
// f32 was the MoE 35B's DEFAULT — either way f32 gated-norm inputs were sent to the
// slow shipped kernel by the former bf16-only guard.
// The kernel below uses the SAME Load/Store helpers as the shipped
// RmsNormGatedRowKernel<Tin,Tout>, so for EVERY (Tin,Tout) the float op sequence is
// byte-for-byte identical to shipped (Load(float) is the identity; Load(bf16) is the
// same __bfloat162float; Store the same rounding) — the bit-identity argument above
// holds unchanged. Only launch geometry (128 vs 256 threads) and the single
// register-cached x load differ. Other dtype/shape keep RmsNormGatedRowKernel.
constexpr int kGatedFastBlock = 128;   // one thread per Dv element (Dv==128)

template <typename Tin, typename Tout>
__global__ void RmsNormGatedRowFastKernel(Tout* __restrict__ out, const Tin* __restrict__ x,
                                          const Tin* __restrict__ gate, const Tin* __restrict__ w,
                                          int64_t gate_group, int64_t gate_outer, float eps,
                                          bool sigmoid_gate) {
  constexpr int d = kGatedFastBlock;  // 128
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  const Tin* zrow = gate + (row / gate_group) * gate_outer + (row % gate_group) * d;
  Tout* orow = out + row * d;

  __shared__ float partial[kGatedFastBlock];
  // Load x ONCE, cache in a register (shipped reloads it in the normalize pass); the
  // cached value is the SAME Load(xrow,tid) shipped squares and later re-reads.
  const float xf = Load(xrow, tid);  // f32 identity / f32(bf16 x) == shipped's term base
  partial[tid] = xf * xf;            // per-element variance term == shipped's
  __syncthreads();
  // shared-memory binary tree from s=64 — reproduces shipped's kBlock=256 tree for
  // d==128 (its s=128 step only adds the provably-zero partials[128..255]).
  for (int s = kGatedFastBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);  // NOT rsqrtf
  const float z = Load(zrow, tid);
  const float act = sigmoid_gate ? 1.0f / (1.0f + expf(-z)) : Silu(z);  // shipped act
  const float wf = Load(w, tid);
  Store(orow, tid, xf * inv * wf * act);  // ((x*inv)*w)*act, shipped multiply order
}

template <typename Tin, typename Tout>
inline void LaunchGatedFast(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args, int64_t gate_group,
                            int64_t gate_outer, unsigned grid) {
  RmsNormGatedRowFastKernel<Tin, Tout><<<grid, kGatedFastBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), gate.Ptr<Tin>(), w.Ptr<Tin>(), gate_group, gate_outer,
      args.eps, args.sigmoid_gate);
}

// Runtime predicate + launch for the gated fast path. Returns true iff it ran.
// Guard: d==128 (the GDN Dv used by both gate models); core/gate/weight share one dtype
// (the caller VT_CHECKs gate.dtype==x.dtype==w.dtype) in {f32,bf16}; out in {f32,bf16}.
// This covers BOTH the bf16 core/z/weight both gate models carry by default since
// #1168 (GdnOutDType stopped branching on model shape, so the 35B is no longer the
// f32 arm) AND the f32 core/z/weight the VT_GDN_OUT_BF16=0 rollback restores on
// either of them; `out` is bf16 exactly when GlueFuseEnabled() and f32
// UNCONDITIONALLY otherwise -- it does NOT follow the input. All three
// non-GlueFuse gated-norm call sites allocate `DBuf dgated(d, DType::kF32, ...)`
// whatever the input dtype (qwen3_5.cpp:3965, :4435, :4886), so this row's own
// VT_GDN_OUT_BF16=1 default plus the documented VT_GLUE_FUSE=0 rollback makes
// bf16-in/f32-out a REACHABLE pair. That is why all four (Tin,Tout)
// instantiations below are live: the dispatch reads out.dtype independently of
// x.dtype, and narrowing it to the diagonal would drop a shipped combination.
// Out-of-scope shapes keep RmsNormGatedRowKernel. The launch
// uses kGatedFastBlock (=128) threads and reproduces shipped's reduction ORDER, so the
// output is bit-identical for every dispatched (Tin,Tout).
inline bool TryLaunchRmsNormGatedFast(cudaStream_t s, Tensor& out, const Tensor& x,
                                      const Tensor& gate, const Tensor& w,
                                      const RmsNormGatedArgs& args, int64_t d, int64_t gate_group,
                                      int64_t gate_outer, unsigned grid) {
  if (!RmsNormGatedFastFlagIsOn(std::getenv("VT_RMSNORM_GATED_FAST"))) return false;
  if (d != kGatedFastBlock) return false;
  const bool in_f32 = x.dtype == DType::kF32;
  const bool in_bf16 = x.dtype == DType::kBF16;
  if (!in_f32 && !in_bf16) return false;
  if (gate.dtype != x.dtype || w.dtype != x.dtype) return false;
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) return false;
  const bool out_f32 = out.dtype == DType::kF32;
  if (in_f32) {
    if (out_f32)
      LaunchGatedFast<float, float>(s, out, x, gate, w, args, gate_group, gate_outer, grid);
    else
      LaunchGatedFast<float, __nv_bfloat16>(s, out, x, gate, w, args, gate_group, gate_outer, grid);
  } else {
    if (out_f32)
      LaunchGatedFast<__nv_bfloat16, float>(s, out, x, gate, w, args, gate_group, gate_outer, grid);
    else
      LaunchGatedFast<__nv_bfloat16, __nv_bfloat16>(s, out, x, gate, w, args, gate_group,
                                                    gate_outer, grid);
  }
  return true;
}

void RmsNormGatedKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda rmsnorm_gated: unsupported input dtype (f32/bf16 only)");
  VT_CHECK(gate.dtype == x.dtype && w.dtype == x.dtype,
           "cuda rmsnorm_gated: gate/weight dtype must match x");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = d == 0 ? 0 : x.Numel() / d;
  if (t == 0 || d == 0) return;
  // Padded-row rank-3 [T,Hv,D] gate: super-row -> (token, head). Contiguous
  // rank-2 gate degenerates to gate_group=1, gate_outer=gate.stride[0]==d.
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  cudaStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(t);
  // Gated decode-fast path (VT_RMSNORM_GATED_FAST, default ON; '0' = rollback): only
  // engages for the bf16 d==128 gated-norm decode launches; every other case keeps
  // RmsNormGatedRowKernel. Output is bit-identical to RmsNormGatedRowKernel by
  // construction.
  if (TryLaunchRmsNormGatedFast(s, out, x, gate, w, args, d, gate_group, gate_outer, grid)) {
    Check(cudaGetLastError(), "rmsnorm_gated fast launch");
    return;
  }
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      RmsNormGatedRowKernel<float, float><<<grid, kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), d, gate_group,
          gate_outer, args.eps, args.sigmoid_gate);
    } else {
      RmsNormGatedRowKernel<float, __nv_bfloat16><<<grid, kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), d,
          gate_group, gate_outer, args.eps, args.sigmoid_gate);
    }
  } else {
    if (out.dtype == DType::kF32) {
      RmsNormGatedRowKernel<__nv_bfloat16, float><<<grid, kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(),
          w.Ptr<__nv_bfloat16>(), d, gate_group, gate_outer, args.eps, args.sigmoid_gate);
    } else {
      RmsNormGatedRowKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(),
          w.Ptr<__nv_bfloat16>(), d, gate_group, gate_outer, args.eps, args.sigmoid_gate);
    }
  }
  Check(cudaGetLastError(), "rmsnorm_gated launch");
}

// ---------------------------------------------------------------------------
// rmsnorm_gated + static fp8 quant, fused (RmsNormGatedQuantFp8). The gated-norm
// analog of cuda_ops.cu's RmsNormQuantFp8RowKernel: emit the fp8 activation directly
// from the gated-norm normalize loop, so the standalone QuantFp8Static pass + its
// bf16 round-trip disappear on the 35B GDN out_proj (W8A8) path. Mirrors vLLM's
// Inductor fusion of the RMSNormGated epilogue with the following RowParallelLinear's
// static-fp8 activation quant (fla layernorm_guard.py + rms_quant_fusion.py's
// static-fp8 pattern applied to the GATED variant).
//
// BIT-IDENTICAL to RmsNormGated(bf16 out)+QuantFp8Static(input_scale): the fp8 is
// taken from the SAME bf16-rounded value the split path quantizes
// (F32ToFp8(__bfloat162float(__float2bfloat16(n)) * inv_scale)), and the variance
// reduction ORDER is the shipped/fast RmsNormGated order (the fast d==128 kernel is
// itself 0-ulp vs shipped, see RmsNormGatedRowFastKernel above). Only the store
// changes (fp8 byte instead of bf16); the arithmetic is unchanged. Same conversion
// as cuda_ops.cu RmsNormF32ToFp8Dev / QuantFp8StaticKernel.
__device__ __forceinline__ uint8_t GatedNormF32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

// General (shipped-order) fused kernel: reproduces RmsNormGatedRowKernel's kBlock=256
// reduction + normalize, storing fp8 instead of Tout. Byte-identical to
// RmsNormGatedRowKernel<Tin,bf16> followed by QuantFp8Static.
template <typename Tin>
__global__ void RmsNormGatedQuantFp8RowKernel(uint8_t* out_fp8, const Tin* x, const Tin* gate,
                                              const Tin* w, int64_t d, int64_t gate_group,
                                              int64_t gate_outer, float eps, bool sigmoid_gate,
                                              float inv_scale) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  const Tin* zrow = gate + (row / gate_group) * gate_outer + (row % gate_group) * d;
  uint8_t* orow = out_fp8 + row * d;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float z = Load(zrow, j);
    const float act = sigmoid_gate ? 1.0f / (1.0f + expf(-z)) : Silu(z);
    // bf16-intermediate (matches RmsNormGated's bf16 store then QuantFp8Static's load).
    const __nv_bfloat16 nb = __float2bfloat16(Load(xrow, j) * inv * Load(w, j) * act);
    orow[j] = GatedNormF32ToFp8Dev(__bfloat162float(nb) * inv_scale);
  }
}

// Fast d==128 fused kernel: reproduces RmsNormGatedRowFastKernel's 128-thread
// reduction (0-ulp vs shipped) + normalize, storing fp8. This is the 35B GDN out_proj
// production path (Dv==128, f32 core/gate/weight). Byte-identical to
// RmsNormGatedRowFastKernel<Tin,bf16> followed by QuantFp8Static.
template <typename Tin>
__global__ void RmsNormGatedQuantFp8RowFastKernel(uint8_t* __restrict__ out_fp8,
                                                  const Tin* __restrict__ x,
                                                  const Tin* __restrict__ gate,
                                                  const Tin* __restrict__ w, int64_t gate_group,
                                                  int64_t gate_outer, float eps, bool sigmoid_gate,
                                                  float inv_scale) {
  constexpr int d = kGatedFastBlock;  // 128
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  const Tin* zrow = gate + (row / gate_group) * gate_outer + (row % gate_group) * d;
  uint8_t* orow = out_fp8 + row * d;

  __shared__ float partial[kGatedFastBlock];
  const float xf = Load(xrow, tid);
  partial[tid] = xf * xf;
  __syncthreads();
  for (int s = kGatedFastBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);  // NOT rsqrtf
  const float z = Load(zrow, tid);
  const float act = sigmoid_gate ? 1.0f / (1.0f + expf(-z)) : Silu(z);
  const float wf = Load(w, tid);
  // Same value RmsNormGatedRowFastKernel stores as bf16, then re-quantized to fp8.
  const __nv_bfloat16 nb = __float2bfloat16(xf * inv * wf * act);
  orow[tid] = GatedNormF32ToFp8Dev(__bfloat162float(nb) * inv_scale);
}

// Dispatch mirroring RmsNormGatedKernelCuda exactly (same fast-path predicate + same
// shipped-kernel fallback), but emitting fp8. Only f32/bf16 x is supported (the GDN
// gated-norm inputs); the fast path engages for d==128 with VT_RMSNORM_GATED_FAST on,
// so it is bit-identical to whichever RmsNormGated kernel the split path would run.
void RmsNormGatedQuantFp8KernelCuda(Queue& q, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                                    const Tensor& w, const RmsNormGatedArgs& args,
                                    float input_scale) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda rmsnorm_gated_quant_fp8: unsupported input dtype (f32/bf16 only)");
  VT_CHECK(gate.dtype == x.dtype && w.dtype == x.dtype,
           "cuda rmsnorm_gated_quant_fp8: gate/weight dtype must match x");
  VT_CHECK(out_fp8.dtype == DType::kI8, "cuda rmsnorm_gated_quant_fp8: out_fp8 must be i8");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = d == 0 ? 0 : x.Numel() / d;
  if (t == 0 || d == 0) return;
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  cudaStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(t);
  const float inv_scale = 1.0f / input_scale;
  uint8_t* op = out_fp8.Ptr<uint8_t>();
  // Fast d==128 path (matches TryLaunchRmsNormGatedFast's predicate for the fp8 case).
  if (RmsNormGatedFastFlagIsOn(std::getenv("VT_RMSNORM_GATED_FAST")) && d == kGatedFastBlock) {
    if (x.dtype == DType::kF32) {
      RmsNormGatedQuantFp8RowFastKernel<float><<<grid, kGatedFastBlock, 0, s>>>(
          op, x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), gate_group, gate_outer, args.eps,
          args.sigmoid_gate, inv_scale);
    } else {
      RmsNormGatedQuantFp8RowFastKernel<__nv_bfloat16><<<grid, kGatedFastBlock, 0, s>>>(
          op, x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(), w.Ptr<__nv_bfloat16>(),
          gate_group, gate_outer, args.eps, args.sigmoid_gate, inv_scale);
    }
    Check(cudaGetLastError(), "rmsnorm_gated_quant_fp8 fast launch");
    return;
  }
  if (x.dtype == DType::kF32) {
    RmsNormGatedQuantFp8RowKernel<float><<<grid, kBlock, 0, s>>>(
        op, x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), d, gate_group, gate_outer, args.eps,
        args.sigmoid_gate, inv_scale);
  } else {
    RmsNormGatedQuantFp8RowKernel<__nv_bfloat16><<<grid, kBlock, 0, s>>>(
        op, x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(), w.Ptr<__nv_bfloat16>(), d,
        gate_group, gate_outer, args.eps, args.sigmoid_gate, inv_scale);
  }
  Check(cudaGetLastError(), "rmsnorm_gated_quant_fp8 launch");
}

// ---------------------------------------------------------------------------
// Gated-delta-rule scan (gdn-semantics.md §7), shared by prefill and decode:
// one BLOCK per (sequence, v-head), threads parallelize over Dv state rows,
// sequential over the sequence's tokens. The f32 state [Dv,Dk] stays in
// GLOBAL memory, updated in place on the state tensor (Dv*Dk*4B = 64KB at
// real dims exceeds the 48KB default shared-memory budget).
//
// Per token: q' = q*scale; S *= exp(g); v' = (v - S@k)*beta;
// S += outer(v',k); o = S@q'. Thread `vi` owns state row vi EXCLUSIVELY:
// S@k is a row-dot with its own row, the decay/rank-1 updates touch only its
// own row, and o[vi] is its own row dotted with q' — no cross-thread hazard
// on S, so no barriers are needed in the row phase. The only barriers guard
// the shared q'/k staging: one after the cooperative load (all threads must
// see the full vectors) and one at the end of the token iteration (no thread
// may overwrite q'/k for token t+1 while another still reads them for t).
//
// qsl == nullptr selects decode: block y is the token index, token range
// [y, y+1), state row y (one single-token sequence per batch row).
// Upstream counterpart: layers/fla/ops/{fused_recurrent,fused_sigmoid_gating}.py
// Triton kernels (chunked variant: chunk.py, lands M2.3) — align post-MVP.

template <typename Tin, typename Tout, typename TState>
__global__ void GdnScanKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                              const float* g, const float* beta, TState* state,
                              const int32_t* qsl, const int32_t* state_idx,
                              int64_t state_slots, int64_t hk_n, int64_t dk,
                              int64_t hv_n, int64_t dv, float scale) {
  const int64_t s = blockIdx.y;   // sequence (prefill) or token (decode)
  const int64_t hv = blockIdx.x;  // v-head
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        Store(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head =
      state + (state_slot * hv_n + hv) * dv * dk;  // [Dv, Dk]
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Load(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Load(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;  // (S * exp(g)) @ k, fused with the decay pass
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Load(s_row, ki) * decay;
        dot += decayed * k_sh[ki];
      }
      const float vp = (Load(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Load(s_row, ki) * decay + vp * k_sh[ki];
        Store(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      Store(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();  // all reads of q_sh/k_sh done before the next token's load
  }
}

template <typename Tin, typename Tout, typename TState>
void LaunchGdnScan(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                   const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                   const int32_t* qsl, const int32_t* state_idx, int64_t n,
                   const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 2 * static_cast<size_t>(dk) * sizeof(float);
  GdnScanKernel<Tin, Tout, TState><<<grid, kBlock, shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<TState>(), qsl, state_idx, state.shape[0],
      hk_n, dk, hv_n, dv, args.scale);
  Check(cudaGetLastError(), "gdn scan launch");
}

template <typename Tin, typename Tout>
void LaunchGdnScanState(cudaStream_t s, Tensor& out, const Tensor& q_in,
                        const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, Tensor& state, const int32_t* qsl,
                        const int32_t* state_idx, int64_t n,
                        const GdnArgs& args) {
  if (state.dtype == DType::kF16) {
    LaunchGdnScan<Tin, Tout, __half>(s, out, q_in, k, v, g, beta, state,
                                     qsl, state_idx, n, args);
  } else if (state.dtype == DType::kBF16) {
    LaunchGdnScan<Tin, Tout, __nv_bfloat16>(
        s, out, q_in, k, v, g, beta, state, qsl, state_idx, n, args);
  } else {
    LaunchGdnScan<Tin, Tout, float>(s, out, q_in, k, v, g, beta, state,
                                    qsl, state_idx, n, args);
  }
}

void GdnScanCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                 const Tensor& g, const Tensor& beta, Tensor& state,
                 const int32_t* qsl_ptr, const int32_t* state_idx,
                 const GdnArgs& args,
                 const char* name);  // fwd decl (corner-dim fallback)

// ---------------------------------------------------------------------------
// Packed pure-decode port of vLLM v0.25.0
// vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-478 @ 702f4814.
// Raw packed q/k/v and row-strided a/b are consumed directly. q/k reduction
// matches the existing L2NormRowKernel tree at the real Dk=128 shape; the
// recurrence reuses the proven NW-split fused decode ordering. All state math
// is f32; output stores round to T and persistent-state stores independently
// round to TState. A_log and dt_bias also carry independent storage dtypes.
// beta deliberately rounds through b.dtype before recurrence; this is distinct
// from mixed/spec decode, which keeps beta f32.
template <typename T, typename TState, int NW>
__global__ void GdnPackedDecodeKernel(
    T* out, const T* mixed_qkv, const T* a, const T* b,
    const void* a_log, DType a_log_dtype, const void* dt_bias,
    DType dt_bias_dtype, TState* state,
    const int32_t* state_idx, int64_t state_slots, int64_t mixed_stride,
    int64_t a_stride, int64_t b_stride, int64_t hk_n, int64_t dk,
    int64_t hv_n, int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;
  const int64_t i_nh = blockIdx.y;
  const int64_t i_n = i_nh / hv_n;
  const int64_t hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t vbase = i_v * bv;
  const int tid = static_cast<int>(threadIdx.x);
  const int vi = tid / NW;
  const int wk = tid % NW;
  const int64_t vrow = vbase + vi;

  const int32_t slot = state_idx[i_n];
  if (slot < 0 || slot >= state_slots) {
    if (vrow < dv && wk == 0)
      Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
    return;
  }

  __shared__ float q_partial[kBlock];
  __shared__ float k_partial[kBlock];
  __shared__ float scalars[4];  // q_inv, k_inv, decay, rounded beta
  extern __shared__ float smem[];
  float* bq = smem;
  float* bk = bq + dk;
  float* sbh = bk + dk;

  const int64_t mixed_row = i_n * mixed_stride;
  const int64_t qbase = mixed_row + hk * dk;
  const int64_t kbase = mixed_row + hk_n * dk + hk * dk;
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int64_t ki = tid; ki < dk; ki += blockDim.x) {
    const float qv = Load(mixed_qkv, qbase + ki);
    const float kv = Load(mixed_qkv, kbase + ki);
    bq[ki] = qv;
    bk[ki] = kv;
    q_sumsq += qv * qv;
    k_sumsq += kv * kv;
  }
  q_partial[tid] = q_sumsq;
  k_partial[tid] = k_sumsq;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
       stride /= 2) {
    if (tid < stride) {
      q_partial[tid] += q_partial[tid + stride];
      k_partial[tid] += k_partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    scalars[0] = 1.0f / sqrtf(q_partial[0] + 1e-6f);
    scalars[1] = 1.0f / sqrtf(k_partial[0] + 1e-6f);
    const float av = Load(a, i_n * a_stride + hv);
    const float bv_raw = Load(b, i_n * b_stride + hv);
    const float x = av + LoadFloating(dt_bias, dt_bias_dtype, hv);
    const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
    const float g = -expf(LoadFloating(a_log, a_log_dtype, hv)) * softplus;
    scalars[2] = expf(g);
    scalars[3] = RoundToStorage<T>(1.0f / (1.0f + expf(-bv_raw)));
  }
  __syncthreads();
  for (int64_t ki = tid; ki < dk; ki += blockDim.x) {
    bq[ki] *= scalars[0] * scale;
    bk[ki] *= scalars[1];
  }

  const int64_t sdk = dk + 1;
  const int64_t vrows = (dv - vbase) < bv ? (dv - vbase) : bv;
  const int64_t tile = vrows * dk;
  TState* s_head = state +
                   (static_cast<int64_t>(slot) * hv_n + hv) * dv * dk +
                   vbase * dk;
  for (int64_t e = tid; e < bv * dk; e += blockDim.x)
    sbh[(e / dk) * sdk + e % dk] = e < tile ? Load(s_head, e) : 0.0f;
  __syncthreads();

  const int64_t ck = (dk + NW - 1) / NW;
  const int64_t c0 = static_cast<int64_t>(wk) * ck;
  const int64_t c1 = (c0 + ck) < dk ? (c0 + ck) : dk;
  float* row = sbh + static_cast<int64_t>(vi) * sdk;
  float partial_dot = 0.0f;
  for (int64_t ki = c0; ki < c1; ++ki) {
    row[ki] *= scalars[2];
    partial_dot += row[ki] * bk[ki];
  }
  float dot = partial_dot;
#pragma unroll
  for (int off = 1; off < NW; off <<= 1)
    dot += __shfl_xor_sync(0xffffffffu, dot, off);
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float vv = vrow < dv ? Load(mixed_qkv, v_offset) : 0.0f;
  const float vp = (vv - dot) * scalars[3];
  float partial_out = 0.0f;
  for (int64_t ki = c0; ki < c1; ++ki) {
    row[ki] += vp * bk[ki];
    partial_out += row[ki] * bq[ki];
  }
  float output = partial_out;
#pragma unroll
  for (int off = 1; off < NW; off <<= 1)
    output += __shfl_xor_sync(0xffffffffu, output, off);
  if (vrow < dv && wk == 0)
    Store(out, (i_n * hv_n + hv) * dv + vrow, output);
  __syncthreads();
  for (int64_t e = tid; e < tile; e += blockDim.x)
    Store(s_head, e, sbh[(e / dk) * sdk + e % dk]);
}

template <typename T, typename TState, int NW>
void LaunchGdnPackedDecode(cudaStream_t stream, Tensor& out,
                           const Tensor& mixed_qkv, const Tensor& a,
                           const Tensor& b, const Tensor& a_log,
                           const Tensor& dt_bias, Tensor& state,
                           const Tensor& state_idx, int64_t bv,
                           const GdnArgs& args) {
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  const int64_t hk_n =
      (mixed_qkv.shape[1] - hv_n * dv) / (2 * dk);
  const int64_t nv = (dv + bv - 1) / bv;
  const dim3 grid(static_cast<unsigned>(nv),
                  static_cast<unsigned>(batch * hv_n));
  const size_t shmem =
      (2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) *
      sizeof(float);
  GdnPackedDecodeKernel<T, TState, NW>
      <<<grid, static_cast<unsigned>(bv * NW), shmem, stream>>>(
          out.Ptr<T>(), mixed_qkv.Ptr<T>(), a.Ptr<T>(), b.Ptr<T>(),
          a_log.data, a_log.dtype, dt_bias.data, dt_bias.dtype,
          state.Ptr<TState>(),
          state_idx.Ptr<int32_t>(), state.shape[0], mixed_qkv.stride[0],
          a.stride[0], b.stride[0], hk_n, dk, hv_n, dv, bv, args.scale);
  Check(cudaGetLastError(), "gdn packed decode launch");
}

template <typename T, typename TState>
void DispatchGdnPackedDecode(cudaStream_t stream, Tensor& out,
                             const Tensor& mixed_qkv, const Tensor& a,
                             const Tensor& b, const Tensor& a_log,
                             const Tensor& dt_bias, Tensor& state,
                             const Tensor& state_idx, int64_t bv,
                             const GdnArgs& args) {
  const int64_t dv = state.shape[2];
  if (dv >= 32) {
    LaunchGdnPackedDecode<T, TState, 8>(stream, out, mixed_qkv, a, b, a_log,
                                        dt_bias, state, state_idx, bv, args);
  } else {
    LaunchGdnPackedDecode<T, TState, 1>(stream, out, mixed_qkv, a, b, a_log,
                                        dt_bias, state, state_idx, bv, args);
  }
}

// ---------------------------------------------------------------------------
// Register-resident single-warp packed pure-decode — a 1:1 port of vLLM FLA's
// fused_recurrent_gated_delta_rule_packed_decode_kernel (num_warps=1,
// num_stages=3), vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-336
// (launch :391-478) @ 702f4814. One warp owns ONE (sequence, v-head, BV value-
// tile) and each lane owns ONE value-row, holding that row's whole [1,BK] state
// slice in REGISTERS across the update: the recurrence never round-trips the
// state through shared memory, the two Dk contractions are lane-local sequential
// sums (no cross-warp __shfl reduction), and there are no __syncthreads
// barriers. Only q'/k are staged in shared (a [dk] broadcast each) and the L2
// norm sums are the single warp-local reductions. This replaces the legacy
// GdnPackedDecodeKernel (shared-staged [BV,Dk] slice, 8 warps, NW-way __shfl_xor
// reduction, two __syncthreads), which reached ~83% of state bandwidth; the
// register-resident structure mirrors vLLM's ~92% (see .agents/specs/
// gdn-packed-decode.md and the 2026-07-16 correct-state kernel comparison).
//
// NUMERIC SEMANTICS ARE IDENTICAL to GdnPackedDecodeKernel and the CPU
// PackedDecodeReference: F32 arithmetic throughout; in-kernel F32 q/k L2
// normalization (1/sqrtf(sumsq+1e-6)); decay = expf(-expf(A_log)*softplus(a+
// dt_bias)); beta = RoundToStorage<T>(sigmoid(b)) rounded THROUGH the activation
// dtype before the recurrence; per row the state is scaled by decay, the dot
// (S·decay)@k and the output S@q' are each a single sequential Dk sum, and the
// output/state stores round to Tout/TState via Store(). Because each lane owns a
// full row the Dk reduction is the reference's sequential order (closer to the
// fixture oracle than the legacy 8-way butterfly), so the BF16 output stays
// bit-exact and the F32 state within the fixture's 1e-4 contract. BK is a
// compile-time constant so the [BK] register row fully unrolls and stays
// register-resident; the launcher instantiates BK == dk (32, or the 128 gate
// dim) and only for bv == 32 (one warp per tile), routing every other geometry
// to the legacy kernel.
template <typename T, typename TState, int BK>
__global__ void GdnPackedDecodeRegTileKernel(
    T* out, const T* mixed_qkv, const T* a, const T* b,
    const void* a_log, DType a_log_dtype, const void* dt_bias,
    DType dt_bias_dtype, TState* state,
    const int32_t* state_idx, int64_t state_slots, int64_t mixed_stride,
    int64_t a_stride, int64_t b_stride, int64_t hk_n, int64_t dk,
    int64_t hv_n, int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;
  const int64_t i_nh = blockIdx.y;
  const int64_t i_n = i_nh / hv_n;
  const int64_t hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t vbase = i_v * bv;
  const int lane = static_cast<int>(threadIdx.x);  // one lane per value-row
  const int64_t vrow = vbase + lane;

  // Persistent-cache indirection (fla ssm_state_indices). Uniform over the block
  // (i_n fixed by blockIdx.y), so every lane branches together — no divergence.
  const int32_t slot = state_idx[i_n];
  if (slot < 0 || slot >= state_slots) {
    if (vrow < dv) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
    return;
  }

  // q' = q*q_inv*scale and k = k*k_inv, staged in shared and broadcast to every
  // lane. All warp lanes cooperate to load, reduce the L2 sums, and normalize.
  __shared__ float bq[BK];
  __shared__ float bk[BK];
  const int nthreads = static_cast<int>(blockDim.x);  // == bv (== 32 here)
  const int64_t mixed_row = i_n * mixed_stride;
  const int64_t qbase = mixed_row + hk * dk;
  const int64_t kbase = mixed_row + hk_n * dk + hk * dk;
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    const float qv = Load(mixed_qkv, qbase + ki);
    const float kv = Load(mixed_qkv, kbase + ki);
    bq[ki] = qv;
    bk[ki] = kv;
    q_sumsq += qv * qv;
    k_sumsq += kv * kv;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    q_sumsq += __shfl_xor_sync(0xffffffffu, q_sumsq, off);
    k_sumsq += __shfl_xor_sync(0xffffffffu, k_sumsq, off);
  }
  // Group the scale into q's inverse-norm exactly as the legacy kernel and the
  // CPU reference do: bq = qv * (q_inv * scale).
  const float q_scaled_inv = (1.0f / sqrtf(q_sumsq + 1e-6f)) * scale;
  const float k_inv = 1.0f / sqrtf(k_sumsq + 1e-6f);
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    bq[ki] *= q_scaled_inv;
    bk[ki] *= k_inv;
  }
  __syncwarp();

  // Per-(token, head) gate scalars — identical for every lane, computed once
  // each (cheap; avoids a shared round-trip and a second barrier).
  const float av = Load(a, i_n * a_stride + hv);
  const float b_raw = Load(b, i_n * b_stride + hv);
  const float x = av + LoadFloating(dt_bias, dt_bias_dtype, hv);
  const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
  const float g = -expf(LoadFloating(a_log, a_log_dtype, hv)) * softplus;
  const float decay = expf(g);
  const float beta = RoundToStorage<T>(1.0f / (1.0f + expf(-b_raw)));

  // Tail lanes (vrow >= dv) helped fill/normalize bq/bk above; they own no state
  // row, so they stop here (mirrors fla mask_v = o_v < V).
  if (vrow >= dv) return;

  // Load this lane's [1,BK] state row into registers, then run the single-step
  // delta-rule recurrence entirely in registers.
  const int64_t s_row =
      ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vrow) * dk;
  float sh[BK];
#pragma unroll
  for (int c = 0; c < BK; ++c) sh[c] = Load(state, s_row + c);
  float dot = 0.0f;
#pragma unroll
  for (int c = 0; c < BK; ++c) {
    sh[c] *= decay;        // b_h *= exp(g)
    dot += sh[c] * bk[c];  // dot = (S·decay) @ k
  }
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float vp = (Load(mixed_qkv, v_offset) - dot) * beta;  // (v - dot) * beta
  float output = 0.0f;
#pragma unroll
  for (int c = 0; c < BK; ++c) {
    sh[c] += vp * bk[c];     // b_h += (v-dot)*beta * k
    output += sh[c] * bq[c];  // o = b_h @ q'
  }
  Store(out, (i_n * hv_n + hv) * dv + vrow, output);
#pragma unroll
  for (int c = 0; c < BK; ++c) Store(state, s_row + c, sh[c]);
}

template <typename T, typename TState, int BK>
void LaunchGdnPackedDecodeRegTile(cudaStream_t stream, Tensor& out,
                                  const Tensor& mixed_qkv, const Tensor& a,
                                  const Tensor& b, const Tensor& a_log,
                                  const Tensor& dt_bias, Tensor& state,
                                  const Tensor& state_idx, int64_t bv,
                                  const GdnArgs& args) {
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  const int64_t hk_n = (mixed_qkv.shape[1] - hv_n * dv) / (2 * dk);
  const int64_t nv = (dv + bv - 1) / bv;
  const dim3 grid(static_cast<unsigned>(nv),
                  static_cast<unsigned>(batch * hv_n));
  // One warp per tile; the [BK] q'/k broadcast is static shared, so no dynamic
  // shared memory is requested (the register-resident state is the whole point).
  GdnPackedDecodeRegTileKernel<T, TState, BK>
      <<<grid, static_cast<unsigned>(bv), 0, stream>>>(
          out.Ptr<T>(), mixed_qkv.Ptr<T>(), a.Ptr<T>(), b.Ptr<T>(),
          a_log.data, a_log.dtype, dt_bias.data, dt_bias.dtype,
          state.Ptr<TState>(),
          state_idx.Ptr<int32_t>(), state.shape[0], mixed_qkv.stride[0],
          a.stride[0], b.stride[0], hk_n, dk, hv_n, dv, bv, args.scale);
  Check(cudaGetLastError(), "gdn packed decode reg-tile launch");
}

// Compile-time BK instantiation for the register row. Only the tested/production
// gate dims are ported (dk == 32 for the small capture matrix, dk == 128 for the
// real 27B/35B gate); an unsupported dk is a caller bug (the selection guard in
// GdnPackedDecodeKernelCuda already restricts to these).
template <typename T, typename TState>
void DispatchGdnPackedDecodeRegTile(cudaStream_t stream, Tensor& out,
                                    const Tensor& mixed_qkv, const Tensor& a,
                                    const Tensor& b, const Tensor& a_log,
                                    const Tensor& dt_bias, Tensor& state,
                                    const Tensor& state_idx, int64_t bv,
                                    int64_t dk, const GdnArgs& args) {
  if (dk == 128) {
    LaunchGdnPackedDecodeRegTile<T, TState, 128>(stream, out, mixed_qkv, a, b,
                                                 a_log, dt_bias, state,
                                                 state_idx, bv, args);
  } else if (dk == 32) {
    LaunchGdnPackedDecodeRegTile<T, TState, 32>(stream, out, mixed_qkv, a, b,
                                                a_log, dt_bias, state,
                                                state_idx, bv, args);
  } else {
    VT_CHECK(false,
             "cuda gdn_packed_decode reg-tile: unsupported dk (expected 32 or "
             "128)");
  }
}

#ifdef VLLM_CPP_TRITON
// Vendored Triton AOT packed-decode fast-path (MEASURED codegen-bound; 27B-only,
// default ON behind VT_GDN_PACKED_DECODE_TRITON, =0 rolls back). Defined after the
// AOT module loaders below; forward-declared here so GdnPackedDecodeKernelCuda
// (early in the file) can attempt it before the hand kernels. Returns true iff it
// launched.
bool TryTritonPackedDecode(cudaStream_t stream, Tensor& out,
                           const Tensor& mixed_qkv, const Tensor& a,
                           const Tensor& b, const Tensor& a_log,
                           const Tensor& dt_bias, Tensor& state,
                           const Tensor& state_idx, const GdnArgs& args);
#endif

void GdnPackedDecodeKernelCuda(Queue& q, Tensor& out,
                               const Tensor& mixed_qkv, const Tensor& a,
                               const Tensor& b, const Tensor& a_log,
                               const Tensor& dt_bias, Tensor& state,
                               const Tensor& state_idx,
                               const GdnArgs& args) {
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  if (batch == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(batch * hv_n <= kMaxGridY,
           "cuda gdn_packed_decode: too many (batch*head) blocks");
  int64_t bv = 1;
  while (bv < dv && bv < 32) bv *= 2;
  const int nw = dv >= 32 ? 8 : 1;
  const size_t total_smem =
      (2 * static_cast<size_t>(kBlock) + 4 +
       2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) *
      sizeof(float);
  VT_CHECK(total_smem <= 48 * 1024,
           "cuda gdn_packed_decode: dimensions exceed shared-memory limit");
  VT_CHECK(bv * nw <= kBlock,
           "cuda gdn_packed_decode: invalid block geometry");
  RecordGdnPackedDecodeLaunch();
  cudaStream_t stream = AsStream(q);

#ifdef VLLM_CPP_TRITON
  // Vendored Triton AOT packed-decode cubin (VT_GDN_PACKED_DECODE_TRITON, default
  // ON; =0 rolls back; 27B-only). It is vLLM's exact token-identical FLA kernel
  // (MIRROR policy). MEASURED codegen-bound: the identical register-resident
  // [BV=32,BK=128] fp32 state tile is REG:205/0-spill under Triton but
  // REG:255+STACK:48 (spills) as hand-CUDA (dgx phase1 2026-07-16). Guards every
  // dtype/stride/shape and falls back to the hand kernels below on any mismatch.
  if (TryTritonPackedDecode(stream, out, mixed_qkv, a, b, a_log, dt_bias, state,
                            state_idx, args)) {
    RecordGdnPackedDecodeTritonLaunch();
    return;
  }
#endif

  // Register-resident tiling (VT_GDN_PACKED_REG_TILE, default OFF after its failed
  // DGX proof — REG:255+STACK:48 spill / c16 700 vs 794 tok/s): one warp per
  // [BV=32, BK] tile with the state block register-resident (vLLM FLA
  // num_warps=1 / num_stages=3). Only the bv==32, dk in {32,128} geometry is
  // ported; a '0'-rollback OR any other shape stays on the legacy shared-memory
  // kernel in the SAME binary. The env is read per call (coarse decode dispatch,
  // negligible getenv — mirrors VT_GDN_DECODE_NW / VT_GDN_CHUNKED) so in-process
  // tests can flip the selection.
  const bool use_reg_tile =
      GdnPackedRegTileFlagIsOn(std::getenv("VT_GDN_PACKED_REG_TILE")) &&
      bv == 32 && (dk == 32 || dk == 128);
  if (use_reg_tile)
    RecordGdnPackedDecodeRegTileLaunch();
  else
    RecordGdnPackedDecodeLegacyLaunch();

  auto dispatch_state = [&](auto value_tag) {
    using T = decltype(value_tag);
    if (state.dtype == DType::kF32) {
      if (use_reg_tile)
        DispatchGdnPackedDecodeRegTile<T, float>(stream, out, mixed_qkv, a, b,
                                                 a_log, dt_bias, state,
                                                 state_idx, bv, dk, args);
      else
        DispatchGdnPackedDecode<T, float>(stream, out, mixed_qkv, a, b, a_log,
                                          dt_bias, state, state_idx, bv, args);
    } else if (state.dtype == DType::kF16) {
      if (use_reg_tile)
        DispatchGdnPackedDecodeRegTile<T, __half>(stream, out, mixed_qkv, a, b,
                                                  a_log, dt_bias, state,
                                                  state_idx, bv, dk, args);
      else
        DispatchGdnPackedDecode<T, __half>(stream, out, mixed_qkv, a, b, a_log,
                                           dt_bias, state, state_idx, bv, args);
    } else {
      if (use_reg_tile)
        DispatchGdnPackedDecodeRegTile<T, __nv_bfloat16>(
            stream, out, mixed_qkv, a, b, a_log, dt_bias, state, state_idx, bv,
            dk, args);
      else
        DispatchGdnPackedDecode<T, __nv_bfloat16>(
            stream, out, mixed_qkv, a, b, a_log, dt_bias, state, state_idx, bv,
            args);
    }
  };
  if (mixed_qkv.dtype == DType::kF32)
    dispatch_state(float{});
  else if (mixed_qkv.dtype == DType::kF16)
    dispatch_state(__half{});
  else
    dispatch_state(__nv_bfloat16{});
}

// ---------------------------------------------------------------------------
// Fused single-step DECODE recurrence — the (batch × Hv × value-tile) parallel
// grid mirror of fla fused_recurrent_gated_delta_rule_fwd_kernel at T==1
// (vllm/model_executor/layers/fla/ops/fused_recurrent.py:27-175; grid
// (NK=1, NV, N*HV) at :217; b_h[BV,BK] register-resident, loaded once :103-120,
// single step :122-149). Each block owns one (sequence, v-head, BV-value-tile).
// vs the sequential GdnScanKernel (which streams the [Dv,Dk] state from GLOBAL
// four times per token, strided by Dk so uncoalesced): here the [BV,Dk] slice is
// staged into shared ONCE via a COALESCED load, updated in place, and written
// back ONCE — two coalesced global passes instead of four strided ones. The grid
// spans n*Hv*NV blocks so decode parallelizes across the whole batch.
//
// state_idx (optional; mirrors fla ssm_state_indices IS_CONTINUOUS_BATCHING
// :104-116): when non-null the block reads/writes persistent-cache row
// state_idx[i_n] (idx<0 == NULL block → zero the output, skip), so the caller
// need not gather/scatter per-request state rows. When null the state is the
// compact [n,Hv,Dv,Dk] buffer and the state row == i_n.
// NW warps cooperate on ONE (seq, v-head, BV-tile)'s [BV,Dk] state by SPLITTING
// the Dk contraction: thread (vi, wk) owns value-row vi and the Dk column slice
// [wk*ceil(Dk/NW), ...). The two contractions over Dk (dot = (S·exp g)@k and
// o = S@q') are each a partial-sum over the thread's slice, then reduced across
// the NW lanes of the row's warp-group with a __shfl_xor butterfly. Shared-mem
// footprint is UNCHANGED (still the one [BV,Dk] slice), so with NW warps per
// block the SM packs NW× the resident warps for the same shared budget — the
// occupancy lever. NW==1 is bit-identical to the original single-warp kernel
// (empty reduction, whole-Dk slice per thread). The NW lanes of a row are the
// NW consecutive lanes [floor(lane/NW)*NW, +NW), aligned inside one warp (NW is
// a power of two dividing 32 and the launcher only uses NW>1 when BV==32, so a
// block is always a whole number of warps) — the xor butterfly stays in-warp.
template <typename Tin, typename Tout, typename TState, int NW, bool SWIZZLED,
          bool REGSTATE>
__global__ void GdnDecodeFusedKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                                     const float* g, const float* beta, TState* state,
                                     const int32_t* state_idx, int64_t hk_n, int64_t dk,
                                     int64_t hv_n, int64_t dv, int64_t bv, float scale) {
  static_assert(!REGSTATE || (SWIZZLED && NW == 8),
                "register state requires the production swizzled NW8 layout");
  const int64_t i_v = blockIdx.x;         // value-dim tile
  const int64_t i_nh = blockIdx.y;        // fused (sequence, v-head)
  const int64_t i_n = i_nh / hv_n;        // sequence == decode token index
  const int64_t hv = i_nh % hv_n;         // v-head
  const int64_t hk = hv / (hv_n / hk_n);  // GQA-mapped k-head
  const int64_t vbase = i_v * bv;
  const int tid = static_cast<int>(threadIdx.x);
  const int vi = tid / NW;          // value-state row within this BV tile
  const int wk = tid % NW;          // Dk-slice owner within the row's warp-group
  const int64_t vrow = vbase + vi;  // this thread's value-state row

  // Persistent-cache indirection (fla ssm_state_indices). idx<0 == NULL block
  // (our state slots are 0-indexed and slot 0 is valid, so the sentinel is < 0).
  // Uniform over the block (i_n is fixed by blockIdx.y) → no shuffle/sync hazard.
  int64_t row = i_n;
  if (state_idx != nullptr) {
    const int32_t si = state_idx[i_n];
    if (si < 0) {
      if (vrow < dv && wk == 0) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
      return;
    }
    row = si;
  }

  const int64_t sdk = SWIZZLED ? dk + NW : dk + 1;
  extern __shared__ float smem[];
  float* bq = smem;       // [dk]  q' = q*scale
  float* bk = bq + dk;    // [dk]  k
  float* sbh = bk + dk;   // [bv * sdk]  padded [BV,Dk] state slice
  const int64_t vrows = (dv - vbase) < bv ? (dv - vbase) : bv;  // valid rows in tile
  const int64_t tile = vrows * dk;                              // valid elems

  // q'(=q*scale) and k for this (token, head) — broadcast to every lane.
  const int64_t qkbase = (i_n * hk_n + hk) * dk;
  for (int64_t i = tid; i < dk; i += blockDim.x) {
    const int64_t si = SWIZZLED ? (i % 16) * 8 + i / 16 : i;
    bq[si] = Load(q, qkbase + i) * scale;
    bk[si] = Load(k, qkbase + i);
  }
  // Coalesced load of the [BV,Dk] state slice into padded shared. The persistent
  // cache TState is bf16 (mirrors vLLM's default mamba_cache_dtype=auto→model
  // dtype; fla fused_recurrent reads bf16→f32 registers→writes bf16) or f32
  // (unit test). Load() upcasts to f32; the recurrence below runs in f32.
  TState* s_head = state + (row * hv_n + hv) * dv * dk + vbase * dk;  // [<=bv, dk]
  for (int64_t e = tid; e < bv * dk; e += blockDim.x) {
    const int64_t c = e % dk;
    const int64_t sc = SWIZZLED ? (c % 16) * 8 + c / 16 : c;
    sbh[(e / dk) * sdk + sc] = e < tile ? Load(s_head, e) : 0.0f;
  }
  __syncthreads();

  float* r = sbh + static_cast<int64_t>(vi) * sdk;
  float rr[16];
  if constexpr (REGSTATE) {
    for (int64_t j = 0; j < 16; ++j) {
      rr[j] = r[GdnDecodeRegisterSharedColumn(wk, j)];
    }
  }

  // This thread's Dk column slice [c0, c1) of value-row vi (partition of [0,dk)).
  const int64_t ck = (dk + NW - 1) / NW;
  const int64_t c0 = static_cast<int64_t>(wk) * ck;
  const int64_t c1 = (c0 + ck) < dk ? (c0 + ck) : dk;

  // Every lane stays live through the shuffles (tail rows compute on the zeroed
  // slice); only the global v-load and o-store are guarded by vrow < dv.
  const float decay = expf(g[i_n * hv_n + hv]);
  const float beta_t = beta[i_n * hv_n + hv];
  float pdot = 0.0f;  // partial (S * exp(g)) @ k over this slice, fused w/ decay
  if constexpr (REGSTATE) {
    for (int64_t j = 0; j < 16; ++j) {
      const int64_t sc = GdnDecodeRegisterSharedColumn(wk, j);
      rr[j] *= decay;
      pdot += rr[j] * bk[sc];
    }
  } else if constexpr (SWIZZLED) {
    for (int64_t j = 0; j < 16; ++j) {
      const int64_t sc = j * 8 + wk;
      r[sc] *= decay;
      pdot += r[sc] * bk[sc];
    }
  } else {
    for (int64_t c = c0; c < c1; ++c) {
      r[c] *= decay;
      pdot += r[c] * bk[c];
    }
  }
  float dot = pdot;  // reduce the partials across the NW lanes of the row-group
#pragma unroll
  for (int off = 1; off < NW; off <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
  const float vv = vrow < dv ? Load(v, (i_n * hv_n + hv) * dv + vrow) : 0.0f;
  const float vp = (vv - dot) * beta_t;
  float po = 0.0f;  // partial (S + outer(v',k)) @ q' over this slice, fused w/ update
  if constexpr (REGSTATE) {
    for (int64_t j = 0; j < 16; ++j) {
      const int64_t sc = GdnDecodeRegisterSharedColumn(wk, j);
      rr[j] += vp * bk[sc];
      po += rr[j] * bq[sc];
    }
  } else if constexpr (SWIZZLED) {
    for (int64_t j = 0; j < 16; ++j) {
      const int64_t sc = j * 8 + wk;
      r[sc] += vp * bk[sc];
      po += r[sc] * bq[sc];
    }
  } else {
    for (int64_t c = c0; c < c1; ++c) {
      r[c] += vp * bk[c];
      po += r[c] * bq[c];
    }
  }
  float o = po;
#pragma unroll
  for (int off = 1; off < NW; off <<= 1) o += __shfl_xor_sync(0xffffffffu, o, off);
  if (vrow < dv && wk == 0) Store(out, (i_n * hv_n + hv) * dv + vrow, o);
  if constexpr (REGSTATE) {
    for (int64_t j = 0; j < 16; ++j) {
      r[GdnDecodeRegisterSharedColumn(wk, j)] = rr[j];
    }
  }
  __syncthreads();

  // Coalesced write-back of the updated slice from f32 registers to the
  // configured fp16/bf16/fp32 temporal cache.
  for (int64_t e = tid; e < tile; e += blockDim.x) {
    const int64_t c = e % dk;
    const int64_t sc = SWIZZLED ? (c % 16) * 8 + c / 16 : c;
    Store(s_head, e, sbh[(e / dk) * sdk + sc]);
  }
}

template <typename Tin, typename Tout, typename TState, int NW>
void LaunchGdnDecodeFusedNW(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                            const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                            const int32_t* state_idx, int64_t n, const GdnArgs& args,
                            const GdnDecodeLaunchContract& contract) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t bv = contract.value_tile;
  const dim3 grid(static_cast<unsigned>(contract.value_tiles),
                  static_cast<unsigned>(n * hv_n));
  if constexpr (NW == 8) {
    if (contract.swizzled) {
      DispatchGdnDecodeStateStorage(
          contract,
          [&](const GdnDecodeLaunchContract&) {
            GdnDecodeFusedKernel<Tin, Tout, TState, NW, true, false>
                <<<grid, static_cast<unsigned>(contract.block_threads),
                   contract.shared_bytes, s>>>(
                    out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(),
                    v.Ptr<Tin>(), g.Ptr<float>(), beta.Ptr<float>(),
                    state.Ptr<TState>(), state_idx, hk_n, dk, hv_n, dv, bv,
                    args.scale);
          },
          [&](const GdnDecodeLaunchContract&) {
            GdnDecodeFusedKernel<Tin, Tout, TState, NW, true, true>
                <<<grid, static_cast<unsigned>(contract.block_threads),
                   contract.shared_bytes, s>>>(
                    out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(),
                    v.Ptr<Tin>(), g.Ptr<float>(), beta.Ptr<float>(),
                    state.Ptr<TState>(), state_idx, hk_n, dk, hv_n, dv, bv,
                    args.scale);
          });
    } else {
      GdnDecodeFusedKernel<Tin, Tout, TState, NW, false, false>
          <<<grid, static_cast<unsigned>(contract.block_threads), contract.shared_bytes, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
          beta.Ptr<float>(), state.Ptr<TState>(), state_idx, hk_n, dk, hv_n, dv, bv,
          args.scale);
    }
  } else {
    GdnDecodeFusedKernel<Tin, Tout, TState, NW, false, false>
        <<<grid, static_cast<unsigned>(contract.block_threads), contract.shared_bytes, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
        beta.Ptr<float>(), state.Ptr<TState>(), state_idx, hk_n, dk, hv_n, dv, bv, args.scale);
  }
  Check(cudaGetLastError(), "gdn decode(fused) launch");
}

// lanes_per_row: cooperative lanes for each value row's Dk split. The launch
// contract keeps each row's NW consecutive lanes together for every value tile;
// smaller dv test corners preserve the shipped NW=1 rule.
template <typename Tin, typename Tout, typename TState>
void LaunchGdnDecodeFused(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const int32_t* state_idx, int64_t n, const GdnArgs& args,
                          const GdnDecodeLaunchContract& contract) {
  switch (contract.lanes_per_row) {
    case 2:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 2>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, contract);
      break;
    case 4:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 4>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, contract);
      break;
    case 8:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 8>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, contract);
      break;
    default:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 1>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, contract);
      break;
  }
}

// Pick the independent persistent temporal-state dtype, then forward to the
// (Tin,Tout)-typed launcher. Arithmetic remains f32 inside every variant.
template <typename Tin, typename Tout>
void LaunchGdnDecodeFusedS(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                           const int32_t* state_idx, int64_t n, const GdnArgs& args,
                           const GdnDecodeLaunchContract& contract) {
  if (state.dtype == DType::kBF16)
    LaunchGdnDecodeFused<Tin, Tout, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, contract);
  else if (state.dtype == DType::kF16)
    LaunchGdnDecodeFused<Tin, Tout, __half>(s, out, q_in, k, v, g, beta,
                                           state, state_idx, n, args, contract);
  else
    LaunchGdnDecodeFused<Tin, Tout, float>(s, out, q_in, k, v, g, beta, state, state_idx, n, args,
                                           contract);
}

// Decode dispatch. state_idx == nullptr: compact [n,Hv,Dv,Dk] state (row==i_n).
// Falls back to the sequential scan for the rare corner dims whose [BV,Dk]
// shared slice would exceed the 48 KB default budget (the real gate Dk=128 and
// every decode unit dim are well under).
void GdnDecodeFusedCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                        const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                        const int32_t* state_idx, const GdnArgs& args) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           "cuda gdn_decode: unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           "cuda gdn_decode: q/k/v dtypes must match");
  const int64_t n = q_in.shape[0], hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(n * hv_n <= kMaxGridY, "cuda gdn_decode: too many (seq×head) blocks (grid.y limit)");
  // Warps-per-block for the Dk-split occupancy lever (default 8 — measured best
  // on GB10 sm_121: raises GdnDecodeFused theoretical occupancy 10.4%→66.7% and
  // cuts per-call time ~2.2× at conc-64; A/B via env, read once per call —
  // negligible vs kernel). NW=1 == the original single-warp kernel bit-for-bit,
  // so VT_GDN_DECODE_NW=1 is the "before" in the same binary.
  int nw = 8;
  if (const char* e = std::getenv("VT_GDN_DECODE_NW")) {
    const int v_nw = std::atoi(e);
    if (v_nw == 1 || v_nw == 2 || v_nw == 4 || v_nw == 8) nw = v_nw;
  }
  cudaStream_t s = AsStream(q);
  auto launch = [&](const GdnDecodeLaunchContract& contract) {
    if (contract.shared_bytes > 48 * 1024) {
      // Corner-dim fallback (no decode test hits this; real dims are 128).
      GdnScanCuda(q, out, q_in, k, v, g, beta, state, nullptr, state_idx,
                  args, "gdn_decode");
      return;
    }
    if (q_in.dtype == DType::kF32) {
      if (out.dtype == DType::kF32)
        LaunchGdnDecodeFusedS<float, float>(s, out, q_in, k, v, g, beta, state, state_idx, n,
                                            args, contract);
      else
        LaunchGdnDecodeFusedS<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                    n, args, contract);
    } else {
      if (out.dtype == DType::kF32)
        LaunchGdnDecodeFusedS<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                    n, args, contract);
      else
        LaunchGdnDecodeFusedS<__nv_bfloat16, __nv_bfloat16>(
            s, out, q_in, k, v, g, beta, state, state_idx, n, args, contract);
    }
  };
  DispatchGdnDecodeValueTile(std::getenv("VT_GDN_DECODE_BV"),
                             std::getenv("VT_GDN_DECODE_SWIZZLE"),
                             std::getenv("VT_GDN_DECODE_REGSTATE"), dv, dk,
                             nw, launch, launch);
}

// Shared wrapper body: qsl_ptr == nullptr → decode (n = batch = state rows;
// ops.cpp validated state.shape[0] == T for decode / query_start_loc [N+1] on
// the queue's device for prefill).
void GdnScanCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                 const Tensor& g, const Tensor& beta, Tensor& state,
                 const int32_t* qsl_ptr, const int32_t* state_idx,
                 const GdnArgs& args, const char* name) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           std::string("cuda ") + name + ": unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           std::string("cuda ") + name + ": q/k/v dtypes must match");
  const int64_t n = qsl_ptr != nullptr ? state.shape[0] : q_in.shape[0];
  const int64_t hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(n <= kMaxGridY, std::string("cuda ") + name + ": too many sequences (grid.y limit)");
  VT_CHECK(2 * static_cast<size_t>(dk) * sizeof(float) <= 48 * 1024,
           std::string("cuda ") + name + ": Dk too large for the shared q'/k staging");
  cudaStream_t s = AsStream(q);
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchGdnScanState<float, float>(s, out, q_in, k, v, g, beta, state,
                                       qsl_ptr, state_idx, n, args);
    } else {
      LaunchGdnScanState<float, __nv_bfloat16>(
          s, out, q_in, k, v, g, beta, state, qsl_ptr, state_idx, n, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchGdnScanState<__nv_bfloat16, float>(
          s, out, q_in, k, v, g, beta, state, qsl_ptr, state_idx, n, args);
    } else {
      LaunchGdnScanState<__nv_bfloat16, __nv_bfloat16>(
          s, out, q_in, k, v, g, beta, state, qsl_ptr, state_idx, n, args);
    }
  }
}

// ===========================================================================
// KDA per-K-channel-decay sequential scan (kKdaGatedDeltaRule).
//
// Byte-for-byte GdnScanKernel EXCEPT the state decay is per-K-channel: GDN uses
// one scalar `decay = expf(g[t*hv_n+hv])` for the whole [Dv,Dk] state; KDA stages
// a per-K vector `decay[ki] = expf(g[(t*hv_n+hv)*dk + ki])` in shared memory and
// applies `s_row[ki] *= decay[ki]`. Ported 1:1 from FLA
// fused_recurrent_gated_delta_rule_fwd_kernel IS_KDA=True
// (`b_h *= exp(b_gk[None, :])`, third_party/flash_linear_attention/ops/
// fused_recurrent.py:136-137 @ pin 555967922). All state math f32; q/k must be
// l2-normalized by the caller (as GdnScan). The shared GDN kernels are untouched.
template <typename Tin, typename Tout, typename TState>
__global__ void KdaScanKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                              const float* g, const float* beta, TState* state,
                              const int32_t* qsl, const int32_t* state_idx,
                              int64_t state_slots, int64_t hk_n, int64_t dk,
                              int64_t hv_n, int64_t dv, float scale) {
  const int64_t s = blockIdx.y;   // sequence
  const int64_t hv = blockIdx.x;  // v-head
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        Store(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q'  then [dk] k  then [dk] per-K decay
  float* q_sh = smem;
  float* k_sh = smem + dk;
  float* d_sh = smem + 2 * dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;  // [Dv, Dk]
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Load(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Load(k, (t * hk_n + hk) * dk + i);
      d_sh[i] = expf(g[(t * hv_n + hv) * dk + i]);  // per-K-channel decay
    }
    __syncthreads();
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;  // (S * exp(g_channel)) @ k, fused with the per-channel decay
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Load(s_row, ki) * d_sh[ki];
        dot += decayed * k_sh[ki];
      }
      const float vp = (Load(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Load(s_row, ki) * d_sh[ki] + vp * k_sh[ki];
        Store(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      Store(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();  // all reads of the staged rows done before next token's load
  }
}

template <typename Tin, typename Tout>
void LaunchKdaScan(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                   const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                   const int32_t* qsl, int64_t n, const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 3 * static_cast<size_t>(dk) * sizeof(float);  // q' + k + decay
  KdaScanKernel<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<float>(), qsl, nullptr, state.shape[0], hk_n, dk, hv_n, dv,
      args.scale);
  Check(cudaGetLastError(), "kda scan launch");
}

void KdaGatedDeltaRuleKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                                 const Tensor& v, const Tensor& g, const Tensor& beta,
                                 Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  constexpr const char* name = "kda_gated_delta_rule";
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           std::string("cuda ") + name + ": unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           std::string("cuda ") + name + ": q/k/v dtypes must match");
  const int64_t n = state.shape[0];
  const int64_t hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(n <= kMaxGridY, std::string("cuda ") + name + ": too many sequences (grid.y limit)");
  VT_CHECK(3 * static_cast<size_t>(dk) * sizeof(float) <= 48 * 1024,
           std::string("cuda ") + name + ": Dk too large for the shared q'/k/decay staging");
  cudaStream_t s = AsStream(q);
  const int32_t* qsl_ptr = qsl.Ptr<int32_t>();
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchKdaScan<float, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n, args);
    else
      LaunchKdaScan<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n, args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchKdaScan<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n, args);
    else
      LaunchKdaScan<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n,
                                                  args);
  }
}

// ===========================================================================
// Chunk-parallel GDN prefill scan (gdn-semantics.md §7 "chunked oracle").
//
// Mirrors vLLM's FLA chunked gated delta rule (fla/ops/chunk.py + sub-ops).
// The sequential GdnScanKernel carries the [Dv,Dk] state one token at a time,
// so a T-token prefill has sequential depth T. This path splits the sequence
// into chunks of kChunk tokens: the intra-chunk work (steps A/B/D below) is
// INDEPENDENT across chunks (parallel across the whole grid), and only the
// cross-chunk state recurrence (step C) is sequential — depth ceil(T/kChunk).
//
// Math (derived from the sequential recurrence; exact up to fp reassociation,
// validated chunked-vs-sequential on GB10):
//   G = local cumsum of the log-decay g within each chunk (inclusive).
//   A[i,j] = beta_i * exp(G_i - G_j) * (k_i . k_j),  j<i   (chunk_scaled_dot_kkt)
//   u = (I+A)^{-1} (beta (.) v)                            (solve_tril+recompute_w_u)
//   w = (I+A)^{-1} (beta (.) exp(G) (.) k)                 (fused via fwd-subst)
//   v_new = u - w @ h_start^T                              (chunk_delta_h)
//   h_end = h_start*exp(G_last) + sum_i k_i (x) v_new_i*exp(G_last-G_i)
//   o_i   = scale * exp(G_i) * ( q_i.h_start[v]
//                              + sum_{j<=i} exp(-G_j)(q_i.k_j) v_new_j )  (chunk_fwd_o)
// (I+A) is unit lower-triangular so u,w are obtained by forward substitution,
// which replaces the explicit solve_tril inverse — numerically equivalent, and
// still fully chunk-parallel because each column solves independently.
//
// vllm.cpp original (inventory deviation §9): an original fused CUDA kernel set
// structured to map onto the FLA sub-ops (cumsum / scaled_dot_kkt+solve_tril+
// wy / chunk_delta_h / chunk_fwd_o) so a future csrc/FLA drop-in is mechanical.

// Step A — chunk_local_cumsum (fla/ops/cumsum.py): inclusive prefix sum of g
// within each chunk, per (chunk, v-head). One block per (chunk, v-head).
__global__ void GdnChunkCumsumKernel(float* gcum, const float* g, const int32_t* tok0a,
                                     const int32_t* lena, int64_t hv_n) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  if (threadIdx.x != 0) return;  // tiny (<=kChunk) serial scan per chunk-head
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  float acc = 0.0f;
  for (int64_t i = 0; i < len; ++i) {
    acc += g[(tok0 + i) * hv_n + hv];
    gcum[(tok0 + i) * hv_n + hv] = acc;
  }
}

// Step B — chunk_scaled_dot_kkt + solve_tril + recompute_w_u fused
// (fla/ops/{chunk_scaled_dot_kkt,solve_tril,wy_fast}.py). One block per
// (chunk, v-head): build K K^T in shared, then forward-substitute the two
// WY columns u (from beta.v) and w (from beta.exp(G).k). Each output column
// (vi for u, ki for w) is an independent forward solve → no cross-thread
// dependency in the substitution.
template <typename Tin, typename TSc>
__global__ void GdnChunkWUKernel(TSc* u, TSc* w, const Tin* k, const Tin* v,
                                 const float* beta, const float* gcum, const int32_t* tok0a,
                                 const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                 int64_t dv) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  __shared__ float kk[kChunk * kChunk];  // K K^T (row i, col j)
  __shared__ float ed[kChunk * kChunk];  // exp(G_i - G_j), j<=i (bounded <=1), else 0
  __shared__ float gs[kChunk];           // G_i (local cumsum)
  __shared__ float bs[kChunk];           // beta_i
  __shared__ float eg[kChunk];           // exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(k, ib + d) * Load(k, jb + d);
    kk[i * kChunk + j] = dot;
    ed[i * kChunk + j] = j <= i ? expf(gs[i] - gs[j]) : 0.0f;  // G non-increasing -> <=1
  }
  __syncthreads();
  // u column vi: u_i = beta_i (v_i - sum_{j<i} exp(G_i-G_j) KK[i,j] u_j)
  for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
    float ucol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += ed[i * kChunk + j] * kk[i * kChunk + j] * ucol[j];
      const float vi_val = Load(v, (tok0 + i) * hv_n * dv + hv * dv + vi);
      ucol[i] = bs[i] * (vi_val - s);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(u, (tok0 + i) * hv_n * dv + hv * dv + vi, ucol[i]);
  }
  // w column ki: w_i = beta_i (exp(G_i) k_i - sum_{j<i} exp(G_i-G_j) KK[i,j] w_j)
  for (int64_t ki = threadIdx.x; ki < dk; ki += blockDim.x) {
    float wcol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += ed[i * kChunk + j] * kk[i * kChunk + j] * wcol[j];
      const float ki_val = Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki);
      wcol[i] = bs[i] * (eg[i] * ki_val - s);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(w, (tok0 + i) * hv_n * dk + hv * dk + ki, wcol[i]);
  }
}

// Step C — chunk_delta_h (fla/ops/chunk_delta_h.py): the cross-chunk state
// recurrence. One block per (sequence, v-head), SEQUENTIAL over the sequence's
// chunks. The running [Dv,Dk] state lives in shared memory; each chunk snapshots
// its start state into hstate (consumed by step D), emits v_new, and advances
// the state. This is the only sequential-across-chunks kernel.
template <typename Tin>
__global__ void GdnChunkDeltaHKernel(float* state, float* hstate, float* v_new, const Tin* k,
                                     const float* u, const float* w, const float* gcum,
                                     const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                     int64_t dk, int64_t hv_n, int64_t dv) {
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  extern __shared__ float hsh[];      // [Dv, Dk] running state
  __shared__ float decay[kChunk];     // exp(G_last - G_i), i<=last (bounded <=1)
  float* s_head = state + (n * hv_n + hv) * sd;
  for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hsh[e] = s_head[e];
  __syncthreads();
  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    float* hstart = hstate + (gc * hv_n + hv) * sd;  // chunk-start snapshot for step D
    for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hstart[e] = hsh[e];
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);
    __syncthreads();
    // v_new[i,vi] = u[i,vi] - sum_ki w[i,ki] * hsh[vi,ki]
    for (int64_t idx = threadIdx.x; idx < len * dv; idx += blockDim.x) {
      const int64_t i = idx / dv, vi = idx % dv;
      float acc = u[(tok0 + i) * hv_n * dv + hv * dv + vi];
      const int64_t wb = (tok0 + i) * hv_n * dk + hv * dk;
      const float* hrow = hsh + vi * dk;
      for (int64_t d = 0; d < dk; ++d) acc -= w[wb + d] * hrow[d];
      v_new[(tok0 + i) * hv_n * dv + hv * dv + vi] = acc;
    }
    __syncthreads();
    // hsh[vi,ki] = hsh[vi,ki]*exp(G_last) + sum_i k[i,ki]*v_new[i,vi]*exp(G_last-G_i)
    for (int64_t idx = threadIdx.x; idx < sd; idx += blockDim.x) {
      const int64_t vi = idx / dk, ki = idx % dk;
      float acc = hsh[idx] * eglast;
      for (int64_t i = 0; i < len; ++i) {
        acc += Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki) *
               v_new[(tok0 + i) * hv_n * dv + hv * dv + vi] * decay[i];
      }
      hsh[idx] = acc;
    }
    __syncthreads();
  }
  for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) s_head[e] = hsh[e];  // final state
}

// Step D — chunk_fwd_o (fla/ops/chunk_o.py): the output. One block per
// (chunk, v-head): the cross-chunk term q.h_start plus the intra-chunk causal
// term sum_{j<=i} exp(G_i-G_j)(q_i.k_j) v_new_j, both scaled.
template <typename Tin, typename Tout>
__global__ void GdnChunkOKernel(Tout* out, const Tin* q, const Tin* k, const float* v_new,
                                const float* hstate, const float* gcum, const int32_t* tok0a,
                                const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                int64_t dv, float scale) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int64_t sd = dv * dk;
  const float* hstart = hstate + (gc * hv_n + hv) * sd;
  __shared__ float qk[kChunk * kChunk];  // q_i . k_j
  __shared__ float ed[kChunk * kChunk];  // exp(G_i - G_j), j<=i (bounded <=1), else 0
  __shared__ float gs[kChunk];           // G_i
  __shared__ float eg[kChunk];           // exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(q, ib + d) * Load(k, jb + d);
    qk[i * kChunk + j] = dot;
    ed[i * kChunk + j] = j <= i ? expf(gs[i] - gs[j]) : 0.0f;
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * dv; idx += blockDim.x) {
    const int64_t i = idx / dv, vi = idx % dv;
    // cross-chunk: exp(G_i) * (q_i . h_start[vi,:])
    float cross = 0.0f;
    const int64_t qb = (tok0 + i) * hk_n * dk + hk * dk;
    const float* hrow = hstart + vi * dk;
    for (int64_t d = 0; d < dk; ++d) cross += Load(q, qb + d) * hrow[d];
    // intra-chunk causal: sum_{j<=i} exp(G_i-G_j) QK[i,j] v_new_j
    float intra = 0.0f;
    for (int64_t j = 0; j <= i; ++j)
      intra += ed[i * kChunk + j] * qk[i * kChunk + j] * v_new[(tok0 + j) * hv_n * dv + hv * dv + vi];
    const float o = scale * (eg[i] * cross + intra);
    Store(out, (tok0 + i) * hv_n * dv + hv * dv + vi, o);
  }
}

// ===========================================================================
// Tensor-core (WMMA) DeltaH + ChunkO — the top prefill cost (nsys: DeltaH ~25%
// + ChunkO ~18% of 35B prefill). These replace the CUDA-core cooperative
// dot-products with WMMA 16x16 matmuls, f32 accumulate, mirroring FLA's tl.dot
// tiling (chunk_delta_h.py:176 v_new = u - w@hᵀ, :278 h += (k⊙decay)ᵀ@v_new;
// chunk_o.py:111 q@hᵀ, :113 q@kᵀ, :137 A@v_new). Templated on the data/scratch
// dtype TD: bf16 inputs use native bf16 fragments (16x16x16); f32 inputs use
// TF32 fragments (16x16x8, mantissa rounded to 19 bits) — the real 35B GDN runs
// f32, so TF32 is the path that actually hits the model. Running state H[Dv,Dk]
// stays f32 in shared (FLA keeps b_h in f32 registers); scratch dtype = TD.
namespace wmma = nvcuda::wmma;
constexpr int kWM = 16;   // WMMA tile M/N
constexpr int kNB = 32;   // DeltaH V1 output column block (Dv must be a multiple)

// Per-dtype WMMA config: fragment types, K-tile width, and a load that rounds
// f32 -> tf32 in place (bf16 loads natively).
template <typename TD>
struct WmmaCfg;
template <>
struct WmmaCfg<__nv_bfloat16> {
// bf16 WMMA fragments are complete types only for __CUDA_ARCH__ >= 800 (bf16
// tensor cores are Ampere+). The MEMBERS are guarded rather than the struct so
// the specialization still exists for every arch; every consumer is a device
// body that is itself guarded, so nothing references these on <sm_80. WK is
// inside the guard too: with the bodies compiled out it would otherwise trip
// #177-D (declared but never referenced), which this build promotes to an error
// under -Werror=all-warnings. sm_80+ is a preprocessor identity ->
// byte-identical. See .agents/specs/cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  static constexpr int WK = 16;
  using Acc = wmma::fragment<wmma::accumulator, kWM, kWM, WK, float>;
  using Arow = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, __nv_bfloat16, wmma::row_major>;
  using Acol = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, __nv_bfloat16, wmma::col_major>;
  using Brow = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, __nv_bfloat16, wmma::row_major>;
  using Bcol = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, __nv_bfloat16, wmma::col_major>;
  template <typename F>
  __device__ static void load(F& f, const __nv_bfloat16* p, int ld) {
    wmma::load_matrix_sync(f, p, ld);
  }
#endif
};
template <>
struct WmmaCfg<float> {
// TF32 is a SECOND Ampere-gated tensor-core type, and it fails harder than bf16:
// `wmma::precision::tf32` is not declared at all below sm_80, so an UNGUARDED
// alias here is a namespace-lookup error at the DEFINITION of this explicit
// specialization (not an incomplete type at a use site), which is why guarding
// only the kernel bodies is insufficient for this struct. Same shape as the bf16
// specialization above; sm_80+ is a preprocessor identity -> byte-identical.
#if __CUDA_ARCH__ >= 800
  static constexpr int WK = 8;
  using Acc = wmma::fragment<wmma::accumulator, kWM, kWM, WK, float>;
  using Arow = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, wmma::precision::tf32, wmma::row_major>;
  using Acol = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, wmma::precision::tf32, wmma::col_major>;
  using Brow = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, wmma::precision::tf32, wmma::row_major>;
  using Bcol = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, wmma::precision::tf32, wmma::col_major>;
  template <typename F>
  __device__ static void load(F& f, const float* p, int ld) {
    wmma::load_matrix_sync(f, p, ld);
    for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
  }
#endif
};

// 128-bit (int4/float4) staging: one coalesced 16-byte transfer moves N=16/
// sizeof(TD) activations (8 bf16 / 4 f32), addressed ONCE per group instead of
// per element. Uf/Sf unpack/pack to f32 lanes with the SAME __bfloat162float /
// __float2bfloat16(RNE) the scalar Load/Store use, so every staged value is
// bit-identical; Cpy is a raw 16-byte copy (bf16↔f32 round-trip is the identity
// for a value already in TD, so a direct copy reproduces Load→Store exactly).
// This is the DeltaH/ChunkO analogue of the prefill-attention K/V-staging win.
template <typename TD>
struct V128;
template <>
struct V128<__nv_bfloat16> {
// 128-bit staging is used ONLY by the WMMA bodies guarded above, so on <sm_80
// every member here is unreferenced and trips #177-D under -Werror=all-warnings.
// Guarded on the same condition; sm_80+ is a preprocessor identity.
#if __CUDA_ARCH__ >= 800
  static constexpr int N = 8;
  __device__ static void Uf(const __nv_bfloat16* p, float* f) {
    const int4 v = *reinterpret_cast<const int4*>(p);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = __bfloat162float(b[i]);
  }
  __device__ static void Sf(__nv_bfloat16* p, const float* f) {
    int4 v;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = __float2bfloat16(f[i]);
    *reinterpret_cast<int4*>(p) = v;
  }
  __device__ static void Sf0(__nv_bfloat16* p) { *reinterpret_cast<int4*>(p) = int4{0, 0, 0, 0}; }
  __device__ static void Cpy(__nv_bfloat16* d, const __nv_bfloat16* s) {
    *reinterpret_cast<int4*>(d) = *reinterpret_cast<const int4*>(s);
  }
#endif
};
template <>
struct V128<float> {
// Same rationale as the bf16 specialization above.
#if __CUDA_ARCH__ >= 800
  static constexpr int N = 4;
  __device__ static void Uf(const float* p, float* f) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    f[0] = v.x;
    f[1] = v.y;
    f[2] = v.z;
    f[3] = v.w;
  }
  __device__ static void Sf(float* p, const float* f) {
    *reinterpret_cast<float4*>(p) = float4{f[0], f[1], f[2], f[3]};
  }
  __device__ static void Sf0(float* p) { *reinterpret_cast<float4*>(p) = float4{0, 0, 0, 0}; }
  __device__ static void Cpy(float* d, const float* s) {
    *reinterpret_cast<float4*>(d) = *reinterpret_cast<const float4*>(s);
  }
#endif
};

// DeltaH (WMMA). One block per (sequence, v-head), SEQUENTIAL over chunks.
// f32 running state Hf[Dv,Dk] resident in dynamic shared; a pool is reused by
// the two matmuls. Per chunk: snapshot start-state -> TD hstate; V1 v_new=u-W@Hᵀ
// (Hᵀ via a bf16/tf32 slice Hb of Hf); V2 folds the per-row decay into K so the
// rank update H = H*exp(G_last) + V_newᵀ@(K⊙decay) reads V_new straight from
// global (no staging). f32 accumulate.
template <typename TD>
__global__ void GdnChunkDeltaHWmmaKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                         const TD* u, const TD* w, const float* gcum,
                                         const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                         int64_t dk, int64_t hv_n, int64_t dv, bool vec) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  using V = V128<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  float* Hf = reinterpret_cast<float*>(smem_raw);       // [Dv,Dk] f32 state
  char* pool = smem_raw + sd * sizeof(float);           // reused by V1/V2
  __shared__ float decay[kChunk];

  float* s_head = state + (n * hv_n + hv) * sd;
  if (vec)
    for (int64_t g = tid; g < sd / 4; g += blockDim.x)
      V128<float>::Cpy(Hf + g * 4, s_head + g * 4);
  else
    for (int64_t e = tid; e < sd; e += blockDim.x) Hf[e] = s_head[e];
  __syncthreads();

  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    TD* hstart = hstate + (gc * hv_n + hv) * sd;
    if (vec)
      for (int64_t g = tid; g < sd / V::N; g += blockDim.x) {
        float f[V::N];
#pragma unroll
        for (int j = 0; j < V::N; ++j) f[j] = Hf[g * V::N + j];
        V::Sf(hstart + g * V::N, f);
      }
    else
      for (int64_t e = tid; e < sd; e += blockDim.x) Store(hstart, e, Hf[e]);
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);
    __syncthreads();

    // --- V1: v_new = u - W @ Hᵀ  ([len,Dk]@[Dk,Dv] -> [len,Dv]) ---
    TD* Hb = reinterpret_cast<TD*>(pool);                        // [kNB,Dk] TD slice of Hf
    float* sTile = reinterpret_cast<float*>(Hb + kNB * dk);      // [kChunk,kNB] f32
    for (int64_t nb = 0; nb < dv; nb += kNB) {
      if (vec)
        for (int64_t g = tid; g < (kNB * dk) / V::N; g += blockDim.x) {
          float f[V::N];
#pragma unroll
          for (int j = 0; j < V::N; ++j) f[j] = Hf[nb * dk + g * V::N + j];
          V::Sf(Hb + g * V::N, f);
        }
      else
        for (int64_t e = tid; e < kNB * dk; e += blockDim.x)
          Store(Hb, e, Hf[(nb + e / dk) * dk + e % dk]);
      __syncthreads();
      const int ntiles = (kChunk / kWM) * (kNB / kWM);
      for (int t = warp; t < ntiles; t += nwarps) {
        const int rt = t % (kChunk / kWM), ct = t / (kChunk / kWM);
        const int i0 = rt * kWM, vi0 = static_cast<int>(nb) + ct * kWM;
        typename Cfg::Acc acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int64_t kk = 0; kk < dk; kk += WK) {
          typename Cfg::Arow a;
          typename Cfg::Bcol b;
          Cfg::load(a, w + (tok0 + i0) * hv_n * dk + hv * dk + kk, hv_n * dk);
          Cfg::load(b, Hb + (vi0 - nb) * dk + kk, dk);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(sTile + i0 * kNB + ct * kWM, acc, kNB, wmma::mem_row_major);
      }
      __syncthreads();
      if (vec)
        for (int64_t g = tid; g < (kChunk * kNB) / V::N; g += blockDim.x) {
          const int64_t base = g * V::N;
          const int64_t i = base / kNB, cvi = base % kNB;
          if (i < len) {
            float uf[V::N];
            V::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uf);
#pragma unroll
            for (int j = 0; j < V::N; ++j) uf[j] -= sTile[i * kNB + cvi + j];
            V::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uf);
          }
        }
      else
        for (int64_t e = tid; e < kChunk * kNB; e += blockDim.x) {
          const int64_t i = e / kNB, cvi = e % kNB;
          if (i < len) {
            const float uv = Load(u, (tok0 + i) * hv_n * dv + hv * dv + nb + cvi);
            Store(v_new, (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uv - sTile[i * kNB + cvi]);
          }
        }
      __syncthreads();
    }

    // --- V2: H = H*exp(G_last) + V_newᵀ @ (K⊙decay)  ([Dv,len]@[len,Dk]) ---
    TD* Kd = reinterpret_cast<TD*>(pool);                       // [kChunk,Dk] TD, decay-scaled
    if (vec)
      for (int64_t g = tid; g < (kChunk * dk) / V::N; g += blockDim.x) {
        const int64_t base = g * V::N;
        const int64_t i = base / dk, ki = base % dk;
        if (i < len) {
          float kf[V::N];
          V::Uf(k + (tok0 + i) * hk_n * dk + hk * dk + ki, kf);
          const float d = decay[i];
#pragma unroll
          for (int j = 0; j < V::N; ++j) kf[j] *= d;
          V::Sf(Kd + base, kf);
        } else {
          V::Sf0(Kd + base);
        }
      }
    else
      for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
        const int64_t i = e / dk, ki = e % dk;
        Store(Kd, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki) * decay[i] : 0.0f);
      }
    __syncthreads();
    const int ntiles2 = (dv / kWM) * (dk / kWM);
    for (int t = warp; t < ntiles2; t += nwarps) {
      const int vt = t % (dv / kWM), kt = t / (dv / kWM);
      const int vi0 = vt * kWM, ki0 = kt * kWM;
      typename Cfg::Acc acc;
      wmma::load_matrix_sync(acc, Hf + vi0 * dk + ki0, dk, wmma::mem_row_major);
      for (int x = 0; x < acc.num_elements; ++x) acc.x[x] *= eglast;
      for (int ii = 0; ii < kChunk; ii += WK) {
        typename Cfg::Acol a;  // A[vi,i] = V_new[i,vi], col-major straight from global v_new
        typename Cfg::Brow b;  // B[i,ki] = Kd[i,ki]
        Cfg::load(a, v_new + (tok0 + ii) * hv_n * dv + hv * dv + vi0, hv_n * dv);
        Cfg::load(b, Kd + ii * dk + ki0, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Hf + vi0 * dk + ki0, acc, dk, wmma::mem_row_major);
    }
    __syncthreads();
  }
  if (vec)
    for (int64_t g = tid; g < sd / 4; g += blockDim.x)
      V128<float>::Cpy(s_head + g * 4, Hf + g * 4);
  else
    for (int64_t e = tid; e < sd; e += blockDim.x) s_head[e] = Hf[e];
#else
  __trap();
#endif
}

// ----------------------------------------------------------------------------
// DeltaH (REGISTER-tiled, vt::tile Rung-1). FAITHFUL 1:1 port of FLA's
// chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (H held in ACCUMULATOR
// REGISTERS across the chunk loop, NOT in 64 KiB of shared memory) — the true
// CUTLASS-style persistent-accumulator threadblock GEMM. Frees smem for the
// N-stage cp.async input ring (STEP 2 / VT_GDN_TILE_PIPE_CPASYNC).
//
// Ported FROM:
//   vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-315
//     chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (pin e24d1b24). Grid
//     (cdiv(V,BV), N*H); b_h1/b_h2 = [BV,64] f32 register accumulators (K-split
//     at 64: K=128 -> two halves); per-seq SEQUENTIAL over chunks.
//   include/cute/arch/copy_sm80.hpp:40-193 (the cp.async ring — STEP 2 only).
// Specialized to the gate shape: K=V=128, BT=64, USE_G=1, USE_GK=0,
// USE_INITIAL_STATE=1, STORE_FINAL_STATE=1, SAVE_NEW_VALUE=1, IS_VARLEN=1,
// USE_EXP2=0. Requires dk==128 (two K-halves), dv%BV==0, blockDim.x==(BV/16)*32.
//
// Warp ownership (persistent accumulators): warp w owns v-rows
// [w*16, w*16+16) of this block's v-slice, ALL K columns -> Cfg::Acc
// h1[NK],h2[NK] (NK=64/16=4) per warp = 8 f32 accumulator fragments, live across
// the whole chunk loop. nwarps == BV/16 (each warp = 16 v-rows).
//
// Per chunk: (a) snapshot start-state -> hstate(bf16); (b) V1 b_v = u - W@b_hᵀ
// (b_hᵀ built by storing the accumulators to a transient smem tile and loading
// it col-major — the ONLY smem the persistent state touches); store v_new; (c)
// decay folded into K + b_h *= exp(g_last) (algebraically identical to FLA's
// decay-on-b_v: per-row scalar, same as GdnChunkDeltaHWmmaKernel, ref-verified);
// (e) V2 b_h += b_vᵀ @ (K⊙decay) accumulated DIRECTLY into the persistent
// fragments. Epilogue: b_h -> state(f32). w/u read straight from global; k
// staged into the decay-scaled Kd tile; v_new kept on-chip (Vbf) for V2.
template <typename TD, int BV>
__global__ void GdnChunkDeltaHRegKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                        const TD* u, const TD* w, const float* gcum,
                                        const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;         // 64
  constexpr int NK = 64 / kWM;       // 4 k-subtiles per K-half
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;

  // Transient smem arena. Region A (Hf32 | Braw | Kd) and Region B (Hbf | Vbf)
  // alias by phase (syncthreads-separated): the persistent state lives in
  // registers, so NONE of this is resident across chunks — that is what frees
  // smem for the STEP-2 ring.
  const int64_t rA0 = BV * dk * (int64_t)sizeof(float);            // Hf32
  const int64_t rA1 = BT * BV * (int64_t)sizeof(float);            // Braw
  const int64_t rA2 = BT * dk * (int64_t)sizeof(TD);               // Kd
  int64_t regA = rA0 > rA1 ? rA0 : rA1;
  regA = regA > rA2 ? regA : rA2;
  extern __shared__ char smem_raw[];
  float* Hf32 = reinterpret_cast<float*>(smem_raw);          // [BV,dk] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_raw);          // [BT,BV] f32 (aliases A)
  TD* Kd = reinterpret_cast<TD*>(smem_raw);                  // [BT,dk] TD  (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_raw + regA);          // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_raw + regA);          // [BT,BV] TD  (aliases B)
  __shared__ float decay[BT];

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;  // this warp's first (global) v-row

  // init: b_h += h0  (USE_INITIAL_STATE=1; state carries h0)
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }

  const int64_t nt = (seqlen + BT - 1) / BT;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot: store the persistent accumulators to Hf32, then convert to
    // hstate(bf16) AND to the bf16 V1 operand Hbf (one fragment-store serves
    // both). f32 path: Hbf is a plain copy (TD==float).
    for (int kt = 0; kt < NK; ++kt) {
      wmma::store_matrix_sync(Hf32 + (warp * kWM) * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
      wmma::store_matrix_sync(Hf32 + (warp * kWM) * dk + 64 + kt * kWM, h2[kt], dk,
                              wmma::mem_row_major);
    }
    __syncthreads();
    // 16-byte-vectorized snapshot: one Hf32 read serves both hstate(bf16) and the
    // V1 operand Hbf. VN = 8 bf16 / 4 f32 per coalesced transfer (V128).
    constexpr int VN = V128<TD>::N;
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
    for (int64_t e = tid * VN; e < BV * dk; e += (int64_t)blockDim.x * VN) {
      float f[VN];
#pragma unroll
      for (int j = 0; j < VN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
        f[j] = v.x;
        f[j + 1] = v.y;
        f[j + 2] = v.z;
        f[j + 3] = v.w;
      }
      const int64_t r = e / dk, c = e % dk;
      V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + c, f);
      V128<TD>::Sf(Hbf + e, f);
    }
    __syncthreads();

    // (b) V1: b_v[BT,BV] = W[BT,K] @ b_hᵀ[K,BV]  (warp w owns BT-rows [w*16:+16])
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;   // W[t,k]
        typename Cfg::Bcol b;   // b_hᵀ[k,v] = Hbf[v,k] loaded col-major
        Cfg::load(a, w + (tok0 + warp * kWM) * hv_n * dk + hv * dk + kk, hv_n * dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u, write v_new(global)+Vbf(on-chip); zero the partial-tail rows.
    // 16-byte-vectorized: a VN-group shares one row i (BV is a multiple of VN).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)blockDim.x * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float uf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, uf);
#pragma unroll
        for (int j = 0; j < VN; ++j) uf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, uf);
        V128<TD>::Sf(Vbf + e, uf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    __syncthreads();

    // (c) decay folded into K (Kd = k⊙decay, tail rows 0) + b_h *= exp(g_last)
    for (int64_t e = tid * VN; e < BT * dk; e += (int64_t)blockDim.x * VN) {
      const int64_t i = e / dk, ki = e % dk;
      if (i < len) {
        float kf[VN];
        V128<TD>::Uf(k + (tok0 + i) * hk_n * dk + hk * dk + ki, kf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) kf[j] *= d;
        V128<TD>::Sf(Kd + e, kf);
      } else {
        V128<TD>::Sf0(Kd + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h[BV,64] += b_vᵀ[BV,BT] @ Kd[BT,64] — accumulate into the
    // persistent fragments directly. matrix_a = Vbf col-major (A[v,t]=Vbf[t,v]).
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kd + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kd + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();
  }
  // epilogue: b_h -> state(f32) directly (fragment f32 -> f32 global, no convert)
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
#else
  __trap();
#endif
}

// ----------------------------------------------------------------------------
// DeltaH (REGISTER-tiled + N-stage cp.async ring, vt::tile Rung-1 STEP 2). Same
// persistent-accumulator FLA blockdim64 port as GdnChunkDeltaHRegKernel, but the
// streamed per-chunk W and K tiles come through an N-stage cp.async software
// pipeline (the ring the freed smem enables): chunk it+STAGES is prefetched while
// chunk it computes, hiding the global-load latency that the 1-block/SM occupancy
// (ring smem => 1 block) cannot hide via warp-switching. bf16 only (the f32 ring
// tiles overflow the 99 KiB opt-in); f32 falls back to the ring-less reg kernel.
//
// Ported FROM: fla/ops/chunk_delta_h.py:43-315 (pin e24d1b24, decay-on-b_v form,
// FLA's actual);  cute/arch/copy_sm80.hpp:40-193 via vt::cuda::tile::cp_async_cg
// (16B ZFILL async copy) + PipelineState. Two smem savings vs the ring-less
// kernel free room for the 2-stage ring: (1) snapshot staged in HALVES (16 KiB
// Hf32 not 32); (2) decay folded into b_v (FLA form) so V2 reads K straight from
// the ring (no separate Kd tile). ZFILL predicate masks the partial-tail rows.
template <typename TD, int BV, int STAGES>
__global__ void GdnChunkDeltaHRegRingKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                            const TD* u, const TD* w, const float* gcum,
                                            const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                            int64_t dk, int64_t hv_n, int64_t dv) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;
  constexpr int NK = 64 / kWM;
  constexpr int VN = V128<TD>::N;
  constexpr int VW = 16 / sizeof(TD);  // TD per 16B cp.async vector (8 bf16)
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  const int64_t rA = BV * 64 * (int64_t)sizeof(float);  // Hf32(half) == Braw == 16 KiB
  const int64_t rB = BV * dk * (int64_t)sizeof(TD);     // Hbf full == 16 KiB
  const int64_t tile = BT * dk * (int64_t)sizeof(TD);   // one W or K tile (16 KiB)
  extern __shared__ char smem_raw[];
  float* Hf32 = reinterpret_cast<float*>(smem_raw);  // [BV,64] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_raw);  // [BT,BV] f32 (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_raw + rA);    // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_raw + rA);    // [BT,BV] TD  (aliases B)
  char* ring = smem_raw + rA + rB;                   // STAGES * (W tile | K tile)
  __shared__ float decay[BT];

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }
  const int64_t nt = (seqlen + BT - 1) / BT;
  const int nvec = BT * (dk / VW);

  // Issue the cp.async loads of chunk `it`'s W and K into ring stage `s` (ZFILL
  // rows >= len for the partial tail). Caller issues the commit fence.
  auto issue = [&](int64_t it, int s) {
    if (it >= nt) return;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    TD* Wd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile);
    TD* Kd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile + tile);
    for (int vi = tid; vi < nvec; vi += nthreads) {
      const int r = vi / (dk / VW), g = vi % (dk / VW);
      const bool pred = r < len;
      vt::cuda::tile::cp_async_cg<16>(Wd + r * dk + g * VW,
                                      w + (tok0 + r) * hv_n * dk + hv * dk + g * VW, pred);
      vt::cuda::tile::cp_async_cg<16>(Kd + r * dk + g * VW,
                                      k + (tok0 + r) * hk_n * dk + hk * dk + g * VW, pred);
    }
  };

  // prologue: prime all STAGES stages
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue(s, s);
    vt::cuda::tile::cp_async_fence();
  }

  for (int64_t it = 0; it < nt; ++it) {
    const int stage = static_cast<int>(it % STAGES);
    vt::cuda::tile::cp_async_wait<STAGES - 1>();
    __syncthreads();
    TD* Wcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile);
    TD* Kcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile + tile);

    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += nthreads)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot in HALVES (16 KiB Hf32) -> hstate(bf16) + V1 operand Hbf.
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int coff = half * 64;
      for (int kt = 0; kt < NK; ++kt)
        wmma::store_matrix_sync(Hf32 + (warp * kWM) * 64 + kt * kWM, half == 0 ? h1[kt] : h2[kt],
                                64, wmma::mem_row_major);
      __syncthreads();
      for (int64_t e = tid * VN; e < BV * 64; e += (int64_t)nthreads * VN) {
        float f[VN];
#pragma unroll
        for (int j = 0; j < VN; j += 4) {
          const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
          f[j] = v.x;
          f[j + 1] = v.y;
          f[j + 2] = v.z;
          f[j + 3] = v.w;
        }
        const int64_t r = e / 64, c = e % 64;
        V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + coff + c, f);
        V128<TD>::Sf(Hbf + r * dk + coff + c, f);
      }
      __syncthreads();
    }

    // (b) V1: b_v[BT,BV] = W @ b_hᵀ  (W straight from the ring)
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;
        typename Cfg::Bcol b;
        Cfg::load(a, Wcur + (warp * kWM) * dk + kk, dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u (global): v_new = undecayed b_v; Vbf = decayed b_v (FLA form).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)nthreads * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float bf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] *= d;
        V128<TD>::Sf(Vbf + e, bf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h += b_vᵀ(decayed) @ K(undecayed, from the ring)
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kcur + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kcur + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();
    // prefetch chunk it+STAGES into the just-consumed stage.
    issue(it + STAGES, stage);
    vt::cuda::tile::cp_async_fence();
  }
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
#else
  __trap();
#endif
}

// DeltaH (REGISTER-tiled + N-stage TMA+mbarrier ring, vt::tile Rung-2). Identical
// persistent-accumulator FLA blockdim64 compute as GdnChunkDeltaHRegRingKernel; the
// ONLY change is the W/K chunk streaming: instead of the sm80 cp.async ring
// (Rung-1), each stage is filled by a TMA bulk-tensor copy (cp.async.bulk.tensor.3d)
// whose completion is signalled on a per-stage mbarrier (transaction-bytes) — the
// Blackwell-native async pipeline Triton emits. This is the DECISIVE A/B: same
// smem layout (plain row-major, swizzle NONE) + same WMMA, so it isolates the
// COPY/PIPELINE mechanism (cp.async vs TMA+mbarrier) from every other variable.
// bf16 only. Behind VT_GDN_TMA (default OFF).
//
// Ported FROM: fla/ops/chunk_delta_h.py:43-315 (compute, decay-on-b_v FLA form) +
// gdn_chunk_cutedsl/kernel_h.py:198-270 (the TMA warp's mbarrier producer loop:
// mbarrier_init(count) / arrive_and_expect_tx(STAGE_SIZE) / simple_tma_copy /
// mbarrier_wait(parity), num_stages ring) mapped to a unified (non-warp-specialized)
// pipeline feeding WMMA (sm_121 has no tcgen05/tmem; the DSL's MMA path is sm_100).
// TMA/mbarrier atoms: vt::cuda::tile::tma_pipeline.cuh (cited to CUTLASS there).
template <typename TD, int BV, int STAGES>
__global__ void GdnChunkDeltaHTmaKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                        const TD* u, const TD* w, const float* gcum,
                                        const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv,
                                        const __grid_constant__ CUtensorMap descW,
                                        const __grid_constant__ CUtensorMap descK) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  namespace tp = vt::cuda::tile;
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;
  constexpr int NK = 64 / kWM;
  constexpr int VN = V128<TD>::N;
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  const int64_t rA = BV * 64 * (int64_t)sizeof(float);  // Hf32(half) == Braw == 16 KiB
  const int64_t rB = BV * dk * (int64_t)sizeof(TD);     // Hbf full == 16 KiB
  const int64_t tile = BT * dk * (int64_t)sizeof(TD);   // one W or K tile (16 KiB)
  extern __shared__ __align__(128) char smem_tma[];     // 128B base for the TMA dst
  float* Hf32 = reinterpret_cast<float*>(smem_tma);  // [BV,64] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_tma);  // [BT,BV] f32 (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_tma + rA);    // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_tma + rA);    // [BT,BV] TD  (aliases B)
  char* ring = smem_tma + rA + rB;                   // STAGES * (W tile | K tile), 128B-aligned
  __shared__ float decay[BT];
  __shared__ uint64_t full_mbar[STAGES];             // one completion barrier per stage

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }
  const int64_t nt = (seqlen + BT - 1) / BT;
  const uint32_t txbytes = static_cast<uint32_t>(2 * BT * dk * (int64_t)sizeof(TD));  // W+K

  // arm the per-stage mbarriers (count=1: the single arrive.expect_tx below), then
  // fence so the init is visible to the async (TMA) proxy before any copy.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) tp::mbarrier_init(&full_mbar[s], 1);
    tp::fence_proxy_async_shared();
  }
  __syncthreads();

  // Producer: one elected thread arms stage `s`'s barrier with the W+K tx bytes and
  // issues the two TMA bulk copies (chunk `it`) into it. TMA hardware zero-fills any
  // OOB tail (partial last chunk) — masked downstream regardless.
  auto issue = [&](int64_t it, int s) {
    if (tid != 0 || it >= nt) return;
    const int64_t tok0 = begin + it * BT;
    TD* Wd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile);
    TD* Kd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile + tile);
    tp::mbarrier_arrive_expect_tx(&full_mbar[s], txbytes);
    tp::tma_load_3d(&descW, &full_mbar[s], Wd, 0, static_cast<int>(hv), static_cast<int>(tok0));
    tp::tma_load_3d(&descK, &full_mbar[s], Kd, 0, static_cast<int>(hk), static_cast<int>(tok0));
  };

  // prologue: prime all STAGES stages
#pragma unroll
  for (int s = 0; s < STAGES; ++s) issue(s, s);

  for (int64_t it = 0; it < nt; ++it) {
    const int stage = static_cast<int>(it % STAGES);
    const uint32_t phase = static_cast<uint32_t>((it / STAGES) & 1);
    tp::mbarrier_wait(&full_mbar[stage], phase);  // all threads block until W+K landed
    __syncthreads();
    TD* Wcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile);
    TD* Kcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile + tile);

    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += nthreads)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot in HALVES (16 KiB Hf32) -> hstate(bf16) + V1 operand Hbf.
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int coff = half * 64;
      for (int kt = 0; kt < NK; ++kt)
        wmma::store_matrix_sync(Hf32 + (warp * kWM) * 64 + kt * kWM, half == 0 ? h1[kt] : h2[kt],
                                64, wmma::mem_row_major);
      __syncthreads();
      for (int64_t e = tid * VN; e < BV * 64; e += (int64_t)nthreads * VN) {
        float f[VN];
#pragma unroll
        for (int j = 0; j < VN; j += 4) {
          const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
          f[j] = v.x;
          f[j + 1] = v.y;
          f[j + 2] = v.z;
          f[j + 3] = v.w;
        }
        const int64_t r = e / 64, c = e % 64;
        V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + coff + c, f);
        V128<TD>::Sf(Hbf + r * dk + coff + c, f);
      }
      __syncthreads();
    }

    // (b) V1: b_v[BT,BV] = W @ b_hᵀ  (W straight from the TMA ring)
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;
        typename Cfg::Bcol b;
        Cfg::load(a, Wcur + (warp * kWM) * dk + kk, dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u (global): v_new = undecayed b_v; Vbf = decayed b_v (FLA form).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)nthreads * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float bf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] *= d;
        V128<TD>::Sf(Vbf + e, bf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h += b_vᵀ(decayed) @ K(undecayed, from the TMA ring)
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kcur + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kcur + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();  // WAR: all consumers done reading Wcur/Kcur before the re-issue overwrites
    // prefetch chunk it+STAGES into the just-consumed stage.
    issue(it + STAGES, stage);
  }
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
#else
  __trap();
#endif
}

// ChunkO (WMMA). One block per (chunk, v-head). cross = Q@Hstartᵀ; A = Q@Kᵀ
// (decay-weighted, causal-masked); o = scale*(exp(G)*cross + A@V_new). Buffers
// alias: Ks (used only for qk) is reused as outc after qk.
template <typename TD, typename Tout>
__global__ void GdnChunkOWmmaKernel(Tout* out, const TD* q, const TD* k, const TD* v_new,
                                    const TD* hstate, const float* gcum, const int32_t* tok0a,
                                    const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                    int64_t dv, float scale, bool vec) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  using V = V128<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int64_t sd = dv * dk;
  const TD* hstart = hstate + (gc * hv_n + hv) * sd;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  const size_t r2 = static_cast<size_t>(kChunk * dk) * sizeof(TD) > static_cast<size_t>(kChunk * dv) * 4
                        ? static_cast<size_t>(kChunk * dk) * sizeof(TD)
                        : static_cast<size_t>(kChunk * dv) * 4;
  TD* Qs = reinterpret_cast<TD*>(smem_raw);                                    // [kChunk,Dk]
  char* p2 = smem_raw + static_cast<size_t>(kChunk * dk) * sizeof(TD);         // Ks | outc
  TD* Ks = reinterpret_cast<TD*>(p2);
  float* outc = reinterpret_cast<float*>(p2);
  float* qks = reinterpret_cast<float*>(p2 + r2);                             // [kChunk,kChunk]
  TD* As = reinterpret_cast<TD*>(qks + kChunk * kChunk);                       // [kChunk,kChunk]
  __shared__ float gs[kChunk], eg[kChunk];

  if (vec)
    for (int64_t g = tid; g < (kChunk * dk) / V::N; g += blockDim.x) {
      const int64_t base = g * V::N;
      const int64_t i = base / dk, d = base % dk;
      if (i < len) {
        V::Cpy(Qs + base, q + (tok0 + i) * hk_n * dk + hk * dk + d);
        V::Cpy(Ks + base, k + (tok0 + i) * hk_n * dk + hk * dk + d);
      } else {
        V::Sf0(Qs + base);
        V::Sf0(Ks + base);
      }
    }
  else
    for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
      const int64_t i = e / dk, d = e % dk;
      const bool in = i < len;
      Store(Qs, e, in ? Load(q, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
      Store(Ks, e, in ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
    }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // qk = Q @ Kᵀ  [kChunk,kChunk]
  const int nqk = (kChunk / kWM) * (kChunk / kWM);
  for (int t = warp; t < nqk; t += nwarps) {
    const int it = t % (kChunk / kWM), jt = t / (kChunk / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t kk = 0; kk < dk; kk += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Qs + i0 * dk + kk, dk);
      Cfg::load(b, Ks + j0 * dk + kk, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(qks + i0 * kChunk + j0, acc, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  // A[i,j] = exp(G_i - G_j) * qk[i,j] for j<=i (causal), else 0.
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) {
    const int64_t i = e / kChunk, j = e % kChunk;
    float a = 0.0f;
    if (i < len && j < len && j <= i) a = expf(gs[i] - gs[j]) * qks[e];
    Store(As, e, a);
  }
  __syncthreads();

  // cross = Q @ Hstartᵀ  [kChunk,Dv] -> outc (aliases Ks, now free)
  const int ncr = (kChunk / kWM) * (dv / kWM);
  for (int t = warp; t < ncr; t += nwarps) {
    const int it = t % (kChunk / kWM), vt = t / (kChunk / kWM);
    const int i0 = it * kWM, vi0 = vt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t kk = 0; kk < dk; kk += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Qs + i0 * dk + kk, dk);
      Cfg::load(b, hstart + vi0 * dk + kk, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(outc + i0 * dv + vi0, acc, dv, wmma::mem_row_major);
  }
  __syncthreads();
  for (int64_t e = tid; e < kChunk * dv; e += blockDim.x) {
    const int64_t i = e / dv;
    outc[e] *= i < len ? eg[i] : 0.0f;
  }
  __syncthreads();
  // outc = eg*cross + A @ V_new ; o = scale * outc
  for (int t = warp; t < ncr; t += nwarps) {
    const int it = t % (kChunk / kWM), vt = t / (kChunk / kWM);
    const int i0 = it * kWM, vi0 = vt * kWM;
    typename Cfg::Acc acc;
    wmma::load_matrix_sync(acc, outc + i0 * dv + vi0, dv, wmma::mem_row_major);
    for (int jj = 0; jj < kChunk; jj += WK) {
      typename Cfg::Arow a;
      typename Cfg::Brow b;
      Cfg::load(a, As + i0 * kChunk + jj, kChunk);
      Cfg::load(b, v_new + (tok0 + jj) * hv_n * dv + hv * dv + vi0, hv_n * dv);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(outc + i0 * dv + vi0, acc, dv, wmma::mem_row_major);
  }
  __syncthreads();
  for (int64_t e = tid; e < kChunk * dv; e += blockDim.x) {
    const int64_t i = e / dv, vi = e % dv;
    if (i < len) Store(out, (tok0 + i) * hv_n * dv + hv * dv + vi, scale * outc[e]);
  }
#else
  __trap();
#endif
}

// Step B (WMMA) — chunk_scaled_dot_kkt Gram on tensor cores. The scalar
// GdnChunkWUKernel spends its bulk in the O(BT²·Dk) Gram K@Kᵀ (prefill's last
// scalar-float GDN chunk kernel, ~22% GPU). This replaces that cooperative
// dot-product with a WMMA 16x16 matmul, f32 accumulate — exactly the qk=Q@Kᵀ
// block of GdnChunkOWmmaKernel but with K in both operands (mirrors FLA
// chunk_scaled_dot_kkt.py:86 b_A += tl.dot(b_kb, tl.trans(b_k))). The
// exp(G_i−G_j) diff is folded into the Gram (chunk_scaled_dot_kkt.py:92
// b_A = b_A*exp(b_g_diff)) so the shared A[i,j] holds exp(G_i−G_j)·(k_i·k_j)
// masked to the strict-lower (causal) triangle. The two WY forward
// substitutions (u from β⊙v, w from β⊙exp(G)⊙k) stay f32 — they carry a
// sequential i→i dependency (solve_tril.py:84 loop) that is not a matmul.
// TD scratch dtype: bf16 native fragments; f32 uses TF32 (WmmaCfg), the path
// the real 35B GDN (f32) hits — same precision regime as DeltaH/O.
template <typename TD>
__global__ void GdnChunkWUWmmaKernel(TD* u, TD* w, const TD* k, const TD* v, const float* beta,
                                     const float* gcum, const int32_t* tok0a, const int32_t* lena,
                                     int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  TD* Ks = reinterpret_cast<TD*>(smem_raw);                 // [kChunk,Dk] staged K (zero-padded)
  float* kk = reinterpret_cast<float*>(Ks + kChunk * dk);   // [kChunk,kChunk] Gram -> A
  __shared__ float gs[kChunk];                              // G_i (local cumsum)
  __shared__ float bs[kChunk];                              // beta_i
  __shared__ float eg[kChunk];                              // exp(G_i)

  // Stage K into shared, zero past len so tail (partial-chunk) tiles are finite
  // and never read past the k input's t_tot rows (mirrors ChunkO's Ks staging).
  for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
    const int64_t i = e / dk, d = e % dk;
    Store(Ks, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // kk = K @ Kᵀ  [kChunk,kChunk]  (Arow = K[i,:], Bcol = K[j,:] read as Kᵀ)
  const int nkk = (kChunk / kWM) * (kChunk / kWM);
  for (int t = warp; t < nkk; t += nwarps) {
    const int it = t % (kChunk / kWM), jt = t / (kChunk / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t d = 0; d < dk; d += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Ks + i0 * dk + d, dk);
      Cfg::load(b, Ks + j0 * dk + d, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(kk + i0 * kChunk + j0, acc, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  // A[i,j] = exp(G_i − G_j) · kk[i,j] for j<i (strict-lower causal), else 0.
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) {
    const int64_t i = e / kChunk, j = e % kChunk;
    kk[e] = (i < len && j < i) ? expf(gs[i] - gs[j]) * kk[e] : 0.0f;
  }
  __syncthreads();

  // u column vi: u_i = beta_i (v_i − sum_{j<i} A[i,j] u_j)  (f32 forward solve)
  for (int64_t vi = tid; vi < dv; vi += blockDim.x) {
    float ucol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float sfs = 0.0f;
      for (int64_t j = 0; j < i; ++j) sfs += kk[i * kChunk + j] * ucol[j];
      const float vi_val = Load(v, (tok0 + i) * hv_n * dv + hv * dv + vi);
      ucol[i] = bs[i] * (vi_val - sfs);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(u, (tok0 + i) * hv_n * dv + hv * dv + vi, ucol[i]);
  }
  // w column ki: w_i = beta_i (exp(G_i) k_i − sum_{j<i} A[i,j] w_j)
  for (int64_t ki = tid; ki < dk; ki += blockDim.x) {
    float wcol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float sfs = 0.0f;
      for (int64_t j = 0; j < i; ++j) sfs += kk[i * kChunk + j] * wcol[j];
      const float ki_val = Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki);
      wcol[i] = bs[i] * (eg[i] * ki_val - sfs);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(w, (tok0 + i) * hv_n * dk + hv * dk + ki, wcol[i]);
  }
#else
  __trap();
#endif
}

// Step B (WMMA, VEC path — VT_GDN_CHUNK_VEC, default). FLA-faithful WY: the
// scalar GdnChunkWUWmmaKernel above spends ~73% of its time (measured GB10
// ablation) in the O(BT²·(Dv+Dk)) per-column forward-substitution — Dv+Dk=256
// independent length-BT serial solves through LOCAL memory (ucol/wcol spill,
// STACK:256), latency/throughput-bound at 256-thread occupancy. This mirrors
// vLLM's FLA instead: fold β into A (chunk_scaled_dot_kkt.py:86-92
// b_kb=b_k*b_beta, b_A*=exp(g_diff), strict-lower mask), compute the [BT,BT]
// triangular inverse T=(I+A)⁻¹ ONCE in shared f32 (solve_tril.py — column
// forward-subst, no local spill, BT² not BT²·(Dv+Dk) work), then apply on the
// TENSOR CORES: u=T@(β⊙v), w=T@(β⊙exp(G)⊙k) (wy_fast.recompute_w_u_fwd). The
// inverse-then-apply is algebraically identical to the fused forward-solve
// ((I+A)u=β⊙v ⇒ u_i=β_i(v_i−Σ_{j<i}exp(G_i−G_j)KK[i,j]u_j)); T is rounded to TD
// (bf16 native / f32→TF32) before the WMMA apply, the same precision regime FLA
// uses (solve_tril output_dtype=k.dtype). Compact aliased shared arena (2 regions,
// reused across the Gram/inverse/apply phases → 2 blocks/SM on GB10, vs a naive
// 1-block/SM layout that leaves the serial inverse + staging occupancy-starved):
//   R0 = max(Ks[BT,Dk], Tf[BT,BT]f32): Gram-K → inverse-T(f32) → apply-operand(TD)
//   R1 = max(Am[BT,BT]f32, Tb[BT,BT]TD + Osh[BT,kOBlk]f32): Gram-out/A → T(TD)+apply-out
// The apply streams u then w in column blocks of kOBlk (per-pass operand build,
// so only ONE [BT,Dcol] operand is resident) — keeps the peak at ~40 KiB (bf16).
constexpr int kOBlk = 64;  // WMMA apply output column block (multiple of kWM)

// Blocked triangular-inverse Schur merge (mirrors FLA solve_tril.py
// merge_16x16_to_64x64_inverse_kernel:356-390, off-diagonal fill). Called by ONE
// warp to compute the [16,16] off-diagonal inverse block
//   T(bi,bj) = -T(bi,bi) @ ( Σ_{k=bj}^{bi-1} A(bi,k) @ T(k,bj) )
// on the tensor cores, where A is the strict-lower [BT,BT] f32 matrix (Am, the
// beta/decay-folded K@Kᵀ) and T is the running [BT,BT] f32 inverse (Tf). The two
// diagonal-inverse operands T(bi,bi)/T(k,bj) and the A(bi,k) blocks must already
// be resident in Tf/Am. `Pw` is a per-warp [16,16] f32 scratch. Result is stored
// (negated) into the (bi,bj) block of Tf. The inverse is carried in f32 shared for
// BOTH kernel dtypes (Am/Tf/Pw are f32; the apply rounds Tf→Tb(TD) afterwards), so
// the merges use the f32/TF32 WMMA config unconditionally — the same precision the
// f32 apply already runs at (solve_tril output_dtype=k.dtype).
// bf16/TF32 WMMA tensor-core code is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Unlike the
// __global__ kernels below, the WHOLE function is guarded rather than just its
// body: every caller is itself guarded out on <sm_80, so a body-only guard would
// leave an emitted-but-uncalled definition that trips #177-D (declared but never
// referenced) under -Werror=all-warnings. sm_80+ is a preprocessor identity ->
// byte-identical. See .agents/specs/cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
__device__ inline void WyMerge(float* Tf, const float* Am, float* Pw, int64_t BT, int bi,
                               int bj) {
  using Cfg = WmmaCfg<float>;
  constexpr int WK = Cfg::WK;
  // P = Σ_{bk=bj}^{bi-1} A(bi,bk) @ T(bk,bj)  (f32 accumulate over the block chain).
  typename Cfg::Acc accP;
  wmma::fill_fragment(accP, 0.0f);
  for (int bk = bj; bk < bi; ++bk) {
    for (int kk = 0; kk < kWM; kk += WK) {
      typename Cfg::Arow a;  // A(bi,bk)[:, kk:kk+WK]
      typename Cfg::Brow b;  // T(bk,bj)[kk:kk+WK, :]
      Cfg::load(a, Am + (bi * kWM) * BT + (bk * kWM) + kk, BT);
      Cfg::load(b, Tf + (bk * kWM + kk) * BT + (bj * kWM), BT);
      wmma::mma_sync(accP, a, b, accP);
    }
  }
  wmma::store_matrix_sync(Pw, accP, kWM, wmma::mem_row_major);
  __syncwarp();
  // T(bi,bj) = -T(bi,bi) @ P.
  typename Cfg::Acc accT;
  wmma::fill_fragment(accT, 0.0f);
  for (int kk = 0; kk < kWM; kk += WK) {
    typename Cfg::Arow a;  // T(bi,bi)[:, kk:kk+WK]
    typename Cfg::Brow b;  // P[kk:kk+WK, :]
    Cfg::load(a, Tf + (bi * kWM) * BT + (bi * kWM) + kk, BT);
    Cfg::load(b, Pw + kk * kWM, kWM);
    wmma::mma_sync(accT, a, b, accT);
  }
  for (int i = 0; i < accT.num_elements; ++i) accT.x[i] = -accT.x[i];
  wmma::store_matrix_sync(Tf + (bi * kWM) * BT + (bj * kWM), accT, BT, wmma::mem_row_major);
}
#endif

template <typename TD>
__global__ void GdnChunkWUWmmaVecKernel(TD* u, TD* w, const TD* k, const TD* v,
                                        const float* beta, const float* gcum,
                                        const int32_t* tok0a, const int32_t* lena, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv, bool blocked) {
// bf16/TF32 WMMA tensor-core body is Ampere+ (bf16 fragments and
// wmma::precision::tf32 are complete only for __CUDA_ARCH__ >= 800). Guard so
// Turing/Volta/Pascal device passes compile this TU; sm_80+ keeps the body
// verbatim (preprocessor identity -> byte-identical). Never launched on <sm_80:
// no tactic registers these for major<8. See cuda-arch-breadth-fp16.md §V0-a.
#if __CUDA_ARCH__ >= 800
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  const int64_t BT = kChunk;
  const size_t s0a = static_cast<size_t>(BT * dk) * sizeof(TD);
  const size_t s0b = static_cast<size_t>(BT * BT) * sizeof(float);
  const size_t s0 = s0a > s0b ? s0a : s0b;
  char* R0 = smem_raw;
  char* R1 = smem_raw + s0;
  TD* Ks = reinterpret_cast<TD*>(R0);       // [BT,Dk] TD  (Gram in)
  float* Tf = reinterpret_cast<float*>(R0);  // [BT,BT] f32 (inverse; aliases Ks)
  TD* Bop = reinterpret_cast<TD*>(R0);      // [BT,Dcol] TD (apply operand; aliases R0)
  float* Am = reinterpret_cast<float*>(R1);  // [BT,BT] f32 (Gram out → A)
  TD* Tb = reinterpret_cast<TD*>(R1);       // [BT,BT] TD  (T; aliases Am after inverse)
  float* Osh = reinterpret_cast<float*>(R1 + static_cast<size_t>(BT * BT) * sizeof(TD));  // [BT,kOBlk] f32
  __shared__ float gs[kChunk], bs[kChunk], eg[kChunk];

  // Stage K into shared (zero past len so tail tiles are finite).
  for (int64_t e = tid; e < BT * dk; e += blockDim.x) {
    const int64_t i = e / dk, d = e % dk;
    Store(Ks, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // Gram Am = K@Kᵀ  [BT,BT] f32 (Arow=K[i,:], Bcol=K[j,:] read as Kᵀ).
  const int nkk = (BT / kWM) * (BT / kWM);
  for (int t = warp; t < nkk; t += nwarps) {
    const int it = t % (BT / kWM), jt = t / (BT / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t d = 0; d < dk; d += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Ks + i0 * dk + d, dk);
      Cfg::load(b, Ks + j0 * dk + d, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(Am + i0 * BT + j0, acc, BT, wmma::mem_row_major);
  }
  __syncthreads();

  // A[i,j] = β_i·exp(G_i−G_j)·KK[i,j] for j<i (strict-lower causal), else 0 (in place).
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) {
    const int64_t i = e / BT, j = e % BT;
    Am[e] = (i < len && j < i) ? bs[i] * expf(gs[i] - gs[j]) * Am[e] : 0.0f;
  }
  __syncthreads();  // Ks (R0) free → Tf aliases it.

  // T = (I + A)⁻¹, unit lower-triangular. Two solvers (VT_GDN_WY_BLOCKED):
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) Tf[e] = 0.0f;
  __syncthreads();
  if (!blocked) {
    // Serial (fallback/default): column-wise forward-substitution in shared f32
    // (no local spill). Column c: T[c,c]=1, T[i,c]=−Σ_{m=c}^{i−1} A[i,m]T[m,c].
    // Only BT of the blockDim.x threads work (one dependent chain per column,
    // the longest ~BT·(BT−1)/2 deep) — the phase the blocked path shortens.
    for (int64_t c = tid; c < BT; c += blockDim.x) {
      Tf[c * BT + c] = 1.0f;
      if (c < len)
        for (int64_t i = c + 1; i < len; ++i) {
          float sfs = 0.0f;
          for (int64_t m = c; m < i; ++m) sfs += Am[i * BT + m] * Tf[m * BT + c];
          Tf[i * BT + c] = -sfs;
        }
    }
    __syncthreads();
  } else {
    // Blocked tensor-core inverse (mirrors FLA solve_tril.py
    // merge_16x16_to_64x64_inverse_kernel:238-390). BT=64 → four 16×16 diagonal
    // blocks solved by a short (≤16-deep) per-block forward-substitution, then six
    // off-diagonal blocks filled by tensor-core Schur merges (T(i,j) = −T(i,i)·
    // Σ_{j≤k<i} A(i,k)·T(k,j)), scheduled by merge distance so each phase's inputs
    // are ready. Cuts the serial critical path from ~BT-deep (~2016 dep FMAs on one
    // thread) to ~16-deep-per-block + a handful of 16×16 MMAs. Same [BT,BT] f32 Tf
    // result → identical apply/store below. Partial tails (len<BT) fall out for
    // free: Am is pre-zeroed past len (above) so out-of-range blocks invert to I /
    // merge to 0, and the apply masks rows ≥ len. Requires BT == 4·kWM (== 64).
    __shared__ __align__(16) float Pw[3][kWM * kWM];  // per-warp Schur scratch (≤3 concurrent)
    // Diagonal: 4 blocks × 16 columns = 64 independent within-block column solves.
    for (int p = tid; p < 4 * kWM; p += blockDim.x) {
      const int bd = p / kWM;         // diagonal block 0..3
      const int base = bd * kWM;      // block covers rows/cols [base, base+16)
      const int c = base + (p % kWM);  // global column
      Tf[c * BT + c] = 1.0f;
      if (c < len) {
        const int64_t iend = base + kWM < len ? base + kWM : len;
        for (int64_t i = c + 1; i < iend; ++i) {
          float sfs = 0.0f;
          for (int64_t m = c; m < i; ++m) sfs += Am[i * BT + m] * Tf[m * BT + c];
          Tf[i * BT + c] = -sfs;
        }
      }
    }
    __syncthreads();
    // Off-diagonal Schur merges, phased by distance (deps: T(i,j) needs T(k,j),
    // k<i, same column). Distance 1: (1,0)(2,1)(3,2); 2: (2,0)(3,1); 3: (3,0).
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 1, 0);
    else if (warp == 1) WyMerge(Tf, Am, Pw[1], BT, 2, 1);
    else if (warp == 2) WyMerge(Tf, Am, Pw[2], BT, 3, 2);
    __syncthreads();
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 2, 0);
    else if (warp == 1) WyMerge(Tf, Am, Pw[1], BT, 3, 1);
    __syncthreads();
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 3, 0);
    __syncthreads();
  }
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) Store(Tb, e, Tf[e]);  // T(f32) → Tb(TD), Am free
  __syncthreads();

  // Apply on the tensor cores: u = T @ (β⊙v), w = T @ (β⊙exp(G)⊙k). Each pass
  // rebuilds ONE [BT,Dcol] operand in R0 (Tf now free), streams the output in
  // kOBlk-wide column blocks through Osh (store_matrix_sync f32 → TD global).
  for (int p = 0; p < 2; ++p) {
    TD* dst = (p == 0 ? u : w);
    const int64_t dcol = (p == 0 ? dv : dk);
    const int64_t stride = (p == 0 ? hv_n * dv : hv_n * dk);
    const int64_t hoff = (p == 0 ? hv * dv : hv * dk);
    for (int64_t e = tid; e < BT * dcol; e += blockDim.x) {
      const int64_t i = e / dcol, c = e % dcol;
      float val = 0.0f;
      if (i < len)
        val = p == 0 ? bs[i] * Load(v, (tok0 + i) * hv_n * dv + hv * dv + c)
                     : bs[i] * eg[i] * Load(k, (tok0 + i) * hk_n * dk + hk * dk + c);
      Store(Bop, e, val);
    }
    __syncthreads();
    for (int64_t cb = 0; cb < dcol; cb += kOBlk) {
      const int bw = static_cast<int>(dcol - cb < kOBlk ? dcol - cb : kOBlk);  // block width
      const int ntiles = (static_cast<int>(BT) / kWM) * (bw / kWM);
      for (int t = warp; t < ntiles; t += nwarps) {
        const int it = t % (static_cast<int>(BT) / kWM), ct = t / (static_cast<int>(BT) / kWM);
        const int i0 = it * kWM, c0 = ct * kWM;
        typename Cfg::Acc acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int64_t jj = 0; jj < BT; jj += WK) {
          typename Cfg::Arow a;  // T[i0:,jj:]
          typename Cfg::Brow b;  // Bop[jj:, cb+c0:]
          Cfg::load(a, Tb + i0 * BT + jj, BT);
          Cfg::load(b, Bop + jj * dcol + cb + c0, dcol);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(Osh + i0 * bw + c0, acc, bw, wmma::mem_row_major);
      }
      __syncthreads();
      for (int64_t e = tid; e < BT * bw; e += blockDim.x) {
        const int64_t i = e / bw, c = e % bw;
        if (i < len) Store(dst, (tok0 + i) * stride + hoff + cb + c, Osh[e]);
      }
      __syncthreads();
    }
  }
#else
  __trap();
#endif
}

// Builds the per-chunk layout (tok0/len) + per-seq base-offset (boh) arrays
// DIRECTLY on the device from the device query_start_loc — the exact loop the
// host used to run after a D2H copy+sync. Single-thread (n_seq is small, ~batch
// size); writes exactly nt_tot entries into tok0/lenv (nt_tot precomputed on the
// host from the same qsl values, so grid/scratch sizing matches bit-for-bit).
// Lets the chunked-prefill path stay device-resident: no D2H sync, no host-vector
// H2D + launch sync (the prefill host-tax; see GdnArgs::query_start_loc_host).
__global__ void GdnBuildChunkMeta(int32_t* tok0, int32_t* lenv, int32_t* boh,
                                  const int32_t* qsl, int64_t n_seq) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t nt = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    boh[n] = nt;
    const int32_t begin = qsl[n];
    const int32_t len = qsl[n + 1] - begin;
    const int32_t ntn = (len + kChunk - 1) / kChunk;
    for (int32_t it = 0; it < ntn; ++it) {
      const int32_t rem = len - it * kChunk;
      tok0[nt] = begin + it * kChunk;
      lenv[nt] = rem < kChunk ? rem : kChunk;
      ++nt;
    }
  }
}

#ifdef VLLM_CPP_TRITON
// FLA chunk_indices [NT_total, 2]: per GLOBAL chunk index g (grouped by sequence,
// matching chunk_offsets/boh), stores (i_n, i_t_local). The Triton chunk_o / WU
// kernels use it to recover (sequence, local chunk) from program_id, mirroring
// fla/ops/index.py prepare_chunk_indices. Built on device from qsl + boh (both
// ready in either dev_meta or fallback mode) — one thread per sequence.
__global__ void GdnBuildChunkIndices(int32_t* cidx, const int32_t* qsl, const int32_t* boh,
                                     int64_t n_seq) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_seq) return;
  const int32_t begin = qsl[n];
  const int32_t len = qsl[n + 1] - begin;
  const int32_t ntn = (len + kChunk - 1) / kChunk;
  const int32_t base = boh[n];
  for (int32_t it = 0; it < ntn; ++it) {
    cidx[(base + it) * 2] = static_cast<int32_t>(n);
    cidx[(base + it) * 2 + 1] = it;
  }
}
#endif  // VLLM_CPP_TRITON

// Slack-only v_new zeroing (default OFF; VT_GDN_SLACK_MEMSET=1 enables). The
// full-buffer cudaMemsetAsync of v_new below zeros [0,t_pad) rows, but DeltaH V1
// writes EVERY real token row [0,t_tot) before any read that needs it, so zeroing
// those is dead (~97% of the buffer at real prefill lengths). Only the per-seq
// last-partial-chunk slack rows the WMMA tiles over-sweep (DeltaH-V2 @ :1268 /
// ChunkO @ :1397 read a full kChunk-row tile per chunk and multiply the slack
// rows by a 0 mask — Kd/As — so 0*NaN=NaN unless those rows are finite) must be
// zeroed. Mirrors FLA masking with boundary_check instead of a memset
// (chunk_delta_h.py:355 v_new=torch.empty_like(u), stored/loaded boundary_check=(0,1)).
bool GdnSlackMemsetEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_SLACK_MEMSET");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Zeros ONLY v_new's per-sequence last-partial-chunk slack rows, exactly the rows
// the WMMA tiles over-read past each sequence's real tokens:
//   [qsl[n]+seqlen_n, qsl[n]+ceil(seqlen_n/kChunk)*kChunk) x [0,dv)
// (tok0 = qsl[n]+it*kChunk tight packing, cf. GdnBuildChunkMeta). One block per
// (sequence, v-head) — grid (n_seq, hv_n), mirroring grid_seq. Idempotent: when a
// slack range overlaps the next sequence's real rows (middle sequences) DeltaH V1
// overwrites them with real finite values anyway; the invariant is only finiteness
// before the masked V2/ChunkO reads. The LAST sequence's slack range covers the
// over-read prefix of the trailing global pad; rows past it (up to t_pad) are
// never read, so they stay uninitialized (as the real-row reads already assume).
// Runs BEFORE DeltaH, same stream ordering as the full memset it replaces.
template <typename TD>
__global__ void GdnChunkVNewSlackZeroKernel(TD* v_new, const int32_t* qsl, int64_t hv_n,
                                            int64_t dv) {
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t begin = qsl[n];
  const int64_t seqlen = qsl[n + 1] - begin;
  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  const int64_t r0 = begin + seqlen;         // first slack row (== qsl[n+1])
  const int64_t r1 = begin + nt * kChunk;    // one past the last over-read row
  const int64_t nrow = r1 - r0;              // == 0 when seqlen is a kChunk multiple
  if (nrow <= 0) return;
  for (int64_t e = threadIdx.x; e < nrow * dv; e += blockDim.x) {
    const int64_t i = e / dv, d = e % dv;
    Store(v_new, (r0 + i) * hv_n * dv + hv * dv + d, 0.0f);
  }
}

#ifdef VLLM_CPP_TRITON
// DEFAULT-ON (2026-07-10) in the VLLM_CPP_TRITON build: gated by token-exact
// test_ops_gdn 31/31 + 27B AND 35B greedy 16/16 through the Triton path in BOTH
// toggle arms, and the measured every-axis win (35B conc64 +2.64% total, TTFT
// −4.1%, TPOT −2.4%, mem equal → 1.0195× vs graphed vLLM = the MVP gate; 27B
// +13pts at conc32). `VT_GDN_*_TRITON=0` opts back into the hand-C++ kernels.
// (Defined BEFORE the first TryTriton* user — main's default-ON commit placed it
// after TryTritonDeltaH, which fails to compile with VLLM_CPP_TRITON=ON.)
static bool GdnTritonEnvOn(const char* n) {
  const char* e = std::getenv(n);
  return e == nullptr || e[0] != '0';
}

int CurrentTritonAotTreeIndex() {
  const DeviceCaps& caps = GetDeviceCaps();
  if (!caps.valid) return -1;
  return TritonAotTreeIndex(caps.sm_major, caps.sm_minor);
}

bool TritonAotAvailableOnCurrentDevice() {
  return CurrentTritonAotTreeIndex() >= 0;
}

// Persistent per-stream scratch for the Triton-AOT GDN build. This mirrors the
// caching allocator vLLM gets through PyTorch: buffers grow to the largest shape
// seen on a stream and are reused by subsequent GDN layers/steps. Reuse is safe
// under single-stream forward ordering; VT_GDN_TRITON_CHUNK_POOL=0 and
// VT_GDN_TRITON_WU_POOL=0 restore per-call cudaMallocAsync/cudaFreeAsync for A/B.
struct GdnScratchBuf {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct GdnChunkScratch {
  GdnScratchBuf gcum;
  GdnScratchBuf u;
  GdnScratchBuf w;
  GdnScratchBuf v_new;
  GdnScratchBuf hstate;
  GdnScratchBuf tok0;
  GdnScratchBuf len;
  GdnScratchBuf boh;
  GdnScratchBuf cidx;
};

struct GdnWuScratch {
  GdnScratchBuf a;
  GdnScratchBuf ai;
};

struct GdnStreamScratch {
  // A shared stream's submissions must not interleave while they reuse the same
  // buffers. Normal engine use is single-threaded; this also makes the ownership
  // contract correct for concurrent host callers.
  std::mutex submit_mu;
  GdnChunkScratch chunk;
  GdnWuScratch wu;
};

struct GdnStreamKey {
  int device = 0;
  cudaStream_t stream = nullptr;

  bool operator==(const GdnStreamKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct GdnStreamKeyHash {
  size_t operator()(const GdnStreamKey& key) const {
    const size_t h1 = std::hash<int>{}(key.device);
    const size_t h2 = std::hash<void*>{}(static_cast<void*>(key.stream));
    return h1 ^ (h2 + 0x9e3779b9U + (h1 << 6U) + (h1 >> 2U));
  }
};

struct GdnDebugCounters {
  std::atomic<bool> enabled{false};
  std::atomic<uint64_t> chunk_o_f32_launches{0};
  std::atomic<uint64_t> chunk_o_bf16_launches{0};
  std::atomic<uint64_t> chunk_o_hand_launches{0};
  std::atomic<uint64_t> chunk_pool_allocations{0};
  std::atomic<uint64_t> chunk_pool_growths{0};
  std::atomic<uint64_t> chunk_pool_reuses{0};
  std::atomic<uint64_t> wu_pool_allocations{0};
  std::atomic<uint64_t> wu_pool_growths{0};
  std::atomic<uint64_t> wu_pool_reuses{0};
};

GdnDebugCounters& GdnCounters() {
  static GdnDebugCounters counters;
  return counters;
}

using GdnScratchMap =
    std::unordered_map<GdnStreamKey, std::shared_ptr<GdnStreamScratch>, GdnStreamKeyHash>;

GdnScratchMap& GdnScratchPool() {
  static GdnScratchMap pool;
  return pool;
}

std::mutex& GdnScratchPoolMutex() {
  static std::mutex mu;
  return mu;
}

std::shared_ptr<GdnStreamScratch> GdnScratchFor(cudaStream_t s) {
  int device = 0;
  Check(cudaGetDevice(&device), "gdn scratch current device");
  std::lock_guard<std::mutex> lk(GdnScratchPoolMutex());
  auto& entry = GdnScratchPool()[GdnStreamKey{device, s}];
  if (!entry) entry = std::make_shared<GdnStreamScratch>();
  return entry;
}

bool GdnTritonChunkPoolEnabled() { return GdnTritonEnvOn("VT_GDN_TRITON_CHUNK_POOL"); }

bool GdnTritonWuPoolEnabled() { return GdnTritonEnvOn("VT_GDN_TRITON_WU_POOL"); }

enum class GdnScratchKind { kChunk, kWu };

void RecordGdnScratchEvent(GdnScratchKind kind, bool allocation, bool growth) {
  GdnDebugCounters& c = GdnCounters();
  if (!c.enabled.load(std::memory_order_relaxed)) return;
  std::atomic<uint64_t>* counter = nullptr;
  if (kind == GdnScratchKind::kChunk) {
    counter = allocation ? &c.chunk_pool_allocations
                         : (growth ? &c.chunk_pool_growths : &c.chunk_pool_reuses);
  } else {
    counter = allocation ? &c.wu_pool_allocations
                         : (growth ? &c.wu_pool_growths : &c.wu_pool_reuses);
  }
  counter->fetch_add(1, std::memory_order_relaxed);
}

void* EnsureGdnScratch(GdnScratchBuf& buf, size_t bytes, cudaStream_t s, const char* what,
                       GdnScratchKind kind) {
  if (bytes > buf.bytes) {
    const bool growth = buf.ptr != nullptr;
    void* replacement = nullptr;
    Check(cudaMallocAsync(&replacement, bytes, s), what);
    void* old = buf.ptr;
    buf.ptr = replacement;
    buf.bytes = bytes;
    // Preserve ownership of the old allocation until replacement succeeds.
    // The stream-ordered free remains after all earlier users of old, while
    // subsequent launches consume replacement.
    if (old != nullptr) Check(cudaFreeAsync(old, s), "gdn scratch grow free");
    RecordGdnScratchEvent(kind, !growth, growth);
  } else {
    RecordGdnScratchEvent(kind, false, false);
  }
  return buf.ptr;
}

template <typename T>
T* EnsureGdnScratch(GdnScratchBuf& buf, size_t count, cudaStream_t s, const char* what,
                    GdnScratchKind kind) {
  return static_cast<T*>(EnsureGdnScratch(buf, count * sizeof(T), s, what, kind));
}

// triton.tools.compile emits unsynchronized lazy initialization around process-
// global CUmodule/CUfunction variables. Guard each stable dispatcher explicitly
// so concurrent first use from separate queues cannot double-load or race those
// globals. CUDA primary-context modules are process-wide for the active device.
struct GdnAotModuleOnce {
  std::array<std::once_flag, 6> deltah_h48;
  std::array<std::once_flag, 6> deltah_h32;
  std::array<std::once_flag, 6> chunko_h48;
  std::array<std::once_flag, 6> chunko_h32;
#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
  std::array<std::once_flag, 6> chunko_bf16_h48;
  std::array<std::once_flag, 6> chunko_bf16_h32;
#endif
  std::array<std::once_flag, 6> kkt_h48;
  std::array<std::once_flag, 6> tril_h48;
  std::array<std::once_flag, 6> wu_h48;
  std::array<std::once_flag, 6> kkt_h32;
  std::array<std::once_flag, 6> tril_h32;
  std::array<std::once_flag, 6> wu_h32;
  std::array<std::once_flag, 6> decode_h48;
  std::array<std::once_flag, 6> decode_h32;
  // KDA chunk-prefill family (Kimi-Linear; H=32).
  std::array<std::once_flag, 6> kda_gate_cumsum;
  std::array<std::once_flag, 6> kda_kkt_inter;
  std::array<std::once_flag, 6> kda_kkt_intra;
  std::array<std::once_flag, 6> kda_wu;
  std::array<std::once_flag, 6> kda_deltah_h32;
  std::array<std::once_flag, 6> kda_gla_o;
};

GdnAotModuleOnce& GdnAotModules() {
  static GdnAotModuleOnce modules;
  return modules;
}

void EnsureGdnDeltaHLoaded(int64_t hv_n) {
  auto& modules = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  if (hv_n == 48) {
    std::call_once(modules.deltah_h48[tree], load_gdn_deltah_h48);
  } else {
    std::call_once(modules.deltah_h32[tree], load_gdn_deltah_h32);
  }
}

void EnsureGdnChunkOF32Loaded(int64_t hv_n) {
  auto& modules = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  if (hv_n == 48) {
    std::call_once(modules.chunko_h48[tree], load_gdn_chunko_h48);
  } else {
    std::call_once(modules.chunko_h32[tree], load_gdn_chunko_h32);
  }
}

#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
void EnsureGdnChunkOBF16Loaded(int64_t hv_n) {
  auto& modules = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  if (hv_n == 48) {
    std::call_once(modules.chunko_bf16_h48[tree], load_gdn_chunko_bf16_h48);
  } else {
    std::call_once(modules.chunko_bf16_h32[tree], load_gdn_chunko_bf16_h32);
  }
}
#endif

void EnsureGdnWULoaded(int64_t hv_n) {
  auto& modules = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  if (hv_n == 48) {
    std::call_once(modules.kkt_h48[tree], load_gdn_kkt_h48);
    std::call_once(modules.tril_h48[tree], load_gdn_tril_h48);
    std::call_once(modules.wu_h48[tree], load_gdn_wu_h48);
  } else {
    std::call_once(modules.kkt_h32[tree], load_gdn_kkt_h32);
    std::call_once(modules.tril_h32[tree], load_gdn_tril_h32);
    std::call_once(modules.wu_h32[tree], load_gdn_wu_h32);
  }
}

// Dense packed-decode cubins cover Hv={48,32}. The 35B MoE model does not
// select packed decode; Hv=32 serves the dense 4B model.
void EnsureGdnPackedDecodeLoaded(int64_t hv_n) {
  auto& modules = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  if (hv_n == 48) {
    std::call_once(modules.decode_h48[tree], load_gdn_decode_h48);
  } else {
    std::call_once(modules.decode_h32[tree], load_gdn_decode_h32);
  }
}

void EnsureAllGdnAotModulesLoaded() {
  if (!TritonAotAvailableOnCurrentDevice()) return;
  for (const int64_t hv_n : {48, 32}) {
    EnsureGdnDeltaHLoaded(hv_n);
    EnsureGdnChunkOF32Loaded(hv_n);
#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
    EnsureGdnChunkOBF16Loaded(hv_n);
#endif
    EnsureGdnWULoaded(hv_n);
  }
  for (const int64_t hv_n : {48, 32}) EnsureGdnPackedDecodeLoaded(hv_n);
}

// SANCTIONED Triton AOT fast-path dispatch for the GDN packed pure-decode
// recurrence (fused_recurrent_gated_delta_rule packed decode). Returns true iff
// it launched the vendored cubin (caller then skips the hand kernels). Runtime
// toggle VT_GDN_PACKED_DECODE_TRITON: DEFAULT ON — the vendored kernel IS vLLM's
// exact token-identical FLA kernel (MIRROR policy; the sibling GDN Triton kernels
// VT_GDN_DELTAH/CHUNKO/WU_TRITON are default-ON the same way). '0' rolls back to
// the hand GdnPackedDecodeKernel in the SAME binary. Fires ONLY for the exact
// dense packed-decode call sites the two AOT specializations were baked to
// (bf16 mixed_qkv/a/b/out, f32 A_log/dt_bias/state, dk=dv=128, hk=16,
// hv in {48,32}, scale=Dk^-0.5, and the pinned contiguous strides); ANY
// dtype/shape/stride mismatch returns false so
// the preserved hand-C++ GdnPackedDecodeKernel (and the CPU reference) still
// handle it — the portable contract is intact. 35B (Hv=32, MoE) never selects
// packed decode at all (ShouldUsePackedGdnDecode requires a dense model,
// qwen3_5.cpp); Hv=32 is the dense 4B specialization.
// The state-index ABI adapter lives in the shim (skip `state_idx < 0`, slot 0
// valid) matching our cache ABI.
bool TryTritonPackedDecode(cudaStream_t stream, Tensor& out,
                           const Tensor& mixed_qkv, const Tensor& a,
                           const Tensor& b, const Tensor& a_log,
                           const Tensor& dt_bias, Tensor& state,
                           const Tensor& state_idx, const GdnArgs& args) {
  // Default ON: fire unless VT_GDN_PACKED_DECODE_TRITON leads with '0' (rollback).
  if (!TritonAotAvailableOnCurrentDevice()) return false;
  if (!GdnPackedDecodeTritonFlagIsOn(
          std::getenv("VT_GDN_PACKED_DECODE_TRITON")))
    return false;

  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  const int64_t hk_n = (mixed_qkv.shape[1] - hv_n * dv) / (2 * dk);
  // Exact dense packed-decode geometries covered by the baked specializations.
  if (dk != 128 || dv != 128 || hk_n != 16 ||
      (hv_n != 48 && hv_n != 32))
    return false;
  // Baked dtypes: bf16 activations/BA/out, f32 SSM state + A_log/dt_bias.
  if (mixed_qkv.dtype != DType::kBF16 || out.dtype != DType::kBF16 ||
      a.dtype != DType::kBF16 || b.dtype != DType::kBF16 ||
      state.dtype != DType::kF32 || a_log.dtype != DType::kF32 ||
      dt_bias.dtype != DType::kF32)
    return false;
  // Baked constexpr strides (see the shim signature). Guard every one; a mismatch
  // (e.g. non-merged BA or a padded conv buffer) falls back to the hand kernel.
  if (mixed_qkv.stride[0] != 2 * hk_n * dk + hv_n * dv ||  // 10240
      a.stride[0] != 2 * hv_n || b.stride[0] != 2 * hv_n ||  // 96 (merged BA view)
      state.stride[0] != hv_n * dv * dk ||                    // 786432
      state_idx.stride[0] != 1 || !out.IsContiguous())
    return false;
  // scale is pinned to Dk^-0.5 in the cubin (Triton AOT mis-packs fp32 scalars);
  // the model always passes exactly this, but guard defensively.
  const float baked_scale = 1.0F / std::sqrt(static_cast<float>(dk));
  if (std::fabs(args.scale - baked_scale) > 1e-6F * baked_scale) return false;

  EnsureGdnPackedDecodeLoaded(hv_n);
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t NBH = static_cast<int32_t>(batch * hv_n);  // grid-y carrier
  // state aliases h0 (read) and ht (written); the full h0 read of a (seq, head)
  // tile precedes its ht write, so passing one pointer for both is safe (mirrors
  // TryTritonDeltaH and FLA's initial_state==final_state packed-decode launch).
  const CUresult r =
      hv_n == 48
          ? gdn_decode_h48_default(
                stream, D(mixed_qkv.data), D(a.data), D(b.data), D(a_log.data),
                D(dt_bias.data), D(out.data), D(state.data), D(state.data),
                D(state_idx.data), NBH)
          : gdn_decode_h32_default(
                stream, D(mixed_qkv.data), D(a.data), D(b.data), D(a_log.data),
                D(dt_bias.data), D(out.data), D(state.data), D(state.data),
                D(state_idx.data), NBH);
  VT_CHECK(r == CUDA_SUCCESS,
           "cuda gdn packed decode(triton): launcher returned non-success");
  return true;
}

// SANCTIONED Triton AOT fast-path dispatch for GDN delta_h (the state recurrence).
// Returns true iff it launched the Triton kernel (caller then skips the hand
// path). Runtime toggle VT_GDN_DELTAH_TRITON: default ON (GdnTritonEnvOn above;
// =0 restores the hand path). Fires ONLY for the exact bf16 gate-model GDN shapes the two
// AOT specializations were pinned to (K=V=128, Hg=16, H in {48,32}); any other
// shape/dtype returns false so the preserved hand-C++ GdnChunkDeltaHRegRingKernel
// (and the CPU reference) still handle it — the portable contract is intact. The
// buffer layout is a verified 1:1 drop-in (strides checked against FLA), so the
// Triton kernel consumes the SAME device buffers our other chunk kernels produced.
bool TryTritonDeltaH(cudaStream_t s, float* state, __nv_bfloat16* hstate, __nv_bfloat16* v_new,
                     const __nv_bfloat16* k, const __nv_bfloat16* u, const __nv_bfloat16* w,
                     const float* gcum, const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                     int64_t dk, int64_t hv_n, int64_t dv, int64_t n_seq, int64_t t_tot) {
  if (!TritonAotAvailableOnCurrentDevice()) return false;
  if (!GdnTritonEnvOn("VT_GDN_DELTAH_TRITON")) return false;  // default ON (see GdnTritonEnvOn); =0 restores hand path
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  EnsureGdnDeltaHLoaded(hv_n);
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);      // overwritten per-seq (IS_VARLEN)
  const int32_t NH = static_cast<int32_t>(n_seq * hv_n);  // grid-y = N*H (grid carrier)
  // state serves as BOTH FLA h0 (loaded before the loop) and ht (stored at the end);
  // the full h0 read precedes any ht write, so aliasing the same pointer is safe.
  const CUresult r =
      hv_n == 48
          ? gdn_deltah_h48_default(s, D(k), D(u), D(w), D(v_new), D(gcum), 0, D(hstate), D(state),
                                   D(state), D(qsl), D(boh), T, NH)
          : gdn_deltah_h32_default(s, D(k), D(u), D(w), D(v_new), D(gcum), 0, D(hstate), D(state),
                                   D(state), D(qsl), D(boh), T, NH);
  VT_CHECK(r == CUDA_SUCCESS, "cuda gdn delta_h(triton): launcher returned non-success");
  return true;
}

// SANCTIONED Triton AOT fast-path for GDN chunk_o (the output kernel). Fires ONLY
// at the exact gate-model shape (K=V=128, Hg=16, H in {48,32}) with f32 out, and
// with bf16 out when VLLM_CPP_TRITON_CHUNKO_BF16 is compiled. Runtime toggle
// VT_GDN_CHUNKO_TRITON (default ON). Consumes the SAME buffers our hand
// GdnChunkOWmmaKernel does (v_new/hstate from delta_h, gcum, q/k, out) — a verified
// 1:1 drop-in — plus chunk_indices (cidx). Returns true iff it launched (caller
// then skips the hand kernel). The hand path + CPU ref remain the fallback.
template <typename Tout>
bool TryTritonChunkO(cudaStream_t s, Tout* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
                     const __nv_bfloat16* v_new, const __nv_bfloat16* hstate, const float* gcum,
                     const int32_t* qsl, const int32_t* cidx, float scale, int64_t hk_n,
                     int64_t dk, int64_t hv_n, int64_t dv, int64_t nt_tot, int64_t t_tot) {
  static_assert(std::is_same<Tout, float>::value || std::is_same<Tout, __nv_bfloat16>::value,
                "Triton chunk_o supports f32 or bf16 output");
  if (!TritonAotAvailableOnCurrentDevice()) return false;
  if (!GdnTritonEnvOn("VT_GDN_CHUNKO_TRITON")) return false;
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  if (cidx == nullptr) return false;
  // scale is PINNED to Dk^-0.5 in the kernel (Triton AOT can't take an fp32 scalar
  // arg — it mis-packs it as a double). The gate model always passes Dk^-0.5; bail
  // to the hand path if a caller ever uses a different scale, keeping it exact.
  if (std::fabs(scale - 1.0f / std::sqrt(static_cast<float>(dk))) > 1e-6f) return false;
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);   // overwritten per-seq (IS_VARLEN)
  const int32_t NT = static_cast<int32_t>(nt_tot);  // grid-y carrier (= total chunks)
  // The check/return tail lives INSIDE each `if constexpr` arm: when the bf16
  // artifacts are not compiled in (no VLLM_CPP_TRITON_CHUNKO_BF16), the bf16
  // instantiation's body is just `return false` — a shared tail after the
  // branch would be unreachable code in that instantiation (nvcc #128-D,
  // promoted to error by warnings-as-errors).
  if constexpr (std::is_same<Tout, float>::value) {
    EnsureGdnChunkOF32Loaded(hv_n);
    const CUresult r =
        hv_n == 48
            ? gdn_chunko_h48_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum), D(out), D(qsl),
                                     D(cidx), T, NT)
            : gdn_chunko_h32_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum), D(out), D(qsl),
                                     D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn chunk_o(triton): launcher returned non-success");
    GdnDebugCounters& counters = GdnCounters();
    if (counters.enabled.load(std::memory_order_relaxed)) {
      counters.chunk_o_f32_launches.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
  } else {
#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
    EnsureGdnChunkOBF16Loaded(hv_n);
    const CUresult r =
        hv_n == 48 ? gdn_chunko_bf16_h48_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum),
                                                 D(out), D(qsl), D(cidx), T, NT)
                   : gdn_chunko_bf16_h32_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum),
                                                 D(out), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn chunk_o(triton): launcher returned non-success");
    GdnDebugCounters& counters = GdnCounters();
    if (counters.enabled.load(std::memory_order_relaxed)) {
      counters.chunk_o_bf16_launches.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
#else
    return false;
#endif
  }
}

// SANCTIONED Triton AOT fast-path for the GDN WU (WY-representation) pipeline — the
// 3 FLA kernels (chunk_scaled_dot_kkt -> solve_tril -> recompute_w_u) that our
// single fused hand GdnChunkWUWmmaVecKernel mirrors. Fires ONLY at the gate shape.
// Runtime toggle VT_GDN_WU_TRITON (default ON; =0 restores the hand path).
// Allocates the two intermediate WY buffers (A raw f32, Ai bf16) as
// stream-ordered scratch, runs the 3 kernels, and
// writes u/w (the SAME buffers the hand path fills, consumed downstream by delta_h)
// — a verified 1:1 drop-in. Returns true iff it launched (caller skips the hand WU).
bool TryTritonWU(cudaStream_t s, __nv_bfloat16* u, __nv_bfloat16* w, const __nv_bfloat16* k,
                 const __nv_bfloat16* v, const float* beta, const float* gcum, const int32_t* qsl,
                 const int32_t* cidx, int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv,
                 int64_t nt_tot, int64_t t_tot, GdnWuScratch* scratch) {
  if (!TritonAotAvailableOnCurrentDevice()) return false;
  if (!GdnTritonEnvOn("VT_GDN_WU_TRITON")) return false;
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  if (cidx == nullptr) return false;
  EnsureGdnWULoaded(hv_n);
  constexpr int BT = kChunk;  // 64
  // WY intermediates [T,H,BT]: A = beta*K*Kᵀ*exp(gΔ) strictly-lower (kkt out, f32);
  // Ai = (I+A)^-1 (solve_tril out, bf16 = k.dtype). Indexed by packed token, so
  // t_tot rows suffice (block-ptr boundary_check clamps the partial-tail slack).
  const size_t na = static_cast<size_t>(t_tot) * static_cast<size_t>(hv_n) * BT;
  float* araw = nullptr;
  __nv_bfloat16* ai = nullptr;
  const bool pooled = scratch != nullptr;
  if (pooled) {
    araw = EnsureGdnScratch<float>(scratch->a, na, s, "gdn wu(triton) A pool alloc",
                                   GdnScratchKind::kWu);
    ai = EnsureGdnScratch<__nv_bfloat16>(scratch->ai, na, s, "gdn wu(triton) Ai pool alloc",
                                         GdnScratchKind::kWu);
  } else {
    Check(cudaMallocAsync(&araw, na * sizeof(float), s), "gdn wu(triton) A alloc");
    Check(cudaMallocAsync(&ai, na * sizeof(__nv_bfloat16), s), "gdn wu(triton) Ai alloc");
  }
  // solve_tril writes ONLY the 10 lower-triangular 16×16 blocks of each [BT,BT]
  // inverse; the 6 upper blocks stay 0 (FLA: `Ai = torch.zeros_like(A)`).
  // recompute_w_u then reads the FULL [BT,BT] Ai block and dots it, so the upper
  // triangle MUST be zeroed — cudaMallocAsync returns dirty pool memory in a busy
  // engine (clean in the op test / under memcheck, which is why it slipped there).
  Check(cudaMemsetAsync(ai, 0, na * sizeof(__nv_bfloat16), s), "gdn wu(triton) Ai zero");
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);
  const int32_t NT = static_cast<int32_t>(nt_tot);
  CUresult r;
  if (hv_n == 48) {
    r = gdn_kkt_h48_default(s, D(k), D(beta), D(gcum), D(araw), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.kkt(triton): non-success");
    r = gdn_tril_h48_default(s, D(araw), D(ai), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.solve_tril(triton): non-success");
    r = gdn_wu_h48_default(s, D(k), D(v), D(beta), D(w), D(u), D(ai), D(gcum), D(qsl), D(cidx), T,
                           NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.recompute(triton): non-success");
  } else {
    r = gdn_kkt_h32_default(s, D(k), D(beta), D(gcum), D(araw), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.kkt(triton): non-success");
    r = gdn_tril_h32_default(s, D(araw), D(ai), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.solve_tril(triton): non-success");
    r = gdn_wu_h32_default(s, D(k), D(v), D(beta), D(w), D(u), D(ai), D(gcum), D(qsl), D(cidx), T,
                           NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.recompute(triton): non-success");
  }
  // The hand WMMA delta_h reads full BT-row w/u tiles into the kChunk-row t_pad
  // over-allocation; the Triton recompute (boundary_check) writes only [0,t_tot),
  // so zero the appended slack rows [t_tot, t_tot+kChunk) to keep them finite
  // (0*NaN=NaN on the tensor core) — mirrors the v_new slack invariant. Disjoint
  // from the written rows, so ordering vs the recompute kernel is irrelevant.
  Check(cudaMemsetAsync(u + static_cast<size_t>(t_tot) * hv_n * dv, 0,
                        static_cast<size_t>(kChunk) * hv_n * dv * sizeof(__nv_bfloat16), s),
        "gdn wu(triton) u slack zero");
  Check(cudaMemsetAsync(w + static_cast<size_t>(t_tot) * hv_n * dk, 0,
                        static_cast<size_t>(kChunk) * hv_n * dk * sizeof(__nv_bfloat16), s),
        "gdn wu(triton) w slack zero");
  if (!pooled) {
    Check(cudaFreeAsync(araw, s), "gdn wu(triton) A free");
    Check(cudaFreeAsync(ai, s), "gdn wu(triton) Ai free");
  }
  return true;
}
#endif  // VLLM_CPP_TRITON

// ===========================================================================
// KDA CHUNK-PREFILL (kKdaChunkPrefill) — the chunked forward of the per-K-channel
// gated-delta linear attention through the vendored FLA Triton-AOT cubins, vLLM's
// ACTUAL prefill path (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate).
// Mirrors LaunchChunkedPrefill's structure but for the KDA per-K-channel kernels;
// decode (T==1) and any non-pinned shape fall back to the recurrence.
// ===========================================================================
#ifdef VLLM_CPP_TRITON
void EnsureKdaChunkLoaded() {
  auto& m = GdnAotModules();
  const std::size_t tree = static_cast<std::size_t>(CurrentTritonAotTreeIndex());
  std::call_once(m.kda_gate_cumsum[tree], load_kda_gate_cumsum);
  std::call_once(m.kda_kkt_inter[tree], load_kda_kkt_inter);
  std::call_once(m.kda_kkt_intra[tree], load_kda_kkt_intra);
  std::call_once(m.tril_h32[tree], load_gdn_tril_h32);  // REUSE gdn_tril_h32 (byte-identical sig)
  std::call_once(m.kda_wu[tree], load_kda_wu);
  std::call_once(m.kda_deltah_h32[tree], load_kda_deltah_h32);
  std::call_once(m.kda_gla_o[tree], load_kda_gla_o);
}

template <typename Tin>
__global__ void KdaToBf16Kernel(const Tin* in, __nv_bfloat16* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(Load(in, i));  // Load handles f32/bf16
}
__global__ void KdaBf16ToF32Kernel(const __nv_bfloat16* in, float* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}
void KdaCastTensorToBf16(cudaStream_t s, const Tensor& t, __nv_bfloat16* out, int64_t n) {
  const unsigned nb = static_cast<unsigned>((n + 255) / 256);
  if (t.dtype == DType::kBF16)
    KdaToBf16Kernel<__nv_bfloat16><<<nb, 256, 0, s>>>(t.Ptr<__nv_bfloat16>(), out, n);
  else
    KdaToBf16Kernel<float><<<nb, 256, 0, s>>>(t.Ptr<float>(), out, n);
  Check(cudaGetLastError(), "kda chunk cast->bf16 launch");
}

// The pinned Kimi KDA chunk-prefill orchestration (H=32, Dk=Dv=128, BT=64), a
// single prefill sequence (state [1,H,Dv,Dk]). Exactly _chunk_kda_fwd_with_
// cumulative_g's launch order. All scratch stream-ordered + freed on the queue.
void LaunchKdaChunkPrefill(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                           const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                           const Tensor& qsl, const GdnArgs& /*args*/) {
  constexpr int64_t H = 32, DK = 128, DV = 128, BT = 64;
  const int64_t T = q_in.shape[0];
  const int64_t NT = (T + BT - 1) / BT;  // chunks (single sequence)
  const int32_t Ti = static_cast<int32_t>(T);
  const int32_t NTi = static_cast<int32_t>(NT);
  const int32_t NHi = static_cast<int32_t>(H);  // n_seq(1) * H
  EnsureKdaChunkLoaded();
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };

  // ── chunk metadata: chunk_indices [NT,2] and chunk_offsets [1]=0 (single seq).
  int32_t* d_cidx = nullptr;
  int32_t* d_boh = nullptr;
  Check(cudaMallocAsync(&d_cidx, static_cast<size_t>(2 * NT) * sizeof(int32_t), s),
        "kda chunk cidx alloc");
  Check(cudaMallocAsync(&d_boh, sizeof(int32_t), s), "kda chunk boh alloc");
  Check(cudaMemsetAsync(d_boh, 0, sizeof(int32_t), s), "kda chunk boh zero");
  GdnBuildChunkIndices<<<1, 32, 0, s>>>(d_cidx, qsl.Ptr<int32_t>(), d_boh, 1);
  Check(cudaGetLastError(), "kda chunk cidx build");

  // ── bf16 casts of the Triton kernel inputs (q/k/v/g_raw/beta).
  const size_t nqk = static_cast<size_t>(T) * H * DK;
  const size_t nv = static_cast<size_t>(T) * H * DV;
  const size_t nbeta = static_cast<size_t>(T) * H;
  __nv_bfloat16 *qb = nullptr, *kb = nullptr, *vb = nullptr, *grb = nullptr, *betab = nullptr;
  Check(cudaMallocAsync(&qb, nqk * sizeof(__nv_bfloat16), s), "kda chunk qb alloc");
  Check(cudaMallocAsync(&kb, nqk * sizeof(__nv_bfloat16), s), "kda chunk kb alloc");
  Check(cudaMallocAsync(&vb, nv * sizeof(__nv_bfloat16), s), "kda chunk vb alloc");
  Check(cudaMallocAsync(&grb, nqk * sizeof(__nv_bfloat16), s), "kda chunk grb alloc");
  Check(cudaMallocAsync(&betab, nbeta * sizeof(__nv_bfloat16), s), "kda chunk betab alloc");
  KdaCastTensorToBf16(s, q_in, qb, static_cast<int64_t>(nqk));
  KdaCastTensorToBf16(s, k, kb, static_cast<int64_t>(nqk));
  KdaCastTensorToBf16(s, v, vb, static_cast<int64_t>(nv));
  KdaCastTensorToBf16(s, g_raw, grb, static_cast<int64_t>(nqk));
  KdaCastTensorToBf16(s, beta, betab, static_cast<int64_t>(nbeta));

  // ── scratch (all [0,T)-indexed; the Triton kernels boundary_check the tail).
  float* g_cum = nullptr;  // [T,H,DK] cumulative gate (exp2-space) f32
  float* a_raw = nullptr;  // [T,H,BT] strictly-lower A f32
  float* aqk = nullptr;    // [T,H,BT] causal Aqk f32
  __nv_bfloat16* a_inv = nullptr;  // [T,H,BT] (I+A)^-1 bf16
  __nv_bfloat16* w = nullptr;      // [T,H,DK] bf16
  __nv_bfloat16* u = nullptr;      // [T,H,DV] bf16
  __nv_bfloat16* kg = nullptr;     // [T,H,DK] bf16
  __nv_bfloat16* hstate = nullptr; // [NT,H,DV,DK] bf16
  __nv_bfloat16* v_new = nullptr;  // [T,H,DV] bf16
  __nv_bfloat16* o_bf = nullptr;   // [T,H,DV] bf16 output
  const size_t na = static_cast<size_t>(T) * H * BT;
  const size_t nh = static_cast<size_t>(NT) * H * DV * DK;
  Check(cudaMallocAsync(&g_cum, nqk * sizeof(float), s), "kda chunk g_cum alloc");
  Check(cudaMallocAsync(&a_raw, na * sizeof(float), s), "kda chunk a_raw alloc");
  Check(cudaMallocAsync(&aqk, na * sizeof(float), s), "kda chunk aqk alloc");
  Check(cudaMallocAsync(&a_inv, na * sizeof(__nv_bfloat16), s), "kda chunk a_inv alloc");
  Check(cudaMallocAsync(&w, nqk * sizeof(__nv_bfloat16), s), "kda chunk w alloc");
  Check(cudaMallocAsync(&u, nv * sizeof(__nv_bfloat16), s), "kda chunk u alloc");
  Check(cudaMallocAsync(&kg, nqk * sizeof(__nv_bfloat16), s), "kda chunk kg alloc");
  Check(cudaMallocAsync(&hstate, nh * sizeof(__nv_bfloat16), s), "kda chunk hstate alloc");
  Check(cudaMallocAsync(&v_new, nv * sizeof(__nv_bfloat16), s), "kda chunk v_new alloc");
  Check(cudaMallocAsync(&o_bf, nv * sizeof(__nv_bfloat16), s), "kda chunk o_bf alloc");
  // kkt writes only the lower/diagonal blocks of A; solve_tril writes only the 10
  // lower 16x16 blocks of A_inv, and recompute_w_u reads the FULL [BT,BT] A_inv —
  // so zero both (cudaMallocAsync returns dirty pool memory in a busy engine).
  Check(cudaMemsetAsync(a_raw, 0, na * sizeof(float), s), "kda chunk a_raw zero");
  Check(cudaMemsetAsync(a_inv, 0, na * sizeof(__nv_bfloat16), s), "kda chunk a_inv zero");

  const CUdeviceptr Dqsl = D(qsl.Ptr<int32_t>());
  const CUdeviceptr Dcidx = D(d_cidx);
  CUresult r;
  // (1) fused decay-gate + chunk cumsum: g_raw_bf16 (+a_log,dt_bias) -> g_cum.
  r = kda_gate_cumsum_default(s, D(grb), D(a_log.data), D(g_cum), D(dt_bias.data), Dqsl, Dcidx,
                              Ti, NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk gate_cumsum: non-success");
  // (2)+(3) per-K-channel-gated K.Kᵀ + q.kᵀ -> A(strictly-lower), Aqk(causal).
  r = kda_kkt_inter_default(s, D(qb), D(kb), D(g_cum), D(betab), D(a_raw), D(aqk), Dqsl, Dcidx, Ti,
                            NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk kkt_inter: non-success");
  r = kda_kkt_intra_default(s, D(qb), D(kb), D(g_cum), D(betab), D(a_raw), D(aqk), Dqsl, Dcidx, Ti,
                            NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk kkt_intra: non-success");
  // (4) invert the strictly-lower A (WY solve) -> A_inv (REUSE gdn_tril_h32).
  r = gdn_tril_h32_default(s, D(a_raw), D(a_inv), Dqsl, Dcidx, Ti, NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk solve_tril: non-success");
  // (5) recompute W, U (+ kg = k*exp2(gn-gk)); q/qg dead (STORE_QG=0).
  r = kda_wu_default(s, 0, D(kb), 0, D(kg), D(vb), D(betab), D(w), D(u), D(a_inv), D(g_cum), Dqsl,
                     Dcidx, Ti, NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk recompute_w_u: non-success");
  // (6) chunked hidden-state scan: k=kg, v=u, gk=g_cum (g dead), h0=ht=state
  //     -> h(snapshots), v_new, final state. state is fresh zeros (caller).
  r = kda_deltah_h32_default(s, D(kg), D(u), D(w), D(v_new), 0, D(g_cum), D(hstate),
                             D(state.data), D(state.data), Dqsl, D(d_boh), Ti, NHi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk delta_h: non-success");
  // (7) GLA-style output with per-K gk decay -> o_bf.
  r = kda_gla_o_default(s, D(qb), D(v_new), D(g_cum), D(hstate), D(o_bf), D(aqk), Dqsl, Dcidx, Ti,
                        NTi);
  VT_CHECK(r == CUDA_SUCCESS, "cuda kda chunk gla_o: non-success");

  // ── out <- o_bf (cast to f32 or copy bf16).
  if (out.dtype == DType::kF32) {
    const unsigned nbo = static_cast<unsigned>((static_cast<int64_t>(nv) + 255) / 256);
    KdaBf16ToF32Kernel<<<nbo, 256, 0, s>>>(o_bf, out.Ptr<float>(), static_cast<int64_t>(nv));
    Check(cudaGetLastError(), "kda chunk out cast launch");
  } else {
    Check(cudaMemcpyAsync(out.data, o_bf, nv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s),
          "kda chunk out copy");
  }

  for (void* p : {static_cast<void*>(o_bf), static_cast<void*>(v_new), static_cast<void*>(hstate),
                  static_cast<void*>(kg), static_cast<void*>(u), static_cast<void*>(w),
                  static_cast<void*>(a_inv), static_cast<void*>(aqk), static_cast<void*>(a_raw),
                  static_cast<void*>(g_cum), static_cast<void*>(betab), static_cast<void*>(grb),
                  static_cast<void*>(vb), static_cast<void*>(kb), static_cast<void*>(qb),
                  static_cast<void*>(d_boh), static_cast<void*>(d_cidx)})
    Check(cudaFreeAsync(p, s), "kda chunk scratch free");
}
#endif  // VLLM_CPP_TRITON

// Fused per-K-channel decay gate (kda_gate_cumsum's per-token gate, pre-cumsum):
// g_dec[t,h,d] = -exp(a_log[h]) * softplus(g_raw[t,h,d] + dt_bias[h,d]) (beta=1).
// Feeds the recurrence fallback when the chunk cubins are unavailable/mismatched.
__global__ void KdaGateFromRawKernel(const float* g_raw, const float* a_log, const float* dt_bias,
                                     bool has_bias, float* g_dec, int64_t T, int64_t H, int64_t Dk) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= T * H * Dk) return;
  const int64_t d = idx % Dk;
  const int64_t h = (idx / Dk) % H;
  float x = g_raw[idx];
  if (has_bias) x += dt_bias[h * Dk + d];
  const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus(beta=1)
  g_dec[idx] = -expf(a_log[h]) * sp;
}

void KdaChunkPrefillKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                               const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                               const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                               const Tensor& qsl, const GdnArgs& args) {
  const int64_t T = q_in.shape[0];
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  // hk_n's ONLY use is the geom_ok check below, which sits inside
  // `#ifdef VLLM_CPP_TRITON`. In a NON-Triton build that use is compiled out and
  // nvcc's #177-D -- an ERROR under -Werror -- fails the whole CUDA build. That
  // configuration dependence is why this stayed green wherever Triton is on.
  (void)hk_n;
  const int64_t hv_n = state.shape[1], dv = state.shape[2];
  if (T == 0 || hv_n == 0 || dv == 0) return;
  cudaStream_t s = AsStream(q);
  const bool has_bias = dt_bias.shape[0] != 0;
#ifdef VLLM_CPP_TRITON
  // Fire the vendored chunk cubins only at the exact pinned Kimi KDA geometry, with
  // the baked scale, a real prompt (T>1), dt_bias present (HAS_BIAS=1 baked), and
  // the runtime toggle on (VT_KDA_CHUNK_TRITON default ON; =0 restores recurrence).
  const float baked_scale = 1.0f / std::sqrt(static_cast<float>(dk));
  const bool geom_ok = (hk_n == 32 && hv_n == 32 && dk == 128 && dv == 128 && has_bias &&
                        std::fabs(args.scale - baked_scale) <= 1e-6f * baked_scale && T > 1);
  if (geom_ok && TritonAotAvailableOnCurrentDevice() &&
      GdnTritonEnvOn("VT_KDA_CHUNK_TRITON")) {
    LaunchKdaChunkPrefill(s, out, q_in, k, v, g_raw, beta, a_log, dt_bias, state, qsl, args);
    return;
  }
#endif
  // Fallback (non-Triton build, non-pinned shape, decode, or no bias): fuse the gate
  // on device then run the exact per-K-channel recurrence (byte-for-byte the decode
  // kernel). Produces the same math as the chunk path up to reduction order.
  float* g_dec = nullptr;
  const int64_t ng = T * hv_n * dk;
  Check(cudaMallocAsync(&g_dec, static_cast<size_t>(ng) * sizeof(float), s),
        "kda chunk fallback g_dec alloc");
  const unsigned nb = static_cast<unsigned>((ng + 255) / 256);
  KdaGateFromRawKernel<<<nb, 256, 0, s>>>(g_raw.Ptr<float>(), a_log.Ptr<float>(),
                                          has_bias ? dt_bias.Ptr<float>() : nullptr, has_bias,
                                          g_dec, T, hv_n, dk);
  Check(cudaGetLastError(), "kda chunk fallback gate launch");
  Tensor g_t = g_raw;  // same [T,Hv,Dk] f32 contiguous metadata
  g_t.data = g_dec;
  KdaGatedDeltaRuleKernelCuda(q, out, q_in, k, v, g_t, beta, state, qsl, args);
  Check(cudaFreeAsync(g_dec, s), "kda chunk fallback g_dec free");
}

template <typename Tin, typename Tout>
void LaunchChunkedPrefill(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const std::vector<int32_t>& tok0h,
                          const std::vector<int32_t>& lenh, const std::vector<int32_t>& bohh,
                          const Tensor& qsl, int64_t n_seq, int64_t nt_tot, bool dev_meta,
                          const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  // hk_n is kept for the shape mapping it documents (q_in is [T, hk_n, dk]), but
  // nothing below reads it, and nvcc's #177-D is an ERROR under -Werror -- so this
  // declaration alone was failing every CUDA build.
  (void)hk_n;
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t t_tot = out.shape[0];

  // Scratch dtype tracks the input dtype: f32 scratch consumed by TF32
  // fragments (the real 35B GDN, f32), bf16 scratch by native bf16 fragments
  // (the coupled bf16 path). The tensor cores read the scratch natively.
  using TSc = Tin;
  // WMMA is used for both dtypes when the head dims tile cleanly (the routing
  // in GdnPrefillKernelCuda guarantees this for bf16; f32 corners fall back to
  // the CUDA-core chunked kernels below).
  const bool wmma = (dk % kWM == 0 && dv % kNB == 0);
  // u/w/v_new are over-allocated by kChunk rows so the WMMA kernels can read
  // full BT-row tiles past a partial-tail chunk without going out of bounds
  // (the extra rows contribute nothing: masked/discarded).
  const int64_t t_pad = t_tot + kChunk;

  // Device scratch (stream-ordered). In Triton-AOT builds this is pooled per stream;
  // otherwise it is freed after the kernels on the same stream.
  float* gcum = nullptr;
  TSc* u = nullptr;
  TSc* w = nullptr;
  TSc* v_new = nullptr;
  TSc* hstate = nullptr;
  int32_t* d_tok0 = nullptr;
  int32_t* d_len = nullptr;
  int32_t* d_boh = nullptr;
#ifdef VLLM_CPP_TRITON
  GdnChunkScratch* chunk_sc = nullptr;
  const bool chunk_pool = GdnTritonChunkPoolEnabled();
  const bool wu_pool = GdnTritonWuPoolEnabled();
  std::shared_ptr<GdnStreamScratch> stream_sc;
  std::unique_lock<std::mutex> stream_scratch_lock;
  if (chunk_pool || wu_pool) {
    stream_sc = GdnScratchFor(s);
    stream_scratch_lock = std::unique_lock<std::mutex>(stream_sc->submit_mu);
  }
  if (chunk_pool) {
    chunk_sc = &stream_sc->chunk;
    gcum = EnsureGdnScratch<float>(chunk_sc->gcum, static_cast<size_t>(t_tot * hv_n), s,
                                   "gdn chunked gcum pool alloc", GdnScratchKind::kChunk);
    u = EnsureGdnScratch<TSc>(chunk_sc->u, static_cast<size_t>(t_pad * hv_n * dv), s,
                              "gdn chunked u pool alloc", GdnScratchKind::kChunk);
    w = EnsureGdnScratch<TSc>(chunk_sc->w, static_cast<size_t>(t_pad * hv_n * dk), s,
                              "gdn chunked w pool alloc", GdnScratchKind::kChunk);
    v_new = EnsureGdnScratch<TSc>(chunk_sc->v_new, static_cast<size_t>(t_pad * hv_n * dv), s,
                                  "gdn chunked v_new pool alloc", GdnScratchKind::kChunk);
    hstate = EnsureGdnScratch<TSc>(chunk_sc->hstate,
                                   static_cast<size_t>(nt_tot * hv_n * dv * dk), s,
                                   "gdn chunked hstate pool alloc", GdnScratchKind::kChunk);
    d_tok0 = EnsureGdnScratch<int32_t>(chunk_sc->tok0, static_cast<size_t>(nt_tot), s,
                                       "gdn chunked tok0 pool alloc", GdnScratchKind::kChunk);
    d_len = EnsureGdnScratch<int32_t>(chunk_sc->len, static_cast<size_t>(nt_tot), s,
                                      "gdn chunked len pool alloc", GdnScratchKind::kChunk);
    d_boh = EnsureGdnScratch<int32_t>(chunk_sc->boh, static_cast<size_t>(n_seq), s,
                                      "gdn chunked boh pool alloc", GdnScratchKind::kChunk);
  } else
#endif
  {
    Check(cudaMallocAsync(&gcum, static_cast<size_t>(t_tot * hv_n) * sizeof(float), s),
          "gdn chunked gcum alloc");
    Check(cudaMallocAsync(&u, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
          "gdn chunked u alloc");
    Check(cudaMallocAsync(&w, static_cast<size_t>(t_pad * hv_n * dk) * sizeof(TSc), s),
          "gdn chunked w alloc");
    Check(cudaMallocAsync(&v_new, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
          "gdn chunked v_new alloc");
    Check(cudaMallocAsync(&hstate, static_cast<size_t>(nt_tot * hv_n * dv * dk) * sizeof(TSc), s),
          "gdn chunked hstate alloc");
    Check(cudaMallocAsync(&d_tok0, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
          "gdn chunked tok0 alloc");
    Check(cudaMallocAsync(&d_len, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
          "gdn chunked len alloc");
    Check(cudaMallocAsync(&d_boh, static_cast<size_t>(n_seq) * sizeof(int32_t), s),
          "gdn chunked boh alloc");
  }
  if (dev_meta) {
    // Device-resident metadata: fill d_tok0/d_len/d_boh on the device from the
    // device qsl (no host-vector H2D, no launch sync — the caller does not sync).
    // tok0h/lenh/bohh are empty here; nt_tot was precomputed on the host from the
    // same qsl values, so grid_chunk.x + the hstate size match bit-for-bit.
    GdnBuildChunkMeta<<<1, 1, 0, s>>>(d_tok0, d_len, d_boh, qsl.Ptr<int32_t>(), n_seq);
    Check(cudaGetLastError(), "gdn chunked build-meta launch");
  } else {
    Check(cudaMemcpyAsync(d_tok0, tok0h.data(), static_cast<size_t>(nt_tot) * sizeof(int32_t),
                          cudaMemcpyHostToDevice, s),
          "gdn chunked tok0 upload");
    Check(cudaMemcpyAsync(d_len, lenh.data(), static_cast<size_t>(nt_tot) * sizeof(int32_t),
                          cudaMemcpyHostToDevice, s),
          "gdn chunked len upload");
    Check(cudaMemcpyAsync(d_boh, bohh.data(), static_cast<size_t>(n_seq) * sizeof(int32_t),
                          cudaMemcpyHostToDevice, s),
          "gdn chunked boh upload");
    // The host index vectors must outlive the async uploads; the caller syncs the
    // stream before they go out of scope.
  }

  // FLA chunk_indices [nt_tot,2] for the Triton chunk_o / WU fast-paths. Allocated
  // + built on device ONLY when a Triton chunk path is toggled on at the gate shape
  // (else it stays null and the hand kernels — which don't need it — run): the OFF
  // build never compiles this and an ON build with the toggles off is byte-inert.
#ifdef VLLM_CPP_TRITON
  int32_t* d_cidx = nullptr;
  const bool want_cidx =
      std::is_same<TSc, __nv_bfloat16>::value && dk == 128 && dv == 128 && hk_n == 16 &&
      (hv_n == 48 || hv_n == 32) &&
      (GdnTritonEnvOn("VT_GDN_CHUNKO_TRITON") || GdnTritonEnvOn("VT_GDN_WU_TRITON"));
  if (want_cidx) {
    if (chunk_pool) {
      d_cidx = EnsureGdnScratch<int32_t>(chunk_sc->cidx, static_cast<size_t>(2 * nt_tot), s,
                                         "gdn chunked cidx pool alloc", GdnScratchKind::kChunk);
    } else {
      Check(cudaMallocAsync(&d_cidx, static_cast<size_t>(2 * nt_tot) * sizeof(int32_t), s),
            "gdn chunked cidx alloc");
    }
    const unsigned nb = static_cast<unsigned>((n_seq + 31) / 32);
    GdnBuildChunkIndices<<<nb, 32, 0, s>>>(d_cidx, qsl.Ptr<int32_t>(), d_boh, n_seq);
    Check(cudaGetLastError(), "gdn chunked cidx build launch");
  }
#endif

  const dim3 grid_chunk(static_cast<unsigned>(nt_tot), static_cast<unsigned>(hv_n));
  const dim3 grid_seq(static_cast<unsigned>(n_seq), static_cast<unsigned>(hv_n));

  GdnChunkCumsumKernel<<<grid_chunk, 32, 0, s>>>(gcum, g.Ptr<float>(), d_tok0, d_len, hv_n);
  Check(cudaGetLastError(), "gdn chunked cumsum launch");

  auto opt_in = [](void* kernel, size_t bytes, const char* what) {
    if (bytes > 48 * 1024)
      Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(bytes)),
            what);
  };

  // Step B (WY u/w). WMMA tensor-core Gram when the WMMA path is active (same
  // dim gate as DeltaH/O); A/B: VT_GDN_WU_WMMA=0 forces the scalar-float kernel
  // (also the f32-corner fallback and the unit-test scalar reference path).
  const char* wu_env = std::getenv("VT_GDN_WU_WMMA");
  const bool wu_wmma = wmma && (wu_env == nullptr || wu_env[0] != '0');
  // VEC path (default): replace the per-column WY forward-substitution (measured
  // ~73% of the scalar-solve WU kernel) with a single [BT,BT] triangular inverse
  // + tensor-core apply (GdnChunkWUWmmaVecKernel, mirrors FLA solve_tril+
  // recompute_w_u). VT_GDN_CHUNK_VEC=0 keeps the WMMA-Gram/scalar-solve kernel.
  const char* vec_env = std::getenv("VT_GDN_CHUNK_VEC");
  // VT_GDN_WY_BLOCKED (DEFAULT ON at the gate shape; =0 restores serial-f32): swap ONLY the vec kernel's triangular-inverse
  // phase for the FLA blocked tensor-core inverse (four 16×16 diagonal forward-subs
  // + six 16×16 Schur merges) instead of the serial ~BT-deep column solve. Requires
  // BT == kChunk == 4·kWM == 64 (the merge_16x16_to_64x64 block structure). Greedy 16/16-vs-oracle
  // token-exact on BOTH gate models (35B single+batched, 27B tie-free prefix); the tf32
  // WMMA Schur merges hold vs vLLM's ieee inverse. test_ops_gdn 423/423.
  const char* blk_env = std::getenv("VT_GDN_WY_BLOCKED");
  const bool wy_blocked = kChunk == 4 * kWM && (blk_env == nullptr || blk_env[0] != '0');
  // Compact aliased arena (see kernel): R0=max(Ks[BT,Dk], Tf[BT,BT]f32),
  // R1=max(Am[BT,BT]f32, Tb[BT,BT]TD + Osh[BT,kOBlk]f32). bf16 ~40 KiB (2 blocks/SM),
  // f32/TF32 ~64 KiB (1 block/SM). Both fit the GB10 99 KiB opt-in; the guard keeps
  // the fallback safe on devices/corners with a smaller opt-in budget.
  const size_t r0 = std::max(static_cast<size_t>(kChunk * dk) * sizeof(TSc),
                             static_cast<size_t>(kChunk * kChunk) * sizeof(float));
  const size_t r1 = std::max(static_cast<size_t>(kChunk * kChunk) * sizeof(float),
                             static_cast<size_t>(kChunk * kChunk) * sizeof(TSc) +
                                 static_cast<size_t>(kChunk * kOBlk) * sizeof(float));
  const size_t vec_bytes = r0 + r1;
  int smem_optin = 48 * 1024;
  cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const bool wu_vec = wu_wmma && (vec_env == nullptr || vec_env[0] != '0') &&
                      vec_bytes <= static_cast<size_t>(smem_optin);
  // SANCTIONED Triton AOT WU fast-path (bf16 gate shape only, default ON;
  // VT_GDN_WU_TRITON=0 restores the hand path). If it fires it produces u/w via
  // the FLA 3-kernel WY pipeline into the SAME buffers and skips the hand WU
  // kernels; otherwise the preserved hand path runs unchanged.
  bool triton_wu = false;
#ifdef VLLM_CPP_TRITON
  if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
    triton_wu = TryTritonWU(s, u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum,
                            qsl.Ptr<int32_t>(), d_cidx, hk_n, dk, hv_n, dv, nt_tot, t_tot,
                            wu_pool ? &stream_sc->wu : nullptr);
  }
#endif
  if (triton_wu) {
    // WU (u,w) already produced by the Triton WY pipeline; skip the hand kernels.
  } else if (wu_vec) {
    Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkWUWmmaVecKernel<TSc>),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(vec_bytes)),
          "gdn chunked wu(vec) shared opt-in");
    GdnChunkWUWmmaVecKernel<TSc><<<grid_chunk, 256, vec_bytes, s>>>(
        u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum, d_tok0, d_len, hk_n, dk, hv_n,
        dv, wy_blocked);
    Check(cudaGetLastError(), "gdn chunked wu(vec) launch");
  } else if (wu_wmma) {
    const size_t wu_bytes = static_cast<size_t>(kChunk * dk) * sizeof(TSc) +
                            static_cast<size_t>(kChunk * kChunk) * sizeof(float);
    // Always opt in: the f32/TF32 dynamic request is exactly 48 KiB, which plus
    // the static gs/bs/eg overflows the 48 KiB default (opt_in's ">" guard would
    // miss the boundary). Setting the max-dynamic attribute unconditionally is a
    // no-op when the request already fits.
    Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkWUWmmaKernel<TSc>),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(wu_bytes)),
          "gdn chunked wu(wmma) shared opt-in");
    GdnChunkWUWmmaKernel<TSc><<<grid_chunk, 256, wu_bytes, s>>>(
        u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum, d_tok0, d_len, hk_n, dk, hv_n,
        dv);
    Check(cudaGetLastError(), "gdn chunked wu(wmma) launch");
  } else {
    GdnChunkWUKernel<Tin, TSc><<<grid_chunk, 128, 0, s>>>(u, w, k.Ptr<Tin>(), v.Ptr<Tin>(),
                                                          beta.Ptr<float>(), gcum, d_tok0, d_len,
                                                          hk_n, dk, hv_n, dv);
    Check(cudaGetLastError(), "gdn chunked wu launch");
  }

  if (wmma) {
    // WMMA tensor-core path (TF32 for f32 in, native bf16 for bf16 in).
    // Occupancy: DeltaH and O are pinned to 1 resident block per SM by their
    // ~96 KiB dynamic shared (GB10: 100 KiB/SM). At the legacy 256-thread
    // (8-warp) block that single block realizes only 8/48 = 16.7% of the SM's
    // peak warp slots, while their registers (DeltaH 80, O 74 per thread) would
    // admit far more resident warps within that one block. Both kernels' tensor-
    // core tile loops are already warp-strided (t = warp; t < ntiles; t += nwarps)
    // and every elementwise/staging loop strides by blockDim.x, so the block
    // simply scales with width — more warps in the one resident block hide the
    // WMMA + global-load latency the 8-warp block could not (mirrors the
    // GdnDecodeFused 1-warp -> multi-warp occupancy win). Measured GB10 optima
    // (35B f32 dims, T=2048): DeltaH is bottlenecked by its 64-tile V2 rank
    // update and keeps scaling to 768 threads (24 warps, 50% occ, -32% per call:
    // 2.70 -> 1.84 ms); its 80 regs/thread cap the block at 65 536/80 = 819, so
    // 768 is the top warp-multiple that still fits 1 block/SM. O saturates at 512
    // threads (16 warps, 33% occ, -25%: 1.65 -> 1.23 ms) — its smaller 16/32-tile
    // phases leave the extra 768-block warps idle, a hair slower. Both stay
    // 1 block/SM (shared-bound). A/B: VT_GDN_OCC_BLOCK forces BOTH to a fixed
    // thread count (=256 recovers the pre-occupancy 8-warp path).
    unsigned occ_delta = 768u, occ_o = 512u;
    if (const char* e = std::getenv("VT_GDN_OCC_BLOCK")) {
      const int v = std::atoi(e);
      if (v >= 32 && v <= 1024) occ_delta = occ_o = (static_cast<unsigned>(v) / 32u) * 32u;
    }
    // The f32/TF32 DeltaH instantiation is register-heavier (96 vs the bf16 path's
    // 70); regs×threads must fit the 64K/SM register file, so 96×768 would exceed
    // it and fail to launch. Cap the f32 block at 640 (still 20 warps). f32 is the
    // correctness-only corner (the benchmarked 35B GDN is bf16), so the marginal
    // occupancy drop is immaterial; the bf16 hot path keeps its tuned 768.
    if (sizeof(TSc) == 4 && occ_delta > 640u) occ_delta = 640u;
    // 128-bit (int4/float4) staging for the DeltaH/ChunkO gmem+smem copy loops
    // (measured ~32%/13% of those kernels): address-once + one coalesced 16-byte
    // transfer per 8 bf16 / 4 f32, no per-element bf16↔f32 round-trip. Values are
    // bit-identical (same RNE convert), so chunked==sequential is unchanged. A/B:
    // VT_GDN_DELTAH_VEC=0 restores the scalar per-element staging in the same binary.
    const char* dvec_env = std::getenv("VT_GDN_DELTAH_VEC");
    const bool dvec = (dvec_env == nullptr || dvec_env[0] != '0');
    // Zero v_new so the over-allocated tail rows the matmuls may sweep
    // (their A operand is 0 there, but 0*NaN=NaN on the tensor core) are finite.
    // VT_GDN_SLACK_MEMSET zeros ONLY the per-seq partial-chunk slack rows (the
    // ~3% actually over-read); default OFF keeps the byte-identical full memset.
    if (GdnSlackMemsetEnabled()) {
      GdnChunkVNewSlackZeroKernel<TSc><<<grid_seq, 128, 0, s>>>(v_new, qsl.Ptr<int32_t>(), hv_n,
                                                                dv);
      Check(cudaGetLastError(), "gdn chunked v_new slack-zero launch");
    } else {
      Check(cudaMemsetAsync(v_new, 0, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
            "gdn chunked v_new zero");
    }
    const size_t sz = sizeof(TSc);
    const size_t v1 = static_cast<size_t>(kNB * dk) * sz + static_cast<size_t>(kChunk * kNB) * 4;
    const size_t v2 = static_cast<size_t>(kChunk * dk) * sz;
    const size_t delta_bytes = static_cast<size_t>(dv * dk) * sizeof(float) + (v1 > v2 ? v1 : v2);
    // REGISTER-tiled delta_h (vt::tile Rung-1): the running state H lives in WMMA
    // ACCUMULATOR REGISTERS across the chunk loop (FLA blockdim64's b_h1/b_h2), NOT
    // in 64 KiB of shared memory — the CUTLASS-style persistent-accumulator GEMM
    // that frees smem for the cp.async ring below. DEFAULT ON at the gate shape
    // (dk==128, dv%BV==0, bf16); VT_GDN_TILE_PIPE=0 restores the baseline
    // GdnChunkDeltaHWmmaKernel for A/B. BV=64 (grid.x=dv/BV, 4 warps/block).
    // MEASURED (35B NVFP4, in1024/out8, GB10): delta_h 441us baseline ->
    // 398us reg-no-ring -> 345us reg+ring (0.78x); e2e in1024/out128 conc32 np192
    // +0.85% total & prefill, TTFT/TPOT lower. Token-exact (test_ops_gdn 423/423).
    // SANCTIONED Triton AOT fast-path (bf16 gate shapes only, default ON;
    // VT_GDN_DELTAH_TRITON=0 restores the hand path). If it fires it launches
    // delta_h into the SAME buffers and skips the hand kernels; otherwise the
    // preserved hand-C++ path runs unchanged.
    bool triton_deltah = false;
#ifdef VLLM_CPP_TRITON
    if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
      triton_deltah =
          TryTritonDeltaH(s, state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum,
                          qsl.Ptr<int32_t>(), d_boh, hk_n, dk, hv_n, dv, n_seq, t_tot);
    }
#endif
    const char* tp_env = std::getenv("VT_GDN_TILE_PIPE");
    const bool tile_pipe = (tp_env == nullptr || tp_env[0] != '0') && dk == 128 && (dv % 64 == 0);
    if (triton_deltah) {
      // delta_h already launched by the Triton fast-path; skip the hand kernels.
    } else if (tile_pipe) {
      constexpr int BV = 64;
      const int64_t rA0 = static_cast<int64_t>(BV * dk) * sizeof(float);
      const int64_t rA1 = static_cast<int64_t>(kChunk * BV) * sizeof(float);
      const int64_t rA2 = static_cast<int64_t>(kChunk * dk) * sz;
      int64_t regA = rA0 > rA1 ? rA0 : rA1;
      regA = regA > rA2 ? regA : rA2;
      const int64_t rB0 = static_cast<int64_t>(BV * dk) * sz;
      const int64_t rB1 = static_cast<int64_t>(kChunk * BV) * sz;
      const size_t reg_bytes = static_cast<size_t>(regA + (rB0 > rB1 ? rB0 : rB1));
      const dim3 grid_reg(static_cast<unsigned>(dv / BV),
                          static_cast<unsigned>(n_seq * hv_n));
      const unsigned reg_block = (BV / kWM) * 32;
      // STEP 2 — cp.async ring (VT_GDN_TILE_PIPE_CPASYNC, default ON for bf16):
      // 2-stage software pipeline over the streamed W/K tiles. bf16 only (the f32
      // ring tiles overflow the 99 KiB opt-in); f32 uses the ring-less reg kernel.
      // The ring runs at 1 block/SM (its smem halves occupancy vs reg-no-ring's 2
      // blocks), but the prefetch hides the load-stalls that low occupancy exposes,
      // netting the 398->345us win. VT_GDN_TILE_PIPE_CPASYNC=0 isolates reg-no-ring.
      const char* ca_env = std::getenv("VT_GDN_TILE_PIPE_CPASYNC");
      const bool use_ring = (ca_env == nullptr || ca_env[0] != '0');
      // STEP 3 (Rung-2) — TMA + mbarrier ring (VT_GDN_TMA, default OFF). Same
      // register-tiled WMMA compute + smem layout as the cp.async ring; only the
      // W/K streaming swaps sm80 cp.async for TMA bulk-tensor copies signalled by
      // per-stage mbarriers (the Blackwell-native pipeline). A/B isolates the copy
      // mechanism. bf16 only; falls back to the cp.async ring if TMA desc setup fails.
      const char* tma_env = std::getenv("VT_GDN_TMA");
      const bool use_tma = (tma_env != nullptr && tma_env[0] == '1');
      bool launched = false;
      if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
        constexpr int STAGES = 2;
        const int64_t rAr = static_cast<int64_t>(BV * 64) * sizeof(float);  // Hf32(half)/Braw f32
        const int64_t rBr = static_cast<int64_t>(BV * dk) * sz;             // Hbf full (TD)
        const int64_t tile = static_cast<int64_t>(kChunk * dk) * sz;
        const size_t ring_bytes = static_cast<size_t>(rAr + rBr + (int64_t)STAGES * 2 * tile);
        if (use_tma) {
          CUtensorMap descW{}, descK{};
          const bool okW = BuildGdnTmaDesc3D(&descW, w, t_pad, hv_n, dk, kChunk);
          const bool okK = BuildGdnTmaDesc3D(&descK, k.Ptr<Tin>(), t_tot, hk_n, dk, kChunk);
          if (okW && okK) {
            Check(cudaFuncSetAttribute(
                      reinterpret_cast<void*>(GdnChunkDeltaHTmaKernel<TSc, BV, STAGES>),
                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ring_bytes)),
                  "gdn chunked delta_h(tma) shared opt-in");
            GdnChunkDeltaHTmaKernel<TSc, BV, STAGES><<<grid_reg, reg_block, ring_bytes, s>>>(
                state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
                d_boh, hk_n, dk, hv_n, dv, descW, descK);
            Check(cudaGetLastError(), "gdn chunked delta_h(tma) launch");
            launched = true;
          }
        }
        if (!launched && use_ring) {
          Check(cudaFuncSetAttribute(
                    reinterpret_cast<void*>(GdnChunkDeltaHRegRingKernel<TSc, BV, STAGES>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ring_bytes)),
                "gdn chunked delta_h(reg-ring) shared opt-in");
          GdnChunkDeltaHRegRingKernel<TSc, BV, STAGES><<<grid_reg, reg_block, ring_bytes, s>>>(
              state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
              d_boh, hk_n, dk, hv_n, dv);
          Check(cudaGetLastError(), "gdn chunked delta_h(reg-ring) launch");
          launched = true;
        }
      }
      if (!launched) {
        // Always opt in: the bf16 request is exactly 48 KiB, which the default
        // 48896-byte cap rejects (opt_in's ">" guard would miss the boundary).
        Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkDeltaHRegKernel<TSc, BV>),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(reg_bytes)),
              "gdn chunked delta_h(reg) shared opt-in");
        GdnChunkDeltaHRegKernel<TSc, BV><<<grid_reg, reg_block, reg_bytes, s>>>(
            state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
            d_boh, hk_n, dk, hv_n, dv);
        Check(cudaGetLastError(), "gdn chunked delta_h(reg) launch");
      }
    } else {
      opt_in(reinterpret_cast<void*>(GdnChunkDeltaHWmmaKernel<TSc>), delta_bytes,
             "gdn chunked delta_h(wmma) shared opt-in");
      GdnChunkDeltaHWmmaKernel<TSc><<<grid_seq, occ_delta, delta_bytes, s>>>(
          state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(), d_boh,
          hk_n, dk, hv_n, dv, dvec);
      Check(cudaGetLastError(), "gdn chunked delta_h(wmma) launch");
    }

    // SANCTIONED Triton AOT chunk_o fast-path (bf16 gate shape + f32 out, plus
    // bf16 out when VLLM_CPP_TRITON_CHUNKO_BF16 is compiled). If it fires it writes
    // `out` directly and we skip the hand kernel; otherwise the preserved hand path
    // runs unchanged.
    bool triton_chunko = false;
#ifdef VLLM_CPP_TRITON
    if constexpr (std::is_same<TSc, __nv_bfloat16>::value &&
                  (std::is_same<Tout, float>::value ||
                   std::is_same<Tout, __nv_bfloat16>::value)) {
      triton_chunko = TryTritonChunkO(s, out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new,
                                      hstate, gcum, qsl.Ptr<int32_t>(), d_cidx, args.scale, hk_n,
                                      dk, hv_n, dv, nt_tot, t_tot);
    }
#endif
    if (!triton_chunko) {
#ifdef VLLM_CPP_TRITON
      GdnDebugCounters& counters = GdnCounters();
      if (counters.enabled.load(std::memory_order_relaxed)) {
        counters.chunk_o_hand_launches.fetch_add(1, std::memory_order_relaxed);
      }
#endif
      const size_t r2 = static_cast<size_t>(kChunk * dk) * sz > static_cast<size_t>(kChunk * dv) * 4
                            ? static_cast<size_t>(kChunk * dk) * sz
                            : static_cast<size_t>(kChunk * dv) * 4;
      const size_t chunko_bytes = static_cast<size_t>(kChunk * dk) * sz + r2 +
                                  static_cast<size_t>(kChunk * kChunk) * 4 +
                                  static_cast<size_t>(kChunk * kChunk) * sz;
      opt_in(reinterpret_cast<void*>(GdnChunkOWmmaKernel<TSc, Tout>), chunko_bytes,
             "gdn chunked o(wmma) shared opt-in");
      GdnChunkOWmmaKernel<TSc, Tout><<<grid_chunk, occ_o, chunko_bytes, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new, hstate, gcum, d_tok0, d_len,
          hk_n, dk, hv_n, dv, args.scale, dvec);
      Check(cudaGetLastError(), "gdn chunked o(wmma) launch");
    }
  } else if constexpr (std::is_same<Tin, float>::value) {
    // CUDA-core fallback — only reached for f32 corner dims (bf16 non-WMMA dims
    // are routed to the sequential scan upstream).
    const size_t hsh_bytes = static_cast<size_t>(dv * dk) * sizeof(float);
    opt_in(reinterpret_cast<void*>(GdnChunkDeltaHKernel<Tin>), hsh_bytes,
           "gdn chunked delta_h shared opt-in");
    GdnChunkDeltaHKernel<Tin><<<grid_seq, 256, hsh_bytes, s>>>(
        state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(), d_boh,
        hk_n, dk, hv_n, dv);
    Check(cudaGetLastError(), "gdn chunked delta_h launch");

    GdnChunkOKernel<Tin, Tout><<<grid_chunk, 256, 0, s>>>(out.Ptr<Tout>(), q_in.Ptr<Tin>(),
                                                          k.Ptr<Tin>(), v_new, hstate, gcum,
                                                          d_tok0, d_len, hk_n, dk, hv_n, dv,
                                                          args.scale);
    Check(cudaGetLastError(), "gdn chunked o launch");
  } else {
    VT_CHECK(false, "cuda gdn_prefill(chunked): bf16 requires WMMA-friendly head dims");
  }

#ifdef VLLM_CPP_TRITON
  if (!chunk_pool)
#endif
  {
    Check(cudaFreeAsync(gcum, s), "gdn chunked gcum free");
    Check(cudaFreeAsync(u, s), "gdn chunked u free");
    Check(cudaFreeAsync(w, s), "gdn chunked w free");
    Check(cudaFreeAsync(v_new, s), "gdn chunked v_new free");
    Check(cudaFreeAsync(hstate, s), "gdn chunked hstate free");
    Check(cudaFreeAsync(d_tok0, s), "gdn chunked tok0 free");
    Check(cudaFreeAsync(d_len, s), "gdn chunked len free");
    Check(cudaFreeAsync(d_boh, s), "gdn chunked boh free");
  }
#ifdef VLLM_CPP_TRITON
  if (d_cidx != nullptr && !chunk_pool) Check(cudaFreeAsync(d_cidx, s), "gdn chunked cidx free");
#endif
}

void GdnPrefillChunkedCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                           const Tensor& qsl, const GdnArgs& args) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           "cuda gdn_prefill(chunked): unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           "cuda gdn_prefill(chunked): q/k/v dtypes must match");
  const int64_t n_seq = state.shape[0], hv_n = state.shape[1], dv = state.shape[2];
  if (n_seq == 0 || hv_n == 0 || dv == 0) return;
  const int64_t hk_n = q_in.shape[1];
  VT_CHECK(hk_n > 0 && hv_n % hk_n == 0,
           "cuda gdn_prefill(chunked): Hv must be a multiple of Hk");
  cudaStream_t s = AsStream(q);

  // Chunk layout from query_start_loc. FAST PATH (dev_meta): the caller handed us
  // the host-resident qsl (args.query_start_loc_host) — the SAME values already
  // materialized on the host by the GDN attention-metadata build — so we read
  // them directly (no D2H copy, no cudaStreamSynchronize) and let a device kernel
  // fill the per-chunk arrays. This kills the per-GDN-layer host↔GPU lockstep that
  // left the prefill ~67% GPU-idle. FALLBACK (host qsl null: op tests / callers
  // without it): the original D2H copy + sync to read qsl on the host.
  const bool dev_meta = args.query_start_loc_host != nullptr;
  std::vector<int32_t> qslh(static_cast<size_t>(n_seq) + 1);
  if (dev_meta) {
    for (int64_t i = 0; i <= n_seq; ++i) qslh[static_cast<size_t>(i)] = args.query_start_loc_host[i];
  } else {
    Check(cudaMemcpyAsync(qslh.data(), qsl.Ptr<int32_t>(),
                          (static_cast<size_t>(n_seq) + 1) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost, s),
          "gdn chunked qsl download");
    Check(cudaStreamSynchronize(s), "gdn chunked qsl sync");
  }

  // nt_tot (total chunks) is always computed on the host — it sizes the grids and
  // the hstate scratch. In dev_meta mode the GdnBuildChunkMeta device kernel
  // recomputes the identical layout from the same qsl values, so nt_tot matches
  // bit-for-bit; the host tok0h/lenh/bohh vectors are then left empty (unused).
  std::vector<int32_t> tok0h, lenh, bohh;
  int32_t nt_tot = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    const int32_t begin = qslh[static_cast<size_t>(n)];
    const int32_t len = qslh[static_cast<size_t>(n) + 1] - begin;
    const int32_t nt = (len + kChunk - 1) / kChunk;
    if (!dev_meta) {
      if (bohh.empty()) bohh.resize(static_cast<size_t>(n_seq));
      bohh[static_cast<size_t>(n)] = nt_tot;
      for (int32_t it = 0; it < nt; ++it) {
        const int32_t rem = len - it * kChunk;
        tok0h.push_back(begin + it * kChunk);
        lenh.push_back(rem < kChunk ? rem : kChunk);
      }
    }
    nt_tot += nt;
  }
  if (nt_tot == 0) return;

  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<float, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh, bohh,
                                         qsl, n_seq, nt_tot, dev_meta, args);
    else
      LaunchChunkedPrefill<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, dev_meta, args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, dev_meta, args);
    else
      LaunchChunkedPrefill<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h,
                                                         lenh, bohh, qsl, n_seq, nt_tot, dev_meta,
                                                         args);
  }
  // FALLBACK only: keep the host index vectors alive until the async uploads
  // finish. dev_meta builds the arrays on-device (no host vectors) so it does NOT
  // sync here — the whole point (host runs ahead, GPU stays fed across layers).
  if (!dev_meta) Check(cudaStreamSynchronize(s), "gdn chunked launch sync");
}

void GdnPrefillKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const Tensor& qsl, const GdnArgs& args) {
  const int64_t dv = state.shape[2], dk = state.shape[3];
  // Chunked path (default) for real head dims; the sequential scan handles the
  // arbitrary-dim corners (the CUDA op tests exercise Dv=Dk=300) and the A/B
  // fallback (VT_GDN_CHUNKED=0). The bf16 chunked path is WMMA (tensor-core),
  // which tiles at 16 and 32 — bf16 dims that are not WMMA-friendly fall back
  // to the sequential scan (real gate dims Dk=Dv=128 satisfy both).
  // ARCH TERM (required, not an optimisation). Every kernel LaunchChunkedPrefill
  // reaches — the WU, delta_h (wmma/reg/regring/tma) and chunk_o bodies — is
  // compiled `#if __CUDA_ARCH__ >= 800` with an `#else __trap()`, because bf16
  // fragments AND wmma::precision::tf32 are Ampere+. That holds for BOTH scratch
  // dtypes: the TSc=float instantiations use the TF32 WmmaCfg, so f32 is not a
  // way around it. This routing is the single chokepoint (all 7 launches live in
  // LaunchChunkedPrefill, itself reached only from here), so gating it sends
  // <sm_80 to GdnScanCuda, the portable sequential scan that already serves the
  // arbitrary-dim corners. Fails safe: caps invalid -> scan. On sm_80+ the term
  // is always true, so the gate models' path is unchanged.
  // See .agents/specs/cuda-arch-breadth-fp16.md §V0-a / W1c.
  const DeviceCaps& gdn_caps = GetDeviceCaps();
  const bool arch_has_mma = gdn_caps.valid && gdn_caps.sm_major >= 8;
  const bool wmma_ok = q_in.dtype != DType::kBF16 || (dk % kWM == 0 && dv % kNB == 0);
  // KERNEL-GDN-CHUNKED-MIRROR D0. The first term now carries the dtype: CUDA
  // used to take the chunked arm for f32 inputs, which is an algorithm vLLM
  // refuses to run at that dtype on either of its implementations
  // (chunk.py:213-215 asserts; csrc/cpu/sgl-kernels/fla.cpp:2205-2207 type-
  // checks bf16). This is a BEHAVIOUR CHANGE on a shipped backend: an f32 CUDA
  // GdnPrefill that ran chunked before this row runs the sequential scan after
  // it. It is also what test_op_parity.cpp:594's ScopedEnv was compensating
  // for; that belt is kept while these braces are new.
  if (GdnUseChunkedPrefill(q_in.dtype) && dk <= kChunkMaxDim && dv <= kChunkMaxDim &&
      args.scale != 0.0f && wmma_ok && arch_has_mma) {
    GdnPrefillChunkedCuda(q, out, q_in, k, v, g, beta, state, qsl, args);
    return;
  }
  GdnScanCuda(q, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(),
              nullptr, args, "gdn_prefill");
}

// ===========================================================================
// SPECULATIVE GDN (SPEC-MTP I4): the T>1 / IS_SPEC_DECODING variants.
//
// Both are NEW ENTRY POINTS. They add no branch to any shipped kernel, so the
// non-speculative production path (the only one reachable today — without a
// SpeculativeConfig num_spec_decodes is always 0) is byte-identical by
// construction; see the default-off proof in the I4 record.
// ===========================================================================

// SPECULATIVE causal_conv1d_update. Ported from
// vllm/model_executor/layers/mamba/ops/causal_conv1d.py @ e24d1b24
// (_causal_conv1d_update_kernel, IS_SPEC_DECODING + IS_VARLEN, :818-1067).
// One thread per (request, channel); the request's 1+k tokens run sequentially
// inside the thread, which mirrors the upstream program structure exactly
// (grid = (batch, cdiv(dim, BLOCK_N)) with a `for idx_token in range(seqlen)`
// loop at :972). Bit-identical to the CPU reference CausalConv1dSpecUpdateKernel
// (src/vt/cpu/cpu_ops.cpp): same bias init, same left-to-right accumulation over
// the widened window, same silu epilogue, same one-tap left shift.
template <typename Tin, typename Tout, typename TState>
__global__ void ConvSpecUpdateKernel(Tout* out, const Tin* x, const Tin* w, const Tin* bias,
                                     TState* conv_state, const int32_t* cache_idx,
                                     const int32_t* num_accepted, const int32_t* cu_seqlens,
                                     int64_t num_reqs, int64_t c_dim, int64_t x_row_stride,
                                     int64_t k, int64_t state_len, int64_t state_slots,
                                     bool silu) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= c_dim) return;
  const int64_t i = blockIdx.y;
  if (i >= num_reqs) return;
  const int32_t slot = cache_idx[i];
  // Local cache ABI: idx < 0 == NULL row → leave `out` and the cache untouched.
  // The upper bound is an independent kernel-side guard (engine metadata is
  // validated on host before upload; re-checking here adds no capture-breaking
  // synchronization).
  if (slot < 0 || static_cast<int64_t>(slot) >= state_slots) return;
  const int64_t lo = cu_seqlens[i], hi = cu_seqlens[i + 1];
  const int64_t seqlen = hi - lo;
  if (seqlen <= 0) return;  // zero-length padded row (:832-833)
  const int64_t width = k - 1;
  // Sliding-window read offset = accepted count - 1 (:846-852).
  const int64_t off = static_cast<int64_t>(num_accepted[i]) - 1;
  TState* srow = conv_state + (static_cast<int64_t>(slot) * c_dim + c) * state_len;
  // Outputs first: the whole window is still the OLD state (the roll below only
  // starts once every token has been emitted).
  for (int64_t t = 0; t < seqlen; ++t) {
    float acc = bias != nullptr ? Load(bias, c) : 0.0f;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t si = t + j;
      const float sv = si < width
                           ? ((off + si) < state_len ? Load(srow, off + si) : 0.0f)
                           : Load(x, (lo + si - width) * x_row_stride + c);
      acc += Load(w, c * k + j) * sv;
    }
    Store(out, (lo + t) * c_dim + c, silu ? Silu(acc) : acc);
  }
  // Roll: new row = old[off+1 .. off+keep] ++ raw x. Writing forward is safe —
  // destination j reads source off+j+1 > j, so no source is clobbered first.
  const int64_t keep = state_len - seqlen;
  for (int64_t j = 0; j < keep; ++j) {
    const int64_t src = off + j + 1;
    Store(srow, j, src < state_len ? Load(srow, src) : 0.0f);
  }
  for (int64_t t = 0; t < seqlen; ++t)
    Store(srow, keep + t, Load(x, (lo + t) * x_row_stride + c));
}

template <typename Tin, typename Tout, typename TState>
void LaunchConvSpecUpdate(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                          const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                          const int32_t* num_accepted, const int32_t* cu_seqlens,
                          int64_t num_reqs, const CausalConv1dArgs& args) {
  const int64_t c = x.shape[1], k = w.shape[1];
  const int64_t state_len = conv_state.shape[2];
  const dim3 grid(static_cast<unsigned>((c + kBlock - 1) / kBlock),
                  static_cast<unsigned>(num_reqs));
  ConvSpecUpdateKernel<Tin, Tout, TState><<<grid, kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
      bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<TState>(), cache_idx,
      num_accepted, cu_seqlens, num_reqs, c, x.stride[0], k, state_len, conv_state.shape[0],
      args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_spec_update launch");
}

template <typename Tin, typename Tout>
void LaunchConvSpecUpdateS(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                           const int32_t* num_accepted, const int32_t* cu_seqlens,
                           int64_t num_reqs, const CausalConv1dArgs& args) {
  if (conv_state.dtype == DType::kBF16)
    LaunchConvSpecUpdate<Tin, Tout, __nv_bfloat16>(s, out, x, w, bias, conv_state, cache_idx,
                                                   num_accepted, cu_seqlens, num_reqs, args);
  else
    LaunchConvSpecUpdate<Tin, Tout, float>(s, out, x, w, bias, conv_state, cache_idx,
                                           num_accepted, cu_seqlens, num_reqs, args);
}

void ConvSpecUpdateKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor& conv_state_indices,
                              const Tensor& num_accepted_tokens, const Tensor& cu_seqlens,
                              const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_spec_update: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_spec_update: weight/bias dtype must match x");
  const int64_t num_reqs = cu_seqlens.shape[0] - 1;
  if (num_reqs == 0 || x.shape[1] == 0) return;
  VT_CHECK(num_reqs <= kMaxGridY,
           "cuda causal_conv1d_spec_update: too many sequences (grid.y limit)");
  cudaStream_t s = AsStream(q);
  const int32_t* ci = conv_state_indices.Ptr<int32_t>();
  const int32_t* na = num_accepted_tokens.Ptr<int32_t>();
  const int32_t* cs = cu_seqlens.Ptr<int32_t>();
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchConvSpecUpdateS<float, float>(s, out, x, w, bias, conv_state, ci, na, cs, num_reqs,
                                          args);
    else
      LaunchConvSpecUpdateS<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, na, cs,
                                                  num_reqs, args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchConvSpecUpdateS<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, ci, na, cs,
                                                  num_reqs, args);
    else
      LaunchConvSpecUpdateS<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, na,
                                                          cs, num_reqs, args);
  }
}

// SPECULATIVE gated-delta-rule recurrence. Ported from
// vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py @ e24d1b24
// (fused_recurrent_gated_delta_rule_fwd_kernel with IS_VARLEN :66-72,
// IS_SPEC_DECODING :103-116 and INPLACE_FINAL_STATE :156-166).
//
// Geometry is GdnDecodeFusedKernel's — one block per (request, v-head, BV value
// tile), the [BV,Dk] state slice staged once into shared and updated in place —
// with two additions:
//   * the INITIAL slice is read from column `num_accepted_tokens[i] - 1` of the
//     request's state-index row (the snapshot after the last ACCEPTED token);
//   * the request's 1+k tokens run as an inner sequential loop, and after EVERY
//     timestep the slice is written back to THAT timestep's own slot, so a later
//     rejection is just an earlier column read.
// The per-token arithmetic is the GdnDecodeFusedKernel body verbatim (same
// decay-fused Dk pass, same shfl_xor reduction, same update order), so a T==1 /
// num_accepted==1 spec call reproduces GdnDecode bit-for-bit.
template <typename Tin, typename Tout, typename TState, int NW>
__global__ void GdnSpecDecodeFusedKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                                         const float* g, const float* beta, TState* state,
                                         const int32_t* cu_seqlens,
                                         const int32_t* state_indices,
                                         const int32_t* num_accepted, int64_t num_cols,
                                         int64_t state_slots, int64_t hk_n, int64_t dk,
                                         int64_t hv_n, int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;         // value-dim tile
  const int64_t i_nh = blockIdx.y;        // fused (request, v-head)
  const int64_t i_n = i_nh / hv_n;        // request
  const int64_t hv = i_nh % hv_n;         // v-head
  const int64_t hk = hv / (hv_n / hk_n);  // GQA-mapped k-head
  const int64_t vbase = i_v * bv;
  const int tid = static_cast<int>(threadIdx.x);
  const int vi = tid / NW;
  const int wk = tid % NW;
  const int64_t vrow = vbase + vi;

  const int64_t lo = cu_seqlens[i_n];
  const int64_t hi = cu_seqlens[i_n + 1];
  // Initial-state column = accepted count - 1 (fused_sigmoid_gating.py:105-107).
  // Uniform over the block (i_n is fixed by blockIdx.y), so every early return
  // below is block-uniform and the shfl butterflies stay well-formed.
  int64_t init_col = static_cast<int64_t>(num_accepted[i_n]) - 1;
  if (init_col < 0) init_col = 0;
  if (init_col >= num_cols) init_col = num_cols - 1;
  const int32_t init_slot = state_indices[i_n * num_cols + init_col];
  if (init_slot < 0 || static_cast<int64_t>(init_slot) >= state_slots) {
    // NULL initial slot ⇒ skip the request, zeroing its output rows
    // (fused_sigmoid_gating.py:112-115 `if state_idx <= 0: return`).
    if (vrow < dv && wk == 0)
      for (int64_t t = lo; t < hi; ++t) Store(out, (t * hv_n + hv) * dv + vrow, 0.0f);
    return;
  }
  if (hi <= lo) return;  // T == 0 early-out (:76-78)

  const int64_t sdk = dk + 1;  // padded shared row stride (kills bank conflicts)
  extern __shared__ float smem[];
  float* bq = smem;
  float* bk = bq + dk;
  float* sbh = bk + dk;  // [bv * sdk]
  const int64_t vrows = (dv - vbase) < bv ? (dv - vbase) : bv;
  const int64_t tile = vrows * dk;

  const TState* s_init =
      state + (static_cast<int64_t>(init_slot) * hv_n + hv) * dv * dk + vbase * dk;
  for (int64_t e = tid; e < bv * dk; e += blockDim.x)
    sbh[(e / dk) * sdk + e % dk] = e < tile ? Load(s_init, e) : 0.0f;

  const int64_t ck = (dk + NW - 1) / NW;
  const int64_t c0 = static_cast<int64_t>(wk) * ck;
  const int64_t c1 = (c0 + ck) < dk ? (c0 + ck) : dk;
  float* r = sbh + static_cast<int64_t>(vi) * sdk;

  for (int64_t t = lo; t < hi; ++t) {
    __syncthreads();  // protects bq/bk and the shared slice across iterations
    const int64_t qkbase = (t * hk_n + hk) * dk;
    for (int64_t i = tid; i < dk; i += blockDim.x) {
      bq[i] = Load(q, qkbase + i) * scale;
      bk[i] = Load(k, qkbase + i);
    }
    __syncthreads();

    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    float pdot = 0.0f;
    for (int64_t c = c0; c < c1; ++c) {
      r[c] *= decay;
      pdot += r[c] * bk[c];
    }
    float dot = pdot;
#pragma unroll
    for (int off = 1; off < NW; off <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float vv = vrow < dv ? Load(v, (t * hv_n + hv) * dv + vrow) : 0.0f;
    const float vp = (vv - dot) * beta_t;
    float po = 0.0f;
    for (int64_t c = c0; c < c1; ++c) {
      r[c] += vp * bk[c];
      po += r[c] * bq[c];
    }
    float o = po;
#pragma unroll
    for (int off = 1; off < NW; off <<= 1) o += __shfl_xor_sync(0xffffffffu, o, off);
    if (vrow < dv && wk == 0) Store(out, (t * hv_n + hv) * dv + vrow, o);

    // Per-timestep SNAPSHOT into timestep t's OWN slot
    // (fused_sigmoid_gating.py:156-166). A NULL column skips only its store.
    __syncthreads();
    const int32_t slot = state_indices[i_n * num_cols + (t - lo)];
    if (slot >= 0 && static_cast<int64_t>(slot) < state_slots) {
      TState* s_dst = state + (static_cast<int64_t>(slot) * hv_n + hv) * dv * dk + vbase * dk;
      for (int64_t e = tid; e < tile; e += blockDim.x)
        Store(s_dst, e, sbh[(e / dk) * sdk + e % dk]);
    }
  }
}

template <typename Tin, typename Tout, typename TState, int NW>
void LaunchGdnSpecDecodeNW(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                           const int32_t* cu_seqlens, const int32_t* state_indices,
                           const int32_t* num_accepted, int64_t num_reqs, int64_t num_cols,
                           const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t bv = dv < 32 ? dv : 32;
  const int64_t nv = (dv + bv - 1) / bv;
  const dim3 grid(static_cast<unsigned>(nv), static_cast<unsigned>(num_reqs * hv_n));
  const size_t shmem =
      (2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) * sizeof(float);
  GdnSpecDecodeFusedKernel<Tin, Tout, TState, NW>
      <<<grid, static_cast<unsigned>(bv * NW), shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
          beta.Ptr<float>(), state.Ptr<TState>(), cu_seqlens, state_indices, num_accepted,
          num_cols, state.shape[0], hk_n, dk, hv_n, dv, bv, args.scale);
  Check(cudaGetLastError(), "gdn spec decode launch");
}

template <typename Tin, typename Tout, typename TState>
void LaunchGdnSpecDecode(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                         const int32_t* cu_seqlens, const int32_t* state_indices,
                         const int32_t* num_accepted, int64_t num_reqs, int64_t num_cols,
                         const GdnArgs& args, int nw) {
  const int64_t dv = v.shape[2];
  const int nw_eff = dv >= 32 ? nw : 1;
  switch (nw_eff) {
    case 2:
      LaunchGdnSpecDecodeNW<Tin, Tout, TState, 2>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                                  state_indices, num_accepted, num_reqs,
                                                  num_cols, args);
      break;
    case 4:
      LaunchGdnSpecDecodeNW<Tin, Tout, TState, 4>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                                  state_indices, num_accepted, num_reqs,
                                                  num_cols, args);
      break;
    case 8:
      LaunchGdnSpecDecodeNW<Tin, Tout, TState, 8>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                                  state_indices, num_accepted, num_reqs,
                                                  num_cols, args);
      break;
    default:
      LaunchGdnSpecDecodeNW<Tin, Tout, TState, 1>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                                  state_indices, num_accepted, num_reqs,
                                                  num_cols, args);
      break;
  }
}

template <typename Tin, typename Tout>
void LaunchGdnSpecDecodeS(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const int32_t* cu_seqlens, const int32_t* state_indices,
                          const int32_t* num_accepted, int64_t num_reqs, int64_t num_cols,
                          const GdnArgs& args, int nw) {
  if (state.dtype == DType::kBF16)
    LaunchGdnSpecDecode<Tin, Tout, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                                  state_indices, num_accepted, num_reqs,
                                                  num_cols, args, nw);
  else if (state.dtype == DType::kF16)
    LaunchGdnSpecDecode<Tin, Tout, __half>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                           state_indices, num_accepted, num_reqs, num_cols, args,
                                           nw);
  else
    LaunchGdnSpecDecode<Tin, Tout, float>(s, out, q_in, k, v, g, beta, state, cu_seqlens,
                                          state_indices, num_accepted, num_reqs, num_cols, args,
                                          nw);
}

void GdnSpecDecodeKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                             const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                             const Tensor& cu_seqlens, const Tensor& state_indices,
                             const Tensor& num_accepted_tokens, const GdnArgs& args) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           "cuda gdn_spec_decode: unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           "cuda gdn_spec_decode: q/k/v dtypes must match");
  const int64_t num_reqs = state_indices.shape[0];
  const int64_t num_cols = state_indices.shape[1];
  const int64_t hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  if (num_reqs == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(num_reqs * hv_n <= kMaxGridY,
           "cuda gdn_spec_decode: too many (request x head) blocks (grid.y limit)");
  const int64_t bv = dv < 32 ? dv : 32;
  const size_t shmem =
      (2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) * sizeof(float);
  VT_CHECK(shmem <= 48 * 1024,
           "cuda gdn_spec_decode: Dk/Dv too large for the shared state slice");
  // Same warps-per-block occupancy lever as GdnDecodeFused (default 8; NW==1 is
  // the single-warp reference). Read once per call.
  int nw = 8;
  if (const char* e = std::getenv("VT_GDN_DECODE_NW")) {
    const int v_nw = std::atoi(e);
    if (v_nw == 1 || v_nw == 2 || v_nw == 4 || v_nw == 8) nw = v_nw;
  }
  cudaStream_t s = AsStream(q);
  const int32_t* cs = cu_seqlens.Ptr<int32_t>();
  const int32_t* si = state_indices.Ptr<int32_t>();
  const int32_t* na = num_accepted_tokens.Ptr<int32_t>();
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchGdnSpecDecodeS<float, float>(s, out, q_in, k, v, g, beta, state, cs, si, na, num_reqs,
                                         num_cols, args, nw);
    else
      LaunchGdnSpecDecodeS<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, cs, si, na,
                                                 num_reqs, num_cols, args, nw);
  } else {
    if (out.dtype == DType::kF32)
      LaunchGdnSpecDecodeS<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, cs, si, na,
                                                 num_reqs, num_cols, args, nw);
    else
      LaunchGdnSpecDecodeS<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, cs,
                                                         si, na, num_reqs, num_cols, args, nw);
  }
}

void GdnDecodeKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                         const Tensor* state_idx, const GdnArgs& args) {
  const int32_t* si = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  // Fused (batch × Hv × value-tile) parallel decode — the fla fused_recurrent
  // mirror. state_idx != null => in-place on the FULL persistent cache at slot
  // state_idx[i_n] (mirrors fla ssm_state_indices); null => compact state (row ==
  // i_n). Replaces the batch-serial GdnScanKernel decode branch (M2.x GDN decode
  // throughput). A/B: VT_GDN_FUSED_DECODE=0 falls back to the sequential scan for
  // same-binary before/after measurement — but ONLY for the compact path (the scan
  // has no ssm_state_indices indirection), so the indexed path always uses fused.
  const char* e = std::getenv("VT_GDN_FUSED_DECODE");
  if (si == nullptr && e != nullptr && e[0] == '0') {
    GdnScanCuda(q, out, q_in, k, v, g, beta, state, nullptr, nullptr,
                args, "gdn_decode");
    return;
  }
  GdnDecodeFusedCuda(q, out, q_in, k, v, g, beta, state, si, args);
}

// Registers the CUDA GDN kernels during static init (pre-main, like the M0.6
// ops in cuda_ops.cu). Filling the op table is harmless on machines without a
// GPU: the kCUDA backend never registers there, so no CUDA queue can exist to
// dispatch with.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&ConvFwdKernelCuda)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&ConvUpdateKernelCuda)));
    RegisterOp(OpId::kL2Norm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernelCuda)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernelCuda)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernelCuda)));
    RegisterOp(
        OpId::kRmsNormGatedQuantFp8, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<RmsNormGatedQuantFp8Fn>(&RmsNormGatedQuantFp8KernelCuda)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernelCuda)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernelCuda)));
    RegisterOp(
        OpId::kGdnSpecDecode, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<GdnSpecDecodeFn>(&GdnSpecDecodeKernelCuda)));
    RegisterOp(
        OpId::kCausalConv1dSpecUpdate, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<CausalConv1dSpecUpdateFn>(&ConvSpecUpdateKernelCuda)));
    RegisterOp(
        OpId::kGdnPackedDecode, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<GdnPackedDecodeFn>(
            &GdnPackedDecodeKernelCuda)));
    RegisterOp(
        OpId::kKdaGatedDeltaRule, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<KdaGatedDeltaRuleFn>(&KdaGatedDeltaRuleKernelCuda)));
    RegisterOp(
        OpId::kKdaChunkPrefill, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<KdaChunkPrefillFn>(&KdaChunkPrefillKernelCuda)));
    RegisterOp(
        OpId::kGdnStateGather, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<GdnStateGatherFn>(&GdnStateGatherKernelCuda)));
    RegisterOp(
        OpId::kGdnStateScatter, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<GdnStateScatterFn>(&GdnStateScatterKernelCuda)));
    RegisterOp(OpId::kIndexSelect, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<IndexSelectFn>(&IndexSelectKernelCuda)));
    RegisterOp(OpId::kIndexCopy, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<IndexCopyFn>(&IndexCopyKernelCuda)));
    RegisterOp(OpId::kVHeadPermute, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<VHeadPermuteFn>(&VHeadPermuteKernelCuda)));
    // Mamba2 / SSD device arm (mamba2-ssd.md W2, #496) — sibling ops, never a
    // parameterisation of the GDN kernels above.
    RegisterOp(OpId::kMamba2ChunkScan, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Mamba2ChunkScanFn>(&mamba2::Mamba2ChunkScanKernelCuda)));
    RegisterOp(OpId::kMamba2StateUpdate, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Mamba2StateUpdateFn>(&mamba2::Mamba2StateUpdateKernelCuda)));
    RegisterOp(OpId::kRmsNormGatedGroup, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<RmsNormGatedGroupFn>(&mamba2::RmsNormGatedGroupKernelCuda)));
  }
} registrar;

}  // namespace

#ifdef VLLM_CPP_TRITON
void ReleaseGdnTritonScratch(int device, void* stream_handle) {
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  std::shared_ptr<GdnStreamScratch> scratch;
  {
    std::lock_guard<std::mutex> lk(GdnScratchPoolMutex());
    auto& pool = GdnScratchPool();
    const auto it = pool.find(GdnStreamKey{device, stream});
    if (it == pool.end()) return;
    scratch = std::move(it->second);
    pool.erase(it);
  }

  std::lock_guard<std::mutex> lk(scratch->submit_mu);
  auto release = [&](GdnScratchBuf& buf) {
    if (buf.ptr == nullptr) return;
    Check(cudaFreeAsync(buf.ptr, stream), "gdn scratch queue-destroy free");
    buf.ptr = nullptr;
    buf.bytes = 0;
  };
  release(scratch->chunk.gcum);
  release(scratch->chunk.u);
  release(scratch->chunk.w);
  release(scratch->chunk.v_new);
  release(scratch->chunk.hstate);
  release(scratch->chunk.tok0);
  release(scratch->chunk.len);
  release(scratch->chunk.boh);
  release(scratch->chunk.cidx);
  release(scratch->wu.a);
  release(scratch->wu.ai);
}

namespace testing {
void WarmGdnTritonAotModules(int device) {
  Check(cudaSetDevice(device), "gdn AOT warm set device");
  EnsureAllGdnAotModulesLoaded();
}

size_t PoisonGdnTritonScratch(Queue& queue, unsigned char value) {
  VT_CHECK(queue.device.type == DeviceType::kCUDA,
           "PoisonGdnTritonScratch requires a CUDA queue");
  Check(cudaSetDevice(queue.device.index), "gdn scratch poison set device");
  const cudaStream_t stream = static_cast<cudaStream_t>(queue.handle);
  std::shared_ptr<GdnStreamScratch> scratch;
  {
    std::lock_guard<std::mutex> lk(GdnScratchPoolMutex());
    const auto& pool = GdnScratchPool();
    const auto it = pool.find(GdnStreamKey{queue.device.index, stream});
    if (it == pool.end()) return 0;
    scratch = it->second;
  }

  std::lock_guard<std::mutex> lk(scratch->submit_mu);
  size_t poisoned = 0;
  auto poison = [&](GdnScratchBuf& buf) {
    if (buf.ptr == nullptr) return;
    Check(cudaMemsetAsync(buf.ptr, value, buf.bytes, stream), "gdn scratch test poison");
    ++poisoned;
  };
  poison(scratch->chunk.gcum);
  poison(scratch->chunk.u);
  poison(scratch->chunk.w);
  poison(scratch->chunk.v_new);
  poison(scratch->chunk.hstate);
  poison(scratch->chunk.tok0);
  poison(scratch->chunk.len);
  poison(scratch->chunk.boh);
  poison(scratch->chunk.cidx);
  poison(scratch->wu.a);
  poison(scratch->wu.ai);
  return poisoned;
}

void ResetGdnTritonDebugStats() {
  GdnDebugCounters& c = GdnCounters();
  c.chunk_o_f32_launches.store(0, std::memory_order_relaxed);
  c.chunk_o_bf16_launches.store(0, std::memory_order_relaxed);
  c.chunk_o_hand_launches.store(0, std::memory_order_relaxed);
  c.chunk_pool_allocations.store(0, std::memory_order_relaxed);
  c.chunk_pool_growths.store(0, std::memory_order_relaxed);
  c.chunk_pool_reuses.store(0, std::memory_order_relaxed);
  c.wu_pool_allocations.store(0, std::memory_order_relaxed);
  c.wu_pool_growths.store(0, std::memory_order_relaxed);
  c.wu_pool_reuses.store(0, std::memory_order_relaxed);
  c.enabled.store(true, std::memory_order_release);
}

GdnTritonDebugStats GetGdnTritonDebugStats() {
  GdnDebugCounters& c = GdnCounters();
  GdnTritonDebugStats out;
  out.chunk_o_f32_launches = c.chunk_o_f32_launches.load(std::memory_order_relaxed);
  out.chunk_o_bf16_launches = c.chunk_o_bf16_launches.load(std::memory_order_relaxed);
  out.chunk_o_hand_launches = c.chunk_o_hand_launches.load(std::memory_order_relaxed);
  out.chunk_pool_allocations = c.chunk_pool_allocations.load(std::memory_order_relaxed);
  out.chunk_pool_growths = c.chunk_pool_growths.load(std::memory_order_relaxed);
  out.chunk_pool_reuses = c.chunk_pool_reuses.load(std::memory_order_relaxed);
  out.wu_pool_allocations = c.wu_pool_allocations.load(std::memory_order_relaxed);
  out.wu_pool_growths = c.wu_pool_growths.load(std::memory_order_relaxed);
  out.wu_pool_reuses = c.wu_pool_reuses.load(std::memory_order_relaxed);
  return out;
}

void DisableGdnTritonDebugStats() {
  GdnCounters().enabled.store(false, std::memory_order_release);
}
}  // namespace testing
#endif

namespace testing {
void ResetGdnPackedDecodeDebugStats() {
  g_gdn_packed_launches.store(0, std::memory_order_relaxed);
  g_gdn_packed_triton_launches.store(0, std::memory_order_relaxed);
  g_gdn_packed_reg_tile_launches.store(0, std::memory_order_relaxed);
  g_gdn_packed_legacy_launches.store(0, std::memory_order_relaxed);
  g_gdn_packed_debug_enabled.store(true, std::memory_order_release);
}

GdnPackedDecodeDebugStats GetGdnPackedDecodeDebugStats() {
  GdnPackedDecodeDebugStats out;
  out.launches = g_gdn_packed_launches.load(std::memory_order_relaxed);
  out.triton_launches =
      g_gdn_packed_triton_launches.load(std::memory_order_relaxed);
  out.reg_tile_launches =
      g_gdn_packed_reg_tile_launches.load(std::memory_order_relaxed);
  out.legacy_launches =
      g_gdn_packed_legacy_launches.load(std::memory_order_relaxed);
  return out;
}

void DisableGdnPackedDecodeDebugStats() {
  g_gdn_packed_debug_enabled.store(false, std::memory_order_release);
}
}  // namespace testing

}  // namespace vt::cuda
