// ROCm GDN recurrence core (BACKEND-ROCM-GDN-KERNELS family 4; issue #41).
// kGdnPrefill + kGdnDecode over the PORTABLE scan — hand-translation of
// cuda_gdn.cu GdnScanKernel (:1856) + its launchers, readable side by side
// against the donor. CPU oracle: src/vt/cpu/cpu_ops.cpp:1331/1368
// (GdnHeadTokenStep); formula reference gdn-semantics.md §7/§8. The scan
// family serves both ops in two mappings (qsl != null => varlen prefill;
// qsl == null => single-step decode with optional state_idx + NULL-block
// zero-out): GdnScanCoopK (warp-per-row, two state passes) and
// GdnScanCoopFusedK (register-fused, one state pass + row z-split,
// bit-identical arithmetic) — the launcher picks per mode and dk.
//
// Deliberately NOT ported (docs/ROCM.md §6): the WMMA chunked prefill
// (`#if __CUDA_ARCH__ >= 800`), the fused/packed/Triton-AOT decode variants —
// NVIDIA-only machinery; the scan is the correctness baseline and the perf
// variants are M5 levers (upstream's ROCm answer, Triton/CK, is a later row).
//
// Gate: the recurrence case in test_backend_cross_device.cpp — NMSE <= 5e-4
// vs the CPU oracle on out AND in-place state; decode NULL-slot zero-out
// EXACT.

#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm gdn: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

// ── Recurrence scan (donor cuda_gdn.cu:1856; serves prefill AND decode) ─────
template <typename Tin, typename Tout, typename TState>
__global__ void GdnScanK(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                         const float* g, const float* beta, TState* state,
                         const int32_t* qsl, const int32_t* state_idx,
                         int64_t state_slots, int64_t hk_n, int64_t dk,
                         int64_t hv_n, int64_t dv, float scale) {
  const int64_t s = blockIdx.y;
  const int64_t hv = blockIdx.x;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        St(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Ld(s_row, ki) * decay;
        dot += decayed * k_sh[ki];
      }
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Ld(s_row, ki) * decay + vp * k_sh[ki];
        St(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}
// ── Cooperative scan (TG200 T6a, opt-in VT_GDN_SCAN_COOP=1) ─────────────────
// Same recurrence, different WORK MAPPING: one WARP per state row instead of
// one thread. The donor mapping prices decode at ~61us/call on gfx1100: only
// min(dv, blockDim) of 256 threads do anything, each walking its row's dk
// SERIALLY through dk*sizeof(TState)-strided addresses, so every access is a
// private cache line and the walk is latency-bound (~17 GB/s effective on a
// 1 MiB/call state stream). Here lanes walk ki COALESCED across the row,
// the two dots reduce through a fixed shfl_down tree, and rows iterate
// warp-strided so every wavefront of the state read/write is dense.
// NUMERIC CONTRACT: the elementwise math is unchanged — decayed is still
// Ld*decay computed twice (pass 1 dot, pass 2 update), vp/o formulas are the
// donor's — but the two dot reductions change ORDER (tree vs serial), so
// outputs are NMSE-equal, not bit-exact: this op is cross_device NMSE-gated,
// and engine-level A/B records near-tie adjudication per campaign doctrine.
template <typename Tin, typename Tout, typename TState>
__global__ void GdnScanCoopK(Tout* out, const Tin* q, const Tin* k,
                             const Tin* v, const float* g, const float* beta,
                             TState* state, const int32_t* qsl,
                             const int32_t* state_idx, int64_t state_slots,
                             int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv,
                             float scale) {
  constexpr int NWARPS = kBlock / 32;
  const int64_t s = blockIdx.y;
  const int64_t hv = blockIdx.x;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        St(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = warp; vi < dv; vi += NWARPS) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;
      for (int64_t ki = lane; ki < dk; ki += 32)
        dot += Ld(s_row, ki) * decay * k_sh[ki];
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down(dot, off);
      dot = __shfl(dot, 0);
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
      for (int64_t ki = lane; ki < dk; ki += 32) {
        const float updated = Ld(s_row, ki) * decay + vp * k_sh[ki];
        St(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) o += __shfl_down(o, off);
      if (lane == 0) St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}

// ── Register-fused cooperative scan (RX-7700XT campaign, decode latency) ────
// Same work mapping as GdnScanCoopK (one warp per state row) with two changes
// for the latency-bound decode step on gfx1101:
//   1. FUSED DOT+UPDATE: the row is loaded ONCE into per-lane registers,
//      dotted, updated in place, stored ONCE. The donor mapping reads the
//      state row twice (dot pass + update pass) and writes once — this arm
//      removes one full state pass per row.
//   2. Row z-split: gridDim.z partitions the dv rows across blocks so a short
//      token walk occupies more SMs. ZSPLIT is a launch parameter; the z=1
//      case degenerates to GdnScanCoopK's work split exactly.
// BIT-IDENTICAL to GdnScanCoopK under the build's -ffp-contract=off: every
// product and sum rounds separately in both arms (rv = Ld*decay, then
// rv*k / rv + vp*k), so the per-row state and outputs match the two-pass
// coop math exactly. A contracting build would fuse the two arms
// differently (fma(vp, k, rv) vs fma(Ld, decay, vp*k)) and lose that.
// Serves BOTH modes: qsl == nullptr => decode (begin=s, end=s+1, optional
// state_idx indirection + NULL-block zero-out); qsl != nullptr => varlen
// prefill (per-seq [qsl[s], qsl[s+1]) walk, initial state read from the
// working-state slot s and the final state left there — the engine gathers/
// scatters the persistent pool outside this op). dk is a template bound
// (KD elems per lane = dk/32); callers dispatch dk in {64, 128}; anything
// else falls back to GdnScanCoopK.
template <typename Tin, typename Tout, typename TState, int KD>
__global__ void GdnScanCoopFusedK(Tout* out, const Tin* q, const Tin* k,
                                  const Tin* v, const float* g, const float* beta,
                                  TState* state, const int32_t* qsl,
                                  const int32_t* state_idx, int64_t state_slots,
                                  int64_t hk_n, int64_t dk, int64_t hv_n,
                                  int64_t dv, float scale, int zsplit) {
  constexpr int NWARPS = kBlock / 32;
  const int64_t s = blockIdx.y;
  const int64_t hv = blockIdx.x;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  // Row slice owned by this z-block: contiguous [row_lo, row_hi).
  const int64_t rows_per = (dv + zsplit - 1) / zsplit;
  const int64_t row_lo = blockIdx.z * rows_per;
  const int64_t row_hi = min(dv, row_lo + rows_per);
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = row_lo + threadIdx.x; vi < row_hi; vi += blockDim.x)
        St(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = row_lo + warp; vi < row_hi; vi += NWARPS) {
      TState* s_row = s_head + vi * dk;
      float rv[KD];
      float dot = 0.0f;
#pragma unroll
      for (int ki = 0; ki < KD; ++ki) {
        const int64_t kk = static_cast<int64_t>(lane) + ki * 32;
        rv[ki] = Ld(s_row, kk) * decay;
        dot += rv[ki] * k_sh[kk];
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down(dot, off);
      dot = __shfl(dot, 0);
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
#pragma unroll
      for (int ki = 0; ki < KD; ++ki) {
        const int64_t kk = static_cast<int64_t>(lane) + ki * 32;
        const float updated = rv[ki] + vp * k_sh[kk];
        St(s_row, kk, updated);
        o += updated * q_sh[kk];
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) o += __shfl_down(o, off);
      if (lane == 0) St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}

template <typename Tin, typename Tout>
void LaunchGdnScanState(hipStream_t s, Tensor& out, const Tensor& q_in,
                        const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, Tensor& state, const int32_t* qsl,
                        const int32_t* state_idx, int64_t n,
                        const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  // T6a opt-in (read once per process like the sibling arms): the
  // warp-per-row cooperative mapping. Default ON: measured +1.7 tok/s
  // decode end-to-end with the sibling coop arms on gfx1100 (Qwen3.8-27B
  // EXL3, 2026-10-02). `VT_GDN_SCAN_COOP=0` keeps the donor walk.
  static const bool scan_coop = EnvBoolStrict("VT_GDN_SCAN_COOP", true);
  // Register-fused arm (same mapping, one state pass, bit-identical to the
  // coop arm — see the kernel header): serves decode (qsl == nullptr) AND
  // varlen prefill (qsl != nullptr) — the kernel handles qsl, state_idx,
  // begin/end and the row-sliced zero-out internally; initial/final state
  // live in the working-state slot per seq, exactly as GdnScanCoopK sees it.
  // dk not in {64,128} stays on GdnScanCoopK. VT_GDN_SCAN_ZSPLIT overrides
  // the decode row split (0 disables the fused arm entirely, both modes);
  // VT_GDN_SCAN_ZSPLIT_PREFILL sets the prefill split (0 keeps prefill on
  // the coop arm).
  // gridDim.z is capped at 65535 and a split wider than dv only wastes idle
  // blocks — clamp rather than launch invalid.
  static const int kZsplitEnv =
      std::max(0, std::min(EnvIntStrict("VT_GDN_SCAN_ZSPLIT", 4), 65535));
  // Prefill z-split default 8: measured on gfx1100 (hk=16, hv=48, dk=dv=128,
  // f32 state) — T>=7 favors z=8 (1.9 us/tok vs coop 14-16), tiny walks
  // (T<=4) favor z=4. gdn_bench.hip, review/GdnPrefillScan/REPORT.md.
  // Read per prefill call (one getenv per GDN layer per prefill step, never
  // on the captured decode path) so a test process can A/B the two arms.
  const int zsplit_prefill = [qsl] {
    if (qsl == nullptr) return 0;
    const int v = EnvIntStrict("VT_GDN_SCAN_ZSPLIT_PREFILL", 8);
    return std::max(0, std::min(v, 65535));
  }();
  const int64_t t_total = q_in.shape[0];
  const bool fused_ok =
      scan_coop && kZsplitEnv > 0 && (dk == 64 || dk == 128) &&
      (qsl == nullptr || zsplit_prefill > 0);
  const int zsplit =
      fused_ok ? std::min<int>(qsl == nullptr ? kZsplitEnv
                                             : (t_total <= 4
                                                    ? std::min(zsplit_prefill, 4)
                                                    : zsplit_prefill),
                               dv)
               : 1;
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n),
                  static_cast<unsigned>(zsplit));
  const size_t shmem = 2 * static_cast<size_t>(dk) * sizeof(float);
  if (state.dtype == DType::kF16) {
    if (fused_ok) {
      if (dk == 128)
        GdnScanCoopFusedK<Tin, Tout, __half, 4><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
      else
        GdnScanCoopFusedK<Tin, Tout, __half, 2><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
    } else if (scan_coop) {
      GdnScanCoopK<Tin, Tout, __half><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl, state_idx,
          state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    } else {
      GdnScanK<Tin, Tout, __half><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl, state_idx,
          state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    }
  } else if (state.dtype == DType::kBF16) {
    if (fused_ok) {
      if (dk == 128)
        GdnScanCoopFusedK<Tin, Tout, __hip_bfloat16, 4><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
      else
        GdnScanCoopFusedK<Tin, Tout, __hip_bfloat16, 2><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
    } else if (scan_coop) {
      GdnScanCoopK<Tin, Tout, __hip_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
          state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    } else {
      GdnScanK<Tin, Tout, __hip_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
          state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    }
  } else {
    if (fused_ok) {
      if (dk == 128)
        GdnScanCoopFusedK<Tin, Tout, float, 4><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
      else
        GdnScanCoopFusedK<Tin, Tout, float, 2><<<grid, kBlock, shmem, s>>>(
            out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
            g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl,
            state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale, zsplit);
    } else if (scan_coop) {
      GdnScanCoopK<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl, state_idx,
          state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    } else {
      GdnScanK<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
          g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl, state_idx,
          state.shape[0], hk_n, dk, hv_n, dv, args.scale);
    }
  }
  Check(hipGetLastError(), "gdn scan launch");
}
template <typename Tin>
void LaunchGdnScanIn(hipStream_t s, Tensor& out, const Tensor& q_in,
                     const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, Tensor& state, const int32_t* qsl,
                     const int32_t* state_idx, int64_t n, const GdnArgs& args) {
  if (out.dtype == DType::kBF16)
    LaunchGdnScanState<Tin, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl,
                                            state_idx, n, args);
  else
    LaunchGdnScanState<Tin, float>(s, out, q_in, k, v, g, beta, state, qsl,
                                   state_idx, n, args);
}
}  // namespace

void GdnPrefillKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta,
                          Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  const int64_t n = state.shape[0];
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                    qsl.Ptr<int32_t>(), nullptr, n, args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(),
                           nullptr, n, args);
  }
}

void GdnDecodeKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor* state_idx, const GdnArgs& args) {
  const int64_t n = q_in.shape[0];
  if (n == 0) return;
  const int32_t* si = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  hipStream_t s = AsStream(q);
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state, nullptr, si, n,
                                    args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, nullptr, si, n, args);
  }
}

}  // namespace vt::rocm
