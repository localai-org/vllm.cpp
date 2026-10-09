// ROCm GDN norm/gate + fused attention preamble (BACKEND-ROCM-GDN-KERNELS
// family 5, the final slice; issue #41). kRmsNormGated (gdn-semantics §5),
// kSigmoidGateBf16 (ops.cpp contract: out bf16, attn f32/bf16, gate f32 — the
// sigmoid input stays unrounded), kAttnQkNormRopeGate (fused full-attention
// preamble: split q|gate + gemma/plain qk-RMSNorm + partial NeoX RoPE-from-
// cache + gate passthrough). Donors: cuda_gdn.cu RmsNormGatedRowKernel and
// cuda_ops.cu:1429 area; CPU oracles cpu_ops.cpp:1210/:2273/:956.
// kSigmoidGateBf16 has NO CUDA registration — the CPU composite semantics are
// the donor of record (the Vulkan lane's only native sibling).
//
// Gate: the norm-gate and preamble arms of test_backend_cross_device.cpp —
// NMSE <= 5e-4 for the norm/preamble arithmetic; SigmoidGateBf16 bit-exact
// (single multiply, same RNE store on both sides).

#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include "vt/rocm/hip_shfl_compat.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdlib>
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
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}
__device__ inline float Silu(float x) { return x / (1.0f + expf(-x)); }
__device__ inline float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// ── Norm/gate (donor cuda_gdn.cu RmsNormGatedRowKernel; CPU :1210/:2273) ─────
template <typename Tx, typename Tout>
__global__ void RmsNormGatedK(Tout* out, const Tx* x, const Tx* gate,
                              const Tx* w, int64_t rows, int64_t d,
                              int64_t gate_group, int64_t gate_outer, float eps,
                              bool sigmoid_gate) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  float sumsq = 0.0f;
  for (int64_t j = 0; j < d; ++j) {
    const float v = Ld(x, i * d + j);
    sumsq += v * v;
  }
  const float inv = 1.0f / sqrtf(sumsq / static_cast<float>(d) + eps);
  const int64_t gbase = (i / gate_group) * gate_outer + (i % gate_group) * d;
  for (int64_t j = 0; j < d; ++j) {
    const float z = Ld(gate, gbase + j);
    const float act = sigmoid_gate ? Sigmoid(z) : Silu(z);
    St(out, i * d + j, Ld(x, i * d + j) * inv * Ld(w, j) * act);
  }
}

// The donor kernel runs ONE THREAD PER ROW (`<<<rows, 1>>>`): each row walks
// d twice serially — 24 launches/tok x 17.25us = 0.414 ms/tok of pure
// single-thread latency. This arm gives each row a whole 256-thread block:
// strided-per-thread sumsq with a wavefront-shfl reduction (width from
// warpSize), then a strided gated store. The float association CHANGES
// (opt-in flag, adjudication owed like its T6/T8 siblings); no quant
// contract exists on this op.
template <typename Tx, typename Tout>
__global__ void RmsNormGatedCoopK(Tout* out, const Tx* x, const Tx* gate,
                                  const Tx* w, int64_t rows, int64_t d,
                                  int64_t gate_group, int64_t gate_outer, float eps,
                                  bool sigmoid_gate) {
  const int64_t i = blockIdx.x;  // one row per block
  if (i >= rows) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int waveSz = warpSize;
  __shared__ float partial[256 / 32];  // sized for the narrowest wavefront
  const Tx* xrow = x + i * d;
  // Pass 1: strided-per-thread sumsq — lanes touch consecutive addresses,
  // which IS the coalesced pattern for a streaming pass.
  float acc = 0.0f;
  for (int64_t j = tid; j < d; j += 256) {
    const float v = Ld(xrow, j);
    acc += v * v;
  }
  for (int off = waveSz / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffULL, acc, off);
  if (tid % waveSz == 0) partial[tid / waveSz] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
    const int nw = 256 / waveSz;
    for (int wv = 0; wv < nw; ++wv) t += partial[wv];
    partial[0] = t;
  }
  __syncthreads();
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);
  // Pass 2: gated scaled store, same strided pattern.
  const int64_t gbase = (i / gate_group) * gate_outer + (i % gate_group) * d;
  for (int64_t j = tid; j < d; j += 256) {
    const float z = Ld(gate, gbase + j);
    const float act = sigmoid_gate ? Sigmoid(z) : Silu(z);
    St(out, i * d + j, Ld(xrow, j) * inv * Ld(w, j) * act);
  }
}

// ops.cpp contract: out bf16, attn f32-or-bf16, gate f32 (unrounded sigmoid
// input). Tattn only.
template <typename Tattn>
__global__ void SigmoidGateK(__hip_bfloat16* out, const Tattn* attn,
                             const float* gate, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    St(out, i, Ld(attn, i) * Sigmoid(gate[i]));
  }
}

template <typename Tsrc, typename Tqk, typename Tgate>
__global__ void AttnQkNormRopeGateK(Tqk* q_out, Tqk* k_out, Tgate* gate_out,
                                    const Tsrc* qgate, const Tsrc* kf,
                                    const float* q_norm, const float* k_norm,
                                    const float* cos_sin, int64_t t, int64_t hq,
                                    int64_t hkv, int64_t dh, int64_t qgate_stride,
                                    int64_t kf_stride, int rot, float eps,
                                    bool gemma) {
  const int64_t half = rot / 2;
  const int64_t items = t * (hq + hkv);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < items; item += step) {
    const int64_t tok = item / (hq + hkv);
    const int64_t h = item % (hq + hkv);
    const float* cs = cos_sin + tok * rot;
    int64_t src_off, out_off;
    if (h < hq) {
      src_off = tok * qgate_stride + h * 2 * dh;
      out_off = (tok * hq + h) * dh;
      for (int64_t j = 0; j < dh; ++j)
        St(gate_out, out_off + j, Ld(qgate, src_off + dh + j));
      float ss = 0.0f;
      for (int64_t j = 0; j < dh; ++j) {
        const float v = Ld(qgate, src_off + j);
        ss += v * v;
      }
      const float inv = 1.0f / sqrtf(ss / static_cast<float>(dh) + eps);
      for (int64_t j = 0; j < dh; ++j) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + j + half), inv, q_norm[j + half], gemma);
          St(q_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(qgate, src_off + i), inv, q_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + i + half), inv, q_norm[i + half], gemma);
          St(q_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(q_out, out_off + j,
             GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma));
        }
      }
    } else {
      const int64_t hk_i = h - hq;
      src_off = tok * kf_stride + hk_i * dh;
      out_off = (tok * hkv + hk_i) * dh;
      float ss = 0.0f;
      for (int64_t j = 0; j < dh; ++j) {
        const float v = Ld(kf, src_off + j);
        ss += v * v;
      }
      const float inv = 1.0f / sqrtf(ss / static_cast<float>(dh) + eps);
      for (int64_t j = 0; j < dh; ++j) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + j + half), inv, k_norm[j + half], gemma);
          St(k_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(kf, src_off + i), inv, k_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + i + half), inv, k_norm[i + half], gemma);
          St(k_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(k_out, out_off + j,
             GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma));
        }
      }
    }
  }
}
// ── Cooperative preamble (TG200 T6b, opt-in VT_ATTN_PREAMBLE_COOP=1) ───────
// Donor mapping prices decode at ~88us/call: items = t*(hq+hkv) = 24 at
// batch 1, so 24 of 256 threads run, each doing three serial dh-loops
// (gate copy, sumsq, rope/norm store) through private cache lines. Here one
// WARP handles one item: lanes stride j for the gate copy and the stores,
// the sumsq reduces through a fixed shfl_down tree and inv is broadcast.
// NUMERIC CONTRACT: every element's math is the donor's given inv; only the
// ss summation ORDER changes (tree vs serial), so outputs are NMSE-equal,
// not bit-exact — op is cross_device NMSE-gated, engine A/B records near-tie
// adjudication per campaign doctrine.
template <typename Tsrc, typename Tqk, typename Tgate>
__global__ void AttnQkNormRopeGateCoopK(Tqk* q_out, Tqk* k_out, Tgate* gate_out,
                                        const Tsrc* qgate, const Tsrc* kf,
                                        const float* q_norm, const float* k_norm,
                                        const float* cos_sin, int64_t t, int64_t hq,
                                        int64_t hkv, int64_t dh, int64_t qgate_stride,
                                        int64_t kf_stride, int rot, float eps,
                                        bool gemma) {
  constexpr int NWARPS = kBlock / 32;
  const int64_t half = rot / 2;
  const int64_t items = t * (hq + hkv);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  for (int64_t item = warp; item < items; item += NWARPS) {
    const int64_t tok = item / (hq + hkv);
    const int64_t h = item % (hq + hkv);
    const float* cs = cos_sin + tok * rot;
    if (h < hq) {
      const int64_t src_off = tok * qgate_stride + h * 2 * dh;
      const int64_t out_off = (tok * hq + h) * dh;
      for (int64_t j = lane; j < dh; j += 32)
        St(gate_out, out_off + j, Ld(qgate, src_off + dh + j));
      float ss = 0.0f;
      for (int64_t j = lane; j < dh; j += 32) {
        const float v = Ld(qgate, src_off + j);
        ss += v * v;
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_down(ss, off);
      const float inv =
          1.0f / sqrtf(__shfl(ss, 0) / static_cast<float>(dh) + eps);
      for (int64_t j = lane; j < dh; j += 32) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + j + half), inv, q_norm[j + half], gemma);
          St(q_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(qgate, src_off + i), inv, q_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + i + half), inv, q_norm[i + half], gemma);
          St(q_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(q_out, out_off + j,
             GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma));
        }
      }
    } else {
      const int64_t hk_i = h - hq;
      const int64_t src_off = tok * kf_stride + hk_i * dh;
      const int64_t out_off = (tok * hkv + hk_i) * dh;
      float ss = 0.0f;
      for (int64_t j = lane; j < dh; j += 32) {
        const float v = Ld(kf, src_off + j);
        ss += v * v;
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_down(ss, off);
      const float inv =
          1.0f / sqrtf(__shfl(ss, 0) / static_cast<float>(dh) + eps);
      for (int64_t j = lane; j < dh; j += 32) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + j + half), inv, k_norm[j + half], gemma);
          St(k_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(kf, src_off + i), inv, k_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + i + half), inv, k_norm[i + half], gemma);
          St(k_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(k_out, out_off + j,
             GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma));
        }
      }
    }
  }
}
}  // namespace
void RmsNormGatedKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "rocm rmsnorm_gated: unsupported input dtype (f32/bf16 only)");
  VT_CHECK(gate.dtype == x.dtype && w.dtype == x.dtype,
           "rocm rmsnorm_gated: gate/weight dtype must match x");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = d == 0 ? 0 : x.Numel() / d;
  if (t == 0 || d == 0) return;
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  hipStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(t);
  // T9 arm (read PER CALL like the sibling flags): one block per row,
  // cooperative reduction. Default ON — measured on gfx1100 (2026-10-02);
  // `VT_GDN_NORMGATED_COOP=0` keeps the donor kernel.
  const bool coop = EnvBoolStrict("VT_GDN_NORMGATED_COOP", true);
  if (coop) {
    if (x.dtype == DType::kF32) {
      if (out.dtype == DType::kF32) {
        RmsNormGatedCoopK<float, float><<<grid, 256, 0, s>>>(
            out.Ptr<float>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), t, d,
            gate_group, gate_outer, args.eps, args.sigmoid_gate);
      } else {
        RmsNormGatedCoopK<float, __hip_bfloat16><<<grid, 256, 0, s>>>(
            out.Ptr<__hip_bfloat16>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(),
            t, d, gate_group, gate_outer, args.eps, args.sigmoid_gate);
      }
    } else {
      if (out.dtype == DType::kF32) {
        RmsNormGatedCoopK<__hip_bfloat16, float><<<grid, 256, 0, s>>>(
            out.Ptr<float>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
            w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
            args.sigmoid_gate);
      } else {
        RmsNormGatedCoopK<__hip_bfloat16, __hip_bfloat16><<<grid, 256, 0, s>>>(
            out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
            w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
            args.sigmoid_gate);
      }
    }
    Check(hipGetLastError(), "rmsnorm_gated coop launch");
    return;
  }
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      RmsNormGatedK<float, float><<<grid, 1, 0, s>>>(
          out.Ptr<float>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), t, d,
          gate_group, gate_outer, args.eps, args.sigmoid_gate);
    } else {
      RmsNormGatedK<float, __hip_bfloat16><<<grid, 1, 0, s>>>(
          out.Ptr<__hip_bfloat16>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(),
          t, d, gate_group, gate_outer, args.eps, args.sigmoid_gate);
    }
  } else {
    if (out.dtype == DType::kF32) {
      RmsNormGatedK<__hip_bfloat16, float><<<grid, 1, 0, s>>>(
          out.Ptr<float>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
          w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
          args.sigmoid_gate);
    } else {
      RmsNormGatedK<__hip_bfloat16, __hip_bfloat16><<<grid, 1, 0, s>>>(
          out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
          w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
          args.sigmoid_gate);
    }
  }
  Check(hipGetLastError(), "rmsnorm_gated launch");
}

void SigmoidGateBf16KernelRocm(Queue& q, Tensor& out, const Tensor& attn,
                               const Tensor& gate) {
  // Contract (ops.cpp:3350): out bf16, attn f32/bf16, gate f32.
  VT_CHECK(out.dtype == DType::kBF16, "rocm sigmoid_gate_bf16: out must be bf16");
  VT_CHECK(gate.dtype == DType::kF32, "rocm sigmoid_gate_bf16: gate must be f32");
  const int64_t n = out.Numel();
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  if (attn.dtype == DType::kBF16) {
    SigmoidGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(),
                                               attn.Ptr<__hip_bfloat16>(),
                                               gate.Ptr<float>(), n);
  } else {
    SigmoidGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(),
                                               attn.Ptr<float>(),
                                               gate.Ptr<float>(), n);
  }
  Check(hipGetLastError(), "sigmoid_gate launch");
}

// ── Gate-free fused attention preamble (kAttnQkNormRope) ────────────────────
// Per-head STANDARD RMSNorm(q) + RMSNorm(k) + partial RoPE-from-cache, the
// Qwen3-dense preamble, in one launch. The normalized value is carried in F32
// through the rotation and narrowed ONCE, at the store: that is the pinned
// primary's boundary -- Qwen3Attention applies q_norm/k_norm and then rotary_emb
// (vllm/model_executor/models/qwen3.py:150-167) and the compiled preamble reads a
// cos|sin cache narrowed to the query dtype
// (vllm/model_executor/layers/rotary_embedding/base.py:105-131), so nothing is
// rounded to bf16 before the rotation. The Tier-0 composite this op realises
// (RmsNorm into a bf16 buffer, then RopeFromCache over that buffer) rounds twice
// and measured 1569 of 8448 Q words against the primary on the L33/C2/R0
// workload; see .agents/specs/rocm-attn-parity.md.
// The RMSNorm reduction is the shipped one -- same kBlock-strided accumulation
// and same shared-memory tree as RmsNormRowKernel (rocm_rmsnorm.hip:118-147) --
// so the f32 carrier keeps the norm's own bits and only the narrowing point
// moves.
template <typename T, typename Tid>
__device__ void AttnQkNormRopeRow(int64_t row, T* x, const T* w, const T* cache,
                                  const Tid* positions, int64_t cache_rows, int64_t heads,
                                  int64_t dh, int64_t row_stride, int rot, float eps, bool neox,
                                  bool gemma) {
  const int64_t pos = static_cast<int64_t>(positions[row / heads]);
  if (rot <= 0 || pos < 0 || pos >= cache_rows) return;
  T* xrow = x + row * row_stride;
  const T* cs = cache + pos * rot;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  const bool compiled256 = gemma && dh == 256;
  // Executing Triton reduction: R0_BLOCK=64, eight adjacent values per lane.
  // Sum positions j, j+64, j+128, j+192 before the eight-value lane reduction.
  const int width = compiled256 ? 64 : kBlock;
  if (static_cast<int>(threadIdx.x) < width) {
    for (int64_t j = threadIdx.x; j < dh; j += width) {
      const float v = Ld(xrow, j);
      acc += v * v;
    }
  }
  float total;
  if (compiled256) {
    const int lane = threadIdx.x & 31;
    float group = __shfl(acc, lane & ~7);
#pragma unroll
    for (int j = 1; j < 8; ++j) group += __shfl(acc, (lane & ~7) + j);
    if (threadIdx.x < 64 && (lane & 7) == 0) partial[threadIdx.x / 8] = group;
    __syncthreads();
    float reduced = lane < 8 ? partial[lane] : 0.f;
#pragma unroll
    for (int stride = 4; stride; stride /= 2) reduced += __shfl_xor(reduced, stride);
    total = __shfl(reduced, 0);
  } else {
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride /= 2) {
      if (static_cast<int>(threadIdx.x) < stride)
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    total = partial[0];
  }
  const float variance = total / static_cast<float>(dh) + eps;
  const float inv = gemma ? __builtin_amdgcn_rsqf(fmaf(total, 1.f / static_cast<float>(dh), eps))
                          : 1.0f / sqrtf(variance);
  const int64_t half = rot / 2;
  // One thread per rotation PAIR, both stores from the values read before either
  // write, because the rotation is in place (RopeFromCacheK, rocm_dense_basic.hip
  // :699-704). The pair is the only place the bf16 narrowing is deferred.
  for (int64_t pair = threadIdx.x; pair < half; pair += kBlock) {
    const int64_t first = neox ? pair : pair * 2;
    const int64_t second = neox ? pair + half : pair * 2 + 1;
    const float c = Ld(cs, pair);
    const float sn = Ld(cs, half + pair);
    const float x1 = Ld(xrow, first) * inv * (Ld(w, first) + (gemma ? 1.f : 0.f));
    const float x2 = Ld(xrow, second) * inv * (Ld(w, second) + (gemma ? 1.f : 0.f));
    St(xrow, first, gemma ? fmaf(x1, c, -(x2 * sn)) : x1 * c - x2 * sn);
    St(xrow, second, gemma ? fmaf(x2, c, x1 * sn) : x1 * sn + x2 * c);
  }
  // Outside the rotary width the normed value is only stored, narrowed once.
  for (int64_t j = rot + threadIdx.x; j < dh; j += kBlock) {
    St(xrow, j, Ld(xrow, j) * inv * (Ld(w, j) + (gemma ? 1.f : 0.f)));
  }
}

template <typename T, typename Tid>
__global__ void AttnQkNormRopeRowK(T* x, const T* w, const T* cache, const Tid* positions,
                                   int64_t cache_rows, int64_t heads, int64_t dh,
                                   int64_t row_stride, int rot, float eps, bool neox, bool gemma) {
  AttnQkNormRopeRow(blockIdx.x, x, w, cache, positions, cache_rows, heads, dh, row_stride, rot, eps,
                    neox, gemma);
}

template <typename T, typename Tid>
__global__ void GemmaQkNormRopeRowsK(T* query, T* key, const T* qw, const T* kw, const T* cache,
                                     const Tid* positions, int64_t cache_rows, int64_t hq,
                                     int64_t hkv, int64_t q_stride, int64_t k_stride, int rot,
                                     float eps, bool neox) {
  const int64_t token = blockIdx.x / (hq + hkv), head = blockIdx.x % (hq + hkv);
  if (head < hq)
    AttnQkNormRopeRow(token * hq + head, query, qw, cache, positions, cache_rows, hq, 256, q_stride,
                      rot, eps, neox, true);
  else
    AttnQkNormRopeRow(token * hkv + head - hq, key, kw, cache, positions, cache_rows, hkv, 256,
                      k_stride, rot, eps, neox, true);
}

void AttnQkNormRopeGateKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                                  const Tensor& qgate, const Tensor& kf,
                                  const Tensor& q_norm, const Tensor& k_norm,
                                  const Tensor& cos_sin, const RmsNormArgs& na,
                                  const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  if (t == 0) return;
  const int64_t items = t * (hq + hkv);
  hipStream_t s = AsStream(q);
  auto launch = [&](auto src_tag, auto qk_tag, auto gate_tag) {
    using Tsrc = decltype(src_tag);
    using Tqk = decltype(qk_tag);
    using Tgate = decltype(gate_tag);
    // T6b arm (read once per process like the sibling arms): warp-per-item
    // cooperative mapping. Default ON — measured on gfx1100 with the GDN
    // coop arms (2026-10-02); `VT_ATTN_PREAMBLE_COOP=0` keeps the donor walk.
    static const bool preamble_coop =
        EnvBoolStrict("VT_ATTN_PREAMBLE_COOP", true);
    if (preamble_coop) {
      AttnQkNormRopeGateCoopK<Tsrc, Tqk, Tgate><<<GridFor(items), kBlock, 0, s>>>(
          q_out.Ptr<Tqk>(), k_out.Ptr<Tqk>(), gate_out.Ptr<Tgate>(), qgate.Ptr<Tsrc>(),
          kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(),
          t, hq, hkv, dh, qgate.stride[0], kf.stride[0], ra.rotary_dim, na.eps,
          na.gemma);
    } else {
      AttnQkNormRopeGateK<Tsrc, Tqk, Tgate><<<GridFor(items), kBlock, 0, s>>>(
          q_out.Ptr<Tqk>(), k_out.Ptr<Tqk>(), gate_out.Ptr<Tgate>(), qgate.Ptr<Tsrc>(),
          kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(),
          t, hq, hkv, dh, qgate.stride[0], kf.stride[0], ra.rotary_dim, na.eps,
          na.gemma);
    }
  };
  // Dispatch on the OUTPUT dtype like the CUDA lane (cuda_ops.cu
  // LaunchAttnPreambleOut): the caller picks f32 out (token-exact path) or bf16
  // out (FA-2 combos) INDEPENDENTLY of the source dtype — the 0.8B bf16 model
  // runs bf16 src + f32 q/k out + f32 gate on the f32-attention path, which the
  // old src-keyed dispatch silently mis-launched as all-bf16 (writing bf16 bits
  // through the f32 out pointer; issue #41 M4 W2 finding). Validated combos
  // (mirror CUDA): (f32 src, f32 out); (bf16 src, f32 out + f32 gate);
  // (bf16 src, bf16 out + bf16 gate); (bf16 src, bf16 out + f32 gate).
  const bool src_bf16 = qgate.dtype == DType::kBF16;
  if (q_out.dtype == DType::kF32) {
    if (src_bf16) {
      launch(__hip_bfloat16{}, float{}, float{});
    } else {
      launch(float{}, float{}, float{});
    }
  } else {
    VT_CHECK(src_bf16, "rocm attn_qk_norm_rope_gate: bf16 out needs bf16 src");
    if (gate_out.dtype == DType::kF32) {
      launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
    } else {
      launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
    }
  }
  Check(hipGetLastError(), "attn_qk_norm_rope_gate launch");
}

// The registered AttnQkNormRopeFn (include/vt/ops.h). The f32 attention arm has
// no intermediate rounding to remove -- RmsNorm(f32) then RopeFromCache(f32) is
// already the primary's boundary -- so it keeps the shipped composite's bytes by
// running that composite, and only the bf16 arm takes the fused kernel.
void AttnQkNormRopeKernelRocm(Queue& q, Tensor& q3, Tensor& k3, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const Tensor& positions, const RmsNormArgs& na,
                              const RopeArgs& ra) {
  const int64_t t = q3.shape[0], hq = q3.shape[1], dh = q3.shape[2];
  const int64_t hkv = k3.shape[1];
  if (t == 0) return;
  VT_CHECK(q3.rank == 3 && k3.rank == 3 && k3.shape[2] == dh,
           "rocm attn_qk_norm_rope: q3/k3 must be [T,H,Dh]");
  VT_CHECK(q_norm.rank == 1 && q_norm.shape[0] == dh && k_norm.rank == 1 &&
               k_norm.shape[0] == dh,
           "rocm attn_qk_norm_rope: q_norm/k_norm must be [Dh]");
  VT_CHECK(cos_sin.rank == 2 && cos_sin.shape[1] == ra.rotary_dim,
           "rocm attn_qk_norm_rope: cos_sin must be [rows, rotary_dim]");
  VT_CHECK(cos_sin.dtype == q3.dtype && q_norm.dtype == q3.dtype && k_norm.dtype == q3.dtype,
           "rocm attn_qk_norm_rope: weights and cache must match the states' dtype");
  if (q3.dtype == DType::kF32) {
    Tensor q2d = q3;
    q2d.rank = 2;
    q2d.shape[0] = t * hq;
    q2d.shape[1] = dh;
    q2d.stride[0] = dh;
    q2d.stride[1] = 1;
    Tensor k2d = k3;
    k2d.rank = 2;
    k2d.shape[0] = t * hkv;
    k2d.shape[1] = dh;
    k2d.stride[0] = dh;
    k2d.stride[1] = 1;
    RmsNorm(q, q2d, q2d, q_norm, na, nullptr);
    RmsNorm(q, k2d, k2d, k_norm, na, nullptr);
    RopeFromCache(q, q3, &k3, positions, cos_sin, ra);
    return;
  }
  VT_CHECK(q3.dtype == DType::kBF16,
           "rocm attn_qk_norm_rope: states must be f32 or bf16");
  const int64_t rot = ra.rotary_dim;
  VT_CHECK(rot > 0 && rot % 2 == 0 && rot <= dh,
           "rocm attn_qk_norm_rope: rotary_dim must be even and <= Dh");
  hipStream_t s = AsStream(q);
  auto launch = [&](auto t_tag, auto id_tag) {
    using T = decltype(t_tag);
    using Tid = decltype(id_tag);
    if (na.gemma && dh == 256 && hq > 0 && hkv > 0) {
      GemmaQkNormRopeRowsK<T, Tid><<<static_cast<unsigned>(t * (hq + hkv)), kBlock, 0, s>>>(
          q3.Ptr<T>(), k3.Ptr<T>(), q_norm.Ptr<T>(), k_norm.Ptr<T>(), cos_sin.Ptr<T>(),
          positions.Ptr<Tid>(), cos_sin.shape[0], hq, hkv, q3.stride[1], k3.stride[1], rot, na.eps,
          ra.is_neox_style);
      return;
    }
    if (hq > 0)
      AttnQkNormRopeRowK<T, Tid><<<static_cast<unsigned>(t * hq), kBlock, 0, s>>>(
          q3.Ptr<T>(), q_norm.Ptr<T>(), cos_sin.Ptr<T>(), positions.Ptr<Tid>(), cos_sin.shape[0],
          hq, dh, q3.stride[1], rot, na.eps, ra.is_neox_style, na.gemma);
    if (hkv > 0)
      AttnQkNormRopeRowK<T, Tid><<<static_cast<unsigned>(t * hkv), kBlock, 0, s>>>(
          k3.Ptr<T>(), k_norm.Ptr<T>(), cos_sin.Ptr<T>(), positions.Ptr<Tid>(), cos_sin.shape[0],
          hkv, dh, k3.stride[1], rot, na.eps, ra.is_neox_style, na.gemma);
  };
  if (positions.dtype == DType::kI32) {
    launch(__hip_bfloat16{}, int32_t{});
  } else {
    launch(__hip_bfloat16{}, int64_t{});
  }
  Check(hipGetLastError(), "attn_qk_norm_rope launch");
}

}  // namespace vt::rocm
