// ROCm GDN post-conv glue (BACKEND-ROCM-GDN-KERNELS family 3; issue #41).
// kGdnPostConv: conv-split + q/k l2norm + g/beta in ONE launch (mirror of fla
// fused_gdn_prefill_post_conv; gdn-semantics.md §4/§6). Hand-translated from
// cuda_gdn.cu:1155, readable side by side against the donor; CPU oracle
// src/vt/cpu/cpu_ops.cpp:2337. This is the fused VT_GLUE_FUSE path the model
// calls by default — the M2 throw after conv lands. Sequential per
// (token, head) thread here, matching the CPU oracle's reduction order; the
// donor's block-tree reduction and the Fast/Split variants are M5 levers.
//
// Gate: the postconv arm of the fused-glue case in
// test_backend_cross_device.cpp — NMSE <= 5e-4 vs the CPU oracle, padded a/b
// row strides included.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <cstdlib>
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

// ── Fused glue (donors cuda_gdn.cu:1155; cpu oracle :956 for the preamble) ──
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvK(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                             float* beta_out, const Tconv* conv, const Tgate* araw,
                             const Tgate* braw, const float* a_log,
                             const float* dt_bias, int64_t t, int64_t hk, int64_t dk,
                             int64_t hv, int64_t dv, int64_t a_row_stride,
                             int64_t b_row_stride, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t items = t * (hk + 1);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < items; item += step) {
    const int64_t tok = item / (hk + 1);
    const int64_t head = item % (hk + 1);
    const int64_t row = tok * conv_dim;
    if (head < hk) {
      const Tconv* qin = conv + row + head * dk;
      const Tconv* kin = conv + row + key_dim + head * dk;
      float qss = 0.0f, kss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) {
        const float qv = Ld(qin, j);
        const float kv = Ld(kin, j);
        qss += qv * qv;
        kss += kv * kv;
      }
      const float qinv = 1.0f / sqrtf(qss + eps);
      const float kinv = 1.0f / sqrtf(kss + eps);
      Tqkv* qo = q_out + (tok * hk + head) * dk;
      Tqkv* ko = k_out + (tok * hk + head) * dk;
      for (int64_t j = 0; j < dk; ++j) {
        St(qo, j, Ld(qin, j) * qinv);
        St(ko, j, Ld(kin, j) * kinv);
      }
    } else {
      const Tconv* vin = conv + row + 2 * key_dim;
      Tqkv* vo = v_out + tok * value_dim;
      for (int64_t j = 0; j < value_dim; ++j) St(vo, j, Ld(vin, j));
      for (int64_t h = 0; h < hv; ++h) {
        const int64_t idx = tok * hv + h;
        const float av = Ld(araw, tok * a_row_stride + h);
        const float bv = Ld(braw, tok * b_row_stride + h);
        const float x = av + dt_bias[h];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        g_out[idx] = -expf(a_log[h]) * sp;
        beta_out[idx] = 1.0f / (1.0f + expf(-bv));
      }
    }
  }
}

// Same work as GdnPostConvK, decomposed so no single thread carries the whole
// v copy. The original splits by (token, head) into t*(hk+1) items, and the ONE
// item with head==hk copies the entire value_dim row -- 4096 elements at
// Hv=32/Dv=128 -- while the other hk items do ~4*dk each. At decode (t=1) that
// is 17 items in a single workgroup with an 8x imbalance, so the kernel's
// duration is set by that one lane.
//
// Here a token's slots are: hk q/k heads, then v_chunks copy chunks of v_chunk
// elements each, then one slot for the gates. Every element operation is
// identical and independent (the v copy is elementwise, the gates are per-h), so
// this reorders WHICH thread does a given element and never the arithmetic --
// output is bit-exact.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvChunkedK(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                                    float* beta_out, const Tconv* conv, const Tgate* araw,
                                    const Tgate* braw, const float* a_log,
                                    const float* dt_bias, int64_t t, int64_t hk, int64_t dk,
                                    int64_t hv, int64_t dv, int64_t a_row_stride,
                                    int64_t b_row_stride, float eps, int64_t v_chunk,
                                    int64_t v_chunks) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  // One slot per gate head too: the shared gate slot was ONE thread running
  // hv serial expf/log1pf chains (~36us measured at hv=48 on gfx1100, the
  // kernel's whole duration at t=1). Every head's output is an independent
  // elementwise result, so splitting the slot is bit-exact.
  const int64_t slots = hk + v_chunks + hv;
  const int64_t items = t * slots;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < items; item += step) {
    const int64_t tok = item / slots;
    const int64_t slot = item % slots;
    const int64_t row = tok * conv_dim;
    if (slot < hk) {
      const int64_t head = slot;
      const Tconv* qin = conv + row + head * dk;
      const Tconv* kin = conv + row + key_dim + head * dk;
      float qss = 0.0f, kss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) {
        const float qv = Ld(qin, j);
        const float kv = Ld(kin, j);
        qss += qv * qv;
        kss += kv * kv;
      }
      const float qinv = 1.0f / sqrtf(qss + eps);
      const float kinv = 1.0f / sqrtf(kss + eps);
      Tqkv* qo = q_out + (tok * hk + head) * dk;
      Tqkv* ko = k_out + (tok * hk + head) * dk;
      for (int64_t j = 0; j < dk; ++j) {
        St(qo, j, Ld(qin, j) * qinv);
        St(ko, j, Ld(kin, j) * kinv);
      }
    } else if (slot < hk + v_chunks) {
      const int64_t c = slot - hk;
      const Tconv* vin = conv + row + 2 * key_dim;
      Tqkv* vo = v_out + tok * value_dim;
      const int64_t j0 = c * v_chunk;
      const int64_t j1 = (j0 + v_chunk) < value_dim ? (j0 + v_chunk) : value_dim;
      for (int64_t j = j0; j < j1; ++j) St(vo, j, Ld(vin, j));
    } else {
      const int64_t h = slot - hk - v_chunks;
      const int64_t idx = tok * hv + h;
      const float av = Ld(araw, tok * a_row_stride + h);
      const float bv = Ld(braw, tok * b_row_stride + h);
      const float x = av + dt_bias[h];
      const float sp = x > 20.0f ? x : log1pf(expf(x));
      g_out[idx] = -expf(a_log[h]) * sp;
      beta_out[idx] = 1.0f / (1.0f + expf(-bv));
    }
  }
}

// One block per token. GdnPostConvChunkedK gives every q/k head and v chunk to
// ONE thread, so at decode (t=1) 112 threads each walk a 128-element row with
// lanes 256 B apart — nothing coalesces and the launch was 26 us on gfx1100.
// Here a warp loads a q/k head coalesced into LDS, one lane per vector runs
// the SAME serial j-order sum of squares, and the warp stores; the v copy is
// block-strided; gate heads are one thread each with the same formula. Every
// output element is computed by the same f32 operations in the same order, so
// all five outputs are byte-identical to GdnPostConvChunkedK.
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ __launch_bounds__(kBlock) void GdnPostConvRowK(
    Tqkv* __restrict__ q_out, Tqkv* __restrict__ k_out, Tqkv* __restrict__ v_out,
    float* __restrict__ g_out, float* __restrict__ beta_out, const Tconv* __restrict__ conv,
    const Tgate* __restrict__ araw, const Tgate* __restrict__ braw,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int64_t hk, int64_t dk,
    int64_t hv, int64_t dv, int64_t a_row_stride, int64_t b_row_stride, float eps) {
  constexpr int kWarps = kBlock / 32;
  extern __shared__ float sh[];  // [kWarps][2][dk] staged q/k, then [kWarps][2] inverses
  const int64_t tok = blockIdx.x;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t row = tok * (2 * key_dim + value_dim);
  float* qs = sh + static_cast<int64_t>(warp) * 2 * dk;
  float* ks = qs + dk;
  float* inv = sh + static_cast<int64_t>(kWarps) * 2 * dk + 2 * warp;

  for (int64_t h0 = 0; h0 < hk; h0 += kWarps) {
    const int64_t head = h0 + warp;
    const bool active = head < hk;
    const Tconv* qin = conv + row + head * dk;
    const Tconv* kin = conv + row + key_dim + head * dk;
    if (active)
      for (int64_t j = lane; j < dk; j += 32) {
        qs[j] = Ld(qin, j);
        ks[j] = Ld(kin, j);
      }
    __syncthreads();
    if (active && lane < 2) {
      const float* x = lane == 0 ? qs : ks;
      float ss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) ss += x[j] * x[j];
      inv[lane] = 1.0f / sqrtf(ss + eps);
    }
    __syncthreads();
    if (active) {
      const float qinv = inv[0], kinv = inv[1];
      Tqkv* qo = q_out + (tok * hk + head) * dk;
      Tqkv* ko = k_out + (tok * hk + head) * dk;
      for (int64_t j = lane; j < dk; j += 32) {
        St(qo, j, qs[j] * qinv);
        St(ko, j, ks[j] * kinv);
      }
    }
    __syncthreads();
  }

  const Tconv* vin = conv + row + 2 * key_dim;
  Tqkv* vo = v_out + tok * value_dim;
#pragma unroll 8
  for (int64_t j = threadIdx.x; j < value_dim; j += kBlock) St(vo, j, Ld(vin, j));

  for (int64_t h = threadIdx.x; h < hv; h += kBlock) {
    const int64_t idx = tok * hv + h;
    const float av = Ld(araw, tok * a_row_stride + h);
    const float bv = Ld(braw, tok * b_row_stride + h);
    const float x = av + dt_bias[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    g_out[idx] = -expf(a_log[h]) * sp;
    beta_out[idx] = 1.0f / (1.0f + expf(-bv));
  }
}

// Read per call so a test can A/B it against the chunked kernel in one
// process. Default ON; VT_ROCM_GDN_POSTCONV_ROW=0 selects the chunked kernel.
inline bool GdnPostConvRowEnabled() {
  const char* e = std::getenv("VT_ROCM_GDN_POSTCONV_ROW");
  return e == nullptr || e[0] != '0';
}

// Same-binary A/B. Default ON; VT_ROCM_GDN_POSTCONV_CHUNK=0 restores the
// original single-thread-per-v-row kernel. Output must be byte-identical.
inline bool GdnPostConvChunkedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_GDN_POSTCONV_CHUNK");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

}  // namespace

void GdnPostConvKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                           Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                           const Tensor& araw, const Tensor& braw, const Tensor& a_log,
                           const Tensor& dt_bias, const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0) return;
  VT_CHECK(q_out.dtype == k_out.dtype && q_out.dtype == v_out.dtype,
           "rocm gdn_post_conv: q/k/v out dtypes must match");
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "rocm gdn_post_conv: q/k/v out must be f32 or bf16");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "rocm gdn_post_conv: conv must be f32 or bf16");
  const int64_t items = t * (hk + 1);
  // Chunked decomposition: split the value_dim copy into dk-sized pieces so it
  // is not one thread's serial loop. dk is the natural chunk -- it is what the
  // q/k slots already do, so every slot carries comparable work.
  const int64_t value_dim = hv * dv;
  const int64_t v_chunk = dk > 0 ? dk : value_dim;
  const int64_t v_chunks = v_chunk > 0 ? (value_dim + v_chunk - 1) / v_chunk : 1;
  const int64_t items_chunked = t * (hk + v_chunks + hv);
  hipStream_t s = AsStream(q);
  auto launch = [&](auto qkv_tag, auto conv_tag, auto gate_tag) {
    using Tqkv = decltype(qkv_tag);
    using Tconv = decltype(conv_tag);
    using Tgate = decltype(gate_tag);
    // The row arm stages kWarps*2*dk floats plus the inv pair; gate on the
    // computed size, not a dk constant — dk=1024 overshoots the 64 KiB
    // per-workgroup LDS limit by 64 B (16384+16 floats) and must fall through
    // to the chunked kernel instead of throwing at launch.
    const size_t shmem =
        (static_cast<size_t>(kBlock / 32) * 2 * dk + 2 * (kBlock / 32)) *
        sizeof(float);
    if (shmem <= 64 * 1024 && GdnPostConvRowEnabled()) {
      GdnPostConvRowK<Tqkv, Tconv, Tgate><<<static_cast<unsigned>(t), kBlock, shmem, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
          a_log.Ptr<float>(), dt_bias.Ptr<float>(), hk, dk, hv, dv, araw.stride[0],
          braw.stride[0], args.eps);
      return;
    }
    if (GdnPostConvChunkedEnabled()) {
      GdnPostConvChunkedK<Tqkv, Tconv, Tgate><<<GridFor(items_chunked), kBlock, 0, s>>>(
          q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
          beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
          a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hk, dk, hv, dv, araw.stride[0],
          braw.stride[0], args.eps, v_chunk, v_chunks);
      return;
    }
    GdnPostConvK<Tqkv, Tconv, Tgate><<<GridFor(items), kBlock, 0, s>>>(
        q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
        beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
        a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hk, dk, hv, dv, araw.stride[0],
        braw.stride[0], args.eps);
  };
  auto launch_gate = [&](auto qkv_tag, auto conv_tag) {
    if (araw.dtype == DType::kBF16)
      launch(qkv_tag, conv_tag, __hip_bfloat16{});
    else
      launch(qkv_tag, conv_tag, float{});
  };
  auto launch_conv = [&](auto qkv_tag) {
    if (conv.dtype == DType::kBF16)
      launch_gate(qkv_tag, __hip_bfloat16{});
    else
      launch_gate(qkv_tag, float{});
  };
  if (q_out.dtype == DType::kBF16)
    launch_conv(__hip_bfloat16{});
  else
    launch_conv(float{});
  Check(hipGetLastError(), "gdn_post_conv launch");
}

}  // namespace vt::rocm
