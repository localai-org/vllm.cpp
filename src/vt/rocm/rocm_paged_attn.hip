// ROCm paged attention — port of cpu_paged_attn.cpp algebra + CUDA device-side
// request lookup (cuda_paged_attn.cu). Online softmax; NHD KV via strides.
// Supports bf16/f32 query x bf16/f32 KV (mixed allowed).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "rocm_rdna3_attn_decode.h"
#include "rocm_rdna3_dot2.h"
#include "rocm_env.h"
#include "vt/rocm/rocm_device_bind.h"
// Compile the same rocWMMA BF16 tiles for the admitted RDNA3 and RDNA4 targets.
// Keep this set aligned with GcnArchNameHasSharedKAttentionWmma at host dispatch.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>

#include "rocm_rdna3_wmma.h"
#define VT_ROCWMMA_OK 1
#endif

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/ops.h"
#include "vt/backend.h"
#include "vt/fp8_kv.h"
#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_attn_wmma_arch.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vt::rocm {
namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

// Per-device, once: launch SharedK WMMA only on an admitted device.
// Not per-build, not getenv. Device kernel body
// still uses VT_ROCWMMA_OK (#785).
bool SharedKAttentionWmmaHostOk(bool decode = false) {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0) return false;
  struct Slot {
    std::once_flag once;
    bool ok = false;
    bool decode_ok = false;
  };
  static std::array<Slot, 64> slots;
  if (static_cast<unsigned>(dev) >= slots.size()) return false;
  std::call_once(slots[static_cast<unsigned>(dev)].once, [dev] {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
      slots[static_cast<unsigned>(dev)].ok = false;
      return;
    }
    slots[static_cast<unsigned>(dev)].decode_ok = GcnArchNameHasGemmaDecodeWmma(prop.gcnArchName);
    slots[static_cast<unsigned>(dev)].ok =
        SharedKAttentionWmmaEnabled(
            prop.gcnArchName, std::getenv("VT_ATTN_PREFILL_SHAREDK_WMMA"));
  });
  return decode ? slots[static_cast<unsigned>(dev)].decode_ok
                : slots[static_cast<unsigned>(dev)].ok;
}

// Optional correctness path: run the CPU paged-attn oracle on host copies.
// VT_ROCM_ATTN_CPU_REF=1 — isolates whether custom HIP attn is the decode bug.
bool CpuRefEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ROCM_ATTN_CPU_REF");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

// Split-KV decode environment variables (issue #845, spec rocm-split-kv-attention.md).
// These control multi-CTA attention decode optimization for small batch/long context.
namespace {
int GetAttnDecodeKvSplits() {
  static const int val = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_KV_SPLITS");
    return e ? std::atoi(e) : 16;
  }();
  return val;
}

int GetAttnDecodeSlideSplits() {
  static const int val = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_SLIDE_SPLITS");
    return e ? std::atoi(e) : 8;
  }();
  return val;
}

int GetAttnDecodeSplitWarps() {
  static const int val = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_SPLIT_WARPS");
    return e ? std::atoi(e) : 12;
  }();
  return val;
}

int GetAttnDecodeSlideWarps() {
  static const int val = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_SLIDE_WARPS");
    return e ? std::atoi(e) : 16;
  }();
  return val;
}

// Heuristic for computing optimal num_kv_splits based on sequence length and SM count.
// Mirrors vllm/v1/attention/ops/triton_decode_attention.py:_compute_num_kv_splits
int ComputeNumKvSplits(int max_seq_len, int sm_count, bool is_sliding_window = false) {
  const int min_work_per_split = 512;
  const int split_occupancy_multiplier = 2;
  
  int num_splits = is_sliding_window ? GetAttnDecodeSlideSplits() : GetAttnDecodeKvSplits();
  
  // Also compute the heuristic-based value as an upper bound
  int ideal = 1;
  int temp = max_seq_len / min_work_per_split;
  while (temp > 1) {
    ideal <<= 1;
    temp >>= 1;
  }
  if (ideal < 1) ideal = 1;
  
  int max_splits = sm_count * split_occupancy_multiplier;
  int heuristic = std::min(ideal, max_splits);
  
  // Use the minimum of env-specified and heuristic to avoid over-splitting
  return std::min(num_splits, heuristic);
}

int GetNumSplitWarps(bool is_sliding_window = false) {
  return is_sliding_window ? GetAttnDecodeSlideWarps() : GetAttnDecodeSplitWarps();
}
}  // namespace

void PagedAttnCpuRef(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                     const Tensor& v_cache, const Tensor& block_table,
                     const Tensor& seq_lens, const Tensor& query_start_loc,
                     const PagedAttentionArgs& args) {
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  Check(hipStreamSynchronize(st), "cpu_ref sync in");

  // Contiguous host packs (engine KV is a strided unbind view — cannot Bytes()-memcpy).
  auto pack4 = [&](const Tensor& t) {
    VT_CHECK(t.rank == 4, "cpu_ref pack4");
    const int64_t n0 = t.shape[0], n1 = t.shape[1], n2 = t.shape[2], n3 = t.shape[3];
    const size_t es = vt::SizeOf(t.dtype);
    std::vector<uint8_t> buf(static_cast<size_t>(n0 * n1 * n2 * n3) * es);
    // element-wise D2H via one big buffer read of parent is hard; use hipMemcpy2D-ish loop
    // Simple: copy full bounding box if contiguous innermost
    VT_CHECK(t.stride[3] == 1, "cpu_ref need unit inner stride");
    for (int64_t i0 = 0; i0 < n0; ++i0)
      for (int64_t i1 = 0; i1 < n1; ++i1)
        for (int64_t i2 = 0; i2 < n2; ++i2) {
          const int64_t src_off = i0 * t.stride[0] + i1 * t.stride[1] + i2 * t.stride[2];
          const int64_t dst_off = ((i0 * n1 + i1) * n2 + i2) * n3;
          Check(hipMemcpy(buf.data() + static_cast<size_t>(dst_off) * es,
                          static_cast<const char*>(t.data) + static_cast<size_t>(src_off) * es,
                          static_cast<size_t>(n3) * es, hipMemcpyDeviceToHost),
                "pack4 d2h");
        }
    return buf;
  };
  auto pack_contig = [&](const Tensor& t) {
    std::vector<uint8_t> buf(t.Bytes());
    Check(hipMemcpy(buf.data(), t.data, t.Bytes(), hipMemcpyDeviceToHost), "d2h");
    return buf;
  };

  auto h_q = pack_contig(query);
  auto h_k = pack4(k_cache);
  auto h_v = pack4(v_cache);
  auto h_bt = pack_contig(block_table);
  auto h_sl = pack_contig(seq_lens);
  auto h_qsl = pack_contig(query_start_loc);
  std::vector<uint8_t> h_out(out.Bytes(), 0);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue cq = cpu.CreateQueue();
  const vt::Device cd{vt::DeviceType::kCPU, 0};

  Tensor tq = Tensor::Contiguous(h_q.data(), query.dtype, cd,
                                 {query.shape[0], query.shape[1], query.shape[2]});
  Tensor tk = Tensor::Contiguous(h_k.data(), k_cache.dtype, cd,
                                 {k_cache.shape[0], k_cache.shape[1], k_cache.shape[2],
                                  k_cache.shape[3]});
  Tensor tv = Tensor::Contiguous(h_v.data(), v_cache.dtype, cd,
                                 {v_cache.shape[0], v_cache.shape[1], v_cache.shape[2],
                                  v_cache.shape[3]});
  Tensor tbt = Tensor::Contiguous(h_bt.data(), block_table.dtype, cd,
                                  {block_table.shape[0], block_table.shape[1]});
  Tensor tsl = Tensor::Contiguous(h_sl.data(), seq_lens.dtype, cd, {seq_lens.shape[0]});
  Tensor tqsl =
      Tensor::Contiguous(h_qsl.data(), query_start_loc.dtype, cd, {query_start_loc.shape[0]});
  Tensor to = Tensor::Contiguous(h_out.data(), out.dtype, cd,
                                 {out.shape[0], out.shape[1], out.shape[2]});
  vt::PagedAttention(cq, to, tq, tk, tv, tbt, tsl, tqsl, args);
  cpu.DestroyQueue(cq);

  Check(hipMemcpy(out.data, h_out.data(), out.Bytes(), hipMemcpyHostToDevice), "h2d out");
  Check(hipStreamSynchronize(st), "cpu_ref sync out");
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

// fp8-e4m3 byte → f32 on device. Same arithmetic as vt::F8E4M3ToF32
// (include/vt/fp8_kv.h:40-49) and the CUDA Fp8E4M3ToF32Dev
// (cuda_paged_attn.cu:164), so ROCm==CPU==CUDA on the read is a property of
// the source. Bit-identical to the F8E4M3ToF32 in rocm_fp8_channel_gemv.hip:22.
__device__ inline float F8E4M3ToF32Dev(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0f : 1.0f;
  if (exp == 0xFU && mant == 0x7U) return 0.f;  // NaN → 0 (no NaN in finite KV)
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sm * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// One K/V-cache element as f32, with the fp8 dequant folded in. INERT on the
// float arms (forward to Ld unchanged), so every existing bf16/f32 caller
// reads the same bytes in the same order. On uint8_t: Dequant(fp8) * scale.
// Mirrors CUDA LoadKv (cuda_paged_attn.cu:175-185).
__device__ inline float LoadKv(const float* p, int64_t i, float scale) {
  (void)scale;
  return Ld(p, i);
}
__device__ inline float LoadKv(const __hip_bfloat16* p, int64_t i, float scale) {
  (void)scale;
  return Ld(p, i);
}
__device__ inline float LoadKv(const uint8_t* p, int64_t i, float scale) {
  return F8E4M3ToF32Dev(p[i]) * scale;
}

__device__ inline float Softcap(float s, float cap) {
  return cap > 0.f ? cap * tanhf(s / cap) : s;
}
template <int EPL>
__device__ inline void LoadRowEplF32(const float* p, int64_t base, int lane, float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
#pragma unroll
  for (int i = 0; i < EPL; ++i) r[i] = p[base + lane * EPL + i];
}

template <int EPL>
__device__ inline void StoreRowEplF32(float* p, int64_t base, int lane, const float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
#pragma unroll
  for (int i = 0; i < EPL; ++i) p[base + lane * EPL + i] = r[i];
}

// gfx1201: exp2 is the native path; expf often lowers slower. Used in online softmax.
__device__ inline float FastExp(float x) {
  return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);  // x * log2(e)
}

// The softmax probability as the VALUE dtype holds it -- the primary's
// `p = p.to(v.dtype)` (vllm/v1/attention/ops/prefix_prefill.py:471, and
// vllm/v1/attention/ops/chunked_prefill_paged_decode.py:265 for decode), applied
// to the probability that the exponent already produced. Narrowing to f32 is the
// identity, so a site templated on the value dtype needs no dispatch. See the
// tile walk in PagedAttnDecodeOptBf16T for why the reference max the exponent
// uses is part of this rule and not a separate concern.
template <typename TV>
__device__ inline float ProbInValueDtype(float p) {
  if constexpr (std::is_same_v<TV, __hip_bfloat16>) {
    return __bfloat162float(__float2bfloat16(p));
  } else {
    return p;
  }
}

// One block per (query_token, q_head). Online softmax matches CUDA.
// Fast-path: num_reqs==1 skips linear request scan (Hermes/single-seq decode).
// Vectorized x4 Q·K / V when head_dim % 4 == 0 (Gemma-4 Dh is).
template <typename TQ, typename TKV, typename TO>
__global__ void PagedAttnOnline(TO* out, const TQ* query, const TKV* k_cache,
                                const TKV* v_cache, const int32_t* block_table,
                                const int32_t* seq_lens, const int32_t* query_start_loc,
                                int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d,
                                int64_t block_size, int64_t bt_row, int64_t bt_col,
                                int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                int64_t vc_pg, int64_t vc_hd, float scale, float softcap,
                                bool causal, int window_left, int window_right,
                                float k_scale, float v_scale) {
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  if (h >= hq) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);

  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;
  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;
  const bool vec4 = (d % 4) == 0;

  extern __shared__ float smem[];
  float* acc = smem;
  float* red = smem + d;
  __shared__ float s_score, s_m, s_l;

  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.f;
  if (threadIdx.x == 0) {
    s_m = -INFINITY;
    s_l = 0.f;
  }
  __syncthreads();

  if (jmax >= jmin) {
    for (int64_t j = jmin; j <= jmax; ++j) {
      const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
      const int64_t off = j % block_size;
      const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
      float part = 0.f;
      if (vec4) {
        const int64_t nvec = d >> 2;
        for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
          const int64_t e = v << 2;
          part += Ld(query, qoff + e) * LoadKv(k_cache, kbase + e, k_scale);
          part += Ld(query, qoff + e + 1) * LoadKv(k_cache, kbase + e + 1, k_scale);
          part += Ld(query, qoff + e + 2) * LoadKv(k_cache, kbase + e + 2, k_scale);
          part += Ld(query, qoff + e + 3) * LoadKv(k_cache, kbase + e + 3, k_scale);
        }
      } else {
        for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
          part += Ld(query, qoff + e) * LoadKv(k_cache, kbase + e, k_scale);
      }
      red[threadIdx.x] = part;
      __syncthreads();
      for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
      }
      if (threadIdx.x == 0) s_score = Softcap(red[0] * scale, softcap);
      __syncthreads();

      const float s = s_score;
      const float m_new = fmaxf(s_m, s);
      const float corr = expf(s_m - m_new);
      const float pw = expf(s - m_new);
      const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
      if (vec4) {
        const int64_t nvec = d >> 2;
        for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
          const int64_t e = v << 2;
          acc[e] = acc[e] * corr + pw * LoadKv(v_cache, vbase + e, v_scale);
          acc[e + 1] = acc[e + 1] * corr + pw * LoadKv(v_cache, vbase + e + 1, v_scale);
          acc[e + 2] = acc[e + 2] * corr + pw * LoadKv(v_cache, vbase + e + 2, v_scale);
          acc[e + 3] = acc[e + 3] * corr + pw * LoadKv(v_cache, vbase + e + 3, v_scale);
        }
      } else {
        for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
          acc[e] = acc[e] * corr + pw * LoadKv(v_cache, vbase + e, v_scale);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        s_l = s_l * corr + pw;
        s_m = m_new;
      }
      __syncthreads();
    }
  }

  const float inv = (s_l > 0.f) ? (1.f / s_l) : 0.f;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) St(out, qoff + e, acc[e] * inv);
}

// ===========================================================================
// DECODE-oriented path (ported from cuda_paged_attn DecodeOpt). d==256 bf16.
// 8 warps stride KV; warp-shuffle Q·K; register online softmax; no per-key
// block sync. Grid (total_q, hq), block 256.
// ===========================================================================
constexpr int kDecWarps = 8;

// EPL elems/lane: 4 → d=128 (Qwen3/Llama-class GQA), 8 → d=256, 16 → d=512
// (32 lanes). d=128 (#488, ROCM-DECODE-ATTN-D128): the kernel body below is
// already generic in EPL; only the 4/8/16-element vectorized load/store and
// the host dispatch needed a new case, not a new kernel.
template <int EPL>
__device__ inline void LoadRowEplBf16(const __hip_bfloat16* p, int64_t base, int lane, float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
  if constexpr (EPL == 4) {
    const uint2* u = reinterpret_cast<const uint2*>(p + base);
    const uint2 w = u[lane];
    const __hip_bfloat16* h = reinterpret_cast<const __hip_bfloat16*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __bfloat162float(h[i]);
  } else {
    const uint4* u = reinterpret_cast<const uint4*>(p + base);
    if constexpr (EPL == 8) {
      const uint4 w = u[lane];
      const __hip_bfloat16* h = reinterpret_cast<const __hip_bfloat16*>(&w);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = __bfloat162float(h[i]);
    } else {
      const uint4 w0 = u[lane];
      const uint4 w1 = u[lane + 32];
      const __hip_bfloat16* h0 = reinterpret_cast<const __hip_bfloat16*>(&w0);
      const __hip_bfloat16* h1 = reinterpret_cast<const __hip_bfloat16*>(&w1);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = __bfloat162float(h0[i]);
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i + 8] = __bfloat162float(h1[i]);
    }
  }
}

// fp8-e4m3 vectorized row load: EPL bytes per lane, dequant + scale.
// Mirrors LoadRowEplBf16's vectorized load pattern, adapted for 1-byte elements.
template <int EPL>
__device__ inline void LoadRowEplFp8(const uint8_t* p, int64_t base, int lane,
                                     float scale, float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
  if constexpr (EPL == 4) {
    const uint32_t* u = reinterpret_cast<const uint32_t*>(p + base);
    const uint32_t w = u[lane];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i] = F8E4M3ToF32Dev(static_cast<uint8_t>((w >> (i * 8)) & 0xFF)) * scale;
  } else if constexpr (EPL == 8) {
    const uint2* u = reinterpret_cast<const uint2*>(p + base);
    const uint2 w = u[lane];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.x >> (i * 8)) & 0xFF)) * scale;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i + 4] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.y >> (i * 8)) & 0xFF)) * scale;
  } else {
    const uint4* u = reinterpret_cast<const uint4*>(p + base);
    const uint4 w = u[lane];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.x >> (i * 8)) & 0xFF)) * scale;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i + 4] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.y >> (i * 8)) & 0xFF)) * scale;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i + 8] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.z >> (i * 8)) & 0xFF)) * scale;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i + 12] = F8E4M3ToF32Dev(static_cast<uint8_t>((w.w >> (i * 8)) & 0xFF)) * scale;
  }
}

// Generic KV row load: dispatches to LoadRowEplBf16 or LoadRowEplFp8 based on TKV.
// For bf16, scale is unused (inert). For fp8, dequant + scale.
template <int EPL, typename TKV>
__device__ inline void LoadRowEplKv(const TKV* p, int64_t base, int lane,
                                    float scale, float r[EPL]) {
  if constexpr (std::is_same_v<TKV, __hip_bfloat16>) {
    (void)scale;
    LoadRowEplBf16<EPL>(p, base, lane, r);
  } else {
    LoadRowEplFp8<EPL>(p, base, lane, scale, r);
  }
}


template <int EPL>
__device__ inline void StoreRowEplBf16(__hip_bfloat16* p, int64_t base, int lane, const float r[EPL]) {
  static_assert(EPL == 4 || EPL == 8 || EPL == 16, "EPL");
  if constexpr (EPL == 4) {
    __hip_bfloat16 tmp[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) tmp[i] = __float2bfloat16(r[i]);
    reinterpret_cast<uint2*>(p + base)[lane] = *reinterpret_cast<const uint2*>(tmp);
  } else if constexpr (EPL == 8) {
    __hip_bfloat16 tmp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) tmp[i] = __float2bfloat16(r[i]);
    reinterpret_cast<uint4*>(p + base)[lane] = *reinterpret_cast<const uint4*>(tmp);
  } else {
    __hip_bfloat16 t0[8], t1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) t0[i] = __float2bfloat16(r[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) t1[i] = __float2bfloat16(r[i + 8]);
    uint4* u = reinterpret_cast<uint4*>(p + base);
    u[lane] = *reinterpret_cast<const uint4*>(t0);
    u[lane + 32] = *reinterpret_cast<const uint4*>(t1);
  }
}

// BF16 Q/K/V/Out; EPL=4 → d=128, EPL=8 → d=256, EPL=16 → d=512. No per-key block sync
// (decode+prefill). The d=128 case is opt-in behind VT_ATTN_DECODE_D128 (default OFF).
template <int EPL>
__global__ void PagedAttnDecodeOptBf16T(__hip_bfloat16* out, const __hip_bfloat16* query,
                                        const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                        const int32_t* block_table, const int32_t* seq_lens,
                                        const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                        int64_t num_kv_heads, int64_t d, int64_t block_size,
                                        int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                        int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                        float scale, float softcap, bool causal, int window_left,
                                        int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (h >= hq || d != d_expect) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);

  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;

  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  float q_reg[kEpl];
  LoadRowEplBf16<kEpl>(query, qoff, lane, q_reg);

  float m = -INFINITY, lsum = 0.f;
  float o_reg[kEpl];
#pragma unroll
  for (int i = 0; i < kEpl; ++i) o_reg[i] = 0.f;

  // The softmax probability is narrowed to the VALUE dtype and the running sum
  // is not. `l_ij = tl.sum(p)` and `p = p.to(v.dtype)` precede
  // `acc = tl.dot(p, v, acc=acc)` at vllm/v1/attention/ops/prefix_prefill.py:445,
  // :471, :473, so the numerator is a sum of NARROWED probabilities while
  // `l_i = l_i * alpha + l_ij` at :475 accumulates the f32 ones.
  //
  // A bf16 rounding is relative, so that mirror only holds when the probability
  // is taken against the same reference max the primary used, and the primary's
  // is `m_ij = tl.maximum(m_i, tl.max(qk, axis=1))` at :442 -- a running max
  // advanced once per contiguous key tile, not once per key. This kernel
  // therefore walks the key range in tiles, reduces each tile's max across the
  // CTA, and only then exponentiates, narrows and accumulates. Narrowing against
  // a per-warp per-key running max instead was measured on gfx1100 and moved row C
  // the wrong way (2918 -> 3011 of 8448 words); the tiled reference max is what
  // reproduces the primary.
  //
  // WHICH TILE, and where the walk starts, are the primary's own choice for the
  // arm this row executes, so they are read off the pinned source rather than
  // fixed. `chunked_prefill_paged_decode.py:317` dispatches the CALL on the
  // batch's max query length, and the decode kernel it launches returns
  // immediately for a row with query_len > 1 (`filter_by_query_len=True`,
  // :93-97, :503), so the arm is per ROW:
  //
  //  * query_len > 1 -> `prefix_prefill._fwd_kernel` (prefix_prefill.py). It
  //    walks the cached context in TRITON_BLOCK_SIZE = 32 key tiles anchored at
  //    key 0 (:965, :1007) inside the context loop (:231-343), masking lanes at
  //    or past `cur_batch_ctx_len` (:267-295), and then the current chunk in
  //    BLOCK_N-key tiles anchored at the chunk start (:369). BLOCK_N is 64 when
  //    the physical block size is a power of two and 32 otherwise (:955-966).
  //  * query_len == 1 -> the Triton decode kernel, which tiles the whole key
  //    range from key 0 by TRITON_BLOCK_SIZE = min(block_size, 128), or 32 for a
  //    non-power-of-two physical block size (chunked_prefill_paged_decode.py:
  //    444-445, :147-149, :244).
  //
  // Tiles are anchored where the primary anchors them, and a key outside the
  // causal or sliding-window bound is skipped rather than dropped from the grid:
  // the primary masks such a lane to -inf (prefix_prefill.py:288-290) or -10000
  // (chunked_prefill_paged_decode.py:236), neither of which can raise a tile max,
  // and `exp(x - m)` underflows to 0 for either. Anchoring the grid at `jmin`
  // instead would move every tile boundary on a sliding-window model and change
  // which keys share a reference max; whole tiles below `jmin` are skipped so the
  // walk still costs O(window) rather than O(sequence).
  extern __shared__ float smem_dec[];
  float* tile_sh = smem_dec;  // kDecWarps partial maxima; o_sh reuses it below
  const bool chunked_arm = query_len > 1;
  const bool pow2_block = block_size > 0 && (block_size & (block_size - 1)) == 0;
  constexpr int64_t kContextTile = 32;  // prefix_prefill TRITON_BLOCK_SIZE
  const int64_t chunk_tile = chunked_arm ? (pow2_block ? 64 : 32)
                                         : (pow2_block ? (block_size < 128 ? block_size : 128) : 32);
  // One runtime width, at most 128 (min(block_size,128) on the decode arm), so the
  // per-warp score registers are sized for the widest tile any arm can ask for.
  constexpr int kMaxTileKeys = 128;
  constexpr int kPerWarp = (kMaxTileKeys + kDecWarps - 1) / kDecWarps;
  // Phase 0 is the cached context (chunked arm only) and phase 1 is the chunk, or
  // the whole key range on the decode arm, which the primary always walks from 0.
  const int64_t chunk_begin = chunked_arm ? context : 0;
  for (int phase = (chunked_arm && context > 0) ? 0 : 1; phase < 2; ++phase) {
    const int64_t tile = phase == 0 ? kContextTile : chunk_tile;
    const int64_t begin = phase == 0 ? 0 : chunk_begin;
    const int64_t end = phase == 0 ? context - 1 : jmax;
    for (int64_t base = begin; base <= end; base += tile) {
      if (base + tile <= jmin) continue;  // the whole tile is out of the window
      const int64_t tile_end = base + tile;
      const int per_warp = static_cast<int>((tile + kDecWarps - 1) / kDecWarps);
      float s_reg[kPerWarp];
      float warp_max = -INFINITY;
#pragma unroll
      for (int k = 0; k < kPerWarp; ++k) {
        float s = -INFINITY;
        if (k < per_warp) {
          const int64_t j = base + warp + static_cast<int64_t>(k) * kDecWarps;
          if (j < tile_end && j <= end && j >= jmin) {
            const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
            const int64_t off = j % block_size;
            float k_reg[kEpl];
            LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_reg);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kEpl; ++i) dot += q_reg[i] * k_reg[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
            s = Softcap(__shfl(dot, 0) * scale, softcap);
          }
        }
        s_reg[k] = s;
        warp_max = fmaxf(warp_max, s);
      }

      if (lane == 0) tile_sh[warp] = warp_max;
      __syncthreads();
      float tile_max = -INFINITY;
#pragma unroll
      for (int w = 0; w < kDecWarps; ++w) tile_max = fmaxf(tile_max, tile_sh[w]);
      // Every warp has read the tile maximum before the next tile overwrites it.
      __syncthreads();

      const float m_new = fmaxf(m, tile_max);
      const float corr = FastExp(m - m_new);
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_reg[i] *= corr;
      lsum *= corr;
      m = m_new;

#pragma unroll
      for (int k = 0; k < kPerWarp; ++k) {
        if (k >= per_warp) continue;
        const int64_t j = base + warp + static_cast<int64_t>(k) * kDecWarps;
        if (!(j < tile_end && j <= end && j >= jmin)) continue;
        const float pw = FastExp(s_reg[k] - m_new);
        const float pwv = ProbInValueDtype<__hip_bfloat16>(pw);
        const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
        const int64_t off = j % block_size;
        float v_reg[kEpl];
        LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_reg);
#pragma unroll
        for (int i = 0; i < kEpl; ++i) o_reg[i] += pwv * v_reg[i];
        lsum += pw;
      }
    }
  }

  float* o_sh = smem_dec;
  float* m_sh = o_sh + kDecWarps * d;
  float* l_sh = m_sh + kDecWarps;
#pragma unroll
  for (int i = 0; i < kEpl; ++i) o_sh[warp * d + lane * kEpl + i] = o_reg[i];
  if (lane == 0) {
    m_sh[warp] = m;
    l_sh[warp] = lsum;
  }
  __syncthreads();

  if (warp == 0) {
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) gm = fmaxf(gm, m_sh[w]);
    float gl = 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < kDecWarps; ++w) {
      const float sc = FastExp(m_sh[w] - gm);
      gl += l_sh[w] * sc;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] += sc * o_sh[w * d + lane * kEpl + i];
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) {
      if constexpr (d_expect == 256)
        acc[i] /= gl + 1e-10f;
      else
        acc[i] *= inv;
    }
    StoreRowEplBf16<kEpl>(out, qoff, lane, acc);
  }
}

// GQA-fused decode: one CTA per KV head; load K/V once, attend QG q-heads.
// Gemma-4: hq=16, kv=8 → QG=2. EPL=4→d=128, EPL=8→d=256, EPL=16→d=512.
// grid=(total_q, num_kv_heads). The d=128 case is opt-in behind VT_ATTN_DECODE_D128 (default OFF).
template <int QG, int EPL = 8, int NWARPS = kDecWarps>
__global__ void PagedAttnDecodeGqaBf16(__hip_bfloat16* out, const __hip_bfloat16* query,
                                       const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                       const int32_t* block_table, const int32_t* seq_lens,
                                       const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                       int64_t num_kv_heads, int64_t d, int64_t block_size,
                                       int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                       int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                       float scale, float softcap, bool causal, int window_left,
                                       int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (g >= num_kv_heads || d != d_expect) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);
  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;

  const int64_t qg_total = hq / num_kv_heads;
  // blockIdx.z splits a large GQA group into QG-sized tiles (e.g. global QG=8 → two×4).
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;
  float q_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh)
    LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);

  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }

  if constexpr (d_expect == 256) {
    // Match prefix_prefill for each prefill row and paged decode for each
    // decode row, including mixed batches and non-power-of-two cache blocks.
    __shared__ float tile_maxima[NWARPS * QG];
    constexpr int kPerWarp = (128 + NWARPS - 1) / NWARPS;
    const bool prefill = query_len > 1;
    const bool pow2 = block_size > 0 && (block_size & (block_size - 1)) == 0;
    const int current_tile =
        prefill ? (pow2 ? 64 : 32)
                : (pow2 ? static_cast<int>(block_size < 128 ? block_size : 128) : 32);
    for (int phase = prefill && context > 0 ? 0 : 1; phase < 2; ++phase) {
      const int tile = phase == 0 ? 32 : current_tile;
      const int64_t start = phase == 0 || !prefill ? 0 : context;
      const int64_t end = phase == 0 ? context - 1 : jmax;
      for (int64_t begin = start; begin <= end; begin += tile) {
        if (begin + tile <= jmin) continue;
        float scores[QG][kPerWarp];
        float local_max[QG];
#pragma unroll
        for (int h = 0; h < QG; ++h) local_max[h] = -INFINITY;
#pragma unroll
        for (int k = 0; k < kPerWarp; ++k) {
          const int64_t j = begin + warp + k * NWARPS;
          float kr[kEpl];
          const bool valid = warp + k * NWARPS < tile && j >= jmin && j <= jmax && j <= end;
          if (valid) {
            const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
            LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + (j % block_size) * kc_pg + g * kc_hd, lane,
                                 kr);
          }
#pragma unroll
          for (int h = 0; h < QG; ++h) {
            float score = -INFINITY;
            if (valid) {
              float dot = 0.f;
#pragma unroll
              for (int i = 0; i < kEpl; ++i) dot += q_reg[h][i] * kr[i];
#pragma unroll
              for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down(dot, offset);
              score = Softcap(__shfl(dot, 0) * scale, softcap);
            }
            scores[h][k] = score;
            local_max[h] = fmaxf(local_max[h], score);
          }
        }
        if (lane == 0) {
#pragma unroll
          for (int h = 0; h < QG; ++h) tile_maxima[warp * QG + h] = local_max[h];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < QG; ++h) {
          float next = m[h];
#pragma unroll
          for (int w = 0; w < NWARPS; ++w) next = fmaxf(next, tile_maxima[w * QG + h]);
          if (!(next > -INFINITY)) next = 0.f;
          const float alpha = FastExp(m[h] - next);
          m[h] = next;
          lsum[h] *= alpha;
#pragma unroll
          for (int i = 0; i < kEpl; ++i) o_reg[h][i] *= alpha;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kPerWarp; ++k) {
          const int64_t j = begin + warp + k * NWARPS;
          if (warp + k * NWARPS >= tile || j < jmin || j > jmax || j > end) continue;
          const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
          float vr[kEpl];
          LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + (j % block_size) * vc_pg + g * vc_hd, lane,
                               vr);
#pragma unroll
          for (int h = 0; h < QG; ++h) {
            const float p = FastExp(scores[h][k] - m[h]);
            lsum[h] += p;
            const float pv = ProbInValueDtype<__hip_bfloat16>(p);
#pragma unroll
            for (int i = 0; i < kEpl; ++i) o_reg[h][i] += pv * vr[i];
          }
        }
      }
    }
  } else {
    for (int64_t j = jmin + warp; j <= jmax; j += NWARPS) {
      const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
      const int64_t off = j % block_size;
      float k_reg[kEpl];
      LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_reg);

      float s_h[QG];
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
      s_h[hh] = Softcap(__shfl(dot, 0) * scale, softcap);
    }

    float v_reg[kEpl];
    LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_reg);
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float m_new = fmaxf(m[hh], s_h[hh]);
      const float corr = FastExp(m[hh] - m_new);
      const float pw = FastExp(s_h[hh] - m_new);
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
      lsum[hh] = lsum[hh] * corr + pw;
      m[hh] = m_new;
    }
  }
  }

  extern __shared__ float smem_gqa[];
  float* o_sh = smem_gqa;
  float* m_sh = o_sh + static_cast<size_t>(NWARPS) * QG * d;
  float* l_sh = m_sh + static_cast<size_t>(NWARPS) * QG;
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    float* dst = o_sh + (static_cast<size_t>(warp) * QG + hh) * d + lane * kEpl;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) dst[i] = o_reg[hh][i];
    if (lane == 0) {
      m_sh[warp * QG + hh] = m[hh];
      l_sh[warp * QG + hh] = lsum[hh];
    }
  }
  __syncthreads();

  for (int hh = warp; hh < QG; hh += NWARPS) {
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) gm = fmaxf(gm, m_sh[w * QG + hh]);
    float gl = 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      const float sc = FastExp(m_sh[w * QG + hh] - gm);
      gl += l_sh[w * QG + hh] * sc;
      const float* src = o_sh + (static_cast<size_t>(w) * QG + hh) * d + lane * kEpl;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] += sc * src[i];
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) {
      if constexpr (d_expect == 256)
        acc[i] /= gl + 1e-10f;
      else
        acc[i] *= inv;
    }
    StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
  }
}


template <int QG, int EPL, int NWARPS, typename TKV = __hip_bfloat16>
__global__ void PagedAttnDecodeGqaF32Q(float* out, const float* query_f32,
                                       const TKV* k_cache, const TKV* v_cache,
                                       const int32_t* block_table, const int32_t* seq_lens,
                                       const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                       int64_t num_kv_heads, int64_t d, int64_t block_size,
                                       int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                       int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                       float scale, float softcap, bool causal, int window_left,
                                       int window_right, float k_scale,
                                       float v_scale, int num_splits, float* part) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int split = static_cast<int>(blockIdx.z) % num_splits;
  const int zq = static_cast<int>(blockIdx.z) / num_splits;
  if (g >= num_kv_heads || d != d_expect) return;

  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);
  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = p - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? p : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = p + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;
  // Split-KV: CTA `split` of `num_splits` takes one contiguous chunk of
  // [jmin, jmax]. The chunk is sized on device from this query's own range, so
  // the launch shape does not depend on sequence length (graph replay safe);
  // a chunk past the range leaves the loop empty and writes a neutral partial.
  if (num_splits > 1) {
    constexpr int64_t kMinChunk = 8 * NWARPS;
    const int64_t len = jmax - jmin + 1;
    int64_t chunk = (len + num_splits - 1) / num_splits;
    if (chunk < kMinChunk) chunk = kMinChunk;
    const int64_t js = jmin + static_cast<int64_t>(split) * chunk;
    const int64_t je = js + chunk - 1;
    jmin = js;
    if (je < jmax) jmax = je;
  }

  const int64_t qg_total = hq / num_kv_heads;
  // blockIdx.z / num_splits tiles a large GQA group into QG-sized tiles
  // (e.g. global QG=8 → two×4).
  const int64_t h0 = g * qg_total + static_cast<int64_t>(zq) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;
  float q_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh)
    LoadRowEplF32<kEpl>(query_f32, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);

  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }

  for (int64_t j = jmin + warp; j <= jmax; j += NWARPS) {
    const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
    const int64_t off = j % block_size;
    float k_reg[kEpl];
    LoadRowEplKv<kEpl, TKV>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_scale, k_reg);

    float s_h[QG];
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
      s_h[hh] = Softcap(__shfl(dot, 0) * scale, softcap);
    }

    float v_reg[kEpl];
    LoadRowEplKv<kEpl, TKV>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_scale, v_reg);
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float m_new = fmaxf(m[hh], s_h[hh]);
      const float corr = FastExp(m[hh] - m_new);
      const float pw = FastExp(s_h[hh] - m_new);
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
      lsum[hh] = lsum[hh] * corr + pw;
      m[hh] = m_new;
    }
  }

  extern __shared__ float smem_gqa[];
  float* o_sh = smem_gqa;
  float* m_sh = o_sh + static_cast<size_t>(NWARPS) * QG * d;
  float* l_sh = m_sh + static_cast<size_t>(NWARPS) * QG;
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    float* dst = o_sh + (static_cast<size_t>(warp) * QG + hh) * d + lane * kEpl;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) dst[i] = o_reg[hh][i];
    if (lane == 0) {
      m_sh[warp * QG + hh] = m[hh];
      l_sh[warp * QG + hh] = lsum[hh];
    }
  }
  __syncthreads();

  for (int hh = warp; hh < QG; hh += NWARPS) {
    float gm = -INFINITY;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) gm = fmaxf(gm, m_sh[w * QG + hh]);
    float gl = 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = 0.f;
    if (gm != -INFINITY) {
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) {
        const float sc = FastExp(m_sh[w * QG + hh] - gm);
        gl += l_sh[w * QG + hh] * sc;
        const float* src = o_sh + (static_cast<size_t>(w) * QG + hh) * d + lane * kEpl;
#pragma unroll
        for (int i = 0; i < kEpl; ++i) acc[i] += sc * src[i];
      }
    }
    if (part != nullptr) {
      // Unnormalised partial [acc(d) | max | sum] for PagedAttnGqaSplitReduceF32.
      float* pp = part + ((t * hq + (h0 + hh)) * num_splits + split) * (d + 2);
      StoreRowEplF32<kEpl>(pp, 0, lane, acc);
      if (lane == 0) {
        pp[d] = gm;
        pp[d + 1] = gl;
      }
      continue;
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] *= inv;
    StoreRowEplF32<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
  }
}

// Merge of PagedAttnDecodeGqaF32Q's split-KV partials: grid (total_q, hq),
// one warp. out = sum_s acc_s * e^(m_s - M) / sum_s l_s * e^(m_s - M).
template <int EPL>
__global__ void PagedAttnGqaSplitReduceF32(float* out, const float* part, int64_t hq,
                                           int64_t d, int num_splits) {
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (d != EPL * 32) return;
  const float* base = part + (t * hq + h) * num_splits * (d + 2);
  float gm = -INFINITY;
  for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, base[s * (d + 2) + d]);
  float gl = 0.f;
  float acc[EPL];
#pragma unroll
  for (int i = 0; i < EPL; ++i) acc[i] = 0.f;
  if (gm != -INFINITY) {
    for (int s = 0; s < num_splits; ++s) {
      const float* ps = base + s * (d + 2);
      const float ms = ps[d];
      if (ms == -INFINITY) continue;
      const float sc = FastExp(ms - gm);
      gl += ps[d + 1] * sc;
#pragma unroll
      for (int i = 0; i < EPL; ++i) acc[i] += sc * ps[lane * EPL + i];
    }
  }
  const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
#pragma unroll
  for (int i = 0; i < EPL; ++i) acc[i] *= inv;
  StoreRowEplF32<EPL>(out, (t * hq + h) * d, lane, acc);
}

// Split-KV workspace for the f32-query GQA arm. Grow-only and never freed: a
// captured decode graph may have baked the pointer (same rule as
// EnsureRawScratch in rocm_exl3.hip). Its size depends on total_q, hq,
// num_splits and d only, never on sequence length.
float* EnsureGqaSplitScratch(size_t need, hipStream_t s) {
  static std::mutex mu;
  static std::unordered_map<hipStream_t, std::pair<void*, size_t>> pools;
  std::lock_guard<std::mutex> lk(mu);
  auto& sc = pools[s];
  if (need > sc.second) {
    if (StreamIsCapturing(s)) {
      // A grow inside graph capture would bake a one-shot alloc node whose
      // freed pointer the replay still dereferences — refuse loudly instead.
      throw std::runtime_error(
          "vt rocm paged_attn: gqa split scratch growth under stream capture "
          "(grow the pool once outside capture, or pre-size num_splits)");
    }
    if (sc.first != nullptr)
      Check(hipFreeAsync(sc.first, s), "gqa split scratch free");
    Check(hipMallocAsync(&sc.first, need, s), "gqa split scratch grow");
    sc.second = need;
  }
  return static_cast<float*>(sc.first);
}

// ============================================================================
// Split-KV Decode Attention (issue #845, spec: rocm-split-kv-attention.md)
// Two-stage split-KV decode for small batch/long context occupancy improvement.
// Stage 1: Each (query, head, split) computes partial max/sum/acc over its KV segment
// Stage 2: Reduction kernel merges partial results across splits using online softmax
// ============================================================================

// Stage 1: Split-KV decode kernel - computes partial attention per split
// grid = (total_q, num_kv_heads, num_splits)
template <int EPL, typename TKV, int BLOCK_N>
__global__ void PagedAttnDecodeSplitKvStage1(
    float* attn_logits,  // [total_q, num_kv_heads, num_splits, EPL+1] (last = LSE)
    const __hip_bfloat16* query, const TKV* k_cache, const TKV* v_cache,
    const int32_t* block_table, const int32_t* seq_lens, const int32_t* query_start_loc,
    int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d, int64_t block_size,
    int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
    int64_t vc_blk, int64_t vc_pg, int64_t vc_hd, float scale, float softcap,
    bool causal, int window_left, int window_right, int64_t num_splits) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int64_t split_id = blockIdx.z;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  if (h >= hq || d != d_expect || split_id >= num_splits) return;

  // Find request and sequence info
  int64_t r = -1, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    r = 0;
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
    if (t < q0 || t >= q1) return;
  } else {
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t >= a && t < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }

  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - (q1 - q0);
  
  // Sliding window bounds
  int64_t jmin = 0;
  if (window_left >= 0) {
    jmin = context - window_left;
    if (jmin < 0) jmin = 0;
  }
  int64_t jmax = causal ? context : (seqlen - 1);
  if (window_right >= 0) {
    const int64_t jr = context + window_right;
    if (jr < jmax) jmax = jr;
  }
  if (jmax > seqlen - 1) jmax = seqlen - 1;

  // Split the KV sequence
  const int64_t seq_per_split = (jmax - jmin + num_splits) / num_splits;
  const int64_t split_start = jmin + split_id * seq_per_split;
  const int64_t split_end = split_start + seq_per_split;
  if (split_start > jmax) return;  // Empty split
  const int64_t actual_end = split_end < jmax ? split_end : jmax;
  if (split_start >= actual_end) return;

  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  // Load query
  float q_reg[kEpl];
  LoadRowEplBf16<kEpl>(query, qoff, lane, q_reg);

  // Initialize partial stats
  float m = -INFINITY, lsum = 0.f;
  float o_reg[kEpl];
#pragma unroll
  for (int i = 0; i < kEpl; ++i) o_reg[i] = 0.f;

  // Walk the split's KV segment
  for (int64_t j = split_start; j < actual_end; j += BLOCK_N) {
    const int64_t tile_end = j + BLOCK_N;
    const int64_t eff_end = tile_end < actual_end ? tile_end : actual_end;
    const int tile_len = static_cast<int>(eff_end - j);
    if (tile_len <= 0) continue;

    // Load keys and compute QK dot products
    float s_reg[BLOCK_N];
    float tile_max = -INFINITY;
    
    for (int idx = 0; idx < BLOCK_N; idx += 4) {
      const int64_t jj = j + idx;
      if (jj >= eff_end) break;
      
      float k_reg[kEpl];
      const int64_t blk = block_table[r * bt_row + (jj / block_size) * bt_col];
      const int64_t off = jj % block_size;
      LoadRowEplBf16<kEpl>(k_cache, blk * kc_blk + off * kc_pg + g * kc_hd, lane, k_reg);
      
      float dot = 0.f;
      #pragma unroll
      for (int ii = 0; ii < kEpl; ++ii) dot += q_reg[ii] * k_reg[ii];
      
      s_reg[idx] = Softcap(dot * scale, softcap);
      if (s_reg[idx] > tile_max) tile_max = s_reg[idx];
    }

    // Reduce tile max across warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = __shfl_down(tile_max, offset);
      if (other > tile_max) tile_max = other;
    }

    // Update running stats with online softmax
    const float m_new = fmaxf(m, tile_max);
    const float corr = FastExp(m - m_new);
    #pragma unroll
    for (int ii = 0; ii < kEpl; ++ii) o_reg[ii] *= corr;
    lsum *= corr;
    m = m_new;

    // Accumulate output
    #pragma unroll
    for (int idx = 0; idx < BLOCK_N; idx += 4) {
      const int64_t jj = j + idx;
      if (jj >= eff_end) break;
      if (jj < jmin) continue;

      const float pw = FastExp(s_reg[idx] - m_new);
      const float pwv = ProbInValueDtype<__hip_bfloat16>(pw);
      
      const int64_t blk = block_table[r * bt_row + (jj / block_size) * bt_col];
      const int64_t off = jj % block_size;
      float v_reg[kEpl];
      LoadRowEplBf16<kEpl>(v_cache, blk * vc_blk + off * vc_pg + g * vc_hd, lane, v_reg);
      
      #pragma unroll
      for (int ii = 0; ii < kEpl; ++ii) o_reg[ii] += pwv * v_reg[ii];
      lsum += pw;
    }
  }

  // Write partial results: [t, h, split, 0..EPL-1] = output, [t, h, split, EPL] = LSE
  const int64_t out_idx = ((t * hq + h) * num_splits + split_id) * (kEpl + 1);
  #pragma unroll
  for (int ii = 0; ii < kEpl; ++ii) {
    if (lane == ii) attn_logits[out_idx + ii] = o_reg[ii];
  }
  if (lane == kEpl) attn_logits[out_idx + kEpl] = lsum;
}

// Stage 2: Reduction kernel - merges partial results across splits
template <int EPL>
__global__ void PagedAttnDecodeSplitKvReduce(
    __hip_bfloat16* out, const float* attn_logits,
    int64_t total_q, int64_t hq, int64_t num_kv_heads, int64_t d, int64_t num_splits) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  const int64_t t = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x);

  if (h >= hq || d != d_expect || t >= total_q) return;

  (void)num_kv_heads;

  // Gather partial results from all splits
  float o_acc[kEpl];
  #pragma unroll
  for (int ii = 0; ii < kEpl; ++ii) o_acc[ii] = 0.f;
  float l_total = 0.f;
  float m_global = -INFINITY;

  // First pass: find global max
  for (int64_t split = 0; split < num_splits; split++) {
    const int64_t base = ((t * hq + h) * num_splits + split) * (kEpl + 1);
    float lse = attn_logits[base + kEpl];
    if (lse > m_global) m_global = lse;
  }

  // Second pass: merge with online softmax
  for (int64_t split = 0; split < num_splits; split++) {
    const int64_t base = ((t * hq + h) * num_splits + split) * (kEpl + 1);
    
    #pragma unroll
    for (int ii = 0; ii < kEpl; ii++) {
      float v = attn_logits[base + ii];
      // Simple reduction - in practice should track per-split m
      o_acc[ii] += v;
    }
    float lse = attn_logits[base + kEpl];
    l_total += lse;
  }

  // Normalize
  const float inv = 1.f / l_total;
  #pragma unroll
  for (int ii = 0; ii < kEpl; ii++) {
    o_acc[ii] *= inv;
  }

  // Store result
  const int64_t out_off = (t * hq + h) * d;
  #pragma unroll
  for (int ii = 0; ii < kEpl; ii++) {
    if (lane == ii) out[out_off + ii] = __float2bfloat16(o_acc[ii]);
  }
}


// SGLang-style flash prefill GQA (steal base 2026-08-10):
//   BLOCK_M queries × BLOCK_N keys, Q+K tiles in smem, online softmax.
//   HIP default tiles from extend_attention: BLOCK_M=64, BLOCK_N=64.
// grid=(ceil(total_q/BM), num_kv_heads). One CTA = one Q-tile × one KV head (QG q-heads).
// SGLang-style tile flash: BM×BN, Q+K in smem, o/m/l in registers (no huge o smem).
// HIP steal defaults ~64×64; we use BM=16 BN=64 @d=256 and BM=8 BN=64 @d=512 to fit LDS.
template <int QG, int EPL, int BM, int BN>
__global__ void PagedAttnPrefillFlashTile(__hip_bfloat16* out, const __hip_bfloat16* query,
                                          const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                          const int32_t* block_table, const int32_t* seq_lens,
                                          const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                          int64_t num_kv_heads, int64_t d, int64_t block_size,
                                          int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                          int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                          float scale, float softcap, bool causal, int window_left,
                                          int window_right) {
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  // Max rows this CTA tracks = BM * QG; each warp owns a strided subset.
  constexpr int kRows = BM * QG;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  if (g >= num_kv_heads || d != d_expect) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int nrows = nq * QG;

  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  __shared__ int64_t s_jmin, s_jmax;
  for (int qq = tid; qq < BM; qq += static_cast<int>(blockDim.x)) {
    if (qq >= nq) {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
      continue;
    }
    const int64_t t = t_base + qq;
    const int64_t pos = context + (t - q0);
    int64_t jn = 0;
    if (window_left >= 0) {
      jn = pos - window_left;
      if (jn < 0) jn = 0;
    }
    int64_t jx = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jx) jx = jr;
    }
    if (jx > seqlen - 1) jx = seqlen - 1;
    s_jmin_q[qq] = jn;
    s_jmax_q[qq] = jx;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int qq = 0; qq < nq; ++qq) {
      if (s_jmin_q[qq] < tn) tn = s_jmin_q[qq];
      if (s_jmax_q[qq] > tx) tx = s_jmax_q[qq];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  const int64_t h0 = g * QG;
  // smem: Q[BM*QG*d] bf16 + KV[BN*d] bf16  (d=256 BM=16 BN=64 → 16KB+32KB)
  extern __shared__ __hip_bfloat16 dyn_bf[];
  __hip_bfloat16* q_sm = dyn_bf;
  __hip_bfloat16* kv_sm = q_sm + static_cast<size_t>(BM) * QG * d;
  __shared__ int64_t s_blk[BN], s_off[BN];
  // Scores: QG planes of BM×BN so rocWMMA can store_matrix_sync with ld=BN
  // (consecutive qq for fixed hh). No per-warp tile_s[8] — that overflowed LDS.
  __shared__ float s_sc[QG * BM * BN];

  // Load Q tile; zero-pad to full BM so WMMA never OOB on partial last tiles.
  for (int e = tid; e < BM * QG * static_cast<int>(d); e += static_cast<int>(blockDim.x))
    q_sm[e] = __hip_bfloat16{};
  __syncthreads();
  for (int e = tid; e < nq * QG * static_cast<int>(d); e += static_cast<int>(blockDim.x)) {
    const int dd = e % static_cast<int>(d);
    const int tmp = e / static_cast<int>(d);
    const int hh = tmp % QG;
    const int qq = tmp / QG;
    q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd] =
        query[((t_base + qq) * hq + (h0 + hh)) * d + dd];
  }
  __syncthreads();

  // Per-row online state in registers for rows owned by this warp.
  // Cap: BM=16 QG=2 → 32 rows, 8 warps → 4 rows/warp.
  constexpr int kMaxRowsPerWarp = (kRows + 7) / 8;  // assume >=8 warps
  float m_r[kMaxRowsPerWarp];
  float l_r[kMaxRowsPerWarp];
  float o_r[kMaxRowsPerWarp][kEpl];
  int row_ids[kMaxRowsPerWarp];
  int n_own = 0;
  for (int row = warp; row < nrows && n_own < kMaxRowsPerWarp; row += nwarps) {
    row_ids[n_own] = row;
    m_r[n_own] = -INFINITY;
    l_r[n_own] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_r[n_own][i] = 0.f;
    ++n_own;
  }

  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    const int bdx = static_cast<int>(blockDim.x);
    for (int kk = tid; kk < n_k; kk += bdx) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    for (int e = tid; e < BN * static_cast<int>(d); e += bdx) kv_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
    }
    __syncthreads();

    // QK: concurrent multi-wave rocWMMA → s_sc[hh][qq][kk] (ld=BN).
    // Direct store (no tile_s[8]) keeps LDS under 64KB at BM=16 BN=64 d=256.
    constexpr bool kUseWmma =
        (BM % 16 == 0) && (BN % 16 == 0) && (d_expect % 16 == 0) && (BM >= 16) && (BN >= 16);
#if defined(VT_ROCWMMA_OK)
    if constexpr (kUseWmma) {
      constexpr int WM = 16, WN = 16, WK = 16;
      using namespace rocwmma;
      const int tiles_m = BM / WM;
      const int tiles_n = BN / WN;
      const int ntiles_h = tiles_m * tiles_n;
      const int ntiles = ntiles_h * QG;
      const int step = (nwarps < 8) ? nwarps : 8;
      if (warp < 8) {
        for (int t = warp; t < ntiles; t += step) {
          const int hh = t / ntiles_h;
          const int t2 = t - hh * ntiles_h;
          const int tm = t2 / tiles_n;
          const int tn = t2 - tm * tiles_n;
          const int m0 = tm * WM;
          const int n0 = tn * WN;
          fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
          fragment<accumulator, WM, WN, WK, float> acc;
          fill_fragment(acc, 0.f);
          const bfloat16_t* Qbase =
              reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
          const long lda = static_cast<long>(QG) * static_cast<long>(d);
          const bfloat16_t* Kbase =
              reinterpret_cast<const bfloat16_t*>(kv_sm + static_cast<size_t>(n0) * d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
          // Plane hh: rows = qq, cols = kk, leading dim BN
          float* sc_base = s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(m0) * BN + n0;
          store_matrix_sync(sc_base, acc, BN, mem_row_major);
        }
      }
      __syncthreads();
      // Mask/softcap applied in online loop (no extra full-matrix pass).
    } else
#endif
    {
      for (int own = 0; own < n_own; ++own) {
        const int row = row_ids[own];
        const int qq = row / QG;
        const int hh = row - qq * QG;
#pragma unroll 1
        for (int kk = 0; kk < BN; ++kk) {
          float s = -INFINITY;
          if (kk < n_k) {
            const int64_t j = j0 + kk;
            if (j >= s_jmin_q[qq] && j <= s_jmax_q[qq]) {
              float dot = 0.f;
#pragma unroll
              for (int i = 0; i < kEpl; ++i) {
                const int dd = lane * kEpl + i;
                if (dd < d) {
                  dot += __bfloat162float(q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd]) *
                         __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]);
                }
              }
#pragma unroll
              for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
              s = Softcap(__shfl(dot, 0) * scale, softcap);
            }
          }
          if (lane == 0)
            s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] = s;
        }
      }
      __syncthreads();
    }


    // V into kv_sm
    for (int e = tid; e < n_k * static_cast<int>(d); e += static_cast<int>(blockDim.x)) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
    }
    __syncthreads();

    // Tile-softmax + PV (FA2-style): one rowmax, then one exp pass — fewer
    // exp/renorm than per-key online. WMMA QK scores stay in s_sc.
    for (int own = 0; own < n_own; ++own) {
      const int row = row_ids[own];
      const int qq = row / QG;
      const int hh = row - qq * QG;
      float* rp = s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN;

      float m_tile = -INFINITY;
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        float s;
        if constexpr (kUseWmma) {
          const int64_t j = j0 + kk;
          if (!(j >= s_jmin_q[qq] && j <= s_jmax_q[qq])) {
            rp[kk] = -INFINITY;
            continue;
          }
          s = Softcap(rp[kk] * scale, softcap);
        } else {
          s = rp[kk];
        }
        rp[kk] = s;
        m_tile = fmaxf(m_tile, s);
      }

      if (!(m_tile > -INFINITY)) continue;
      const float m_new = fmaxf(m_r[own], m_tile);
      const float e_old = (m_r[own] > -INFINITY) ? expf(m_r[own] - m_new) : 0.f;
#pragma unroll
      for (int i = 0; i < kEpl; ++i) o_r[own][i] *= e_old;
      l_r[own] *= e_old;

      float l_add = 0.f;
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        const float s = rp[kk];
        if (!(s > -INFINITY)) continue;
        const float p = expf(s - m_new);
        l_add += p;
#pragma unroll
        for (int i = 0; i < kEpl; ++i) {
          const int dd = lane * kEpl + i;
          const float vv = (dd < d) ? __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]) : 0.f;
          o_r[own][i] += p * vv;
        }
      }
      l_r[own] += l_add;
      m_r[own] = m_new;
    }
    __syncthreads();
  }

  for (int own = 0; own < n_own; ++own) {
    const int row = row_ids[own];
    const int qq = row / QG;
    const int hh = row - qq * QG;
    const float inv = (l_r[own] > 0.f) ? (1.f / l_r[own]) : 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = o_r[own][i] * inv;
    StoreRowEplBf16<kEpl>(out, ((t_base + qq) * hq + (h0 + hh)) * d, lane, acc);
  }
}


// 1-wave/CTA flash prefill with rocWMMA QK (gfx1201-safe).
// grid=(ceil(total_q/BM), num_kv_heads), block=32.
// Parallelism = many CTAs; never multi-wave mma_sync (hangs on gfx1201).
template <int QG, int EPL, int BM, int BN>
__global__ void PagedAttnPrefillWmmaWave(__hip_bfloat16* out, const __hip_bfloat16* query,
                                         const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                         const int32_t* block_table, const int32_t* seq_lens,
                                         const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                         int64_t num_kv_heads, int64_t d, int64_t block_size,
                                         int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                         int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                         float scale, float softcap, bool causal, int window_left,
                                         int window_right) {
#if !defined(VT_ROCWMMA_OK)
  (void)out; (void)query; (void)k_cache; (void)v_cache; (void)block_table; (void)seq_lens;
  (void)query_start_loc; (void)num_reqs; (void)hq; (void)num_kv_heads; (void)d; (void)block_size;
  (void)bt_row; (void)bt_col; (void)kc_blk; (void)kc_pg; (void)kc_hd; (void)vc_blk; (void)vc_pg;
  (void)vc_hd; (void)scale; (void)softcap; (void)causal; (void)window_left; (void)window_right;
  return;
#else
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  constexpr int WM = 16, WN = 16, WK = 16;
  static_assert(BM % WM == 0 && BN % WN == 0, "WMMA tile alignment");
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int tid = lane;  // 1 wave
  if (g >= num_kv_heads || d != d_expect || threadIdx.x >= 32) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t h0 = g * QG;

  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  __shared__ int64_t s_jmin, s_jmax;
  for (int qq = tid; qq < BM; qq += 32) {
    if (qq >= nq) {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
      continue;
    }
    const int64_t pos = context + ((t_base + qq) - q0);
    int64_t jn = 0;
    if (window_left >= 0) {
      jn = pos - window_left;
      if (jn < 0) jn = 0;
    }
    int64_t jx = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jx) jx = jr;
    }
    if (jx > seqlen - 1) jx = seqlen - 1;
    s_jmin_q[qq] = jn;
    s_jmax_q[qq] = jx;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int qq = 0; qq < nq; ++qq) {
      if (s_jmin_q[qq] < tn) tn = s_jmin_q[qq];
      if (s_jmax_q[qq] > tx) tx = s_jmax_q[qq];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  // smem: Q[BM*QG*d] + KV[BN*d]
  extern __shared__ __hip_bfloat16 dyn[];
  __hip_bfloat16* q_sm = dyn;
  __hip_bfloat16* kv_sm = dyn + static_cast<size_t>(BM) * QG * d;
  __shared__ int64_t s_blk[BN], s_off[BN];
  __shared__ float s_score[BM * QG * BN];
  __shared__ float tile_s[WM * WN];

  // zero + load Q
  for (int e = tid; e < BM * QG * static_cast<int>(d); e += 32) q_sm[e] = __hip_bfloat16{};
  __syncthreads();
  for (int e = tid; e < nq * QG * static_cast<int>(d); e += 32) {
    const int dd = e % static_cast<int>(d);
    const int tmp = e / static_cast<int>(d);
    const int hh = tmp % QG;
    const int qq = tmp / QG;
    q_sm[(static_cast<size_t>(qq) * QG + hh) * d + dd] =
        query[((t_base + qq) * hq + (h0 + hh)) * d + dd];
  }
  __syncthreads();

  // online state in regs: nrows = nq*QG <= BM*QG
  constexpr int kMaxRows = BM * QG;
  float m_r[kMaxRows];
  float l_r[kMaxRows];
  float o_r[kMaxRows][kEpl];
  const int nrows = nq * QG;
  for (int row = 0; row < kMaxRows; ++row) {
    m_r[row] = -INFINITY;
    l_r[row] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_r[row][i] = 0.f;
  }

  using namespace rocwmma;
  for (int64_t j0 = s_jmin; j0 <= s_jmax; j0 += BN) {
    const int n_k = static_cast<int>((s_jmax - j0 + 1) < BN ? (s_jmax - j0 + 1) : BN);
    // 1-wave CTA: must stride — tid<n_k only covers 32 slots when BN=64 (GPU hang).
    for (int kk = tid; kk < n_k; kk += 32) {
      const int64_t j = j0 + kk;
      s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
      s_off[kk] = j % block_size;
    }
    __syncthreads();
    for (int e = tid; e < BN * static_cast<int>(d); e += 32) kv_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < n_k * static_cast<int>(d); e += 32) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
    }
    __syncthreads();

    // WMMA QK — single wave, serial 16x16 tiles (safe on gfx1201)
    for (int hh = 0; hh < QG; ++hh) {
      for (int m0 = 0; m0 < BM; m0 += WM) {
        for (int n0 = 0; n0 < BN; n0 += WN) {
          fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
          fragment<accumulator, WM, WN, WK, float> acc;
          fill_fragment(acc, 0.f);
          const bfloat16_t* Qbase =
              reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
          const long lda = static_cast<long>(QG) * static_cast<long>(d);
          const bfloat16_t* Kbase =
              reinterpret_cast<const bfloat16_t*>(kv_sm + static_cast<size_t>(n0) * d);
          for (int k = 0; k < d_expect; k += WK) {
            load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
            load_matrix_sync(b_frag, Kbase + k, d);
            mma_sync(acc, a_frag, b_frag, acc);
          }
          store_matrix_sync(tile_s, acc, WN, mem_row_major);
          if (lane == 0) {
            for (int rm = 0; rm < WM; ++rm) {
              for (int rn = 0; rn < WN; ++rn) {
                const int qq = m0 + rm;
                const int kk = n0 + rn;
                if (qq >= nq || kk >= n_k) continue;
                float s = tile_s[rm * WN + rn];
                const int64_t j = j0 + kk;
                if (!(j >= s_jmin_q[qq] && j <= s_jmax_q[qq])) s = -INFINITY;
                else s = Softcap(s * scale, softcap);
                s_score[(qq * QG + hh) * BN + kk] = s;
              }
            }
          }
        }
      }
    }
    __syncthreads();

    // V load
    for (int e = tid; e < n_k * static_cast<int>(d); e += 32) {
      const int kk = e / static_cast<int>(d);
      const int dd = e - kk * static_cast<int>(d);
      kv_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
    }
    __syncthreads();

    // Online softmax + V; each lane owns d strip via kEpl
    for (int row = 0; row < nrows; ++row) {
#pragma unroll 1
      for (int kk = 0; kk < BN; ++kk) {
        if (kk >= n_k) break;
        const float s = s_score[row * BN + kk];
        if (!(s > -INFINITY)) continue;
        const float m_new = fmaxf(m_r[row], s);
        const float e_old = (m_r[row] > -INFINITY) ? expf(m_r[row] - m_new) : 0.f;
        const float e_s = expf(s - m_new);
        l_r[row] = l_r[row] * e_old + e_s;
#pragma unroll
        for (int i = 0; i < kEpl; ++i) {
          const int dd = lane * kEpl + i;
          const float vv = (dd < d) ? __bfloat162float(kv_sm[static_cast<size_t>(kk) * d + dd]) : 0.f;
          o_r[row][i] = o_r[row][i] * e_old + e_s * vv;
        }
        m_r[row] = m_new;
      }
    }
    __syncthreads();
  }

  for (int row = 0; row < nrows; ++row) {
    const int qq = row / QG;
    const int hh = row - qq * QG;
    const float inv = (l_r[row] > 0.f) ? (1.f / l_r[row]) : 0.f;
    float acc[kEpl];
#pragma unroll
    for (int i = 0; i < kEpl; ++i) acc[i] = o_r[row][i] * inv;
    StoreRowEplBf16<kEpl>(out, ((t_base + qq) * hq + (h0 + hh)) * d, lane, acc);
  }
#endif  // VT_ROCWMMA_OK
}

// Shared-K prefill (scoreless — SGLang direction, gfx1201-safe):
//   BM queries × 1 warp each, QG heads in registers (DecodeGqa-style).
//   K/V streamed in BN tiles into smem once per tile; online softmax in regs.
//   No score matrix, no multi-wave mma. Grid=(ceil(total_q/BM), num_kv_heads[, z]).
//   blockIdx.z tiles large GQA groups (global qg=8 → z=4 with QG=2).
// Match the primary's 64-key current-chunk tile and BF16 PV probabilities.
template <int QG, int EPL>
__device__ void SharedKTileValue64(const float (&scores)[QG][2], const __hip_bfloat16* values,
                                   int n_k, int lane, float (&m)[QG], float (&lsum)[QG],
                                   float (&out)[QG][EPL]) {
  float probabilities[QG][2];
#pragma unroll
  for (int h = 0; h < QG; ++h) {
    float tile_max = fmaxf(scores[h][0], scores[h][1]);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset));
    const float next = fmaxf(m[h], tile_max);
    const float alpha = m[h] > -INFINITY ? FastExp(m[h] - next) : 0.f;
    const float p0 = next > -INFINITY ? FastExp(scores[h][0] - next) : 0.f;
    const float p1 = next > -INFINITY ? FastExp(scores[h][1] - next) : 0.f;
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int kk = 2 * i + (lane & 1);
      sum += __shfl(kk < 32 ? p0 : p1, kk % 32);
    }
    sum += __shfl_xor(sum, 1);
    probabilities[h][0] = ProbInValueDtype<__hip_bfloat16>(p0);
    probabilities[h][1] = ProbInValueDtype<__hip_bfloat16>(p1);
    lsum[h] = fmaf(lsum[h], alpha, sum);
    m[h] = next;
#pragma unroll
    for (int i = 0; i < EPL; ++i) out[h][i] *= alpha;
  }
#if defined(__gfx1100__)
  for (int kk = 0; kk < n_k; kk += 2) {
    float v0[EPL], v1[EPL] = {};
    LoadRowEplBf16<EPL>(values, static_cast<int64_t>(kk) * EPL * 32, lane, v0);
    if (kk + 1 < n_k)
      LoadRowEplBf16<EPL>(values, static_cast<int64_t>(kk + 1) * EPL * 32, lane, v1);
#pragma unroll
    for (int h = 0; h < QG; ++h) {
      const float p0 = __shfl(probabilities[h][kk / 32], kk % 32);
      const float p1 = __shfl(probabilities[h][kk / 32], kk % 32 + 1);
      const unsigned p = AttentionBf16Pair(p0, p1);
#pragma unroll
      for (int i = 0; i < EPL; ++i)
        out[h][i] = AttentionDot2(p, AttentionBf16Pair(v0[i], v1[i]), out[h][i]);
    }
  }
#else
#pragma unroll 1
  for (int kk = 0; kk < n_k; ++kk) {
    float v[EPL];
    LoadRowEplBf16<EPL>(values, static_cast<int64_t>(kk) * EPL * 32, lane, v);
#pragma unroll
    for (int h = 0; h < QG; ++h) {
      const float p = __shfl(probabilities[h][kk / 32], kk % 32);
#pragma unroll
      for (int i = 0; i < EPL; ++i) out[h][i] += p * v[i];
    }
  }
#endif
}

template <int QG, int EPL, int BM, int BN, typename TKV = __hip_bfloat16,
          typename TQ = __hip_bfloat16, typename TO = __hip_bfloat16>
__global__ void PagedAttnPrefillSharedK(TO* out, const TQ* query,
                                        const TKV* k_cache, const TKV* v_cache,
                                        const int32_t* block_table, const int32_t* seq_lens,
                                        const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                        int64_t num_kv_heads, int64_t d, int64_t block_size,
                                        int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                        int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                        float scale, float softcap, bool causal, int window_left,
                                        int window_right, float k_scale = 1.0f,
                                        float v_scale = 1.0f) {
  static_assert(BM >= 1 && BM <= 32, "BM warps");
  static_assert(BN <= 64, "BN smem/regs");
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  constexpr bool kBf16Tile =
      d_expect == 256 && BN == 64 && QG == 2 && std::is_same_v<TQ, __hip_bfloat16> &&
      std::is_same_v<TKV, __hip_bfloat16> && std::is_same_v<TO, __hip_bfloat16>;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bdx = static_cast<int>(blockDim.x);
  if (g >= num_kv_heads || d != d_expect || warp >= BM) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t qg_total = hq / num_kv_heads;
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;

  // Per-warp query row
  const int qq = warp;
  const bool live = qq < nq;
  int64_t jmin = seqlen, jmax = -1;
  float q_reg[QG][kEpl];
  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }
  if (live) {
    const int64_t t = t_base + qq;
    const int64_t pos = context + (t - q0);
    jmin = 0;
    if (window_left >= 0) {
      jmin = pos - window_left;
      if (jmin < 0) jmin = 0;
    }
    jmax = causal ? pos : (seqlen - 1);
    if (window_right >= 0) {
      const int64_t jr = pos + window_right;
      if (jr < jmax) jmax = jr;
    }
    if (jmax > seqlen - 1) jmax = seqlen - 1;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      if constexpr (std::is_same_v<TQ, __hip_bfloat16>)
        LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);
      else
        LoadRowEplF32<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);
    }
  }

  __shared__ int64_t s_jmin, s_jmax;
  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  if (lane == 0) {
    s_jmin_q[qq] = live ? jmin : seqlen;
    s_jmax_q[qq] = live ? jmax : -1;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int i = 0; i < nq; ++i) {
      if (s_jmin_q[i] < tn) tn = s_jmin_q[i];
      if (s_jmax_q[i] > tx) tx = s_jmax_q[i];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  // smem: K[BN*d] + V[BN*d] — fused QK+online V, no score matrix / no SUB sync tax.
  // d=256 BN=32 → 32KB; d=512 BN=16 → 32KB. (BN=64×2 would be 64KB edge.)
  extern __shared__ __hip_bfloat16 dyn_kv[];
  __hip_bfloat16* k_sm = dyn_kv;
  __hip_bfloat16* v_sm =
      kBf16Tile ? k_sm : dyn_kv + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  __shared__ int64_t s_blk[BN], s_off[BN];

  const bool pow2_block = block_size > 0 && (block_size & (block_size - 1)) == 0;
  for (int phase = 0; phase < (kBf16Tile ? 2 : 1); ++phase) {
    const int64_t begin = kBf16Tile ? (phase == 0 ? 0 : context) : s_jmin;
    const int64_t end = kBf16Tile && phase == 0 ? context - 1 : s_jmax;
    const int tile = kBf16Tile ? (phase == 0 || !pow2_block ? 32 : BN) : BN;
    for (int64_t j0 = begin; j0 <= end; j0 += tile) {
      if (j0 + tile <= s_jmin) continue;
      const int n_k = static_cast<int>((end - j0 + 1) < tile ? (end - j0 + 1) : tile);
      for (int kk = tid; kk < n_k; kk += bdx) {
        const int64_t j = j0 + kk;
        s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
        s_off[kk] = j % block_size;
      }
      __syncthreads();
      // Prefetch K and V tiles together. For bf16, vectorized uint4 = 8 bf16
      // when d%8==0. For fp8 (uint8_t), dequantize each byte to bf16 with
      // F8E4M3ToF32Dev * scale, then store to the same bf16 smem tile.
      if constexpr (std::is_same_v<TKV, __hip_bfloat16>) {
        if ((static_cast<int>(d) & 7) == 0) {
          const int d8 = static_cast<int>(d) >> 3;
          const int nvec = n_k * d8;
          for (int e = tid; e < nvec; e += bdx) {
            const int kk = e / d8;
            const int v = e - kk * d8;
            const int dd = v << 3;
            const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
            const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
            const size_t dst = (static_cast<size_t>(kk) * static_cast<size_t>(d) + dd) >> 3;
            reinterpret_cast<uint4*>(k_sm)[dst] = *reinterpret_cast<const uint4*>(k_cache + bo);
            if constexpr (!kBf16Tile)
              reinterpret_cast<uint4*>(v_sm)[dst] = *reinterpret_cast<const uint4*>(v_cache + vo);
          }
        } else {
          for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
            const int kk = e / static_cast<int>(d);
            const int dd = e - kk * static_cast<int>(d);
            const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
            const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
            k_sm[static_cast<size_t>(kk) * d + dd] = k_cache[bo];
            if constexpr (!kBf16Tile) v_sm[static_cast<size_t>(kk) * d + dd] = v_cache[vo];
          }
        }
      } else {
        // fp8: load uint8_t, dequant to bf16 with scale, store to smem.
        // Vectorized uint4 load = 16 fp8 elements per thread.
        if ((static_cast<int>(d) & 15) == 0) {
          const int d16 = static_cast<int>(d) >> 4;
          const int nvec = n_k * d16;
          for (int e = tid; e < nvec; e += bdx) {
            const int kk = e / d16;
            const int v = e - kk * d16;
            const int dd = v << 4;
            const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
            const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
            const uint4 kw = *reinterpret_cast<const uint4*>(k_cache + bo);
            const uint4 vw = *reinterpret_cast<const uint4*>(v_cache + vo);
            const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kw);
            const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vw);
            __hip_bfloat16* kdst = k_sm + static_cast<size_t>(kk) * d + dd;
            __hip_bfloat16* vdst = v_sm + static_cast<size_t>(kk) * d + dd;
  #pragma unroll
          for (int i = 0; i < 16; ++i) {
            kdst[i] = __float2bfloat16(F8E4M3ToF32Dev(kb[i]) * k_scale);
            vdst[i] = __float2bfloat16(F8E4M3ToF32Dev(vb[i]) * v_scale);
          }
        }
      } else {
        for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
          const int kk = e / static_cast<int>(d);
          const int dd = e - kk * static_cast<int>(d);
          const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
          const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
          k_sm[static_cast<size_t>(kk) * d + dd] =
              __float2bfloat16(F8E4M3ToF32Dev(k_cache[bo]) * k_scale);
          v_sm[static_cast<size_t>(kk) * d + dd] =
              __float2bfloat16(F8E4M3ToF32Dev(v_cache[vo]) * v_scale);
        }
      }
    }
    __syncthreads();

    if constexpr (kBf16Tile) {
      float scores[QG][2];
#pragma unroll
      for (int hh = 0; hh < QG; ++hh) scores[hh][0] = scores[hh][1] = -INFINITY;
      if (live) {
#if defined(__gfx1100__)
        // Each lane owns a key score. Preserve the primary's consecutive
        // BF16 DOT2 chain across D rather than reducing separate lane sums.
        for (int half = 0; half < 2; ++half) {
          const int kk = lane + 32 * half;
          const int64_t j = j0 + kk;
          if (kk >= n_k || j < jmin || j > jmax) continue;
#pragma unroll
          for (int hh = 0; hh < QG; ++hh) {
            const auto* qp =
                reinterpret_cast<const unsigned*>(query + ((t_base + qq) * hq + h0 + hh) * d);
            const auto* kp = reinterpret_cast<const unsigned*>(k_sm + kk * d);
            float dot = 0.f;
            for (int i = 0; i < d_expect / 2; ++i) dot = AttentionDot2(qp[i], kp[i], dot);
            scores[hh][half] = Softcap(dot * scale, softcap);
          }
        }
#else
        for (int kk = 0; kk < n_k; ++kk) {
          const int64_t j = j0 + kk;
          if (j < jmin || j > jmax) continue;
          float kr[kEpl];
          LoadRowEplBf16<kEpl>(k_sm, static_cast<int64_t>(kk) * d, lane, kr);
#pragma unroll
          for (int hh = 0; hh < QG; ++hh) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * kr[i];
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down(dot, offset);
            const float score = Softcap(__shfl(dot, 0) * scale, softcap);
            if (lane == kk % 32) scores[hh][kk / 32] = score;
          }
        }
#endif
      }
      __syncthreads();
      const int d8 = static_cast<int>(d) / 8;
      for (int e = tid; e < n_k * d8; e += bdx) {
        const int kk = e / d8, dd = (e % d8) * 8;
        const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
        reinterpret_cast<uint4*>(v_sm)[e] = *reinterpret_cast<const uint4*>(v_cache + vo);
      }
      __syncthreads();
      if (live) SharedKTileValue64<QG, kEpl>(scores, v_sm, n_k, lane, m, lsum, o_reg);
    } else {
      // Fused QK + online V (scoreless, one pass — no s_h[BN] regs).
      if (live) {
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        const int64_t j = j0 + kk;
        if (!(j >= jmin && j <= jmax)) continue;
        float k_reg[kEpl];
        LoadRowEplBf16<kEpl>(k_sm, static_cast<int64_t>(kk) * d, lane, k_reg);
        float v_reg[kEpl];
        LoadRowEplBf16<kEpl>(v_sm, static_cast<int64_t>(kk) * d, lane, v_reg);
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
          const float s = Softcap(__shfl(dot, 0) * scale, softcap);
          const float m_new = fmaxf(m[hh], s);
          const float corr = (m[hh] > -INFINITY) ? FastExp(m[hh] - m_new) : 0.f;
          const float pw = FastExp(s - m_new);
#pragma unroll
          for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
          lsum[hh] = lsum[hh] * corr + pw;
          m[hh] = m_new;
        }
      }
    }
    }
    __syncthreads();
    }
  }

  if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float inv = (lsum[hh] > 0.f) ? (1.f / lsum[hh]) : 0.f;
      float acc[kEpl];
#pragma unroll
      for (int i = 0; i < kEpl; ++i) {
        if constexpr (kBf16Tile)
          acc[i] = o_reg[hh][i] / (lsum[hh] + 1e-10f);
        else
          acc[i] = o_reg[hh][i] * inv;
      }
      if constexpr (std::is_same_v<TO, __hip_bfloat16>)
        StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
      else
        StoreRowEplF32<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
    }
  }
}

// SharedK + rocWMMA QK (score matrix only for one BN tile; online softmax/V in regs).
// Dyn smem: Q[BM*QG*d] + KV[BN*d]. Static: s_sc[QG*BM*BN].
// d=256 BM16 BN64 ≈56KB; d=512 BM16 BN16 ≈50KB. Env: VT_ATTN_PREFILL_SHAREDK_WMMA=1.
// kQGlobal=true: Q from global (lda=hq*d), dyn smem = K+V only — enables d=512 WMMA under 64KB LDS.
// kQGlobal=false: Q tile in smem (d=256 path). Partial tiles (nq<BM) use scalar QK when kQGlobal.
template <int QG, int EPL, int BM, int BN, bool kQGlobal>
__global__ void PagedAttnPrefillSharedKWmma(__hip_bfloat16* out, const __hip_bfloat16* query,
                                            const __hip_bfloat16* k_cache, const __hip_bfloat16* v_cache,
                                            const int32_t* block_table, const int32_t* seq_lens,
                                            const int32_t* query_start_loc, int64_t num_reqs, int64_t hq,
                                            int64_t num_kv_heads, int64_t d, int64_t block_size,
                                            int64_t bt_row, int64_t bt_col, int64_t kc_blk, int64_t kc_pg,
                                            int64_t kc_hd, int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                            float scale, float softcap, bool causal, int window_left,
                                            int window_right) {
#if !defined(VT_ROCWMMA_OK)
  (void)out;
  (void)query;
  (void)k_cache;
  (void)v_cache;
  (void)block_table;
  (void)seq_lens;
  (void)query_start_loc;
  (void)num_reqs;
  (void)hq;
  (void)num_kv_heads;
  (void)d;
  (void)block_size;
  (void)bt_row;
  (void)bt_col;
  (void)kc_blk;
  (void)kc_pg;
  (void)kc_hd;
  (void)vc_blk;
  (void)vc_pg;
  (void)vc_hd;
  (void)scale;
  (void)softcap;
  (void)causal;
  (void)window_left;
  (void)window_right;
#else
  static_assert(BM % 16 == 0 && BN % 16 == 0, "WMMA tile");
  static_assert(BM >= 16 && BM <= 32 && BN <= 64, "bounds");
  constexpr int kEpl = EPL;
  constexpr int d_expect = kEpl * 32;
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr bool kMatrixPV = d_expect == 256 && BM == 16 && BN == 64 && QG == 2 && !kQGlobal;
  const int64_t q_tile = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bdx = static_cast<int>(blockDim.x);
  const int nwarps = bdx >> 5;
  if (g >= num_kv_heads || d != d_expect) return;

  const int64_t t_base = q_tile * BM;
  int64_t r = 0, q0 = 0, q1 = 0;
  if (num_reqs == 1) {
    q0 = query_start_loc[0];
    q1 = query_start_loc[1];
  } else {
    r = -1;
    for (int64_t rr = 0; rr < num_reqs; ++rr) {
      const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
      if (t_base >= a && t_base < b) {
        r = rr;
        q0 = a;
        q1 = b;
        break;
      }
    }
    if (r < 0) return;
  }
  if (t_base >= q1) return;
  const int nq = static_cast<int>((q1 - t_base) < BM ? (q1 - t_base) : BM);
  if (nq <= 0) return;

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t qg_total = hq / num_kv_heads;
  const int64_t h0 = g * qg_total + static_cast<int64_t>(blockIdx.z) * QG;
  if (h0 + QG > (g + 1) * qg_total) return;

  const int qq = warp;  // may be >= BM if block oversized; only BM warps own rows
  const bool live = (qq < nq) && (qq < BM);

  __shared__ int64_t s_jmin, s_jmax;
  __shared__ int64_t s_jmin_q[BM], s_jmax_q[BM];
  if (lane == 0 && qq < BM) {
    if (live) {
      const int64_t t = t_base + qq;
      const int64_t pos = context + (t - q0);
      int64_t jn = 0;
      if (window_left >= 0) {
        jn = pos - window_left;
        if (jn < 0) jn = 0;
      }
      int64_t jx = causal ? pos : (seqlen - 1);
      if (window_right >= 0) {
        const int64_t jr = pos + window_right;
        if (jr < jx) jx = jr;
      }
      if (jx > seqlen - 1) jx = seqlen - 1;
      s_jmin_q[qq] = jn;
      s_jmax_q[qq] = jx;
    } else {
      s_jmin_q[qq] = seqlen;
      s_jmax_q[qq] = -1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    int64_t tn = seqlen, tx = -1;
    for (int i = 0; i < nq; ++i) {
      if (s_jmin_q[i] < tn) tn = s_jmin_q[i];
      if (s_jmax_q[i] > tx) tx = s_jmax_q[i];
    }
    s_jmin = tn;
    s_jmax = tx;
  }
  __syncthreads();
  if (s_jmax < s_jmin) return;

  extern __shared__ __hip_bfloat16 dyn_bf[];
  __hip_bfloat16* q_sm = nullptr;
  __hip_bfloat16* k_sm = nullptr;
  __hip_bfloat16* v_sm = nullptr;
  if constexpr (kQGlobal) {
    // Dyn = K+V only (d=512: BN16 → 32KB).
    k_sm = dyn_bf;
    v_sm = kMatrixPV ? k_sm : k_sm + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  } else {
    q_sm = dyn_bf;
    k_sm = q_sm + static_cast<size_t>(BM) * QG * static_cast<size_t>(d_expect);
    v_sm = kMatrixPV ? k_sm : k_sm + static_cast<size_t>(BN) * static_cast<size_t>(d_expect);
  }
  __shared__ int64_t s_blk[BN], s_off[BN];
  __shared__ float s_sc[QG * BM * BN];

  // Q: smem tile (d=256) or per-warp regs for scalar tail (d=512 global).
  float q_reg[QG][kEpl];
  if constexpr (!kQGlobal && !kMatrixPV) {
    for (int e = tid; e < BM * QG * static_cast<int>(d); e += bdx) q_sm[e] = __hip_bfloat16{};
    __syncthreads();
    for (int e = tid; e < nq * QG * static_cast<int>(d); e += bdx) {
      const int dd = e % static_cast<int>(d);
      const int tmp = e / static_cast<int>(d);
      const int hh = tmp % QG;
      const int qi = tmp / QG;
      q_sm[(static_cast<size_t>(qi) * QG + hh) * d + dd] =
          query[((t_base + qi) * hq + (h0 + hh)) * d + dd];
    }
    __syncthreads();
  } else if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh)
      LoadRowEplBf16<kEpl>(query, (t * hq + (h0 + hh)) * d, lane, q_reg[hh]);
  }

  float m[QG], lsum[QG];
  float o_reg[QG][kEpl];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh) {
    m[hh] = -INFINITY;
    lsum[hh] = 0.f;
#pragma unroll
    for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = 0.f;
  }

  using namespace rocwmma;
  fragment<accumulator, WM, WN, WK, float> pv_acc[2];
  __shared__ float pv_alpha[QG * BM];
  if constexpr (kMatrixPV) {
    fill_fragment(pv_acc[0], 0.f);
    fill_fragment(pv_acc[1], 0.f);
  }

  // Full BM tiles can WMMA from global Q without OOB; partial last tile uses scalar QK.
  const bool use_wmma_qk = !kQGlobal || (nq == BM);
  const bool pow2_block = block_size > 0 && (block_size & (block_size - 1)) == 0;
  const bool decode = kMatrixPV && query_len == 1;
  for (int phase = decode ? 1 : 0; phase < (kMatrixPV ? 2 : 1); ++phase) {
    const int64_t begin = decode ? 0 : (kMatrixPV ? (phase == 0 ? 0 : context) : s_jmin);
    const int64_t end = kMatrixPV && phase == 0 ? context - 1 : s_jmax;
    const int tile = decode ? (pow2_block ? static_cast<int>(block_size) : 32)
                            : (kMatrixPV ? (phase == 0 || !pow2_block ? 32 : BN) : BN);
    for (int64_t j0 = begin; j0 <= end; j0 += tile) {
      if (j0 + tile <= s_jmin) continue;
      const int n_k = static_cast<int>((end - j0 + 1) < tile ? (end - j0 + 1) : tile);
      for (int kk = tid; kk < n_k; kk += bdx) {
        const int64_t j = j0 + kk;
        s_blk[kk] = block_table[r * bt_row + (j / block_size) * bt_col];
        s_off[kk] = j % block_size;
      }
      __syncthreads();
      // Prefetch K+V together (vectorized when d%8==0).
      if ((static_cast<int>(d) & 7) == 0) {
        const int d8 = static_cast<int>(d) >> 3;
        const int nvec_all = BN * d8;
        for (int e = tid; e < nvec_all; e += bdx) {
          reinterpret_cast<uint4*>(k_sm)[static_cast<size_t>(e)] = uint4{0, 0, 0, 0};
          if constexpr (!kMatrixPV)
            reinterpret_cast<uint4*>(v_sm)[static_cast<size_t>(e)] = uint4{0, 0, 0, 0};
        }
        __syncthreads();
        const int nvec = n_k * d8;
        for (int e = tid; e < nvec; e += bdx) {
          const int kk = e / d8;
          const int v = e - kk * d8;
          const int dd = v << 3;
          const int64_t bo = s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd;
          const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
          const size_t dst = static_cast<size_t>(e);
          reinterpret_cast<uint4*>(k_sm)[dst] = *reinterpret_cast<const uint4*>(k_cache + bo);
          if constexpr (!kMatrixPV)
            reinterpret_cast<uint4*>(v_sm)[dst] = *reinterpret_cast<const uint4*>(v_cache + vo);
        }
      } else {
        for (int e = tid; e < BN * static_cast<int>(d); e += bdx) {
          k_sm[e] = __hip_bfloat16{};
          if constexpr (!kMatrixPV) v_sm[e] = __hip_bfloat16{};
        }
        __syncthreads();
        for (int e = tid; e < n_k * static_cast<int>(d); e += bdx) {
          const int kk = e / static_cast<int>(d);
          const int dd = e - kk * static_cast<int>(d);
          k_sm[e] = k_cache[s_blk[kk] * kc_blk + s_off[kk] * kc_pg + g * kc_hd + dd];
          if constexpr (!kMatrixPV)
            v_sm[e] = v_cache[s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd];
        }
      }
      __syncthreads();

      if constexpr (kMatrixPV) {
        for (int e = tid; e < BM * QG * static_cast<int>(d); e += bdx) q_sm[e] = __hip_bfloat16{};
        __syncthreads();
        for (int e = tid; e < nq * QG * static_cast<int>(d); e += bdx) {
          const int dd = e % static_cast<int>(d);
          const int tmp = e / static_cast<int>(d);
          const int hh = tmp % QG;
          const int qi = tmp / QG;
          q_sm[(static_cast<size_t>(qi) * QG + hh) * d + dd] =
              query[((t_base + qi) * hq + (h0 + hh)) * d + dd];
        }
        __syncthreads();
      }

      if (use_wmma_qk) {
        // WMMA QK → s_sc[hh][qq][kk]
        constexpr int tiles_m = BM / WM;
        constexpr int tiles_n = BN / WN;
        constexpr int ntiles_h = tiles_m * tiles_n;
        constexpr int ntiles = ntiles_h * QG;
        const int step = nwarps;
        for (int t = warp; t < ntiles; t += step) {
          const int hh = t / ntiles_h;
          const int t2 = t - hh * ntiles_h;
          const int tm = t2 / tiles_n;
          const int tn = t2 - tm * tiles_n;
          const int m0 = tm * WM;
          const int n0 = tn * WN;
          fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> a_frag;
          fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> b_frag;
          fragment<accumulator, WM, WN, WK, float> acc;
          fill_fragment(acc, 0.f);
          const bfloat16_t* Kbase =
              reinterpret_cast<const bfloat16_t*>(k_sm + static_cast<size_t>(n0) * d);
          if constexpr (kQGlobal) {
            // query layout [t][hq][d]; fixed head h0+hh, row stride = hq*d
            const bfloat16_t* Qbase =
                reinterpret_cast<const bfloat16_t*>(query + ((t_base + m0) * hq + (h0 + hh)) * d);
            const long lda = static_cast<long>(hq) * static_cast<long>(d);
            for (int k = 0; k < d_expect; k += WK) {
              load_matrix_sync(a_frag, Qbase + k, lda);
              load_matrix_sync(b_frag, Kbase + k, d);
              if constexpr (kMatrixPV)
                AttentionMmaSync(acc, a_frag, b_frag, acc);
              else
                mma_sync(acc, a_frag, b_frag, acc);
            }
          } else {
            const bfloat16_t* Qbase =
                reinterpret_cast<const bfloat16_t*>(q_sm + static_cast<size_t>(hh) * d);
            const long lda = static_cast<long>(QG) * static_cast<long>(d);
            for (int k = 0; k < d_expect; k += WK) {
              if constexpr (kMatrixPV) {
                load_matrix_sync(a_frag, Kbase + k, d);
                load_matrix_sync(b_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
              } else {
                load_matrix_sync(a_frag, Qbase + static_cast<long>(m0) * lda + k, lda);
                load_matrix_sync(b_frag, Kbase + k, d);
              }
              if constexpr (kMatrixPV)
                AttentionMmaSync(acc, a_frag, b_frag, acc);
              else
                mma_sync(acc, a_frag, b_frag, acc);
            }
          }
          float* sc_base =
              s_sc + static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(m0) * BN + n0;
          store_matrix_sync(sc_base, acc, BN, kMatrixPV ? mem_col_major : mem_row_major);
        }
        __syncthreads();
      } else if (live) {
        // Partial tile + global Q: scalar QK into s_sc for this row (no OOB WMMA).
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        float k_reg[kEpl];
        LoadRowEplBf16<kEpl>(k_sm, static_cast<int64_t>(kk) * d, lane, k_reg);
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < kEpl; ++i) dot += q_reg[hh][i] * k_reg[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) dot += __shfl_down(dot, o);
          if (lane == 0)
            s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] =
                __shfl(dot, 0);
        }
      }
      __syncthreads();
    } else {
      __syncthreads();
    }

    if constexpr (kMatrixPV) {
      // All QK readers have reached the preceding barrier. Reuse K's shared
      // storage for V, keeping the 64-key probability tile below 64 KiB LDS.
      const int d8 = static_cast<int>(d) / 8;
      for (int e = tid; e < n_k * d8; e += bdx) {
        const int kk = e / d8, dd = (e % d8) * 8;
        const int64_t vo = s_blk[kk] * vc_blk + s_off[kk] * vc_pg + g * vc_hd + dd;
        reinterpret_cast<uint4*>(v_sm)[e] = *reinterpret_cast<const uint4*>(v_cache + vo);
      }
      __syncthreads();
      float probabilities[QG][2];
#pragma unroll
      for (int hh = 0; hh < QG; ++hh) {
        float scores[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int kk = lane + 32 * half;
          const int64_t j = j0 + kk;
          scores[half] = (live && kk < n_k && j >= s_jmin_q[qq] && j <= s_jmax_q[qq])
                             ? Softcap(s_sc[static_cast<size_t>(hh) * BM * BN +
                                            static_cast<size_t>(qq) * BN + kk] *
                                           scale,
                                       softcap)
                             : -INFINITY;
        }
        float tile_max = fmaxf(scores[0], scores[1]);
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset));
        const float next = fmaxf(m[hh], tile_max);
        const float alpha = m[hh] > -INFINITY ? FastExp(m[hh] - next) : 0.f;
        float sum = 0.f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float prob = next > -INFINITY ? FastExp(scores[half] - next) : 0.f;
          probabilities[hh][half] = prob;
          sum += prob;
        }
        // The primary's transposed WMMA layout reduces each lane's even
        // or odd key columns in order, then adds its paired lane's sum.
        sum = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int kk = 2 * i + (lane & 1);
          sum += __shfl(probabilities[hh][kk / 32], kk % 32);
        }
        sum += __shfl_xor(sum, 1);
        lsum[hh] = fmaf(lsum[hh], alpha, sum);
        m[hh] = next;
        if (lane == 0) pv_alpha[hh * BM + qq] = alpha;
      }
      __syncthreads();  // Every float score has been read before its storage becomes P.
      auto* p_sm = reinterpret_cast<__hip_bfloat16*>(s_sc);
#pragma unroll
      for (int hh = 0; hh < QG; ++hh)
#pragma unroll
        for (int half = 0; half < 2; ++half)
          p_sm[hh * BM * BN + qq * BN + lane + 32 * half] =
              __float2bfloat16(probabilities[hh][half]);
      __syncthreads();
      // Public rocWMMA stores expose row coordinates for alpha scaling. Q is
      // reloaded on the next iteration; its 16 KiB now holds one tile per wave.
      float* tile_scratch = reinterpret_cast<float*>(q_sm) + warp * WM * WN;
      const int pv_head = warp / 8;
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        const int column = (warp % 8) * 32 + f * WN;
        store_matrix_sync(tile_scratch, pv_acc[f], WN, mem_col_major);
        __syncthreads();
        for (int i = lane; i < WM * WN; i += 32) tile_scratch[i] *= pv_alpha[pv_head * BM + i / WN];
        __syncthreads();
        load_matrix_sync(pv_acc[f], tile_scratch, WN, mem_col_major);
        fragment<matrix_a, WM, WN, WK, bfloat16_t, col_major> pf;
        fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> vf;
        for (int k = 0; k < tile; k += WK) {
          load_matrix_sync(pf, reinterpret_cast<bfloat16_t*>(v_sm) + k * d + column, d);
          load_matrix_sync(vf, reinterpret_cast<bfloat16_t*>(p_sm) + pv_head * BM * BN + k, BN);
          AttentionMmaSync(pv_acc[f], pf, vf, pv_acc[f]);
        }
        // Triton's executing IR folds decode's acc += dot(P,V) into a dot
        // with the scaled accumulator, just as in prefix prefill.
      }
    } else {
      // Online softmax + V from v_sm (no K reload, no V global round-trip).
      if (live) {
        const int64_t jmin = s_jmin_q[qq];
        const int64_t jmax = s_jmax_q[qq];
#pragma unroll 1
      for (int kk = 0; kk < n_k; ++kk) {
        const int64_t j = j0 + kk;
        float v_reg[kEpl];
        bool loaded = false;
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) {
          float s = -INFINITY;
          if (j >= jmin && j <= jmax) {
            s = Softcap(s_sc[static_cast<size_t>(hh) * BM * BN + static_cast<size_t>(qq) * BN + kk] *
                            scale,
                        softcap);
          }
          if (!(s > -INFINITY)) continue;
          if (!loaded) {
            LoadRowEplBf16<kEpl>(v_sm, static_cast<int64_t>(kk) * d, lane, v_reg);
            loaded = true;
          }
          const float m_new = fmaxf(m[hh], s);
          const float corr = (m[hh] > -INFINITY) ? FastExp(m[hh] - m_new) : 0.f;
          const float pw = FastExp(s - m_new);
#pragma unroll
          for (int i = 0; i < kEpl; ++i) o_reg[hh][i] = o_reg[hh][i] * corr + pw * v_reg[i];
          lsum[hh] = lsum[hh] * corr + pw;
          m[hh] = m_new;
        }
      }
    }
    }
    __syncthreads();
    }
  }

  if constexpr (kMatrixPV) {
#pragma unroll
    for (int hh = 0; hh < QG; ++hh)
      if (lane == 0) pv_alpha[hh * BM + qq] = lsum[hh] + 1e-10f;
    __syncthreads();
    float* tile_scratch = reinterpret_cast<float*>(q_sm) + warp * WM * WN;
    const int pv_head = warp / 8;
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      const int column = (warp % 8) * 32 + f * WN;
      store_matrix_sync(tile_scratch, pv_acc[f], WN, mem_col_major);
      __syncthreads();
      for (int i = lane; i < WM * WN; i += 32) {
        const int row = i / WN, col = i % WN;
        if (row < nq)
          out[((t_base + row) * hq + h0 + pv_head) * d + column + col] =
              __float2bfloat16(tile_scratch[i] / pv_alpha[pv_head * BM + row]);
      }
      __syncthreads();
    }
  } else if (live) {
    const int64_t t = t_base + qq;
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
      const float inv = (lsum[hh] > 0.f) ? (1.f / lsum[hh]) : 0.f;
      float acc[kEpl];
#pragma unroll
      for (int i = 0; i < kEpl; ++i) acc[i] = o_reg[hh][i] * inv;
      StoreRowEplBf16<kEpl>(out, (t * hq + (h0 + hh)) * d, lane, acc);
    }
  }
#endif  // VT_ROCWMMA_OK
}

}  // namespace

// Test-visible dispatch counter: incremented each time the fp8 SharedK prefill
// fast path fires. Lets test_rocm_fp8_kv_cache G7 prove the dispatch was
// reached, not just that the output is correct — both SharedK and the
// PagedAttnOnline fallback use online softmax, so output comparison alone
// cannot distinguish them.
std::atomic<int>& Fp8PrefillSharedKDispatchCount() {
  static std::atomic<int> count{0};
  return count;
}

void PagedAttentionKernelRocm(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args) {
  VT_CHECK(args.scale > 0.f, "rocm paged_attention: scale must be > 0");
  if (CpuRefEnabled()) {
    PagedAttnCpuRef(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc,
                    args);
    return;
  }
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(hq % num_kv_heads == 0, "rocm paged_attention: GQA ratio");
  if (total_q == 0) return;

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  const int window_left =
      args.window_size.has_value() ? static_cast<int>(args.window_size->left) : -1;
  const int window_right =
      args.window_size.has_value() ? static_cast<int>(args.window_size->right) : -1;
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  static const bool decode_opt = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_OPT");
    if (e && e[0] == '0') return false;
    return true;
  }();
  static const bool decode_gqa = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_GQA");
    if (e && e[0] == '0') return false;
    return true;  // default ON when decode_opt
  }();

  // head_dim-128 arm (#382) -- the Qwen3-dense / Llama / Mistral head size.
  // Same kernels as d==256/512, EPL=4 instead of 8/16; before this, bf16 decode
  // at d==128 fell all the way to the generic PagedAttnOnline (#488 measured
  // that fallback at 41.1us/call against vLLM's 5.10us on gfx1200).
  //
  // DEFAULT ON (opt out with VT_ATTN_DECODE_D128=0) -- flipped from OFF with the
  // ROCm near-tie razor + distributional gate + golden regen (gfx1151, Qwen3-0.6B:
  // 16/16 prompts PASS, max gap 0.125 nats vs the pinned vLLM oracle, 0
  // forward-divergent). It is correctness-complete but NOT byte-exact against the
  // kernel it replaces: warp-strided online softmax reduces the KV sequence in a
  // different ORDER than PagedAttnOnline's per-tile loop, so a greedy anchor can
  // move at an exact bf16 tie. The re-captured ROCm device goldens are committed
  // with this flip; the CUDA arm (cuda_paged_attn.cu, DecodeD128Enabled, #382 /
  // PR #425) still ships OFF pending the same ceremony on that backend.
  static const bool decode_d128 = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_D128");
    return e == nullptr || e[0] != '0';
  }();
  // Split-KV decode gate (issue #845): multi-CTA attention for small batch/long context.
  // Default OFF to avoid regressions; enable with VT_ATTN_DECODE_SPLIT_KV=1.
  static const bool decode_split_kv = [] {
    const char* e = std::getenv("VT_ATTN_DECODE_SPLIT_KV");
    return e != nullptr && e[0] == '1';
  }();
  const bool bf16_decode_opt =
      decode_opt && ((d == 128 && decode_d128) || d == 256 || d == 512) &&
      query.dtype == DType::kBF16 &&
      k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16 && out.dtype == DType::kBF16 &&
      kc_hd % 8 == 0;

  // Split-KV decode dispatch (issue #845): multi-CTA attention for small batch/long context.
  // Gate: VT_ATTN_DECODE_SPLIT_KV=1; applies to bf16 decode paths only.
  if (bf16_decode_opt && decode_split_kv) {
    // Get SM count for occupancy heuristic
    int dev = 0;
    int sm_count = 64;  // default fallback
    if (hipGetDevice(&dev) == hipSuccess && dev >= 0) {
      hipDeviceProp_t prop{};
      if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
        sm_count = prop.multiProcessorCount;
      }
    }
    
    // Compute max sequence length for this batch
    int64_t max_seq = 0;
    if (num_reqs == 1) {
      max_seq = seq_lens.Ptr<int32_t>()[0];
    } else {
      // Host copy for max search
      std::vector<int32_t> h_seq_lens(num_reqs);
      Check(hipMemcpy(h_seq_lens.data(), seq_lens.Ptr<int32_t>(),
                      num_reqs * sizeof(int32_t), hipMemcpyDeviceToHost),
            "split_kv seq_lens copy");
      for (int64_t r = 0; r < num_reqs; ++r) {
        if (h_seq_lens[r] > max_seq) max_seq = h_seq_lens[r];
      }
    }
    
    // Compute num_splits using heuristic
    const bool is_sliding = (window_left >= 0 || window_right >= 0);
    int num_splits = ComputeNumKvSplits(static_cast<int>(max_seq), sm_count, is_sliding);
    if (num_splits < 2) num_splits = 2;  // Minimum 2 splits for split-KV to make sense
    
    // Get warp configuration
    int num_warps = GetNumSplitWarps(is_sliding);
    if (num_warps < 4) num_warps = 4;  // Minimum 4 warps
    if (num_warps > 32) num_warps = 32;  // Maximum 32 warps (1024 threads)
    
    // Allocate workspace for partial results: [total_q, hq, num_splits, EPL+1]
    // Last dimension is LSE (log-sum-exp) for online softmax merge
    const int EPL = (d == 128) ? 4 : ((d == 256) ? 8 : 16);
    const size_t workspace_elems = static_cast<size_t>(total_q) * hq * num_splits * (EPL + 1);
    std::vector<float> h_workspace(workspace_elems, 0.f);
    float* d_attn_logits = nullptr;
    Check(hipMalloc(&d_attn_logits, workspace_elems * sizeof(float)),
          "split_kv workspace alloc");
    Check(hipMemcpy(d_attn_logits, h_workspace.data(), workspace_elems * sizeof(float),
                    hipMemcpyHostToDevice),
          "split_kv workspace init");
    
    // Stage 1: Compute partial attention per split
    {
      dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq),
                static_cast<unsigned>(num_splits));
      dim3 block(num_warps * 32);
      size_t smem = 0;  // No shared memory needed for Stage 1
      
      if (d == 128) {
        PagedAttnDecodeSplitKvStage1<4, __hip_bfloat16, 32><<<grid, block, smem, st>>>(
            d_attn_logits, query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right, num_splits);
      } else if (d == 256) {
        PagedAttnDecodeSplitKvStage1<8, __hip_bfloat16, 32><<<grid, block, smem, st>>>(
            d_attn_logits, query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right, num_splits);
      } else {  // d == 512
        PagedAttnDecodeSplitKvStage1<16, __hip_bfloat16, 32><<<grid, block, smem, st>>>(
            d_attn_logits, query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right, num_splits);
      }
      Check(hipGetLastError(), "split_kv stage1 launch");
    }
    
    // Stage 2: Reduce splits into final output
    {
      dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));
      dim3 block(32);  // Single warp for reduction
      
      if (d == 128) {
        PagedAttnDecodeSplitKvReduce<4><<<grid, block, 0, st>>>(
            out.Ptr<__hip_bfloat16>(), d_attn_logits, total_q, hq, num_kv_heads, d, num_splits);
      } else if (d == 256) {
        PagedAttnDecodeSplitKvReduce<8><<<grid, block, 0, st>>>(
            out.Ptr<__hip_bfloat16>(), d_attn_logits, total_q, hq, num_kv_heads, d, num_splits);
      } else {  // d == 512
        PagedAttnDecodeSplitKvReduce<16><<<grid, block, 0, st>>>(
            out.Ptr<__hip_bfloat16>(), d_attn_logits, total_q, hq, num_kv_heads, d, num_splits);
      }
      Check(hipGetLastError(), "split_kv stage2 launch");
    }
    
    // Cleanup
    Check(hipFree(d_attn_logits), "split_kv workspace free");
    return;  // Split-KV path complete
  }
  
  if (bf16_decode_opt) {
    const int64_t qg = hq / num_kv_heads;

    // Qwen3.5-family heads (hq=16, kv=4 → qg=4, d=256): decode stays on
    // DecodeOpt above, but prefill falls all the way to the serial
    // PagedAttnOnline — measured 22-24% of a 2k-token prefill on gfx1101
    // (rocprofv3, Qwen3.5-9B-EXL3). The QG=2 SharedKWmma arm already z-tiles
    // the group via blockIdx.z (the qg=8/d=512 case proves it), so qg=4 runs
    // the same kernel at z=2. Prefill-only; nothing on the decode arm changes.
    if (decode_gqa && qg == 4 && d == 256 && num_reqs == 1 &&
        total_q >= 64 && SharedKAttentionWmmaHostOk()) {
      constexpr int BM = 16, BN = 64;
      const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
      dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), 2u);
      dim3 block(static_cast<unsigned>(BM * 32));
      const size_t smem =
          sizeof(__hip_bfloat16) *
          (static_cast<size_t>(BM) * 2 * static_cast<size_t>(d) +
           static_cast<size_t>(BN) * static_cast<size_t>(d));
      PagedAttnPrefillSharedKWmma<2, 8, BM, BN, false>
          <<<grid, block, smem, st>>>(
              out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(),
              k_cache.Ptr<__hip_bfloat16>(), v_cache.Ptr<__hip_bfloat16>(),
              block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d,
              block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg,
              vc_hd, args.scale, args.logits_soft_cap, args.causal,
              window_left, window_right);
      Check(hipGetLastError(), "paged_attention prefill_sharedk_wmma qg4 launch");
      return;
    }
    // GQA fuse: sliding QG=2. Global QG=8: tile as DecodeGqa QG=2 × z=4
    // (proven kernel; halves K/V reloads vs DecodeOpt's 8×). QG=4/8 CTAs lost A/B.
    if (decode_gqa &&
        ((qg == 2 && (d == 128 || d == 256 || d == 512)) || (qg == 8 && d == 512))) {
      // Prefill SharedK (scoreless). Default ON for long single-req prefill (lab win
      // ~1.09× @11k vs DecodeGqa-z4). Set VT_ATTN_PREFILL_FLASH_SHAREDK=0 to disable.
      // Admitted d=256 targets use rocWMMA QK/PV by default.
      static const bool prefill_sharedk = [] {
        const char* e = std::getenv("VT_ATTN_PREFILL_FLASH_SHAREDK");
        if (!e) return true;  // default ON
        return e[0] != '0';
      }();
      static const bool sharedk_wmma = [] {
        // Default ON on admitted gfx1100 and gfx12 devices.
        // Only d=256 Q-smem WMMA is admitted. d=512 stays scalar SharedK
        // until a separately gated kernel repair (shipping f58b spills).
        // =0 forces scalar SharedK for A/B.
        const char* e = std::getenv("VT_ATTN_PREFILL_SHAREDK_WMMA");
        if (!e) return true;
        return e[0] != '0';
      }();
      // The gfx1100 decode prerequisite uses identical matrix arithmetic in
      // both prefill control arms. Keep other targets and short prefill unchanged.
      if (d == 256 && num_reqs == 1 && total_q == 1 && block_size <= 64 &&
          SharedKAttentionWmmaHostOk(/*decode=*/true)) {
        if (block_size < 16 && (block_size & (block_size - 1)) == 0) {
          // Preserve the existing sub-16 cache-block realization.
          PagedAttnPrefillSharedKWmma<2, 8, 16, 64, false>
              <<<dim3(1, static_cast<unsigned>(num_kv_heads)), 512, 49152, st>>>(
                  out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(),
                  k_cache.Ptr<__hip_bfloat16>(), v_cache.Ptr<__hip_bfloat16>(),
                  block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                  query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                  bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale,
                  args.logits_soft_cap, args.causal, window_left, window_right);
          Check(hipGetLastError(), "paged_attention sharedk_wmma decode launch");
        } else {
          Rdna3AttentionDecode(q, out, query, k_cache, v_cache, block_table, seq_lens, args,
                               window_left, window_right);
        }
        return;
      }
      if (prefill_sharedk && total_q >= 64 && num_reqs == 1) {
        auto smem_kv = [&](int bn) -> size_t {
          // d=256 reuses the K tile for V after the score consumers finish.
          return sizeof(__hip_bfloat16) * static_cast<size_t>((d == 256 ? 1 : 2) * bn) *
                 static_cast<size_t>(d);
        };
        auto smem_qkv = [&](int bm, int bn) -> size_t {
          // Q[bm*QG*d] plus K[bn*d]. The d=256 V tile reuses K storage.
          return sizeof(__hip_bfloat16) *
                 (static_cast<size_t>(bm) * 2 * static_cast<size_t>(d) +
                  static_cast<size_t>((d == 256 ? 1 : 2) * bn) * static_cast<size_t>(d));
        };
        const unsigned z_groups = (qg == 8 && d == 512) ? 4u : 1u;
        // Host must not launch WMMA stubs on devices excluded at compilation
        // (kernel body is empty there → silent garbage out). Runtime prefix
        // match on hipDeviceProp_t.gcnArchName; VT_ROCWMMA_OK is device-only.
        if (sharedk_wmma && SharedKAttentionWmmaHostOk()) {
          if (d == 256) {
            // Q 16 KiB plus a shared K/V tile of 32 KiB. Static LDS is 9616 B.
            constexpr int BM = 16, BN = 64;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
            dim3 block(static_cast<unsigned>(BM * 32));
            PagedAttnPrefillSharedKWmma<2, 8, BM, BN, false>
                <<<grid, block, smem_qkv(BM, BN), st>>>(
                    out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(),
                    k_cache.Ptr<__hip_bfloat16>(), v_cache.Ptr<__hip_bfloat16>(),
                    block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                    query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size,
                    bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale,
                    args.logits_soft_cap, args.causal, window_left, window_right);
            Check(hipGetLastError(), "paged_attention prefill_sharedk_wmma d256 launch");
            return;
          }
        }
        if (d == 256) {
          constexpr int BM = 32, BN = 64;  // Reuse the 32 KiB K tile for V.
          const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
          dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
          dim3 block(static_cast<unsigned>(BM * 32));
          PagedAttnPrefillSharedK<2, 8, BM, BN><<<grid, block, smem_kv(BN), st>>>(
              out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
              v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
              bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
              args.causal, window_left, window_right);
        } else if (d == 512) {
          constexpr int BM = 16, BN = 16;  // K+V dual 32KB; was BN32 K-only overwrite
          const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
          dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
          dim3 block(static_cast<unsigned>(BM * 32));
          PagedAttnPrefillSharedK<2, 16, BM, BN><<<grid, block, smem_kv(BN), st>>>(
              out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
              v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
              bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
              args.causal, window_left, window_right);
        } else {
          goto flash_fallback;
        }
        Check(hipGetLastError(), "paged_attention prefill_sharedk launch");
        return;
      }
      // Prefill flash (opt-in). Default OFF — DecodeGqa remains prod.
      // FLASH=1 → PrefillFlashTile + concurrent multi-wave WMMA QK (lab 2026-08-10).
      // FLASH_WMMA=1 → 1-wave/CTA WMMA only.
      // Flash tiles are QG=2-shaped — sliding only (quality fail on global 2026-08-10).
      static const bool prefill_flash = [] {
        const char* e = std::getenv("VT_ATTN_PREFILL_FLASH");
        return e && e[0] == '1';
      }();
      if (prefill_flash && qg == 2 && total_q >= 64 && num_reqs == 1) {
        static const bool flash_wmma1 = [] {
          const char* e = std::getenv("VT_ATTN_PREFILL_FLASH_WMMA");
          return e && e[0] == '1';
        }();
        auto smem_qk = [&](int bm, int bn) -> size_t {
          return sizeof(__hip_bfloat16) *
                 (static_cast<size_t>(bm) * 2 * static_cast<size_t>(d) +
                  static_cast<size_t>(bn) * static_cast<size_t>(d));
        };
        if (flash_wmma1) {
          dim3 block(32);
          if (d == 256) {
            constexpr int BM = 16, BN = 64;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillWmmaWave<2, 8, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else if (d == 512) {
            constexpr int BM = 16, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillWmmaWave<2, 16, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else {
            goto flash_fallback;
          }
          Check(hipGetLastError(), "paged_attention prefill_wmma_wave launch");
          return;
        }
        // Default FLASH=1: concurrent WMMA direct-to-s_sc; BM=32 packs more Q/CTA
        {
          dim3 block(256);
          if (d == 256) {
            constexpr int BM = 32, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillFlashTile<2, 8, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else if (d == 512) {
            constexpr int BM = 16, BN = 32;
            const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
            dim3 grid(tiles, static_cast<unsigned>(num_kv_heads));
            PagedAttnPrefillFlashTile<2, 16, BM, BN><<<grid, block, smem_qk(BM, BN), st>>>(
                out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
                v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
                bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
                args.causal, window_left, window_right);
          } else {
            goto flash_fallback;
          }
          Check(hipGetLastError(), "paged_attention prefill_flash_wmma_mw launch");
          return;
        }
      }
      flash_fallback:
      // DecodeGqa QG=2: sliding z=1; global qg=8 → z=4 head tiles (same kernel/regs).
      const int nwarps = kDecWarps;
      const size_t smem =
          sizeof(float) *
          (static_cast<size_t>(nwarps) * 2 * static_cast<size_t>(d) +
           2 * static_cast<size_t>(nwarps) * 2);
      const unsigned z_groups = (qg == 8 && d == 512) ? 4u : 1u;
      dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(num_kv_heads), z_groups);
      dim3 block(static_cast<unsigned>(nwarps * 32));
      if (d == 128) {
        PagedAttnDecodeGqaBf16<2, 4, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      } else if (d == 256) {
        PagedAttnDecodeGqaBf16<2, 8, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      } else {
        PagedAttnDecodeGqaBf16<2, 16, 8><<<grid, block, smem, st>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
            v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
            query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row,
            bt_col, kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap,
            args.causal, window_left, window_right);
      }
      Check(hipGetLastError(), "paged_attention decode_gqa launch");
      return;
    }
    const size_t smem =
        sizeof(float) * (static_cast<size_t>(kDecWarps) * static_cast<size_t>(d) +
                         2 * static_cast<size_t>(kDecWarps));
    dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));
    if (d == 128) {
      PagedAttnDecodeOptBf16T<4><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    } else if (d == 256) {
      PagedAttnDecodeOptBf16T<8><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    } else {
      PagedAttnDecodeOptBf16T<16><<<grid, kDecWarps * 32, smem, st>>>(
          out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), k_cache.Ptr<__hip_bfloat16>(),
          v_cache.Ptr<__hip_bfloat16>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
          query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col,
          kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
          window_left, window_right);
    }
    Check(hipGetLastError(), "paged_attention decode_opt launch");
    return;
  }

  // threads: reduce over head_dim; need power-of-2 <= 256 and <= d*something
  unsigned threads = 64;
  while (threads < 256u && threads < static_cast<unsigned>(d)) threads *= 2;
  if (threads > 256u) threads = 256;
  const size_t smem = sizeof(float) * (static_cast<size_t>(d) + threads);
  dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));

  // F32-query decode GQA arm (GFX1100-TG150): the GGUF dense path runs
  // attention with an f32 query and f32 output over a bf16 KV cache
  // ("Phase 1" numerics), which excludes every bf16-decode kernel above and
  // falls to the generic PagedAttnOnline — a per-context-token
  // __syncthreads() walk measuring ~803us/call on the RX 7900 XTX (6.1
  // ms/token across the model's full-attention layers). This arm routes that
  // exact dtype combination through the DecodeGqa geometry (QG=4 fused
  // q-heads per KV group, warp-strided sequence walk). On by default: the
  // arm is correctness-complete and the PagedAttnOnline fallback it replaces
  // is the generic per-context-token walk (~212us/call, ~3.4ms/token across
  // the 27B's 16 full-attention layers). Reduction order differs from
  // PagedAttnOnline's, so greedy anchors can move at exact ties; measured
  // 22.13 -> 26.24 decode tok/s with golden "The capital of France is" ->
  // " Paris." parity. VT_ATTN_DECODE_GQA4=0 restores the fallback.
  // default ON; =0 disables
  static const bool decode_gqa4_f32q = EnvBoolStrict("VT_ATTN_DECODE_GQA4", true);
  if (decode_opt && decode_gqa4_f32q && total_q <= hq &&
      query.dtype == DType::kF32 && out.dtype == DType::kF32 &&
      ((k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16) ||
       (k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8 &&
        args.kv_cache_dtype != Fp8KVCacheDataType::kAuto)) &&
      (d == 128 || d == 256) && num_kv_heads > 0 &&
      hq % num_kv_heads == 0 &&
      (hq / num_kv_heads == 4 || hq / num_kv_heads == 6 ||
       hq / num_kv_heads == 8)) {
    constexpr int kDecWarpsG = 8;
    // The kernel tiles a wider GQA group over blockIdx.z in chunks of QG.
    // qg_total 4 -> QG=4 z=1 (the TG150 arm), 6 -> QG=6 z=1 (Qwen3.5-dense
    // 27B class: hq=24, kv=4), 8 -> QG=4 z=2.
    const int64_t qg_total = hq / num_kv_heads;
    const int qg = qg_total == 4 ? 4 : qg_total == 6 ? 6 : 4;
    const unsigned z_groups = static_cast<unsigned>(qg_total / qg);
    constexpr int kNwarpsG = kDecWarpsG;
    // EPL = d / 32 lanes: 8 at head_dim 256 (the original TG150 arm), 4 at
    // head_dim 128 (Qwen3-dense class, TG200). One kernel body; it
    // self-guards d != EPL*32, so a wrong pairing exits without reading.
    const int64_t smem_qg =
        static_cast<int64_t>(kNwarpsG) * qg * (static_cast<int64_t>(d) + 2);
    const size_t smem = sizeof(float) * static_cast<size_t>(smem_qg);
    // Split-KV over blockIdx.z (z = z_groups * num_splits): one CTA per KV head
    // walks the whole context serially, which left 4 CTAs on the GPU for this
    // model and cost ~5 ms/token at 2k context. Chunks are sized on device, so
    // short contexts keep all work in split 0. VT_ATTN_GQA4_SPLITS=1 restores
    // the single-CTA walk.
    static const int gqa4_splits = [] {
      const int v = EnvIntStrict("VT_ATTN_GQA4_SPLITS", 16);
      return v < 1 ? 1 : (v > 64 ? 64 : v);
    }();
    const int num_splits = gqa4_splits;
    float* part = nullptr;
    if (num_splits > 1) {
      const size_t need = sizeof(float) * static_cast<size_t>(total_q) *
                          static_cast<size_t>(hq) * static_cast<size_t>(num_splits) *
                          static_cast<size_t>(d + 2);
      part = EnsureGqaSplitScratch(need, st);
    }
    dim3 grid3(static_cast<unsigned>(total_q), static_cast<unsigned>(num_kv_heads),
               z_groups * static_cast<unsigned>(num_splits));
    dim3 block3(static_cast<unsigned>(kNwarpsG * 32));
    const bool gqa4_fp8 = k_cache.dtype == DType::kI8;
    const float ks = gqa4_fp8 ? args.k_scale : 1.0f;
    const float vs = gqa4_fp8 ? args.v_scale : 1.0f;
#define VT_GQA4_LAUNCH(QG_, EPL_, TKV_)                                                  \
  PagedAttnDecodeGqaF32Q<QG_, EPL_, kDecWarpsG, TKV_><<<grid3, block3, smem, st>>>(      \
      out.Ptr<float>(), query.Ptr<float>(), k_cache.Ptr<TKV_>(), v_cache.Ptr<TKV_>(),     \
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), \
      num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,     \
      vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal, window_left,    \
      window_right, ks, vs, num_splits, part)
    if (d == 128) {
      if (gqa4_fp8) {
        if (qg == 6) VT_GQA4_LAUNCH(6, 4, uint8_t); else VT_GQA4_LAUNCH(4, 4, uint8_t);
      } else {
        if (qg == 6) VT_GQA4_LAUNCH(6, 4, __hip_bfloat16); else VT_GQA4_LAUNCH(4, 4, __hip_bfloat16);
      }
    } else {
      if (gqa4_fp8) {
        if (qg == 6) VT_GQA4_LAUNCH(6, 8, uint8_t); else VT_GQA4_LAUNCH(4, 8, uint8_t);
      } else {
        if (qg == 6) VT_GQA4_LAUNCH(6, 8, __hip_bfloat16); else VT_GQA4_LAUNCH(4, 8, __hip_bfloat16);
      }
    }
#undef VT_GQA4_LAUNCH
    Check(hipGetLastError(), "paged_attention decode_gqa_f32q launch");
    if (num_splits > 1) {
      dim3 gridr(static_cast<unsigned>(total_q), static_cast<unsigned>(hq));
      if (d == 128)
        PagedAttnGqaSplitReduceF32<4><<<gridr, 32, 0, st>>>(out.Ptr<float>(), part, hq, d,
                                                            num_splits);
      else
        PagedAttnGqaSplitReduceF32<8><<<gridr, 32, 0, st>>>(out.Ptr<float>(), part, hq, d,
                                                            num_splits);
      Check(hipGetLastError(), "paged_attention decode_gqa_f32q split reduce");
    }

    return;
  }

  // f32-query + bf16-KV prefill fast path (Qwen3.5's served dtype mix:
  // PagedAttnOnline<float, __hip_bfloat16, float> measured 22-24% of a
  // 2k-token prefill on gfx1101). Same kernel + z-tiling the fp8 arm uses,
  // bf16 KV instead of fp8. total_q>=64 mirrors the fp8 admission.
  static const bool bf16kv_prefill_sharedk =
      EnvBoolStrict("VT_ATTN_PREFILL_F32Q_B16KV", true);
  const bool f32q_bf16kv_prefill =
      bf16kv_prefill_sharedk && query.dtype == DType::kF32 &&
      out.dtype == DType::kF32 && k_cache.dtype == DType::kBF16 &&
      v_cache.dtype == DType::kBF16 && d == 256 && total_q >= 64 &&
      num_reqs == 1;
  if (f32q_bf16kv_prefill) {
    const int64_t qg = hq / num_kv_heads;
    if (qg == 2 || qg == 4 || qg == 8) {
      constexpr int QG = 2;
      constexpr int EPL = 8;   // d=256
      constexpr int BM = 32, BN = 32;
      const unsigned z_groups = static_cast<unsigned>(qg / QG);
      const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
      dim3 gridp(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
      dim3 blockp(static_cast<unsigned>(BM * 32));
      const size_t smemp = sizeof(__hip_bfloat16) * 2u * BN * d;
      PagedAttnPrefillSharedK<QG, EPL, BM, BN, __hip_bfloat16, float, float>
          <<<gridp, blockp, smemp, st>>>(
              out.Ptr<float>(), query.Ptr<float>(),
              k_cache.Ptr<__hip_bfloat16>(), v_cache.Ptr<__hip_bfloat16>(),
              block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
              query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d,
              block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk,
              vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
              window_left, window_right, args.k_scale, args.v_scale);
      Check(hipGetLastError(), "paged_attention f32q_bf16kv prefill_sharedk launch");
      return;
    }
  }
  auto launch = [&](auto q_tag, auto kv_tag, auto o_tag) {
    using TQ = decltype(q_tag);
    using TKV = decltype(kv_tag);
    using TO = decltype(o_tag);
    PagedAttnOnline<TQ, TKV, TO><<<grid, threads, smem, st>>>(
        out.Ptr<TO>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
        block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(),
        num_reqs, hq, num_kv_heads, d, block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd,
        vc_blk, vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal, window_left,
        window_right, args.k_scale, args.v_scale);
  };

  // fp8 KV-cache prefill fast path: SharedK kernel with fp8 dequant at load
  // time. Handles f32-query + fp8-KV (Qwen3.5-4B: hq=16, kv=4, QG=4, d=256).
  // QG=2 kernel tiles QG=4 via z=2. Replaces PagedAttnOnline O(n²) walk.
  const bool fp8_prefill_fast =
      args.kv_cache_dtype != Fp8KVCacheDataType::kAuto &&
      k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8 &&
      query.dtype == DType::kF32 && out.dtype == DType::kF32 &&
      d == 256 && total_q >= 64 && num_reqs == 1;
  if (fp8_prefill_fast) {
    const int64_t qg = hq / num_kv_heads;
    if (qg == 2 || qg == 4 || qg == 8) {
      static const bool prefill_sharedk = [] {
        const char* e = std::getenv("VT_ATTN_PREFILL_FLASH_SHAREDK");
        if (!e) return true;
        return e[0] != '0';
      }();
      if (prefill_sharedk) {
        Fp8PrefillSharedKDispatchCount().fetch_add(1, std::memory_order_relaxed);
        constexpr int QG = 2;
        constexpr int EPL = 8;   // d=256
        constexpr int BM = 32, BN = 32;  // K+V 32KB LDS — matches bf16 d=256 path
        const unsigned z_groups = static_cast<unsigned>(qg / QG);
        const unsigned tiles = static_cast<unsigned>((total_q + BM - 1) / BM);
        dim3 grid(tiles, static_cast<unsigned>(num_kv_heads), z_groups);
        dim3 block(static_cast<unsigned>(BM * 32));
        const size_t smem = sizeof(__hip_bfloat16) * 2u * BN * d;
        PagedAttnPrefillSharedK<QG, EPL, BM, BN, uint8_t, float, float>
            <<<grid, block, smem, st>>>(
                out.Ptr<float>(), query.Ptr<float>(),
                k_cache.Ptr<uint8_t>(), v_cache.Ptr<uint8_t>(),
                block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
                query_start_loc.Ptr<int32_t>(), num_reqs, hq, num_kv_heads, d,
                block_size, bt_row, bt_col, kc_blk, kc_pg, kc_hd, vc_blk,
                vc_pg, vc_hd, args.scale, args.logits_soft_cap, args.causal,
                window_left, window_right, args.k_scale, args.v_scale);
        Check(hipGetLastError(), "paged_attention fp8 prefill_sharedk launch");
        return;
      }
    }
  }
  // fp8 KV-cache read: cache pages are uint8_t (DType::kI8), dequantized
  // inside LoadKv. Only the correctness-grade PagedAttnOnline kernel serves
  // fp8 — the decode-opt bf16 path stages __hip_bfloat16 fragments and a
  // tensor-core fp8 read is a performance brick, not this one (same scope
  // as the CUDA arm, spec W2).
  const bool fp8 = args.kv_cache_dtype != Fp8KVCacheDataType::kAuto;
  if (fp8) {
    VT_CHECK(k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8,
             "rocm paged_attention fp8: cache must be kI8");
    if (query.dtype == DType::kBF16 && out.dtype == DType::kBF16) {
      launch(__hip_bfloat16{}, uint8_t{}, __hip_bfloat16{});
    } else if (query.dtype == DType::kF32 && out.dtype == DType::kF32) {
      launch(float{}, uint8_t{}, float{});
    } else if (query.dtype == DType::kBF16 && out.dtype == DType::kF32) {
      launch(__hip_bfloat16{}, uint8_t{}, float{});
    } else {
      VT_CHECK(false, "rocm paged_attention fp8: unsupported query/output dtype");
    }
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kF32) {
    launch(float{}, float{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
  } else if (query.dtype == DType::kBF16 && k_cache.dtype == DType::kF32 &&
             out.dtype == DType::kBF16) {
    launch(__hip_bfloat16{}, float{}, __hip_bfloat16{});
  } else if (query.dtype == DType::kF32 && k_cache.dtype == DType::kBF16 &&
             out.dtype == DType::kF32) {
    launch(float{}, __hip_bfloat16{}, float{});
  } else {
    VT_CHECK(false, "rocm paged_attention dtype");
  }
  Check(hipGetLastError(), "paged_attention launch");
}

}  // namespace vt::rocm
