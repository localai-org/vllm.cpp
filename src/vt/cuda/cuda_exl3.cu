// EXL3 (exllamav3 trellis) device kernels, CUDA arm — MODEL-DSV4-EXL3 W2a/W2b.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT).
// vLLM implements no EXL3 at the parity pin, so exllamav3 is this format's
// secondary oracle (AGENTS.md "When vLLM has no implementation"); see
// `.agents/specs/model-dsv4-exl3.md`, whose `## W2 design` fixes the parity
// contract, the output dtype and the shape policy this file obeys.
//
// The files this is a port of, and what each contributed:
//   exllamav3_ext/ptx.cuh:6-16,52-74,103-139,159-212  fragments, mma, the
//       global barrier, cp.async and ldmatrix
//   exllamav3_ext/quant/codebook.cuh:44-54,92-116     the MCG (cb 1) codebook
//   exllamav3_ext/quant/exl3_dq.cuh:5-13,96-161,254-293  the tail-biting window
//       read and dq_dispatch
//   exllamav3_ext/quant/hadamard_inner.cuh:17-44,93-279  shuffle_had_f4x32 and
//       the hf / ff / fh inners
//   exllamav3_ext/quant/exl3_gemm_inner.cuh:22-733    the pipelined tile loop
//   exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-80     the fused had -> gemm ->
//       had grid-cooperative kernel
//   exllamav3_ext/quant/exl3_gemm.cu:110-309          the launcher
//   exllamav3_ext/quant/exl3_gemv.cu:29-169           the GEMV try-launch (W2c)
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh:1-402    the GEMV kernel (W2c)
//   exllamav3_ext/quant/exl3_moe.cu:99-301            the MoE launcher (W2d)
//   exllamav3_ext/quant/exl3_moe_kernel.cuh:17-283    the fused MoE kernel (W2d)
//   exllamav3_ext/quant/hadamard_inner.cuh:284-473    its guad / d epilogues (W2d)
//   exllamav3_ext/quant/exl3_devctx.cu:59-70          the per-device lock buffer
//
// DELIBERATE DIVERGENCES FROM THE UPSTREAM SOURCE, all recorded:
//   * The torch/ATen layer is gone. Tensors arrive as `vt::Tensor`, the stream
//     comes from the `vt::Queue`, and every TORCH_CHECK is a VT_CHECK in
//     src/vt/ops.cpp, so the refusals are shared with the CPU arm rather than
//     duplicated per backend.
//   * `register` is dropped from the fragment declarations. The keyword is
//     REMOVED in C++17 and this tree compiles CUDA at -std=c++20.
//   * The shape table lives in src/vt/exl3_policy.cpp as pure host code, so it
//     is gated on a machine with no GPU. This file calls it; it does not carry
//     a second copy.
//   * THREE (bits, codebook) arms are INSTANTIATED: (3, 0), (3, 1) and (6, 0).
//     It was ONE, (3, 1), which was the whole of the DeepSeek-V4-Flash 3.0bpw
//     artifact and NOTHING ELSE -- `LinearEXL3` derives the codebook from tensor
//     PRESENCE (`exl3.py:74-77`), so every stock `turboderp/*-exl3` artifact
//     ships no marker and is cb 0, and the device arm refused all of them
//     (QUANT-EXL3, #2181). (6, 0) is the stock `lm_head`, which is SIX-bit over
//     a 3-bit body and is 21% of a 1B model's weights. Every other pair still
//     refuses BY NAME from the launcher and is recorded owed in the row's spec;
//     the CPU arm stays generic over bits, so the reference is not narrowed with
//     the kernel.
//   * The grid sync is a hand-rolled sense-reversing barrier rather than
//     `cooperative_groups`. NOT a preference: `grid_group::sync()` lowers to a
//     `cudadevrt` call and therefore requires `-rdc=true` device linking, which
//     this tree sets nowhere (no target carries CUDA_SEPARABLE_COMPILATION and
//     nothing else under src/vt/cuda/ includes the cooperative-groups header).
//     Upstream itself replaces the same sync with a hand-rolled barrier on newer
//     architectures (`ptx.cuh:319-348` `group_barrier`, taken by
//     `exl3_gemm_kernel.cuh:166-170`), so this is upstream's own primitive and
//     not an invention. `cudaLaunchCooperativeKernel` is still what makes the
//     barrier legal: it is what guarantees every block is CO-RESIDENT, and a
//     spin barrier over blocks that are not co-resident deadlocks.
//   * The autotuner (`coop_autotune.cuh`) is NOT ported. Shape selection goes
//     through the table, and the GEMV arm is tried before it exactly as
//     `exl3_gemm.cu:220-236` tries it.
//   * W2c added the m<=8 GEMV (`exl3_gemv.cu` + `exl3_gemv_kernel.cuh`) and W2d
//     the fused MoE mgemm (`exl3_moe.cu` + `exl3_moe_kernel.cuh`). Both are
//     narrowed to bits == 3, codebook == 1; the REGULAR kernel is no longer so
//     narrowed (it carries (3,0), (3,1) and (6,0)), and
//     the GEMV's fp16 accumulation gives it its OWN bound (tier 3c, spec
//     `## W2cd design` W2c-3) rather than the regular kernel's tier 3.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/cuda/cublas_lt_hgemm.h"
#include "vt/cuda/cuda_device_caps.h"
#include "vt/cuda/cuda_exl3_internal.h"
#include "vt/cuda/graph_safe_scratch.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda exl3: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

#define EXL3_MIN(a, b) ((a) < (b) ? (a) : (b))
#define EXL3_MAX(a, b) ((a) > (b) ? (a) : (b))
#define EXL3_CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))

// exl3_gemm_inner.cuh:6-7. 90 KiB is the largest dynamic shared-memory opt-in
// that every architecture in this tree's fat build accepts (sm_86 caps at
// 99 KiB, sm_89 at 100 KiB, GB10 reports 101376).
constexpr int kBaseThreads = 256;
constexpr int kSmemMax = 90 * 1024;

// exl3_devctx.cuh:7-9. The per-device lock buffer is indexed by output tile
// column, and one int per 16-wide column block covers any n this tree loads. Two
// more ints past the end are the grid barrier's counter and sense, which is
// where upstream's BARRIER_LOCKS_OFFSET puts its own.
constexpr size_t kMaxTilesC = 1024 * 1024;
// exl3_devctx.cuh:8-15. Past the tile locks come `2 * MAX_BARRIERS` ints for the
// group barriers (a counter and a sense per group; the single-group GEMM uses
// group 0, which is where upstream's BARRIER_LOCKS_OFFSET puts its own), and
// past those the MoE scheduler's self-resetting ticket state: [0] next ticket,
// [1] retired groups, [2 + g] the ticket published to group g.
constexpr size_t kMaxBarriers = 1024;
constexpr size_t kBarrierLocksOffset = kMaxTilesC;
constexpr size_t kMoeSchedOffset = kMaxTilesC + 2 * kMaxBarriers;
constexpr size_t kMoeSchedInts = 2 + 64;  // MOE_SCHED_INTS, MOE_MAX_GROUPS == 64
constexpr size_t kLockInts = kMoeSchedOffset + kMoeSchedInts;

// ── compat types (exllamav3_ext/compat.cuh, util.cuh) ────────────────────────

struct __align__(8) half4 {
  half2 x;
  half2 y;
};

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
};

// util.cuh:83-90. The 16-bit sibling, which only the mul1 codebook needs: cb 2
// REINTERPRETS an integer sum as an fp16 bit pattern rather than arriving at a
// half through arithmetic.
union half_uint16 {
  uint16_t as_uint16;
  half as_half;
  __device__ half_uint16(uint16_t val) : as_uint16(val) {}
};

// ── fragments and PTX (ptx.cuh) ──────────────────────────────────────────────

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
// ptx.cuh:16. The FP16-accumulate fragment the m<=8 GEMV arm uses (W2c).
using FragC_h = Vec<half2, 2>;
// EXL3_GEMV_MAX_M (exl3_gemv_kernel.cuh:31). The host copy is vt::kExl3GemvMaxM
// in include/vt/ops.h; this one is what the kernel's template arithmetic reads,
// and a static_assert below pins the two together.
constexpr int kExl3GemvMaxMDev = 8;

// ptx.cuh:52-74. FP16 @ FP16 + FP32 -> FP32.
__device__ inline void ptx_mma_m16n8k16(const FragA& frag_a, const FragB& frag_b,
                                        FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  const float* d = reinterpret_cast<const float*>(&frag_c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(d[0]), "f"(d[1]),
        "f"(d[2]), "f"(d[3]));
}

// ptx.cuh:103-139. The split-K column barrier: threadblocks in one output column
// hand the partial sum down in reverse k order, and the LAST one resets the lock.
__device__ inline void barrier_acquire(int* lock, int stage) {
  if (threadIdx.x == 0) {
    volatile int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    } while (state != stage);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, int val, bool reset) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      *lock = 0;
      return;
    }
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

// The whole-grid barrier, in the shape of upstream's `group_barrier`
// (ptx.cuh:319-348) with `cuda::atomic_ref` spelled as the plain atomics +
// `__threadfence()` this tree already uses, and specialised to ONE group (every
// block). `cs[0]` is the arrival counter and `cs[1]` the sense; both live in the
// zero-initialised per-device lock buffer and the barrier restores the counter
// to 0, so successive launches need no re-zeroing.
//
// CORRECTNESS RESTS ON CO-RESIDENCY. Every block spins until the last one
// arrives, which only terminates because `cudaLaunchCooperativeKernel` refuses a
// grid that does not fit on the device at once. No block can lap another:
// round N+1's flip needs every block to have arrived in round N+1, which needs
// each of them to have left round N first.
__device__ inline void group_barrier(int* barrier_cs, int group_id, int group_size) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int* cs = barrier_cs + group_id * 2;
    volatile int* sense = cs + 1;
    const int old_sense = *sense;
    __threadfence();  // publish this block's writes before announcing arrival
    const int old = atomicAdd(cs, 1);
    if (old == group_size - 1) {
      atomicExch(cs, 0);
      __threadfence();
      *sense = 1 - old_sense;  // release: everyone may proceed
    } else {
      while (*sense == old_sense) __nanosleep(32);
      __threadfence();  // acquire: see what the other blocks wrote
    }
  }
  __syncthreads();
}

// The whole-grid case is one group. Kept as a named entry point so the GEMM
// kernel below reads the way it did before the MoE arm needed the general form.
__device__ inline void grid_barrier(int* cs, int group_size) {
  group_barrier(cs, 0, group_size);
}

// ptx.cuh:159-212.
__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr) {
  const int bytes = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
               "}\n" ::"r"(smem),
               "l"(glob_ptr), "n"(bytes));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// ── the MCG codebook, cb == 1 (codebook.cuh:44-54,109-116) ───────────────────
//
// `lop3.b32 ... 0x6a` is `(a & b) ^ c`, so the three instructions are
// `x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` and the two fp16 halves
// are then summed in fp16. W1a's host `Exl3DecodeMcg` is the same three, and
// tier 1 of the parity contract requires the two to agree bit for bit.
__device__ inline half2 decode_mcg_product_2(uint32_t x0, uint32_t x1) {
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  half2_uint32 xu0(x0);
  half2_uint32 xu1(x1);
  half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
  half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
  return __hadd2(d0, d1);
}

// ── the mul1 codebook, cb == 2 (codebook.cuh:25-41,76-89) ────────────────────
//
// A DIFFERENT SHAPE, not a third multiplier. cb 0 and cb 1 mask, xor and sum
// the two fp16 halves of the product; cb 2 sums the product's four UNSIGNED
// BYTES into a fixed accumulator, reinterprets that sum as an fp16 bit pattern,
// and maps it with a fused fp16 affine. So it cannot be reached by widening the
// arm above, and getting it wrong is the quiet failure this family's records
// document: a weight with the right distribution and no correlation to the true
// one.
//
// `__dp4a(x, 0x01010101u, acc)` is `acc + b0 + b1 + b2 + b3` over the unsigned
// bytes of `x`. Upstream notes it is bit-identical to the `vabsdiff4(x, 0, acc)`
// it replaced, and native on Blackwell where vabsdiff4 is emulated.
//
// `acc == 0x6400` is load-bearing rather than arbitrary, and upstream's own
// comment says so: fp16 has an ULP of exactly 1.0 across the whole
// `[1024, 2048)` binade, and the byte sum cannot exceed `4 * 255 == 1020`, so
// `0x6400 + sum` READ AS AN FP16 PATTERN is exactly the integer `1024 + sum`.
//
// The two constants are BIT PATTERNS, and the decimals in upstream's comments
// are rounded: `0x1eee` is `887/131072 == 0.00676727294921875`, not "0.00677",
// and `0xc931` is `-1329/128 == -10.3828125`, not "-10.39". W1a's host
// `Exl3DecodeCodeword(cw, 2)` is the same arithmetic and the two are required to
// agree bit for bit, which `tests/vt/test_exl3_gemm.cpp` gates.
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1) {
  const uint32_t acc = 0x6400u;
  uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
  uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
  half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
  half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
  half_uint16 h0(static_cast<uint16_t>(sum0));
  half_uint16 h1(static_cast<uint16_t>(sum1));
  return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// `decode_3inst<cb>` (`codebook.cuh:56-90`), two codewords at a time.
//
// ALL THREE ARMS, because which one a checkpoint uses is decided by tensor
// PRESENCE and no arm is the default in practice. `LinearEXL3` reads
// `mcg_tensor is not None` and `mul1_tensor is not None` (`exl3.py:74-77`), so
// every stock `turboderp/*-exl3` artifact -- shipping neither marker -- is cb 0,
// the original QTIP 3INST; the SparkInfer DeepSeek-V4 artifact ships an `mcg`
// marker and is cb 1; `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships a `mul1` marker
// on every quantized linear and is cb 2 (#2495). Instantiating only cb 1 made
// the device arm refuse every ordinary EXL3 checkpoint (QUANT-EXL3, #2181), and
// omitting cb 2 refused that 27B one whole (QUANT-EXL3-MUL1, #2495).
template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
  static_assert(cb == 0 || cb == 1 || cb == 2,
                "exl3: upstream defines codebooks 0 (3INST), 1 (mcg) and 2 (mul1) and "
                "nothing else -- `decode_3inst<cb>` (codebook.cuh:56-90) falls off the "
                "end for any other value");
  // `else` rather than an early return, so that no arm of the other codebook is
  // even instantiated for cb 2 -- the two tails are different functions, not two
  // multipliers into one.
  if constexpr (cb == 2) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    return decode_mul1_product_2(x0, x1);
  } else {
    if constexpr (cb == 0) {
      x0 *= 89226354u;
      x0 += 64248484u;
      x1 *= 89226354u;
      x1 += 64248484u;
    } else {
      x0 *= 0xCBAC1FEDu;
      x1 *= 0xCBAC1FEDu;
    }
    return decode_mcg_product_2(x0, x1);
  }
}

// ── the tail-biting window read (exl3_dq.cuh) ────────────────────────────────

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
  uint64_t merged = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
  return static_cast<uint32_t>(merged >> shift);
}

// exl3_dq.cuh:96-161. `align` is how many of the eight consecutive codewords
// share one funnel shift; for bits == 3 upstream picks 4 (`:267`).
template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0,
                                    FragB& frag1) {
  int b1 = (t_offset + 257) * bits;
  int b0 = b1 - 16;
  int b2 = b1 + bits * 7;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;

  uint32_t a = ptr[i0 % (bits * 256 / 32)];
  uint32_t b = ptr[i2 % (bits * 256 / 32)];
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  if constexpr (align == 4) {
    w7 = fshift(b, a, s2);
    w6 = w7 >> bits;
    w5 = w6 >> bits;
    w4 = w5 >> bits;
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = w3 >> bits;
    w1 = w2 >> bits;
    w0 = w1 >> bits;
  } else {
    w7 = fshift(b, a, s2);
    w6 = fshift(b, a, s2 + bits);
    w5 = fshift(b, a, s2 + bits * 2);
    w4 = fshift(b, a, s2 + bits * 3);
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = fshift(b, a, s2 + bits * 5);
    w1 = fshift(b, a, s2 + bits * 6);
    w0 = fshift(b, a, s2 + bits * 7);
  }
  frag0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
  frag0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
  frag1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
  frag1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// exl3_dq.cuh:37-56. FOUR codewords per funnel shift instead of eight, and the
// reason is arithmetic rather than taste: `dq8` spans `16 + bits*7` bits across
// the two uint32 words it merges, which is 58 bits at bits == 6 and overflows
// the 64-bit funnel once the shift is added. Upstream therefore routes bits
// 5/6/8 through this one (`exl3_dq.cuh:274-293`), and bits 7 through `dq2x2`.
template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag) {
  int b0 = (t_offset + 257) * bits - 16;  // start of the first word
  int b1 = b0 + 3 * bits;                 // start of the last word
  int b2 = b1 + 16;                       // end of the last word
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;

  uint32_t a = ptr[i0 % (bits * 256 / 32)];
  uint32_t b = ptr[i2 % (bits * 256 / 32)];
  uint32_t w3 = fshift(b, a, s2) & 0xffff;
  uint32_t w2 = fshift(b, a, s2 + bits) & 0xffff;
  uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
  uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
  frag[0] = decode_3inst_2<cb>(w0, w1);
  frag[1] = decode_3inst_2<cb>(w2, w3);
}

// exl3_dq.cuh:254-293, over the widths this tree instantiates. THE ROUTE PER
// WIDTH IS UPSTREAM'S AND IS ARITHMETIC, NOT TASTE:
//
//   bits 3, 4  ->  dq8, which merges TWO uint32 words and peels eight 16-bit
//                  windows out of them. The eight windows span `16 + bits*7`
//                  bits, which is 37 at bits 3 and 44 at bits 4, and both stay
//                  inside the 64-bit funnel at any start offset.
//   bits 5, 6  ->  TWO dq4s. The same span is 51 bits at bits 5 and 58 at bits
//                  6, and once the start shift is added the window crosses a
//                  third uint32, so the two-word load no longer covers it.
//                  Halving it gives `16 + bits*3`, which is 31 and 34, and both
//                  fit. Upstream routes 5/6/8 this way for exactly that reason
//                  (`exl3_dq.cuh:273-292`).
//
// `align` is how many consecutive windows share one funnel shift. Upstream
// picks 4 for bits 3 (`:267`) and hand-writes an align-4 form for bits 4
// (`dq8_aligned_4bits`, `:164`); the generic `dq8<bits, cb, 4>` here computes
// the same eight windows, and the hand-written version differs only in using
// immediate rather than register shifts.
//
// Every other width is refused at the launcher, so the refusal names the row and
// the missing instantiation rather than firing here.
//
// None of these widths is exotic. The stock `turboderp/*-exl3` artifacts
// quantize the BODY at 3 bits and the `lm_head` at 6, so a device arm without
// bits 6 leaves 21% of the weights of a 1B model on a CPU queue; and
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` is 270 tensors at bits 4, one at 5 and one
// at 6 (#2495) -- a mixed-width artifact where NO single width would serve it.
template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0,
                                            FragB& frag1) {
  static_assert(bits == 3 || bits == 4 || bits == 5 || bits == 6,
                "exl3: only bits 3, 4, 5 and 6 are instantiated on the device arm");
  if constexpr (bits == 3 || bits == 4) {
    dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
  } else {
    dq4<bits, cb>(ptr, idx, frag0);
    dq4<bits, cb>(ptr, idx + 4, frag1);
  }
}

// ── the Hadamard inners (hadamard_inner.cuh) ─────────────────────────────────

// hadamard_inner.cuh:17-44. Levels 4..64 of the 128-point transform, carried by
// five xor-partner shuffles. The sign flip is an XOR of the f32 SIGN BIT, which
// is what src/vt/cpu/cpu_exl3_kernels.cpp reproduces so the two arms are byte
// equal (spec `## W2 design` §1 tier 2).
__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3,
                                         const int lane_id) {
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint32_t i0 = __float_as_uint(h0);
    uint32_t i1 = __float_as_uint(h1);
    uint32_t i2 = __float_as_uint(h2);
    uint32_t i3 = __float_as_uint(h3);
    uint64_t h01 = static_cast<uint64_t>(i0) | (static_cast<uint64_t>(i1) << 32);
    uint64_t h23 = static_cast<uint64_t>(i2) | (static_cast<uint64_t>(i3) << 32);
    uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
    uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
    float ph0 = __uint_as_float(static_cast<uint32_t>(ph01 & 0xffffffff));
    float ph1 = __uint_as_float(static_cast<uint32_t>(ph01 >> 32));
    float ph2 = __uint_as_float(static_cast<uint32_t>(ph23 & 0xffffffff));
    float ph3 = __uint_as_float(static_cast<uint32_t>(ph23 >> 32));
    int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
    i0 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i1 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i2 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    i3 ^= static_cast<uint32_t>(sfm) & 0x80000000u;
    h0 = __uint_as_float(i0) + ph0;
    h1 = __uint_as_float(i1) + ph1;
    h2 = __uint_as_float(i2) + ph2;
    h3 = __uint_as_float(i3) + ph3;
  }
}

// hadamard_inner.cuh:78-89. The half2 variant: five xor-partner shuffles with a
// sign flip that XORs the fp16 sign bit in BOTH lanes (0x80008000). Used by the
// fused reconstruct kernel's Hadamard butterfly (reconstruct.cu:269,300-301).
__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id) {
  for (int i = 1; i < 32; i <<= 1) {
    half2 pv = __shfl_xor_sync(0xffffffff, v, i);
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
    *vi ^= static_cast<uint32_t>(sfm) & 0x80008000u;
    v = __hadd2(v, pv);
  }
  return v;
}

// hadamard_inner.cuh:93-147. Half vector, half scales, half out.
template <bool pre_scale, bool post_scale>
__device__ inline void had_hf_r_128_inner(const half* __restrict__ input_ptr,
                                          half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  half4 v = reinterpret_cast<const half4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);
  }
  float v0 = __half2float(__low2half(v.x));
  float v1 = __half2float(__high2half(v.x));
  float v2 = __half2float(__low2half(v.y));
  float v3 = __half2float(__high2half(v.y));
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);
  }
  reinterpret_cast<half4*>(output_ptr)[t] = v;
}

// hadamard_inner.cuh:151-212. Float vector, half scales, float out.
template <bool pre_scale, bool post_scale>
__device__ inline void had_ff_r_128_inner(const float* __restrict__ input_ptr,
                                          float* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  float4 v = reinterpret_cast<const float4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  float v0 = v.x;
  float v1 = v.y;
  float v2 = v.z;
  float v3 = v.w;
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  // DEVIATION, recorded and DELIBERATE: upstream's float inner runs two
  // shuffle_had_f2x32 passes over (x,y) and (z,w) (hadamard_inner.cuh:192-193)
  // while the half inner runs one shuffle_had_f4x32 over all four. The two are
  // the SAME butterfly in the same order — f2x32 packs two values per shuffle,
  // f4x32 four — so using f4x32 here is byte-identical and keeps ONE
  // implementation of the level-4..64 stage for a byte gate to point at.
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v.x = h0 * r_scale;
  v.y = h1 * r_scale;
  v.z = h2 * r_scale;
  v.w = h3 * r_scale;
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  reinterpret_cast<float4*>(output_ptr)[t] = v;
}

// hadamard_inner.cuh:216-279. Float vector, half scales, HALF out — the arm the
// GEMM's fp16 destination takes, where the post-scale applies AFTER the round.
template <bool pre_scale, bool post_scale>
__device__ inline void had_fh_r_128_inner(const float* __restrict__ input_ptr,
                                          half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale,
                                          const int scale_block) {
  (void)scale;
  (void)scale_block;
  int t = threadIdx.x & 31;
  float4 v = reinterpret_cast<const float4*>(input_ptr)[t];
  if constexpr (pre_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
  }
  float v0 = v.x;
  float v1 = v.y;
  float v2 = v.z;
  float v3 = v.w;
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  half4 o;
  o.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  o.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
  if constexpr (post_scale) {
    int i = scale_block * 32 + t;
    half4 scales = reinterpret_cast<const half4*>(scale)[i];
    o.x = __hmul2(o.x, scales.x);
    o.y = __hmul2(o.y, scales.y);
  }
  reinterpret_cast<half4*>(output_ptr)[t] = o;
}

// ── the standalone had_r_128 kernels (hadamard.cu:9-37) ──────────────────────

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_hf_r_128_kernel(const half* __restrict__ input_ptr,
                                                          half* __restrict__ output_ptr,
                                                          const half* __restrict__ scale,
                                                          const float r_scale) {
  const size_t off = static_cast<size_t>(gridDim.y) * 128 * blockIdx.x + blockIdx.y * 128;
  had_hf_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale,
                                            static_cast<int>(blockIdx.y));
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32) void had_ff_r_128_kernel(const float* __restrict__ input_ptr,
                                                          float* __restrict__ output_ptr,
                                                          const half* __restrict__ scale,
                                                          const float r_scale) {
  const size_t off = static_cast<size_t>(gridDim.y) * 128 * blockIdx.x + blockIdx.y * 128;
  had_ff_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale,
                                            static_cast<int>(blockIdx.y));
}

// ── the pipelined tile loop (exl3_gemm_inner.cuh:22-733) ─────────────────────

// SHMEM_OUT_HAD is upstream's `shmem_out_had` (exl3_gemm_inner.cuh:22). It
// selects the epilogue: `true` stages the finished tile in shared memory and
// applies the output Hadamard against `post_scale`; `false` stores the f32
// accumulator straight to global memory as fp16. Only the caller knows whether
// a `post_scale` exists, so the branch is a template parameter and not a
// run-time test on the pointer.
template <int bits, bool c_fp32, int cb, int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
          int SH_STAGES, int FRAG_STAGES, bool SHMEM_OUT_HAD>
__device__ inline void exl3_gemm_kernel_inner(const half* __restrict__ A,
                                              const uint16_t* __restrict__ B,
                                              void* __restrict__ C, const int size_m,
                                              const int size_k, const int size_n,
                                              int* __restrict__ locks,
                                              const half* post_scale) {
  const int TILEBLOCKS_M = TILESIZE_M / 16;
  const int TILEBLOCKS_K = TILESIZE_K / 16;
  const int TILEBLOCKS_N = TILESIZE_N / 16;
  const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (kBaseThreads / 32);

  const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                        // in halfs
  const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;  // in uint16s
  const int sh_c_size = EXL3_MAX(4 * kBaseThreads * FRAGS_N_PER_WARP, TILESIZE_N * TILESIZE_M);

  const int A_COLS = TILESIZE_K / 8;
  const int A_SWIZZLE_MASK = A_COLS - 1;
  const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

  static_assert(kBaseThreads == 256);
  // Upstream carries FRAG_STAGES 1..5; the four shipped shapes use 5 (shape 1)
  // and 3 (shapes 2-4), so only those two ladders are transcribed. Without this
  // assert a shape asking for 2 or 4 would compile to a kernel whose main loop
  // is EMPTY — a silent no-op rather than a build error.
  static_assert(FRAG_STAGES == 1 || FRAG_STAGES == 3 || FRAG_STAGES == 5,
                "MODEL-DSV4-EXL3 W2 transcribes the FRAG_STAGES 1/3/5 ladders only");
  static_assert(TILESIZE_M == 16, "Invalid kernel params");
  static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
  static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
  static_assert(kSmemMax >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) +
                                4 * sh_c_size,
                "Invalid kernel params (insufficient shared memory for shape)");

  extern __shared__ half shared[];
  half* sh_a = shared;
  uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + SH_STAGES * sh_a_stage_size);
  float* sh_c = reinterpret_cast<float*>(sh_b + sh_b_stage_size * SH_STAGES);

  int t = threadIdx.x % kBaseThreads;
  int sub_k = threadIdx.x / kBaseThreads;
  int warp_id = t / 32;
  int lane_id = t % 32;

  int tiles_k = size_k / TILESIZE_K;
  int tiles_n = size_n / TILESIZE_N;
  int blocks_n = tiles_n * TILEBLOCKS_N;

  int num_slices = gridDim.x;
  int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
  int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
  int slice_len = slice_end - slice_beg;
  if (slice_len < 1) return;

  auto index_k = [&](int slice_i) { return (slice_i % tiles_k); };
  auto index_n = [&](int slice_i) { return (slice_i / tiles_k); };

  const int slice_m = 0;

  int slice0_k = index_k(slice_beg);
  int slice0_n = index_n(slice_beg);
  int slice0_iters = slice_len;

  int gl_a_stride_m = TILESIZE_M * size_k;
  const int gl_a_stride_k = TILESIZE_K;
  const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
  const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
  half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

  const int load_a_iters = EXL3_CEIL_DIVIDE(sh0_a_stride_m / 8, kBaseThreads);
  bool pred_a_gl[load_a_iters];
  int load_a_gl[load_a_iters];
  int load_a_sh[load_a_iters];
  for (int i = 0; i < load_a_iters; ++i) {
    int k = (i * kBaseThreads + t) % (gl_a_stride_k / 8);
    int m = (i * kBaseThreads + t) / (gl_a_stride_k / 8);
    load_a_gl[i] = m * size_k / 8 + k;
    load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
    pred_a_gl[i] = m < size_m;
  }

  int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
  const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
  const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
  const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
  uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

  const int load_b_iters = EXL3_CEIL_DIVIDE(sh0_b_stride_k / 8, kBaseThreads);
  bool pred_b_gl[load_b_iters];
  int load_b_gl[load_b_iters];
  for (int i = 0; i < load_b_iters; ++i) {
    int n = (i * kBaseThreads + t) % (gl_b_stride_n / 8);
    int k = (i * kBaseThreads + t) / (gl_b_stride_n / 8);
    load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
    pred_b_gl[i] = i * kBaseThreads + t < sh0_b_stride_k / 8;
  }

  auto advance0 = [&]() {
    slice0_k++;
    slice0_iters--;
    int stage = slice0_iters % SH_STAGES;
    sh0_a_ptr = sh_a + stage * sh_a_stage_size;
    sh0_b_ptr = sh_b + stage * sh_b_stage_size;
    if (slice0_k >= tiles_k) {
      slice0_k = 0;
      slice0_n++;
      gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
      gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    } else {
      gl_a_ptr += gl_a_stride_k;
      gl_b_ptr += gl_b_stride_k;
    }
  };

  int slice1_k = slice0_k;
  int slice1_iters = slice0_iters;
  half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
  uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

  auto advance1 = [&]() {
    slice1_k++;
    slice1_iters--;
    int stage = slice1_iters % SH_STAGES;
    sh1_a_ptr = sh_a + stage * sh_a_stage_size;
    sh1_b_ptr = sh_b + stage * sh_b_stage_size;
    if (slice1_k >= tiles_k) slice1_k = 0;
  };

  int slice2_k = slice0_k;
  int slice2_k0 = slice0_k;
  int slice2_n = slice0_n;
  int slice2_iters = slice0_iters;

  int gl_c_stride_n = TILESIZE_N;
  int gl_c_stride_m = TILESIZE_M * size_n;

  half* gl_c_ptr_16 = static_cast<half*>(C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
  float* gl_c_ptr_32 =
      static_cast<float*>(C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

  FragA frag_a[FRAG_STAGES];
  FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
  FragC frag_c[FRAGS_N_PER_WARP];

  auto advance2 = [&]() {
    slice2_k++;
    slice2_iters--;
    if (slice2_k >= tiles_k) {
      slice2_k = 0;
      slice2_k0 = 0;
      slice2_n++;
      if constexpr (c_fp32)
        gl_c_ptr_32 += gl_c_stride_n;
      else
        gl_c_ptr_16 += gl_c_stride_n;
    }
  };

  auto async_load_gl = [&]() {
    if (sub_k) {
      cp_async_fence();
      return;
    }
    if (slice0_iters) {
      {
        const int4* gl = reinterpret_cast<const int4*>(gl_a_ptr);
        int4* sh = reinterpret_cast<int4*>(sh0_a_ptr);
#pragma unroll
        for (int i = 0; i < load_a_iters; ++i)
          if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
      }
      {
        const int4* gl = reinterpret_cast<const int4*>(gl_b_ptr);
        int4* sh = reinterpret_cast<int4*>(sh0_b_ptr);
#pragma unroll
        for (int i = 0; i < load_b_iters; ++i)
          if (pred_b_gl[i]) cp_async(sh + kBaseThreads * i + t, gl + load_b_gl[i]);
      }
      advance0();
    }
    cp_async_fence();
  };

  auto load_frags = [&](int buf) {
    if (!slice1_iters) return;
    {
      int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
      int base_c = lane_id / 16 + sub_k * 2;
#pragma unroll
      for (int m = 0; m < TILEBLOCKS_M; ++m) {
        int R = r + m * 16;
        int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
        ldsm4(frag_a[buf], reinterpret_cast<int4*>(sh1_a_ptr) + R * A_COLS + c_swizzled);
      }
    }
#pragma unroll
    for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2) {
      int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
      const uint32_t* shb =
          reinterpret_cast<const uint32_t*>(sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
      dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
    }
    __syncthreads();
    advance1();
  };

  auto clear_frag_c = [&]() {
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) frag_c[n] = {};
  };

  auto threadblock_reduce = [&]() {
    auto store = [&](int i) {
      if (sub_k == i) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
#pragma unroll
          for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
        }
      }
      __syncthreads();
    };
    auto add = [&](int i) {
      if (sub_k == i) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
#pragma unroll
          for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
        }
      }
    };
    auto store_small = [&](int i) {
      if (sub_k == i && lane_id / 4 < size_m) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
          *sh_red++ = frag_c[n][0];
          *sh_red++ = frag_c[n][1];
        }
      }
      __syncthreads();
    };
    auto add_small = [&](int i) {
      if (sub_k == i && lane_id / 4 < size_m) {
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
#pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
          frag_c[n][0] += *sh_red++;
          frag_c[n][1] += *sh_red++;
        }
      }
    };

    if (size_m <= 8) {
      if constexpr (TILEBLOCKS_K == 2) {
        store_small(1);
        add_small(0);
      }
      if constexpr (TILEBLOCKS_K == 3) {
        store_small(1);
        add_small(0);
        store_small(2);
        add_small(0);
      }
      if constexpr (TILEBLOCKS_K == 4) {
        store_small(3);
        add_small(2);
        store_small(1);
        add_small(0);
        store_small(2);
        add_small(0);
      }
    } else {
      if constexpr (TILEBLOCKS_K == 2) {
        store(1);
        add(0);
      }
      if constexpr (TILEBLOCKS_K == 3) {
        store(1);
        add(0);
        store(2);
        add(0);
      }
      if constexpr (TILEBLOCKS_K == 4) {
        store(3);
        add(2);
        store(1);
        add(0);
        store(2);
        add(0);
      }
    }
  };

  auto write_sum_tile_sh = [&]() {
    const int n0 = warp_id * FRAGS_N_PER_WARP;
    const int r0 = lane_id / 4;
    const int r1 = r0 + 8;
    if (r0 < size_m) {
      const int c = (lane_id % 4) * 2;
#pragma unroll
      for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
        float* c_ptr = sh_c + r0 * TILESIZE_N + (n0 + n) * 8 + c;
        *c_ptr++ = frag_c[n][0];
        *c_ptr++ = frag_c[n][1];
      }
    }
    if (r1 < size_m) {
      const int c = (lane_id % 4) * 2;
#pragma unroll
      for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
        float* c_ptr = sh_c + r1 * TILESIZE_N + (n0 + n) * 8 + c;
        *c_ptr++ = frag_c[n][2];
        *c_ptr++ = frag_c[n][3];
      }
    }
  };

  // exl3_gemm_inner.cuh:456-480. The output transform reads the finished tile
  // out of shared memory and applies svh. `scale_block` is the GLOBAL 128-column
  // index, which upstream reaches through blockIdx.y in the standalone kernel
  // and through the post_scale POINTER here.
  auto output_had_sh_gl = [&]() {
    int sh_warp = warp_id;
    constexpr int active_warps = kBaseThreads / 32;
    for (;; sh_warp += active_warps) {
      int col = sh_warp % (TILESIZE_N / 128);
      int row = sh_warp / (TILESIZE_N / 128);
      if (row >= size_m) break;
      const float* had_in = sh_c + row * TILESIZE_N + col * 128;
      const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;
      if constexpr (c_fp32) {
        float* had_out = gl_c_ptr_32 + row * size_n + col * 128;
        had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f, 0);
      } else {
        half* had_out = gl_c_ptr_16 + row * size_n + col * 128;
        had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f, 0);
      }
    }
  };

  auto read_sum_gl = [&]() {
    int n0 = warp_id * FRAGS_N_PER_WARP;
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
      int r0 = lane_id / 4;
      int r1 = r0 + 8;
      int c = (lane_id % 4) * 2;
      if (r0 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
          frag_c[n][0] += *c_ptr++;
          frag_c[n][1] += *c_ptr++;
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
          float2 interm = __half22float2(*c_ptr);
          frag_c[n][0] += interm.x;
          frag_c[n][1] += interm.y;
        }
      }
      if (r1 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
          frag_c[n][2] += *c_ptr++;
          frag_c[n][3] += *c_ptr++;
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
          float2 interm = __half22float2(*c_ptr);
          frag_c[n][2] += interm.x;
          frag_c[n][3] += interm.y;
        }
      }
    }
  };

  auto write_sum_gl = [&]() {
    int n0 = warp_id * FRAGS_N_PER_WARP;
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
      int r0 = lane_id / 4;
      int r1 = r0 + 8;
      int c = (lane_id % 4) * 2;
      if (r0 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
          *c_ptr++ = frag_c[n][0];
          *c_ptr++ = frag_c[n][1];
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
          *c_ptr = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
        }
      }
      if (r1 < size_m) {
        if constexpr (c_fp32) {
          float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
          *c_ptr++ = frag_c[n][2];
          *c_ptr++ = frag_c[n][3];
        } else {
          half2* c_ptr = reinterpret_cast<half2*>(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
          *c_ptr = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
        }
      }
    }
  };

  auto reduce = [&]() {
    threadblock_reduce();
    int lock_i = tiles_k - slice2_k - 1;
    int lock_d = slice2_k - slice2_k0 + 1;
    int* lock = &locks[slice_m * blocks_n + slice2_n];
    barrier_acquire(lock, lock_i);
    bool first = lock_i == 0;
    bool last = lock_i + lock_d == tiles_k;
    if (!sub_k && !first) read_sum_gl();
    if (!sub_k && !last) write_sum_gl();
    // exl3_gemm_inner.cuh:610-623. The last block in the column stages the tile
    // for the output Hadamard only when the caller supplied a `post_scale`.
    // Without one it takes the same global store as the intermediate blocks,
    // which rounds the f32 accumulator to fp16 through __floats2half2_rn.
    if (!sub_k && last) {
      if constexpr (SHMEM_OUT_HAD)
        write_sum_tile_sh();
      else
        write_sum_gl();
    }
    if constexpr (SHMEM_OUT_HAD) {
      if (last) __syncthreads();
      if (!sub_k && last) output_had_sh_gl();
    }
    barrier_release(lock, lock_d, last);
    clear_frag_c();
  };

  auto wait_stage = [&]() {
    cp_async_wait<SH_STAGES - 2>();
    __syncthreads();
  };

  auto matmul = [&](int buf) {
#pragma unroll
    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
      ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
  };

#pragma unroll
  for (int i = 0; i < SH_STAGES - 1; ++i) async_load_gl();
  wait_stage();

  clear_frag_c();
  if constexpr (FRAG_STAGES > 1) load_frags(0);

#define EXL3_FSTAGE_OLD(_load, _mul)                                              \
  async_load_gl();                                                                \
  wait_stage();                                                                   \
  load_frags(_load);                                                              \
  matmul(_mul);                                                                   \
  if (slice2_k == tiles_k - 1 || slice2_iters == 1) {                             \
    reduce();                                                                     \
    slice2_k0 = slice2_k + 1;                                                     \
  }                                                                               \
  advance2();                                                                     \
  if (!slice2_iters) break;

#define EXL3_FSTAGE(_load, _mul)                                                  \
  async_load_gl();                                                                \
  wait_stage();                                                                   \
  matmul(_mul);                                                                   \
  if (slice2_k == tiles_k - 1 || slice2_iters == 1) {                             \
    reduce();                                                                     \
    slice2_k0 = slice2_k + 1;                                                     \
  }                                                                               \
  advance2();                                                                     \
  if (!slice2_iters) break;                                                       \
  load_frags(_load);

  if constexpr (FRAG_STAGES == 1) {
    while (true) {
      EXL3_FSTAGE_OLD(0, 0)
    }
  }
  if constexpr (FRAG_STAGES == 3) {
    while (true) {
      EXL3_FSTAGE(1, 0)
      EXL3_FSTAGE(2, 1)
      EXL3_FSTAGE(0, 2)
    }
  }
  if constexpr (FRAG_STAGES == 5) {
    while (true) {
      EXL3_FSTAGE(1, 0)
      EXL3_FSTAGE(2, 1)
      EXL3_FSTAGE(3, 2)
      EXL3_FSTAGE(4, 3)
      EXL3_FSTAGE(0, 4)
    }
  }
#undef EXL3_FSTAGE_OLD
#undef EXL3_FSTAGE
}

// ── the grid-cooperative fused kernel (exl3_gemm_kernel.cuh:8-80) ────────────
//
// One launch does all three steps: the whole grid transforms A into A_had, meets
// at the grid barrier, then walks the output in 16-row bands, and the output
// transform rides the tail of the tile loop. The barrier is why this must be a
// COOPERATIVE launch: it spins, so every block has to be resident.
template <int bits, bool c_fp32, int cb, int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
          int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(kBaseThreads* TILESIZE_K / 16) void exl3_gemm_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks,
    const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh) {
  // The two ints the grid barrier owns sit past the tile locks, which is exactly
  // where upstream puts its own barrier counters (`exl3_devctx.cuh:9`,
  // BARRIER_LOCKS_OFFSET == MAX_TILES_C).
  int* barrier_cs = locks + kBarrierLocksOffset;
  const int blocks = static_cast<int>(gridDim.x);

  {
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid)
      had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                      suh + (this_warp * 128) % size_k, 0.088388347648f, 0);
    grid_barrier(barrier_cs, blocks);
    A = A_had;
  }

  int size_m_ = size_m;
  const half* A_ = A;
  void* C_ = C;

  while (size_m_ > 0) {
    // exl3_gemm_kernel.cuh:40 instantiates shmem_out_had = true here, and `svh`
    // is the post_scale the output Hadamard reads.
    exl3_gemm_kernel_inner<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES,
                           FRAG_STAGES, true>(A_, B, C_, EXL3_MIN(size_m_, 16), size_k, size_n,
                                              locks, svh);
    A_ += 16 * size_k;
    if constexpr (c_fp32)
      C_ = static_cast<void*>(static_cast<float*>(C_) + 16 * size_n);
    else
      C_ = static_cast<void*>(static_cast<half*>(C_) + 16 * size_n);
    size_m_ -= 16;
    if (size_m_ > 0 || svh) grid_barrier(barrier_cs, blocks);
  }
}


// ── the m<=8 GEMV arm (exl3_gemv_kernel.cuh:1-402) ───────────────────────────
//
// A QTIP-style small-m path on the unmodified EXL3 format. Warps split k and
// never synchronise in the main loop; B streams to registers behind a prefetch
// ring; the two-word bit windows are resolved in-warp by lane shuffles; one
// m16n8k16 MMA pair per 16x16 weight tile.
//
// IT IS A DIFFERENT NUMERIC ARM, not only a faster one, and that is why it has
// its own bound. The accumulate is `mma...f16.f16.f16.f16` — FP16 — folded to an
// f32 pair only every FOLD iterations, so an fp16 accumulator absorbs up to
// FOLD*16 k-elements. The regular kernel accumulates in f32 throughout. The
// spec's `## W2cd design` W2c-3 states the resulting bound (tier 3c, 6.0e-3
// relative RMS) rather than reusing tier 3's 1.0e-3, which a correct kernel
// here could not meet.
//
// NARROWED to bits 3 and 4, over codebooks 1 (mcg) and 2 (mul1), which is NOT
// the regular kernel's set: that one is seven pairs wide. Upstream's own GEMV
// list is `(4,0) (4,1) (4,2) (2,1) (2,2) (3,1) (3,2)`, so cb 0 at 3 bpw is
// excluded there too and its envelope refuses it before any kernel is chosen.
// Upstream instantiates 2/3/4 bpw over three codebooks; every other width
// DECLINES this arm and falls through to the shape table, which is upstream's
// own failure mode (`exl3_gemv_select_kernel` returns nullptr and
// `exl3_gemv_try_launch` returns false) and not an unimplemented refusal.
//
// THE WIDTH IS SPECIALIZED, THE CODEBOOK IS NOT, AND THE TWO COST DIFFERENT
// WORK. `cb` is a free template argument that reaches exactly one call, the
// `dq8_regs_*bits<cb>` extractor, so `(3, 2)` cost one template argument
// (QUANT-EXL3-PERF slice A, #2570). `bits` is a geometry argument: `LSTRIDE`,
// the uint32 stride per warp load, is `bits == 3 ? 24 : 32`
// (`exl3_gemv_kernel.cuh:153`), the 24-lane load guard applies to bits 3 alone
// (`:228-231`), and the bit-window extraction is a different function per width
// (`dq8_regs_3bits` vs `dq8_regs_4bits`, `:87`, `:121`). Slice B ports the
// bits-4 half of all three (QUANT-EXL3-PERF slice B, #2570). 2 bpw stays
// unported: no 2-bit EXL3 artifact has reached this tree, and its `LOADS`
// halving and two-tiles-per-word shuffle (`:104-117`, `:302-310`) are a third
// geometry with nothing to gate it against.

// mma.m16n8k16 with the A operand as two FragB halves, FP16 accumulate
// (exl3_gemv_kernel.cuh:37-52).
__device__ inline void ptx_mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b,
                                    FragC_h& c) {
  const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
  const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
  const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
  uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(cc[0]), "+r"(cc[1])
      : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]), "r"(bb[0]), "r"(bb[1]));
}

// The register form of `dq8<3, cb, 4>` with the per-lane funnel alignment
// precomputed (exl3_gemv_kernel.cuh:120-134). Same eight codewords the shared
// -memory form reads, from two already-loaded words.
template <int cb>
__device__ inline void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  w7 = fshift(b, a, s2);
  w6 = w7 >> 3;
  w5 = w6 >> 3;
  w4 = w5 >> 3;
  w3 = fshift(b, a, s2 + 12);
  w2 = w3 >> 3;
  w1 = w2 >> 3;
  w0 = w1 >> 3;
  f0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
  f0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
  f1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
  f1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// The two immediate-operand PTX forms `dq8_regs_4bits` is written around
// (`ptx.cuh:314-315`). They are macros rather than functions because the shift
// amount and the field offset are ASSEMBLER IMMEDIATES: `shf.r.wrap.b32` and
// `bfe.u32` take them in the instruction text, so a runtime argument cannot
// reach them and a template non-type argument cannot be stringified into the
// `asm` literal without one. `#undef`d immediately below, so neither name
// escapes this arm.
//
// `VT_EXL3_FSHF_IMM(d, lo, hi, i)` is `fshift(lo, hi, i)` with `i` fixed, and
// `VT_EXL3_BFE16_IMM(d, s, i)` is `(s >> i) & 0xffff`. Spelled as upstream
// spells them so the ported sequence below reads line for line against
// `exl3_gemv_kernel.cuh:87-100`.
#define VT_EXL3_FSHF_IMM(dst, lo, hi, imm) \
  asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define VT_EXL3_BFE16_IMM(dst, src, imm) \
  asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// The register form of `dq8_aligned_4bits` (exl3_gemv_kernel.cuh:86-100), which
// is itself `exl3_dq.cuh:163-184` reading from shared memory. Same eight
// codewords, from two already-loaded words.
//
// WHY THE 4-BIT WINDOW READ IS A DIFFERENT FUNCTION AND NOT A PARAMETER. At
// bits 3 the eight codewords do NOT align to the uint32 grid: a tile is 24
// words for 256 three-bit codewords, so the window start moves by 3 bits per
// codeword and every window needs its own funnel shift off a LANE-COMPUTED word
// pair (`x_src_a`, `x_src_b`, `x_s2`). At bits 4 a tile is exactly 32 words,
// one per lane, four codewords sit in each word aligned to the nibble grid, and
// the pair is always `(lane + 31) & 31` and `lane`. So the two extractors take
// different arguments and do different work; only the trailing decode is
// shared. That is why `(4, 2)` was a KERNEL PORT and `(3, 2)` was one template
// argument.
template <int cb>
__device__ inline void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1) {
  uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
  VT_EXL3_FSHF_IMM(s, b, a, 20);
  w7 = b & 0xffff;
  VT_EXL3_BFE16_IMM(w6, b, 4);
  VT_EXL3_BFE16_IMM(w5, b, 8);
  VT_EXL3_BFE16_IMM(w4, b, 12);
  VT_EXL3_BFE16_IMM(w3, b, 16);
  w2 = s & 0xffff;
  VT_EXL3_BFE16_IMM(w1, s, 4);
  VT_EXL3_BFE16_IMM(w0, s, 8);
  // Upstream routes cb 2 through `decode_pair_cb2_dp4a_` and everything else
  // through `decode_3inst_2<cb>` (`exl3_gemv_kernel.cuh:65-83`). This tree's
  // `decode_3inst_2<2>` IS that dp4a form -- it multiplies by `0x83DCD12D` and
  // calls `decode_mul1_product_2`, which is the same `__dp4a` byte sum with the
  // same `0x6400` accumulator and the same two fp16 constants -- so the split is
  // already inside one function here and the arms stay bit-identical.
  f0[0] = decode_3inst_2<cb>(w0, w1);
  f0[1] = decode_3inst_2<cb>(w2, w3);
  f1[0] = decode_3inst_2<cb>(w4, w5);
  f1[1] = decode_3inst_2<cb>(w6, w7);
}

#undef VT_EXL3_BFE16_IMM
#undef VT_EXL3_FSHF_IMM

// exl3_gemv_kernel.cuh:138-402, narrowed to bits 3 and 4. CFG 0 is the "narrow"
// config (512 threads, 2 n-tiles per warp, 16 k-splits) and CFG 1 the "wide" one
// (256 threads, 4 n-tiles, 8 k-splits). MMODE 0 is the m == 1 fast path and
// MMODE 1 covers 2 <= m <= 8 with row-guarded fragment loads.
template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256) void exl3_gemv_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks,
    const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh) {
  // The GEMV is SPECIALIZED to its widths rather than merely asserted on them,
  // which is why widening it is a kernel port and not an instantiation. Three
  // things below are per-width: `LSTRIDE`, the uint32 stride per warp load
  // (`exl3_gemv_kernel.cuh:153`); the 24-lane load and store guards, which exist
  // because a bits-3 tile is 24 words and eight lanes of the warp have nothing
  // to carry (`:228-231`, `:270`); and the bit-window extractor. Upstream also
  // carries a 2 bpw arm with a fourth geometry -- `LOADS` halved and two tiles
  // per loaded word (`:152`, `:302-310`) -- which is NOT ported, because no
  // 2-bit EXL3 artifact has reached this tree to gate it against.
  //
  // Every other width DECLINES the GEMV and falls through to the regular shape
  // table, which is upstream's own arrangement for a declined GEMV
  // (`exl3_gemm.cu:220-236`) and matches its `if (K < 2 || K > 4) return false;`
  // (`exl3_gemv.cu:110-111`). So the 6-bit lm_head loses its m<=8 fast path and
  // not its device arm (QUANT-EXL3, #2181).
  static_assert(bits == 3 || bits == 4,
                "exl3: the GEMV kernel is specialized to the 3 and 4 bpw arms");
  constexpr int WK = CFG == 0 ? 16 : 8;    // k-split (warps per block)
  constexpr int WNT = CFG == 0 ? 2 : 4;    // adjacent n-tiles per warp
  constexpr int PF = CFG == 0 ? 4 : 2;     // prefetch ring depth
  constexpr int FOLD = CFG == 0 ? 4 : 2;   // fp16 -> fp32 fold cadence (divides PF)
  constexpr int THREADS = WK * 32;
  constexpr int ROWS = MMODE == 0 ? 1 : kExl3GemvMaxMDev;
  constexpr int COLS = WNT * 16;
  constexpr int TWORDS = 8 * bits;   // uint32 per 16x16 tile
  constexpr int LOADS = WNT;         // warp loads per k-slice (bits != 2)
  // exl3_gemv_kernel.cuh:153. It equals TWORDS at both ported widths, which is
  // what lets one warp load cover one whole 16x16 tile and what makes the
  // `t * TWORDS` shared-memory tile base below agree with the `l * LSTRIDE`
  // store stride.
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;  // uint32 per load
  static_assert(LSTRIDE == TWORDS, "exl3 gemv: one warp load must cover one tile");

  int* barrier_cs = locks + kBarrierLocksOffset;
  const int blocks = static_cast<int>(gridDim.x);

  // The input scales and Hadamard, identical to the regular kernel's.
  {
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid)
      had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                      suh + (this_warp * 128) % size_k, 0.088388347648f, 0);
    grid_barrier(barrier_cs, blocks);
    A = A_had;
  }

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;
  const int chunk = EXL3_CEIL_DIVIDE(kslices, WK);
  const int ks0 = warp * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = static_cast<size_t>(ntiles) * TWORDS;
  const half2* A2 = reinterpret_cast<const half2*>(A);
  const half2 hzero = __half2half2(__ushort_as_half(0));

  const int r0 = lane >> 2;
  const size_t a_row0 = static_cast<size_t>(r0) * (size_k / 2);
  const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

  // Per-lane extraction constants, the bits == 3 arm of :199-216. At bits 4 the
  // window pair is `(lane + 31) & 31` and `lane` with no funnel offset, so these
  // three have no bits-4 meaning and upstream leaves them at zero and
  // `[[maybe_unused]]` (`:199`). Reproduced here rather than hoisted into the
  // dispatch, because reading them under `if constexpr (bits == 4)` would be a
  // silent wrong answer instead of a compile error.
  [[maybe_unused]] int x_s2 = 0, x_src_a = 0, x_src_b = 0;
  if constexpr (bits == 3) {
    const int t_offset = lane << 3;
    const int b1 = (t_offset + 257) * 3;
    const int b2 = b1 + 21;
    const int i0 = (b1 - 16) / 32;
    const int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][ROWS][COLS];
  // `[[maybe_unused]]` is upstream's own annotation (exl3_gemv_kernel.cuh:214):
  // the staging buffer collapses to one word when SMEM_STAGE is off and nothing
  // reads it, which -Werror would otherwise call a defect.
  [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1]
                                               [SMEM_STAGE ? LOADS * LSTRIDE : 1];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + static_cast<size_t>(ks0) * slice_stride + group * WNT * TWORDS +
                         lane;
    // exl3_gemv_kernel.cuh:226-232. The 24-lane guard is the BITS-3 arm only: a
    // bits-3 tile is 24 uint32 and lanes 24..31 have nothing to load, while a
    // bits-4 tile is exactly 32 and every lane carries one word.
    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __ldcs(bp + static_cast<size_t>(i) * slice_stride + l * LSTRIDE) : 0u;
      else
        return __ldcs(bp + static_cast<size_t>(i) * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;
        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }
        if constexpr (SMEM_STAGE) {
          __syncwarp();
#pragma unroll
          for (int l = 0; l < LOADS; ++l)
            if (bits != 3 || lane < 24) sh_stage[warp][l * LSTRIDE + lane] = bw[l];
          __syncwarp();
        }
        const size_t a_col = static_cast<size_t>(ks0 + i) * 8 + (lane & 3);
        FragB a01, a23;
        a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
        a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        a01[1] = hzero;
        a23[1] = hzero;
#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          // exl3_gemv_kernel.cuh:287-316. The window pair is per-width: bits 3
          // takes two LANE-COMPUTED words and a funnel offset, bits 4 takes the
          // lane's own word and its left neighbour with no offset.
          if constexpr (SMEM_STAGE) {
            const uint32_t* tp = &sh_stage[warp][t * TWORDS];
            if constexpr (bits == 4)
              dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
            else
              dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
          } else if constexpr (bits == 4) {
            const uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else {
            const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
            const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }
          ptx_mma_ab_h(a01, a23, f0, ch[t][0]);
          ptx_mma_ab_h(a01, a23, f1, ch[t][1]);
        }
        // The FOLD cadence. This is the whole reason tier 3c exists: between
        // folds the accumulator is fp16.
        if ((d + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
          for (int t = 0; t < WNT; ++t)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              acc0[t][f].x += __low2float(ch[t][f][0]);
              acc0[t][f].y += __high2float(ch[t][f][0]);
              ch[t][f][0] = hzero;
            }
        }
      }
    }

    // Cross-warp reduction over the k splits (:335-360).
    {
      const int c0 = 2 * (lane & 3);
      const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
      const int sr0 = MMODE == 0 ? 0 : r0;
      if (store0) {
#pragma unroll
        for (int t = 0; t < WNT; ++t)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + c0;
            sh_red[warp][sr0][col + 0] = acc0[t][f].x;
            sh_red[warp][sr0][col + 1] = acc0[t][f].y;
          }
      }
    }
    __syncthreads();

    const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
      const int r = idx / COLS;
      const int c = idx % COLS;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
      const int col = group * COLS + c;
      if constexpr (c_fp32)
        static_cast<float*>(C)[static_cast<size_t>(r) * size_n + col] = sum;
      else
        static_cast<half*>(C)[static_cast<size_t>(r) * size_n + col] = __float2half_rn(sum);
    }
    __syncthreads();
  }

  // The output scales and Hadamard, same semantics as the inner GEMM epilogue.
  {
    grid_barrier(barrier_cs, blocks);
    int total_warps = size_m * size_n / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid) {
      if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>(static_cast<const float*>(C) + this_warp * 128,
                                        static_cast<float*>(C) + this_warp * 128,
                                        svh + (this_warp * 128) % size_n, 0.088388347648f, 0);
      else
        had_hf_r_128_inner<false, true>(static_cast<const half*>(C) + this_warp * 128,
                                        static_cast<half*>(C) + this_warp * 128,
                                        svh + (this_warp * 128) % size_n, 0.088388347648f, 0);
    }
  }
}

// ── the fused MoE epilogues (hadamard_inner.cuh:284-473) ─────────────────────
//
// Two inners the standalone GEMM never needs. `guad` is the whole middle of the
// MoE step in one pass — the output Hadamards for gate and up, the activation
// and the gate multiply, the down pre-scale and the down input Hadamard — and
// `d` is the epilogue that turns the down GEMM's fp16 store into the f32
// accumulator with an atomicAdd.
//
// The scale pointers arrive PRE-OFFSET by the caller (`128 * token_off`), which
// is what upstream's `blockIdx.y == 0` reduces to inside the MoE kernel, so both
// index their scale arrays at lane `t`.

// The 128-point transform on the four halves a lane holds, as `had` inside
// `had_hf_r_128_guad_inner` (hadamard_inner.cuh:299-321).
__device__ inline void moe_had4(half4& v, int t, float r_scale) {
  float v0 = __half2float(__low2half(v.x));
  float v1 = __half2float(__high2half(v.x));
  float v2 = __half2float(__low2half(v.y));
  float v3 = __half2float(__high2half(v.y));
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
}

__device__ inline half2 moe_silu_h2(const half2& x) {
  // hadamard_inner.cuh:323-332, in fp16 exactly as upstream computes it.
  half2 one = __float2half2_rn(1.0f);
  half2 e = h2exp(__hneg2(x));
  return __hmul2(x, h2rcp(__hadd2(one, e)));
}

__device__ inline half2 moe_gelu_h2(const half2& x) {
  // hadamard_inner.cuh:334-342.
  float2 xf = __half22float2(x);
  const float c = 0.797884560803f;  // sqrt(2/pi)
  xf.x = 0.5f * xf.x * (1.0f + tanhf(c * (xf.x + 0.044715f * xf.x * xf.x * xf.x)));
  xf.y = 0.5f * xf.y * (1.0f + tanhf(c * (xf.y + 0.044715f * xf.y * xf.y * xf.y)));
  return __float22half2_rn(xf);
}

// `had_hf_r_128_guad_inner` (hadamard_inner.cuh:284-413), plus ONE arm that is
// not upstream's: `kExl3MoeActSiluAndMulClamp` is vLLM's `SiluAndMulWithClamp`
// (activation.py:197-201), which clamps the gate BEFORE the silu and applies the
// limit unconditionally. AGENTS.md makes vLLM the authority wherever it
// implements the behavior; see `.agents/specs/model-dsv4-exl3.md` `## W2cd
// design` W2d-2 for the number the two orders differ by.
constexpr int kExl3MoeActSilu = 0;
constexpr int kExl3MoeActGelu = 1;
constexpr int kExl3MoeActRelu2NoGate = 2;
constexpr int kExl3MoeActSiluAndMulClamp = 3;

__device__ inline void had_hf_r_128_guad_inner(const half* __restrict__ in_g,
                                               const half* __restrict__ in_u,
                                               half* __restrict__ out, const half* __restrict__ sv_g,
                                               const half* __restrict__ sv_u,
                                               const half* __restrict__ su_d, const float r_scale,
                                               const float act_limit, const int act_function) {
  const int t = threadIdx.x & 31;
  const bool gated = act_function != kExl3MoeActRelu2NoGate;

  half4 vg = {};
  half4 vu = reinterpret_cast<const half4*>(in_u)[t];
  moe_had4(vu, t, r_scale);
  half4 su = reinterpret_cast<const half4*>(sv_u)[t];
  vu.x = __hmul2(vu.x, su.x);
  vu.y = __hmul2(vu.y, su.y);

  if (gated) {
    vg = reinterpret_cast<const half4*>(in_g)[t];
    moe_had4(vg, t, r_scale);
    half4 sg = reinterpret_cast<const half4*>(sv_g)[t];
    vg.x = __hmul2(vg.x, sg.x);
    vg.y = __hmul2(vg.y, sg.y);
  }

  if (act_function == kExl3MoeActSiluAndMulClamp) {
    // vLLM's order: clamp gate to the limit (MAX only), clamp up to +/- limit,
    // THEN silu the clamped gate. Unconditional — a zero limit clamps to zero,
    // which is vLLM's own degenerate case and not upstream's "no limit".
    const half2 lo = __float2half2_rn(-act_limit);
    const half2 hi = __float2half2_rn(act_limit);
    vg.x = __hmin2(vg.x, hi);
    vg.y = __hmin2(vg.y, hi);
    vu.x = __hmin2(__hmax2(vu.x, lo), hi);
    vu.y = __hmin2(__hmax2(vu.y, lo), hi);
    vg.x = moe_silu_h2(vg.x);
    vg.y = moe_silu_h2(vg.y);
  } else {
    switch (act_function) {
      case kExl3MoeActSilu:
        vg.x = moe_silu_h2(vg.x);
        vg.y = moe_silu_h2(vg.y);
        break;
      case kExl3MoeActGelu:
        vg.x = moe_gelu_h2(vg.x);
        vg.y = moe_gelu_h2(vg.y);
        break;
      case kExl3MoeActRelu2NoGate:
        vg.x = __hmax2(vu.x, __float2half2_rn(0.0f));
        vg.y = __hmax2(vu.y, __float2half2_rn(0.0f));
        break;
      default:
        break;
    }
    // hadamard_inner.cuh:389-397: upstream's limit is optional and lands AFTER
    // the activation.
    if (act_limit != 0.0f) {
      const half2 lo = __float2half2_rn(-act_limit);
      const half2 hi = __float2half2_rn(act_limit);
      vu.x = __hmin2(__hmax2(vu.x, lo), hi);
      vu.y = __hmin2(__hmax2(vu.y, lo), hi);
      vg.x = __hmin2(vg.x, hi);
      vg.y = __hmin2(vg.y, hi);
    }
  }

  vg.x = __hmul2(vg.x, vu.x);  // the gate multiply
  vg.y = __hmul2(vg.y, vu.y);
  half4 sd = reinterpret_cast<const half4*>(su_d)[t];  // the down PRE scale
  vg.x = __hmul2(vg.x, sd.x);
  vg.y = __hmul2(vg.y, sd.y);
  moe_had4(vg, t, r_scale);
  reinterpret_cast<half4*>(out)[t] = vg;
}

// `had_hf_r_128_d_inner` (hadamard_inner.cuh:418-473). Half in, f32 out, the
// post-scale in f32 and an atomicAdd into the accumulator. The reshuffle through
// shared memory is what makes the four atomicAdds coalesced.
//
// DEVIATION, recorded: upstream declares its own `extern __shared__ float
// temp_shared[]` inside this function and slices it by `threadIdx.x / 32`. Here
// the warp-private 128-float slice arrives as a PARAMETER, so the translation
// unit carries ONE `extern __shared__` declaration instead of two of different
// types. Same memory, same slice, one fewer thing for a reader to reconcile.
__device__ inline void had_hf_r_128_d_inner(const half* __restrict__ in, float* __restrict__ out,
                                            const half* __restrict__ post, const float r_scale,
                                            float* sh) {
  const int t = threadIdx.x & 31;
  half4 v = reinterpret_cast<const half4*>(in)[t];
  float v0 = __half2float(__low2half(v.x));
  float v1 = __half2float(__high2half(v.x));
  float v2 = __half2float(__low2half(v.y));
  float v3 = __half2float(__high2half(v.y));
  float s0 = v0 + v1;
  float d0 = v0 - v1;
  float s1 = v2 + v3;
  float d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, t);
  h0 *= r_scale;
  h1 *= r_scale;
  h2 *= r_scale;
  h3 *= r_scale;
  half4 sc = reinterpret_cast<const half4*>(post)[t];
  h0 *= __low2float(sc.x);
  h1 *= __high2float(sc.x);
  h2 *= __low2float(sc.y);
  h3 *= __high2float(sc.y);
  sh[t * 4 + 0] = h0;
  sh[t * 4 + 1] = h1;
  sh[t * 4 + 2] = h2;
  sh[t * 4 + 3] = h3;
  __syncwarp();
  atomicAdd(out + 0 + t, sh[0 + t]);
  atomicAdd(out + 32 + t, sh[32 + t]);
  atomicAdd(out + 64 + t, sh[64 + t]);
  atomicAdd(out + 96 + t, sh[96 + t]);
}

// ── the fused MoE kernel (exl3_moe_kernel.cuh:17-283) ────────────────────────
//
// A persistent cooperative launch: `num_groups` groups of `group_size` blocks
// each draw expert tickets from a self-resetting scheduler in the lock buffer
// and meet at a group barrier between stages. The group barrier is legal for the
// same reason the GEMM's grid barrier is — `cudaLaunchCooperativeKernel`
// guarantees every block is co-resident, and a spin barrier over blocks that are
// not co-resident deadlocks.
//
// The five stages are the ones `src/vt/cpu/cpu_exl3_kernels.cpp` reproduces on
// the host, in the same order and with the same roundings. Both GEMM calls pass
// a NULL post_scale, so the tile loop stores its f32 accumulator as fp16 without
// an output Hadamard, and stages 3 and 5 own those.
constexpr int kMoeSmsPerExpert = 8;       // MOE_SMS_PER_EXPERT
constexpr int kMoeMaxSmsPerExpert = 32;   // MOE_MAX_SMS_PER_EXPERT
constexpr int kMoeMaxGroups = 64;         // MOE_MAX_GROUPS
constexpr int kMoeTilesizeK = 32;         // MOE_TILESIZE_K
constexpr int kMoeTilesizeM = 16;
constexpr int kMoeShStages = 3;
constexpr int kMoeFragStages = 3;

template <int bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(kBaseThreads* kMoeTilesizeK / 16) void exl3_moe_kernel(
    const half* __restrict__ hidden_state, half* __restrict__ temp_state_g,
    half* __restrict__ temp_state_u, half* __restrict__ temp_intermediate_g,
    half* __restrict__ temp_intermediate_u, float* __restrict__ output_state,
    const uint16_t* const* __restrict__ gate_trellis, const half* const* __restrict__ gate_suh,
    const half* const* __restrict__ gate_svh, const uint16_t* const* __restrict__ up_trellis,
    const half* const* __restrict__ up_suh, const half* const* __restrict__ up_svh,
    const uint16_t* const* __restrict__ down_trellis, const half* const* __restrict__ down_suh,
    const half* const* __restrict__ down_svh, const int64_t* __restrict__ expert_count,
    const int64_t* __restrict__ token_sorted, const half* __restrict__ weight_sorted,
    const int hidden_dim, const int intermediate_dim, const int num_experts,
    const int max_tokens_per_expert, const float act_limit, const int act_function,
    const int K_gate, const int K_up, const int K_down, int* __restrict__ locks) {
  const int group_idx = blockIdx.z;
  const int block_idx = blockIdx.x;
  const int group_size = gridDim.x;
  const int num_groups = gridDim.z;
  const int block_threads = kBaseThreads * kMoeTilesizeK / 16;
  const int group_threads = group_size * block_threads;
  const int warp_id = threadIdx.x / 32;
  const int warps_per_group = group_threads / 32;
  const int warps_per_block = block_threads / 32;
  const int warp_idx0 = block_idx * warps_per_block + warp_id;

  temp_state_g += static_cast<size_t>(group_idx) * max_tokens_per_expert * hidden_dim;
  temp_state_u += static_cast<size_t>(group_idx) * max_tokens_per_expert * hidden_dim;
  temp_intermediate_g += static_cast<size_t>(group_idx) * max_tokens_per_expert * intermediate_dim;
  temp_intermediate_u += static_cast<size_t>(group_idx) * max_tokens_per_expert * intermediate_dim;

  int* barrier_cs = locks + kBarrierLocksOffset;
  int* sched = locks + kMoeSchedOffset;
  locks += static_cast<size_t>(group_idx) * EXL3_MAX(hidden_dim, intermediate_dim) / 128;

  extern __shared__ float moe_shared[];

  // exl3_moe_kernel.cuh:47-50. Active experts are numbered in scan order and a
  // group takes the one matching its ticket; after finishing it draws the next
  // unclaimed ticket, so the load balances greedily without assuming a uniform
  // cost per expert.
  int ticket = group_idx;
  int start = 0;
  int end = 0;
  int expert_idx_assign = 0;
  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    start = end;
    end += static_cast<int>(expert_count[expert_idx]);
    const int token_count = end - start;
    if (token_count == 0) continue;
    if (token_count > max_tokens_per_expert) continue;
    if (expert_idx_assign++ != ticket) continue;

    const uint16_t* e_g_tr = gate_trellis[expert_idx];
    const half* e_g_su = gate_suh[expert_idx];
    const half* e_g_sv = gate_svh[expert_idx];
    const uint16_t* e_u_tr = up_trellis[expert_idx];
    const half* e_u_su = up_suh[expert_idx];
    const half* e_u_sv = up_svh[expert_idx];
    const uint16_t* e_d_tr = down_trellis[expert_idx];
    const half* e_d_su = down_suh[expert_idx];
    const half* e_d_sv = down_svh[expert_idx];

    const bool gated = act_function != kExl3MoeActRelu2NoGate;

    // stage 1: gather + input Hadamard (:85-114).
    {
      const int warps_per_token = hidden_dim / 128;
      const int total_warps = token_count * warps_per_token;
      const int64_t* top_x = token_sorted + start;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_idx = static_cast<int>(top_x[wi / warps_per_token]);
        const int token_off = wi % warps_per_token;
        const half* in_ptr = hidden_state + static_cast<size_t>(token_idx) * hidden_dim +
                             static_cast<size_t>(token_off) * 128;
        if (gated)
          had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + 128 * wi,
                                          e_g_su + 128 * token_off, 0.088388347648f, 0);
        had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + 128 * wi,
                                        e_u_su + 128 * token_off, 0.088388347648f, 0);
      }
      group_barrier(barrier_cs, group_idx, group_size);
    }

    // stage 2: the gate and up GEMMs (:117-161). `post_scale` is nullptr, so the
    // tile loop skips the output transform and rounds its f32 accumulator to
    // fp16 at the store.
    //
    // `K` is upstream's per-projection bit width, and upstream switches on it at
    // RUN TIME because its `t_bits == 0` instance leaves all three free
    // (`exl3_moe_kernel.cuh:139-149`). This port instantiates one width, so the
    // switch would be a ladder with one live arm; the parameter is kept because
    // the launcher's refusal is what enforces the equality, and dropping it here
    // would move that contract out of sight.
    auto gemm_band = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, int K,
                         int size_k, int size_n) {
      (void)K;
      int size_m = token_count;
      while (size_m > 0) {
        // exl3_moe_kernel.cuh:138 and :212 instantiate shmem_out_had = false in
        // both MoE bands, which is what makes the NULL post_scale legal.
        exl3_gemm_kernel_inner<bits, false, cb, kMoeTilesizeM, kMoeTilesizeK, MOE_TILESIZE_N,
                               kMoeShStages, kMoeFragStages, false>(
            in_addr, trellis, static_cast<void*>(out_addr), EXL3_MIN(size_m, 16), size_k, size_n,
            locks, nullptr);
        in_addr += 16 * size_k;
        out_addr += 16 * size_n;
        size_m -= 16;
      }
    };

    if (gated)
      gemm_band(temp_state_g, temp_intermediate_g, e_g_tr, K_gate, hidden_dim, intermediate_dim);
    gemm_band(temp_state_u, temp_intermediate_u, e_u_tr, K_up, hidden_dim, intermediate_dim);
    group_barrier(barrier_cs, group_idx, group_size);

    // stage 3: the fused guad pass (:165-188).
    {
      const int warps_per_token = intermediate_dim / 128;
      const int total_warps = token_count * warps_per_token;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_off = wi % warps_per_token;
        had_hf_r_128_guad_inner(temp_intermediate_g + 128 * wi, temp_intermediate_u + 128 * wi,
                                temp_intermediate_g + 128 * wi, e_g_sv + 128 * token_off,
                                e_u_sv + 128 * token_off, e_d_su + 128 * token_off,
                                0.088388347648f, act_limit, act_function);
      }
      group_barrier(barrier_cs, group_idx, group_size);
    }

    // stage 4: the down GEMM (:191-233), into `temp_state_g`.
    gemm_band(temp_intermediate_g, temp_state_g, e_d_tr, K_down, intermediate_dim, hidden_dim);
    group_barrier(barrier_cs, group_idx, group_size);

    // stage 5: the output Hadamard for d, times the routing weight, scatter-added
    // into the f32 accumulator (:237-259).
    {
      const int warps_per_token = hidden_dim / 128;
      const int total_warps = token_count * warps_per_token;
      const int64_t* top_x = token_sorted + start;
      const half* weights = weight_sorted + start;
      float* sh = moe_shared + warp_id * 128;
      for (int wi = warp_idx0; wi < total_warps; wi += warps_per_group) {
        const int token_idx = static_cast<int>(top_x[wi / warps_per_token]);
        const half weight = weights[wi / warps_per_token];
        const int token_off = wi % warps_per_token;
        float* out_ptr = output_state + static_cast<size_t>(token_idx) * hidden_dim +
                         static_cast<size_t>(token_off) * 128;
        had_hf_r_128_d_inner(temp_state_g + 128 * wi, out_ptr, e_d_sv + 128 * token_off,
                             0.088388347648f * __half2float(weight), sh);
      }
    }

    // Draw the next ticket and publish it through the end-of-expert barrier,
    // which also protects the temp buffers for reuse (:261-266).
    if (block_idx == 0 && threadIdx.x == 0)
      sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
    group_barrier(barrier_cs, group_idx, group_size);
    ticket = sched[2 + group_idx];
  }

  // Retire; the last group out resets the scheduler for the next launch
  // (:269-282). The fence is what orders every group's ticket grabs before the
  // reset, so a straggler's in-flight grab cannot leak into the next launch.
  // Upstream spells the ordering with `cuda::atomic_ref` acq_rel; this file
  // already spells every such barrier as the plain atomics plus `__threadfence`
  // this tree uses, for the reason the header records.
  if (block_idx == 0 && threadIdx.x == 0) {
    __threadfence();
    const int retired = atomicAdd(&sched[1], 1);
    if (retired == num_groups - 1) {
      atomicExch(&sched[0], 0);
      atomicExch(&sched[1], 0);
    }
  }
}

// ── the per-device lock buffer (exl3_devctx.cu:59-70) ────────────────────────

int* DeviceLocks(int device) {
  static std::mutex mtx;
  static int* locks[16] = {};
  std::lock_guard<std::mutex> lock(mtx);
  if (device < 0 || device >= 16)
    throw std::runtime_error("vt cuda exl3: device ordinal out of range");
  if (locks[device] == nullptr) {
    void* p = nullptr;
    Check(cudaMalloc(&p, kLockInts * sizeof(int)), "cudaMalloc exl3 locks");
    Check(cudaMemset(p, 0, kLockInts * sizeof(int)), "cudaMemset exl3 locks");
    locks[device] = static_cast<int*>(p);
  }
  return locks[device];
}

// exl3_gemm.cu:38,251-256. Opting a kernel into 90 KiB of dynamic shared memory
// is a per-(device, function) property; doing it per launch costs a driver
// round-trip on the hot path.
void EnsureSmemOptIn(int device, const void* kernel) {
  static std::mutex mtx;
  static std::set<std::pair<int, const void*>> done;
  std::lock_guard<std::mutex> lock(mtx);
  if (done.insert({device, kernel}).second) {
    // Diagnose the shortfall BY NAME rather than letting cudaFuncSetAttribute
    // answer `invalid argument`, which is the lesson cuda_paged_attn.cu:116-140
    // already records for this exact call.
    if (!DynamicSmemFits(static_cast<long long>(kSmemMax))) {
      const DeviceCaps& caps = GetDeviceCaps(device);
      throw std::runtime_error(
          "vt cuda exl3: exl3_gemm needs " + std::to_string(kSmemMax) +
          " B of dynamic shared memory, but sm_" + std::to_string(caps.sm_arch()) +
          " caps opt-in shared memory at " +
          std::to_string(caps.max_shared_memory_per_block_optin) +
          " B. Upstream's SMEM_MAX is 90 KiB (exl3_gemm_inner.cuh:7) and every shape's "
          "static_assert is written against it, so a smaller ceiling needs its own tile "
          "table (MODEL-DSV4-EXL3).");
    }
    Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax),
          "cudaFuncSetAttribute exl3_gemm");
  }
}

// ── the launchers ────────────────────────────────────────────────────────────

void Exl3HadR128KernelCuda(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  const int rows = static_cast<int>(in.shape[0]);
  const int cols = static_cast<int>(in.shape[1]);
  if (rows == 0 || cols == 0) return;
  const int blocks = cols / 128;
  const float r_scale = args.scale * 0.088388347648f;  // hadamard.cu:107
  const dim3 block_dim(32);
  const dim3 grid_dim(static_cast<unsigned>(rows), static_cast<unsigned>(blocks));
  cudaStream_t stream = AsStream(q);
  const half* sc = args.pre_scale != nullptr  ? args.pre_scale->Ptr<half>()
                   : args.post_scale != nullptr ? args.post_scale->Ptr<half>()
                                                : nullptr;
  const bool pre = args.pre_scale != nullptr;
  const bool post = args.post_scale != nullptr;

  if (in.dtype == DType::kF16) {
    const half* ip = in.Ptr<half>();
    half* op = out.Ptr<half>();
    if (pre)
      had_hf_r_128_kernel<true, false><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else if (post)
      had_hf_r_128_kernel<false, true><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else
      had_hf_r_128_kernel<false, false>
          <<<grid_dim, block_dim, 0, stream>>>(ip, op, nullptr, r_scale);
  } else {
    const float* ip = in.Ptr<float>();
    float* op = out.Ptr<float>();
    if (pre)
      had_ff_r_128_kernel<true, false><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else if (post)
      had_ff_r_128_kernel<false, true><<<grid_dim, block_dim, 0, stream>>>(ip, op, sc, r_scale);
    else
      had_ff_r_128_kernel<false, false>
          <<<grid_dim, block_dim, 0, stream>>>(ip, op, nullptr, r_scale);
  }
  Check(cudaGetLastError(), "exl3_had_r_128 launch");
}

// THE INSTANTIATED (bits, codebook) ARMS. Every other pair refuses BY NAME
// rather than being silently decoded as if it were one of these.
//
//   (3, 0)  the body of every stock `turboderp/*-exl3` artifact
//   (3, 1)  the SparkInfer DeepSeek-V4 artifact, which ships an `mcg` marker
//   (6, 0)  the `lm_head` of those stock artifacts, which is SIX-bit over a
//           3-bit body
//   (3, 2)  137 of the 409 trellis modules of
//           `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`, which are EVERY MLP
//           projection of the layers quantized at the low end of its 3.5 bpw
//           average (QUANT-EXL3-MUL1 slice F, #2574)
//   (4, 2)  270 more of that artifact's modules: its whole GDN tower, its
//           dense-attention projections, the other 55 MLP projections and its
//           `mtp` head (QUANT-EXL3-MUL1, #2495)
//   (5, 2)  one tensor of that checkpoint, and ALL 36 of its DFlash2 draft
//   (6, 2)  its `lm_head`, the same 6-over-narrow-body shape the stock
//           artifacts use, in the mul1 codebook
//
// (3, 2) IS THE PAIR THIS LIST HAD TO LEARN TWICE. The mul1 widths were first
// instantiated against a census that read "270 tensors at 4 bpw, one at 5, one
// at 6" -- 272 modules, when the artifact has 409. A count taken from the local
// safetensors headers rather than by range request gives {3: 137, 4: 270,
// 5: 1, 6: 1}, all 409 carrying `mul1` and none carrying `mcg`. The 137 the
// first count missed refused BY NAME, one MLP projection at a time, so the
// checkpoint could not execute a single decoder layer on a CUDA device while
// its loader arms were complete. The lesson is in the census and not in the
// table: widen this list from the artifact, never from a summary of it.
//
// AND (3, 1) IS NOW ITS NEIGHBOUR, which is the confusable pair. Both are three
// bits wide, both take the same `dq8` route, both use the same tile shapes, and
// nothing about a `cb` threaded wrongly between them fails to compile or
// changes a shape: it decodes with the other codebook's multiplier and yields a
// weight with the RIGHT DISTRIBUTION and no correlation to the true one. The
// device case in `tests/vt/test_exl3_gemm.cpp` gates both pairs against the CPU
// arm for exactly that reason.
//
// Three arms and not one, because the original single arm (3, 1) was the
// EXCEPTION rather than the rule: `LinearEXL3` derives the codebook from tensor
// PRESENCE (`exl3.py:74-77`), so an artifact with no marker is cb 0, and the
// device arm refused every ordinary EXL3 checkpoint (QUANT-EXL3, #2181).
//
// SEVEN AND NOT TWENTY-FOUR. Upstream's table is DENSE -- K in 1..8 by cb in
// 0..2, 24 translation units of 16 instantiations each -- and it can be, because
// it splits per (K, cb) into `comp_units/exl3_comp_unit_K_cbX.cu` and compiles
// for one architecture at a time. This tree has ONE translation unit and the fat
// build compiles it for ten architectures, so a dense table here would be more
// than three times the kernels of the arm set below in a single `.cu`.
//
// THIS LINE SAID SIX, AND IT SAID THE NEXT PAIR SHOULD CARRY THE PER-K SPLIT
// WITH IT. The seventh pair arrived and does not carry the split, and the reason
// is not that the argument was wrong: (3, 2) is not a new artifact's width, it
// is the SAME artifact's largest width population, missed by a census that read
// 272 modules where there are 409. Six was never the right number for #2495;
// seven was, and the split was owed at three pairs exactly as much as it is
// owed at seven. What holds unchanged is the rule the split answers: a pair
// that a NEW artifact needs carries the split with it, and this one is not that.
//
// THE WIDTHS ARE NOT FREE-FORM EITHER. `dq_dispatch` above routes 3 and 4
// through `dq8` and 5 and 6 through two `dq4`s, because the eight-window span
// `16 + bits*7` leaves the 64-bit funnel once the start shift is added at bits
// 5. A width this tree has no route for fails to COMPILE on that
// `static_assert` rather than reading the trellis wrongly.
//
// THE WIDTH IS LOAD-BEARING ON SHARED MEMORY, and the guard is already there
// rather than added here: every B stride in `exl3_gemm_kernel` is
// `256 / 16 * bits` uint16 per tile, so `sh_b_stage_size` DOUBLES from bits 3
// to bits 6, and the `static_assert(kSmemMax >= ...)` at the top of that kernel
// is what refuses a shape whose staged tiles no longer fit. A width that
// overflows it fails to COMPILE with that assert rather than silently
// mis-staging, which is why widening here is safe to attempt: the failure mode
// is loud.
constexpr bool Exl3ArmInstantiated(int bits, int cb) {
  return (bits == 3 && (cb == 0 || cb == 1)) || (bits == 6 && cb == 0) ||
         ((bits == 3 || bits == 4 || bits == 5 || bits == 6) && cb == 2);
}

template <int BITS, int CB, bool c_fp32>
const void* GemmKernelForArm(int shape_idx) {
  switch (shape_idx) {
    case 1:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<BITS, c_fp32, CB, 16, 16, 128, 6, 5>);
    case 2:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<BITS, c_fp32, CB, 16, 32, 128, 4, 3>);
    case 3:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<BITS, c_fp32, CB, 16, 32, 256, 4, 3>);
    case 4:
      return reinterpret_cast<const void*>(
          &exl3_gemm_kernel<BITS, c_fp32, CB, 16, 16, 512, 4, 3>);
    default:
      return nullptr;
  }
}

template <bool c_fp32>
const void* GemmKernelForShape(int bits, int cb, int shape_idx) {
  if (bits == 3 && cb == 0) return GemmKernelForArm<3, 0, c_fp32>(shape_idx);
  if (bits == 3 && cb == 1) return GemmKernelForArm<3, 1, c_fp32>(shape_idx);
  if (bits == 6 && cb == 0) return GemmKernelForArm<6, 0, c_fp32>(shape_idx);
  if (bits == 3 && cb == 2) return GemmKernelForArm<3, 2, c_fp32>(shape_idx);
  if (bits == 4 && cb == 2) return GemmKernelForArm<4, 2, c_fp32>(shape_idx);
  if (bits == 5 && cb == 2) return GemmKernelForArm<5, 2, c_fp32>(shape_idx);
  if (bits == 6 && cb == 2) return GemmKernelForArm<6, 2, c_fp32>(shape_idx);
  return nullptr;
}

// ── the GEMV try-launch (exl3_gemv.cu:92-169) ────────────────────────────────
//
// Returns true when it launched. A false is a DECLINE, not a failure: the caller
// falls through to the shape table unchanged, which is upstream's own
// arrangement at `exl3_gemm.cu:220-236`.
//
// The occupancy query is cached per (device, kernel) because it is a driver
// round-trip and it feeds the shape heuristic on every call
// (`exl3_gemv.cu:125-135`).
template <int BITS, int CB>
const void* GemvKernelForArm(bool c_fp32, int mmode, int cfg, bool smem) {
#define VT_EXL3_GEMV_SEL(fp32_, mm_, cfg_, sm_)                                            \
  if (c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_)                       \
    return reinterpret_cast<const void*>(                                                  \
        &exl3_gemv_kernel<BITS, fp32_, CB, mm_, cfg_, sm_>);
#define VT_EXL3_GEMV_ROW(sm_)                                                              \
  VT_EXL3_GEMV_SEL(false, 0, 0, sm_) VT_EXL3_GEMV_SEL(false, 0, 1, sm_)                    \
  VT_EXL3_GEMV_SEL(false, 1, 0, sm_) VT_EXL3_GEMV_SEL(false, 1, 1, sm_)                    \
  VT_EXL3_GEMV_SEL(true, 0, 0, sm_) VT_EXL3_GEMV_SEL(true, 0, 1, sm_)                      \
  VT_EXL3_GEMV_SEL(true, 1, 0, sm_) VT_EXL3_GEMV_SEL(true, 1, 1, sm_)
  VT_EXL3_GEMV_ROW(false)
  VT_EXL3_GEMV_ROW(true)
#undef VT_EXL3_GEMV_ROW
#undef VT_EXL3_GEMV_SEL
  return nullptr;
}


// THE GEMV ARM SET IS `(3, 1)`, `(3, 2)` AND `(4, 2)`, AND CODEBOOK 0 IS EXCLUDED ON PURPOSE
// rather than as an oversight. Upstream's own envelope refuses it:
// `exl3_gemv.cu`'s try-launch carries `if (K != 4 && cb == 0) return false;`,
// which `Exl3GemvHardEligible` transcribes at `exl3_policy.cpp:148`, and
// upstream's `SEL_GRID` list instantiates `(4,0) (4,1) (4,2) (2,1) (2,2)
// (3,1) (3,2)` — no `(3, 0)`. `tests/vt/test_exl3_gemv.cpp:130` already
// asserted that refusal before this row existed.
//
// A first cut of W3 added `(3, 0)` here anyway, which is 16 kernels that can
// never launch in a translation unit the fat build compiles for ten
// architectures, with three comments claiming they were reachable. The claim
// even reached a measurement: a `VT_EXL3_GEMV=1` vs `=0` A/B on a stock
// codebook-0 checkpoint was reported as an 8% GEMV effect when neither arm
// could take the GEMV at all — it ran the same path twice. A fresh review
// caught it.
//
// SO A STOCK `turboderp/*-exl3` CHECKPOINT HAS NO GEMV FAST PATH, on this tree
// or upstream's, and takes the regular shape table at `m == 1`. That is
// upstream's behaviour, not a gap this row opened.
//
// THE SAME IS TRUE OF THE mul1 WIDTHS, AND FOR TWO DIFFERENT REASONS
// (QUANT-EXL3-MUL1, #2495):
//
//   bits 5, 6, cb 2 -- upstream HAS NO GEMV EITHER. Its try-launch opens with
//     `if (K < 2 || K > 4) return false;` (`exl3_gemv.cu:107-121`) and its
//     `SEL_GRID` list is `(4,0) (4,1) (4,2) (2,1) (2,2) (3,1) (3,2)`. Falling
//     to the regular shape table at these widths is upstream's arrangement.
//   bits 2, any cb -- upstream HAS it and this tree does not. `LOADS` halves to
//     `WNT / 2` and one loaded word carries TWO tiles (`:152`, `:302-310`),
//     which is a third geometry, and no 2-bit EXL3 artifact has reached this
//     tree to gate it against. Owed, and named so it is not rediscovered.
//
// bits 3, cb 2 IS AN INSTANTIATION, and it landed with slice A of
// QUANT-EXL3-PERF (#2570). It is 137 of those 409 modules -- every MLP
// projection quantized at the low end of that artifact's 3.5 bpw average -- and
// until that line the GEMV's ONLY arm was `(3, 1)`, of which that checkpoint
// contains ZERO tensors. The fast path was unreachable on the whole model the
// benchmark is about, one module at a time and silently, because a declined
// GEMV falls through instead of refusing. Nothing about the kernel's geometry
// depends on `cb`: `LSTRIDE`, `TWORDS`, `FOLD`, `PF` and `LOADS` are functions
// of `bits`, `CFG` and `MMODE` alone, and the single decode site is
// `dq8_regs_3bits<cb>`, which has carried all three codebooks since
// QUANT-EXL3-MUL1 slice A.
//
// bits 4, cb 2 WAS A KERNEL PORT, and it is now here (QUANT-EXL3-PERF slice B,
// #2570). It is 270 of the 409 modules, the LARGEST single population in that
// artifact and the one #2570 leads with. What it cost, against upstream: a
// per-width `LSTRIDE` (`exl3_gemv_kernel.cuh:153`), the 24-lane load and store
// guards narrowed to bits 3 (`:228-231`, `:270`), the per-lane funnel constants
// made bits-3-only (`:199-216`), and `dq8_regs_4bits` (`:86-100`), which is a
// different extractor and not a parameter of the 3-bit one.
//
// (4, 0) and (4, 1) are NOT instantiated even though `Exl3GemvHardEligible`
// admits `(4, 0)` -- upstream's `K != 4 && cb == 0` refusal exempts this width
// alone -- and even though nothing in the kernel above depends on `cb`, so it
// SHOULD compile for both. Nothing builds either, so that last clause is an
// inference and is written as one. They are 16 kernels each
// in a translation unit the fat build compiles for ten architectures, and no
// artifact in this tree carries a 4-bit tensor at either codebook: the #2495
// checkpoint's 270 are all `mul1`. Adding them would repeat the `(3, 0)`
// mistake this comment block already records. Owed with the artifact that needs
// them.
//
// AN INSTANTIATION IS NECESSARY AND IT IS NOT SUFFICIENT, which is the part
// that is easy to bank and wrong. `Exl3GemvSelectConfig` returns -1 to DECLINE,
// and on Blackwell the only branch that can admit ANY of this checkpoint's
// shapes -- at either width -- is `size_n / 32 <= narrow_coresident`, an
// OCCUPANCY QUERY. For bits 3, `if (K == 3) return -1;` closes the door right
// after it. For bits 4 the door stays open one line longer, and it leads
// nowhere here: `size_n >= 8192 && size_k <= 4096` is the wide-config band and
// this artifact's smallest 4-bit `k` is 5120. So the widths differ in what the
// envelope SAYS and not in what it DOES on this checkpoint, and both rest on
// the same occupancy term. What bits 4 does buy is lower thresholds:
// `narrow_coresident >= 32` admits 34 modules, `>= 160` a further 85 and
// `>= 192` a further 48, where the bits-3 shapes need 160 and 544.
// `tests/vt/test_exl3_gemv.cpp` pins every one of those thresholds, so whether
// an arm actually RUNS on a given device is a lookup and not a guess, and a
// zero end-to-end effect with the arm declined stays distinguishable from a
// zero with the arm taken.
//
// The cost of declining is NOT asserted here: it is the `m <= 8` fast path
// only, and quantifying it needs the checkpoint on a device. `## Owed` of
// `.agents/specs/quant-exl3-perf.md` carries what is measured and what is not.
//
// A null here is a DECLINE, and `TryGemv` turns it into a fall-through rather
// than a failure.
constexpr bool Exl3GemvArmInstantiated(int bits, int cb) {
  return (bits == 3 && (cb == 1 || cb == 2)) || (bits == 4 && cb == 2);
}

const void* GemvKernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem) {
  if (bits == 3 && cb == 1) return GemvKernelForArm<3, 1>(c_fp32, mmode, cfg, smem);
  if (bits == 3 && cb == 2) return GemvKernelForArm<3, 2>(c_fp32, mmode, cfg, smem);
  if (bits == 4 && cb == 2) return GemvKernelForArm<4, 2>(c_fp32, mmode, cfg, smem);
  return nullptr;
}

int GemvOccupancy(int device, const void* kernel, int block_dim) {
  static std::mutex mtx;
  static std::map<std::pair<int, const void*>, int> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find({device, kernel});
  if (it != cache.end()) return it->second;
  int blocks_per_sm = 0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor exl3_gemv");
  cache[{device, kernel}] = blocks_per_sm;
  return blocks_per_sm;
}

bool Exl3GemvTryLaunch(Queue& q, int device, Exl3Cc cc, int num_sms, void** kernel_args,
                       int size_m, int size_k, int size_n, const Exl3GemmArgs& args,
                       bool c_fp32) {
  static_assert(kExl3GemvMaxMDev == kExl3GemvMaxM,
                "the device and host copies of EXL3_GEMV_MAX_M must agree");
  // exl3_gemv.cu:108-116: the free integer tests first, then the env read.
  // A HALF-INTEGER rate (K+0.5, mul1) declines first of all: its bits = KA
  // would pass the integer tests below as e.g. (3, 2) and decode the frac
  // bitstream through the integer-3 GEMV — silently wrong. No CUDA frac arm
  // exists (BACKEND-ROCM, spec `## Owed`).
  if (args.half) return false;
  if (args.force_gemv == 0) return false;
  if (!Exl3GemvArmInstantiated(args.bits, args.codebook)) return false;
  if (!Exl3GemvHardEligible(size_m, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true, /*half=*/args.half))
    return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  const int mmode = size_m == 1 ? 0 : 1;
  const bool smem = Exl3GemvSmemMode() == 1;

  const void* narrow = GemvKernel(args.bits, args.codebook, c_fp32, mmode, 0, smem);
  if (narrow == nullptr) return false;
  const int narrow_coresident = GemvOccupancy(device, narrow, 512) * num_sms;
  const int cfg = Exl3GemvSelectConfig(cc, size_m, size_k, size_n, args.bits, args.codebook, mode,
                                       narrow_coresident);
  if (cfg < 0) return false;
  const void* kernel =
      cfg == 0 ? narrow : GemvKernel(args.bits, args.codebook, c_fp32, mmode, cfg, smem);
  if (kernel == nullptr) return false;

  const int block_dim = cfg == 0 ? 512 : 256;
  const int cols = cfg == 0 ? 32 : 64;
  const int max_blocks = GemvOccupancy(device, kernel, block_dim) * num_sms;
  const int grid = EXL3_MIN(size_n / cols, max_blocks);
  if (grid < 1) return false;

  Check(cudaLaunchCooperativeKernel(kernel, dim3(static_cast<unsigned>(grid)),
                                    dim3(static_cast<unsigned>(block_dim)), kernel_args, 0,
                                    AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_gemv");
  Check(cudaGetLastError(), "exl3_gemv launch");
  return true;
}

void Exl3GemmKernelCuda(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                        const Tensor& suh, const Tensor& svh, Tensor& a_had,
                        const Exl3GemmArgs& args) {
  // The HALF-INTEGER rates (K+0.5, mul1) have NO CUDA arm — every decode arm
  // below is `template <int BITS>` over integer widths, and a frac tensor
  // carried as bits = KA would name an existing (KA, cb) instantiation and
  // decode garbage. It refuses by name, consistent with the repo's
  // unimplemented-arm rule; the ROCm dot arm and Exl3GemmK fallback serve
  // these tensors (BACKEND-ROCM frac rates).
  VT_CHECK(!args.half,
           "vt cuda exl3: half-integer rates (K+0.5, mul1) have no CUDA arm — the frac "
           "decode (dq8_half, frac.cu) is ported on ROCm only. This is an unimplemented "
           "arm, not an invalid tensor (BACKEND-ROCM, ## Owed).");
  if (!Exl3ArmInstantiated(args.bits, args.codebook)) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_gemm is instantiated for (bits, codebook) in "
        "{(3, 0), (3, 1), (6, 0), (3, 2), (4, 2), (5, 2), (6, 2)} only; got bits " +
        std::to_string(args.bits) + " codebook " + std::to_string(args.codebook) +
        ". Those seven are the body and head of the stock exl3 artifacts (cb 0), the "
        "SparkInfer DeepSeek-V4 one (cb 1), and the four mul1 widths of "
        "Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw and its DFlash2 draft (cb 2): 137 modules at "
        "3 bpw, 270 at 4, one at 5 and one at 6. Upstream's own "
        "table is dense over bits 1..8 by codebook 0..2 because it splits per (K, cb) "
        "into one translation unit each; widening this one further is that split, and "
        "belongs with the artifact that needs it (QUANT-EXL3-MUL1, #2495, #2574). The "
        "CPU arm decodes every width and serves them on a CPU queue meanwhile.");
  }
  // NOT const: cudaLaunchCooperativeKernel takes `void**`, so each argument has
  // to be a modifiable lvalue whose address can be taken as `void*`.
  int size_m = static_cast<int>(a.shape[0]);
  int size_k = static_cast<int>(a.shape[1]);
  int size_n = static_cast<int>(c.shape[1]);
  if (size_m == 0 || size_k == 0 || size_n == 0) return;

  int device = 0;
  Check(cudaGetDevice(&device), "cudaGetDevice");
  const DeviceCaps& caps = GetDeviceCaps(device);
  const Exl3Cc cc = Exl3CcFromSm(caps.sm_major, caps.sm_minor);

  const half* a_ptr0 = a.Ptr<half>();
  const uint16_t* b_ptr0 = trellis.Ptr<uint16_t>();
  void* c_ptr0 = c.data;
  int* locks0 = DeviceLocks(device);
  const half* suh_ptr0 = suh.Ptr<half>();
  half* a_had_ptr0 = a_had.Ptr<half>();
  const half* svh_ptr0 = svh.Ptr<half>();
  void* gemv_args[] = {static_cast<void*>(&a_ptr0),   static_cast<void*>(&b_ptr0),
                       static_cast<void*>(&c_ptr0),   static_cast<void*>(&size_m),
                       static_cast<void*>(&size_k),   static_cast<void*>(&size_n),
                       static_cast<void*>(&locks0),   static_cast<void*>(&suh_ptr0),
                       static_cast<void*>(&a_had_ptr0), static_cast<void*>(&svh_ptr0)};
  // MODEL-DSV4-EXL3 W2c. Upstream tries the GEMV only when the caller forced
  // neither a shape nor an SM count (`exl3_gemm.cu:222`), because forcing either
  // is a request for a specific regular-kernel launch; `force_gemv` is the
  // separate lever that asks for THIS arm.
  if (args.force_shape_idx <= 0 || args.force_gemv > 0) {
    if (Exl3GemvTryLaunch(q, device, cc, caps.multiprocessor_count > 0
                                            ? caps.multiprocessor_count
                                            : 1,
                          gemv_args, size_m, size_k, size_n, args, c.dtype == DType::kF32))
      return;
  }
  if (args.force_gemv > 0) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_gemm was asked to force the m<=8 GEMV arm (force_gemv=1) but the "
        "call is not hard-eligible for it: m=" +
        std::to_string(size_m) + " k=" + std::to_string(size_k) + " n=" +
        std::to_string(size_n) + " bits=" + std::to_string(args.bits) +
        ". Upstream's own direct entry point refuses the same way (exl3_gemv.cu:238).");
  }

  int shape_idx = args.force_shape_idx > 0
                      ? args.force_shape_idx
                      : Exl3SelectGemmShape(cc, size_m, size_k, size_n, args.bits, false);
  if (shape_idx <= 0 || !Exl3GemmShapeCompat(shape_idx, size_k, size_n)) {
    throw std::runtime_error("vt cuda exl3: no compatible exl3_gemm kernel shape for k=" +
                             std::to_string(size_k) + " n=" + std::to_string(size_n) +
                             " (selected shape " + std::to_string(shape_idx) + ")");
  }
  const Exl3GemmShape shape = Exl3GemmShapeParams(shape_idx);
  const bool c_fp32 = c.dtype == DType::kF32;
  const void* kernel = c_fp32 ? GemmKernelForShape<true>(args.bits, args.codebook, shape_idx)
                              : GemmKernelForShape<false>(args.bits, args.codebook, shape_idx);
  if (kernel == nullptr)
    throw std::runtime_error("vt cuda exl3: exl3_gemm shape " + std::to_string(shape_idx) +
                             " has no instantiation");

  int num_sms = Exl3GemmNumSms(shape_idx, size_k, size_n,
                               caps.multiprocessor_count > 0 ? caps.multiprocessor_count : 1);
  EnsureSmemOptIn(device, kernel);

  const half* a_ptr = a.Ptr<half>();
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  void* c_ptr = c.data;
  int* locks = DeviceLocks(device);
  const half* suh_ptr = suh.Ptr<half>();
  half* a_had_ptr = a_had.Ptr<half>();
  const half* svh_ptr = svh.Ptr<half>();
  void* kernel_args[] = {
      static_cast<void*>(&a_ptr),  static_cast<void*>(&b_ptr),     static_cast<void*>(&c_ptr),
      static_cast<void*>(&size_m), static_cast<void*>(&size_k),    static_cast<void*>(&size_n),
      static_cast<void*>(&locks),  static_cast<void*>(&suh_ptr),   static_cast<void*>(&a_had_ptr),
      static_cast<void*>(&svh_ptr)};
  Check(cudaLaunchCooperativeKernel(kernel, dim3(static_cast<unsigned>(num_sms)),
                                    dim3(static_cast<unsigned>(shape.block_dim)), kernel_args,
                                    kSmemMax, AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_gemm");
  Check(cudaGetLastError(), "exl3_gemm launch");
}

// ── the reconstruct kernels (reconstruct.cu:11-308) ─────────────────────────
//
// The EXL3 prefill path for M > 144 (AUTO_RECONSTRUCT_THRESHOLD, exl3.py:10).
// Dequantizes the trellis to a full fp16 weight matrix, then runs cuBLAS fp16
// GEMM. Two variants:
//
//   reconstruct_kernel     unfused: dequantizes a 16x128 tile to row-major fp16.
//                          The caller applies the input/output Hadamards as
//                          standalone launches (suh on input, svh on output).
//
//   reconstruct_had_kernel fused: dequantizes + applies BOTH Hadamards + sign
//                          vectors inside the kernel, emitting original-basis
//                          weights. The GEMM then runs on the raw input and no
//                          standalone Hadamard launches are needed (~14% of
//                          long-chunk prefill GPU time saved, exl3.py:171-175).
//
// Both kernels share `dq_dispatch` (line 451) and `shuffle_had_h2x32`
// (line 499) with the cooperative GEMM kernel. The arm set is the same seven
// pairs as `Exl3ArmInstantiated` above, because `dq_dispatch` static_asserts
// bits 3..6 and the codebook set matches the GEMM's.

// reconstruct.cu:11-84. Unfused reconstruct: dequantize one 16x128 tile and
// store in row-major fp16.
template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_kernel(half* __restrict__ g_unpacked,
                        const uint16_t* __restrict__ g_packed,
                        int packed_blocks_n, int packed_n_offset) {
  constexpr int packed_size = 256 * K / 16;  // in uint16s

  int t = threadIdx.x;
  int lane_id = t % 32;
  int warp_id = t / 32;
  int k = blockIdx.y;
  int n = blockIdx.x * 8;
  int tiles_n = gridDim.x;
  int out_blocks_n = tiles_n * 8;

  // Load packed 16*128 tile
  __shared__ uint32_t s_packed[8][packed_size / 2];
  g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
  if (t < packed_size)
    ((int4*)s_packed)[t] = ((int4*)g_packed)[t];
  __syncthreads();

  // Dequant
  FragB frag[2];
  dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

  // Shuffle from tensor core layout to row major tile
  __shared__ half2 tile[16][8][8];

  half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
  half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
  half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
  half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

  if (!(lane_id & 4)) {
    half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
    half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
    half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
    half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
    half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
    half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
    half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
    half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
    int r0 = (lane_id % 4) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c0 = lane_id / 8;
    int c1 = c0 + 4;
    tile[r0][warp_id][c0] = m0;
    tile[r1][warp_id][c0] = m1;
    tile[r2][warp_id][c0] = m2;
    tile[r3][warp_id][c0] = m3;
    tile[r0][warp_id][c1] = m4;
    tile[r1][warp_id][c1] = m5;
    tile[r2][warp_id][c1] = m6;
    tile[r3][warp_id][c1] = m7;
  }
  __syncthreads();

  // Store unpacked tile
  int r = t / 16;
  int c = t % 16;
  int4* tile_int4 = (reinterpret_cast<int4*>(tile));
  int4* out_int4 = ((int4*)g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
  *out_int4 = tile_int4[t];
}

// reconstruct.cu:147-308. Fused reconstruct + both-side Hadamard: emits
// W = diag(suh) . H128 . W_hat . H128 . diag(svh) (per 128x128 tile, 1/sqrt(128)
// per side), i.e. the ORIGINAL-basis weights, so the GEMM runs on raw input.
//
// The 128x128 tile lives in shared memory with an XOR swizzle on the half4
// chunk index (chunk ^= (row >> 2) & 31) making all phases bank-conflict-free
// except the 16-lane dequant scatter; both transforms reuse the natural-order
// Sylvester butterfly (symmetric, so column-then-row order is H W H).
template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_had_kernel(half* __restrict__ g_unpacked,
                            const uint16_t* __restrict__ g_packed,
                            const half* __restrict__ suh,
                            const half* __restrict__ svh,
                            int packed_blocks_n, int packed_n_offset) {
  constexpr int packed_size = 256 * K / 16;
  constexpr float r_scale = 0.08838834764831845f;

  int t = threadIdx.x;
  int lane_id = t % 32;
  int warp_id = t / 32;
  int kb = blockIdx.y;
  int nb = blockIdx.x;
  int n = nb * 8;
  int row_len = gridDim.x * 128;

  __shared__ uint32_t s_packed[8][8][packed_size / 2];
  __shared__ half2 stile[128 * 64];

  auto tix = [&](int R, int q, int p) {
    return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
  };

  constexpr int j_int4 = packed_size / 8;
  for (int u = t; u < 8 * 8 * j_int4; u += 256) {
    int j = u / (8 * j_int4);
    int r = u % (8 * j_int4);
    const uint16_t* gp = g_packed +
        ((size_t)((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
    ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
  }
  __syncthreads();

  for (int jj = 0; jj < 8 * 8 / (256 / 32); ++jj) {
    int j = (warp_id / 8) * (8 / (256 / 256)) + jj;
    int wn = warp_id % 8;
    FragB frag[2];
    dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4)) {
      half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
      half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
      half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
      half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
      half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
      half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
      half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
      half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
      int r0 = j * 16 + (lane_id % 4) * 2;
      int r1 = r0 + 1;
      int r2 = r0 + 8;
      int r3 = r0 + 9;
      int c0 = lane_id / 8;
      int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
      int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
      stile[tix(r0, q0, p0)] = m0;
      stile[tix(r1, q0, p0)] = m1;
      stile[tix(r2, q0, p0)] = m2;
      stile[tix(r3, q0, p0)] = m3;
      stile[tix(r0, q1, p1)] = m4;
      stile[tix(r1, q1, p1)] = m5;
      stile[tix(r2, q1, p1)] = m6;
      stile[tix(r3, q1, p1)] = m7;
    }
  }
  __syncthreads();

  const half2 rs2 = __float2half2_rn(r_scale);
  constexpr int CHUNKS_PW = 32 / (256 / 32);
#pragma unroll
  for (int qq = 0; qq < CHUNKS_PW; ++qq) {
    int q = warp_id * CHUNKS_PW + qq;
    int qs = q ^ lane_id;
    half2 a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v = *((const half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2));
      a[i] = v.x;
      b[i] = v.y;
    }
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      half2* v = x == 0 ? a : b;
      half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
      half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
      v[0] = __hmul2(__hadd2(s0, s1), rs2);
      v[1] = __hmul2(__hadd2(d0, d1), rs2);
      v[2] = __hmul2(__hsub2(s0, s1), rs2);
      v[3] = __hmul2(__hsub2(d0, d1), rs2);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v[i] = shuffle_had_h2x32(v[i], lane_id);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v;
      v.x = a[i];
      v.y = b[i];
      *((half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
    }
  }
  __syncthreads();

  // Row transform fused with the store: after the lane butterfly, lane l holds
  // the FINAL columns 4l..4l+3 of row R. Apply the sign scales in registers and
  // write the coalesced 256-byte row directly.
  constexpr int ROWS_PW = 128 / (256 / 32);
  const half4 sv4 = ((const half4*)svh)[nb * 32 + lane_id];
#pragma unroll
  for (int rr = 0; rr < ROWS_PW; ++rr) {
    int R = warp_id * ROWS_PW + rr;
    int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
    half2 v01 = stile[base];
    half2 v23 = stile[base + 1];
    float v0 = __low2float(v01), v1 = __high2float(v01);
    float v2 = __low2float(v23), v3 = __high2float(v23);
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
    half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
    h01 = shuffle_had_h2x32(h01, lane_id);
    h23 = shuffle_had_h2x32(h23, lane_id);
    half2 su2 = __half2half2(suh[kb * 128 + R]);
    half4 v;
    v.x = __hmul2(__hmul2(h01, su2), sv4.x);
    v.y = __hmul2(__hmul2(h23, su2), sv4.y);
    *((half4*)(g_unpacked + (size_t)(kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
  }
}

// ── the reconstruct launchers (reconstruct.cu:98-144, 324-373) ───────────────
//
// Same arm set as Exl3ArmInstantiated (the 7 pairs dq_dispatch admits), dispatched
// with a switch rather than the reference's dense 24-entry array, matching the
// GEMM's own sparse dispatch (GemmKernelForArm, line 2137).

#define VT_RECONSTRUCT_DISPATCH(K, CB)                                     \
  reconstruct_kernel<K, CB><<<grid_dim, block_dim, 0, stream>>>(           \
      w_ptr, trellis_ptr_u16, packed_blocks_n, packed_n_offset);           \
  return

void ReconstructSlice(Queue& q, Tensor& w, const Tensor& trellis, int bits,
                      int codebook, int64_t n_offset) {
  const int rows = static_cast<int>(trellis.shape[0]);      // k/16
  const int packed_blocks_n = static_cast<int>(trellis.shape[1]);  // n/16
  if (rows == 0 || packed_blocks_n == 0) return;
  const int cols = static_cast<int>(w.shape[1]) / 16;      // n_slice/16
  const int packed_n_offset = static_cast<int>(n_offset / 16);
  const dim3 block_dim(256);
  const dim3 grid_dim(static_cast<unsigned>(cols / 8), static_cast<unsigned>(rows));
  cudaStream_t stream = AsStream(q);
  half* w_ptr = w.Ptr<half>();
  const uint16_t* trellis_ptr_u16 = trellis.Ptr<uint16_t>();

  if (bits == 3 && codebook == 0) { VT_RECONSTRUCT_DISPATCH(3, 0); }
  if (bits == 3 && codebook == 1) { VT_RECONSTRUCT_DISPATCH(3, 1); }
  if (bits == 6 && codebook == 0) { VT_RECONSTRUCT_DISPATCH(6, 0); }
  if (bits == 3 && codebook == 2) { VT_RECONSTRUCT_DISPATCH(3, 2); }
  if (bits == 4 && codebook == 2) { VT_RECONSTRUCT_DISPATCH(4, 2); }
  if (bits == 5 && codebook == 2) { VT_RECONSTRUCT_DISPATCH(5, 2); }
  if (bits == 6 && codebook == 2) { VT_RECONSTRUCT_DISPATCH(6, 2); }
  throw std::runtime_error(
      "vt cuda exl3: reconstruct is instantiated for (bits, codebook) in "
      "{(3, 0), (3, 1), (6, 0), (3, 2), (4, 2), (5, 2), (6, 2)} only; got bits " +
      std::to_string(bits) + " codebook " + std::to_string(codebook) +
      ". The CPU arm decodes every width over all three codebooks.");
}
#undef VT_RECONSTRUCT_DISPATCH

#define VT_RECONSTRUCT_HAD_DISPATCH(K, CB)                               \
  reconstruct_had_kernel<K, CB><<<grid_dim, block_dim, 0, stream>>>(     \
      w_ptr, trellis_ptr_u16, suh_ptr, svh_ptr, packed_blocks_n,         \
      packed_n_offset);                                                  \
  return

void ReconstructHadSlice(Queue& q, Tensor& w, const Tensor& trellis,
                         const Tensor& suh, const Tensor& svh, int bits,
                         int codebook, int64_t n_offset) {
  const int k_dim = static_cast<int>(w.shape[0]);       // k (full)
  const int n_slice = static_cast<int>(w.shape[1]);     // n_slice
  if (k_dim == 0 || n_slice == 0) return;
  const int packed_blocks_n = static_cast<int>(trellis.shape[1]);  // n/16 (full)
  const int packed_n_offset = static_cast<int>(n_offset / 16);
  const dim3 block_dim(256);
  const dim3 grid_dim(static_cast<unsigned>(n_slice / 128),
                      static_cast<unsigned>(k_dim / 128));
  cudaStream_t stream = AsStream(q);
  half* w_ptr = w.Ptr<half>();
  const uint16_t* trellis_ptr_u16 = trellis.Ptr<uint16_t>();
  const half* suh_ptr = suh.Ptr<half>();
  const half* svh_ptr = svh.Ptr<half>();

  if (bits == 3 && codebook == 0) { VT_RECONSTRUCT_HAD_DISPATCH(3, 0); }
  if (bits == 3 && codebook == 1) { VT_RECONSTRUCT_HAD_DISPATCH(3, 1); }
  if (bits == 6 && codebook == 0) { VT_RECONSTRUCT_HAD_DISPATCH(6, 0); }
  if (bits == 3 && codebook == 2) { VT_RECONSTRUCT_HAD_DISPATCH(3, 2); }
  if (bits == 4 && codebook == 2) { VT_RECONSTRUCT_HAD_DISPATCH(4, 2); }
  if (bits == 5 && codebook == 2) { VT_RECONSTRUCT_HAD_DISPATCH(5, 2); }
  if (bits == 6 && codebook == 2) { VT_RECONSTRUCT_HAD_DISPATCH(6, 2); }
  throw std::runtime_error(
      "vt cuda exl3: reconstruct_had is instantiated for (bits, codebook) in "
      "{(3, 0), (3, 1), (6, 0), (3, 2), (4, 2), (5, 2), (6, 2)} only; got bits " +
      std::to_string(bits) + " codebook " + std::to_string(codebook) +
      ". The CPU arm decodes every width over all three codebooks.");
}
#undef VT_RECONSTRUCT_HAD_DISPATCH

// ── the reconstruct+cuBLAS launcher (exl3.py:161-218) ────────────────────────
//
// Mirrors exllamav3's `reconstruct_hgemm` dispatch: dequantize the trellis to
// fp16, then cuBLAS fp16 GEMM. Two sub-paths:
//   FUSED (M >= 1024, both dims 128-divisible): reconstruct_had_slice emits
//     original-basis weights (both Hadamards + sign vectors folded in), so the
//     GEMM runs on raw `a` with no standalone Hadamard launches.
//   UNFUSED (M < 1024): reconstruct (plain dequant) + input Hadamard (suh) on
//     `a` + cuBLAS + output Hadamard (svh) on `c` in-place.
//
// For N > 32768 (MAX_RECONSTRUCT_SLICE_N), the weight is reconstructed and
// GEMMed in 32768-column slices (exl3.py:199-211).
constexpr int kMaxReconstructSliceN = 32768;

// ── the persistent reconstruct scratch (QUANT-EXL3 W7) ───────────────────────
//
// `.agents/specs/quant-exl3-recon-scratch.md`. ONE grow-only fp16 buffer per
// (device, stream), owned here and never by the DevicePool. #3150 drew it as a
// pool `DBuf` per call, and a pool block a CUDA-graph step demands is PINNED
// for that graph's lifetime (`device_pool.h` PinForGraph), so every lazily
// captured verify width held one block of every [k, min(n, 32768)] class: about
// 786 MiB per slot on Qwen3.8-27B EXL3, and a host OOM on GB10 at c = 32
// (ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ). exllamav3's `torch.empty` in
// `reconstruct_hgemm` (exl3.py:161-217) is one block shared by every call and
// every graph under torch's caching allocator; this restores that.
//
// SHARING IS SAFE because the contents are dead once the GEMM that reads them is
// enqueued, and every consumer of one stream runs in that stream's order. The
// premise is that one host thread at a time enqueues EXL3 linears on a stream;
// the engine busy loop is the only such thread today (spec §6).
//
// THE HOUSE RULES for a per-stream grow-only scratch, and one refinement:
//   * one lock across the capacity check and the publish
//     (vt/grow_only_stream_scratch.h, rule 1);
//   * growth is FORBIDDEN while the stream is capturing, by name, as
//     cuda_dropin.cu's workspace and cuda_mla_attn.cu's attn-logits scratch
//     refuse it: a `cudaMallocAsync` inside capture is a graph-owned allocation
//     whose lifetime is the graph's execution;
//   * on growth the old block is RETIRED when a capture may have baked it
//     (graph_safe_scratch.h). The refinement: a block that was never handed out
//     while its stream was capturing cannot be referenced by any graph, so it is
//     freed on the stream instead. The buffer's size depends on the weight and
//     not on m, so it reaches the model's largest class within the first eager
//     reconstruct step, and retiring every intermediate class of that step would
//     keep them all resident for the process for no graph at all.
struct Exl3ReconScratch {
  void* buf = nullptr;
  size_t bytes = 0;
  bool exposed_to_capture = false;
};

std::mutex& Exl3ReconScratchMutex() {
  static std::mutex mu;
  return mu;
}

std::map<std::pair<int, cudaStream_t>, Exl3ReconScratch>& Exl3ReconScratchTable() {
  static std::map<std::pair<int, cudaStream_t>, Exl3ReconScratch> table;
  return table;
}

// A contiguous f16 [k, w_cols] view of this queue's persistent scratch.
Tensor Exl3PersistentReconScratch(const Queue& q, int64_t k, int64_t w_cols) {
  const cudaStream_t stream = AsStream(q);
  const size_t need = static_cast<size_t>(k) * static_cast<size_t>(w_cols) * sizeof(uint16_t);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  Check(cudaStreamIsCapturing(stream, &capture), "exl3 reconstruct scratch capture-status query");
  const bool capturing = capture != cudaStreamCaptureStatusNone;

  std::lock_guard<std::mutex> lock(Exl3ReconScratchMutex());
  Exl3ReconScratch& sc = Exl3ReconScratchTable()[{q.device.index, stream}];
  if (need > sc.bytes) {
    if (capturing) {
      throw std::runtime_error(
          "vt cuda exl3: exl3 reconstruct scratch growth is forbidden during CUDA graph "
          "capture (have " + std::to_string(sc.bytes) + " bytes, need " +
          std::to_string(need) + " for [" + std::to_string(k) + ", " +
          std::to_string(w_cols) + "] f16); run this shape eagerly on the stream first");
    }
    void* grown = nullptr;
    Check(cudaMallocAsync(&grown, need, stream), "cudaMallocAsync(exl3 reconstruct scratch)");
    if (sc.buf != nullptr) {
      if (sc.exposed_to_capture) {
        RetireGraphScratch(sc.buf);
      } else {
        Check(cudaFreeAsync(sc.buf, stream), "cudaFreeAsync(exl3 reconstruct scratch)");
      }
    }
    sc.buf = grown;
    sc.bytes = need;
    sc.exposed_to_capture = false;
  }
  if (capturing) sc.exposed_to_capture = true;
  return Tensor::Contiguous(sc.buf, DType::kF16, q.device, {k, w_cols});
}

void Exl3ReconstructGemmKernelCuda(Queue& q, Tensor& c, const Tensor& a,
                                    const Tensor& trellis, const Tensor& suh,
                                    const Tensor& svh, Tensor& a_had,
                                    Tensor& w_scratch, const Exl3GemmArgs& args) {
  // The shared seam (ops.cpp) already refuses args.half for reconstruct; this
  // names the CUDA side of the same missing arm so a direct route cannot slip
  // a frac tensor into the integer-KA decode.
  VT_CHECK(!args.half,
           "vt cuda exl3: half-integer rates (K+0.5, mul1) have no CUDA reconstruct arm "
           "(BACKEND-ROCM, ## Owed).");
  if (!Exl3ArmInstantiated(args.bits, args.codebook)) {
    throw std::runtime_error(
        "vt cuda exl3: reconstruct+cuBLAS is instantiated for (bits, codebook) "
        "in {(3, 0), (3, 1), (6, 0), (3, 2), (4, 2), (5, 2), (6, 2)} only; got "
        "bits " + std::to_string(args.bits) + " codebook " +
        std::to_string(args.codebook) + ". The CPU arm decodes every width.");
  }
  const int size_m = static_cast<int>(a.shape[0]);
  const int size_k = static_cast<int>(a.shape[1]);
  const int size_n = static_cast<int>(c.shape[1]);
  if (size_m == 0 || size_k == 0 || size_n == 0) return;

  // QUANT-EXL3 W7: an EMPTY scratch selects this stream's persistent buffer
  // (vt::Exl3ReconstructGemm without w_scratch). Resolved before any launch, so
  // a growth refused under capture has enqueued nothing of this call.
  Tensor persistent_scratch;
  Tensor* scratch = &w_scratch;
  if (w_scratch.data == nullptr) {
    persistent_scratch = Exl3PersistentReconScratch(
        q, size_k, EXL3_MIN(size_n, kMaxReconstructSliceN));
    scratch = &persistent_scratch;
  }

  // exl3.py:176-184. Fused requires both dims 128-divisible (always true for
  // EXL3 tensors, which are Hadamard-transformed at quant time) and M >= 1024.
  const bool dims_128 = (size_k % 128 == 0) && (size_n % 128 == 0);
  const bool use_fused = dims_128 && size_m >= 1024;

  // exl3.py:186-190. Input Hadamard (unfused only).
  Tensor* xh = const_cast<Tensor*>(&a);
  if (!use_fused) {
    Exl3HadArgs had_args;
    had_args.pre_scale = const_cast<Tensor*>(&suh);
    had_args.scale = 1.0f;
    Exl3HadR128(q, a_had, a, had_args);
    xh = &a_had;
  }

  // exl3.py:192-211. Reconstruct + GEMM, optionally sliced along N.
  if (size_n <= kMaxReconstructSliceN) {
    // Single slice: reconstruct the full weight, then GEMM.
    if (use_fused) {
      ReconstructHadSlice(q, *scratch, trellis, suh, svh, args.bits, args.codebook, 0);
    } else {
      ReconstructSlice(q, *scratch, trellis, args.bits, args.codebook, 0);
    }
    CublasLtHgemm(q, c, *xh, *scratch);
  } else {
    // exl3.py:199-211. Reconstruct + GEMM in 32768-column slices, reusing
    // w_scratch.
    for (int n_start = 0; n_start < size_n; n_start += kMaxReconstructSliceN) {
      const int n_end = EXL3_MIN(n_start + kMaxReconstructSliceN, size_n);
      const int n_slice = n_end - n_start;
      // Slice w_scratch to [k, n_slice] for this iteration.
      Tensor w_slice = scratch->Slice(1, 0, n_slice);
      if (use_fused) {
        // svh must be pre-offset by the caller; slice it from n_start.
        Tensor svh_slice = svh.Slice(0, n_start, n_end);
        ReconstructHadSlice(q, w_slice, trellis, suh, svh_slice, args.bits,
                            args.codebook, n_start);
      } else {
        ReconstructSlice(q, w_slice, trellis, args.bits, args.codebook, n_start);
      }
      // GEMM into the corresponding column slice of c.
      Tensor c_slice = c.Slice(1, n_start, n_end);
      CublasLtHgemm(q, c_slice, *xh, w_slice);
    }
  }

  // exl3.py:213-214. Output Hadamard (unfused only), applied in-place on c.
  // Safe: each 128-element block is processed independently.
  if (!use_fused) {
    Exl3HadArgs had_args;
    had_args.post_scale = const_cast<Tensor*>(&svh);
    had_args.scale = 1.0f;
    Exl3HadR128(q, c, c, had_args);
  }
}

// THE INSTANTIATED FUSED-MoE ARMS, which are NOT the GEMM's arms above.
//
// Upstream's table is `exl3_moe_kernel_instances[]` (`exl3_moe.cu:22-33`),
// indexed at `:226` as `[4 * K + 2 * cb_idx + N_off]` with K in 0..8, cb_idx in
// {0, 1} and N_off in {0, 1}. TWO of upstream's own bounds are load-bearing here
// and the old refusal message stated neither:
//
//   CODEBOOK 0 IS NOT A MoE ARM UPSTREAM EITHER. `exl3_moe.cu:184` is
//   `TORCH_CHECK(gate_mcg != gate_mul1, "MoE kernel: Only mcg and mul1 codebooks
//   are supported")` and `:185` derives `cb_idx = gate_mul1 ? 1 : 0`. A 3INST
//   checkpoint is refused before the table is indexed. So the MoE's reachable
//   codebook set is at most {1, 2} -- it is NOT the GEMM's {0, 1, 2}, and a
//   message implying otherwise is wrong about upstream rather than merely terse.
//
//   K == 0 IS UPSTREAM'S RUNTIME-WIDTH INSTANCE, for a tower whose gate/up/down
//   widths differ (`exl3_moe_kernel.cuh:139-149`). This port takes K_gate, K_up
//   and K_down as kernel arguments and discards them (`(void)K;` in
//   `exl3_moe_kernel`'s `gemm_band`), so it serves only Kg == Ku == Kd. That is
//   owed separately from the width set. NOTHING ELSE CATCHES IT: the only width
//   check above this one is `deepseek_v4.cpp`'s, and it compares each projection
//   ACROSS EXPERTS (`xe.w1.bits == e0.w1.bits` and its two siblings) rather than
//   the three projections against each other, so a tower whose gate and down
//   widths differ passes it. The launcher refuses that below.
//
// WIDTHS 3..6 AND CODEBOOK 1, and each half of that has its own reason.
//
// The WIDTHS are four and not eight because `dq_dispatch` above static_asserts
// `bits == 3 || 4 || 5 || 6`, and that bound is arithmetic: the eight-window span
// `16 + bits*7` leaves the 64-bit funnel once the start shift is added at bits 5.
// Widths 1, 2, 7 and 8 are a DECODER question this file's GEMM shares, not a MoE
// question, and widening them is not this arm's slice.
//
// The CODEBOOK is one and not two because a cb-2 MoE kernel would be DEAD CODE.
// `vt::Exl3MoeMlp` has two production callers, both in `Exl3FusedMoePass`
// (`deepseek_v4.cpp`), and FIVE sites pin the codebook to 1 before the kernel is
// ever chosen: the loader refuses any marker but `mcg`, three separate
// `args.codebook = 1` assignments, and `ops.cpp`'s own
// `VT_CHECK(args.codebook == 1, ...)` in the SHARED seam -- which refuses for
// every backend, the CPU reference included. Widening here alone would
// instantiate a kernel nothing can reach.
// QUANT-EXL3-MUL1 slice G records it as owed, as a LOADER slice that ends in a
// kernel (#2756).
//
// THE WIDTHS, BY CONTRAST, ARE REACHABLE TODAY AND REFUSED. `deepseek_v4.cpp`
// reads the bit width per projection off the checkpoint (`e0.w1.bits` and its
// siblings) and assigns it into `Exl3MoeArgs` unchanged; nothing between the
// loader and this launcher clamps it. Before this arm set, an expert tower
// quantized at 4, 5 or 6 bits reached here with that width and was REFUSED, with
// no second path: `MoeBlock` calls `Exl3FusedMoePass` unguarded so the exception
// leaves the forward, this arm is default-ON, and the `VT_DSV4_EXL3_FUSED_MOE=0`
// rollback lands on `Exl3ArmInstantiated`, which has no `(4, 1)`, `(5, 1)` or
// `(6, 1)` either. Such a tower could not run on a CUDA queue by ANY path.
//
// SHARED MEMORY ADMITS EVERY ONE OF THEM, and the guard is the one already in
// `exl3_gemm_kernel_inner` rather than a new check. At the MoE shape
// (TILESIZE_M 16, TILESIZE_K 32, SH_STAGES 3) its `static_assert` resolves to
// `kSmemMax >= 3*(2*512 + 2*512*bits) + 4*4096` for MOE_TILESIZE_N 256, which is
// 28672 bytes at bits 3 and 37888 at bits 6 against a kSmemMax of 92160; the 128
// form is smaller. A width that did not fit would fail to COMPILE there rather
// than mis-stage silently.
constexpr bool Exl3MoeArmInstantiated(int bits, int cb) {
  return cb == 1 && (bits == 3 || bits == 4 || bits == 5 || bits == 6);
}

// Upstream's `[4 * K + 2 * cb_idx + N_off]` lookup, written as a switch because
// this table is sparse where upstream's is dense. A pair the predicate admits
// and this returns `nullptr` for is a BUG rather than a refusal, and the caller
// treats it as one: the two can never disagree silently.
template <int MOE_TILESIZE_N>
const void* MoeKernelN(int bits, int cb) {
  if (cb != 1) return nullptr;
  switch (bits) {
    case 3:
      return reinterpret_cast<const void*>(&exl3_moe_kernel<3, MOE_TILESIZE_N, 1>);
    case 4:
      return reinterpret_cast<const void*>(&exl3_moe_kernel<4, MOE_TILESIZE_N, 1>);
    case 5:
      return reinterpret_cast<const void*>(&exl3_moe_kernel<5, MOE_TILESIZE_N, 1>);
    case 6:
      return reinterpret_cast<const void*>(&exl3_moe_kernel<6, MOE_TILESIZE_N, 1>);
    default:
      return nullptr;
  }
}

const void* MoeKernel(int bits, int cb, bool n256) {
  return n256 ? MoeKernelN<256>(bits, cb) : MoeKernelN<128>(bits, cb);
}

// ── the fused MoE launcher (exl3_moe.cu:99-301) ──────────────────────────────
//
// Every validation upstream performs lives in `src/vt/ops.cpp`, shared with the
// CPU arm; what is left here is the launch geometry and the instantiation
// choice, which are the only device-specific parts.
void Exl3MoeMlpKernelCuda(Queue& q, Tensor& output_state, const Tensor& hidden_state,
                          const Exl3MoeExpertTables& tables, const Exl3MoeRouting& routing,
                          const Exl3MoeTemps& temps, const Exl3MoeArgs& args) {
  // `exl3_moe.cu:188-189` -- upstream is `int K = 0;` then
  // `if (K_gate == K_up && K_up == K_down) K = K_gate;`. A tower whose widths
  // disagree therefore leaves K at 0 and takes the runtime-width instance,
  // whose per-band `switch (K)` is `exl3_moe_kernel.cuh:139-149`. This port does
  // not carry that instance, so a disagreeing tower refuses here rather than
  // decoding one band with another's width.
  if (args.bits_gate != args.bits_up || args.bits_up != args.bits_down) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe carries ONE width for all three projections; got bits (" +
        std::to_string(args.bits_gate) + ", " + std::to_string(args.bits_up) + ", " +
        std::to_string(args.bits_down) +
        "). Upstream's K == 0 instance switches the width at RUN TIME "
        "(exl3_moe_kernel.cuh:139-149) and is NOT ported; QUANT-EXL3-MUL1 slice G records it "
        "as owed (#2756). The CPU arm carries a width per projection and serves this on a "
        "CPU queue.");
  }
  if (!Exl3MoeArmInstantiated(args.bits_gate, args.codebook)) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe is instantiated for bits in {3, 4, 5, 6} at codebook 1 (mcg); "
        "got bits " +
        std::to_string(args.bits_gate) + " codebook " + std::to_string(args.codebook) +
        ". Widths 1, 2, 7 and 8 have no device decode route at all -- `dq_dispatch` "
        "static_asserts 3..6 for the GEMM too. Codebook 2 (mul1) is an upstream MoE arm "
        "(exl3_moe.cu:184-185 admits mcg and mul1, and refuses 3INST) that this tree cannot "
        "REACH: the loader accepts only an `mcg` marker and three call sites pin "
        "`args.codebook` to 1, so the kernel would be dead code; QUANT-EXL3-MUL1 slice G "
        "records it as owed (#2756). The CPU arm decodes every width over all three "
        "codebooks and can serve them on a CPU queue.");
  }
  // NOT const: cudaLaunchCooperativeKernel takes `void**`, so every argument has
  // to be a modifiable lvalue whose address can be taken as `void*`.
  int bsz = static_cast<int>(hidden_state.shape[0]);
  int hidden_dim = static_cast<int>(hidden_state.shape[1]);
  int intermediate_dim = static_cast<int>(temps.intermediate_g->shape[2]);
  int num_experts = static_cast<int>(routing.expert_count->Numel() - 1);
  int max_tokens_per_expert = static_cast<int>(temps.state_g->shape[1]);
  int concurrency = static_cast<int>(temps.state_g->shape[0]);
  if (bsz == 0 || num_experts == 0) return;

  int device = 0;
  Check(cudaGetDevice(&device), "cudaGetDevice");
  const DeviceCaps& caps = GetDeviceCaps(device);
  const int num_sms = caps.multiprocessor_count > 0 ? caps.multiprocessor_count : 1;

  // exl3_moe.cu:210-222. Every block of the grid must be co-resident for the
  // group barriers, so groups * width <= num_sms. With a known active-expert
  // count, launch only as many groups as there are experts and widen them into
  // the freed SMs.
  if (static_cast<long long>(concurrency) * kMoeSmsPerExpert > num_sms) {
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe was given temp buffers for concurrency " +
        std::to_string(concurrency) + ", which needs " +
        std::to_string(concurrency * kMoeSmsPerExpert) + " co-resident SMs but this device has " +
        std::to_string(num_sms) +
        ". Size the buffers with vt::Exl3MoeMaxConcurrency(device_sms).");
  }
  int num_groups = EXL3_MIN(concurrency, kMoeMaxGroups);
  int group_size = kMoeSmsPerExpert;
  if (args.num_active > 0) {
    num_groups = EXL3_MIN(num_groups, args.num_active);
    group_size = EXL3_MIN(num_sms / num_groups, kMoeMaxSmsPerExpert);
  }
  if (num_groups < 1 || group_size < 1) return;

  // exl3_moe.cu:224-226. The N tile is 256 when both dims allow it.
  const bool n256 = (hidden_dim % 256 == 0) && (intermediate_dim % 256 == 0);
  const void* kernel = MoeKernel(args.bits_gate, args.codebook, n256);
  // A pair `Exl3MoeArmInstantiated` admitted and `MoeKernel` has no entry for is
  // the two tables disagreeing. That is a defect in this file, not a refusal of
  // the caller, and it must not reach `cudaLaunchCooperativeKernel` as a null.
  if (kernel == nullptr)
    throw std::runtime_error(
        "vt cuda exl3: exl3_moe arm (bits " + std::to_string(args.bits_gate) + ", codebook " +
        std::to_string(args.codebook) +
        ") passed Exl3MoeArmInstantiated but MoeKernel has no entry for it. The two tables "
        "disagree; this is a bug in cuda_exl3.cu, not a property of the call.");
  EnsureSmemOptIn(device, kernel);

  const half* hid = hidden_state.Ptr<half>();
  half* st_g = temps.state_g->Ptr<half>();
  half* st_u = temps.state_u->Ptr<half>();
  half* in_g = temps.intermediate_g->Ptr<half>();
  half* in_u = temps.intermediate_u->Ptr<half>();
  float* out = output_state.Ptr<float>();
  auto tbl16 = [](const Tensor* tt) {
    return reinterpret_cast<const uint16_t* const*>(tt->data);
  };
  auto tblh = [](const Tensor* tt) { return reinterpret_cast<const half* const*>(tt->data); };
  const uint16_t* const* g_tr = tbl16(tables.gate_trellis);
  const half* const* g_su = tblh(tables.gate_suh);
  const half* const* g_sv = tblh(tables.gate_svh);
  const uint16_t* const* u_tr = tbl16(tables.up_trellis);
  const half* const* u_su = tblh(tables.up_suh);
  const half* const* u_sv = tblh(tables.up_svh);
  const uint16_t* const* d_tr = tbl16(tables.down_trellis);
  const half* const* d_su = tblh(tables.down_suh);
  const half* const* d_sv = tblh(tables.down_svh);
  const int64_t* cnt = routing.expert_count->Ptr<int64_t>();
  const int64_t* tok = routing.token_sorted->Ptr<int64_t>();
  const half* wgt = routing.weight_sorted->Ptr<half>();
  int* locks = DeviceLocks(device);
  float act_limit = args.act_limit;
  int act_function = static_cast<int>(args.act);
  int k_gate = args.bits_gate, k_up = args.bits_up, k_down = args.bits_down;

  void* kernel_args[] = {
      static_cast<void*>(&hid),        static_cast<void*>(&st_g),
      static_cast<void*>(&st_u),       static_cast<void*>(&in_g),
      static_cast<void*>(&in_u),       static_cast<void*>(&out),
      static_cast<void*>(&g_tr),       static_cast<void*>(&g_su),
      static_cast<void*>(&g_sv),       static_cast<void*>(&u_tr),
      static_cast<void*>(&u_su),       static_cast<void*>(&u_sv),
      static_cast<void*>(&d_tr),       static_cast<void*>(&d_su),
      static_cast<void*>(&d_sv),       static_cast<void*>(&cnt),
      static_cast<void*>(&tok),        static_cast<void*>(&wgt),
      static_cast<void*>(&hidden_dim),  static_cast<void*>(&intermediate_dim),
      static_cast<void*>(&num_experts),
      static_cast<void*>(&max_tokens_per_expert),
      static_cast<void*>(&act_limit),   static_cast<void*>(&act_function),
      static_cast<void*>(&k_gate),      static_cast<void*>(&k_up),
      static_cast<void*>(&k_down),      static_cast<void*>(&locks)};

  const int block_dim = kBaseThreads * kMoeTilesizeK / 16;
  Check(cudaLaunchCooperativeKernel(
            kernel, dim3(static_cast<unsigned>(group_size), 1, static_cast<unsigned>(num_groups)),
            dim3(static_cast<unsigned>(block_dim)), kernel_args, kSmemMax, AsStream(q)),
        "cudaLaunchCooperativeKernel exl3_moe");
  Check(cudaGetLastError(), "exl3_moe launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kExl3HadR128, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3HadR128Fn>(&Exl3HadR128KernelCuda)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelCuda)));
    RegisterOp(OpId::kExl3MoeMlp, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Exl3MoeMlpFn>(&Exl3MoeMlpKernelCuda)));
    RegisterOp(OpId::kExl3ReconstructGemm, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Exl3ReconstructGemmFn>(&Exl3ReconstructGemmKernelCuda)));
  }
} registrar;

}  // namespace

// QUANT-EXL3 W7 review repair: the queue-teardown release, the house pattern of
// ReleaseFa2Scratch (cuda_flash_attn_fa2.cu) and ReleaseGdnTritonScratch.
// Without it a destroyed queue kept its [k, min(n, 32768)] block for the process,
// and a new stream whose handle value equals the old one inherited the entry.
void ReleaseExl3ReconScratch(int device, void* stream_handle) {
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  Exl3ReconScratch sc;
  {
    std::lock_guard<std::mutex> lock(Exl3ReconScratchMutex());
    auto& table = Exl3ReconScratchTable();
    const auto it = table.find({device, stream});
    if (it == table.end()) return;
    sc = it->second;
    table.erase(it);
  }
  if (sc.buf == nullptr) return;
  // Every launch this stream enqueued against the block completes before the
  // block leaves the entry's ownership.
  Check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(exl3 reconstruct scratch release)");
  if (sc.exposed_to_capture) {
    // A graph captured on this stream may outlive the queue and still hold the
    // pointer; the retire list keeps it valid for the process.
    RetireGraphScratch(sc.buf);
  } else {
    Check(cudaFreeAsync(sc.buf, stream), "cudaFreeAsync(exl3 reconstruct scratch release)");
  }
}

namespace testing {

size_t Exl3ReconScratchBytesForTesting(int device, void* stream_handle) {
  std::lock_guard<std::mutex> lock(Exl3ReconScratchMutex());
  const auto& table = Exl3ReconScratchTable();
  const auto it = table.find({device, static_cast<cudaStream_t>(stream_handle)});
  return it == table.end() ? 0 : it->second.bytes;
}

void* Exl3ReconScratchPtrForTesting(int device, void* stream_handle) {
  std::lock_guard<std::mutex> lock(Exl3ReconScratchMutex());
  const auto& table = Exl3ReconScratchTable();
  const auto it = table.find({device, static_cast<cudaStream_t>(stream_handle)});
  return it == table.end() ? nullptr : it->second.buf;
}

size_t Exl3ReconScratchLiveBytesForTesting() {
  std::lock_guard<std::mutex> lock(Exl3ReconScratchMutex());
  size_t total = 0;
  for (const auto& entry : Exl3ReconScratchTable()) total += entry.second.bytes;
  return total;
}

size_t RetiredGraphScratchCountForTesting() { return RetiredGraphScratchCount(); }

}  // namespace testing
}  // namespace vt::cuda
