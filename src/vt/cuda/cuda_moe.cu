// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA MoE ops (M0.8 Task 3): router top-k (softmax + greedy top-k +
// renormalize) and weighted combine. Correctness-grade — plain kernels
// matching the CPU reference math in src/vt/cpu/cpu_ops.cpp element for
// element; formulas from .agents/specs/moe-semantics.md (§3 router, §4/§6 combine).
//
// Upstream counterpart: layers/fused_moe/ (fused_topk / moe_align + grouped
// GEMM Triton/cutlass kernels — M2.2 replaces this correctness-grade path).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/cuda/moe_router_warp.h"
#include "vt/ops.h"
#include "vt/op_provider.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline void Store(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// ---------------------------------------------------------------------------
// moe_router_topk (moe-semantics.md §3): one BLOCK per token. The softmax is a
// block reduction (max-subtracted, f32, over all E experts). The greedy top-k
// is PARALLEL across the block (the default path), mirroring vLLM's
// topk_softmax_kernels.cu moeTopK/topkGating argmax reduction
// (csrc/libtorch_stable/moe/topk_softmax_kernels.cu:192-242, :494-537 @ vLLM
// e24d1b24 — "We want lower indices to win in every thread so we break ties
// this way"): each thread does a local strict-`>` argmax over its strided
// experts, then a shared-memory tree reduction resolves the block argmax with
// the identical lowest-index tie-break. The `Serial` template path keeps the
// original single-threaded greedy scan as the byte-exact parity reference; the
// two paths are byte-identical BY CONSTRUCTION — the softmax is untouched (so
// sp[] is bit-identical), the argmax is comparison-only over those same values
// with the same tie-break, and thread 0 accumulates the renorm denom in the
// same k order. Probs live in dynamic shared memory [E]; `red[kBlock]` /
// `redi[kBlock]` are the reduction scratch. lowest-index tie-break matches the
// CPU reference (cpu_ops.cpp MoeRouterTopKKernel) bit-for-bit.

template <typename Tin, bool Serial>
__global__ void MoeRouterTopKKernel(float* weights, int32_t* indices, const Tin* logits,
                                    int64_t e, int k, bool renormalize) {
  const int64_t row = blockIdx.x;
  const Tin* lrow = logits + row * e;
  extern __shared__ float sp[];  // [e] softmax probs
  __shared__ float red[kBlock];

  // Max over E (max-subtraction, topk_softmax_kernels.cu / cpu_ops.cpp §3).
  float m = -INFINITY;
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) m = fmaxf(m, Load(lrow, j));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  // exp(logit - max) into shared, block-summed for the denominator.
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
    const float ex = expf(Load(lrow, j) - mx);
    sp[j] = ex;
    acc += ex;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  __syncthreads();

  // Normalize with the sum>0 guard + NaN/Inf clamp (cpu_ops.cpp §3, .cu:136).
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
    float pj = sum > 0.0f ? sp[j] / sum : 0.0f;
    if (!isfinite(pj)) pj = 0.0f;
    sp[j] = pj;
  }
  __syncthreads();

  if constexpr (Serial) {
    // Reference path (retained for the byte-exact parity test): single-threaded
    // greedy argmax, strict `>` over ascending idx -> lowest expert index wins
    // ties. Probs are finite >= 0; masking a winner with -INFINITY excludes it.
    if (threadIdx.x == 0) {
      float denom = 0.0f;
      for (int j = 0; j < k; ++j) {
        int64_t best = -1;
        float best_v = -INFINITY;
        for (int64_t idx = 0; idx < e; ++idx) {
          if (sp[idx] > best_v) {
            best_v = sp[idx];
            best = idx;
          }
        }
        sp[best] = -INFINITY;  // exclude from subsequent rounds
        weights[row * k + j] = best_v;
        indices[row * k + j] = static_cast<int32_t>(best);
        denom += best_v;
      }
      if (renormalize) {
        if (!(denom > 0.0f)) denom = 1.0f;  // denom<=0 -> 1 guard (.cu:245-253)
        for (int j = 0; j < k; ++j) weights[row * k + j] /= denom;
      }
    }
  } else {
    // Parallel greedy top-k (default). Each round: every thread computes a
    // local argmax over its strided experts (ascending idx + strict `>` -> the
    // lowest index at the subset max, matching the serial ascending scan), then
    // a tree reduction resolves the block argmax with the same lower-index
    // tie-break. Only thread 0 mutates sp[]/writes results and accumulates the
    // renorm denom in k order, so the output is byte-identical to `Serial`.
    __shared__ int redi[kBlock];
    float denom = 0.0f;  // meaningful on thread 0 only
    for (int j = 0; j < k; ++j) {
      float lv = -INFINITY;
      int li = -1;
      for (int64_t idx = threadIdx.x; idx < e; idx += blockDim.x) {
        const float v = sp[idx];
        if (v > lv) {  // strict `>`, ascending stride -> lowest index at max
          lv = v;
          li = static_cast<int>(idx);
        }
      }
      // Warp-shuffle argmax, then ONE cross-warp pass. This replaces a
      // block-wide tree that cost log2(kBlock) __syncthreads PER ROUND: at
      // decode the grid is one block per token, so a k=8 top-k over 256 experts
      // spent ~64 barriers on a single SM and the measured kernel was 19.5 us
      // for work that is a few hundred comparisons (4.7% of the 35B decode
      // step). Barriers per round drop from log2(kBlock)+1 to 2.
      //
      // BYTE-IDENTICAL, and the reason is worth stating: this is an ARGMAX
      // reduction, not an arithmetic one. The comparison below is exactly the
      // one the tree used -- higher value wins, and on an exact tie the lower
      // expert index wins -- and argmax under a total order is associative and
      // commutative, so ANY reduction order yields the same (value, index).
      // The softmax max and sum reductions above are arithmetic and their tree
      // is deliberately left untouched, because changing THEIR order would
      // change the denominator in the last ulp.
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, lv, off);
        const int oi = __shfl_down_sync(0xffffffffu, li, off);
        if (ov > lv || (ov == lv && oi >= 0 && (li < 0 || oi < li))) {
          lv = ov;
          li = oi;
        }
      }
      constexpr int kWarps = kBlock / 32;
      if ((threadIdx.x & 31u) == 0u) {
        red[threadIdx.x >> 5] = lv;
        redi[threadIdx.x >> 5] = li;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        float best_v = red[0];
        int best = redi[0];
        for (int w = 1; w < kWarps; ++w) {
          const float ov = red[w];
          const int oi = redi[w];
          if (ov > best_v || (ov == best_v && oi >= 0 && (best < 0 || oi < best))) {
            best_v = ov;
            best = oi;
          }
        }
        if (best >= 0) sp[best] = -INFINITY;  // exclude from subsequent rounds
        weights[row * k + j] = best_v;
        indices[row * k + j] = static_cast<int32_t>(best);
        denom += best_v;
      }
      __syncthreads();  // sp[best]=-INF visible + red/redi reusable next round
    }
    if (threadIdx.x == 0 && renormalize) {
      if (!(denom > 0.0f)) denom = 1.0f;  // denom<=0 -> 1 guard (.cu:245-253)
      for (int j = 0; j < k; ++j) weights[row * k + j] /= denom;
    }
  }
}

// ─── Single-warp router top-k (KERNEL-MOE-ROUTER-WARP, issue #378) ─────────
// One WARP per token with the whole logit row in registers: no shared memory,
// no __syncthreads(), and ONE global read of the row instead of two. At E=256
// the block kernel above spends 37 barriers and 3 KiB of dynamic shared memory
// on a few hundred comparisons; this spends none.
//
// BYTE-IDENTICAL to MoeRouterTopKKernel<Tin,false>, and the reason is
// structural, not an appeal to associativity. In that kernel's block reduction
// the levels s = 128, 64, 32 are all multiples of the warp width, so `t` and
// `t + s` always share a lane id: those three levels never cross a lane, and
// they combine exactly red[L + 32q] by the standard halving recursion on q.
// Levels s = 16..1 live inside warp 0 and are what __shfl_down_sync
// reproduces. So `MoeRouterWarpExpert(lane, q) == lane + 32q` plus
// `MoeRouterWarpTree{Sum,Max}`'s halving tree performs the IDENTICAL float
// operations on the IDENTICAL operands in the IDENTICAL association. The full
// derivation, one VPT at a time, is in
// .agents/specs/moe-router-topk-single-warp.md §5; the reduction-order claim is
// EXECUTED (with no GPU needed) by tests/vt/test_moe_router_warp_map.cpp.
//
// This is where 6a8c5cf9's "vLLM's topkGating reorders the softmax reduction so
// it is off-limits" (see :156-163) is too strong. It is true of vLLM's OWN lane
// map -- at topkGating<8,256,4,16,32,...> lane L owns the CONTIGUOUS experts
// [8L, 8L+8) (topk_softmax_kernels.cu:344-346), which genuinely reassociates --
// and false of this one.
//
// SHAPE port only. Every arithmetic decision below is the block kernel's, NOT
// vLLM's: the divide (not a reciprocal multiply), the sum>0 guard, the isfinite
// clamp after normalize, the -INFINITY mask, the -INFINITY max seed that erases
// NaN, the denom<=0 -> 1 guard, the trailing divide, and the best<0 sentinel.
// vLLM does five of those differently and porting any of them changes tokens;
// the spec §4 tabulates each against its upstream file:line.
template <typename Tin, int VPT>
__launch_bounds__(kMoeRouterWarpsPerCta* kMoeRouterWarpWidth) __global__
    void MoeRouterTopKWarpKernel(float* weights, int32_t* indices, const Tin* logits, int64_t t,
                                 int k, bool renormalize) {
  constexpr int kE = kMoeRouterWarpWidth * VPT;
  const int lane = static_cast<int>(threadIdx.x);
  // One token per warp; the exit is warp-UNIFORM, so a warp that survives it
  // has all 32 lanes active and the 0xffffffffu shuffle masks are valid.
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kMoeRouterWarpsPerCta +
                      static_cast<int64_t>(threadIdx.y);
  if (row >= t) return;
  const Tin* lrow = logits + row * kE;

  // The ONLY read of the logit row. Slot q covers the 32 consecutive experts
  // [32q, 32q+32), so every load is fully coalesced.
  float p[VPT];
#pragma unroll
  for (int q = 0; q < VPT; ++q) p[q] = Load(lrow, MoeRouterWarpExpert(lane, q));

  // Max: per-lane halving tree (with the -INFINITY seed that erases NaN),
  // then the five in-warp levels. Congruent to :70-79.
  float m = MoeRouterWarpTreeMax<VPT>(p);
#pragma unroll
  for (int off = kMoeRouterWarpWidth / 2; off > 0; off >>= 1) {
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
  }
  const float mx = __shfl_sync(0xffffffffu, m, 0);  // the block read red[0] here

  // exp(logit - max) stays in registers -- no second global read, no sp[].
  // Congruent to :82-94.
#pragma unroll
  for (int q = 0; q < VPT; ++q) p[q] = expf(p[q] - mx);
  float s = MoeRouterWarpTreeSum<VPT>(p);
#pragma unroll
  for (int off = kMoeRouterWarpWidth / 2; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffffu, s, off);
  }
  const float sum = __shfl_sync(0xffffffffu, s, 0);

  // Normalize: the DIVIDE, the sum>0 guard, the clamp after (:97-103).
#pragma unroll
  for (int q = 0; q < VPT; ++q) {
    float pj = sum > 0.0f ? p[q] / sum : 0.0f;
    if (!isfinite(pj)) pj = 0.0f;
    p[q] = pj;
  }

  // Greedy top-k. The argmax is a reduction over the total order "higher value,
  // then lower expert index", which IS associative and commutative, so this
  // grouping matches the block's per-thread/warp/leader grouping exactly --
  // the argument already recorded at :156-163 and unchanged here.
  float denom = 0.0f;  // meaningful on lane 0 only, accumulated in k order
  for (int j = 0; j < k; ++j) {
    float lv = -INFINITY;
    int li = -1;
#pragma unroll
    for (int q = 0; q < VPT; ++q) {  // ascending expert index within the lane
      if (p[q] > lv) {               // strict `>` -> lowest index at the max
        lv = p[q];
        li = MoeRouterWarpExpert(lane, q);
      }
    }
#pragma unroll
    for (int off = kMoeRouterWarpWidth / 2; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, lv, off);
      const int oi = __shfl_down_sync(0xffffffffu, li, off);
      if (ov > lv || (ov == lv && oi >= 0 && (li < 0 || oi < li))) {
        lv = ov;
        li = oi;
      }
    }
    const float best_v = __shfl_sync(0xffffffffu, lv, 0);
    const int best = __shfl_sync(0xffffffffu, li, 0);
    // Exclude the winner from later rounds, in the owning lane's own register.
    // The slot compare is UNROLLED on purpose: a runtime index into a per-thread
    // array forces it to local memory, which would spill the whole row and cost
    // exactly the register residency this kernel exists for.
    if (best >= 0 && lane == (best & (kMoeRouterWarpWidth - 1))) {
      const int slot = best / kMoeRouterWarpWidth;
#pragma unroll
      for (int q = 0; q < VPT; ++q) {
        if (q == slot) p[q] = -INFINITY;
      }
    }
    if (lane == 0) {
      weights[row * k + j] = best_v;                     // -INFINITY when best < 0
      indices[row * k + j] = static_cast<int32_t>(best);  // -1 when best < 0
      denom += best_v;
    }
  }
  if (lane == 0 && renormalize) {
    if (!(denom > 0.0f)) denom = 1.0f;  // denom<=0 -> 1 guard, as :197
    for (int j = 0; j < k; ++j) weights[row * k + j] /= denom;
  }
}

// ─── Grouped-topk (`noaux_tc`) router (W3) ─────────────────────────────────
// Mirrors the CPU reference (cpu_ops.cpp MoeRouterGroupedTopKKernel), itself a
// 1:1 port of grouped_topk_router.py:106-161 @ e24d1b24. This is a SEPARATE
// kernel from MoeRouterTopKKernel above: the ungrouped path is not touched, so
// the 27B / 35B / Coder / Qwen3-dense routers stay byte-identical.
//
// Structure: the SCORING pass is parallel (identical shape to the ungrouped
// kernel, so the softmax tree-sum matches it); the group scoring, group mask,
// top-k and renorm/scale run on thread 0. At DeepSeek-V3's real dimensions
// (E=256, n_group=8, topk_group=4, top_k=8) that is a few thousand serial ops
// per token — correctness-grade, deterministic, and bit-identical to the CPU
// reference by construction. Speed work belongs to W9, after the numerics are
// gated. Dynamic shared memory holds [sel(e) | orig(e) | gscore(n_group)].
template <typename Tin>
__global__ void MoeRouterGroupedTopKKernel(float* weights, int32_t* indices,
                                           const Tin* logits, const float* bias, int64_t e,
                                           int k, bool renormalize, bool sigmoid,
                                           int64_t n_group, int topk_group,
                                           float routed_scaling_factor) {
  const int64_t row = blockIdx.x;
  const Tin* lrow = logits + row * e;
  extern __shared__ float smem[];
  float* sel = smem;              // [e] SELECTION score (biased)
  float* orig = smem + e;         // [e] WEIGHT score (unbiased)
  float* gscore = smem + 2 * e;             // [n_group]
  float* gkeep = smem + 2 * e + n_group;    // [n_group] 0/1 mask
  __shared__ float red[kBlock];
  __shared__ int redi[kBlock];  // argmax partner for red, step (4)

  // (1) scores = softmax(logits, -1) | sigmoid(logits)  (:110-117)
  if (sigmoid) {
    // ELEMENTWISE — no cross-expert normalization (the V3/R1 path), so no
    // reduction and nothing to diverge from the CPU reference on.
    for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
      orig[j] = 1.0f / (1.0f + expf(-Load(lrow, j)));
    }
  } else {
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j < e; j += blockDim.x) m = fmaxf(m, Load(lrow, j));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) {
        red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
      }
      __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    float acc = 0.0f;
    for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
      const float ex = expf(Load(lrow, j) - mx);
      orig[j] = ex;
      acc += ex;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    const float sum = red[0];
    __syncthreads();
    for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
      float pj = sum > 0.0f ? orig[j] / sum : 0.0f;
      if (!isfinite(pj)) pj = 0.0f;
      orig[j] = pj;
    }
  }
  __syncthreads();

  // (2) the bias shifts the SELECTION score only; the WEIGHT stays unbiased.
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
    sel[j] = orig[j] + (bias != nullptr ? bias[j] : 0.0f);
  }
  __syncthreads();

  // Steps (2) and (3) stay serial on thread 0: they are O(n_group * group_size)
  // ONCE per token and n_group is 1 or 8 for the shapes we serve. The block is
  // NOT retired here any more, because step (4) below now uses it.
  const int64_t group_size = e / n_group;
  if (threadIdx.x == 0) {
  // Group score: top-2 SUM with a bias (:124-126), else the group MAX (:128-131).
  for (int64_t g = 0; g < n_group; ++g) {
    const int64_t base = g * group_size;
    if (bias != nullptr) {
      float b0 = -INFINITY, b1 = -INFINITY;
      for (int64_t j = 0; j < group_size; ++j) {
        const float v = sel[base + j];
        if (v > b0) {
          b1 = b0;
          b0 = v;
        } else if (v > b1) {
          b1 = v;
        }
      }
      gscore[g] = b0 + b1;
    } else {
      float m = -INFINITY;
      for (int64_t j = 0; j < group_size; ++j) m = fmaxf(m, sel[base + j]);
      gscore[g] = m;
    }
  }
  // (3) keep the top `topk_group` groups, mask the rest to -inf (:133-145).
  // Strict `>` over ascending g -> lowest group index wins an exact tie. Uses an
  // explicit keep mask (NOT an in-place sentinel), matching the CPU reference
  // exactly — an all-`-inf` group row must still be selectable.
  for (int64_t g = 0; g < n_group; ++g) gkeep[g] = 0.0f;
  for (int gi = 0; gi < topk_group; ++gi) {
    int64_t best = -1;
    float best_v = -INFINITY;
    for (int64_t g = 0; g < n_group; ++g) {
      if (gkeep[g] != 0.0f) continue;
      if (best < 0 || gscore[g] > best_v) {  // first unkept index seeds the scan
        best_v = gscore[g];
        best = g;
      }
    }
    if (best < 0) break;  // fewer groups than topk_group (wrapper forbids it)
    gkeep[best] = 1.0f;
  }
    for (int64_t g = 0; g < n_group; ++g) {
      if (gkeep[g] != 0.0f) continue;
      for (int64_t j = 0; j < group_size; ++j) sel[g * group_size + j] = -INFINITY;
    }
  }
  __syncthreads();  // the group mask in sel[] must be visible to the block

  // (4) top-k over the masked selection scores; weight from the unbiased score.
  //
  // BLOCK-PARALLEL, and byte-identical to the serial scan it replaces. This was
  // `for idx in [0,e)` on THREAD 0 alone, k times: at Kimi-Linear's shape
  // (e=256, k=8) that is ~2048 serial compares on one lane of one block, which
  // the kernel's own comment deferred to "W9, after the numerics are gated".
  // The identity argument is the same one used for the ungrouped router: this
  // is an ARGMAX over a total order (higher score wins, exact tie to the lower
  // expert index, which the serial ascending scan with strict `>` also gives),
  // and argmax is associative and commutative, so any reduction order returns
  // the same (value, index). Everything ARITHMETIC is untouched and still runs
  // on thread 0 in the same order: the `denom` accumulation in k order, the
  // renormalize, and the routed_scaling_factor.
  float denom = 0.0f;  // meaningful on thread 0 only
  constexpr int kWarps = kBlock / 32;
  for (int j = 0; j < k; ++j) {
    float lv = -INFINITY;
    int li = -1;
    for (int64_t idx = threadIdx.x; idx < e; idx += blockDim.x) {
      const float v = sel[idx];
      if (v > lv) {  // strict `>`, ascending stride -> lowest index at the max
        lv = v;
        li = static_cast<int>(idx);
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, lv, off);
      const int oi = __shfl_down_sync(0xffffffffu, li, off);
      if (ov > lv || (ov == lv && oi >= 0 && (li < 0 || oi < li))) {
        lv = ov;
        li = oi;
      }
    }
    if ((threadIdx.x & 31u) == 0u) {
      red[threadIdx.x >> 5] = lv;
      redi[threadIdx.x >> 5] = li;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float best_v = red[0];
      int best = redi[0];
      for (int w2 = 1; w2 < kWarps; ++w2) {
        const float ov = red[w2];
        const int oi = redi[w2];
        if (ov > best_v || (ov == best_v && oi >= 0 && (best < 0 || oi < best))) {
          best_v = ov;
          best = oi;
        }
      }
      if (best < 0) best = 0;  // all -inf: the serial scan also fell back to 0
      sel[best] = -INFINITY;
      const float w = orig[best];
      weights[row * k + j] = w;
      indices[row * k + j] = static_cast<int32_t>(best);
      denom += w;
    }
    __syncthreads();  // sel[best]=-INF visible + red/redi reusable next round
  }
  if (threadIdx.x != 0) return;
  // (5) renormalize (:156-157) THEN routed_scaling_factor (:159-160).
  if (renormalize) {
    if (!(denom > 0.0f)) denom = 1.0f;
    for (int j = 0; j < k; ++j) weights[row * k + j] /= denom;
  }
  if (routed_scaling_factor != 1.0f) {
    for (int j = 0; j < k; ++j) weights[row * k + j] *= routed_scaling_factor;
  }
}

template <typename Tin>
void LaunchGroupedRouter(cudaStream_t s, Tensor& weights, Tensor& indices,
                         const Tensor& logits, const float* bias, int64_t t, int64_t e,
                         const MoeRouterTopKArgs& args) {
  // Dynamic shared memory layout, in floats: [sel(e) | orig(e) | gscore(G) |
  // gkeep(G)]. BOTH per-group arrays must be counted — an earlier version
  // allocated only ONE of them and `compute-sanitizer memcheck` caught the
  // resulting out-of-bounds __shared__ write on gkeep (the unit tests still
  // PASSED, since the stray write landed outside the live data; a green test is
  // not evidence of memory safety).
  const size_t shmem = (static_cast<size_t>(2 * e) +
                        2 * static_cast<size_t>(args.num_expert_group)) *
                       sizeof(float);
  MoeRouterGroupedTopKKernel<Tin><<<static_cast<unsigned>(t), kBlock, shmem, s>>>(
      weights.Ptr<float>(), indices.Ptr<int32_t>(), logits.Ptr<Tin>(), bias, e, args.top_k,
      args.renormalize, args.scoring_func == MoeScoringFunc::kSigmoid,
      args.num_expert_group, args.topk_group, args.routed_scaling_factor);
  Check(cudaGetLastError(), "moe_router_grouped_topk launch");
}

// VT_MOE_ROUTER_WARP (default ON, "0" restores the block kernel for a
// same-binary A/B). Read fresh per launch -- a getenv on a host path that runs
// once per MoE layer per step -- so an in-process test can flip it, matching
// Fa2PrefillEnabled() (cuda_paged_attn.cu:2504-2507). Under CUDA-graph capture
// it is read at capture time and the graph bakes the chosen kernel, which is
// how every other lever in this backend behaves.
bool MoeRouterWarpEnabled() {
  return MoeRouterWarpFlagIsOn(std::getenv("VT_MOE_ROUTER_WARP"));
}

// Returns false when this (E) is not one of the widths whose byte-exactness is
// derived, so the caller falls through to the UNCHANGED block kernel.
template <typename Tin>
bool LaunchRouterWarp(cudaStream_t s, Tensor& weights, Tensor& indices, const Tensor& logits,
                      int64_t t, int64_t e, int k, bool renorm) {
  const int vpt = MoeRouterWarpValuesPerThread(e);
  if (vpt == 0) return false;  // decide BEFORE touching the tensors
  const dim3 block(kMoeRouterWarpWidth, kMoeRouterWarpsPerCta);
  const unsigned grid =
      static_cast<unsigned>((t + kMoeRouterWarpsPerCta - 1) / kMoeRouterWarpsPerCta);
  float* w = weights.Ptr<float>();
  int32_t* idx = indices.Ptr<int32_t>();
  const Tin* lg = logits.Ptr<Tin>();
  switch (vpt) {
    case 1:
      MoeRouterTopKWarpKernel<Tin, 1><<<grid, block, 0, s>>>(w, idx, lg, t, k, renorm);
      break;
    case 2:
      MoeRouterTopKWarpKernel<Tin, 2><<<grid, block, 0, s>>>(w, idx, lg, t, k, renorm);
      break;
    case 4:
      MoeRouterTopKWarpKernel<Tin, 4><<<grid, block, 0, s>>>(w, idx, lg, t, k, renorm);
      break;
    case 8:
      MoeRouterTopKWarpKernel<Tin, 8><<<grid, block, 0, s>>>(w, idx, lg, t, k, renorm);
      break;
    default:
      return false;
  }
  Check(cudaGetLastError(), "moe_router_topk_warp launch");
  return true;
}

template <typename Tin>
void LaunchRouter(cudaStream_t s, Tensor& weights, Tensor& indices, const Tensor& logits,
                  int64_t t, int64_t e, int k, bool renorm, bool serial) {
  // The single-warp kernel is byte-identical (see MoeRouterTopKWarpKernel) but
  // NEVER replaces `serial`: that path is the byte-exact ORACLE the parity test
  // compares against, so changing it would invalidate the oracle instead of
  // testing the candidate.
  if (!serial && MoeRouterWarpEnabled() &&
      LaunchRouterWarp<Tin>(s, weights, indices, logits, t, e, k, renorm)) {
    return;
  }
  const size_t shmem = static_cast<size_t>(e) * sizeof(float);
  if (serial) {
    MoeRouterTopKKernel<Tin, true><<<static_cast<unsigned>(t), kBlock, shmem, s>>>(
        weights.Ptr<float>(), indices.Ptr<int32_t>(), logits.Ptr<Tin>(), e, k, renorm);
  } else {
    MoeRouterTopKKernel<Tin, false><<<static_cast<unsigned>(t), kBlock, shmem, s>>>(
        weights.Ptr<float>(), indices.Ptr<int32_t>(), logits.Ptr<Tin>(), e, k, renorm);
  }
  Check(cudaGetLastError(), "moe_router_topk launch");
}

void RouterDispatch(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                    const MoeRouterTopKArgs& args, const Tensor* bias_t, bool serial) {
  VT_CHECK(logits.dtype == DType::kF32 || logits.dtype == DType::kBF16,
           "cuda moe_router_topk: unsupported logits dtype (f32/bf16 only)");
  const int64_t t = logits.shape[0], e = logits.shape[1];
  if (t == 0 || e == 0) return;
  cudaStream_t s = AsStream(q);
  if (args.num_expert_group > 0) {  // W3 grouped-topk (`noaux_tc`) path
    const float* bias = bias_t != nullptr ? bias_t->Ptr<float>() : nullptr;
    if (logits.dtype == DType::kF32) {
      LaunchGroupedRouter<float>(s, weights, indices, logits, bias, t, e, args);
    } else {
      LaunchGroupedRouter<__nv_bfloat16>(s, weights, indices, logits, bias, t, e, args);
    }
    return;
  }
  if (logits.dtype == DType::kF32) {
    LaunchRouter<float>(s, weights, indices, logits, t, e, args.top_k, args.renormalize, serial);
  } else {
    LaunchRouter<__nv_bfloat16>(s, weights, indices, logits, t, e, args.top_k, args.renormalize,
                                serial);
  }
}

void MoeRouterTopKKernelCuda(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args, const Tensor* bias) {
  RouterDispatch(q, weights, indices, logits, args, bias, /*serial=*/false);
}

// ---------------------------------------------------------------------------
// moe_combine (moe-semantics.md §4/§6): thread per (token, output-dim). Sums
// the k expert contributions weighted by the router weights (f32 accumulation),
// adds the optional shared term in f32, single store-rounding — same as the CPU
// reference (cpu_ops.cpp MoeCombineKernel), so CPU and CUDA agree bit-for-bit.
// (No upstream double-round here; that M0.9 decision is separate.)
// Upstream counterpart: layers/fused_moe/ (moe_sum reduction over the topk
// weighted w2 outputs) — M2.2 replaces this correctness-grade path.

// `routed_scale` multiplies the ROUTED sum only, BEFORE the shared term is added
// — upstream's apply_routed_scale_to_output arm (layers/fused_moe/runner/
// moe_runner.py:390-407 (:402-406) scales `fused_output`, leaves `shared_output` alone,
// then :722-725 adds them). Applied in the same f32 accumulator the CPU
// reference (cpu_ops.cpp MoeCombineKernel) uses, in the same order. The scale
// itself is ONE standalone f32 multiply on the finished accumulator, with
// nothing adjacent to contract into, so THIS step is bit-identical to the CPU
// reference. That does not extend to the `acc += w * Load(...)` reduction above
// it: only CXX/HIP/OBJCXX carry -ffp-contract=off (CMakeLists.txt:55, :393,
// :467) and nothing passes nvcc --fmad=false, so device code may contract that
// multiply-add into an FMA where the host may not. See #591, which tracks that
// repo-wide flag gap. Default 1.0f == the landed fold-into-weights arm.
// Like the CPU reference it scales the ASSEMBLED sum, not each router weight
// (:404 `fused_output *= factor` is one multiply on the finished tensor); the
// fold is equal in exact arithmetic and a different f32 value. Upstream's fp16
// arm (:403-406, divide `shared_output` instead) is unreachable here — `out` is
// gated to f32/bf16 by `IsOutFloat` (ops.cpp:22). See cpu_ops.cpp for the full
// note.
template <typename Teo, typename Tsh, typename Tout>
__global__ void MoeCombineKernel(Tout* out, const Teo* expert_out, const float* weights,
                                 const Tsh* shared, int64_t t, int64_t h, int k,
                                 float routed_scale) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    const int64_t col = idx % h;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
      acc += weights[row * k + j] * Load(expert_out, (row * k + j) * h + col);
    if (routed_scale != 1.0f) acc *= routed_scale;
    if (shared != nullptr) acc += Load(shared, idx);
    Store(out, idx, acc);
  }
}

template <typename Teo, typename Tsh, typename Tout>
void LaunchCombine(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                   const Tensor* shared, int64_t t, int64_t h, int k, float routed_scale) {
  MoeCombineKernel<Teo, Tsh, Tout><<<GridFor(t * h), kBlock, 0, s>>>(
      out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(),
      shared != nullptr ? shared->Ptr<Tsh>() : nullptr, t, h, k, routed_scale);
  Check(cudaGetLastError(), "moe_combine launch");
}

// Dispatch shared dtype (or the no-shared path, where Tsh is unused).
template <typename Teo, typename Tout>
void DispatchShared(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor* shared, int64_t t, int64_t h, int k, float routed_scale) {
  if (shared == nullptr || shared->dtype == DType::kF32) {
    LaunchCombine<Teo, float, Tout>(s, out, expert_out, weights, shared, t, h, k, routed_scale);
  } else {
    LaunchCombine<Teo, __nv_bfloat16, Tout>(s, out, expert_out, weights, shared, t, h, k,
                                            routed_scale);
  }
}

template <typename Teo>
void DispatchOut(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                 const Tensor* shared, int64_t t, int64_t h, int k, float routed_scale) {
  if (out.dtype == DType::kF32) {
    DispatchShared<Teo, float>(s, out, expert_out, weights, shared, t, h, k, routed_scale);
  } else {
    DispatchShared<Teo, __nv_bfloat16>(s, out, expert_out, weights, shared, t, h, k, routed_scale);
  }
}

void MoeCombineKernelCuda(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                          const Tensor* shared, float routed_scale) {
  VT_CHECK(expert_out.dtype == DType::kF32 || expert_out.dtype == DType::kBF16,
           "cuda moe_combine: unsupported expert_out dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "cuda moe_combine: unsupported out dtype (f32/bf16 only)");
  VT_CHECK(shared == nullptr || shared->dtype == DType::kF32 || shared->dtype == DType::kBF16,
           "cuda moe_combine: unsupported shared dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  if (t == 0 || h == 0) return;
  cudaStream_t s = AsStream(q);
  if (expert_out.dtype == DType::kF32) {
    DispatchOut<float>(s, out, expert_out, weights, shared, t, h, static_cast<int>(k),
                       routed_scale);
  } else {
    DispatchOut<__nv_bfloat16>(s, out, expert_out, weights, shared, t, h, static_cast<int>(k),
                               routed_scale);
  }
}

// ---------------------------------------------------------------------------
// moe_combine_gate (MoE glue fusion): MoeCombine with the shared-expert gate
// fused inline. Instead of a pre-materialized bf16 `shared` buffer (produced by
// a separate SharedExpertGate launch + read back here), it takes sd [T,H] f32
// and gl [T,1] f32 and computes the shared term per element as
//   bf16(sigmoid(gl[row]) * sd[idx])  -> re-added in f32,
// which is bit-identical to SharedExpertGate's store (Store<bf16>, round-to-
// nearest-even) followed by MoeCombine's Load(shared) (bf16 -> f32). Saves one
// kernel launch and the shared [T,H] global write+read per MoE layer. Mirrors
// vLLM's fused weight-and-reduce (layers/fused_moe/moe_fused_mul_sum.py,
// topk_weight_and_reduce.py moe_sum) extended to fold the shared contribution.
__device__ inline float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// `sd` is read through Tsd: the f32 the shared down-proj used to be cast to, or
// the bf16 the Marlin GEMM actually produced. Widening bf16 to float in-kernel
// is EXACT, and the value is immediately re-rounded through bf16 below, so both
// forms are bit-identical -- the bf16 one just skips a whole [T,H] f32 cast.
template <typename Teo, typename Tsd, typename Tout>
__global__ void MoeCombineGateKernel(Tout* out, const Teo* expert_out, const float* weights,
                                     const Tsd* sd, const float* gl, int64_t t, int64_t h,
                                     int k) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    const int64_t col = idx % h;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
      acc += weights[row * k + j] * Load(expert_out, (row * k + j) * h + col);
    // Shared-expert gate, rounded through bf16 exactly as SharedExpertGate's
    // store, then re-added in f32 (matches MoeCombine's Load(shared) bf16->f32).
    const float sv = SigmoidF(gl[row]) * Load(sd, idx);
    acc += __bfloat162float(__float2bfloat16(sv));
    Store(out, idx, acc);
  }
}

template <typename Teo, typename Tout>
void LaunchCombineGate(cudaStream_t s, Tensor& out, const Tensor& expert_out,
                       const Tensor& weights, const Tensor& sd, const Tensor& gl, int64_t t,
                       int64_t h, int k) {
  if (sd.dtype == DType::kBF16) {
    MoeCombineGateKernel<Teo, __nv_bfloat16, Tout><<<GridFor(t * h), kBlock, 0, s>>>(
        out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(),
        sd.Ptr<__nv_bfloat16>(), gl.Ptr<float>(), t, h, k);
  } else {
    MoeCombineGateKernel<Teo, float, Tout><<<GridFor(t * h), kBlock, 0, s>>>(
        out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(), sd.Ptr<float>(),
        gl.Ptr<float>(), t, h, k);
  }
  Check(cudaGetLastError(), "moe_combine_gate launch");
}

template <typename Teo>
void DispatchOutGate(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                     const Tensor& sd, const Tensor& gl, int64_t t, int64_t h, int k) {
  if (out.dtype == DType::kF32) {
    LaunchCombineGate<Teo, float>(s, out, expert_out, weights, sd, gl, t, h, k);
  } else {
    LaunchCombineGate<Teo, __nv_bfloat16>(s, out, expert_out, weights, sd, gl, t, h, k);
  }
}

void MoeCombineGateKernelCuda(Queue& q, Tensor& out, const Tensor& expert_out,
                              const Tensor& weights, const Tensor& sd, const Tensor& gl) {
  VT_CHECK(expert_out.dtype == DType::kF32 || expert_out.dtype == DType::kBF16,
           "cuda moe_combine_gate: unsupported expert_out dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "cuda moe_combine_gate: unsupported out dtype (f32/bf16 only)");
  VT_CHECK(sd.dtype == DType::kF32 || sd.dtype == DType::kBF16,
           "cuda moe_combine_gate: unsupported sd dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  if (t == 0 || h == 0) return;
  cudaStream_t s = AsStream(q);
  if (expert_out.dtype == DType::kF32) {
    DispatchOutGate<float>(s, out, expert_out, weights, sd, gl, t, h, static_cast<int>(k));
  } else {
    DispatchOutGate<__nv_bfloat16>(s, out, expert_out, weights, sd, gl, t, h,
                                   static_cast<int>(k));
  }
}

// ---------------------------------------------------------------------------
// moe_silu_mul (moe-semantics.md §4): out[i] = silu(gate[i]) * up[i], the fused
// activation between the grouped gate/up and down GEMMs. f32 math (silu via
// expf), rounded on store — the same accepted expf-vs-std::exp deviation the CPU
// reference carries (the routed sum is bf16-robust; the greedy gate is stable).
template <typename Tg, typename Tu, typename Tout>
__global__ void MoeSiluMulKernel(Tout* out, const Tg* gate, const Tu* up, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const float g = Load(gate, i);
    // vLLM a7c23ac96d activation_kernels.cu:165-167 returns the gate type.
    const float s = static_cast<float>(static_cast<Tg>(g / (1.0f + expf(-g))));
    Store(out, i, s * Load(up, i));
  }
}

template <typename Tg, typename Tu, typename Tout>
void LaunchSiluMul(cudaStream_t s, Tensor& out, const Tensor& gate, const Tensor& up, int64_t n) {
  MoeSiluMulKernel<Tg, Tu, Tout>
      <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<Tout>(), gate.Ptr<Tg>(), up.Ptr<Tu>(), n);
  Check(cudaGetLastError(), "moe_silu_mul launch");
}

template <typename Tg, typename Tu>
void SiluMulByOut(cudaStream_t s, Tensor& out, const Tensor& gate, const Tensor& up, int64_t n) {
  if (out.dtype == DType::kF32) {
    LaunchSiluMul<Tg, Tu, float>(s, out, gate, up, n);
  } else if (out.dtype == DType::kF16) {
    LaunchSiluMul<Tg, Tu, __half>(s, out, gate, up, n);
  } else {
    LaunchSiluMul<Tg, Tu, __nv_bfloat16>(s, out, gate, up, n);
  }
}

template <typename Tg>
void SiluMulByUp(cudaStream_t s, Tensor& out, const Tensor& gate, const Tensor& up, int64_t n) {
  if (up.dtype == DType::kF32) {
    SiluMulByOut<Tg, float>(s, out, gate, up, n);
  } else if (up.dtype == DType::kF16) {
    SiluMulByOut<Tg, __half>(s, out, gate, up, n);
  } else {
    SiluMulByOut<Tg, __nv_bfloat16>(s, out, gate, up, n);
  }
}

void MoeSiluMulKernelCuda(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up) {
  VT_CHECK(gate.dtype == DType::kF32 || gate.dtype == DType::kBF16 || gate.dtype == DType::kF16,
           "cuda moe_silu_mul: unsupported gate dtype (f32/f16/bf16 only)");
  VT_CHECK(up.dtype == DType::kF32 || up.dtype == DType::kBF16 || up.dtype == DType::kF16,
           "cuda moe_silu_mul: unsupported up dtype (f32/f16/bf16 only)");
  const int64_t n = out.Numel();
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  if (gate.dtype == DType::kF32) {
    SiluMulByUp<float>(s, out, gate, up, n);
  } else if (gate.dtype == DType::kF16) {
    SiluMulByUp<__half>(s, out, gate, up, n);
  } else {
    SiluMulByUp<__nv_bfloat16>(s, out, gate, up, n);
  }
}

// ---------------------------------------------------------------------------
// moe_relu2: out[i] = relu(x[i])^2, the NON-GATED MoE activation (NemotronH's
// expert epilogue — nemotron_h.py:227 activation_without_mul("relu2") ->
// MoEActivation.RELU2_NO_MUL). Sibling of moe_silu_mul with ONE input, because a
// non-gated expert has no gate half. Dtype order is upstream's relu_squared_kernel
// (csrc/libtorch_stable/activation_kernels.cu:673-678) verbatim: widen to f32,
// clamp at zero in f32, square in f32, ONE round on the store — so a bf16 input
// with an f32 output keeps the full f32 square. Byte-identical to the CPU
// reference (cpu_ops.cpp MoeRelu2Kernel): both are exact f32 ops, no expf —
// and unlike the combine reduction above there is no multiply-add here for
// nvcc to contract into an FMA (a compare-select and ONE multiply), so this
// one holds without --fmad=false. See #591 for the flag gap itself.
template <typename Tx, typename Tout>
__global__ void MoeRelu2Kernel(Tout* out, const Tx* x, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const float f = Load(x, i);
    const float v = f > 0.0f ? f : 0.0f;
    Store(out, i, v * v);
  }
}

template <typename Tx, typename Tout>
void LaunchRelu2(cudaStream_t s, Tensor& out, const Tensor& x, int64_t n) {
  MoeRelu2Kernel<Tx, Tout><<<GridFor(n), kBlock, 0, s>>>(out.Ptr<Tout>(), x.Ptr<Tx>(), n);
  Check(cudaGetLastError(), "moe_relu2 launch");
}

template <typename Tx>
void Relu2ByOut(cudaStream_t s, Tensor& out, const Tensor& x, int64_t n) {
  if (out.dtype == DType::kF32) {
    LaunchRelu2<Tx, float>(s, out, x, n);
  } else {
    LaunchRelu2<Tx, __nv_bfloat16>(s, out, x, n);
  }
}

void MoeRelu2KernelCuda(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda moe_relu2: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "cuda moe_relu2: unsupported out dtype (f32/bf16 only)");
  const int64_t n = out.Numel();
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    Relu2ByOut<float>(s, out, x, n);
  } else {
    Relu2ByOut<__nv_bfloat16>(s, out, x, n);
  }
}

// Registers the CUDA MoE kernels during static init (pre-main, like the M0.6
// ops in cuda_ops.cu). Filling the op table is harmless on machines without a
// GPU: the kCUDA backend never registers there, so no CUDA queue can dispatch.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelCuda)));
    RegisterOp(OpId::kMoeCombine, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernelCuda)));
    RegisterOp(OpId::kMoeCombineGate, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeCombineGateFn>(&MoeCombineGateKernelCuda)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernelCuda)));
    RegisterOpScalarType(OpId::kMoeSiluMul, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(OpId::kMoeRelu2, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeRelu2Fn>(&MoeRelu2KernelCuda)));
  }
} registrar;

}  // namespace

// Test-only reference (external linkage): launches the original single-threaded
// greedy top-k so the byte-exact routing parity test can prove the parallel
// production path is byte-identical. Declared in include/vt/cuda/moe_decode_ref.h.
void MoeRouterTopKSerialCuda(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args) {
  RouterDispatch(q, weights, indices, logits, args, /*bias_t=*/nullptr, /*serial=*/true);
}

}  // namespace vt::cuda
