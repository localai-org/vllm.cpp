// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Paged attention op unit tests. Semantics ported from
// vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward @ e24d1b24
// and tests/v1/attention/test_attention_backends.py:745-867 (causal decoder and
// symmetric encoder sliding-window masks over paged K/V; scale = self.scale).
// Kernel vectors additionally carry tests/kernels/attention/test_flash_attn.py:
// 95-217 (paged varlen, GQA and window boundaries).
// The cache is the
// NHD FlashAttention layout — the two dim-1 unbind slices of get_kv_cache_shape's
// (num_blocks, 2, block_size, num_kv_heads, head_size).
//
// Golden strategy (M1.6 Task-3 review): COMPOSE the reference math, do NOT dump
// backend cache bytes.
//   1. ANCHOR (strongest): on a single contiguous sequence, paged_attention MUST
//      agree with M0.9's dense vt::Attention. The K/V are written into a paged
//      cache via vt::ReshapeAndCache, so the whole pipeline is exercised. No new
//      oracle needed.
//   2. Batched varlen (2 reqs: prefill len 4 + decode at seq_len 6) validated
//      against a composed per-token causal GQA softmax reference (host math).
//   3. GQA head mapping, causal masking, block-spanning (seq > block_size).
//   4. CUDA-vs-CPU parity (guarded by HasCuda; dgx-pending on CPU-only boxes).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#ifdef VLLM_CPP_FLASH_ATTN
namespace vt::cuda::testing {
int Fa2DecodeNumSplitsForTesting(int batch_nheads_mblocks, int num_sms,
                                 int num_n_blocks, int max_splits);
void ResetFa2DecodeDebugCounters();
void DisableFa2DecodeDebugCounters();
uint64_t Fa2DecodeLaunchesForTesting();
uint64_t Fa2DecodeSplitLaunchesForTesting();
uint64_t Fa2DecodeNoSplitLaunchesForTesting();
uint64_t Fa2DecodeSwapLaunchesForTesting();
uint64_t Fa2DecodeScratchAllocationsForTesting();
uint64_t Fa2DecodeScratchReusesForTesting();
size_t Fa2DecodeScratchShapeCountForTesting(int device, void* stream);
}  // namespace vt::cuda::testing
#endif

using vt::AttentionArgs;
using vt::AttentionWindow;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

// Rank-4 strided view into a single (num_blocks, 2, block_size, H, D) allocation
// at element offset off_elems — the unbind(1) slice K/V really are (block stride
// = 2*bs*H*D, NOT bs*H*D). Mirrors the reshape_and_cache strided-slice tests.
template <typename T>
Tensor StridedCache(T* data, DType dt, Device dev, int64_t off_elems,
                    const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  Tensor t;
  t.data = data + off_elems;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride[static_cast<size_t>(i)];
  }
  return t;
}

Tensor F32(std::vector<float>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kF32, Cpu(), shape);
}
Tensor I32(std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kI32, Cpu(), shape);
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  // Deterministic LCG in [-2,2); matches the other vt op tests.
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

// Host composed reference: per-token causal/non-causal GQA softmax over K/V gathered from a
// contiguous NHD cache (block b, offset o at flat index ((b*bs+o)*H+g)*D+e). This
// is the M0.9-style reference, independent of the op's stride arithmetic.
std::vector<float> ComposedPagedRef(const std::vector<float>& q, const std::vector<float>& kc,
                                    const std::vector<float>& vc,
                                    const std::vector<int32_t>& block_table, int64_t max_blocks,
                                    const std::vector<int32_t>& seq_lens,
                                    const std::vector<int32_t>& qsl, int64_t hq, int64_t hk,
                                    int64_t d, int64_t block_size, float scale, bool causal,
                                    std::optional<AttentionWindow> window = std::nullopt) {
  const int64_t num_reqs = static_cast<int64_t>(seq_lens.size());
  const int64_t num_tokens = qsl.back();
  const int64_t qpk = hq / hk;
  std::vector<float> out(static_cast<size_t>(num_tokens * hq * d), 0.0f);
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[static_cast<size_t>(r)], q1 = qsl[static_cast<size_t>(r + 1)];
    const int64_t qlen = q1 - q0;
    const int64_t seqlen = seq_lens[static_cast<size_t>(r)];
    const int64_t context = seqlen - qlen;
    for (int64_t local = 0; local < qlen; ++local) {
      const int64_t t = q0 + local;
      const int64_t p = context + local;
      const int64_t jmin =
          window.has_value() ? std::max<int64_t>(0, p - window->left) : 0;
      int64_t jmax = causal ? p : seqlen - 1;
      if (window.has_value()) jmax = std::min(jmax, p + window->right);
      jmax = std::min(jmax, seqlen - 1);
      for (int64_t h = 0; h < hq; ++h) {
        const int64_t g = h / qpk;
        const int64_t qoff = (t * hq + h) * d;
        std::vector<float> sc(static_cast<size_t>(jmax - jmin + 1));
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t j = jmin; j <= jmax; ++j) {
          const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
          const int64_t off = j % block_size;
          const int64_t kbase = ((blk * block_size + off) * hk + g) * d;
          float dot = 0.0f;
          for (int64_t e = 0; e < d; ++e)
            dot += q[static_cast<size_t>(qoff + e)] * kc[static_cast<size_t>(kbase + e)];
          dot *= scale;
          sc[static_cast<size_t>(j - jmin)] = dot;
          if (dot > m) m = dot;
        }
        float denom = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const float e = std::exp(sc[static_cast<size_t>(j - jmin)] - m);
          sc[static_cast<size_t>(j - jmin)] = e;
          denom += e;
        }
        const float inv = 1.0f / denom;
        for (int64_t e = 0; e < d; ++e) {
          float a = 0.0f;
          for (int64_t j = jmin; j <= jmax; ++j) {
            const int64_t blk = block_table[static_cast<size_t>(r * max_blocks + j / block_size)];
            const int64_t off = j % block_size;
            const int64_t vbase = ((blk * block_size + off) * hk + g) * d;
            a += sc[static_cast<size_t>(j - jmin)] * inv * vc[static_cast<size_t>(vbase + e)];
          }
          out[static_cast<size_t>(qoff + e)] = a;
        }
      }
    }
  }
  return out;
}

}  // namespace

// ===========================================================================
// ANCHOR: single contiguous sequence — paged_attention == dense vt::Attention.
// Build one sequence, write its K/V into a paged cache via ReshapeAndCache, run
// both ops, assert equal. The strongest correctness gate (no new oracle).
// ===========================================================================
TEST_CASE("paged_attention anchor: single sequence agrees with dense vt::Attention") {
  const int64_t T = 7, Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  auto q = RandF32(static_cast<size_t>(T * Hq * D), 11);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), 22);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), 33);
  Queue qq = Q();

  // Dense reference (M0.9): contiguous single-sequence causal GQA attention.
  std::vector<float> dense(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor td = F32(dense, {T, Hq, D});
  vt::Attention(qq, td, tq, tk, tv, AttentionArgs{scale, /*causal=*/true});

  // Paged path: allocate a cache big enough for T tokens, write K/V at slots
  // 0..T-1 via ReshapeAndCache, then run PagedAttention over the one request.
  const int64_t num_blocks = (T + block_size - 1) / block_size + 1;  // slack block
  const int64_t page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slots[static_cast<size_t>(i)] = i;  // contiguous, block 0..
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {T});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  const int64_t max_blocks = num_blocks;
  std::vector<int32_t> block_table(static_cast<size_t>(max_blocks));
  for (int64_t b = 0; b < max_blocks; ++b) block_table[static_cast<size_t>(b)] = b;  // identity
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};

  std::vector<float> paged(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tp = F32(paged, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < dense.size(); ++i)
    CHECK(paged[i] == doctest::Approx(dense[i]).epsilon(1e-5));
}

// ===========================================================================
// ANCHOR on the REAL strided K/V unbind slices: same agreement must hold when
// k_cache/v_cache are the two dim-1 slices of one (nb,2,bs,H,D) allocation
// (block stride 2*bs*H*D). This is the layout the runner actually hands us.
// ===========================================================================
TEST_CASE("paged_attention anchor on strided unbind-slice cache == dense") {
  const int64_t T = 5, Hq = 2, Hk = 1, D = 4, block_size = 2;
  const float scale = 0.35f;
  auto q = RandF32(static_cast<size_t>(T * Hq * D), 101);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), 202);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), 303);
  Queue qq = Q();

  std::vector<float> dense(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor td = F32(dense, {T, Hq, D});
  vt::Attention(qq, td, tq, tk, tv, AttentionArgs{scale, true});

  const int64_t nb = (T + block_size - 1) / block_size + 1;
  const int64_t page = Hk * D;
  const int64_t within_block = block_size * page;
  const int64_t blk_stride = 2 * within_block;
  std::vector<float> buf(static_cast<size_t>(nb * 2 * block_size * page), 0.0f);
  const std::vector<int64_t> cshape = {nb, block_size, Hk, D};
  const std::vector<int64_t> cstride = {blk_stride, page, D, 1};
  Tensor tkc = StridedCache(buf.data(), DType::kF32, Cpu(), 0, cshape, cstride);
  Tensor tvc = StridedCache(buf.data(), DType::kF32, Cpu(), within_block, cshape, cstride);
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slots[static_cast<size_t>(i)] = i;
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {T});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table(static_cast<size_t>(nb));
  for (int64_t b = 0; b < nb; ++b) block_table[static_cast<size_t>(b)] = b;
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};
  std::vector<float> paged(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tp = F32(paged, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, nb});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < dense.size(); ++i)
    CHECK(paged[i] == doctest::Approx(dense[i]).epsilon(1e-5));
}

// ===========================================================================
// Batched varlen (M1.5 oracle shape): 2 reqs — req0 prefill len 4 (seq_len 4),
// req1 decode at seq_len 6 (query_len 1, context 5). query_start_loc=[0,4,5],
// seq_lens=[4,6]. Non-identity block tables + block-spanning (req1 uses 2 blocks
// with block_size 4). Validated against the composed host reference.
// ===========================================================================
TEST_CASE("paged_attention batched varlen (prefill + decode) vs composed reference") {
  const int64_t Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t num_tokens = 5;  // 4 (req0) + 1 (req1)
  std::vector<int32_t> qsl = {0, 4, 5};
  std::vector<int32_t> seq_lens = {4, 6};
  Queue qq = Q();

  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 44);

  // A paged cache with enough blocks; give the two requests DISJOINT,
  // non-identity blocks. req0 (seq 4) → 1 block; req1 (seq 6) → 2 blocks.
  const int64_t num_blocks = 6, page = Hk * D, max_blocks = 2;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  // block_table[req, blk_idx]: req0 uses block 3; req1 uses blocks 1 and 4.
  std::vector<int32_t> block_table = {3, 0,  /* req0: only block 3 needed */
                                      1, 4}; /* req1: pos 0..3 → blk1, pos 4..5 → blk4 */

  // Fill the cache directly (host) at the exact (block,offset,head) the tables
  // address, so the composed reference reads the same values the op will.
  auto set_cache = [&](std::vector<float>& cache, int64_t blk, int64_t off, uint32_t seed) {
    auto vals = RandF32(static_cast<size_t>(page), seed);
    const int64_t base = ((blk * block_size + off) * Hk) * D;
    for (int64_t i = 0; i < page; ++i) cache[static_cast<size_t>(base + i)] = vals[static_cast<size_t>(i)];
  };
  // req0 positions 0..3 → block 3, offsets 0..3.
  for (int64_t j = 0; j < 4; ++j) {
    set_cache(kc, 3, j, 500u + static_cast<uint32_t>(j));
    set_cache(vc, 3, j, 600u + static_cast<uint32_t>(j));
  }
  // req1 positions 0..3 → block 1 offsets 0..3; positions 4..5 → block 4 offsets 0..1.
  for (int64_t j = 0; j < 4; ++j) {
    set_cache(kc, 1, j, 700u + static_cast<uint32_t>(j));
    set_cache(vc, 1, j, 800u + static_cast<uint32_t>(j));
  }
  for (int64_t j = 0; j < 2; ++j) {
    set_cache(kc, 4, j, 900u + static_cast<uint32_t>(j));
    set_cache(vc, 4, j, 950u + static_cast<uint32_t>(j));
  }

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, /*causal=*/true);

  Tensor tq = F32(q, {num_tokens, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {2, max_blocks});
  Tensor tsl = I32(seq_lens, {2});
  Tensor tqsl = I32(qsl, {3});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  Tensor tp = F32(got, {num_tokens, Hq, D});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

// ===========================================================================
// Causal masking: a decode query (context 2, seq_len 3) must ignore nothing up
// to its own position, but a prefill query at position 0 must see only key 0
// even when a huge future V sits at position 1.
// ===========================================================================
TEST_CASE("paged_attention causal mask hides future keys from an early query") {
  const int64_t Hq = 1, Hk = 1, D = 2, block_size = 4;
  const int64_t T = 2;  // one request, prefill of 2 tokens
  const float scale = 1.0f;
  std::vector<float> q = {5, 0, 0, 5};  // token0 q=[5,0], token1 q=[0,5]
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 1, 1000, 1000};  // huge future v at pos 1
  Queue qq = Q();

  const int64_t num_blocks = 1, page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tk = F32(k, {T, Hk, D}), tv = F32(v, {T, Hk, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots = {0, 1};
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {2});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table = {0};
  std::vector<int32_t> seq_lens = {2};
  std::vector<int32_t> qsl = {0, 2};
  std::vector<float> out(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor tq = F32(q, {T, Hq, D}), tp = F32(out, {T, Hq, D});
  Tensor tbt = I32(block_table, {1, 1}), tsl = I32(seq_lens, {1}), tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  CHECK(out[0] == doctest::Approx(1.0f));  // token 0 sees only key 0
  CHECK(out[1] == doctest::Approx(1.0f));
}

// ===========================================================================
// SPEC-DFLASH2 W10 (#1857): the UNIFORM SPECULATIVE VERIFY shape — q>1 query
// rows per request over an existing context, exactly the batch a spec-decode
// verify presents. Two guarantees, one case:
//   1. INTRA-STEP CAUSALITY vs the composed reference: verify row `local` of a
//      request attends keys 0..context+local and nothing further. The values
//      are O(1) randoms, so one extra admitted position moves the softmax far
//      beyond the 1e-5 comparison band — a mask that admits one extra position
//      reds this case.
//   2. THE LENGTH BOUND: every cache slot past each request's seq_len carries a
//      poison V (1e6). Any read past the per-request K length drags the output
//      to the poison magnitude and reds the comparison.
// `uniform_spec_query_len` is set the way the production runner sets it (the
// classified verify width). On the CPU backend it is a no-op by design — the
// reference math IS the semantics — which this case pins by matching the same
// composed reference used everywhere else in this file.
// ===========================================================================
TEST_CASE("paged_attention uniform spec verify (q>1 per request) matches composed reference") {
  const int64_t Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t q_len = 3, B = 2;
  const int64_t num_tokens = q_len * B;  // 6
  std::vector<int32_t> qsl = {0, 3, 6};
  std::vector<int32_t> seq_lens = {5, 8};  // contexts 2 and 5
  Queue qq = Q();

  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 91);

  const int64_t num_blocks = 6, page = Hk * D, max_blocks = 2;
  // Poison EVERY slot first; the per-position fills below overwrite exactly the
  // in-length slots, so whatever remains poisoned is out of every request's
  // K-length bound and must never be read.
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 1.0e6f);
  std::vector<int32_t> block_table = {3, 5,   /* req0: seq 5 → blocks 3,5 */
                                      1, 4};  /* req1: seq 8 → blocks 1,4 */
  auto set_cache = [&](std::vector<float>& cache, int64_t blk, int64_t off, uint32_t seed) {
    auto vals = RandF32(static_cast<size_t>(page), seed);
    const int64_t base = ((blk * block_size + off) * Hk) * D;
    for (int64_t i = 0; i < page; ++i)
      cache[static_cast<size_t>(base + i)] = vals[static_cast<size_t>(i)];
  };
  for (int64_t j = 0; j < 5; ++j) {  // req0 positions 0..4
    const int64_t blk = block_table[static_cast<size_t>(j / block_size)];
    set_cache(kc, blk, j % block_size, 1500u + static_cast<uint32_t>(j));
    set_cache(vc, blk, j % block_size, 1600u + static_cast<uint32_t>(j));
  }
  for (int64_t j = 0; j < 8; ++j) {  // req1 positions 0..7
    const int64_t blk = block_table[static_cast<size_t>(2 + j / block_size)];
    set_cache(kc, blk, j % block_size, 1700u + static_cast<uint32_t>(j));
    set_cache(vc, blk, j % block_size, 1800u + static_cast<uint32_t>(j));
  }

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl,
                                            Hq, Hk, D, block_size, scale, /*causal=*/true);

  Tensor tq = F32(q, {num_tokens, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {2, max_blocks});
  Tensor tsl = I32(seq_lens, {2});
  Tensor tqsl = I32(qsl, {3});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  Tensor tp = F32(got, {num_tokens, Hq, D});
  PagedAttentionArgs args{scale, true};
  args.query_start_loc_host = qsl.data();
  args.max_seq_len = 8;
  args.uniform_spec_query_len = static_cast<int32_t>(q_len);
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, args);

  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
  }
}

// ===========================================================================
// GQA head mapping: q-head h reads kv-head h/(Hq/Hk). Single key so softmax=1 →
// each q-head's output is exactly its mapped kv-head's V.
// ===========================================================================
TEST_CASE("paged_attention GQA maps q-head h to kv-head h/(Hq/Hk)") {
  const int64_t Hq = 4, Hk = 2, D = 1, block_size = 2;
  const float scale = 1.0f;
  std::vector<float> q = {1, 2, 3, 4};  // [T=1, Hq=4, D=1]; values irrelevant (1 key)
  std::vector<float> k = {1, 1};        // [T=1, Hk=2, D=1]
  std::vector<float> v = {7, 9};        // kv0 → 7, kv1 → 9
  Queue qq = Q();

  const int64_t num_blocks = 1, page = Hk * D;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  Tensor tk = F32(k, {1, Hk, D}), tv = F32(v, {1, Hk, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  std::vector<int64_t> slots = {0};
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  std::vector<int32_t> block_table = {0};
  std::vector<int32_t> seq_lens = {1};
  std::vector<int32_t> qsl = {0, 1};
  std::vector<float> out(static_cast<size_t>(Hq * D), 0.0f);
  Tensor tq = F32(q, {1, Hq, D}), tp = F32(out, {1, Hq, D});
  Tensor tbt = I32(block_table, {1, 1}), tsl = I32(seq_lens, {1}), tqsl = I32(qsl, {2});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  CHECK(out[0] == doctest::Approx(7.0f));  // head 0 → kv 0
  CHECK(out[1] == doctest::Approx(7.0f));  // head 1 → kv 0
  CHECK(out[2] == doctest::Approx(9.0f));  // head 2 → kv 1
  CHECK(out[3] == doctest::Approx(9.0f));  // head 3 → kv 1
}

// ===========================================================================
// Block-spanning: a single decode query at seq_len 5 with block_size 2 reads
// keys across 3 blocks (positions 0-1 blk A, 2-3 blk B, 4 blk C) via the block
// table. Validated against the composed reference. Confirms j/block_size and
// j%block_size drive the block hop correctly.
// ===========================================================================
TEST_CASE("paged_attention block-spanning decode reads across multiple blocks") {
  const int64_t Hq = 2, Hk = 1, D = 4, block_size = 2;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t seqlen = 5;
  auto q = RandF32(static_cast<size_t>(Hq * D), 55);  // 1 decode token

  const int64_t num_blocks = 4, page = Hk * D, max_blocks = 3;
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  // Non-identity, non-contiguous blocks: pos 0-1 → blk 2, pos 2-3 → blk 0, pos 4 → blk 3.
  std::vector<int32_t> block_table = {2, 0, 3};
  auto set_cache = [&](std::vector<float>& cache, int64_t blk, int64_t off, uint32_t seed) {
    auto vals = RandF32(static_cast<size_t>(page), seed);
    const int64_t base = ((blk * block_size + off) * Hk) * D;
    for (int64_t i = 0; i < page; ++i) cache[static_cast<size_t>(base + i)] = vals[static_cast<size_t>(i)];
  };
  for (int64_t j = 0; j < seqlen; ++j) {
    const int64_t blk = block_table[static_cast<size_t>(j / block_size)];
    const int64_t off = j % block_size;
    set_cache(kc, blk, off, 1000u + static_cast<uint32_t>(j));
    set_cache(vc, blk, off, 2000u + static_cast<uint32_t>(j));
  }

  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seqlen)};
  std::vector<int32_t> qsl = {0, 1};  // one decode token
  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, true);

  Queue qq = Q();
  Tensor tq = F32(q, {1, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  std::vector<float> got(static_cast<size_t>(Hq * D), 0.0f);
  Tensor tp = F32(got, {1, Hq, D});
  vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{scale, true});

  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
}

// Port of tests/v1/attention/test_attention_backends.py::
// test_sliding_window_backend_correctness. One ragged batch combines prefill,
// decode and context-bearing chunked prefill; W straddles page boundaries.
TEST_CASE("paged_attention sliding-window decoder matches bottom-right causal mask") {
  const int64_t Hq = 4, Hk = 2, D = 8, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  std::vector<int32_t> qsl = {0, 5, 6, 9};
  std::vector<int32_t> seq_lens = {5, 9, 11};
  const int64_t num_tokens = qsl.back();
  const int64_t num_blocks = 12, max_blocks = 3;
  std::vector<int32_t> block_table = {
      2, 3, 0,   // five-token prefill
      4, 7, 8,   // decode at absolute position 8
      1, 9, 10,  // three-token chunk at positions 8..10
  };
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 3101);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3102);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3103);

  Queue qq = Q();
  Tensor tq = F32(q, {num_tokens, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {3, max_blocks});
  Tensor tsl = I32(seq_lens, {3});
  Tensor tqsl = I32(qsl, {4});

  for (const int32_t width : {1, 3, 4, 5}) {
    CAPTURE(width);
    const AttentionWindow window{width - 1, 0};
    const std::vector<float> ref = ComposedPagedRef(
        q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
        block_size, scale, /*causal=*/true, window);
    std::vector<float> got(ref.size(), 0.0f);
    Tensor tout = F32(got, {num_tokens, Hq, D});
    PagedAttentionArgs args{scale, true};
    args.window_size = window;
    vt::PagedAttention(qq, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
    for (size_t i = 0; i < ref.size(); ++i) {
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
    }
  }
}

// Port of test_sliding_window_encoder_backend_correctness. Non-causal encoder
// windows are symmetric, and the absolute query positions are bottom-right
// aligned when Q is shorter than K (context = seqlen-query_len).
TEST_CASE("paged_attention sliding-window encoder is symmetric and bottom-right aligned") {
  const int64_t Hq = 2, Hk = 1, D = 4, block_size = 4;
  const int64_t query_len = 5, seqlen = 7;
  const int64_t num_blocks = 4, max_blocks = 2;
  const float scale = 0.5f;
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(query_len)};
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seqlen)};
  std::vector<int32_t> block_table = {3, 1};
  auto q = RandF32(static_cast<size_t>(query_len * Hq * D), 3201);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3202);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), 3203);
  const AttentionWindow window{/*left=*/1, /*right=*/1};  // W=2
  const std::vector<float> ref = ComposedPagedRef(
      q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, /*causal=*/false, window);

  Queue qq = Q();
  Tensor tq = F32(q, {query_len, Hq, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(),
                      {num_blocks, block_size, Hk, D});
  Tensor tbt = I32(block_table, {1, max_blocks});
  Tensor tsl = I32(seq_lens, {1});
  Tensor tqsl = I32(qsl, {2});
  std::vector<float> got(ref.size(), 0.0f);
  Tensor tout = F32(got, {query_len, Hq, D});
  PagedAttentionArgs args{scale, false};
  args.window_size = window;
  vt::PagedAttention(qq, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-5));
  }

  // The first query is at absolute p=2 and may see keys 1..3 only. Poison key
  // 0 and keys 4..6; its output must stay bit-identical while later rows may use
  // some of those keys. This directly proves both local bounds are enforced.
  std::vector<float> poisoned_v = vc;
  for (const int64_t key : {0, 4, 5, 6}) {
    const int64_t blk = block_table[static_cast<size_t>(key / block_size)];
    const int64_t off = key % block_size;
    const int64_t base = (blk * block_size + off) * Hk * D;
    for (int64_t e = 0; e < Hk * D; ++e) {
      poisoned_v[static_cast<size_t>(base + e)] = 100000.0f + static_cast<float>(e);
    }
  }
  Tensor poisoned_tvc = Contig(poisoned_v.data(), DType::kF32, Cpu(),
                               {num_blocks, block_size, Hk, D});
  std::vector<float> poisoned_out(got.size(), 0.0f);
  Tensor poisoned_tout = F32(poisoned_out, {query_len, Hq, D});
  vt::PagedAttention(qq, poisoned_tout, tq, tkc, poisoned_tvc, tbt, tsl,
                     tqsl, args);
  for (int64_t i = 0; i < Hq * D; ++i) {
    CHECK(poisoned_out[static_cast<size_t>(i)] == got[static_cast<size_t>(i)]);
  }
}

TEST_CASE("paged_attention validates shapes/args") {
  const int64_t block_size = 2, Hk = 1, D = 2;
  std::vector<float> kc(static_cast<size_t>(1 * block_size * Hk * D), 0.0f);
  std::vector<float> q(static_cast<size_t>(1 * 2 * D), 0.0f);
  std::vector<float> out(static_cast<size_t>(1 * 2 * D), 0.0f);
  std::vector<int32_t> bt = {0}, sl = {1}, qsl = {0, 1};
  Queue qq = Q();
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {1, block_size, Hk, D});
  Tensor tvc = Contig(kc.data(), DType::kF32, Cpu(), {1, block_size, Hk, D});
  Tensor tbt = I32(bt, {1, 1}), tsl = I32(sl, {1}), tqsl = I32(qsl, {2});

  // Hq not a multiple of Hk (3 q-heads, 1 kv-head is fine; use Hk=2 mismatch).
  {
    std::vector<float> kc2(static_cast<size_t>(1 * block_size * 2 * D), 0.0f);
    Tensor tkc2 = Contig(kc2.data(), DType::kF32, Cpu(), {1, block_size, 2, D});
    Tensor tvc2 = Contig(kc2.data(), DType::kF32, Cpu(), {1, block_size, 2, D});
    std::vector<float> q3(static_cast<size_t>(3 * D), 0.0f), o3(static_cast<size_t>(3 * D), 0.0f);
    Tensor tq3 = F32(q3, {1, 3, D}), to3 = F32(o3, {1, 3, D});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, to3, tq3, tkc2, tvc2, tbt, tsl, tqsl, PagedAttentionArgs{1.0f, true}),
        std::runtime_error);
  }
  // scale must be > 0.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, PagedAttentionArgs{0.0f, true}),
        std::runtime_error);
  }
  // query_start_loc wrong length (must be num_reqs+1).
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int32_t> bad = {0};
    Tensor tqsl_bad = I32(bad, {1});
    CHECK_THROWS_AS(vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl_bad,
                                       PagedAttentionArgs{1.0f, true}),
                    std::runtime_error);
  }
  // seq_lens must be i32.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int64_t> sl64 = {1};
    Tensor tsl_bad = Contig(sl64.data(), DType::kI64, Cpu(), {1});
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl_bad, tqsl, PagedAttentionArgs{1.0f, true}),
        std::runtime_error);
  }
  // A present window is an actual local window, so both inclusive distances
  // must be non-negative (full attention is represented by nullopt).
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    PagedAttentionArgs bad{1.0f, true};
    bad.window_size = AttentionWindow{-1, 0};
    CHECK_THROWS_AS(
        vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl, tqsl, bad),
        std::runtime_error);
  }
  // #1390 / #1394: A BLOCK TABLE THAT CANNOT ADDRESS `seq_lens` IS AN
  // OUT-OF-BOUNDS READ, not a wrong number, and the refusal that stops it had
  // no test of its own. `tbt` is one column at block_size 2, so it addresses
  // positions 0..1; seq_len 3 makes the kernel read `block_table[0, 1]`, one
  // int32 past the row, and multiply that byte pattern by the cache's block
  // stride. Measured at `5f68e60df`: that read returned 0 when the crashing
  // case ran alone and 1021459670 when one earlier case in its file had run
  // first, so the same source faulted at 0x533fbc625590 in the natural suite
  // order and looked correct in isolation.
  //
  // WHERE THE BOUND LIVES, AND WHERE THIS CASE ENTERS IT. The refusal is in the
  // CPU kernel, `src/vt/cpu/cpu_paged_attn.cpp`, beside the read it protects,
  // and it holds for the CPU backend ONLY. This case is written against
  // `vt::PagedAttention` because that is the call a caller writes and the one
  // entry point all six registered backends share -- NOT because the bound is
  // enforced there for all six. It is not: the same unbounded index sits in 19
  // device sites, and #1406 records why the seam cannot hold one bound for them
  // today.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int32_t> sl_over = {3};  // needs 2 columns, the table has 1
    Tensor tsl_over = I32(sl_over, {1});
    CHECK_THROWS_AS(vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl_over, tqsl,
                                       PagedAttentionArgs{1.0f, true}),
                    std::runtime_error);
    // And a batch that FITS still runs: seq_len 2 == 1 column x block_size 2.
    // Without this line the case above is satisfied by a refusal that refuses
    // everything.
    std::vector<int32_t> sl_fit = {2};
    Tensor tsl_fit = I32(sl_fit, {1});
    CHECK_NOTHROW(vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt, tsl_fit, tqsl,
                                     PagedAttentionArgs{1.0f, true}));
  }
  // A PADDED ROW IS NOT A CALLER MISTAKE. Request 1 carries no query tokens, so
  // no kernel reads its `seq_lens` entry and its over-long context addresses
  // nothing. A bound that read every row would reject a batch that runs
  // correctly today, which is #1394's second measurement and the reason its
  // refusal skips a row the token loop never visits.
  {
    Tensor tq = F32(q, {1, 2, D}), tp = F32(out, {1, 2, D});
    std::vector<int32_t> bt2 = {0, 0}, sl2 = {1, 99}, qsl2 = {0, 1, 1};
    Tensor tbt2 = I32(bt2, {2, 1}), tsl2 = I32(sl2, {2}), tqsl2 = I32(qsl2, {3});
    CHECK_NOTHROW(vt::PagedAttention(qq, tp, tq, tkc, tvc, tbt2, tsl2, tqsl2,
                                     PagedAttentionArgs{1.0f, true}));
  }
}

// ===========================================================================
// CUDA parity: the CUDA paged kernel must match the CPU reference on random
// inputs (batched varlen + block-spanning). Guarded by HasCuda; dgx-pending on
// CPU-only boxes.
// ===========================================================================
namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

void CheckCudaWindowCase(const std::vector<int32_t>& qsl,
                         const std::vector<int32_t>& seq_lens, bool causal,
                         AttentionWindow window, uint32_t seed) {
  const int64_t Hq = 4, Hk = 2, D = 32, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t num_reqs = static_cast<int64_t>(seq_lens.size());
  const int64_t num_tokens = qsl.back();
  const int32_t max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
  const int64_t max_blocks = (max_seq + block_size - 1) / block_size;
  const int64_t num_blocks = num_reqs * max_blocks;
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t b = 0; b < max_blocks; ++b) {
      block_table[static_cast<size_t>(r * max_blocks + b)] =
          static_cast<int32_t>(r * max_blocks + b);
    }
  }
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), seed);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), seed + 1);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * Hk * D), seed + 2);
  const std::vector<float> ref = ComposedPagedRef(
      q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, causal, window);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32,
                   {num_blocks, block_size, Hk, D}, kc.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32,
                   {num_blocks, block_size, Hk, D}, vc.data());
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks},
                   block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  PagedAttentionArgs args{scale, causal};
  args.window_size = window;
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), dkc.tensor(),
                     dvc.tensor(), dbt.tensor(), dsl.tensor(), dqsl.tensor(),
                     args);
  std::vector<float> got(ref.size(), 0.0f);
  dout.Download(g.q, got.data());
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
  }
}

}  // namespace

TEST_CASE("paged_attention CUDA matches CPU (batched varlen, GQA, block-spanning)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention CUDA parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 32, block_size = 4;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // 3 reqs: prefill len 6 (seq 6), decode seq 9, prefill len 2 (seq 11 chunk).
  std::vector<int32_t> qsl = {0, 6, 7, 9};
  std::vector<int32_t> seq_lens = {6, 9, 11};
  const int64_t num_tokens = 9;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 16, page = Hk * D, max_blocks = 3;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 4242);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 71);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 91);
  // block tables (num_reqs x max_blocks), disjoint-ish blocks.
  std::vector<int32_t> block_table = {5, 0, 0,  /* req0 seq6 → 2 blocks: 5,0 */
                                      2, 7, 0,  /* req1 seq9 → 3 blocks: 2,7,0 */
                                      9, 3, 6}; /* req2 seq11 → 3 blocks: 9,3,6 */

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, true);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, kc.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, vc.data());
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), dkc.tensor(), dvc.tensor(), dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
}

TEST_CASE("paged_attention CUDA sliding-window masks match upstream decoder and encoder vectors") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping sliding-window CUDA parity (dgx-pending)");
    return;
  }

  // Mixed prefill/decode/chunked-prefill takes the portable tiled prefill path.
  for (const int32_t width : {1, 3, 4, 5}) {
    CAPTURE(width);
    CheckCudaWindowCase({0, 5, 6, 9}, {5, 9, 11}, /*causal=*/true,
                        AttentionWindow{width - 1, 0},
                        5000u + static_cast<uint32_t>(width));
  }
  CheckCudaWindowCase({0, 129}, {129}, /*causal=*/true,
                      AttentionWindow{4095, 0}, 5050);
  // All query lengths are one, so this exercises the graph-safe decode path.
  CheckCudaWindowCase({0, 1, 2}, {9, 11}, /*causal=*/true,
                      AttentionWindow{4, 0}, 5100);
  // Q<K bottom-right alignment with the symmetric encoder window.
  CheckCudaWindowCase({0, 5}, {7}, /*causal=*/false,
                      AttentionWindow{2, 2}, 5200);
}

// ===========================================================================
// bf16 TENSOR-CORE (WMMA) prefill parity. The WMMA flash path fires only for a
// bf16 KV cache (the deployment path — vLLM's bf16 flash_attn store); it casts
// the f32 query to bf16 and runs QKᵀ / P·V on the tensor cores with f32 online
// softmax. Validate at the gate config (head_dim 256, GQA 16q/2kv, multi-tile
// prefill) against the f32 composed reference computed on the SAME bf16-rounded
// K/V — so the residual is query-bf16 + bf16-matmul rounding only (bf16 tol).
// Guarded by HasCuda; dgx-pending on CPU-only boxes.
// ===========================================================================
namespace {
// f32 -> bf16 round-to-nearest-even (host; the test .cpp is not nvcc-compiled).
inline uint16_t F32ToBf16Bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>((x + rounding) >> 16);
}
inline float Bf16BitsToF32(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}
}  // namespace

TEST_CASE("paged_attention CUDA WMMA (bf16 cache) matches f32 ref at head_dim 256") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention WMMA bf16 parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};

  // Round K/V to bf16 (bit-identical to what the kernel reads), and build the
  // f32 reference on those same values so only the compute rounding remains.
  std::vector<uint16_t> kc_b(kc.size()), vc_b(vc.size());
  std::vector<float> kc_r(kc.size()), vc_r(vc.size());
  for (size_t i = 0; i < kc.size(); ++i) {
    kc_b[i] = F32ToBf16Bits(kc[i]);
    kc_r[i] = Bf16BitsToF32(kc_b[i]);
    vc_b[i] = F32ToBf16Bits(vc[i]);
    vc_r[i] = Bf16BitsToF32(vc_b[i]);
  }
  std::vector<float> ref = ComposedPagedRef(q, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl,
                                            Hq, Hk, D, block_size, scale, true);

  // Build the ENGINE cache layout: a single (num_blocks, 2, block_size, Hk, D)
  // bf16 allocation; K = unbind(1)[0], V = unbind(1)[1] (block stride 2*bs*Hk*D,
  // V at element offset bs*Hk*D). This is the M1.6 layout trap — the strided
  // slice the runner really feeds PagedAttention (KvSlice, qwen3_5.cpp:778).
  const int64_t within = block_size * Hk * D;
  std::vector<uint16_t> combined(static_cast<size_t>(num_blocks * 2 * within), 0);
  for (int64_t b = 0; b < num_blocks; ++b)
    for (int64_t e = 0; e < within; ++e) {
      combined[static_cast<size_t>((b * 2 + 0) * within + e)] = kc_b[static_cast<size_t>(b * within + e)];
      combined[static_cast<size_t>((b * 2 + 1) * within + e)] = vc_b[static_cast<size_t>(b * within + e)];
    }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  // Allocate as rank-1 (kMaxRank=4; the true (nb,2,bs,Hk,D) is rank 5) then build
  // strided rank-4 K/V views into it.
  DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks * 2 * within}, combined.data());
  auto SliceView = [&](int which) {
    Tensor t = dcache.tensor();
    t.data = static_cast<char*>(t.data) +
             static_cast<size_t>(which) * static_cast<size_t>(within) * vt::SizeOf(DType::kBF16);
    t.rank = 4;
    t.shape[0] = num_blocks;
    t.shape[1] = block_size;
    t.shape[2] = Hk;
    t.shape[3] = D;
    t.stride[0] = 2 * within;
    t.stride[1] = Hk * D;
    t.stride[2] = D;
    t.stride[3] = 1;
    return t;
  };
  Tensor kview = SliceView(0);
  Tensor vview = SliceView(1);
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  double max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    max_abs = std::max(max_abs, std::abs(static_cast<double>(got[i]) - ref[i]));
  MESSAGE("WMMA bf16 prefill max_abs_err vs f32 ref = " << max_abs);
  CHECK(max_abs < 5e-2);

  // The same optimized WMMA ladder must honor the local lower bound; FA-2 is
  // ineligible here because query/out are f32.
  const AttentionWindow local_window{31, 0};
  const std::vector<float> local_ref = ComposedPagedRef(
      q, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
      block_size, scale, /*causal=*/true, local_window);
  DeviceTensor local_out(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  PagedAttentionArgs local_args{scale, true};
  local_args.window_size = local_window;
  vt::PagedAttention(g.q, local_out.tensor(), dq.tensor(), kview, vview,
                     dbt.tensor(), dsl.tensor(), dqsl.tensor(), local_args);
  std::vector<float> local_got(local_ref.size(), 0.0f);
  local_out.Download(g.q, local_got.data());
  double local_max_abs = 0.0;
  for (size_t i = 0; i < local_ref.size(); ++i) {
    local_max_abs = std::max(
        local_max_abs,
        std::abs(static_cast<double>(local_got[i]) - local_ref[i]));
  }
  MESSAGE("WMMA sliding-window max_abs_err vs f32 ref = " << local_max_abs);
  CHECK(local_max_abs < 5e-2);
}

// ===========================================================================
// CUDA parity at the GATE-MODEL config: head_dim 256, GQA 16q/2kv. This is the
// M2.4 flash prefill path's real shape — head_dim 256 means the warp-per-row
// kernel distributes 8 elements per lane (epl=8), and prefill lengths that span
// several BN key-tiles exercise the cross-tile online-softmax rescale. The
// smaller D=32 case above only covers epl=1, so this pins the production path.
// Guarded by HasCuda; dgx-pending on CPU-only boxes.
// ===========================================================================
TEST_CASE("paged_attention CUDA matches CPU at head_dim 256 (gate config, epl=8)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping paged_attention D=256 parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // 3 reqs: a long prefill (seq 100, spans multiple BN=32 key-tiles), a decode
  // at seq 133, and a short prefill chunk (len 3, seq 140).
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  // block tables (num_reqs x max_blocks); non-identity, disjoint blocks.
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,   // req0 seq100 → 7 blocks
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,   // req1 seq133 → 9 blocks
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};  // req2 seq140 → 9 blocks

  std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens, qsl, Hq,
                                            Hk, D, block_size, scale, true);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {num_tokens, Hq, D}, q.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, kc.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32, {num_blocks, block_size, Hk, D}, vc.data());
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout.tensor(), dq.tensor(), dkc.tensor(), dvc.tensor(), dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<float> got(static_cast<size_t>(num_tokens * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
}

// ===========================================================================
// Vendored FlashAttention-2 prefill (VLLM_CPP_FLASH_ATTN build + VT_FA2_PREFILL
// runtime): the exact flash_fwd_splitkv kernel vLLM runs on GB10, engaged for
// the natively-bf16 combo (bf16 query + bf16 KV + bf16 out, head_dim 256,
// prefill). Validated at the gate config against the f32 composed reference on
// the SAME bf16-rounded q/K/V (residual = FA-2's bf16 tensor-core matmuls +
// f32-softmax rounding + the bf16 output round, same tolerance class as the
// WMMA bf16 test). Also pins the sync-free host-metadata path: passing
// query_start_loc_host + a max_seq_len UPPER BOUND must be bit-identical to the
// fallback (D2H+sync) run — the host values size grids only, never geometry.
// ===========================================================================
#ifdef VLLM_CPP_FLASH_ATTN
namespace {
// Set/restore an env var for the duration of a scope (exception-safe).
struct EnvGuard {
  const char* name;
  std::optional<std::string> previous;
  explicit EnvGuard(const char* n, const char* value) : name(n) {
    if (const char* old = std::getenv(name); old != nullptr) previous = old;
    setenv(name, value, 1);
  }
  ~EnvGuard() {
    if (previous.has_value()) {
      setenv(name, previous->c_str(), 1);
    } else {
      unsetenv(name);
    }
  }
  EnvGuard(const EnvGuard&) = delete;
  EnvGuard& operator=(const EnvGuard&) = delete;
};
}  // namespace

TEST_CASE("paged_attention CUDA FA-2 prefill (bf16 q/kv/out) matches f32 ref at head_dim 256") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 prefill parity (dgx-pending)");
    return;
  }
  const int64_t Hq = 16, Hk = 2, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // Ragged 3-req batch at the gate config: a long prefill, a decode row, and a
  // short chunked-prefill tail (context 137) — the mixed batch LaunchPaged
  // treats as one prefill call.
  std::vector<int32_t> qsl = {0, 100, 101, 104};
  std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104;
  const int64_t num_reqs = 3;
  const int64_t num_blocks = 64, page = Hk * D, max_blocks = 9;
  auto qf = RandF32(static_cast<size_t>(num_tokens * Hq * D), 2024);
  auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 137);
  auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page), 179);
  std::vector<int32_t> block_table = {5,  0,  11, 3,  0,  0, 0, 0, 0,
                                      2,  17, 9,  20, 1,  8, 0, 0, 0,
                                      30, 4,  22, 15, 6, 19, 7, 12, 0};

  // Round q AND K/V to bf16 (bit-identical to what FA-2 reads) and build the
  // f32 reference on those same values, so only FA-2's compute rounding + the
  // bf16 output round remain.
  std::vector<uint16_t> q_b(qf.size()), kc_b(kc.size()), vc_b(vc.size());
  std::vector<float> q_r(qf.size()), kc_r(kc.size()), vc_r(vc.size());
  for (size_t i = 0; i < qf.size(); ++i) {
    q_b[i] = F32ToBf16Bits(qf[i]);
    q_r[i] = Bf16BitsToF32(q_b[i]);
  }
  for (size_t i = 0; i < kc.size(); ++i) {
    kc_b[i] = F32ToBf16Bits(kc[i]);
    kc_r[i] = Bf16BitsToF32(kc_b[i]);
    vc_b[i] = F32ToBf16Bits(vc[i]);
    vc_r[i] = Bf16BitsToF32(vc_b[i]);
  }
  std::vector<float> ref = ComposedPagedRef(q_r, kc_r, vc_r, block_table, max_blocks, seq_lens,
                                            qsl, Hq, Hk, D, block_size, scale, true);

  // ENGINE cache layout: one (num_blocks, 2, block_size, Hk, D) bf16 allocation,
  // K/V = the dim-1 unbind slices (block stride 2*bs*Hk*D) — what the runner
  // really feeds PagedAttention (KvSlice).
  const int64_t within = block_size * Hk * D;
  std::vector<uint16_t> combined(static_cast<size_t>(num_blocks * 2 * within), 0);
  for (int64_t b = 0; b < num_blocks; ++b)
    for (int64_t e = 0; e < within; ++e) {
      combined[static_cast<size_t>((b * 2 + 0) * within + e)] =
          kc_b[static_cast<size_t>(b * within + e)];
      combined[static_cast<size_t>((b * 2 + 1) * within + e)] =
          vc_b[static_cast<size_t>(b * within + e)];
    }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kBF16, {num_tokens, Hq, D}, q_b.data());
  DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks * 2 * within}, combined.data());
  auto SliceView = [&](int which) {
    Tensor t = dcache.tensor();
    t.data = static_cast<char*>(t.data) +
             static_cast<size_t>(which) * static_cast<size_t>(within) * vt::SizeOf(DType::kBF16);
    t.rank = 4;
    t.shape[0] = num_blocks;
    t.shape[1] = block_size;
    t.shape[2] = Hk;
    t.shape[3] = D;
    t.stride[0] = 2 * within;
    t.stride[1] = Hk * D;
    t.stride[2] = D;
    t.stride[3] = 1;
    return t;
  };
  Tensor kview = SliceView(0);
  Tensor vview = SliceView(1);
  DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
  DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
  DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());

  EnvGuard fa2_on("VT_FA2_PREFILL", "1");

  // Run 1: no host metadata -> the launcher's D2H+sync fallback sizes the grid.
  DeviceTensor dout1(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout1.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
  std::vector<uint16_t> got1(static_cast<size_t>(num_tokens * Hq * D), 0);
  dout1.Download(g.q, got1.data());

  // Run 2: sync-free host metadata, with max_seq_len an UPPER BOUND (160 > 140)
  // — must be bit-identical to run 1 (host values size grids, not geometry).
  PagedAttentionArgs host_args{scale, true};
  host_args.query_start_loc_host = qsl.data();
  host_args.max_seq_len = 160;
  DeviceTensor dout2(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
  vt::PagedAttention(g.q, dout2.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                     dsl.tensor(), dqsl.tensor(), host_args);
  std::vector<uint16_t> got2(static_cast<size_t>(num_tokens * Hq * D), 0);
  dout2.Download(g.q, got2.data());

  double max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs,
                       std::abs(static_cast<double>(Bf16BitsToF32(got1[i])) - ref[i]));
  }
  MESSAGE("FA-2 bf16 prefill max_abs_err vs f32 ref = " << max_abs);
  CHECK(max_abs < 5e-2);

  size_t mism = 0;
  for (size_t i = 0; i < got1.size(); ++i)
    if (got1[i] != got2[i]) ++mism;
  MESSAGE("FA-2 host-metadata vs fallback mismatches = " << mism);
  CHECK(mism == 0);

  // Finite windows must dispatch FA-2's LOCAL specialization. In particular,
  // the decoder keeps causal semantics via right=0 while is_causal is normalized
  // false for the local template; leaving the causal template selected would
  // silently ignore the lower bound in the pinned dependency.
  auto check_local = [&](bool causal, AttentionWindow window,
                         const char* label) {
    const std::vector<float> local_ref = ComposedPagedRef(
        q_r, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl, Hq, Hk, D,
        block_size, scale, causal, window);
    PagedAttentionArgs local_args{scale, causal};
    local_args.window_size = window;
    DeviceTensor local_out(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
    vt::PagedAttention(g.q, local_out.tensor(), dq.tensor(), kview, vview,
                       dbt.tensor(), dsl.tensor(), dqsl.tensor(), local_args);
    std::vector<uint16_t> local_got(local_ref.size(), 0);
    local_out.Download(g.q, local_got.data());
    double local_max_abs = 0.0;
    for (size_t i = 0; i < local_ref.size(); ++i) {
      local_max_abs = std::max(
          local_max_abs,
          std::abs(static_cast<double>(Bf16BitsToF32(local_got[i])) -
                   local_ref[i]));
    }
    INFO(label);
    CHECK(local_max_abs < 5e-2);
  };
  check_local(/*causal=*/true, AttentionWindow{31, 0},
              "FA-2 causal decoder local window");
  check_local(/*causal=*/false, AttentionWindow{31, 31},
              "FA-2 symmetric encoder local window");
}

// Qwen3-dense VARLEN d128 prefill parity (the DOMINANT TTFT lever): vLLM runs
// Qwen3 prefill on the same flash_attn_varlen_func FA2 family, so the generalized
// LaunchPrefillFA2Bf16 (num_splits=1, Split=false) must reproduce the paged
// causal GQA reduction. Validate at both Qwen3 ratios (0.6B Hq/Hkv=16/8, 4B 32/8)
// against the f32 composed reference on the SAME bf16-rounded q/K/V; pin the
// sync-free host-metadata path (bit-identical to the D2H+sync fallback), and the
// same-binary opt-out (VT_FA2_PREFILL_QWEN3=0 -> scalar CUDA-core prefill).
TEST_CASE("paged_attention CUDA FA-2 prefill (bf16 q/kv/out) matches f32 ref at head_dim 128") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 d128 prefill parity (dgx-pending)");
    return;
  }
  const int64_t D = 128, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  // Ragged 3-req prefill batch: a long prefill, a decode row, and a short
  // chunked-prefill tail — the mixed batch LaunchPaged treats as one prefill.
  const std::vector<int32_t> qsl = {0, 100, 101, 104};
  const std::vector<int32_t> seq_lens = {100, 133, 140};
  const int64_t num_tokens = 104, num_reqs = 3;
  // kBlockN=128 for d128: covers up to ceil(140/128)*128 = 256 keys = 16 pages;
  // size the block table to that so no column is read past its row (memcheck).
  const int64_t max_blocks = 16, num_blocks = 64;
  // ★ {32, 2} IS NEMOTRON-3.5-LIGHTNING, AND IT WAS NOT HERE (#1157).
  // Every case in this file measured ngroups 2 (16/8) and 4 (32/8), the two
  // Qwen3-dense gate configs the varlen d128 decode path was written for. The
  // path is DEFAULT ON for ANY bf16 causal pure-decode at head_dim 128, so the
  // first model to arrive with a different ratio reaches an untested grid — and
  // NemotronH-3.5-Lightning-30B is 32 query heads over 2 KV heads, ngroups 16,
  // four times the widest group count ever measured here. Its A3 token gate
  // failed 6/96 on the device while the same binary decodes the same checkpoint
  // token-exact on CPU, which is what sent the search here.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{16, 8},
                            std::pair<int64_t, int64_t>{32, 8},
                            std::pair<int64_t, int64_t>{32, 2}}) {
    const int64_t Hq = ratio.first, Hk = ratio.second, page = Hk * D;
    CAPTURE(Hq);
    auto qf = RandF32(static_cast<size_t>(num_tokens * Hq * D),
                      5024U + static_cast<uint32_t>(Hq));
    auto kc = RandF32(static_cast<size_t>(num_blocks * block_size * page),
                      337U + static_cast<uint32_t>(Hq));
    auto vc = RandF32(static_cast<size_t>(num_blocks * block_size * page),
                      379U + static_cast<uint32_t>(Hq));
    std::vector<int32_t> block_table(static_cast<size_t>(num_reqs * max_blocks));
    for (int64_t r = 0; r < num_reqs; ++r)
      for (int64_t b = 0; b < max_blocks; ++b)
        block_table[static_cast<size_t>(r * max_blocks + b)] =
            static_cast<int32_t>((r * 13 + b * 7) % num_blocks);

    std::vector<uint16_t> q_b(qf.size()), kc_b(kc.size()), vc_b(vc.size());
    std::vector<float> q_r(qf.size()), kc_r(kc.size()), vc_r(vc.size());
    for (size_t i = 0; i < qf.size(); ++i) {
      q_b[i] = F32ToBf16Bits(qf[i]);
      q_r[i] = Bf16BitsToF32(q_b[i]);
    }
    for (size_t i = 0; i < kc.size(); ++i) {
      kc_b[i] = F32ToBf16Bits(kc[i]);
      kc_r[i] = Bf16BitsToF32(kc_b[i]);
      vc_b[i] = F32ToBf16Bits(vc[i]);
      vc_r[i] = Bf16BitsToF32(vc_b[i]);
    }
    const std::vector<float> ref =
        ComposedPagedRef(q_r, kc_r, vc_r, block_table, max_blocks, seq_lens, qsl,
                         Hq, Hk, D, block_size, scale, true);

    const int64_t within = block_size * Hk * D;
    std::vector<uint16_t> combined(static_cast<size_t>(num_blocks * 2 * within), 0);
    for (int64_t b = 0; b < num_blocks; ++b)
      for (int64_t e = 0; e < within; ++e) {
        combined[static_cast<size_t>((b * 2 + 0) * within + e)] =
            kc_b[static_cast<size_t>(b * within + e)];
        combined[static_cast<size_t>((b * 2 + 1) * within + e)] =
            vc_b[static_cast<size_t>(b * within + e)];
      }

    Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard g(gpu);
    DeviceTensor dq(gpu, g.q, DType::kBF16, {num_tokens, Hq, D}, q_b.data());
    DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks * 2 * within}, combined.data());
    auto SliceView = [&](int which) {
      Tensor t = dcache.tensor();
      t.data = static_cast<char*>(t.data) +
               static_cast<size_t>(which) * static_cast<size_t>(within) *
                   vt::SizeOf(DType::kBF16);
      t.rank = 4;
      t.shape[0] = num_blocks;
      t.shape[1] = block_size;
      t.shape[2] = Hk;
      t.shape[3] = D;
      t.stride[0] = 2 * within;
      t.stride[1] = Hk * D;
      t.stride[2] = D;
      t.stride[3] = 1;
      return t;
    };
    Tensor kview = SliceView(0), vview = SliceView(1);
    DeviceTensor dbt(gpu, g.q, DType::kI32, {num_reqs, max_blocks}, block_table.data());
    DeviceTensor dsl(gpu, g.q, DType::kI32, {num_reqs}, seq_lens.data());
    DeviceTensor dqsl(gpu, g.q, DType::kI32, {num_reqs + 1}, qsl.data());

    EnvGuard fa2_on("VT_FA2_PREFILL_QWEN3", "1");
    // Run 1: no host metadata -> the launcher's D2H+sync fallback sizes the grid.
    DeviceTensor dout1(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
    vt::PagedAttention(g.q, dout1.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                       dsl.tensor(), dqsl.tensor(), PagedAttentionArgs{scale, true});
    std::vector<uint16_t> got1(static_cast<size_t>(num_tokens * Hq * D), 0);
    dout1.Download(g.q, got1.data());

    // Run 2: sync-free host metadata, max_seq_len an UPPER BOUND (160 > 140) —
    // must be bit-identical to run 1 (host values size grids, not geometry).
    PagedAttentionArgs host_args{scale, true};
    host_args.query_start_loc_host = qsl.data();
    host_args.max_seq_len = 160;
    DeviceTensor dout2(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
    vt::PagedAttention(g.q, dout2.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                       dsl.tensor(), dqsl.tensor(), host_args);
    std::vector<uint16_t> got2(static_cast<size_t>(num_tokens * Hq * D), 0);
    dout2.Download(g.q, got2.data());

    double max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
      max_abs = std::max(max_abs,
                         std::abs(static_cast<double>(Bf16BitsToF32(got1[i])) - ref[i]));
    MESSAGE("FA-2 bf16 d128 prefill max_abs_err vs f32 ref = " << max_abs);
    CHECK(max_abs < 5e-2);
    size_t mism = 0;
    for (size_t i = 0; i < got1.size(); ++i)
      if (got1[i] != got2[i]) ++mism;
    CHECK(mism == 0);

    // Same-binary opt-out: toggle 0 -> scalar CUDA-core prefill, still correct.
    EnvGuard fa2_off("VT_FA2_PREFILL_QWEN3", "0");
    DeviceTensor dout3(gpu, g.q, DType::kBF16, {num_tokens, Hq, D});
    vt::PagedAttention(g.q, dout3.tensor(), dq.tensor(), kview, vview, dbt.tensor(),
                       dsl.tensor(), dqsl.tensor(), host_args);
    std::vector<uint16_t> got3(static_cast<size_t>(num_tokens * Hq * D), 0);
    dout3.Download(g.q, got3.data());
    double fb_max = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
      fb_max = std::max(fb_max,
                        std::abs(static_cast<double>(Bf16BitsToF32(got3[i])) - ref[i]));
    CHECK(fb_max < 5e-2);
  }
}

namespace {

// Host/device fixture for the pure-decode vectors ported from
// tests/kernels/attention/test_flash_attn.py::test_varlen_with_paged_kv
// (vLLM 702f481, lines 95-217). Every request has qlen=1. Block-table rows use
// shared, permuted physical blocks so long 2k-token vectors stay compact while
// retaining nontrivial paged addressing and head-specific data.
struct Fa2DecodeCase {
  int64_t hq;
  int64_t hk;
  int64_t d = 256;
  int64_t block_size = 16;
  int64_t batch;
  // W10 (#1857): query rows per request. 1 is the shipped pure-decode shape;
  // q>1 is the uniform speculative verify (each request's rows are the LAST
  // q_len positions of its seq_len, matching the verify's bottom-right causal
  // geometry). The query/qsl sizing below follows it; everything else is
  // shape-agnostic.
  int64_t q_len = 1;
  int64_t max_blocks;
  int64_t num_blocks;
  float scale = std::pow(256.0F, -0.5F);
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> qsl;
  std::vector<int32_t> block_table;
  std::vector<uint16_t> query_bf16;
  std::vector<uint16_t> combined_cache;
  std::vector<float> query_rounded;
  std::vector<float> key_rounded;
  std::vector<float> value_rounded;

  Fa2DecodeCase(int64_t query_heads, int64_t kv_heads,
                std::vector<int32_t> lengths, uint32_t seed,
                int64_t capacity_blocks = 0, int64_t head_dim = 256,
                int64_t query_rows_per_req = 1)
      : hq(query_heads),
        hk(kv_heads),
        batch(static_cast<int64_t>(lengths.size())),
        q_len(query_rows_per_req),
        seq_lens(std::move(lengths)) {
    d = head_dim;
    scale = std::pow(static_cast<float>(head_dim), -0.5F);
    const int32_t max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
    max_blocks = std::max<int64_t>(capacity_blocks,
                                   (max_seq + block_size - 1) / block_size);
    // FA-2's split-KV kernel resolves the last (partial) kBlockN tile at
    // kGmemRowsPerThread granularity, so it can index a block_table column up to
    // the kBlockN-rounded page count (kBlockN=128 for d128, 64 for d256 => up to
    // 128/block_size pages). A real KV allocator (vLLM) sizes the block table to
    // cover that rounding; mirror it here so short sequences don't read a column
    // past a tightly-sized row (the extra pages are masked out of the softmax).
    const int64_t pages_per_tile = 128 / block_size;  // >= any kBlockN/page_size
    if (pages_per_tile > 0)
      max_blocks = (max_blocks + pages_per_tile - 1) / pages_per_tile * pages_per_tile;
    num_blocks = max_blocks + 17;
    qsl.resize(static_cast<size_t>(batch + 1));
    for (int64_t i = 0; i <= batch; ++i)
      qsl[static_cast<size_t>(i)] = static_cast<int32_t>(i * q_len);

    block_table.resize(static_cast<size_t>(batch * max_blocks));
    for (int64_t r = 0; r < batch; ++r) {
      for (int64_t b = 0; b < max_blocks; ++b) {
        block_table[static_cast<size_t>(r * max_blocks + b)] =
            static_cast<int32_t>((r * 13 + b * 7) % num_blocks);
      }
    }

    const auto qf = RandF32(static_cast<size_t>(batch * q_len * hq * d), seed);
    const auto kf = RandF32(
        static_cast<size_t>(num_blocks * block_size * hk * d), seed + 1);
    const auto vf = RandF32(kf.size(), seed + 2);
    query_bf16.resize(qf.size());
    query_rounded.resize(qf.size());
    std::vector<uint16_t> key_bf16(kf.size()), value_bf16(vf.size());
    key_rounded.resize(kf.size());
    value_rounded.resize(vf.size());
    for (size_t i = 0; i < qf.size(); ++i) {
      query_bf16[i] = F32ToBf16Bits(qf[i]);
      query_rounded[i] = Bf16BitsToF32(query_bf16[i]);
    }
    for (size_t i = 0; i < kf.size(); ++i) {
      key_bf16[i] = F32ToBf16Bits(kf[i]);
      value_bf16[i] = F32ToBf16Bits(vf[i]);
      key_rounded[i] = Bf16BitsToF32(key_bf16[i]);
      value_rounded[i] = Bf16BitsToF32(value_bf16[i]);
    }

    const int64_t within = block_size * hk * d;
    combined_cache.assign(static_cast<size_t>(num_blocks * 2 * within), 0);
    for (int64_t block = 0; block < num_blocks; ++block) {
      for (int64_t e = 0; e < within; ++e) {
        const size_t source = static_cast<size_t>(block * within + e);
        combined_cache[static_cast<size_t>((block * 2) * within + e)] =
            key_bf16[source];
        combined_cache[static_cast<size_t>((block * 2 + 1) * within + e)] =
            value_bf16[source];
      }
    }
  }

  Tensor CacheView(DeviceTensor& cache, int which) const {
    const int64_t within = block_size * hk * d;
    Tensor view = cache.tensor();
    view.data = static_cast<char*>(view.data) +
                static_cast<size_t>(which * within) * vt::SizeOf(DType::kBF16);
    view.rank = 4;
    view.shape[0] = num_blocks;
    view.shape[1] = block_size;
    view.shape[2] = hk;
    view.shape[3] = d;
    view.stride[0] = 2 * within;
    view.stride[1] = hk * d;
    view.stride[2] = d;
    view.stride[3] = 1;
    return view;
  }

  std::vector<float> Reference(
      const std::vector<int32_t>& lengths,
      std::optional<AttentionWindow> window = std::nullopt) const {
    return ComposedPagedRef(query_rounded, key_rounded, value_rounded,
                            block_table, max_blocks, lengths, qsl, hq, hk, d,
                            block_size, scale, /*causal=*/true, window);
  }
};

void CheckBf16AgainstReference(const std::vector<uint16_t>& got,
                               const std::vector<float>& reference,
                               const char* label) {
  REQUIRE(got.size() == reference.size());
  size_t violations = 0;
  double max_abs = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double actual = Bf16BitsToF32(got[i]);
    const double error = std::abs(actual - static_cast<double>(reference[i]));
    max_abs = std::max(max_abs, error);
    if (error > 1.5e-2 + 1.0e-2 * std::abs(static_cast<double>(reference[i]))) {
      ++violations;
    }
  }
  INFO(label);
  INFO(max_abs);
  CHECK(violations == 0);
}

struct Fa2DecodeRunStats {
  uint64_t launches = 0;
  uint64_t split_launches = 0;
  uint64_t no_split_launches = 0;
  uint64_t swap_launches = 0;
};

Fa2DecodeRunStats RunFa2DecodeCase(Fa2DecodeCase& c, const char* toggle,
                                   bool expect_fa2,
                                   std::optional<AttentionWindow> window = std::nullopt) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor query(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch},
                        c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d});

  // All d256 decode arms share this harness. Drive their independent flags from
  // one test toggle so each topology has the same fallback contract.
  EnvGuard decode4_toggle("VT_FA2_DECODE_4B", toggle);
  EnvGuard decode_toggle("VT_FA2_DECODE", toggle);
  EnvGuard decode35_toggle("VT_FA2_DECODE_35B", toggle);
  vt::cuda::testing::ResetFa2DecodeDebugCounters();
  PagedAttentionArgs args{c.scale, true};
  args.window_size = window;
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> got(c.query_bf16.size(), 0);
  out.Download(guard.q, got.data());

  const Fa2DecodeRunStats stats{
      vt::cuda::testing::Fa2DecodeLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeNoSplitLaunchesForTesting()};
  vt::cuda::testing::DisableFa2DecodeDebugCounters();
  CheckBf16AgainstReference(got, c.Reference(c.seq_lens, window),
                            expect_fa2 ? "FA2 decode" : "paged fallback");
  CHECK(stats.launches == (expect_fa2 ? 1U : 0U));
  CHECK(stats.split_launches + stats.no_split_launches == stats.launches);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == (expect_fa2 ? 1U : 0U));
  return stats;
}

// VARLEN d128 decode harness (Qwen3-dense): the flash_attn_varlen_func path
// (VT_FA2_DECODE_QWEN3), head_dim 128. `swap_toggle` drives VT_FA2_DECODE_GQA_SWAP
// so one harness covers both the plain-varlen reduction and vLLM's decode
// seqlenq_ngroups_swapped grid. `expect_swap` asserts the swapped launcher engaged
// (or did NOT, for the eligibility guards). Both validated against the same f32
// ComposedPagedRef. When `out_bits` is non-null the downloaded bf16 output is
// copied out so callers can cross-check swap-ON vs swap-OFF (the near-tie razor).
Fa2DecodeRunStats RunFa2VarlenDecodeCase(Fa2DecodeCase& c, const char* toggle,
                                         bool expect_fa2, const char* swap_toggle = "0",
                                         bool expect_swap = false,
                                         std::vector<uint16_t>* out_bits = nullptr,
                                         const char* nsplits_cap = nullptr) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor query(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch}, c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d});

  EnvGuard qwen3_toggle("VT_FA2_DECODE_QWEN3", toggle);
  EnvGuard swap_env("VT_FA2_DECODE_GQA_SWAP", swap_toggle);
  // VT_FA2_NSPLITS_CAP is read fresh per launch, so within-process A/B works.
  // Passing nullptr leaves it OFF (default); an explicit value drives the cap for
  // the num_splits regime-change proof.
  EnvGuard cap_env("VT_FA2_NSPLITS_CAP", nsplits_cap == nullptr ? "0" : nsplits_cap);
  vt::cuda::testing::ResetFa2DecodeDebugCounters();
  PagedAttentionArgs args{c.scale, true};
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> got(c.query_bf16.size(), 0);
  out.Download(guard.q, got.data());

  const Fa2DecodeRunStats stats{
      vt::cuda::testing::Fa2DecodeLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeNoSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSwapLaunchesForTesting()};
  vt::cuda::testing::DisableFa2DecodeDebugCounters();
  CheckBf16AgainstReference(got, c.Reference(c.seq_lens),
                            expect_fa2 ? "FA2 varlen decode d128" : "paged fallback");
  CHECK(stats.launches == (expect_fa2 ? 1U : 0U));
  // The swap grid MUST have engaged (or not) as expected — a plain-path fallback
  // that still matched the reference would otherwise pass the port vacuously.
  CHECK(stats.swap_launches == (expect_swap ? 1U : 0U));
  if (out_bits != nullptr) *out_bits = std::move(got);
  return stats;
}

}  // namespace

// Qwen3-dense VARLEN d128 decode parity: matches the f32 composed reference at
// both gate configs (0.6B Hq/Hkv=16/8, 4B 32/8), across batch sizes and context
// lengths short (num_splits==1, the strict-gate regime) and long (num_splits>1,
// exercising the split combine). Opt-in via VT_FA2_DECODE_QWEN3.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 decode parity (dgx-pending)");
    return;
  }
  // ★ {32, 2} IS NEMOTRON-3.5-LIGHTNING, AND IT WAS NOT HERE (#1157).
  // Every case in this file measured ngroups 2 (16/8) and 4 (32/8), the two
  // Qwen3-dense gate configs the varlen d128 decode path was written for. The
  // path is DEFAULT ON for ANY bf16 causal pure-decode at head_dim 128, so the
  // first model to arrive with a different ratio reaches an untested grid — and
  // NemotronH-3.5-Lightning-30B is 32 query heads over 2 KV heads, ngroups 16,
  // four times the widest group count ever measured here. Its A3 token gate
  // failed 6/96 on the device while the same binary decodes the same checkpoint
  // token-exact on CPU, which is what sent the search here.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{16, 8},
                            std::pair<int64_t, int64_t>{32, 8},
                            std::pair<int64_t, int64_t>{32, 2}}) {
    for (const int batch : {1, 2, 4, 8}) {
      for (const int base_len : {5, 21, 1024}) {  // short => num_splits==1; long => split
        CAPTURE(ratio.first);
        CAPTURE(batch);
        CAPTURE(base_len);
        std::vector<int32_t> lengths(static_cast<size_t>(batch));
        for (int i = 0; i < batch; ++i)
          lengths[static_cast<size_t>(i)] = base_len + i * 3;
        Fa2DecodeCase c(ratio.first, ratio.second, std::move(lengths),
                        7100U + static_cast<uint32_t>(batch * 31 + base_len),
                        /*capacity_blocks=*/0, /*head_dim=*/128);
        RunFa2VarlenDecodeCase(c, "1", /*expect_fa2=*/true);
      }
    }
  }
}

// Same-binary A/B off arm: with VT_FA2_DECODE_QWEN3=0 the d128 decode stays on
// the portable fallback (no varlen-FA2 launch). The default is now ON (the
// shipped Qwen3-dense decode), so this pins the explicit-off toggle path.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode toggle-off uses fallback") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 opt-in check (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/16, /*Hkv=*/8, {21, 24, 27}, 7333,
                  /*capacity_blocks=*/0, /*head_dim=*/128);
  RunFa2VarlenDecodeCase(c, "0", /*expect_fa2=*/false);
}

// Qwen3-dense VARLEN d128 decode GQA GROUP-SWAP (VT_FA2_DECODE_GQA_SWAP): vLLM's
// seqlenq_ngroups_swapped decode grid ported into the varlen launcher (#47). The
// port must (a) match the same f32 composed reference within the near-tie band at
// both gate ratios (0.6B 16/8 => ngroups 2, 4B 32/8 => ngroups 4), across batch
// and both num_splits regimes, AND (b) PROVE the swapped grid actually engaged
// (swap_launches==1) — a silent fallback to the plain arm that still matched the
// reference would pass the port vacuously. Grid-sensitivity (the RED anchor): a
// wrong swapped stride / seqlen_q / head count corrupts the output and trips
// CheckBf16AgainstReference, so a GREEN here means the layout is read correctly.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode GQA group-swap matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 group-swap parity (dgx-pending)");
    return;
  }
  // ★ {32, 2} IS NEMOTRON-3.5-LIGHTNING, AND IT WAS NOT HERE (#1157).
  // Every case in this file measured ngroups 2 (16/8) and 4 (32/8), the two
  // Qwen3-dense gate configs the varlen d128 decode path was written for. The
  // path is DEFAULT ON for ANY bf16 causal pure-decode at head_dim 128, so the
  // first model to arrive with a different ratio reaches an untested grid — and
  // NemotronH-3.5-Lightning-30B is 32 query heads over 2 KV heads, ngroups 16,
  // four times the widest group count ever measured here. Its A3 token gate
  // failed 6/96 on the device while the same binary decodes the same checkpoint
  // token-exact on CPU, which is what sent the search here.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{16, 8},
                            std::pair<int64_t, int64_t>{32, 8},
                            std::pair<int64_t, int64_t>{32, 2}}) {
    for (const int batch : {1, 2, 4, 8}) {
      for (const int base_len : {5, 21, 1024}) {  // short => num_splits==1; long => split
        CAPTURE(ratio.first);
        CAPTURE(batch);
        CAPTURE(base_len);
        std::vector<int32_t> lengths(static_cast<size_t>(batch));
        for (int i = 0; i < batch; ++i)
          lengths[static_cast<size_t>(i)] = base_len + i * 3;
        Fa2DecodeCase c(ratio.first, ratio.second, std::move(lengths),
                        7700U + static_cast<uint32_t>(batch * 31 + base_len),
                        /*capacity_blocks=*/0, /*head_dim=*/128);
        RunFa2VarlenDecodeCase(c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1",
                               /*expect_swap=*/true);
      }
    }
  }
}

// Near-tie razor: the swap arm and the plain-varlen arm read the SAME q/k/v, so
// their bf16 outputs must agree within the split-reduction near-tie band (they are
// only non-byte-exact when num_splits>1). Directly comparing the two engine
// outputs (not just each vs the host reference) pins that the swap is a numerics-
// preserving reshape of vLLM's own reduction, not a different computation.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode swap near-ties the plain-varlen arm") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 swap-vs-plain near-tie (dgx-pending)");
    return;
  }
  // ★ {32, 2} IS NEMOTRON-3.5-LIGHTNING, AND IT WAS NOT HERE (#1157).
  // Every case in this file measured ngroups 2 (16/8) and 4 (32/8), the two
  // Qwen3-dense gate configs the varlen d128 decode path was written for. The
  // path is DEFAULT ON for ANY bf16 causal pure-decode at head_dim 128, so the
  // first model to arrive with a different ratio reaches an untested grid — and
  // NemotronH-3.5-Lightning-30B is 32 query heads over 2 KV heads, ngroups 16,
  // four times the widest group count ever measured here. Its A3 token gate
  // failed 6/96 on the device while the same binary decodes the same checkpoint
  // token-exact on CPU, which is what sent the search here.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{16, 8},
                            std::pair<int64_t, int64_t>{32, 8},
                            std::pair<int64_t, int64_t>{32, 2}}) {
    for (const int batch : {2, 8}) {
      for (const int base_len : {21, 1024}) {
        CAPTURE(ratio.first);
        CAPTURE(batch);
        CAPTURE(base_len);
        std::vector<int32_t> lengths(static_cast<size_t>(batch));
        for (int i = 0; i < batch; ++i)
          lengths[static_cast<size_t>(i)] = base_len + i * 3;
        Fa2DecodeCase c(ratio.first, ratio.second, std::move(lengths),
                        7900U + static_cast<uint32_t>(batch * 31 + base_len),
                        /*capacity_blocks=*/0, /*head_dim=*/128);
        std::vector<uint16_t> plain, swapped;
        RunFa2VarlenDecodeCase(c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"0",
                               /*expect_swap=*/false, &plain);
        RunFa2VarlenDecodeCase(c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1",
                               /*expect_swap=*/true, &swapped);
        REQUIRE(plain.size() == swapped.size());
        double max_abs = 0.0;
        for (size_t i = 0; i < plain.size(); ++i) {
          const double diff = static_cast<double>(Bf16BitsToF32(plain[i])) -
                              static_cast<double>(Bf16BitsToF32(swapped[i]));
          max_abs = std::max(max_abs, std::abs(diff));
        }
        INFO("swap vs plain max_abs = " << max_abs);
        CHECK(max_abs < 2.0e-2);
      }
    }
  }
}

// GB10 num_splits cap (VT_FA2_NSPLITS_CAP): the FA2 heuristic OVER-SPLITS the
// batch-1 decode (fed num_sms*2, all waves<1 => runs to max eligible splits). The
// cap clamps the split count. This must (a) still match the same f32 composed
// reference at BOTH gate ratios (0.6B 16/8, 4B 32/8) in the capped regime, and
// (b) PROVE the cap engaged: at a long context that normally splits (base_len
// 1024 => num_splits>1), forcing cap="1" collapses the launch onto the no-split
// path (no_split_launches==1, split_launches==0), while the uncapped run splits
// (split_launches==1). A cap that silently no-op'd would fail this regime check.
// The swap is ON here (the shipped default), so this covers the production grid.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode num_splits cap engages and stays correct") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 num_splits-cap check (dgx-pending)");
    return;
  }
  // ★ {32, 2} IS NEMOTRON-3.5-LIGHTNING, AND IT WAS NOT HERE (#1157).
  // Every case in this file measured ngroups 2 (16/8) and 4 (32/8), the two
  // Qwen3-dense gate configs the varlen d128 decode path was written for. The
  // path is DEFAULT ON for ANY bf16 causal pure-decode at head_dim 128, so the
  // first model to arrive with a different ratio reaches an untested grid — and
  // NemotronH-3.5-Lightning-30B is 32 query heads over 2 KV heads, ngroups 16,
  // four times the widest group count ever measured here. Its A3 token gate
  // failed 6/96 on the device while the same binary decodes the same checkpoint
  // token-exact on CPU, which is what sent the search here.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{16, 8},
                            std::pair<int64_t, int64_t>{32, 8},
                            std::pair<int64_t, int64_t>{32, 2}}) {
    for (const int batch : {1, 2, 4}) {
      CAPTURE(ratio.first);
      CAPTURE(batch);
      // base_len 1024 => the uncapped heuristic splits (num_splits>1).
      std::vector<int32_t> lengths(static_cast<size_t>(batch));
      for (int i = 0; i < batch; ++i)
        lengths[static_cast<size_t>(i)] = 1024 + i * 3;
      Fa2DecodeCase c(ratio.first, ratio.second, lengths,
                      7500U + static_cast<uint32_t>(batch * 31), /*capacity_blocks=*/0,
                      /*head_dim=*/128);

      // Uncapped (default OFF) with swap ON: the long context splits.
      std::vector<uint16_t> uncapped;
      const Fa2DecodeRunStats base = RunFa2VarlenDecodeCase(
          c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1", /*expect_swap=*/true,
          &uncapped, /*nsplits_cap=*/nullptr);
      CHECK(base.split_launches == 1U);
      CHECK(base.no_split_launches == 0U);

      // Cap to 1: the same launch collapses onto the no-split kernel, and the
      // output still matches the composed reference (asserted inside the helper).
      std::vector<uint16_t> capped1;
      const Fa2DecodeRunStats cap1 = RunFa2VarlenDecodeCase(
          c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1", /*expect_swap=*/true,
          &capped1, /*nsplits_cap=*/"1");
      CHECK(cap1.split_launches == 0U);
      CHECK(cap1.no_split_launches == 1U);

      // Cap to 2: a DIFFERENT split count than the uncapped heuristic, still
      // splitting and still correct — near-ties the uncapped output (the cap is a
      // reduction-order change over the SAME q/k/v, not a different computation).
      std::vector<uint16_t> capped2;
      const Fa2DecodeRunStats cap2 = RunFa2VarlenDecodeCase(
          c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1", /*expect_swap=*/true,
          &capped2, /*nsplits_cap=*/"2");
      CHECK(cap2.split_launches == 1U);
      REQUIRE(uncapped.size() == capped2.size());
      double max_abs = 0.0;
      for (size_t i = 0; i < uncapped.size(); ++i) {
        const double diff = static_cast<double>(Bf16BitsToF32(uncapped[i])) -
                            static_cast<double>(Bf16BitsToF32(capped2[i]));
        max_abs = std::max(max_abs, std::abs(diff));
      }
      INFO("uncapped vs cap=2 max_abs = " << max_abs);
      CHECK(max_abs < 2.0e-2);
    }
  }
}

// Eligibility guard: with the swap env ON but a NON-GQA topology (MHA, qpk==1) the
// launcher must stay on the plain-varlen path — there are no query groups to pack.
// Proves the dispatcher/launcher gate, not just the happy path.
TEST_CASE("paged_attention CUDA FA-2 varlen d128 decode swap is inert for MHA (qpk==1)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 varlen d128 swap MHA-ineligibility (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/8, /*Hkv=*/8, {21, 24, 27}, 7811,
                  /*capacity_blocks=*/0, /*head_dim=*/128);
  RunFa2VarlenDecodeCase(c, "1", /*expect_fa2=*/true, /*swap_toggle=*/"1",
                         /*expect_swap=*/false);
}

TEST_CASE("paged_attention CUDA FA-2 split heuristic mirrors upstream") {
  using vt::cuda::testing::Fa2DecodeNumSplitsForTesting;
  CHECK(Fa2DecodeNumSplitsForTesting(32, 40, 18, 128) == 1);
  CHECK(Fa2DecodeNumSplitsForTesting(4, 40, 18, 128) == 9);
  CHECK(Fa2DecodeNumSplitsForTesting(8, 40, 32, 128) == 5);
  CHECK(Fa2DecodeNumSplitsForTesting(4, 40, 18, 2) == 2);
  CHECK(Fa2DecodeNumSplitsForTesting(1, 40, 1, 128) == 1);
}

TEST_CASE("paged_attention CUDA pure-decode upstream paged vector retains fallback correctness") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping upstream FA-2 pure-decode vector (dgx-pending)");
    return;
  }
  // Exact upstream sequence-length vector. Hq/Hkv=8/2 is outside W3-G's first
  // ratio-6 slice, so it proves honest fallback rather than broadened support.
  Fa2DecodeCase c(/*Hq=*/8, /*Hkv=*/2, {523, 37, 2011}, 6120);
  RunFa2DecodeCase(c, "1", /*expect_fa2=*/false);
}

TEST_CASE("paged_attention CUDA FA-2 ratio-6 pure decode matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 ratio-6 decode parity (dgx-pending)");
    return;
  }
  for (const int batch : {1, 2, 4, 8, 16}) {
    CAPTURE(batch);
    std::vector<int32_t> lengths(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) lengths[static_cast<size_t>(i)] = 1024 + i * 7;
    Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, std::move(lengths),
                    6200U + static_cast<uint32_t>(batch));
    RunFa2DecodeCase(c, "1", /*expect_fa2=*/true);
  }
}

TEST_CASE("paged_attention CUDA FA-2 ratio-4 Qwen3.5-4B decode matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 ratio-4 decode parity (dgx-pending)");
    return;
  }
  for (const int batch : {1, 2, 4, 8, 16}) {
    CAPTURE(batch);
    std::vector<int32_t> lengths(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) lengths[static_cast<size_t>(i)] = 1024 + i * 7;
    Fa2DecodeCase c(/*Hq=*/16, /*Hkv=*/4, std::move(lengths),
                    6450U + static_cast<uint32_t>(batch));
    RunFa2DecodeCase(c, "1", /*expect_fa2=*/true);
  }
}

TEST_CASE("paged_attention CUDA FA-2 ratio-8 pure decode matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 ratio-8 decode parity (dgx-pending)");
    return;
  }
  // Qwen3.6-35B Hq/Hkv=16/2 hd-256 (CLAIM-35B-FA2-DECODE-1): the same vendored
  // split-KV path as ratio-6, now enabled for the 35B full-attention layers.
  for (const int batch : {1, 2, 4, 8, 16}) {
    CAPTURE(batch);
    std::vector<int32_t> lengths(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) lengths[static_cast<size_t>(i)] = 1024 + i * 7;
    Fa2DecodeCase c(/*Hq=*/16, /*Hkv=*/2, std::move(lengths),
                    6500U + static_cast<uint32_t>(batch));
    RunFa2DecodeCase(c, "1", /*expect_fa2=*/true);
  }
}

// SPEC-DFLASH2 W10 (#1857): the UNIFORM-QLEN SPECULATIVE VERIFY on the d256
// DECODE lane. Same harness family as RunFa2DecodeCase, with q_len query rows
// per request and the classified `uniform_spec_query_len` set the way the
// production runner sets it. `expect_fa2` is the ROUTING pin: the decode
// counters must move exactly when the spec-as-decode lane served the batch —
// before W10 this batch rode the prefill ladder and they stayed 0, and with
// VT_FA2_SPEC_DECODE=0 they must stay 0 again (the same-binary A/B the spec's
// numerics section names). Correctness is the SAME composed f32 reference every
// decode case here uses: the verify rows are the last q_len positions of each
// request's seq_len, so the reference's bottom-right causal geometry IS the
// packed draft mask's semantics.
Fa2DecodeRunStats RunFa2SpecDecodeCase(Fa2DecodeCase& c, const char* toggle,
                                       bool expect_fa2) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  const int64_t total_q = c.batch * c.q_len;
  DeviceTensor query(gpu, guard.q, DType::kBF16, {total_q, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch},
                        c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {total_q, c.hq, c.d});

  // The three d256 topologies stay on their shipped defaults (ON); the toggle
  // under test is the spec-as-decode admission itself.
  EnvGuard decode4_toggle("VT_FA2_DECODE_4B", "1");
  EnvGuard decode_toggle("VT_FA2_DECODE", "1");
  EnvGuard decode35_toggle("VT_FA2_DECODE_35B", "1");
  EnvGuard spec_toggle("VT_FA2_SPEC_DECODE", toggle);
  vt::cuda::testing::ResetFa2DecodeDebugCounters();
  PagedAttentionArgs args{c.scale, true};
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
  args.uniform_spec_query_len = static_cast<int32_t>(c.q_len);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> got(c.query_bf16.size(), 0);
  out.Download(guard.q, got.data());

  const Fa2DecodeRunStats stats{
      vt::cuda::testing::Fa2DecodeLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeNoSplitLaunchesForTesting(),
      vt::cuda::testing::Fa2DecodeSwapLaunchesForTesting()};
  vt::cuda::testing::DisableFa2DecodeDebugCounters();
  CheckBf16AgainstReference(got, c.Reference(c.seq_lens),
                            expect_fa2 ? "FA2 spec-as-decode" : "prefill-lane route");
  CHECK(stats.launches == (expect_fa2 ? 1U : 0U));
  // The spec lane is the PLAIN (non-swapped) presentation: a swap row is a head
  // group, and a causal mask across head groups would be wrong.
  CHECK(stats.swap_launches == 0U);
  return stats;
}

TEST_CASE("paged_attention CUDA FA-2 uniform spec verify routes DECODE and matches composed reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 spec-as-decode parity (dgx-pending)");
    return;
  }
  // All three shipped d256 topologies; q=9 is the #1857 measured verify width
  // (k=8), q=2 the narrowest verify. Short contexts keep num_splits==1 (the
  // bit-path today's prefill route takes); 1024 drives the split combine.
  for (const auto& ratio : {std::pair<int64_t, int64_t>{24, 4},
                            std::pair<int64_t, int64_t>{16, 4},
                            std::pair<int64_t, int64_t>{16, 2}}) {
    for (const int batch : {1, 2}) {
      for (const int base_len : {33, 1024}) {
        for (const int q_rows : {2, 9}) {
          CAPTURE(ratio.first);
          CAPTURE(batch);
          CAPTURE(base_len);
          CAPTURE(q_rows);
          std::vector<int32_t> lengths(static_cast<size_t>(batch));
          for (int i = 0; i < batch; ++i)
            lengths[static_cast<size_t>(i)] = base_len + i * 7;
          Fa2DecodeCase c(ratio.first, ratio.second, std::move(lengths),
                          8600U + static_cast<uint32_t>(batch * 131 + base_len + q_rows),
                          /*capacity_blocks=*/0, /*head_dim=*/256,
                          /*query_rows_per_req=*/q_rows);
          RunFa2SpecDecodeCase(c, "1", /*expect_fa2=*/true);
        }
      }
    }
  }
}

TEST_CASE("paged_attention CUDA FA-2 spec verify kill switch restores the prefill route") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 spec-as-decode kill-switch check (dgx-pending)");
    return;
  }
  // VT_FA2_SPEC_DECODE=0: the same batch must route exactly as before W10
  // (prefill ladder — decode counters silent) and still match the reference.
  Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, {1024}, 8700U,
                  /*capacity_blocks=*/0, /*head_dim=*/256,
                  /*query_rows_per_req=*/9);
  RunFa2SpecDecodeCase(c, "0", /*expect_fa2=*/false);
  // An UNCLASSIFIED batch (field 0) with q>1 rows is a plain prefill and must
  // never reach the decode lane even with the toggle ON.
  Fa2DecodeCase c2(/*Hq=*/24, /*Hkv=*/4, {1024}, 8710U,
                   /*capacity_blocks=*/0, /*head_dim=*/256,
                   /*query_rows_per_req=*/9);
  {
    Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard guard(gpu);
    const int64_t total_q = 9;
    DeviceTensor query(gpu, guard.q, DType::kBF16, {total_q, c2.hq, c2.d},
                       c2.query_bf16.data());
    DeviceTensor cache(gpu, guard.q, DType::kBF16,
                       {static_cast<int64_t>(c2.combined_cache.size())},
                       c2.combined_cache.data());
    Tensor key = c2.CacheView(cache, 0);
    Tensor value = c2.CacheView(cache, 1);
    DeviceTensor block_table(gpu, guard.q, DType::kI32, {c2.batch, c2.max_blocks},
                             c2.block_table.data());
    DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c2.batch}, c2.seq_lens.data());
    std::vector<int32_t> qsl_v = {0, 9};
    DeviceTensor qsl(gpu, guard.q, DType::kI32, {c2.batch + 1}, qsl_v.data());
    DeviceTensor out(gpu, guard.q, DType::kBF16, {total_q, c2.hq, c2.d});
    EnvGuard decode_toggle("VT_FA2_DECODE", "1");
    EnvGuard spec_toggle("VT_FA2_SPEC_DECODE", "1");
    vt::cuda::testing::ResetFa2DecodeDebugCounters();
    PagedAttentionArgs args{c2.scale, true};
    args.query_start_loc_host = qsl_v.data();
    args.max_seq_len = static_cast<int>(c2.max_blocks * c2.block_size);
    // uniform_spec_query_len stays 0: not classified.
    vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                       block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
    const uint64_t launches = vt::cuda::testing::Fa2DecodeLaunchesForTesting();
    vt::cuda::testing::DisableFa2DecodeDebugCounters();
    CHECK(launches == 0U);
  }
}

TEST_CASE("paged_attention CUDA FA-2 decode toggle and invalid eligibility use fallback") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 decode fallback vectors (dgx-pending)");
    return;
  }
  Fa2DecodeCase ratio6(/*Hq=*/24, /*Hkv=*/4, {1024, 1057}, 6300);
  RunFa2DecodeCase(ratio6, "0", /*expect_fa2=*/false);
  RunFa2DecodeCase(ratio6, "1", /*expect_fa2=*/false,
                   AttentionWindow{127, 0});

  // The 35B ratio-8 topology is now supported; VT_FA2_DECODE_35B=0 and a finite
  // window each fall back to the paged kernel in the same binary.
  Fa2DecodeCase ratio8(/*Hq=*/16, /*Hkv=*/2, {1024, 1057}, 6310);
  RunFa2DecodeCase(ratio8, "0", /*expect_fa2=*/false);
  RunFa2DecodeCase(ratio8, "1", /*expect_fa2=*/false,
                   AttentionWindow{127, 0});

  // The Qwen3.5-4B ratio-4 topology has its own rollback and finite-window
  // fallback.
  Fa2DecodeCase ratio4_4b(/*Hq=*/16, /*Hkv=*/4, {1024, 1057}, 6315);
  RunFa2DecodeCase(ratio4_4b, "0", /*expect_fa2=*/false);
  RunFa2DecodeCase(ratio4_4b, "1", /*expect_fa2=*/false,
                   AttentionWindow{127, 0});

  // A different ratio-4 geometry is still outside the bounded admission.
  Fa2DecodeCase ratio4(/*Hq=*/8, /*Hkv=*/2, {1024, 1057}, 6320);
  RunFa2DecodeCase(ratio4, "1", /*expect_fa2=*/false);
}

TEST_CASE("paged_attention CUDA FA-2 decode scratch is capture-stable across replay") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 capture/replay vector (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, {257, 389}, 6400,
                  /*capacity_blocks=*/72);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor query(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d},
                     c.query_bf16.data());
  DeviceTensor cache(gpu, guard.q, DType::kBF16,
                     {static_cast<int64_t>(c.combined_cache.size())},
                     c.combined_cache.data());
  Tensor key = c.CacheView(cache, 0);
  Tensor value = c.CacheView(cache, 1);
  DeviceTensor block_table(gpu, guard.q, DType::kI32,
                           {c.batch, c.max_blocks}, c.block_table.data());
  DeviceTensor seq_lens(gpu, guard.q, DType::kI32, {c.batch},
                        c.seq_lens.data());
  DeviceTensor qsl(gpu, guard.q, DType::kI32, {c.batch + 1}, c.qsl.data());
  DeviceTensor out(gpu, guard.q, DType::kBF16, {c.batch, c.hq, c.d});
  PagedAttentionArgs args{c.scale, true};
  args.query_start_loc_host = c.qsl.data();
  args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);

  EnvGuard decode_on("VT_FA2_DECODE", "1");
  vt::cuda::testing::ResetFa2DecodeDebugCounters();

  // Cold eager: materialize the one capture-stable shape entry.
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  std::vector<uint16_t> warm(c.query_bf16.size(), 0);
  out.Download(guard.q, warm.data());
  CheckBf16AgainstReference(warm, c.Reference(c.seq_lens), "FA2 cold eager");
  CHECK(vt::cuda::testing::Fa2DecodeScratchAllocationsForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchReusesForTesting() == 0);

  // Capture must be a pure pool hit: no pointer changes or allocator calls.
  gpu.BeginCapture(guard.q);
  vt::PagedAttention(guard.q, out.tensor(), query.tensor(), key, value,
                     block_table.tensor(), seq_lens.tensor(), qsl.tensor(), args);
  void* graph = gpu.EndCaptureGraph(guard.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchAllocationsForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchReusesForTesting() == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == 1);

  // Grow actual sequence lengths inside the fixed block-table capacity. The
  // captured geometry and scratch addresses stay fixed; seqused_k changes.
  const std::vector<int32_t> grown_lens = {511, 887};
  gpu.Copy(guard.q, seq_lens.tensor().data, grown_lens.data(),
           grown_lens.size() * sizeof(int32_t));
  gpu.ReplayGraph(guard.q, graph);
  std::vector<uint16_t> replay1(c.query_bf16.size(), 0);
  out.Download(guard.q, replay1.data());
  CheckBf16AgainstReference(replay1, c.Reference(grown_lens), "FA2 replay 1");
  gpu.ReplayGraph(guard.q, graph);
  std::vector<uint16_t> replay2(c.query_bf16.size(), 0);
  out.Download(guard.q, replay2.data());
  CHECK(replay1 == replay2);
  gpu.DestroyGraph(graph);
  vt::cuda::testing::DisableFa2DecodeDebugCounters();

  // A new block-table column capacity gets a second stable entry; the first is
  // retained rather than grown/freed because an independently owned graph may
  // still reference it in production.
  Fa2DecodeCase wider(/*Hq=*/24, /*Hkv=*/4, {511, 887}, 6410,
                      /*capacity_blocks=*/80);
  DeviceTensor query2(gpu, guard.q, DType::kBF16,
                      {wider.batch, wider.hq, wider.d},
                      wider.query_bf16.data());
  DeviceTensor cache2(gpu, guard.q, DType::kBF16,
                      {static_cast<int64_t>(wider.combined_cache.size())},
                      wider.combined_cache.data());
  Tensor key2 = wider.CacheView(cache2, 0);
  Tensor value2 = wider.CacheView(cache2, 1);
  DeviceTensor block_table2(gpu, guard.q, DType::kI32,
                            {wider.batch, wider.max_blocks},
                            wider.block_table.data());
  DeviceTensor seq_lens2(gpu, guard.q, DType::kI32, {wider.batch},
                         wider.seq_lens.data());
  DeviceTensor qsl2(gpu, guard.q, DType::kI32, {wider.batch + 1},
                    wider.qsl.data());
  DeviceTensor out2(gpu, guard.q, DType::kBF16,
                    {wider.batch, wider.hq, wider.d});
  PagedAttentionArgs args2{wider.scale, true};
  args2.query_start_loc_host = wider.qsl.data();
  args2.max_seq_len = static_cast<int>(wider.max_blocks * wider.block_size);
  vt::PagedAttention(guard.q, out2.tensor(), query2.tensor(), key2, value2,
                     block_table2.tensor(), seq_lens2.tensor(), qsl2.tensor(),
                     args2);
  gpu.Synchronize(guard.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(
            guard.q.device.index, guard.q.handle) == 2);
}

TEST_CASE("paged_attention CUDA FA-2 decode scratch is queue-owned and released") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping FA-2 queue lifecycle vector (dgx-pending)");
    return;
  }
  Fa2DecodeCase c(/*Hq=*/24, /*Hkv=*/4, {1024}, 6500);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard first(gpu);
  QueueGuard second(gpu);
  const auto launch = [&](Queue& queue) {
    DeviceTensor query(gpu, queue, DType::kBF16, {c.batch, c.hq, c.d},
                       c.query_bf16.data());
    DeviceTensor cache(gpu, queue, DType::kBF16,
                       {static_cast<int64_t>(c.combined_cache.size())},
                       c.combined_cache.data());
    Tensor key = c.CacheView(cache, 0);
    Tensor value = c.CacheView(cache, 1);
    DeviceTensor block_table(gpu, queue, DType::kI32,
                             {c.batch, c.max_blocks}, c.block_table.data());
    DeviceTensor seq_lens(gpu, queue, DType::kI32, {c.batch},
                          c.seq_lens.data());
    DeviceTensor qsl(gpu, queue, DType::kI32, {c.batch + 1}, c.qsl.data());
    DeviceTensor out(gpu, queue, DType::kBF16, {c.batch, c.hq, c.d});
    PagedAttentionArgs args{c.scale, true};
    args.query_start_loc_host = c.qsl.data();
    args.max_seq_len = static_cast<int>(c.max_blocks * c.block_size);
    vt::PagedAttention(queue, out.tensor(), query.tensor(), key, value,
                       block_table.tensor(), seq_lens.tensor(), qsl.tensor(),
                       args);
    gpu.Synchronize(queue);
  };

  EnvGuard decode_on("VT_FA2_DECODE", "1");
  launch(first.q);
  launch(second.q);
  void* first_handle = first.q.handle;
  void* second_handle = second.q.handle;
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, first_handle) == 1);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 1);

  gpu.DestroyQueue(first.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, first_handle) == 0);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 1);
  gpu.DestroyQueue(second.q);
  CHECK(vt::cuda::testing::Fa2DecodeScratchShapeCountForTesting(0, second_handle) == 0);
}
#else
TEST_CASE("paged_attention CUDA FA-2 prefill (bf16 q/kv/out) matches f32 ref at head_dim 256") {
  MESSAGE("built without VLLM_CPP_FLASH_ATTN; FA-2 prefill parity skipped");
}
#endif  // VLLM_CPP_FLASH_ATTN

// ROCm f32-query / bf16-KV decode arm (PagedAttnDecodeGqaF32Q + split-KV
// reduce) at the Qwen3.8-27B full-attention shape: hq=24, kv=4 (QG=6), d=256.
// The CUDA parity cases above skip on ROCm, so without this case nothing on the
// ROCm backend reaches that kernel. Contexts straddle the device-side split
// chunking (64 positions minimum per split, 16 splits by default).
namespace {

bool HasRocmBackend() {
  try {
    vt::GetBackend(DeviceType::kROCM);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct RocmBuf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  RocmBuf(Backend& be, Queue& q, DType dt, const std::vector<int64_t>& shape,
          const void* host = nullptr)
      : b(be) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    if (host != nullptr) b.Copy(q, p, host, bytes);
    t.data = p;
    t.dtype = dt;
    t.device = Device{DeviceType::kROCM, 0};
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
  }
  ~RocmBuf() { b.Free(p); }
  RocmBuf(const RocmBuf&) = delete;
  RocmBuf& operator=(const RocmBuf&) = delete;
};

void CheckRocmF32QGqa(const std::vector<int32_t>& qsl, const std::vector<int32_t>& seq_lens,
                      uint32_t seed) {
  const int64_t Hq = 24, Hk = 4, D = 256, block_size = 16;
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t num_reqs = static_cast<int64_t>(seq_lens.size());
  const int64_t num_tokens = qsl.back();
  const int32_t max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
  const int64_t max_blocks = (max_seq + block_size - 1) / block_size;
  const int64_t num_blocks = num_reqs * max_blocks;
  // Reversed block ids so a request's pages are not contiguous in the pool.
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i)
    block_table[static_cast<size_t>(i)] = static_cast<int32_t>(num_blocks - 1 - i);
  auto q = RandF32(static_cast<size_t>(num_tokens * Hq * D), seed);
  const size_t kv_n = static_cast<size_t>(num_blocks * block_size * Hk * D);
  auto kc = RandF32(kv_n, seed + 1);
  auto vc = RandF32(kv_n, seed + 2);
  std::vector<uint16_t> kb(kv_n), vb(kv_n);
  for (size_t i = 0; i < kv_n; ++i) {
    kb[i] = vt::F32ToBF16(kc[i]);
    vb[i] = vt::F32ToBF16(vc[i]);
    kc[i] = vt::BF16ToF32(kb[i]);
    vc[i] = vt::BF16ToF32(vb[i]);
  }
  const std::vector<float> ref = ComposedPagedRef(q, kc, vc, block_table, max_blocks, seq_lens,
                                                  qsl, Hq, Hk, D, block_size, scale, true);

  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  {
    RocmBuf dq(gpu, gq, DType::kF32, {num_tokens, Hq, D}, q.data());
    RocmBuf dkc(gpu, gq, DType::kBF16, {num_blocks, block_size, Hk, D}, kb.data());
    RocmBuf dvc(gpu, gq, DType::kBF16, {num_blocks, block_size, Hk, D}, vb.data());
    RocmBuf dbt(gpu, gq, DType::kI32, {num_reqs, max_blocks}, block_table.data());
    RocmBuf dsl(gpu, gq, DType::kI32, {num_reqs}, seq_lens.data());
    RocmBuf dqsl(gpu, gq, DType::kI32, {num_reqs + 1}, qsl.data());
    RocmBuf dout(gpu, gq, DType::kF32, {num_tokens, Hq, D});
    vt::PagedAttention(gq, dout.t, dq.t, dkc.t, dvc.t, dbt.t, dsl.t, dqsl.t,
                       PagedAttentionArgs{scale, true});
    std::vector<float> got(ref.size(), 0.0f);
    gpu.Copy(gq, got.data(), dout.p, dout.bytes);
    gpu.Synchronize(gq);
    size_t bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
      const float e = std::fabs(got[i] - ref[i]);
      if (!(e <= 1e-4f * (1.0f + std::fabs(ref[i])))) ++bad;
      if (!(e <= max_err)) max_err = e;
    }
    INFO("seq_lens[0]=" << seq_lens[0] << " reqs=" << num_reqs << " tokens=" << num_tokens
                        << " max_abs_err=" << max_err);
    CHECK(bad == 0);
  }
  gpu.DestroyQueue(gq);
}

}  // namespace

TEST_CASE("paged_attention ROCm f32-query GQA decode (hq=24 kv=4 d=256) matches CPU") {
  if (!HasRocmBackend()) {
    MESSAGE("no ROCm backend; skipping the f32-query GQA decode parity case");
    return;
  }
  for (int32_t len : {1, 7, 63, 64, 65, 129, 300, 1000, 1024, 1025, 2048, 4097})
    CheckRocmF32QGqa({0, 1}, {len}, 1000u + static_cast<uint32_t>(len));
  // Three decodes of very different length in one batch.
  CheckRocmF32QGqa({0, 1, 2, 3}, {5, 777, 2600}, 77u);
  // A 6-token prefill chunk on top of 300 context, next to a decode.
  CheckRocmF32QGqa({0, 6, 7}, {306, 900}, 91u);
}
