// Ported from: vllm/v1/attention/backends/flash_attn.py @ e24d1b24
//   (FlashAttentionImpl.forward SEMANTICS: causal/non-causal GQA softmax over
//    the paged K/V, optional bottom-right-aligned window_size, softmax_scale =
//    self.scale, cu_seqlens_q = query_start_loc, seqused_k = seq_lens,
//    block_table = block_table_tensor). The cache READ is the NHD
//    layout FlashAttentionBackend::get_kv_cache_shape allocates
//    (num_blocks, 2, block_size, num_kv_heads, head_size), indexed by TENSOR
//    STRIDES — NOT cpu_attn.py's HND arithmetic (see the M1.6 Task-3 layout trap).
//
// Correctness-grade CPU reference: a clear per-(request, token, q-head) loop with
// a two-pass max-subtracted f32 softmax — algebraically identical to the dense
// M0.9 AttentionKernel, so on a single contiguous sequence the two agree. The
// current step's K/V are assumed already written into the cache
// (vt::ReshapeAndCache), so the read is entirely from the paged blocks.
#include <algorithm>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "cpu_matmul_elem.h"
#include "cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// PERF-CPU-ATTN-DTYPE. The inner loops below used to call a per-ELEMENT
// `LoadF32(tensor, offset)` dtype switch — a branch on a value that is constant
// for the whole tensor, taken once per element, inside the K/V reduction. That
// is the same defect the elementwise GEMM already removed (see the
// MatmulOneChunk header, cpu_ops.cpp:98-107), so this mirrors that shape rather
// than inventing a new one:
//   1. the QUERY row is widened to f32 ONCE per token with the shared
//      `WidenRowToF32` (cpu_matmul_elem.h:110), documented bit-identical to a
//      per-element LoadF32, and reused by every q-head and every key; and
//   2. the K/V cache dtype is resolved ONCE per kernel invocation into a TYPED
//      element accessor, exactly as the GEMM resolves the weight dtype once per
//      chunk into a typed micro-kernel.
// Nothing about the arithmetic moves: the same f32 products accumulate over the
// same indices in the same strictly sequential order, so this is bit-identical
// by construction (asserted against a captured pre-change reference in
// tests/vt/test_ops_paged_attn_dtype.cpp).

// The element encodings a paged K/V cache may carry, resolved once per call.
// kFp8 is the KV-FP8 W1 path: 1-byte fp8 pages dequantized by k_scale/v_scale.
enum class KvKind { kF32, kF16, kBF16, kFp8 };

// One K/V element, with the dtype decision already made at COMPILE time. The
// f32/f16/bf16 arms are literally the old switch arms (same `Ptr<T>()[off]`,
// same F16ToF32/BF16ToF32) and the fp8 arm is the old `LoadKvFp8E4M3` branch,
// so every arm returns the identical float the per-element form returned.
template <KvKind kKind>
inline float KvElem(const void* base, int64_t off, float scale) {
  if constexpr (kKind == KvKind::kF32) {
    (void)scale;
    return static_cast<const float*>(base)[off];
  } else if constexpr (kKind == KvKind::kF16) {
    (void)scale;
    return F16ToF32(static_cast<const uint16_t*>(base)[off]);
  } else if constexpr (kKind == KvKind::kBF16) {
    (void)scale;
    return BF16ToF32(static_cast<const uint16_t*>(base)[off]);
  } else {
    return vt::LoadKvFp8E4M3(static_cast<const uint8_t*>(base)[off], scale);
  }
}

// KvKind of a non-fp8 cache. Same supported set — and same failure message — as
// the per-element switch it replaces; only the moment the branch is taken moves.
KvKind KvKindOf(DType dt) {
  switch (dt) {
    case DType::kF32: return KvKind::kF32;
    case DType::kF16: return KvKind::kF16;
    case DType::kBF16: return KvKind::kBF16;
    default: VT_CHECK(false, "paged_attention LoadF32: unsupported dtype"); return KvKind::kF32;
  }
}

// The output counterpart of `WidenRowToF32`: narrow `n` f32 accumulators into
// `t`'s storage dtype with ONE branch for the row instead of one per element.
// Same supported set, same rounding (`F32ToBF16`), same message as the
// per-element StoreF32 it replaces.
void StoreRowF32(const Tensor& t, int64_t elem_offset, int64_t n, const float* src) {
  switch (t.dtype) {
    case DType::kF32: {
      float* dst = t.Ptr<float>() + elem_offset;
      for (int64_t i = 0; i < n; ++i) dst[i] = src[i];
      break;
    }
    case DType::kBF16: {
      uint16_t* dst = t.Ptr<uint16_t>() + elem_offset;
      for (int64_t i = 0; i < n; ++i) dst[i] = F32ToBF16(src[i]);
      break;
    }
    default: VT_CHECK(false, "paged_attention StoreF32: unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// PERF-CPU-ATTN-NEON. The K dot-product and the V accumulation below run the
// SAME math as the scalar reference, one float32x4 at a time, aarch64 only
// (default on; `VT_CPU_PAGED_ATTN_NEON=0` rolls back to the scalar body, the
// reference and the non-aarch64 path). Two consequences the scalar body does
// not have:
//   1. The K reduction order changes (four interleaved lanes summed by
//      vaddvq_f32 instead of a strictly sequential f32 chain), so the NEON path
//      is NOT bit-identical to the scalar one. It is adjudicated by the model
//      gates, not by a memcmp: the unit sweep bounds the score/output drift
//      (tests/vt/test_ops_paged_attn_neon.cpp) and the kolibri battery —
//      including the W3 near-tie ARGMAX chain — is the contract.
//   2. Every width the vector path serves must be a multiple of the 4-lane
//      group in BOTH d (K/Q width) and d_v (V width); anything else, and any
//      non-aarch64 build, falls back to the untouched scalar body below.
// The element encodings are decoded with the same helpers the scalar arm uses
// (F16ToF32 / BF16ToF32 / LoadKvFp8E4M3), four elements at a time; the bf16
// lane widens by a 16-bit shift into the high half, exactly BitsToF32 does.
// ---------------------------------------------------------------------------
#if defined(__aarch64__)
bool PagedAttnNeonActive() {
  // ROLLED BACK to opt-in (ISSUE-LOCAL-01M4J4QNTE166QFG5MN7V3H4DG): under
  // thread contention (host load ~18) the lane serves a DEGENERATE chain
  // (prefill anchor 33382, not 101807) while the scalar path on the same
  // host, same 32 threads, is correct — a load-dependent race in the lane's
  // parallel dispatch that quiet-host gates (W3 900/900) never see. The
  // scalar body is the default again; an explicit `=1` opts into the lane
  // for the race hunt. `=0` keeps working.
  const char* e = std::getenv("VT_CPU_PAGED_ATTN_NEON");
  return e != nullptr && e[0] == '1';
}

// Four consecutive K/V elements as an f32 vector, the vector counterpart of
// KvElem. Same scales, same converters, four lanes.
template <KvKind kKind>
inline float32x4_t KvElem4(const void* base, int64_t off, float scale) {
  if constexpr (kKind == KvKind::kF32) {
    (void)scale;
    return vld1q_f32(static_cast<const float*>(base) + off);
  } else if constexpr (kKind == KvKind::kF16) {
    (void)scale;
    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t*>(base) + off));
  } else if constexpr (kKind == KvKind::kBF16) {
    (void)scale;
    uint16x4_t u = vld1_u16(static_cast<const uint16_t*>(base) + off);
    return vreinterpretq_f32_u32(vshll_n_u16(u, 16));
  } else {
    float tmp[4];
    for (int i = 0; i < 4; ++i)
      tmp[i] = vt::LoadKvFp8E4M3(static_cast<const uint8_t*>(base)[off + i], scale);
    return vld1q_f32(tmp);
  }
}
#endif  // __aarch64__

// Paged causal GQA attention. For request r, query token at global index t and
// local index `local`, the absolute position is p = context + local where
// context = seq_lens[r] - query_len_r. Keys 0..p (causal) are intersected with
// [p-left,p+right] for local attention, then gathered from the paged cache:
// position j lives in block_table[r, j/block_size], offset j%block_size, at
// [block, offset, kv_head, :] — read via cache strides.
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  // MiMoV2: V head dim can differ from K head dim (v_head_dim=128 vs
  // head_dim=192). d == K head dim (QK dot-product width); d_v is the V
  // accumulation width, read from v_cache.shape[3].
  const int64_t d_v = v_cache.shape[3];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t qpk = hq / num_kv_heads;  // q-heads per kv-head (GQA ratio)
  const float scale = args.scale;
  // Attention logit soft-cap (vLLM Attention(logits_soft_cap=...), gemma2.py:202):
  // score' = cap * tanh(score / cap). 0.0 (default) leaves the plain scaled dot.
  const float softcap = args.logits_soft_cap;
  const int64_t window_left =
      args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right =
      args.window_size.has_value() ? args.window_size->right : -1;

  // MiMoV2: attention sink bias. A per-q-head scalar that joins the softmax
  // denominator and contributes NO value — expressed by seeding the running
  // max at the sink and the running denominator at exp(0)=1. Absent
  // (attn_sink == nullptr, every caller before MiMoV2) the seeds stay -inf
  // and 0, so the loop, the reduction ORDER and the output are
  // bit-identical to the pre-sink kernel.
  const float* sink_p =
      args.attn_sink != nullptr ? args.attn_sink->Ptr<float>() : nullptr;

  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  // THE BLOCK TABLE HAS TO REACH THE LAST POSITION `seq_lens` DECLARES, and
  // nothing checked that it did (#1394). The j loop below reads
  // `btab[r * bt_row + (j / block_size) * bt_col]` for every j < seq_lens[r], so
  // a table with fewer columns than `ceil(seq_lens[r] / block_size)` is read past
  // its own last element. That read is SILENT: it lands inside whatever the
  // scratch pool put next to the table, and then either attends to the WRONG page
  // or dies, depending on what those bytes decode to.
  // `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp` supplied exactly that.
  // At `origin/main` it SIGSEGVs deterministically in one measured build (exit
  // 139, three runs of three) and reads green in another, which is the shape of
  // the hazard: the read is out of bounds unconditionally, and whether it faults
  // is the allocator's business rather than the caller's. One compare per
  // request, outside the token loop, turns it into a refusal that names the
  // caller's mistake instead.
  const int64_t bt_cols = block_table.rank >= 2 ? block_table.shape[1] : 0;
  for (int64_t r = 0; r < num_reqs; ++r) {
    // Only the requests the token loop actually visits. A padded or inert row
    // carries no query tokens, so its `seq_lens` entry is never read, and
    // refusing on one would reject a batch the kernel handles correctly today.
    if (qsl[r + 1] - qsl[r] <= 0) continue;
    const int64_t need = (slens[r] + block_size - 1) / block_size;
    VT_CHECK(need <= bt_cols,
             "paged_attention: the block table is shorter than the sequence it "
             "must address (a request needs more logical blocks than the table "
             "has columns)");
  }
  // Cache strides (unbind-slice aware): block / page(offset) / head / elem.
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  // fp8 KV read (KV-FP8 W1): when the cache is fp8 (kv_cache_dtype != kAuto) the
  // pages are 1-byte fp8 (kI8) and each read is DEQUANTIZED as Dequant(fp8) *
  // k_scale|v_scale before the f32 softmax (scaled_vec_conversion<float,uint8_t>,
  // quant_utils.cuh:302-308). kAuto keeps the float path byte-identical.
  const bool kv_fp8 = args.kv_cache_dtype != vt::Fp8KVCacheDataType::kAuto;
  const float k_scale = args.k_scale;
  const float v_scale = args.v_scale;
  // Resolved ONCE per invocation, replacing both the per-element `kv_fp8 ? ...`
  // branch and the per-element dtype switch underneath it. ONE kind covers both
  // caches because `vt::PagedAttention` already requires it: k_cache and v_cache
  // must share one float dtype (ops.cpp:3043-3045), and the fp8 path requires
  // both to be kI8 (ops.cpp:3050-3051).
  const void* k_base = k_cache.data;
  const void* v_base = v_cache.data;
  const KvKind kv_kind = kv_fp8 ? KvKind::kFp8 : KvKindOf(k_cache.dtype);
  // Query rows are widened with the shared `WidenRowToF32`; its byte cursor and
  // element size are the flat element indexing the per-element load did. An f32
  // query is already its own widened form — `WidenRowToF32` would `memcpy` it
  // onto itself — so that case reads the tensor in place and copies nothing.
  const uint8_t* q_bytes = static_cast<const uint8_t*>(query.data);
  const size_t q_esize = SizeOf(query.dtype);
  const float* q_f32 = query.dtype == DType::kF32 ? query.Ptr<float>() : nullptr;

  // Flatten the (request, local-token) nest into the global query-token index so
  // the work is one embarrassingly-parallel axis: at c1 prefill num_reqs==1, so
  // the request loop alone is serial (kPagedAttention profiled at 10% of prefill,
  // single-threaded). Precompute each token's absolute position p, its request's
  // seqlen, and its request row (for the block table) once on the caller — cheap
  // O(total_q) — then chunk the token rows across the pool. Each output row
  // out[(t*hq+h)*d] is produced by exactly one thread with the same per-element
  // math and j-reduction order as the serial code: bit-identical by construction.
  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;  // past positions before this chunk
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  // NEON lane, resolved ONCE per invocation. Opt-in again (the default-on
  // flip raced under load; ISSUE-LOCAL-01M4J4QNTE166QFG5MN7V3H4DG): an
  // explicit `=1` selects the lane, anything else runs the scalar body.
  // The scalar body below stays the reference and the non-aarch64 path.
#if defined(__aarch64__)
  const bool neon_active = PagedAttnNeonActive();
#else
  const bool neon_active = false;
  (void)neon_active;
#endif

  // The token loop, with the K and V element encodings bound at compile time.
  // Body text is unchanged from the per-element form apart from the two loads.
  auto run = [&](auto kv_tag) {
    ParallelForRows(CurrentThreadpool(), total_q, [&](int64_t t0, int64_t t1) {
      std::vector<float> probs;
      std::vector<float> acc(static_cast<size_t>(d));
      // The token's whole query row (hq*d contiguous elements), widened once and
      // reused by every q-head and every key of that token. Unused (and unsized)
      // when the query is already f32.
      std::vector<float> qrow;
      if (q_f32 == nullptr) qrow.resize(static_cast<size_t>(hq * d));
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t r = tok_req[static_cast<size_t>(t)];
        const int64_t p = tok_pos[static_cast<size_t>(t)];  // absolute position
        const int64_t seqlen = tok_slen[static_cast<size_t>(t)];
        const int64_t jmin = window_left >= 0 ? std::max<int64_t>(0, p - window_left) : 0;
        int64_t jmax = args.causal ? p : seqlen - 1;
        if (window_right >= 0) jmax = std::min(jmax, p + window_right);
        jmax = std::min(jmax, seqlen - 1);
        if (jmax < jmin) continue;
        probs.assign(static_cast<size_t>(jmax - jmin + 1), 0.0f);
        const float* qtok;
        if (q_f32 != nullptr) {
          qtok = q_f32 + t * hq * d;
        } else {
          WidenRowToF32(query.dtype, q_bytes + static_cast<size_t>(t * hq * d) * q_esize, hq * d,
                        qrow.data());
          qtok = qrow.data();
        }
        for (int64_t h = 0; h < hq; ++h) {
          const int64_t g = h / qpk;
          const int64_t qoff = (t * hq + h) * d_v;
          const float* q = qtok + h * d;  // == &query[(t*hq+h)*d], as f32
          // Pass 1: scores + running max.
          // SEEDED BY THE PER-Q-HEAD ATTENTION SINK when one is present.
          // A sink is one extra logit that joins the DENOMINATOR and
          // contributes NO value — expressed by starting the running max at
          // the sink and the running denominator at exp(0)=1. Absent
          // (sink_p == nullptr, every caller before MiMoV2) the seeds stay
          // -inf and 0, so the loop and output are bit-identical to the
          // pre-sink kernel.
          float m = sink_p != nullptr ? sink_p[h] : -std::numeric_limits<float>::infinity();
#if defined(__aarch64__)
          if (neon_active && d % 4 == 0 && d_v % 4 == 0) {
            // The NEON lane: same scores, same masking, same softmax passes.
            // Only the K reduction and the V accumulation run four lanes wide.
            for (int64_t j = jmin; j <= jmax; ++j) {
              const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
              const int64_t off = j % block_size;
              const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
              float32x4_t vacc = vdupq_n_f32(0.0f);
              for (int64_t e = 0; e < d; e += 4)
                vacc = vfmaq_f32(vacc, vld1q_f32(q + e),
                                 KvElem4<decltype(kv_tag)::value>(k_base, kbase + e, k_scale));
              // Lane-serial-in-block, block-sequential: NOT the scalar order.
              float dot = vaddvq_f32(vacc);
              dot *= scale;
              if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
              probs[static_cast<size_t>(j - jmin)] = dot;
              if (dot > m) m = dot;
            }
            float denom = sink_p != nullptr ? 1.0f : 0.0f;
            for (int64_t j = jmin; j <= jmax; ++j) {
              const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
              probs[static_cast<size_t>(j - jmin)] = e;
              denom += e;
            }
            const float inv = 1.0f / denom;
            for (int64_t e = 0; e < d_v; ++e) acc[static_cast<size_t>(e)] = 0.0f;
            for (int64_t j = jmin; j <= jmax; ++j) {
              const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
              const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
              const int64_t off = j % block_size;
              const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
              const float32x4_t pwv = vdupq_n_f32(pw);
              for (int64_t e = 0; e < d_v; e += 4) {
                float32x4_t a = vld1q_f32(acc.data() + e);
                a = vfmaq_f32(a, pwv, KvElem4<decltype(kv_tag)::value>(v_base, vbase + e, v_scale));
                vst1q_f32(acc.data() + e, a);
              }
            }
            StoreRowF32(out, qoff, d_v, acc.data());
          } else
#endif
          {
          for (int64_t j = jmin; j <= jmax; ++j) {
            const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
            const int64_t off = j % block_size;
            const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
            float dot = 0.0f;
            for (int64_t e = 0; e < d; ++e)
              dot += q[e] *
                     KvElem<decltype(kv_tag)::value>(k_base, kbase + e, k_scale);
            dot *= scale;
            if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
            probs[static_cast<size_t>(j - jmin)] = dot;
            if (dot > m) m = dot;
          }
          // Pass 2: exp + denominator. Seeded at 1.0 when a sink is
          // present (the sink's own exp(sink - m) contribution).
          float denom = sink_p != nullptr ? 1.0f : 0.0f;
          for (int64_t j = jmin; j <= jmax; ++j) {
            const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
            probs[static_cast<size_t>(j - jmin)] = e;
            denom += e;
          }
          const float inv = 1.0f / denom;  // every valid decoder/encoder window has >= 1 key
          // Pass 3: weighted sum of V (f32 accumulation), stored at out's dtype.
          for (int64_t e = 0; e < d_v; ++e) acc[static_cast<size_t>(e)] = 0.0f;
          for (int64_t j = jmin; j <= jmax; ++j) {
            const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
            const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
            const int64_t off = j % block_size;
            const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
            for (int64_t e = 0; e < d_v; ++e)
              acc[static_cast<size_t>(e)] +=
                  pw * KvElem<decltype(kv_tag)::value>(v_base, vbase + e, v_scale);
          }
          StoreRowF32(out, qoff, d_v, acc.data());
          }
        }
      }
    });
  };

  // ONE 4-way switch for the whole call, instead of a branch per element.
#define VT_KV_KIND_CASE(kind)                                \
  case KvKind::kind:                                         \
    run(std::integral_constant<KvKind, KvKind::kind>{});     \
    break
  switch (kv_kind) {
    VT_KV_KIND_CASE(kF32);
    VT_KV_KIND_CASE(kF16);
    VT_KV_KIND_CASE(kBF16);
    VT_KV_KIND_CASE(kFp8);
  }
#undef VT_KV_KIND_CASE
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kPagedAttention, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
