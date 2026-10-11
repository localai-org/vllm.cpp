// Kolibri-1 — CPU forward pass (MODEL-TEXT-kolibri-1 W2).
//
// Implements the hybrid forward the oracle pins in
// `aleph_alpha_inference/kolibri1.py` (pinned 049a6a7bd240):
//   - Sandwich (post-norm) decoder layer (kolibri1.py:236-253):
//     input_layernorm -> attention -> post_attn_norm -> residual add via
//     post_attention_layernorm -> MoE -> post_ffn_norm -> residual. The
//     residual stream accumulates the POST-NORMED attention output and the
//     post-ffn output, exactly as vLLM's residual-carrying RMSNorm contract
//     rounds it.
//   - Attention (kolibri1.py:39-123): GQA 48/4, per-head qk-norm BEFORE RoPE
//     (:107-118), scale head_dim**-0.5 (:100), full rotary on the SLIDING
//     layers (:91-95) and NO positional encoding at all on the full-attention
//     layers — RNoPE, :81-83. Sliding window 513 per-layer (:85-106).
//   - MoE (kolibri1.py:126-207): fp32 router logits, top-6 selected on
//     `logits + e_score_correction_bias`, weights = SIGMOID of the UNBIASED
//     logits, norm_topk_prob=false means NO renormalisation, plus one UNGATED
//     shared expert always added (:146-188).
//   - Untied lm_head; final norm carries the residual.
//
// CPU path only (the row's scope): the queue must be the CPU device; a GPU
// queue is refused by name — the GPU arm is a separate owed row.
//
// FP8 DISPOSITION (spec risk R1, owed record): the W1 loader stores fp8-block
// bytes RAW beside the f32 `weight_scale_inv` grid. This forward DEQUANTS EACH
// BLOCK WEIGHT TO BF16 on first use (scratch dequant, per call) and runs plain
// bf16 GEMMs — dequant-at-load semantics, the minimum-complete arm. A CPU
// fp8-block GEMM arm would avoid the 2x transient and is recorded as owed.
// Dequant is the standard w = fp8(w) * scale_inv[n/block_n, k/block_k] grid.

#include "vllm/model_executor/models/kolibri1_forward.h"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // StepInputs, KvSlice
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vllm/model_executor/models/kolibri1_dequant_cache.h"
#include "vllm/model_executor/models/kolibri1_bfdot_gemv.h"
#include "vllm/model_executor/models/kolibri1_shared.h"  // the routing + step inputs, shared with the TT row
#include "vllm/model_executor/models/kv_cache_route.h"     // WriteKvCache
#include "vllm/model_executor/models/host_parallel.h"  // the ONE pool (#1664)
#include "vt/ops.h"

// SCRATCH DEBUG (layer-1 moe localization): the CPU arm's last router logits,
// read by the TT arm's RouteDump for the direct per-expert diff.
std::vector<float> g_kolibri1_dbg_route_logits;
int64_t g_kolibri1_dbg_route_layer = -1;
std::vector<uint16_t> g_kolibri1_dbg_dhn;
int64_t g_kolibri1_dbg_dhn_layer = -1;

namespace vllm {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

namespace {

constexpr const char* kGpuRefusal =
    "Kolibri1ForCausalLM: the Kolibri-1 forward is implemented on the CPU "
    "path only in this row. Row MODEL-TEXT-kolibri-1, spec "
    ".agents/specs/kolibri-1-cpu.md — the GPU (CUDA/Tenstorrent) arm is a "
    "separate owed row.";

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ── Stage profiler (VT_KOLIBRI1_PROFILE=1; off by default, zero cost) ───────
// Wall-clock attribution for the CPU-profile evidence note. Accumulators are
// touched only from the serial caller thread, so plain statics suffice.
namespace prof {

std::map<std::string, double>& Acc() {
  static std::map<std::string, double> m;
  return m;
}
double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
bool On() {
  static const bool on = std::getenv("VT_KOLIBRI1_PROFILE") != nullptr;
  return on;
}
struct Scope {
  const char* k;
  double t0;
  explicit Scope(const char* key) : k(key), t0(Now()) {}
  ~Scope() {
    if (On()) Acc()[k] += Now() - t0;
  }
};
void Report(int64_t calls) {
  std::fprintf(stderr, "[kolibri1-profile] after %lld forwards:\n",
               static_cast<long long>(calls));
  for (const auto& [k, v] : Acc())
    std::fprintf(stderr, "[kolibri1-profile]   %-20s %10.2f s\n", k.c_str(), v);
  const auto& c = kolibri1_dequant_cache::Counters();
  std::fprintf(stderr,
               "[kolibri1-profile]   dequant_cache hits=%llu misses=%llu "
               "evictions=%llu decode_calls=%llu\n",
               static_cast<unsigned long long>(c.hits),
               static_cast<unsigned long long>(c.misses),
               static_cast<unsigned long long>(c.evictions),
               static_cast<unsigned long long>(c.decode_calls));
}

}  // namespace prof

// ── SCRATCH GEMV-site profiler (VT_GEMV_PROF; row/kolibri-gemv-overhead) ────
// Not for landing. Per-GEMV-site attribution: calls, elements, total wall in
// MatmulBT, and (from the pool counters in cpu_threadpool.cpp) the busy compute
// inside the kick vs the dispatch+sync remainder.
namespace gemvprof {

struct Rec {
  uint64_t calls = 0, elems = 0, wall_ns = 0, pool_ns = 0, busy_ns = 0, kicks = 0;
};
std::map<std::string, Rec>& Map() {
  static std::map<std::string, Rec> m;
  return m;
}
bool On() { static const bool on = std::getenv("VT_GEMV_PROF") != nullptr; return on; }
uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct Call {  // wraps ONE MatmulBT; destructs right after it returns
  const char* site;
  uint64_t elems, t0;
  vt::cpu::GemvProfSnap s0;
  Call(const char* s, uint64_t e)
      : site(s), elems(e), t0(NowNs()), s0(vt::cpu::GemvProfTake()) {}
  ~Call() {
    if (!On()) return;
    auto s1 = vt::cpu::GemvProfTake();
    Rec& r = Map()[site];
    r.calls++;
    r.elems += elems;
    r.wall_ns += NowNs() - t0;
    r.pool_ns += s1.pool_ns - s0.pool_ns;
    r.busy_ns += s1.busy_ns - s0.busy_ns;
    r.kicks += s1.kicks - s0.kicks;
  }
};
void Reset() { Map().clear(); vt::cpu::GemvProfReset(); }
void Report() {
  std::fprintf(stderr,
               "[gemvprof] %-12s %8s %14s %10s %10s %10s %8s\n", "site",
               "calls", "Melem", "wall-ms", "busy-ms", "disp-ms", "kicks");
  uint64_t c = 0, e = 0, w = 0, b = 0, k = 0;
  for (const auto& [s, r] : Map()) {
    uint64_t disp = r.pool_ns > r.busy_ns ? r.pool_ns - r.busy_ns : 0;
    std::fprintf(stderr,
                 "[gemvprof] %-12s %8llu %14.2f %10.2f %10.2f %10.2f %8llu\n",
                 s.c_str(), (unsigned long long)r.calls, r.elems / 1e6,
                 r.wall_ns / 1e6, r.busy_ns / 1e6, disp / 1e6,
                 (unsigned long long)r.kicks);
    c += r.calls; e += r.elems; w += r.wall_ns; b += r.busy_ns;
    k += r.kicks;
    uint64_t d = r.pool_ns > r.busy_ns ? r.pool_ns - r.busy_ns : 0;
    (void)d;
  }
  uint64_t disp = w > b ? w - b : 0;
  std::fprintf(stderr,
               "[gemvprof] %-12s %8llu %14.2f %10.2f %10.2f %10.2f %8llu\n",
               "TOTAL", (unsigned long long)c, e / 1e6, w / 1e6, b / 1e6,
               disp / 1e6, (unsigned long long)k);
  std::fprintf(stderr,
               "[gemvprof] Gelem/s: wall=%.2f busy=%.2f\n",
               e / (double)w, b ? e / (double)b : 0.0);
}
void ExitReport() { std::fprintf(stderr, "[gemvprof] ---- atexit ----\n"); Report(); }
bool RegisterExit() {
  if (On()) std::atexit(ExitReport);
  return On();
}

}  // namespace gemvprof

// ── FP8 block dequant (R1 disposition: dequant-to-bf16 scratch) ─────────────
// w[n, k] = fp8_e4m3(packed[n, k]) * scale_inv[n / block_n, k / block_k].
// Dequant happens in f32 per element and rounds ONCE to bf16 on store — the
// same single-rounding the loader's converting-copy semantics use.
DBuf DequantFp8Block(Dev d, const Fp8BlockWeight& w) {
  VT_CHECK(w.n > 0 && w.k > 0 && w.block_n > 0 && w.block_k > 0,
           "kolibri1 forward: degenerate fp8 block weight");
  const int64_t scale_cols = CDiv(w.k, w.block_k);
  const int64_t scale_rows = CDiv(w.n, w.block_n);
  VT_CHECK(static_cast<int64_t>(w.scale.bytes.size()) ==
               scale_rows * scale_cols * 4,
           "kolibri1 forward: fp8 scale grid is not f32 [cdiv(n,bn), "
           "cdiv(k,bk)]");
  VT_CHECK(static_cast<int64_t>(w.packed.bytes.size()) == w.n * w.k,
           "kolibri1 forward: fp8 packed bytes are not [n, k]");
  DBuf out(d, DType::kBF16, {w.n, w.k});
  prof::Scope prof("dequant_fp8_block");
  const auto* src = w.packed.bytes.data();
  const auto* sc = reinterpret_cast<const float*>(w.scale.bytes.data());
  auto* dst = reinterpret_cast<uint16_t*>(out.ptr());
  // THREADED over output rows through the ONE pool (host_parallel.h): the
  // loop is elementwise and partitions OUTPUT elements only, so the result is
  // bit-identical to the serial loop BY CONSTRUCTION (the pool determinism
  // contract). The row is the unit of work, never a K-chunk, so no
  // accumulation order moves. This dequant is the profiled #1 stage: the
  // fp8-block arm re-dequants every used weight PER CALL (~1.4 G elements per
  // decode step), all serial before this.
  host_parallel::ForOutputRows(
      w.n, w.k, [&](int64_t n0, int64_t n1) {
        // The NEON decode (aarch64) / scalar fallback body lives in
        // kolibri1_fp8_dequant.h, bit-identical to this file's previous
        // scalar loop elementwise (the bitwise doctest pins all 256 e4m3
        // bytes x the scale grid).
        kolibri1_fp8::DequantRowsBf16(src, sc, scale_cols, n0, n1, w.k,
                                      w.block_n, w.block_k, dst);
      });
  return out;
}

// One linear projection out[T, N] = x[T, K] @ W[N, K]^T (MatmulBT; the loader
// stores every projection in the raw [N, K] on-disk orientation). The bf16
// arm (the router gate) and the fp8-block arm (everything else) resolve here.
DBuf LinearBTRaw(Dev d, const Tensor& x, const OwnedTensor& wt_raw, int64_t t,
                 DType out_dtype = DType::kBF16,
                 const char* gsite = "other_bf16") {
  prof::Scope prof("linear_gemm");
  Tensor wt = ResidentWeight(d, wt_raw);
  DBuf out(d, out_dtype, {t, wt.shape[0]});
  gemvprof::Call gc(gsite, static_cast<uint64_t>(t) * wt.shape[0] * wt.shape[1]);
  vt::MatmulBT(d.q, out.t(), x, wt);
  return out;
}

// One MatmulBT weight, resolved the way LinearBT resolves it (bf16 arm
// resident; fp8 block arm through the dequant cache lease or the threaded
// decode — the same ownership comment as LinearBT's applies: the lease/buffer
// must outlive every GEMV that reads wt).
struct ResolvedBT {
  Tensor wt;
  DBuf wt_buf;
  kolibri1_dequant_cache::Lease dequant_lease;
};

ResolvedBT ResolveBT(Dev d, const Kolibri1Projection& w) {
  ResolvedBT r;
  if (w.IsBf16()) {
    r.wt = ResidentWeight(d, w.bf16);
    return r;
  }
  if (kolibri1_dequant_cache::ProcessCache().enabled()) {
    prof::Scope prof("dequant_fp8_block");  // cache lookups + cold decodes
    r.dequant_lease =
        kolibri1_dequant_cache::ProcessCache().GetOrDequant(w.fp8_block);
    r.wt = dense_attn::MakeTensor(const_cast<uint16_t*>(r.dequant_lease.data()),
                                  DType::kBF16, d.q.device,
                                  {w.fp8_block.n, w.fp8_block.k});
  } else {
    r.wt_buf = DequantFp8Block(d, w.fp8_block);
    r.wt = r.wt_buf.t();
  }
  return r;
}

DBuf LinearBT(Dev d, const Tensor& x, const Kolibri1Projection& w,              int64_t t, DType out_dtype = DType::kBF16,
              const char* gsite = "other") {
  if (w.IsBf16()) return LinearBTRaw(d, x, w.bf16, t, out_dtype, gsite);
  // The dequant cache (ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2 re-land, with
  // the 2026-10-08 review repair ISSUE-LOCAL-01M4CVDDHAFD7R1QCK9F493SWZ).
  // The cache is the fp8 decode path when enabled; the decoded block comes
  // back as a LEASE that co-owns the bytes, so the lease is held across the
  // GEMM below and released only after MatmulBT returns — the raw pointer
  // the cache returned before was unprotected the moment GetOrDequant
  // returned (a concurrent caller could evict and reuse the bytes mid-GEMM,
  // maint-bot P1b on PR #3414). The entries OWN their bytes (independent
  // allocations, never pool blocks — that ownership defect was v1's
  // corruption). Budget 0 (the default) DISABLES the cache and keeps the
  // original threaded pool decode, so main-line behavior is byte-for-byte
  // unchanged unless the env var is set.
  prof::Scope prof("linear_gemm");
  ResolvedBT r = ResolveBT(d, w);
  // BFDOT bf16 GEMV DRAFT arm, strictly opt-in (VT_CPU_KOLIBRI_BFDOT_GEMV=1,
  // default OFF) and t==1 decode-only: the warm-cache tensor is ALREADY bf16
  // [n, k] row-major (the dequant cache decode rounds to bf16 on store), so
  // the arm feeds those bytes straight into vbfdotq_f32 (bf16 in, f32
  // accumulate). This is a rounding-policy change versus the f32
  // widen+fma MatmulBT below — the chain WILL move under the knob; measured,
  // not hidden. Requires DetectArmBf16 (KnobActive refuses by name
  // otherwise); deliberately dotprod-free. The residency precondition is the
  // dequant cache itself: the arm exists to re-use the warm bf16 bytes, so
  // with the cache disabled the knob refuses and the f32-widen path below
  // runs. Covers every fp8-block projection that resolves here at decode:
  // attention q/k/v/o and the routed-expert gate/up/down. The router gate
  // (IsBf16) keeps its production path — it never was an fp8 warm-cache
  // tensor.
  if (t == 1 && kolibri1_bfdot_gemv::KnobActive() &&
      kolibri1_dequant_cache::ProcessCache().enabled()) {
    prof::Scope prof("linear_gemm");
    DBuf out(d, out_dtype, {1, w.fp8_block.n});
    kolibri1_bfdot_gemv::Gemv(static_cast<const uint16_t*>(r.wt.data),
                              static_cast<const uint16_t*>(x.data),
                              static_cast<uint16_t*>(out.ptr()),
                              w.fp8_block.n, w.fp8_block.k);
    return out;
  }
  DBuf out(d, out_dtype, {t, w.fp8_block.n});
  gemvprof::Call gc(gsite, static_cast<uint64_t>(t) * w.fp8_block.n *
                               w.fp8_block.k);
  vt::MatmulBT(d.q, out.t(), x, r.wt);
  return out;
}

// SwiGLU expert MLP: silu(x @ gate^T) * (x @ up^T), then down. The shared
bool BatchGemvOn();  // defined with the attention block; the A/B knob

// expert uses this (the routed experts ride the batched path in MoeBlock).
// With the batching knob on, gate+up share ONE dispatch (stage (b) sweep);
// same rows, same kernel — bit-identical.
DBuf ExpertMlp(Dev d, const Kolibri1ExpertWeights& e, const Tensor& x,
               int64_t t, int64_t inter, const char* gsite = "expert") {
  std::string sg = std::string(gsite) + "_gate", su = std::string(gsite) + "_up",
              sd = std::string(gsite) + "_down";
  if (BatchGemvOn()) {
    std::vector<ResolvedBT> wg, wu;
    wg.push_back(ResolveBT(d, e.gate_proj));
    wu.push_back(ResolveBT(d, e.up_proj));
    DBuf g(d, DType::kBF16, {t, inter});
    DBuf u(d, DType::kBF16, {t, inter});
    Tensor gt = g.t(), ut = u.t(), xx = x;
    std::vector<vt::MatmulBTJob> jobs{{&gt, &xx, &wg[0].wt},
                                      {&ut, &xx, &wu[0].wt}};
    gemvprof::Call gc(sg.c_str(), static_cast<uint64_t>(t) * inter *
                                      wg[0].wt.shape[1] * 2);
    vt::MatmulBTBatch(d.q, jobs.data(), static_cast<int64_t>(jobs.size()));
    DBuf a(d, DType::kBF16, {t, inter});
    vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
    return LinearBT(d, a.t(), e.down_proj, t, DType::kBF16, sd.c_str());
  }
  DBuf g = LinearBT(d, x, e.gate_proj, t, DType::kBF16, sg.c_str());  // [t, I]
  DBuf u = LinearBT(d, x, e.up_proj, t, DType::kBF16, su.c_str());    // [t, I]
  DBuf a(d, DType::kBF16, {t, inter});
  vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
  return LinearBT(d, a.t(), e.down_proj, t, DType::kBF16, sd.c_str());  // [t, H]
}

// ── Router: sigmoid-logit-add (kolibri1.py:126-142) ─────────────────────────
// The routing math lives in kolibri1_shared.h (a PURE RELOCATION out of this
// TU, shared verbatim with the Tenstorrent B2b-i device forward); the CPU
// row's op sequence is byte-identical to the pre-extraction form.

// ── Attention block (kolibri1.py:39-123) ────────────────────────────────────
// Same-binary A/B knob for the dispatch batching (stage (a) routed experts,
// stage (b) attention q/k/v). Default ON; =0 restores the per-call structure.
bool BatchGemvOn() {
  static const bool on = [] {
    const char* ev = std::getenv("VT_KOLIBRI1_BATCH_GEMV");
    return ev == nullptr || ev[0] != '0';
  }();
  return on;
}

// The batched q/k/v projection's output width for one projection.
int64_t BTOutN(const Kolibri1Projection& w) {
  return w.IsBf16() ? w.bf16.shape[0] : w.fp8_block.n;
}

DBuf AttentionBlock(Dev d, const Kolibri1AttnWeights& w, const Kolibri1Params& p,
                    bool is_sliding, const Tensor& dhn, const Tensor& positions,
                    const dense_attn::StepInputs& si, const PagedKvCache& kv,
                    int64_t t) {
  const int64_t hq = p.num_attention_heads;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t dh = p.head_dim;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dh)));

  // q/k/v projections (separate linears upstream, :64-79). Stage (b) of the
  // GEMV-overhead restructure: one vt::MatmulBTBatch dispatch instead of
  // three (the k/v matvecs were the two smallest kicks/layer). Same resolved
  // weights, same rows, same MatmulOneChunk kernel — bit-identical; the knob
  // restores the three per-call projections.
  DBuf q(d, DType::kBF16, {t, BTOutN(w.q_proj)});
  DBuf k(d, DType::kBF16, {t, BTOutN(w.k_proj)});
  DBuf v(d, DType::kBF16, {t, BTOutN(w.v_proj)});
  if (!BatchGemvOn()) {
    q = LinearBT(d, dhn, w.q_proj, t, DType::kBF16, "attn_q");
    k = LinearBT(d, dhn, w.k_proj, t, DType::kBF16, "attn_k");
    v = LinearBT(d, dhn, w.v_proj, t, DType::kBF16, "attn_v");
  } else {
    std::vector<ResolvedBT> wq, wk, wv;
    wq.push_back(ResolveBT(d, w.q_proj));
    wk.push_back(ResolveBT(d, w.k_proj));
    wv.push_back(ResolveBT(d, w.v_proj));
    Tensor qt = q.t(), kt = k.t(), vt_t = v.t(), x = dhn;
    std::vector<vt::MatmulBTJob> jobs{
        {&qt, &x, &wq[0].wt}, {&kt, &x, &wk[0].wt}, {&vt_t, &x, &wv[0].wt}};
    const uint64_t elems = static_cast<uint64_t>(t) *
                           (BTOutN(w.q_proj) + BTOutN(w.k_proj) +
                            BTOutN(w.v_proj)) *
                           BTOutN(w.q_proj);  // K == Hq*Dh for all three
    gemvprof::Call gc("attn_qkv", elems);
    vt::MatmulBTBatch(d.q, jobs.data(), static_cast<int64_t>(jobs.size()));
  }

  // Per-head qk-norm BEFORE RoPE (:107-118): RMSNorm(head_dim) per head.
  // Reshape [T, H*Dh] -> [T*H, Dh] and run the shared RmsNorm with the
  // per-head gamma — every head shares one gamma vector.
  Tensor q3 = Reshape(q.t(), {t, hq, dh});
  Tensor k3 = Reshape(k.t(), {t, hkv, dh});
  Tensor v3 = Reshape(v.t(), {t, hkv, dh});
  // qn/kn OWN the storage q3/k3 are re-pointed at below: RoPE (:260) and the
  // KV-cache write both read it, so the DBufs must live in THIS scope, not
  // the inner one. A DBuf destroyed at a closing brace returns its block to
  // the DevicePool — freed outright under VT_POOL_BYPASS=1 — while later ops
  // still read it (ASan/TSan heap-use-after-free,
  // ISSUE-LOCAL-01M4FK2A9T7BPB0KHPAA8TN1VA). The house pattern for exactly
  // this norm-then-use structure is kimi_linear_device.cpp:708-717, which
  // declares its qn/kn in the enclosing scope.
  DBuf qn(d, DType::kBF16, {t * hq, dh});
  DBuf kn(d, DType::kBF16, {t * hkv, dh});
  {
    Tensor qh = Reshape(q.t(), {t * hq, dh});
    Tensor kh = Reshape(k.t(), {t * hkv, dh});
    Tensor qw = ResidentWeight(d, w.q_norm);
    Tensor kw = ResidentWeight(d, w.k_norm);
    prof::Scope prof("norms");
    vt::RmsNorm(d.q, qn.t(), qh, qw, vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    vt::RmsNorm(d.q, kn.t(), kh, kw, vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    q3 = Reshape(qn.t(), {t, hq, dh});
    k3 = Reshape(kn.t(), {t, hkv, dh});
  }

  // RNoPE (:81-95): RoPE on the SLIDING layers only; full-attention layers
  // carry NO positional encoding. Full rotary: rotary_dim == head_dim.
  if (is_sliding) {
    vt::RopeArgs ra{};
    ra.base = static_cast<float>(p.rope_theta);
    ra.rotary_dim = static_cast<int>(dh);
    ra.is_neox_style = true;
    prof::Scope prof("attn_rope");
    vt::RopeNeox(d.q, q3, k3, positions, ra);
  }

  // KV cache write + paged attention. Sliding layers cap the window at 513
  // (:85-106): AttentionWindow{left = W-1, right = 0} is the causal decoder
  // window of W tokens (the mimo-v2 house convention).
  Tensor k_cache = dense_attn::KvSlice(kv, d.q.device, 0);
  Tensor v_cache = dense_attn::KvSlice(kv, d.q.device, 1);
  DBuf attn(d, DType::kBF16, {t, hq, dh});
  {
    prof::Scope prof("attn_core");
    vt::PagedAttentionArgs pa{};
    pa.scale = scale;
    pa.causal = true;
    if (is_sliding) {
      pa.window_size = vt::AttentionWindow{
          static_cast<int32_t>(p.sliding_window - 1), 0};
    } else {
      pa.window_size = std::nullopt;  // full attention, RNoPE or not
    }
    dense_attn::ApplyKvCacheQuant(pa, kv);
    dense_attn::WriteKvCache(d.q, kv, k3, v3, k_cache, v_cache,
                             si.slot_mapping.t());
    vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache,
                       si.block_table.t(), si.seq_lens.t(),
                       si.query_start_loc.t(), pa);
  }

  Tensor o_in = Reshape(attn.t(), {t, hq * dh});
  if (std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") != nullptr) {
    std::vector<uint8_t> tmp(static_cast<size_t>(t) * hq * dh * 2);
    attn.Download(d, tmp.data());
    const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
    double s = 0;
    auto val = [](uint16_t bits) {
      uint32_t u = static_cast<uint32_t>(bits) << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const int64_t n = t * hq * dh;
    for (int64_t i = 0; i < n; ++i) s += val(bf[i]);
    std::fprintf(stderr, "[STAGE][CPU] pa_out sum=%.6f first=%.6f\n", s, val(bf[0]));
  }
  return LinearBT(d, o_in, w.o_proj, t, DType::kBF16, "attn_o");  // [T, H]
}

// ── MoE block (kolibri1.py:126-207) ─────────────────────────────────────────

// SCRATCH DEBUG (layer-1 moe localization, defined after StageDump below;
// forward-declared here because MoeBlock calls it):
void StageDump(const char* tag, int64_t layer, DBuf& buf, Dev d);
void RouteDump(int64_t layer, const std::vector<float>& logits,
               const Kolibri1HostRouting& route, int64_t t, int64_t e,
               int64_t top_k);

DBuf MoeBlock(Dev d, const Kolibri1MoeWeights& w, const Kolibri1Params& p,
              const Tensor& dhn, int64_t t, int64_t layer) {
  const int64_t h = p.hidden_size;
  const int64_t e = p.num_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t inter = p.moe_intermediate_size;

  // Router logits, f32 (:171-177).
  DBuf dlog = LinearBTRaw(d, dhn, w.router_gate, t, DType::kF32, "router");
  std::vector<float> logits(static_cast<size_t>(t * e));
  {
    prof::Scope prof("moe_d2h");
    dlog.Download(d, logits.data());
  }
  std::vector<float> bias(static_cast<size_t>(e));
  {
    prof::Scope prof("moe_d2h");
    Tensor bt = ResidentWeight(d, w.e_score_correction_bias, {e});
    vt::Backend& be = vt::GetBackend(d.q.device.type);
    be.Copy(d.q, bias.data(), bt.data, static_cast<size_t>(e) * sizeof(float));
    be.Synchronize(d.q);
  }

  if (std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") != nullptr) {
    static std::set<int64_t> seen;
    if (!seen.count(layer)) {
      seen.insert(layer);
      Tensor bt3 = ResidentWeight(d, w.router_gate, {e, h});
      const auto* bf3 = reinterpret_cast<const uint16_t*>(bt3.data);
      uint32_t hh = 2166136261u;
      double ss = 0;
      for (int64_t i = 0; i < e * h; ++i) {
        hh = (hh ^ bf3[i]) * 16777619u;
        uint32_t u = static_cast<uint32_t>(bf3[i]) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        ss += f;
      }
      std::fprintf(stderr, "[STAGE][CPU] router_gate hash=%08x sum=%.4f\n",
                   hh, ss);
      // Host reference logits from the same dhn + router_gate bytes.
      const int64_t hk2 = w.router_gate.shape[1];
      std::vector<uint8_t> xb(static_cast<size_t>(t) * hk2 * 2);
      std::memcpy(xb.data(), dhn.data, xb.size());
      const auto* xb16 = reinterpret_cast<const uint16_t*>(xb.data());
      std::vector<float> href(static_cast<size_t>(t) * e, 0.f);
      for (int64_t n0 = 0; n0 < e; ++n0) {
        const auto* wr =
            reinterpret_cast<const uint16_t*>(bt3.data) +
            static_cast<size_t>(n0) * hk2;
        for (int64_t i = 0; i < t; ++i) {
          double acc = 0;
          for (int64_t k = 0; k < hk2; ++k) {
            auto dec = [](uint16_t b) {
              uint32_t u = static_cast<uint32_t>(b) << 16;
              float f;
              std::memcpy(&f, &u, 4);
              return static_cast<double>(f);
            };
            acc += dec(wr[k]) * dec(xb16[static_cast<size_t>(i) * hk2 + k]);
          }
          href[static_cast<size_t>(i) * e + n0] = static_cast<float>(acc);
        }
      }
      double madd = 0, mlog = 0;
      for (size_t k = 0; k < href.size(); ++k) {
        madd = std::max(madd, std::fabs(static_cast<double>(href[k]) -
                                        static_cast<double>(logits[k])));
        mlog = std::max(
            mlog,
            std::fabs(static_cast<double>(href[k]) -
                      static_cast<double>(g_kolibri1_dbg_route_logits.empty()
                                              ? 0.f
                                              : href[k])));
      }
      std::fprintf(stderr, "[STAGE][CPU] router_href maxdiff_dev=%.6f\n",
                   madd);
    }
  }
  Kolibri1HostRouting route;
  {
    prof::Scope prof("moe_topk");
    route = SigmoidLogitAddRouting(logits, bias, t, e, top_k);
  }
  RouteDump(layer, logits, route, t, e, top_k);

  DBuf dtw(d, DType::kF32, {t, top_k},
           const_cast<float*>(route.weights.data()));
  DBuf dtid(d, DType::kI32, {t, top_k},
            const_cast<int32_t*>(route.ids.data()));
  // Shared expert: UNGATED, always added (:146-188).
  DBuf shared = ExpertMlp(d, w.shared_experts, dhn, t,
                          p.shared_expert_intermediate_size, "shared");
  StageDump("moe_shared", layer, shared, d);

  // Routed experts: the CPU-reference per-expert gather/scatter loop (the
  // mimo-v2 MoeBlock pattern), restructured for dispatch batching
  // (row MODEL-TEXT-kolibri-1, stage (a) of the GEMV-overhead restructure).
  // The gather order, the per-expert token/slot mapping, the per-row GEMV
  // math (same MatmulOneChunk kernel, same sequential-over-K accumulation)
  // and the scatter are EXACTLY the per-expert loop's; the only change is
  // which pool dispatch carries the rows: all active experts' gate+up GEMVs
  // ride ONE vt::MatmulBTBatch, the down GEMVs another (2 kicks/layer
  // instead of 18). BIT-IDENTICAL by construction; gated byte-for-byte by
  // tests/vt/test_ops_matmul_batch.cpp.
  prof::Scope prof_glue("moe_glue");
  // Same-binary A/B for the dispatch batching (see BatchGemvOn). OFF restores
  // the per-expert loop byte-for-byte.
  const bool kBatchGemv = BatchGemvOn();
  DBuf expert_out(d, DType::kBF16, {t, top_k, h});
  expert_out.Zero(d);
  if (!kBatchGemv) {
    // The CPU-reference per-expert gather/scatter loop (the mimo-v2 MoeBlock
    // pattern), unchanged.
    for (int64_t ex = 0; ex < e; ++ex) {
      std::vector<int32_t> token_rows;
      std::vector<int32_t> slot_rows;
      for (int64_t i = 0; i < t; ++i) {
        for (int64_t kk = 0; kk < top_k; ++kk) {
          const int64_t idx = i * top_k + kk;
          if (route.ids[static_cast<size_t>(idx)] == static_cast<int32_t>(ex)) {
            token_rows.push_back(static_cast<int32_t>(i));
            slot_rows.push_back(static_cast<int32_t>(idx));
            break;
          }
        }
      }
      if (token_rows.empty()) continue;
      const int64_t ne = static_cast<int64_t>(token_rows.size());
      DBuf gathered(d, DType::kBF16, {ne, h});
      {
        const size_t rb = static_cast<size_t>(h) * vt::SizeOf(DType::kBF16);
        auto* dp = static_cast<char*>(gathered.ptr());
        const auto* sp = static_cast<const char*>(dhn.data);
        for (size_t s = 0; s < token_rows.size(); ++s)
          d.b.Copy(d.q, dp + s * rb,
                   sp + static_cast<size_t>(token_rows[s]) * rb, rb);
      }
      DBuf o = ExpertMlp(d, w.experts[static_cast<size_t>(ex)], gathered.t(),
                         ne, inter, "routed");
      for (int64_t i = 0; i < ne; ++i) {
        d.b.Copy(d.q,
                 static_cast<char*>(expert_out.ptr()) +
                     static_cast<size_t>(slot_rows[static_cast<size_t>(i)]) *
                         static_cast<size_t>(h) * vt::SizeOf(DType::kBF16),
                 static_cast<const char*>(o.ptr()) +
                     static_cast<size_t>(i) * static_cast<size_t>(h) *
                         vt::SizeOf(DType::kBF16),
                 static_cast<size_t>(h) * vt::SizeOf(DType::kBF16));
      }
    }
  } else {
  // Routed experts: the CPU-reference per-expert gather/scatter loop (the
  // mimo-v2 MoeBlock pattern), restructured for dispatch batching
  // (row MODEL-TEXT-kolibri-1, stage (a) of the GEMV-overhead restructure).
  // The gather order, the per-expert token/slot mapping, the per-row GEMV
  // math (same MatmulOneChunk kernel, same sequential-over-K accumulation)
  // and the scatter are EXACTLY the per-expert loop's; the only change is
  // which pool dispatch carries the rows: all active experts' gate+up GEMVs
  // ride ONE vt::MatmulBTBatch, the down GEMVs another (2 kicks/layer
  // instead of 18). BIT-IDENTICAL by construction; gated byte-for-byte by
  // tests/vt/test_ops_matmul_batch.cpp.
  struct RoutedRun {
    int64_t ex;
    std::vector<int32_t> token_rows;
    std::vector<int32_t> slot_rows;
    int64_t row0;  // this run's rows inside the batched gather/act/out buffers
  };
  std::vector<RoutedRun> runs;
  int64_t total_rows = 0;
  for (int64_t ex = 0; ex < e; ++ex) {
    RoutedRun run;
    run.ex = ex;
    for (int64_t i = 0; i < t; ++i) {
      for (int64_t kk = 0; kk < top_k; ++kk) {
        const int64_t idx = i * top_k + kk;
        if (route.ids[static_cast<size_t>(idx)] == static_cast<int32_t>(ex)) {
          run.token_rows.push_back(static_cast<int32_t>(i));
          run.slot_rows.push_back(static_cast<int32_t>(idx));
          break;
        }
      }
    }
    if (run.token_rows.empty()) continue;
    run.row0 = total_rows;
    total_rows += static_cast<int64_t>(run.token_rows.size());
    runs.push_back(std::move(run));
  }
  if (!runs.empty()) {
    // A rank-2 row slice [nrows, cols] of `src` starting at `row0`. Views
    // only — the underlying DBuf owns the bytes and outlives the batch call.
    auto rows_view = [](const Tensor& src, int64_t row0, int64_t nrows) {
      Tensor v = src;
      v.rank = 2;
      v.shape[0] = nrows;
      v.shape[1] = src.shape[1];
      v.stride[0] = src.stride[0];
      v.stride[1] = src.stride[1];
      v.data = static_cast<char*>(v.data) +
               row0 * src.stride[0] * vt::SizeOf(src.dtype);
      return v;
    };

    DBuf gathered(d, DType::kBF16, {total_rows, h});
    for (const RoutedRun& run : runs) {
      const size_t rb = static_cast<size_t>(h) * vt::SizeOf(DType::kBF16);
      auto* dp = static_cast<char*>(gathered.ptr()) +
                 static_cast<size_t>(run.row0) * rb;
      const auto* sp = static_cast<const char*>(dhn.data);
      for (size_t s = 0; s < run.token_rows.size(); ++s)
        d.b.Copy(d.q, dp + s * rb,
                 sp + static_cast<size_t>(run.token_rows[s]) * rb, rb);
    }

    // Resolve every active expert's weights BEFORE the first batch call: the
    // dequant-cache leases must outlive both dispatches.
    std::vector<ResolvedBT> wg, wu, wd;
    wg.reserve(runs.size());
    wu.reserve(runs.size());
    wd.reserve(runs.size());
    for (const RoutedRun& run : runs) {
      const Kolibri1ExpertWeights& exw = w.experts[static_cast<size_t>(run.ex)];
      wg.push_back(ResolveBT(d, exw.gate_proj));
      wu.push_back(ResolveBT(d, exw.up_proj));
      wd.push_back(ResolveBT(d, exw.down_proj));
    }

    DBuf gout(d, DType::kBF16, {total_rows, inter});
    DBuf uout(d, DType::kBF16, {total_rows, inter});
    {
      std::vector<Tensor> ag, au, og, ou;
      ag.reserve(runs.size());
      au.reserve(runs.size());
      og.reserve(runs.size());
      ou.reserve(runs.size());
      std::vector<vt::MatmulBTJob> jobs;
      jobs.reserve(runs.size() * 2);
      uint64_t elems = 0;
      for (size_t i = 0; i < runs.size(); ++i) {
        const int64_t ne =
            static_cast<int64_t>(runs[i].token_rows.size());
        ag.push_back(rows_view(gathered.t(), runs[i].row0, ne));
        au.push_back(rows_view(gathered.t(), runs[i].row0, ne));
        og.push_back(rows_view(gout.t(), runs[i].row0, ne));
        ou.push_back(rows_view(uout.t(), runs[i].row0, ne));
        jobs.push_back({&og.back(), &ag.back(), &wg[i].wt});
        jobs.push_back({&ou.back(), &au.back(), &wu[i].wt});
        elems += static_cast<uint64_t>(ne) * inter * h;
      }
      gemvprof::Call gc("routed_gate", elems);  // gate+up share the dispatch
      vt::MatmulBTBatch(d.q, jobs.data(), static_cast<int64_t>(jobs.size()));
    }

    // silu(x @ gate^T) * (x @ up^T) per expert run — the same MoeSiluMul on
    // the same operand rows as the per-expert loop, one call per run.
    DBuf act(d, DType::kBF16, {total_rows, inter});
    for (RoutedRun& run : runs) {
      const int64_t ne = static_cast<int64_t>(run.token_rows.size());
      Tensor ag = rows_view(gout.t(), run.row0, ne);
      Tensor au = rows_view(uout.t(), run.row0, ne);
      Tensor aa = rows_view(act.t(), run.row0, ne);
      vt::MoeSiluMul(d.q, aa, ag, au);
    }

    DBuf dout(d, DType::kBF16, {total_rows, h});
    {
      std::vector<Tensor> aa, od;
      aa.reserve(runs.size());
      od.reserve(runs.size());
      std::vector<vt::MatmulBTJob> jobs;
      jobs.reserve(runs.size());
      uint64_t elems = 0;
      for (size_t i = 0; i < runs.size(); ++i) {
        const int64_t ne =
            static_cast<int64_t>(runs[i].token_rows.size());
        aa.push_back(rows_view(act.t(), runs[i].row0, ne));
        od.push_back(rows_view(dout.t(), runs[i].row0, ne));
        jobs.push_back({&od.back(), &aa.back(), &wd[i].wt});
        elems += static_cast<uint64_t>(ne) * inter * h;
      }
      gemvprof::Call gc("routed_down", elems);
      vt::MatmulBTBatch(d.q, jobs.data(), static_cast<int64_t>(jobs.size()));
    }

    // Scatter each run's rows to their (token, top-k) slots — unchanged.
    for (const RoutedRun& run : runs) {
      for (int64_t i = 0; i < static_cast<int64_t>(run.slot_rows.size()); ++i) {
        d.b.Copy(d.q,
                 static_cast<char*>(expert_out.ptr()) +
                     static_cast<size_t>(run.slot_rows[static_cast<size_t>(i)]) *
                         static_cast<size_t>(h) * vt::SizeOf(DType::kBF16),
                 static_cast<const char*>(dout.ptr()) +
                     static_cast<size_t>(run.row0 + i) *
                         static_cast<size_t>(h) * vt::SizeOf(DType::kBF16),
                 static_cast<size_t>(h) * vt::SizeOf(DType::kBF16));
      }
    }
  }
  }  // kBatchGemv batched path

  // Weighted combine + the always-added shared term. routed_scale 1.0: the
  // router weights ARE the mixture weights (no renormalisation, no scaling).
  DBuf out(d, DType::kBF16, {t, h});
  Tensor shared_t = shared.t();
  vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared_t,
                 /*routed_scale=*/1.0f);
  return out;
}

// SCRATCH DEBUG (root-cause the host-free decode corruption): per-stage
// activation checksums mirroring the TT arm's dump (same tags, same env), so
// the TT stage outputs diff against the CPU reference stage by stage.
bool StageDumpOn() {
  static const bool on = std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") != nullptr;
  return on;
}

void StageDump(const char* tag, int64_t layer, DBuf& buf, Dev d) {
  if (!StageDumpOn()) return;
  const Tensor& t = buf.t();
  const int64_t n = t.shape[0] * t.shape[1];
  std::vector<uint8_t> tmp(static_cast<size_t>(n) * 2);
  buf.Download(d, tmp.data());
  const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
  double sum = 0, sq = 0;
  float mx = 0;
  uint32_t hash = 2166136261u;
  auto val = [](uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  for (int64_t i = 0; i < n; ++i) {
    const float v = val(bf[i]);
    sum += v;
    sq += static_cast<double>(v) * v;
    if (std::fabs(v) > mx) mx = std::fabs(v);
    hash = (hash ^ bf[i]) * 16777619u;
  }
  std::fprintf(stderr, "[STAGE][CPU] L%lld %s sum=%.6f sq=%.6e hash=%08x max=%.6f first=[%.6f %.6f %.6f %.6f]\n",
               static_cast<long long>(layer), tag, sum, sq, hash, mx, val(bf[0]), val(bf[1]),
               val(bf[2]), val(bf[3]));
}
// SCRATCH DEBUG (layer-1 moe localization, same env as StageDump): the
// router's raw logits, top-k selection and weights per token — the
// discrete-selection leg of the OFF-vs-CPU stage diff.
void RouteDump(int64_t layer, const std::vector<float>& logits,
               const Kolibri1HostRouting& route, int64_t t, int64_t e,
               int64_t top_k) {
  if (std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") == nullptr) return;
  for (int64_t i = 0; i < t; ++i) {
    double ls = 0;
    for (int64_t j = 0; j < e; ++j)
      ls += logits[static_cast<size_t>(i * e + j)];
    std::string ids, ws;
    for (int64_t k = 0; k < top_k; ++k) {
      const size_t idx = static_cast<size_t>(i * top_k + k);
      ids += (k ? "," : "") + std::to_string(route.ids[idx]);
      ws += (k ? "," : "") +
            [](float v) {
              char b[32];
              std::snprintf(b, sizeof b, "%.6f", v);
              return std::string(b);
            }(route.weights[idx]);
    }
    std::fprintf(stderr,
                 "[STAGE][CPU] L%lld route tok=%lld logit_sum=%.6f ids=[%s] "
                 "w=[%s]\n",
                 static_cast<long long>(layer), static_cast<long long>(i), ls,
                 ids.c_str(), ws.c_str());
  }
  g_kolibri1_dbg_route_logits = logits;
  g_kolibri1_dbg_route_layer = layer;
}


}  // namespace

ForwardLogits ForwardKolibri1Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Kolibri1Weights& weights,
    const MultiKvCacheIndex* multi_kv, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU, kGpuRefusal);
  const Kolibri1Params& p = weights.params;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t t = static_cast<int64_t>(token_ids.size());
  const int64_t h = p.hidden_size;
  const int64_t vocab = p.vocab_size;

  VT_CHECK(t > 0, "kolibri1: empty token batch");
  VT_CHECK(static_cast<int64_t>(positions.size()) == t,
           "kolibri1: positions size mismatch");
  VT_CHECK(static_cast<int64_t>(attn_meta.slot_mapping.size()) == t,
           "kolibri1: slot_mapping size mismatch");
  const double prof_t0 = prof::Now();

  // Embedding (bf16 table, [vocab, H] raw orientation).
  DBuf hidden_buf(d, DType::kBF16, {t, h});
  {
    DBuf ids(d, DType::kI32, {t}, const_cast<int32_t*>(token_ids.data()));
    Tensor tab = ResidentWeight(d, weights.embed_tokens, {vocab, h});
    vt::Embedding(d.q, hidden_buf.t(), tab, ids.t());
  }
  StageDump("emb", -2, hidden_buf, d);
  Tensor hidden = hidden_buf.t();
  DBuf res(d, DType::kBF16, {t, h});
  res.Zero(d);

  const float eps = static_cast<float>(p.rms_norm_eps);
  std::shared_ptr<void> hidden_hold;

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const Kolibri1LayerWeights& lw = weights.layers[static_cast<size_t>(l)];

    const PagedKvCache* kv_ptr = nullptr;
    if (multi_kv != nullptr) {
      const std::string name =
          "model.layers." + std::to_string(l) + ".self_attn";
      const int64_t idx = multi_kv->Find(name);
      VT_CHECK(idx >= 0 && idx < static_cast<int64_t>(attn_kv.size()),
               "kolibri1: KV cache not found for layer " + std::to_string(l));
      kv_ptr = &attn_kv[static_cast<size_t>(idx)];
    } else {
      VT_CHECK(l < static_cast<int64_t>(attn_kv.size()),
               "kolibri1: KV cache missing for layer " + std::to_string(l));
      kv_ptr = &attn_kv[static_cast<size_t>(l)];
    }

    dense_attn::StepInputs si = Kolibri1BuildStepInputs(d, attn_meta, positions);

    // input_layernorm + residual (the vLLM fused add-norm contract).
    DBuf dhn(d, DType::kBF16, {t, h});
    Tensor w_in = ResidentWeight(d, lw.input_layernorm, {h});
    if (StageDumpOn() && l == 0) {
      // SCRATCH: the gamma bytes as the device arm would see them (bf16).
      double s = 0;
      auto val = [](uint16_t bits) {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
      };
      std::vector<uint16_t> bf(static_cast<size_t>(h));
      if (w_in.dtype == vt::DType::kBF16) {
        const auto* p16 = static_cast<const uint16_t*>(w_in.data);
        for (int64_t i = 0; i < h; ++i) bf[static_cast<size_t>(i)] = p16[i];
      } else {
        const auto* pf = static_cast<const float*>(w_in.data);
        for (int64_t i = 0; i < h; ++i) {
          float v = pf[i];
          std::memcpy(&bf[static_cast<size_t>(i)], &v, 2);  // low bits (trunc)
        }
      }
      for (int64_t i = 0; i < h; ++i) s += val(bf[static_cast<size_t>(i)]);
      std::fprintf(stderr, "[STAGE][CPU] gam sum=%.6f first=[%.6f %.6f %.6f %.6f]\n",
                   s, val(bf[0]), val(bf[1]), val(bf[2]), val(bf[3]));
    }
    Tensor dhn_t = dhn.t();
    Tensor res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      prof::Scope prof("norms");
      vt::FusedChain(d.q, dhn_t, hidden, w_in, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, dhn_t, hidden, w_in,
                  vt::RmsNormArgs{eps, false}, &res_t);
    }

    // Attention -> post_attn_norm (no residual).
    StageDump("dhn", l, dhn, d);
    if (l == 0 && std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") != nullptr) {
      const int64_t hn = dhn.t().shape[0] * dhn.t().shape[1];
      g_kolibri1_dbg_dhn.assign(
          reinterpret_cast<const uint16_t*>(dhn.t().data),
          reinterpret_cast<const uint16_t*>(dhn.t().data) + hn);
      g_kolibri1_dbg_dhn_layer = l;
    }
    DBuf attn = AttentionBlock(d, lw.attn, p, lw.is_sliding, dhn.t(),
                               si.positions.t(), si, *kv_ptr, t);
    StageDump("attn", l, attn, d);
    DBuf attn_n(d, DType::kBF16, {t, h});
    Tensor w_pa = ResidentWeight(d, lw.post_attn_norm, {h});
    {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, vt::RmsNormArgs{eps, false});
    }
    StageDump("attn_n", l, attn_n, d);

    // post_attention_layernorm carries the residual: residual += the
    // POST-NORMED attention output (kolibri1.py:250).
    DBuf dh2(d, DType::kBF16, {t, h});
    Tensor w_pal = ResidentWeight(d, lw.post_attention_layernorm, {h});
    Tensor dh2_t = dh2.t();
    res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      prof::Scope prof("norms");
      vt::FusedChain(d.q, dh2_t, attn_n.t(), w_pal, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, dh2_t, attn_n.t(), w_pal, vt::RmsNormArgs{eps, false},
                  &res_t);
    }
    StageDump("dh2", l, dh2, d);

    // MoE on EVERY layer -> post_ffn_norm (no residual).
    DBuf moe = MoeBlock(d, lw.moe, p, dh2.t(), t, l);
    StageDump("moe", l, moe, d);
    DBuf moe_n(d, DType::kBF16, {t, h});
    Tensor w_pf = ResidentWeight(d, lw.post_ffn_norm, {h});
    {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, moe_n.t(), moe.t(), w_pf, vt::RmsNormArgs{eps, false});
    }
    StageDump("moe_n", l, moe_n, d);

    auto* held = new DBuf(std::move(moe_n));
    hidden = held->t();
    hidden_hold = std::shared_ptr<void>(held, [](void* q) {
      delete static_cast<DBuf*>(q);
    });
  }

  // Final norm carries the residual (Qwen3MoeModel.norm(hidden, residual)).
  DBuf dnorm(d, DType::kBF16, {t, h});
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {h});
  Tensor dnorm_t = dnorm.t();
  Tensor res_t = res.t();
  if (dense_attn::FusedChainAdoptEnabled()) {
    prof::Scope prof("norms");
    vt::FusedChain(d.q, dnorm_t, hidden, w_fn, &res_t,
                   vt::kFusedAddRmsNormStd, eps);
  } else {
    prof::Scope prof("norms");
    vt::RmsNorm(d.q, dnorm_t, hidden, w_fn, vt::RmsNormArgs{eps, false},
                &res_t);
  }

  // logits_indices gather, then the UNTIED lm_head.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < t;
  const int64_t n_idx = static_cast<int64_t>(logits_indices.size());
  DBuf dgather(d, DType::kBF16, {do_gather ? n_idx : int64_t{0}, h});
  Tensor src = dnorm.t();
  if (do_gather) {
    const size_t rb = static_cast<size_t>(h) * vt::SizeOf(DType::kBF16);
    auto* dp = static_cast<char*>(dgather.ptr());
    const auto* sp = static_cast<const char*>(dnorm.t().data);
    for (size_t s = 0; s < logits_indices.size(); ++s)
      d.b.Copy(d.q, dp + s * rb,
               sp + static_cast<size_t>(logits_indices[s]) * rb, rb);
    src = dgather.t();
  }
  const int64_t n_out = do_gather ? n_idx : t;

  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, h});
  DBuf logits(d, DType::kF32, {n_out, vocab});
  {
    prof::Scope prof("lm_head");
    gemvprof::Call gc("lm_head", static_cast<uint64_t>(n_out) * vocab * h);
    vt::MatmulBT(d.q, logits.t(), src, lm);
  }

  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = vocab;
  fl.device_tensor = logits.t();
  fl.device_storage = logits.ReleaseShared();
  if (gemvprof::On()) {
    static bool exit_registered = gemvprof::RegisterExit();
    (void)exit_registered;
    if (t > 1) {
      gemvprof::Reset();  // drop the prefill; count decode steps only
    } else {
      static int64_t steps = 0;
      if (++steps % 16 == 0) {
        std::fprintf(stderr, "[gemvprof] ---- window after %lld decode steps "
                             "(cumulative since last reset) ----\n",
                     static_cast<long long>(steps));
        gemvprof::Report();
        gemvprof::Reset();
      }
    }
  }
  if (prof::On()) {
    prof::Acc()["total_forward"] += prof::Now() - prof_t0;
    static int64_t calls = 0;
    if (++calls % 10 == 0) prof::Report(calls);
  }
  return fl;
}

}  // namespace vllm
