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
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // StepInputs, KvSlice
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/kv_cache_route.h"     // WriteKvCache
#include "vt/fp8_kv.h"                                     // F8E4M3ToF32
#include "vllm/model_executor/models/host_parallel.h"  // the ONE pool (#1664)
#include "vt/ops.h"

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
}

}  // namespace prof

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
        for (int64_t n = n0; n < n1; ++n) {
          const int64_t sr = n / w.block_n;
          for (int64_t k = 0; k < w.k; ++k) {
            const float v = vt::F8E4M3ToF32(src[n * w.k + k]) *
                            sc[sr * scale_cols + k / w.block_k];
            dst[n * w.k + k] = vt::F32ToBF16(v);
          }
        }
      });
  return out;
}

// One linear projection out[T, N] = x[T, K] @ W[N, K]^T (MatmulBT; the loader
// stores every projection in the raw [N, K] on-disk orientation). The bf16
// arm (the router gate) and the fp8-block arm (everything else) resolve here.
DBuf LinearBTRaw(Dev d, const Tensor& x, const OwnedTensor& wt_raw, int64_t t,
                 DType out_dtype = DType::kBF16) {
  prof::Scope prof("linear_gemm");
  Tensor wt = ResidentWeight(d, wt_raw);
  DBuf out(d, out_dtype, {t, wt.shape[0]});
  vt::MatmulBT(d.q, out.t(), x, wt);
  return out;
}

DBuf LinearBT(Dev d, const Tensor& x, const Kolibri1Projection& w,
              int64_t t, DType out_dtype = DType::kBF16) {
  if (w.IsBf16()) return LinearBTRaw(d, x, w.bf16, t, out_dtype);
  DBuf wt = DequantFp8Block(d, w.fp8_block);
  prof::Scope prof("linear_gemm");
  DBuf out(d, out_dtype, {t, w.fp8_block.n});
  vt::MatmulBT(d.q, out.t(), x, wt.t());
  return out;
}

// SwiGLU expert MLP: silu(x @ gate^T) * (x @ up^T), then down. Routed experts
// and the shared expert share the shape.
DBuf ExpertMlp(Dev d, const Kolibri1ExpertWeights& e, const Tensor& x,
               int64_t t, int64_t inter) {
  DBuf g = LinearBT(d, x, e.gate_proj, t);  // [t, I]
  DBuf u = LinearBT(d, x, e.up_proj, t);    // [t, I]
  DBuf a(d, DType::kBF16, {t, inter});
  vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
  return LinearBT(d, a.t(), e.down_proj, t);  // [t, H]
}

// ── Router: sigmoid-logit-add (kolibri1.py:126-142) ─────────────────────────
// Selection on logits + bias, weights = sigmoid of the UNBIASED logits, NO
// renormalisation (norm_topk_prob=false). Router logits are f32 (the gate
// linear's out_dtype=torch.float32 upstream, :171-177). Ties break to the
// LOWER expert index, matching torch.topk's stable CPU order for the exact
// ties a test constructs deliberately.
struct HostRouting {
  std::vector<int32_t> ids;      // [T, top_k]
  std::vector<float> weights;    // [T, top_k]
};

HostRouting SigmoidLogitAddRouting(const std::vector<float>& logits,
                                   const std::vector<float>& bias, int64_t t,
                                   int64_t num_experts, int64_t top_k) {
  HostRouting r;
  r.ids.resize(static_cast<size_t>(t * top_k));
  r.weights.resize(static_cast<size_t>(t * top_k));
  for (int64_t i = 0; i < t; ++i) {
    const float* row = &logits[static_cast<size_t>(i * num_experts)];
    // Partial selection: top_k passes of argmax over the biased scores.
    std::vector<char> taken(static_cast<size_t>(num_experts), 0);
    for (int64_t kk = 0; kk < top_k; ++kk) {
      int64_t best = -1;
      float best_score = 0.0f;
      for (int64_t e = 0; e < num_experts; ++e) {
        if (taken[static_cast<size_t>(e)]) continue;
        const float score = row[e] + bias[static_cast<size_t>(e)];
        if (best < 0 || score > best_score) {
          best = e;
          best_score = score;
        }
      }
      taken[static_cast<size_t>(best)] = 1;
      // The WEIGHT reads the unbiased logit (docstring kolibri1.py:134-136).
      r.weights[static_cast<size_t>(i * top_k + kk)] =
          1.0f / (1.0f + std::exp(-row[best]));
      r.ids[static_cast<size_t>(i * top_k + kk)] = static_cast<int32_t>(best);
    }
    // norm_topk_prob=false: the renormalisation branch (:140-141) is skipped.
  }
  return r;
}

// ── Attention block (kolibri1.py:39-123) ────────────────────────────────────
DBuf AttentionBlock(Dev d, const Kolibri1AttnWeights& w, const Kolibri1Params& p,
                    bool is_sliding, const Tensor& dhn, const Tensor& positions,
                    const dense_attn::StepInputs& si, const PagedKvCache& kv,
                    int64_t t) {
  const int64_t hq = p.num_attention_heads;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t dh = p.head_dim;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dh)));

  // q/k/v projections (separate linears upstream, :64-79).
  DBuf q = LinearBT(d, dhn, w.q_proj, t);  // [T, Hq*Dh]
  DBuf k = LinearBT(d, dhn, w.k_proj, t);  // [T, Hkv*Dh]
  DBuf v = LinearBT(d, dhn, w.v_proj, t);  // [T, Hkv*Dh]

  // Per-head qk-norm BEFORE RoPE (:107-118): RMSNorm(head_dim) per head.
  // Reshape [T, H*Dh] -> [T*H, Dh] and run the shared RmsNorm with the
  // per-head gamma — every head shares one gamma vector.
  Tensor q3 = Reshape(q.t(), {t, hq, dh});
  Tensor k3 = Reshape(k.t(), {t, hkv, dh});
  Tensor v3 = Reshape(v.t(), {t, hkv, dh});
  {
    Tensor qh = Reshape(q.t(), {t * hq, dh});
    Tensor kh = Reshape(k.t(), {t * hkv, dh});
    Tensor qw = ResidentWeight(d, w.q_norm);
    Tensor kw = ResidentWeight(d, w.k_norm);
    DBuf qn(d, DType::kBF16, {t * hq, dh});
    DBuf kn(d, DType::kBF16, {t * hkv, dh});
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
  return LinearBT(d, o_in, w.o_proj, t);  // [T, H]
}

// ── MoE block (kolibri1.py:126-207) ─────────────────────────────────────────
DBuf MoeBlock(Dev d, const Kolibri1MoeWeights& w, const Kolibri1Params& p,
              const Tensor& dhn, int64_t t) {
  const int64_t h = p.hidden_size;
  const int64_t e = p.num_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t inter = p.moe_intermediate_size;

  // Router logits, f32 (:171-177).
  DBuf dlog = LinearBTRaw(d, dhn, w.router_gate, t, DType::kF32);
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

  HostRouting route;
  {
    prof::Scope prof("moe_topk");
    route = SigmoidLogitAddRouting(logits, bias, t, e, top_k);
  }

  DBuf dtw(d, DType::kF32, {t, top_k},
           const_cast<float*>(route.weights.data()));
  DBuf dtid(d, DType::kI32, {t, top_k},
            const_cast<int32_t*>(route.ids.data()));
  // Shared expert: UNGATED, always added (:146-188).
  DBuf shared = ExpertMlp(d, w.shared_experts, dhn, t,
                          p.shared_expert_intermediate_size);

  // Routed experts: the CPU-reference per-expert gather/scatter loop (the
  // mimo-v2 MoeBlock pattern).
  prof::Scope prof_glue("moe_glue");
  DBuf expert_out(d, DType::kBF16, {t, top_k, h});
  expert_out.Zero(d);
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
    DBuf o = ExpertMlp(d, w.experts[static_cast<size_t>(ex)], gathered.t(), ne,
                       inter);
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

  // Weighted combine + the always-added shared term. routed_scale 1.0: the
  // router weights ARE the mixture weights (no renormalisation, no scaling).
  DBuf out(d, DType::kBF16, {t, h});
  Tensor shared_t = shared.t();
  vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared_t,
                 /*routed_scale=*/1.0f);
  return out;
}

dense_attn::StepInputs BuildStepInputs(Dev d,
                                       const v1::CommonAttentionMetadata& meta,
                                       const std::vector<int32_t>& positions) {
  const int64_t t = static_cast<int64_t>(positions.size());
  DBuf d_positions(d, DType::kI32, {t}, positions.data());
  DBuf d_slot_mapping(d, DType::kI64,
                      {static_cast<int64_t>(meta.slot_mapping.size())},
                      meta.slot_mapping.data());
  DBuf d_block_table(d, DType::kI32,
                     {meta.num_reqs, meta.block_table_num_cols},
                     const_cast<int32_t*>(meta.block_table_tensor.data()));
  DBuf d_seq_lens(d, DType::kI32, {meta.num_reqs},
                  const_cast<int32_t*>(meta.seq_lens.data()));
  DBuf d_query_start_loc(d, DType::kI32, {meta.num_reqs + 1},
                         const_cast<int32_t*>(meta.query_start_loc.data()));
  dense_attn::StepInputs si;
  si.positions = std::move(d_positions);
  si.slot_mapping = std::move(d_slot_mapping);
  si.block_table = std::move(d_block_table);
  si.seq_lens = std::move(d_seq_lens);
  si.query_start_loc = std::move(d_query_start_loc);
  return si;
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

    dense_attn::StepInputs si = BuildStepInputs(d, attn_meta, positions);

    // input_layernorm + residual (the vLLM fused add-norm contract).
    DBuf dhn(d, DType::kBF16, {t, h});
    Tensor w_in = ResidentWeight(d, lw.input_layernorm, {h});
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
    DBuf attn = AttentionBlock(d, lw.attn, p, lw.is_sliding, dhn.t(),
                               si.positions.t(), si, *kv_ptr, t);
    DBuf attn_n(d, DType::kBF16, {t, h});
    Tensor w_pa = ResidentWeight(d, lw.post_attn_norm, {h});
    {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, vt::RmsNormArgs{eps, false});
    }

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

    // MoE on EVERY layer -> post_ffn_norm (no residual).
    DBuf moe = MoeBlock(d, lw.moe, p, dh2.t(), t);
    DBuf moe_n(d, DType::kBF16, {t, h});
    Tensor w_pf = ResidentWeight(d, lw.post_ffn_norm, {h});
    {
      prof::Scope prof("norms");
      vt::RmsNorm(d.q, moe_n.t(), moe.t(), w_pf, vt::RmsNormArgs{eps, false});
    }

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
    vt::MatmulBT(d.q, logits.t(), src, lm);
  }

  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = vocab;
  fl.device_tensor = logits.t();
  fl.device_storage = logits.ReleaseShared();
  if (prof::On()) {
    prof::Acc()["total_forward"] += prof::Now() - prof_t0;
    static int64_t calls = 0;
    if (++calls % 10 == 0) prof::Report(calls);
  }
  return fl;
}

}  // namespace vllm
