// OLMo-2 dense (`Olmo2ForCausalLM`, OLMo-2-0425-1B) forward — task W3. A plain
// dense GQA/MHA decoder COMPOSED from public vt:: ops, with exactly two OLMo-2
// wiring facts over the Qwen3/GLM-4 dense substrate, BOTH zero-kernel:
//   1. PURE POST-NORM placement (olmo2.py:261-277): attention and MLP run on the
//      RAW residual (NO input/pre norm); a STANDALONE vt::RmsNorm hits each
//      sublayer OUTPUT; a PLAIN vt::Add re-joins the residual. Distinct from the
//      pre-norm (Qwen3) and sandwich (GLM-4/Gemma) layouts.
//   2. FULL-WIDTH QK-norm (olmo2.py:113-117,160-172): standalone vt::RmsNorm over
//      the WHOLE [T,q_size]/[T,kv_size] q/k vector (all heads folded) BEFORE a
//      standard NeoX RoPE — NOT the per-head head_dim norm the fused recipe uses.
//
// Grounding: vllm/model_executor/models/olmo2.py @ e24d1b24 — Olmo2Attention
// (:67-185), Olmo2MLP (:188-230), Olmo2DecoderLayer.forward (:261-277), Olmo2Model
// (:280-343), untied lm_head (:377-382). See .agents/specs/sweep-olmo2.md §0.1-0.3.
//
// Numeric contract (mirrors GLM-4-9B/Qwen3-dense's token-exact bf16 path): the
// residual stream is bf16; the qkv GEMM, full-width q/k RMSNorm (variance in f32,
// rounded to bf16), rope (bf16 cos/sin cache), paged FA2 attention, o_proj and the
// whole MLP flow bf16; the standalone post-norms compute variance in f32 and round
// to bf16; the plain residual adds compute in f32 and round to bf16 (vt::Add).
// Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/models/olmo2.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // shared device glue (Dev/DBuf/...)
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool
#include "vllm/model_executor/models/qwen3_5_common.h"     // HostLogits
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kAttnQkNormRopeFullWidth (D3 full-width qk-norm fold)

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// Reuse the shared dense device glue VERBATIM: Dev/DBuf/ResidentWeight/KvSlice/
// StepInputs/BuildStepInputs/MakeRopeArgs/RopeCacheEnabled/Reshape/Pool. We
// deliberately do NOT call dense_attn::AttnBlock (it hard-codes PRE-norm placement
// and per-head qk-norm — two OLMo-2 violations); the OLMo-2 self-attention block
// below is written fresh, reusing only the glue (OPT/GLM precedent, spike D2).
using namespace dense_attn;

// OLMo-2 dense SwiGLU MLP (olmo2.py::Olmo2MLP): merged gate_up MatmulBT ->
// SiluAndMul -> down MatmulBT. `h` is the RAW residual hidden [T,H] bf16 (NO pre-FF
// norm — post-norm placement). bf16-only.
DBuf Olmo2MlpBlock(Dev d, const Olmo2MlpWeights& w, const HfConfig& cfg,
                   const Tensor& h, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — folds OLMo-2 onto the qwen3.cpp MlpBlock exemplar so it
  // inherits the nvfp4 GateUpFusedMarlinD arm for free once a quantized
  // checkpoint ships. (Tier-A1 fold, arch-fusion-fold-plan-2026-07-30.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, h);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One OLMo-2 self-attention block (olmo2.py::Olmo2Attention.forward :174-185). `h`
// is the RAW residual hidden [T,H] bf16 (NO input norm — post-norm placement);
// returns the o_proj output [T,H] bf16.
//
// OLMo-2 deltas vs dense_attn::AttnBlock: (a) NO input norm (the caller applies the
// post-norm); (b) FULL-WIDTH q/k RMSNorm over [T,q_size]/[T,kv_size] (not per-head);
// (c) NO biases; (d) full NeoX RoPE (rotary_dim == head_dim).
DBuf Olmo2AttnBlock(Dev d, const Olmo2AttnWeights& w, const HfConfig& cfg,
                    const Tensor& h, const StepInputs& si,
                    const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                    int64_t T, bool use_yarn, const Tensor* yarn_cache,
                    std::optional<int64_t> sliding_window) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "olmo2: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "olmo2: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (bias=False): one MatmulBT over the [q|k|v] owner,
  // then split into contiguous q/k/v shards (olmo2.py:179-180).
  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), h, wqkv);
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // FULL-WIDTH q/k RMSNorm (olmo2.py:166-167 _apply_qk_norm): one RMSNorm over the
  // WHOLE q vector [T,q_size] and the WHOLE k vector [T,kv_size] — all heads folded
  // into one normalization statistic. This is applied to the [T,qdim]/[T,kdim]
  // views BEFORE reshaping to heads, so it differs fundamentally from the per-head
  // head_dim norm (Qwen3). Weight dtype == q dtype (bf16 · bf16, variance in f32).
  Tensor wqn = ResidentWeight(d, w.q_norm, {qdim});
  Tensor wkn = ResidentWeight(d, w.k_norm, {kdim});
  Tensor q2 = q.t();  // [T, qdim]
  Tensor k2 = k.t();  // [T, kdim]
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});

  // D3 FOLD (arch-fusion-fold-plan-2026-07-30, Tier-D3): the FULL-WIDTH q/k
  // RMSNorm + standard NeoX RoPE preamble folds onto the SHARED fusion catalog
  // recipe vt::kAttnQkNormRopeFullWidth — the `norm_full_width` shape-param
  // variant of Qwen3-dense's kAttnQkNormRope (recipes.h). Its Tier-0 composite
  // dispatches EXACTLY the standalone RmsNorm(q,[qdim]) + RmsNorm(k,[kdim]) +
  // RopeFromCache sequence OLMo-2 hand-called before this fold, so adoption is
  // byte-identical (SACRED-gated). The fold requires RoPE FROM A CACHE (the
  // recipe's third step is RopeFromCache): the YaRN full-attention path and the
  // default-ON bf16 cos/sin cache both qualify; the in-place RopeNeox fallback
  // (VT_QWEN3_ROPE_CACHE=0) does NOT — routing it through the recipe would
  // silently swap one RoPE impl for another (mirrors the qwen3.cpp guard), so it
  // keeps the standalone sequence. Both norms are full-width in EVERY branch.
  const bool rope_from_cache = rot > 0 && (use_yarn || RopeCacheEnabled());
  const bool fused_preamble = FusedChainAdoptEnabled() && rope_from_cache;
  if (fused_preamble) {
    // yarn (OLMo-3 full-attn layers) rope against the precomputed YaRN cache
    // indexed by the REAL positions; the default path (OLMo-2, OLMo-3 sliding)
    // ropes against the per-step bf16 cos/sin cache indexed by the identity row
    // (the position map is baked into the cache rows).
    Tensor rope_cache = use_yarn ? *yarn_cache : si.cos_sin_bf16.t();
    Tensor rope_idx = use_yarn ? si.positions.t() : si.rope_row_idx.t();
    vt::RopeArgs ra;
    if (use_yarn) {
      ra.rotary_dim = rot;
      ra.is_neox_style = true;  // base unused (YaRN inv_freq baked into the cache)
    } else {
      ra = MakeRopeArgs(cfg);
    }
    vt::FusedBinding b;
    b.op[0] = &q2;   // q [T,qdim]   (full-width norm in place)
    b.op[1] = &wqn;  // q_norm [qdim]
    b.op[2] = &k2;   // k [T,kdim]   (full-width norm in place)
    b.op[3] = &wkn;  // k_norm [kdim]
    b.op[4] = &q3;   // q3 [T,Hq,Dh] (per-head rope view, aliases q2)
    b.op[5] = &k3;   // k3 [T,Hkv,Dh] (per-head rope view, aliases k2)
    b.op[6] = &rope_cache;
    b.op[7] = &rope_idx;
    b.n = 8;
    vt::FusedParams p;
    p.eps = eps;
    p.rope = ra;
    vt::FusedChain(d.q, vt::kAttnQkNormRopeFullWidth, b, p);
  } else {
    // Standalone fallback: full-width q/k RMSNorm then in-place NeoX RoPE (base
    // rope_theta, per-element fp64 cos/sin). Byte-identical to the pre-fold code.
    vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
    if (rot > 0) vt::RopeNeox(d.q, q3, k3, si.positions.t(), MakeRopeArgs(cfg));
  }

  // Write rope'd K + V into the paged cache (bf16 == cache dtype on the production
  // path, no cast), then causal GQA/MHA paged attention.
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
  Tensor kw = k3;
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != DType::kBF16) {
    vt::CastF32(d.q, kcast.t(), k3);
    vt::CastF32(d.q, vcast.t(), v3);
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, DType::kBF16, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // OLMo-3 SLIDING-attention layer: mask to the last `sliding_window` keys (FA
  // window convention (W-1, 0)). Inert for contexts < sliding_window (the gate
  // battery is short), so the SACRED gate is unaffected; correct for long contexts.
  // ENG-ATTENTION-WINDOW (#2388). No kill switch here either; see W3.
  if (sliding_window.has_value() && *sliding_window > 0)
    pa.window_size = ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, sliding_window,
        v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/DisableSlidingWindowActive());
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, NO bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// One OLMo-2 decoder layer (olmo2.py::Olmo2DecoderLayer.forward :261-277), the
// PURE POST-NORM pattern:
//   residual = h
//   h = self_attn(h)                     # attention on the RAW residual (NO norm)
//   h = post_attention_layernorm(h)      # STANDALONE norm on the attn output
//   h = h + residual                     # PLAIN residual add
//   residual = h
//   h = mlp(h)                           # MLP on the RAW residual (NO pre-FF norm)
//   h = post_feedforward_layernorm(h)    # STANDALONE norm on the mlp output
//   h = residual + h                     # PLAIN residual add
// `hidden` (bf16 [T,H]) is BOTH the residual and the running stream (there is no
// separate fused residual buffer — the plain add keeps the residual in `hidden`).
void RunLayer(Dev d, const Olmo2LayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T,
              bool use_yarn, const Tensor* yarn_cache,
              std::optional<int64_t> sliding_window) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // Attention sub-block on the RAW residual (NO input norm).
  DBuf attn = Olmo2AttnBlock(d, layer.attn, cfg, hidden.t(), si, meta, kv, T,
                             use_yarn, yarn_cache, sliding_window);
  // post_attention_layernorm (STANDALONE): attn_n = norm(attn).
  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, vt::RmsNormArgs{eps, false});
  // Plain residual add: hidden = hidden + attn_n (commutative; out aliases a).
  vt::Add(d.q, hidden.t(), hidden.t(), attn_n.t());

  // MLP sub-block on the RAW residual (NO pre-FF norm).
  DBuf mlp = Olmo2MlpBlock(d, layer.mlp, cfg, hidden.t(), T);
  // post_feedforward_layernorm (STANDALONE): mlp_n = norm(mlp).
  Tensor w_pf = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
  DBuf mlp_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, mlp_n.t(), mlp.t(), w_pf, vt::RmsNormArgs{eps, false});
  // Plain residual add: hidden = hidden + mlp_n.
  vt::Add(d.q, hidden.t(), hidden.t(), mlp_n.t());
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// embed -> N layers -> final standalone RMSNorm -> lm_head. Returns [n_out, vocab] f32.
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv, const Olmo2Weights& weights,
                 const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const int rot = static_cast<int>(config.rotary_dim);
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "olmo2: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "olmo2: one PagedKvCache per layer required");

  // Embed: hidden[T,H] bf16 = embed_tokens[token_ids].
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "olmo2 embed");
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);
  // OLMo-2 ropes off the bf16 cos/sin cache on the default path (RopeCacheEnabled);
  // guarantee the bf16 cache is populated (BuildStepInputs only casts it when the
  // flag is on, which is the default). Idempotent when already built.
  if (rot > 0 && RopeCacheEnabled())
    vt::CastBf16(d.q, si.cos_sin_bf16.t(), si.cos_sin.t());

  // OLMo-3 (`Olmo3ForCausalLM`): per-layer interleaved sliding window + YaRN rope on
  // FULL-attention layers ONLY (olmo2.py:121-149). Routed by config.layer_types.
  // OLMo-2 has EMPTY layer_types -> has_layer_types is false -> every layer takes the
  // plain-rope / no-window path (byte-identical to before). The YaRN cache is
  // resident-uploaded once and reused; EMPTY for OLMo-2.
  const bool has_layer_types = !config.layer_types.empty();
  const bool has_yarn = !weights.rope_cos_sin_yarn.Empty();
  Tensor yarn_cache;
  if (has_yarn) yarn_cache = ResidentWeight(d, weights.rope_cos_sin_yarn);
  const std::optional<int64_t> window_len =
      config.sliding_window.has_value() ? config.sliding_window : std::nullopt;

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    bool use_yarn = false;
    std::optional<int64_t> sliding_window;
    if (has_layer_types) {
      const std::string& lt = config.layer_types[static_cast<size_t>(l)];
      if (lt == "sliding_attention") {
        sliding_window = window_len;  // plain rope, finite window
      } else {
        use_yarn = has_yarn;  // full_attention: YaRN rope, no window
      }
    }
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T, use_yarn,
             use_yarn ? &yarn_cache : nullptr, sliding_window);
  }

  // Final standalone RMSNorm over the residual stream (olmo2.py:342), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false});

  // lm_head. OLMo-2-1B is UNTIED: the loaded Matmul-B [H, vocab] via vt::Matmul.
  // Tied fallback aliases embed_tokens via MatmulBT (for a tied OLMo-2 checkpoint).
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
  Tensor src = dnorm.t();
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  (void)rot;
  return logits;
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Olmo2Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Olmo2Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config,
                  logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Olmo2Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Olmo2Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config,
                  logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
