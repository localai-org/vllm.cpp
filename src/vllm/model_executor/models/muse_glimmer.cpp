// Muse Glimmer TEXT tower forward (W1). Composed from the public vt:: ops and the
// shared dense-attention device glue (dense_attn_block.h), structurally mirroring
// `gemma2.cpp` — Muse's decoder layer is the SAME sandwich-norm shape — with six
// deliberate deltas, each of which is a place a wrong port stays plausible and
// coherent instead of erroring (spec §9):
//
//   1. EMBED. Gemma multiplies the embedding by sqrt(hidden); Muse instead applies
//      a WEIGHTLESS RMSNorm (`embed_norm`, muse_glimmer.py:1286). Different
//      operation, same slot. Weightless is realized as vt::RmsNorm against a ones
//      weight (spec §9: no weightless variant exists in the op set).
//   2. SPLIT EPS. The two PRE norms take `rms_norm_eps`, the two POST norms
//      `post_norm_eps` (:1236-1247); gemma2 threads ONE eps everywhere. The POST
//      norms stay STANDALONE — they are sublayer-output norms with no residual add,
//      so folding them onto kFusedAddRmsNorm would be an incorrect fold (the hazard
//      gemma2.cpp documents). The FINAL norm (:1296) has NO `+1` offset, unlike all
//      four sandwich norms — so it runs the gemma=false recipe.
//   3. ATTENTION SCALE. Plain `head_dim ** -0.5` (:1112), NOT gemma2's
//      `query_pre_attn_scalar ** -0.5`. The query pre-scale is a SEPARATE multiply
//      on q after QK-norm (:1192), never folded into the softmax scale.
//   4. iRoPE. `no_rope_layers[l] == 1` => RoPE AND sliding window; `== 0` => NoPE
//      AND full attention (:1114-1116, :1167-1168). RoPE and the window travel
//      TOGETHER — gemma2's sliding split is independent of RoPE.
//   5. OUTPUT GATE. `attn * sigmoid(output_gate_proj(x))` where x is the NORMED
//      LAYER INPUT, not the attention output (:1203-1206) — via the shared
//      vt::SigmoidGateBf16 seam Qwen3.5 already uses.
//   6. SwiGLU (not GeGLU), UNTIED lm_head, then `output_multiplier` BEFORE the
//      final logit soft-cap (:1615-1621).
//
// ─── OFF-PIN HONESTY ─────────────────────────────────────────────────────────
// Every `file:line` above is vllm#51655 head `075d645af`, an OPEN and CI-red
// upstream PR — NOT the parity pin `555967922`, which contains no muse_glimmer at
// all. See porting-inventory §9 deviation 16 and specs/muse-glimmer.md §0.
//
// Consequently: the pinned oracle CANNOT load this model, so there is NO golden and
// NO speed denominator. W1 establishes STRUCTURAL and PER-MECHANISM correctness
// (tests/vllm/models/test_muse_glimmer_text.cpp checks the whole text forward
// against an independently written fp32 reference transcribed from
// muse_glimmer.py, plus scale-invariance and A/B properties for each mechanism).
// It does NOT establish token-exact e2e correctness against the HF reference —
// that is W2 — and it establishes NOTHING about speed.
//
// Numeric contract: bf16 per-op, matching vLLM's stores (dense_attn_block.h). The
// output gate's sigmoid argument stays f32 (vt::SigmoidGateBf16's contract: the
// sigmoid input must not be rounded), the same convention qwen3_5.cpp uses.
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/models/muse_glimmer.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"            // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNorm / kFusedAddRmsNormStd

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev/DBuf/ResidentWeight/KvSlice/StepInputs/Reshape

constexpr const char* kNoWeights =
    "MuseGlimmer forward: the text tower's weights are not materialized on this "
    "MuseGlimmerWeights (text_loaded == false). This is the W0 params-only "
    "accounting form; load a checkpoint through "
    "LoadMuseGlimmerForConditionalGenerationWeights first. See "
    ".agents/specs/muse-glimmer.md §3.";

// Per-layer routing derived from the resolved params. Everything here is READ from
// the config — nothing is inferred — because each field is a place where a wrong
// value produces coherent-but-wrong text rather than an error.
struct MuseGlimmerLayout {
  float pre_eps = 1e-6f;        // input_layernorm + pre_feedforward_layernorm
  float post_eps = 1e-6f;       // post_attention + post_feedforward
  float norm_eps = 1e-6f;       // embed_norm, qk_norm, final norm (rms_norm_eps)
  float attn_scale = 1.0f;      // head_dim ** -0.5  (:1112)
  double scale_query_by = 1.0;  // post-QK-norm query pre-scale (:1192)
  double rope_theta = 500000.0;
  int64_t sliding_window = 0;
  bool use_qk_norm = true;
  bool use_output_gate = true;
  float output_multiplier = 1.0f;
  float final_logit_softcap = 0.0f;
  const std::vector<int64_t>* no_rope_layers = nullptr;

  // muse_glimmer.py:1114-1116 — `no_rope_layers[l] == 1` means this layer USES
  // RoPE; `== 0` is a NoPE layer. The window rides the SAME flag (:1167-1168).
  bool UsesRope(int64_t l) const {
    return no_rope_layers != nullptr && l >= 0 &&
           static_cast<size_t>(l) < no_rope_layers->size() &&
           (*no_rope_layers)[static_cast<size_t>(l)] == 1;
  }
};

MuseGlimmerLayout MakeLayout(const MuseGlimmerTextParams& t) {
  MuseGlimmerLayout g;
  g.pre_eps = t.rms_norm_eps;
  g.post_eps = t.post_norm_eps;
  g.norm_eps = t.rms_norm_eps;
  g.attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(t.head_dim)));
  g.scale_query_by = t.scale_query_by;
  g.rope_theta = t.rope_theta;
  g.sliding_window = t.sliding_window;
  g.use_qk_norm = t.use_qk_norm;
  g.use_output_gate = t.use_attn_output_gate;
  g.output_multiplier = static_cast<float>(t.output_multiplier);
  g.final_logit_softcap = static_cast<float>(t.final_logit_softcapping);
  g.no_rope_layers = &t.no_rope_layers;
  return g;
}

// A bf16 ones vector, the weight for the two WEIGHTLESS RMSNorms (`embed_norm`
// :1286 and the per-head `qk_norm` :1121). bf16 1.0 is exact, so `x/rms * 1` is
// bit-identical to the weightless `_norm(x.float()).type_as(x)` upstream computes.
DBuf OnesBf16(Dev d, int64_t n) {
  std::vector<uint16_t> host(static_cast<size_t>(n), vt::F32ToBF16(1.0f));
  return DBuf(d, DType::kBF16, {n}, host.data());
}

// One Muse Glimmer self-attention block (muse_glimmer.py::MuseGlimmerAttention
// .forward, :1177-1215). `dhn` is the input-normed hidden [T,H] bf16 — it is BOTH
// the qkv input AND the output gate's input (:1204). Returns o_proj out [T,H] bf16.
DBuf MuseGlimmerAttnBlock(Dev d, const MuseGlimmerAttnWeights& w,
                          const MuseGlimmerTextParams& t, const MuseGlimmerLayout& g,
                          const Tensor& dhn, const Tensor& ones_head,
                          const StepInputs& si, const CommonAttentionMetadata& meta,
                          const PagedKvCache& kv, int64_t T, bool use_rope) {
  const int64_t H = t.hidden_size;
  const int64_t Hq = t.num_attention_heads;
  const int64_t Hkv = t.num_key_value_heads;
  const int64_t Dh = t.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "muse_glimmer: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "muse_glimmer: KV cache head dims mismatch config");

  // Merged QKVParallelLinear, no bias (:1126-1135). Same D1 merged-owner + QkvSplit
  // fold every bf16 dense model uses; VT_QWEN3_QKV_MERGE=0 gives the byte-identical
  // 3-shard form.
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  {
    Tensor wqkv = ResidentWeight(d, w.qkv_proj);
    if (MergedQkvEnabled()) {
      DBuf qkv(d, adt, {T, qdim + 2 * kdim});
      vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
      vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
    } else {
      Tensor wq = wqkv.Slice(0, 0, qdim);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      vt::MatmulBT(d.q, q.t(), dhn, wq);
      vt::MatmulBT(d.q, k.t(), dhn, wk);
      vt::MatmulBT(d.q, v.t(), dhn, wv);
    }
  }

  // WEIGHTLESS QK-norm over head_dim, fp32 accumulation, applied BEFORE RoPE
  // (:1189-1196), then the query pre-scale on q ONLY (:1192). Upstream rounds the
  // norm back to the activation dtype before the scalar multiply
  // (MuseGlimmerRMSNorm returns `out.type_as(hidden_states)`), which is exactly
  // what a bf16-out RmsNorm followed by a bf16 MulScalar does.
  //
  // The softmax scale stays head_dim**-0.5 and is NOT folded together with the
  // pre-scale: they are two different multiplies on two different tensors, and
  // conflating them is the ~11.3x query blow-up the config parse already guards.
  if (g.use_qk_norm) {
    // Normed IN PLACE over the [T*H,Dh] 2-D row view, the same aliasing the shared
    // kAttnQkNormRope preamble relies on (RmsNorm reads a whole row before storing).
    Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
    const vt::RmsNormArgs weightless{g.norm_eps, /*gemma=*/false};
    vt::RmsNorm(d.q, q2, q2, ones_head, weightless);
    vt::RmsNorm(d.q, k2, k2, ones_head, weightless);
    vt::MulScalar(d.q, q.t(), q.t(), g.scale_query_by);
  }

  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  // iRoPE (:1163-1174): NeoX RoPE over the full head_dim on RoPE layers, and
  // NOTHING at all on NoPE layers — `self.rotary_emb` is None there.
  if (use_rope) {
    vt::RopeArgs ra;
    ra.base = static_cast<float>(g.rope_theta);
    ra.rotary_dim = static_cast<int>(Dh);
    vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);
  }

  // Write (rope'd) K + V into the paged cache, casting to the cache dtype.
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
  Tensor kw = k3;
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    }
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  // Paged GQA attention. scale = head_dim**-0.5; NO attention logit soft-cap
  // (upstream passes logits_soft_cap=None, :1170). The sliding window rides the
  // RoPE flag: RoPE layers are windowed, NoPE layers are full attention (:1167).
  DBuf attn(d, adt, {T, Hq, Dh});
  vt::PagedAttentionArgs pa{g.attn_scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // ENG-ATTENTION-WINDOW (#2388). `use_rope` is this model's own routing
  // predicate and stays: a NoPE layer takes no window at all, which is a
  // different question from how wide the window is.
  if (use_rope && g.sliding_window > 0)
    pa.window_size = ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, g.sliding_window,
        v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/DisableSlidingWindowActive());
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // Attention OUTPUT GATE (:1202-1206): attn * sigmoid(output_gate_proj(x)) where x
  // is the NORMED LAYER INPUT `dhn`, NOT the attention output. Gating on the wrong
  // tensor is the archetypal plausible-but-wrong port here. Routed through the
  // shared vt::SigmoidGateBf16 seam (the Qwen3.5 gated-attention op); the gate GEMM
  // lands in f32 because that op's contract keeps the sigmoid input unrounded.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  DBuf gated(d, adt, {T, qdim});
  if (g.use_output_gate) {
    VT_CHECK(!w.output_gate_proj.Empty(),
             "muse_glimmer: use_attn_output_gate is on but output_gate_proj is absent");
    Tensor wg = ResidentWeight(d, w.output_gate_proj);
    DBuf gate(d, DType::kF32, {T, qdim});
    vt::MatmulBT(d.q, gate.t(), dhn, wg);
    vt::SigmoidGateBf16(d.q, gated.t(), o_in, gate.t());
    o_in = gated.t();
  }

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// SwiGLU MLP (muse_glimmer.py::MuseGlimmerMLP, :1046-1079): merged gate_up ->
// SiluAndMul -> down, via the SHARED bf16 gate-up MLP seam. gemma2 uses the
// ...GeluMethod sibling; Muse asserts `silu` at config parse and uses this one.
DBuf MuseGlimmerMlpBlock(Dev d, const MuseGlimmerMlpWeights& w,
                         const MuseGlimmerTextParams& t, const Tensor& dh2,
                         int64_t T) {
  DBuf act =
      layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, t.intermediate_size).Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);
  DBuf down(d, DType::kBF16, {T, t.hidden_size});
  vt::MatmulBT(d.q, down.t(), act.t(), wd);
  return down;
}

// One decoder layer (muse_glimmer.py::MuseGlimmerDecoderLayer.forward, :1249-1277).
// Upstream writes the residual add EXPLICITLY after each post-norm:
//   res = h;  h = in_norm(h);  h = attn(h);  h = post_attn_norm(h);  h = res + h
//   res = h;  h = pre_ff_norm(h); h = mlp(h); h = post_ff_norm(h);   h = res + h
// We carry the deferred-residual form gemma2.cpp uses, which is the SAME algebra:
// `hidden` holds the last post-norm output and `res` the running residual stream,
// so `res += hidden` inside the next norm reproduces upstream's add exactly.
void RunLayer(Dev d, const MuseGlimmerLayerWeights& layer,
              const MuseGlimmerTextParams& t, const MuseGlimmerLayout& g, int64_t l,
              DBuf& hidden, DBuf& res, const Tensor& ones_head, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = t.hidden_size;
  const vt::RmsNormArgs pre{g.pre_eps, /*gemma=*/true};
  const vt::RmsNormArgs post{g.post_eps, /*gemma=*/true};

  // input_layernorm: fused residual-add + (1+w) RMSNorm at `rms_norm_eps` (:1236).
  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNorm,
                   g.pre_eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, pre, &res.t());

  DBuf attn = MuseGlimmerAttnBlock(d, layer.attn, t, g, dhn.t(), ones_head, si, meta,
                                   kv, T, g.UsesRope(l));

  // post_attention_layernorm: SANDWICH sublayer-output post-norm at `post_norm_eps`
  // (:1239) with NO residual add — NOT fusable onto kFusedAddRmsNorm (that recipe's
  // step0 IS a residual add this site does not have). STANDALONE by design.
  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, post);

  // pre_feedforward_layernorm: fused residual-add + (1+w) RMSNorm at
  // `rms_norm_eps` (:1242).
  Tensor w_pf = ResidentWeight(d, layer.pre_feedforward_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn_n.t(), w_pf, &res.t(), vt::kFusedAddRmsNorm,
                   g.pre_eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn_n.t(), w_pf, pre, &res.t());

  DBuf mlp = MuseGlimmerMlpBlock(d, layer.mlp, t, dh2.t(), T);

  // post_feedforward_layernorm: STANDALONE post-norm at `post_norm_eps` (:1245).
  Tensor w_pff = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
  hidden = DBuf(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, hidden.t(), mlp.t(), w_pff, post);
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// `inputs_embeds`, when non-null, is the W4 MULTIMODAL entry: the caller has
// already run `embed_input_ids` and masked-scattered the vision soft tokens, so the
// hidden stream starts from those [T, H] bf16 DEVICE rows (a BORROWED view —
// ENG-MM-INPUT-PIPELINE P1, #1358/#2300) and the embedding + weightless
// `embed_norm` are SKIPPED — exactly the branch upstream's model forward takes on
// `inputs_embeds is not None` (muse_glimmer.py:1311-1315). Null on every text step,
// which leaves the text path byte-identical: the pointer only gates which of two
// ways `hidden` is filled, and nothing after it reads the flag.
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const MuseGlimmerWeights& weights,
                 const std::vector<int32_t>& logits_indices,
                 const Tensor* inputs_embeds = nullptr) {
  const MuseGlimmerTextParams& t = weights.params.text;
  const int64_t T = inputs_embeds != nullptr
                        ? static_cast<int64_t>(positions.size())
                        : static_cast<int64_t>(token_ids.size());
  const int64_t H = t.hidden_size;
  const int64_t vocab = t.vocab_size;
  VT_CHECK(weights.text_loaded, kNoWeights);
  VT_CHECK(inputs_embeds == nullptr ||
               (inputs_embeds->data != nullptr && inputs_embeds->Numel() == T * H),
           "muse_glimmer mm forward: inputs_embeds must be [positions, hidden_size]");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "muse_glimmer: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(t.num_hidden_layers),
           "muse_glimmer: one PagedKvCache per layer required");
  VT_CHECK(weights.layers.size() == static_cast<size_t>(t.num_hidden_layers),
           "muse_glimmer: one layer weight set per layer required");

  const MuseGlimmerLayout g = MakeLayout(t);

  // The two WEIGHTLESS norms' ones weights, built ONCE per forward: `embed_norm`
  // over hidden_size and the per-head `qk_norm` over head_dim.
  DBuf ones_hidden = OnesBf16(d, H);
  DBuf ones_head = OnesBf16(d, t.head_dim);

  // Embed, then the WEIGHTLESS embed_norm (:1286, :1298-1299). This is the slot
  // where Gemma multiplies by sqrt(hidden); Muse normalizes instead. Because
  // RMSNorm is scale-invariant, the two are not a constant apart — swapping them is
  // a different function, not a different constant.
  DBuf hidden(d, DType::kBF16, {T, H});
  if (inputs_embeds != nullptr) {
    // The mm branch: the merged embeds ALREADY carry embed_norm on their text rows
    // and the un-normalized soft tokens on the placeholder rows. Re-applying
    // embed_norm here would re-normalize the vision features, which upstream never
    // does (:1312-1313 assigns inputs_embeds straight through). This is a D2D copy
    // of the caller's BORROWED view rather than an alias: `RunLayer` replaces
    // `hidden` on its last line today, so nothing writes the seed, but that
    // invariant is stated nowhere and the runner will hand this seam a window into
    // a buffer it reuses across steps (#2300).
    d.b.Copy(d.q, hidden.ptr(), inputs_embeds->data, hidden.bytes());
  } else {
    DBuf emb(d, DType::kBF16, {T, H});
    EmbedGather(d, emb, token_ids, weights.embed_tokens, vocab, H,
                "muse_glimmer embed");
    vt::RmsNorm(d.q, hidden.t(), emb.t(), ones_hidden.t(),
                vt::RmsNormArgs{g.norm_eps, /*gemma=*/false});
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  HfConfig step_cfg;
  step_cfg.num_attention_heads = t.num_attention_heads;
  step_cfg.num_key_value_heads = t.num_key_value_heads;
  step_cfg.head_dim = t.head_dim;
  step_cfg.rotary_dim = t.head_dim;
  step_cfg.rope_theta = t.rope_theta;
  StepInputs si = BuildStepInputs(d, positions, attn_meta, step_cfg);

  for (int64_t l = 0; l < t.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], t, g, l, hidden, res,
             ones_head.t(), si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final norm (:1296): residual-add + RMSNorm with the weight applied as `w`, NOT
  // `1+w` — MuseGlimmerRMSNorm's default weight_offset is 0 and only the four
  // sandwich norms pass weight_offset=1. Hence the gemma=FALSE recipe here.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd,
                   g.norm_eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn,
                vt::RmsNormArgs{g.norm_eps, /*gemma=*/false}, &res.t());

  // UNTIED lm_head (:1480). A tied checkpoint (none ships today) would carry no
  // `lm_head.weight`; fall back to the embedding table exactly as every tied model
  // does, so the shape stays resolvable rather than reading an empty tensor.
  const bool tied = t.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
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

  // compute_logits (:1615-1621): the output multiplier is applied BEFORE the final
  // tanh soft-cap. Both are monotone, so greedy argmax is invariant to either — they
  // are applied for faithfulness, and the ORDER matters for the logprobs a sampler
  // sees (a soft-cap applied first would clamp a differently-scaled logit).
  if (g.output_multiplier != 1.0f)
    vt::MulScalar(d.q, logits.t(), logits.t(), g.output_multiplier);
  if (g.final_logit_softcap > 0.0f)
    vt::SoftCap(d.q, logits.t(), logits.t(), g.final_logit_softcap);
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

std::vector<float> MuseGlimmerModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * weights.params.text.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits MuseGlimmerModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, weights.params.text.vocab_size);
}

std::vector<float> MuseGlimmerModel::ForwardMm(
    const vt::Tensor& inputs_embeds,
    const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // `token_ids` is unused on this branch (T comes from `positions`); pass an empty
  // vector so the mm seam never dereferences it.
  const std::vector<int32_t> no_tokens;
  DBuf dlogits = ForwardBody(d, no_tokens, positions, attn_meta, attn_kv, weights,
                             logits_indices, &inputs_embeds);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * weights.params.text.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

}  // namespace vllm
