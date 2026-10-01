// GLM-4-0414 dense (`Glm4ForCausalLM`, GLM-4-9B-0414) forward — task G2/W2, the
// first GLM-family model. A standard dense transformer COMPOSED from public vt::
// ops, with exactly two GLM deltas over the Qwen3-dense path (qwen3.cpp):
//   1. PARTIAL + INTERLEAVED RoPE (rotary_dim 64, is_neox_style=false): route the
//      EXISTING RopeFromCache primitive with is_neox_style=false. The kernel
//      (cuda_ops.cu:697-698 / cpu_ops.cpp:744-746) rotates only the leading
//      rotary_dim slice in adjacent-pair layout and passes the tail through — the
//      same interleaved path DeepSeek-V2's decoupled RoPE is already gated on.
//   2. SANDWICH NORMS (glm4.py:206,211): standalone vt::RmsNorm on the attn / mlp
//      sublayer OUTPUT before the residual add.
// Plus BIASED qkv (attention_bias=true) and an UNTIED lm_head. bf16-only.
//
// Grounding: vllm/model_executor/models/glm4.py @ e24d1b24 — Glm4Attention
// (:55-140), Glm4DecoderLayer.forward (:189-213), Glm4MLP=LlamaMLP, Glm4Model=
// LlamaModel, untied lm_head (:255-267). See
// .agents/specs/glm-dsa-latest-deepseek.md §0.4.3.
//
// Numeric contract (mirrors Qwen3-dense's token-exact bf16 path): the residual
// stream is bf16 (vLLM's fused_add_rms_norm residual); the qkv GEMM + bias, rope
// (bf16 cos/sin cache), paged FA2 attention, o_proj and the whole MLP flow bf16;
// the standalone sandwich RMSNorms compute variance in f32 and round to bf16.
// Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/glm4.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // shared device glue (Dev/DBuf/...)
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool
#include "vllm/model_executor/models/qwen3_5_common.h"     // HostLogits
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// Reuse the shared dense device glue VERBATIM: Dev/DBuf/ResidentWeight/KvSlice/
// StepInputs/BuildStepInputs/MakeRopeArgs/Reshape/Pool. We deliberately do NOT
// call dense_attn::AttnBlock (it asserts qkv_bias.Empty() + hardcodes NeoX rope +
// per-head qk-norm — three GLM violations); the GLM self-attention block below is
// written fresh, reusing only the glue (OPT precedent, spike D4).
using namespace dense_attn;

// GLM-4 dense SwiGLU MLP (glm4.py Glm4MLP=LlamaMLP): merged gate_up MatmulBT ->
// SiluAndMul -> down MatmulBT. `dh2` is the post-norm hidden [T,H] bf16. bf16-only.
DBuf Glm4MlpBlock(Dev d, const Glm4MlpWeights& w, const HfConfig& cfg,
                  const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — the seam's Apply IS {ResidentWeight; MatmulBT[2I,H];
  // SiluAndMul} with M = x.shape[0] == T. Mirrors upstream's single
  // MergedColumnParallelLinear + SiluAndMul (glm4.py:46 aliases llama.py:79
  // LlamaMLP verbatim), and picks up the nvfp4 GateUpFusedMarlinD arm for free
  // once a quantized checkpoint ships. (FUSION-DENSE-MIGRATE.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One GLM-4 self-attention block (glm4.py::Glm4Attention.forward). `dhn` is the
// input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
//
// GLM deltas vs dense_attn::AttnBlock: (a) BIASED merged qkv (MatmulBT + row-
// broadcast bias, like OPT's BiasedProj); (b) NO per-head q/k RMSNorm; (c) PARTIAL
// + INTERLEAVED RoPE (rotary_dim 64, is_neox_style=false) via the bf16 cos/sin
// cache; (d) o_proj has NO bias.
DBuf Glm4AttnBlock(Dev d, const Glm4AttnWeights& w, const HfConfig& cfg,
                   const Tensor& dhn, const StepInputs& si,
                   const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                   int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(!w.qkv_bias.Empty(),
           "glm4: attention_bias=true expected — qkv bias must be present");
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "glm4: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "glm4: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (bias=True): one MatmulBT over the [q|k|v] owner +
  // row-broadcast bias, then split into contiguous q/k/v shards (glm4.py:130-131).
  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
  Tensor bqkv = ResidentWeight(d, w.qkv_bias, {qdim + 2 * kdim});
  vt::Add(d.q, qkv.t(), qkv.t(), bqkv);  // rank-1 row-broadcast, in place
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // Partial + INTERLEAVED RoPE. NO per-head q/k norm (GLM has none): the raw q/k
  // rotate directly. RopeFromCache rotates only the leading rotary_dim slice in
  // adjacent-pair (is_neox_style=false) layout, passing dims [rotary_dim,Dh)
  // through — exactly vLLM get_rope(is_neox_style=False) with partial_rotary_factor.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  if (rot > 0) {
    vt::RopeArgs ra = MakeRopeArgs(cfg);
    ra.is_neox_style = false;  // GLM: adjacent-pair (GPT-J) rotation
    Tensor k3v = k3;
    vt::RopeFromCache(d.q, q3, &k3v, si.rope_row_idx.t(), si.cos_sin_bf16.t(), ra);
  }

  // Write rope'd K + V into the paged cache (bf16 == cache dtype, no cast on the
  // production path), then causal GQA paged attention.
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
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, NO bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// One GLM-4 decoder layer (glm4.py::Glm4DecoderLayer.forward :189-213), the
// Gemma2 SANDWICH pattern:
//   hidden, res = input_layernorm(hidden, res)          # fused add+RMSNorm
//   attn        = self_attn(hidden)
//   attn        = post_self_attn_layernorm(attn)         # STANDALONE norm
//   dh2 , res   = post_attention_layernorm(attn, res)    # fused add+RMSNorm
//   mlp         = mlp(dh2)
//   hidden      = post_mlp_layernorm(mlp)                # STANDALONE norm
// `hidden` (bf16 [T,H]) is the sublayer delta; `res` (bf16 [T,H]) the residual.
void RunLayer(Dev d, const Glm4LayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // input_layernorm (fused add+RMSNorm, std/non-gemma): res += hidden; dhn=norm(res)
  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

  // self-attention
  DBuf attn = Glm4AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  // post_self_attn_layernorm (STANDALONE, sandwich): attn_n = norm(attn)
  Tensor w_psa = ResidentWeight(d, layer.post_self_attn_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_psa, vt::RmsNormArgs{eps, false});

  // post_attention_layernorm (fused add+RMSNorm): res += attn_n; dh2=norm(res)
  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn_n.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  // MLP
  DBuf mlp = Glm4MlpBlock(d, layer.mlp, cfg, dh2.t(), T);

  // post_mlp_layernorm (STANDALONE, sandwich): hidden = norm(mlp)  (next delta)
  Tensor w_pm = ResidentWeight(d, layer.post_mlp_layernorm, {H});
  hidden = DBuf(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, hidden.t(), mlp.t(), w_pm, vt::RmsNormArgs{eps, false});
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

// embed -> N layers -> final RMSNorm -> lm_head. Returns [n_out, vocab] f32.
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv, const Glm4Weights& weights,
                 const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const int rot = static_cast<int>(config.rotary_dim);
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "glm4: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "glm4: one PagedKvCache per layer required");

  // Embed: hidden[T,H] bf16 = embed_tokens[token_ids].
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "glm4 embed");
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);
  // GLM always ropes off the bf16 cos/sin cache (is_neox_style=false). Guarantee
  // the bf16 cache is populated regardless of VT_QWEN3_ROPE_CACHE (BuildStepInputs
  // only casts it when that flag is on); idempotent when already built.
  if (rot > 0) vt::CastBf16(d.q, si.cos_sin_bf16.t(), si.cos_sin.t());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  // lm_head (UNTIED for GLM-4-9B-0414): the loaded Matmul-B [H, vocab] via
  // vt::Matmul. (Tied fallback aliases embed_tokens via MatmulBT, for parity with
  // the qwen3 template — not used by GLM-4-9B-0414.)
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
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

std::vector<float> Glm4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Glm4Weights& weights, const HfConfig& config, vt::Queue& queue,
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

ForwardLogits Glm4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Glm4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config,
                  logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
