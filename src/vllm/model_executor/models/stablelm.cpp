// StableLM (`StableLmForCausalLM`, stabilityai/stablelm-2-1_6b) forward — a
// pre-norm dense GQA/MHA decoder with nn.LayerNorm (weight+bias, NON-fused
// residual add, like OPT) + partial NeoX RoPE from a plain cache + optional merged
// qkv bias + SiLU-SwiGLU. ZERO new kernel; composed from public vt:: ops reusing
// the shared dense device glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs).
//
// Grounding: vllm/model_executor/models/stablelm.py @ e24d1b24
//   StablelmDecoderLayer.forward (:198-215): residual = h; h = input_layernorm(h);
//     h = self_attn(h); h = residual + h; residual = h;
//     h = post_attention_layernorm(h); h = mlp(h); h = residual + h.
//   StablelmAttention.forward (:158-174): qkv (+bias) -> split -> rotary_emb (partial
//     NeoX) -> attn(scale head_dim**-0.5) -> o_proj.
//   StablelmMLP.forward (:84-89): gate_up -> SiluAndMul -> down.
//
// Numeric contract (mirrors OPT/phi3 bf16 path): bf16 residual stream; qkv GEMM,
// merged bias add, partial RoPE (bf16 cos/sin cache), paged FA2 attention, o_proj,
// MLP flow bf16; each nn.LayerNorm accumulates mean/variance in f32 and rounds to
// bf16. Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/stablelm.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // shared device glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool
#include "vllm/model_executor/models/qwen3_5_common.h"     // HostLogits
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

using namespace dense_attn;

// nn.LayerNorm over the [T,H] stream (weight+bias, always affine for StableLM).
void LayerNormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& weight,
                   const OwnedTensor& bias, float eps, int64_t H) {
  Tensor wt = ResidentWeight(d, weight, {H});
  Tensor bt = ResidentWeight(d, bias, {H});
  vt::LayerNorm(d.q, out.t(), x, &wt, &bt, vt::LayerNormArgs{eps});
}

// StableLM SwiGLU MLP (merged gate_up -> SiluAndMul -> down). `dh2` is the
// post-norm hidden [T,H] bf16.
DBuf StablelmMlpBlock(Dev d, const StablelmMlpWeights& w, const HfConfig& cfg,
                      const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — folds StableLM onto the qwen3.cpp MlpBlock exemplar so it
  // inherits the nvfp4 GateUpFusedMarlinD arm for free once a quantized
  // checkpoint ships. (Tier-A1 fold, arch-fusion-fold-plan-2026-07-30.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One StableLM self-attention block. `dhn` is the input-LayerNormed hidden [T,H]
// bf16; returns the o_proj output [T,H] bf16. Merged qkv (+ optional bias) ->
// split -> partial NeoX RoPE from the plain cache (real positions, rotary_dim <
// head_dim) -> paged FA2. NO qk-norm, standard 1/sqrt(Dh) scale, o_proj bias-free.
DBuf StablelmAttnBlock(Dev d, const StablelmAttnWeights& w, const Tensor& rope_cache,
                       const HfConfig& cfg, const Tensor& dhn, const StepInputs& si,
                       const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                       int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "stablelm: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "stablelm: KV cache head dims mismatch config");

  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
  if (!w.qkv_bias.Empty()) {
    Tensor b = ResidentWeight(d, w.qkv_bias, {qdim + 2 * kdim});
    vt::Add(d.q, qkv.t(), qkv.t(), b);  // rank-1 row-broadcast, in place
  }
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // Partial NeoX RoPE from the plain cos/sin cache: gather cache[position] (REAL
  // positions) and rotate the leading rotary_dim dims of each head.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  if (rot > 0) {
    vt::RopeArgs ra;
    ra.rotary_dim = rot;
    ra.is_neox_style = true;
    Tensor k3v = k3;
    vt::RopeFromCache(d.q, q3, &k3v, si.positions.t(), rope_cache, ra);
  }

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

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// One StableLM decoder layer (pre-norm, non-fused residual add — OPT-style). The
// full residual stream lives in `hidden`; there is no separate accumulator.
void RunLayer(Dev d, const StablelmLayerWeights& layer, const Tensor& rope_cache,
              const HfConfig& cfg, float eps, DBuf& hidden, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;

  // ---- self attention: residual = h; h = LN(h); h = attn(h); h = residual + h --
  DBuf normed(d, DType::kBF16, {T, H});
  LayerNormInto(d, normed, hidden.t(), layer.input_layernorm,
                layer.input_layernorm_bias, eps, H);
  DBuf attn = StablelmAttnBlock(d, layer.attn, rope_cache, cfg, normed.t(), si, meta,
                                kv, T);
  vt::Add(d.q, attn.t(), attn.t(), hidden.t());  // residual + hidden_states
  hidden = std::move(attn);

  // ---- fully connected: residual = h; h = LN(h); h = mlp(h); h = residual + h ---
  DBuf normed2(d, DType::kBF16, {T, H});
  LayerNormInto(d, normed2, hidden.t(), layer.post_attention_layernorm,
                layer.post_attention_layernorm_bias, eps, H);
  DBuf mlp = StablelmMlpBlock(d, layer.mlp, cfg, normed2.t(), T);
  vt::Add(d.q, mlp.t(), mlp.t(), hidden.t());  // residual + mlp
  hidden = std::move(mlp);
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const StablelmWeights& weights, const HfConfig& config,
                 float eps, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "stablelm: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "stablelm: one PagedKvCache per layer required");

  Tensor rope_cache = ResidentWeight(d, weights.rope_cos_sin);

  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "stablelm embed");
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], rope_cache, config, eps,
             hidden, si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  DBuf dnorm(d, DType::kBF16, {T, H});
  LayerNormInto(d, dnorm, hidden.t(), weights.final_norm, weights.final_norm_bias,
                eps, H);

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

std::vector<float> StablelmModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const StablelmWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = StablelmLayerNormEps(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits StablelmModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const StablelmWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = StablelmLayerNormEps(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
