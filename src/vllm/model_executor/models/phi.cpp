// Phi-1 / Phi-1.5 / Phi-2 (`PhiForCausalLM`, microsoft/phi-2) forward — a GPT-J-style
// PARALLEL-residual dense MHA decoder with nn.LayerNorm (weight+bias), biased
// q/k/v/dense projections, partial NeoX RoPE, a NON-gated NewGELU MLP (fc1 -> gelu_new
// -> fc2, both biased), a final nn.LayerNorm and an UNTIED lm_head carrying a bias.
// ZERO new kernel; composed from public vt:: ops reusing the shared dense device
// glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs).
//
// Grounding: vllm/model_executor/models/phi.py @ e24d1b24
//   PhiLayer.forward (:189-202): residual = h; h = input_layernorm(h);
//     attn_out = self_attn(h); ffn_out = mlp(h); h = attn_out + ffn_out + residual.
//     ONE nn.LayerNorm feeds BOTH attn AND mlp (parallel residual).
//   PhiAttention.forward (:126-136): qkv (+bias) -> chunk -> rotary_emb (partial
//     NeoX, is_neox_style default True) -> attn(scale head_size**-0.5) -> dense (+bias).
//   PhiMLP.forward (:165-169): fc1 (+bias) -> gelu_new -> fc2 (+bias).
//   PhiForCausalLM.compute_logits (:316-321): logits = lm_head(h) + lm_head.bias.
//
// gelu_new (activation.py:516-519) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) —
// bit-identical to vt::GeluTanh (the Qwen3-VL vision tower unary), so the non-gated
// MLP needs NO new op.
//
// Numeric contract (mirrors the OPT/StableLM bf16 path): bf16 residual stream; qkv
// GEMM + merged bias, partial RoPE (bf16 cos/sin cache), paged FA2 attention, dense
// (+bias), fc1 (+bias), gelu, fc2 (+bias) all bf16; each nn.LayerNorm accumulates
// mean/variance in f32 and rounds to bf16. Final logits GEMM in f32; the lm_head
// bias is added into the f32 logits (bf16 broadcast, computed in f32). Returns
// [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/phi.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

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

// nn.LayerNorm over the [T,H] stream (weight+bias, always affine for Phi).
void LayerNormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& weight,
                   const OwnedTensor& bias, float eps, int64_t H) {
  Tensor wt = ResidentWeight(d, weight, {H});
  Tensor bt = ResidentWeight(d, bias, {H});
  vt::LayerNorm(d.q, out.t(), x, &wt, &bt, vt::LayerNormArgs{eps});
}

// A biased projection: out = x @ W^T + bias (nn.Linear-shaped Column/Row/QKV
// ParallelLinear with bias=True). Mirrors the OPT BiasedProj helper.
DBuf BiasedProj(Dev d, const OwnedTensor& weight, const OwnedTensor& bias,
                const Tensor& x, int64_t T, int64_t n_out) {
  Tensor w = ResidentWeight(d, weight);
  DBuf out(d, DType::kBF16, {T, n_out});
  vt::MatmulBT(d.q, out.t(), x, w);
  if (!bias.Empty()) {
    Tensor b = ResidentWeight(d, bias, {n_out});
    vt::Add(d.q, out.t(), out.t(), b);  // rank-1 row-broadcast, in place
  }
  return out;
}

// Phi NON-gated GELU MLP (phi.py::PhiMLP): fc1 (+bias) -> gelu_new -> fc2 (+bias).
// `dhn` is the input-LayerNormed hidden [T,H] bf16 (the SAME norm attention reads).
DBuf PhiMlpBlock(Dev d, const PhiMlpWeights& w, int64_t ffn, int64_t H,
                 const Tensor& dhn, int64_t T) {
  DBuf h = BiasedProj(d, w.fc1, w.fc1_bias, dhn, T, ffn);
  vt::GeluTanh(d.q, h.t(), h.t());  // gelu_new (NewGELU tanh-approx), in place
  return BiasedProj(d, w.fc2, w.fc2_bias, h.t(), T, H);
}

// One Phi self-attention block. `dhn` is the input-LayerNormed hidden [T,H] bf16;
// returns the dense (o_proj) output [T,H] bf16. Merged qkv (+bias) -> split ->
// partial NeoX RoPE from the plain cache (real positions, rotary_dim < head_dim)
// -> paged FA2. NO qk-norm, standard 1/sqrt(Dh) scale, dense carries a bias.
DBuf PhiAttnBlock(Dev d, const PhiAttnWeights& w, const Tensor& rope_cache,
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
           "phi: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "phi: KV cache head dims mismatch config");

  DBuf qkv = BiasedProj(d, w.qkv_proj, w.qkv_bias, dhn, T, qdim + 2 * kdim);
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
  return BiasedProj(d, w.dense, w.dense_bias, o_in, T, H);
}

// One Phi decoder layer (GPT-J PARALLEL residual, phi.py:189-202). ONE nn.LayerNorm
// (weight+bias) feeds BOTH attention and MLP; their outputs are summed back onto the
// residual TOGETHER: residual = h; n = LayerNorm(h); h = residual + attn(n) + mlp(n).
void RunLayer(Dev d, const PhiLayerWeights& layer, const Tensor& rope_cache,
              const HfConfig& cfg, float eps, int64_t ffn, DBuf& hidden,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;

  // ONE input LayerNorm (weight+bias) -> feeds BOTH attn and MLP.
  DBuf normed(d, DType::kBF16, {T, H});
  LayerNormInto(d, normed, hidden.t(), layer.input_layernorm,
                layer.input_layernorm_bias, eps, H);

  DBuf attn = PhiAttnBlock(d, layer.attn, rope_cache, cfg, normed.t(), si, meta, kv, T);
  DBuf mlp = PhiMlpBlock(d, layer.mlp, ffn, H, normed.t(), T);

  // Parallel re-join: hidden = residual + attn + mlp (phi.py:201).
  vt::Add(d.q, hidden.t(), hidden.t(), attn.t());
  vt::Add(d.q, hidden.t(), hidden.t(), mlp.t());
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
                 const std::vector<PagedKvCache>& attn_kv, const PhiWeights& weights,
                 const HfConfig& config, float eps, int64_t ffn,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "phi: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "phi: one PagedKvCache per layer required");

  Tensor rope_cache = ResidentWeight(d, weights.rope_cos_sin);

  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "phi embed");
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], rope_cache, config, eps, ffn,
             hidden, si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  DBuf dnorm(d, DType::kBF16, {T, H});
  LayerNormInto(d, dnorm, hidden.t(), weights.final_norm, weights.final_norm_bias,
                eps, H);

  // Untied lm_head with a per-vocab bias (phi.py asserts NOT tie_word_embeddings).
  Tensor lm = ResidentWeight(d, weights.lm_head);  // [H, vocab] Matmul-B

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
  vt::Matmul(d.q, logits.t(), src, lm);
  // lm_head bias (phi.py:320): logits += lm_head.bias, broadcast over rows. Added
  // in f32 (bf16 bias upcast) — the [vocab] row-broadcast Add.
  if (!weights.lm_head_bias.Empty()) {
    Tensor b = ResidentWeight(d, weights.lm_head_bias, {vocab});
    vt::Add(d.q, logits.t(), logits.t(), b);
  }
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

std::vector<float> PhiModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const PhiWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = PhiLayerNormEps(config);
  const int64_t ffn = PhiFfnDim(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, ffn, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits PhiModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const PhiWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = PhiLayerNormEps(config);
  const int64_t ffn = PhiFfnDim(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, ffn, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
