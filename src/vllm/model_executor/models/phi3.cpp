// Phi-3 / Phi-4 dense (`Phi3ForCausalLM`, Phi-4-mini-instruct) forward — the Llama
// dense forward with the precomputed LongRoPE cache (ZERO new kernel). A pre-norm
// dense GQA decoder COMPOSED from public vt:: ops, reusing the shared dense device
// glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs). Written fresh (not
// dense_attn::AttnBlock) ONLY because the RoPE must read the LongRoPE cache
// (partial rotary, LONG rescale + mscale) that MakeRopeArgs cannot express.
//
// Grounding: vllm/model_executor/models/phi3.py @ e24d1b24 (LlamaForCausalLM
// subclass, pre-fused packed_modules_mapping) + llama.py decoder (pre-norm fused
// add+RMSNorm, 1/sqrt(Dh) scale, SwiGLU) + Phi3LongRoPEScaledRotaryEmbedding.
//
// Numeric contract (mirrors Qwen3-dense/GLM-4 bf16 path): bf16 residual stream;
// qkv GEMM, RoPE (bf16 cos/sin cache), paged FA2 attention, o_proj and MLP flow
// bf16; the fused add+RMSNorm computes variance in f32 and rounds to bf16. Returns
// [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/phi3.h"

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

// Phi-3 SwiGLU MLP (pre-fused gate_up_proj): merged gate_up MatmulBT -> SiluAndMul
// -> down MatmulBT. `dh2` is the post-norm hidden [T,H] bf16.
DBuf Phi3MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
                  const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — the seam's Apply IS {ResidentWeight; MatmulBT[2I,H];
  // SiluAndMul} with M = x.shape[0] == T. Phi-3 IS Llama upstream (phi3.py:10
  // Phi3ForCausalLM(LlamaForCausalLM) -> llama.py:79 LlamaMLP), i.e. one
  // MergedColumnParallelLinear + SiluAndMul. Direct Unquantized arm, not the
  // factory: no *_fp4 is ever populated here (spec §Port map).
  // (FUSION-DENSE-MIGRATE.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One Phi-3 self-attention block. `dhn` is the input-normed hidden [T,H] bf16;
// returns the o_proj output [T,H] bf16. RoPE reads the precomputed LongRoPE cache
// (resident) indexed by the REAL positions; partial rotary (rotary_dim < head_dim).
// NO qk-norm, NO biases, standard 1/sqrt(Dh) scale.
DBuf Phi3AttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const Tensor& rope_cache,
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
           "phi3: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "phi3: KV cache head dims mismatch config");

  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // Partial NeoX RoPE from the precomputed LongRoPE cache: gather cache[position]
  // (REAL positions) and rotate the first rotary_dim dims of each head.
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

// One Phi-3 decoder layer (pre-norm Llama): fused add+RMSNorm -> attention ->
// fused add+RMSNorm -> MLP. `hidden` is the delta, `res` the residual accumulator.
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const Tensor& rope_cache,
              const HfConfig& cfg, DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

  DBuf attn = Phi3AttnBlock(d, layer.attn, rope_cache, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  hidden = Phi3MlpBlock(d, layer.mlp, cfg, dh2.t(), T);
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
                 const std::vector<PagedKvCache>& attn_kv, const Phi3Weights& weights,
                 const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "phi3: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "phi3: one PagedKvCache per layer required");

  const Qwen3DenseWeights& dw = weights.dense;
  Tensor rope_cache = ResidentWeight(d, weights.rope_cos_sin);

  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, dw.embed_tokens, vocab, H,
                "phi3 embed");
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, dw.layers[static_cast<size_t>(l)], rope_cache, config, hidden, res,
             si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  Tensor w_fn = ResidentWeight(d, dw.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  const bool tied = dw.tie_word_embeddings || dw.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, dw.embed_tokens, {vocab, H})
                   : ResidentWeight(d, dw.lm_head);

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

std::vector<float> Phi3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Phi3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Phi3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Phi3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
