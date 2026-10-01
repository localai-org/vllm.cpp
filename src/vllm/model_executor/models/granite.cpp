// Granite-3 (`GraniteForCausalLM`, granite-3.3-2b-instruct) forward — the Llama
// dense forward plus FOUR default-1 scalar multipliers (ZERO new kernel). A plain
// pre-norm dense GQA decoder COMPOSED from public vt:: ops, reusing only the shared
// dense device glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs) — NOT
// dense_attn::AttnBlock, which hard-codes the 1/sqrt(Dh) attention scale and the
// fused residual add that Granite's attention_multiplier / residual_multiplier
// both violate.
//
// Grounding: vllm/model_executor/models/granite.py @ e24d1b24 — GraniteAttention
// (scale = config.attention_multiplier, :137,165), GraniteDecoderLayer.forward
// (:224-246, scaled residual adds), GraniteModel.forward (:305-315, embedding
// multiplier), GraniteForCausalLM logit_scale /= logits_scaling (:371-372).
//
// Numeric contract (mirrors GLM-4/Qwen3-dense bf16 path): residual stream bf16;
// qkv GEMM, rope, paged FA2 attention, o_proj and MLP flow bf16; the standalone
// (non-gemma) RMSNorms compute variance in f32 and round to bf16; the scalar
// multiplies + residual adds run in f32 (vt::MulScalar / vt::Add) and round to
// bf16. Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/granite.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // MakeMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // shared device glue
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

// Reuse the shared dense device glue VERBATIM (Dev/DBuf/ResidentWeight/KvSlice/
// StepInputs/BuildStepInputs/MakeRopeArgs/RopeCacheEnabled/Reshape/Pool). We do
// NOT call dense_attn::AttnBlock (custom scale + no fused residual). Granite's rope
// is plain NeoX (rope_scaling null), so BuildStepInputs' MakeRopeArgs cos/sin cache
// is exactly right.
using namespace dense_attn;

// Read a top-level double scalar from config.json (Granite scalars are flat),
// defaulting when absent/null/non-numeric.
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}

// Granite dense SwiGLU MLP (granite.py::GraniteMLP): merged gate_up MatmulBT ->
// SiluAndMul -> down MatmulBT. `h` is the post-norm hidden [T,H] bf16.
DBuf GraniteMlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
                     const Tensor& h, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED gate-up MLP seam. Granite's
  // MLP weights are literally Qwen3DenseMlpWeights (the factory's operand type),
  // so this is the exact exemplar (qwen3.cpp MlpBlock): byte-for-byte the same op
  // sequence, and MakeMlpGateUpMethod selects the nvfp4 GateUpFusedMarlinD arm
  // automatically the moment a quantized Granite checkpoint populates *_fp4.
  // (Tier-A1 fold, arch-fusion-fold-plan-2026-07-30.)
  auto gate_up = layers::MakeMlpGateUpMethod(w.gate_up_proj, w.gate_proj_fp4,
                                             w.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, h);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One Granite self-attention block (granite.py::GraniteAttention). `h` is the
// input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16. Deltas vs
// dense_attn::AttnBlock: (a) the softmax scale is config.attention_multiplier
// (NOT 1/sqrt(Dh)); (b) NO per-head qk-norm; (c) NO biases; (d) plain NeoX RoPE.
DBuf GraniteAttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
                      double attention_multiplier, const Tensor& h,
                      const StepInputs& si, const CommonAttentionMetadata& meta,
                      const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "granite: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "granite: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (bias=False): one MatmulBT over the [q|k|v] owner,
  // then split into contiguous q/k/v shards.
  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), h, wqkv);
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // Plain NeoX RoPE (no qk-norm). Two paths (RopeCacheEnabled default-ON): the
  // bf16 cos/sin cache off BuildStepInputs (vLLM-faithful) or in-place RopeNeox.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  if (RopeCacheEnabled() && rot > 0) {
    Tensor k3v = k3;
    vt::RopeFromCache(d.q, q3, &k3v, si.rope_row_idx.t(), si.cos_sin_bf16.t(),
                      MakeRopeArgs(cfg));
  } else if (rot > 0) {
    vt::RopeNeox(d.q, q3, k3, si.positions.t(), MakeRopeArgs(cfg));
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
  // Granite: scale = config.attention_multiplier (NOT 1/sqrt(Dh)).
  vt::PagedAttentionArgs pa{static_cast<float>(attention_multiplier), meta.causal};
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

// One Granite decoder layer (granite.py::GraniteDecoderLayer.forward :224-246):
//   normed = input_layernorm(res)          # STANDALONE RMSNorm
//   attn   = self_attn(normed)             # scale = attention_multiplier
//   res    = res + attn * residual_multiplier
//   normed = post_attention_layernorm(res)
//   mlp    = mlp(normed)
//   res    = res + mlp * residual_multiplier
// `res` (bf16 [T,H]) is the running residual stream. The scaled residual add is a
// vt::MulScalar into a scratch + vt::Add (no fused-add-RMSNorm — the multiplier
// forbids it).
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              double attention_multiplier, double residual_multiplier, DBuf& res,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf normed(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed.t(), res.t(), w_in, vt::RmsNormArgs{eps, false});

  DBuf attn = GraniteAttnBlock(d, layer.attn, cfg, attention_multiplier, normed.t(),
                               si, meta, kv, T);
  DBuf scaled(d, DType::kBF16, {T, H});
  vt::MulScalar(d.q, scaled.t(), attn.t(), residual_multiplier);
  vt::Add(d.q, res.t(), res.t(), scaled.t());

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf normed2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed2.t(), res.t(), w_post, vt::RmsNormArgs{eps, false});

  DBuf mlp = GraniteMlpBlock(d, layer.mlp, cfg, normed2.t(), T);
  vt::MulScalar(d.q, scaled.t(), mlp.t(), residual_multiplier);
  vt::Add(d.q, res.t(), res.t(), scaled.t());
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
                 const GraniteWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const int rot = static_cast<int>(config.rotary_dim);
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "granite: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "granite: one PagedKvCache per layer required");

  const double embedding_multiplier =
      RawDouble(config.raw, "embedding_multiplier", 1.0);
  const double residual_multiplier =
      RawDouble(config.raw, "residual_multiplier", 1.0);
  const double attention_multiplier = RawDouble(
      config.raw, "attention_multiplier",
      1.0 / std::sqrt(static_cast<double>(config.head_dim)));
  const double logits_scaling = RawDouble(config.raw, "logits_scaling", 1.0);

  // Embed then scale by embedding_multiplier (granite.py:313).
  DBuf res(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, res, token_ids, weights.embed_tokens, vocab, H,
                "granite embed");
  }
  vt::MulScalar(d.q, res.t(), res.t(), embedding_multiplier);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);
  if (rot > 0 && RopeCacheEnabled())
    vt::CastBf16(d.q, si.cos_sin_bf16.t(), si.cos_sin.t());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config,
             attention_multiplier, residual_multiplier, res, si, attn_meta,
             attn_kv[static_cast<size_t>(l)], T);

  // Final standalone RMSNorm (granite.py GraniteModel.norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), res.t(), w_fn, vt::RmsNormArgs{eps, false});

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
  // logits /= logits_scaling (granite.py:371-372). A positive divisor does not
  // change greedy argmax, but is applied for fidelity (f32 scale).
  if (logits_scaling != 1.0 && logits_scaling > 0.0)
    vt::MulScalar(d.q, logits.t(), logits.t(), 1.0 / logits_scaling);
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

std::vector<float> GraniteModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const GraniteWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits GraniteModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const GraniteWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
