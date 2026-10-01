// MiniCPM (`MiniCPMForCausalLM`, MiniCPM-2B-sft-bf16) forward — the Llama dense
// forward plus THREE scalar deltas (ZERO new kernel). A plain pre-norm dense
// (M)HA decoder COMPOSED from public vt:: ops, reusing only the shared dense device
// glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs) — NOT dense_attn::AttnBlock,
// which hard-codes the fused residual add that MiniCPM's scale_depth violates.
//
// Grounding: vllm/model_executor/models/minicpm.py @ e24d1b24 — MiniCPMAttention
// (scale = head_dim**-0.5, :266,310), MiniCPMDecoderLayer.forward (:378-394, scaled
// residual adds with scale_depth/sqrt(num_hidden_layers)), MiniCPMModel.embed_input_ids
// (:441-443, scale_emb), MiniCPMForCausalLM (scale_width = hidden/dim_model_base
// :604; hidden /= scale_width :633,640 before lm_head).
//
// Numeric contract (mirrors Granite/GLM-4/Qwen3-dense bf16 path): residual stream
// bf16; qkv GEMM, rope, paged FA2 attention, o_proj and MLP flow bf16; the
// standalone (non-gemma) RMSNorms compute variance in f32 and round to bf16; the
// scalar multiplies + residual adds run in f32 (vt::MulScalar / vt::Add) and round
// to bf16. Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/minicpm.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
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
// StepInputs/BuildStepInputs/MakeRopeArgs/RopeCacheEnabled/Reshape/Pool). We do NOT
// call dense_attn::AttnBlock (no fused residual add; scale_depth needs the scaled
// standalone add). MiniCPM's rope is plain NeoX (rope_scaling null), so
// BuildStepInputs' MakeRopeArgs cos/sin cache is exactly right.
using namespace dense_attn;

// Read a top-level double scalar from config.json (MiniCPM scalars are flat),
// defaulting when absent/null/non-numeric.
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}

// MiniCPM dense SwiGLU MLP (minicpm.py::MiniCPMMLP, hidden_act "silu"): merged
// gate_up MatmulBT -> SiluAndMul -> down MatmulBT. `h` is the post-norm hidden
// [T,H] bf16.
DBuf MiniCPMMlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
                     const Tensor& h, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — the seam's Apply IS {ResidentWeight; MatmulBT[2I,H];
  // SiluAndMul} with M = x.shape[0] == T. Mirrors upstream's single
  // MergedColumnParallelLinear + SiluAndMul (minicpm.py:204-211,219), and picks up
  // the nvfp4 GateUpFusedMarlinD arm for free once a quantized checkpoint ships.
  // The DIRECT Unquantized arm rather than MakeMlpGateUpMethod: MiniCPM's loader
  // never populates Qwen3DenseMlpWeights::{gate,up}_proj_fp4, so the factory would
  // add an untestable quantized branch (spec §Port map). (FUSION-DENSE-MIGRATE.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, h);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One MiniCPM self-attention block (minicpm.py::MiniCPMAttention). `h` is the
// input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16. Deltas vs
// dense_attn::AttnBlock: (a) NO per-head qk-norm; (b) NO biases; (c) plain NeoX
// RoPE. The softmax scale IS the standard 1/sqrt(Dh) (minicpm.py:266).
DBuf MiniCPMAttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
                      const Tensor& h, const StepInputs& si,
                      const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                      int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "minicpm: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "minicpm: KV cache head dims mismatch config");

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
  // MiniCPM: standard softmax scale = 1/sqrt(head_dim) (minicpm.py:266).
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
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

// One MiniCPM decoder layer (minicpm.py::MiniCPMDecoderLayer.forward :378-394):
//   normed = input_layernorm(res)          # STANDALONE RMSNorm
//   attn   = self_attn(normed)             # scale = head_dim**-0.5
//   res    = res + attn * residual_scale    (residual_scale = scale_depth/sqrt(L))
//   normed = post_attention_layernorm(res)
//   mlp    = mlp(normed)
//   res    = res + mlp * residual_scale
// `res` (bf16 [T,H]) is the running residual stream. The scaled residual add is a
// vt::MulScalar into a scratch + vt::Add (no fused-add-RMSNorm — scale_depth forbids
// it).
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              double residual_scale, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv,
              int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf normed(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed.t(), res.t(), w_in, vt::RmsNormArgs{eps, false});

  DBuf attn = MiniCPMAttnBlock(d, layer.attn, cfg, normed.t(), si, meta, kv, T);
  DBuf scaled(d, DType::kBF16, {T, H});
  vt::MulScalar(d.q, scaled.t(), attn.t(), residual_scale);
  vt::Add(d.q, res.t(), res.t(), scaled.t());

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf normed2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed2.t(), res.t(), w_post, vt::RmsNormArgs{eps, false});

  DBuf mlp = MiniCPMMlpBlock(d, layer.mlp, cfg, normed2.t(), T);
  vt::MulScalar(d.q, scaled.t(), mlp.t(), residual_scale);
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
                 const MiniCPMWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const int rot = static_cast<int>(config.rotary_dim);
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "minicpm: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "minicpm: one PagedKvCache per layer required");

  // The three MiniCPM scalar deltas (minicpm.py). scale_emb: embedding scale.
  // scale_depth/sqrt(num_hidden_layers): per-sublayer residual scale. scale_width =
  // hidden_size/dim_model_base: the pre-lm_head logit divisor (dim_model_base>0).
  const double scale_emb = RawDouble(config.raw, "scale_emb", 1.0);
  const double scale_depth = RawDouble(config.raw, "scale_depth", 1.0);
  const double residual_scale =
      scale_depth / std::sqrt(static_cast<double>(config.num_hidden_layers));
  const double dim_model_base = RawDouble(config.raw, "dim_model_base",
                                          static_cast<double>(H));
  const double scale_width =
      dim_model_base > 0.0 ? static_cast<double>(H) / dim_model_base : 1.0;

  // Embed then scale by scale_emb (minicpm.py:441-443).
  DBuf res(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, res, token_ids, weights.embed_tokens, vocab, H,
                "minicpm embed");
  }
  vt::MulScalar(d.q, res.t(), res.t(), scale_emb);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);
  if (rot > 0 && RopeCacheEnabled())
    vt::CastBf16(d.q, si.cos_sin_bf16.t(), si.cos_sin.t());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, residual_scale, res,
             si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final standalone RMSNorm (minicpm.py MiniCPMModel.norm), then hidden /=
  // scale_width (minicpm.py:633,640) before lm_head. Dividing the hidden by a
  // positive scalar before the linear lm_head is identical to dividing the logits
  // after (lm_head has no bias); done on the hidden to mirror vLLM's exact op order.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), res.t(), w_fn, vt::RmsNormArgs{eps, false});
  if (scale_width != 1.0 && scale_width > 0.0)
    vt::MulScalar(d.q, dnorm.t(), dnorm.t(), 1.0 / scale_width);

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

std::vector<float> MiniCPMModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const MiniCPMWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits MiniCPMModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const MiniCPMWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
