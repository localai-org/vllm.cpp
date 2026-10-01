// Command-R / Cohere (`CohereForCausalLM`) forward — a GPT-J-style PARALLEL-residual
// dense GQA decoder with a weight-only (bias-free) mean-centred Cohere LayerNorm,
// full-width GPT-J RoPE from a plain cache, SiLU-SwiGLU, tied embeddings and a
// logit_scale scalar. ZERO new kernel; composed from public vt:: ops reusing the
// shared dense device glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs).
//
// Grounding: vllm/model_executor/models/commandr.py @ e24d1b24
//   CohereDecoderLayer.forward (:257-273): residual = h; h = input_layernorm(h);
//     h_attn = self_attn(h); h_mlp = mlp(h); h = residual + h_attn + h_mlp.
//     ONE norm per layer feeds BOTH attn AND mlp (parallel residual).
//   CohereAttention.forward (:214-231): qkv (no bias) -> split -> rope (GPT-J,
//     is_neox_style=False, applied for v1) -> attn(scale head_dim**-0.5) -> o_proj.
//   CohereMLP.forward (:119-122): gate_up -> SiluAndMul -> down.
//   layer_norm_func (:65-73): f32 mean-subtract, rsqrt(var+eps), weight scale
//     (NO bias), round to input dtype.
//   CohereForCausalLM (:376,404-413): logits = logit_scale * (embed_tokens @ h).
//
// Numeric contract (mirrors the OPT/stablelm bf16 path): bf16 residual stream; qkv
// GEMM, full-width GPT-J RoPE (bf16 cos/sin cache), paged FA2 attention, o_proj,
// MLP flow bf16; each Cohere LayerNorm accumulates mean/variance in f32 and rounds
// to bf16 (bias pointer null). Final logits GEMM in f32, then a single f32
// MulScalar by logit_scale. Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/commandr.h"

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

// Cohere LayerNorm over the [T,H] stream (weight-only, NO learnable bias). The
// null bias pointer selects the mean-centred, weight-scaled path the CUDA/CPU
// LayerNorm kernels already implement (they branch on bias==nullptr).
void LayerNormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& weight,
                   float eps, int64_t H) {
  Tensor wt = ResidentWeight(d, weight, {H});
  vt::LayerNorm(d.q, out.t(), x, &wt, /*bias=*/nullptr, vt::LayerNormArgs{eps});
}

// Command-R SwiGLU MLP (merged gate_up -> SiluAndMul -> down). `dhn` is the
// input-LayerNormed hidden [T,H] bf16 (the SAME norm output attention reads).
DBuf CommandrMlpBlock(Dev d, const CommandrMlpWeights& w, const HfConfig& cfg,
                      const Tensor& dhn, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — the seam's Apply IS {ResidentWeight; MatmulBT[2I,H];
  // SiluAndMul} with M = x.shape[0] == T. Mirrors upstream's single
  // MergedColumnParallelLinear + SiluAndMul (commandr.py:102-108,116), and picks up
  // the nvfp4 GateUpFusedMarlinD arm for free once a quantized checkpoint ships.
  // (FUSION-DENSE-MIGRATE, specs/fusion-dense-migrate.md.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dhn);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One Command-R self-attention block. `dhn` is the input-LayerNormed hidden [T,H]
// bf16; returns the o_proj output [T,H] bf16. Merged qkv (NO bias) -> split ->
// full-width GPT-J RoPE from the plain cache (real positions, rotary_dim ==
// head_dim, is_neox_style=False) -> paged FA2. NO qk-norm (Cohere2-only, rejected
// at config parse), standard 1/sqrt(Dh) scale, o_proj bias-free.
DBuf CommandrAttnBlock(Dev d, const CommandrAttnWeights& w, const Tensor& rope_cache,
                       const HfConfig& cfg, const Tensor& dhn, const StepInputs& si,
                       const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                       int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "commandr: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "commandr: KV cache head dims mismatch config");

  DBuf qkv(d, DType::kBF16, {T, qdim + 2 * kdim});
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);  // no bias (QKVParallelLinear bias=False)
  DBuf q(d, DType::kBF16, {T, qdim});
  DBuf k(d, DType::kBF16, {T, kdim});
  DBuf v(d, DType::kBF16, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  // Full-width GPT-J RoPE from the plain cos/sin cache: gather cache[position]
  // (REAL positions) and rotate ALL head_dim dims of each head with the adjacent
  // (2i,2i+1) pair rule (is_neox_style=False), mirroring get_rope(..., is_neox_
  // style=False) at commandr.py:174-179.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  {
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(Dh);  // full rotary
    ra.is_neox_style = false;              // GPT-J adjacent-pair
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
  vt::MatmulBT(d.q, o.t(), o_in, wo);  // no bias (RowParallelLinear bias=False)
  return o;
}

// One Command-R decoder layer (GPT-J PARALLEL residual). ONE LayerNorm feeds BOTH
// attention and MLP; their outputs are summed back onto the residual TOGETHER:
//   residual = h; n = LayerNorm(h); h = residual + attn(n) + mlp(n).
void RunLayer(Dev d, const CommandrLayerWeights& layer, const Tensor& rope_cache,
              const HfConfig& cfg, float eps, DBuf& hidden, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;

  // ONE input LayerNorm (weight-only, no bias) -> feeds BOTH attn and MLP.
  DBuf normed(d, DType::kBF16, {T, H});
  LayerNormInto(d, normed, hidden.t(), layer.input_layernorm, eps, H);

  DBuf attn = CommandrAttnBlock(d, layer.attn, rope_cache, cfg, normed.t(), si, meta,
                                kv, T);
  DBuf mlp = CommandrMlpBlock(d, layer.mlp, cfg, normed.t(), T);

  // Parallel re-join: hidden = residual + attn + mlp (commandr.py:271).
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
                 const std::vector<PagedKvCache>& attn_kv,
                 const CommandrWeights& weights, const HfConfig& config,
                 float eps, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "commandr: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "commandr: one PagedKvCache per layer required");

  Tensor rope_cache = ResidentWeight(d, weights.rope_cos_sin);

  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "commandr embed");
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], rope_cache, config, eps,
             hidden, si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  DBuf dnorm(d, DType::kBF16, {T, H});
  LayerNormInto(d, dnorm, hidden.t(), weights.final_norm, eps, H);

  // Tied embeddings: logits = embed_tokens @ h (commandr.py asserts tie=True).
  Tensor lm = ResidentWeight(d, weights.embed_tokens, {vocab, H});

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
  vt::MatmulBT(d.q, logits.t(), src, lm);
  // logit_scale: logits *= config.logit_scale (commandr.py:376). Applied as a
  // single f32 scalar over the [n_out, vocab] logits (fold into the lm_head
  // epilogue) — mirrors LogitsProcessor(scale=logit_scale).
  if (weights.logit_scale != 1.0)
    vt::MulScalar(d.q, logits.t(), logits.t(), weights.logit_scale);
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

std::vector<float> CommandrModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const CommandrWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = CommandrLayerNormEps(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits CommandrModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const CommandrWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = CommandrLayerNormEps(config);
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, eps, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
