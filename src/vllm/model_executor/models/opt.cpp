// OPT (`OPTForCausalLM`) forward — the CROSS-FAMILY additivity canary, W3.
//
// Grounding: vllm/model_executor/models/opt.py @ e24d1b24 —
//   OPTLearnedPositionalEmbedding (:59-68), OPTAttention (:71-122),
//   OPTDecoderLayer (:125-195), OPTDecoder (:198-293), OPTForCausalLM
//   (:327-394). See .agents/specs/sweep-opt-125m.md §Port map.
//
// This forward REUSES the landed device glue verbatim — `Dev`/`DBuf`/the shared
// DevicePool, `ResidentWeight`, `KvSlice`, `MakeTensor`/`Reshape` from
// include/vllm/model_executor/models/dense_attn_block.h — but deliberately does
// NOT reuse `dense_attn::AttnBlock` or `dense_attn::BuildStepInputs`. That is
// the honest additivity result, not an oversight: `AttnBlock` hard-codes the
// Qwen attention preamble (per-head q/k RMSNorm, NeoX RoPE, and an explicit
// `VT_CHECK(w.qkv_bias.Empty())`), and `BuildStepInputs` exists mostly to build
// the RoPE cos|sin cache. OPT has NO q/k norm, NO RoPE, and REQUIRES qkv bias,
// so ~none of that preamble applies; what OPT does reuse is the *device glue*
// underneath it. See the spec's "seams that held / seams that leaked" §L1.
//
// Per decoder layer (do_layer_norm_before == true, the 125m/1.7B/.../175B
// placement — opt.py:168-195):
//   res = hidden ; hidden = LayerNorm(hidden, w, b)
//   -> qkv MatmulBT + merged bias -> slice q|k|v -> ReshapeAndCache
//   -> paged causal MHA -> out_proj + bias ; hidden = res + hidden
//   res = hidden ; hidden = LayerNorm(hidden, w, b)
//   -> fc1 + bias -> ReLU -> fc2 + bias ; hidden = res + hidden
// With do_layer_norm_before == false (the 350m placement) each LayerNorm moves
// to AFTER its residual join — both branches are implemented, but only the
// pre-LN branch has a checkpoint on the box to gate (spec R2).
//
// Numeric contract: the whole stream is bf16 — matching vLLM's per-op bf16
// stores under `--dtype bfloat16`, the arm the SACRED gate runs on (spec D1) —
// with LayerNorm accumulating mean/variance in f32 and rounding once on store,
// exactly as torch's `acc_type<bfloat16> == float` LayerNorm does. The paged KV
// cache is written bf16 and the query enters vt::PagedAttention bf16. Returns
// [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/opt.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/KvSlice glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// Device glue (Dev, DBuf, ResidentWeight, KvSlice, MakeTensor, Reshape, the
// DevicePool policy) — reused verbatim from the shared dense header.
using namespace dense_attn;

// Per-step device inputs OPT actually needs. This is the RoPE-free analog of
// dense_attn::StepInputs: OPT has no rotary embedding at all, so there is no
// cos|sin cache, no bf16 cache mirror and no identity row-index table — but it
// DOES need the position ids ON DEVICE (Qwen only needed them for RoPE), because
// the learned positional embedding is an on-device table lookup.
struct OPTStepInputs {
  DBuf pos_ids;          // i32 [T]  == positions + 2 (the fairseq offset)
  DBuf slot_mapping;     // i64 [T]
  DBuf block_table;      // i32 [num_reqs, cols]
  DBuf seq_lens;         // i32 [num_reqs]
  DBuf query_start_loc;  // i32 [num_reqs+1]
};

// OPTLearnedPositionalEmbedding.forward is literally `super().forward(positions
// + self.offset)` with offset == 2 (opt.py:64-68). The offset is applied HOST-side
// while building the step inputs (positions are already a host vector), so the
// embedding lookup itself is an unmodified vt::Embedding over the [P+2,H] table.
OPTStepInputs BuildOPTStepInputs(Dev d, const std::vector<int32_t>& positions,
                                 const CommonAttentionMetadata& am) {
  const int64_t T = static_cast<int64_t>(positions.size());
  std::vector<int32_t> pos_plus_offset(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    pos_plus_offset[static_cast<size_t>(i)] = positions[static_cast<size_t>(i)] + 2;
  return OPTStepInputs{
      DBuf(d, DType::kI32, {T}, pos_plus_offset.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
  };
}

// A biased projection: out = x @ W^T + bias (vLLM's `nn.Linear`-shaped
// Column/Row/QKVParallelLinear with bias=True). `bias` may be empty, in which
// case this is the plain MatmulBT every Qwen projection uses.
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

// One OPT self-attention block (opt.py::OPTAttention.forward :114-122). `dhn` is
// the LayerNormed hidden [T,H] bf16; returns the out_proj output [T,H] bf16.
//
// OPT is pre-GQA multi-head: num_key_value_heads == num_attention_heads, so the
// merged qkv is exactly [3*H, H] and q/k/v are three equal H-wide slices.
DBuf OPTAttnBlock(Dev d, const OPTAttnWeights& w, const HfConfig& cfg, const Tensor& dhn,
                  const OPTStepInputs& si, const CommonAttentionMetadata& meta,
                  const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  VT_CHECK(Hkv == Hq, "opt: multi-head attention only (OPT predates GQA)");
  VT_CHECK(Hq * Dh == H, "opt: num_attention_heads * head_dim must equal hidden_size");
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "opt: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "opt: KV cache head dims mismatch config");

  // ONE merged qkv GEMM + merged bias, then split — mirrors opt.py:118-119
  // (`qkv, _ = self.qkv_proj(hidden_states); q, k, v = qkv.chunk(3, dim=-1)`).
  // vt::QkvSplit does the chunk into DENSE per-shard buffers (a strided Slice
  // view over the merged [T,3H] would not satisfy ReshapeAndCache's contiguity
  // requirement).
  DBuf qkv = BiasedProj(d, w.qkv_proj, w.qkv_bias, dhn, T, 3 * H);
  DBuf qd(d, DType::kBF16, {T, Hq, Dh});
  DBuf kd(d, DType::kBF16, {T, Hkv, Dh});
  DBuf vd(d, DType::kBF16, {T, Hkv, Dh});
  vt::QkvSplit(d.q, qd.t(), kd.t(), vd.t(), qkv.t());

  // NO q/k norm and NO RoPE — the q/k that come out of the projection go
  // straight into the cache and the attention. This is the whole reason OPT
  // cannot reuse dense_attn::AttnBlock.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kd.t(), vd.t(), k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, DType::kBF16, {T, Hq, Dh});
  // scaling = head_dim ** -0.5 (opt.py:88) — the standard scale.
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attn.t(), qd.t(), k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  return BiasedProj(d, w.out_proj, w.out_bias, o_in, T, H);
}

// OPT's plain MLP (opt.py:188-190): fc1 + bias -> ReLU -> fc2 + bias. No gating.
DBuf OPTMlpBlock(Dev d, const OPTMlpWeights& w, const OPTConfigExtras& extras,
                 const HfConfig& cfg, const Tensor& dhn, int64_t T) {
  DBuf h = BiasedProj(d, w.fc1, w.fc1_bias, dhn, T, extras.ffn_dim);
  vt::Relu(d.q, h.t(), h.t());  // in place
  return BiasedProj(d, w.fc2, w.fc2_bias, h.t(), T, cfg.hidden_size);
}

// A LayerNorm over the [T,H] stream (weight+bias may be empty ==
// elementwise_affine=False).
void LayerNormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& weight,
                   const OwnedTensor& bias, const OPTConfigExtras& extras, int64_t H) {
  Tensor wt;
  Tensor bt;
  const bool affine = !weight.Empty();
  if (affine) {
    wt = ResidentWeight(d, weight, {H});
    bt = ResidentWeight(d, bias, {H});
  }
  vt::LayerNorm(d.q, out.t(), x, affine ? &wt : nullptr, affine ? &bt : nullptr,
                vt::LayerNormArgs{extras.layer_norm_eps});
}

// One OPT decoder layer (opt.py::OPTDecoderLayer.forward :168-195). `hidden`
// carries the full residual stream (OPT adds the residual explicitly rather than
// threading a separate accumulator the way the Qwen fused_add_rms_norm path
// does), so there is no `res` DBuf to maintain across layers.
void RunLayer(Dev d, const OPTLayerWeights& layer, const OPTConfigExtras& extras,
              const HfConfig& cfg, DBuf& hidden, const OPTStepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;

  // ---- self attention ------------------------------------------------------
  DBuf normed(d, DType::kBF16, {T, H});
  if (extras.do_layer_norm_before) {
    LayerNormInto(d, normed, hidden.t(), layer.self_attn_layer_norm,
                  layer.self_attn_layer_norm_bias, extras, H);
  } else {
    // POST-LN (350m): attention consumes the RAW residual stream.
    d.b.Copy(d.q, normed.ptr(), hidden.ptr(), hidden.bytes());
  }
  DBuf attn = OPTAttnBlock(d, layer.attn, cfg, normed.t(), si, meta, kv, T);
  vt::Add(d.q, attn.t(), attn.t(), hidden.t());  // residual + hidden_states
  if (extras.do_layer_norm_before) {
    hidden = std::move(attn);
  } else {
    LayerNormInto(d, hidden, attn.t(), layer.self_attn_layer_norm,
                  layer.self_attn_layer_norm_bias, extras, H);
  }

  // ---- fully connected -----------------------------------------------------
  DBuf normed2(d, DType::kBF16, {T, H});
  if (extras.do_layer_norm_before) {
    LayerNormInto(d, normed2, hidden.t(), layer.final_layer_norm,
                  layer.final_layer_norm_bias, extras, H);
  } else {
    d.b.Copy(d.q, normed2.ptr(), hidden.ptr(), hidden.bytes());
  }
  DBuf mlp = OPTMlpBlock(d, layer.mlp, extras, cfg, normed2.t(), T);
  vt::Add(d.q, mlp.t(), mlp.t(), hidden.t());
  if (extras.do_layer_norm_before) {
    hidden = std::move(mlp);
  } else {
    LayerNormInto(d, hidden, mlp.t(), layer.final_layer_norm, layer.final_layer_norm_bias,
                  extras, H);
  }
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`
// (identical to the qwen3.cpp helper — a device-side row gather).
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// embed(+positions) -> N layers -> final LayerNorm -> lm_head. Returns
// [n_out, vocab] f32 as a device DBuf (no host Download).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv, const OPTWeights& weights,
                 const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const OPTConfigExtras extras = GetOPTConfigExtras(config);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "opt: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "opt: one PagedKvCache per layer required");
  VT_CHECK(extras.word_embed_proj_dim == H,
           "opt: word_embed_proj_dim != hidden_size (project_in/project_out) is "
           "not supported — no checkpoint on the box to gate it (spec R2)");

  // hidden = embed_tokens[ids] + embed_positions[positions + 2]
  // (opt.py:274-279; project_in is a no-op when word_embed_proj_dim == H).
  OPTStepInputs si = BuildOPTStepInputs(d, positions, attn_meta);
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "opt embed");

    const int64_t p_rows = weights.embed_positions.shape[0];
    Tensor ptab = ResidentWeight(d, weights.embed_positions, {p_rows, H});
    DBuf pos_embeds(d, DType::kBF16, {T, H});
    vt::Embedding(d.q, pos_embeds.t(), ptab, si.pos_ids.t());
    vt::Add(d.q, hidden.t(), hidden.t(), pos_embeds.t());
  }

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], extras, config, hidden, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Decoder-level final LayerNorm (present only under the pre-LN placement
  // without _remove_final_layer_norm — opt.py:289-290).
  DBuf normed(d, DType::kBF16, {T, H});
  Tensor final_hidden = hidden.t();
  if (!weights.final_layer_norm.Empty()) {
    LayerNormInto(d, normed, hidden.t(), weights.final_layer_norm,
                  weights.final_layer_norm_bias, extras, H);
    final_hidden = normed.t();
  }

  // lm_head. Tied (the OPT default): logits = hidden @ embed_tokens^T via
  // MatmulBT over the [vocab,H] embed table. Untied: the loaded Matmul-B
  // [H,vocab] lm_head via vt::Matmul.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = final_hidden;
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), final_hidden, logits_indices, H);
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

ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  // The pool block's lifetime moves into a shared_ptr whose deleter returns it
  // to the DevicePool (mirrors qwen3.cpp/qwen3_5.cpp WrapDeviceLogits).
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

}  // namespace

std::vector<float> OPTModel::Forward(const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const CommonAttentionMetadata& attn_meta,
                                     const std::vector<PagedKvCache>& attn_kv,
                                     const OPTWeights& weights, const HfConfig& config,
                                     vt::Queue& queue,
                                     const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits OPTModel::ForwardDevice(const std::vector<int32_t>& token_ids,
                                      const std::vector<int32_t>& positions,
                                      const CommonAttentionMetadata& attn_meta,
                                      const std::vector<PagedKvCache>& attn_kv,
                                      const OPTWeights& weights, const HfConfig& config,
                                      vt::Queue& queue,
                                      const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
