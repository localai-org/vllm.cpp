// MiniCPM3 (`MiniCPM3ForCausalLM`, MiniCPM3-4B) forward — the MiniCPM scalar
// dense skeleton (scale_emb / scaled residual add / dim_model_base logit scale)
// with its attention swapped to the landed DeepSeek-V2 **MLA** block. ZERO new
// kernel: it composes mla::ForwardMlaAttentionBlock (which itself composes the
// MLA cache write, the absorbed MQA decode and the materialized-MHA chunked
// prefill) with the MiniCPM scalar wiring reused verbatim from minicpm.cpp.
//
// Grounding (@ pin e24d1b24), file:line on both sides:
//   ForwardBody / layer loop   <- minicpm3.py:207-233 (MiniCPM3Model, inheriting
//                                 MiniCPMModel) + minicpm.py:430-450
//   RunLayer                   <- minicpm.py:378-394 MiniCPMDecoderLayer.forward
//                                 (input_layernorm -> self_attn -> scaled add ->
//                                  post_attention_layernorm -> mlp -> scaled add),
//                                 with self_attn the MLA block (minicpm3.py:136-183)
//   the three scalars          <- minicpm.py:441-443 (scale_emb), :384-393
//                                 (scale_depth/sqrt(L)), :604,633,640 (scale_width)
//   BuildMlaStep               <- mla_attention.py:1652-1830 (non-DCP), the SAME
//                                 metadata build DeepSeek-V2 runs; reused via the
//                                 shared BuildMlaBatchSplit + mla:: chunked helpers.
//
// This is the EAGER MLA forward (no decode CUDA-graph driver yet — SPEED PENDING;
// the correctness gate runs eager). The MLA block, the batch-split ordering
// invariant and the chunked-context loop are all the landed DeepSeek-V2 code.
#include "vllm/model_executor/models/minicpm3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/mla_chunked_context.h"
#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/deepseek_v2.h"       // BuildMlaBatchSplit/MlaBatchSplit
#include "vllm/model_executor/models/device_pool.h"       // Pool()
#include "vllm/model_executor/models/mla_attention.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

using namespace dense_attn;  // Dev / DBuf / MakeTensor / Reshape / ResidentWeight

// ─── per-step device inputs the MLA block needs ─────────────────────────────
// Mirrors deepseek_v2.cpp's MlaStep, but eager-only (every upload is a plain
// UploadInto — there is no CUDA-graph capture here, so the UploadRange
// dangling-source contract does not apply).
struct MlaStep {
  std::vector<DBuf> owned;
  Tensor positions;
  Tensor slot_mapping;
  mla::MlaBlockMetadata meta;
  MlaBatchSplit split;
  const Tensor* rope_cache = nullptr;
};

template <typename T>
Tensor UploadInto(Dev d, std::vector<DBuf>& owned, DType dt,
                  const std::vector<int64_t>& shape, const std::vector<T>& host) {
  owned.emplace_back(d, dt, shape, host.data());
  return owned.back().t();
}

// `MLACommonMetadataBuilder.build` (mla_attention.py:1652-1830), non-DCP branch —
// the same construction deepseek_v2.cpp::BuildMlaStep runs, reusing the shared
// BuildMlaBatchSplit ordering invariant and the mla:: chunked-context helpers.
MlaStep BuildMlaStep(Dev d, const std::vector<int32_t>& positions,
                     const CommonAttentionMetadata& am, int64_t block_size,
                     int64_t max_model_len) {
  MlaStep s;
  s.split = BuildMlaBatchSplit(am);
  const MlaBatchSplit& sp = s.split;
  const int64_t T = static_cast<int64_t>(positions.size());

  s.positions = UploadInto(d, s.owned, DType::kI32, {T}, positions);
  s.slot_mapping = UploadInto(d, s.owned, DType::kI64, {T}, am.slot_mapping);

  s.meta.num_decode_tokens = sp.num_decode_tokens;
  const int64_t cols = am.block_table_num_cols;

  // --- decode half (triton_mla.py:214-216, 245-246) ---
  if (sp.num_decodes > 0) {
    std::vector<int32_t> bt(am.block_table_tensor.begin(),
                            am.block_table_tensor.begin() +
                                static_cast<size_t>(sp.num_decodes * cols));
    std::vector<int32_t> sl(am.seq_lens.begin(),
                            am.seq_lens.begin() + sp.num_decodes);
    s.meta.decode.block_table =
        UploadInto(d, s.owned, DType::kI32, {sp.num_decodes, cols}, bt);
    s.meta.decode.seq_lens = UploadInto(d, s.owned, DType::kI32, {sp.num_decodes}, sl);
    s.meta.decode.max_seq_len = sp.decode_max_seq_len;
  }

  // --- prefill half (mla_attention.py:1652-1682) ---
  if (sp.num_prefills > 0) {
    s.meta.prefill_cu_seqlens_q = UploadInto(d, s.owned, DType::kI32,
                                             {sp.num_prefills + 1},
                                             sp.prefill_cu_seqlens_q);
    std::vector<int32_t> pbt(
        am.block_table_tensor.begin() + static_cast<size_t>(sp.num_decodes * cols),
        am.block_table_tensor.begin() +
            static_cast<size_t>((sp.num_decodes + sp.num_prefills) * cols));
    s.meta.prefill_block_table =
        UploadInto(d, s.owned, DType::kI32, {sp.num_prefills, cols}, pbt);
    s.meta.max_query_len = sp.prefill_max_query_len;

    if (sp.num_prefills_with_context > 0) {
      const int64_t workspace = mla::DetermineChunkedPrefillWorkspaceSize(
          max_model_len, am.num_reqs, block_size);
      const mla::MlaChunkedContextMetadata cm = mla::BuildMlaChunkedContext(
          sp.prefill_context_lens, sp.prefill_cu_seqlens_q, workspace, block_size);
      s.meta.prefill_tokens_with_context = cm.prefill_tokens_with_context;
      s.meta.chunk_workspace_tokens = workspace;
      const int64_t np = cm.num_prefills;
      const int32_t row = std::max<int32_t>(cm.max_token_num_over_chunk, 1);
      for (int32_t i = 0; i < cm.num_chunks; ++i) {
        const std::vector<int32_t> cu(
            cm.cu_seq_lens.begin() + static_cast<size_t>(i) * (np + 1),
            cm.cu_seq_lens.begin() + static_cast<size_t>(i + 1) * (np + 1));
        const std::vector<int32_t> starts(
            cm.starts.begin() + static_cast<size_t>(i) * np,
            cm.starts.begin() + static_cast<size_t>(i + 1) * np);
        const std::vector<int32_t> t2s(
            cm.token_to_seq.begin() + static_cast<size_t>(i) * row,
            cm.token_to_seq.begin() + static_cast<size_t>(i + 1) * row);
        mla::MlaChunkDeviceMetadata cd;
        cd.cu_seq_lens = UploadInto(d, s.owned, DType::kI32, {np + 1}, cu);
        cd.starts = UploadInto(d, s.owned, DType::kI32, {np}, starts);
        cd.token_to_seq = UploadInto(d, s.owned, DType::kI32, {row}, t2s);
        cd.total_tokens = cm.chunk_total_token[static_cast<size_t>(i)];
        cd.max_seq_len = cm.max_seq_lens[static_cast<size_t>(i)];
        s.meta.chunks.push_back(cd);
      }
    }
  }
  return s;
}

// The MLA block's device-resident weight views for one layer (q_lora branch only).
mla::MlaBlockWeights ResidentMla(Dev d, const MiniCPM3MlaWeights& w,
                                 const mla::MlaBlockDims& dm,
                                 const Tensor& rope_cache) {
  mla::MlaBlockWeights m;
  m.fused_qkv_a_proj = ResidentWeight(d, w.fused_qkv_a_proj);
  m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm, {dm.q_lora_rank});
  m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm, {dm.kv_lora_rank});
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  m.rope_cos_sin_cache = rope_cache;
  return m;
}

// MiniCPM3 dense SwiGLU MLP (minicpm.py::MiniCPMMLP): merged gate_up MatmulBT ->
// SiluAndMul -> down MatmulBT.
DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const Tensor& h, int64_t T,
              int64_t H, int64_t I) {
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran — the seam's Apply IS {ResidentWeight; MatmulBT[2I,H];
  // SiluAndMul} with M = x.shape[0] == T. MiniCPM3 inherits MiniCPMMLP verbatim
  // upstream (minicpm3.py:186 -> minicpm.py:193), i.e. one
  // MergedColumnParallelLinear + SiluAndMul; the MLA attention half is untouched.
  // Direct Unquantized arm, not the factory: no *_fp4 is ever populated here
  // (spec §Port map). (FUSION-DENSE-MIGRATE.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, h);
  Tensor wd = ResidentWeight(d, w.down_proj);  // [H, I]
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wd);
  return out;
}

// One MiniCPM3 decoder layer (minicpm.py:378-394, self_attn = the MLA block).
// The scaled residual add (scale_depth/sqrt(L)) is a MulScalar into scratch +
// Add — the fused add+RMSNorm form is NOT usable (scale_depth forbids it), same
// as MiniCPM.
void RunLayer(Dev d, const MiniCPM3LayerWeights& layer, const MiniCPM3Params& p,
              double residual_scale, DBuf& res, const MlaStep& step,
              Tensor& kv_cache, v1::TritonMLAImpl& impl, int64_t T) {
  const int64_t H = p.hidden_size;
  const float eps = p.rms_norm_eps;

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf normed(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed.t(), res.t(), w_in, vt::RmsNormArgs{eps, false});

  DBuf attn(d, DType::kBF16, {T, H});
  Tensor attn_t = attn.t();
  const mla::MlaBlockWeights mw = ResidentMla(d, layer.attn, p.mla, *step.rope_cache);
  mla::ForwardMlaAttentionBlock(d, p.mla, mw, normed.t(), step.positions, kv_cache,
                                step.slot_mapping, step.meta, impl, attn_t);
  DBuf scaled(d, DType::kBF16, {T, H});
  vt::MulScalar(d.q, scaled.t(), attn.t(), residual_scale);
  vt::Add(d.q, res.t(), res.t(), scaled.t());

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf normed2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed2.t(), res.t(), w_post, vt::RmsNormArgs{eps, false});

  DBuf mlp = MlpBlock(d, layer.mlp, normed2.t(), T, H, p.intermediate_size);
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
                 const CommonAttentionMetadata& am,
                 const std::vector<PagedKvCache>& attn_kv,
                 const MiniCPM3Weights& weights,
                 const std::vector<int32_t>& logits_indices) {
  const MiniCPM3Params& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t vocab = p.vocab_size;
  const float eps = p.rms_norm_eps;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "minicpm3: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(p.num_hidden_layers),
           "minicpm3: one MLA PagedKvCache per layer required");
  VT_CHECK(T > 0, "minicpm3: empty batch");

  const double residual_scale = p.residual_scale();
  const double scale_width = p.scale_width();

  // Embed then scale by scale_emb (minicpm.py:441-443).
  DBuf res(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, res, token_ids, weights.embed_tokens, vocab, H,
                "minicpm3 embed");
  }
  vt::MulScalar(d.q, res.t(), res.t(), p.scale_emb);

  const int64_t block_size = attn_kv[0].block_size;
  MlaStep step = BuildMlaStep(d, positions, am, block_size, p.max_position_embeddings);
  const Tensor rope = ResidentWeight(d, weights.rope_cos_sin_cache);
  step.rope_cache = &rope;

  v1::TritonMLAImpl impl;
  const int64_t head_size = p.mla.head_size();
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == head_size,
             "minicpm3: the MLA cache must be 1-head, "
             "kv_lora_rank + qk_rope_head_dim wide (MLAAttentionSpec)");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, head_size});
    RunLayer(d, weights.layers[static_cast<size_t>(l)], p, residual_scale, res, step,
             kv_cache, impl, T);
  }

  // Final RMSNorm then hidden /= scale_width before the tied lm_head
  // (minicpm.py:633,640).
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), res.t(), w_fn, vt::RmsNormArgs{eps, false});
  if (scale_width != 1.0 && scale_width > 0.0)
    vt::MulScalar(d.q, dnorm.t(), dnorm.t(), 1.0 / scale_width);

  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
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

ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

}  // namespace

std::vector<float> MiniCPM3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MiniCPM3Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * weights.params.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits MiniCPM3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MiniCPM3Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

}  // namespace vllm
