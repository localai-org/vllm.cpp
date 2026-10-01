// Qwen3 DENSE (`Qwen3ForCausalLM`) forward — the first ADDITIVE-MODEL bring-up
// W3 (the capstone). A pure standard-dense transformer forward COMPOSED from the
// public vt:: ops + the fusion catalog (kFusedAddRmsNormStd / kAttnQkNormRope,
// include/vt/recipes.h), with NO GDN, NO MoE and NO attention output gate. It is
// the Qwen3.6-dense full-attention path (qwen3_5.cpp DenseForwardLayers /
// FullAttnBlockPaged) stripped to the pure-dense subset:
//   - ONE full-attention KV group per layer (no MambaSpec/GDN, no hybrid split);
//   - STANDARD (non-gemma) RMSNorm at input/post/final norms;
//   - per-head q_norm/k_norm (RMSNorm(head_dim), non-gemma) applied BEFORE RoPE;
//   - NO attention gate (Qwen3 has none);
//   - a TIED lm_head (aliases embed_tokens).
//
// Grounding: vllm/model_executor/models/qwen3.py @ e24d1b24
//   Qwen3Attention (:65-168), Qwen3MLP=Qwen2MLP (:58), Qwen3DecoderLayer
//   (:171-242), Qwen3Model=Qwen2Model (:260), tied lm_head (:294-295).
// See .agents/specs/first-additive-model-qwen3-dense.md §2/§4/§6.
//
// Numeric contract (mirrors the qwen3_5 full-attention FALLBACK path — the
// token-exact paged==dense anchor): the residual stream is the model dtype (bf16,
// matching vLLM's fused_add_rms_norm residual); the qkv GEMM emits f32 q/k/v so
// the per-head q/k RMSNorm + RoPE run in f32; the paged KV cache is written bf16
// (down-cast K/V) while the query stays f32 into vt::PagedAttention; o_proj and
// the whole MLP flow bf16. Returns [n_out, vocab] f32 logits.
//
// Self-contained device glue (Dev/DBuf/ResidentWeight): the DBuf here draws its
// scratch from the SHARED DevicePool (include/vllm/model_executor/models/
// device_pool.h — extracted verbatim from qwen3_5.cpp), so the dense forward
// reuses freed blocks instead of a per-op cudaMalloc/cudaFree. This is a pure
// allocation-source change (identical computation ⇒ byte-identical output; all
// gate models unchanged). NOTE: a clean same-binary A/B (Qwen3-4B c1+c8)
// measured the pool PERF-NEUTRAL on this model — the async scheduler already
// overlaps the host-side alloc syncs with GPU compute; it is kept as byte-safe
// hygiene + code sharing, not a measured TTFT lever. The real dense-TTFT lever
// is the RoPE cos|sin cache below.
#include <chrono>

#include "vllm/model_executor/models/qwen3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/quantization/exl3.h"
#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // LinearMethod seam
#include "vllm/model_executor/models/decode_graph_sizes.h"  // DecodeGraphSizes/PadToCaptureSize
#include "vllm/model_executor/models/dense_attn_block.h"  // shared AttnBlock + device glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // NVFP4 W4A16 dispatch
#include "vllm/model_executor/models/device_pool.h"     // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIds seam
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vllm/model_executor/models/dense_attn_graph_break.h"  // CopyOutput(optional<DBuf>)
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK: the break-point seam
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"  // WarmRopeCosSin (item-5 TT-only)
#include "vt/recipes.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// The dense self-attention block + all its device glue (Dev/DBuf/pool policy,
// ResidentWeight[F32]/WeightF32, KvSlice, StepInputs/BuildStepInputs, the
// env-flag readers and AttnBlock itself) were EXTRACTED VERBATIM to the shared
// header include/vllm/model_executor/models/dense_attn_block.h so the first
// full-attention MoE (Qwen3-Coder `Qwen3MoeForCausalLM`, qwen3_moe.cpp W3)
// reuses the exact same attention preamble. This is a PURE RELOCATION: the
// definitions are byte-for-byte the same and the dense-only MLP / decoder-layer
// / forward-body machinery below composes them via `using namespace dense_attn`,
// so the Qwen3-dense (0.6B/4B) forward is byte-identical (same vt:: op order).
using namespace dense_attn;

// TT-only dump. The getenv is paid only on kTENSTORRENT so a CUDA replay
// step does not walk the environment twice.
bool TtDumpKv(const Dev& d) {
  return d.q.device.type == vt::DeviceType::kTENSTORRENT &&
         std::getenv("VT_TT_DUMP_KV") != nullptr;
}

// Near-tie adjudication for the [TT-DUMP-LOGITS] prints: the top-2 RAW-logit
// gap IS the top-2 logprob gap in nats (softmax is monotone), so an argmax
// flip between two arms whose own top-2 gaps are both tiny is a bf16 near-tie
// resolution difference, not a forward divergence — the same bar
// scripts/qwen3-neartie-gap.py applies (gap <= ~0.5 nats = structurally
// correct). Prints the pair as `top2=[id1:v1 id2:v2] gap2=g`.
void TtDumpTop2(const float* v, int64_t n, char* out, size_t out_n) {
  int64_t i1 = 0, i2 = -1;
  float v1 = v[0], v2 = -1.0e30f;
  for (int64_t i = 1; i < n; ++i) {
    if (v[i] > v1) {
      i2 = i1; v2 = v1; i1 = i; v1 = v[i];
    } else if (v[i] > v2) {
      i2 = i; v2 = v[i];
    }
  }
  if (i2 < 0) i2 = 0;  // n==1 degenerate
  std::snprintf(out, out_n, "top2=[%lld:%.6f %lld:%.6f] gap2=%.6g",
                (long long)i1, static_cast<double>(v1),
                (long long)i2, static_cast<double>(v2),
                static_cast<double>(v1 - v2));
}

// Dense SwiGLU MLP (qwen3.py::Qwen3MLP=Qwen2MLP): merged gate_up_proj ->
// SiluAndMul -> down_proj. `dh2` is the post-norm hidden [T,H] bf16.
//
// Routed through the LinearMethod seam (S4): the gate_up+SiluAndMul and the
// down projection each go through a method chosen ONCE by the checkpoint's
// scheme (bf16 UnquantizedLinearMethod vs NVFP4 W4A16), so this forward no
// longer carries the `IsNvfp4()` probe, the `device == kCUDA` gate or the
// `#ifdef VT_MARLIN_NVFP4` fused-path dispatch — they live in the shared
// quantization scheme headers. Byte-identical: the methods run the exact same
// vt:: ops in the same order the inline path did.
DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
              const Tensor& dh2, int64_t /*T*/,
              const TensorParallel* tp = nullptr) {
  const int64_t I = cfg.intermediate_size;
  // gate_up is a MergedColumnParallelLinear (sharded on I, no comm); down is a
  // RowParallelLinear whose per-rank partial [T,H] products are all-reduced below
  // (linear.py:1766). tp_size==1 ⇒ whole tensors + the all-reduce is a no-op, so
  // this is byte-identical to the single-GPU MLP.
  // QUANT-EXL3 (#2181): the scheme is chosen ONCE from the populated weights,
  // by the same factory shape the fp4 arm uses. Exactly one of {bf16, fp4,
  // exl3} is populated per layer.
  auto gate_up =
      w.IsExl3() ? layers::MakeMlpGateUpMethod(w.gate_up_proj, w.gate_proj_exl3,
                                               w.up_proj_exl3, I)
                 : layers::MakeMlpGateUpMethod(w.gate_up_proj, w.gate_proj_fp4,
                                               w.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, dh2);
  auto down = w.IsExl3() ? layers::MakeLinearMethod(w.down_proj, w.down_proj_exl3)
                         : layers::MakeLinearMethod(w.down_proj, w.down_proj_fp4);
  DBuf out = down->Apply(d, act.t(), DType::kBF16);
  Tensor ot = out.t();
  TpAllReduceSum(tp, d.q, ot);
  return out;
}

// One dense decoder layer (qwen3.py::Qwen3DecoderLayer): input norm (std
// add+RMSNorm) -> attention -> post norm (std add+RMSNorm) -> MLP. `hidden` (bf16
// [T,H]) is the delta; `res` (bf16 [T,H]) the residual accumulator.
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T,
              const TensorParallel* tp = nullptr) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // ENG-CUDAGRAPH-BREAK (#1163) W1 (#1192): THE DENSE ATTENTION ENTRY IS A BREAK
  // POINT. The boundary is vLLM's, not ours to invent: its v1 default splits at
  // `splitting_ops`, defaulted to the attention family
  // (`vllm/config/compilation.py:517,764-772,1145` @ pin `5559679229`). The
  // registration form is SGLang's, because vLLM gets its split from Dynamo and
  // FX and we have no compiler: one line at the site, exactly as
  // `layers/radix_attention.py:256` @ `f63458b5be`. THE SITE IS THE
  // REGISTRATION.
  //
  // Destination form, because `AttnBlock` returns a FRESH pooled buffer on every
  // call. The destination is a `vt::BreakSlot`, not a local `std::optional<DBuf>`:
  // the following segment bakes the destination's address, and a plain local
  // dies on this function's `return` while the pooled block it named goes back
  // on the `DevicePool` free list — a host use-after-scope plus D1's reuse
  // hazard, at the one site that has to obey the rule. The slot hands its
  // storage to the seam on the capturing path and keeps it inline on the
  // pass-through path.
  //
  // Outside a capture scope — which is every production step today, until W2
  // migrates this model's decode driver onto the seam — `GraphBreak` moves the
  // result into the slot and returns, so this is byte-identical to the
  // `DBuf attn = AttnBlock(...)` it replaces, allocates nothing extra, and makes
  // zero backend calls.
  vt::BreakSlot<std::optional<DBuf>> attn;
  vt::GraphBreak([&] { return AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T, tp); },
                 attn);
  DBuf& attn_buf = attn->value();

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn_buf.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn_buf.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  hidden = MlpBlock(d, layer.mlp, cfg, dh2.t(), T, tp);
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

// Embed: hidden[T,H] bf16 = embed_tokens[token_ids]. KEPT OUTSIDE THE
// CUDA-GRAPH (mirrors qwen3_moe.cpp / qwen3_5.cpp EmbedInto): the CUDA Embedding
// op allocates a device bounds-check flag (cudaMalloc/cudaFree) and syncs the
// stream, both illegal inside a capture region — and it consumes the HOST
// token_ids. The graph driver runs this per step into its PERSISTENT hidden buffer,
// then captures/replays ForwardLayers over that fixed hidden address. The table
// residency and the async device-id override are `EmbedGather`'s.
void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const Qwen3DenseWeights& weights, const HfConfig& config) {
  EmbedGather(d, hidden, token_ids, weights.embed_tokens, config.vocab_size,
              config.hidden_size, "qwen3 dense embed");
}

// The CAPTURABLE region: everything AFTER the embedding — the residual stream
// (res=0), the N dense decoder layers, the final RMSNorm and the (tied/untied)
// lm_head — returning [n_out, vocab] f32 as a device DBuf (no host Download). Split
// out of the eager forward body so the exact op sequence is what the decode graph
// captures/replays; every per-step-varying input is read from a HOST vector
// argument (positions / the attention-metadata vectors, via BuildStepInputs) whose
// host->device copies are capturable on GB10, and which the graph driver keeps
// persistent + mutates in place so a replay picks up the new step's inputs.
//
// `hidden_in` is the embedded input (a view over the graph's persistent hidden
// buffer on the replay path). It is COPIED into a working buffer so the per-layer
// `hidden` DBuf reassignment (RunLayer's `hidden = MlpBlock(...)`) never disturbs
// the persistent embedding — the copy is a pure device->device data move, so the
// layer sequence and its output are BYTE-IDENTICAL to the pre-split forward.
// `return_hidden` (ARCH-ONE-SURFACE ROW 6, default false = byte-identical
// text path): when true, STOP after the final RMSNorm (+ the logits_indices
// gather) and return the [n_out, H] hidden rows upcast to f32 — the pooling
// forward of an embedding conversion, whose model has NO lm_head at all
// (adapters.py:135-151: as_embedding_model replaces the output layer with a
// missing-layer stage; the pooler consumes the post-final-norm hidden). Every
// existing caller leaves the default, so the lm_head tail is untouched.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& attn_meta,
                   const std::vector<PagedKvCache>& attn_kv,
                   const Qwen3DenseWeights& weights, const HfConfig& config,
                   const std::vector<int32_t>& logits_indices,
                   bool return_hidden = false,
                   std::optional<DBuf>* out_hidden = nullptr) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 dense: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 dense: one PagedKvCache per layer required");

  // Working copy of the embedded hidden (device->device; captured). RunLayer
  // reassigns `hidden` per layer, so it must NOT alias the persistent buffer.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16, do_gather ? std::vector<int64_t>{
                                                static_cast<int64_t>(logits_indices.size()), H}
                                          : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  // ARCH-ONE-SURFACE ROW 6 pooling tail: the post-final-norm hidden rows,
  // upcast bf16 -> f32 (vt::CastF32), with NO lm_head — an embedding-converted
  // checkpoint has no output layer to multiply by. Never taken by any text
  // caller (return_hidden defaults false).
  if (return_hidden) {
    DBuf dhid(d, DType::kF32, {n_out, H});
    vt::CastF32(d.q, dhid.t(), src);
    return dhid;
  }

  // MODEL-MUSIC-MUSIC3 W2: the SAME post-final-norm rows as the pooling tail
  // above, emitted ALONGSIDE the logits instead of instead of them. MiniMax-
  // Music3's autoregressive stage needs both from ONE forward — it reads
  // `output.last_hidden_state[:, -1]` and then applies `lm_head` to that very
  // row (encoders.py:311-313, :318, :353) — and running the 8.6B stack twice to
  // get the two halves is the alternative this branch removes.
  //
  // `nullptr` (every existing caller, including the decode-graph capture) adds
  // NO op: the branch is not taken, so the captured sequence is byte-identical.
  if (out_hidden != nullptr) {
    DBuf dhid(d, DType::kF32, {n_out, H});
    vt::CastF32(d.q, dhid.t(), src);
    *out_hidden = std::move(dhid);
  }

  // lm_head. Tied (Qwen3-0.6B): logits = hidden @ embed_tokens^T via MatmulBT
  // over the [vocab,H] embed table (== [N=vocab,K=H]). Untied: the loaded
  // Matmul-B [H,vocab] lm_head via vt::Matmul.
  // QUANT-EXL3 (#2181): an EXL3 checkpoint ships a REAL quantized head and does
  // not tie it. Its width is its own -- 6-bit against a 3-bit body in the
  // published 3.0bpw quant -- which is why nothing here reads a config scalar.
  if (!weights.lm_head_exl3.Empty()) {
    return dense_attn::Exl3MatmulD(d, src, weights.lm_head_exl3, DType::kF32);
  }
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

// Full eager forward body: embed (host token_ids) then the capturable layer region.
// Used by Qwen3DenseModel::Forward/ForwardDevice and by the graph driver's eager
// fallback / cold-size pre-warm step (one contiguous stream, no capture). Byte-
// identical op sequence to the graph (eager output == replay output).
//
// `inputs_embeds_bf16` (MODEL-MUSIC-MUSIC3 W2) is the ADDITIVE `inputs_embeds`
// entry, mirroring what the Qwen3 family already has on its multimodal siblings
// (qwen3_vl.h:145,159 `inputs_embeds_bf16`; gemma4.cpp:400-447
// `inputs_embeds_override`; muse_glimmer.cpp:346-381): the caller has already
// built the [T, H] bf16 hidden rows and the stream STARTS from them, so
// `EmbedInto` — and with it the token-id lookup — is skipped entirely. Upstream
// transformers exposes exactly this door (`Qwen3Model.forward(inputs_embeds=)`,
// which encoders.py:311 and :353 are the only Music3 callers of), and it is the
// door a continuous frame-feedback embedding needs because it corresponds to no
// vocabulary row. NULL on every existing caller ⇒ the token path is untouched.
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3DenseWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices,
                 bool return_hidden = false,
                 const std::vector<uint16_t>* inputs_embeds_bf16 = nullptr,
                 std::optional<DBuf>* out_hidden = nullptr) {
  const int64_t H = config.hidden_size;
  const int64_t T = inputs_embeds_bf16 != nullptr
                        ? static_cast<int64_t>(inputs_embeds_bf16->size()) / H
                        : static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "qwen3 dense: a forward needs at least one input row");
  VT_CHECK(inputs_embeds_bf16 == nullptr ||
               static_cast<int64_t>(inputs_embeds_bf16->size()) == T * H,
           "qwen3 dense: inputs_embeds must be [num_tokens, hidden_size]");
  DBuf hidden(d, DType::kBF16, {T, H});
  if (inputs_embeds_bf16 != nullptr) {
    // The rows ARE the embedding; nothing scales or norms them here, exactly as
    // `Qwen3Model.forward` assigns `inputs_embeds` straight through.
    d.b.Copy(d.q, hidden.ptr(), inputs_embeds_bf16->data(),
             static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
  } else {
    EmbedInto(d, hidden, token_ids, weights, config);
  }
  return ForwardLayers(d, hidden.t(), positions, attn_meta, attn_kv, weights, config,
                       logits_indices, return_hidden, out_hidden);
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  // The pool block's lifetime moves into a shared_ptr whose deleter returns it to
  // the DevicePool — no per-step cudaMalloc/cudaFree, and the buffer safely
  // outlives sampling (mirrors qwen3_5.cpp WrapDeviceLogits).
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

// NON-OWNING [rows, vocab] f32 view over a buffer the graph slot keeps alive
// (mirrors qwen3_moe.cpp / qwen3_5.cpp ViewDeviceLogits). Stream ordering
// guarantees the sampler's later reads see the replay's writes; the next same-size
// replay overwrites the buffer, so in-place sampler mutation is safe.
ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                               int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  return fl;
}

// Overwrite dst's CONTENTS from src WITHOUT changing dst.data() when the sizes
// already match (preserves the fixed address a captured host->device copy reads
// from); reallocate only when the shape actually changed (qwen3_moe.cpp CopyInPlace).
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// ATTENTION-ONLY analogue of qwen3_5.cpp's BuildPaddedDecode (pure dense has no GDN
// metadata) — byte-for-byte the qwen3_moe.cpp BuildPaddedDecodeAttn.
//
// The decode forward is ROW-INDEPENDENT (paged attention is per-request causal; the
// norm / SwiGLU MLP / lm_head are per-token with no cross-row reduction), so
// appending S-B INERT rows cannot perturb the real rows' logits. The padding rows
// are made inert exactly as vLLM's cudagraph padding:
//   * token id / position 0 (the embed row is discarded);
//   * slot_mapping = -1 -> ReshapeAndCache skips the KV write, so no real KV block
//     is touched;
//   * seq_lens = 1 + block_table row 0 -> paged attention does a valid in-bounds
//     read of block 0 whose output row is discarded (never returned).
// The real prefix [0,B) is copied verbatim, so at S==B this is a bit-identical
// rebuild of the eager inputs.
void BuildPaddedDecodeAttn(int64_t S, const std::vector<int32_t>& tok,
                           const std::vector<int32_t>& pos,
                           const CommonAttentionMetadata& am,
                           std::vector<int32_t>& tok_out,
                           std::vector<int32_t>& pos_out,
                           CommonAttentionMetadata& am_out) {
  const int64_t cols = am.block_table_num_cols;

  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  // W10 (#1857): a pure-decode rewrite is never spec-classified. Belt on the
  // vt shape guard's braces (S == q*S only at q == 1).
  am_out.uniform_spec_query_len = 0;
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);
}

}  // namespace

std::vector<float> Qwen3DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  if (TtDumpKv(d)) {
    int argmax = 0;
    for (int64_t i = 1; i < n_out * config.vocab_size; ++i)
      if (logits[static_cast<size_t>(i)] > logits[static_cast<size_t>(argmax)])
        argmax = static_cast<int>(i);
    char top2[96];
    TtDumpTop2(logits.data(), n_out * config.vocab_size, top2, sizeof(top2));
    fprintf(stderr, "[TT-DUMP-LOGITS] Forward eager argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
            argmax, logits[0], logits[1], logits[2], logits[3], logits[4], top2);
  }
  return logits;
}

ForwardLogits Qwen3DenseModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  if (TtDumpKv(d)) {
    std::vector<float> logits_dump(static_cast<size_t>(n_out * config.vocab_size));
    dlogits.Download(d, logits_dump.data());
    int argmax = 0;
    for (size_t i = 1; i < logits_dump.size(); ++i)
      if (logits_dump[i] > logits_dump[static_cast<size_t>(argmax)])
        argmax = static_cast<int>(i);
    char top2[96];
    TtDumpTop2(logits_dump.data(), static_cast<int64_t>(logits_dump.size()), top2, sizeof(top2));
    fprintf(stderr, "[TT-DUMP-LOGITS] ForwardDevice eager argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
            argmax, logits_dump[0], logits_dump[1], logits_dump[2], logits_dump[3], logits_dump[4], top2);
  }
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

ForwardLogits Qwen3DenseModel::ForwardHidden(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  // ARCH-ONE-SURFACE ROW 6: the POOLING forward — the same embed + layer stack
  // as Forward/ForwardDevice, stopping after the final RMSNorm (+ gather) with
  // NO lm_head, mirroring an as_embedding_model conversion whose output layer
  // is a missing-layer stage (adapters.py:135-151). The [n_out, H] f32 rows are
  // downloaded to the host carrier: the landed pooler ops are host-side, and an
  // embedding batch is one prefill (no per-step decode loop to keep resident).
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dhidden = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, /*return_hidden=*/true);
  const int64_t n_out = dhidden.t().shape[0];
  const int64_t H = config.hidden_size;
  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = H;  // the carrier's row width IS the hidden size on this path
  fl.host.resize(static_cast<size_t>(n_out) * static_cast<size_t>(H));
  dhidden.Download(d, fl.host.data());
  return fl;
}

std::vector<float> Qwen3DenseModel::ForwardEmbeds(
    const std::vector<uint16_t>& inputs_embeds_bf16, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices, std::vector<float>* out_hidden) {
  // MODEL-MUSIC-MUSIC3 W2. The layer stack, the final RMSNorm, the gather and
  // the lm_head are the SAME ones Forward runs — only the first step differs,
  // and it differs exactly as `Qwen3Model.forward(inputs_embeds=...)` does.
  Dev d{vt::GetBackend(queue.device.type), queue};
  std::optional<DBuf> dhidden;
  DBuf dlogits = ForwardBody(d, /*token_ids=*/{}, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, /*return_hidden=*/false,
                             &inputs_embeds_bf16, out_hidden != nullptr ? &dhidden : nullptr);
  const int64_t n_out = dlogits.t().shape[0];
  if (out_hidden != nullptr) {
    VT_CHECK(dhidden.has_value(), "qwen3 dense: the hidden rows were not produced");
    out_hidden->resize(static_cast<size_t>(n_out) * static_cast<size_t>(config.hidden_size));
    dhidden->Download(d, out_hidden->data());
  }
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

// ─── Qwen3DenseDecodeGraph (shared pure-dense decode CUDA-graph driver) ───────
// The pure-dense sibling of Qwen3MoeDecodeGraph (qwen3_moe.cpp) — SAME cold ->
// warm -> capture -> replay state machine, SAME padded-batch capture set
// (decode_graph_sizes.h) and SAME persistent fixed-address host inputs + persistent
// embed/logits buffers, driving the dense forward (ForwardLayers over EmbedInto)
// with a dense SwiGLU MLP instead of the MoE block. NO GDN (attention-only).
//
// Ported from: vllm/v1/worker/gpu_model_runner.py::GPUModelRunner @ e24d1b24
//   (`_dummy_run` warm-up then capture, then graph dispatch per decode step) +
//   vllm/compilation/cuda_graph.py (`CUDAGraphWrapper.__call__`: pad the batch to a
//   captured size, replay, else run eager).
//
// GRAPH-SAFETY AUDIT of the bf16 dense decode path (capture requires stable pointers
// and no host sync / stream-ordered alloc inside the region) — identical to the
// already-shipped Qwen3MoeDecodeGraph (its d128 full-attention capture path IS this
// one, minus the MoE-only scratch):
//   * Embedding (device flag cudaMalloc + stream sync) stays OUTSIDE (EmbedInto).
//   * All device scratch comes from the shared DevicePool, whose blocks are recycled
//     (never returned to the driver) — the cold pre-warm step at this exact size
//     populates every size class the capture then reuses, so capture itself performs
//     no cudaMalloc.
//   * The graph-safe persistent RoPE row-index table (BuildStepInputs, W7) is baked
//     once per T and never moved.
//   * ResidentWeight uploads every weight once, on first touch (pre-warm).
//   * The FA-2 varlen-decode launcher's per-shape scratch throws if it misses during
//     capture (cuda_flash_attn_fa2.cu) — the pre-warm step at the same padded size
//     populates it. Its host `max_seq_len` only sizes the split-KV grid; the
//     per-request causal geometry is read from the DEVICE seq_lens, and each split's
//     KV range is derived in-kernel from `seqused_k`, so a captured graph stays
//     CORRECT as the sequences grow (identical contract to the shipped decode graphs).
//   * cuBLASLt's workspace is a one-time per-context cudaMalloc.
struct Qwen3DenseDecodeGraph::Impl {
  Impl(const Qwen3DenseWeights& w, const HfConfig& c, vt::Queue q, int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    // ENG-CUDAGRAPH-BREAK W2 (#1261): the kill switch is the SEAM's, not this
    // driver's. `vt::GraphCaptureEnabled()` reads `VLLM_CPP_CUDAGRAPH` once per
    // process into a function-local static; six drivers each read that variable
    // for themselves before this stage, and there was no one switch that turned
    // capture off. This driver no longer owns a copy of it.
    Backend& b = vt::GetBackend(queue.device.type);
    // #1625/#2812: this ONE graph class serves the Qwen3-dense registries AND
    // the Mistral/Llama/InternLM2 registries. The architecture-aware overload
    // defaults capture for exactly Qwen3ForCausalLM (the evidence family: the
    // 0.6B capture pair, the deterministic 4B) and keeps the explicit opt-in
    // for the rest until their captured arms are brought up (#2812 — Mistral's
    // captured arm drifts from its committed eager pair at p[1] t=8 today).
    enabled = vt::GraphCaptureEnabled() &&
              platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !platforms::GetPlatform(queue.device.type)
                   .static_graph_requires_opt_in(config.architectures) &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr) {
      std::string extra;
      if (replay_steps > 0) {
        extra = "; replay branch avg " +
                std::to_string(static_cast<double>(replay_ns) / 1e6 /
                               static_cast<double>(replay_steps)) +
                " ms/step over " + std::to_string(replay_steps) + " steps";
      }
      std::fprintf(stderr,
                   "[Qwen3DenseDecodeGraph] dense decode graph: %lld total replays "
                   "across %zu captured size(s)%s\n",
                   static_cast<long long>(replays), slots.size(), extra.c_str());
    }
    // No DestroyGraph loop: every segment handle belongs to the slot's
    // `vt::BreakableGraph`, whose destructor releases it through
    // `Backend::DestroyGraph`. That routing is what lets ENG-CUDAGRAPH-DEDUP
    // (#1162) interpose at the backend later without editing this driver.
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses, so each size needs
  // its own fixed-address buffers), its persistent embed target + logits output, and
  // its instantiated graph.
  struct SizeSlot {
    std::vector<int32_t> token_ids;  // [S]
    std::vector<int32_t> positions;  // [S]
    CommonAttentionMetadata attn_meta;
    std::unique_ptr<DBuf> hidden;  // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;  // [S,vocab] f32 held graph output
    // ENG-CUDAGRAPH-BREAK W2 (#1261): the instantiated graph, its handle
    // ownership, its release and its `captured()` state now live in the shared
    // seam instead of in a raw `void*` plus a `bool` this driver maintained by
    // hand. `vt::BreakableGraph` is non-copyable and is constructed in place by
    // `slots[S]`, so the map still owns one per padded size.
    vt::BreakableGraph graph;
    int fa_cols = -1;  // captured block-table column count
    bool warm = false;
    int64_t replays = 0;
    // #2469: the on-device cur_pos value THIS slot's trace expects. A seeding
    // step (cold/warm/capture, or a boundary step the predicate below forces)
    // sets it to seq_lens-1; every completed replay's in-trace plus_one adds
    // one. A steady same-request step has seq_lens-1 == expected_cur_pos; a
    // mismatch is a request boundary (or a missed warm) and must re-seed.
    int64_t expected_cur_pos = -1;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const CommonAttentionMetadata& am) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
    }
  };

  const Qwen3DenseWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
  // Steady-state timing (VT_DECODE_GRAPH_STATS): wall time of the replay
  // branch (warm copies + ReplayGraph; excludes the caller's logits readback).
  int64_t replay_ns = 0;
  int64_t replay_steps = 0;
};

Qwen3DenseDecodeGraph::Qwen3DenseDecodeGraph(const Qwen3DenseWeights& weights,
                                             const HfConfig& config, vt::Queue queue,
                                             int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3DenseDecodeGraph::~Qwen3DenseDecodeGraph() = default;

bool Qwen3DenseDecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3DenseDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3DenseDecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv) {
  const int64_t B = static_cast<int64_t>(token_ids.size());
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // Pure decode passes identity logits_indices (gather is a no-op), so the
  // capturable region returns the full [S,vocab].
  const std::vector<int32_t> kNoGather;
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv,
                          impl_->weights, impl_->config, kNoGather);
    if (TtDumpKv(d)) {
      std::vector<float> logits_dump(static_cast<size_t>(vocab));
      lg.Download(d, logits_dump.data());
      int argmax = 0;
      for (int64_t i = 1; i < vocab; ++i)
        if (logits_dump[static_cast<size_t>(i)] > logits_dump[static_cast<size_t>(argmax)])
          argmax = static_cast<int>(i);
      char top2[96];
      TtDumpTop2(logits_dump.data(), vocab, top2, sizeof(top2));
      fprintf(stderr, "[TT-DUMP-LOGITS] eager path argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
              argmax, logits_dump[0], logits_dump[1], logits_dump[2], logits_dump[3], logits_dump[4], top2);
    }
    return WrapDeviceLogits(d, std::move(lg), B, vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then refresh
  // THIS size's persistent host buffers in place.
  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  CommonAttentionMetadata pam;
  BuildPaddedDecodeAttn(S, token_ids, positions, attn_meta, ptok, ppos, pam);

  // Debug: dump attention metadata for comparison
  if (TtDumpKv(d)) {
    fprintf(stderr, "[TT-DUMP-META] S=%lld B=%lld\n", (long long)S, (long long)B);
    fprintf(stderr, "[TT-DUMP-META] real: num_reqs=%d num_tokens=%d slot0=%lld seq_len0=%d bt_cols=%d\n",
            attn_meta.num_reqs, attn_meta.num_actual_tokens,
            attn_meta.slot_mapping.empty() ? -1LL : (long long)attn_meta.slot_mapping[0],
            attn_meta.seq_lens.empty() ? -1 : attn_meta.seq_lens[0],
            attn_meta.block_table_num_cols);
    fprintf(stderr, "[TT-DUMP-META] pad:  num_reqs=%d num_tokens=%d slot0=%lld seq_len0=%d bt_cols=%d\n",
            pam.num_reqs, pam.num_actual_tokens,
            pam.slot_mapping.empty() ? -1LL : (long long)pam.slot_mapping[0],
            pam.seq_lens.empty() ? -1 : pam.seq_lens[0],
            pam.block_table_num_cols);
    fprintf(stderr, "[TT-DUMP-META] real pos0=%d pad pos0=%d\n",
            positions.empty() ? -1 : positions[0],
            ppos.empty() ? -1 : ppos[0]);
  }

  // A block-table column-count change reallocates the persistent block_table (the
  // captured H2D copy's source address moves) -> invalidate this slot's graph and
  // re-warm/re-capture.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  // #2469: continuation is the per-slot boundary predicate — seq_lens-1 must
  // equal the cur_pos this slot's replays produced. Read by the WarmDecodePos
  // regime AND by the boundary re-capture below (both need the same answer).
  bool seq_continuation = true;  // no seq_lens -> no boundary to detect
  s.Refresh(ptok, ppos, pam);
  // HOST-FREE-FORWARD item 5 (TT only): populate the persistent device
  // rope cos/sin tensors for THIS step's UNPADDED positions (the same T-row
  // `positions` vector the captured RopeNeox reads via StepInputs), outside
  // capture, so the captured rope cache-HITs on content. Not ppos.
  if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
    vt::tenstorrent::WarmRopeCosSin(
        positions.data(), static_cast<int64_t>(positions.size()),
        impl_->config.num_attention_heads,
        impl_->config.num_key_value_heads, impl_->config.rotary_dim,
        static_cast<double>(impl_->config.rope_theta));
    // ITEM 5 (RAC): stage the persistent device idx/page-table tensors for
    // the PADDED slot mapping the captured ReshapeAndCache will see. The
    // kernel keys its cache on si.slot_mapping's host buffer; si builds from
    // attn_meta (pam here) so this is the same buffer content.
    // ITEM 5: prime paged-KV device shadows for EVERY layer (MUST run before
    // WarmRacIdx, which builds the persistent sharded input from the shadows).
    for (const auto& kv : attn_kv) {
      const int64_t max_slot = pam.slot_mapping.empty() ? 0
          : *std::max_element(pam.slot_mapping.begin(), pam.slot_mapping.end());
      const int64_t used = (max_slot < 0) ? 1
          : std::max<int64_t>(1, max_slot / kv.block_size + 1);
      const size_t half = static_cast<size_t>(kv.block_size * kv.num_kv_heads *
                                               kv.head_size) * vt::SizeOf(kv.dtype);
      char* base = static_cast<char*>(kv.data);
      vt::tenstorrent::WarmPagedKvShadow(
          base, base + half, kv.num_blocks, kv.block_size,
          kv.num_kv_heads, kv.head_size, used);
    }
    // R2: seed the on-device-advanced cur_pos BEFORE WarmRacIdx, so the RAC
    // path can alias update_idxs to it (eliminating the per-replay
    // update_idxs copy_to_device — the toxic ~38-replay hang class).
    // replay_regime = this slot's graph is captured (the warm hooks run
    // BEFORE the boundary Reset below): replay steps leave cur_pos to the
    // captured plus_one; cold/warm/capture steps re-seed it, which is what
    // makes a post-boundary RE-capture read the right position (#1476).
    if (!pam.seq_lens.empty()) {
      // #2469: replay regime requires this step to CONTINUE the request the
      // trace was last driven with: seq_lens-1 must equal the on-device
      // cur_pos this slot's replays produced. A bare captured() flag also
      // holds on a request boundary, where every re-seed path is already
      // disabled (WarmDecodePos's replay branch, WarmRacIdx's steady state,
      // WarmPaMeta's process-global r2_steady), so the recycled size slot
      // replays the previous request's final cur_pos: PA attends the dead
      // request's decode KV rows still in the recycled physical block and RAC
      // writes at the stale virtual position. A mismatch forces the existing
      // seeding path; RAC and PA alias the same DecodePos buffer, so the one
      // copy_to_device fixes both consumers.
      const bool continuation =
          (static_cast<int64_t>(pam.seq_lens[0]) - 1) == s.expected_cur_pos;
      seq_continuation = continuation;
      vt::tenstorrent::WarmDecodePos(
          pam.seq_lens.data(), static_cast<int64_t>(pam.num_reqs),
          /*replay_regime=*/s.graph.captured() && continuation);
      // A seeding step leaves the device holding seq_lens-1; a continuation
      // step already did (that is the predicate). Either way this is the
      // value the trace's plus_one starts this step from.
      s.expected_cur_pos = pam.seq_lens[0] - 1;
    }
    vt::tenstorrent::WarmRacIdx(
        pam.slot_mapping.data(), pam.slot_mapping.data(),
        static_cast<int64_t>(pam.slot_mapping.size()),
        attn_kv.empty() ? 32 : attn_kv[0].block_size,
        pam.block_table_tensor.data(),
        static_cast<int64_t>(pam.block_table_num_cols),
        pam.seq_lens.data());
    // ITEM 5 (PA): warm persistent page_table + cur_pos device tensors.
    if (!pam.block_table_tensor.empty() && !pam.seq_lens.empty()) {
      vt::tenstorrent::WarmPaMeta(
          pam.block_table_tensor.data(),
          static_cast<int64_t>(pam.num_reqs),
          static_cast<int64_t>(pam.block_table_num_cols),
          static_cast<int64_t>(pam.block_table_num_cols), 1,
          pam.seq_lens.data());
    }
  }
  s.fa_cols = cols;
  // #2469: a same-size request boundary must re-capture, not seed-and-replay.
  // With the boundary seed alone the first post-boundary replay still drifts
  // (near-tie wrong token, flaky across builds: 374 unfixed; 13/11/264 with
  // the seed verified correct on device), so the recycled trace replays some
  // state the per-step warms do not refresh. Route the boundary through the
  // same proven lane as a cols change (#1476): destroy the trace, let THIS
  // step run eagerly against the freshly seeded cur_pos, and re-capture on
  // the next step, which records addresses built from the new request's own
  // state. Costs one eager step and one capture per request boundary.
  const bool tt_boundary =
      d.q.device.type == vt::DeviceType::kTENSTORRENT && !seq_continuation;
  if ((cols_changed || tt_boundary) && s.graph.captured()) {
    // Reset() releases every segment through Backend::DestroyGraph and returns
    // the container to its as-constructed state, which is also what lets the
    // next capture open a scope on it (the scope refuses a container that
    // already holds one).
    s.graph.Reset();
    s.warm = false;
  }

  // Fast path: this size's graph is captured. On TT, refresh the persistent
  // decode-ids tensor (allocation-free) and replay — the embedding itself is
  // INSIDE the captured region, so a replay step performs zero eager device
  // allocations (eager alloc/free churn around a live trace hung the device
  // ~60 replays in). CUDA keeps the outside-the-graph EmbedInto.
  // VT_TT_RECAPTURE_EVERY=N (TT only): destroy and re-capture the graph every
  // N replays. WORKAROUND for the deterministic ~38-replay completion hang on
  // this tt-metal build: a replayed mesh trace stops completing (futex wait
  // in the post-replay readback) after ~38 replays of one trace id,
  // independent of interleaved eager-copy count and readback count.
  // Re-capturing resets the per-trace device state; the eager re-warm step
  // and capture step run with NO live trace (DestroyGraph releases it), so
  // eager allocations are legal there.
  bool do_replay = s.graph.captured();
  if (do_replay && d.q.device.type == vt::DeviceType::kTENSTORRENT) {
    const char* rc_env = std::getenv("VT_TT_RECAPTURE_EVERY");
    const int rc_n = rc_env != nullptr ? std::atoi(rc_env) : 0;
    if (rc_n > 0 && s.replays >= rc_n) {
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
        std::fprintf(stderr,
                     "[TT-STEP] recapture: destroying graph after %lld replays "
                     "(every %d)\n",
                     static_cast<long long>(s.replays), rc_n);
      s.graph.Reset();
      s.warm = false;
      do_replay = false;
    }
  }
  if (do_replay) {
    const auto replay_t0 = std::chrono::steady_clock::now();
    if (d.q.device.type == vt::DeviceType::kTENSTORRENT) {
      vt::tenstorrent::WarmDecodeIds(
          s.token_ids.data(), static_cast<int64_t>(s.token_ids.size()));
    } else {
      EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    }
    if (TtDumpKv(d)) {
      std::vector<float> pre(static_cast<size_t>(vocab));
      s.logits->Download(d, pre.data());
      fprintf(stderr, "[TT-DUMP-LOGITS] pre-replay first5=[%f,%f,%f,%f,%f]\n",
              pre[0], pre[1], pre[2], pre[3], pre[4]);
    }
    // Through the seam's container, never `Backend::ReplayGraph` directly: the
    // container replays its segments in order (one, here, because a decode
    // capture is kFull) and owns the G3 replay counter that the reachability
    // gate reads.
    s.graph.Replay(impl_->queue);
    ++s.replays;
    ++s.expected_cur_pos;  // the replay's in-trace plus_one advanced cur_pos
    ++impl_->replays;
    impl_->replay_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - replay_t0)
                            .count();
    ++impl_->replay_steps;
    if (TtDumpKv(d)) {
      std::vector<float> logits_dump(static_cast<size_t>(vocab));
      s.logits->Download(d, logits_dump.data());
      int argmax = 0;
      for (int64_t i = 1; i < vocab; ++i)
        if (logits_dump[static_cast<size_t>(i)] > logits_dump[static_cast<size_t>(argmax)])
          argmax = static_cast<int>(i);
      char top2[96];
      TtDumpTop2(logits_dump.data(), vocab, top2, sizeof(top2));
      fprintf(stderr, "[TT-DUMP-LOGITS] replay step argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
              argmax, logits_dump[0], logits_dump[1], logits_dump[2], logits_dump[3], logits_dump[4], top2);
    }
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + weight residency + per-shape kernel scratch were warmed for
  // this size by the previous (eager) step. CAPTURE the layer region once,
  // instantiate the graph, then launch it.
  if (s.warm) {
    const bool tt_dev = d.q.device.type == vt::DeviceType::kTENSTORRENT;
    if (tt_dev) {
      // Stage ids for the captured embedding (outside capture).
      vt::tenstorrent::WarmDecodeIds(
          s.token_ids.data(), static_cast<int64_t>(s.token_ids.size()));
      // HOST-FREE-FORWARD R4 (#1105): _dummy_run mirror for the capture-only
      // embed segment. The eager step's EmbedInto never runs the device
      // embedding or the hidden-shadow copy, so both programs are cold at
      // first capture and tt-metal refuses to load new binaries mid-trace
      // (TT_FATAL mesh_workload.cpp:153 !is_capturing_trace). Run the exact
      // captured segment once OUTSIDE the scope; the captured pass then hits
      // the program cache. Safe to run twice: the hold in
      // EmbedDeviceIdsInto replaces the previous out tensor, and nothing
      // references the dummy run's output.
      Tensor dtab = ResidentWeight(d, impl_->weights.embed_tokens,
                                   {impl_->config.vocab_size,
                                    impl_->config.hidden_size});
      vt::tenstorrent::EmbedDeviceIdsInto(
          s.hidden->ptr(), S, impl_->config.hidden_size, dtab.data,
          impl_->config.vocab_size, impl_->config.hidden_size,
          static_cast<int64_t>(s.token_ids.size()));
    } else {
      EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    }
    // ENG-CUDAGRAPH-BREAK W2 (#1261): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope owns
    // the segment, the handle, its release, the drain that a mid-capture throw
    // needs (three drivers each hand-rolled that same
    // `try { EndCaptureGraph(); } catch (...) {}`) and the G3 counters.
    //
    // kFULL, and the mode is the whole argument. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63`) is documented at
    // `:630-632` as a FULL graph for DECODE batches and a piecewise one for
    // prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the full
    // half. This is a decode driver, so its capture is one segment with the
    // attention break points INSIDE it — byte-identical in shape to the region
    // this line replaces. Opening it kPiecewise would turn every layer's
    // attention into an eager call between two graph replays, which is not
    // vLLM's decode behaviour and which nothing in this row's record supports.
    // The piecewise arm reaches this driver at W6, when the eligibility
    // predicate moves off `pure_decode`; `## Owed` in the spec names what has to
    // be true first.
    std::optional<DBuf> lg;
    {
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      if (tt_dev) {
        // Capture-safe embedding over the persistent ids tensor, writing the
        // persistent hidden shadow the layer region reads.
        Tensor dtab = ResidentWeight(d, impl_->weights.embed_tokens,
                                     {impl_->config.vocab_size,
                                      impl_->config.hidden_size});
        vt::tenstorrent::EmbedDeviceIdsInto(
            s.hidden->ptr(), S, impl_->config.hidden_size, dtab.data,
            impl_->config.vocab_size, impl_->config.hidden_size,
            static_cast<int64_t>(s.token_ids.size()));
      }
      lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                         impl_->weights, impl_->config, kNoGather);
      // R2: advance cur_pos on-device (plus_one) INSIDE the captured trace, at
      // the END of the body (after all reads of cur_pos in sdpa_decode/RAC).
      // The NEXT replay sees cur_pos+1 — eliminating the per-replay cur_pos /
      // update_idxs copy_to_device (the toxic ~38-replay hang class).
      if (tt_dev && !pam.seq_lens.empty()) {
        vt::tenstorrent::CaptureDecodePosAdvance(
            static_cast<int64_t>(pam.num_reqs));
      }
    }  // ~GraphCaptureScope closes the segment and files it on s.graph
    // NOT CAPTURED covers TWO states, and returning `*lg` is correct for exactly
    // one of them. `~GraphCaptureScope` must swallow a throwing
    // `EndCaptureGraph` — a destructor that propagates terminates — so a FAILED
    // capture leaves the container reporting what an INERT scope reports.
    //
    //   * INERT (`capture_failed() == false`): the backend cannot capture, or
    //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, the layer
    //     region above ran EAGERLY, and `*lg` is a real result. Return it and
    //     go back to cold so the next same-size step re-warms.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph`
    //     threw — `cudaStreamEndCapture` returning
    //     `cudaErrorStreamCaptureInvalidated`/`WrongThread`, or a failing
    //     `cudaGraphInstantiate` (`src/vt/cuda/cuda_backend.cu:229`, `Check()`
    //     at `:50`). Under stream capture NOTHING between `BeginCapture` and
    //     the throw executed: every kernel was RECORDED, so `*lg` is
    //     pool-recycled memory. Returning it would hand this step uncomputed
    //     device memory as its logits — silently wrong tokens, no fault, and a
    //     token gate cannot see it.
    //
    // So the failure PROPAGATES, carrying the runtime's own exception. This is
    // what the pre-seam driver did (`s.graph = b.EndCaptureGraph(...)` was
    // unguarded), and it is not a recovery this stage can justify inventing: a
    // stream whose capture was INVALIDATED has not told us it is usable, and the
    // three hand-rolled drains this migration cites drain and RETHROW THE
    // ORIGINAL ERROR (`qwen3_5.cpp:10184`, `:10609`, `qwen3_dflash.cpp:1106`)
    // rather than returning a value. Gated at
    // `tests/vllm/models/test_qwen3_decode_graph_seam.cpp`.
    if (!s.graph.captured()) {
      s.warm = false;  // either way this slot goes back to cold
      if (s.graph.capture_failed()) {
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();  // clear the failure with the graph it described
        // The runtime's OWN diagnosis where the seam holds it. It is empty only
        // on the arm where an exception was already propagating THROUGH the
        // scope, which cannot reach this line; the refusal below is what makes
        // that unreachability an assertion rather than a claim.
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "Qwen3DenseDecodeGraph: the decode capture was ABANDONED and its logits "
                 "were never computed; refusing to return uncaptured device memory");
      }
      ForwardLogits drained = WrapDeviceLogits(d, std::move(*lg), B, vocab);
      drained.device_tensor =
          MakeTensor(drained.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
      return drained;
    }
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[Qwen3DenseDecodeGraph] captured dense decode graph for padded "
                   "size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    s.graph.Replay(impl_->queue);
    s.replays = 1;
    ++s.expected_cur_pos;  // the capture step's launch ran the trace once
    ++impl_->replays;
    if (TtDumpKv(d)) {
      std::vector<float> logits_dump(static_cast<size_t>(vocab));
      s.logits->Download(d, logits_dump.data());
      int argmax = 0;
      for (int64_t i = 1; i < vocab; ++i)
        if (logits_dump[static_cast<size_t>(i)] > logits_dump[static_cast<size_t>(argmax)])
          argmax = static_cast<int>(i);
      char top2[96];
      TtDumpTop2(logits_dump.data(), vocab, top2, sizeof(top2));
      fprintf(stderr, "[TT-DUMP-LOGITS] capture step argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
              argmax, logits_dump[0], logits_dump[1], logits_dump[2], logits_dump[3], logits_dump[4], top2);
    }
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool size classes, the
  // resident weights, and the FA-2 per-shape scratch for this size) and defer
  // capture to the next same-size step. This is a real decode step — nothing wasted.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                          impl_->weights, impl_->config, kNoGather);
  // Debug: dump logits first 5 values
  if (TtDumpKv(d)) {
    std::vector<float> logits_dump(static_cast<size_t>(vocab));
    lg.Download(d, logits_dump.data());
    int argmax = 0;
    for (int64_t i = 1; i < vocab; ++i)
      if (logits_dump[static_cast<size_t>(i)] > logits_dump[static_cast<size_t>(argmax)])
        argmax = static_cast<int>(i);
    char top2[96];
    TtDumpTop2(logits_dump.data(), vocab, top2, sizeof(top2));
    fprintf(stderr, "[TT-DUMP-LOGITS] cold step argmax=%d first5=[%f,%f,%f,%f,%f] %s\n",
            argmax, logits_dump[0], logits_dump[1], logits_dump[2], logits_dump[3], logits_dump[4], top2);
  }
  s.warm = true;
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), B, vocab);
  fl.device_tensor =
      MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  return fl;
}

// Per-family gate (see qwen3.h). DEFAULT ON (row QUANT-CT-MXFP4-MARLIN-STRUCT step 1,
// parity-enabler): the shared dense decode CUDA-graph is byte-coherent + token-exact
// vs the eager forward on both dense checkpoints — test_qwen3_paged_engine 184/184
// (Qwen3-0.6B near-tie + Qwen3-4B) and test_qwen3_dense_async_serving 82/82, graph
// ON == OFF — and on the Qwen3-8B-MXFP4 #44 smoke (deterministic 3/3 token-exact +
// coherent), all captured on GB10. An explicit VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0
// opts back out to the eager path (byte-identical to the pre-graph forward); the
// framework kill switch VLLM_CPP_CUDAGRAPH=0 additionally forces eager inside the
// driver (Impl::enabled), so the graph never captures under either opt-out.
bool DenseDecodeGraphEnabled() {
  const char* value = std::getenv("VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH");
  return !(value != nullptr && value[0] == '0');
}

std::optional<ForwardLogits> DenseDecodeGraphForward(
    std::unique_ptr<Qwen3DenseDecodeGraph>& graph,
    const Qwen3DenseWeights& weights, const ModelForwardInput& input) {
  // Only a graph-eligible PURE decode on a CUDA static-graph platform routes here;
  // otherwise the caller runs its existing eager Forward/ForwardDevice path. The
  // driver itself caps the padded batch at max_num_reqs and falls back internally
  // for larger batches / when the framework kill switch is set.
  if (!DenseDecodeGraphEnabled() || !input.pure_decode ||
      !platforms::GetPlatform(input.queue.device.type).support_static_graph_mode()) {
    return std::nullopt;
  }
  // #323 — CORRECTNESS FIRST. `Step()` below replays against the HOST
  // `input.token_ids` and never reads `input.device_token_ids`. On the depth-2
  // async path the combine has patched the DEVICE ids and `token_ids` is
  // deliberately stale for decode rows (runner.cpp), so the replay generates
  // from stale ids and every concurrent request past slot 0 degenerates — the
  // #31 signature, reproduced on Mistral-7B-v0.3 and InternLM2-chat-1.8B and
  // latent for EVERY classic-dense model, since the graph is default-ON.
  //
  // Measured, same binary, 4-concurrent battery vs a batch-1 sync anchor:
  //   depth-1, graph ON   PASS 78/78      (no async pipelining)
  //   depth-2, graph OFF  PASS 82/82      (eager path honours the scope)
  //   depth-2, graph ON   FAIL, slots 1-3 degenerate
  // Both conditions are required. THE MECHANISM THIS COMMENT USED TO RECORD FOR
  // THAT — "the registry-level DeviceTokenIdsScope (60e71a0e) did not close it:
  // this path returns BEFORE the eager forward ever runs" — IS FALSE, and
  // ENG-CUDAGRAPH-BREAK W4 (#1307) established it by reading the tree at the
  // commit that wrote the sentence. At `338cbbfd1^` the scope is constructed at
  // `qwen3_dense.cpp:96-97`, BEFORE `DenseDecodeGraphForward` at `:102`, and all
  // three of this driver's arms (`:610,621,644`) call `EmbedInto`, which applies
  // the override through `ApplyDeviceTokenIdsOverride`. The override was LIVE on
  // the graph path. `60e71a0e`'s three registry sites (`mistral_registry.cpp:89`,
  // `internlm2_registry.cpp:92`, `llama_registry.cpp:101`) all have the same
  // order, and two of them are the models the battery reproduced on.
  //
  // So the measured failure is real and its recorded CAUSE is not the one that
  // produced it, and nobody has identified what did. That is why the decline
  // STANDS rather than why it should go: a mitigation whose failure mode is
  // unexplained is not retired by a refactor that plausibly addresses an
  // explanation no one has confirmed. Removing it needs the battery
  // (`tests/parity/test_qwen3_dense_async_serving.cpp`) run twice — as it stands,
  // and with the decline deleted, because only the second can fail.
  //
  // THE FIX THIS COMMENT NAMES IS NOW HALF-BUILT, AND NOT IN THIS DRIVER. When W4
  // measured it, no decode graph carried token ids to the device at all:
  // `StepDevInputs` (`qwen3_5.cpp`) has no token-id member, its pinned sibling's
  // `token_ids` block was allocated and filled every step and NEVER uploaded and
  // never read (W4 removed it), and every batched driver embedded OUTSIDE the
  // captured region from the HOST vector. #1305 changed the DESTINATION half for
  // two of the nine drivers: `Qwen3MoeDecodeGraph` and `DeepseekV2DecodeGraph`
  // now hold their step identifiers in a `vllm::StepTokenIds`
  // (`include/vllm/model_executor/models/step_token_ids.h`) whose device address
  // is stable for the life of the slot, and refresh it through
  // `vt::PersistentStepInput::RefreshFromDevice` from the runner's mirror.
  //
  // THAT BUYS THIS DRIVER NOTHING, and it does not weaken the decline. This
  // driver has no such destination: W2 (#1261) migrating its capture onto the
  // shared seam did not move the inputs, and #1305 did not touch it. And even in
  // the two drivers that now have one, the refresh runs OUTSIDE the capture, once
  // per step, because `vt::Embedding` allocates a device bounds-check flag and
  // synchronizes the stream and therefore cannot be captured. Reading the
  // identifiers at REPLAY time — this comment's own wording for the fix — still
  // exists in no driver, which is why #1305 records the graph half of the defect
  // as unsettled rather than closing it.
  //
  // Declining the graph while the mirror is live falls back to the proven-correct
  // eager path. This is a MITIGATION, not the end state, and a correct stream
  // outranks the graph's throughput until the two battery runs above say
  // otherwise. Owner: row `ENG-CUDAGRAPH-BREAK`, the stage that gets a `dgx`
  // window WITH the Qwen3-0.6B/4B checkpoints the battery needs; recorded under
  // `## Owed` in `.agents/specs/eng-cudagraph-break.md`, tracked by #1179 and
  // #323, and gated in `tests/vllm/models/test_qwen3_decode_graph_seam.cpp` so
  // neither arm can change silently.
  if (input.device_token_ids != nullptr) {
    return std::nullopt;
  }
  // gdn_state_slots carries max_num_reqs for EVERY arch (the runner sets it from
  // max_num_reqs_ regardless of whether the model has GDN layers), so a pure
  // full-attention model reads its capture-size cap from it unchanged.
  if (!graph) {
    graph = std::make_unique<Qwen3DenseDecodeGraph>(weights, input.config,
                                                    input.queue, input.gdn_state_slots);
  }
  return graph->Step(input.token_ids, input.positions, input.attn_meta,
                     input.attn_kv);
}

}  // namespace vllm
