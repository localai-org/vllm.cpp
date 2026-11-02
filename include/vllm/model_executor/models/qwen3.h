// Qwen3 DENSE (`Qwen3ForCausalLM`) — the first ADDITIVE-MODEL bring-up.
// (Upstream: vllm/model_executor/models/qwen3.py @ e24d1b24; config
// Qwen3-0.6B/config.json — a pure standard-dense transformer: NO GDN, NO MoE,
// standard (non-gemma) RMSNorm, per-head q/k norm, tied lm_head, sliding_window
// null → one full-attention KV group only.)
//
// This header carries the registry-facing declarations shared by the Qwen3
// registry TU (qwen3_dense.cpp): the per-family config hook and the
// full-attention-ONLY KV-cache spec builder. The heavy dense forward machinery
// (Qwen3DenseModel::Forward/ForwardDevice) and the on-disk weight name map land
// in W2/W3 (qwen3.cpp / qwen3_weights.cpp) — see
// .agents/specs/first-additive-model-qwen3-dense.md §6. W0/W1 deliberately do
// NOT implement the forward; the registered forward hook is a clear-throwing
// stub until W3.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// One Qwen3 dense self-attention block. Mirrors vLLM `Qwen3Attention`
// (qwen3.py:65-168): merged QKV (packed_modules_mapping qkv_proj<-[q,k,v]) and
// per-head q/k RMSNorm (RMSNorm(head_dim)) applied before RoPE. Qwen3 has NO
// attention output gate (unlike the Qwen3.6 gated full-attention), so the merged
// QKV width is exactly Hq*Dh + 2*Hkv*Dh (no doubling).
//
// Projections are kept RAW in the on-disk torch Linear [N=out, K=in] orientation
// (nk=true), consumed by vt::MatmulBT — the cuBLASLt TN fast path vLLM's
// F.linear hits (mirrors the qwen3_5-dense in_proj_qkvz choice, notes §3.6).
struct Qwen3DenseAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  // Optional original GGUF shards in q/k/v order, including mixed encodings.
  std::vector<OwnedTensor> qkv_proj_parts;
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
  OwnedTensor q_norm;    // bf16 [head_dim]  (per-head RMSNorm, non-gemma)
  OwnedTensor k_norm;    // bf16 [head_dim]
  // Merged QKV bias [Hq*Dh + 2*Hkv*Dh], present only when config attention_bias
  // is true. EMPTY for Qwen3-0.6B (attention_bias=false).
  OwnedTensor qkv_bias;

  // NVFP4 W4A16 alternatives to the two BF16 projections above (compressed-
  // tensors `nvfp4-pack-quantized`, e.g. RedHatAI/Qwen3-32B-NVFP4A16). EXACTLY
  // ONE of {qkv_proj, qkv_proj_fp4} is populated per layer, likewise
  // {o_proj, o_proj_fp4} — the loader probes `.weight_packed` and the forward
  // dispatches on `Empty()`. Same merged ownership as the BF16 fields: the
  // fp4 qkv is ONE [Hq*Dh + 2*Hkv*Dh, H] operand (rows q|k|v).
  Nvfp4Weight qkv_proj_fp4;  // [N=Hq*Dh + 2*Hkv*Dh, K=H]
  Nvfp4Weight o_proj_fp4;    // [N=H, K=Hq*Dh]

  // EXL3 trellis alternatives (QUANT-EXL3 W1b, #2181). Same ownership rule as
  // the NVFP4 fields: exactly one of {bf16, fp4, exl3} is populated per layer
  // and the forward dispatches on `Empty()`.
  //
  // q/k/v are held SEPARATELY where bf16 and NVFP4 hold one merged operand,
  // because that is how the checkpoint stores them and merging them is a real
  // transform rather than a concatenation: the trellis is
  // `[k/16, n/16, 32*bits]`, so joining on the output dim interleaves per
  // input tile. It is a VALID transform for this family -- `had_r_128` blocks
  // the output in 128s and Llama-3.2-1B's `q` (2048) and `k`/`v` (512) are each
  // a multiple of 128, so no Hadamard block would straddle two matrices -- but
  // it is a wave with its own gate, recorded under `## Owed` in
  // `specs/quant-exl3-shared.md`, not something to slip in beside a bring-up.
  Exl3Weight q_proj_exl3;  // [K=H, N=Hq*Dh]
  Exl3Weight k_proj_exl3;  // [K=H, N=Hkv*Dh]
  Exl3Weight v_proj_exl3;  // [K=H, N=Hkv*Dh]
  Exl3Weight o_proj_exl3;  // [K=Hq*Dh, N=H]

  // True when this block's projections are NVFP4 W4A16.
  bool IsNvfp4() const { return !qkv_proj_fp4.Empty(); }
  // True when this block's projections are EXL3 trellis (QUANT-EXL3, #2181).
  bool IsExl3() const { return !q_proj_exl3.Empty(); }
};

// Dense SwiGLU MLP. Mirrors vLLM `Qwen3MLP` = `Qwen2MLP` (qwen3.py:58): merged
// gate_up (packed_modules_mapping gate_up_proj<-[gate,up]) -> SiluAndMul ->
// down_proj. Raw-NK like the attention projections.
struct Qwen3DenseMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  // Optional original GGUF shards in gate/up order.
  std::vector<OwnedTensor> gate_up_proj_parts;
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk

  // NVFP4 W4A16 alternatives (see Qwen3DenseAttnWeights). gate and up are kept
  // as SEPARATE fp4 operands so the forward can choose vLLM's fused merged
  // gate_up Marlin layout (ONE GEMM, size_n=2I — what
  // prepare_fp4_layer_for_marlin does to the merged parameter) or the split
  // two-GEMM A/B fallback, exactly like the 35B shared expert.
  Nvfp4Weight gate_proj_fp4;  // [N=I, K=H]
  Nvfp4Weight up_proj_fp4;    // [N=I, K=H]

  // EXL3 trellis alternatives (QUANT-EXL3 W1b, #2181), gate and up separate for
  // the same reason as q/k/v above.
  Exl3Weight gate_proj_exl3;  // [K=H, N=I]
  Exl3Weight up_proj_exl3;    // [K=H, N=I]
  Exl3Weight down_proj_exl3;  // [K=I, N=H]
  Nvfp4Weight down_proj_fp4;  // [N=H, K=I]

  bool IsNvfp4() const { return !down_proj_fp4.Empty(); }
  bool IsExl3() const { return !down_proj_exl3.Empty(); }
};

// One Qwen3 dense decoder layer: input/post standard (non-gemma) RMSNorm +
// attention + dense SwiGLU MLP (qwen3.py:171-242).
struct Qwen3DenseLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  Qwen3DenseAttnWeights attn;
  Qwen3DenseMlpWeights mlp;
};

// Whole Qwen3 dense text-model weights. When `tie_word_embeddings` is true
// (Qwen3-0.6B), `lm_head` is EMPTY and the output projection aliases
// `embed_tokens` — mirroring vLLM `Qwen3ForCausalLM.__init__` (`self.lm_head =
// self.model.embed_tokens`, qwen3.py:294-295) and its loader skip_prefixes
// (`["lm_head."]`, qwen3.py:339) which drops the checkpoint's redundant
// lm_head.weight. When false, `lm_head` holds the ParallelLMHead weight in
// Matmul-B [H, vocab] layout.
struct Qwen3DenseWeights {
  bool tie_word_embeddings = true;
  bool attention_bias = false;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied

  // EXL3 trellis lm_head (QUANT-EXL3 W1b, #2181). An EXL3 checkpoint does NOT
  // tie its head -- `turboderp/Llama-3.2-1B-Instruct-exl3` sets
  // `tie_word_embeddings: false` and ships a real quantized head where the bf16
  // Llama-3.2-1B ties them -- so the bf16 loader's `skip_prefixes(["lm_head."])`
  // must not be applied to an EXL3 load.
  //
  // ITS WIDTH IS NOT THE BODY'S. That head is SIX-bit while the layers are
  // three, so it is the one tensor in the checkpoint that proves the per-tensor
  // `bits` rule pays. The CUDA arm instantiates `bits == 3` only and refuses
  // anything else by name, so on a device this head is the first thing to
  // refuse; the CPU arm is generic over all eight widths and runs it. Widening
  // the device arm is W3.
  Exl3Weight lm_head_exl3;   // [K=H, N=vocab]

  std::vector<Qwen3DenseLayerWeights> layers;
};

// Load `Qwen3ForCausalLM` (Qwen3-0.6B) safetensors into Qwen3DenseWeights. Name
// map: model.embed_tokens.weight, model.norm.weight, and per layer
// model.layers.N.{input_layernorm,post_attention_layernorm}.weight,
// .self_attn.{q,k,v,o}_proj.weight (+ per-head .self_attn.{q,k}_norm.weight),
// .mlp.{gate,up,down}_proj.weight. q/k/v merged into one qkv_proj and gate/up
// into one gate_up_proj (vLLM packed_modules_mapping). The checkpoint's
// lm_head.weight is intentionally SKIPPED when tie_word_embeddings (aliased to
// embed_tokens). Reuses the shared dense_weight_loaders.h helpers. Text path
// only; no vision tower (Qwen3-0.6B is text-only).
Qwen3DenseWeights LoadQwen3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The DENSE Qwen3 (`Qwen3ForCausalLM`) forward — the additive-model W3 capstone.
// A pure standard-dense transformer built by COMPOSING the public vt:: ops + the
// fusion catalog (kFusedAddRmsNormStd / kAttnQkNormRope, recipes.h), with NO GDN,
// NO MoE and NO attention gate. Structurally the Qwen3.6-dense full-attention
// path minus the hybrid extras (mirrors qwen3_5.cpp's DenseForwardLayers /
// FullAttnBlockPaged, stripped to one full-attention KV group, standard
// (non-gemma) RMSNorm, per-head q/k RMSNorm before RoPE, and a tied lm_head).
//
// Per decoder layer (qwen3.py::Qwen3DecoderLayer @ e24d1b24):
//   input_layernorm (std add+RMSNorm) -> qkv_proj (BF16 MatmulBT) -> split q/k/v
//   -> per-head q_norm/k_norm (RMSNorm(head_dim), non-gemma) BEFORE RoPE
//   -> RoPE (NeoX, theta 1e6) -> FA2 causal paged attention -> o_proj
//   -> post_attention_layernorm (std add+RMSNorm) -> gate_up_proj -> SiluAndMul
//   -> down_proj. Then final_norm (std RMSNorm) -> lm_head (TIED to embed_tokens).
//
// The residual stream is kept in the model dtype (bf16, matching vLLM's
// fused_add_rms_norm residual); q/k-norm + RoPE run in f32 (the qwen3_5 fallback
// numeric anchor); the paged KV cache is written bf16; the query stays f32 into
// vt::PagedAttention. Returns [n_out, vocab] f32 logits (n_out == num_reqs when
// logits_indices gather-before-lm_head, else num_actual_tokens).
class Qwen3DenseModel {
 public:
  // Batched PAGED dense forward. token_ids/positions are the flattened
  // length-num_actual_tokens step inputs; attn_meta the full-attention KV group's
  // CommonAttentionMetadata; attn_kv one PagedKvCache per layer (all layers are
  // full-attention). `logits_indices` (optional): the per-request last-token row
  // indices — when a proper subset of the T rows, the final hidden rows are
  // gathered on-device BEFORE lm_head so the return is [num_reqs, vocab].
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant (sampler-on-device hot path): same contract as
  // Forward but returns the lm_head output as a device buffer with no full-logits
  // D2H (ForwardLogits::device_*).
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // POOLING forward (ARCH-ONE-SURFACE ROW 6): the same embed + layer stack,
  // stopping after the final RMSNorm (+ the logits_indices gather) with NO
  // lm_head — the forward of an as_embedding_model conversion
  // (vllm/model_executor/models/adapters.py:135-151 replaces the output layer
  // with a missing-layer stage; the pooler consumes the post-final-norm
  // hidden). Returns a HOST ForwardLogits carrier of [n_out, hidden_size] f32
  // rows (`vocab` == hidden_size on this path); the engine's pooling branch
  // hands them to the landed PoolingRunner. Additive: no text caller routes
  // here, and the lm_head tail above is byte-identical.
  static ForwardLogits ForwardHidden(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // The `inputs_embeds` ENTRY (MODEL-MUSIC-MUSIC3 W2, #672) — additive, and the
  // door the Qwen3 family already has everywhere except here.
  //
  // `Qwen3VLForConditionalGeneration` takes `inputs_embeds_bf16` after scattering
  // the vision tower's rows into it (qwen3_vl.h:145,159; qwen3_vl_text.h:62-65);
  // Gemma-4 and Muse-Glimmer do the same (gemma4.h:210-218,
  // muse_glimmer.h:369-380). All of them exist because upstream's own
  // `Qwen3Model.forward` accepts EITHER `input_ids` OR `inputs_embeds`, and the
  // DENSE registration only ever wired the first. MiniMax-Music3's autoregressive
  // loop is the caller that needs the second on the plain text model: its frame
  // feedback is `_embed_audio_frame` (encoders.py:106-115), a SUM of one language-
  // model row and seven depth-decoder rows scaled by num_codebooks^-0.5, which
  // corresponds to no vocabulary entry and so cannot be spelled as a token id.
  //
  // `inputs_embeds_bf16` is the ALREADY-BUILT [num_tokens * hidden_size] host bf16
  // hidden stream; `token_ids` has no counterpart here and is not taken. The layer
  // stack, the final RMSNorm, the `logits_indices` gather and the (tied/untied)
  // lm_head are the SAME ones Forward runs, so this is one branch at the embed
  // step and nothing else — every token-id caller is byte-identical, which
  // tests/vllm/models/test_qwen3_forward.cpp asserts by feeding the embedding of
  // the same ids through both entries.
  //
  // `out_hidden`, when non-null, additionally receives the [n_out, hidden_size]
  // f32 post-final-norm rows — the SAME rows ForwardHidden returns — from the one
  // forward that produced the logits. Music3 reads `last_hidden_state[:, -1]` and
  // applies `lm_head` to that same row (encoders.py:311-318), so a second 8.6B
  // pass to fetch the other half would be pure waste. Null ⇒ not computed.
  static std::vector<float> ForwardEmbeds(
      const std::vector<uint16_t>& inputs_embeds_bf16,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      std::vector<float>* out_hidden = nullptr);
};

// SHARED pure-dense decode CUDA-graph driver — the sibling of Qwen3MoeDecodeGraph
// (qwen3_moe.cpp) and Qwen3_5DenseDecodeGraph (qwen3_5.cpp), driving the SHARED
// Qwen3DenseModel forward. Because `LlamaModel`/`MistralModel`/`InternLM2Model` are
// type aliases of `Qwen3DenseModel` over `Qwen3DenseWeights`, ONE driver serves all
// FIVE registrations (Qwen3ForCausalLM, LlamaForCausalLM, InternLM3ForCausalLM,
// MistralForCausalLM, InternLM2ForCausalLM).
//
// Same cold -> warm -> capture -> replay state machine + same padded-batch capture
// set (DecodeGraphSizes / PadToCaptureSize) as the siblings; the ATTENTION-ONLY
// analogue (no GDN metadata — pure full-attention). Captures the pure-decode dense
// forward once per PADDED batch size and replays it per token, removing the per-step
// host launch tax (~hundreds of tiny kernel launches/step) the eager decode paid.
// The embedding stays OUTSIDE the capture region (run per step into a persistent
// hidden buffer); every per-call device scratch is pool-backed / resident (the
// shared DevicePool + resident weights + the FA-2 varlen-decode per-shape scratch)
// so a cold pre-warm makes the captured region do ZERO cudaMalloc. Bit-identical to
// Qwen3DenseModel::Forward for the same inputs/caches (same op sequence: EmbedInto +
// ForwardLayers). The d128 full-attention capture path is exactly the one the
// already-shipped Qwen3MoeDecodeGraph (Qwen3-Coder) proves capture-safe.
//
// VLLM_CPP_CUDAGRAPH=0 (framework kill switch) disables capture; the per-family
// opt-in env VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH gates whether the registry routes
// here at all (see DenseDecodeGraphEnabled). The 27B/35B/Qwen3-Coder graphs are
// separate drivers and are untouched.
class Qwen3DenseDecodeGraph {
 public:
  // max_num_reqs == the runner's max_num_seqs (carried in ModelForwardInput::
  // gdn_state_slots for every arch); the padded decode batch is capped at it so it
  // never exceeds the captured size set (mirrors vLLM's decode cudagraph dispatch).
  Qwen3DenseDecodeGraph(const Qwen3DenseWeights& weights, const HfConfig& config,
                        vt::Queue queue, int64_t max_num_reqs);
  ~Qwen3DenseDecodeGraph();
  Qwen3DenseDecodeGraph(const Qwen3DenseDecodeGraph&) = delete;
  Qwen3DenseDecodeGraph& operator=(const Qwen3DenseDecodeGraph&) = delete;

  // One PURE-DECODE step. Returns the [B, vocab] f32 logits as a DEVICE-resident
  // ForwardLogits (the captured graph's output stays on device — a view over the
  // slot's persistent logits buffer; the eager fallback owns a pool block), fed
  // straight to the sampler with NO full-logits D2H. Bit-identical to
  // Qwen3DenseModel::Forward for the same inputs/caches. The caller must only route
  // pure-decode batches here (all query_len==1, no prefill).
  ForwardLogits Step(const std::vector<int32_t>& token_ids,
                     const std::vector<int32_t>& positions,
                     const v1::CommonAttentionMetadata& attn_meta,
                     const std::vector<PagedKvCache>& attn_kv);

  // Diagnostics (A/B + tests): is a graph currently captured, and how many replays
  // have run since the last (re)capture.
  bool captured() const;
  int64_t replay_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Per-family gate for the shared dense decode CUDA-graph. Reads
// VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH (DEFAULT ON as of row QUANT-CT-MXFP4-MARLIN-STRUCT
// step 1 — its per-model SACRED token-exact gate PASSED on GB10: paged-engine 184/184
// + async 82/82 graph ON == OFF, and the Qwen3-8B-MXFP4 #44 smoke 3/3 token-exact +
// coherent). An explicit =0 opts back out to eager (byte-identical to the pre-graph
// forward); the framework kill switch VLLM_CPP_CUDAGRAPH=0 also forces eager inside
// the driver. When false the dense factories' forward is LITERALLY unchanged (never
// constructs or routes through a graph).
bool DenseDecodeGraphEnabled();

// SHARED routing helper used by all five dense factory forwards. When this step is
// a graph-eligible pure decode (DenseDecodeGraphEnabled() && input.pure_decode &&
// CUDA static-graph platform), lazily builds `graph` on the caller's LoadedModel
// and returns the graphed [num_reqs, vocab] device logits; otherwise returns
// std::nullopt so the caller runs its existing eager Forward/ForwardDevice path.
std::optional<ForwardLogits> DenseDecodeGraphForward(
    std::unique_ptr<Qwen3DenseDecodeGraph>& graph,
    const Qwen3DenseWeights& weights, const ModelForwardInput& input);

// Per-family config hook (mirrors ParseQwen3_5Config). LoadHfConfig already
// materializes the consumed Qwen3 fields (num_key_value_heads, head_dim,
// rope_theta, intermediate_size, rms_norm_eps, ...); this explicit no-op hook is
// where the family would add normalization/validation without touching the
// registry/runner contract. (tie_word_embeddings / attention_bias are consumed
// by the W2/W3 loader+forward, not by W0/W1.)
void ParseQwen3ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder for the pure-dense arch: exactly ONE full-attention KV
// group, NO MambaSpec/GDN group (Qwen3 dense has no linear-attention layers).
// This is what forces — and validates — the runner's full-attention-only
// generalization (W1): a KVCacheConfig with no mamba group and an empty
// layer_types must allocate + build metadata without the hybrid GDN path.
v1::KVCacheConfig MakeQwen3ForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks);

}  // namespace vllm
