// Gemma-4 text backbone (`Gemma4ForConditionalGeneration`, the language_model
// stack of `unsloth/gemma-4-E4B-it`) — MODEL-GEMMA4 G1. A NEW-primitive dense
// decoder that reuses the Gemma vocabulary but diverges from Gemma-2/3 in FIVE
// load-bearing ways (all grounded in vllm/model_executor/models/gemma4.py @
// 555967922 / 0.26.0.dev0, cross-checked against gemma4.py @ e24d1b24):
//
//   1. PLAIN RMSNorm, NOT GemmaRMSNorm. gemma4.py imports
//      `from vllm.model_executor.layers.layernorm import RMSNorm` (:45) and uses
//      RMSNorm(...) at EVERY norm site — so the weight is applied as `x_norm * w`
//      (NOT the Gemma `(1+w)` offset gemma2/3 use). vt::RmsNormArgs{eps,false}.
//   2. Per-Layer Embeddings (PLE). A second embedding table
//      `embed_tokens_per_layer` [vocab, hidden_size_per_layer_input*num_layers]
//      (gemma4.py:999-1004), a `per_layer_model_projection` [total_ple, hidden]
//      (:1015-1023), a shared `per_layer_projection_norm` (:1025-1028), and per
//      decoder layer a gate/projection/norm (:680-704) that fuses a per-token,
//      per-layer embedding into the residual AFTER the MLP block (:753-761).
//   3. YOCO KV-sharing. The last `num_kv_shared_layers` (18 of 42 for E4B) do
//      NOT cache their own K/V; they attend to the last non-shared layer of the
//      same attention type (gemma4.py:463-489, forward :535-548). For E4B every
//      shared sliding layer targets layer 22, every shared full layer targets 23.
//   4. Heterogeneous head_dim. full_attention layers use global_head_dim=512,
//      sliding_attention layers head_dim=256 (gemma4.py:572-578), with a common
//      num_key_value_heads=2 and scaling=1.0 (Q/K norms carry the scale, :408-411).
//   5. Proportional partial-RoPE on full layers. rope_type "proportional"
//      (gemma4_rope.py) with partial_rotary_factor=0.25 → rotary_dim=128 of a 512
//      head, inv_freq denominator = head_dim (not rotary_dim), non-rotated pairs
//      zero-padded to identity; sliding layers use standard rope (theta 1e4,
//      full 256 rotary). Plus GeGLU MLP (gelu_pytorch_tanh), a sqrt(hidden) embed
//      normalizer, a per-layer learned `layer_scalar` [1] multiply (:707,:765),
//      tied lm_head, and a final logit soft-cap = 30.0.
//
//   E4B disables the harder Gemma-4 primitives: enable_moe_block=false (NO
//   Gemma-4 MoE router / per_expert_scale), attention_k_eq_v=false (NO k_eq_v),
//   use_double_wide_mlp=false. Those stay OUT of this bring-up (recorded in the
//   spec as the follow-on for the ≥12B MoE Gemma-4 checkpoints).
//
// Config (unsloth/gemma-4-E4B-it text_config): hidden 2560, 42L, GQA 8/2,
// head_dim 256 / global_head_dim 512, intermediate 10240,
// hidden_size_per_layer_input 256, num_kv_shared_layers 18, sliding_window 512,
// final_logit_softcapping 30.0, vocab 262144, tie_word_embeddings true.
//
// Numeric contract mirrors the bf16 dense path (dense_attn_block.h): the residual
// stream, all GEMMs, norms, RoPE, flash attention and MLP flow bf16 per-op; paged
// KV bf16. NOTE (honest, G1): the PLE projection combine is accumulated in f32 in
// vLLM (bf16*f32-scalar promotes); we round per-op in bf16 — a named to-verify
// nuance for the strict gate.
#pragma once

#include "vllm/model_executor/models/gemma4_vision.h"
#include "vllm/model_executor/models/gemma4_audio.h"
#include <optional>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/models/gemma4_moe.h"       // MoE AWQ layer weights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// One Gemma-4 self-attention block (gemma4.py::Gemma4Attention). Merged QKV
// (q,k,v_proj), o_proj, per-head Q/K RMSNorm (plain, NOT 1+w) on head_dim, plus a
// WEIGHT-LESS V-norm (has_weight=false, no checkpoint tensor). No bias
// (attention_bias=false). head_dim is per-layer (256 sliding / 512 full). RAW-NK.
struct Gemma4AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh]
  OwnedTensor q_norm;    // bf16 [Dh]  (per-head plain RMSNorm)
  OwnedTensor k_norm;    // bf16 [Dh]
  // v_norm has_weight=false → no tensor; the V-norm is a pure normalization.
};

// Gemma-4 GeGLU MLP (gemma4.py::Gemma4MLP): merged gate_up -> GeluAndMul(tanh) ->
// down. Raw-NK. (E4B: intermediate 10240, single width — no double-wide MLP.)
struct Gemma4MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One Gemma-4 decoder layer (gemma4.py::Gemma4DecoderLayer). Four PLAIN RMSNorms
// (input/post_attention/pre_feedforward/post_feedforward) in a
// standalone-norm-then-explicit-add residual layout (NOT the gemma2/3 fused
// add-norm sandwich), the PLE gate/projection/norm, and the per-layer scalar.
struct Gemma4LayerWeights {
  OwnedTensor input_layernorm;             // bf16 [H]  (standalone plain)
  OwnedTensor post_attention_layernorm;    // bf16 [H]
  OwnedTensor pre_feedforward_layernorm;   // bf16 [H]
  OwnedTensor post_feedforward_layernorm;  // bf16 [H]
  // PLE per-layer components (gemma4.py:680-704).
  OwnedTensor per_layer_input_gate;   // bf16 raw-NK [ple_dim, H]  (Linear H->256)
  OwnedTensor per_layer_projection;   // bf16 raw-NK [H, ple_dim]  (Linear 256->H)
  OwnedTensor post_per_layer_input_norm;  // bf16 [H]  (plain RMSNorm)
  OwnedTensor layer_scalar;               // bf16 [1]  (learned per-layer scalar)
  Gemma4AttnWeights attn;
  Gemma4MlpWeights mlp;
  // Parallel MoE (26B-A4B): empty when enable_moe_block=false (12B dense).
  Gemma4MoeLayerWeights moe;
  bool is_full_attention = false;  // layer_type == "full_attention"
  bool is_kv_shared = false;       // layer_idx >= num_layers - num_kv_shared_layers
  bool k_eq_v = false;             // no v_proj; V shares K (attention_k_eq_v)
  int64_t head_dim = 0;            // 512 full / 256 sliding
  int64_t num_kv_heads = 0;        // may differ full vs sliding (global vs local GQA)
  int64_t kv_target_layer = -1;    // for shared layers: source of K/V (-1 = self)
};

// Whole Gemma-4 text-model weights. tie_word_embeddings defaults TRUE (lm_head
// aliases embed_tokens; the checkpoint has no lm_head.weight). PLE tables live at
// model level (gemma4.py:986-1063).
// Staged connector storage: owned raw [text_hidden, encoder_output] BF16.
// Encoder execution and hook integration remain owed by the E4B baseline spec.
struct Gemma4ModalityWeights {
  nlohmann::json config;  // null when absent; retain encoder and clipping fields
  OwnedTensor embedding_projection;
};

// Only frontend fields are validated here; not an executable full tower config.
struct Gemma4AudioFrontendConfig {
  int64_t hidden_size, sub_ch0, sub_ch1, input_width;
  float rms_norm_eps;
};

struct Gemma4Weights {
  // Complete owned host-F32 audio tower inventory; historical member name.
  multimodal::Gemma4AudioWeights audio_frontend_weights;
  std::optional<multimodal::Gemma4AudioConfig> audio_config;
  std::optional<Gemma4AudioFrontendConfig> audio_frontend_config;
  // Partial inventory: patch tensors and blocks; clipping/execution remain owed.
  multimodal::Gemma4VisionWeights vision_patch_weights;
  // Inventory configuration, not a claim of GQA execution by the MHA forward.
  std::optional<multimodal::Gemma4VisionConfig> vision_config;
  Gemma4ModalityWeights vision;
  Gemma4ModalityWeights audio;
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;             // bf16 [vocab, H]  (scaled by sqrt(H) at use)
  OwnedTensor embed_tokens_per_layer;   // bf16 [vocab, ple_dim*num_layers]
  OwnedTensor per_layer_model_projection;  // bf16 raw-NK [ple_dim*num_layers, H]
  OwnedTensor per_layer_projection_norm;   // bf16 [ple_dim]  (plain RMSNorm)
  OwnedTensor final_norm;               // bf16 [H]  (model.norm, plain RMSNorm)
  OwnedTensor lm_head;                  // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Gemma4LayerWeights> layers;
  // Keeps safetensors mmaps alive for borrowed fused expert tensors (26B MoE).
  std::shared_ptr<const void> shards_keepalive;
};

// Dense / small loads (shards need not outlive weights).
Gemma4Weights LoadGemma4ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// MoE BF16: experts mmap-borrowed; pass FromSafetensorsOwned shared_ptr.
Gemma4Weights LoadGemma4ForConditionalGenerationWeightsOwned(
    std::shared_ptr<const std::vector<SafetensorsFile>> shards,
    const HfConfig& config);

// The Gemma-4 text backbone forward. Per decoder layer (gemma4.py:709-767):
//   r=h; input_layernorm(h) -> attn(Q/K/V plain-norm, V weight-less norm, dual
//   proportional/standard RoPE, per-layer head_dim, YOCO KV-share) ->
//   post_attention_layernorm -> h = attn_n + r; r=h;
//   pre_feedforward_layernorm -> GeGLU MLP -> post_feedforward_layernorm ->
//   h = mlp_n + r; PLE(gate=gelu(gate_lin(h)); h += pln(proj(gate*ple_in)));
//   h *= layer_scalar. Then model.norm (plain) -> tied lm_head -> soft-cap 30.
// The token embeddings are scaled by sqrt(hidden) (bf16) before layer 0, and the
// PLE inputs are precomputed once (gemma4.py:848-898). Returns [n_out, vocab] f32.
//
// G1 HONEST STATUS: the forward is grounded + compiles, but the strict 32/32
// end-to-end gate is BLOCKED on the runner allocating one uniform KV head_dim
// (runner.cpp:600-646) — Gemma-4's per-layer 256/512 heads are not representable
// without a shared-path change to attn_kv_ construction. See gemma4-multimodal.md.
// MODEL-DIFFUSION-LTX25 L3. What `output_hidden_states=True` returns, for the
// consumers that condition on the WHOLE stack rather than on the logits — LTX-2.5's
// text encoder is one (base_encoder.py:68-71: it runs the inner model, takes
// `outputs.hidden_states`, and never touches lm_head).
//
// `hidden_states` has `num_hidden_layers + 1` entries, each [T, H] host f32, in
// transformers' own append order:
//   [0]   the input embeddings, already sqrt(hidden)-scaled
//   [i]   the output of decoder layer i-1, for i in 1..num_hidden_layers-1
//   [L]   model.norm(output of the LAST decoder layer)
// The RAW output of the last decoder layer is NOT in the tuple. A consumer that
// assumes it is gets 49 finite tensors of the right shape and the wrong content,
// which is why this order is stated here and gated in
// tests/vllm/models/test_ltx2_text_encoder.cpp rather than left to the reader.
struct Gemma4HiddenStatesResult {
  // HOST f32, and that is a WIDENING of upstream's dtype — recorded here rather
  // than left as bare fact, per AGENTS.md's dtype-polarity rule. Upstream runs the
  // whole text tower in ONE resolved dtype, bf16 by default
  // (`LTXGemmaTextEncoder.__init__`, base_encoder.py:41
  // `dtype: torch.dtype = torch.bfloat16`), and the `hidden_states` tuple it
  // passes to the feature extractor is bf16. These states are downloaded from BF16
  // device buffers (gemma4.cpp:571-578) and widened on the way out because this is
  // the LTX-2.5 PARITY ARM's carrier: f32 is the dtype the text-tower goldens were
  // produced in, and this tuple is what that gate compares against upstream
  // executed in torch float32. It is NOT the only arm any more. Since
  // LTX25-A24-TEXT-TOWER-BF16 the LTX text entry points accept bf16 as well and
  // refuse everything else BY NAME (ltx2_text_encoder.h, the DTYPE note); bf16 is
  // upstream's own resolved dtype (distilled.py:109) and the arm the render path
  // runs, and it takes these f32 states through `Ltx2StackHiddenStatesBf16`, which
  // narrows them at the boundary rather than computing at the wrong width. The
  // FP8 and NVFP4 arms are A22 — upstream's quantization policies, not its default
  // dtype — and are still OWED, not shipped. Cost of the widening, stated so it is
  // not discovered later: at the shipped 49 x 1024 x 3840 x 4B this holds ~771 MB
  // host where upstream holds ~385 MB, and narrowing this tuple to bf16 at the
  // source is the row that would remove it.
  std::vector<std::vector<float>> hidden_states;
  std::vector<float> logits;  // [n_out, vocab], as Forward() returns
};

class Gemma4Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // The same forward as Forward(), additionally returning every hidden state in
  // transformers' order (see Gemma4HiddenStatesResult). Capture is OFF on every
  // other entry point, so no shipped path changes shape or cost.
  static Gemma4HiddenStatesResult ForwardHiddenStates(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // CLAIM-GEMMA4-MM-E2E: one MULTIMODAL forward. The hidden stream starts from the
  // already-merged bf16 DEVICE inputs_embeds [T, H] (text rows sqrt(H)-scaled,
  // image/audio soft-token rows the embed_vision/embed_audio projector output), and
  // the PLE embed_tokens_per_layer lookup uses the DEVICE `ple_token_ids` [T] (mm
  // rows masked to 0 + the vocab_size_per_layer_input range mask) —
  // gemma4_mm.py:1962-1973 + gemma4.py get_per_layer_inputs/
  // project_per_layer_inputs (:848-928). Both handles are BORROWED device views
  // (ENG-MM-INPUT-PIPELINE P1, #1358/#2300); see MultiModalForwardInput. Positions
  // are Gemma-4's standard 1-D positions (NO MRoPE, NO DeepStack). Returns
  // [n_out, vocab] host f32 (n_out = |logits_indices|, 1 for the last row).
  static std::vector<float> ForwardMm(
      const vt::Tensor& inputs_embeds,
      const vt::Tensor& ple_token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// ── CLAIM-GEMMA4-MM-E2E: the registered engine mm-forward driver + adapters ──

// Registered-path greedy IMAGE->text generation for Gemma-4: EVERY decoder step is
// driven through ModelRegistry::Forward(model, input) with input.mm carrying the
// merged inputs_embeds + Gemma-4 PLE-masked ids (positions are the 1-D
// ModelForwardInput::positions) — i.e. it exercises the ENGINE registered
// mm-forward (ForwardGemma4ForConditionalGeneration's mm branch). Mirrors
// Qwen3VLGenerateGreedyViaRegistry (the Qwen3-VL fold): the prefill builds the
// merged embeds (embed(prompt_ids)*sqrt(H) with `mm_projected` scattered into the
// image-placeholder rows), then greedy-decodes through the registered forward.
//
// prompt_ids  : placeholder-expanded model input ids (image_token_id repeated N
//               times at the image span).
// mm_projected: the vision embed_vision projector output [N, text_hidden] (== the
//               masked-scatter merge input), host f32, bf16-rounded before merge.
// image_token_id / eos_token_id : the Gemma-4 <image> soft-token id + eos.
// `model` MUST be the registered Gemma-4 LoadedModel loaded from `weights`.
// `out_margins` (optional): per generated token, the greedy top1-top2 logit margin
// — used by the e2e gate to classify a divergence as a bf16 near-tie (small margin)
// vs a structural error (large margin), per the ratified near-tie methodology.
std::vector<int32_t> Gemma4GenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_projected, int32_t image_token_id,
    int32_t eos_token_id, const Gemma4Weights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens,
    std::vector<float>* out_margins = nullptr);

// The FULL-attention layers' "proportional" rope cos|sin table, on host in f32 —
// the exact values the forward builds (BuildProportionalRopeCache rounds them to
// bf16 to match the q/k it rotates; nothing else differs). Returns
// [max_pos + 1, head_dim]: the first head_dim/2 columns are cos and the second
// half sin, over the head_dim/2 DISTINCT angle pairs, mirroring upstream's
// `emb = cat((freqs, freqs))` with each angle stored once
// (`Gemma4UnifiedTextRotaryEmbedding.forward`, modeling_gemma4_unified.py:259-275
// — the `cat` itself is :271; the inv_freq it consumes comes from
// modeling_rope_utils.py:187-254).
//
// This is a GATE SURFACE, and it exists because of a measurement rather than a
// preference. `partial_rotary_factor` decides how many angle pairs are rotated
// and how many are zero-padded to identity, and it is the one field on this path
// that the tower's hidden states cannot resolve: on the reduced LTX tower
// fixture, forcing it from the config's 0.25 to 1.0 displaces the worst hidden
// state by 1.09e-01 against that state's measured bf16 noise floor of 9.99e-02
// — a ratio of 1.09, inside the tolerance the states are gated at — and a LARGER
// fixture makes it worse, not better (0.65 at head_dim 16/32, seq 32), because
// bf16 accumulation noise grows at least as fast as the rope contribution. So
// the states are the wrong instrument and this table is the right one: f32, no
// accumulation, compared element-wise against the oracle's own rotary embedding.
std::vector<float> Gemma4ProportionalRopeCosSin(const HfConfig& config,
                                                int64_t head_dim, int64_t max_pos);

// Wrap already-loaded Gemma-4 weights in the registered LoadedModel so a caller
// that owns the weights (the mm e2e gate) can drive ModelRegistry::Forward without
// re-reading the checkpoint. `Make` OWNS the moved weights; `Borrow` does NOT own
// `weights` (it must outlive the model — used so the driver and the model share ONE
// Gemma4Weights on the unified-memory box). Mirrors Make/BorrowQwen3VLLoadedModel.
std::unique_ptr<LoadedModel> MakeGemma4LoadedModel(Gemma4Weights weights);
std::unique_ptr<LoadedModel> BorrowGemma4LoadedModel(const Gemma4Weights& weights);

// Per-family config hook (mirrors ParseGemma3ForCausalLMConfig). The engine's
// HfConfig loader already descends into `text_config` (hf_config.cpp:103-113), so
// the typed fields + config.raw are the text sub-dict; the Gemma-4-specific
// scalars (layer_types, rope_parameters, hidden_size_per_layer_input,
// num_kv_shared_layers, global_head_dim, final_logit_softcapping) are read from
// config.raw by the loader/forward. No-op hook.
void ParseGemma4ForConditionalGenerationConfig(const HfConfig& config);

// KV-cache spec builder. Emits TWO groups reflecting the true topology: sliding
// layers (head_dim 256) and full-attention layers (head_dim 512). NOTE: the
// current runner reads a single uniform head_dim, so consuming a two-head-dim
// spec is the named G-next shared-path change; this builder documents the intent.
v1::KVCacheConfig MakeGemma4ForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks);

// T03B: typed read-only view of the Gemma-4 multimodal special token IDs from
// the wrapper config.json. The eight multimodal soft-token IDs and the
// `eos_token_id` array live at the top level of the ForConditionalGeneration
// wrapper; `bos_token_id`, `pad_token_id` and `vocab_size` live under the
// nested `text_config` sub-dict (the same split LoadHfConfig already resolves
// for the typed text fields). This is a pure read of `config.raw`; it does not
// change any serving/forwarding path.
struct Gemma4SpecialTokenIds {
  int32_t audio_token_id = 0;
  int32_t boa_token_id = 0;
  int32_t boi_token_id = 0;
  int32_t eoa_token_id = 0;
  int32_t eoa_token_index = 0;
  int32_t eoi_token_id = 0;
  int32_t image_token_id = 0;
  int32_t video_token_id = 0;
  int32_t bos_token_id = 0;
  int32_t pad_token_id = 0;
  int32_t vocab_size = 0;
  // Top-level `eos_token_id` is a JSON array (e.g. [1, 106]).
  std::vector<int32_t> eos_token_ids;

  explicit Gemma4SpecialTokenIds(const HfConfig& config);
};

// T03D: build the Gemma-4 video token replacement sequence for a list of frames.
// Mirrors Python get_video_repl in gemma4_mm.py:412-447. For each frame i with
// timestamp ts (seconds) and n soft tokens, emits encode(prefix) where prefix is
// "MM:SS " for i==0 and " MM:SS " for i>0 (minutes=int(ts//60), seconds=int(ts%60),
// both zero-padded to 2 digits), followed by boi, video repeated n times, eoi.
// `encode` tokenizes the prefix string. Returns the concatenated token ids.
// Throws (via VT_CHECK) if timestamps and num_soft_tokens_per_frame differ in
// length. Empty inputs return an empty result.
std::vector<int32_t> Gemma4VideoReplacement(
    const std::vector<double>& timestamps,
    const std::vector<int32_t>& num_soft_tokens_per_frame,
    int32_t boi_token_id, int32_t eoi_token_id, int32_t video_token_id,
    const std::function<std::vector<int32_t>(std::string_view)>& encode);

}  // namespace vllm
