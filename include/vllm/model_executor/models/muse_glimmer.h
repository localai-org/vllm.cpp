// Muse Glimmer (`MuseGlimmerForConditionalGeneration`) — the ADDITIVE model TU
// skeleton for the Muse Glimmer structural bring-up (`CLAIM-MUSE-GLIMMER-W0`).
// This header defines the config parse (flat/nested normalization + the iRoPE
// layer mask + the dual query-pre-scale schema), the structural weight name map
// for both checkpoint conventions, and the forward / KV-cache seams. The forward
// REFUSES-by-name (`VT_CHECK(false, ...)`) exactly like `kimi_k3.{h,cpp}` and
// `deepseek_v4.{h,cpp}` — the TU BUILDS and the config/name-map structure is
// unit-testable, but a forward LOUDLY reports the pending brick rather than
// returning a silent wrong answer.
//
// ─── OFF-PIN HONESTY (up front) ──────────────────────────────────────────────
// Meta released Muse Glimmer on 2026-08-08, AFTER the parity pin `555967922`
// (2026-07-26). There is no `muse_glimmer` code at the pin, and none on vLLM
// `main` either: the ONLY upstream implementation is the still-OPEN, approved-
// but-CI-red PR vllm#51655 at head `075d645af`. Every `file:line` below points at
// that BRANCH HEAD, not at the pin — a deliberate exception on explicit developer
// direction, recorded as porting-inventory §9 deviation 16.
//
// Consequently the pinned oracle CANNOT load this model, there is no honest
// throughput denominator, and **no speed axis is claimable** for Muse Glimmer
// until #51655 merges and the pin advances. Correctness gates against the HF
// reference instead. A green CPU build is STRUCTURE, not execution evidence.
//
// ─── WHAT THIS IS A PORT OF (file:line, @ vllm#51655 head 075d645af) ──────────
//   OURS                              <-  UPSTREAM
//   MuseGlimmerTextParams             <-  transformers_utils/configs/muse_glimmer.py:29-126
//                                         (MuseGlimmerTextConfig)
//   MuseGlimmerVisionParams           <-  configs/muse_glimmer.py:129-176
//                                         (MuseGlimmerVisionConfig)
//   ParseMuseGlimmerParams            <-  configs/muse_glimmer.py:186-305 (the flat ->
//                                         canonical normalization) + :20-26
//                                         (_default_no_rope_layers)
//   ResolveMuseGlimmerQueryPreScale   <-  models/muse_glimmer.py:472-517
//   MuseGlimmerUseQkNorm/OutputGate   <-  models/muse_glimmer.py:456-469
//   EnumerateMuseGlimmerTensors       <-  models/muse_glimmer.py:1271-1300 (text),
//                                         :692-739 + :651-668 (vision tower),
//                                         :1036-1044 (adapter), :1440-1486 (wrapper)
//   NormalizeMuseGlimmerWeightName    <-  models/muse_glimmer.py:1389-1425
//                                         (hf_to_vllm_mapper)
//   MuseGlimmerModel::Forward         <-  models/muse_glimmer.py:1304-1345,
//                                         :1604-1613 — REFUSE-by-name
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/muse_glimmer_vision.h"  // the W3 perception tower
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Which checkpoint naming convention a weight name came from. The discriminator
// is the PREFIX, and it is load-bearing: the two families give the sandwich norms
// DIFFERENT MEANINGS (see `NormalizeMuseGlimmerWeightName`).
enum class MuseGlimmerCheckpointConvention {
  kCanonical,  // `model.language_model.layers.N.*` — norms already correct
  kLegacyGuac,  // `model.layers.N.*` — `post_attention_layernorm` is really the
                // PRE-feedforward norm; `post_attn_norm` is the true post-attn one
};

// ─── THE ARCHITECTURE'S OWN CONSTANTS (issue #412) ───────────────────────────
// An ABSENT key falls back to the ARCHITECTURE's value, never to a neutral one
// (identity / zero / "off"). Both independent references agree on all six, and
// both express them as CLASS DEFAULTS on the config — which is precisely the
// value a checkpoint that omits the key gets:
//
//   vLLM #51655 @ 075d645af  transformers_utils/configs/muse_glimmer.py
//                              :45 rms_norm_eps            1e-5
//                              :56 sliding_window          2048
//                              :58 final_logit_softcapping 20.0
//                              :62 qk_scale_factor         43.7840518911
//                              :65 output_multiplier       0.19611613513818404
//                              :67 post_norm_eps           1e-8
//   SGLang @ 38a1bc5d2f      python/sglang/srt/configs/muse_glimmer.py
//                              :90, :93, :102 (output_soft_cap_temp), :98, :101, :91
//
// Defaulting them to neutral values does not error and does not produce
// gibberish — it silently runs a DIFFERENT model. The released 30B `config.json`
// carries all six, which is exactly why the defaults went untested; the two
// checkpoints that DO omit keys are the released GGUF (no post-norm epsilon key
// at all, so both sandwich post-norms ran 1000x too loose) and the DFlash
// drafter's `config.json` (no `qk_scale_factor`, `post_norm_eps`,
// `output_multiplier`, `final_logit_softcapping`, `vocab_size`).
inline constexpr float kMuseGlimmerDefaultRmsNormEps = 1e-5f;
inline constexpr float kMuseGlimmerDefaultPostNormEps = 1e-8f;
inline constexpr int64_t kMuseGlimmerDefaultSlidingWindow = 2048;
inline constexpr double kMuseGlimmerDefaultOutputMultiplier = 0.19611613513818404;
inline constexpr double kMuseGlimmerDefaultFinalLogitSoftcapping = 20.0;
// The NATIVE (unfolded) query scale. What the forward applies is this divided by
// `sqrt(head_dim)` — exactly 3.87 at the released head_dim 128 — so unlike the
// five above it cannot be a plain member default: it depends on the geometry.
// `ResolveMuseGlimmerQueryPreScale` resolves it.
inline constexpr double kMuseGlimmerDefaultQkScaleFactor = 43.7840518911;

// The Muse Glimmer text tower. Values in comments are the released
// `meta-models/Muse-Glimmer-30B` scale.
struct MuseGlimmerTextParams {
  int64_t vocab_size = 0;             // 202048
  int64_t hidden_size = 0;            // 6656
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;      // 52
  int64_t num_attention_heads = 0;    // 32
  int64_t num_key_value_heads = 0;    // 2  (GQA 16:1)
  int64_t head_dim = 0;               // 128
  int64_t max_position_embeddings = 0;  // 131072
  // 0 means NO WINDOW AT ALL (muse_glimmer.cpp:229), which is why it cannot be
  // the absent-key fallback — that is a different model, not a looser one.
  int64_t sliding_window = kMuseGlimmerDefaultSlidingWindow;
  double rope_theta = 500000.0;
  bool tie_word_embeddings = false;

  // Sandwich norms: pre-norms use `rms_norm_eps`, post-norms use the separate,
  // SMALLER `post_norm_eps` (muse_glimmer.py:1236-1247) — 1e-5 against 1e-8 in
  // the released config, three orders apart. Both norms carry a baked `+1`
  // weight offset and are computed in fp32.
  float rms_norm_eps = kMuseGlimmerDefaultRmsNormEps;
  float post_norm_eps = kMuseGlimmerDefaultPostNormEps;

  // iRoPE mask, one entry per layer (configs/muse_glimmer.py:20-26, 108-112):
  //   0 => NoPE   AND full attention
  //   1 => RoPE   AND sliding-window attention
  // Default is NoPE every 4th layer counted BACKWARD from the last layer.
  std::vector<int64_t> no_rope_layers;

  // Post-QK-norm query pre-scale, already normalized across the two config
  // schemas (see `ResolveMuseGlimmerQueryPreScale`). The softmax scaling stays
  // `head_dim ** -0.5` and is NOT folded in here.
  double scale_query_by = 1.0;

  // Both default to TRUE. The modular schema OMITS them, so they read as null
  // and only an explicit `false` disables them (muse_glimmer.py:456-469).
  bool use_qk_norm = true;
  bool use_attn_output_gate = true;

  // Defaults TRUE when absent, like the two flags above (#405): the released
  // config omits it, and upstream gates `perception_emb_norm` on it.
  bool normalize_tok_embeddings = true;
  double output_multiplier = kMuseGlimmerDefaultOutputMultiplier;
  // 0 == none. As with `sliding_window`, that is a state a config reaches only
  // by saying so explicitly (`"final_logit_softcapping": null`), never by
  // omitting the key.
  double final_logit_softcapping = kMuseGlimmerDefaultFinalLogitSoftcapping;
  std::string hidden_activation = "silu";
};

// The perception encoder. Parsed in full so the arch RESOLVES and the tower's
// tensor names are enumerable; the tower forward is a W3 residual.
struct MuseGlimmerVisionParams {
  bool present = false;
  int64_t patch_size = 14;
  int64_t pos_emb_height = 32;
  int64_t pos_emb_width = 32;
  int64_t num_attention_heads = 16;
  int64_t num_hidden_layers = 50;
  int64_t hidden_size = 1536;
  int64_t intermediate_size = 8960;
  int64_t merge_kernel_size = 2;
  int64_t output_dim = 6144;
  int64_t patch_temporal = 2;
  int64_t adapter_dim = 4096;
  float layer_norm_eps = 1e-5f;
  // "full_attention" / "sliding_attention" per layer. Upstream's vision default
  // is full every 4th layer AND on the last layer (configs/muse_glimmer.py:168-176)
  // — note that is a DIFFERENT rule from the text tower's backward-counted mask.
  std::vector<std::string> layer_types;
};

struct MuseGlimmerParams {
  MuseGlimmerTextParams text;
  MuseGlimmerVisionParams vision;
  int64_t image_token_id = 200092;
  int64_t video_token_id = 200091;
};

// Resolve + validate MuseGlimmerParams from an HfConfig. Pure/host —
// unit-testable without a checkpoint. Accepts BOTH the canonical nested layout
// (`text_config` / `vision_config`) and the older FLAT layout, normalizing the
// latter (configs/muse_glimmer.py:186-305). That normalization is not cosmetic:
// without it a flat config deserializes to an ALL-DEFAULT text config, silently
// ignoring every checkpoint value and building a wrong-shaped model with no
// error. Throws with a precise message on a missing required field.
MuseGlimmerParams ParseMuseGlimmerParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`).
void ParseMuseGlimmerConfig(const HfConfig& config);

// The post-QK-norm query pre-scale, normalized across the two config schemas
// (muse_glimmer.py:472-517). HF native modeling ships the RAW `qk_scale_factor`
// (~43.784 at head_dim 128) and folds `1/sqrt(head_dim)` itself; the modular
// `text_config` ships the value ALREADY folded (~3.87). Both must yield the same
// ~3.87. Upstream disambiguates by MAGNITUDE against `sqrt(head_dim)`, and an
// explicit `scale_query_by` wins outright. Getting this wrong scales every query
// by ~11.3x, so it is exercised RED-first from both schemas.
double ResolveMuseGlimmerQueryPreScale(double qk_scale_factor,
                                       bool has_qk_scale_factor,
                                       double explicit_scale_query_by,
                                       bool has_explicit_scale, int64_t head_dim);

// The default iRoPE mask when the checkpoint omits `no_rope_layers`:
// NoPE every 4th layer counted BACKWARD from the last (configs/muse_glimmer.py:20-26).
std::vector<int64_t> DefaultMuseGlimmerNoRopeLayers(int64_t num_hidden_layers);

// Normalize ONE checkpoint weight name to our canonical internal name, mirroring
// upstream's `hf_to_vllm_mapper` (muse_glimmer.py:1389-1425). Returns false when
// the name is dropped outright (upstream maps `model.rotary_emb.` to None).
//
// TWO ordering hazards are load-bearing here and are covered by tests:
//   1. `.self_attn.gate_proj` -> `.self_attn.output_gate_proj` MUST be applied
//      before any `.gate_proj` -> `.gate_up_proj` MLP stacking rule. The
//      attention OUTPUT GATE and the MLP gate share a suffix; folding the former
//      into `gate_up_proj` would silently corrupt both.
//   2. For a kLegacyGuac checkpoint the sandwich norms must be renamed in the
//      right order: legacy `post_attention_layernorm` is really the
//      PRE-feedforward norm and must be renamed FIRST, before legacy
//      `post_attn_norm` becomes `post_attention_layernorm` — otherwise the
//      second rule's output is re-captured by the first and the two norms swap.
bool NormalizeMuseGlimmerWeightName(const std::string& name, std::string* out);

// Which convention a raw checkpoint name belongs to, by PREFIX
// (muse_glimmer.py:1379-1381).
MuseGlimmerCheckpointConvention MuseGlimmerConventionOf(const std::string& name);

// The structural weight name map, in our canonical post-normalization names.
// Grounded 1:1 in the module tree at vllm#51655 head `075d645af`. Weightless
// modules (`embed_norm`, the per-head `qk_norm`, `perception_emb_norm`) contribute
// NO tensor, which is why the counts are smaller than a Gemma-shaped tower's.
std::vector<std::string> EnumerateMuseGlimmerTensors(const MuseGlimmerParams& params);

// One Muse Glimmer self-attention block (muse_glimmer.py:1082-1215). Merged QKV
// (`qkv_proj` <- [q,k,v]_proj, no bias), and — the delta vs every Gemma — a
// SEPARATE per-head output-gate projection whose input is the NORMED LAYER INPUT,
// not the attention output (:1203-1206). The QK-norm is WEIGHTLESS (:1121), so it
// contributes no tensor here.
struct MuseGlimmerAttnWeights {
  OwnedTensor qkv_proj;         // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;           // bf16 raw-NK [H, Hq*Dh]
  OwnedTensor output_gate_proj;  // bf16 raw-NK [Hq*Dh, H]; EMPTY when the gate is off
};

// Muse Glimmer SwiGLU MLP (muse_glimmer.py:1046-1079): merged gate_up ->
// SiluAndMul -> down. `hidden_activation` is asserted `silu` at config parse —
// this is the delta vs Gemma-2's GeGLU sibling.
struct MuseGlimmerMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One decoder layer (muse_glimmer.py:1218-1277). FOUR sandwich RMSNorms, all with
// the baked `+1` weight offset, but on TWO different epsilons: the two PRE norms
// take `rms_norm_eps`, the two POST norms `post_norm_eps` (:1236-1247).
struct MuseGlimmerLayerWeights {
  OwnedTensor input_layernorm;             // bf16 [H]  pre  (eps = rms_norm_eps)
  OwnedTensor post_attention_layernorm;    // bf16 [H]  post (eps = post_norm_eps)
  OwnedTensor pre_feedforward_layernorm;   // bf16 [H]  pre  (eps = rms_norm_eps)
  OwnedTensor post_feedforward_layernorm;  // bf16 [H]  post (eps = post_norm_eps)
  MuseGlimmerAttnWeights attn;
  MuseGlimmerMlpWeights mlp;
};

// The materialized perception encoder (W4 WIRING). The tower itself is the W3
// `vllm::multimodal` code, which owns host-f32 weight structs, so the loader
// converts the checkpoint's bf16 bytes to f32 here rather than inventing a second
// tower representation. That costs 2x the tower's on-disk footprint in host RAM
// (~3.7 GiB bf16 -> ~7.4 GiB f32 at the released 30B scale) — a named residual,
// not a design claim: collapsing it means teaching the W3 tower a bf16 weight
// struct, which is a change to the sibling-owned tower gate.
//
// `vision_projection` is torch storage order [text_hidden, adapter_dim]
// (muse_glimmer.py:1464-1468, `nn.Linear(adapter_dim, hidden_size, bias=False)`).
// `perception_emb_norm` (:1469-1473) is a WEIGHTLESS RMSNorm applied only when
// `normalize_tok_embeddings` is set, and `nn.Identity()` otherwise — it holds no
// tensor either way, which is why it is absent from the enumeration.
struct MuseGlimmerVisionTower {
  bool loaded = false;
  multimodal::MuseGlimmerVisionConfig cfg{};
  multimodal::MuseGlimmerVisionWeights encoder{};
  multimodal::MuseGlimmerVisionAdapterWeights adapter{};
  std::vector<float> projection;  // [text_hidden, adapter_dim]
};

// Whole Muse Glimmer weights. W0 carried only the resolved params + the loader's
// structural accounting; W1 added the materialized TEXT tower; W4 adds the
// materialized PERCEPTION ENCODER (`vision`), so a Muse Glimmer forward is no
// longer text-only.
//
// `model.embed_norm` (:1286) and the per-head `qk_norm` (:1121) are WEIGHTLESS and
// deliberately hold no tensor; the forward realizes them as `vt::RmsNorm` against a
// ones weight (spec §9).
struct MuseGlimmerWeights {
  MuseGlimmerParams params{};
  int64_t accounted_tensors = 0;
  int64_t enumerated_tensors = 0;

  // W1 text tower. `text_loaded` is false for a params-only struct (the W0
  // accounting form, and every scaffold-level unit test) — the forward refuses on
  // it BY NAME rather than reading empty tensors.
  bool text_loaded = false;
  OwnedTensor embed_tokens;  // bf16 [V,H]   (:1280)
  OwnedTensor final_norm;    // bf16 [H]     (:1296) — NOTE: NO `+1` offset here
  OwnedTensor lm_head;       // bf16 [H,V] Matmul-B (:1480) — UNTIED
  std::vector<MuseGlimmerLayerWeights> layers;

  // W4: the perception encoder. `vision.loaded` is false for a text-only
  // checkpoint (no `vision_config`) and for the params-only accounting form.
  MuseGlimmerVisionTower vision;

  // #607 L3, the TOWER SKIP. True when this checkpoint DOES carry a perception
  // encoder and the load deliberately did not read it, because every modality
  // it serves was at limit 0 (interfaces.py:288-293). `vision.loaded` is false
  // in that case too, which is why the two are separate: they have different
  // fixes, and the refusal message must be able to tell a reader which one they
  // hit. `params.vision.present && vision_skipped` is "you asked for zero
  // limits"; `!params.vision.present` is "this checkpoint has no encoder".
  bool vision_skipped = false;
};

// Load `MuseGlimmerForConditionalGeneration` safetensors. Performs the W0
// structural accounting pass, materializes the W1 text tower (both checkpoint
// naming conventions, via `NormalizeMuseGlimmerWeightName`), and — W4 — the
// perception encoder when the config carries a `vision_config`. The vision
// attention's separate on-disk `q/k/v` shards are FUSED here into the tower's
// merged `[3*hidden, hidden]` operand, in q|k|v row order, mirroring upstream's
// `packed_modules_mapping` (muse_glimmer.py:1427-1430).
//
// `mm_config` (#607 L3) is the engine's multimodal input limits, BORROWED. When
// every modality the perception encoder serves is at limit 0 the encoder is
// CONSTRUCTED — its geometry is still parsed off `vision_config`, so a refusal
// can still name what is missing — but its tensors are never read, mirroring
// `_mark_tower_model`'s `no_init_weights` (interfaces.py:288-293). NULL, the
// default, loads the encoder exactly as before.
MuseGlimmerWeights LoadMuseGlimmerForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    const MultiModalConfig* mm_config = nullptr);

// The Muse Glimmer forward. W1 implements the TEXT tower
// (muse_glimmer.py:1218-1345, :1604-1613):
//   embed -> WEIGHTLESS embed_norm (NOT Gemma's sqrt(hidden) scale, :1286)
//   per layer: input_layernorm (fused add, +1, rms_norm_eps)
//              -> attn: merged qkv, weightless per-head QK-norm in fp32 BEFORE
//                 RoPE, query pre-scale `scale_query_by` on q only, iRoPE
//                 (`no_rope_layers[l]==1` => RoPE AND sliding window; `==0` =>
//                 NoPE AND full attention), softmax scale head_dim**-0.5,
//                 attn * sigmoid(output_gate_proj(normed layer input)), o_proj
//              -> post_attention_layernorm (STANDALONE, +1, post_norm_eps)
//              -> pre_feedforward_layernorm (fused add, +1, rms_norm_eps)
//              -> SwiGLU MLP
//              -> post_feedforward_layernorm (STANDALONE, +1, post_norm_eps)
//   final norm (fused add, NO offset, rms_norm_eps) -> UNTIED lm_head
//   -> * output_multiplier -> final_logit_softcapping
// Returns [n_out, vocab] f32.
//
// W4 WIRING adds `ForwardMm`, the `inputs_embeds` branch upstream's model forward
// already has (:1311-1315), so the perception encoder is REACHABLE: an image or
// video prompt now runs instead of refusing.
//
// NOT ESTABLISHED: no token-exact e2e claim, for text OR for image/video. The
// pinned oracle cannot load `muse_glimmer` at all (spec §0), so there is neither a
// golden nor a speed denominator; the evidence here is structural + per-mechanism
// unit level. "The tower is reachable and its output lands on the placeholder
// rows" is NOT "an image produces the right tokens".
class MuseGlimmerModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  // W4 WIRING: the MULTIMODAL entry. Takes the ALREADY-MERGED input embeddings
  // (bf16, [T, hidden] row-major, on DEVICE — a BORROWED view, see
  // MultiModalForwardInput and ENG-MM-INPUT-PIPELINE P1 #1358/#2300) instead of
  // token ids, exactly as upstream's `MuseGlimmerModel.forward` takes
  // `inputs_embeds` (muse_glimmer.py:1311-1315).
  //
  // NOTE the asymmetry that upstream's two branches encode and that the caller
  // therefore owns: `embed_input_ids` (:1301-1302) is `embed_norm(embed_tokens(ids))`,
  // but the `inputs_embeds` branch applies NO embed_norm. The vision soft tokens
  // must not be re-normalized (their own `perception_emb_norm` already ran, or is
  // Identity), so the norm belongs to the TEXT rows only — which is precisely what
  // `MuseGlimmerMergeMultimodalEmbeds` below builds. Everything downstream of the
  // embedding is shared with the text path, so a text-only prompt routed through
  // here is BIT-IDENTICAL to `Forward` (gated in test_muse_glimmer_wiring.cpp).
  static std::vector<float> ForwardMm(
      const vt::Tensor& inputs_embeds,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// The perception encoder, adapter and projection ALONE, read from the shards
// through the same name normalization and slot loader the full-model load uses
// (`LoadMuseGlimmerForConditionalGenerationWeights` calls the same
// `LoadVisionTower`). A vision reference run needs only these ~3.7 GB, not the
// 60 GB text tower. Throws BY NAME when a tensor is missing or the checkpoint
// declares no `vision_config`.
MuseGlimmerVisionTower LoadMuseGlimmerVisionTower(const std::vector<SafetensorsFile>& shards,
                                                  const MuseGlimmerParams& params);

// The perception-encoder geometry, bridged from the resolved model config to the
// W3 tower's own config struct. `compute_dtype` stays the tower default (bf16 —
// what the checkpoint ships and what production runs).
multimodal::MuseGlimmerVisionConfig MuseGlimmerVisionConfigOf(
    const MuseGlimmerParams& params);

// Which prompt rows are multimodal placeholders. BOTH the image (200092) and the
// video (200091) token are placeholders and share one soft-token stream, mirroring
// `configure_mm_token_handling(vocab, [image_token_id, video_token_id])`
// (muse_glimmer.py:1496-1498) and `embed_multimodal` appending both modalities into
// ONE `MultiModalEmbeddings` list (:1592-1602).
std::vector<bool> MuseGlimmerMultimodalMask(const std::vector<int32_t>& token_ids,
                                            const MuseGlimmerParams& params);

// `_encode_pixel_groups` (muse_glimmer.py:1548-1569): encoder -> adapter ->
// vision_projection -> perception_emb_norm. Returns host f32 [total_tokens,
// text_hidden] — the soft tokens, in image order, ready to scatter. Throws BY NAME
// when the checkpoint carries no perception encoder (:1553-1559).
std::vector<float> MuseGlimmerEncodePixelGroups(
    const std::vector<multimodal::MuseGlimmerVisionImage>& images,
    const MuseGlimmerWeights& weights, vt::Queue& queue);

// `SupportsMultiModal.embed_input_ids`: `embed_norm(embed_tokens(ids))` with the
// placeholder rows masked-scattered from `mm_embeds` [N, hidden] host f32. Returns
// bf16 bits [T, hidden], the exact input `ForwardMm` wants. `mm_embeds` may be
// empty, in which case this is plain `embed_input_ids` and the result reproduces
// the text path bit-for-bit.
std::vector<uint16_t> MuseGlimmerMergeMultimodalEmbeds(
    const std::vector<int32_t>& token_ids, const std::vector<float>& mm_embeds,
    const MuseGlimmerWeights& weights, vt::Queue& queue);

// Single-sequence greedy image->text driver, mirroring the Gemma-4 fold
// (`Gemma4GenerateGreedyViaRegistry`, gemma4_mm.cpp): EVERY step goes through
// `ModelRegistry::Forward` with `ModelForwardInput.mm` set, so the ENGINE's
// registered mm branch drives decode rather than a bespoke in-TU forward.
//
// HONESTY: this makes an image prompt RUN. It does NOT establish that the tokens
// are correct — the pinned oracle cannot load `muse_glimmer` at all, so there is
// no reference decode to compare against and no speed denominator either
// (specs/muse-glimmer.md §0).
std::vector<int32_t> MuseGlimmerGenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<multimodal::MuseGlimmerVisionImage>& images,
    int32_t eos_token_id, const MuseGlimmerWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens);

// KV-cache spec builder: ONE full-attention group over the uniform GQA geometry.
// W1 RESOLVED the W0 placeholder note: the RoPE layers are sliding-window and the
// NoPE layers are full attention (muse_glimmer.py:1167-1168), but BOTH classes have
// the SAME num_key_value_heads and head_dim, so the only per-layer difference is
// the WINDOW — which is applied at the attention-kernel level
// (`vt::PagedAttentionArgs::window_size`), exactly as Gemma-2/Laguna do for their
// interleaved sliding layers. No heterogeneous per-layer spec is needed; the
// Gemma-4 per-layer seam exists for models whose KV GEOMETRY differs per layer,
// which Muse Glimmer's does not.
v1::KVCacheConfig MakeMuseGlimmerKVCache(const HfConfig& config, int block_size,
                                         int num_blocks);

}  // namespace vllm
