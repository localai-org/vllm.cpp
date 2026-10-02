// Muse Glimmer config resolution + structural weight name map (W0). This TU
// implements `ParseMuseGlimmerParams` (flat/nested config normalization, the
// iRoPE mask, the dual query-pre-scale schema), `NormalizeMuseGlimmerWeightName`
// (the two checkpoint conventions), the pure `EnumerateMuseGlimmerTensors`
// structural map, and the loader's accounting entry.
//
// ─── OFF-PIN HONESTY ─────────────────────────────────────────────────────────
// Ported from vllm#51655 head `075d645af`, an OPEN and CI-red upstream PR, NOT
// from the parity pin `555967922` (Muse Glimmer did not exist at the pin). See
// porting-inventory §9 deviation 16. The pinned oracle cannot load this model, so
// there is no speed denominator and no speed claim is available from this row.
//
// ─── CANONICAL NAME MAP (post-normalization, muse_glimmer.py @ 075d645af) ─────
//   MODEL LEVEL   model.embed_tokens.weight            (:1280)
//                 model.norm.weight                    (:1296)
//                 lm_head.weight                       (:1480, absent if tied)
//     NOTE `model.embed_norm` is WEIGHTLESS (:1286) — contributes no tensor.
//   PER LAYER N   model.layers.N.input_layernorm.weight            (:1236)
//                 model.layers.N.post_attention_layernorm.weight   (:1239)
//                 model.layers.N.pre_feedforward_layernorm.weight  (:1242)
//                 model.layers.N.post_feedforward_layernorm.weight (:1245)
//                 model.layers.N.self_attn.{q,k,v,o}_proj.weight   (:1126-1141)
//                 model.layers.N.self_attn.output_gate_proj.weight (:1145, gated)
//                 model.layers.N.mlp.{gate,up,down}_proj.weight    (:1047-1075)
//     NOTE the per-head `qk_norm` is WEIGHTLESS (:1121) — contributes no tensor.
//   VISION        vision_encoder.conv1_linear.weight               (:710)
//                 vision_encoder.positional_embedding_vlm          (:711)
//                 vision_encoder.{ln_pre,ln_post}.{weight,bias}    (:714,:732)
//                 vision_encoder.transformer.N.{ln_1,ln_2}.{weight,bias} (:661,:666)
//                 vision_encoder.transformer.N.attn.{q,k,v,o}_proj.{weight,bias}
//                                                                  (:574-589,:1397)
//                   — SEPARATE on disk, WITH bias; upstream's merged `qkv_proj`
//                     module is a LOAD-time fusion (packed_modules_mapping,
//                     :1427-1430), which is what our loader reproduces.
//                 vision_encoder.transformer.N.mlp.{c_fc,c_proj}.{weight,bias} (:643-644)
//                 vision_adapter.{c_fc,c_proj}.weight              (:1038-1039)
//                 vision_projection.weight                         (:1464)
#include "vllm/model_executor/models/muse_glimmer.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/interfaces.h"  // #607 L3 SkipTowerForModalities

namespace vllm {
namespace {

const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  if (!doc.is_object()) return nullptr;
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
// PRESENT, including an explicit `null`. Two upstream fields are `X | None`
// (configs/muse_glimmer.py:56, :58) where None is a real state DIFFERENT from
// absent: absent takes the architecture's default, an explicit `null` turns the
// mechanism off. `Field` collapses the two, so those fields ask this instead.
bool HasKey(const nlohmann::json& doc, const char* key) {
  return doc.is_object() && doc.find(key) != doc.end();
}
const nlohmann::json* Object(const nlohmann::json& doc, const char* key) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_object()) ? f : nullptr;
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}
// Tri-state: MuseGlimmer's modular schema OMITS `use_qk_norm` /
// `use_attn_output_gate`, so "absent" must mean TRUE, not false
// (muse_glimmer.py:456-469). Only an explicit `false` disables.
bool RawBoolDefaultTrue(const nlohmann::json& doc, const char* key) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : true;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}
std::vector<int64_t> RawIntArray(const nlohmann::json& doc, const char* key) {
  std::vector<int64_t> out;
  const nlohmann::json* f = Field(doc, key);
  if (f != nullptr && f->is_array())
    for (const auto& v : *f)
      if (v.is_number()) out.push_back(v.get<int64_t>());
  return out;
}
std::vector<double> RawDoubleArray(const nlohmann::json& doc, const char* key) {
  std::vector<double> out;
  const nlohmann::json* f = Field(doc, key);
  if (f != nullptr && f->is_array())
    for (const auto& v : *f)
      if (v.is_number()) out.push_back(v.get<double>());
  return out;
}
std::vector<std::string> RawStringArray(const nlohmann::json& doc, const char* key) {
  std::vector<std::string> out;
  const nlohmann::json* f = Field(doc, key);
  if (f != nullptr && f->is_array())
    for (const auto& v : *f)
      if (v.is_string()) out.push_back(v.get<std::string>());
  return out;
}

bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// Replace the FIRST occurrence of `from` with `to`. Returns whether it fired.
bool ReplaceFirst(std::string* s, const std::string& from, const std::string& to) {
  const std::string::size_type at = s->find(from);
  if (at == std::string::npos) return false;
  s->replace(at, from.size(), to);
  return true;
}

bool ReplacePrefix(std::string* s, const std::string& from, const std::string& to) {
  if (!StartsWith(*s, from)) return false;
  *s = to + s->substr(from.size());
  return true;
}

// configs/muse_glimmer.py:186-305. A flat config has NO `text_config` but DOES
// carry text-level fields at the top level. Detecting this is load-bearing:
// without normalization every checkpoint value is ignored and the model silently
// builds at ALL-DEFAULT shape.
bool LooksFlat(const nlohmann::json& raw) {
  if (Object(raw, "text_config") != nullptr) return false;
  return Field(raw, "hidden_size") != nullptr ||
         Field(raw, "num_hidden_layers") != nullptr;
}

// Hoist the flat text fields into a canonical text object, applying upstream's
// renames (configs/muse_glimmer.py:207-247).
nlohmann::json HoistFlatText(const nlohmann::json& raw) {
  static const char* kKeys[] = {
      "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
      "num_attention_heads", "num_key_value_heads", "head_dim",
      "max_position_embeddings", "initializer_range", "rms_norm_eps", "use_cache",
      "tie_word_embeddings", "rope_parameters", "rope_theta", "attention_bias",
      "attention_dropout", "query_pre_attn_scalar", "sliding_window", "layer_types",
      "attn_logit_softcapping", "use_bidirectional_attention", "qk_scale_factor",
      "use_qk_norm", "use_attn_output_gate", "output_multiplier",
      "normalize_tok_embeddings", "post_norm_eps", "no_rope_layers"};
  nlohmann::json text = nlohmann::json::object();
  for (const char* k : kKeys) {
    // Carry an explicit `null` THROUGH rather than dropping the key: for the two
    // `X | None` fields a dropped null would be re-defaulted on the other side,
    // so a flat config could not express "no window" / "no soft-cap" at all.
    if (HasKey(raw, k)) text[k] = raw[k];
  }
  // Renames (configs/muse_glimmer.py:207-210).
  if (const nlohmann::json* f = Field(raw, "hidden_act");
      f != nullptr && !text.contains("hidden_activation"))
    text["hidden_activation"] = *f;
  // `final_logit_softcapping` is nullable, so these two carry a null through for
  // the same reason the loop above does.
  if (HasKey(raw, "output_soft_cap_temp") && !text.contains("final_logit_softcapping"))
    text["final_logit_softcapping"] = raw["output_soft_cap_temp"];
  if (const nlohmann::json* f = Field(raw, "hidden_activation"); f != nullptr)
    text["hidden_activation"] = *f;
  if (HasKey(raw, "final_logit_softcapping"))
    text["final_logit_softcapping"] = raw["final_logit_softcapping"];
  return text;
}

// configs/muse_glimmer.py:249-260, 279-299.
nlohmann::json HoistFlatVision(const nlohmann::json& raw) {
  static const std::pair<const char*, const char*> kRenames[] = {
      {"vision_latent_dim", "hidden_size"},
      {"vision_heads", "num_attention_heads"},
      {"vision_layers", "num_hidden_layers"},
      {"vision_output_dim", "output_dim"},
      {"vision_patch_size", "patch_size"},
      {"vision_patch_temporal", "patch_temporal"},
      {"vision_adapter_dim", "adapter_dim"},
      {"vision_pos_emb_grid_h", "pos_emb_height"},
      {"vision_pos_emb_grid_w", "pos_emb_width"},
      {"vision_downsample_factor", "merge_kernel_size"}};
  nlohmann::json vision = nlohmann::json::object();
  for (const auto& [flat, canon] : kRenames) {
    const nlohmann::json* f = Field(raw, flat);
    if (f != nullptr) vision[canon] = *f;
  }
  if (const nlohmann::json* f = Field(raw, "vision_mlp_ratio");
      f != nullptr && f->is_number()) {
    const int64_t hidden =
        vision.contains("hidden_size") ? vision["hidden_size"].get<int64_t>() : 1536;
    vision["intermediate_size"] =
        static_cast<int64_t>(f->get<double>() * static_cast<double>(hidden));
  }
  if (const nlohmann::json* f = Field(raw, "vision_sparse_attention_factor");
      f != nullptr && f->is_number()) {
    const int64_t stride = f->get<int64_t>();
    if (stride <= 0)
      throw std::runtime_error(
          "MuseGlimmer: vision_sparse_attention_factor must be positive");
    const int64_t layers = vision.contains("num_hidden_layers")
                               ? vision["num_hidden_layers"].get<int64_t>()
                               : 50;
    nlohmann::json types = nlohmann::json::array();
    for (int64_t i = 0; i < layers; ++i)
      types.push_back(((i + 1) % stride == 0 || i == layers - 1) ? "full_attention"
                                                                 : "sliding_attention");
    vision["layer_types"] = types;
  }
  return vision;
}

}  // namespace

std::vector<int64_t> DefaultMuseGlimmerNoRopeLayers(int64_t num_hidden_layers) {
  // configs/muse_glimmer.py:20-26 — NoPE every 4th layer counted BACKWARD from
  // the LAST layer, so the final layer is always NoPE/full-attention.
  std::vector<int64_t> mask;
  mask.reserve(static_cast<size_t>(num_hidden_layers));
  for (int64_t i = 0; i < num_hidden_layers; ++i)
    mask.push_back(((num_hidden_layers - 1 - i) % 4 == 0) ? 0 : 1);
  return mask;
}

double ResolveMuseGlimmerQueryPreScale(double qk_scale_factor,
                                       bool has_qk_scale_factor,
                                       double explicit_scale_query_by,
                                       bool has_explicit_scale, int64_t head_dim) {
  // muse_glimmer.py:472-517. An explicit `scale_query_by` is ALREADY the final
  // factor and wins outright.
  if (has_explicit_scale) return explicit_scale_query_by;
  // ABSENT is not "no scaling" (#412). `qk_scale_factor` is a class default in
  // both references (configs/muse_glimmer.py:62; SGLang srt/configs/muse_glimmer.py:98),
  // so a checkpoint that omits it — the released DFlash drafter does — still runs
  // the architecture's scale. The identity here leaves `use_qk_norm` on with no
  // pre-scale: coherent output, collapsed acceptance, nothing to see in a log.
  // It goes through the SAME magnitude disambiguation below because what is
  // defaulted is the NATIVE (unfolded) value, exactly as upstream stores it.
  const double qk =
      has_qk_scale_factor ? qk_scale_factor : kMuseGlimmerDefaultQkScaleFactor;
  const double sqrt_hd = std::sqrt(static_cast<double>(head_dim));
  // Native raw (~43.784 at head_dim 128) is ~sqrt(head_dim)x larger than the
  // modular pre-folded value (~3.87). Disambiguate by magnitude against
  // sqrt(head_dim): at/above it the value is native and must be divided.
  if (qk >= sqrt_hd) return qk / sqrt_hd;
  return qk;
}

MuseGlimmerParams ParseMuseGlimmerParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;

  // Normalize the flat (older converter) layout into the canonical nested one
  // BEFORE reading anything (configs/muse_glimmer.py:186-305).
  const bool flat = LooksFlat(raw);
  const nlohmann::json flat_text = flat ? HoistFlatText(raw) : nlohmann::json::object();
  const nlohmann::json flat_vision =
      flat ? HoistFlatVision(raw) : nlohmann::json::object();
  const nlohmann::json* nested_text = Object(raw, "text_config");
  const nlohmann::json* nested_vision = Object(raw, "vision_config");
  const nlohmann::json& text = flat ? flat_text
                                    : (nested_text != nullptr ? *nested_text : raw);

  MuseGlimmerParams p;
  MuseGlimmerTextParams& t = p.text;

  t.vocab_size = RawInt(text, "vocab_size", 0);
  t.hidden_size = RawInt(text, "hidden_size", 0);
  t.intermediate_size = RawInt(text, "intermediate_size", 0);
  t.num_hidden_layers = RawInt(text, "num_hidden_layers", 0);
  t.num_attention_heads = RawInt(text, "num_attention_heads", 0);
  t.num_key_value_heads =
      RawInt(text, "num_key_value_heads", t.num_attention_heads);
  t.head_dim = RawInt(text, "head_dim", 0);
  t.max_position_embeddings = RawInt(text, "max_position_embeddings", 0);
  // `sliding_window` is `int | None` upstream (configs/muse_glimmer.py:56) with a
  // 2048 DEFAULT, and 0 here means no window at all — so absent must take 2048
  // and only an explicit `null` disables the window. Defaulting to 0 turned every
  // RoPE layer into global attention (#412).
  t.sliding_window = HasKey(text, "sliding_window")
                         ? RawInt(text, "sliding_window", 0)
                         : kMuseGlimmerDefaultSlidingWindow;
  t.tie_word_embeddings = RawBool(text, "tie_word_embeddings", false);
  t.rms_norm_eps = static_cast<float>(
      RawDouble(text, "rms_norm_eps",
                static_cast<double>(kMuseGlimmerDefaultRmsNormEps)));
  // The post-norms use their OWN, SMALLER eps, and its default is 1e-8 — NOT
  // `rms_norm_eps` (configs/muse_glimmer.py:67; SGLang srt/configs/muse_glimmer.py:91).
  // The released GGUF carries no post-norm key at all, so the old fallback ran
  // both sandwich post-norms at 1e-5, a thousand times too large (#412).
  t.post_norm_eps = static_cast<float>(
      RawDouble(text, "post_norm_eps",
                static_cast<double>(kMuseGlimmerDefaultPostNormEps)));
  t.hidden_activation = RawString(text, "hidden_activation", "silu");
  // ABSENT means TRUE, the same tri-state as use_qk_norm / use_attn_output_gate
  // above (configs/muse_glimmer.py:66). The released config omits the key, so the
  // default is the only value that ships; defaulting it false made
  // `perception_emb_norm` a silent no-op on the vision path (#405).
  t.normalize_tok_embeddings = RawBoolDefaultTrue(text, "normalize_tok_embeddings");
  t.output_multiplier =
      RawDouble(text, "output_multiplier", kMuseGlimmerDefaultOutputMultiplier);
  // `float | None` with a 20.0 default (configs/muse_glimmer.py:58; SGLang's
  // `output_soft_cap_temp`, srt/configs/muse_glimmer.py:102). 0 is our "no cap"
  // sentinel, so — as with the window — absent takes 20.0 and only an explicit
  // `null` removes the cap.
  t.final_logit_softcapping =
      HasKey(text, "final_logit_softcapping")
          ? RawDouble(text, "final_logit_softcapping", 0.0)
          : kMuseGlimmerDefaultFinalLogitSoftcapping;

  // RoPE theta: an explicit `rope_parameters` dict wins, else a bare
  // `rope_theta`, else upstream's 500000 default (configs/muse_glimmer.py:91-98).
  t.rope_theta = 500000.0;
  if (const nlohmann::json* rp = Object(text, "rope_parameters"); rp != nullptr)
    t.rope_theta = RawDouble(*rp, "rope_theta", 500000.0);
  else
    t.rope_theta = RawDouble(text, "rope_theta", 500000.0);

  if (t.hidden_size <= 0 || t.num_hidden_layers <= 0 || t.num_attention_heads <= 0)
    throw std::runtime_error(
        "MuseGlimmer config is missing required text geometry (hidden_size / "
        "num_hidden_layers / num_attention_heads)");
  if (t.head_dim <= 0) t.head_dim = t.hidden_size / t.num_attention_heads;

  // iRoPE mask, CHECKPOINT-FIRST. The released meta-models/Muse-Glimmer-30B
  // config ships NO `no_rope_layers` at all; it encodes the same split TWICE:
  //   text_config.layer_rope_theta[i] == 0        marks a NoPE layer
  //   text_config.layer_types[i] == "full_attention" marks the same layer
  // (verified against the released config.json, 2026-08-10: both put the NoPE
  // layers at 3, 7, 11, ... 51 for L=52). Deriving from the checkpoint rather than
  // from the counted default matters because the default only HAPPENS to agree for
  // this depth; a checkpoint with a different schedule would be silently mis-split
  // into wrong-RoPE, wrong-window layers that still emit fluent text.
  //
  // Precedence: an explicit `no_rope_layers` wins; else derive from the checkpoint's
  // own fields; else fall back to upstream's backward-counted default
  // (configs/muse_glimmer.py:20-26).
  const std::vector<double> layer_rope_theta = RawDoubleArray(text, "layer_rope_theta");
  const std::vector<std::string> text_layer_types = RawStringArray(text, "layer_types");
  const size_t L = static_cast<size_t>(t.num_hidden_layers);

  std::vector<int64_t> from_theta;
  if (layer_rope_theta.size() == L) {
    for (double th : layer_rope_theta) from_theta.push_back(th != 0.0 ? 1 : 0);
    // Every RoPE layer must use the model's single theta: we thread ONE theta into
    // the forward, so a per-layer value that disagrees would be applied wrongly.
    for (double th : layer_rope_theta)
      if (th != 0.0 && th != t.rope_theta)
        throw std::runtime_error(
            "MuseGlimmer layer_rope_theta carries a per-layer theta that disagrees "
            "with rope_parameters.rope_theta; only a single theta is supported");
  }
  std::vector<int64_t> from_types;
  if (text_layer_types.size() == L)
    for (const std::string& ty : text_layer_types)
      from_types.push_back(ty == "full_attention" ? 0 : 1);

  // Both present => they must AGREE. A disagreement is a config we do not
  // understand, and guessing which one wins is exactly the silent-wrong-model risk.
  if (!from_theta.empty() && !from_types.empty() && from_theta != from_types)
    throw std::runtime_error(
        "MuseGlimmer layer_rope_theta and layer_types disagree about which layers "
        "are NoPE/full-attention");

  t.no_rope_layers = RawIntArray(text, "no_rope_layers");
  if (t.no_rope_layers.empty() && !from_theta.empty()) t.no_rope_layers = from_theta;
  if (t.no_rope_layers.empty() && !from_types.empty()) t.no_rope_layers = from_types;
  if (t.no_rope_layers.empty())
    t.no_rope_layers = DefaultMuseGlimmerNoRopeLayers(t.num_hidden_layers);
  if (static_cast<int64_t>(t.no_rope_layers.size()) != t.num_hidden_layers)
    throw std::runtime_error(
        "MuseGlimmer no_rope_layers length does not match num_hidden_layers");

  // The two flags default to TRUE when absent (muse_glimmer.py:456-469).
  t.use_qk_norm = RawBoolDefaultTrue(text, "use_qk_norm");
  t.use_attn_output_gate = RawBoolDefaultTrue(text, "use_attn_output_gate");

  const nlohmann::json* qk = Field(text, "qk_scale_factor");
  const nlohmann::json* sq = Field(text, "scale_query_by");
  t.scale_query_by = ResolveMuseGlimmerQueryPreScale(
      (qk != nullptr && qk->is_number()) ? qk->get<double>() : 0.0,
      qk != nullptr && qk->is_number(),
      (sq != nullptr && sq->is_number()) ? sq->get<double>() : 0.0,
      sq != nullptr && sq->is_number(), t.head_dim);

  // --- vision ---
  const nlohmann::json* vision_obj =
      flat ? (flat_vision.empty() ? nullptr : &flat_vision) : nested_vision;
  if (vision_obj != nullptr) {
    MuseGlimmerVisionParams& v = p.vision;
    v.present = true;
    v.patch_size = RawInt(*vision_obj, "patch_size", 14);
    v.pos_emb_height = RawInt(*vision_obj, "pos_emb_height", 32);
    v.pos_emb_width = RawInt(*vision_obj, "pos_emb_width", 32);
    v.num_attention_heads = RawInt(*vision_obj, "num_attention_heads", 16);
    v.num_hidden_layers = RawInt(*vision_obj, "num_hidden_layers", 50);
    v.hidden_size = RawInt(*vision_obj, "hidden_size", 1536);
    v.intermediate_size = RawInt(*vision_obj, "intermediate_size", 8960);
    // FIELD SPELLINGS, verified against the released config.json (2026-08-10):
    // the vision block ships `merge_size` (not `merge_kernel_size`) and ships
    // NEITHER `output_dim` NOR `adapter_dim` — those live at the TOP level as
    // `out_hidden_size` and `projector_hidden_size`. Reading only the old spellings
    // fell back to defaults that COINCIDENTALLY equal the real values, so the
    // `output_dim == hidden * merge^2` check below passed by luck rather than by
    // reading the checkpoint. Both spellings are accepted, checkpoint-first.
    v.merge_kernel_size =
        RawInt(*vision_obj, "merge_size", RawInt(*vision_obj, "merge_kernel_size", 2));
    v.output_dim =
        RawInt(*vision_obj, "output_dim", RawInt(raw, "out_hidden_size", 6144));
    v.patch_temporal = RawInt(*vision_obj, "patch_temporal", 2);
    v.adapter_dim =
        RawInt(*vision_obj, "adapter_dim", RawInt(raw, "projector_hidden_size", 4096));
    v.layer_norm_eps =
        static_cast<float>(RawDouble(*vision_obj, "layer_norm_eps", 1e-5));
    v.layer_types = RawStringArray(*vision_obj, "layer_types");
    if (v.layer_types.empty()) {
      // configs/muse_glimmer.py:168-176 — full every 4th layer AND on the last.
      // NOTE this rule differs from the text tower's backward-counted mask. The
      // non-full spelling is "window_attention", which is what the released
      // config.json ships (verified 2026-08-10); every consumer compares against
      // "full_attention" as upstream does, so either spelling reads the same.
      for (int64_t i = 0; i < v.num_hidden_layers; ++i)
        v.layer_types.push_back(
            ((i + 1) % 4 == 0 || i == v.num_hidden_layers - 1) ? "full_attention"
                                                               : "window_attention");
    }
    // muse_glimmer.py:734-739 — the pixel-shuffle output width is structural.
    const int64_t expected = v.hidden_size * v.merge_kernel_size * v.merge_kernel_size;
    if (v.output_dim != expected)
      throw std::runtime_error(
          "MuseGlimmer vision output_dim does not match the pixel-shuffle output "
          "(hidden_size * merge_kernel_size^2)");
  }

  p.image_token_id = RawInt(raw, "image_token_id", RawInt(raw, "patch_token_id", 200092));
  p.video_token_id = RawInt(raw, "video_token_id", 200091);
  return p;
}

void ParseMuseGlimmerConfig(const HfConfig& config) {
  const MuseGlimmerParams p = ParseMuseGlimmerParams(config);
  if (p.text.hidden_activation != "silu")
    throw std::runtime_error(
        "MuseGlimmer uses `silu` as the hidden activation; got `" +
        p.text.hidden_activation + "`");
}

MuseGlimmerCheckpointConvention MuseGlimmerConventionOf(const std::string& name) {
  // muse_glimmer.py:1379-1381 — the PREFIX is the unambiguous discriminator.
  if (StartsWith(name, "model.language_model.") || StartsWith(name, "language_model."))
    return MuseGlimmerCheckpointConvention::kCanonical;
  if (StartsWith(name, "model.layers."))
    return MuseGlimmerCheckpointConvention::kLegacyGuac;
  return MuseGlimmerCheckpointConvention::kCanonical;
}

bool NormalizeMuseGlimmerWeightName(const std::string& name, std::string* out) {
  std::string s = name;

  // Dropped outright (muse_glimmer.py:1403).
  if (StartsWith(s, "model.rotary_emb.")) return false;

  // (1) LEGACY sandwich-norm remap, BEFORE anything strips the prefix. Order is
  // load-bearing: legacy `post_attention_layernorm` is really the PRE-feedforward
  // norm, so it must be renamed FIRST or the next rule's output is re-captured
  // and the two norms silently swap (muse_glimmer.py:1364-1388).
  if (MuseGlimmerConventionOf(s) == MuseGlimmerCheckpointConvention::kLegacyGuac) {
    if (!ReplaceFirst(&s, ".post_attention_layernorm.", ".pre_feedforward_layernorm."))
      (void)0;
    if (!ReplaceFirst(&s, ".post_attn_norm.", ".post_attention_layernorm."))
      (void)0;
    (void)ReplaceFirst(&s, ".post_ffn_norm.", ".post_feedforward_layernorm.");
  }

  // (2) The attention OUTPUT GATE rename MUST precede any `.gate_proj` MLP
  // stacking rule (muse_glimmer.py:1400). Our loader keeps gate/up separate, so
  // no stacking rule runs here, but the rename still has to fire so the tensor
  // lands on `output_gate_proj` rather than being mistaken for an MLP gate.
  (void)ReplaceFirst(&s, ".self_attn.gate_proj", ".self_attn.output_gate_proj");

  // (3) Vision substring renames (muse_glimmer.py:1391-1399).
  (void)ReplaceFirst(&s,
                     "model.vision_tower.patch_embedder.position_embedding_table.weight",
                     "model.vision_tower.positional_embedding_vlm");
  (void)ReplaceFirst(&s, "model.vision_tower.layers.", "model.vision_tower.transformer.");
  (void)ReplaceFirst(&s, ".norm1.", ".ln_1.");
  (void)ReplaceFirst(&s, ".norm2.", ".ln_2.");
  (void)ReplaceFirst(&s, ".attn.proj.", ".attn.o_proj.");
  (void)ReplaceFirst(&s, ".mlp.fc1.", ".mlp.c_fc.");
  (void)ReplaceFirst(&s, ".mlp.fc2.", ".mlp.c_proj.");

  // (4) Prefix renames, most specific first (muse_glimmer.py:1402-1417).
  (void)(ReplacePrefix(&s, "model.vision_tower.patch_embedder.patch_embedding.",
                       "model.vision_tower.conv1_linear.") ||
         ReplacePrefix(&s, "model.vision_adapter.fc1.", "model.vision_adapter.c_fc.") ||
         ReplacePrefix(&s, "model.vision_adapter.fc2.", "model.vision_adapter.c_proj."));
  (void)(ReplacePrefix(&s, "model.vision_tower.", "vision_encoder.") ||
         ReplacePrefix(&s, "model.vision_encoder.", "vision_encoder.") ||
         ReplacePrefix(&s, "vision_tower.", "vision_encoder.") ||
         ReplacePrefix(&s, "model.vision_adapter.", "vision_adapter.") ||
         ReplacePrefix(&s, "model.vision_projection.", "vision_projection.") ||
         ReplacePrefix(&s, "model.perception_emb_norm.", "perception_emb_norm.") ||
         ReplacePrefix(&s, "model.language_model.", "model.") ||
         ReplacePrefix(&s, "language_model.", "model."));

  *out = s;
  return true;
}

std::vector<std::string> EnumerateMuseGlimmerTensors(const MuseGlimmerParams& params) {
  const MuseGlimmerTextParams& t = params.text;
  std::vector<std::string> names;

  names.push_back("model.embed_tokens.weight");
  for (int64_t i = 0; i < t.num_hidden_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    // Sandwich norms (all four are real tensors; the +1 offset is baked at use).
    names.push_back(p + "input_layernorm.weight");
    names.push_back(p + "post_attention_layernorm.weight");
    names.push_back(p + "pre_feedforward_layernorm.weight");
    names.push_back(p + "post_feedforward_layernorm.weight");
    names.push_back(p + "self_attn.q_proj.weight");
    names.push_back(p + "self_attn.k_proj.weight");
    names.push_back(p + "self_attn.v_proj.weight");
    names.push_back(p + "self_attn.o_proj.weight");
    // The per-head qk_norm is WEIGHTLESS — deliberately no tensor here.
    if (t.use_attn_output_gate)
      names.push_back(p + "self_attn.output_gate_proj.weight");
    names.push_back(p + "mlp.gate_proj.weight");
    names.push_back(p + "mlp.up_proj.weight");
    names.push_back(p + "mlp.down_proj.weight");
  }
  names.push_back("model.norm.weight");
  // `model.embed_norm` is weightless — deliberately absent.
  if (!t.tie_word_embeddings) names.push_back("lm_head.weight");

  if (params.vision.present) {
    const MuseGlimmerVisionParams& v = params.vision;
    names.push_back("vision_encoder.conv1_linear.weight");
    names.push_back("vision_encoder.positional_embedding_vlm");
    names.push_back("vision_encoder.ln_pre.weight");
    names.push_back("vision_encoder.ln_pre.bias");
    for (int64_t i = 0; i < v.num_hidden_layers; ++i) {
      const std::string p = "vision_encoder.transformer." + std::to_string(i) + ".";
      names.push_back(p + "ln_1.weight");
      names.push_back(p + "ln_1.bias");
      names.push_back(p + "ln_2.weight");
      names.push_back(p + "ln_2.bias");
      // W4 CORRECTION, verified against the released `meta-models/Muse-Glimmer-30B`
      // checkpoint (1436 tensors, revision f84ecc3a0e; the header-only projection is
      // committed as tests/vllm/models/fixtures/muse_glimmer_30b/index.json). W0
      // enumerated ONE merged `attn.qkv_proj.weight` per vision layer and NO vision
      // attention bias at all, mirroring upstream's `QKVParallelLinear` MODULE name
      // rather than the checkpoint's on-disk names. The checkpoint ships SEPARATE
      // `attn.{q,k,v}_proj` and `attn.proj` (-> `attn.o_proj`), each WITH a bias —
      // upstream reaches them through its `packed_modules_mapping`
      // (muse_glimmer.py:1427-1430), which fuses the three shards at LOAD time.
      //
      // The enumeration is the checkpoint's contract, so it names what the file
      // ships; the MERGE stays an internal representation and happens in the loader
      // (LoadVisionTower), which is where upstream does it too. Getting this
      // backwards is not cosmetic: the loader would demand a tensor no checkpoint
      // contains, and the structural accounting pass would report the tower as
      // partly missing instead of saying so.
      for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + "attn." + proj + ".weight");
        names.push_back(p + "attn." + proj + ".bias");
      }
      names.push_back(p + "mlp.c_fc.weight");
      names.push_back(p + "mlp.c_fc.bias");
      names.push_back(p + "mlp.c_proj.weight");
      names.push_back(p + "mlp.c_proj.bias");
    }
    names.push_back("vision_encoder.ln_post.weight");
    names.push_back("vision_encoder.ln_post.bias");
    names.push_back("vision_adapter.c_fc.weight");
    names.push_back("vision_adapter.c_proj.weight");
    names.push_back("vision_projection.weight");
    // `perception_emb_norm` is weightless — deliberately absent.
  }
  return names;
}

namespace {

// Where a CANONICAL (post-`NormalizeMuseGlimmerWeightName`) name lives.
struct TensorSite {
  const SafetensorsFile* shard = nullptr;
  std::string raw_name;
};

// One decoder layer's tensors, in the canonical names EnumerateMuseGlimmerTensors
// declares (muse_glimmer.py:1218-1277).
MuseGlimmerLayerWeights LoadLayer(const TensorResolver& get, int64_t layer,
                                  bool use_attn_output_gate) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  MuseGlimmerLayerWeights w;
  w.input_layernorm = dense_loaders::LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      dense_loaders::LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.pre_feedforward_layernorm =
      dense_loaders::LoadBf16Direct(get, base + "pre_feedforward_layernorm.weight");
  w.post_feedforward_layernorm =
      dense_loaders::LoadBf16Direct(get, base + "post_feedforward_layernorm.weight");

  // Merged QKVParallelLinear, rows q|k|v (muse_glimmer.py:1126-1135).
  w.attn.qkv_proj = dense_loaders::LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = dense_loaders::LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // The attention OUTPUT GATE (:1145-1152). Its checkpoint name collides with the
  // MLP gate by suffix, which is why NormalizeMuseGlimmerWeightName renames
  // `.self_attn.gate_proj` FIRST; here the canonical name is unambiguous.
  if (use_attn_output_gate)
    w.attn.output_gate_proj =
        dense_loaders::LoadMergedBf16RawNK(get, {sa + "output_gate_proj.weight"});

  // MergedColumnParallelLinear gate|up (:1052-1058) — SwiGLU, not GeGLU.
  w.mlp.gate_up_proj = dense_loaders::LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = dense_loaders::LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

// ── W4: the perception encoder ───────────────────────────────────────────────
// The W3 tower (muse_glimmer_vision.{h,cpp}) owns HOST f32 weight structs in torch
// storage order, so the loader's job here is a bf16 -> f32 widening plus ONE
// structural fold: the checkpoint's separate `attn.{q,k,v}_proj` shards become the
// tower's merged `[3*hidden, hidden]` operand. Row order is q|k|v and it is
// load-bearing — upstream's `QKVParallelLinear` output is viewed as
// `(tokens, 3, heads, head_dim)` and unbound on dim 1 (muse_glimmer.py:611-618),
// so shard k landing where q is expected silently permutes the attention rather
// than erroring.
std::vector<float> Bf16TensorToF32(const TensorResolver& get, const std::string& name,
                                   const std::vector<int64_t>& expect) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16",
           "muse_glimmer vision: expected BF16 for " + name + ", got " + t.dtype);
  VT_CHECK(t.shape.size() == expect.size(),
           "muse_glimmer vision: rank mismatch for " + name);
  int64_t n = 1;
  for (size_t i = 0; i < expect.size(); ++i) {
    VT_CHECK(t.shape[i] == expect[i],
             "muse_glimmer vision: shape mismatch for " + name);
    n *= expect[i];
  }
  VT_CHECK(t.nbytes >= static_cast<size_t>(n) * 2,
           "muse_glimmer vision: truncated tensor " + name);
  const auto* src = static_cast<const uint8_t*>(t.data);
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    uint16_t bits = 0;
    std::memcpy(&bits, src + static_cast<size_t>(i) * 2, sizeof(bits));
    out[static_cast<size_t>(i)] = vt::BF16ToF32(bits);
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return out;
}

// Concatenate the q|k|v shards along output rows into one [3*H, H] operand.
std::vector<float> MergeQkvF32(const TensorResolver& get, const std::string& prefix,
                               const std::string& suffix,
                               const std::vector<int64_t>& shard_shape) {
  std::vector<float> out;
  for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {
    const std::vector<float> shard =
        Bf16TensorToF32(get, prefix + proj + suffix, shard_shape);
    out.insert(out.end(), shard.begin(), shard.end());
  }
  return out;
}

MuseGlimmerVisionTower LoadVisionTower(const TensorResolver& get,
                                       const MuseGlimmerParams& params) {
  const MuseGlimmerVisionParams& v = params.vision;
  const int64_t VH = v.hidden_size;
  const int64_t VI = v.intermediate_size;
  const int64_t patch_dim = v.patch_temporal * 3 * v.patch_size * v.patch_size;

  MuseGlimmerVisionTower tower;
  tower.cfg = MuseGlimmerVisionConfigOf(params);
  tower.encoder.conv1_w =
      Bf16TensorToF32(get, "vision_encoder.conv1_linear.weight", {VH, patch_dim});
  tower.encoder.pos_emb =
      Bf16TensorToF32(get, "vision_encoder.positional_embedding_vlm",
                      {v.pos_emb_height * v.pos_emb_width, VH});
  tower.encoder.ln_pre_w = Bf16TensorToF32(get, "vision_encoder.ln_pre.weight", {VH});
  tower.encoder.ln_pre_b = Bf16TensorToF32(get, "vision_encoder.ln_pre.bias", {VH});
  tower.encoder.ln_post_w = Bf16TensorToF32(get, "vision_encoder.ln_post.weight", {VH});
  tower.encoder.ln_post_b = Bf16TensorToF32(get, "vision_encoder.ln_post.bias", {VH});

  tower.encoder.blocks.reserve(static_cast<size_t>(v.num_hidden_layers));
  for (int64_t l = 0; l < v.num_hidden_layers; ++l) {
    const std::string p = "vision_encoder.transformer." + std::to_string(l) + ".";
    multimodal::MuseGlimmerVisionBlockWeights b;
    b.ln_1_w = Bf16TensorToF32(get, p + "ln_1.weight", {VH});
    b.ln_1_b = Bf16TensorToF32(get, p + "ln_1.bias", {VH});
    b.ln_2_w = Bf16TensorToF32(get, p + "ln_2.weight", {VH});
    b.ln_2_b = Bf16TensorToF32(get, p + "ln_2.bias", {VH});
    b.qkv_w = MergeQkvF32(get, p + "attn.", ".weight", {VH, VH});
    b.qkv_b = MergeQkvF32(get, p + "attn.", ".bias", {VH});
    b.o_w = Bf16TensorToF32(get, p + "attn.o_proj.weight", {VH, VH});
    b.o_b = Bf16TensorToF32(get, p + "attn.o_proj.bias", {VH});
    b.c_fc_w = Bf16TensorToF32(get, p + "mlp.c_fc.weight", {VI, VH});
    b.c_fc_b = Bf16TensorToF32(get, p + "mlp.c_fc.bias", {VI});
    b.c_proj_w = Bf16TensorToF32(get, p + "mlp.c_proj.weight", {VH, VI});
    b.c_proj_b = Bf16TensorToF32(get, p + "mlp.c_proj.bias", {VH});
    tower.encoder.blocks.push_back(std::move(b));
  }

  // The adapter's two projections are BIAS-FREE (:1039-1040), and so is the
  // vision_projection (:1464-1468) — enumerating a bias for either would demand a
  // tensor the checkpoint does not ship.
  tower.adapter.c_fc_w =
      Bf16TensorToF32(get, "vision_adapter.c_fc.weight", {v.adapter_dim, v.output_dim});
  tower.adapter.c_proj_w = Bf16TensorToF32(get, "vision_adapter.c_proj.weight",
                                           {v.adapter_dim, v.adapter_dim});
  tower.projection = Bf16TensorToF32(get, "vision_projection.weight",
                                     {params.text.hidden_size, v.adapter_dim});
  tower.loaded = true;
  return tower;
}

}  // namespace

MuseGlimmerVisionTower LoadMuseGlimmerVisionTower(const std::vector<SafetensorsFile>& shards,
                                                  const MuseGlimmerParams& params) {
  VT_CHECK(params.vision.present,
           "muse_glimmer: this config declares no vision_config, so there is no "
           "perception encoder to load");
  std::unordered_map<std::string, TensorSite> where;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& raw_name : shard.Names()) {
      std::string canonical;
      if (!NormalizeMuseGlimmerWeightName(raw_name, &canonical)) continue;
      where.emplace(canonical, TensorSite{&shard, raw_name});
    }
  }
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "muse_glimmer: tensor not found: " + name);
    return it->second.shard->Get(it->second.raw_name);
  };
  return LoadVisionTower(get, params);
}

MuseGlimmerWeights LoadMuseGlimmerForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    const MultiModalConfig* mm_config) {
  MuseGlimmerWeights w;
  w.params = ParseMuseGlimmerParams(config);
  const std::vector<std::string> expected = EnumerateMuseGlimmerTensors(w.params);
  w.enumerated_tensors = static_cast<int64_t>(expected.size());

  // STRUCTURAL accounting (W0): normalize every checkpoint name through the same
  // mapper the forward uses, and report how much of the enumerated structure is
  // present. The same index is what the W1 materialization reads through, so the
  // accounting and the load can never disagree about a name.
  std::unordered_map<std::string, TensorSite> where;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& raw_name : shard.Names()) {
      std::string canonical;
      if (!NormalizeMuseGlimmerWeightName(raw_name, &canonical)) continue;
      where.emplace(canonical, TensorSite{&shard, raw_name});
    }
  }
  for (const std::string& name : expected)
    if (where.count(name) != 0) ++w.accounted_tensors;

  // W1: materialize the TEXT tower. The perception encoder's tensors are accounted
  // above but NOT materialized — that is W3, and a Muse Glimmer forward is text-only
  // until it lands.
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "muse_glimmer: tensor not found: " + name);
    return it->second.shard->Get(it->second.raw_name);
  };

  const MuseGlimmerTextParams& t = w.params.text;
  w.embed_tokens = dense_loaders::LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = dense_loaders::LoadBf16Direct(get, "model.norm.weight");
  // UNTIED lm_head (:1480). Muse Glimmer is NOT a Gemma here: `tie_word_embeddings`
  // is false in the released config, and only an explicitly tied checkpoint (which
  // ships no `lm_head.weight`) falls back to the embedding table.
  if (!t.tie_word_embeddings)
    w.lm_head = dense_loaders::LoadBf16Transposed(get, "lm_head.weight");

  w.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
  for (int64_t l = 0; l < t.num_hidden_layers; ++l)
    w.layers.push_back(LoadLayer(get, l, t.use_attn_output_gate));
  w.text_loaded = true;

  // W4: the perception encoder, when the checkpoint carries one. Loaded by DEFAULT
  // — a capability behind an opt-in flag is a capability nobody reaches. A
  // text-only Muse Glimmer checkpoint (no `vision_config`) leaves `vision.loaded`
  // false and the mm forward refuses BY NAME rather than reading empty vectors.
  //
  // #607 L3, the TOWER SKIP: with every modality the encoder serves at limit 0,
  // its tensors are never read. `w.params.vision` above already parsed the whole
  // geometry off `vision_config`, which is upstream's construct-then-do-not-
  // initialise exactly (interfaces.py:288-293 over `torch.device("meta")`); only
  // the storage is skipped. The perception encoder covers image AND video
  // (muse_glimmer_registry.cpp:37-39), which is the modality set `_mark_tower_
  // model` would be called with, so BOTH must be 0 — `--limit-mm-per-prompt
  // '{"image":0}'` alone keeps the tower.
  if (w.params.vision.present) {
    if (SkipTowerForModalities(mm_config, {"image", "video"})) {
      w.vision_skipped = true;
    } else {
      w.vision = LoadVisionTower(get, w.params);
    }
  }
  return w;
}

}  // namespace vllm
