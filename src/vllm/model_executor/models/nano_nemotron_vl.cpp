// `NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3`: config parse and
// vision-side weight load. See include/vllm/model_executor/models/nano_nemotron_vl.h.
#include "vllm/model_executor/models/nano_nemotron_vl.h"

#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

using json = nlohmann::json;

constexpr const char* kSpec = ".agents/specs/nano-nemotron-vl-radio.md";

[[noreturn]] void Refuse(const std::string& arch, const std::string& what) {
  throw std::runtime_error("Model architecture " + arch + ": " + what + " (see " +
                           kSpec + ")");
}

const json* Obj(const json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || !it->is_object()) return nullptr;
  return &*it;
}

bool Has(const json& doc, const char* key) {
  auto it = doc.find(key);
  return it != doc.end() && !it->is_null();
}

double Num(const json& doc, const char* key, double fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}

std::string Str(const json& doc, const char* key, const std::string& fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || !it->is_string()) return fallback;
  return it->get<std::string>();
}

// Python round(): half to even.
int64_t PyRound(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

}  // namespace

NanoNemotronVLParams ParseNanoNemotronVLParams(const HfConfig& config) {
  const json& doc = config.raw;
  NanoNemotronVLParams p;
  p.architecture =
      config.architectures.empty() ? std::string("NemotronH_Nano_VL_V2") : config.architectures[0];
  const std::string& arch = p.architecture;

  const json* vision = Obj(doc, "vision_config");
  if (vision == nullptr) Refuse(arch, "config.json carries no `vision_config`");
  const json* args = Obj(*vision, "args");
  if (args == nullptr) Refuse(arch, "`vision_config.args` is missing");

  // `use_dynamic_resolution` (processors/...:622-625) is the PRESENCE of the
  // key. The 12B `NemotronH_Nano_VL_V2` release does not carry it (checked on
  // nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 @ ca9543b1), so that checkpoint
  // takes the static arm. A present-but-null value would crash upstream's
  // tiler; it is refused here instead.
  if (!Has(*args, "min_num_patches")) {
    Refuse(arch,
           "`vision_config.args.min_num_patches` is absent or null, which "
           "selects the static InternVL tiling arm (`dynamic_preprocess` with "
           "`max_num_tiles` tiles plus a thumbnail, processors/nano_nemotron_vl.py:"
           "94-143). Only the dynamic-resolution tiler is ported");
  }

  const std::string ps = Str(doc, "ps_version", "v1");
  if (ps != "v2") {
    Refuse(arch, "`ps_version` is '" + ps +
                     "'; only the v2 pixel shuffle is ported (v1 transposes "
                     "the image, nano_nemotron_vl.py:1020-1027)");
  }
  const double ds = Num(doc, "downsample_ratio", 0.0);
  if (!(ds > 0.0) || std::abs(1.0 / ds - 2.0) > 1e-9) {
    Refuse(arch, "`downsample_ratio` must be 0.5 (the dynamic tiler asserts a "
                 "reduction factor of 2, processors/nano_nemotron_vl.py:276-283)");
  }
  p.reduction = 2;

  // --- RADIO (get_vit_model_from_radio_config, :1568-1598) ---
  const std::string model_name = Str(*args, "model", "");
  struct Dims { int64_t h, l, nh, i; };
  static const std::map<std::string, Dims> kVit = {
      {"vit_small_patch16_224", {384, 12, 6, 1536}},
      {"vit_base_patch16_224", {768, 12, 12, 3072}},
      {"vit_large_patch16_224", {1024, 24, 16, 4096}},
      {"vit_huge_patch16_224", {1280, 32, 16, 5120}},
  };
  auto dims = kVit.find(model_name);
  if (dims == kVit.end()) {
    Refuse(arch, "RADIO model '" + model_name +
                     "' is not in VIT_TIMM_DIM_BY_NAME (configs/radio.py:12-17)");
  }
  multimodal::RadioVisionConfig& r = p.radio;
  r.hidden_size = dims->second.h;
  r.num_hidden_layers = dims->second.l;
  r.num_attention_heads = dims->second.nh;
  r.intermediate_size = dims->second.i;
  r.patch_size = static_cast<int64_t>(Num(*vision, "patch_size", 16));
  // RadioConfig(**args) lets `args` override every RadioConfig default.
  r.layer_norm_eps = static_cast<float>(Num(*args, "layer_norm_eps", 1e-6));
  if (Has(*args, "qk_normalization") && args->at("qk_normalization").is_boolean() &&
      args->at("qk_normalization").get<bool>()) {
    Refuse(arch, "RADIO qk_normalization is on; the tower ports the plain qkv path");
  }
  if (Str(*args, "norm_type", "layer_norm") != "layer_norm") {
    Refuse(arch, "RADIO norm_type '" + Str(*args, "norm_type", "") + "' is not ported");
  }
  if (Str(*args, "hidden_act", "gelu") != "gelu") {
    Refuse(arch, "RADIO hidden_act '" + Str(*args, "hidden_act", "") + "' is not ported");
  }
  if (Num(*args, "initializer_factor", 1.0) != 1.0) {
    Refuse(arch, "RADIO initializer_factor (the ls1/ls2 layer scale) is not 1.0");
  }
  if (Has(*args, "qkv_bias") && args->at("qkv_bias").is_boolean() &&
      !args->at("qkv_bias").get<bool>()) {
    Refuse(arch, "RADIO qkv_bias is off; the released towers carry a qkv bias");
  }

  // ViTPatchGenerator geometry (radio.py:540-558, :132-158).
  int64_t image_size = 224;
  if (const auto it = vision->find("preferred_resolution");
      it != vision->end() && it->is_array() && !it->empty() && (*it)[0].is_number()) {
    image_size = (*it)[0].get<int64_t>();
  }
  const double cpe_max = Num(*args, "cpe_max_size", 2048);
  const int64_t max_img =
      PyRound(cpe_max / static_cast<double>(r.patch_size)) * r.patch_size;
  const int64_t max_dim =
      static_cast<int64_t>(std::ceil(static_cast<double>(max_img) / r.patch_size)) * r.patch_size;
  r.pos_rows = r.pos_cols = max_dim / r.patch_size;
  r.cpe_mode = max_dim != image_size;

  // ClsToken (radio.py:60-106): one CLS per UNIQUE teacher when
  // cls_token_per_teacher, then registers up to register_multiple.
  std::set<std::string> teachers;
  if (const auto it = args->find("teachers"); it != args->end() && it->is_array()) {
    for (const json& t : *it) {
      if (t.is_object() && t.contains("name") && t.at("name").is_string())
        teachers.insert(t.at("name").get<std::string>());
    }
  }
  const bool per_teacher = args->contains("cls_token_per_teacher") &&
                           args->at("cls_token_per_teacher").is_boolean() &&
                           args->at("cls_token_per_teacher").get<bool>();
  r.num_cls_tokens = per_teacher ? static_cast<int64_t>(teachers.size()) : 1;
  const int64_t reg_mult = static_cast<int64_t>(Num(*args, "register_multiple", 0));
  r.num_registers = reg_mult > 0 ? reg_mult - (r.num_cls_tokens % reg_mult) : 0;

  // --- projector (mlp1, :955-976) ---
  const int64_t vit_hidden = static_cast<int64_t>(Num(doc, "vit_hidden_size", 0));
  if (vit_hidden != r.hidden_size) {
    Refuse(arch, "`vit_hidden_size` " + std::to_string(vit_hidden) +
                     " does not match the RADIO width " + std::to_string(r.hidden_size));
  }
  p.projector.in_dim = vit_hidden * p.reduction * p.reduction;
  p.projector.hidden_dim = static_cast<int64_t>(Num(doc, "projector_hidden_size", 0));
  p.projector.out_dim = config.hidden_size;
  if (p.projector.hidden_dim <= 0 || p.projector.out_dim <= 0) {
    Refuse(arch, "`projector_hidden_size` or the language tower's hidden_size is missing");
  }

  // --- processor (processors/...:582-619) ---
  multimodal::NanoNemotronVLProcessorConfig& pc = p.processor;
  pc.patch_size = static_cast<int64_t>(Num(doc, "patch_size", 16));
  if (pc.patch_size != r.patch_size) {
    Refuse(arch, "the processor patch_size and the RADIO patch_size disagree");
  }
  pc.reduction = p.reduction;
  pc.min_num_patches = static_cast<int64_t>(Num(*args, "min_num_patches", 0));
  pc.max_num_patches = static_cast<int64_t>(Num(*args, "max_num_patches", 0));
  for (const char* key : {"norm_mean", "norm_std"}) {
    const auto it = doc.find(key);
    if (it == doc.end() || !it->is_array() || it->size() != 3) {
      Refuse(arch, std::string("`") + key + "` must be a 3-element list");
    }
    for (size_t c = 0; c < 3; ++c) {
      (std::string(key) == "norm_mean" ? pc.norm_mean : pc.norm_std)[c] =
          (*it)[c].get<float>();
    }
  }

  p.img_start_token = Str(doc, "img_start_token", "<img>");
  p.img_end_token = Str(doc, "img_end_token", "</img>");
  p.img_context_token = Str(doc, "img_context_token", "<image>");
  p.has_sound = Obj(doc, "sound_config") != nullptr;
  p.has_video_embedder =
      Num(*vision, "video_temporal_patch_size", 1) > 1;
  return p;
}

HfConfig NanoNemotronVLTextConfig(const HfConfig& config) {
  const json& doc = config.raw;
  const json* text = Obj(doc, "llm_config");
  if (text == nullptr) text = Obj(doc, "text_config");
  if (text == nullptr) {
    Refuse(config.architectures.empty() ? std::string("NemotronH_Nano_VL_V2")
                                        : config.architectures[0],
           "config.json carries neither `llm_config` nor `text_config`");
  }
  json t = *text;
  if (Has(doc, "quantization_config") && !Has(t, "quantization_config")) {
    t["quantization_config"] = doc.at("quantization_config");
  }
  return ParseHfConfig(t, "llm_config");
}

NanoNemotronVLVisionLoad LoadNanoNemotronVLVisionWeights(
    const std::vector<SafetensorsFile>& shards, const NanoNemotronVLParams& params) {
  const std::string& arch = params.architecture;
  const multimodal::RadioVisionConfig& r = params.radio;
  std::map<std::string, const StTensor*> index;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      if (name.rfind("vision_model.", 0) == 0 || name.rfind("mlp1.", 0) == 0) {
        index.emplace(name, &shard.Get(name));
      }
    }
  }
  NanoNemotronVLVisionLoad out;
  out.shipped = static_cast<int64_t>(index.size());
  std::set<std::string> consumed;

  auto take = [&](const std::string& name, std::vector<int64_t> shape) {
    auto it = index.find(name);
    if (it == index.end()) Refuse(arch, "the checkpoint does not ship '" + name + "'");
    const StTensor& t = *it->second;
    if (t.dtype != "BF16") {
      Refuse(arch, "'" + name + "' is " + t.dtype +
                       "; the vision tower and projector load bf16 only");
    }
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    int64_t have = 1;
    for (int64_t s : t.shape) have *= s;
    if (have != numel) {
      Refuse(arch, "'" + name + "' has " + std::to_string(have) +
                       " elements, expected " + std::to_string(numel));
    }
    std::vector<uint16_t> v(static_cast<size_t>(numel));
    std::memcpy(v.data(), t.data, static_cast<size_t>(numel) * sizeof(uint16_t));
    consumed.insert(name);
    return v;
  };

  const int64_t H = r.hidden_size;
  const int64_t I = r.intermediate_size;
  const std::string pg = "vision_model.radio_model.model.patch_generator.";
  out.radio.embedder_w = take(pg + "embedder.weight", {H, r.patch_dim()});
  out.radio.pos_embed = take(pg + "pos_embed", {r.pos_rows * r.pos_cols, H});
  out.radio.cls_token = take(pg + "cls_token.token", {r.num_skip(), H});
  out.radio.blocks.resize(static_cast<size_t>(r.num_hidden_layers));
  for (int64_t l = 0; l < r.num_hidden_layers; ++l) {
    const std::string b = "vision_model.radio_model.model.blocks." + std::to_string(l) + ".";
    multimodal::RadioBlockWeights& w = out.radio.blocks[static_cast<size_t>(l)];
    w.norm1_w = take(b + "norm1.weight", {H});
    w.norm1_b = take(b + "norm1.bias", {H});
    w.qkv_w = take(b + "attn.qkv.weight", {3 * H, H});
    w.qkv_b = take(b + "attn.qkv.bias", {3 * H});
    w.proj_w = take(b + "attn.proj.weight", {H, H});
    w.proj_b = take(b + "attn.proj.bias", {H});
    w.norm2_w = take(b + "norm2.weight", {H});
    w.norm2_b = take(b + "norm2.bias", {H});
    w.fc1_w = take(b + "mlp.fc1.weight", {I, H});
    w.fc1_b = take(b + "mlp.fc1.bias", {I});
    w.fc2_w = take(b + "mlp.fc2.weight", {H, I});
    w.fc2_b = take(b + "mlp.fc2.bias", {H});
  }
  const multimodal::NanoNemotronVLProjectorConfig& pc = params.projector;
  out.projector.norm_w = take("mlp1.0.weight", {pc.in_dim});
  out.projector.fc1_w = take("mlp1.1.weight", {pc.hidden_dim, pc.in_dim});
  out.projector.fc2_w = take("mlp1.3.weight", {pc.out_dim, pc.hidden_dim});
  out.materialized = static_cast<int64_t>(consumed.size());

  // Deferred BY NAME, and nothing else may be left over.
  for (const auto& [name, t] : index) {
    (void)t;
    if (consumed.count(name) != 0) continue;
    if (name == "vision_model.radio_model.input_conditioner.norm_mean" ||
        name == "vision_model.radio_model.input_conditioner.norm_std") {
      // radio.py:715-718: the processor normalizes with the config's values.
      out.deferred.push_back(name + " (input conditioner; applied by the processor)");
      continue;
    }
    if (name == pg + "video_embedder.weight" ||
        name == "vision_model.radio_model.summary_idxs") {
      out.deferred.push_back(name + " (video modality / summary head, not ported)");
      continue;
    }
    Refuse(arch, "the checkpoint ships '" + name +
                     "', which no slot of the RADIO tower or mlp1 names");
  }
  return out;
}

}  // namespace vllm
