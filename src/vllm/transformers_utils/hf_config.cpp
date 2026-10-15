// vllm.cpp original container reader. Sliding-window normalization mirrors
// vllm/config/model.py:542-559,654-660,723-726,1232-1234; typed RoPE
// normalization mirrors vllm/transformers_utils/config.py:439-509 and
// model_executor/layers/rotary_embedding/__init__.py:33-112,200-230,243-283,
// 315-335
// @ e24d1b24fe96.
#include "vllm/transformers_utils/hf_config.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {

namespace {

int64_t GetInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<int64_t>();
}

double GetDouble(const nlohmann::json& doc, const char* key, double fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<double>();
}

std::string GetString(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::string>();
}

std::string CanonicalDType(std::string dtype) {
  if (dtype == "half" || dtype == "fp16" || dtype == "torch.float16")
    return "float16";
  if (dtype == "bf16" || dtype == "torch.bfloat16")
    return "bfloat16";
  if (dtype == "float" || dtype == "fp32" || dtype == "torch.float32")
    return "float32";
  return dtype;
}

std::vector<std::string> GetStringArray(const nlohmann::json& doc,
                                        const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::vector<std::string>>();
}

bool GetBool(const nlohmann::json& doc, const char* key, bool fallback) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return fallback;
  return it->get<bool>();
}

std::vector<int64_t> GetIntArray(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::vector<int64_t>>();
}

std::vector<double> GetDoubleArray(const nlohmann::json& doc,
                                   const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return {};
  return it->get<std::vector<double>>();
}

std::optional<double> GetOptionalDouble(const nlohmann::json& doc,
                                        const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return std::nullopt;
  return it->get<double>();
}

std::optional<int64_t> GetOptionalInt(const nlohmann::json& doc,
                                      const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return std::nullopt;
  return it->get<int64_t>();
}

std::optional<int64_t> GetSlidingWindow(const nlohmann::json& doc) {
  auto it = doc.find("sliding_window");
  if (it == doc.end() || it->is_null()) return std::nullopt;
  const int64_t window = it->get<int64_t>();
  // Pinned ModelConfig normalizes checkpoint sliding_window=0 to None before
  // max-length verification and backend construction (config/model.py:654-660).
  return window == 0 ? std::nullopt : std::optional<int64_t>(window);
}

void RequireKey(const nlohmann::json& doc, const char* key,
                const std::string& path) {
  auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) {
    throw std::runtime_error("hf_config: missing required field \"" +
                             std::string(key) + "\" in " + path);
  }
}

// Returns a pointer to the object member `key` of `doc` if it exists and is a
// JSON object, else nullptr.
const nlohmann::json* FindObject(const nlohmann::json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end() || !it->is_object()) return nullptr;
  return &*it;
}

// Resolves the effective "text config" json object, mirroring upstream
// PretrainedConfig.get_text_config() + the _CONFIG_ATTRS_MAPPING alias
// {"llm_config": "text_config"} (vllm/transformers_utils/config.py:134) and the
// thinker_config.text_config path (thinker_uses_mrope, config.py:529). Composite
// (multimodal wrapper) configs nest the text-model fields under a `text_config`
// (or `llm_config`, or `thinker_config.text_config`) sub-dict; plain dense
// configs have no such nesting and resolve to the top-level doc itself.
const nlohmann::json& ResolveTextConfig(const nlohmann::json& doc) {
  if (const nlohmann::json* text = FindObject(doc, "text_config")) return *text;
  if (const nlohmann::json* llm = FindObject(doc, "llm_config")) return *llm;
  if (const nlohmann::json* thinker = FindObject(doc, "thinker_config")) {
    if (const nlohmann::json* text = FindObject(*thinker, "text_config")) {
      return *text;
    }
  }
  return doc;
}

// True for the Qwen3.5 / Qwen3-Next family, whose upstream config classes
// default partial_rotary_factor to 0.25 (qwen3_next.py:240, qwen3_5_moe.py:92).
// Both the wrapper's top-level model_type ("qwen3_5_moe") and the nested text
// model_type ("qwen3_5_moe_text") carry the signal, so we check either.
bool IsQwen35Family(const std::string& model_type) {
  return model_type == "qwen3_next" || model_type == "qwen3_5" ||
         model_type == "qwen3_5_moe" || model_type == "qwen3_5_text" ||
         model_type == "qwen3_5_moe_text";
}

bool LooksLikeNestedRopeParameters(const nlohmann::json& params) {
  if (params.empty()) return false;
  for (const auto& item : params.items()) {
    if (!item.value().is_object()) return false;
  }
  return true;
}

RopeParameters ParseRopeParameters(const nlohmann::json& text,
                                   double default_partial_rotary_factor,
                                   const std::string& path,
                                   bool* has_parameters) {
  RopeParameters params;
  params.rope_theta = GetDouble(text, "rope_theta", 10000.0);
  params.partial_rotary_factor = GetDouble(
      text, "partial_rotary_factor", default_partial_rotary_factor);

  // Transformers v5 exposes rope_parameters. Older checkpoints use
  // rope_scaling; pinned patch_rope_parameters standardizes either dictionary
  // before get_rope sees it. Prefer the modern spelling when both are present.
  const nlohmann::json* raw = FindObject(text, "rope_parameters");
  if (raw == nullptr) raw = FindObject(text, "rope_scaling");
  *has_parameters = raw != nullptr;
  if (raw == nullptr) return params;

  if (LooksLikeNestedRopeParameters(*raw)) {
    // Per-layer-TYPE nested rope (Gemma-4: rope_parameters =
    // {full_attention:{rope_theta,partial_rotary_factor,rope_type:"proportional"},
    //  sliding_attention:{rope_theta,rope_type:"default"}}). vLLM keeps these as
    // per-layer-type rope configs on the model, and our Gemma-4 forward mirrors
    // that by reading them directly from config.raw (gemma4.cpp::RopeField /
    // MakeLayout — theta 1e6/1e4, proportional partial-RoPE). There is no single
    // flat typed rope for such a model, so we DO NOT synthesize one (the
    // "proportional" per-type rope_type is not a flat get_rope type either):
    // record that rope parameters exist and return the top-level defaults, which
    // the heterogeneous-rope model does not consume. ADDITIVE + byte-neutral:
    // every existing model previously ERRORED on this branch (nested rope was
    // unsupported), so no gate's typed rope can change — only a model that owns
    // its per-type rope in-forward (Gemma-4) now loads instead of aborting.
    *has_parameters = true;
    return params;
  }

  std::string modern_type = GetString(*raw, "rope_type");
  const std::string legacy_type = GetString(*raw, "type");
  if (!modern_type.empty() && !legacy_type.empty() &&
      modern_type != legacy_type &&
      !(legacy_type == "su" && modern_type == "longrope") &&
      !(legacy_type == "mrope" && modern_type == "default")) {
    throw std::runtime_error(
        "hf_config: conflicting rope_type '" + modern_type + "' and type '" +
        legacy_type + "' in " + path);
  }
  if (modern_type.empty()) modern_type = legacy_type;
  if (modern_type.empty()) modern_type = "default";
  if (modern_type == "su") modern_type = "longrope";
  if (modern_type == "mrope") {
    if (raw->find("mrope_section") == raw->end()) {
      throw std::runtime_error(
          "hf_config: legacy rope type 'mrope' requires mrope_section in " +
          path);
    }
    modern_type = "default";
  }
  params.rope_type = modern_type;

  params.rope_theta = GetDouble(*raw, "rope_theta", params.rope_theta);
  params.partial_rotary_factor = GetDouble(
      *raw, "partial_rotary_factor", params.partial_rotary_factor);
  params.rope_dim = GetOptionalInt(*raw, "rope_dim");
  if (params.rope_dim.has_value() && *params.rope_dim == 0) {
    // Python's `if rotary_dim := ...` treats zero as absent.
    params.rope_dim.reset();
  }
  params.factor = GetOptionalDouble(*raw, "factor");
  params.original_max_position_embeddings =
      GetOptionalInt(*raw, "original_max_position_embeddings");
  // Phi-3/Phi-4 longrope (and some other checkpoints) place
  // original_max_position_embeddings at the TOP LEVEL of config.json rather than
  // inside the rope dict; transformers' standardize_rope_params folds it into the
  // rope parameters before vLLM's get_rope sees it. Mirror that: when the rope dict
  // omits it, fall back to the top-level field. Inert for every checkpoint whose
  // rope dict already carries the field (yarn/llama3 configs) and for default-rope
  // models (the field is unused unless rope_type consumes it).
  if (!params.original_max_position_embeddings.has_value())
    params.original_max_position_embeddings =
        GetOptionalInt(text, "original_max_position_embeddings");
  params.low_freq_factor = GetOptionalDouble(*raw, "low_freq_factor");
  params.high_freq_factor = GetOptionalDouble(*raw, "high_freq_factor");
  params.short_factor = GetDoubleArray(*raw, "short_factor");
  params.long_factor = GetDoubleArray(*raw, "long_factor");
  params.short_mscale = GetOptionalDouble(*raw, "short_mscale");
  params.long_mscale = GetOptionalDouble(*raw, "long_mscale");
  params.alpha = GetOptionalDouble(*raw, "alpha");
  params.max_trained_positions =
      GetOptionalInt(*raw, "max_trained_positions");
  params.extrapolation_factor =
      GetDouble(*raw, "extrapolation_factor", 1.0);
  params.attn_factor = GetDouble(*raw, "attn_factor", 1.0);
  params.beta_fast = GetInt(*raw, "beta_fast", 32);
  params.beta_slow = GetInt(*raw, "beta_slow", 1);
  params.apply_yarn_scaling =
      GetBool(*raw, "apply_yarn_scaling", true);
  params.truncate = GetBool(*raw, "truncate", true);
  params.mrope_section = GetIntArray(*raw, "mrope_section");
  params.mrope_interleaved = GetBool(*raw, "mrope_interleaved", false);

  if (params.rope_type == "linear") {
    if (!params.factor.has_value() || !std::isfinite(*params.factor) || *params.factor <= 0.) {
      throw std::runtime_error("hf_config: linear rope requires factor finite and positive in " + path);
    }
  } else if (params.rope_type == "yarn") {
    if (!params.factor.has_value() ||
        !params.original_max_position_embeddings.has_value()) {
      throw std::runtime_error(
          "hf_config: yarn rope requires factor and "
          "original_max_position_embeddings in " +
          path);
    }
    if (!(*params.factor > 0.0) ||
        *params.original_max_position_embeddings <= 0) {
      throw std::runtime_error(
          "hf_config: yarn factor and original_max_position_embeddings must "
          "be positive in " +
          path);
    }
  } else if (params.rope_type == "llama3") {
    if (!params.factor.has_value() ||
        !params.original_max_position_embeddings.has_value() ||
        !params.low_freq_factor.has_value() ||
        !params.high_freq_factor.has_value()) {
      throw std::runtime_error(
          "hf_config: llama3 rope requires factor, low_freq_factor, "
          "high_freq_factor, and original_max_position_embeddings in " +
          path);
    }
    if (!std::isfinite(*params.factor) || !(*params.factor > 0.0) ||
        !std::isfinite(*params.low_freq_factor) ||
        !(*params.low_freq_factor > 0.0) ||
        !std::isfinite(*params.high_freq_factor) ||
        !(*params.high_freq_factor > 0.0) ||
        *params.original_max_position_embeddings <= 0) {
      throw std::runtime_error(
          "hf_config: llama3 scaling and frequency factors plus "
          "original_max_position_embeddings must be finite and positive in " +
          path);
    }
  } else if (params.rope_type == "longrope") {
    if (!params.original_max_position_embeddings.has_value() ||
        params.short_factor.empty() || params.long_factor.empty()) {
      throw std::runtime_error(
          "hf_config: longrope requires short_factor, long_factor, and "
          "original_max_position_embeddings in " +
          path);
    }
    if (*params.original_max_position_embeddings <= 0) {
      throw std::runtime_error(
          "hf_config: longrope original_max_position_embeddings must be "
          "positive in " +
          path);
    }
  } else if (params.rope_type == "dynamic") {
    if (!params.alpha.has_value() && !params.factor.has_value()) {
      throw std::runtime_error(
          "hf_config: dynamic rope requires either alpha or factor in " +
          path);
    }
    if (params.max_trained_positions.has_value() &&
        *params.max_trained_positions <= 0) {
      throw std::runtime_error(
          "hf_config: dynamic max_trained_positions must be positive in " +
          path);
    }
  } else if (params.rope_type != "default") {
    throw std::runtime_error(
        "hf_config: checkpoint declares rope type '" + params.rope_type +
        "' which vllm.cpp does not implement yet (supported: default, yarn, "
        "llama3, longrope, dynamic, linear) in " + path);
  }

  if (!params.rope_dim.has_value() &&
      (!(params.partial_rotary_factor > 0.0) ||
       params.partial_rotary_factor > 1.0)) {
    throw std::runtime_error(
        "hf_config: partial_rotary_factor must be in (0, 1] in " + path);
  }
  return params;
}

}  // namespace

std::vector<std::string> PeekHfArchitectures(const std::string& path) {
  // Non-throwing by contract (see the header): any problem yields {} so the
  // caller's ordinary LoadHfConfig path keeps owning every diagnostic.
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  nlohmann::json doc = nlohmann::json::parse(in, /*cb=*/nullptr,
                                             /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return {};
  const auto it = doc.find("architectures");
  if (it == doc.end() || !it->is_array()) {
    // cua-s1-forms models (MODEL-CUA-S1-FORMS) ship cua-s1-forms.json
    // with a "config" envelope wrapping width/layers/heads and no
    // architectures field. Detect by the envelope signature and synthesize
    // the architecture so the registry can resolve it.
    if (doc.contains("config") && doc["config"].is_object() &&
        doc["config"].contains("width") && doc["config"].contains("layers")) {
      return {"CuaS1Forms"};
    }
    // Laya models (MODEL-LAYA) ship rl_agent_config.json with no
    // architectures field. Detect by the encoder + head_layers signature
    // and synthesize the architecture so the registry can resolve it.
    if (doc.contains("encoder") && doc.contains("head_layers")) {
      return {"LayaModel"};
    }
    return {};
  }
  std::vector<std::string> archs;
  for (const auto& a : *it) {
    if (!a.is_string()) return {};
    archs.push_back(a.get<std::string>());
  }
  return archs;
}

namespace {

// The sibling generation_config.json of a config.json path (HF checkpoint
// layout). `path` is the config.json file itself.
std::string SiblingGenerationConfigPath(const std::string& path) {
  const auto slash = path.find_last_of("/\\");
  const std::string dir = (slash == std::string::npos) ? std::string(".")
                                                       : path.substr(0, slash);
  return dir + "/generation_config.json";
}

// generation_config.json's eos_token_id (int OR list) as a sorted unique list.
// Upstream ModelConfig.try_get_generation_config loads this file for the
// default --generation-config=auto; a missing or malformed file is not an
// error there either (try_get_generation_config returns {}), so every failure
// path here yields an empty list rather than throwing.
std::vector<int32_t> ReadGenerationConfigEosIds(const std::string& path) {
  std::vector<int32_t> out;
  std::ifstream in(SiblingGenerationConfigPath(path), std::ios::binary);
  if (!in) return out;
  nlohmann::json gen = nlohmann::json::parse(in, /*cb=*/nullptr,
                                             /*allow_exceptions=*/false);
  if (gen.is_discarded() || !gen.is_object()) return out;
  const auto it = gen.find("eos_token_id");
  if (it == gen.end() || it->is_null()) return out;
  std::set<int32_t> ids;
  if (it->is_number_integer()) {
    ids.insert(it->get<int32_t>());
  } else if (it->is_array()) {
    for (const auto& e : *it) {
      if (e.is_number_integer()) ids.insert(e.get<int32_t>());
    }
  }
  out.assign(ids.begin(), ids.end());
  return out;
}

// tokenizer_config.json's eos_token beside `path` (config.json): a string, or
// an AddedToken object whose "content" is the string. Empty on any failure,
// with the same never-throw polarity as the generation_config.json reads.
std::string ReadTokenizerConfigEosToken(const std::string& path) {
  const auto slash = path.find_last_of("/\\");
  const std::string dir = (slash == std::string::npos) ? std::string(".")
                                                       : path.substr(0, slash);
  std::ifstream in(dir + "/tokenizer_config.json", std::ios::binary);
  if (!in) return {};
  nlohmann::json doc = nlohmann::json::parse(in, /*cb=*/nullptr,
                                             /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return {};
  const auto it = doc.find("eos_token");
  if (it == doc.end()) return {};
  if (it->is_string()) return it->get<std::string>();
  if (it->is_object()) {
    const auto content = it->find("content");
    if (content != it->end() && content->is_string()) {
      return content->get<std::string>();
    }
  }
  return {};
}

// One optional numeric key out of a generation_config.json object. Absent,
// null and wrong-typed all read as "the checkpoint did not declare it", which
// is upstream's polarity: try_get_generation_config never raises, and
// get_diff_sampling_param's `if config.get(p) is not None` drops the key.
template <typename T>
std::optional<T> OptionalNumber(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return std::nullopt;
  return it->get<T>();
}

// The six sampling keys of an already-parsed generation_config.json document.
GenerationConfigSampling SamplingFromDoc(const nlohmann::json& gen) {
  GenerationConfigSampling out;
  if (!gen.is_object()) return out;
  out.repetition_penalty = OptionalNumber<double>(gen, "repetition_penalty");
  out.temperature = OptionalNumber<double>(gen, "temperature");
  out.top_k = OptionalNumber<int>(gen, "top_k");
  out.top_p = OptionalNumber<double>(gen, "top_p");
  out.min_p = OptionalNumber<double>(gen, "min_p");
  out.max_new_tokens = OptionalNumber<int>(gen, "max_new_tokens");
  return out;
}

// generation_config.json's sampling keys, read from an explicit file path.
GenerationConfigSampling ReadGenerationConfigSamplingAt(const std::string& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  nlohmann::json gen = nlohmann::json::parse(in, /*cb=*/nullptr,
                                             /*allow_exceptions=*/false);
  if (gen.is_discarded()) return {};
  return SamplingFromDoc(gen);
}

}  // namespace

namespace {

// The whole parse, shared by the path and the in-memory entry points. `path` is
// what error messages name; `sibling_generation_config` is false when there is
// no file and therefore no sibling generation_config.json to read.
HfConfig ParseHfConfigDoc(nlohmann::json doc, const std::string& path,
                          bool sibling_generation_config) {
  if (!doc.is_object()) {
    throw std::runtime_error("hf_config: top-level JSON is not an object in " +
                             path);
  }

  // cua-s1-forms models (MODEL-CUA-S1-FORMS) ship cua-s1-forms.json with
  // a "config" envelope wrapping the model params (width, layers, heads,
  // context_tokens, option_tokens, rank) and no model_type or architectures
  // field. Unwrap the envelope, inject the identity, and skip the
  // required-field checks (params are parsed from config.raw in
  // cua_s1_weights.cpp). Done before ResolveTextConfig so the reference is
  // taken on the unwrapped doc.
  if (!doc.contains("model_type") && doc.contains("config") &&
      doc["config"].is_object() && doc["config"].contains("width") &&
      doc["config"].contains("layers")) {
    doc = doc["config"];
    doc["model_type"] = "cua_s1_forms";
    doc["architectures"] = nlohmann::json::array({"CuaS1Forms"});
  }

  // Resolve the effective text config: for multimodal wrapper configs (e.g.
  // Qwen3_5MoeForConditionalGeneration) the text-model fields are nested under
  // `text_config`; for plain dense configs `text` aliases `doc`. `architectures`
  // and `model_type` are always read from the top-level wrapper doc.
  const nlohmann::json& text = ResolveTextConfig(doc);

  // Laya models (MODEL-LAYA) ship rl_agent_config.json with no model_type.
  // Detect by the encoder + head_layers signature and inject the identity
  // so the registry can resolve it. The encoder params are inferred from
  // weight shapes in laya_weights.cpp, so no hidden_size/num_hidden_layers.
  if (!doc.contains("model_type") && doc.contains("encoder") &&
      doc.contains("head_layers")) {
    doc["model_type"] = "laya";
    doc["architectures"] = nlohmann::json::array({"LayaModel"});
  }

  RequireKey(doc, "model_type", path);

  // GLiNER2.5 configs (model_type "extractor") carry no encoder fields —
  // hidden_size and num_hidden_layers are inferred from weight shapes in
  // gliner2_weights.cpp::InferEncoderParams, so skip the requirement here.
  // cua-s1-forms configs (model_type "cua_s1_forms") are the same: params are
  // parsed from config.raw in cua_s1_weights.cpp.
  // Laya configs (model_type "laya") are the same: inferred from weight shapes.
  const bool is_extractor = GetString(doc, "model_type") == "extractor";
  const bool is_cua_s1 = GetString(doc, "model_type") == "cua_s1_forms";
  const bool is_laya = GetString(doc, "model_type") == "laya";
  if (!is_extractor && !is_cua_s1 && !is_laya) {
    RequireKey(text, "hidden_size", path);
    RequireKey(text, "num_hidden_layers", path);
  }

  HfConfig cfg;
  try {
    cfg.model_type = GetString(doc, "model_type");
    cfg.architectures = GetStringArray(doc, "architectures");
    cfg.hidden_size = GetInt(text, "hidden_size", 0);
    cfg.num_hidden_layers = GetInt(text, "num_hidden_layers", 0);
    cfg.vocab_size = GetInt(text, "vocab_size", 0);
    cfg.num_attention_heads = GetInt(text, "num_attention_heads", 0);
    // Absent -> MHA, per upstream convention.
    cfg.num_key_value_heads =
        GetInt(text, "num_key_value_heads", cfg.num_attention_heads);
    int64_t derived_head_dim =
        cfg.num_attention_heads > 0 ? cfg.hidden_size / cfg.num_attention_heads
                                    : 0;
    // Upstream only honors an explicit head_dim when it is > 0
    // (model_arch_config_convertor.py:61-75); absent or non-positive falls
    // back to hidden_size / num_attention_heads.
    cfg.head_dim = GetInt(text, "head_dim", 0);
    if (cfg.head_dim <= 0) cfg.head_dim = derived_head_dim;
    cfg.sliding_window = GetSlidingWindow(text);
    cfg.layer_types = GetStringArray(text, "layer_types");
    cfg.intermediate_size = GetInt(text, "intermediate_size", 0);

    cfg.num_experts = GetInt(text, "num_experts", 0);
    cfg.num_experts_per_tok = GetInt(text, "num_experts_per_tok", 0);
    cfg.moe_intermediate_size = GetInt(text, "moe_intermediate_size", 0);
    cfg.shared_expert_intermediate_size =
        GetInt(text, "shared_expert_intermediate_size", 0);

    cfg.linear_num_key_heads = GetInt(text, "linear_num_key_heads", 0);
    cfg.linear_num_value_heads = GetInt(text, "linear_num_value_heads", 0);
    cfg.linear_key_head_dim = GetInt(text, "linear_key_head_dim", 0);
    cfg.linear_value_head_dim = GetInt(text, "linear_value_head_dim", 0);
    cfg.linear_conv_kernel_dim = GetInt(text, "linear_conv_kernel_dim", 0);
    // GDN output-gate activation, mirroring
    // vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:452-456
    // @555967922:
    //   output_gate_type = getattr(config, "output_gate_type", "silu")
    //   if output_gate_type == "swish": output_gate_type = "silu"
    //   assert output_gate_type in ["silu", "swish", "sigmoid"]
    // Read from the RESOLVED text config so a nested VL wrapper and a flat
    // text-only config behave alike, and canonicalized HERE so no consumer can
    // reintroduce the default by forgetting to normalize. Upstream asserts on
    // an unrecognized value; we refuse at load naming the key and the accepted
    // set, because a silent fallback to silu is a numerics change no token gate
    // over today's (all-silu) checkpoints could ever see.
    //
    // ABSENT and PRESENT-BUT-UNUSABLE are different states, so this cannot go
    // through GetString(), which flattens both to "": `getattr` substitutes the
    // default ONLY when the attribute is missing, and a present None / "" /
    // non-string is handed straight to the assert and errors. Probing for the
    // key keeps null and "" on the refusal path.
    //
    // The refusal is unconditional rather than gated on the architecture being
    // GDN, where upstream's assert lives. No checkpoint we know of carries the
    // key outside the GDN family, and refusing a value nothing can honor is the
    // safer direction; if one ever appears, that is a scoped follow-up with its
    // own test, not a silent widening here.
    const auto gate_it = text.find("output_gate_type");
    if (gate_it == text.end()) {
      cfg.output_gate_type = "silu";  // upstream's getattr default
    } else {
      // A non-string value is dumped verbatim (`null`, `3`) so the refusal
      // names what was actually found; it can never match silu/sigmoid.
      cfg.output_gate_type =
          gate_it->is_string() ? gate_it->get<std::string>() : gate_it->dump();
      if (cfg.output_gate_type == "swish") cfg.output_gate_type = "silu";
      if (cfg.output_gate_type != "silu" && cfg.output_gate_type != "sigmoid") {
        throw std::runtime_error(
            "hf_config: unsupported output_gate_type \"" +
            cfg.output_gate_type +
            "\" (expected one of: silu, swish, sigmoid) in " + path);
      }
    }
    cfg.mamba_ssm_dtype = GetString(text, "mamba_ssm_dtype");

    // Kimi-Linear (`KimiLinearForCausalLM`) KV enablement for the shared paged
    // runner (kimi-linear.md §20.3 B1). Kimi's config carries NO `layer_types`
    // and NONE of the explicit qwen3_5-style `linear_*` keys read above: its
    // KDA/full-attn split and GDN-group geometry live in the nested
    // `linear_attn_config` (upstream transformers_utils/configs/kimi_linear.py
    // :34-148; `is_kda_layer(l) := (l+1) in kda_layers` :144-148 — the layer
    // lists are 1-INDEXED; `num_heads`/`head_dim`/`short_conv_kernel_size`
    // :109-119 are what MambaStateShapeCalculator.kda_state_shape derives the
    // conv/recurrent state from, mamba_utils.py:270-294). Synthesize the typed
    // runner-facing fields from it so the runner's MambaSpec consistency check
    // (runner.cpp `expected_conv_shape`/`expected_ssm_shape`) and its per-layer
    // linear-attention/full-attention allocation loop see the same geometry
    // `MakeKimiLinearKVCache` declares. ADDITIVE by construction: configs that
    // carry the explicit fields (the qwen3_5/qwen3-next family) never enter —
    // the explicit branch above already populated them — and configs with no
    // `linear_attn_config` (every other arch) skip it entirely. KDA has
    // num_k_heads == num_v_heads == num_heads and Dk == Dv == head_dim
    // (kimi_gdn_linear_attn.py:120-141), so conv_dim = 2*Hk*Dk + Hv*Dv equals
    // kda_state_shape's 3*num_heads*head_dim.
    if (cfg.linear_num_key_heads == 0 && text.contains("linear_attn_config") &&
        text["linear_attn_config"].is_object()) {
      const nlohmann::json& lac = text["linear_attn_config"];
      const int64_t kda_heads = GetInt(lac, "num_heads", 0);
      const int64_t kda_head_dim = GetInt(lac, "head_dim", 0);
      const int64_t kda_conv = GetInt(lac, "short_conv_kernel_size", 0);
      const bool has_kda_layers = lac.contains("kda_layers") &&
                                  lac["kda_layers"].is_array() &&
                                  !lac["kda_layers"].empty();
      if (kda_heads > 0 && kda_head_dim > 0 && kda_conv > 0 && has_kda_layers) {
        cfg.linear_num_key_heads = kda_heads;
        cfg.linear_num_value_heads = kda_heads;
        cfg.linear_key_head_dim = kda_head_dim;
        cfg.linear_value_head_dim = kda_head_dim;
        cfg.linear_conv_kernel_dim = kda_conv;
        if (cfg.layer_types.empty() && cfg.num_hidden_layers > 0) {
          std::vector<bool> is_kda(static_cast<size_t>(cfg.num_hidden_layers),
                                   false);
          for (const nlohmann::json& e : lac["kda_layers"]) {
            if (!e.is_number_integer()) continue;
            const int64_t one_indexed = e.get<int64_t>();
            if (one_indexed >= 1 && one_indexed <= cfg.num_hidden_layers) {
              is_kda[static_cast<size_t>(one_indexed - 1)] = true;
            }
          }
          cfg.layer_types.reserve(static_cast<size_t>(cfg.num_hidden_layers));
          for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
            cfg.layer_types.push_back(is_kda[static_cast<size_t>(l)]
                                          ? "linear_attention"
                                          : "full_attention");
          }
        }
      }
    }

    // Partial rotary factor. When the key is absent, upstream Qwen-family
    // config classes default it to 0.25 (qwen3_next.py:240, qwen3_5_moe.py:92);
    // all other models default to full rotary (1.0). The wrapper carries the
    // qwen signal on the top-level model_type ("qwen3_5_moe") while the nested
    // text config carries it as "qwen3_5_moe_text" -- check either.
    double default_partial_rotary_factor = 1.0;
    if (IsQwen35Family(cfg.model_type) ||
        IsQwen35Family(GetString(text, "model_type"))) {
      default_partial_rotary_factor = 0.25;
    }
    cfg.rope_parameters = ParseRopeParameters(
        text, default_partial_rotary_factor, path, &cfg.has_rope_parameters);
    cfg.rope_theta = cfg.rope_parameters.rope_theta;
    // Upstream get_rope gives a truthy explicit rope_dim precedence; otherwise
    // it truncates int(head_dim * partial_rotary_factor).
    cfg.rotary_dim = cfg.rope_parameters.rope_dim.has_value()
                         ? *cfg.rope_parameters.rope_dim
                         : static_cast<int64_t>(
                               cfg.rope_parameters.partial_rotary_factor *
                               static_cast<double>(cfg.head_dim));

    cfg.rms_norm_eps = GetDouble(text, "rms_norm_eps", 0.0);
    cfg.max_position_embeddings = GetInt(text, "max_position_embeddings", 0);
    // Modern HF configs declare `dtype`; older ones use `torch_dtype`.
    // Resolve once for the text model and reject conflicting declarations,
    // including disagreements between a wrapper and its nested text config.
    const auto read_dtype = [&cfg, &path](const nlohmann::json& owner,
                                          const char* key,
                                          const std::string& source) {
      const std::string value = CanonicalDType(GetString(owner, key));
      if (value.empty()) return;
      if (!cfg.torch_dtype.empty() && cfg.torch_dtype != value) {
        throw std::runtime_error(
            "hf_config: conflicting dtype declarations in " + path +
            " (" + cfg.dtype_source + "=" + cfg.torch_dtype + ", " +
            source + "=" + value + ")");
      }
      if (cfg.torch_dtype.empty()) {
        cfg.torch_dtype = value;
        cfg.dtype_source = source;
      }
    };
    read_dtype(text, "dtype", &text == &doc ? "dtype" : "text.dtype");
    read_dtype(text, "torch_dtype", &text == &doc ? "torch_dtype"
                                                     : "text.torch_dtype");
    if (&text != &doc) {
      read_dtype(doc, "dtype", "dtype");
      read_dtype(doc, "torch_dtype", "torch_dtype");
    }
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf_config: bad field type in " + path + ": " +
                             e.what());
  }

  // from_model_config's eos, read before `doc` moves into raw, sorted and
  // unique like the file's. The OUTER config's eos_token_id wins; the text
  // config's is used only when the outer one is unset or null, as
  // GenerationConfig.from_model_config resolves it.
  std::vector<int32_t> model_config_eos_ids;
  {
    const auto named = [](const nlohmann::json& j) -> const nlohmann::json* {
      const auto it = j.find("eos_token_id");
      return (it == j.end() || it->is_null()) ? nullptr : &*it;
    };
    const nlohmann::json* eos = named(doc);
    if (eos == nullptr) eos = named(text);
    if (eos != nullptr) {
      std::set<int32_t> ids;
      if (eos->is_number_integer()) {
        ids.insert(eos->get<int32_t>());
      } else if (eos->is_array()) {
        for (const auto& e : *eos) {
          if (e.is_number_integer()) ids.insert(e.get<int32_t>());
        }
      }
      model_config_eos_ids.assign(ids.begin(), ids.end());
    }
  }

  cfg.raw = std::move(doc);
  if (sibling_generation_config) {
    const bool has_generation_config =
        std::ifstream(SiblingGenerationConfigPath(path), std::ios::binary).good();
    cfg.generation_config_eos_ids = ReadGenerationConfigEosIds(path);
    if (!has_generation_config) {
      cfg.model_config_eos_ids = std::move(model_config_eos_ids);
    }
    cfg.tokenizer_eos_token = ReadTokenizerConfigEosToken(path);
    cfg.generation_config_sampling =
        ReadGenerationConfigSamplingAt(SiblingGenerationConfigPath(path));
  }
  return cfg;
}

}  // namespace

HfConfig LoadHfConfig(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("hf_config: cannot open " + path);
  }
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf_config: JSON parse error in " + path + ": " +
                             e.what());
  }
  return ParseHfConfigDoc(std::move(doc), path, /*sibling_generation_config=*/true);
}

HfConfig ParseHfConfig(const nlohmann::json& doc, const std::string& source) {
  return ParseHfConfigDoc(doc, source, /*sibling_generation_config=*/false);
}

GenerationConfigSampling ReadGenerationConfigSamplingFile(const std::string& path) {
  return ReadGenerationConfigSamplingAt(path);
}

}  // namespace vllm
