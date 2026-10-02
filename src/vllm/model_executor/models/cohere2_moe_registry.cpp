// Cohere2MoeForCausalLM registry TU: the config hook, the KV-cache spec, the
// LoadedModel subclass and the factory, self-registered through ONE
// REGISTER_VLLM_MODEL line (no shared-array edit).
//
// Config grounding: vllm/model_executor/models/cohere2_moe.py @ a7c23ac96d and
// the HF config class it instantiates, transformers' `Cohere2MoeConfig`
// (`model_type: cohere2_moe`). vLLM reads the config through that class, so a
// key the checkpoint omits reaches the model file as the CLASS default. That is
// why the defaults below are the class's (logit_scale 0.0625, sliding_window
// 4096, expert_selection_fn "softmax", norm_topk_prob True, head_dim 128,
// shared_expert_combination_strategy "average", sliding_window_pattern 4) and
// not the `getattr` fallbacks the model file spells, which the class makes
// unreachable.
#include "vllm/model_executor/models/model_registry.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/cohere2_moe.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kCohere2MoeInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Cohere2MoeLoadedModel final : public LoadedModel {
 public:
  Cohere2MoeLoadedModel(const ModelRegistration& registration, Cohere2MoeWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Cohere2MoeWeights& weights() const { return weights_; }

 private:
  Cohere2MoeWeights weights_;
};

std::unique_ptr<LoadedModel> LoadCohere2MoeForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    // llama.cpp b10451 defines `cohere2moe`; the GGUF loader arm is owed
    // (ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY).
    throw std::runtime_error(
        "Model architecture Cohere2MoeForCausalLM does not support GGUF weights yet: "
        "the llama.cpp `cohere2moe` loader arm is not implemented");
  }
  if (source.safetensors == nullptr)
    throw std::runtime_error("safetensors model source is empty");
  return std::make_unique<Cohere2MoeLoadedModel>(
      registration, LoadCohere2MoeWeights(*source.safetensors, config));
}

void PrepareCohere2MoeForCausalLM(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardCohere2MoeForCausalLM(LoadedModel& model,
                                           const ModelForwardInput& input) {
  const auto& cm = ModelAs<Cohere2MoeLoadedModel>(model, "Cohere2MoeForCausalLM");
  const Cohere2MoeWeights& weights = cm.weights();
  if (input.gather_logits) {
    return Cohere2MoeModel::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  return HostLogits(Cohere2MoeModel::Forward(input.token_ids, input.positions,
                                             input.attn_meta, input.attn_kv, weights,
                                             input.queue, input.logits_indices),
                    weights.params.vocab_size);
}

const ModelFactory kCohere2MoeFactory{
    .parse_config = &ParseCohere2MoeForCausalLMConfig,
    .load_weights = &LoadCohere2MoeForCausalLM,
    .prepare = &PrepareCohere2MoeForCausalLM,
    .forward = &ForwardCohere2MoeForCausalLM,
    .make_kv_cache = &MakeCohere2MoeKVCache,
    .is_dense_model = false,
};

const nlohmann::json* Find(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &*it;
}

double NumberOr(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* v = Find(doc, key);
  if (v == nullptr) return fallback;
  VT_CHECK(v->is_number(), std::string("cohere2_moe: config key ") + key +
                               " must be a number");
  return v->get<double>();
}

int64_t IntOr(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* v = Find(doc, key);
  if (v == nullptr) return fallback;
  VT_CHECK(v->is_number_integer(), std::string("cohere2_moe: config key ") + key +
                                       " must be an integer");
  return v->get<int64_t>();
}

bool BoolOr(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* v = Find(doc, key);
  if (v == nullptr) return fallback;
  VT_CHECK(v->is_boolean(), std::string("cohere2_moe: config key ") + key +
                                " must be a boolean");
  return v->get<bool>();
}

std::string StringOr(const nlohmann::json& doc, const char* key,
                     const std::string& fallback) {
  const nlohmann::json* v = Find(doc, key);
  if (v == nullptr) return fallback;
  VT_CHECK(v->is_string(), std::string("cohere2_moe: config key ") + key +
                               " must be a string");
  return v->get<std::string>();
}

std::vector<std::string> StringList(const nlohmann::json& doc, const char* key) {
  const nlohmann::json* v = Find(doc, key);
  std::vector<std::string> out;
  if (v == nullptr) return out;
  VT_CHECK(v->is_array(), std::string("cohere2_moe: config key ") + key +
                              " must be a list");
  for (const auto& e : *v) {
    VT_CHECK(e.is_string(), std::string("cohere2_moe: config key ") + key +
                                " must be a list of strings");
    out.push_back(e.get<std::string>());
  }
  return out;
}

}  // namespace

Cohere2MoeParams ParseCohere2MoeParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  Cohere2MoeParams p;
  p.hidden_size = config.hidden_size;
  p.num_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.num_heads = config.num_attention_heads;
  p.num_kv_heads = config.num_key_value_heads;
  // `head_dim: int = 128` on the config class; the model file's
  // `getattr(config, "head_dim", hidden // heads)` (:166-168) never falls back.
  p.head_dim = IntOr(raw, "head_dim", 128);
  VT_CHECK(p.hidden_size > 0 && p.num_layers > 0 && p.vocab_size > 0,
           "cohere2_moe: hidden_size, num_hidden_layers and vocab_size must be positive");
  VT_CHECK(p.num_heads > 0 && p.num_kv_heads > 0 && p.head_dim > 0,
           "cohere2_moe: head counts and head_dim must be positive");
  VT_CHECK(p.num_heads % p.num_kv_heads == 0,
           "cohere2_moe: num_attention_heads must be divisible by num_key_value_heads");

  // Refusals by name. `use_qk_norm` is not a mechanism of the pinned
  // cohere2_moe.py at all, so a checkpoint that asks for it cannot be mirrored.
  VT_CHECK(!BoolOr(raw, "use_qk_norm", false),
           "cohere2_moe: use_qk_norm=true is not supported: the pinned vLLM "
           "cohere2_moe.py (a7c23ac96d) has no q/k norm to mirror");
  VT_CHECK(Find(raw, "quantization_config") == nullptr,
           "cohere2_moe: quantized checkpoints (quantization_config) are not "
           "supported yet: the FP8 / W4A16 / NVFP4 arms are owed; load the bf16 "
           "checkpoint");
  VT_CHECK(config.rope_parameters.rope_type == "default",
           "cohere2_moe: rope_scaling '" + config.rope_parameters.rope_type +
               "' is not supported; only the default rope is ported");
  VT_CHECK(config.rope_parameters.partial_rotary_factor == 1.0,
           "cohere2_moe: partial_rotary_factor != 1 is not supported (get_rope "
           "receives the full head_dim at cohere2_moe.py:198-203)");
  p.rope_theta = config.rope_parameters.rope_theta;

  // max_position (cohere2_moe.py:178-180): model_max_length or
  // max_position_embeddings (class default 8192).
  p.max_position = IntOr(raw, "model_max_length", 0);
  if (p.max_position <= 0) p.max_position = IntOr(raw, "max_position_embeddings", 8192);
  VT_CHECK(p.max_position > 0, "cohere2_moe: max_position_embeddings must be positive");

  // Norm (cohere2_moe.py:97-103).
  const nlohmann::json* rms = Find(raw, "rms_norm_eps");
  p.use_rms_norm = rms != nullptr;
  p.norm_eps = static_cast<float>(p.use_rms_norm ? NumberOr(raw, "rms_norm_eps", 0.0)
                                                 : NumberOr(raw, "layer_norm_eps", 1e-5));

  p.logit_scale = NumberOr(raw, "logit_scale", 0.0625);

  // MoE (cohere2_moe.py:249-317).
  p.intermediate_size = config.intermediate_size;
  VT_CHECK(p.intermediate_size > 0, "cohere2_moe: intermediate_size must be positive");
  p.prefix_dense_intermediate_size =
      IntOr(raw, "prefix_dense_intermediate_size", p.intermediate_size);
  p.num_experts = IntOr(raw, "num_experts", 8);
  p.top_k = IntOr(raw, "num_experts_per_tok", 2);
  VT_CHECK(p.num_experts > 0 && p.top_k > 0 && p.top_k <= p.num_experts,
           "cohere2_moe: num_experts_per_tok must be in [1, num_experts]");
  p.num_shared_experts = IntOr(raw, "num_shared_experts", 0);
  VT_CHECK(p.num_shared_experts >= 0, "cohere2_moe: num_shared_experts must be >= 0");
  if (p.num_shared_experts > 0) {
    const std::string strategy =
        StringOr(raw, "shared_expert_combination_strategy", "average");
    VT_CHECK(strategy == "average" || strategy == "sum",
             "cohere2_moe: shared_expert_combination_strategy must be one of "
             "['average', 'sum'] (got '" + strategy + "')");
    p.shared_average = strategy == "average";
  }
  p.sigmoid_router = StringOr(raw, "expert_selection_fn", "softmax") == "sigmoid";
  p.norm_topk_prob = BoolOr(raw, "norm_topk_prob", true);

  // layer_types / mlp_layer_types. The config class derives layer_types from
  // first_k_dense_replace with prefix_dense_sliding_window_pattern and
  // sliding_window_pattern when absent; cohere2_moe.py:410-418 derives
  // mlp_layer_types from first_k_dense_replace when absent.
  const int64_t L = p.num_layers;
  const int64_t first_k_dense = IntOr(raw, "first_k_dense_replace", 0);
  VT_CHECK(first_k_dense >= 0 && first_k_dense <= L,
           "cohere2_moe: first_k_dense_replace must be in [0, num_hidden_layers]");
  const int64_t prefix_pattern = IntOr(raw, "prefix_dense_sliding_window_pattern", 1);
  VT_CHECK(prefix_pattern > 0,
           "cohere2_moe: prefix_dense_sliding_window_pattern must be positive");
  std::vector<std::string> layer_types = StringList(raw, "layer_types");
  if (layer_types.empty()) {
    const int64_t pattern = IntOr(raw, "sliding_window_pattern", 4);
    VT_CHECK(pattern > 0, "cohere2_moe: sliding_window_pattern must be positive");
    for (int64_t i = 0; i < first_k_dense; ++i)
      layer_types.push_back((i + 1) % prefix_pattern != 0 ? "sliding_attention"
                                                          : "full_attention");
    for (int64_t i = 0; i < L - first_k_dense; ++i)
      layer_types.push_back((i + 1) % pattern != 0 ? "sliding_attention"
                                                   : "full_attention");
  }
  VT_CHECK(static_cast<int64_t>(layer_types.size()) == L,
           "cohere2_moe: layer_types must have num_hidden_layers entries");
  std::vector<std::string> mlp_types = StringList(raw, "mlp_layer_types");
  if (mlp_types.empty()) {
    for (int64_t i = 0; i < L; ++i)
      mlp_types.push_back(i < first_k_dense ? "dense" : "sparse");
  }
  VT_CHECK(static_cast<int64_t>(mlp_types.size()) == L,
           "cohere2_moe: mlp_layer_types must have num_hidden_layers entries");

  // sliding_window: class default 4096 when the key is absent. An explicit
  // null, or 0 (which ModelConfig rewrites to None, config/model.py:762-764),
  // reaches `config.sliding_window + 1` (:211) as None and fails upstream.
  int64_t sliding_window = 4096;
  if (raw.contains("sliding_window")) {
    const nlohmann::json& sw = raw.at("sliding_window");
    if (sw.is_null()) {
      sliding_window = 0;
    } else if (sw.is_number_integer()) {
      sliding_window = sw.get<int64_t>();
    } else {
      // An integral float (4096.0) is the same window to Python's `+ 1`; a
      // fractional or non-numeric value is not a window at all.
      VT_CHECK(sw.is_number_float() && std::floor(sw.get<double>()) == sw.get<double>(),
               "cohere2_moe: sliding_window must be an integer (got " + sw.dump() + ")");
      sliding_window = static_cast<int64_t>(sw.get<double>());
    }
  }

  p.window.assign(static_cast<size_t>(L), std::nullopt);
  p.rope.assign(static_cast<size_t>(L), false);
  p.dense.assign(static_cast<size_t>(L), false);
  bool prefix_dense = true;  // is_prefix_dense_layer (cohere2_moe.py:49-53)
  for (int64_t l = 0; l < L; ++l) {
    const size_t i = static_cast<size_t>(l);
    const std::string& lt = layer_types[i];
    VT_CHECK(lt == "sliding_attention" || lt == "full_attention",
             "cohere2_moe: layer_types[" + std::to_string(l) + "] must be "
             "sliding_attention or full_attention (got '" + lt + "')");
    if (lt == "sliding_attention") {
      VT_CHECK(sliding_window > 0,
               "cohere2_moe: layer " + std::to_string(l) +
                   " is sliding_attention but sliding_window is null or 0; the "
                   "pinned cohere2_moe.py computes sliding_window + 1 and cannot run it");
      p.window[i] = sliding_window + 1;
    }
    p.dense[i] = mlp_types[i] == "dense";
    prefix_dense = prefix_dense && p.dense[i];
    const bool force_rope = prefix_dense && prefix_pattern == 1;
    p.rope[i] = p.window[i].has_value() || force_rope;
  }
  return p;
}

void ParseCohere2MoeForCausalLMConfig(const HfConfig& config) {
  (void)ParseCohere2MoeParams(config);
}

v1::KVCacheConfig MakeCohere2MoeKVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // One FULL-ATTENTION group over every layer; the sliding layers are masked at
  // the attention kernel by their per-layer window. Upstream gives a sliding
  // layer a SlidingWindowSpec (attention.py:615-660 @ a7c23ac96d), which only
  // lets the allocator free blocks behind the window: a memory optimization with
  // no effect on the attended keys. This is the tree's convention for every
  // interleaved-window model (gemma3_registry.cpp, laguna_registry.cpp).
  const Cohere2MoeParams p = ParseCohere2MoeParams(config);
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, static_cast<int>(p.num_kv_heads),
                                              static_cast<int>(p.head_dim),
                                              v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(cohere2_moe, "Cohere2MoeForCausalLM", kCohere2MoeFactory,
                    kCohere2MoeInfo)

}  // namespace vllm
