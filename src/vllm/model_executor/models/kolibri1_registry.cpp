// Kolibri1 (`Kolibri1ForCausalLM`) registry TU — MODEL-TEXT-kolibri-1 W1,
// spec .agents/specs/kolibri-1-cpu.md. A NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array, mirroring the
// mimo_v2_registry.cpp seam.
//
// W1 owns: the config hook (config-descent validation), the KV-cache spec
// (two groups: full-attention + sliding-window, uniform head_dim 128, window
// 513), and the 32-shard FP8-block weight loader (kolibri1_weights.cpp).
// GGUF is refused BY NAME per the row's scope, and the forward pass refuses
// with a message naming the missing part — the CPU forward is a later wave
// of the same row.

#include "vllm/model_executor/models/kolibri1.h"
#include "vllm/model_executor/models/kolibri1_dequant_cache.h"  // BumpModelGeneration
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/model_executor/models/kolibri1_tt_forward.h"  // the B2b-i device arm
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, ModelForwardInput
#include "vllm/v1/kv_cache_dtype.h"  // ResolveKvCacheDType
#include "vllm/v1/kv_cache_interface.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm {

namespace {

int64_t GetInt(const nlohmann::json& j, const char* key, int64_t def = 0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number_integer())
    return it->get<int64_t>();
  if (const auto it = j.find(key); it != j.end() && it->is_number_float())
    return static_cast<int64_t>(it->get<double>());
  return def;
}

double GetDouble(const nlohmann::json& j, const char* key, double def = 0.0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number())
    return it->get<double>();
  return def;
}

bool GetBool(const nlohmann::json& j, const char* key, bool def = false) {
  if (const auto it = j.find(key); it != j.end() && it->is_boolean())
    return it->get<bool>();
  return def;
}

std::string GetString(const nlohmann::json& j, const char* key,
                      const char* def = "") {
  if (const auto it = j.find(key); it != j.end() && it->is_string())
    return it->get<std::string>();
  return def;
}

}  // namespace

Kolibri1Params ParseKolibri1Params(const HfConfig& config) {
  Kolibri1Params p;
  const auto& raw = config.raw;

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.num_attention_heads = config.num_attention_heads;
  p.num_key_value_heads = config.num_key_value_heads;

  // Attention geometry
  p.head_dim = GetInt(raw, "head_dim", 0);
  p.sliding_window = GetInt(raw, "sliding_window", 0);
  p.use_sliding_window = GetBool(raw, "use_sliding_window", false);

  // Hybrid layer kinds: "sliding_attention" / "full_attention" per layer.
  // RNoPE (kolibri1.py:81-95): RoPE lives ONLY on the sliding layers. Any
  // other string is refused rather than mapped to a guess.
  if (const auto it = raw.find("layer_types"); it != raw.end() && it->is_array()) {
    for (const auto& el : *it) {
      VT_CHECK(el.is_string(),
               "kolibri1: layer_types entries must be strings");
      const std::string kind = el.get<std::string>();
      VT_CHECK(kind == "sliding_attention" || kind == "full_attention",
               "kolibri1: layer_types entry '" + kind +
                   "' is neither sliding_attention nor full_attention");
      p.layer_types.push_back(kind);
    }
  }

  // RoPE (R2): the checkpoint ships a flat rope_theta; the plugin reads the
  // same value through rope_parameters["rope_theta"] on the transformers
  // Qwen3MoeConfig base. Accept both spellings, flat first.
  p.rope_theta = GetDouble(raw, "rope_theta", 0.0);
  if (p.rope_theta == 0.0) {
    if (const auto it = raw.find("rope_parameters"); it != raw.end() &&
                                                      it->is_object()) {
      p.rope_theta = GetDouble(*it, "rope_theta", 0.0);
    }
  }

  // Rotary extent: FULL rotary over head_dim (rotary_dim == head_dim == 128;
  // there is no partial rotary in this architecture). A partial_rotary_factor
  // that narrows it is refused rather than silently applied.
  const double prf = GetDouble(raw, "partial_rotary_factor", 1.0);
  VT_CHECK(prf == 1.0,
           "kolibri1: partial_rotary_factor " + std::to_string(prf) +
               " is not supported — the architecture is full rotary over "
               "head_dim");
  p.rotary_dim = p.head_dim;

  // MoE
  p.num_experts = GetInt(raw, "num_experts", 0);
  p.num_experts_per_tok = GetInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = GetInt(raw, "moe_intermediate_size", 0);
  p.shared_expert_intermediate_size =
      GetInt(raw, "shared_expert_intermediate_size", 0);
  p.norm_topk_prob = GetBool(raw, "norm_topk_prob", false);

  // Norm / activation
  p.rms_norm_eps = GetDouble(raw, "rms_norm_eps", 1e-6);
  p.hidden_act = GetString(raw, "hidden_act", "silu");
  p.head_dtype = GetString(raw, "head_dtype", "");
  p.tie_word_embeddings = GetBool(raw, "tie_word_embeddings", false);

  // ---- Validation: everything unrepresentable throws ----
  VT_CHECK(p.hidden_size > 0, "kolibri1: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "kolibri1: num_hidden_layers must be positive");
  VT_CHECK(p.vocab_size > 0, "kolibri1: vocab_size must be positive");
  VT_CHECK(p.num_attention_heads > 0,
           "kolibri1: num_attention_heads must be positive");
  VT_CHECK(p.num_key_value_heads > 0,
           "kolibri1: num_key_value_heads must be positive");
  VT_CHECK(p.num_attention_heads % p.num_key_value_heads == 0,
           "kolibri1: num_attention_heads must be a multiple of "
           "num_key_value_heads (GQA)");
  VT_CHECK(p.head_dim > 0, "kolibri1: head_dim must be positive");
  VT_CHECK(!p.layer_types.empty(),
           "kolibri1: layer_types must be present — the hybrid RNoPE pattern "
           "cannot be derived");
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == p.num_hidden_layers,
           "kolibri1: layer_types length must equal num_hidden_layers");
  if (p.use_sliding_window) {
    VT_CHECK(p.sliding_window > 0,
             "kolibri1: use_sliding_window requires a positive sliding_window");
  }
  VT_CHECK(!p.IsSlidingLayer(0) || p.sliding_window > 0,
           "kolibri1: sliding layers require a positive sliding_window");
  VT_CHECK(p.rope_theta > 0.0, "kolibri1: rope_theta must be positive");
  VT_CHECK(p.rotary_dim == p.head_dim,
           "kolibri1: rotary_dim must equal head_dim (full rotary)");
  VT_CHECK(p.num_experts > 0, "kolibri1: num_experts must be positive");
  VT_CHECK(p.num_experts_per_tok > 0 &&
               p.num_experts_per_tok <= p.num_experts,
           "kolibri1: num_experts_per_tok must be in (0, num_experts]");
  VT_CHECK(p.moe_intermediate_size > 0,
           "kolibri1: moe_intermediate_size must be positive");
  VT_CHECK(p.shared_expert_intermediate_size > 0,
           "kolibri1: shared_expert_intermediate_size must be positive");
  // The sigmoid-logit-add router (kolibri1.py:126-142) renormalizes nothing:
  // norm_topk_prob=true would need a different routing path that does not
  // exist here.
  VT_CHECK(!p.norm_topk_prob,
           "kolibri1: norm_topk_prob=true is not implemented — the "
           "sigmoid-logit-add router runs with no weight renormalisation");
  VT_CHECK(p.rms_norm_eps > 0.0, "kolibri1: rms_norm_eps must be positive");
  VT_CHECK(p.hidden_act == "silu",
           "kolibri1: hidden_act '" + p.hidden_act +
               "' is not the silu the architecture defines");
  VT_CHECK(!p.tie_word_embeddings,
           "kolibri1: tied word embeddings are not implemented — the "
           "checkpoint ships an untied lm_head");
  VT_CHECK(p.head_dtype.empty() || p.head_dtype == "float32",
           "kolibri1: head_dtype '" + p.head_dtype +
               "' is not a known value (the checkpoint carries float32)");

  return p;
}

void ParseKolibri1Config(const HfConfig& config) {
  // The resolve IS the validation: it throws on everything unrepresentable.
  (void)ParseKolibri1Params(config);
}

// ---- KV-cache spec ----
v1::KVCacheConfig MakeKolibri1KVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  const Kolibri1Params p = ParseKolibri1Params(config);
  const vt::DType dtype = v1::ResolveKvCacheDType();

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // Two KV-cache groups: full-attention (RNoPE — no rope, full span) and
  // sliding-window (window 513). Uniform head_dim 128 and 4 KV heads across
  // both groups.
  std::vector<std::string> full_layer_names;
  std::vector<std::string> swa_layer_names;
  for (int64_t i = 0; i < static_cast<int64_t>(p.layer_types.size()); ++i) {
    const std::string name =
        "model.layers." + std::to_string(i) + ".self_attn";
    if (p.IsSlidingLayer(i)) {
      swa_layer_names.push_back(name);
    } else {
      full_layer_names.push_back(name);
    }
  }

  if (!full_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(full_layer_names),
        std::make_shared<v1::FullAttentionSpec>(
            block_size, static_cast<int>(p.num_key_value_heads),
            static_cast<int>(p.head_dim), dtype));
  }
  if (!swa_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(swa_layer_names),
        std::make_shared<v1::SlidingWindowSpec>(
            block_size, static_cast<int>(p.num_key_value_heads),
            static_cast<int>(p.head_dim), dtype,
            static_cast<int>(p.sliding_window)));
  }
  return kv;
}

// ---- LoadedModel subclass ----

class Kolibri1LoadedModel final : public LoadedModel {
 public:
  Kolibri1LoadedModel(const ModelRegistration& registration,
                      Kolibri1Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {
    // Scope the process-wide dequant cache to THIS model incarnation
    // (maint-bot P1a on PR #3414, ISSUE-LOCAL-01M4CVDDHAFD7R1QCK9F493SWZ):
    // the cache key carries the model generation, so a later load that
    // reuses this model's freed weight-buffer addresses can never be served
    // this model's cached decodes.
    kolibri1_dequant_cache::BumpModelGeneration();
  }
  const Kolibri1Weights& weights() const { return weights_; }

  // The B2b-i device-resident compute context (the memoized bf16 dequants
  // of the resident fp8-block projections), built ONCE on the model's first
  // Tenstorrent use — eagerly by prepare, lazily by the forward otherwise.
  // The CPU arm never builds one.
  Kolibri1TTResidentDeviceContext& tt_context(vt::Queue& queue) {
    if (tt_ctx_ == nullptr) {
      vt::Backend& be = vt::GetBackend(queue.device.type);
      tt_ctx_ = BuildKolibri1TTResidentDeviceContext(be, queue, weights_);
    }
    return *tt_ctx_;
  }

 private:
  Kolibri1Weights weights_;
  std::unique_ptr<Kolibri1TTResidentDeviceContext> tt_ctx_;
};

// ---- Load / Prepare / Forward ----

std::unique_ptr<LoadedModel> LoadKolibri1ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    throw std::runtime_error(
        "Kolibri1ForCausalLM: GGUF is not supported for this architecture. "
        "Row MODEL-TEXT-kolibri-1, spec .agents/specs/kolibri-1-cpu.md; the "
        "checkpoint is the FP8 block-quantized safetensors index and the "
        "GGUF arm is a separate owed row.");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Kolibri1LoadedModel>(
      registration, LoadKolibri1Weights(*source.safetensors, config));
}

void PrepareKolibri1ForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  // The CPU arm needs no device materialization (the forward wave's
  // residency seam uploads lazily). A Tenstorrent queue materializes the
  // B2b-i device context eagerly at prepare — the resident slice's fp8
  // projections dequanted once into device-resident bf16 buffers (spec
  // .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i).
  if (queue.device.type != vt::DeviceType::kTENSTORRENT) return;
  (void)config;
  auto& m = ModelAs<Kolibri1LoadedModel>(model, "Kolibri1ForCausalLM");
  (void)m.tt_context(queue);
}

ForwardLogits ForwardKolibri1ForCausalLM(LoadedModel& model,
                                         const ModelForwardInput& input) {
  auto& m = ModelAs<Kolibri1LoadedModel>(model, "Kolibri1ForCausalLM");
  // The device dispatch: the CPU row owns the CPU arm (unchanged); the
  // Tenstorrent B2b-i dense-resident slice owns the TT arm; every other
  // device is refused by name here, before any model code runs.
  switch (input.queue.device.type) {
    case vt::DeviceType::kCPU:
      // W2: the CPU hybrid forward (RNoPE, qk-norm, sandwich norms,
      // sigmoid-logit-add MoE) lives in kolibri1_forward.cpp.
      return ForwardKolibri1Forward(input.token_ids, input.positions,
                                    input.attn_meta, input.attn_kv, m.weights(),
                                    input.multi_kv, input.queue,
                                    input.logits_indices);
    case vt::DeviceType::kTENSTORRENT:
      // B2b-i: the dense-resident device forward (spec addendum, slice i).
      return ForwardKolibri1TTResidentForward(
          input.token_ids, input.positions, input.attn_meta, input.attn_kv,
          m.weights(), input.multi_kv, input.queue, input.logits_indices,
          m.tt_context(input.queue));
    default:
      throw std::runtime_error(
          std::string("Kolibri1ForCausalLM: the ") +
          vt::DeviceTypeName(input.queue.device.type) +
          " forward arm is not implemented. The landed arms are the CPU row "
          "(MODEL-TEXT-kolibri-1, spec .agents/specs/kolibri-1-cpu.md) and "
          "the Tenstorrent B2b-i dense-resident slice "
          "(MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md "
          "### B2 scope — B2b addendum, slice i); the " +
          vt::DeviceTypeName(input.queue.device.type) +
          " arm is a separate owed row.");
  }
}

// ---- ModelInfo / ModelFactory ----
inline constexpr ModelInfo kKolibri1Info{
    .is_text_generation_model = true,
    .is_hybrid = true,
    .supports_multimodal = false,
};

inline constexpr ModelFactory kKolibri1Factory{
    .parse_config = &ParseKolibri1Config,
    .load_weights = &LoadKolibri1ForCausalLM,
    .prepare = &PrepareKolibri1ForCausalLM,
    .forward = &ForwardKolibri1ForCausalLM,
    .make_kv_cache = &MakeKolibri1KVCache,
    .is_dense_model = false,
    .consumes_multi_kv = true,
};

REGISTER_VLLM_MODEL(kolibri1, "Kolibri1ForCausalLM", kKolibri1Factory,
                    kKolibri1Info)

// The checked accessor over the registry's loaded model (the B2b device
// waves read the weights back out of the LoadedModel they are handed).
const Kolibri1Weights& Kolibri1LoadedModelWeights(LoadedModel& model) {
  return ModelAs<Kolibri1LoadedModel>(model, "Kolibri1ForCausalLM").weights();
}

// The checked accessor over the registry's loaded model's B2b-i device
// context (built by prepare on a Tenstorrent queue, lazily otherwise).
Kolibri1TTResidentDeviceContext& Kolibri1LoadedModelTTContext(
    LoadedModel& model, vt::Queue& queue) {
  return ModelAs<Kolibri1LoadedModel>(model, "Kolibri1ForCausalLM")
      .tt_context(queue);
}

}  // namespace vllm
