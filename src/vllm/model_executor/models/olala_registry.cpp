// Olala (`OlalaForCausalLM`) registry TU — NEW-CANDIDATE-PORTS W2b.
// Self-registers "OlalaForCausalLM" via REGISTER_VLLM_MODEL (one in-TU line,
// ZERO shared-array edit), mirroring kolibri1_registry.cpp / glm4_registry.cpp.
//
// W2b owns: the config hook, the hybrid KV-cache spec (7 V layers sliding-window
// FA + 29 M layers MambaSpec recurrent state), the GGUF bf16/F32 weight loader
// (olala_gguf_weights.cpp, W1 census-validated), and the forward boundary that
// refuses BY NAME with the unported ops listed (W2c ports them against the
// olala-transformers oracle). The safetensors (HF) arm refuses by name — its
// name map is W2c work; the GGUF quantized arms refuse at load (same row).
//
// Reference: modeling_olala.py @ f781f124 (OVHaiLLM/Olala-v1.0-mini-7A1B).
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/olala.h"
#include "vllm/model_executor/models/olala_gguf_config.h"
#include "vllm/model_executor/models/olala_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits (shared carrier)
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// Hybrid Mamba3-MIMO + differential-attention MoE: text generation, hybrid
// (29 recurrent + 7 attention layers), inner state (the M layers' SSM state).
inline constexpr ModelInfo kOlalaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = true,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class OlalaLoadedModel final : public LoadedModel {
 public:
  OlalaLoadedModel(const ModelRegistration& registration, OlalaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const OlalaWeights& weights() const { return weights_; }

 private:
  OlalaWeights weights_;
};

void ParseOlalaForCausalLMConfig(const HfConfig& config) {
  const OlalaParams p = ParseOlalaParams(config);
  VT_CHECK(p.hidden > 0 && p.layers > 0 && p.vocab > 0,
           "olala: hidden/layers/vocab must be positive");
  VT_CHECK(p.heads > 0 && p.kv_heads > 0 && p.head_dim > 0,
           "olala: attention head geometry must be positive");
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == p.layers,
           "olala: layer_types length must equal block_count");
  VT_CHECK(p.experts > 0 && p.top_k > 0 && p.top_k <= p.experts,
           "olala: expert counts inconsistent");
  VT_CHECK(p.latent > 0 && p.expert_ff > 0 && p.shared_ff > 0,
           "olala: MoE dims must be positive");
  VT_CHECK(p.ssm_inner > 0 && p.d_state > 0 && p.mimo_dim > 0 &&
               p.headdim > 0 && p.rope_angles > 0,
           "olala: Mamba3-MIMO geometry must be positive");
}

std::unique_ptr<LoadedModel> LoadOlalaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kGguf || source.gguf == nullptr) {
    throw std::runtime_error(
        "olala: the safetensors (HF) arm is not ported yet and refuses BY NAME "
        "— owed to NEW-CANDIDATE-PORTS W2c (the HF name map against "
        "modeling_olala.py). Load a GGUF from OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF "
        "(the bf16 arm works today; the k-quant arms refuse at load until W2c).");
  }
  return std::make_unique<OlalaLoadedModel>(
      registration, LoadOlalaGgufWeights(*source.gguf, config));
}

void PrepareOlalaForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardOlalaForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // Refuse-by-name stub: the runner-routing drift scan SKIPs a VT_CHECK(false)
  // forward (it decodes nothing), so this registers cleanly while W2c ports
  // the ops. The message names each missing op.
  VT_CHECK(false, OlalaForwardRefusalOps());
}

v1::KVCacheConfig MakeOlalaForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks) {
  const OlalaParams p = ParseOlalaParams(config);
  const vt::DType kv_dtype = v1::ResolveKvCacheDType();

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // V layers (differential attention, sliding window slw_wsize): one SWA group.
  std::vector<std::string> v_layers;
  // M layers (Mamba3-MIMO): one MambaSpec recurrent-state group. State shapes
  // per modeling_olala.py:1372-1404 (allocate_inference_cache): ssm_state
  // [nheads, headdim, d_state] f32, k_state [mimo, nheads, d_state], v_state
  // [nheads, headdim], angle state [nheads, num_rope_angles] f32. nheads =
  // 2*hidden/headdim = 48 on the pinned artifact. The exact per-tensor dtype
  // split is pinned by W2c against the reference; W2b publishes the f32 state
  // and the model-dtype k/v state.
  std::vector<std::string> m_layers;
  const int64_t nheads = 2 * p.hidden / p.headdim;
  for (int64_t l = 0; l < p.layers; ++l) {
    const std::string name =
        "model.layers." + std::to_string(l) + "." +
        (p.layer_types[static_cast<size_t>(l)] == 'M' ? "mixer" : "self_attn");
    if (p.layer_types[static_cast<size_t>(l)] == 'M') {
      m_layers.push_back(name);
    } else {
      v_layers.push_back(name);
    }
  }
  if (!v_layers.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(v_layers),
        std::make_shared<v1::SlidingWindowSpec>(
            block_size, static_cast<int>(p.kv_heads), static_cast<int>(p.head_dim),
            kv_dtype, static_cast<int>(p.window)));
  }
  if (!m_layers.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(m_layers),
        std::make_shared<v1::MambaSpec>(
            block_size,
            std::vector<std::vector<int64_t>>{
                {nheads, p.headdim, p.d_state},               // ssm_state, f32
                {p.mimo_dim, nheads, p.d_state},              // k_state
                {nheads, p.headdim},                          // v_state
                {nheads, p.rope_angles}},                     // angle state, f32
            std::vector<vt::DType>{vt::DType::kF32, vt::DType::kBF16,
                                   vt::DType::kBF16, vt::DType::kF32}));
  }
  return kv;
}

const ModelFactory kOlalaFactory{
    .parse_config = &ParseOlalaForCausalLMConfig,
    .load_weights = &LoadOlalaForCausalLM,
    .prepare = &PrepareOlalaForCausalLM,
    .forward = &ForwardOlalaForCausalLM,
    .make_kv_cache = &MakeOlalaForCausalLMKVCache,
    .is_dense_model = false,
    .consumes_multi_kv = true,
};

}  // namespace

REGISTER_VLLM_MODEL(olala, "OlalaForCausalLM", kOlalaFactory, kOlalaInfo)

}  // namespace vllm
