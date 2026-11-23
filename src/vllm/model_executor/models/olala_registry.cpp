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
#include "vllm/model_executor/models/olala_paged.h"
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
  // W2f (ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W): the production decode path.
  // The runner supplies its paged KV + recurrent caches on EVERY engine step,
  // so a non-empty cache pair routes to the paged forward (olala_paged.cpp) —
  // the same selection NemotronH (nemotron_h_registry.cpp:200-202) and
  // Kimi-Linear (kimi_linear_registry.cpp:99-102) make. Direct callers with no
  // runner caches refuse by name: the block generate host (OlalaGenerateHost)
  // stays the test/oracle operand, not an engine seam.
  auto& olala = ModelAs<OlalaLoadedModel>(model, "OlalaForCausalLM");
  if (!input.attn_kv.empty() && !input.gdn_state.empty()) {
    return OlalaPagedForward(olala.weights(), ParseOlalaParams(input.config),
                             input);
  }
  VT_CHECK(false,
           "olala: the host block generate (OlalaGenerateHost) is the "
           "test/oracle operand; engine decode runs through the paged forward "
           "with runner-allocated caches. Direct forward calls without caches "
           "are not a supported path.");
}

v1::KVCacheConfig MakeOlalaForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks) {
  const OlalaParams p = ParseOlalaParams(config);
  // W2f: f32, NOT ResolveKvCacheDType(). The W2d bf16 token gate is EXACT
  // against llama-cpp-olala, whose cache is f32; the V-layer shift recurrence
  // (each token's k_prev is the previous token's RAW pre-shift k/v) and the
  // Mamba3 state would both bf16-round in a model-dtype cache and buy a
  // plausible near-tie drift into the production path. The page is small
  // (7 V layers × window × 2·KV·D). This is the annotated f32 exception
  // AGENTS.md asks for on a model-path buffer.
  const vt::DType kv_dtype = vt::DType::kF32;

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // V layers (differential attention, sliding window slw_wsize): one SWA group.
  std::vector<std::string> v_layers;
  // M layers (Mamba3-MIMO): one MambaSpec recurrent-state group. State shapes
  // per modeling_olala.py:1372-1404 (allocate_inference_cache): ssm_state
  // [nheads, headdim, d_state], k_state [mimo, nheads, d_state], v_state
  // [nheads, headdim], angle state [nheads, num_rope_angles]. All four rows
  // are f32 (see the kv_dtype note above): the paged forward scatters the
  // step host's f32 state back verbatim, so the engine path is bit-identical
  // to the W2d EXACT host gate.
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
    // W2f: TWO head planes (2·kv_heads). Heads [0, KV) hold the post-shift
    // post-norm k/v the attention reads; heads [KV, 2·KV) hold that token's
    // RAW pre-shift k/v — the shift recurrence's k_prev/v_prev
    // (modeling_olala.py:619). Storing the raw rows per token keeps the whole
    // V-layer recurrence in the paged cache, so the forward carries no
    // private per-request state. olala_paged.cpp is the only consumer.
    kv.kv_cache_groups.emplace_back(
        std::move(v_layers),
        std::make_shared<v1::SlidingWindowSpec>(
            block_size, static_cast<int>(2 * p.kv_heads),
            static_cast<int>(p.head_dim), kv_dtype,
            static_cast<int>(p.window)));
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
            std::vector<vt::DType>{vt::DType::kF32, vt::DType::kF32,
                                   vt::DType::kF32, vt::DType::kF32}));
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
