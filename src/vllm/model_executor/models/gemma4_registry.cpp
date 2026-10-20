// Gemma-4 text backbone (`Gemma4ForConditionalGeneration`) registry TU —
// MODEL-GEMMA4 G1. Self-registers "Gemma4ForConditionalGeneration" via
// REGISTER_VLLM_MODEL and owns the arch entry points (config hook, KV-cache spec,
// LoadedModel subclass + factory). Mirrors the gemma3_registry.cpp seam (new TU +
// one in-TU REGISTER line -> ZERO shared-array edit). The engine's HfConfig
// loader descends into `text_config` for the TYPED fields (hf_config.cpp:103-113),
// but keeps `config.raw` as the FULL config.json (hf_config.cpp:414). Gemma-4's
// per-arch scalars (global_head_dim, layer_types, hidden_size_per_layer_input,
// rope_parameters, num_kv_shared_layers) are nested under raw["text_config"] in
// the mm wrapper, so every raw read here / in gemma4.cpp / gemma4_weights.cpp
// goes through a `TextCfg`/text_config view (top-level for a plain config).
//
// G1 SCOPE: the text backbone only. The vision (SigLIP) + audio (USM-Conformer)
// towers are G2/G3 (skipped by the weight loader). See gemma4-multimodal.md.
#include "vllm/model_executor/models/model_registry.h"

#include <set>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vt/fused_ops.h"
#include "vllm/model_executor/models/gemma4_loaded.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace vllm {
namespace {

// registry.py _ModelInfo: text generation via the Gemma-4 language_model stack.
// CLAIM-GEMMA4-MM-E2E: supports_multimodal is now TRUE — the registered forward
// consumes ModelForwardInput.mm (the SigLIP2 vision tower's projected soft tokens,
// masked-scattered into the <image> rows) via the mm branch below, exactly like the
// Qwen3-VL fold. mm is nullopt on every TEXT step ⇒ the text path is byte-identical
// by construction (the SACRED text 32/32 gate is re-proven post-flip).
inline constexpr ModelInfo kGemma4Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// Owned-or-borrowed weights (mirrors Qwen3VLLoadedModel): the mm e2e gate BORROWS
// the loaded weights so the driver and the model share ONE Gemma4Weights (no
// multi-GB copy on the unified-memory box).

std::unique_ptr<LoadedModel> LoadGemma4ForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Gemma4ForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Gemma4LoadedModel>(
      registration,
      source.safetensors_owned
          ? LoadGemma4ForConditionalGenerationWeightsOwned(source.safetensors_owned,
                                                           config)
          : LoadGemma4ForConditionalGenerationWeights(*source.safetensors, config));
}

void PrepareGemma4ForConditionalGeneration(LoadedModel& model,
                                           const HfConfig& config,
                                           vt::Queue& queue) {
  (void)config;
  (void)queue;
  const char* env = std::getenv("VT_GEMMA4_RESIDENT_EXPERTS");
  if (env == nullptr || env[0] != '1') return;
  auto& gemma = ModelAs<Gemma4LoadedModel>(model, "Gemma4ForConditionalGeneration");
  Gemma4Weights& w = const_cast<Gemma4Weights&>(gemma.weights());
  int ngpu = 2;  // Upload clamps to hipGetDeviceCount
  if (const char* g = std::getenv("VT_GEMMA4_RESIDENT_GPUS"))
    ngpu = std::max(1, std::atoi(g));
  UploadGemma4ExpertsResidentForWeights(w, ngpu);
  // Pre-warm ExpertGeGLU scratch on every GPU that holds resident experts (no mid-decode malloc).
  {
    std::set<int> devs;
    int G = 8, I = 0, H = 0;
    for (const auto& layer : w.layers) {
      if (!layer.moe.enabled || !layer.moe.experts.fp8_native_resident) continue;
      devs.insert(layer.moe.experts.dev_id);
      if (I == 0) {
        I = static_cast<int>(layer.moe.experts.intermediate);
        H = static_cast<int>(layer.moe.experts.hidden);
        G = layer.moe.top_k > 0 ? layer.moe.top_k : 8;
      }
    }
    if (I > 0 && H > 0) {
      for (int d : devs) {
        if (d < 0) continue;
        if (!vt::PrewarmExpertGeGLUFp8TopK(d, G, I, H)) {
          std::fprintf(stderr, "gemma4: ExpertGeGLU prewarm failed on gpu %d\n", d);
        } else {
          std::fprintf(stderr, "gemma4: ExpertGeGLU prewarm ok gpu %d G=%d I=%d H=%d\n", d, G, I, H);
        }
      }
    }
  }
}

ForwardLogits ForwardGemma4ForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  const auto& gemma = ModelAs<Gemma4LoadedModel>(model, "Gemma4ForConditionalGeneration");
  const Gemma4Weights& weights = gemma.weights();
  // CLAIM-GEMMA4-MM-E2E: the multimodal branch. When ModelForwardInput.mm is set
  // (the Gemma4GenerateGreedyViaRegistry driver / the runner mm-path) the hidden
  // stream starts from the ALREADY-MERGED inputs_embeds and the PLE lookup uses the
  // mm-masked ids; positions are the 1-D ModelForwardInput::positions (NO MRoPE / NO
  // DeepStack). nullopt on every text step ⇒ the text path below is byte-identical.
  if (input.mm.has_value()) {
    const MultiModalForwardInput& mm = *input.mm;
    // ENG-MM-INPUT-PIPELINE P1: both handles are DEVICE views (#1358, #2300).
    VT_CHECK(mm.inputs_embeds.data != nullptr && mm.ple_token_ids.data != nullptr,
             "Gemma-4 mm forward: null merged-embeds / ple_token_ids device handle "
             "on ModelForwardInput.mm");
    return HostLogits(
        Gemma4Model::ForwardMm(mm.inputs_embeds, mm.ple_token_ids,
                               input.positions, input.attn_meta, input.attn_kv,
                               weights, input.config, input.queue,
                               input.logits_indices),
        input.config.vocab_size);
  }
  if (input.gather_logits) {
    return Gemma4Model::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      Gemma4Model::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGemma4Factory = [] {
  auto factory = kGemma4MmHooks;
  factory.parse_config = &ParseGemma4ForConditionalGenerationConfig;
  factory.load_weights = &LoadGemma4ForConditionalGeneration;
  factory.prepare = &PrepareGemma4ForConditionalGeneration;
  factory.forward = &ForwardGemma4ForConditionalGeneration;
  factory.make_kv_cache = &MakeGemma4ForConditionalGenerationKVCache;
  factory.is_dense_model = true;
  return factory;
}();

}  // namespace

void ParseGemma4ForConditionalGenerationConfig(const HfConfig& config) {
  (void)config;  // Gemma-4 scalars are read from raw["text_config"] by loader/forward.
}

v1::KVCacheConfig MakeGemma4ForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks) {
  // TRUE topology (G1b LANDED): TWO head dims — sliding layers head_dim=256,
  // full_attention layers global_head_dim=512, both num_key_value_heads=2, plus
  // YOCO KV-sharing on the last num_kv_shared_layers. The runner now consumes a
  // PER-LAYER attention spec (`per_layer_attn_specs`, runner.cpp G1b), so each
  // non-GDN layer allocates + views its own head_dim. The single group below
  // still drives the (head_dim-independent) block table / KV manager; its spec
  // carries the LARGER head_dim as an honest per-page upper bound.
  // HfConfig::raw is the FULL config.json (hf_config.cpp:414); Gemma-4's
  // global_head_dim / layer_types are nested under `text_config` in the mm
  // wrapper, so resolve that view (top-level for a plain config).
  const nlohmann::json& raw =
      (config.raw.contains("text_config") && config.raw.at("text_config").is_object())
          ? config.raw.at("text_config")
          : config.raw;
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int num_kv_heads_full = [&]() {
    if (const auto it = raw.find("num_global_key_value_heads");
        it != raw.end() && it->is_number_integer()) {
      return it->get<int>();
    }
    return num_kv_heads;
  }();
  const int head_dim_sliding = static_cast<int>(config.head_dim);  // 256
  int head_dim_full = head_dim_sliding;                            // 512
  if (const auto it = raw.find("global_head_dim");
      it != raw.end() && it->is_number_integer()) {
    head_dim_full = it->get<int>();
  }
  const vt::DType kv_dtype = v1::ResolveKvCacheDType();
  const int64_t L = config.num_hidden_layers;

  std::vector<bool> is_full(static_cast<size_t>(L), false);
  if (const auto it = raw.find("layer_types");
      it != raw.end() && it->is_array()) {
    for (int64_t l = 0; l < L && static_cast<size_t>(l) < it->size(); ++l) {
      is_full[static_cast<size_t>(l)] =
          it->at(static_cast<size_t>(l)).is_string() &&
          it->at(static_cast<size_t>(l)).get<std::string>() == "full_attention";
    }
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, std::max(num_kv_heads, num_kv_heads_full),
          std::max(head_dim_sliding, head_dim_full), kv_dtype));
  kv.per_layer_attn_specs.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    const bool full = is_full[static_cast<size_t>(l)];
    const int hd = full ? head_dim_full : head_dim_sliding;
    const int hkv = full ? num_kv_heads_full : num_kv_heads;
    kv.per_layer_attn_specs.push_back(std::make_shared<v1::FullAttentionSpec>(
        block_size, hkv, hd, kv_dtype));
  }
  return kv;
}

// CLAIM-GEMMA4-MM-E2E: own/borrow adapters (mirror Make/BorrowQwen3VLLoadedModel).
std::unique_ptr<LoadedModel> MakeGemma4LoadedModel(Gemma4Weights weights) {
  return std::make_unique<Gemma4LoadedModel>(
      RegistrationFor("Gemma4ForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowGemma4LoadedModel(const Gemma4Weights& weights) {
  return std::make_unique<Gemma4LoadedModel>(
      RegistrationFor("Gemma4ForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(gemma4, "Gemma4ForConditionalGeneration", kGemma4Factory,
                    kGemma4Info)
// google/gemma-4-12B-it (and other "unified" HF exports) advertise this arch
// name with model_type gemma4_unified. Same text backbone factory; the weight
// loader tolerates no-PLE dense layouts (hidden_size_per_layer_input==0).
REGISTER_VLLM_MODEL(gemma4_unified, "Gemma4UnifiedForConditionalGeneration",
                    kGemma4Factory, kGemma4Info)

}  // namespace vllm
