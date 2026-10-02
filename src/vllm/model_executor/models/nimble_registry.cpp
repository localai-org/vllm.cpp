// Nimble decision model registry (MODEL-NIMBLE).
//
// Self-registers "NimbleModel". The checkpoint is Qwen3.5-9B with the
// Bespoke-Nimble LoRA merged at CONVERT time by scripts/convert-nimble.py, so
// this loader reads one set of Qwen3.5 dense shards, like kev. NimbleDecide
// runs one ForwardDenseLastLogits per schema field and reads the candidate
// letter logits (inference.py candidate_logits, logits_to_keep=1).
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/nimble_inference.h"
#include "vllm/model_executor/models/qwen3_5.h"          // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"   // ParseQwen3_5Config, MakeQwen3_5KVCache, HostLogits
#include "vllm/model_executor/models/qwen3_5_dense.h"    // LoadQwen3_5Dense, ForwardDenseLastLogits
#include "vllm/tokenizer/tokenizer.h"

namespace vllm {
namespace {

// A decision model: /v1/systemone and vllm_decide, no generation route. The
// backbone is the Qwen3.5 GDN hybrid.
inline constexpr ModelInfo kNimbleInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

class NimbleLoadedModel final : public LoadedModel {
 public:
  NimbleLoadedModel(const ModelRegistration& registration,
                    Qwen3_5DenseWeights weights, HfConfig config,
                    double temperature, int64_t max_length)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        config_(std::move(config)),
        temperature_(temperature),
        max_length_(max_length) {}

  const Qwen3_5DenseWeights& weights() const { return weights_; }
  const HfConfig& config() const { return config_; }
  double temperature() const { return temperature_; }
  int64_t max_length() const { return max_length_; }
  void set_queue(vt::Queue q) { queue_ = q; }
  vt::Queue queue() const { return queue_; }

 private:
  Qwen3_5DenseWeights weights_;
  HfConfig config_;
  double temperature_;
  int64_t max_length_;
  vt::Queue queue_;
};

std::unique_ptr<LoadedModel> LoadNimble(const ModelRegistration& registration,
                                        const HfConfig& config,
                                        const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture NimbleModel loads merged safetensors only; GGUF "
        "k-quant arms are owed (.agents/specs/nimble.md)");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  // The converter records the checkpoint's own temperature. There is no safe
  // default: Bespoke-Nimble-9B uses 1.0 and Bespoke-Nimble-9B-v2 uses 2.179.
  const auto t = config.raw.find("nimble_temperature");
  if (t == config.raw.end() || !t->is_number() || !(t->get<double>() > 0.0) ||
      !std::isfinite(t->get<double>())) {
    throw std::runtime_error(
        "NimbleModel: config.json has no positive nimble_temperature. Convert "
        "the adapter with scripts/convert-nimble.py, which copies it from the "
        "adapter's temperature_config.json");
  }
  int64_t max_length = nimble::kDefaultMaxLength;
  const auto ml = config.raw.find("nimble_max_length");
  if (ml != config.raw.end() && ml->is_number_integer()) {
    max_length = ml->get<int64_t>();
  }
  return std::make_unique<NimbleLoadedModel>(
      registration,
      LoadQwen3_5Dense(*source.safetensors, config, source.load_queue), config,
      t->get<double>(), max_length);
}

void PrepareNimble(LoadedModel& model, const HfConfig& config, vt::Queue& queue) {
  (void)config;
  ModelAs<NimbleLoadedModel>(model, "NimbleModel").set_queue(queue);
}

// The registry forward is the plain dense causal-LM forward, so a runner that
// reaches this model through ModelRegistry::Forward gets ordinary logits. The
// decision readout is NimbleDecide.
ForwardLogits ForwardNimble(LoadedModel& model, const ModelForwardInput& input) {
  auto& m = ModelAs<NimbleLoadedModel>(model, "NimbleModel");
  return HostLogits(
      Qwen3_5DenseModel::Forward(input.token_ids, input.positions,
                                 input.attn_meta, input.gdn_meta, input.attn_kv,
                                 input.gdn_state, m.weights(), input.config,
                                 input.queue, input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kNimbleFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadNimble,
    .prepare = &PrepareNimble,
    .forward = &ForwardNimble,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
};

}  // namespace

NimbleResponse NimbleDecide(const LoadedModel& model,
                            const tok::Tokenizer& tokenizer,
                            const nlohmann::ordered_json& body) {
  const auto& m = ModelAs<NimbleLoadedModel>(model, "NimbleModel");
  const nimble::Request request = nimble::CompileRequest(body);
  const std::vector<std::string> prompts = nimble::BuildPrompts(request);
  // One ForwardDenseLastLogits per field (inference.py candidate_logits,
  // logits_to_keep=1). Nothing is sampled, so output_tokens stays 0.
  vt::Queue queue = m.queue();
  const decision_scorer::CandidateLogitsFn last_row =
      [&m, &queue](const std::vector<std::vector<int32_t>>& prompt_ids,
                   const std::vector<std::vector<int32_t>>& candidate_ids) {
        decision_scorer::CandidateLogits out;
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
          std::vector<int32_t> positions(prompt_ids[i].size());
          std::iota(positions.begin(), positions.end(), 0);
          const std::vector<float> logits =
              Qwen3_5DenseModel::ForwardDenseLastLogits(
                  prompt_ids[i], positions, m.weights(), m.config(), queue);
          std::vector<double> candidate;
          candidate.reserve(candidate_ids[i].size());
          for (const int32_t id : candidate_ids[i]) {
            candidate.push_back(static_cast<double>(logits[static_cast<size_t>(id)]));
          }
          out.logits.push_back(std::move(candidate));
        }
        return out;
      };
  return decision_scorer::ScoreRequest(tokenizer, request, prompts, "</think>\n\n",
                                       m.temperature(), m.max_length(), "nimble",
                                       last_row);
}

std::unique_ptr<LoadedModel> MakeNimbleLoadedModel(Qwen3_5DenseWeights weights,
                                                   HfConfig config,
                                                   double temperature,
                                                   int64_t max_length,
                                                   vt::Queue queue) {
  auto model = std::make_unique<NimbleLoadedModel>(
      RegistrationFor("NimbleModel"), std::move(weights), std::move(config),
      temperature, max_length);
  model->set_queue(queue);
  return model;
}

REGISTER_VLLM_MODEL(nimble_model, "NimbleModel", kNimbleFactory, kNimbleInfo)

}  // namespace vllm
