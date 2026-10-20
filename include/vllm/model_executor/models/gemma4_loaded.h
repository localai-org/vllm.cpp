#pragma once
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/model_registry.h"
// Defined once in qwen3_5_common.h; a reference needs no definition here.
namespace vllm { struct BorrowedWeightsTag; }
namespace vllm {
class Gemma4LoadedModel final : public LoadedModel {
 public:
  Gemma4LoadedModel(const ModelRegistration& registration, Gemma4Weights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Gemma4LoadedModel(const ModelRegistration& registration,
                    const Gemma4Weights& weights, const BorrowedWeightsTag&)
      : LoadedModel(registration), weights_(&weights) {}
  const Gemma4Weights& weights() const { return *weights_; }

 private:
  std::optional<Gemma4Weights> owned_weights_;
  const Gemma4Weights* weights_ = nullptr;
};

// Production hooks; CPU reference dispatch is not runner readiness.
extern const ModelFactory kGemma4MmHooks;
MmForwardBuffers EmbedMmGemma4(LoadedModel&, const HfConfig&, vt::Queue&,
                              const MmEmbedInputs&);

MmEncoderOutput EncodeMmGemma4Audio(LoadedModel&, const HfConfig&, vt::Queue&,
    const multimodal::MultiModalFeatureSpec&);
}
