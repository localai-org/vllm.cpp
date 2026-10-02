// `Tev1Model` registry TU — thin alias over the Qwen3.5 dense factory.
//
// Tev1 (`togethercomputer/Tev1-4B-experimental`) is an SFT of Qwen3.5-4B-Base
// via LoRA (rank=8, alpha=16), full merged weights. It is an autoregressive
// text-generation model that produces a single option letter via
// /v1/chat/completions (temperature=0, max_tokens=8, enable_thinking=false).
//
// The registration is a thin alias: the Qwen3.5 dense text-only factory with a
// Tev1-specific ModelInfo. Naming it in config.json also opts the engine in to
// /v1/systemone and vllm_decide (MODEL-TEV1 Phase 6, tev1_inference.h), which
// score through this same generation engine.
//
// We cannot extern kQwen3_5DenseFactory directly because it lives in an
// anonymous namespace in qwen3_5_dense.cpp. Instead, at first-use time we look
// up the already-registered "Qwen3_5ForCausalLM" entry and borrow its factory
// pointer. The pointer is stable for the process lifetime (the factory has
// static storage in qwen3_5_dense.cpp's anonymous namespace).
#include "vllm/model_executor/models/model_registry.h"

#include <string>
#include <string_view>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

// Tev1 is text-generation (autoregressive). has_inner_state = true per spec
// (GDN recurrent state in the Qwen3.5 backbone).
inline constexpr ModelInfo kTev1Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = true,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Resolve the Qwen3.5 dense factory lazily. kQwen3_5DenseFactory is in an
// anonymous namespace inside qwen3_5_dense.cpp, so we cannot reference it by
// name. Instead we read it from the already-registered
// "Qwen3_5ForCausalLM" ModelRegistration. The function-local static ensures
// this resolves on first call (after qwen3_5_dense.cpp's static init has run),
// not at cross-TU static-init time where ordering is unspecified.
const ModelFactory& Tev1Factory() {
  const ModelFactory& factory = *RegistrationFor("Qwen3_5ForCausalLM").factory;
  return factory;
}

}  // namespace vllm

REGISTER_VLLM_MODEL(tev1_model, "Tev1Model", vllm::Tev1Factory(),
                    vllm::kTev1Info)
