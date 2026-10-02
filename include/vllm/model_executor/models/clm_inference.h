// CLM decision entry point (MODEL-CLM, .agents/specs/clm.md).
//
// ClmDecide answers a whole /v1/systemone request: it is the reference's
// server.py systemone + Engine.answer, with the Qwen3 backbone's last-token
// hidden state as the encoder. The HTTP server registers it through
// ApiServer::set_systemone_request and vllm_decide calls it, so the two
// surfaces cannot drift. It is request-level, like NimbleDecide, because the
// reference renders the raw state and criteria JSON itself (to_text), which
// the shared per-question DecisionFn strings cannot carry.
#pragma once

#include <cstdint>
#include <memory>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/clm.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

class LoadedModel;

namespace tok {
class Tokenizer;
}  // namespace tok

// The reference embedder truncates each text to its first 2048 tokens
// (embedder.py max_tokens=2048 as truncate_prompt_tokens; the Qwen3 tokenizer
// truncates on the right).
inline constexpr int64_t kClmMaxTokens = 2048;

// Returns {"answers": {...}, "usage": {"billing_units", "input_tokens",
// "output_tokens"}}. Throws clm::RequestError for a request the reference
// refuses and std::runtime_error for an engine fault.
nlohmann::ordered_json ClmDecide(const LoadedModel& model,
                                 const tok::Tokenizer& tokenizer,
                                 const nlohmann::ordered_json& body);

// The last-token post-norm hidden state of `token_ids` through a ClmModel's
// backbone, one single-sequence prefill. Exposed for tests.
std::vector<float> ClmLastHidden(const LoadedModel& model,
                                 const std::vector<int32_t>& token_ids);

// Wrap already-built weights as a ClmModel (synthetic tests), the same shape
// as MakeNimbleLoadedModel.
std::unique_ptr<LoadedModel> MakeClmLoadedModel(Qwen3DenseWeights weights,
                                                clm::HeadWeights heads,
                                                clm::HeadParams params,
                                                HfConfig config, vt::Queue queue);

}  // namespace vllm
