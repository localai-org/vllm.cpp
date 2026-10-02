// Nimble decision model (MODEL-NIMBLE): the ONE library seam behind both
// POST /v1/systemone and vllm_decide for a "NimbleModel" engine.
//
// Nimble (bespokelabs/Bespoke-Nimble-9B) is a LoRA on the Qwen3.5-9B dense
// backbone, merged at convert time by scripts/convert-nimble.py. It answers
// each schema field with one forward: the last-position logits of the
// candidate letter tokens, then softmax(logits / T). Nothing is sampled.
//
// The seam is request-level, not per-question, because every field's prompt
// embeds the WHOLE schema (parallel_schema.py prepare_prompts). References:
//   prompt   bespokelabs/Bespoke-Nimble-9B @ bd792f44 parallel_schema.py
//   mapping  bespokelabsai/nimble @ 62076b4f nimble/serving/compiler.py
//   answer   ekzhang/openjev-sglang @ 7f84bedc src/openjev/scoring.py
// See .agents/specs/nimble.md.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/decision_scorer.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

class LoadedModel;
namespace tok {
class Tokenizer;
}

namespace nimble {

// parallel_schema.py serves at most 26 one-letter codes. Wider fields switch
// to extended_schema.py, which this port does not implement yet.
inline constexpr int kMaxChoices = 26;
// openjev defaults.py MAX_QUESTIONS and MAX_ANSWERS.
inline constexpr int kMaxQuestions = decision_scorer::kMaxQuestions;
inline constexpr int kMaxAnswers = decision_scorer::kMaxAnswers;
// schema_config.json max_length; the converter records the checkpoint's own.
inline constexpr int64_t kDefaultMaxLength = 8192;

// parallel_schema.py SYSTEM_PROMPT, verbatim.
extern const char* const kSystemPrompt;

// The compilation, the answer and the refusal type are the shared scorer's
// (decision_scorer.h); Nimble names them here so its callers read unchanged.
using decision_scorer::AnswerFromLogits;
using decision_scorer::EntropyConfidence;
using decision_scorer::Field;
using decision_scorer::Request;
using decision_scorer::RequestError;
using decision_scorer::Serialize;

// Python json.dumps(ensure_ascii=False) with the default ", " / ": "
// separators, then "<" and ">" replaced by \u003c / \u003e (safe_json).
std::string SafeJson(const nlohmann::ordered_json& value);

// Validate and compile a /v1/systemone body with Nimble's 26-choice contract.
// Throws RequestError.
Request CompileRequest(const nlohmann::ordered_json& body);

// The rendered chat prompt for every field, in field order.
std::vector<std::string> BuildPrompts(const Request& request);

}  // namespace nimble

// The answers and token accounting for one request. output_tokens is always 0
// for Nimble: nothing is sampled.
using NimbleResponse = decision_scorer::ScoredRequest;

// Run a whole /v1/systemone request on a NimbleModel engine. Throws
// nimble::RequestError for a request the reference refuses, and
// std::runtime_error for an engine fault.
NimbleResponse NimbleDecide(const LoadedModel& model,
                            const tok::Tokenizer& tokenizer,
                            const nlohmann::ordered_json& body);

// Wrap already-built weights as a NimbleModel (synthetic tests), the same
// shape as MakeQwen3_5DenseLoadedModel. `queue` is the device the forward runs
// on; a loaded engine gets it from the factory's prepare callback instead.
std::unique_ptr<LoadedModel> MakeNimbleLoadedModel(Qwen3_5DenseWeights weights,
                                                   HfConfig config,
                                                   double temperature,
                                                   int64_t max_length,
                                                   vt::Queue queue);

}  // namespace vllm
