// The request-level SystemOne scorer shared by the candidate-letter decision
// models (MODEL-NIMBLE, MODEL-TEV1 Phase 6).
//
// A /v1/systemone request names typed questions (choice, noul, score). Each
// question becomes one prompt whose answer is one letter; the model's
// next-token logits at those letters, softmaxed, are the answer. The models
// differ only in how a question is rendered into a prompt and in where the
// logits come from. Everything else lives here, once:
//   compile   the Jev question -> field mapping (bespokelabsai/nimble
//             compiler.py; Ollama decision/systemone.go compileField agrees)
//   letters   the check that every letter is one ordinary token at the answer
//             boundary (parallel_schema.py; Ollama llama_server_score.go)
//   answer    openjev scoring.py / Ollama Answer: softmax, noul, score,
//             legend, entropy confidence
//   request   tokenize every prompt, refuse an over-long one, gather the
//             candidate logits, answer every field
// See .agents/specs/nimble.md and .agents/specs/tev1.md (Phase 6).
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {

namespace tok {
class Tokenizer;
}

namespace decision_scorer {

// openjev defaults.py MAX_QUESTIONS and MAX_ANSWERS; Ollama allows the same
// 1-64 questions.
inline constexpr int kMaxQuestions = 64;
inline constexpr int kMaxAnswers = 64;

// A refusal of the request itself (HTTP 400, VLLM_ERR_INVALID_ARGUMENT).
class RequestError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// One field, compiled from one question.
struct Field {
  std::string name;         // the question id
  std::string type;         // "noul" | "choice" | "score"
  std::string description;  // Serialize(instructions)
  // The schema values in candidate order: false/true for noul, the criteria
  // keys for choice, "0".."L-1" for score.
  std::vector<nlohmann::ordered_json> values;
  // One description per value (the compiler always supplies one).
  std::vector<std::string> value_descriptions;
  // Answer keys, in the same order: "false"/"true", criteria keys, "0"..
  std::vector<std::string> keys;
};

struct Request {
  std::string model;             // empty when the request names none
  nlohmann::ordered_json state;  // the request's state value, unchanged
  std::string context;           // Serialize(state)
  std::vector<Field> fields;
};

// How a model narrows the shared compilation.
struct CompileLimits {
  // The widest field the model's prompt contract serves (one letter each).
  int max_choices = 26;
  // Appended to "<question>: <n> choices; " when a field is wider.
  std::string too_wide_reason;
};

// Python json.dumps(ensure_ascii=False) with the default ", " / ": "
// separators. Throws RequestError on a non-finite number (allow_nan=False).
std::string JsonDumps(const nlohmann::ordered_json& value);

// Python str.strip() == "" over ASCII whitespace only; a string of only
// non-ASCII whitespace (U+00A0) is NOT blank here, where Python would say it is.
bool IsBlank(std::string_view s);

// compiler.py serialize(): a string as-is, anything else JsonDumps.
std::string Serialize(const nlohmann::ordered_json& value);

// Validate and compile a /v1/systemone body. Throws RequestError.
Request CompileRequest(const nlohmann::ordered_json& body,
                       const CompileLimits& limits);

// The answer for one field from its candidate logits: p = softmax(logits / T).
nlohmann::ordered_json AnswerFromLogits(const Field& field,
                                        const std::vector<double>& logits,
                                        double temperature);

// clamp(1 - H(p) / ln(n), 0, 1).
double EntropyConfidence(const std::vector<double>& probabilities);

// The token id of each letter "A".. (count of them) as it follows a prompt
// that ends with `boundary`. Throws std::runtime_error, naming `model`, when a
// letter is not exactly one ordinary token there or two letters share an id.
std::vector<int32_t> CandidateIds(const tok::Tokenizer& tokenizer,
                                  std::string_view boundary, size_t count,
                                  std::string_view model);

// The logits of each field's candidates, from whatever runs the model.
struct CandidateLogits {
  std::vector<std::vector<double>> logits;  // [field][candidate]
  int64_t output_tokens = 0;                // tokens sampled to get them
};
using CandidateLogitsFn = std::function<CandidateLogits(
    const std::vector<std::vector<int32_t>>& prompt_ids,
    const std::vector<std::vector<int32_t>>& candidate_ids)>;

// Scores one compiled request: tokenizes every prompt first and refuses the
// request, naming the longest prompt, when it exceeds max_prompt_tokens (the
// input is never truncated); then asks `logits_fn` for every field's
// candidates and answers each field at `temperature`.
struct ScoredRequest {
  nlohmann::ordered_json answers = nlohmann::ordered_json::object();
  int64_t input_tokens = 0;
  int64_t output_tokens = 0;
};
ScoredRequest ScoreRequest(const tok::Tokenizer& tokenizer,
                           const Request& request,
                           const std::vector<std::string>& prompts,
                           std::string_view boundary, double temperature,
                           int64_t max_prompt_tokens, std::string_view model,
                           const CandidateLogitsFn& logits_fn);

}  // namespace decision_scorer
}  // namespace vllm
