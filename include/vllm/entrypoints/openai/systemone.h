// SystemOne API helpers shared by the HTTP server (api_server.cpp) and the
// C ABI (vllm_c.cpp). Extracted from the former anonymous namespace so
// vllm_decide can parse and format the same request/response
// shapes without going through the ApiServer callback layer.
//
// Ported from kev/api.py and rl_agent_api.py. The helpers live in a
// sub-namespace so they do not collide with ApiServer members.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SYSTEMONE_H_
#define VLLM_ENTRYPOINTS_OPENAI_SYSTEMONE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai::systemone {

// ── Result types ───────────────────────────────────────────────────────────
// Mirrors the former ApiServer nested structs. ApiServer now aliases these so
// server_main.cpp and tests keep compiling unchanged.

struct NerEntity {
  std::string label;
  std::string text;
  int64_t start = 0;
  int64_t end = 0;
  float confidence = 0.0F;
};

struct NerResult {
  std::vector<NerEntity> entities;
  int64_t prompt_tokens = 0;
};

struct ScoreResult {
  std::vector<float> probabilities;
  int64_t winner = 0;
  float confidence = 0.0F;
  int64_t prompt_tokens = 0;
};

struct DecisionResult {
  std::vector<float> scores;
  std::vector<float> act_logits;
  int64_t prompt_tokens = 0;
};

// ── Math helpers ───────────────────────────────────────────────────────────

// r2(x) — round to 2 decimal places (kev/api.py:r2).
double R2(double x);

// r4(x) — round to 4 decimal places. Used by the cua-s1 score path
// (rl_agent_api.py:round(float(v), 4)).
double R4(double x);

// choice_confidence(p) — normalized margin (kev/api.py:choice_confidence).
double ChoiceConfidence(const std::vector<float>& p);

// score_confidence(p) — 1 - E|level - mode| / (L - 1)
// (kev/api.py:score_confidence).
double ScoreConfidence(const std::vector<float>& p);

// Numerically stable softmax.
std::vector<float> Softmax(const std::vector<float>& scores);

// ── JSON rendering ─────────────────────────────────────────────────────────

// render(v, indent) — flatten JSON content into text. Ported from
// kev/api.py:render. Field names are kept as labels.
std::string RenderJson(const nlohmann::ordered_json& v, int indent = 0);

// option_text(name, desc) — kev/api.py:option_text. "name" if desc is
// null/empty, else "name: rendered_desc".
std::string OptionText(const std::string& name,
                        const nlohmann::ordered_json& desc);

// GetInstructions(qj) — kev uses "instructions"; accept "instr" as alias.
std::string GetInstructions(const nlohmann::ordered_json& qj);

// ── Request parsing ────────────────────────────────────────────────────────

// Parsed question from a SystemOneRequest.
struct SystemOneQuestion {
  std::string id;
  std::string type;  // "noul", "choice", "score"
  std::string instructions;
  std::vector<std::string> keys;   // option names (choice) / level text (score)
  std::vector<std::string> labels;  // NER labels / option text (decision)
};

// Parsed SystemOneRequest body.
struct ParsedSystemOne {
  std::string text;
  std::string model;
  float threshold = 0.5F;
  int64_t max_width = 12;
  std::vector<SystemOneQuestion> questions;
  std::vector<std::string> all_labels;
  bool ok = true;
  int error_status = 400;
  std::string error_type = "BadRequestError";
  std::string error_msg;
};

ParsedSystemOne ParseSystemOneBody(const nlohmann::ordered_json& body);

// The label set the GLiNER NER callback extracts for one question. A noul
// question's NER label is its instruction (GLiNER extracts entities FOR the
// question text, so "is there a person" extracts the label "person");
// choice and score questions extract their option texts, which ARE their
// q.labels. The DECISION path must not use this: kev/api.py to_record scores
// a noul over the two option texts option_text("no", criteria["false"]) /
// option_text("yes", criteria["true"]), which is what ParseSystemOneBody
// puts in q.labels, and the two meanings diverged when MODEL-KEV aligned
// q.labels with that reference.
std::vector<std::string> NerLabels(const SystemOneQuestion& q);

// ── Answer building ───────────────────────────────────────────────────────

// Build one kev answer from NER results for one question.
nlohmann::json BuildSystemOneAnswer(const SystemOneQuestion& q,
                                     const NerResult& result);

// Build one kev answer from decision scores (MODEL-LAYA / MODEL-KEV). Mirrors
// BuildSystemOneAnswer but takes per-option logits directly from the decision
// head instead of extracting NER confidences.
nlohmann::json BuildSystemOneAnswerDecision(const SystemOneQuestion& q,
                                             const DecisionResult& result);

// Render option texts for the Laya decision path, matching
// rl_common.py:render_options. The GLiNER NER path uses NerLabels(q)
// (the instruction for noul, the option texts for choice/score); the Laya
// decision path needs the reference option format.
std::vector<std::string> RenderDecisionOptions(const SystemOneQuestion& q);

}  // namespace vllm::entrypoints::openai::systemone

#endif  // VLLM_ENTRYPOINTS_OPENAI_SYSTEMONE_H_
