// Tev1 on /v1/systemone and vllm_decide (MODEL-TEV1 Phase 6).
//
// Ollama 0.35 serves Tev1 (togethercomputer/Tev1-4B-experimental and
// -0.8B-experimental) on /v1/systemone: each typed question is scored by the
// next-token logits of its answer letters. This file holds the Tev1 prompt
// contract and the one production entry, Tev1Decide, that both the server's
// /v1/systemone route and vllm_decide call. The compilation and the answer are
// the shared decision_scorer's (also Nimble's).
//
// The prompt is the model's own, not Ollama's: one question per prompt, the
// system prompt and JSON user turn of togethercomputer/tev1 @ 1dde7782
// examples/decide.py, rendered with enable_thinking=False. Ollama renders
// Nimble's schema turn for every decision model; .agents/specs/tev1.md
// (Phase 6) records each difference.
//
// The logits come from the ENGINE: one request per question with
// max_tokens=1 and logprob_token_ids = the answer letters, which is vLLM's
// generative scoring (entrypoints/generate/generative_scoring/serving.py:
// 247-255, softmax over the labels at :456-470). The decision therefore shares
// the scheduler and KV cache with chat traffic on the same engine.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/decision_scorer.h"

namespace vllm {

namespace tok {
class Tokenizer;
}
namespace v1 {
class AsyncLLM;
}

namespace tev1 {

// decide.py: "Supply 2-24 options", labels "ABCDEFGHIJKLMNOPQRSTUVWX".
inline constexpr int kMaxOptions = 24;

// decide.py SYSTEM, verbatim.
extern const char* const kSystemPrompt;

// The text every prompt ends with; each answer letter must be one ordinary
// token after it.
inline constexpr const char* kAnswerBoundary = "</think>\n\n";

using decision_scorer::Field;
using decision_scorer::Request;
using decision_scorer::RequestError;

// Validate and compile a /v1/systemone body under Tev1's contract: at most 24
// options per question, and a nonempty description for every option
// (decide.py). Throws RequestError.
Request CompileRequest(const nlohmann::ordered_json& body);

// The rendered chat prompt for every field, in field order.
std::vector<std::string> BuildPrompts(const Request& request);

}  // namespace tev1

using Tev1Response = decision_scorer::ScoredRequest;

// Run a whole /v1/systemone request on a Tev1 engine. max_model_len is the
// engine's; a prompt must leave room for the one sampled token. Throws
// tev1::RequestError for a request the contract refuses, std::runtime_error
// for an engine fault.
Tev1Response Tev1Decide(v1::AsyncLLM& engine, const tok::Tokenizer& tokenizer,
                        int64_t max_model_len,
                        const nlohmann::ordered_json& body);

// The same pipeline over any candidate-logits source.
Tev1Response Tev1DecideWith(const tok::Tokenizer& tokenizer,
                            int64_t max_model_len,
                            const nlohmann::ordered_json& body,
                            const decision_scorer::CandidateLogitsFn& logits_fn);

// The engine logits source: one request per prompt, submitted as one wave,
// greedy, max_tokens=1, no detokenization, logprob_token_ids = the prompt's
// candidates. Returns their raw logprobs (the logit less one per-row
// constant, which the softmax removes) and one output token per prompt.
decision_scorer::CandidateLogitsFn EngineCandidateLogits(v1::AsyncLLM& engine);

}  // namespace vllm
