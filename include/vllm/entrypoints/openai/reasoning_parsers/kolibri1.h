// Ported from: aleph-alpha-inference @ 049a6a7bd240
// (aleph_alpha_inference/reasoning.py) — the model-author vLLM plugin for
// Kolibri, the serving oracle for the kolibri1 architecture
// (.agents/oracles/aleph-alpha-inference.md; upstream vLLM implements nothing
// for it).
//
// reasoning.py:36-48 thinking_enabled — the template's own switch as a
// function: a reasoning_effort that is not None WINS (only "none" disables
// thinking), and only absent-or-None effort falls through to a literal
// enable_thinking == false test. The stock qwen3 reasoning parser reads
// enable_thinking alone, which is wrong on exactly the two arms the plugin
// exists for: effort arriving in chat_template_kwargs, and effort
// contradicting an explicit enable_thinking (reasoning.py:12-22).
//
// reasoning.py:51-58 Kolibri1Parser — the Qwen3 grammar with the starting
// state chosen that way; reasoning.py:61 Kolibri1ParserReasoningAdapter — the
// engine-backed reasoning face over it. The upstream ctor reads the request's
// chat_template_kwargs at parser construction; this seam constructs parsers
// per server name (reasoning_parsers/abstract.cpp), so the kolibri1 adapter
// derives the state lazily from the FIRST request it sees and re-derives it
// on every call (both extract methods carry the request; is_reasoning_end is
// thinking-independent for the Qwen3 grammar — qwen3.cpp).
#ifndef VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_KOLIBRI1_H_
#define VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_KOLIBRI1_H_

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.h"

namespace vllm::entrypoints::openai {

// reasoning.py:36 (thinking_enabled).
bool Kolibri1ThinkingEnabled(
    const nlohmann::ordered_json& chat_template_kwargs);

// reasoning.py:61 (Kolibri1ParserReasoningAdapter). The Qwen3 engine face
// with the starting state derived from the request's chat_template_kwargs the
// way the Kolibri template renders its generation prompt.
class Kolibri1ParserReasoningAdapter final : public ParserEngineReasoningAdapter {
 public:
  Kolibri1ParserReasoningAdapter();

  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // Point engine_ at the state the request's kwargs select. Idempotent after
  // the first call for a given state; a mid-stream flip is impossible because
  // both extract feeds derive the same kwargs every call.
  void EnsureEngine(const ChatCompletionRequest& request);
  bool state_chosen_ = false;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_KOLIBRI1_H_
