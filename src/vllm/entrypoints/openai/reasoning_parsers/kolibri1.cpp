// See kolibri1.h. Ported from aleph-alpha-inference @ 049a6a7bd240
// (aleph_alpha_inference/reasoning.py:36-61).
#include "vllm/entrypoints/openai/reasoning_parsers/kolibri1.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/qwen3.h"

namespace vllm::entrypoints::openai {

namespace pe = vllm::parser::engine;

// reasoning.py:36-48. `effort = kwargs.get("reasoning_effort")` — a JSON null
// IS Python None, so `effort is not None` is `effort != nullptr` here; only
// the literal "none" disables. Otherwise `kwargs.get("enable_thinking") is
// not False`: the template tests `enable_thinking is false`, so anything but
// the literal false (absent included) thinks.
bool Kolibri1ThinkingEnabled(
    const nlohmann::ordered_json& chat_template_kwargs) {
  const auto effort = chat_template_kwargs.find("reasoning_effort");
  if (effort != chat_template_kwargs.end() && !effort->is_null()) {
    // reasoning.py:47 `effort != "none"` — any non-None value other than the
    // literal "none" enables thinking, of whatever JSON type a client sends.
    return !(effort->is_string() && effort->get<std::string>() == "none");
  }
  const auto enable = chat_template_kwargs.find("enable_thinking");
  return !(enable != chat_template_kwargs.end() && enable->is_boolean() &&
           enable->get<bool>() == false);
}

Kolibri1ParserReasoningAdapter::Kolibri1ParserReasoningAdapter()
    // Upstream default (reasoning.py: no kwargs -> thinking on); replaced on
    // the first request by EnsureEngine when the kwargs say otherwise.
    : ParserEngineReasoningAdapter(std::make_unique<vllm::parser::Qwen3Parser>(
          pe::qwen3_config(true, "qwen3"), true)) {}

void Kolibri1ParserReasoningAdapter::EnsureEngine(
    const ChatCompletionRequest& request) {
  const bool thinking = Kolibri1ThinkingEnabled(request.chat_template_kwargs);
  if (state_chosen_) {
    // The state is a property of the request's kwargs, which do not change
    // mid-request; both extract feeds derive the same value every call.
    return;
  }
  state_chosen_ = true;
  if (!thinking) {
    engine_ = std::make_unique<vllm::parser::Qwen3Parser>(
        pe::qwen3_config(false, "qwen3"), false);
  }
}

ExtractedReasoning Kolibri1ParserReasoningAdapter::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& request) {
  EnsureEngine(request);
  return ParserEngineReasoningAdapter::extract_reasoning(model_output, request);
}

std::optional<DeltaMessage>
Kolibri1ParserReasoningAdapter::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  EnsureEngine(request);
  return ParserEngineReasoningAdapter::extract_reasoning_streaming(
      previous_text, current_text, delta_text, request);
}

}  // namespace vllm::entrypoints::openai
