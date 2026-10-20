#include "vllm/entrypoints/openai/reasoning_parsers/request.h"

namespace vllm::entrypoints::openai {
namespace {
// Python truthiness for JSON values, not Kolibri's reasoning-effort policy.
// vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94 parser/gemma4.py:424-429.
bool JsonTruthy(const nlohmann::ordered_json& value) {
  if (value.is_null()) return false;
  if (value.is_boolean()) return value == true;
  if (value.is_number()) return value != 0;
  if (value.is_string()) return !value.get_ref<const std::string&>().empty();
  return !value.empty();
}
}  // namespace

std::unique_ptr<ReasoningParser> MakeRequestReasoningParser(
    const std::string& name, const ChatCompletionRequest& request) {
  if (name != "gemma4") return get_reasoning_parser(name);
  const auto& kwargs = request.chat_template_kwargs;
  const auto thinking = kwargs.find("enable_thinking");
  const bool enabled = thinking != kwargs.end() && JsonTruthy(*thinking);
  return get_reasoning_parser(name, enabled, nullptr);
}
}  // namespace vllm::entrypoints::openai
