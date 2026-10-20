// Request-local construction; template server defaults are not merged here.
#pragma once
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {
std::unique_ptr<ReasoningParser> MakeRequestReasoningParser(
    const std::string& name, const ChatCompletionRequest& request);
}  // namespace vllm::entrypoints::openai
