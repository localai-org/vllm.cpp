// Ported from: vllm/reasoning/abs_reasoning_parsers.py @ e24d1b24
// (ReasoningParserManager.get_reasoning_parser). Hand-wired factory over the T0
// reasoning-parser formats, mirroring tool_parsers/abstract.cpp get_tool_parser.
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.h"
#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/reasoning_parsers/kolibri1.h"
#include "vllm/entrypoints/openai/reasoning_parsers/think_auto.h"
#include "vllm/entrypoints/openai/reasoning_parsers/minimax_m2.h"
#include "vllm/entrypoints/openai/reasoning_parsers/mistral.h"
#include "vllm/entrypoints/openai/reasoning_parsers/muse_glimmer.h"
#include "vllm/entrypoints/openai/reasoning_parsers/olmo3.h"
#include "vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.h"
#include "vllm/entrypoints/openai/reasoning_parsers/step3.h"

namespace vllm::entrypoints::openai {

std::unique_ptr<ReasoningParser> get_reasoning_parser(const std::string& name) {
  if (name == "think_auto") {
    return std::make_unique<ThinkAutoReasoningParser>();
  }
  if (name == "deepseek_r1") {
    return std::make_unique<DeepSeekR1ReasoningParser>();
  }
  // deepseek_v3_reasoning_parser.py:20 (name "deepseek_v3"). Hybrid-thinking
  // delegate; the name-only factory mirrors upstream's DEFAULT construction
  // (thinking=False -> Identity passthrough). See deepseek_v3.h for the deviation.
  if (name == "deepseek_v3") {
    return std::make_unique<DeepSeekV3ReasoningParser>(false);
  }
  // __init__.py:71 registers "holo2" -> DeepSeekV3ReasoningWithThinkingParser
  // (deepseek_v3_reasoning_parser.py:83), the thinking-default variant -> R1 split.
  if (name == "holo2") {
    return std::make_unique<DeepSeekV3ReasoningWithThinkingParser>();
  }
  if (name == "mistral") {
    return std::make_unique<MistralReasoningParser>();
  }
  if (name == "minimax_m2") {
    return std::make_unique<MiniMaxM2ReasoningParser>();
  }
  if (name == "minimax_m2_append_think") {
    return std::make_unique<MiniMaxM2AppendThinkReasoningParser>();
  }
  if (name == "step3") {
    return std::make_unique<Step3ReasoningParser>();
  }
  if (name == "olmo3") {
    return std::make_unique<Olmo3ReasoningParser>();
  }
  // muse_glimmer_reasoning_parser.py:106 (register_module("muse_glimmer")).
  if (name == "muse_glimmer") {
    return std::make_unique<MuseGlimmerReasoningParser>();
  }
  // __init__.py:115 ("qwen3") and :87 ("mimo") register the SAME class,
  // Qwen3ParserReasoningAdapter (qwen3_engine_reasoning_parser.py, a re-export
  // of registered_adapters.py:48) — the engine-backed reasoning face, NOT a
  // text splitter. Constructed with upstream's default thinking=True
  // (chat_template_kwargs threading is W4, see specs/reasoning-parsers.md).
  if (name == "qwen3" || name == "mimo") {
    return std::make_unique<Qwen3ParserReasoningAdapter>();
  }
  // aleph-alpha-inference @ 049a6a7bd240 __init__.py:44-48 registers
  // "kolibri1" -> Kolibri1ParserReasoningAdapter (reasoning.py:61): the qwen3
  // engine face with the starting state derived from the request's
  // chat_template_kwargs the way the Kolibri template switches thinking
  // (reasoning_effort "none" / literal enable_thinking false; see kolibri1.h).
  if (name == "kolibri1") {
    return std::make_unique<Kolibri1ParserReasoningAdapter>();
  }
  return nullptr;
}

// The enumeration of the factory above (see abstract.h). ADD A NAME HERE IN THE
// SAME CHANGE THAT ADDS ITS FACTORY BRANCH.
const std::vector<std::string>& reasoning_parser_names() {
  static const std::vector<std::string> names = {
      "think_auto", "deepseek_r1", "deepseek_v3", "holo2",
      "mistral", "minimax_m2", "minimax_m2_append_think", "step3", "olmo3",
      "muse_glimmer", "qwen3", "mimo", "kolibri1",
  };
  return names;
}

}  // namespace vllm::entrypoints::openai
