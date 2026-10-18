// See detect.h. ORIGINAL packaging-layer component (no upstream mirror).
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

namespace {

// ORDER MATTERS: first match wins, more specific first. "[THINK]" is the
// Mistral special-token literal and cannot appear in a <think>-family
// template; "<think>" is the broad Qwen/DeepSeek-family convention and sits
// last. The "<think>" row selects think_auto (NOT deepseek_r1): generic
// <think> templates include hybrid-thinking families that may answer WITHOUT
// a think block, and R1 semantics would swallow such answers whole as
// reasoning; think_auto degrades to plain content when no marker appears and
// is R1-identical otherwise. R1-style models keep explicit selection.
// Families whose markers are not template-stable (minimax_m2's
// end-token-only </think>, olmo3's plain-vocab <think>) stay EXPLICIT-ONLY:
// a template-level probe cannot distinguish them from deepseek_r1, and
// deepseek_r1's split behavior is the correct default for a plain
// <think>...</think> stream. muse_glimmer is rowed on the ATEM literal
// "<atem:function_calls>" that its chat template writes into the
// tool-definition preamble: a full literal, shared with nothing else here,
// and the same tell the tool-parser table uses (the two parsers are always
// selected together).
// kolibri1: the Kolibri template's no-reasoning system sentence
// ("Reasoning is disabled. Proceed straight to answering…") is a full literal
// shared with no other row and contained by no other row's marker. It must
// precede the generic "<think>" row, which the same template also contains
// (the plugin serves kolibri1's OWN reasoning parser, not think_auto:
// aleph-alpha-inference __init__.py:44-48).
constexpr ReasoningParserMarker kReasoningParserMarkers[] = {
    {"muse_glimmer", "<atem:function_calls>"},
    {"kolibri1",
     "Reasoning is disabled. Proceed straight to answering"},
    {"mistral", "[THINK]"},
    {"think_auto", "<think>"},
};

constexpr std::size_t kReasoningParserMarkerCount =
    sizeof(kReasoningParserMarkers) / sizeof(kReasoningParserMarkers[0]);

}  // namespace

std::string DetectReasoningParser(const std::string& chat_template,
                                  const ReasoningParserMarker* table,
                                  std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (table[i].template_marker != nullptr &&
        chat_template.find(table[i].template_marker) != std::string::npos) {
      return table[i].parser;
    }
  }
  return std::string();
}

std::string DetectReasoningParser(const std::string& chat_template) {
  return DetectReasoningParser(chat_template, kReasoningParserMarkers,
                               kReasoningParserMarkerCount);
}

const ReasoningParserMarker* ReasoningParserMarkerTable(std::size_t* out_count) {
  if (out_count != nullptr) *out_count = kReasoningParserMarkerCount;
  return kReasoningParserMarkers;
}

std::string ResolveReasoningParserName(const std::string& requested,
                                       const std::string& chat_template) {
  if (requested.empty() || requested == "none") return std::string();
  const std::string name =
      requested == "auto" ? DetectReasoningParser(chat_template) : requested;
  // "auto" with no matching marker legitimately resolves to "" (disabled).
  if (name.empty()) return name;
  if (get_reasoning_parser(name) == nullptr) {
    std::string msg = "unknown reasoning parser \"" + name +
                      "\" (registered parsers: ";
    const std::vector<std::string>& names = reasoning_parser_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i != 0) msg += ", ";
      msg += names[i];
    }
    msg += "; \"auto\" detects from the chat template, \"none\" disables)";
    throw std::invalid_argument(msg);
  }
  return name;
}

}  // namespace vllm::entrypoints::openai
