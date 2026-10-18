// Gate for the kolibri1 tool parser alias. The serving oracle
// (aleph-alpha-inference @ 049a6a7bd240) registers the kolibri1 tool-call
// parser as vLLM's Hermes parser verbatim:
//   aleph_alpha_inference/__init__.py:50-54
//     ToolParserManager.register_lazy_module(
//         name="kolibri1",
//         module_path="vllm.tool_parsers.hermes_tool_parser",
//         class_name="Hermes2ProToolParser")
// with the comment "Kolibri 1 currently shares the Hermes
// `<tool_call>...</tool_call>` format." So the local kolibri1 name MUST
// resolve to the same HermesToolParser behavior, byte for byte — the alias is
// the mirror of the plugin's registration, not a new dialect.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::get_tool_parser;
using vllm::entrypoints::openai::ResolveToolParserName;

namespace {

// The exact shape the Kolibri chat template instructs the model to emit
// (tokenizer_config.json chat_template, assistant tool_calls branch: the
// `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` wrapper).
const std::string kKolibriToolCall =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
    "{\"city\": \"Berlin\"}}\n</tool_call>";

}  // namespace

TEST_CASE("kolibri1 tool parser resolves as a registry name") {
  CHECK(get_tool_parser("kolibri1") != nullptr);
  CHECK(ResolveToolParserName("kolibri1", "") == "kolibri1");
}

TEST_CASE("kolibri1 parses the Hermes <tool_call> format like hermes") {
  auto kolibri = get_tool_parser("kolibri1");
  auto hermes = get_tool_parser("hermes");
  REQUIRE(kolibri != nullptr);
  REQUIRE(hermes != nullptr);

  const ChatCompletionRequest request;
  const auto k = kolibri->extract_tool_calls(kKolibriToolCall, request);
  const auto h = hermes->extract_tool_calls(kKolibriToolCall, request);
  REQUIRE(k.tool_calls.size() == 1);
  REQUIRE(h.tool_calls.size() == 1);
  CHECK(k.tool_calls[0].function.name == h.tool_calls[0].function.name);
  CHECK(k.tool_calls[0].function.name == "get_weather");
  CHECK(k.tool_calls[0].function.arguments ==
        h.tool_calls[0].function.arguments);
  CHECK(k.content == h.content);
}

TEST_CASE("kolibri1 streams the Hermes format like hermes") {
  auto kolibri = get_tool_parser("kolibri1");
  auto hermes = get_tool_parser("hermes");
  REQUIRE(kolibri != nullptr);
  REQUIRE(hermes != nullptr);

  // Feed the canonical Hermes delta cadence (the one the hermes gate drives
  // in test_tool_parsers.cpp): wrapper tokens atomic, body one fragment.
  const std::vector<std::string> deltas = {
      "<tool_call>",  "{\"name\": \"get_",          "weather\", ",
      "\"arguments\": {\"ci", "ty\": \"Ber", "lin\"}}", "</tool_call>"};
  const ChatCompletionRequest request;
  vllm::entrypoints::openai::DeltaMessage k;
  vllm::entrypoints::openai::DeltaMessage h;
  k.tool_calls = std::vector<vllm::entrypoints::openai::DeltaToolCall>{};
  h.tool_calls = std::vector<vllm::entrypoints::openai::DeltaToolCall>{};
  std::string previous;
  for (const auto& delta : deltas) {
    const std::string current = previous + delta;
    if (auto dm = kolibri->extract_tool_calls_streaming(previous, current,
                                                        delta, request)) {
      if (dm->tool_calls.has_value()) {
        k.tool_calls->insert(k.tool_calls->end(), dm->tool_calls->begin(), dm->tool_calls->end());
      }
    }
    if (auto dm = hermes->extract_tool_calls_streaming(previous, current,
                                                       delta, request)) {
      if (dm->tool_calls.has_value()) {
        h.tool_calls->insert(h.tool_calls->end(), dm->tool_calls->begin(), dm->tool_calls->end());
      }
    }
    previous = current;
  }
  // The streamed shape is name-first then argument DIFFS (the hermes gate
  // pins this), so compare like the hermes gate does: one name, arguments
  // concatenating to the complete object.
  REQUIRE(k.tool_calls.has_value());
  REQUIRE(h.tool_calls.has_value());
  auto collect = [](std::vector<vllm::entrypoints::openai::DeltaToolCall>& v) {
    std::optional<std::string> name;
    std::string args;
    for (const auto& tc : v) {
      if (tc.function.name.has_value()) name = tc.function.name;
      if (tc.function.arguments.has_value()) args += *tc.function.arguments;
    }
    return std::pair<std::optional<std::string>, std::string>{name, args};
  };
  const auto [kname, kargs] = collect(*k.tool_calls);
  const auto [hname, hargs] = collect(*h.tool_calls);
  REQUIRE(kname.has_value());
  CHECK(*kname == "get_weather");
  CHECK(*kname == hname);
  CHECK(kargs == hargs);
  CHECK(kargs == "{\"city\": \"Berlin\"}");
}