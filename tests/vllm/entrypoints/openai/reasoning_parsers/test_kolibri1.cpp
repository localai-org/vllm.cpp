// Gate for the kolibri1 reasoning parser: the plugin-reference behavior of
// aleph-alpha-inference @ 049a6a7bd240 (aleph_alpha_inference/reasoning.py),
// the model-author vLLM plugin that IS the serving oracle for Kolibri
// (.agents/oracles/aleph-alpha-inference.md). Upstream vLLM implements
// nothing for this architecture; every expected value below comes from the
// plugin source or from the switch its chat template encodes.
//
// Mirrored behavior (reasoning.py):
//   - thinking_enabled (:36-48): a reasoning_effort that is not None wins and
//     only "none" disables thinking; without it, only a literal
//     enable_thinking: false disables thinking (the template tests
//     `enable_thinking is false`).
//   - Kolibri1Parser (:51): the Qwen3 grammar with the starting state derived
//     the way the template renders the generation prompt — thinking on leaves
//     `<|im_start|>assistant\n` open and the output starts INSIDE <think>;
//     thinking off renders the closed `<think>\n\n</think>\n\n` block and the
//     whole output is content (qwen3.py:247 passthrough).
//   - Kolibri1ParserReasoningAdapter (:61): the engine-backed reasoning face
//     over that parser; the request's chat_template_kwargs reach the split.
//
// Harness: the same TEXT-ONLY seam as test_qwen3.cpp (reasoning_test_utils.h);
// deltas isolate the think markers the way a detokenizer surfaces special
// tokens. The chat_template_kwargs arrive on the ChatCompletionRequest — the
// field the OpenAI protocol already carries (protocol.py:341).
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "reasoning_test_utils.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::reasoning_test::Extracted;
using vllm::entrypoints::openai::reasoning_test::RunStreaming;

namespace {

// One assistant turn that thinks, then answers, then calls a tool.
const std::vector<std::string> kThinkToolAnswerDeltas = {
    "<think>", "Let me check.", "</think>", "\n\nChecking. ",
    "<tool_call>", "\n{\"name\": \"get_weather\", \"arguments\": "
                   "{\"city\": \"Berlin\"}}\n",
    "</tool_call>",
};

ChatCompletionRequest RequestWithKwargs(nlohmann::ordered_json kwargs) {
  ChatCompletionRequest request;
  request.chat_template_kwargs = std::move(kwargs);
  return request;
}

}  // namespace

TEST_CASE("kolibri1 reasoning parser is registered") {
  CHECK(get_reasoning_parser("kolibri1") != nullptr);
}

TEST_CASE("kolibri1: default kwargs leave thinking ON") {
  // reasoning.py:44-48 — no reasoning_effort, no enable_thinking: the
  // template stops at `<|im_start|>assistant\n` and the parser starts inside
  // the think block (initial state REASONING).
  auto parser = get_reasoning_parser("kolibri1");
  REQUIRE(parser != nullptr);
  const Extracted out = RunStreaming(*parser, kThinkToolAnswerDeltas);
  CHECK(out.reasoning.value_or("") == "Let me check.");
  // The reasoning face runs under _skip_tool_parsing (adapters.py:51): the
  // tool body stays verbatim CONTENT for the tool parser to handle later.
  CHECK(out.content.value_or("") ==
        "\n\nChecking. <tool_call>\n{\"name\": \"get_weather\", "
        "\"arguments\": {\"city\": \"Berlin\"}}\n</tool_call>");
}

TEST_CASE("kolibri1: enable_thinking false starts in CONTENT") {
  // The template renders the closed empty block; the whole output is content,
  // tool markers included (qwen3.py:247 passthrough).
  auto parser = get_reasoning_parser("kolibri1");
  REQUIRE(parser != nullptr);
  const ChatCompletionRequest request = RequestWithKwargs(
      nlohmann::ordered_json{{"enable_thinking", false}});
  std::string joined;
  for (const auto& d : kThinkToolAnswerDeltas) joined += d;
  const ExtractedReasoning er = parser->extract_reasoning(joined, request);
  CHECK_FALSE(er.reasoning.has_value());
  CHECK(er.content.value_or("") == joined);
}

TEST_CASE("kolibri1: reasoning_effort drives the switch like the template") {
  struct Case {
    const char* name;
    nlohmann::ordered_json kwargs;
    bool thinking;
  };
  const Case cases[] = {
      {"effort none", {{"reasoning_effort", "none"}}, false},
      {"effort low", {{"reasoning_effort", "low"}}, true},
      {"effort medium", {{"reasoning_effort", "medium"}}, true},
      {"effort high", {{"reasoning_effort", "high"}}, true},
      {"effort minimal", {{"reasoning_effort", "minimal"}}, true},
      {"effort xhigh", {{"reasoning_effort", "xhigh"}}, true},
      {"effort max", {{"reasoning_effort", "max"}}, true},
      // reasoning.py:46 — effort is not None, so it WINS over the explicit
      // enable_thinking:false (the stock qwen3 parser goes wrong here).
      {"effort low beats enable_thinking false",
       {{"reasoning_effort", "low"}, {"enable_thinking", false}}, true},
      {"effort none beats enable_thinking true",
       {{"reasoning_effort", "none"}, {"enable_thinking", true}}, false},
  };
  for (const Case& c : cases) {
    CAPTURE(c.name);
    auto parser = get_reasoning_parser("kolibri1");
    REQUIRE(parser != nullptr);
    const ChatCompletionRequest request = RequestWithKwargs(c.kwargs);
    std::string joined;
    for (const auto& d : kThinkToolAnswerDeltas) joined += d;
    const ExtractedReasoning er = parser->extract_reasoning(joined, request);
    if (c.thinking) {
      CHECK(er.reasoning.value_or("") == "Let me check.");
      CHECK(er.content.value_or("") ==
            "\n\nChecking. <tool_call>\n{\"name\": \"get_weather\", "
            "\"arguments\": {\"city\": \"Berlin\"}}\n</tool_call>");
    } else {
      CHECK_FALSE(er.reasoning.has_value());
      CHECK(er.content.value_or("") == joined);
    }
  }
}

TEST_CASE("kolibri1: streaming matches the non-streaming split (thinking on)") {
  auto parser = get_reasoning_parser("kolibri1");
  REQUIRE(parser != nullptr);
  const ChatCompletionRequest request =
      RequestWithKwargs(nlohmann::ordered_json{{"reasoning_effort", "low"}});
  Extracted acc;
  std::string previous;
  for (const auto& delta : kThinkToolAnswerDeltas) {
    const std::string current = previous + delta;
    const std::optional<DeltaMessage> dm =
        parser->extract_reasoning_streaming(previous, current, delta, request);
    if (!dm.has_value()) {
      previous = current;
      continue;
    }
    if (dm->reasoning.has_value()) {
      acc.reasoning = acc.reasoning.value_or("") + *dm->reasoning;
    }
    if (dm->content.has_value()) {
      acc.content = acc.content.value_or("") + *dm->content;
    }
    previous = current;
  }
  CHECK(acc.reasoning.value_or("") == "Let me check.");
  CHECK(acc.content.value_or("") ==
        "\n\nChecking. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Berlin\"}}\n</tool_call>");
}

TEST_CASE("kolibri1: reasoning_effort null behaves like absent") {
  // reasoning.py:45-47 — `effort is not None` gates the override; an explicit
  // JSON null IS None, so enable_thinking decides.
  auto parser = get_reasoning_parser("kolibri1");
  REQUIRE(parser != nullptr);
  const ChatCompletionRequest request = RequestWithKwargs(
      nlohmann::ordered_json{{"reasoning_effort", nullptr},
                             {"enable_thinking", false}});
  std::string joined;
  for (const auto& d : kThinkToolAnswerDeltas) joined += d;
  const ExtractedReasoning er = parser->extract_reasoning(joined, request);
  CHECK_FALSE(er.reasoning.has_value());
  CHECK(er.content.value_or("") == joined);
}

TEST_CASE("kolibri1: streaming without kwargs defaults to thinking on") {
  // Thinking on starts the engine in REASONING, so a marker-less stream is
  // reasoning end to end (the template guarantees the model opens <think>
  // itself when the generation prompt stops at `<|im_start|>assistant\n`).
  auto parser = get_reasoning_parser("kolibri1");
  REQUIRE(parser != nullptr);
  const Extracted out = RunStreaming(*parser, {"Just answer."});
  CHECK(out.reasoning.value_or("") == "Just answer.");
  CHECK_FALSE(out.content.has_value());
}
