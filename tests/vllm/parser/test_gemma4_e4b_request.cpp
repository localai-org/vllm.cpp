// Expectations: Python bool(JSON value), vLLM a7c23ac gemma4.py:424-429.
// Text-only probes, not original Google vocabulary or served HTTP evidence.
#include <doctest/doctest.h>
#include <typeinfo>
#include "vllm/entrypoints/openai/reasoning_parsers/request.h"
using namespace vllm::entrypoints::openai;

TEST_CASE("Gemma request settings use Python JSON truthiness") {
  struct Row { const char* json; bool thinking; };
  const Row rows[] = {
      {R"json(null)json", false},
      {R"json(false)json", false},
      {R"json(0)json", false},
      {R"json(0.0)json", false},
      {R"json(-0.0)json", false},
      {R"json("")json", false},
      {R"json([])json", false},
      {R"json({})json", false},
      {R"json(true)json", true},
      {R"json(1)json", true},
      {R"json(-1)json", true},
      {R"json(0.5)json", true},
      {R"json(-0.5)json", true},
      {R"json("false")json", true},
      {R"json("auto")json", true},
      {R"json("0")json", true},
      {R"json(" ")json", true},
      {R"json([false])json", true},
      {R"json({"x": false})json", true},
      {R"json(18446744073709551615)json", true},
  };
  for (const auto& row : rows) {
    CAPTURE(row.json);
    ChatCompletionRequest request;
    request.chat_template_kwargs["enable_thinking"] =
        nlohmann::ordered_json::parse(row.json);
    request.chat_template_kwargs["reasoning_effort"] = "none";
    auto parser = MakeRequestReasoningParser("gemma4", request);
    REQUIRE(parser);
    for (const auto* text : {"", "plain", "<channel|><|turn>model\n",
                             "<|channel>old<|tool_response>done"})
      CHECK(parser->is_reasoning_end(text) == !row.thinking);
    CHECK_FALSE(parser->is_reasoning_end("<|channel>thought\n"));
    CHECK(parser->is_reasoning_end("<channel|>answer"));
    const auto result = parser->extract_reasoning("plain answer", request);
    CHECK_FALSE(result.reasoning.has_value());
    REQUIRE(result.content.has_value());
    CHECK(*result.content == "plain answer");
  }
}

TEST_CASE("Gemma missing request thinking defaults false") {
  for (const auto* kwargs : {"{}", "null", "{\"unrelated\":true}"}) {
    ChatCompletionRequest request;
    request.chat_template_kwargs = nlohmann::ordered_json::parse(kwargs);
    auto parser = MakeRequestReasoningParser("gemma4", request);
    REQUIRE(parser);
    CHECK(parser->is_reasoning_end("plain"));
  }
}

TEST_CASE("Request seam preserves other reasoning factories") {
  using Legacy = std::unique_ptr<ReasoningParser> (*)(const std::string&);
  Legacy legacy = &get_reasoning_parser;
  for (const auto& name : reasoning_parser_names()) {
    if (name == "gemma4") continue;
    for (bool thinking : {false, true}) {
      CAPTURE(name); CAPTURE(thinking);
      ChatCompletionRequest request;
      request.chat_template_kwargs["enable_thinking"] = thinking;
      auto expected = legacy(name);
      auto actual = MakeRequestReasoningParser(name, request);
      REQUIRE(expected);
      REQUIRE(actual);
      CHECK(typeid(*actual) == typeid(*expected));
      for (const auto* text : {"", "plain", "<think>x", "</think>",
                               "<|channel>thought\n", "<channel|>"})
        CHECK(actual->is_reasoning_end(text) == expected->is_reasoning_end(text));
    }
  }
  for (const auto* name : {"", "unknown", "none"}) {
    CHECK_FALSE(legacy(name));
    CHECK_FALSE(MakeRequestReasoningParser(name, {}));
  }
}
