// Text-form port of vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94:
// tests/reasoning/test_gemma4_reasoning_parser.py:265-288 and
// parser/engine/parser_engine.py:650-674. Not original-vocabulary parity.
#include <doctest/doctest.h>
#include "vllm/parser/parser_manager.h"

namespace {
// Artificial IDs only; no claim about the Google tokenizer vocabulary.
class SyntheticTokenizer : public vllm::parser::engine::EngineTokenizer {
 public:
  std::map<std::string, int> vocab{{"<|channel>", 1}, {"<channel|>", 2},
      {"<|tool_call>", 3}, {"<tool_call|>", 4}, {"<|turn>", 5},
      {"<|tool_response>", 6}};
  const std::map<std::string, int>& get_vocab() const override { return vocab; }
  std::string decode(int id) const override {
    for (const auto& [text, value] : vocab) if (value == id) return text;
    return "";
  }
};
}

TEST_CASE("Gemma canonical new-turn reasoning end through factory") {
  const std::string text = "<|channel>thought\n1st thought<channel|>1st content<turn|>\n"
      "<|turn>user\nThanks<|turn>model\n";
  auto omitted = vllm::parser::get_parser_engine("gemma4");
  REQUIRE(omitted);
  CHECK(omitted->is_reasoning_end(text));
  SyntheticTokenizer tok;
  for (bool thinking : {false, true}) {
    auto parser = vllm::parser::get_parser_engine("gemma4", thinking, &tok);
    REQUIRE(parser);
    CHECK(parser->is_reasoning_end(text) == !thinking);
  }
}

TEST_CASE("Gemma latest literal decides before boundary or thinking fallback") {
  struct Row { const char* text; bool off; bool on; };
  const Row rows[] = {
      {"", true, false}, {"plain answer", true, false},
      {"thought\n", true, false}, {"<|channel>thought\n", false, false},
      {"<channel|>", true, true}, {"<|tool_call>call:f{}", true, true},
      {"<|channel>x<|tool_call>call:f{}", true, true},
      {"<channel|><|channel>x", false, false},
      {"<|tool_call><|channel>x", false, false},
      {"<|channel>x<channel|>", true, true},
      {"<channel|><|turn>model\n", true, false},
      {"<channel|><|tool_response>ok", true, false},
      {"<|channel>x<|turn>model\n", true, false},
      {"<|channel>x<|tool_response>ok", true, false},
      {"<|tool_call><|turn>model\n", true, false},
      {"<|tool_call><|tool_response>ok", true, false},
      {"<|turn>model\n<|channel>x", false, false},
      {"<|tool_response>ok<|channel>x", false, false},
      {"<|turn>model\n<channel|>", true, true},
      {"<|tool_response>ok<|tool_call>", true, true},
      {"<channel|><|turn>model<|tool_response>ok", true, false},
      {"<|channel>x<turn|>", false, false}, // turn END is not a boundary
      {"<tool_call|>", true, false}, // tool END is not reasoning end
      {"<|chan", true, false}, // partial literals are plain text
  };
  SyntheticTokenizer tok;
  auto omitted = vllm::parser::get_parser_engine("gemma4");
  REQUIRE(omitted);
  for (const auto& row : rows) {
    CAPTURE(row.text);
    CHECK(omitted->is_reasoning_end(row.text) == row.off);
    for (bool thinking : {false, true}) {
      auto parser = vllm::parser::get_parser_engine("gemma4", thinking, &tok);
      REQUIRE(parser);
      CHECK(parser->is_reasoning_end(row.text) == (thinking ? row.on : row.off));
    }
  }
}

TEST_CASE("Gemma thinking leaves plain output in initial CONTENT") {
  SyntheticTokenizer tok;
  for (bool thinking : {false, true}) {
    auto parser = vllm::parser::get_parser_engine("gemma4", thinking, &tok);
    REQUIRE(parser);
    auto result = parser->extract_reasoning("plain answer", {});
    CHECK_FALSE(result.first.has_value());
    REQUIRE(result.second.has_value());
    CHECK(*result.second == "plain answer");
  }
}

TEST_CASE("Engine factory preserves other-family omitted defaults") {
  for (const auto* name : {"qwen3", "seed_oss", "kimi_k2", "minimax_m2",
           "deepseek_v4", "deepseek_v32", "nemotron_v3", "glm47_moe", "inkling"}) {
    CAPTURE(name);
    auto omitted = vllm::parser::get_parser_engine(name);
    auto enabled = vllm::parser::get_parser_engine(name, true);
    REQUIRE(omitted);
    REQUIRE(enabled);
    for (const auto* text : {"", "plain", "<think>x", "</think>",
             "<seed:think>x", "</seed:think>"})
      CHECK(omitted->is_reasoning_end(text) == enabled->is_reasoning_end(text));
  }
  CHECK_FALSE(vllm::parser::get_parser_engine("unknown"));
}

TEST_CASE("Gemma retained streaming end-detection parameters in text form") {
  // All nine TestGemma4IsReasoningEnd parameters, with artificial token IDs
  // represented by their literals. This does not port streaming/prompt hooks.
  struct Row { const char* text; bool thinking; bool ended; };
  const Row rows[] = {
      {"<|channel>x<channel|>", true, true},
      {"<|channel>x<|tool_call>", true, true},
      {"<|channel>x", true, false},
      {"<|turn>model", true, false},
      {"<channel|><|turn>model", true, false},
      {"<|turn>model", false, true},
      {"<|turn>model<|channel>x<channel|>", false, true},
      {"<|channel>x", false, false},
      {"<|channel>x<channel|><|tool_call>call", true, true},
  };
  SyntheticTokenizer tok;
  for (const auto& row : rows) {
    CAPTURE(row.text);
    auto parser = vllm::parser::get_parser_engine("gemma4", row.thinking, &tok);
    REQUIRE(parser);
    CHECK(parser->is_reasoning_end(row.text) == row.ended);
  }
}
