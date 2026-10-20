// Gemma registration: pinned vLLM a7c23ac96 reasoning/__init__.py:55-58.
#include <algorithm>
#include "doctest/doctest.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
namespace oai = vllm::entrypoints::openai;
TEST_CASE("Gemma reasoning factory registers exactly once") {
  CHECK(oai::get_reasoning_parser("gemma4") != nullptr);
  const auto& names = oai::reasoning_parser_names();
  CHECK(std::count(names.begin(), names.end(), "gemma4") == 1);
  for (const auto& name : names) CHECK(oai::get_reasoning_parser(name) != nullptr);
}

#include "vllm/parser/engine/parser_engine.h"
namespace {
using vllm::parser::engine::EngineTokenizer;
class PromptTokenizer : public EngineTokenizer {
 public:
  std::map<std::string, int> vocab{{"<|channel>", 101}, {"<channel|>", 102},
      {"<|tool_call>", 103}, {"<tool_call|>", 104}, {"<|turn>", 105},
      {"<|tool_response>", 106}};
  const std::map<std::string, int>& get_vocab() const override { return vocab; }
  std::string decode(int id) const override {
    for (const auto& [text, value] : vocab) if (value == id) return text;
    return "";
  }
};
}

TEST_CASE("Gemma reasoning ABC forwards open prompt before first small delta") {
  PromptTokenizer tok;
  for (bool thinking : {false, true}) {
    auto p = oai::get_reasoning_parser("gemma4", thinking, &tok);
    REQUIRE(p);
    p->adjust_initial_state_from_prompt({101, 3000, 3001});
    auto d = p->extract_reasoning_streaming("", "Sure", "Sure", {});
    REQUIRE(d);
    CHECK(d->reasoning.value_or("") == "Sure");
    CHECK_FALSE(d->content.has_value());
  }
}

#include "vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.h"
#include "vllm/parser/gemma4.h"
namespace {
const std::string reason =
    "The user is asking for the current weather in Dallas, Texas, "
    "and specifically requests the temperature in Fahrenheit. "
    "I have a tool `get_current_weather` that can provide this "
    "information. I should call this tool with `city='Dallas'`, "
    "`state='TX'`, and `unit='fahrenheit'`.";
const std::string tool = "<|tool_call>call:get_current_weather{city:<|\"|>Dallas<|\"|>,"
    "state:<|\"|>TX<|\"|>,unit:<|\"|>fahrenheit<|\"|>}<tool_call|>";
const std::string full = "<|channel>thought\n" + reason + "<channel|>" + tool;
}
// Retained TestAdapterExtractReasoning1112-1150; expected text is Python input,
// not captured C++ output. Existing guard behavior is regression coverage.
TEST_CASE("Gemma reasoning adapter preserves canonical tool text") {
  for (bool thinking : {false, true}) {
    auto p = oai::get_reasoning_parser("gemma4", thinking, nullptr);
    auto out = p->extract_reasoning(full, {});
    CHECK(out.reasoning == reason);
    CHECK(out.content == tool);
    auto plain = oai::get_reasoning_parser("gemma4", thinking, nullptr)
        ->extract_reasoning("Hello world, no thinking here.", {});
    CHECK_FALSE(plain.reasoning.has_value());
    CHECK(plain.content == "Hello world, no thinking here.");
    for (std::size_t chunk : {1, 2, 10, 1000}) {
      auto streaming = oai::get_reasoning_parser("gemma4", thinking, nullptr);
      std::string reasoning, content;
      for (std::size_t i = 0; i < full.size(); i += chunk) {
        auto d = streaming->extract_reasoning_streaming("", "", full.substr(i, chunk), {});
        if (d) { reasoning += d->reasoning.value_or(""); content += d->content.value_or(""); }
      }
      CHECK(reasoning == reason);
      CHECK(content == tool);
    }
  }
}
TEST_CASE("Gemma adapter restores skip flag and protects prepared initialization") {
  PromptTokenizer tok;
  for (bool saved : {false, true}) {
    auto engine = std::make_unique<vllm::parser::Gemma4Parser>(&tok);
    auto* raw = engine.get();
    oai::ParserEngineReasoningAdapter p(std::move(engine));
    raw->set_skip_tool_parsing(saved);
    p.extract_reasoning(full, {});
    CHECK(raw->skip_tool_parsing() == saved);
    p.adjust_initial_state_from_prompt({101});
    raw->initialize_streaming();
    auto d = p.extract_reasoning_streaming("", "", "Sure", {});
    REQUIRE(d);
    CHECK(d->reasoning == "Sure");
    CHECK(raw->skip_tool_parsing() == saved);
  }
}
TEST_CASE("Gemma ABC prompt boundaries missing IDs and null tokenizer") {
  PromptTokenizer tok;
  for (bool thinking : {false, true}) {
    for (int boundary : {102, 103, 105, 106}) {
      for (bool opened : {false, true}) {
        auto p = oai::get_reasoning_parser("gemma4", thinking, &tok);
        p->adjust_initial_state_from_prompt(opened ? std::vector<int>{boundary,101}
                                                  : std::vector<int>{101,boundary});
        auto d = p->extract_reasoning_streaming("", "", "Sure", {});
        REQUIRE(d);
        CHECK(d->reasoning.value_or("") == (opened ? "Sure" : ""));
        CHECK(d->content.value_or("") == (opened ? "" : "Sure"));
      }
    }
    for (bool nulltok : {false,true}) {
      PromptTokenizer missing; missing.vocab.erase("<|channel>");
      auto p = oai::get_reasoning_parser("gemma4", thinking, nulltok ? nullptr : &missing);
      p->adjust_initial_state_from_prompt({101});
      auto d = p->extract_reasoning_streaming("", "", "Sure", {});
      REQUIRE(d); CHECK(d->content == "Sure"); CHECK_FALSE(d->reasoning.has_value());
    }
  }
}
TEST_CASE("Gemma adapter prompt prefix and explicit repeat remain caller owned") {
  PromptTokenizer tok;
  for (bool thinking : {false,true}) {
    auto p = oai::get_reasoning_parser("gemma4", thinking, &tok);
    p->adjust_initial_state_from_prompt({101});
    std::string reasoning,content;
    for (const auto& text : {"<|channel>", "thought\n", "Sure", "<channel|>", "Hello"}) {
      auto d=p->extract_reasoning_streaming("", "", text, {});
      if(d) { reasoning+=d->reasoning.value_or(""); content+=d->content.value_or(""); }
    }
    CHECK(reasoning == "Sure"); CHECK(content == "Hello");
    p->adjust_initial_state_from_prompt({101});
    auto d=p->extract_reasoning_streaming("", "", "Again", {});
    REQUIRE(d); CHECK(d->reasoning == "Again");
  }
}
TEST_CASE("Reasoning context preserves old symbol defaults and noop hooks") {
  std::unique_ptr<oai::ReasoningParser> (*old_factory)(const std::string&) =
      &oai::get_reasoning_parser;
  CHECK(old_factory("gemma4")->is_reasoning_end(""));
  CHECK_FALSE(oai::get_reasoning_parser("gemma4",true,nullptr)->is_reasoning_end(""));
  CHECK_FALSE(old_factory("unknown"));
  for (const auto& name : oai::reasoning_parser_names()) {
    if(name == "gemma4") continue;
    auto a=old_factory(name); auto b=oai::get_reasoning_parser(name,false,nullptr);
    a->adjust_initial_state_from_prompt({101});
    auto x=a->extract_reasoning("Hello",{}); auto y=b->extract_reasoning("Hello",{});
    CHECK(x.reasoning == y.reasoning); CHECK(x.content == y.content);
    CHECK(a->is_reasoning_end("") == b->is_reasoning_end(""));
  }
}
