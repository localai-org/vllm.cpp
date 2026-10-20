// Pinned vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94 gemma4.py:480-526.
// Synthetic vocabulary, not original Google token parity. Expected fields come
// from retained TestGemma4PromptOpenReasoning, PreInit and ChannelLess tests.
#include <doctest/doctest.h>
#include "vllm/parser/parser_manager.h"
#include "vllm/parser/engine/configs.h"

namespace {
using namespace vllm::parser;
using namespace vllm::parser::engine;
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
struct Fields {
  std::string reasoning, content;
  void add(const std::optional<oai::DeltaMessage>& d) {
    if (!d) return;
    reasoning += d->reasoning.value_or("");
    content += d->content.value_or("");
  }
};
const std::vector<int> open{101, 3000, 3001};
}
TEST_CASE("Gemma prompt-open first small delta is reasoning before close") {
  PromptTokenizer tok;
  for (bool thinking : {false, true}) {
    auto p = get_parser_engine("gemma4", thinking, &tok);
    Fields f;
    f.add(p->parse_delta("Sure", {}, {}, false, &open));
    CHECK(f.reasoning == "Sure");
    CHECK(f.content.empty());
    f.add(p->parse_delta(", the answer is 42", {}, {}, false, &open));
    CHECK(f.reasoning == "Sure, the answer is 42");
    CHECK(f.content.empty());
    f.add(p->parse_delta("<channel|>", {}, {}, false, &open));
    f.add(p->parse_delta("Hello", {}, {}, true, &open));
    CHECK(f.reasoning == "Sure, the answer is 42");
    CHECK(f.content == "Hello");
  }
}

TEST_CASE("Gemma prompt batches preserve generated and redundant openers") {
  PromptTokenizer tok;
  for (bool thinking : {false, true}) {
    for (int chunk : {1, 2, 10, 1000}) {
      for (bool generated : {false, true}) {
        for (bool prompt_open : {false, true}) {
          if (!generated && !prompt_open) continue;
          auto p = get_parser_engine("gemma4", thinking, &tok);
          const std::vector<int> prompt = prompt_open ? open : std::vector<int>{105, 9100, 9101};
          const std::vector<std::string> tokens = generated
              ? std::vector<std::string>{"<|channel>", "thought", "\n", "Reason", "ing", " body", "<channel|>", "Final", " content"}
              : std::vector<std::string>{"Sure", ", the", " answer", " is", " 42", "<channel|>", "Hello", " world"};
          Fields f;
          for (std::size_t i = 0; i < tokens.size();) {
            std::string text;
            for (int j = 0; j < chunk && i < tokens.size(); ++j) text += tokens[i++];
            f.add(p->parse_delta(text, {}, {}, i == tokens.size(), &prompt));
          }
          CHECK(f.reasoning == (generated ? "Reasoning body" : "Sure, the answer is 42"));
          CHECK(f.content == (generated ? "Final content" : "Hello world"));
        }
      }
    }
  }
}

TEST_CASE("Gemma prompt boundary order and channel-less nonstream agreement") {
  PromptTokenizer tok;
  for (bool thinking : {false, true}) {
    for (int boundary : {102, 103, 105, 106}) {
      for (bool newer_open : {false, true}) {
        const std::vector<int> prompt = newer_open ? std::vector<int>{boundary, 101, 3000}
                                                  : std::vector<int>{101, 3000, boundary, 9100};
        auto p = get_parser_engine("gemma4", thinking, &tok);
        Fields f;
        f.add(p->parse_delta("Plain answer", {}, {}, true, &prompt));
        CHECK(f.reasoning == (newer_open ? "Plain answer" : ""));
        CHECK(f.content == (newer_open ? "" : "Plain answer"));
      }
    }
    for (int chunk : {1, 2, 10, 1000}) {
      const std::vector<int> prompt{101, 102, 105, 9100, 9101};
      auto p = get_parser_engine("gemma4", thinking, &tok);
      const std::string plain = "This is a direct final answer without channel markers.";
      const auto nonstream = p->extract_reasoning(plain, {});
      CHECK_FALSE(nonstream.first);
      CHECK(nonstream.second == plain);
      p = get_parser_engine("gemma4", thinking, &tok);
      Fields f;
      for (std::size_t i = 0; i < plain.size(); i += chunk)
        f.add(p->parse_delta(plain.substr(i, chunk), {}, {}, i + chunk >= plain.size(), &prompt));
      CHECK(f.reasoning.empty());
      CHECK(f.content == plain);
    }
  }
}

TEST_CASE("Gemma missing vocabulary never guesses prompt markers") {
  for (const auto* missing : {"<|channel>", "<channel|>", "<|tool_call>", "<|turn>", "<|tool_response>"}) {
    PromptTokenizer tok;
    int removed = tok.vocab.at(missing);
    tok.vocab.erase(missing);
    const std::vector<int> prompt{101, removed, 3000};
    auto p = get_parser_engine("gemma4", false, &tok);
    Fields f;
    f.add(p->parse_delta("Sure", {}, {}, true, &prompt));
    const bool opens = removed != 101;
    CHECK(f.reasoning == (opens ? "Sure" : ""));
    CHECK(f.content == (opens ? "" : "Sure"));
  }
  auto p = get_parser_engine("gemma4", true, nullptr);
  Fields f;
  f.add(p->parse_delta("Sure", {}, {}, true, &open));
  CHECK(f.reasoning.empty());
  CHECK(f.content == "Sure");
}

TEST_CASE("Gemma absent versus empty preparation and finish lifecycle") {
  PromptTokenizer tok;
  const std::vector<int> empty;
  for (bool present : {false, true}) {
    auto p = get_parser_engine("gemma4", false, &tok);
    p->parse_delta("", {}, {}, false, present ? &empty : nullptr);
    Fields f;
    f.add(p->parse_delta("Sure", {}, {}, false, &open));
    CHECK(f.reasoning == (present ? "" : "Sure"));
    CHECK(f.content == (present ? "Sure" : ""));
  }
  auto p = get_parser_engine("gemma4", false, &tok);
  Fields f;
  f.add(p->parse_delta("Sure", {}, {}, false, &open));
  f.add(p->parse_delta("<channel|>Hello", {}, {}, false, &open));
  f.add(p->finish_streaming());
  f.add(p->parse_delta(" again", {}, {}, true, &open));
  CHECK(f.reasoning == "Sure");
  CHECK(f.content == "Hello again");
}

TEST_CASE("Gemma direct prepare survives initialize without clearing prefix bookkeeping") {
  PromptTokenizer tok;
  auto p = get_parser_engine("gemma4", false, &tok);
  p->adjust_initial_state_from_prompt(open);
  p->initialize_streaming();
  Fields f;
  f.add(p->parse_delta("Sure", {}, {}, false));
  CHECK(f.reasoning == "Sure");
  CHECK(f.content.empty());
  // A second direct preparation resets only the underlying engine. Once the
  // prefix was consumed, literal thought must not be stripped a second time.
  p->adjust_initial_state_from_prompt(open);
  f.add(p->parse_delta("thought\nbody<channel|>Hello", {}, {}, true));
  CHECK(f.reasoning == "Surethought\nbody");
  CHECK(f.content == "Hello");
  // extract_reasoning performs the real virtual reset, clearing prefix and
  // prompt-prepared bookkeeping for a subsequent request.
  p->extract_reasoning("plain", {});
  Fields next;
  next.add(p->parse_delta("thought\nNew", {}, {}, false, &open));
  next.add(p->parse_delta("<channel|>End", {}, {}, true, &open));
  CHECK(next.reasoning == "New");
  CHECK(next.content == "End");
}

namespace {
class ResetHook : public ParserEngine {
 public:
  ResetHook() : ParserEngine(gemma4_config()) {}
  int calls = 0;
  void adjust_initial_state_from_prompt(const std::vector<int>&) override {
    ++calls;
    reset(ParserState::REASONING);
  }
  void restart() { reset(std::nullopt); }
};
}
TEST_CASE("Base prompt hook marks after reset and only once until reset") {
  ResetHook p;
  const std::vector<int> empty;
  p.parse_delta("", {}, {}, false);
  p.parse_delta("", {}, {}, false, nullptr);
  CHECK(p.calls == 0);
  p.parse_delta("Sure", {}, {}, false, &empty);
  CHECK(p.calls == 1);
  p.parse_delta("<channel|>Hello", {}, {}, false, &open);
  p.finish_streaming();
  p.parse_delta("again", {}, {}, true, &open);
  CHECK(p.calls == 1);
  p.restart();
  p.parse_delta("Sure", {}, {}, true, &open);
  CHECK(p.calls == 2);
}
