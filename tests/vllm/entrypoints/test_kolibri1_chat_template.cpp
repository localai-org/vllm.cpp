// Gate for the Kolibri-1 chat template as served. The template is NOT code
// this repo ships: it rides in the checkpoint's tokenizer_config.json
// (chat_template key) and reaches the renderer through
// LoadChatTemplateFromConfig -> apply_chat_template (the vendored minja
// engine). The serving oracle is the model-author plugin
// (aleph-alpha-inference @ 049a6a7bd240): it ships no template of its own —
// its reasoning.py:3-26 documents the exact switch the checkpoint template
// encodes (reasoning_effort "none" disables thinking; else a literal
// enable_thinking false does; thinking on stops the generation prompt at
// `<|im_start|>assistant\n`, thinking off renders the closed
// `<think>\n\n</think>\n\n` block). The reference outputs are CPython jinja2
// renderings of the SAME template text under transformers' whitespace policy,
// captured by tests/fixtures/gen-kolibri1-chat-template-references.py (see
// the fixture's provenance field).
//
// What this gate proves: minja renders the checkpoint template byte-identical
// to the reference for every scenario the plugin's switch distinguishes, and
// the detection seams pick the kolibri1 parsers off that template text.
#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

using namespace vllm::entrypoints;
using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::ChatCompletionToolsParam;

namespace {

const nlohmann::ordered_json& References() {
  static const nlohmann::ordered_json refs = [] {
    const char* dir = std::getenv("KOLIBRI1_TEMPLATE_FIXTURE_DIR");
#ifdef KOLIBRI1_TEMPLATE_FIXTURE_DIR
    if (dir == nullptr) dir = KOLIBRI1_TEMPLATE_FIXTURE_DIR;
#endif
    REQUIRE(dir != nullptr);
    std::ifstream in(std::string(dir) +
                     "/kolibri1_chat_template_references.json");
    REQUIRE(in.good());
    return nlohmann::ordered_json::parse(in);
  }();
  return refs;
}

std::vector<openai::ChatMessage> ToMessages(const nlohmann::json& arr) {
  std::vector<openai::ChatMessage> out;
  for (const auto& m : arr) out.push_back(m.get<openai::ChatMessage>());
  return out;
}

std::vector<ChatCompletionToolsParam> ToTools(const nlohmann::json& tools) {
  std::vector<ChatCompletionToolsParam> out;
  for (const auto& t : tools) {
    out.push_back(t.get<ChatCompletionToolsParam>());
  }
  return out;
}

// The kolibri1 template tell-tale: the "Reasoning is disabled" sentence the
// template embeds as its no-reasoning system sentence. No other template in
// the marker tables contains it. Mirrors how the plugin selects its own
// parsers: by the kolibri1 registration name, which detection must resolve to.
const char* kKolibriMarker =
    "Reasoning is disabled. Proceed straight to answering";

}  // namespace

TEST_CASE("kolibri1: minja renders the checkpoint template like jinja2") {
  for (const auto& c : References().at("cases")) {
    CAPTURE(c.at("name").get<std::string>());
    const std::string rendered = apply_chat_template(
        LoadChatTemplateFromConfig(
            "/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json"),
        ToMessages(c.at("messages")),
        c.at("add_generation_prompt").get<bool>(), "", "",
        c.contains("tools") && !c.at("tools").is_null()
            ? ToTools(c.at("tools"))
            : std::vector<ChatCompletionToolsParam>{},
        c.at("chat_template_kwargs"));
    CHECK(rendered == c.at("expected").get<std::string>());
  }
}

TEST_CASE("kolibri1: the template's thinking switch matches the plugin's") {
  // The closed-block vs open-prompt boundary is the contract between the
  // template and the kolibri1 reasoning parser (reasoning.py:3-26). Asserted
  // on the reference strings so the parser switch cannot drift from what the
  // template actually renders.
  const auto& cases = References().at("cases");
  auto find = [&](const std::string& name) {
    for (const auto& c : cases) {
      if (c.at("name") == name) return c;
    }
    throw std::runtime_error("missing case " + name);
  };
  const std::string kClosed = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  for (const char* off :
       {"enable_thinking_false", "reasoning_effort_none",
        "effort_none_overrides_enable_thinking_default"}) {
    CAPTURE(off);
    CHECK(find(off).at("expected").get<std::string>().find(kClosed) !=
          std::string::npos);
  }
  for (const char* on :
       {"default_user_only", "reasoning_effort_low", "reasoning_effort_medium",
        "reasoning_effort_high", "effort_low_overrides_enable_thinking_false"}) {
    CAPTURE(on);
    CHECK(find(on).at("expected").get<std::string>().find(kClosed) ==
          std::string::npos);
    CHECK(find(on).at("expected").get<std::string>().ends_with(
        "<|im_start|>assistant\n"));
  }
}

TEST_CASE("kolibri1: detection rows select kolibri1 off the template text") {
  const std::string template_str = LoadChatTemplateFromConfig(
      "/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json");
  REQUIRE(template_str.find(kKolibriMarker) != std::string::npos);
  CHECK(DetectReasoningParser(template_str) == "kolibri1");
  CHECK(DetectToolParser(template_str) == "kolibri1");
}

TEST_CASE("kolibri1: the marker row precedes and does not shadow <think>") {
  // The template contains "<think>" too; the kolibri1 row must win by ORDER,
  // and the marker must not appear inside any other family's template tell.
  CHECK(std::string(kKolibriMarker).find("<think>") == std::string::npos);
  CHECK(DetectReasoningParser(std::string("..<think>..")) == "think_auto");
}
