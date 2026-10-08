// Reasoning-parser auto-detection (ABI v5): the template marker table and its
// safe empty (disabled) default.
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"

#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

using vllm::entrypoints::openai::DetectReasoningParser;
using vllm::entrypoints::openai::get_reasoning_parser;
using vllm::entrypoints::openai::ResolveReasoningParserName;
using vllm::entrypoints::openai::ReasoningParserMarker;
using vllm::entrypoints::openai::ReasoningParserMarkerTable;

TEST_CASE("reasoning detect: [THINK] selects mistral, <think> selects deepseek_r1") {
  CHECK(DetectReasoningParser("..[THINK]..") == "mistral");
  CHECK(DetectReasoningParser("..<think>..") == "think_auto");
}

TEST_CASE("reasoning detect: [THINK] wins over <think> when both appear") {
  CHECK(DetectReasoningParser("[THINK] and <think>") == "mistral");
}

TEST_CASE("reasoning detect: no marker means DISABLED, not a fallback parser") {
  CHECK(DetectReasoningParser("no reasoning markers here").empty());
}

TEST_CASE("reasoning detect: every built-in row names a registered parser") {
  std::size_t n = 0;
  const ReasoningParserMarker* table = ReasoningParserMarkerTable(&n);
  REQUIRE(table != nullptr);
  REQUIRE(n >= 2);
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    REQUIRE(table[i].parser != nullptr);
    REQUIRE(table[i].template_marker != nullptr);
    CHECK(get_reasoning_parser(table[i].parser) != nullptr);
  }
}

// ---------------------------------------------------------------------------
// ResolveReasoningParserName — the behaviour of the server's --reasoning-parser
// flag (examples/server/main.cpp), the sibling of ResolveToolParserName.
// ---------------------------------------------------------------------------

TEST_CASE("--reasoning-parser: the DEFAULT is byte-identical to the old hardcode") {
  // Before the flag existed the server passed reasoning_parser_name="" — i.e.
  // reasoning extraction DISABLED. The flag defaults to "none", which resolves
  // to that same empty name, for ANY chat template. A <think> template that
  // "auto" would light up stays off unless the operator asks.
  CHECK(ResolveReasoningParserName("none", "..<think>..").empty());
  CHECK(ResolveReasoningParserName("none", "").empty());
  CHECK(ResolveReasoningParserName("", "..[THINK]..").empty());
  // Proof the templates above WOULD have detected something under "auto".
  CHECK(DetectReasoningParser("..<think>..") == "think_auto");
}

TEST_CASE("--reasoning-parser: explicit names and auto-detection") {
  CHECK(ResolveReasoningParserName("deepseek_r1", "") == "deepseek_r1");
  CHECK(ResolveReasoningParserName("olmo3", "") == "olmo3");
  CHECK(ResolveReasoningParserName("minimax_m2_append_think", "") ==
        "minimax_m2_append_think");
  CHECK(ResolveReasoningParserName("auto", "..<think>..") == "think_auto");
  CHECK(ResolveReasoningParserName("auto", "..[THINK]..") == "mistral");
  // "auto" with no matching marker legitimately stays DISABLED (unlike tool
  // detection, a reasoning parser actively splits text, so silence is safest).
  CHECK(ResolveReasoningParserName("auto", "plain template").empty());
}

TEST_CASE("--reasoning-parser: an unknown name FAILS LOUDLY, listing the registry") {
  CHECK_THROWS_AS(ResolveReasoningParserName("deepseek-r1", ""),
                  std::invalid_argument);
  CHECK_THROWS_AS(ResolveReasoningParserName("think", ""),
                  std::invalid_argument);
  std::string what;
  try {
    ResolveReasoningParserName("nope", "");
  } catch (const std::invalid_argument& e) {
    what = e.what();
  }
  CHECK(what.find("nope") != std::string::npos);
  for (const std::string& name :
       vllm::entrypoints::openai::reasoning_parser_names()) {
    CAPTURE(name);
    CHECK(what.find(name) != std::string::npos);
  }
}

TEST_CASE("Registry: every enumerated reasoning-parser name resolves") {
  const std::vector<std::string>& names =
      vllm::entrypoints::openai::reasoning_parser_names();
  for (const std::string& name : names) {
    CAPTURE(name);
    CHECK(get_reasoning_parser(name) != nullptr);
    CHECK(ResolveReasoningParserName(name, "") == name);
  }
  // Pinned count: adding a factory branch without listing it fails here.
  // 2026-07-28 (CLAIM-SAMPLE-REASONING): 7 -> 9, adding the deepseek_v3 +
  // holo2 thinking-gated delegates (see specs/reasoning-parsers.md W1).
  // 2026-08-10 (MODEL-MUSE-GLIMMER-W7): 9 -> 10, adding "muse_glimmer".
  // 2026-08-13 (SAMPLE-REASONING W3, #605): 10 -> 12, adding the engine-backed
  // "qwen3" + its "mimo" alias (ONE class, two upstream registry names).
  // 2026-10-08 (MODEL-TEXT-kolibri-1 serving completion): 12 -> 13, adding
  // "kolibri1" (the model-author plugin's reasoning parser name).
  CHECK(names.size() == 13);
  std::size_t marker_count = 0;
  const ReasoningParserMarker* markers = ReasoningParserMarkerTable(&marker_count);
  for (std::size_t i = 0; i < marker_count; ++i) {
    CAPTURE(markers[i].parser);
    CHECK(get_reasoning_parser(markers[i].parser) != nullptr);
  }
}
