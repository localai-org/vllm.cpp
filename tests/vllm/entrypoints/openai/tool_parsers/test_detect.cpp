// Unit tests for the ABI v4 tool-parser AUTO-detector
// (src/vllm/entrypoints/openai/tool_parsers/detect.*): marker-hit selection, the
// no-marker fallback, first-match ordering semantics (proven with a synthetic
// table since only one family is registered today), and the built-in table's
// invariants. Pure string-level tests — no engine, no disk model.
#include "vllm/entrypoints/openai/tool_parsers/detect.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

using vllm::entrypoints::openai::DetectToolParser;
using vllm::entrypoints::openai::get_tool_parser;
using vllm::entrypoints::openai::ResolveToolParserName;
using vllm::entrypoints::openai::ToolParserMarker;
using vllm::entrypoints::openai::ToolParserMarkerTable;

TEST_CASE("detect: the hermes <tool_call> marker selects hermes") {
  // A minimal template that wraps a call in the Hermes/Qwen surface.
  const std::string tmpl =
      "{% for m in messages %}{{ m.role }}{% endfor %}"
      "<tool_call>{\"name\": \"f\"}</tool_call>";
  CHECK(DetectToolParser(tmpl) == "hermes");
}

TEST_CASE("detect: a template with no known marker falls back to hermes") {
  const std::string tmpl =
      "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}";
  CHECK(DetectToolParser(tmpl) == "hermes");
  // The empty template is the "no template resolved" shape.
  CHECK(DetectToolParser("") == "hermes");
}

TEST_CASE("detect: first matching row wins (ordering semantics)") {
  // A synthetic two-row table: the SPECIFIC marker precedes the GENERIC one it
  // is a superset of. DetectToolParser must return the first row that matches,
  // so a template carrying BOTH markers resolves to the specific parser.
  const ToolParserMarker table[] = {
      {"specific", "<specific_call>"},
      {"generic", "<call>"},
  };
  const std::size_t n = sizeof(table) / sizeof(table[0]);

  // Contains both markers -> the earlier (specific) row wins.
  CHECK(DetectToolParser("prefix <specific_call> and <call> suffix", table, n) ==
        "specific");
  // Only the generic marker -> the second row.
  CHECK(DetectToolParser("only <call> here", table, n) == "generic");
  // Neither marker -> the hermes fallback (not a row in this table).
  CHECK(DetectToolParser("nothing to see", table, n) == "hermes");
}

TEST_CASE("detect: the built-in marker table holds its invariants") {
  std::size_t n = 0;
  const ToolParserMarker* table = ToolParserMarkerTable(&n);
  REQUIRE(table != nullptr);
  REQUIRE(n >= 1);  // at least the hermes row.
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    // Every row names a NON-EMPTY parser + marker...
    REQUIRE(table[i].parser != nullptr);
    REQUIRE(table[i].template_marker != nullptr);
    CHECK(std::string(table[i].parser).size() > 0);
    CHECK(std::string(table[i].template_marker).size() > 0);
    // ...and every named parser is actually REGISTERED, so detection can never
    // hand EnsureChatServing a name get_tool_parser rejects.
    CHECK(get_tool_parser(table[i].parser) != nullptr);
  }
}

TEST_CASE("detect: null count pointer is tolerated") {
  CHECK(ToolParserMarkerTable(nullptr) != nullptr);
}

TEST_CASE("detect: every ported family resolves from its template marker") {
  CHECK(DetectToolParser("...<longcat_tool_call>...") == "longcat");
  CHECK(DetectToolParser("...<｜tool▁calls▁begin｜>...") == "deepseek_v3");
  CHECK(DetectToolParser("...[TOOL_CALLS]...") == "mistral");
  CHECK(DetectToolParser("...<function_call>...") == "granite-20b-fc");
  CHECK(DetectToolParser("...<|tool_call|>...") == "granite");
  CHECK(DetectToolParser("...<|python_start|>...") == "llama4_pythonic");
  CHECK(DetectToolParser("...<|python_tag|>...") == "llama3_json");
  CHECK(DetectToolParser("...<tool_call>...") == "hermes");
}

TEST_CASE("detect: specific markers beat the generic hermes tail row") {
  // A template that carries BOTH a family marker and a plain <tool_call>
  // must resolve to the specific family (first-match-wins ordering).
  CHECK(DetectToolParser("<longcat_tool_call> and <tool_call>") == "longcat");
  CHECK(DetectToolParser("[TOOL_CALLS] then <tool_call>") == "mistral");
}

TEST_CASE("detect: wave B2 families resolve from their template markers") {
  CHECK(DetectToolParser("..<｜DSML｜function_calls>..") == "deepseek_v32");
  CHECK(DetectToolParser("..<｜DSML｜tool_calls>..") == "deepseek_v4");
  CHECK(DetectToolParser("..<｜tool_calls_begin｜>..") == "step3");
  CHECK(DetectToolParser("..<function name=\"x\">..") == "minicpm5");
  CHECK(DetectToolParser("..<function_calls>[f(x=1)]..") == "olmo3");
  CHECK(DetectToolParser("..<|action_start|><|plugin|>..") == "internlm");
  CHECK(DetectToolParser("..functools[..") == "phi4_mini_json");
  CHECK(DetectToolParser("..<tool_calls>[..") == "jamba");
  CHECK(DetectToolParser("..<tool_callsSFX>..") == "hy_v3");
  CHECK(DetectToolParser("..<tool_call><function=f>..") == "step3p5");
}

TEST_CASE("detect: B2 ordering - exact jamba beats the hy_v3 prefix, step3p5 beats hermes") {
  // jamba's exact "<tool_calls>" row precedes hy_v3's prefix probe.
  CHECK(DetectToolParser("<tool_calls>") == "jamba");
  // A suffixed hy_v3 tag misses jamba's exact literal and lands on the prefix.
  CHECK(DetectToolParser("<tool_calls_v1>") == "hy_v3");
  // step3p5's inner "<function=" wins over the generic hermes wrapper.
  CHECK(DetectToolParser("<tool_call><function=get>") == "step3p5");
  // A plain hermes template still falls through to hermes.
  CHECK(DetectToolParser("<tool_call>{\"name\"") == "hermes");
}

TEST_CASE("detect: wave B4 families resolve from their template markers") {
  CHECK(DetectToolParser("..<|tool_call_start|>..") == "lfm2");
  CHECK(DetectToolParser("..<start_function_call>..") == "functiongemma");
  CHECK(DetectToolParser("..<|tools_prefix|>..") == "apertus");
  CHECK(DetectToolParser("..<|function_call|>..") == "gigachat3");
  // gigachat's piped literal must not shadow granite-20b-fc's plain one.
  CHECK(DetectToolParser("..<function_call>..") == "granite-20b-fc");
}

TEST_CASE("detect: ENG-wave families resolve from their template markers") {
  CHECK(DetectToolParser("..<|tool_calls_section_begin|>..") == "kimi_k2");
  CHECK(DetectToolParser("..<minimax:tool_call>..") == "minimax_m2");
  CHECK(DetectToolParser("..<|tool_call>call:get..") == "gemma4");
  CHECK(DetectToolParser("..<seed:tool_call>..<function=f>..") == "seed_oss");
  // seed_oss's wrapper must beat step3p5's inner "<function=" row.
  CHECK(DetectToolParser("<seed:tool_call><function=f>") != "step3p5");
  // gemma4 and lfm2 share a "<|tool_call" prefix but stay distinct.
  CHECK(DetectToolParser("..<|tool_call_start|>..") == "lfm2");
}

// ---------------------------------------------------------------------------
// ResolveToolParserName — the behaviour of the server's --tool-call-parser flag
// (examples/server/main.cpp). The flag's whole semantics live in this function
// so they are testable without launching a server.
// ---------------------------------------------------------------------------

TEST_CASE("--tool-call-parser: the DEFAULT is byte-identical to the old hardcode") {
  // Before the flag existed, examples/server/main.cpp constructed
  // OpenAIServingChat with the literal "hermes". The flag's default value is
  // that same literal, so an invocation that does not name the flag resolves to
  // exactly "hermes" REGARDLESS of the model's chat template — no detection, no
  // behaviour change for any existing deployment.
  const std::string deepseek_tmpl = "..<｜tool▁calls▁begin｜>..";
  CHECK(ResolveToolParserName("hermes", deepseek_tmpl) == "hermes");
  CHECK(ResolveToolParserName("hermes", "") == "hermes");
  // Proof the template above WOULD have detected as something else: this is
  // what "auto" opts into, and what the default deliberately does not do.
  CHECK(DetectToolParser(deepseek_tmpl) == "deepseek_v3");
}

TEST_CASE("--tool-call-parser: an explicit name selects that dialect") {
  CHECK(ResolveToolParserName("qwen3_coder", "") == "qwen3_coder");
  CHECK(ResolveToolParserName("mistral", "") == "mistral");
  CHECK(ResolveToolParserName("glm47", "") == "glm47");
  // Aliases resolve to themselves (the factory maps several names onto one impl).
  CHECK(ResolveToolParserName("mimo", "") == "mimo");
  CHECK(ResolveToolParserName("llama4_json", "") == "llama4_json");
}

TEST_CASE("--tool-call-parser: auto detects, none disables") {
  CHECK(ResolveToolParserName("auto", "..<minimax:tool_call>..") == "minimax_m2");
  CHECK(ResolveToolParserName("auto", "..<tool_call>..") == "hermes");
  // No template to sniff: detection falls back to hermes, never to nothing.
  CHECK(ResolveToolParserName("auto", "") == "hermes");
  // "none" and "" both mean DISABLED (empty name => MakeToolParser returns null).
  CHECK(ResolveToolParserName("none", "..<tool_call>..").empty());
  CHECK(ResolveToolParserName("", "..<tool_call>..").empty());
}

TEST_CASE("--tool-call-parser: an unknown name FAILS LOUDLY, listing the registry") {
  // The failure must be an exception at startup, not a silent fallback: a
  // server that quietly disabled tool parsing for a typo'd dialect would look
  // healthy while emitting raw tool-call text to every client.
  CHECK_THROWS_AS(ResolveToolParserName("hermez", ""), std::invalid_argument);
  CHECK_THROWS_AS(ResolveToolParserName("Hermes", ""), std::invalid_argument);
  CHECK_THROWS_AS(ResolveToolParserName("qwen3-coder", ""), std::invalid_argument);

  // The message enumerates the registry (so a user can fix the typo from it),
  // and it is enumerated FROM tool_parser_names(), not hand-written here.
  std::string what;
  try {
    ResolveToolParserName("nope", "");
  } catch (const std::invalid_argument& e) {
    what = e.what();
  }
  CHECK(what.find("nope") != std::string::npos);
  for (const std::string& name : vllm::entrypoints::openai::tool_parser_names()) {
    CAPTURE(name);
    CHECK(what.find(name) != std::string::npos);
  }
  CHECK(what.find("auto") != std::string::npos);
  CHECK(what.find("none") != std::string::npos);
}

TEST_CASE("Registry: every enumerated tool-parser name resolves") {
  // The list backing the flag's error message must not drift from the factory.
  // Every enumerated name builds a real parser...
  const std::vector<std::string>& names =
      vllm::entrypoints::openai::tool_parser_names();
  for (const std::string& name : names) {
    CAPTURE(name);
    CHECK(get_tool_parser(name) != nullptr);
    CHECK(ResolveToolParserName(name, "") == name);
  }
  // ...and the count is PINNED, so adding a factory branch without listing its
  // name here fails this case instead of silently shipping an unreachable
  // dialect. 40 accepted names over 36 parser families (aliases:
  // llama3_json/llama4_json, qwen3_coder/qwen3_xml/mimo, glm45/glm47).
  // 2026-08-10 (MODEL-MUSE-GLIMMER-W7): 40 -> 41, adding "muse_glimmer".
  // 2026-08-13 (TOOLS-PARSER-BREADTH W1, #608): 41 -> 42, adding "inkling".
  // 2026-10-08 (MODEL-TEXT-kolibri-1 serving completion): 42 -> 43, adding
  // "kolibri1" — an ALIAS to the hermes family, per the model-author plugin's
  // registration (aleph-alpha-inference __init__.py:50-54).
  CHECK(names.size() == 43);
  // Every name the marker table can emit must itself be a registered name.
  std::size_t marker_count = 0;
  const ToolParserMarker* markers = ToolParserMarkerTable(&marker_count);
  for (std::size_t i = 0; i < marker_count; ++i) {
    CAPTURE(markers[i].parser);
    CHECK(get_tool_parser(markers[i].parser) != nullptr);
  }
}
