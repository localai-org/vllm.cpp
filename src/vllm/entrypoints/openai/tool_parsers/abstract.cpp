// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v31.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v32.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v4.h"
#include "vllm/entrypoints/openai/tool_parsers/functiongemma.h"
#include "vllm/entrypoints/openai/tool_parsers/glm47.h"
#include "vllm/entrypoints/openai/tool_parsers/gemma4.h"
#include "vllm/entrypoints/openai/tool_parsers/granite.h"
#include "vllm/entrypoints/openai/tool_parsers/granite4.h"
#include "vllm/entrypoints/openai/tool_parsers/granite_20b_fc.h"
#include "vllm/entrypoints/openai/tool_parsers/apertus.h"
#include "vllm/entrypoints/openai/tool_parsers/ernie45.h"
#include "vllm/entrypoints/openai/tool_parsers/gigachat3.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.h"
#include "vllm/entrypoints/openai/tool_parsers/internlm.h"
#include "vllm/entrypoints/openai/tool_parsers/jamba.h"
#include "vllm/entrypoints/openai/tool_parsers/kimi_k2.h"
#include "vllm/entrypoints/openai/tool_parsers/hy_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/lfm2.h"
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"
#include "vllm/entrypoints/openai/tool_parsers/minicpm5.h"
#include "vllm/entrypoints/openai/tool_parsers/minimax_m2.h"
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"
#include "vllm/entrypoints/openai/tool_parsers/muse_glimmer.h"
#include "vllm/entrypoints/openai/tool_parsers/phi4_mini.h"
#include "vllm/entrypoints/openai/tool_parsers/poolside_v1.h"
#include "vllm/entrypoints/openai/tool_parsers/xlam.h"
#include "vllm/entrypoints/openai/tool_parsers/llama.h"
#include "vllm/entrypoints/openai/tool_parsers/llama4_pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/olmo3.h"
#include "vllm/entrypoints/openai/tool_parsers/parser_engine_adapter.h"
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3_coder.h"
#include "vllm/entrypoints/openai/tool_parsers/seed_oss.h"
#include "vllm/entrypoints/openai/tool_parsers/step3.h"
#include "vllm/entrypoints/openai/tool_parsers/step3p5.h"

namespace vllm::entrypoints::openai {

// abstract_tool_parser.py:218 — the base raises NotImplementedError; derived
// streaming-capable parsers (Hermes) override this.
std::optional<DeltaMessage> ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& /*delta_text*/, const ChatCompletionRequest& /*request*/) {
  throw std::runtime_error(
      "ToolParser::extract_tool_calls_streaming has not been implemented!");
}

std::string make_tool_call_id() {
  // Upstream: f"chatcmpl-tool-{random_uuid()}" with random_uuid() == the hex of
  // a uuid4 (32 hex chars). We emit 32 random hex chars (uniqueness only).
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  std::ostringstream oss;
  oss << "chatcmpl-tool-" << std::hex << std::setfill('0') << std::setw(16)
      << hi << std::setw(16) << lo;
  return oss.str();
}

std::unique_ptr<ToolParser> get_tool_parser(const std::string& name) {
  // Ported from ToolParserManager.get_tool_parser (abstract_tool_parser.py:235).
  // Registry: "hermes" -> Hermes2ProToolParser; "qwen3" -> the gate model's
  // Hermes-format parser; "deepseek_v3"/"deepseek_v31" -> the DeepSeek marker
  // formats; "longcat" -> the Hermes subclass with the <longcat_tool_call> tag.
  // Unknown name -> nullptr (upstream raises KeyError).
  //
  // DETECTION MARKERS (for any auto-detect layer built on top of this factory;
  // upstream vLLM selects by explicit --tool-call-parser NAME, so ordering does
  // not affect this factory itself):
  //   - DeepSeek: the OUTER begin marker <｜tool▁calls▁begin｜> (fullwidth ｜ +
  //     U+2581) is unique and unambiguous — probe it directly.
  //   - longcat: probe <longcat_tool_call>. VERIFIED against the upstream chat
  //     template (huggingface.co/meituan-longcat/LongCat-Flash-Chat
  //     tokenizer_config.json): LongCat wraps tool calls EXCLUSIVELY in
  //     <longcat_tool_call>…</longcat_tool_call> and never emits the bare Hermes
  //     <tool_call>. Since "<tool_call>" is NOT a substring of
  //     "<longcat_tool_call>", a hermes-first check would NOT false-positive on
  //     standard longcat output; still, a loose/substring auto-detector should
  //     prefer the more specific <longcat_tool_call> marker before <tool_call>.
  // Hermes-format parser; "llama3_json"/"llama4_json" -> LlamaToolParser;
  // "pythonic" -> PythonicToolParser; "llama4_pythonic" ->
  // Llama4PythonicToolParser. Unknown name -> nullptr (upstream raises KeyError).
  if (name == "hermes") {
    return std::make_unique<HermesToolParser>();
  }
  // kolibri1 IS the Hermes class upstream: the plugin registers the kolibri1
  // name over vllm.tool_parsers.hermes_tool_parser.Hermes2ProToolParser
  // verbatim (aleph-alpha-inference @ 049a6a7bd240 __init__.py:50-54, "shares
  // the Hermes <tool_call>...</tool_call> format"), so this is an alias to
  // the same implementation, not a parallel dialect.
  if (name == "kolibri1") {
    return std::make_unique<HermesToolParser>();
  }
  if (name == "qwen3") {
    return std::make_unique<Qwen3ToolParser>();
  }
  if (name == "deepseek_v3") {
    return std::make_unique<DeepSeekV3ToolParser>();
  }
  if (name == "deepseek_v31") {
    return std::make_unique<DeepSeekV31ToolParser>();
  }
  // deepseek_v32 (DSML <｜DSML｜function_calls>…) + deepseek_v4 (the same DSML
  // grammar wrapped in <｜DSML｜tool_calls>). deepseek_v4 subclasses v32.
  if (name == "deepseek_v32") {
    return std::make_unique<DeepSeekV32ToolParser>();
  }
  if (name == "deepseek_v4") {
    return std::make_unique<DeepSeekV4ToolParser>();
  }
  if (name == "longcat") {
    return std::make_unique<LongcatToolParser>();
  }
  // "mistral" -> the modern v11 name-first form (mistral_tool_parser.py:96).
  // The pre-v11 bracketed-array form is selected via MistralToolParser(true).
  if (name == "mistral") {
    return std::make_unique<MistralToolParser>();
  }
  // Granite family (granite_tool_parser.py / granite4_tool_parser.py /
  // granite_20b_fc_tool_parser.py @ e24d1b24).
  if (name == "granite") {
    return std::make_unique<GraniteToolParser>();
  }
  if (name == "granite4") {
    return std::make_unique<Granite4ToolParser>();
  }
  if (name == "granite-20b-fc") {
    return std::make_unique<Granite20bFCToolParser>();
  }
  // llama_tool_parser.py registers Llama3JsonToolParser under BOTH names
  // (llama3_json + llama4_json); they share one implementation.
  if (name == "llama3_json" || name == "llama4_json") {
    return std::make_unique<LlamaToolParser>();
  }
  if (name == "pythonic") {
    return std::make_unique<PythonicToolParser>();
  }
  if (name == "llama4_pythonic") {
    return std::make_unique<Llama4PythonicToolParser>();
  }
  // xlam_tool_parser.py (name "xlam") - fenced / [TOOL_CALLS] / <tool_call> /
  // bare-list envelopes around a JSON tool-call array.
  if (name == "xlam") {
    return std::make_unique<xLAMToolParser>();
  }
  // phi4mini_tool_parser.py (name "phi4_mini_json") - `functools[...]`.
  if (name == "phi4_mini_json") {
    return std::make_unique<Phi4MiniJsonToolParser>();
  }
  // internlm2_tool_parser.py (name "internlm") - <|action_start|><|plugin|>{}<|action_end|>.
  if (name == "internlm") {
    return std::make_unique<Internlm2ToolParser>();
  }
  // jamba_tool_parser.py (name "jamba") - <tool_calls>[...]</tool_calls>.
  if (name == "jamba") {
    return std::make_unique<JambaToolParser>();
  }
  // StepFun step3 (opaque fullwidth-｜ markers) + step3.5 (XML-parametrized
  // <tool_call><function=…><parameter=…> streaming state machine).
  if (name == "step3") {
    return std::make_unique<Step3ToolParser>();
  }
  if (name == "step3p5") {
    return std::make_unique<Step3p5ToolParser>();
  }
  // Qwen3-Coder XML surface (qwen3_engine_tool_parser.py registers "qwen3_coder",
  // "qwen3_xml" and "mimo" onto the same Qwen3EngineToolParser); reimplemented
  // from the qwen3.py wire format as a text parser (qwen3_coder.h). NOTE: the
  // existing "qwen3" name above is the Hermes-JSON Qwen variant, a DIFFERENT
  // dialect - these three names are the XML `<function=…><parameter=…>` surface.
  if (name == "qwen3_coder" || name == "qwen3_xml" || name == "mimo") {
    return std::make_unique<Qwen3CoderToolParser>();
  }
  // olmo3_tool_parser.py: the pythonic-inside-<function_calls> wrapper format.
  if (name == "olmo3") {
    return std::make_unique<Olmo3ToolParser>();
  }
  // minicpm5xml_tool_parser.py: the <function name="x"><param ...> XML dialect.
  if (name == "minicpm5") {
    return std::make_unique<MiniCPM5ToolParser>();
  }
  // hy_v3_tool_parser.py: the suffix-parametrized <tool_calls..><tool_call..>
  // <tool_sep..><arg_key..><arg_value..> XML dialect (default empty suffix).
  if (name == "hy_v3") {
    return std::make_unique<HYV3ToolParser>();
  }
  // hunyuan_a13b_tool_parser.py: <tool_calls>[{"name":..,"arguments":..}]
  // </tool_calls> (a JSON array; <think>-guarded; bot_string + consume_space).
  if (name == "hunyuan_a13b") {
    return std::make_unique<HunyuanA13BToolParser>();
  }
  // apertus_tool_parser.py: <|tools_prefix|>[{name:{args}}]<|tools_suffix|>.
  if (name == "apertus") {
    return std::make_unique<ApertusToolParser>();
  }
  // ernie45_tool_parser.py: <tool_call>{json}</tool_call> (thinking/response
  // interplay; upstream vocab-aware, reworked to text - see ernie45.h).
  if (name == "ernie45") {
    return std::make_unique<Ernie45ToolParser>();
  }
  // gigachat3_tool_parser.py: "function call<|role_sep|>\n{json}" or
  // "<|function_call|>{json}"; generation may end with "</s>".
  if (name == "gigachat3") {
    return std::make_unique<GigaChat3ToolParser>();
  }
  // lfm2_tool_parser.py (name "lfm2") - pythonic `[func(arg=val), ...]` list
  // wrapped in <|tool_call_start|>...<|tool_call_end|> sentinels.
  if (name == "lfm2") {
    return std::make_unique<Lfm2ToolParser>();
  }
  // poolside_v1_tool_parser.py (name "poolside_v1") - GLM-4-derived bare
  // <tool_call> with a NAME + <arg_key>/<arg_value> XML body.
  if (name == "poolside_v1") {
    return std::make_unique<PoolsideV1ToolParser>();
  }
  // functiongemma_tool_parser.py (name "functiongemma") - Google FunctionGemma
  // <start_function_call>call:NAME{KEY:<escape>VALUE<escape>}<end_function_call>.
  if (name == "functiongemma") {
    return std::make_unique<FunctionGemmaToolParser>();
  }
  // kimi_k2.py (name "kimi_k2") - <|tool_calls_section_begin|><|tool_call_begin|>
  // functions.NAME:IDX <|tool_call_argument_begin|>{args}<|tool_call_end|>
  // <|tool_calls_section_end|>; the native functions.NAME:IDX header IS the
  // ToolCall id (reimplemented from the wire format - see kimi_k2.h).
  if (name == "kimi_k2") {
    return std::make_unique<KimiK2ToolParser>();
  }
  // glm47_moe_tool_parser.py registers Glm47MoeModelToolParser under BOTH "glm45"
  // and "glm47" (test_glm4_moe_tool_parser asserts both resolve to it). The GLM
  // XML <tool_call>NAME<arg_key>..<arg_value>..</tool_call> wire surface,
  // reimplemented from the token-id ParserEngine (see glm47.h).
  if (name == "glm45" || name == "glm47") {
    return std::make_unique<Glm47ToolParser>();
  }
  // minimax_m2_tool_parser.py (name "minimax_m2") - the <minimax:tool_call> XML
  // wrapper with <invoke name=..> blocks + <parameter name=..> args (minimax_m2.h).
  if (name == "minimax_m2") {
    return std::make_unique<MinimaxM2ToolParser>();
  }
  // gemma4.py + gemma4_engine_tool_parser.py (name "gemma4") - Google Gemma4
  // <|tool_call>call:NAME{KEY:<|"|>VALUE<|"|>,..}<tool_call|> (the native
  // key:value dialect, reimplemented from the wire format; see gemma4.h).
  if (name == "gemma4") {
    return std::make_unique<Gemma4ToolParser>();
  }
  // seed_oss.py + seed_oss_engine_tool_parser.py (name "seed_oss") - ByteDance
  // Seed-OSS: the Qwen3 XML body (<function=..><parameter=..>) wrapped in
  // <seed:tool_call> (reimplemented from the wire format; see seed_oss.h).
  if (name == "seed_oss") {
    return std::make_unique<SeedOssToolParser>();
  }
  // muse_glimmer_tool_parser.py:183 (register_module("muse_glimmer")) - the
  // channel-scoped ATEM <atem:function_calls>/<atem:invoke> dialect.
  if (name == "muse_glimmer") {
    return std::make_unique<MuseGlimmerToolParser>();
  }
  // inkling_tool_parser.py:7 (name "inkling", __init__.py:177) - Inkling's typed
  // content blocks (<|content_thinking|>/<|content_text|>/
  // <|content_invoke_tool_json|>) parsed by the SHARED ParserEngine, not a
  // hand-rolled text parser: upstream's class is a bare
  // make_adapters(InklingParser) tool adapter, and our engine + config already
  // ship (parser/inkling.cpp, engine/configs.cpp inkling_config, golden-gated in
  // test_parser_engine_assembly). This branch is the registry face over it -
  // without it the name threw at startup and the ported dialect was unreachable.
  if (name == "inkling") {
    return std::make_unique<InklingEngineToolParser>();
  }
  return nullptr;
}

// The enumeration of the factory above (see abstract.h). Listed in factory
// order, aliases individually. ADD A NAME HERE IN THE SAME CHANGE THAT ADDS ITS
// FACTORY BRANCH — the registry test resolves every entry and pins the count, so
// an unlisted parser is caught by the suite rather than by a user who cannot
// find it in --tool-call-parser's error message.
const std::vector<std::string>& tool_parser_names() {
  static const std::vector<std::string> names = {
      "hermes",         "qwen3",
      "deepseek_v3",    "deepseek_v31",
      "deepseek_v32",   "deepseek_v4",
      "longcat",        "mistral",
      "granite",        "granite4",
      "granite-20b-fc", "llama3_json",
      "llama4_json",    "pythonic",
      "llama4_pythonic", "xlam",
      "phi4_mini_json", "internlm",
      "jamba",          "step3",
      "step3p5",        "qwen3_coder",
      "qwen3_xml",      "mimo",
      "olmo3",          "minicpm5",
      "hy_v3",          "hunyuan_a13b",
      "apertus",        "ernie45",
      "gigachat3",      "lfm2",
      "poolside_v1",    "functiongemma",
      "kimi_k2",        "glm45",
      "glm47",          "minimax_m2",
      "gemma4",         "seed_oss",
      "muse_glimmer",   "inkling",
      "kolibri1",
  };
  return names;
}

}  // namespace vllm::entrypoints::openai
