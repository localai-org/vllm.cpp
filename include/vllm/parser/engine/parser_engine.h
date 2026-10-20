// Ported from: vllm/parser/engine/parser_engine.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// The parser ASSEMBLY layer: a ParserEngine consumes the SemanticEvent stream
// produced by the (already-ported) StreamingParserEngine and turns it into the
// serving-visible output — streaming `DeltaMessage`s (reasoning/content +
// tool-call name-first then argument deltas) and the one-shot
// `ExtractedToolCallInformation`. This is the 0.26 replacement for the legacy
// per-family hand-rolled streaming tool parsers.
//
// SCOPE (ROAD-V1-C8 TOOLS-STREAMING-PARSER-ASSEMBLY). Ported: the full event ->
// delta assembly, held-back streaming-arg prefix logic, tool-index increment,
// finish() flush, and the one-shot extract path, for the three engine configs
// (qwen3 / seed_oss / kimi_k2). JSON-schema tool-argument type coercion
// (_fix_arg_types / _streamable_string_keys with a non-empty tool schema) is now
// PORTED: ParserTool carries the function `parameters` schema, so a request whose
// tools declare typed params has its assembled arguments coerced to the declared
// types (parser_engine.py:365 _fix_arg_types); with no schema the path is identity
// (byte-identical, unchanged). RESIDUAL (tracked, .agents/specs/
// parser-assembly-c8.md): responses hooks. Prompt preparation is a base no-op
// overridden by Gemma for engine-local parse_delta calls.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/parser/engine/events.h"
#include "vllm/parser/engine/parser_engine_config.h"
#include "vllm/parser/engine/streaming_parser_engine.h"
#include "vllm/parser/engine/token_id_scanner.h"

namespace vllm::parser::engine {

namespace oai = vllm::entrypoints::openai;

// Minimal subset of ChatCompletionRequest / ResponsesRequest that ParserEngine
// actually reads (parser_engine.py: include_reasoning, tool_choice, tools, and
// the history-tool-call count). Models only what the assembly path consumes; a
// documented deviation from binding the full request type (mirrors the existing
// tool_parsers/abstract.h subset deviation).
struct ParserTool {
  std::string name;  // function tool name (find_tool_name / find_tool_properties)
  // The function's JSON-Schema `parameters` object (FunctionDefinition.parameters,
  // upstream dict[str,Any]|None). nullopt / absent `properties` => no schema, so
  // _fix_arg_types is identity and _streamable_string_keys is None (unchanged).
  std::optional<nlohmann::json> parameters;
};

struct ParserRequest {
  bool include_reasoning = true;      // chat_completion/protocol.py:242 (default True)
  std::string tool_choice = "auto";   // "none" | "auto" | "required" | "function"
  std::vector<ParserTool> tools;      // request.tools (function tools)
  int history_tool_call_cnt = 0;      // count_history_tool_calls(request)
};

// The ChatCompletionRequest -> ParserRequest projection. Upstream has no such
// function (chat_completion/serving.py hands the request object straight to
// parse_delta / parse); it exists here only because ParserRequest models the
// subset the assembly path reads. Shared so the serving path and the
// tool_parsers ParserEngineToolAdapter cannot drift into two projections of the
// same request. history_tool_call_cnt stays 0 for a fresh request, matching
// abstract_parser.py _initialize_history_tool_call_cnt.
ParserRequest ParserRequestFromChatCompletion(
    const oai::ChatCompletionRequest& request);

// Ported from: parser_engine.py:48 (ToolCallSlot).
struct ToolCallSlot {
  std::string id;
  std::string name;
  bool name_sent = false;
  // string_keys: keys whose trailing string values can safely stream. nullopt
  // means "no schema" (all string values keep JSON string form). See
  // _streamable_string_keys.
  std::optional<std::vector<std::string>> string_keys;
  std::string streamed_json;

  const std::string& args() const {
    if (!args_joined_) args_joined_ = join_parts();
    return *args_joined_;
  }
  void append_args(const std::string& value) {
    args_parts_.push_back(value);
    args_joined_.reset();
  }

 private:
  std::vector<std::string> args_parts_;
  mutable std::optional<std::string> args_joined_{std::string()};
  std::string join_parts() const {
    std::string s;
    for (const auto& p : args_parts_) s += p;
    return s;
  }
};

// Ported from: parser_engine.py:79 (ParserEngine).
class ParserEngine {
 public:
  // id_type -> tool-call id. Default mirrors chat_utils.make_tool_call_id
  // (random uuid for "random"; "functions.<name>:<idx>" for "kimi_k2"). The
  // gate injects a deterministic factory so ids are exactly comparable.
  using IdFactory = std::function<std::string(
      const std::string& id_type, const std::string& func_name, int idx)>;

  explicit ParserEngine(ParserEngineConfig config,
                        const EngineTokenizer* tokenizer = nullptr);
  virtual ~ParserEngine() = default;

  ParserEngine(const ParserEngine&) = delete;
  ParserEngine& operator=(const ParserEngine&) = delete;

  // Test/serving seam: override the tool-call id generator.
  void set_id_factory(IdFactory f) { id_factory_ = std::move(f); }
  void set_tool_call_id_type(std::string t) { tool_call_id_type_ = std::move(t); }

  bool reasoning_ended() const { return reasoning_ended_; }
  bool skip_tool_parsing() const { return engine_.skip_tool_parsing; }
  void set_skip_tool_parsing(bool v) { engine_.skip_tool_parsing = v; }
  std::optional<std::string> reasoning_start_str() const;
  std::optional<std::string> reasoning_end_str() const;

  // ── Engine lifecycle ──────────────────────────────────────────────
  void initialize_streaming(
      std::optional<ParserState> initial_state = std::nullopt);
  std::optional<oai::DeltaMessage> finish_streaming();
  virtual void adjust_initial_state_from_prompt(
      const std::vector<int>& prompt_token_ids);

  // ── Streaming ─────────────────────────────────────────────────────
  std::optional<oai::DeltaMessage> parse_delta(
      const std::string& delta_text, const std::vector<int>& delta_token_ids,
      const ParserRequest& request, bool finished);

  // Borrowed for this call only: nullptr means absent; an empty vector is present.
  // Existing four-argument callers retain their symbol. Untyped address-taking
  // must select an overload; virtual/layout changes require a consumer rebuild.
  std::optional<oai::DeltaMessage> parse_delta(
      const std::string& delta_text, const std::vector<int>& delta_token_ids,
      const ParserRequest& request, bool finished,
      const std::vector<int>* prompt_token_ids);


  // parser_engine.py:519 extract_reasoning_streaming — the REASONING face of the
  // incremental parse: initialize, feed the delta, assemble. Upstream takes the
  // previous/current text and the three token-ID spans and reads NONE of them
  // (only delta_text and delta_token_ids reach the lexer), so the C++ signature
  // carries just what is consumed. Unlike extract_tool_calls_streaming this does
  // NOT run _check_skip_tool_parsing: the reasoning adapter suppresses tool
  // parsing around the call instead (adapters.py:51 _skip_tool_parsing).
  std::optional<oai::DeltaMessage> extract_reasoning_streaming(
      const std::string& delta_text,
      const std::vector<int>& delta_token_ids = {});

  std::optional<oai::DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text, const ParserRequest& request);

  // ── Non-streaming ─────────────────────────────────────────────────
  // virtual: gemma4 overrides to strip the intrinsic `thought\n` channel prefix
  // (gemma4.py:572).
  virtual std::pair<std::optional<std::string>, std::optional<std::string>>
  extract_reasoning(const std::string& model_output,
                    const ParserRequest& request);

  oai::ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output, const ParserRequest& request);

  oai::ExtractedToolCallInformation extract_tool_calls_from_content(
      const std::string& content, const ParserRequest& request);

  std::tuple<std::optional<std::string>, std::optional<std::string>,
             std::optional<std::vector<oai::FunctionCall>>>
  parse(const std::string& model_output, const ParserRequest& request);

  // ── Reasoning state query ─────────────────────────────────────────
  // parser_engine.py:595 is_reasoning_end, in the TEXT form our reasoning seam
  // uses (reasoning_parsers/abstract.h documents the token-ID -> text
  // deviation; the think markers are self-delimiting so a substring scan is
  // equivalent for a well-formed stream). virtual: qwen3 also ends reasoning on
  // an unpaired tool-call marker (qwen3.py:256).
  virtual bool is_reasoning_end(const std::string& text) const;

 protected:
  // Overridable per-family hooks (kimi_k2 overrides several).
  virtual void emit_name_delta(int idx, std::vector<oai::DeltaToolCall>& deltas,
                               const std::optional<std::string>& name);
  virtual void handle_tool_name(const SemanticEvent& event);
  virtual void handle_arg_chunk(const SemanticEvent& event,
                                std::vector<oai::DeltaToolCall>& deltas);
  virtual void handle_tool_end(const SemanticEvent& event,
                               std::vector<oai::DeltaToolCall>& deltas);
  virtual std::string extract_args_json(const std::string& raw_args,
                                        const std::string& func_name);

  // parser_engine.py:210 _preprocess_feed — a first-feed hook that rewrites the
  // (delta_text, delta_token_ids) before the streaming engine sees them. Base is
  // the identity (inert for all families that don't override); gemma4 injects the
  // `<|channel>` opener (gemma4.py:424).
  virtual std::pair<std::string, std::vector<int>> preprocess_feed(
      const std::string& delta_text, const std::vector<int>& delta_token_ids);

  // parser_engine.py:706 _events_to_delta — virtual so gemma4 can post-process
  // the reasoning delta (strip the intrinsic `thought\n` prefix; gemma4.py:530).
  virtual std::optional<oai::DeltaMessage> events_to_delta(
      const std::vector<SemanticEvent>& events, bool finished);

  // parser_engine.py:645 _single_pass_parse — virtual so inkling can flush the
  // trailing text block that follows a tool-call block (inkling.py:376). The
  // tool_call_info the Python tuple carries is stashed in last_extracted_.
  virtual std::pair<std::optional<std::string>, std::optional<std::string>>
  single_pass_parse(const std::string& text, std::optional<ParserState> initial);

  // parser_engine.py:1064 _extract_args_value key list, refactored to a virtual
  // hook: inkling adds the "args" wrapper key (inkling.py:402). Base returns the
  // fixed {"arguments","parameters"} the upstream base inlines.
  virtual std::vector<std::string> args_wrapper_keys() const;

  // parser_engine.py:195 _reset — virtual so gemma4 can also clear its
  // prefix-strip / first-feed state (gemma4.py:418).
  virtual void reset(std::optional<ParserState> initial_state = std::nullopt);

  // Helpers reused by family overrides.
  void ensure_tool_id(ToolCallSlot& slot, const std::string& name);
  bool is_valid_tool_name(const std::string& name) const;
  std::optional<std::string> strip_content_whitespace(const std::string& content,
                                                      bool tools_called) const;
  ParserState engine_state() const { return engine_.state(); }
  // Prepare only the lexer/state machine, preserving assembly bookkeeping.
  void prepare_streaming_state(ParserState state);

  ParserEngineConfig config_;
  std::vector<ToolCallSlot> tool_slots_;
  std::string deferred_content_;
  // parser_engine.py:141-149 _reasoning_start_token_id / _reasoning_end_token_id
  // (resolved from the tokenizer vocab). Read by gemma4's _preprocess_feed.
  std::optional<int> reasoning_start_token_id_;
  std::optional<int> reasoning_end_token_id_;
  // Stashes the last _build_extracted_result (mirrors the tuple returned by
  // upstream _single_pass_parse, read by the non-streaming entrypoints).
  oai::ExtractedToolCallInformation last_extracted_;

 private:
  std::vector<SemanticEvent> feed(const std::string& delta_text,
                                  const std::vector<int>& delta_token_ids);
  void check_skip_tool_parsing(const ParserRequest& request);
  void initialize_history_tool_call_cnt(const ParserRequest& request);

  std::optional<oai::DeltaMessage> strip_trailing_reasoning(
      std::optional<oai::DeltaMessage> delta);

  void ensure_slot(int idx);
  std::optional<std::string> try_extract_name(int idx) const;
  std::optional<std::string> compute_arg_delta(int idx,
                                               const std::string& raw_delta);
  std::optional<std::string> flush_arg_converter(int idx);
  std::string fix_arg_types(const std::string& args_json,
                            const std::string& func_name) const;
  std::optional<std::vector<std::string>> streamable_string_keys(
      const std::string& func_name) const;
  // tool_parsers/utils.py:271 find_tool_properties — the named function tool's
  // `parameters.properties` object, or an empty object when the tool / params /
  // properties are absent. (Models only the FunctionTool form the assembly
  // ParserRequest carries; NamespaceTool / Responses variants unmodeled.)
  nlohmann::json find_tool_properties(const std::string& func_name) const;

  static std::vector<oai::DeltaToolCall> coalesce_tool_call_deltas(
      const std::vector<oai::DeltaToolCall>& deltas);

  oai::ExtractedToolCallInformation build_extracted_result(
      const std::vector<const oai::DeltaMessage*>& deltas);
  std::pair<std::string, std::string> extract_name_and_args(
      const std::string& raw_body) const;

  StreamingParserEngine engine_;

  IdFactory id_factory_;
  std::string tool_call_id_type_ = "random";
  int history_tool_call_cnt_ = 0;
  bool history_tool_call_cnt_initialized_ = false;

  bool has_reasoning_ = false;
  bool reasoning_ended_ = true;
  bool streaming_initialized_ = false;
  bool prompt_streaming_prepared_ = false;

  std::vector<ParserTool> tools_;
  std::string deferred_reasoning_;
  bool content_has_nonws_ = false;
  bool suppress_tool_calls_ = false;
};

}  // namespace vllm::parser::engine
