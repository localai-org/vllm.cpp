// Ported from: vllm/parser/engine/parser_engine.py @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/engine/parser_engine.h"

#include <cctype>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"
#include "vllm/parser/engine/py_json.h"

namespace vllm::parser::engine {

namespace {

namespace oai = vllm::entrypoints::openai;

// Python-type category of a JSON value, used to model parser_engine.py:264
// `type(coerced) is not type(value)`: CPython has a single `int` type, so
// nlohmann's number_integer / number_unsigned collapse to one rank (bool is a
// distinct Python type from int, mirroring isinstance semantics here).
int py_type_rank(const nlohmann::json& v) {
  if (v.is_boolean()) return 1;
  if (v.is_number_integer() || v.is_number_unsigned()) return 2;
  if (v.is_number_float()) return 3;
  if (v.is_string()) return 4;
  if (v.is_null()) return 5;
  if (v.is_array()) return 6;
  if (v.is_object()) return 7;
  return 0;
}

std::pair<nlohmann::ordered_json, bool> coerce_value(
    const nlohmann::ordered_json& value, const nlohmann::json& schema);

// parser_engine.py:269 _coerce_dict — coerce every value in `args` whose key has
// a schema-object property. Mutates `args`; returns whether anything changed.
bool coerce_dict(nlohmann::ordered_json& args, const nlohmann::json& properties) {
  bool changed = false;
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (!properties.is_object() || !properties.contains(it.key())) continue;
    const nlohmann::json& prop = properties.at(it.key());
    if (!prop.is_object()) continue;  // properties.get(key) not a dict -> skip
    auto [coerced, val_changed] = coerce_value(it.value(), prop);
    if (val_changed) {
      it.value() = std::move(coerced);
      changed = true;
    }
  }
  return changed;
}

// parser_engine.py:227 _coerce_value — coerce a single value against its schema,
// recursing into nested object `properties` and array `items`.
std::pair<nlohmann::ordered_json, bool> coerce_value(
    const nlohmann::ordered_json& value, const nlohmann::json& schema) {
  if (value.is_string()) {
    const std::set<std::string> types = oai::extract_types_from_schema(schema);
    nlohmann::ordered_json coerced =
        oai::coerce_to_schema_type(value.get<std::string>(), types);
    // Upstream tests `coerced is not value`: coerce_to_schema_type returns the
    // SAME string only for the string candidate / parse-failure fallback, so a
    // string result whose content equals the input means "unchanged".
    const bool changed =
        !(coerced.is_string() && coerced.get<std::string>() == value.get<std::string>());
    if (changed) return {std::move(coerced), true};
    return {value, false};
  }
  if (value.is_object()) {
    if (schema.contains("properties") && schema.at("properties").is_object()) {
      nlohmann::ordered_json nested = value;
      const bool changed = coerce_dict(nested, schema.at("properties"));
      return {std::move(nested), changed};
    }
    return {value, false};
  }
  if (value.is_array()) {
    if (schema.contains("items") && schema.at("items").is_object()) {
      nlohmann::ordered_json arr = value;
      bool changed = false;
      for (auto& item : arr) {
        auto [coerced, item_changed] = coerce_value(item, schema.at("items"));
        if (item_changed) {
          item = std::move(coerced);
          changed = true;
        }
      }
      return {std::move(arr), changed};
    }
    return {value, false};
  }
  // Non-string scalar (number / bool / null): re-serialize and re-coerce.
  const std::set<std::string> types = oai::extract_types_from_schema(schema);
  const std::string as_str = python_json_dumps(value);
  nlohmann::ordered_json coerced = oai::coerce_to_schema_type(as_str, types);
  if (py_type_rank(coerced) != py_type_rank(value) || coerced != value) {
    return {std::move(coerced), true};
  }
  return {value, false};
}

// Python str.strip()/rstrip() over ASCII whitespace (the values the parser
// handles are ASCII/UTF-8; whitespace boundaries are ASCII).
bool is_py_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string rstrip(const std::string& s) {
  std::size_t end = s.size();
  while (end > 0 && is_py_space(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(0, end);
}
std::string strip(const std::string& s) {
  std::size_t begin = 0;
  while (begin < s.size() && is_py_space(static_cast<unsigned char>(s[begin])))
    ++begin;
  std::size_t end = s.size();
  while (end > begin && is_py_space(static_cast<unsigned char>(s[end - 1])))
    --end;
  return s.substr(begin, end - begin);
}
bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// parser_engine.py:999 _NAME_RE.
const std::regex& name_re() {
  static const std::regex re(R"RE("name"\s*:\s*"([^"]*)")RE");
  return re;
}

// parser_engine.py:283 _safe_arg_prefix — prefix of json_str up to the last
// top-level value; the trailing value is excluded unless it is a streamable
// string (string_keys). string_keys == nullopt means "no schema": stream the
// trailing string's content up to (but excluding) its closing quote.
std::string safe_arg_prefix(
    const std::string& json_str,
    const std::optional<std::vector<std::string>>& string_keys) {
  long last_colon = -1;
  std::optional<std::string> last_key;
  std::optional<std::string> pending_key;
  bool in_string = false;
  bool escape = false;
  long string_start = -1;
  int depth = 0;
  const long n = static_cast<long>(json_str.size());
  for (long i = 0; i < n; ++i) {
    char c = json_str[static_cast<std::size_t>(i)];
    if (escape) {
      escape = false;
      continue;
    }
    if (in_string) {
      if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
        if (depth == 1 && string_start >= 0) {
          pending_key = json_str.substr(static_cast<std::size_t>(string_start + 1),
                                        static_cast<std::size_t>(i - string_start - 1));
        }
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      string_start = i;
    } else if (c == '{' || c == '[') {
      depth += 1;
    } else if (c == '}' || c == ']') {
      depth -= 1;
    } else if (c == ':' && depth == 1) {
      last_colon = i;
      last_key = pending_key;
      pending_key.reset();
    }
  }
  if (last_colon < 0) return "";
  long end = last_colon + 1;
  while (end < n) {
    char c = json_str[static_cast<std::size_t>(end)];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
      ++end;
    else
      break;
  }
  if (end >= n || json_str[static_cast<std::size_t>(end)] != '"')
    return json_str.substr(0, static_cast<std::size_t>(end));
  if (string_keys.has_value()) {
    bool contains = false;
    if (last_key)
      for (const auto& k : *string_keys)
        if (k == *last_key) {
          contains = true;
          break;
        }
    if (!contains) return json_str.substr(0, static_cast<std::size_t>(end));
  }
  escape = false;
  for (long i = end + 1; i < n; ++i) {
    char c = json_str[static_cast<std::size_t>(i)];
    if (escape) {
      escape = false;
      continue;
    }
    if (c == '\\') {
      escape = true;
      continue;
    }
    if (c == '"') return json_str.substr(0, static_cast<std::size_t>(i));
  }
  return json_str;
}

}  // namespace

// See parser_engine.h. Relocated here from serving_chat.cpp's anonymous
// namespace so the tool_parsers ParserEngineToolAdapter shares ONE projection
// with the serving path rather than carrying a second copy of it.
ParserRequest ParserRequestFromChatCompletion(
    const oai::ChatCompletionRequest& request) {
  ParserRequest pr;
  pr.include_reasoning = request.include_reasoning;
  pr.tool_choice = request.tool_choice.has_value() ? request.tool_choice->mode
                                                   : std::string("auto");
  if (request.tools.has_value()) {
    for (const oai::ChatCompletionToolsParam& t : *request.tools) {
      ParserTool pt;
      pt.name = t.function.name;
      // Carry the function's JSON-Schema parameters so the assembly can coerce
      // argument values to their declared types (parser_engine.py _fix_arg_types /
      // find_tool_properties). Absent parameters => no schema (identity path).
      pt.parameters = t.function.parameters;
      pr.tools.push_back(std::move(pt));
    }
  }
  pr.history_tool_call_cnt = 0;
  return pr;
}

ParserEngine::ParserEngine(ParserEngineConfig config,
                           const EngineTokenizer* tokenizer)
    : config_(std::move(config)),
      engine_(config_, tokenizer, std::nullopt,
              tokenizer ? &tokenizer->get_vocab() : nullptr) {
  id_factory_ = [](const std::string& id_type, const std::string& func_name,
                   int idx) -> std::string {
    if (id_type == "kimi_k2")
      return "functions." + func_name + ":" + std::to_string(idx);
    return oai::make_tool_call_id();  // random "chatcmpl-tool-<uuid>"
  };
  // parser_engine.py:111 _has_reasoning.
  has_reasoning_ = config_.token_id_terminals.count("THINK_END") > 0 ||
                   config_.terminals.count("THINK_START") > 0 ||
                   config_.terminals.count("THINK_END") > 0 ||
                   config_.initial_state == ParserState::REASONING;
  reasoning_ended_ = !has_reasoning_;

  // parser_engine.py:141-149 — resolve the reasoning start/end token ids from the
  // tokenizer vocab (used by per-family _preprocess_feed hooks, e.g. gemma4).
  if (tokenizer) {
    const std::map<std::string, int>& vocab = tokenizer->get_vocab();
    auto resolve = [&](const char* key) -> std::optional<int> {
      auto it = config_.token_id_terminals.find(key);
      if (it == config_.token_id_terminals.end() || it->second.empty())
        return std::nullopt;
      auto v = vocab.find(it->second);
      if (v == vocab.end()) return std::nullopt;
      return v->second;
    };
    reasoning_start_token_id_ = resolve("THINK_START");
    reasoning_end_token_id_ = resolve("THINK_END");
  }
}

std::optional<std::string> ParserEngine::reasoning_start_str() const {
  auto it = config_.terminals.find("THINK_START");
  if (it == config_.terminals.end()) return std::nullopt;
  return it->second;
}
std::optional<std::string> ParserEngine::reasoning_end_str() const {
  auto it = config_.terminals.find("THINK_END");
  if (it == config_.terminals.end()) return std::nullopt;
  return it->second;
}

void ParserEngine::initialize_streaming(
    std::optional<ParserState> initial_state) {
  if (!streaming_initialized_) {
    streaming_initialized_ = true;
    reset(initial_state);
  }
}

void ParserEngine::reset(std::optional<ParserState> initial_state) {
  engine_.reset(initial_state);
  prompt_streaming_prepared_ = false;
  reasoning_ended_ = !has_reasoning_;
  tool_slots_.clear();
  deferred_content_.clear();
  deferred_reasoning_.clear();
  content_has_nonws_ = false;
}

std::pair<std::string, std::vector<int>> ParserEngine::preprocess_feed(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  // parser_engine.py:210 — base identity. Inert for every family that does not
  // override it (all but gemma4).
  return {delta_text, delta_token_ids};
}

std::vector<std::string> ParserEngine::args_wrapper_keys() const {
  // parser_engine.py:1064 _extract_args_value — the base key list. inkling
  // prepends "args" (inkling.py:402).
  return {"arguments", "parameters"};
}

std::vector<SemanticEvent> ParserEngine::feed(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  // parser_engine.py:217 _feed — run the (overridable) preprocess hook first.
  auto [text, ids] = preprocess_feed(delta_text, delta_token_ids);
  return engine_.feed(text, ids);
}

void ParserEngine::initialize_history_tool_call_cnt(
    const ParserRequest& request) {
  if (!history_tool_call_cnt_initialized_) {
    history_tool_call_cnt_ = request.history_tool_call_cnt;
    history_tool_call_cnt_initialized_ = true;
  }
}

void ParserEngine::check_skip_tool_parsing(const ParserRequest& request) {
  if (!request.tools.empty()) tools_ = request.tools;
  if (!skip_tool_parsing() && !suppress_tool_calls_) {
    if (request.tool_choice == "none" && !request.tools.empty())
      suppress_tool_calls_ = true;
  }
}

std::optional<oai::DeltaMessage> ParserEngine::finish_streaming() {
  std::vector<SemanticEvent> events = engine_.finish();
  if (!events.empty() || !deferred_content_.empty())
    return events_to_delta(events, /*finished=*/true);
  return std::nullopt;
}

void ParserEngine::adjust_initial_state_from_prompt(
    const std::vector<int>& /*prompt_token_ids*/) {}

void ParserEngine::prepare_streaming_state(ParserState state) {
  engine_.reset(state);
  streaming_initialized_ = true;
}

std::optional<oai::DeltaMessage> ParserEngine::parse_delta(
    const std::string& delta_text, const std::vector<int>& delta_token_ids,
    const ParserRequest& request, bool finished) {
  return parse_delta(delta_text, delta_token_ids, request, finished, nullptr);
}

std::optional<oai::DeltaMessage> ParserEngine::parse_delta(
    const std::string& delta_text, const std::vector<int>& delta_token_ids,
    const ParserRequest& request, bool finished,
    const std::vector<int>* prompt_token_ids) {
  initialize_history_tool_call_cnt(request);
  // a7c23ac96d parser_engine.py:455-461: the hook may reset this flag.
  if (!prompt_streaming_prepared_ && prompt_token_ids) {
    adjust_initial_state_from_prompt(*prompt_token_ids);
    prompt_streaming_prepared_ = true;
  }
  check_skip_tool_parsing(request);
  std::vector<SemanticEvent> events = feed(delta_text, delta_token_ids);
  if (finished) {
    std::vector<SemanticEvent> fin = engine_.finish();
    events.insert(events.end(), fin.begin(), fin.end());
  }
  std::optional<oai::DeltaMessage> result = events_to_delta(events, finished);
  result = strip_trailing_reasoning(std::move(result));

  if (result && !request.include_reasoning) {
    result->reasoning = std::nullopt;
    bool no_tools = !result->tool_calls || result->tool_calls->empty();
    if (!result->content && no_tools) result = std::nullopt;
  }
  return result;
}

// parser_engine.py:519 extract_reasoning_streaming.
std::optional<oai::DeltaMessage> ParserEngine::extract_reasoning_streaming(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  initialize_streaming();
  std::vector<SemanticEvent> events = feed(delta_text, delta_token_ids);
  return strip_trailing_reasoning(events_to_delta(events, /*finished=*/false));
}

// parser_engine.py:595 is_reasoning_end — TEXT form. Upstream walks the token
// ids BACKWARDS and returns True at the first reasoning-end token, False at the
// first reasoning-start token. Over text that is exactly "the LAST end marker
// lies after the LAST start marker" (neither literal is a substring of the
// other, so their occurrences cannot alias).
bool ParserEngine::is_reasoning_end(const std::string& text) const {
  const std::optional<std::string> end = reasoning_end_str();
  if (!end) return reasoning_ended_;
  if (text.empty()) return config_.initial_state != ParserState::REASONING;
  const std::size_t pe = text.rfind(*end);
  if (pe == std::string::npos) return false;
  const std::optional<std::string> start = reasoning_start_str();
  if (!start) return true;
  const std::size_t ps = text.rfind(*start);
  return ps == std::string::npos || pe > ps;
}

std::optional<oai::DeltaMessage> ParserEngine::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ParserRequest& request) {
  initialize_streaming();
  check_skip_tool_parsing(request);
  std::vector<SemanticEvent> events = feed(delta_text, {});
  return strip_trailing_reasoning(events_to_delta(events, /*finished=*/false));
}

std::optional<oai::DeltaMessage> ParserEngine::strip_trailing_reasoning(
    std::optional<oai::DeltaMessage> delta) {
  if (!config_.strip_trailing_reasoning_whitespace) return delta;
  if (delta && delta->reasoning) {
    std::string combined = deferred_reasoning_ + *delta->reasoning;
    std::string trimmed = rstrip(combined);
    deferred_reasoning_ = combined.substr(trimmed.size());
    delta->reasoning = trimmed.empty() ? std::optional<std::string>()
                                       : std::optional<std::string>(trimmed);
    bool no_tools = !delta->tool_calls || delta->tool_calls->empty();
    if (!delta->reasoning && !delta->content && no_tools)
      return std::nullopt;
  } else if (!deferred_reasoning_.empty() && reasoning_ended_) {
    deferred_reasoning_.clear();
  }
  return delta;
}

std::optional<std::string> ParserEngine::strip_content_whitespace(
    const std::string& content_in, bool tools_called) const {
  std::string content = content_in;
  if (tools_called) {
    if (config_.strip_content_whitespace_with_tools) {
      content = strip(content);
    } else if (config_.drop_whitespace_only_content_before_tools &&
               strip(content).empty()) {
      content = "";
    }
  }
  if (content.empty()) return std::nullopt;
  return content;
}

// ── Event-to-delta conversion (parser_engine.py:706) ──────────────────
std::optional<oai::DeltaMessage> ParserEngine::events_to_delta(
    const std::vector<SemanticEvent>& events, bool finished) {
  if (events.empty() && deferred_content_.empty()) return std::nullopt;

  std::vector<oai::DeltaToolCall> tool_call_deltas;
  std::vector<std::string> content_parts;
  std::vector<std::string> reasoning_parts;

  bool seen_tool_event = false;
  const bool suppress = suppress_tool_calls_;
  for (const SemanticEvent& event : events) {
    switch (event.type) {
      case EventType::TEXT_CHUNK:
        if (seen_tool_event)
          deferred_content_ += event.value;
        else
          content_parts.push_back(event.value);
        break;
      case EventType::REASONING_CHUNK:
        reasoning_parts.push_back(event.value);
        break;
      case EventType::REASONING_END:
        reasoning_ended_ = true;
        break;
      case EventType::TOOL_CALL_START:
        if (!suppress) {
          seen_tool_event = true;
          ensure_slot(event.tool_index);
        }
        break;
      case EventType::TOOL_NAME:
        if (!suppress) {
          seen_tool_event = true;
          handle_tool_name(event);
        }
        break;
      case EventType::ARG_VALUE_CHUNK:
        if (!suppress) {
          seen_tool_event = true;
          handle_arg_chunk(event, tool_call_deltas);
        }
        break;
      case EventType::TOOL_CALL_END:
        if (!suppress) {
          seen_tool_event = true;
          handle_tool_end(event, tool_call_deltas);
        }
        break;
      case EventType::REASONING_START:
        break;  // no delta-level effect
    }
  }

  if (tool_call_deltas.size() > 1)
    tool_call_deltas = coalesce_tool_call_deltas(tool_call_deltas);

  if (!deferred_content_.empty() &&
      (!seen_tool_event || tool_call_deltas.empty())) {
    content_parts.insert(content_parts.begin(), deferred_content_);
    deferred_content_.clear();
  }

  std::string content_str;
  for (const auto& p : content_parts) content_str += p;

  if (content_has_nonws_) {
    // pass
  } else if (!content_str.empty()) {
    std::string stripped = strip(content_str);
    if (!stripped.empty()) {
      content_has_nonws_ = true;
    } else if (!tool_slots_.empty()) {
      if (config_.drop_whitespace_only_content_before_tools) content_str.clear();
    } else if (!finished) {
      deferred_content_ = content_str;
      content_str.clear();
    }
  }

  std::optional<std::string> content =
      content_str.empty() ? std::optional<std::string>()
                          : std::optional<std::string>(content_str);
  std::string reasoning_str;
  for (const auto& p : reasoning_parts) reasoning_str += p;
  std::optional<std::string> reasoning =
      reasoning_str.empty() ? std::optional<std::string>()
                            : std::optional<std::string>(reasoning_str);

  if (content || !tool_call_deltas.empty() || reasoning) {
    oai::DeltaMessage msg;
    if (content) msg.content = content;
    if (reasoning) msg.reasoning = reasoning;
    if (!tool_call_deltas.empty()) msg.tool_calls = std::move(tool_call_deltas);
    return msg;
  }
  return std::nullopt;
}

void ParserEngine::ensure_slot(int idx) {
  while (static_cast<int>(tool_slots_.size()) <= idx)
    tool_slots_.emplace_back();
}

void ParserEngine::ensure_tool_id(ToolCallSlot& slot, const std::string& name) {
  if (slot.id.empty()) {
    slot.id = id_factory_(tool_call_id_type_, name, history_tool_call_cnt_);
    history_tool_call_cnt_ += 1;
  }
}

void ParserEngine::handle_tool_name(const SemanticEvent& event) {
  tool_slots_[static_cast<std::size_t>(event.tool_index)].name += event.value;
}

nlohmann::json ParserEngine::find_tool_properties(
    const std::string& func_name) const {
  // tool_parsers/utils.py:271 — return the named tool's parameters.properties,
  // else an empty object.
  for (const ParserTool& t : tools_) {
    if (t.name != func_name) continue;
    if (!t.parameters.has_value()) return nlohmann::json::object();
    const nlohmann::json& params = *t.parameters;
    if (!params.is_object() || !params.contains("properties"))
      return nlohmann::json::object();
    const nlohmann::json& props = params.at("properties");
    if (!props.is_object()) return nlohmann::json::object();
    return props;
  }
  return nlohmann::json::object();
}

std::optional<std::vector<std::string>> ParserEngine::streamable_string_keys(
    const std::string& func_name) const {
  // parser_engine.py:348 _streamable_string_keys — no schema (empty properties)
  // yields None (all string values keep their JSON string form); otherwise the
  // set of keys whose schema type is EXACTLY {"string"} (safe to stream before
  // the value closes, since coercion won't change their serialized form).
  const nlohmann::json props = find_tool_properties(func_name);
  if (!props.is_object() || props.empty()) return std::nullopt;
  std::vector<std::string> keys;
  for (auto it = props.begin(); it != props.end(); ++it) {
    const std::set<std::string> types = oai::extract_types_from_schema(it.value());
    if (types.size() == 1 && *types.begin() == "string") keys.push_back(it.key());
  }
  return keys;
}

bool ParserEngine::is_valid_tool_name(const std::string& name) const {
  if (!config_.validate_tool_names) return true;
  if (tools_.empty()) return true;
  for (const auto& t : tools_)
    if (t.name == name) return true;
  return false;
}

void ParserEngine::emit_name_delta(int idx,
                                   std::vector<oai::DeltaToolCall>& deltas,
                                   const std::optional<std::string>& name) {
  if (!name || name->empty() || !is_valid_tool_name(*name)) return;
  ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];
  slot.name = *name;
  slot.name_sent = true;
  slot.string_keys = streamable_string_keys(*name);
  ensure_tool_id(slot, *name);
  oai::DeltaToolCall d;
  d.index = idx;
  d.id = slot.id;
  d.type = "function";
  d.function.name = *name;
  deltas.push_back(std::move(d));
}

void ParserEngine::handle_arg_chunk(const SemanticEvent& event,
                                    std::vector<oai::DeltaToolCall>& deltas) {
  const int idx = event.tool_index;
  ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];
  if (!event.value.empty()) slot.append_args(event.value);

  if (!slot.name_sent) {
    if (!slot.name.empty()) {
      emit_name_delta(idx, deltas, slot.name);
    } else if (!event.value.empty()) {
      emit_name_delta(idx, deltas, try_extract_name(idx));
    }
  } else if (!event.value.empty()) {
    std::optional<std::string> arg_delta = compute_arg_delta(idx, event.value);
    if (arg_delta && !arg_delta->empty()) {
      oai::DeltaToolCall d;
      d.index = idx;
      d.function.arguments = *arg_delta;
      deltas.push_back(std::move(d));
    }
  }
}

void ParserEngine::handle_tool_end(const SemanticEvent& event,
                                   std::vector<oai::DeltaToolCall>& deltas) {
  const int idx = event.tool_index;
  if (idx >= static_cast<int>(tool_slots_.size())) return;

  std::optional<std::string> remaining = flush_arg_converter(idx);
  ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];

  if (!slot.name_sent) {
    std::optional<std::string> name =
        !slot.name.empty() ? std::optional<std::string>(slot.name)
                           : try_extract_name(idx);
    if (name && !name->empty() && is_valid_tool_name(*name)) {
      slot.name = *name;
      slot.name_sent = true;
      slot.string_keys = streamable_string_keys(*name);
      ensure_tool_id(slot, *name);
      oai::DeltaToolCall d;
      d.index = idx;
      d.id = slot.id;
      d.type = "function";
      d.function.name = *name;
      d.function.arguments = remaining ? *remaining : std::string("");
      deltas.push_back(std::move(d));
      remaining.reset();
    }
  }

  if (remaining && !remaining->empty() && slot.name_sent) {
    oai::DeltaToolCall d;
    d.index = idx;
    d.function.arguments = *remaining;
    deltas.push_back(std::move(d));
  }
}

std::vector<oai::DeltaToolCall> ParserEngine::coalesce_tool_call_deltas(
    const std::vector<oai::DeltaToolCall>& deltas) {
  // Merge entries that share the same index into one per index, preserving
  // first-seen index order (parser_engine.py:900).
  std::vector<int> order;
  std::vector<oai::DeltaToolCall> merged;
  auto find = [&](int index) -> oai::DeltaToolCall* {
    for (std::size_t i = 0; i < order.size(); ++i)
      if (order[i] == index) return &merged[i];
    return nullptr;
  };
  for (const oai::DeltaToolCall& tc : deltas) {
    oai::DeltaToolCall* existing = find(tc.index);
    if (existing == nullptr) {
      order.push_back(tc.index);
      merged.push_back(tc);
      continue;
    }
    if (tc.id && !existing->id) existing->id = tc.id;
    if (tc.type && !existing->type) existing->type = tc.type;
    if (tc.function.name && !existing->function.name)
      existing->function.name = tc.function.name;
    if (tc.function.arguments) {
      if (!existing->function.arguments)
        existing->function.arguments = tc.function.arguments;
      else
        *existing->function.arguments += *tc.function.arguments;
    }
  }
  if (merged.size() == deltas.size()) return deltas;
  return merged;
}

std::optional<std::string> ParserEngine::compute_arg_delta(
    int idx, const std::string& raw_delta) {
  if (!config_.arg_converter) return raw_delta;
  if (!config_.stream_arg_deltas) return std::nullopt;

  if (config_.arg_structural_chars) {
    const std::set<char>& chars = *config_.arg_structural_chars;
    bool any = false;
    for (char c : raw_delta)
      if (chars.count(c)) {
        any = true;
        break;
      }
    if (!any) return std::nullopt;  // structural.isdisjoint(raw_delta)
  }

  ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];
  std::string current_json;
  try {
    current_json = config_.arg_converter(slot.args(), true);
  } catch (...) {
    return std::nullopt;
  }
  if (current_json.empty()) return std::nullopt;
  if (!slot.name.empty()) current_json = fix_arg_types(current_json, slot.name);

  const std::string& prev = slot.streamed_json;
  std::string safe_json = safe_arg_prefix(current_json, slot.string_keys);
  if (safe_json.empty() || safe_json == prev) return std::nullopt;

  std::string diff;
  if (!prev.empty()) {
    if (!starts_with(safe_json, prev)) return std::nullopt;
    diff = safe_json.substr(prev.size());
  } else {
    diff = safe_json;
  }
  if (!diff.empty()) {
    slot.streamed_json = safe_json;
    return diff;
  }
  return std::nullopt;
}

std::optional<std::string> ParserEngine::flush_arg_converter(int idx) {
  if (!config_.arg_converter) return std::nullopt;
  ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];
  std::string final_json;
  try {
    final_json = config_.arg_converter(slot.args(), false);
  } catch (...) {
    return std::nullopt;
  }
  if (!final_json.empty()) final_json = fix_arg_types(final_json, slot.name);

  const std::string& prev = slot.streamed_json;
  if (!final_json.empty() && final_json.size() > prev.size()) {
    if (!prev.empty() && !starts_with(final_json, prev)) return std::nullopt;
    std::string diff = final_json.substr(prev.size());
    slot.streamed_json = final_json;
    return diff;
  }
  return std::nullopt;
}

std::string ParserEngine::fix_arg_types(const std::string& args_json,
                                        const std::string& func_name) const {
  // parser_engine.py:365 _fix_arg_types. No tools / no name / unparseable /
  // non-object / no schema all pass the raw string through unchanged (identity);
  // only a successful, schema-changed coercion is re-serialized.
  if (tools_.empty() || func_name.empty()) return args_json;
  nlohmann::ordered_json args;
  try {
    args = nlohmann::ordered_json::parse(args_json);
  } catch (const std::exception&) {
    return args_json;
  }
  if (!args.is_object()) return args_json;
  const nlohmann::json properties = find_tool_properties(func_name);
  if (!properties.is_object() || properties.empty()) return args_json;
  if (coerce_dict(args, properties)) return python_json_dumps(args);
  return args_json;
}

std::optional<std::string> ParserEngine::try_extract_name(int idx) const {
  std::smatch m;
  const std::string& args = tool_slots_[static_cast<std::size_t>(idx)].args();
  if (std::regex_search(args, m, name_re())) {
    std::string name = m[1].str();
    if (!name.empty()) return name;
  }
  return std::nullopt;
}

std::string ParserEngine::extract_args_json(const std::string& raw_args,
                                            const std::string& /*func_name*/) {
  if (strip(raw_args).empty()) return "{}";
  auto [name, args] = extract_name_and_args(raw_args);
  (void)name;
  return args;
}

std::pair<std::string, std::string> ParserEngine::extract_name_and_args(
    const std::string& raw_body_in) const {
  std::string raw_body = strip(raw_body_in);
  nlohmann::ordered_json parsed;
  try {
    parsed = nlohmann::ordered_json::parse(raw_body);
  } catch (...) {
    return {"", raw_body};
  }
  if (!parsed.is_object()) return {"", raw_body};

  std::string name;
  if (parsed.contains("name") && parsed["name"].is_string())
    name = parsed["name"].get<std::string>();

  // parser_engine.py:1088 args = self._extract_args_value(parsed): iterate the
  // (overridable) wrapper-key list; inkling adds "args".
  for (const std::string& key : args_wrapper_keys()) {
    if (parsed.contains(key)) {
      const auto& val = parsed[key];
      if (val.is_string()) return {name, val.get<std::string>()};
      return {name, python_json_dumps(val)};
    }
  }
  nlohmann::ordered_json without_name = nlohmann::ordered_json::object();
  for (auto it = parsed.begin(); it != parsed.end(); ++it)
    if (it.key() != "name") without_name[it.key()] = it.value();
  return {name, python_json_dumps(without_name)};
}

// ── Build ExtractedToolCallInformation (parser_engine.py:1011) ─────────
oai::ExtractedToolCallInformation ParserEngine::build_extracted_result(
    const std::vector<const oai::DeltaMessage*>& deltas) {
  std::vector<std::string> content_parts;
  for (const oai::DeltaMessage* d : deltas)
    if (d && d->content) content_parts.push_back(*d->content);

  std::vector<oai::ToolCall> tool_calls;
  for (std::size_t idx = 0; idx < tool_slots_.size(); ++idx) {
    ToolCallSlot& slot = tool_slots_[idx];
    if (slot.name.empty() && slot.args().empty()) continue;

    std::string name = strip(slot.name);
    const std::string& raw_body = slot.args();
    std::string args_json;

    if (name.empty() && !strip(raw_body).empty()) {
      auto [n, a] = extract_name_and_args(raw_body);
      name = n;
      args_json = a;
    } else if (!strip(raw_body).empty()) {
      if (config_.arg_converter) {
        try {
          args_json = config_.arg_converter(raw_body, false);
        } catch (...) {
          args_json = extract_args_json(raw_body, name);
        }
      } else {
        args_json = extract_args_json(raw_body, name);
      }
    } else {
      args_json = "{}";
    }

    if (!name.empty() && is_valid_tool_name(name)) {
      ensure_tool_id(slot, name);
      args_json = fix_arg_types(args_json, name);
      oai::ToolCall tc;
      tc.id = slot.id;
      tc.function = oai::FunctionCall{name, args_json};
      tool_calls.push_back(std::move(tc));
    }
  }

  std::string content_str;
  for (const auto& p : content_parts) content_str += p;
  std::optional<std::string> content =
      strip_content_whitespace(content_str, !tool_calls.empty());

  oai::ExtractedToolCallInformation info;
  info.tools_called = !tool_calls.empty();
  info.tool_calls = std::move(tool_calls);
  info.content = content;
  return info;
}

// ── Non-streaming entrypoints ─────────────────────────────────────────
std::pair<std::optional<std::string>, std::optional<std::string>>
ParserEngine::extract_reasoning(const std::string& model_output,
                                const ParserRequest& /*request*/) {
  reset();
  std::vector<SemanticEvent> events = feed(model_output, {});
  std::vector<SemanticEvent> fin = engine_.finish();
  events.insert(events.end(), fin.begin(), fin.end());

  std::string reasoning_parts;
  std::string content_parts;
  for (const SemanticEvent& event : events) {
    if (event.type == EventType::REASONING_CHUNK)
      reasoning_parts += event.value;
    else if (event.type == EventType::TEXT_CHUNK)
      content_parts += event.value;
    else if (event.type == EventType::REASONING_END)
      reasoning_ended_ = true;
  }
  std::string raw_reasoning = reasoning_parts;
  if (config_.strip_trailing_reasoning_whitespace)
    raw_reasoning = rstrip(raw_reasoning);
  std::optional<std::string> reasoning =
      raw_reasoning.empty() ? std::optional<std::string>()
                            : std::optional<std::string>(raw_reasoning);
  std::optional<std::string> content =
      content_parts.empty() ? std::optional<std::string>()
                            : std::optional<std::string>(content_parts);
  return {reasoning, content};
}

std::pair<std::optional<std::string>, std::optional<std::string>>
ParserEngine::single_pass_parse(const std::string& text,
                                std::optional<ParserState> initial) {
  reset(initial);
  std::vector<SemanticEvent> events = feed(text, {});
  std::vector<SemanticEvent> fin = engine_.finish();
  events.insert(events.end(), fin.begin(), fin.end());

  std::optional<oai::DeltaMessage> delta =
      events_to_delta(events, /*finished=*/true);
  // build_extracted_result reads tool state populated by events_to_delta.
  last_extracted_ = build_extracted_result({});

  std::optional<std::string> reasoning = delta ? delta->reasoning : std::nullopt;
  if (reasoning && config_.strip_trailing_reasoning_whitespace) {
    std::string r = rstrip(*reasoning);
    reasoning = r.empty() ? std::optional<std::string>()
                          : std::optional<std::string>(r);
  }
  std::optional<std::string> content = delta ? delta->content : std::nullopt;
  if (content)
    content = strip_content_whitespace(*content, last_extracted_.tools_called);
  return {reasoning, content};
}

oai::ExtractedToolCallInformation ParserEngine::extract_tool_calls(
    const std::string& model_output, const ParserRequest& request) {
  reset();
  streaming_initialized_ = true;
  std::optional<oai::DeltaMessage> result =
      extract_tool_calls_streaming("", model_output, model_output, request);
  std::optional<oai::DeltaMessage> finish_delta = finish_streaming();
  std::vector<const oai::DeltaMessage*> deltas;
  if (result) deltas.push_back(&*result);
  if (finish_delta) deltas.push_back(&*finish_delta);
  return build_extracted_result(deltas);
}

oai::ExtractedToolCallInformation ParserEngine::extract_tool_calls_from_content(
    const std::string& content, const ParserRequest& request) {
  check_skip_tool_parsing(request);
  auto [reasoning, parsed_content] =
      single_pass_parse(content, ParserState::CONTENT);
  (void)reasoning;
  oai::ExtractedToolCallInformation info = last_extracted_;
  if (parsed_content && !info.content) info.content = parsed_content;
  return info;
}

std::tuple<std::optional<std::string>, std::optional<std::string>,
           std::optional<std::vector<oai::FunctionCall>>>
ParserEngine::parse(const std::string& model_output,
                    const ParserRequest& request) {
  initialize_history_tool_call_cnt(request);
  check_skip_tool_parsing(request);
  auto [reasoning, content] = single_pass_parse(model_output, std::nullopt);

  std::optional<std::vector<oai::FunctionCall>> tool_calls;
  if (last_extracted_.tools_called) {
    std::vector<oai::FunctionCall> calls;
    for (const oai::ToolCall& tc : last_extracted_.tool_calls)
      calls.push_back(
          oai::FunctionCall{tc.function.name, tc.function.arguments});
    tool_calls = std::move(calls);
  }
  return {reasoning, content, tool_calls};
}

}  // namespace vllm::parser::engine
