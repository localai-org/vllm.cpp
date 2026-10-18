// Ported from: vllm/tool_parsers/step3_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/step3.h"

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

// Python str.strip() over ASCII whitespace.
bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsSpace(s[b])) ++b;
  while (e > b && IsSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}
std::string Lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// step3_tool_parser.py:63-79 (_parse_steptml_invoke). Returns (name, params).
// name is nullopt when no <steptml:invoke name="..."> is present. params maps
// each parameter name to its stripped raw text value.
std::pair<std::optional<std::string>, std::vector<std::pair<std::string, std::string>>>
ParseSteptmlInvoke(const std::string& action_text) {
  std::vector<std::pair<std::string, std::string>> params;
  static const std::regex name_re(R"RE(<steptml:invoke name="([^"]+)">)RE");
  std::smatch m;
  if (!std::regex_search(action_text, m, name_re)) {
    return {std::nullopt, params};
  }
  const std::string func_name = m[1].str();

  static const std::regex param_re(
      R"RE(<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>)RE");
  for (auto it = std::sregex_iterator(action_text.begin(), action_text.end(),
                                      param_re);
       it != std::sregex_iterator(); ++it) {
    params.emplace_back((*it)[1].str(), Strip((*it)[2].str()));
  }
  return {func_name, params};
}

// step3_tool_parser.py:81-113 (_cast_arguments). Coerce each raw string value
// according to the matching tool's JSON-schema property type. Preserves author
// (insertion) order via ordered_json.
ojson CastArguments(
    const std::string& func_name,
    const std::vector<std::pair<std::string, std::string>>& params,
    const ChatCompletionRequest& request) {
  // Locate the schema properties for func_name (if any).
  const nlohmann::ordered_json* properties = nullptr;
  if (request.tools.has_value()) {
    for (const ChatCompletionToolsParam& tool : *request.tools) {
      if (tool.function.name == func_name) {
        if (tool.function.parameters.has_value() &&
            tool.function.parameters->is_object() &&
            tool.function.parameters->contains("properties") &&
            (*tool.function.parameters)["properties"].is_object()) {
          properties = &(*tool.function.parameters)["properties"];
        }
        break;
      }
    }
  }

  ojson out = ojson::object();
  for (const auto& [key, value] : params) {
    std::string typ;
    if (properties != nullptr && properties->contains(key) &&
        (*properties)[key].is_object() && (*properties)[key].contains("type") &&
        (*properties)[key]["type"].is_string()) {
      typ = (*properties)[key]["type"].get<std::string>();
    }

    if (typ == "string") {
      out[key] = Strip(value);
    } else if (typ == "integer") {
      try {
        std::size_t pos = 0;
        long long iv = std::stoll(value, &pos);
        if (pos == value.size()) {
          out[key] = iv;
        } else {
          out[key] = value;  // contextlib.suppress(ValueError) -> keep string
        }
      } catch (const std::exception&) {
        out[key] = value;
      }
    } else if (typ == "number") {
      try {
        std::size_t pos = 0;
        double dv = std::stod(value, &pos);
        if (pos == value.size()) {
          out[key] = dv;
        } else {
          out[key] = value;
        }
      } catch (const std::exception&) {
        out[key] = value;
      }
    } else if (typ == "boolean") {
      const std::string lv = Lower(value);
      if (lv == "true") {
        out[key] = true;
      } else if (lv == "false") {
        out[key] = false;
      } else {
        out[key] = value;
      }
    } else if (typ == "null") {
      if (Lower(value) == "null") {
        out[key] = nullptr;
      } else {
        out[key] = value;
      }
    } else {
      // No type / unknown type: leave the raw string value (upstream leaves the
      // dict entry untouched, which for a steptml parameter is the raw string).
      out[key] = value;
    }
  }
  return out;
}

// Python str.split(sep) semantics for the subset step3 uses.
std::vector<std::string> SplitAll(const std::string& s, const std::string& sep) {
  std::vector<std::string> parts;
  std::size_t pos = 0;
  for (;;) {
    std::size_t next = s.find(sep, pos);
    if (next == std::string::npos) {
      parts.push_back(s.substr(pos));
      break;
    }
    parts.push_back(s.substr(pos, next - pos));
    pos = next + sep.size();
  }
  return parts;
}

}  // namespace

ExtractedToolCallInformation Step3ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  const std::string begin = kToolCallsBegin;
  const std::string end = kToolCallsEnd;
  const std::string call_begin = kToolCallBegin;
  const std::string call_end = kToolCallEnd;
  const std::string sep = kToolSep;

  // step3_tool_parser.py:260-263 - no outer marker -> plain content.
  const std::size_t begin_pos = model_output.find(begin);
  if (begin_pos == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  // step3_tool_parser.py:265-271 - split off the tool block.
  const std::string pre_text = model_output.substr(0, begin_pos);
  const std::string rest = model_output.substr(begin_pos + begin.size());
  const std::size_t end_pos = rest.find(end);
  if (end_pos == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
  const std::string tool_block = rest.substr(0, end_pos);
  const std::string post_text = rest.substr(end_pos + end.size());
  const std::string content = Strip(pre_text + post_text);

  // step3_tool_parser.py:274-298 - each call requires `function<｜tool_sep｜>`
  // (a DeepSeek-style prefix the real step3 format never emits; preserved).
  std::vector<ToolCall> tool_calls;
  for (const std::string& part : SplitAll(tool_block, call_begin)) {
    if (part.empty() || part.find(call_end) == std::string::npos) continue;
    const std::string call_content = part.substr(0, part.find(call_end));
    if (call_content.find(sep) == std::string::npos) continue;
    const std::size_t sp = call_content.find(sep);
    const std::string type_part = call_content.substr(0, sp);
    const std::string invoke_part = call_content.substr(sp + sep.size());
    if (Strip(type_part) != "function") continue;

    auto [name, params] = ParseSteptmlInvoke(invoke_part);
    if (name.has_value()) {
      ojson args = CastArguments(*name, params, request);
      ToolCall tc;
      tc.function.name = *name;
      tc.function.arguments = args.dump();
      tool_calls.push_back(std::move(tc));
    }
  }

  if (!tool_calls.empty()) {
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty()) info.content = content;
    return info;
  }
  return ExtractedToolCallInformation{false, {}, model_output};
}

std::optional<DeltaMessage> Step3ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/, const ChatCompletionRequest& request) {
  const std::string begin = kToolCallsBegin;
  const std::string end = kToolCallsEnd;
  const std::string call_begin = kToolCallBegin;
  const std::string call_end = kToolCallEnd;
  const std::string sep = kToolSep;

  // step3_tool_parser.py:126 - process the stream from the last known position.
  for (;;) {
    if (position_ >= current_text.size()) return std::nullopt;

    std::string unprocessed = current_text.substr(position_);

    // STATE: all tools done -> the rest is content.
    if (tool_block_finished_) {
      position_ = current_text.size();
      DeltaMessage m;
      m.content = unprocessed;
      return m;
    }

    // STATE: before the tool block has started.
    if (!tool_block_started_) {
      if (unprocessed.rfind(begin, 0) == 0) {
        position_ += begin.size();
        tool_block_started_ = true;
        continue;
      }
      const std::size_t start_pos = unprocessed.find(begin);
      if (start_pos == std::string::npos) {
        // Hold back a partial OUTER-begin marker prefix (step3:146-152):
        // `TOOL_CALLS_BEGIN.startswith(unprocessed.strip()) and unprocessed`.
        const std::string stripped = Strip(unprocessed);
        if (begin.rfind(stripped, 0) == 0 && !unprocessed.empty()) {
          return std::nullopt;  // it's a prefix, wait
        }
        position_ = current_text.size();
        DeltaMessage m;
        m.content = unprocessed;
        return m;
      }
      const std::string content = unprocessed.substr(0, start_pos);
      position_ += content.size();
      DeltaMessage m;
      m.content = content;
      return m;
    }

    // STATE: inside the main tool block. Skip leading whitespace
    // (step3_tool_parser.py:159-161: offset = len - len(lstrip); position +=).
    std::size_t lead = 0;
    while (lead < unprocessed.size() && IsSpace(unprocessed[lead])) ++lead;
    unprocessed = unprocessed.substr(lead);
    position_ += lead;

    if (unprocessed.rfind(end, 0) == 0) {
      position_ += end.size();
      tool_block_finished_ = true;
      current_tool_id = -1;
      continue;
    }

    // Are we between tool calls?
    bool tool_finished = false;
    if (current_tool_id != -1 &&
        current_tool_id < static_cast<int>(prev_tool_call_arr.size())) {
      const nlohmann::json& e = prev_tool_call_arr[current_tool_id];
      tool_finished = e.contains("finished") && e["finished"].get<bool>();
    }
    if (current_tool_id == -1 || tool_finished) {
      if (unprocessed.rfind(call_begin, 0) == 0) {
        position_ += call_begin.size();
        if (current_tool_id == -1) {
          current_tool_id = 0;
        } else {
          current_tool_id += 1;
        }
        current_tool_name_sent = false;
        while (static_cast<int>(prev_tool_call_arr.size()) <= current_tool_id) {
          prev_tool_call_arr.push_back(nlohmann::json::object());
        }
        prev_tool_call_arr[current_tool_id]["finished"] = false;
        continue;
      }
      if (call_begin.rfind(unprocessed, 0) == 0) return std::nullopt;
    }

    // STATE: parsing an active tool call.
    bool active =
        current_tool_id != -1 &&
        current_tool_id < static_cast<int>(prev_tool_call_arr.size()) &&
        !(prev_tool_call_arr[current_tool_id].contains("finished") &&
          prev_tool_call_arr[current_tool_id]["finished"].get<bool>());
    if (active) {
      const std::size_t end_tool_pos = unprocessed.find(call_end);
      std::string tool_body =
          end_tool_pos == std::string::npos
              ? unprocessed
              : unprocessed.substr(0, end_tool_pos);

      if (end_tool_pos == std::string::npos &&
          call_end.rfind(tool_body, 0) == 0) {
        return std::nullopt;
      }

      auto [function_name, arguments] = ParseSteptmlInvoke(tool_body);
      if (!function_name.has_value()) return std::nullopt;

      // Send the function name as soon as it is parsed.
      if (!current_tool_name_sent) {
        current_tool_name_sent = true;
        prev_tool_call_arr[current_tool_id]["name"] = *function_name;
        DeltaMessage m;
        DeltaToolCall d;
        d.index = current_tool_id;
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = *function_name;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        return m;
      }

      prev_tool_call_arr[current_tool_id]["name"] = *function_name;

      // Only send arguments when the tool call is complete.
      if (end_tool_pos != std::string::npos) {
        position_ += end_tool_pos + call_end.size();
        prev_tool_call_arr[current_tool_id]["finished"] = true;

        ojson final_args = CastArguments(*function_name, arguments, request);
        if (!final_args.empty()) {
          DeltaMessage m;
          DeltaToolCall d;
          d.index = current_tool_id;
          d.function.arguments = final_args.dump();
          m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
          return m;
        }
      }
      return std::nullopt;
    }

    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
