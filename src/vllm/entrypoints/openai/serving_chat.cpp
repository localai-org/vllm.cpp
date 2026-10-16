// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// See serving_chat.h for scope, the chat-prompt seam and deferrals.
#include "vllm/entrypoints/openai/serving_chat.h"

#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/beam_search.h"
#include "vllm/entrypoints/openai/chat_mm.h"  // HasMultiModalParts (mm seam gate)
#include "vllm/entrypoints/openai/request_logger.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/entrypoints/openai/tool_parsers/structural_tags.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/validation_error.h"

namespace vllm::entrypoints::openai {

namespace {
// get_chat_request_role (chat_completion/serving.py:399): the response role is
// self.response_role ("assistant") when add_generation_prompt (the T0 default).
constexpr const char* kAssistantRole = "assistant";

void ChatDbg(const std::string& id, const std::string& msg) {
  LogRequestStage(id, msg);
}

// Whether tool_choice selects a single named function (finish_reason stays the
// model's own — "stop" — for named calls; chat_completion/serving.py:688,935).
bool IsNamedToolChoice(const ChatCompletionRequest& request) {
  return request.tool_choice.has_value() &&
         request.tool_choice->mode == "function";
}

}  // namespace

std::string DefaultChatPromptFallback(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt,
    const std::vector<ChatCompletionToolsParam>& /*tools*/,
    const nlohmann::ordered_json& /*chat_template_kwargs*/) {
  // T0 SEAM (M3.2 swaps in the real chat-template renderer). A simple
  // "<role>: <content>\n" join + an "assistant:" generation prompt. This is NOT
  // a model chat template — it exists only so the chat path is end-to-end
  // exercisable. `tools` are ignored here (rendered by the real template only).
  std::string prompt;
  for (const ChatMessage& m : messages) {
    prompt += m.role;
    prompt += ": ";
    if (m.content.has_value()) prompt += *m.content;
    prompt += "\n";
  }
  if (add_generation_prompt) {
    prompt += "assistant:";
  }
  return prompt;
}

bool ToolsEnabled(const ChatCompletionRequest& request) {
  if (!request.tools.has_value() || request.tools->empty()) return false;
  // tool_choice absent → defaults to "auto" when tools present (enabled).
  if (request.tool_choice.has_value() &&
      request.tool_choice->mode == "none") {
    return false;
  }
  return true;
}

std::optional<nlohmann::json> ToolChoiceStructuralTagSpec(
    const ChatCompletionRequest& request) {
  // Thin wrapper preserved for source compatibility (existing callers/tests).
  // The Hermes builder now lives in the per-family registry (structural_tags.h);
  // this delegates to the "hermes" family exactly.
  return ToolChoiceStructuralTagSpecFor("hermes", request);
}

void ApplyToolChoiceStructuredOutput(const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params) {
  // Thin wrapper preserved for source compatibility — delegates to the Hermes
  // family. create_chat_completion calls the per-family overload with the active
  // tool_parser_name so the constraint matches the model's native syntax.
  ApplyToolChoiceStructuredOutput("hermes", request, sampling_params);
}

ShapedChatMessage ShapeChatMessage(
    const std::string& role, const std::string& model_output,
    std::optional<std::string> output_finish_reason,
    const ChatCompletionRequest& request, ToolParser* parser,
    ReasoningParser* reasoning_parser) {
  ShapedChatMessage shaped;
  shaped.message.role = role;

  // chat_completion/serving.py:858-866. Reasoning runs FIRST: strip the
  // chain-of-thought, then tool detection operates ONLY on the user-visible
  // content. `content_span` is what the tool parser (and a plain message) see;
  // `has_content_span` distinguishes an absent post-reasoning span (pure
  // reasoning, content=None) from an empty one.
  std::string content_span = model_output;
  bool has_content_span = true;
  std::optional<std::string> reasoning;
  if (reasoning_parser != nullptr) {
    const ExtractedReasoning er =
        reasoning_parser->extract_reasoning(model_output, request);
    reasoning = er.reasoning;
    has_content_span = er.content.has_value();
    content_span = er.content.value_or(std::string());
  }
  const auto attach_reasoning = [&]() {
    if (reasoning.has_value() && !reasoning->empty()) {
      // Same contract as every content assignment below: the non-stream path
      // splits the RAW detokenizer output, so the reasoning span needs the
      // same UTF-8 sanitization content gets (think markers are ASCII, so a
      // multi-byte sequence can never straddle the split boundary).
      shaped.message.reasoning = SanitizeUtf8(*reasoning);
    }
  };

  // chat_completion/serving.py:899-923 (the "auto" path). When tools are active
  // and a parser exists, extract; on tools_called, the finish_reason becomes
  // "tool_calls" (:936).
  if (parser != nullptr && ToolsEnabled(request)) {
    const ExtractedToolCallInformation info =
        parser->extract_tool_calls(content_span, request);
    if (info.tools_called && !info.tool_calls.empty()) {
      if (info.content.has_value()) {
        shaped.message.content = SanitizeUtf8(*info.content);
      } else {
        shaped.message.content = std::nullopt;
      }
      shaped.message.tool_calls = info.tool_calls;
      shaped.finish_reason = "tool_calls";
      attach_reasoning();
      return shaped;
    }
    // tools_called=false → fall through to a plain-content message with the
    // parser's content (the post-reasoning span).
    shaped.message.content = SanitizeUtf8(info.content.value_or(content_span));
    shaped.finish_reason = std::move(output_finish_reason);
    if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
    attach_reasoning();
    return shaped;
  }

  // No tools: plain content message (:881). With a reasoning parser, an absent
  // post-reasoning span means content=null (pure reasoning turn).
  if (reasoning_parser != nullptr && !has_content_span) {
    shaped.message.content = std::nullopt;
  } else {
    shaped.message.content = SanitizeUtf8(content_span);
  }
  shaped.finish_reason = std::move(output_finish_reason);
  if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
  attach_reasoning();
  return shaped;
}

std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser,
                                           ReasoningParser* reasoning_parser) {
  // chat_completion/serving.py:587-613. Reasoning runs BEFORE tools. The
  // reasoning parser splits this delta into a reasoning span (rides on
  // DeltaMessage.reasoning) and a post-reasoning CONTENT span; only that content
  // span is routed to the tool parser, so the tool parse never sees the thoughts.
  if (reasoning_parser != nullptr) {
    std::optional<DeltaMessage> rd = reasoning_parser->extract_reasoning_streaming(
        previous_text, current_text, delta_text, request);
    if (!rd.has_value()) return std::nullopt;  // lone marker swallowed

    const bool tools = (parser != nullptr && ToolsEnabled(request));
    if (!tools) return rd;  // reasoning and/or content deltas straight through

    const std::string content_delta = rd->content.value_or(std::string());
    const std::optional<std::string> reasoning_piece = rd->reasoning;
    if (content_delta.empty()) {
      if (reasoning_piece.has_value()) {
        DeltaMessage m;
        m.reasoning = reasoning_piece;
        return m;
      }
      return std::nullopt;
    }

    // Content-space offsets for the (stateful) tool parser, derived from the
    // reasoning split of the accumulated prefix: the content the user has seen
    // so far, without any reasoning.
    const std::string content_prev =
        reasoning_parser->extract_reasoning(previous_text, request)
            .content.value_or(std::string());
    const std::string content_curr = content_prev + content_delta;
    std::optional<DeltaMessage> td = parser->extract_tool_calls_streaming(
        content_prev, content_curr, content_delta, request);

    DeltaMessage out = td.value_or(DeltaMessage{});
    if (reasoning_piece.has_value()) out.reasoning = reasoning_piece;
    if (!td.has_value() && !reasoning_piece.has_value() &&
        !out.content.has_value()) {
      return std::nullopt;
    }
    return out;
  }

  // No reasoning parser: with a tool parser, run the streaming parse; otherwise
  // emit a plain content delta.
  if (parser != nullptr && ToolsEnabled(request)) {
    return parser->extract_tool_calls_streaming(previous_text, current_text,
                                                delta_text, request);
  }
  DeltaMessage msg;
  msg.content = delta_text;
  return msg;
}

namespace {

// The ChatCompletionRequest -> engine ParserRequest projection (the minimal
// subset parser_engine.py reads: include_reasoning, tool_choice, tools, and the
// history tool-call count). chat_completion/serving.py passes the request object
// straight to parse_delta / parse; we model only the fields the assembly path
// consumes. The body now lives next to ParserRequest itself
// (parser_engine.h ParserRequestFromChatCompletion) so the tool_parsers
// ParserEngineToolAdapter projects the request the SAME way this path does.
vllm::parser::engine::ParserRequest ToParserRequest(
    const ChatCompletionRequest& request) {
  return vllm::parser::engine::ParserRequestFromChatCompletion(request);
}

}  // namespace

std::optional<DeltaMessage> ShapeChatDeltaEngine(
    vllm::parser::engine::ParserEngine* parser, const std::string& delta_text,
    const std::vector<int>& delta_token_ids,
    const ChatCompletionRequest& request, bool finished) {
  // chat_completion/serving.py:607 — the unified parser drives BOTH reasoning
  // and tool parsing over the raw delta; no separate reasoning parser runs.
  if (parser == nullptr) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }
  const vllm::parser::engine::ParserRequest req = ToParserRequest(request);
  return parser->parse_delta(delta_text, delta_token_ids, req, finished);
}

ShapedChatMessage ShapeChatMessageEngine(
    const std::string& role, const std::string& model_output,
    std::optional<std::string> output_finish_reason,
    const ChatCompletionRequest& request,
    vllm::parser::engine::ParserEngine* parser) {
  ShapedChatMessage shaped;
  shaped.message.role = role;

  // chat_completion/serving.py:893 — parser.parse(text, request) returns the
  // (reasoning, content, tool_calls) triple with reasoning already split off.
  const vllm::parser::engine::ParserRequest req = ToParserRequest(request);
  auto [reasoning, content, tool_calls] = parser->parse(model_output, req);

  const auto attach_reasoning = [&]() {
    if (reasoning.has_value() && !reasoning->empty()) {
      shaped.message.reasoning = SanitizeUtf8(*reasoning);
    }
  };

  // :908-923 — a tool call flips finish_reason to "tool_calls"; ids come from
  // make_tool_call_id (:918) since parse() returns id-less FunctionCalls.
  if (ToolsEnabled(request) && tool_calls.has_value() && !tool_calls->empty()) {
    if (content.has_value()) {
      shaped.message.content = SanitizeUtf8(*content);
    } else {
      shaped.message.content = std::nullopt;
    }
    std::vector<ToolCall> calls;
    calls.reserve(tool_calls->size());
    for (const FunctionCall& fc : *tool_calls) {
      calls.push_back(ToolCall{make_tool_call_id(), "function", fc});
    }
    shaped.message.tool_calls = std::move(calls);
    shaped.finish_reason = "tool_calls";
    attach_reasoning();
    return shaped;
  }

  // No tool call: plain-content message carrying the reasoning span.
  shaped.message.content = SanitizeUtf8(content.value_or(model_output));
  shaped.finish_reason = std::move(output_finish_reason);
  if (!shaped.finish_reason.has_value()) shaped.finish_reason = "stop";
  attach_reasoning();
  return shaped;
}

namespace {

// chat_completion_stream_generator (serving.py:404-802) as W2's live,
// pull-based SSE source. It waits for and buffers the first result before the
// role frame, in every usage mode, mirroring upstream's first-iteration
// ordering (#1982); continuous usage reads that result's native prompt-token
// count. Subsequent calls block only on this request's collector.
class ChatSseStream final : public SseStream {
 public:
  ChatSseStream(v1::AsyncLLM& engine, v1::AsyncRequest async_request,
                std::string response_id, int64_t created, std::string model,
                ChatCompletionRequest request,
                std::unique_ptr<ToolParser> parser,
                std::unique_ptr<vllm::parser::engine::ParserEngine> engine_parser,
                std::unique_ptr<ReasoningParser> reasoning_parser,
                bool named_tool_choice, StreamUsageSelection usage)
      : engine_(engine),
        async_request_(std::move(async_request)),
        response_id_(std::move(response_id)),
        created_(created),
        model_(std::move(model)),
        request_(std::move(request)),
        parser_(std::move(parser)),
        engine_parser_(std::move(engine_parser)),
        reasoning_parser_(std::move(reasoning_parser)),
        named_tool_choice_(named_tool_choice),
        usage_(usage) {}

  ~ChatSseStream() override { abort(); }

  // Timed wait on this request's collector. On timeout with pings enabled,
  // writes a pure SSE comment frame to `ping_chunk` and returns false so next()
  // can deliver it alone (never concatenated with a data frame). On data,
  // moves into `out` and returns true.
  bool WaitOutput(RequestOutput& out, std::string& ping_chunk) {
    const int ping_s = SsePingIntervalSec();
    if (ping_s <= 0) {
      out = engine_.get_output(async_request_);
      return true;
    }
    auto ready = engine_.get_output_for(
        async_request_, std::chrono::milliseconds(ping_s * 1000));
    // Shared framing with completion: ping is a standalone comment frame.
    return AssignSseWaitResult(std::move(ready), out, ping_chunk);
  }

  bool next(std::string& chunk) override {
    if (complete_) return false;
    if (role_pending_) {
      // Upstream emits the role frame on the FIRST ENGINE RESULT, in every
      // usage mode: chat_completion/serving.py:487 builds the role chunk under
      // `if first_iteration:` inside `async for res in result_generator:`
      // (:477). Its comment at :484-486 gives the reason — "if there are
      // exceptions in the result_generator, it needs to be sent as the FIRST
      // response (by the try...catch)" — and that reason does not depend on
      // usage. So this buffering runs unconditionally (#1982).
      //
      // It was scoped to continuous usage, which needs the native prompt count
      // for the frame it attaches. The narrow reading cost two things. A
      // request that died before its first token had already been answered 200
      // with a role frame, so the error could not be the first response. And
      // `vllm/benchmarks/lib/endpoint_request_func.py:404-408` stamps TTFT on
      // the first chunk carrying a `choices` key whatever `delta.content`
      // holds, so `vllm bench serve --backend openai-chat` measured the HTTP
      // round trip to this empty frame instead of a time to first token.
      //
      // The frame itself is unchanged, including its position ahead of every
      // content frame. Only its arrival time moves, and the first TOKEN is not
      // delayed: the token that used to ride in frame two now rides in frame
      // three, at the same instant, out of the result buffered right here.
      for (;;) {
        RequestOutput response;
        if (!WaitOutput(response, chunk)) {
          // WaitOutput filled chunk with a pure SSE ping — return it first.
          return true;
        }
        prompt_tokens_ = static_cast<int>(response.prompt_token_ids.size());
        if (!response.outputs.empty() || response.finished) {
          buffered_response_ = std::move(response);
          break;
        }
      }
      role_pending_ = false;
      ChatCompletionResponseStreamChoice choice;
      choice.index = 0;
      choice.delta.role = kAssistantRole;
      choice.delta.content = "";
      choice.finish_reason = std::nullopt;
      ChatCompletionStreamResponse frame;
      frame.id = response_id_;
      frame.created = created_;
      frame.model = model_;
      frame.choices.push_back(std::move(choice));
      if (usage_.include_continuous_usage) {
        frame.usage = UsageInfo{prompt_tokens_, prompt_tokens_, 0};
      }
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";
      return true;
    }
    if (usage_pending_) {
      ChatCompletionStreamResponse frame;
      frame.id = response_id_;
      frame.created = created_;
      frame.model = model_;
      frame.usage = UsageInfo{prompt_tokens_,
                              prompt_tokens_ + previous_num_tokens_,
                              previous_num_tokens_};
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";
      usage_pending_ = false;
      done_pending_ = true;
      return true;
    }
    if (done_pending_) {
      chunk = "data: [DONE]\n\n";
      done_pending_ = false;
      complete_ = true;
      return true;
    }

    for (;;) {
      RequestOutput response;
      if (buffered_response_.has_value()) {
        response = std::move(*buffered_response_);
        buffered_response_.reset();
      } else {
        if (!WaitOutput(response, chunk)) {
          return true;  // pure ping frame
        }
      }
      prompt_tokens_ = static_cast<int>(response.prompt_token_ids.size());
      if (response.outputs.empty()) {
        if (response.finished) {
          engine_finished_ = true;
          if (usage_.include_usage) {
            usage_pending_ = true;
          } else {
            done_pending_ = true;
          }
          return next(chunk);
        }
        continue;
      }

      const CompletionOutput& output = response.outputs.front();
      const std::string delta_text = SanitizeUtf8(output.text);
      if (delta_text.empty() && output.token_ids.empty() &&
          previous_num_tokens_ == 0 && !response.finished) {
        continue;
      }
      previous_num_tokens_ += static_cast<int>(output.token_ids.size());
      const std::string current_text = previous_text_ + delta_text;
      const bool finished = output.finish_reason.has_value() || response.finished;
      if (GetRequestLogConfig().debug_stages) {
        const auto now = std::chrono::steady_clock::now();
        if (previous_num_tokens_ <= 1 || finished ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_dbg_)
                    .count() >= 1000) {
          ChatDbg(response_id_,
                  "stage=async_sse prompt_tok=" + std::to_string(prompt_tokens_) +
                      " gen_tok=" + std::to_string(previous_num_tokens_) +
                      " finished=" + std::string(finished ? "1" : "0"));
          last_dbg_ = now;
        }
      }
      std::optional<DeltaMessage> delta =
          engine_parser_ != nullptr
              ? ShapeChatDeltaEngine(
                    engine_parser_.get(), delta_text,
                    std::vector<int>(output.token_ids.begin(),
                                     output.token_ids.end()),
                    request_, finished)
              : ShapeChatDelta(previous_text_, current_text, delta_text, request_,
                               parser_.get(), reasoning_parser_.get());
      previous_text_ = current_text;

      if (delta.has_value() && delta->tool_calls.has_value() &&
          !delta->tool_calls->empty()) {
        tools_streamed_ = true;
      }
      if (!delta.has_value()) {
        if (!response.finished) continue;
        delta = DeltaMessage{};
      }

      ChatCompletionResponseStreamChoice choice;
      choice.index = output.index;
      choice.delta = std::move(*delta);
      if (response.finished) {
        choice.finish_reason = tools_streamed_ && !named_tool_choice_
                                   ? std::optional<std::string>("tool_calls")
                                   : output.finish_reason;
      } else {
        choice.finish_reason = std::nullopt;
      }

      ChatCompletionStreamResponse frame;
      frame.id = response_id_;
      frame.created = created_;
      frame.model = model_;
      frame.choices.push_back(std::move(choice));
      if (usage_.include_continuous_usage) {
        frame.usage = UsageInfo{prompt_tokens_,
                                prompt_tokens_ + previous_num_tokens_,
                                previous_num_tokens_};
      }
      chunk = "data: " + nlohmann::json(frame).dump() + "\n\n";
      if (response.finished) {
        engine_finished_ = true;
        if (usage_.include_usage) {
          usage_pending_ = true;
        } else {
          done_pending_ = true;
        }
      }
      return true;
    }
  }

  void abort() override {
    if (complete_ || engine_finished_ || aborted_) return;
    aborted_ = true;
    engine_.abort(async_request_.request_id);
  }

 private:
  v1::AsyncLLM& engine_;
  v1::AsyncRequest async_request_;
  std::string response_id_;
  int64_t created_ = 0;
  std::string model_;
  ChatCompletionRequest request_;
  std::unique_ptr<ToolParser> parser_;
  std::unique_ptr<vllm::parser::engine::ParserEngine> engine_parser_;
  std::unique_ptr<ReasoningParser> reasoning_parser_;
  bool named_tool_choice_ = false;
  StreamUsageSelection usage_;
  std::optional<RequestOutput> buffered_response_;
  int prompt_tokens_ = 0;
  bool role_pending_ = true;
  bool usage_pending_ = false;
  bool done_pending_ = false;
  bool complete_ = false;
  bool engine_finished_ = false;
  bool aborted_ = false;
  bool tools_streamed_ = false;
  int previous_num_tokens_ = 0;
  std::string previous_text_;
  std::chrono::steady_clock::time_point last_dbg_{std::chrono::steady_clock::now()};
};

}  // namespace

std::unique_ptr<ToolParser> OpenAIServingChat::MakeToolParser(
    const ChatCompletionRequest& request) const {
  if (tool_parser_name_.empty() || !ToolsEnabled(request)) return nullptr;
  // Lab/Hermes: free-form Gemma4 often emits bare `call:NAME{ARGS}` (or
  // detokenized <|tool_call>… text) without the engine's special-token IDs.
  // Prefer the text-seam gemma4 tool parser so both shapes extract to tool_calls.
  // Engine-backed gemma4 remains available for other entrypoints via
  // get_parser_engine("gemma4").
  if (tool_parser_name_ == "gemma4") {
    return get_tool_parser("gemma4");
  }
  // An engine-backed name is served by MakeParserEngine, not the legacy seam.
  if (vllm::parser::get_parser_engine(tool_parser_name_) != nullptr) {
    return nullptr;
  }
  return get_tool_parser(tool_parser_name_);
}

std::unique_ptr<vllm::parser::engine::ParserEngine>
OpenAIServingChat::MakeParserEngine(const ChatCompletionRequest& request) const {
  if (tool_parser_name_.empty() || !ToolsEnabled(request)) return nullptr;
  if (tool_parser_name_ == "gemma4") {
    return nullptr;  // OpenAI chat uses text-seam MakeToolParser above.
  }
  // parser_manager get_parser_engine returns nullptr for a name that is not an
  // engine-backed format, so the legacy MakeToolParser seam handles those. This
  // is the name-selected dispatch swap: engine-backed (qwen3/seed_oss/kimi_k2)
  // formats drive the unified ParserEngine, every other name keeps the legacy
  // tool_parsers path unchanged (parser/parser_manager.py:76).
  return vllm::parser::get_parser_engine(tool_parser_name_);
}

std::unique_ptr<ReasoningParser> OpenAIServingChat::MakeReasoningParser() const {
  if (reasoning_parser_name_.empty()) return nullptr;
  return get_reasoning_parser(reasoning_parser_name_);
}

OpenAIServingChat::OpenAIServingChat(v1::LLMEngine& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn,
                                     std::string tool_parser_name,
                                     std::string reasoning_parser_name,
                                     bool enable_force_include_usage)
    : sync_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)),
      tool_parser_name_(std::move(tool_parser_name)),
      reasoning_parser_name_(std::move(reasoning_parser_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

OpenAIServingChat::OpenAIServingChat(v1::AsyncLLM& engine,
                                     std::string served_model_name,
                                     ChatPromptFn prompt_fn,
                                     std::string tool_parser_name,
                                     std::string reasoning_parser_name,
                                     bool enable_force_include_usage)
    : async_engine_(&engine),
      served_model_name_(std::move(served_model_name)),
      prompt_fn_(std::move(prompt_fn)),
      tool_parser_name_(std::move(tool_parser_name)),
      reasoning_parser_name_(std::move(reasoning_parser_name)),
      enable_force_include_usage_(enable_force_include_usage) {}

std::size_t OperatorMaxPromptChars() {
  const char* e = std::getenv("VT_SERVER_MAX_PROMPT_CHARS");
  if (e && e[0]) return static_cast<std::size_t>(std::strtoull(e, nullptr, 10));
  return 0;
}

ChatCompletionResult OpenAIServingChat::create_chat_completion(
    const ChatCompletionRequest& request) {
  // request_id = f"chatcmpl-{...}" (chat_completion/serving.py:268); created =
  // int(time.time()) (:416 / :816).
  const std::string request_id =
      "chatcmpl-" + std::to_string(request_counter_.fetch_add(1));
  const auto created_time = static_cast<int64_t>(std::time(nullptr));
  const std::string model_name =
      request.model.has_value() ? *request.model : served_model_name_;
  const StreamUsageSelection usage = ShouldIncludeUsage(
      request.stream_options, enable_force_include_usage_);

  // Build the prompt from messages via the seam (add_generation_prompt is the
  // upstream default True). The tools are passed through so the chat template's
  // `{% if tools %}` branch renders the function schemas (upstream
  // apply_chat_template(..., tools=...)); empty when tools are absent/disabled.
  const std::vector<ChatCompletionToolsParam> tools =
      ToolsEnabled(request) ? *request.tools
                            : std::vector<ChatCompletionToolsParam>{};
  // #1681: the request's chat_template_kwargs reach the renderer here, which is
  // the only place they can. Upstream builds them in
  // ChatCompletionRequest.build_chat_params (chat_completion/protocol.py:545-556)
  // and hands them to the renderer the same way.
  const std::string prompt =
      prompt_fn_(request.messages, /*add_generation_prompt=*/true, tools,
                 request.chat_template_kwargs);

  const int max_tok_log =
      request.max_completion_tokens.has_value()
          ? *request.max_completion_tokens
          : request.max_tokens.value_or(-1);
  std::string roles_summary;
  for (size_t i = 0; i < request.messages.size(); ++i) {
    if (i) roles_summary += ",";
    roles_summary += request.messages[i].role;
    size_t clen = request.messages[i].content.has_value()
                      ? request.messages[i].content->size()
                      : 0;
    roles_summary += "(" + std::to_string(clen) + ")";
  }
  LogRequestReceived(request_id, "/v1/chat/completions", model_name, request.stream,
                     max_tok_log, static_cast<int>(request.messages.size()),
                     static_cast<int>(tools.size()), prompt.size(), prompt,
                     roles_summary);
  ChatDbg(request_id, "stage=templated prompt_chars=" + std::to_string(prompt.size()) +
                           " stream=" + std::string(request.stream ? "1" : "0"));
  const auto req_t0 = std::chrono::steady_clock::now();

  // SERVE-REQUEST-LENGTH-GUARD. The pre-tokenization bound on the RENDERED
  // prompt, template and tools included, which is where vLLM's _text_len_check
  // runs: render_messages(tokenize=False), then apply_pre_tokenization
  // (vllm/renderers/params.py:342-370,386-399 @ e126687a9a). The bound is
  // max_model_len * MaxTokenBytes(), vLLM's max_input_tokens *
  // max_chars_per_token, from Tokenizer::MaxPromptBytes, the same derivation
  // ApiServer's #1541 guard uses. That guard sums only messages[].content before
  // the template renders, so tool descriptions and schemas, tool_calls
  // arguments and template framing reach this point unmeasured. The bound reads
  // this handler's own engine, so the C ABI (vllm_chat / vllm_chat_stream) and
  // RunBatch get it as well as HTTP. Differences from vLLM: it counts bytes, not
  // characters, and uses max_model_len, not max_model_len minus the requested
  // output tokens. A prompt of B bytes costs at least B / MaxTokenBytes()
  // tokens, so the bound refuses no prompt that fits in max_model_len, with one
  // exception: a SentencePiece tokenizer with fuse_unk and no byte fallback
  // encodes a run of unknown characters of any length as ONE <unk>
  // (tokenizer.cpp, the fuse_unk_ branches of the SentencePiece encode), so such
  // a prompt can fit in tokens and still exceed the byte bound. vLLM's
  // max_chars_per_token bound has the same exception. The check runs before
  // every encode this handler can reach: the engine encode, the beam-search
  // encode and the multimodal seam below.
  {
    const v1::InputProcessor& processor =
        async_engine_ != nullptr ? async_engine_->input_processor()
                                 : sync_engine_->input_processor();
    const size_t max_prompt_bytes = processor.max_prompt_bytes();
    if (max_prompt_bytes > 0 && prompt.size() > max_prompt_bytes) {
      const int64_t max_model_len = processor.max_model_len();
      std::ostringstream err;
      err << "This model's maximum context length is " << max_model_len
          << " tokens. However, your prompt contains " << prompt.size()
          << " bytes (more than " << max_prompt_bytes
          << " bytes, which is the upper bound for " << max_model_len
          << " input tokens). Please reduce the length of the input prompt.";
      LogRequestError(request_id, "/v1/chat/completions", err.str());
      throw vllm::v1::InputValidationError(err.str());
    }
  }

  // VT_SERVER_MAX_PROMPT_CHARS is an OPTIONAL operator ceiling on the rendered
  // prompt, in bytes; unset or 0 sets none (ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN).
  // It used to default to a fixed 200,000, which is not derived from the
  // context and refused a 262,144-token server's prompts above ~48k tokens.
  // The default bound is the derived one above; vLLM has no such variable.
  // The refusal is the request's fault: InputValidationError is HTTP 400
  // BadRequestError (error_response.py:39-41).
  // VT_SERVER_MAX_NEW_TOKENS keeps its 4096 default; its divergence from vLLM is
  // ISSUE-LOCAL-01M37A3S7N7GSZC37JH515QXFE.
  const size_t max_prompt_chars = OperatorMaxPromptChars();
  static const int kMaxNewTokensCap = [] {
    const char* e = std::getenv("VT_SERVER_MAX_NEW_TOKENS");
    if (e && e[0]) return std::atoi(e);
    return 4096;  // 0 disables
  }();
  if (max_prompt_chars > 0 && prompt.size() > max_prompt_chars) {
    std::ostringstream err;
    err << "prompt length " << prompt.size()
        << " bytes exceeds the operator-set limit VT_SERVER_MAX_PROMPT_CHARS="
        << max_prompt_chars
        << " on the rendered chat prompt. Shorten the messages or tools, or "
           "ask the operator to raise or unset the limit. The request is "
           "refused, not truncated.";
    LogRequestError(request_id, "/v1/chat/completions", err.str());
    throw vllm::v1::InputValidationError(err.str());
  }
  if (prompt.size() > 32000) {
    ChatDbg(request_id,
            "note=large_prompt prefix_caching=ON — first request pays full prefill; "
            "identical system+tools prefix on later turns should hit APC");
  }

  // ── Multimodal (MM-SERVE-ENGINE) ─────────────────────────────────────────
  // When the mm seam is set AND a message carries a mm content part, decode +
  // route the media through the mm processor and carry the placeholder-EXPANDED
  // MultiModalInputs to the engine mm overload. Unset seam OR a text-only
  // request leaves mm_inputs empty and every path below is byte-identical to the
  // text-only server (the RED-line inertness). Both response modes carry the
  // expanded prompt and image features to the same engine overload.
  std::optional<multimodal::MultiModalInputs> mm_inputs;
  if (mm_chat_fn_ || mm_request_chat_fn_) {
    bool has_mm = false;
    for (const ChatMessage& m : request.messages) {
      if (HasMultiModalParts(m)) {
        has_mm = true;
        break;
      }
    }
    if (has_mm) {
      mm_inputs = mm_request_chat_fn_
          ? mm_request_chat_fn_(request.messages, tools, request.chat_template_kwargs)
          : mm_chat_fn_(request.messages);
    }
  }

  // ── use_beam_search (chat_completion/serving.py:319-343) ─────────────────
  // Route the rendered chat prompt through the merged BeamSearch driver;
  // beam_width == n, returns the n best beams as assistant choices. The
  // production server holds an AsyncLLM → BeamSearchAsync (online.py); the sync
  // LLMEngine seam → BeamSearch (offline.py). Both call the SAME scoring, so the
  // choices are identical. Streaming beam search is rejected as upstream.
  if (request.use_beam_search) {
    if (request.stream) {
      throw std::runtime_error(
          "Streaming is not currently supported with beam search");
    }
    if (beam_tokenizer_ == nullptr ||
        (async_engine_ == nullptr && sync_engine_ == nullptr)) {
      throw std::runtime_error(
          "beam search requires an engine and a tokenizer");
    }
    const int max_tok =
        request.max_completion_tokens.has_value()
            ? *request.max_completion_tokens
            : request.max_tokens.value_or(16);
    const BeamSearchParams params =
        request.to_beam_search_params(max_tok, &default_sampling_params_);
    num_beam_prompt_encodes_.fetch_add(1, std::memory_order_relaxed);
    const std::vector<int32_t> prompt_ids = beam_tokenizer_->Encode(prompt);
    const BeamSearchOutput beams =
        async_engine_ != nullptr
            ? BeamSearchAsync(*async_engine_, prompt_ids, params,
                              beam_eos_token_id_, beam_tokenizer_)
            : BeamSearch(*sync_engine_, prompt_ids, params, beam_eos_token_id_,
                         beam_tokenizer_);

    ChatCompletionResponse response;
    response.id = request_id;
    response.created = created_time;
    response.model = model_name;
    int num_generated_tokens = 0;
    for (std::size_t i = 0; i < beams.sequences.size(); ++i) {
      const BeamSearchSequence& beam = beams.sequences[i];
      ChatCompletionResponseChoice choice;
      choice.index = static_cast<int>(i);
      choice.message.role = kAssistantRole;
      choice.message.content = SanitizeUtf8(beam.text.value_or(""));
      choice.finish_reason = beam.finish_reason;
      response.choices.push_back(std::move(choice));
      if (beam.tokens.size() > prompt_ids.size()) {
        num_generated_tokens +=
            static_cast<int>(beam.tokens.size() - prompt_ids.size());
      }
    }
    const int num_prompt_tokens = static_cast<int>(prompt_ids.size());
    response.usage.prompt_tokens = num_prompt_tokens;
    response.usage.completion_tokens = num_generated_tokens;
    response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

    ChatCompletionResult result;
    result.streaming = false;
    result.response = std::move(response);
    return result;
  }

  // One tool parser per request (the streaming parse is stateful); null when the
  // request has no tools (or the parser is disabled). When `tool_parser_name_`
  // is an engine-backed format (qwen3/seed_oss/kimi_k2) `parser` stays null and
  // `engine_parser` is used instead — the name-selected 0.26 dispatch swap.
  std::unique_ptr<ToolParser> parser = MakeToolParser(request);
  std::unique_ptr<vllm::parser::engine::ParserEngine> engine_parser =
      MakeParserEngine(request);
  // One reasoning parser per request (streaming may be stateful, olmo3); null
  // when disabled (empty reasoning_parser_name_). Independent of tools. The
  // engine-backed parser does reasoning itself, so it is bypassed there.
  std::unique_ptr<ReasoningParser> reasoning_parser =
      engine_parser != nullptr ? nullptr : MakeReasoningParser();
  const bool named_tool_choice = IsNamedToolChoice(request);

  SamplingParams sampling_params =
      request.to_sampling_params(std::nullopt, &default_sampling_params_);
  if (kMaxNewTokensCap > 0) {
    const int before = sampling_params.max_tokens.value_or(kMaxNewTokensCap);
    if (before > kMaxNewTokensCap) {
      ChatDbg(request_id, "clamp max_tokens " + std::to_string(before) + " -> " +
                              std::to_string(kMaxNewTokensCap));
      sampling_params.max_tokens = kMaxNewTokensCap;
    }
  }

  // tool_choice -> a structural-tag constraint (structured_outputs.structural_tag)
  // before add_request, built for the ACTIVE tool parser family
  // (tool_parser_name_) so required/named force the model's OWN native tool
  // syntax (not always Hermes). auto -> a LAZY tag (free text until the family's
  // begin marker, then the call is constrained — NOT forced); required/named ->
  // a FORCED tag (a valid, parser-extractable native call from token 0); an
  // unmapped family or none / no tools: no-op. Mirrors upstream's per-model
  // xgrammar StructuralTag (chat_completion/serving.py ->
  // tool_parsers/structural_tag_registry.py::get_model_structural_tag).
  ApplyToolChoiceStructuredOutput(tool_parser_name_, request, sampling_params);

  // Single prompt → sub_request_id == request_id (chat_completion/serving.py:293).
  const std::string engine_request_id = request_id;

  if (request.stream) {
    ChatDbg(request_id, "stage=stream_begin engine=" +
                            std::string(async_engine_ != nullptr ? "async" : "sync"));
    if (async_engine_ != nullptr) {
      v1::AsyncRequest async_request = mm_inputs.has_value()
          ? async_engine_->add_request(engine_request_id, std::move(*mm_inputs),
                                       std::move(sampling_params), request.priority)
          : async_engine_->add_request(engine_request_id, prompt,
                                       std::move(sampling_params), request.priority);
      ChatDbg(request_id, "stage=async_queued");
      ChatCompletionResult result;
      result.streaming = true;
      try {
        result.sse_stream = std::make_shared<ChatSseStream>(
            *async_engine_, async_request, request_id, created_time, model_name,
            request, std::move(parser), std::move(engine_parser),
            std::move(reasoning_parser), named_tool_choice, usage);
      } catch (...) {
        async_engine_->abort(async_request.request_id);
        throw;
      }
      return result;
    }

    // ── Streaming (chat_completion_stream_generator, :404) ────────────────
    ChatCompletionResult result;
    result.streaming = true;

    // Content / tool-call deltas, then the finish chunk (:559-716). The tool
    // parser (when present) drives per-delta shaping; `previous_text` is the
    // accumulated output the streaming parser re-parses each step (:615).
    int previous_num_tokens = 0;
    int num_prompt_tokens = 0;
    std::string previous_text;
    bool tools_streamed = false;
    bool role_emitted = false;
    if (sync_engine_ == nullptr) {
      throw std::runtime_error("chat handler has no engine");
    }
    ChatDbg(request_id, "stage=sync_add_request (prefill may take a while on long prompts)");
    if (mm_inputs.has_value()) {
      sync_engine_->add_request(engine_request_id, std::move(*mm_inputs),
                               std::move(sampling_params), request.priority);
    } else {
      sync_engine_->add_request(engine_request_id, prompt,
                               std::move(sampling_params), request.priority);
    }
    ChatDbg(request_id, "stage=sync_step_loop");
    int step_i = 0;
    auto last_prog = std::chrono::steady_clock::now();
    while (sync_engine_->has_unfinished_requests()) {
      for (const RequestOutput& res : sync_engine_->step()) {
        if (res.request_id != engine_request_id) continue;
        num_prompt_tokens = static_cast<int>(res.prompt_token_ids.size());
        ++step_i;
        const auto now = std::chrono::steady_clock::now();
        if (step_i == 1 || previous_num_tokens == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prog)
                    .count() >= 1000) {
          ChatDbg(request_id,
                  "stage=sync_step n=" + std::to_string(step_i) +
                      " prompt_tok=" + std::to_string(num_prompt_tokens) +
                      " gen_tok=" + std::to_string(previous_num_tokens) +
                      " finished=" + std::string(res.finished ? "1" : "0"));
          last_prog = now;
        }
        if (!role_emitted) {
          role_emitted = true;
          ChatCompletionResponseStreamChoice role_choice;
          role_choice.index = 0;
          role_choice.delta.role = kAssistantRole;
          role_choice.delta.content = "";
          role_choice.finish_reason = std::nullopt;
          ChatCompletionStreamResponse role_chunk;
          role_chunk.id = request_id;
          role_chunk.created = created_time;
          role_chunk.model = model_name;
          role_chunk.choices.push_back(std::move(role_choice));
          if (usage.include_continuous_usage) {
            role_chunk.usage = UsageInfo{num_prompt_tokens,
                                         num_prompt_tokens, 0};
          }
          result.sse_chunks.push_back(
              "data: " + nlohmann::json(role_chunk).dump() + "\n\n");
        }
        for (const CompletionOutput& output : res.outputs) {
          // SanitizeUtf8 (see serving_utils.h): raw-byte deltas may carry an
          // invalid/split multibyte run that would make dump() below throw.
          const std::string delta_text = SanitizeUtf8(output.text);
          // :579-585 chunked-prefill: skip empty chunks.
          if (delta_text.empty() && output.token_ids.empty() &&
              previous_num_tokens == 0) {
            continue;
          }
          previous_num_tokens += static_cast<int>(output.token_ids.size());
          const bool finished = output.finish_reason.has_value();

          const std::string current_text = previous_text + delta_text;
          std::optional<DeltaMessage> delta_message =
              engine_parser != nullptr
                  ? ShapeChatDeltaEngine(
                        engine_parser.get(), delta_text,
                        std::vector<int>(output.token_ids.begin(),
                                         output.token_ids.end()),
                        request, finished)
                  : ShapeChatDelta(previous_text, current_text, delta_text,
                                   request, parser.get(), reasoning_parser.get());
          previous_text = current_text;

          // :598-599 — a tool-call delta flips the finish_reason to tool_calls.
          if (delta_message.has_value() && delta_message->tool_calls.has_value() &&
              !delta_message->tool_calls->empty()) {
            tools_streamed = true;
          }

          // :624-633 — a null delta (parser withholding) skips the chunk unless
          // this is the terminal step, where an empty delta carries the finish.
          if (!delta_message.has_value()) {
            if (!finished) continue;
            delta_message = DeltaMessage{};
          }

          ChatCompletionResponseStreamChoice choice;
          choice.index = 0;
          choice.delta = std::move(*delta_message);
          if (finished) {
            // :688-693 — "tool_calls" for auto/required, else the model's own.
            if (tools_streamed && !named_tool_choice) {
              choice.finish_reason = "tool_calls";
            } else {
              choice.finish_reason = output.finish_reason.value_or("stop");
            }
          } else {
            choice.finish_reason = std::nullopt;
          }

          ChatCompletionStreamResponse chunk;
          chunk.id = request_id;
          chunk.created = created_time;
          chunk.model = model_name;
          chunk.choices.push_back(std::move(choice));
          if (usage.include_continuous_usage) {
            chunk.usage = UsageInfo{num_prompt_tokens,
                                    num_prompt_tokens + previous_num_tokens,
                                    previous_num_tokens};
          }
          result.sse_chunks.push_back(
              "data: " + nlohmann::json(chunk).dump() + "\n\n");
        }
      }
    }
    if (usage.include_usage) {
      ChatCompletionStreamResponse usage_chunk;
      usage_chunk.id = request_id;
      usage_chunk.created = created_time;
      usage_chunk.model = model_name;
      usage_chunk.usage =
          UsageInfo{num_prompt_tokens,
                    num_prompt_tokens + previous_num_tokens,
                    previous_num_tokens};
      result.sse_chunks.push_back(
          "data: " + nlohmann::json(usage_chunk).dump() + "\n\n");
    }
    result.sse_chunks.push_back("data: [DONE]\n\n");
    ChatDbg(request_id, "stage=stream_done prompt_tok=" +
                            std::to_string(num_prompt_tokens) +
                            " gen_tok=" + std::to_string(previous_num_tokens));
    {
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - req_t0)
              .count();
      LogRequestFinished(request_id, num_prompt_tokens, previous_num_tokens, "stream",
                         elapsed, previous_text);
    }
    return result;
  }

  // ── Non-streaming (chat_completion_full_generator, :804) ────────────────
  // With mm inputs, drive the engine mm overload (placeholder-expanded prompt +
  // mm_features); otherwise the text-only string overload byte-identically. The
  // mm forward on the GPU worker consumes the mm_features (MM-SERVE-E2E).
  ChatDbg(request_id, "stage=generate_blocking begin (prefill+decode; long prompts stall here)");
  const RequestOutput final_res =
      mm_inputs.has_value()
          ? (async_engine_ != nullptr
                 ? async_engine_->generate(std::move(*mm_inputs),
                                           std::move(sampling_params),
                                           engine_request_id, request.priority)
                 : sync_engine_->generate(std::move(*mm_inputs),
                                          std::move(sampling_params),
                                          engine_request_id, request.priority))
          : (async_engine_ != nullptr
                 ? async_engine_->generate(prompt, std::move(sampling_params),
                                           engine_request_id, request.priority)
                 : sync_engine_->generate(prompt, std::move(sampling_params),
                                          engine_request_id, request.priority));
  ChatDbg(request_id, "stage=generate_blocking done outputs=" +
                          std::to_string(final_res.outputs.size()) +
                          " prompt_tok=" +
                          std::to_string(final_res.prompt_token_ids.size()));

  ChatCompletionResponse response;
  response.id = request_id;
  response.created = created_time;
  response.model = model_name;

  int num_generated_tokens = 0;
  // SAMPLE-BEST-OF: keep the top-n children by cumulative logprob when best_of >
  // n. Guarded on request.best_of so the default (and plain n>1) path binds
  // `outs` to final_res.outputs with NO copy/re-rank (child indices preserved).
  //
  // NO STREAM DOWNGRADE HERE, and that is DELIBERATE, not an omission.
  // /v1/completions downgrades `best_of != n` + `stream` to a single-frame
  // non-streamed response (serving_completion.cpp), mirroring
  // serving_completion.py:253-260 @ 56e96b37e4^. Chat has no such rule
  // upstream: at that same revision ChatCompletionRequest DOES carry `best_of`
  // (protocol.py:568, class at :528) and DOES forward it to SamplingParams
  // (:892), yet serving_chat.py:355 branches on the RAW `if request.stream:`
  // with no best_of and no beam-search guard at all. Only the completions route
  // ever downgraded, so adding one here would invent a rule vLLM declined to
  // write. The asymmetry is gated in test_api_server.cpp and recorded in
  // .agents/specs/async-parallel-sampling.md.
  std::vector<CompletionOutput> selected_outputs;
  const bool trim_best_of =
      request.best_of.has_value() && *request.best_of > request.n.value_or(1);
  const std::vector<CompletionOutput>& outs =
      trim_best_of
          ? (selected_outputs =
                 SelectBestOf(final_res.outputs, request.n.value_or(1)))
          : final_res.outputs;
  for (const CompletionOutput& output : outs) {
    ChatCompletionResponseChoice choice;
    choice.index = output.index;
    // Tool shaping (:857-966): with a parser + active tools, attach tool_calls
    // and set finish_reason="tool_calls"; else a plain-content assistant message
    // (finish_reason = output.finish_reason or "stop", :956-960).
    ShapedChatMessage shaped =
        engine_parser != nullptr
            ? ShapeChatMessageEngine(kAssistantRole, output.text,
                                     output.finish_reason, request,
                                     engine_parser.get())
            : ShapeChatMessage(kAssistantRole, output.text, output.finish_reason,
                               request, parser.get(), reasoning_parser.get());
    choice.message = std::move(shaped.message);
    // logprobs (chat_completion/serving.py:875-887): when request.logprobs is
    // set, attach the ChatCompletionLogProbs payload built from the generated
    // tokens' SampleLogprobs (top_logprobs count == request.top_logprobs).
    if (request.logprobs && output.logprobs.has_value()) {
      choice.logprobs = BuildChatLogprobs(output.token_ids, *output.logprobs,
                                          request.top_logprobs);
    }
    choice.finish_reason = std::move(shaped.finish_reason);
    response.choices.push_back(std::move(choice));
    num_generated_tokens += static_cast<int>(output.token_ids.size());
  }

  // prompt_logprobs (chat_completion/serving.py:1070): TOP-LEVEL on the chat
  // response, not per choice — one rendered prompt is shared by every choice.
  response.prompt_logprobs = final_res.prompt_logprobs;
  ClampPromptLogprobs(response.prompt_logprobs);

  const int num_prompt_tokens =
      static_cast<int>(final_res.prompt_token_ids.size());
  response.usage.prompt_tokens = num_prompt_tokens;
  response.usage.completion_tokens = num_generated_tokens;
  response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

  {
    std::string finish = response.choices.empty()
                             ? "?"
                             : response.choices[0].finish_reason.value_or("?");
    std::string out_text;
    if (!response.choices.empty() && response.choices[0].message.content.has_value()) {
      out_text = *response.choices[0].message.content;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - req_t0)
            .count();
    LogRequestFinished(request_id, num_prompt_tokens, num_generated_tokens, finish,
                       elapsed, out_text);
  }

  ChatCompletionResult result;
  result.streaming = false;
  result.response = std::move(response);
  return result;
}

}  // namespace vllm::entrypoints::openai
