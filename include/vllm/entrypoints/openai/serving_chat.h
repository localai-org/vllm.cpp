// Ported from: vllm/entrypoints/openai/chat_completion/serving.py @ e24d1b24
// (OpenAIServingChat.create_chat_completion + chat_completion_stream_generator +
// chat_completion_full_generator).
//
// SCOPE (M3.1 Task 2 / T0): the OpenAI /v1/chat/completions serving logic
// DECOUPLED from HTTP. Same shape as OpenAIServingCompletion, with the chat
// specifics:
//   - the prompt is built from `messages` via a SEAM (see ChatPromptFn). At T0
//     the real chat-template renderer (Task 3 / M3.2) is not wired, so the
//     default is a SIMPLE fallback join ("<role>: <content>\n" per message +
//     an "assistant:" generation prompt). Task 3 swaps a template renderer in
//     by constructing the handler with a custom ChatPromptFn — this is the
//     Task-3 integration point.
//   - streaming cadence: FIRST chunk = role delta ({role:"assistant",
//     content:""}), THEN content deltas ({content: piece}), THEN the finish
//     chunk (last content delta + finish_reason), THEN `data: [DONE]\n\n`
//     (chat_completion/serving.py:485-520, :663-738, :802).
//
// DEFERRED (marked; matches upstream): tools / tool_choice / grammars
// (M3.3/M3.4); reasoning parser; logprobs payload; echo; n > 1; beam search;
// LoRA; multimodal.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/multimodal/inputs.h"  // multimodal::MultiModalInputs (mm seam)
#include "vllm/entrypoints/openai/serving_completion.h"  // SseStream
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/parser/parser_manager.h"  // engine-backed ParserEngine dispatch
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm {
namespace tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h (beam-search prompt tok/detok)
}  // namespace tok
}  // namespace vllm

namespace vllm::entrypoints::openai {

// The result of create_chat_completion. Mirrors upstream's
//   AsyncGenerator[str] | ChatCompletionResponse
// return union (chat_completion/serving.py:233).
struct ChatCompletionResult {
  bool streaming = false;
  std::optional<ChatCompletionResponse> response;   // set when !streaming
  std::vector<std::string> sse_chunks;              // set when streaming
  std::shared_ptr<SseStream> sse_stream;             // live AsyncLLM path
};

// The chat-prompt SEAM: messages + add_generation_prompt + tools → the prompt
// string the completion path tokenizes. The M3.2 chat-template renderer injects
// here (MakeChatTemplatePromptFn); the default is DefaultChatPromptFallback. The
// `tools` arg is what upstream passes to apply_chat_template(..., tools=...) so
// the template's `{% if tools %}` branch renders the function schemas
// (chat_completion/serving.py → chat_utils.apply_hf_chat_template).
// The fourth argument is the request's `chat_template_kwargs`
// (chat_completion/protocol.py:341), merged over the server default by the
// renderer. It is a distinct parameter rather than server state because the
// value is per REQUEST: both competitor serve recipes for the Qwen3.8 family
// send {"chat_template_kwargs":{"enable_thinking":false}}, and before #1681
// there was no way for that to reach the template at all.
using ChatPromptFn = std::function<std::string(
    const std::vector<ChatMessage>&, bool,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&)>;

// The MULTIMODAL chat SEAM (MM-SERVE-ENGINE): given the request messages, decode
// + route any mm content parts (image_url / input_audio) through the mm
// processor and return the placeholder-EXPANDED MultiModalInputs (prompt ids +
// mm_features) to hand to the engine mm add_request/generate overload; nullopt
// when no message carries a mm part. This is the injection point where the model
// tokenizer + the qwen3vl / whisper processors live (the production/E2E wiring
// constructs it); the DEFAULT is unset, so a server without it renders the
// text-only path byte-identically (the mm parts fall back to the joined-text
// content, exactly the MM-SERVE-PARSE behavior). Mirrors upstream's serving
// layer building the MultiModalDataDict and passing it to the engine alongside
// the rendered prompt.
using MultiModalChatFn = std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&)>;

// Request-aware counterpart: forwards the same enabled tools and template
// options as the text renderer. Legacy architecture callbacks remain supported.
using MultiModalRequestChatFn = std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&)>;

// The T0 fallback template (marked seam). Concatenates "<role>: <content>\n"
// for each message; when add_generation_prompt, appends "assistant:". Ignores
// `tools` (the fallback is not a model template). Exposed for unit testing.
// `chat_template_kwargs` is accepted and ignored, like `tools`: the fallback is
// not a model template and has no Jinja variables to bind.
std::string DefaultChatPromptFallback(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt,
    const std::vector<ChatCompletionToolsParam>& tools = {},
    const nlohmann::ordered_json& chat_template_kwargs =
        nlohmann::ordered_json::object());

// Whether tool extraction is active for `request`: tools present (non-empty) and
// tool_choice is not the explicit "none" (chat_completion/serving.py:896-905 —
// the "auto/required/named" branches). tool_choice defaults to auto when tools
// are present (protocol.py:828-831).
bool ToolsEnabled(const ChatCompletionRequest& request);

// Non-stream tool shaping (chat_completion/serving.py:857-966). Given the model
// output text + its finish_reason, produce the response choice's message + the
// finish_reason. When `parser` is non-null AND ToolsEnabled(request), run
// extract_tool_calls: on tools_called, attach tool_calls (+ the leading content)
// and set finish_reason="tool_calls"; otherwise a plain-content message.
//
// REASONING (chat_completion/serving.py:858-866): when `reasoning_parser` is
// non-null it runs FIRST: extract_reasoning splits `model_output` into a
// reasoning span (attached to message.reasoning) and the remaining CONTENT, and
// only that content is then handed to the tool parser. Order: reasoning strips
// the chain-of-thought, then tool extraction runs over what the user actually
// sees, exactly as upstream `parser.parse()` returns (reasoning, content,
// tool_calls) with reasoning removed before tool detection.
struct ShapedChatMessage {
  ChatMessage message;
  std::optional<std::string> finish_reason;
};
ShapedChatMessage ShapeChatMessage(const std::string& role,
                                   const std::string& model_output,
                                   std::optional<std::string> output_finish_reason,
                                   const ChatCompletionRequest& request,
                                   ToolParser* parser,
                                   ReasoningParser* reasoning_parser = nullptr);

// Per-delta stream tool shaping (chat_completion/serving.py:589-613). When
// `parser` is non-null AND ToolsEnabled(request), drive the STATEFUL streaming
// parser (extract_tool_calls_streaming) → a DeltaMessage or nullopt (withhold).
// Otherwise a plain-content DeltaMessage carrying `delta_text`.
//
// REASONING: when `reasoning_parser` is non-null it runs FIRST on the raw delta
// (extract_reasoning_streaming). The reasoning span rides on the returned
// DeltaMessage.reasoning; the post-reasoning CONTENT span (once </think> passes)
// is what feeds the tool parser, with content-space previous/current offsets
// derived from the reasoning split so the tool parse never sees the thoughts.
std::optional<DeltaMessage> ShapeChatDelta(const std::string& previous_text,
                                           const std::string& current_text,
                                           const std::string& delta_text,
                                           const ChatCompletionRequest& request,
                                           ToolParser* parser,
                                           ReasoningParser* reasoning_parser = nullptr);

// ENGINE-BACKED per-delta stream shaping (chat_completion/serving.py:607 —
// `parser.parse_delta(delta_text, delta_token_ids, request, finished)`). The
// 0.26 replacement for the legacy ShapeChatDelta seam: when an engine-backed
// tool-call parser (qwen3 / seed_oss / kimi_k2) is selected, the unified
// ParserEngine consumes the RAW (delta_text, delta_token_ids) stream and emits
// the streamed DeltaMessage (reasoning + content + tool-call name-first then
// argument deltas), or nullopt when it is withholding. The ParserEngine does
// BOTH reasoning and tool parsing, so no separate ReasoningParser runs on this
// path (mirrors vLLM's composed Parser). `finished` flushes any held-back tail.
std::optional<DeltaMessage> ShapeChatDeltaEngine(
    vllm::parser::engine::ParserEngine* parser, const std::string& delta_text,
    const std::vector<int>& delta_token_ids,
    const ChatCompletionRequest& request, bool finished);

// ENGINE-BACKED non-stream shaping (chat_completion/serving.py:893 —
// `parser.parse(output.text, request)` -> (reasoning, content, tool_calls)).
// The one-shot analogue of ShapeChatMessage for the engine-backed parser. On a
// tool call: attach tool_calls (ids from make_tool_call_id, :918) + leading
// content and set finish_reason="tool_calls"; else a plain-content assistant
// message carrying the reasoning span.
ShapedChatMessage ShapeChatMessageEngine(
    const std::string& role, const std::string& model_output,
    std::optional<std::string> output_finish_reason,
    const ChatCompletionRequest& request,
    vllm::parser::engine::ParserEngine* parser);

// Ported from: vllm/tool_parsers/structural_tag_registry.py @ e24d1b24
// (get_hermes_structural_tag:237-269 + _hermes_tool_tags:213-234).
//
// THIN WRAPPERS over the per-family STRUCTURAL-TAG registry
// (tool_parsers/structural_tags.h): both delegate to the "hermes" family and are
// kept only for source compatibility with existing callers/tests. New code
// should call ToolChoiceStructuralTagSpecFor(tool_parser_name, request) /
// ApplyToolChoiceStructuredOutput(tool_parser_name, request, sampling_params) so
// the DECODE constraint matches the ACTIVE model family's native tool syntax
// instead of always Hermes. See structural_tags.h for the native spec shape, the
// per-family coverage table and the tool_choice (auto/required/named) semantics.
std::optional<nlohmann::json> ToolChoiceStructuralTagSpec(
    const ChatCompletionRequest& request);

// Thin wrapper: applies the "hermes" family structural tag onto
// `sampling_params` (structured_outputs.structural_tag). create_chat_completion
// itself calls the per-family overload (structural_tags.h) with the active
// tool_parser_name. Kept for source compatibility.
void ApplyToolChoiceStructuredOutput(const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params);

// VT_SERVER_MAX_PROMPT_CHARS: the OPTIONAL operator ceiling, in bytes, on the
// rendered chat prompt. Unset, empty or 0 returns 0, which means NO fixed
// ceiling: the only default bound is the derived one, max_model_len *
// Tokenizer::MaxTokenBytes() (ISSUE-LOCAL-01M37A34NTK8A98KYWA5SA5GNN). vLLM has
// no such variable. Read on every call, so one process can exercise both arms.
std::size_t OperatorMaxPromptChars();

class OpenAIServingChat {
 public:
  // `prompt_fn` defaults to DefaultChatPromptFallback (the M3.2 seam).
  // `tool_parser_name` selects the tool-call parser (get_tool_parser) used when
  // a request carries tools; default "hermes" (the gate model's format — Qwen3.6
  // shares it, see qwen3.h). Empty disables tool parsing.
  // `reasoning_parser_name` selects the reasoning parser (get_reasoning_parser),
  // mirroring the tool_parser_name pattern; default "" DISABLES reasoning
  // extraction (the C ABI / capi wires the model-specific selection separately).
  OpenAIServingChat(v1::LLMEngine& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes",
                    std::string reasoning_parser_name = "",
                    bool enable_force_include_usage = false);
  OpenAIServingChat(v1::AsyncLLM& engine, std::string served_model_name,
                    ChatPromptFn prompt_fn = DefaultChatPromptFallback,
                    std::string tool_parser_name = "hermes",
                    std::string reasoning_parser_name = "",
                    bool enable_force_include_usage = false);

  // create_chat_completion (chat_completion/serving.py:229).
  ChatCompletionResult create_chat_completion(
      const ChatCompletionRequest& request);

  // See OpenAIServingCompletion::uses_async_engine().
  bool uses_async_engine() const { return async_engine_ != nullptr; }

  // SAMPLE-BEAM: attach the tokenizer + eos id the beam-search path needs (it
  // tokenizes the rendered chat prompt and detokenizes each beam). See
  // OpenAIServingCompletion::set_beam_search_tokenizer.
  void set_beam_search_tokenizer(const vllm::tok::Tokenizer* tokenizer,
                                 std::optional<int32_t> eos_token_id) {
    beam_tokenizer_ = tokenizer;
    beam_eos_token_id_ = eos_token_id;
  }

  // How many rendered chat prompts the beam-search path has handed to
  // beam_tokenizer_->Encode. The beam path encodes the prompt itself and never
  // calls InputProcessor's text process_inputs, so num_prompt_encodes() cannot
  // see it. A test asserts this stays unchanged when a prompt is refused before
  // the encode (SERVE-REQUEST-LENGTH-GUARD). Relaxed, because it orders nothing.
  uint64_t num_beam_prompt_encodes() const {
    return num_beam_prompt_encodes_.load(std::memory_order_relaxed);
  }

  // Attach the multimodal chat seam (see MultiModalChatFn). Unset (default)
  // keeps the text-only path byte-identical. When set AND a request carries a mm
  // content part, create_chat_completion routes the request through the engine
  // mm add_request/generate overload with the seam's MultiModalInputs.
  void set_multimodal_chat_fn(MultiModalChatFn fn) {
    mm_chat_fn_ = std::move(fn);
    mm_request_chat_fn_ = {};
  }
  void set_request_multimodal_chat_fn(MultiModalRequestChatFn fn) {
    mm_request_chat_fn_ = std::move(fn);
    mm_chat_fn_ = {};
  }

  // The chat-prompt renderer this handler applies to `messages` (the same seam
  // create_chat_completion tokenizes through). Exposed so the /tokenize chat
  // form (serve/tokenize/serving.py:70-92, TokenizeChatRequest) renders through
  // the IDENTICAL chat template as chat-completions instead of reinventing it.
  const ChatPromptFn& prompt_fn() const { return prompt_fn_; }

  // See OpenAIServingCompletion::set_default_sampling_params (#1985).
  void set_default_sampling_params(vllm::DefaultSamplingParams defaults) {
    default_sampling_params_ = std::move(defaults);
  }
  const vllm::DefaultSamplingParams& default_sampling_params() const {
    return default_sampling_params_;
  }

 private:
  // Build the per-request tool parser (get_tool_parser) when ToolsEnabled and a
  // parser name is configured; else nullptr. ONE instance per request (the
  // streaming parse is stateful).
  std::unique_ptr<ToolParser> MakeToolParser(
      const ChatCompletionRequest& request) const;

  // Build the per-request ENGINE-BACKED parser (parser_manager get_parser_engine)
  // when ToolsEnabled and `tool_parser_name_` names an engine-backed format
  // (qwen3 / seed_oss / kimi_k2); else nullptr — a nullptr here means the legacy
  // MakeToolParser seam is used instead (the dispatch is name-selected, and the
  // default no-tool-parser path stays on the legacy seam byte-for-byte). ONE
  // instance per request (the streaming parse is stateful). Mirrors vLLM 0.26
  // ParserManager.get_parser (parser/parser_manager.py:76).
  std::unique_ptr<vllm::parser::engine::ParserEngine> MakeParserEngine(
      const ChatCompletionRequest& request) const;

  // Build the per-request reasoning parser (get_reasoning_parser) when a parser
  // name is configured; else nullptr. ONE instance per request (the streaming
  // parse may be stateful, olmo3). Unlike tools, reasoning does not gate on
  // ToolsEnabled: a request without tools can still carry a chain-of-thought.
  std::unique_ptr<ReasoningParser> MakeReasoningParser() const;

  v1::LLMEngine* sync_engine_ = nullptr;
  v1::AsyncLLM* async_engine_ = nullptr;
  std::string served_model_name_;
  ChatPromptFn prompt_fn_;
  std::string tool_parser_name_;
  std::string reasoning_parser_name_;
  bool enable_force_include_usage_ = false;
  // Beam-search context (see set_beam_search_tokenizer). Null => beam search
  // unavailable on this handler.
  const vllm::tok::Tokenizer* beam_tokenizer_ = nullptr;
  std::optional<int32_t> beam_eos_token_id_;
  // See set_default_sampling_params. Empty => every knob falls to the neutral
  // OpenAI default, byte-identical to the behaviour before #1985.
  vllm::DefaultSamplingParams default_sampling_params_;
  // Multimodal chat seam (see set_multimodal_chat_fn). Null => the text-only
  // path runs unchanged (mm parts drop to the joined-text content).
  MultiModalChatFn mm_chat_fn_;
  MultiModalRequestChatFn mm_request_chat_fn_;
  // request_id is "chatcmpl-<counter>" (upstream f"chatcmpl-{random_uuid()}").
  std::atomic<int64_t> request_counter_{0};
  // See num_beam_prompt_encodes().
  std::atomic<uint64_t> num_beam_prompt_encodes_{0};
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_CHAT_H_
