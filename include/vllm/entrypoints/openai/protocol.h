// Ported from: vllm/entrypoints/openai/protocol.py @ e24d1b24
//
// NOTE ON UPSTREAM LAYOUT: at e24d1b24 the historical monolithic
// `vllm/entrypoints/openai/protocol.py` has been split into per-endpoint
// modules — the shared/base shapes live in
// `vllm/entrypoints/openai/engine/protocol.py` (OpenAIBaseModel, ErrorInfo,
// ErrorResponse, UsageInfo, DeltaMessage, StreamOptions), the completion
// shapes in `.../completion/protocol.py` (CompletionRequest + its
// to_sampling_params, CompletionResponse[Choice], CompletionStreamResponse +
// StreamChoice), and the chat shapes in `.../chat_completion/protocol.py`
// (ChatCompletionRequest + its to_sampling_params, ChatCompletionResponse,
// ChatMessage, ChatCompletionResponse[Stream]Choice, ChatCompletionResponse,
// ChatCompletionStreamResponse). We collapse the T0 subset of all three into a
// single mirrored `protocol.{h,cpp}` (the pre-split upstream path) — the
// Ported-from headers above the individual sections cite the exact current
// files + lines.
//
// SCOPE (T0): text-completion + chat-completion request parsing → our
// SamplingParams, plus the response / stream-chunk / usage / error JSON shapes
// the OpenAI SDK + LocalAI depend on. Field NAMES and the `object` string
// literals are load-bearing for client compat and are mirrored 1:1.
//
// WIRED (ROAD-V1-C7 SAMPLE-BEST-OF / SAMPLE-BEAM): best_of + use_beam_search are
// now mapped on both request types (best_of → ParentRequest fan-out + top-n rank;
// use_beam_search → BeamSearchParams + the merged BeamSearch driver). See the
// per-field notes below for the honest vLLM-0.26 endpoint-availability finding.
//
// DEFERRED (parsed-and-ignored via OpenAI's extra="allow" — nlohmann simply
// does not read unknown keys — or explicitly marked below):
//   - logit_bias, allowed_token_ids, suffix, user, prompt_embeds
//   - response_format / structured_outputs / tools / tool_choice / functions
//     (M3.3 tool calling, M3.4 grammars)
//   - logprobs *payload* shapes (CompletionLogProbs / ChatCompletionLogProbs);
//     the request-side `logprobs` counts ARE mapped into SamplingParams.
//   - multimodal message content parts (T0 chat content is a bare string)
//   - array / token-id `prompt` forms (T0 completion prompt is a bare string)
//   - kv_transfer_params, vllm_xargs, cache_salt, etc.
//     (`priority` IS parsed now — it feeds the priority scheduler; see below.)
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/beam_search.h"  // BeamSearchParams (use_beam_search)
#include "vllm/logprobs.h"  // vllm::PromptLogprobs (prompt_logprobs payload)
#include "vllm/sampling_params.h"

namespace vllm::entrypoints::openai {

// ---------------------------------------------------------------------------
// Shared shapes — engine/protocol.py
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:111 (UsageInfo).
// `completion_tokens` is Optional[int]=0 upstream; T0 always populates it.
struct UsageInfo {
  int prompt_tokens = 0;
  int total_tokens = 0;
  int completion_tokens = 0;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:241
// (StreamOptions). Explicit null values resolve to the upstream defaults
// during request parsing, so the serving path consumes plain booleans.
struct StreamOptions {
  bool include_usage = false;
  bool continuous_usage_stats = false;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:60 (ErrorInfo).
struct ErrorInfo {
  std::string message;
  std::string type;
  std::optional<std::string> param;
  // Upstream `code: int` (an HTTP status code). Optional here so callers may
  // omit it; serialized as null when unset.
  std::optional<int> code;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:67 (ErrorResponse).
struct ErrorResponse {
  ErrorInfo error;
};

// ---------------------------------------------------------------------------
// response_format — engine/protocol.py (shared by both request types)
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:123
// (JsonSchemaResponseFormat). The wire `schema` field is aliased to
// `json_schema` (pydantic name-conflict workaround upstream); we keep the schema
// as a raw nlohmann::json (the constraint layer serializes it). `strict` /
// `description` are parsed but not load-bearing at T0.
struct JsonSchemaResponseFormat {
  std::string name;
  std::optional<std::string> description;
  // The JSON Schema object (wire key "schema").
  std::optional<nlohmann::json> json_schema;
  std::optional<bool> strict;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:156 (ResponseFormat).
// `type` is one of "text" | "json_object" | "json_schema". structural_tag is
// deferred (M3.4 STRUCTURAL_TAG stub); a structural_tag response_format parses
// but does not map to a constraint at T0.
struct ResponseFormat {
  std::string type;
  std::optional<JsonSchemaResponseFormat> json_schema;
};

// ---------------------------------------------------------------------------
// Tool / function calling — engine/protocol.py + chat_completion/protocol.py
// (M3.3 Task 1)
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/engine/protocol.py:246
// (FunctionDefinition). `parameters` is the JSON-Schema object for the
// function's arguments (upstream dict[str,Any]|None). `strict` / `defer_loading`
// are parsed-and-ignored at T0 (deferred).
struct FunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  // Order-preserving ON PURPOSE: the pinned renderer serializes tool schemas
  // into the prompt with sort_keys=False (transformers 5.14.1
  // utils/chat_template_utils.py:481), so the request document's key order is
  // the prompt's key order (pinned vLLM types this field dict[str, Any],
  // protocol.py:326, and its json parse preserves document order).
  // nlohmann::json would sort the keys away at parse (std::map).
  std::optional<nlohmann::ordered_json> parameters;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:165
// (ChatCompletionToolsParam). `defer_loading` deferred.
struct ChatCompletionToolsParam {
  std::string type = "function";
  FunctionDefinition function;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:218
// (ChatCompletionRequest.tool_choice) union + :188
// (ChatCompletionNamedToolChoiceParam). The upstream union is
// "none"|"auto"|"required"|{type:"function",function:{name}}. We model both the
// string and named-object forms as one struct: `mode` is one of
// "none"/"auto"/"required"/"function"; `function_name` is set only for the named
// (mode=="function") form.
struct ToolChoice {
  std::string mode;  // "none" | "auto" | "required" | "function"
  std::optional<std::string> function_name;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:310 (FunctionCall).
// The internal `id` field (excluded from serialization upstream) is deferred.
struct FunctionCall {
  std::string name;
  std::string arguments;  // JSON-encoded arguments string (OpenAI spec).
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:319 (ToolCall).
struct ToolCall {
  std::string id;
  std::string type = "function";
  FunctionCall function;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:325
// (DeltaFunctionCall).
struct DeltaFunctionCall {
  std::optional<std::string> name;
  std::optional<std::string> arguments;
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:331 (DeltaToolCall).
// `index` is required; id/type/function are per-chunk optional (name-first, then
// arguments deltas).
struct DeltaToolCall {
  int index = 0;
  std::optional<std::string> id;
  std::optional<std::string> type;
  DeltaFunctionCall function;
};

// ---------------------------------------------------------------------------
// Completions — completion/protocol.py
// ---------------------------------------------------------------------------

// Ported from: vllm/entrypoints/openai/completion/protocol.py:45
// (CompletionRequest). T0 subset. `stop` (upstream str|list) is normalized to
// list-form on parse (see from_json). `prompt` T0 = bare string.
struct CompletionRequest {
  std::optional<std::string> model;
  // T0: bare-string prompt. Array / token-id forms deferred (see header note).
  std::string prompt;

  // max_tokens: upstream field default is 16, and its `normalize_null_max_tokens`
  // validator rewrites an explicit `null` back to that default — so this is
  // effectively always populated. Kept optional to model "field absent".
  std::optional<int> max_tokens = 16;
  int n = 1;

  // ROAD-V1-C7 SAMPLE-BEST-OF / SAMPLE-BEAM (the OpenAI-endpoint surface).
  //
  // best_of (OpenAI Completions `best_of`): generate `best_of` sequences and
  // return the `n` highest-cumulative-logprob ones (best_of >= n; best_of == n
  // is the default no-op). HONEST FINDING: vLLM 0.26 (555967922) has DROPPED
  // best_of from the live path — it is NOT on CompletionRequest / the regular
  // ChatCompletionRequest and there is no SamplingParams.best_of; the only
  // surviving `best_of` is a vestigial, NEVER-consumed field on
  // BatchChatCompletionRequest (chat_completion/protocol.py:1048). We therefore
  // implement the CLASSIC OpenAI-spec / vLLM-V0 best_of contract (fan-out via the
  // merged ParentRequest path, rank by cumulative logprob, return top-n), gated
  // against our OWN deterministic fan-out (there is no 0.26 best_of oracle).
  std::optional<int> best_of;

  // use_beam_search (completion/protocol.py:73): route the request through the
  // BeamSearch driver instead of the sampler. beam_width == n; returns the n
  // best beams as choices. REAL vLLM-0.26 OpenAI surface
  // (completion/serving.py:173-205 → to_beam_search_params → self.beam_search).
  bool use_beam_search = false;
  // length_penalty (completion/protocol.py:77): beam-score length-penalty exponent.
  double length_penalty = 1.0;

  // Sampling knobs are Optional upstream (None => resolve to _DEFAULT_SAMPLING_
  // PARAMS in to_sampling_params); mirrored as std::optional here.
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> top_k;
  std::optional<double> min_p;
  std::optional<double> repetition_penalty;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;           // normalized list-form
  std::vector<int32_t> stop_token_ids;
  bool stream = false;
  std::optional<StreamOptions> stream_options;
  std::optional<int> logprobs;
  std::optional<int> prompt_logprobs;
  bool echo = false;                       // parsed; behavior deferred
  int min_tokens = 0;
  bool ignore_eos = false;
  bool include_stop_str_in_output = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  // stream_interval (completion/protocol.py:236 @ vllm#49754): per-request
  // override of the server `--stream-interval`; only raises it (clamped up).
  std::optional<int> stream_interval;

  // ROAD-V1-C7 SAMPLE-LOGIT-FILTERS (completion/protocol.py:58,93,108). OpenAI
  // `logit_bias` has STRING token-id keys; to_sampling_params converts them to
  // int and clamps each bias to [-100, 100] (sampling_params.py:388-413).
  std::optional<std::map<std::string, double>> logit_bias;
  std::optional<std::vector<int32_t>> allowed_token_ids;
  std::vector<std::string> bad_words;  // default_factory=list upstream

  // priority (completion/protocol.py): the request's scheduling priority for
  // the priority policy (lower = handled first). Default 0. Not a sampling
  // param — the serving layer forwards it to engine add_request/process_inputs.
  int priority = 0;

  // response_format (completion/protocol.py:106): {type: "text"|"json_object"|
  // "json_schema", json_schema?} — normalized into structured_outputs in
  // to_sampling_params (M3.4 Task 5).
  std::optional<ResponseFormat> response_format;

  // to_sampling_params — completion/protocol.py:260. Maps each OpenAI field 1:1
  // onto our SamplingParams (resolving None sampling knobs to _DEFAULT_SAMPLING_
  // PARAMS), then runs PostInit(). `default_max_tokens` is the serving-resolved
  // fallback used only when the request omits max_tokens (Task 2 supplies the
  // model-derived value; unset => our SamplingParams default).
  // `defaults` is the server-wide DefaultSamplingParams derived from the
  // checkpoint's generation_config.json (config/generation.h). Upstream's
  // resolution order, mirrored exactly: an explicitly sent request field wins;
  // an omitted one takes the checkpoint value; only if the checkpoint declares
  // nothing does the neutral OpenAI default apply
  // (CompletionRequest._DEFAULT_SAMPLING_PARAMS). nullptr means no
  // server-provided defaults, and reproduces the pre-#1985 behaviour byte for
  // byte, which is what keeps every existing caller unchanged.
  SamplingParams to_sampling_params(
      std::optional<int> default_max_tokens = std::nullopt,
      const DefaultSamplingParams* defaults = nullptr) const;

  // to_beam_search_params — completion/protocol.py:260. beam_width == n, the
  // resolved max_tokens, ignore_eos, temperature (None => the checkpoint's, then
  // 1.0) and length_penalty.
  vllm::BeamSearchParams to_beam_search_params(
      int max_tokens, const DefaultSamplingParams* defaults = nullptr) const;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:580-584
// (CompletionLogProbs). The /v1/completions logprobs payload: per generated (or
// echoed) token, its text, sampled logprob, the running text offset, and the
// {token_str -> logprob} top-N map. token_logprobs / top_logprobs are None at a
// position with no logprobs.
struct CompletionLogProbs {
  std::vector<int> text_offset;
  std::vector<std::optional<float>> token_logprobs;
  std::vector<std::string> tokens;
  std::vector<std::optional<std::map<std::string, float>>> top_logprobs;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:75-78
// (ChatCompletionLogProb).
struct ChatCompletionLogProb {
  std::string token;
  float logprob = -9999.0f;
  // bytes: UTF-8 bytes of `token` (None only when the token has no decoded form).
  std::optional<std::vector<int>> bytes;
};

// Ported from: chat_completion/protocol.py:81-85 (ChatCompletionLogProbsContent,
// extends ChatCompletionLogProb with the per-token top_logprobs alternatives).
struct ChatCompletionLogProbsContent {
  std::string token;
  float logprob = -9999.0f;
  std::optional<std::vector<int>> bytes;
  std::vector<ChatCompletionLogProb> top_logprobs;
};

// Ported from: chat_completion/protocol.py:88-89 (ChatCompletionLogProbs).
struct ChatCompletionLogProbs {
  std::optional<std::vector<ChatCompletionLogProbsContent>> content;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:519
// (CompletionResponseChoice). stop_reason / token payloads deferred.
struct CompletionResponseChoice {
  int index = 0;
  std::string text;
  std::optional<CompletionLogProbs> logprobs;
  std::optional<std::string> finish_reason;
  // prompt_logprobs (completion/protocol.py:601):
  // `list[dict[int, Logprob] | None] | None`. The per-prompt-position
  // distribution, populated from RequestOutput.prompt_logprobs when the request
  // asked for it. Upstream sets it on EVERY choice, inside the per-output loop
  // (completion/serving.py:588), so an n>1 response repeats one prompt payload.
  std::optional<vllm::PromptLogprobs> prompt_logprobs;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:547
// (CompletionResponse). object == "text_completion".
struct CompletionResponse {
  std::string id;
  std::string object = "text_completion";
  int64_t created = 0;
  std::string model;
  std::vector<CompletionResponseChoice> choices;
  UsageInfo usage;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:563
// (CompletionResponseStreamChoice).
struct CompletionResponseStreamChoice {
  int index = 0;
  std::string text;
  std::optional<CompletionLogProbs> logprobs;
  std::optional<std::string> finish_reason;
};

// Ported from: vllm/entrypoints/openai/completion/protocol.py:582
// (CompletionStreamResponse). object == "text_completion"; usage optional.
struct CompletionStreamResponse {
  std::string id;
  std::string object = "text_completion";
  int64_t created = 0;
  std::string model;
  std::vector<CompletionResponseStreamChoice> choices;
  std::optional<UsageInfo> usage;
};

// ---------------------------------------------------------------------------
// Chat completions — chat_completion/protocol.py
// ---------------------------------------------------------------------------

// One parsed multimodal content part of a chat message (ROAD-V1-MM serving W1).
//
// Ported from: vllm/entrypoints/chat_utils.py:1478 (MM_PARSER_MAP) +
// _parse_chat_message_content_mm_part:1524. The OpenAI chat `content` may be a
// bare string (T0) OR an array of typed parts; this struct is ONE parsed part.
// We mirror the field NAMES / part `type` literals 1:1 (client-compat load-
// bearing). Only the raw wire strings live here (pure JSON, no multimodal dep on
// protocol.h); the base64/data-URI DECODE + processor ROUTE live in chat_mm.cpp.
//
// Coverage (this brick): "text", "image_url" ({image_url:{url}}), "input_audio"
// ({input_audio:{data,format}}), "audio_url" ({audio_url:{url}}). Other part
// kinds (image_embeds / video_url / prompt_embeds / image_pil / tool_reference)
// are parsed as their `type` with empty payload and are named residuals.
struct ChatContentPart {
  std::string type;          // part `type` literal (chat_utils.py MM_PARSER_MAP key)
  std::string text;          // set for type=="text" (the "text" field)
  std::string url;           // image_url.url / audio_url.url (may be a data: URI)
  std::string audio_data;    // input_audio.data (base64 payload)
  std::string audio_format;  // input_audio.format ("wav" | "mp3" | ...)
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:57
// (ChatMessage). Bare-string content OR (ROAD-V1-MM serving) a multimodal
// content-part array; refusal / function_call (legacy) deferred. A response
// message carries EITHER content OR tool_calls; when tool_calls is empty the key
// is omitted (upstream _serialize pop).
struct ChatMessage {
  std::string role;
  // Default member initializers so a 2-field aggregate init
  // `ChatMessage{role, content}` (used across the serving/chat-template tests)
  // does not trip -Werror=missing-field-initializers now that tool_calls exists.
  std::optional<std::string> content{};
  // Multimodal content-part array (chat_utils.py). nullopt for a bare-string
  // request (the T0 path is byte-IDENTICAL). When the wire `content` is an
  // array, this holds the parsed parts AND `content` is set to the joined text
  // spans so the existing text prompt path keeps working; the mm parts are
  // routed to the processor by chat_mm.cpp (ParseMultiModalContent).
  std::optional<std::vector<ChatContentPart>> content_parts{};
  std::optional<std::vector<ToolCall>> tool_calls{};
  // reasoning (chat_completion/protocol.py:67). The chain-of-thought span split
  // off by the reasoning parser. Serialized only when present + non-empty, and
  // kept after the positional aggregate-init fields above.
  std::optional<std::string> reasoning{};
  // Multi-turn tool-conversation identity (chat_completion/protocol.py: the
  // role="tool" reply carries the id of the assistant call it answers, and
  // optionally the function name). Without these a second turn after tool
  // execution is malformed for templates that render tool_call_id/name.
  std::optional<std::string> tool_call_id{};
  std::optional<std::string> name{};
};

// Ported from: vllm/entrypoints/openai/engine/protocol.py:350 (DeltaMessage).
// `tool_calls` is a list defaulting to empty upstream (serialized only when
// non-empty); modeled here as optional to mean "absent/empty".
struct DeltaMessage {
  std::optional<std::string> role;
  std::optional<std::string> content;
  // reasoning (engine/protocol.py:353). The streamed chain-of-thought span;
  // emitted on its own delta before the content deltas. Serialized when present.
  std::optional<std::string> reasoning;
  std::optional<std::vector<DeltaToolCall>> tool_calls;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:193
// (ChatCompletionRequest). T0 subset. messages content T0 = bare string.
struct ChatCompletionRequest {
  std::vector<ChatMessage> messages;
  std::optional<std::string> model;

  // Upstream: max_tokens (deprecated) + max_completion_tokens, both default
  // None; serving prefers max_completion_tokens (chat_completion/serving.py:299).
  std::optional<int> max_tokens;
  std::optional<int> max_completion_tokens;
  std::optional<int> n = 1;

  // ROAD-V1-C7 SAMPLE-BEST-OF / SAMPLE-BEAM. See CompletionRequest for the honest
  // best_of finding (vLLM 0.26 dropped it from the live path; classic semantics
  // implemented). use_beam_search / length_penalty mirror
  // chat_completion/protocol.py:249,597 (→ to_beam_search_params → self.beam_search,
  // chat_completion/serving.py:319-343). best_of on chat is a DEVIATION: upstream
  // carries it only on BatchChatCompletionRequest (protocol.py:1048), we expose it
  // on ChatCompletionRequest too for endpoint symmetry with /v1/completions.
  std::optional<int> best_of;
  bool use_beam_search = false;
  double length_penalty = 1.0;

  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<int> top_k;
  std::optional<double> min_p;
  std::optional<double> repetition_penalty;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;           // normalized list-form
  std::vector<int32_t> stop_token_ids;
  bool stream = false;
  std::optional<StreamOptions> stream_options;
  // Upstream chat `logprobs` is a BOOL flag; `top_logprobs` is the count.
  bool logprobs = false;
  int top_logprobs = 0;
  bool echo = false;
  std::optional<int> prompt_logprobs;
  int min_tokens = 0;
  bool ignore_eos = false;
  bool include_stop_str_in_output = false;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  // stream_interval (chat_completion/protocol.py @ vllm#49754): per-request
  // override of the server `--stream-interval`; only raises it (clamped up).
  std::optional<int> stream_interval;

  // ROAD-V1-C7 SAMPLE-LOGIT-FILTERS (chat_completion/protocol.py:202,282,283).
  // See CompletionRequest for the logit_bias string-key -> int + clamp contract.
  std::optional<std::map<std::string, double>> logit_bias;
  std::optional<std::vector<int32_t>> allowed_token_ids;
  std::vector<std::string> bad_words;

  // priority (chat_completion/protocol.py). See CompletionRequest.priority.
  int priority = 0;

  // response_format (chat_completion/protocol.py:210). See CompletionRequest.
  std::optional<ResponseFormat> response_format;

  // tools / tool_choice (chat_completion/protocol.py:217-224). `tools` is the
  // list of available functions; `tool_choice` selects the calling mode. When
  // absent both stay nullopt (backward compat). The serving-layer default
  // (tool_choice="auto" when tools present, else "none") is applied downstream
  // (chat_completion/protocol.py:828-831), not here.
  // DEFERRED (parsed-and-ignored): parallel_tool_calls (protocol.py:240) and the
  // legacy `functions` / `function_call` fields.
  std::optional<std::vector<ChatCompletionToolsParam>> tools;
  std::optional<ToolChoice> tool_choice;

  // chat_template_kwargs (chat_completion/protocol.py:341, default None):
  // "additional keyword args to pass to the template renderer", accessible by
  // the chat template. Carried as the raw JSON object so an arbitrary key
  // reaches the renderer, which is what upstream does -- it filters against the
  // template's own undeclared variables rather than a fixed key list
  // (vllm/renderers/hf.py:633-661). An absent object leaves EVERY name unbound,
  // so `{% if enable_thinking is undefined %}` answers true, which is the
  // Qwen3.8 family's own default and was unreachable before #1681.
  nlohmann::ordered_json chat_template_kwargs =
      nlohmann::ordered_json::object();

  // include_reasoning (chat_completion/protocol.py:242, default True). When
  // false the parser drops the reasoning span (parser_engine.py:451); the
  // engine-backed serving path maps this onto ParserRequest.include_reasoning.
  bool include_reasoning = true;

  // to_sampling_params — chat_completion/protocol.py:585. See CompletionRequest.
  SamplingParams to_sampling_params(
      std::optional<int> default_max_tokens = std::nullopt,
      const DefaultSamplingParams* defaults = nullptr) const;

  // to_beam_search_params — chat_completion/protocol.py:589. See CompletionRequest.
  vllm::BeamSearchParams to_beam_search_params(
      int max_tokens, const DefaultSamplingParams* defaults = nullptr) const;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:94
// (ChatCompletionResponseChoice). finish_reason default "stop" upstream.
struct ChatCompletionResponseChoice {
  int index = 0;
  ChatMessage message;
  std::optional<ChatCompletionLogProbs> logprobs;
  std::optional<std::string> finish_reason = "stop";
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:117
// (ChatCompletionResponse). object == "chat.completion".
struct ChatCompletionResponse {
  std::string id;
  std::string object = "chat.completion";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionResponseChoice> choices;
  UsageInfo usage;
  // prompt_logprobs (chat_completion/protocol.py:126): a vLLM-specific field
  // outside the OpenAI spec, and TOP-LEVEL on the response rather than per
  // choice — the chat prompt is one rendered string shared by every choice
  // (chat_completion/serving.py:1070).
  std::optional<vllm::PromptLogprobs> prompt_logprobs;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:138
// (ChatCompletionResponseStreamChoice).
struct ChatCompletionResponseStreamChoice {
  int index = 0;
  DeltaMessage delta;
  std::optional<ChatCompletionLogProbs> logprobs;
  std::optional<std::string> finish_reason;
};

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:148
// (ChatCompletionStreamResponse). object == "chat.completion.chunk".
struct ChatCompletionStreamResponse {
  std::string id;
  std::string object = "chat.completion.chunk";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionResponseStreamChoice> choices;
  std::optional<UsageInfo> usage;
};

// ---------------------------------------------------------------------------
// nlohmann/json (de)serialization — ADL free functions.
//   Requests: from_json (parse).  Responses/usage/error: to_json (serialize).
// ---------------------------------------------------------------------------
void from_json(const nlohmann::json& j, CompletionRequest& r);
void from_json(const nlohmann::json& j, ChatMessage& m);
void from_json(const nlohmann::json& j, ChatCompletionToolsParam& t);
void from_json(const nlohmann::json& j, ChatCompletionRequest& r);
// Order-preserving twin of the ChatCompletionToolsParam overload: a body
// parsed as nlohmann::ordered_json keeps `parameters` in document order,
// which is the order the pinned renderer dumps into the prompt.
void from_json(const nlohmann::ordered_json& j, ChatCompletionToolsParam& t);

// Re-read a parsed request's `tools` from an order-preserving parse of the
// same body. A body parsed as nlohmann::json sorts object keys (std::map),
// which loses the tool schemas' document order; the pinned renderer
// preserves that order into the prompt bytes (sort_keys=False tojson), so
// every entry point that renders chat prompts calls this after its regular
// parse. No-op when the request carries no tools. Throws whatever the
// ordered parse throws (the regular parse of the same body already
// succeeded by the time callers reach this).
void RestoreToolSchemaOrder(const std::string& request_body,
                            ChatCompletionRequest& r);

void to_json(nlohmann::json& j, const UsageInfo& u);
void to_json(nlohmann::json& j, const ErrorInfo& e);
void to_json(nlohmann::json& j, const ErrorResponse& e);
void to_json(nlohmann::json& j, const CompletionLogProbs& lp);
void to_json(nlohmann::json& j, const ChatCompletionLogProb& lp);
void to_json(nlohmann::json& j, const ChatCompletionLogProbsContent& lp);
void to_json(nlohmann::json& j, const ChatCompletionLogProbs& lp);
void to_json(nlohmann::json& j, const CompletionResponseChoice& c);
void to_json(nlohmann::json& j, const CompletionResponse& r);
void to_json(nlohmann::json& j, const CompletionResponseStreamChoice& c);
void to_json(nlohmann::json& j, const CompletionStreamResponse& r);
void to_json(nlohmann::json& j, const FunctionCall& f);
void to_json(nlohmann::json& j, const ToolCall& t);
void to_json(nlohmann::json& j, const DeltaFunctionCall& f);
void to_json(nlohmann::json& j, const DeltaToolCall& t);
void to_json(nlohmann::json& j, const ChatMessage& m);
void to_json(nlohmann::json& j, const DeltaMessage& m);
void to_json(nlohmann::json& j, const ChatCompletionResponseChoice& c);
void to_json(nlohmann::json& j, const ChatCompletionResponse& r);
void to_json(nlohmann::json& j, const ChatCompletionResponseStreamChoice& c);
void to_json(nlohmann::json& j, const ChatCompletionStreamResponse& r);

}  // namespace vllm::entrypoints::openai
