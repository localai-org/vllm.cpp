// Ported from: vllm/entrypoints/openai/protocol.py @ e24d1b24
// (split upstream into engine/completion/chat_completion/protocol.py — see the
// header for the exact per-shape source citations).

#include "vllm/entrypoints/openai/protocol.h"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm::entrypoints::openai {

namespace {

// completion/protocol.py:231 / chat_completion/protocol.py:559
// (_DEFAULT_SAMPLING_PARAMS). Identical for both request types.
constexpr double kDefaultRepetitionPenalty = 1.0;
constexpr double kDefaultTemperature = 1.0;
constexpr double kDefaultTopP = 1.0;
constexpr int kDefaultTopK = 0;
constexpr double kDefaultMinP = 0.0;

// Read a JSON value that may be absent or explicit null into an optional<T>.
// Templated on the json type so the order-preserving (ordered_json) parse
// path can reuse the same readers.
template <typename Json, typename T>
void GetOpt(const Json& j, const char* key, std::optional<T>& out) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) {
    out = it->template get<T>();
  }
}

// Read a scalar with a fallback when the key is absent or null.
template <typename Json, typename T>
void GetOr(const Json& j, const char* key, T& out) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) {
    out = it->template get<T>();
  }
}

// check_logprobs, the SHARED PREFIX of the two `mode="before"` model validators:
// completion/protocol.py:470-494 and chat_completion/protocol.py:759-783. The two
// are byte-identical up to the point where each checks its own per-token count,
// and they DIVERGE there — see the two call sites, which carry their own suffix
// exactly as upstream does.
//
// Upstream runs this on the raw request dict before any coercion and raises
// VLLMValidationError, which the API layer answers as a 400. Our parser throws
// and api_server.cpp maps any exception out of the body parse to 400
// BadRequestError, so the message text is the part that has to match.
//
// `count_field` names the endpoint's per-token count — `logprobs` on
// /v1/completions, `top_logprobs` on /v1/chat/completions — because the
// integer-ness loop covers it on both. It does NOT decide the count's range;
// that rule differs and lives at the call site.
//
// #249 owns porting this validation; the cap half already landed in
// src/vllm/v1/engine/input_processor.cpp:137.
void ValidateLogprobsPrefix(const nlohmann::json& j, const char* count_field) {
  // :474-481 / :763-771 — a JSON string would reach the comparisons below and
  // raise a TypeError -> HTTP 500 upstream, so it is refused here as a clean
  // 400. A JSON number (int OR float) passes, mirroring
  // `isinstance(v, (int, float))`; a JSON bool is Python's int subclass and
  // passes there too.
  for (const char* field : {"prompt_logprobs", count_field}) {
    auto it = j.find(field);
    if (it == j.end() || it->is_null()) continue;
    if (!it->is_number() && !it->is_boolean()) {
      throw std::invalid_argument(std::string("`") + field +
                                  "` must be an integer.");
    }
  }
  auto plp = j.find("prompt_logprobs");
  if (plp == j.end() || plp->is_null()) return;
  const int value = plp->get<int>();
  // :483-488 / :772-777 — streaming cannot carry the prompt payload. The
  // condition is `> 0 or == -1`, so an explicit 0 (rank only, no alternatives)
  // streams.
  auto stream = j.find("stream");
  const bool streaming =
      stream != j.end() && !stream->is_null() && stream->get<bool>();
  if (streaming && (value > 0 || value == -1)) {
    throw std::invalid_argument(
        "`prompt_logprobs` are not available when `stream=True`.");
  }
  // :490-494 / :779-783 — -1 is the whole-vocabulary sentinel; every other
  // negative is refused.
  if (value < 0 && value != -1) {
    throw std::invalid_argument(
        "`prompt_logprobs` must be a positive value or -1.");
  }
}

// Normalize the OpenAI `stop` union (str | list[str] | null) to list-form —
// mirrors SamplingParams.__post_init__ stop normalization upstream.
std::vector<std::string> ParseStop(const nlohmann::json& j) {
  auto it = j.find("stop");
  if (it == j.end() || it->is_null()) return {};
  if (it->is_string()) return {it->get<std::string>()};
  if (it->is_array()) return it->get<std::vector<std::string>>();
  return {};
}

// ROAD-V1-C7 SAMPLE-LOGIT-FILTERS: parse the shared logit_bias /
// allowed_token_ids / bad_words request fields (completion/protocol.py:58,93,108;
// chat_completion/protocol.py:202,282,283). `logit_bias` keeps its OpenAI string
// keys here; to_sampling_params converts + clamps (from_optional:388-413).
void ParseLogitFilters(const nlohmann::json& j,
                       std::optional<std::map<std::string, double>>& logit_bias,
                       std::optional<std::vector<int32_t>>& allowed_token_ids,
                       std::vector<std::string>& bad_words) {
  if (auto it = j.find("logit_bias"); it != j.end() && it->is_object()) {
    std::map<std::string, double> lb;
    for (auto e = it->begin(); e != it->end(); ++e) {
      lb[e.key()] = e.value().get<double>();
    }
    logit_bias = std::move(lb);
  }
  if (auto it = j.find("allowed_token_ids");
      it != j.end() && it->is_array()) {
    allowed_token_ids = it->get<std::vector<int32_t>>();
  }
  if (auto it = j.find("bad_words"); it != j.end() && it->is_array()) {
    bad_words = it->get<std::vector<std::string>>();
  }
}

// ROAD-V1-C7 SAMPLE-LOGIT-FILTERS: from_optional's logit_bias conversion +
// clamp (sampling_params.py:388-413) plus the allowed_token_ids / bad_words
// carry. String keys are parsed to int token ids; each bias is clamped to
// [-100, 100]. A key that does not parse to an integer throws (VLLMValidationError
// upstream -> std::runtime_error here, surfaced as a 400).
void ApplyLogitFilters(
    SamplingParams& sp,
    const std::optional<std::map<std::string, double>>& logit_bias,
    const std::optional<std::vector<int32_t>>& allowed_token_ids,
    const std::vector<std::string>& bad_words) {
  if (logit_bias.has_value()) {
    std::map<int32_t, float> converted;
    for (const auto& [token, bias] : *logit_bias) {
      int32_t token_id = 0;
      try {
        size_t consumed = 0;
        token_id = static_cast<int32_t>(std::stol(token, &consumed));
        if (consumed != token.size()) throw std::invalid_argument("trailing");
      } catch (const std::exception&) {
        throw std::runtime_error(
            "logit_bias contains key(s) that cannot be converted to integer "
            "token IDs: ['" +
            token + "']");
      }
      converted[token_id] =
          static_cast<float>(std::min(100.0, std::max(-100.0, bias)));
    }
    sp.logit_bias = std::move(converted);
  }
  sp.allowed_token_ids = allowed_token_ids;
  sp.bad_words = bad_words;
}

// Serialize an optional string as either its value or JSON null.
nlohmann::json OrNull(const std::optional<std::string>& v) {
  if (v.has_value()) return *v;
  return nullptr;
}

// Parse the OpenAI `response_format` object (engine/protocol.py:156 ResponseFormat
// + :123 JsonSchemaResponseFormat). Mirrors validate_response_format
// (completion/protocol.py:387 / chat_completion/protocol.py:700): a present
// response_format needs a string `type`, and `json_schema` type requires the
// json_schema field (its `schema` alias). Throws on a malformed shape (surfaces
// as a 400 bad request).
void ParseResponseFormat(const nlohmann::json& j,
                         std::optional<ResponseFormat>& out) {
  auto it = j.find("response_format");
  if (it == j.end() || it->is_null()) return;
  const nlohmann::json& rf = *it;
  if (!rf.is_object()) {
    throw std::runtime_error("response_format must be an object");
  }
  ResponseFormat r;
  if (auto t = rf.find("type"); t != rf.end() && t->is_string()) {
    r.type = t->get<std::string>();
  } else {
    throw std::runtime_error("response_format requires a string 'type'");
  }
  if (auto js = rf.find("json_schema"); js != rf.end() && js->is_object()) {
    JsonSchemaResponseFormat jsr;
    GetOr(*js, "name", jsr.name);
    GetOpt(*js, "description", jsr.description);
    // The schema is carried on the wire as `schema` (aliased to json_schema).
    if (auto sc = js->find("schema"); sc != js->end() && !sc->is_null()) {
      jsr.json_schema = *sc;
    }
    GetOpt(*js, "strict", jsr.strict);
    r.json_schema = std::move(jsr);
  }
  // validate_response_format: json_schema type must carry the json_schema field.
  if (r.type == "json_schema" && !r.json_schema.has_value()) {
    throw std::runtime_error(
        "When response_format type is 'json_schema', the 'json_schema' field "
        "must be provided.");
  }
  out = std::move(r);
}

// engine/protocol.py:241-243 (StreamOptions) plus completion/protocol.py:
// 471-478 / chat_completion/protocol.py:731-737 (validate_stream_options).
// The upstream before-validator rejects a non-empty options object unless the
// request is streaming. Null booleans use the StreamOptions defaults.
void ParseStreamOptions(const nlohmann::json& j, bool stream,
                        std::optional<StreamOptions>& out) {
  auto it = j.find("stream_options");
  if (it == j.end() || it->is_null()) return;
  if (!it->is_object()) {
    throw std::runtime_error("stream_options must be an object");
  }
  if (!stream && !it->empty()) {
    throw std::runtime_error(
        "Stream options can only be defined when `stream=True`.");
  }
  StreamOptions options;
  GetOr(*it, "include_usage", options.include_usage);
  GetOr(*it, "continuous_usage_stats", options.continuous_usage_stats);
  out = options;
}

// Normalize a parsed response_format into SamplingParams.structured_outputs
// (completion/protocol.py:309-338 / chat_completion/protocol.py:629-658):
//   json_object -> structured_outputs.json_object = true
//   json_schema -> structured_outputs.json = the schema (serialized)
//   text / absent -> no structured-output constraint.
// Merges onto any existing structured_outputs (upstream `replace(...)`), though
// at T0 nothing else populates it.
void ApplyResponseFormat(const std::optional<ResponseFormat>& rf,
                         SamplingParams& sp) {
  if (!rf.has_value()) return;
  StructuredOutputsParams so =
      sp.structured_outputs.value_or(StructuredOutputsParams{});
  bool enabled = false;
  if (rf->type == "json_object") {
    so.json_object = true;
    enabled = true;
  } else if (rf->type == "json_schema") {
    // Parse-time validation guarantees json_schema is present; a json_schema
    // WITHOUT a `schema` payload means "any JSON object" (upstream passes
    // json=None through, which the engine treats as json_object); mirror that by
    // requiring the schema here and falling back only when absent.
    if (rf->json_schema.has_value() &&
        rf->json_schema->json_schema.has_value()) {
      so.json = rf->json_schema->json_schema->dump();
    } else {
      so.json_object = true;
    }
    enabled = true;
  }
  // "text" (or any other type, e.g. deferred structural_tag) -> no constraint.
  if (enabled) sp.structured_outputs = std::move(so);
}

// Parse the OpenAI `tool_choice` union (chat_completion/protocol.py:218-224):
// a string "none"|"auto"|"required", or a named-function object
// {type:"function", function:{name}}. Mirrors the object-form validation
// (chat_completion/protocol.py:864-884): an object tool_choice requires a
// `function` object carrying a string `name`. Throws on a malformed shape
// (surfaces as a 400). An absent key leaves tool_choice unset (the serving layer
// applies the "auto when tools present else none" default upstream).
void ParseToolChoice(const nlohmann::json& j, std::optional<ToolChoice>& out) {
  auto it = j.find("tool_choice");
  if (it == j.end() || it->is_null()) return;
  const nlohmann::json& tc = *it;
  ToolChoice choice;
  if (tc.is_string()) {
    choice.mode = tc.get<std::string>();
  } else if (tc.is_object()) {
    // Named-function choice: {type:"function", function:{name}}.
    auto fn = tc.find("function");
    if (fn == tc.end() || !fn->is_object()) {
      throw std::runtime_error(
          "Expected field `function` in `tool_choice`! Correct usage: "
          "`{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`");
    }
    auto name = fn->find("name");
    if (name == fn->end() || !name->is_string()) {
      throw std::runtime_error(
          "Expected field `name` in `function` in `tool_choice`! Correct usage: "
          "`{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`");
    }
    choice.mode = "function";
    choice.function_name = name->get<std::string>();
  } else {
    throw std::runtime_error("Invalid value for `tool_choice`.");
  }
  out = std::move(choice);
}

// SAMPLE-BEST-OF: map the OpenAI `best_of` onto the engine fan-out count.
// Classic OpenAI/vLLM-V0 semantics (vLLM 0.26 dropped best_of from the live path
// — see protocol.h): best_of >= n; when best_of > n, generate `best_of` children
// (sp.n = best_of, via the existing ParentRequest fan-out) and let the serving
// layer rank by cumulative logprob and return the top-n. Ranking needs the
// per-sequence cumulative logprob, which our engine only accumulates when
// logprobs are computed (v1/engine/logprobs.cpp:37-39), so force the sampled-token
// logprob (logprobs=0, no user-visible payload) when the caller didn't ask for it.
// INERT when best_of is unset or <= n (sp.n unchanged; no forced logprobs). Reject
// best_of < n (OpenAI: best_of must be >= n). The greedy restriction (n>1 requires
// non-greedy sampling) is enforced downstream by SamplingParams::PostInit().
void ApplyBestOf(SamplingParams& sp, const std::optional<int>& best_of, int n) {
  if (!best_of.has_value()) return;
  if (*best_of < n) {
    throw std::runtime_error(
        "best_of must be greater than or equal to n");
  }
  if (*best_of > n) {
    sp.n = *best_of;
    if (!sp.logprobs.has_value()) sp.logprobs = 0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Request parsing (from_json)
// ---------------------------------------------------------------------------

void from_json(const nlohmann::json& j, CompletionRequest& r) {
  GetOpt(j, "model", r.model);
  // T0: bare-string prompt (array / token-id forms deferred).
  if (auto it = j.find("prompt"); it != j.end() && it->is_string()) {
    r.prompt = it->get<std::string>();
  }
  // normalize_null_max_tokens (completion/protocol.py:377): an explicit null
  // max_tokens is rewritten to the field default (16); an absent key keeps the
  // default; a value overrides it.
  if (auto it = j.find("max_tokens"); it != j.end()) {
    if (it->is_null()) {
      r.max_tokens = 16;
    } else {
      r.max_tokens = it->get<int>();
    }
  }
  GetOr(j, "n", r.n);
  // SAMPLE-BEST-OF / SAMPLE-BEAM (completion/protocol.py:73,77 + best_of).
  GetOpt(j, "best_of", r.best_of);
  GetOr(j, "use_beam_search", r.use_beam_search);
  GetOr(j, "length_penalty", r.length_penalty);
  GetOpt(j, "temperature", r.temperature);
  GetOpt(j, "top_p", r.top_p);
  GetOpt(j, "top_k", r.top_k);
  GetOpt(j, "min_p", r.min_p);
  GetOpt(j, "repetition_penalty", r.repetition_penalty);
  GetOr(j, "presence_penalty", r.presence_penalty);
  GetOr(j, "frequency_penalty", r.frequency_penalty);
  GetOpt(j, "seed", r.seed);
  r.stop = ParseStop(j);
  GetOr(j, "stop_token_ids", r.stop_token_ids);
  GetOr(j, "stream", r.stream);
  ParseStreamOptions(j, r.stream, r.stream_options);
  // completion/protocol.py:445 (`check_logprobs`, mode="before") — runs on the
  // raw body BEFORE the values are read, exactly as upstream does.
  ValidateLogprobsPrefix(j, "logprobs");
  // :495-499, the completion-only suffix. `logprobs` here has NO -1 sentinel:
  // any negative is a 400. This is deliberately NOT the chat rule below, and it
  // closes the divergence `.agents/specs/logprobs-all-sentinel.md` records —
  // we used to accept `logprobs: -1` on this endpoint and then emit empty
  // top_logprobs maps, because `BuildCompletionLogProbs`'s `idx > -1` breaks on
  // the first entry.
  if (auto lp = j.find("logprobs");
      lp != j.end() && !lp->is_null() && lp->is_number() && lp->get<int>() < 0) {
    throw std::invalid_argument("`logprobs` must be a positive value.");
  }
  GetOpt(j, "logprobs", r.logprobs);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "echo", r.echo);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOpt(j, "stream_interval", r.stream_interval);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
  GetOr(j, "priority", r.priority);
  ParseLogitFilters(j, r.logit_bias, r.allowed_token_ids, r.bad_words);
  ParseResponseFormat(j, r.response_format);
}

// Parse ONE multimodal content part (chat_utils.py:1478 MM_PARSER_MAP +
// _parse_chat_message_content_mm_part:1524). Reads the `type`-keyed payload into
// a ChatContentPart. Unknown/absent type with a direct URL field is not modeled
// here (the typed form is what OpenAI clients emit); such parts keep an empty
// payload under their detected `type` and are named residuals.
static ChatContentPart ParseChatContentPart(const nlohmann::json& part) {
  ChatContentPart p;
  if (!part.is_object()) return p;
  GetOr(part, "type", p.type);
  if (p.type == "text" || p.type == "input_text" || p.type == "output_text") {
    // _TextParser(part).get("text") — normalize the response-input aliases to
    // "text" so the joined prompt text is uniform.
    if (auto t = part.find("text"); t != part.end() && t->is_string()) {
      p.text = t->get<std::string>();
    }
    p.type = "text";
  } else if (p.type == "image_url") {
    // _ImageParser(part).get("image_url", {}).get("url") — the url lives under a
    // nested object {url: "data:image/...;base64,..."}.
    if (auto iu = part.find("image_url"); iu != part.end()) {
      if (iu->is_object()) {
        if (auto u = iu->find("url"); u != iu->end() && u->is_string())
          p.url = u->get<std::string>();
      } else if (iu->is_string()) {
        p.url = iu->get<std::string>();  // simple-image param form
      }
    }
  } else if (p.type == "audio_url") {
    if (auto au = part.find("audio_url"); au != part.end()) {
      if (au->is_object()) {
        if (auto u = au->find("url"); u != au->end() && u->is_string())
          p.url = u->get<std::string>();
      } else if (au->is_string()) {
        p.url = au->get<std::string>();
      }
    }
  } else if (p.type == "input_audio") {
    // _InputAudioParser(part).get("input_audio") — {data: <base64>, format: str}.
    if (auto ia = part.find("input_audio"); ia != part.end() && ia->is_object()) {
      if (auto d = ia->find("data"); d != ia->end() && d->is_string())
        p.audio_data = d->get<std::string>();
      if (auto f = ia->find("format"); f != ia->end() && f->is_string())
        p.audio_format = f->get<std::string>();
    }
  }
  return p;
}

void from_json(const nlohmann::json& j, ChatMessage& m) {
  GetOr(j, "role", m.role);
  // Content may be a bare string (T0, byte-identical) OR a multimodal
  // content-part array (ROAD-V1-MM serving W1; chat_utils.py
  // _parse_chat_message_content_parts). The array form fills content_parts AND
  // sets `content` to the joined text spans so the downstream text prompt path
  // is unchanged; the mm parts are routed to the processor by chat_mm.cpp.
  if (auto it = j.find("content"); it != j.end()) {
    if (it->is_string()) {
      m.content = it->get<std::string>();
    } else if (it->is_array()) {
      std::vector<ChatContentPart> parts;
      std::string joined_text;
      for (const nlohmann::json& part : *it) {
        // A bare string element is treated as a text part (upstream accepts a
        // list of strings/dicts; a str element is text content).
        if (part.is_string()) {
          ChatContentPart tp;
          tp.type = "text";
          tp.text = part.get<std::string>();
          if (!joined_text.empty()) joined_text += "\n";
          joined_text += tp.text;
          parts.push_back(std::move(tp));
          continue;
        }
        ChatContentPart p = ParseChatContentPart(part);
        if (p.type == "text") {
          if (!joined_text.empty()) joined_text += "\n";
          joined_text += p.text;
        }
        parts.push_back(std::move(p));
      }
      m.content = std::move(joined_text);
      m.content_parts = std::move(parts);
    }
  }
  // Multi-turn tool conversations: the assistant-history tool_calls and the
  // role="tool" reply's tool_call_id/name must survive parsing, or the chat
  // template cannot associate a tool result with the call that produced it.
  if (auto it = j.find("tool_calls"); it != j.end() && it->is_array()) {
    std::vector<ToolCall> calls;
    for (const nlohmann::json& c : *it) {
      ToolCall tc;
      GetOr(c, "id", tc.id);
      GetOr(c, "type", tc.type);
      if (auto fn = c.find("function"); fn != c.end() && fn->is_object()) {
        GetOr(*fn, "name", tc.function.name);
        GetOr(*fn, "arguments", tc.function.arguments);
      }
      calls.push_back(std::move(tc));
    }
    if (!calls.empty()) m.tool_calls = std::move(calls);
  }
  if (auto it = j.find("tool_call_id"); it != j.end() && it->is_string()) {
    m.tool_call_id = it->get<std::string>();
  }
  if (auto it = j.find("name"); it != j.end() && it->is_string()) {
    m.name = it->get<std::string>();
  }
  // Prior-turn reasoning round-trips so templates that re-render it can.
  if (auto it = j.find("reasoning"); it != j.end() && it->is_string()) {
    m.reasoning = it->get<std::string>();
  }
}

// Ported from: vllm/entrypoints/openai/chat_completion/protocol.py:165
// (ChatCompletionToolsParam) + engine/protocol.py:246 (FunctionDefinition).
// Templated on the json type: the ordered_json instantiation keeps
// `parameters` in the request document's key order (see protocol.h).
template <typename Json>
void FromJsonToolsParam(const Json& j, ChatCompletionToolsParam& t) {
  GetOr(j, "type", t.type);  // defaults to "function".
  if (auto it = j.find("function"); it != j.end() && it->is_object()) {
    const Json& fn = *it;
    GetOr(fn, "name", t.function.name);
    GetOpt(fn, "description", t.function.description);
    // `parameters` is the JSON-Schema object (kept as raw json, order
    // preserved for the ordered_json parse path).
    if (auto p = fn.find("parameters"); p != fn.end() && !p->is_null()) {
      t.function.parameters = *p;
    }
  }
}

void from_json(const nlohmann::json& j, ChatCompletionToolsParam& t) {
  FromJsonToolsParam(j, t);
}

void from_json(const nlohmann::ordered_json& j, ChatCompletionToolsParam& t) {
  FromJsonToolsParam(j, t);
}

void RestoreToolSchemaOrder(const std::string& request_body,
                            ChatCompletionRequest& r) {
  if (!r.tools.has_value() || r.tools->empty()) return;
  nlohmann::ordered_json ordered_body =
      nlohmann::ordered_json::parse(request_body);
  if (auto it = ordered_body.find("tools");
      it != ordered_body.end() && it->is_array()) {
    r.tools = it->get<std::vector<ChatCompletionToolsParam>>();
  }
}

void from_json(const nlohmann::json& j, ChatCompletionRequest& r) {
  if (auto it = j.find("messages"); it != j.end() && it->is_array()) {
    r.messages = it->get<std::vector<ChatMessage>>();
  }
  GetOpt(j, "model", r.model);
  GetOpt(j, "max_tokens", r.max_tokens);
  GetOpt(j, "max_completion_tokens", r.max_completion_tokens);
  GetOpt(j, "n", r.n);
  // SAMPLE-BEST-OF / SAMPLE-BEAM (chat_completion/protocol.py:249,597 + best_of).
  GetOpt(j, "best_of", r.best_of);
  GetOr(j, "use_beam_search", r.use_beam_search);
  GetOr(j, "length_penalty", r.length_penalty);
  GetOpt(j, "temperature", r.temperature);
  GetOpt(j, "top_p", r.top_p);
  GetOpt(j, "top_k", r.top_k);
  GetOpt(j, "min_p", r.min_p);
  GetOpt(j, "repetition_penalty", r.repetition_penalty);
  GetOr(j, "presence_penalty", r.presence_penalty);
  GetOr(j, "frequency_penalty", r.frequency_penalty);
  GetOpt(j, "seed", r.seed);
  r.stop = ParseStop(j);
  GetOr(j, "stop_token_ids", r.stop_token_ids);
  GetOr(j, "stream", r.stream);
  ParseStreamOptions(j, r.stream, r.stream_options);
  // chat_completion/protocol.py:739 (`check_logprobs`, mode="before"). The chat
  // endpoint's per-token count is `top_logprobs`; `logprobs` there is a bool.
  ValidateLogprobsPrefix(j, "top_logprobs");
  // :784-796, the chat-only suffix, and it is NOT the completion rule.
  // `top_logprobs` DOES carry the -1 "give me every vocabulary entry" sentinel
  // (a capability this tree already serves end to end — `ChatTopLogprobs` in
  // serving_utils.cpp reads -1 as "keep every entry"), and a count that would
  // emit a payload requires the `logprobs` bool to be set.
  if (auto tlp = j.find("top_logprobs");
      tlp != j.end() && !tlp->is_null() && tlp->is_number()) {
    const int count = tlp->get<int>();
    if (count < 0 && count != -1) {
      throw std::invalid_argument(
          "`top_logprobs` must be a positive value or -1.");
    }
    auto flag = j.find("logprobs");
    const bool logprobs_set =
        flag != j.end() && !flag->is_null() && flag->get<bool>();
    if ((count == -1 || count > 0) && !logprobs_set) {
      throw std::invalid_argument(
          "when using `top_logprobs`, `logprobs` must be set to true.");
    }
  }
  GetOr(j, "logprobs", r.logprobs);
  GetOr(j, "top_logprobs", r.top_logprobs);
  GetOr(j, "echo", r.echo);
  GetOpt(j, "prompt_logprobs", r.prompt_logprobs);
  GetOr(j, "min_tokens", r.min_tokens);
  GetOpt(j, "stream_interval", r.stream_interval);
  GetOr(j, "ignore_eos", r.ignore_eos);
  GetOr(j, "include_stop_str_in_output", r.include_stop_str_in_output);
  GetOr(j, "skip_special_tokens", r.skip_special_tokens);
  GetOr(j, "spaces_between_special_tokens", r.spaces_between_special_tokens);
  GetOr(j, "priority", r.priority);
  ParseResponseFormat(j, r.response_format);
  // tools / tool_choice (chat_completion/protocol.py:217-224).
  if (auto it = j.find("tools"); it != j.end() && it->is_array()) {
    r.tools = it->get<std::vector<ChatCompletionToolsParam>>();
  }
  // chat_template_kwargs (chat_completion/protocol.py:341). A non-object is
  // ignored rather than refused, matching every other optional field here; the
  // renderer then binds nothing and the template sees its own defaults.
  if (auto it = j.find("chat_template_kwargs");
      it != j.end() && it->is_object()) {
    r.chat_template_kwargs = nlohmann::ordered_json::parse(it->dump());
  }
  ParseLogitFilters(j, r.logit_bias, r.allowed_token_ids, r.bad_words);
  ParseToolChoice(j, r.tool_choice);
  // include_reasoning (chat_completion/protocol.py:242, default True).
  GetOr(j, "include_reasoning", r.include_reasoning);
  // DEFERRED: parallel_tool_calls, legacy functions / function_call — parsed and
  // ignored (nlohmann skips unknown keys; no field mapped here at T0).
}

// ---------------------------------------------------------------------------
// to_sampling_params
// ---------------------------------------------------------------------------

// completion/protocol.py:288-317 (CompletionRequest.to_sampling_params) and its
// chat twin, as one function per knob:
//
//   if (value := request.field) is None:
//       value = default_sampling_params.get(name, _DEFAULT_SAMPLING_PARAMS[name])
//
// The three-way order is the whole point and is easy to get subtly wrong, so it
// is written once: an explicitly SENT request value wins, an OMITTED one takes
// the checkpoint's server-wide default, and only a knob neither of them
// declares reaches the neutral OpenAI value. A request that explicitly sends
// the neutral value (top_k = 0, "turn top-k off") is a SENT value and is not
// overridden -- which is why the request fields are optionals upstream and here.
template <typename T, typename Member>
T ResolveSamplingKnob(const std::optional<T>& requested,
                      const DefaultSamplingParams* defaults, Member member,
                      T neutral) {
  if (requested.has_value()) return *requested;
  if (defaults != nullptr && (defaults->*member).has_value()) {
    return static_cast<T>(*(defaults->*member));
  }
  return neutral;
}


SamplingParams CompletionRequest::to_sampling_params(
    std::optional<int> default_max_tokens, const DefaultSamplingParams* defaults) const {
  // completion/protocol.py:260. None sampling knobs resolve to
  // _DEFAULT_SAMPLING_PARAMS (no server-provided default_sampling_params in T0).
  SamplingParams sp;
  sp.n = n;
  sp.presence_penalty = presence_penalty;
  sp.frequency_penalty = frequency_penalty;
  sp.repetition_penalty = ResolveSamplingKnob(
      repetition_penalty, defaults, &DefaultSamplingParams::repetition_penalty,
      kDefaultRepetitionPenalty);
  sp.temperature = ResolveSamplingKnob(
      temperature, defaults, &DefaultSamplingParams::temperature, kDefaultTemperature);
  sp.top_p =
      ResolveSamplingKnob(top_p, defaults, &DefaultSamplingParams::top_p, kDefaultTopP);
  sp.top_k =
      ResolveSamplingKnob(top_k, defaults, &DefaultSamplingParams::top_k, kDefaultTopK);
  sp.min_p =
      ResolveSamplingKnob(min_p, defaults, &DefaultSamplingParams::min_p, kDefaultMinP);
  sp.seed = seed;
  sp.stop = stop;
  sp.stop_token_ids = stop_token_ids;
  sp.logprobs = logprobs;
  // prompt_logprobs: fall back to logprobs when echo is set (protocol.py:303).
  sp.prompt_logprobs =
      prompt_logprobs.has_value() ? prompt_logprobs : (echo ? logprobs : std::nullopt);
  sp.ignore_eos = ignore_eos;
  // A non-positive max_tokens means "no client-side limit": Hermes and some
  // OpenAI clients send -1 for that. It is UNSET, not a number -- the engine
  // then generates to max_model_len - seq_len (v1/engine/input_processor.cpp,
  // mirroring vllm input_processor.py:317-321). Substituting any constant here
  // would silently truncate exactly the long-context request that asked to be
  // left unlimited, and PostInit rejects <1, which is what made a clamp look
  // necessary.
  sp.max_tokens = (max_tokens.has_value() && *max_tokens > 0)
                      ? max_tokens
                      : default_max_tokens;
  sp.min_tokens = min_tokens;
  sp.skip_special_tokens = skip_special_tokens;
  sp.spaces_between_special_tokens = spaces_between_special_tokens;
  sp.include_stop_str_in_output = include_stop_str_in_output;
  sp.output_kind =
      stream ? RequestOutputKind::kDelta : RequestOutputKind::kFinalOnly;
  // stream_interval (completion/protocol.py:378 @ vllm#49754).
  sp.stream_interval = stream_interval;
  // logit_bias / allowed_token_ids / bad_words (completion/protocol.py:369-371).
  ApplyLogitFilters(sp, logit_bias, allowed_token_ids, bad_words);
  // response_format -> structured_outputs (completion/protocol.py:309-338).
  ApplyResponseFormat(response_format, sp);
  // best_of: fan-out count + forced ranking logprob (AFTER sp.logprobs is set so
  // the forced logprob is not overwritten). PostInit() then runs the greedy n
  // check on the (possibly best_of) n.
  ApplyBestOf(sp, best_of, n);
  sp.PostInit();
  return sp;
}

SamplingParams ChatCompletionRequest::to_sampling_params(
    std::optional<int> default_max_tokens, const DefaultSamplingParams* defaults) const {
  // chat_completion/protocol.py:585.
  SamplingParams sp;
  sp.n = n.value_or(1);
  sp.presence_penalty = presence_penalty;
  sp.frequency_penalty = frequency_penalty;
  sp.repetition_penalty = ResolveSamplingKnob(
      repetition_penalty, defaults, &DefaultSamplingParams::repetition_penalty,
      kDefaultRepetitionPenalty);
  sp.temperature = ResolveSamplingKnob(
      temperature, defaults, &DefaultSamplingParams::temperature, kDefaultTemperature);
  sp.top_p =
      ResolveSamplingKnob(top_p, defaults, &DefaultSamplingParams::top_p, kDefaultTopP);
  sp.top_k =
      ResolveSamplingKnob(top_k, defaults, &DefaultSamplingParams::top_k, kDefaultTopK);
  sp.min_p =
      ResolveSamplingKnob(min_p, defaults, &DefaultSamplingParams::min_p, kDefaultMinP);
  sp.seed = seed;
  sp.stop = stop;
  sp.stop_token_ids = stop_token_ids;
  // logprobs = top_logprobs if logprobs(bool) else None (protocol.py:677).
  sp.logprobs = logprobs ? std::optional<int>(top_logprobs) : std::nullopt;
  sp.prompt_logprobs =
      prompt_logprobs.has_value() ? prompt_logprobs : (echo ? std::optional<int>(top_logprobs) : std::nullopt);
  sp.ignore_eos = ignore_eos;
  // serving prefers max_completion_tokens over max_tokens
  // (chat_completion/serving.py:299).
  std::optional<int> req_max =
      max_completion_tokens.has_value() ? max_completion_tokens : max_tokens;
  // Same rule as the completions path above: non-positive means UNSET.
  sp.max_tokens =
      (req_max.has_value() && *req_max > 0) ? req_max : default_max_tokens;
  sp.min_tokens = min_tokens;
  sp.skip_special_tokens = skip_special_tokens;
  sp.spaces_between_special_tokens = spaces_between_special_tokens;
  sp.include_stop_str_in_output = include_stop_str_in_output;
  sp.output_kind =
      stream ? RequestOutputKind::kDelta : RequestOutputKind::kFinalOnly;
  // stream_interval (chat_completion/protocol.py @ vllm#49754).
  sp.stream_interval = stream_interval;
  // logit_bias / allowed_token_ids / bad_words (chat_completion/protocol.py:694-697).
  ApplyLogitFilters(sp, logit_bias, allowed_token_ids, bad_words);
  // response_format -> structured_outputs (chat_completion/protocol.py:629-658).
  ApplyResponseFormat(response_format, sp);
  // best_of: see CompletionRequest::to_sampling_params.
  ApplyBestOf(sp, best_of, n.value_or(1));
  sp.PostInit();
  return sp;
}

// ---------------------------------------------------------------------------
// to_beam_search_params
// ---------------------------------------------------------------------------

vllm::BeamSearchParams CompletionRequest::to_beam_search_params(
    int max_tokens_in, const DefaultSamplingParams* defaults) const {
  // completion/protocol.py:260-279. beam_width == n; temperature None => 1.0.
  vllm::BeamSearchParams bp;
  bp.beam_width = n;
  bp.max_tokens = max_tokens_in;
  bp.ignore_eos = ignore_eos;
  bp.temperature = ResolveSamplingKnob(
      temperature, defaults, &DefaultSamplingParams::temperature, kDefaultTemperature);
  bp.length_penalty = length_penalty;
  return bp;
}

vllm::BeamSearchParams ChatCompletionRequest::to_beam_search_params(
    int max_tokens_in, const DefaultSamplingParams* defaults) const {
  // chat_completion/protocol.py:589-606. beam_width == n; temperature None => 1.0.
  vllm::BeamSearchParams bp;
  bp.beam_width = n.value_or(1);
  bp.max_tokens = max_tokens_in;
  bp.ignore_eos = ignore_eos;
  bp.temperature = ResolveSamplingKnob(
      temperature, defaults, &DefaultSamplingParams::temperature, kDefaultTemperature);
  bp.length_penalty = length_penalty;
  return bp;
}

// ---------------------------------------------------------------------------
// Response serialization (to_json)
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const UsageInfo& u) {
  j = nlohmann::json{
      {"prompt_tokens", u.prompt_tokens},
      {"total_tokens", u.total_tokens},
      {"completion_tokens", u.completion_tokens},
  };
}

void to_json(nlohmann::json& j, const ErrorInfo& e) {
  j = nlohmann::json{
      {"message", e.message},
      {"type", e.type},
      {"param", OrNull(e.param)},
      {"code", e.code.has_value() ? nlohmann::json(*e.code) : nlohmann::json(nullptr)},
  };
}

void to_json(nlohmann::json& j, const ErrorResponse& e) {
  j = nlohmann::json{{"error", e.error}};
}

// completion/protocol.py:580 (CompletionLogProbs). token_logprobs / top_logprobs
// carry explicit nulls at positions without logprobs (list[float|None] /
// list[dict|None]); text_offset + tokens are plain arrays.
void to_json(nlohmann::json& j, const CompletionLogProbs& lp) {
  j = nlohmann::json::object();
  j["text_offset"] = lp.text_offset;
  j["tokens"] = lp.tokens;
  nlohmann::json tl = nlohmann::json::array();
  for (const auto& v : lp.token_logprobs) {
    tl.push_back(v.has_value() ? nlohmann::json(*v) : nlohmann::json(nullptr));
  }
  j["token_logprobs"] = std::move(tl);
  nlohmann::json top = nlohmann::json::array();
  for (const auto& m : lp.top_logprobs) {
    if (!m.has_value()) {
      top.push_back(nullptr);
      continue;
    }
    nlohmann::json obj = nlohmann::json::object();
    for (const auto& [tok, val] : *m) obj[tok] = val;
    top.push_back(std::move(obj));
  }
  j["top_logprobs"] = std::move(top);
}

// chat_completion/protocol.py:75 (ChatCompletionLogProb). bytes is null only
// when the token has no decoded form.
void to_json(nlohmann::json& j, const ChatCompletionLogProb& lp) {
  j = nlohmann::json{{"token", lp.token}, {"logprob", lp.logprob}};
  j["bytes"] =
      lp.bytes.has_value() ? nlohmann::json(*lp.bytes) : nlohmann::json(nullptr);
}

// chat_completion/protocol.py:81 (ChatCompletionLogProbsContent).
void to_json(nlohmann::json& j, const ChatCompletionLogProbsContent& lp) {
  j = nlohmann::json{{"token", lp.token}, {"logprob", lp.logprob}};
  j["bytes"] =
      lp.bytes.has_value() ? nlohmann::json(*lp.bytes) : nlohmann::json(nullptr);
  j["top_logprobs"] = lp.top_logprobs;
}

// chat_completion/protocol.py:88 (ChatCompletionLogProbs).
void to_json(nlohmann::json& j, const ChatCompletionLogProbs& lp) {
  j = nlohmann::json::object();
  j["content"] = lp.content.has_value() ? nlohmann::json(*lp.content)
                                        : nlohmann::json(nullptr);
}

// PromptLogprobs (`list[dict[int, Logprob] | None]`, vllm/logprobs.py:27) as the
// server dumps it: a JSON array whose entries are `null` (a position with no
// distribution — the first prompt token has no predecessor) or an object keyed
// by the DECIMAL token id, each value the `Logprob` dataclass
// {logprob, rank, decoded_token}. `model_dump()` keeps a None field, so `rank`
// and `decoded_token` are explicit nulls rather than omitted keys.
//
// The key order is the insertion order our LogprobsOnePosition records, which is
// the Python dict order upstream serializes (`order` + `entries`, see
// include/vllm/logprobs.h).
static nlohmann::json PromptLogprobsToJson(const vllm::PromptLogprobs& plp) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& pos : plp) {
    if (!pos.has_value()) {
      arr.push_back(nullptr);
      continue;
    }
    nlohmann::json obj = nlohmann::json::object();
    for (const int32_t token_id : pos->order) {
      const vllm::Logprob* lp = pos->find(token_id);
      if (lp == nullptr) continue;
      nlohmann::json entry = nlohmann::json::object();
      entry["logprob"] = lp->logprob;
      entry["rank"] =
          lp->rank.has_value() ? nlohmann::json(*lp->rank) : nlohmann::json(nullptr);
      entry["decoded_token"] = lp->decoded_token.has_value()
                                   ? nlohmann::json(*lp->decoded_token)
                                   : nlohmann::json(nullptr);
      obj[std::to_string(token_id)] = std::move(entry);
    }
    arr.push_back(std::move(obj));
  }
  return arr;
}

void to_json(nlohmann::json& j, const CompletionResponseChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"text", c.text},
      {"logprobs",
       c.logprobs.has_value() ? nlohmann::json(*c.logprobs) : nlohmann::json(nullptr)},
      {"finish_reason", OrNull(c.finish_reason)},
  };
  // completion/protocol.py:601 — present as an explicit null when unset.
  j["prompt_logprobs"] = c.prompt_logprobs.has_value()
                             ? PromptLogprobsToJson(*c.prompt_logprobs)
                             : nlohmann::json(nullptr);
}

void to_json(nlohmann::json& j, const CompletionResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
      {"usage", r.usage},
  };
}

void to_json(nlohmann::json& j, const CompletionResponseStreamChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"text", c.text},
      {"logprobs",
       c.logprobs.has_value() ? nlohmann::json(*c.logprobs) : nlohmann::json(nullptr)},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const CompletionStreamResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
  };
  if (r.usage.has_value()) j["usage"] = *r.usage;
}

// engine/protocol.py:310 (FunctionCall). Only name + arguments on the wire (the
// internal id is excluded upstream).
void to_json(nlohmann::json& j, const FunctionCall& f) {
  j = nlohmann::json{{"name", f.name}, {"arguments", f.arguments}};
}

// engine/protocol.py:319 (ToolCall). {id, type:"function", function{...}}.
void to_json(nlohmann::json& j, const ToolCall& t) {
  j = nlohmann::json{
      {"id", t.id},
      {"type", t.type},
      {"function", t.function},
  };
}

// engine/protocol.py:325 (DeltaFunctionCall). name/arguments emitted only when
// present (per-chunk deltas).
void to_json(nlohmann::json& j, const DeltaFunctionCall& f) {
  j = nlohmann::json::object();
  if (f.name.has_value()) j["name"] = *f.name;
  if (f.arguments.has_value()) j["arguments"] = *f.arguments;
}

// engine/protocol.py:331 (DeltaToolCall). index is always present; id/type
// emitted only on the chunk that introduces the call.
void to_json(nlohmann::json& j, const DeltaToolCall& t) {
  j = nlohmann::json{{"index", t.index}, {"function", t.function}};
  if (t.id.has_value()) j["id"] = *t.id;
  if (t.type.has_value()) j["type"] = *t.type;
}

// chat_completion/protocol.py:57 (ChatMessage). content is always serialized
// (null when a tool call carries no text); tool_calls is emitted only when
// non-empty (upstream _serialize pops an empty list).
void to_json(nlohmann::json& j, const ChatMessage& m) {
  j = nlohmann::json{{"role", m.role}, {"content", OrNull(m.content)}};
  // reasoning (chat_completion/protocol.py:67): present + non-empty only.
  if (m.reasoning.has_value() && !m.reasoning->empty()) {
    j["reasoning"] = *m.reasoning;
  }
  // Round-trip symmetry for tool-turn identity (input-side fields; response
  // messages never carry them, so present-only serialization is a no-op there).
  if (m.tool_call_id.has_value()) j["tool_call_id"] = *m.tool_call_id;
  if (m.name.has_value()) j["name"] = *m.name;
  if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
    j["tool_calls"] = *m.tool_calls;
  }
}

// engine/protocol.py:350 (DeltaMessage). tool_calls emitted only when non-empty.
void to_json(nlohmann::json& j, const DeltaMessage& m) {
  j = nlohmann::json::object();
  if (m.role.has_value()) j["role"] = *m.role;
  if (m.content.has_value()) j["content"] = *m.content;
  // reasoning (engine/protocol.py:353): mirror content; emitted when present.
  if (m.reasoning.has_value()) j["reasoning"] = *m.reasoning;
  if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
    j["tool_calls"] = *m.tool_calls;
  }
}

void to_json(nlohmann::json& j, const ChatCompletionResponseChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"message", c.message},
      {"logprobs",
       c.logprobs.has_value() ? nlohmann::json(*c.logprobs) : nlohmann::json(nullptr)},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const ChatCompletionResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
      {"usage", r.usage},
  };
  // chat_completion/protocol.py:126 — top-level, and an explicit null when unset.
  j["prompt_logprobs"] = r.prompt_logprobs.has_value()
                             ? PromptLogprobsToJson(*r.prompt_logprobs)
                             : nlohmann::json(nullptr);
}

void to_json(nlohmann::json& j, const ChatCompletionResponseStreamChoice& c) {
  j = nlohmann::json{
      {"index", c.index},
      {"delta", c.delta},
      {"logprobs",
       c.logprobs.has_value() ? nlohmann::json(*c.logprobs) : nlohmann::json(nullptr)},
      {"finish_reason", OrNull(c.finish_reason)},
  };
}

void to_json(nlohmann::json& j, const ChatCompletionStreamResponse& r) {
  j = nlohmann::json{
      {"id", r.id},
      {"object", r.object},
      {"created", r.created},
      {"model", r.model},
      {"choices", r.choices},
  };
  if (r.usage.has_value()) j["usage"] = *r.usage;
}

}  // namespace vllm::entrypoints::openai
