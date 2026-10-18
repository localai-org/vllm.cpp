// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 + the
// per-endpoint api_router modules. See api_server.h for scope + the cpp-httplib
// dependency deviation.
#include "vllm/entrypoints/openai/api_server.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <utility>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/http_transport_abi.h"
#include "vllm/entrypoints/chat_template.h"  // ChatTemplateError -> HTTP 400
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/request_logger.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"  // InputValidationError -> HTTP 400
#include "vllm/v1/metrics/loggers.h"

namespace vllm::entrypoints::openai {

namespace {

// Build the OpenAI ErrorResponse JSON body for a failed request
// (serve/utils/error_response.py::create_error_response). `code` == the HTTP
// status code (upstream ErrorInfo.code carries it).
ApiServer::DispatchResult MakeError(int status, const std::string& type,
                                    const std::string& message) {
  ErrorResponse err;
  err.error.message = message;
  err.error.type = type;
  err.error.code = status;
  ApiServer::DispatchResult r;
  r.status = status;
  r.content_type = "application/json";
  r.body = nlohmann::json(err).dump();
  return r;
}

}  // namespace

#ifdef VLLM_WITH_DIARIZATION
static ApiServer::DispatchResult HandleAudioDiarizations(
    const ApiServer& server,
    const std::string& file_bytes,
    const std::string& /*response_format*/) {
  auto diarizer = server.diarizer_callback();
  if (!diarizer) {
    return MakeError(404, "NotFoundError", "diarization not enabled");
  }
  try {
    auto segs = diarizer(reinterpret_cast<const uint8_t*>(file_bytes.data()),
                         file_bytes.size());
    nlohmann::json j = nlohmann::json::array();
    for (const auto& s : segs) {
      j.push_back({{"speaker", s.speaker},
                    {"start", s.start},
                    {"end", s.end}});
    }
    ApiServer::DispatchResult r;
    r.status = 200;
    r.content_type = "application/json";
    r.body = j.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

static ApiServer::DispatchResult HandleAudioSas(
    const ApiServer& server,
    const std::string& file_bytes,
    const std::string& /*response_format*/) {
  auto sas_fn = server.sas_callback();
  if (!sas_fn) {
    return MakeError(404, "NotFoundError", "speaker-attributed ASR not enabled");
  }
  try {
    auto result = sas_fn(reinterpret_cast<const uint8_t*>(file_bytes.data()),
                         file_bytes.size());
    nlohmann::json j;
    j["segments"] = nlohmann::json::array();
    for (const auto& u : result.utterances) {
      j["segments"].push_back({
        {"speaker", u.speaker},
        {"text", u.text},
        {"start", u.start},
        {"end", u.end},
        {"confidence", u.conf}
      });
    }
    ApiServer::DispatchResult r;
    r.status = 200;
    r.content_type = "application/json";
    r.body = j.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}
#endif


// SystemOne helpers (ParseSystemOneBody, BuildSystemOneAnswer*, R2, R4, etc.)
// are defined in systemone.h/.cpp. Imported here so the handlers below can
// call them unqualified, matching the former anonymous-namespace usage.
using namespace systemone;

namespace {

size_t HttpWorkerCount(size_t max_concurrent_streams) {
  if (max_concurrent_streams == 0) {
    throw std::invalid_argument("max_concurrent_streams must be positive");
  }
  if (max_concurrent_streams >
      std::numeric_limits<size_t>::max() -
          ApiServer::kControlWorkerHeadroom) {
    throw std::invalid_argument("max_concurrent_streams is too large");
  }
  return max_concurrent_streams + ApiServer::kControlWorkerHeadroom;
}

size_t HttpWorkerCount(size_t max_concurrent_streams,
                       ApiServer::HttpWorkerPoolMode mode) {
  const size_t fixed_count = HttpWorkerCount(max_concurrent_streams);
  return mode == ApiServer::HttpWorkerPoolMode::kCapacityFixed ? fixed_count
                                                               : 0;
}

}  // namespace

// The listener type this server constructs, named ONCE so that
// `vllm::ApiServerListenerIsTls()` below and the member two dozen lines down
// cannot disagree. The same `CPPHTTPLIB_OPENSSL_SUPPORT` define that gives the
// HuggingFace fetcher `https` also makes `httplib::SSLServer` compilable, and
// ENG-HF-MODEL-DOWNLOAD W5 deliberately does NOT enable a listener: that is a
// separate decision with its own certificate, key, flags and documentation.
// Changing this alias changes the listener AND the reported value together.
using ApiServerListener = httplib::Server;

// Opaque httplib::Server (pimpl — keeps httplib.h out of api_server.h).
struct ApiServer::Impl {
  Impl(size_t max_concurrent_streams, HttpWorkerPoolMode mode)
      : http_worker_count(HttpWorkerCount(max_concurrent_streams, mode)) {
    // cpp-httplib's default pool starts at hardware_concurrency()-1 and only
    // grows if idle_thread_count_ is exactly zero at enqueue. A burst can queue
    // accepted sockets while that counter is stale-positive; long-lived SSE
    // jobs then prevent the queued sockets from ever being read. A fixed floor
    // derived from the configured stream capacity removes that race and makes
    // resource use reproducible.
    if (http_worker_count != 0) {
      server.new_task_queue = [workers = http_worker_count]() {
        return new httplib::ThreadPool(workers);
      };
    }
    // Mirror vLLM's serving transport: vLLM serves through uvicorn over asyncio
    // (entrypoints/launcher.py:71,76), and asyncio disables Nagle on every
    // accepted TCP stream socket by default (CPython asyncio/base_events.py
    // _set_nodelay → setsockopt(IPPROTO_TCP, TCP_NODELAY, 1), invoked from
    // selector_events.py _SelectorSocketTransport). cpp-httplib defaults it off
    // (third_party/httplib/httplib.h:142) and applies it to the accepted socket
    // only when tcp_nodelay_ is set (httplib.h:12083). Per-token SSE frames are
    // tiny writes, so enabling TCP_NODELAY here puts each streamed frame on the
    // wire immediately instead of coalescing it under Nagle/delayed-ACK.
    server.set_tcp_nodelay(true);
  }

  ApiServerListener server;
  size_t http_worker_count;
  // The legacy LLMEngine serving constructors remain for small synthetic
  // tests and embedding compatibility. Unlike AsyncLLM, that engine is driven
  // synchronously by its caller, so retain one shared lock for that seam only.
  // Production handlers use AsyncLLM and never take this request-level lock.
  std::mutex legacy_engine_mutex;
};

ApiServer::ApiServer(OpenAIServingCompletion& completion,
                     OpenAIServingChat& chat, OpenAIServingModels& models,
                     std::string version, size_t max_concurrent_streams,
                     HttpWorkerPoolMode worker_pool_mode)
    : completion_(&completion),
      chat_(&chat),
      models_(models),
      version_(std::move(version)),
      impl_(std::make_unique<Impl>(max_concurrent_streams, worker_pool_mode)) {}

// Serving-less construction (transcription-only servers, ARCH-ONE-SURFACE
// ROW 1): no AsyncLLM exists, so the generate handlers stay null and their
// routes are not registered — vLLM's task-conditional route registration
// (api_server.py:255-265) expressed at construction.
ApiServer::ApiServer(OpenAIServingModels& models, std::string version,
                     size_t max_concurrent_streams,
                     HttpWorkerPoolMode worker_pool_mode)
    : models_(models),
      version_(std::move(version)),
      impl_(std::make_unique<Impl>(max_concurrent_streams, worker_pool_mode)) {}

ApiServer::~ApiServer() {
  // Drain the async /v1/videos workers before the job store they write into is
  // destroyed. Threads are joined, never detached, precisely so this ordering is
  // guaranteed rather than hoped for.
  std::vector<std::thread> workers;
  {
    std::lock_guard<std::mutex> lock(video_workers_mutex_);
    workers.swap(video_workers_);
  }
  for (auto& worker : workers) {
    if (worker.joinable()) worker.join();
  }
}

// ── SERVE-REQUEST-LENGTH-GUARD (#1541) ──────────────────────────────────────
// A REFUSING byte bound at the request boundary, checked BEFORE any
// tokenization. Its two constraints come from
// .agents/specs/bpe-quadratic-merge.md `## Defence in depth`:
//
//   1. It refuses and NEVER truncates. Shortening a prompt returns model output
//      for text the caller did not send.
//   2. It is here, at the boundary, and not in
//      vllm::v1::InputProcessor::ValidatePromptLen, which needs the token count
//      the expensive step produces and so cannot run before that step.
//
// vLLM HAS NO EQUIVALENT for a prompt BYTE bound, and this is checked rather
// than assumed at pin 555967922:
//   - vllm/entrypoints/openai/cli_args.py:292-294 `h11_max_incomplete_event_size`
//     (default 4 MB, vllm/entrypoints/serve/utils/constants.py:9) is an h11
//     PARSER limit. Its own docstring says "header or body", but h11 applies it
//     to the undrained receive buffer only (h11/_connection.py:485, whose
//     comment reads "431 is Request header fields too large which is pretty
//     much the only situation where we can get here"), so a body is not bounded
//     by it.
//   - vllm/entrypoints/openai/completion/protocol.py:536-553
//     `validate_prompt_list_length` bounds the COUNT of prompts in a list
//     (VLLM_MAX_COMPLETION_PROMPTS, default 1024, vllm/envs.py:110), not their
//     bytes. It is the REGISTER this refusal mirrors: refused during request
//     validation, ahead of the router, with the limit named in the message.
//   - vllm/entrypoints/speech_to_text/base/utils.py:38-46
//     `read_upload_with_limit` IS a refusing byte bound at the request boundary
//     (VLLM_MAX_AUDIO_CLIP_FILESIZE_MB, default 25, vllm/envs.py:79) -- on the
//     audio-upload surface, not on a text prompt.
// So the shape is mirrored and the number is ours, which is why it is derived
// below rather than picked.
std::optional<ApiServer::DispatchResult> ApiServer::refuse_oversized_prompt(
    size_t prompt_bytes) const {
  if (max_prompt_bytes_ == 0 || prompt_bytes <= max_prompt_bytes_) {
    return std::nullopt;
  }
  return MakeError(
      400, "BadRequestError",
      "prompt length " + std::to_string(prompt_bytes) +
          " bytes exceeds the maximum allowed prompt length of " +
          std::to_string(max_prompt_bytes_) + " bytes (max_model_len " +
          std::to_string(max_model_len_) + " x " +
          std::to_string(max_token_bytes_) +
          " bytes, the longest token in this tokenizer's vocabulary). A prompt "
          "this long cannot fit in " +
          std::to_string(max_model_len_) +
          " tokens, so it is refused here rather than tokenized first. The "
          "request is refused, not truncated.");
}

void ApiServer::set_tokenizer(const vllm::tok::Tokenizer* tokenizer,
                              int64_t max_model_len) {
  tokenizer_ = tokenizer;
  max_model_len_ = max_model_len;
  // The bound is exactly `max_model_len * MaxTokenBytes()` -- see
  // max_prompt_bytes(). Tokenizer::MaxPromptBytes holds the one copy of the
  // derivation: 0 (unbounded) whenever either factor is unknown, and clamped
  // rather than wrapped on the overflow a hostile max_model_len could produce.
  max_token_bytes_ = tokenizer != nullptr ? tokenizer->MaxTokenBytes() : 0;
  max_prompt_bytes_ =
      tokenizer != nullptr ? tokenizer->MaxPromptBytes(max_model_len) : 0;
}

ApiServer::DispatchResult ApiServer::handle_completions(
    const std::string& request_body) {
  if (completion_ == nullptr) {
    // vLLM's api_router `if handler is None: raise NotImplementedError` mirror
    // for a serving-less (transcription-only) server; the socket layer never
    // registers the route in that mode, so this answers direct dispatch only.
    return MakeError(500, "InternalServerError",
                     "The model does not support Completions API "
                     "(transcription-only server)");
  }
  // completion/api_router.py:46 (create_completion): parse → check_model →
  // handler → JSON (non-stream) or text/event-stream (stream).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  CompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  // SERVE-REQUEST-LENGTH-GUARD (#1541): before check_model and before the
  // encode inside create_completion. Ahead of the model lookup because the
  // bound is a property of the REQUEST, which is where vLLM decides its own
  // analogue too -- validate_prompt_list_length is a pydantic model_validator
  // (completion/protocol.py:536), so it runs at body validation, ahead of the
  // router's check_model.
  if (auto refusal = refuse_oversized_prompt(request.prompt.size())) {
    return *refusal;
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  CompletionResult result;
  try {
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!completion_->uses_async_engine()) legacy_lock.lock();
    result = completion_->create_completion(request);
  } catch (const vllm::v1::InputValidationError& e) {
    // The request itself is unservable (today: a prompt at or past
    // max_model_len). Upstream raises ValueError from _validate_prompt_len and
    // create_error_response maps ValueError to BadRequestError / 400
    // (serve/utils/error_response.py:62-65). Caught AHEAD of the generic arm
    // below, which would otherwise report a client mistake as a server fault.
    return MakeError(400, "BadRequestError", e.what());
  } catch (const std::exception& e) {
    // DISCRIMINATOR: attribute a 500 to its endpoint + model + raw cause so a
    // benchmark driver that only sees the generic HTTP body can still recover
    // the true failure. std::cerr only (survives SIGKILL escalation).
    std::cerr << "api-server: 500 endpoint=/v1/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_chat_completions(
    const std::string& request_body) {
  if (chat_ == nullptr) {
    return MakeError(500, "InternalServerError",
                     "The model does not support Chat Completions API "
                     "(transcription-only server)");
  }
  {
    LogHttpIngress("POST", "/v1/chat/completions", request_body.size());
  }
  // chat_completion/api_router.py:53 (create_chat_completion).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  ChatCompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  // The pinned renderer serializes tool schemas into the prompt with
  // sort_keys=False, preserving the request document's key order; the
  // nlohmann::json parse above sorts object keys (std::map), so re-read
  // `tools` from an order-preserving parse (no-op without tools).
  RestoreToolSchemaOrder(request_body, request);
  // SERVE-REQUEST-LENGTH-GUARD (#1541). The measured quantity is the SUM of the
  // message texts, because that is what the chat template concatenates into the
  // one prompt the tokenizer then sees. `content` carries the joined text spans
  // even when the wire form was a content-part ARRAY (protocol.h ChatMessage),
  // so inline base64 media -- which lives in `content_parts`, is never
  // tokenized as text, and would make a raw body-byte bound refuse legitimate
  // multimodal requests -- is correctly not counted here.
  size_t prompt_bytes = 0;
  for (const ChatMessage& m : request.messages) {
    if (m.content.has_value()) prompt_bytes += m.content->size();
  }
  if (auto refusal = refuse_oversized_prompt(prompt_bytes)) {
    return *refusal;
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  ChatCompletionResult result;
  try {
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!chat_->uses_async_engine()) legacy_lock.lock();
    result = chat_->create_chat_completion(request);
  } catch (const vllm::v1::InputValidationError& e) {
    // Same mapping as /v1/completions above (error_response.py:62-65).
    LogRequestError("", "/v1/chat/completions", e.what());
    return MakeError(400, "BadRequestError", e.what());
  } catch (const vllm::entrypoints::ChatTemplateError& e) {
    // A render failure is a CLIENT error, because the conversation and the
    // chat_template_kwargs that reached the template are the request's. Upstream
    // reaches 400 twice over: safe_apply_chat_template wraps ANY exception out
    // of apply_chat_template into a ValueError (vllm/renderers/hf.py:785-789 @
    // 555967922), and create_error_response maps ValueError/TypeError
    // (error_response.py:48-52) AND jinja2.TemplateError and its subclasses
    // (error_response.py:61-65) to BadRequestError. Without this arm the render
    // fell through to the generic 500 below, and /tokenize -- which already
    // answers 400 for the identical body -- disagreed with this endpoint.
    LogRequestError("", "/v1/chat/completions", e.what());
    return MakeError(400, "BadRequestError", e.what());
  } catch (const std::exception& e) {
    std::cerr << "api-server: 500 endpoint=/v1/chat/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    LogRequestError("", "/v1/chat/completions", e.what());
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_models() const {
  // models/api_router.py:21 (show_available_models).
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json(models_.show_available_models()).dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_health() const {
  // Upstream calls engine_client.check_health() before returning an empty 200.
  // This bounded server currently exposes process liveness only.
  DispatchResult out;
  out.status = 200;
  out.content_type = "text/plain";
  out.body.clear();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_version() const {
  // serve/instrumentator/basic.py:53 — {"version": <ver>}.
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json{{"version", version_}}.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_ping() const {
  // sagemaker/api_router.py:47-50 — GET/POST /ping is a liveness probe that
  // returns the same empty 200 as /health.
  return handle_health();
}

namespace {

ApiServer::DispatchResult VideoJsonOk(std::string body) {
  ApiServer::DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = std::move(body);
  return out;
}

}  // namespace

std::string ApiServer::video_model_warning(
    const ::vllm::openai::VideoRequest& request) const {
  // OpenAI clients send the SORA model name ("sora-2-pro"); this server generates
  // with whatever video model it was started with, whose name they cannot know.
  // Refusing would defeat the compatibility, and ignoring would hide a real
  // mismatch, so the request is honoured and the divergence is STATED on the job.
  if (request.model.empty() || models_.is_base_model(request.model)) return {};
  return "requested model '" + request.model +
         "' is not a served model ('" + models_.model_name() +
         "'); generated with the video model this server was started with";
}

ApiServer::DispatchResult ApiServer::handle_audio_transcriptions(
    const std::string& file_bytes, const std::string& response_format) const {
  // Mirror of vLLM speech_to_text/transcription: api_router.py:31
  // `create_transcriptions` reads the multipart upload
  // (read_upload_with_limit) and hands the bytes to
  // OpenAIServingTranscription.create_transcription (serving.py:50), which
  // answers TranscriptionResponse {"text": ...} for response_format json and
  // the raw text otherwise. The transcription itself runs through the ONE
  // library seam (ParakeetTranscriber) — the same code path vllm_transcribe
  // drives, so HTTP and FFI cannot drift.
  if (!transcriber_) {
    // The api_router `if handler is None: raise NotImplementedError` mirror;
    // the socket layer never registers the route without a transcriber.
    return MakeError(500, "InternalServerError",
                     "The model does not support Transcriptions API");
  }
  if (file_bytes.empty()) {
    return MakeError(400, "BadRequestError",
                     "Expected a non-empty `file` upload (16-bit PCM mono "
                     "RIFF/WAVE)");
  }
  const std::string fmt = response_format.empty() ? "json" : response_format;
  if (fmt != "json" && fmt != "text") {
    // verbose_json / srt / vtt are NAMED RESIDUALS of this fold (protocol.py
    // AudioResponseFormat lists them; nothing here produces segment timing).
    return MakeError(400, "BadRequestError",
                     "response_format '" + fmt +
                         "' is not supported (supported: json, text; "
                         "verbose_json/srt/vtt are named residuals)");
  }
  try {
    const ::vllm::multimodal::ParakeetTranscription result = transcriber_(
        reinterpret_cast<const uint8_t*>(file_bytes.data()), file_bytes.size());
    if (!result.has_text) {
      return MakeError(500, "InternalServerError",
                       "the checkpoint ships no tokenizer.json, so ids-only "
                       "transcription has no OpenAI response shape");
    }
    DispatchResult r;
    if (fmt == "text") {
      r.content_type = "text/plain; charset=utf-8";
      r.body = result.text;
    } else {
      r.body = nlohmann::json{{"text", result.text}}.dump();
    }
    return r;
  } catch (const std::exception& e) {
    // Undecodable audio (not RIFF/WAVE, not PCM16 mono, wrong sample rate) is
    // a caller error; the pipeline names the cause.
    return MakeError(400, "BadRequestError", e.what());
  }
}

// ── Speaker diarization handler (ABI v30) ────────────────────────────────
#ifdef VLLM_WITH_DIARIZATION
ApiServer::DispatchResult ApiServer::handle_audio_diarizations(
    const std::string& file_bytes,
    const std::string& /*response_format*/) const {
  if (!diarizer_) {
    return MakeError(500, "InternalServerError",
                     "The model does not support Diarization API");
  }
  if (file_bytes.empty()) {
    return MakeError(400, "BadRequestError",
                     "Expected a non-empty `file` upload (16-bit PCM mono RIFF/WAVE)");
  }
  try {
    auto segs = diarizer_(
        reinterpret_cast<const uint8_t*>(file_bytes.data()), file_bytes.size());
    nlohmann::json j = nlohmann::json::array();
    for (const auto& s : segs) {
      j.push_back({{"speaker", s.speaker},
                    {"start", s.start},
                    {"end", s.end}});
    }
    DispatchResult r;
    r.content_type = "application/json";
    r.body = j.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError", e.what());
  }
}

ApiServer::DispatchResult ApiServer::handle_audio_sas(
    const std::string& file_bytes,
    const std::string& /*response_format*/) const {
  if (!sas_) {
    return MakeError(500, "InternalServerError",
                     "The model does not support Speaker-Attributed ASR API");
  }
  if (file_bytes.empty()) {
    return MakeError(400, "BadRequestError",
                     "Expected a non-empty `file` upload (16-bit PCM mono RIFF/WAVE)");
  }
  try {
    auto result = sas_(
        reinterpret_cast<const uint8_t*>(file_bytes.data()), file_bytes.size());
    nlohmann::json j;
    j["segments"] = nlohmann::json::array();
    for (const auto& u : result.utterances) {
      j["segments"].push_back({
        {"speaker", u.speaker},
        {"text", u.text},
        {"start", u.start},
        {"end", u.end},
        {"confidence", u.conf}
      });
    }
    DispatchResult r;
    r.content_type = "application/json";
    r.body = j.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError", e.what());
  }
}
#endif

ApiServer::DispatchResult ApiServer::handle_embeddings(
    const std::string& request_body) const {
  // Mirror of vLLM pooling/embed/api_router.py:28 `create_embedding` over the
  // EmbeddingCompletionRequest shape (embed/protocol.py:34: `model` + `input`
  // as ONE string or an ARRAY of strings) and the EmbeddingResponse shape
  // (embed/protocol.py:173-185). The embedding itself runs through the ONE
  // engine path (LLMEngine::embed -> registry forward -> PoolingRunner) — the
  // same code path vllm_embed drives, so HTTP and FFI cannot drift.
  if (!embedder_) {
    // The api_router `if handler is None` mirror (embed/api_router.py:22-25);
    // the socket layer never registers the route without an embedder.
    return MakeError(500, "InternalServerError",
                     "The model does not support Embeddings API");
  }
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("invalid JSON body: ") + e.what());
  }
  if (!body.is_object()) {
    return MakeError(400, "BadRequestError", "request body must be an object");
  }
  // model: honoured like every other serving handler — an unknown name is 404.
  if (body.contains("model") && body["model"].is_string() &&
      !models_.is_base_model(body["model"].get<std::string>())) {
    return MakeError(404, "NotFoundError",
                     "The model `" + body["model"].get<std::string>() +
                         "` does not exist.");
  }
  // encoding_format: float (the default) only; base64 is a NAMED residual.
  if (body.contains("encoding_format") && body["encoding_format"].is_string() &&
      body["encoding_format"].get<std::string>() != "float") {
    return MakeError(400, "BadRequestError",
                     "encoding_format '" +
                         body["encoding_format"].get<std::string>() +
                         "' is not supported (supported: float; base64 is a "
                         "named residual)");
  }
  if (body.contains("dimensions") && !body["dimensions"].is_null()) {
    // Matryoshka truncation is a NAMED residual of this fold (the pooler op
    // supports it; the request plumb does not yet).
    return MakeError(400, "BadRequestError",
                     "dimensions is not supported yet (named residual)");
  }
  // input: ONE string or an ARRAY of strings (embed/protocol.py:34
  // EmbeddingCompletionRequest via CompletionRequestMixin). Token-array
  // inputs are a NAMED residual.
  std::vector<std::string> inputs;
  if (!body.contains("input")) {
    return MakeError(400, "BadRequestError", "input is required");
  }
  if (body["input"].is_string()) {
    inputs.push_back(body["input"].get<std::string>());
  } else if (body["input"].is_array()) {
    for (const nlohmann::json& item : body["input"]) {
      if (!item.is_string()) {
        return MakeError(400, "BadRequestError",
                         "input must be a string or an array of strings "
                         "(token-array inputs are a named residual)");
      }
      inputs.push_back(item.get<std::string>());
    }
    if (inputs.empty()) {
      return MakeError(400, "BadRequestError",
                       "input must contain at least one string");
    }
  } else {
    return MakeError(400, "BadRequestError",
                     "input must be a string or an array of strings");
  }

  try {
    const EmbeddingBatch batch = embedder_(inputs);
    if (batch.embeddings.size() != inputs.size()) {
      return MakeError(500, "InternalServerError",
                       "embedder returned a mismatched batch");
    }
    nlohmann::json data = nlohmann::json::array();
    for (size_t i = 0; i < batch.embeddings.size(); ++i) {
      data.push_back(nlohmann::json{
          {"index", static_cast<int64_t>(i)},
          {"object", "embedding"},
          {"embedding", batch.embeddings[i]},
      });
    }
    // id: "embd-<counter>" (upstream f"embd-{random_uuid()}",
    // embed/protocol.py:180 — the serving_completion.h counter stand-in).
    static std::atomic<uint64_t> embd_counter{0};
    DispatchResult r;
    r.body = nlohmann::json{
        {"id", "embd-" + std::to_string(embd_counter.fetch_add(1))},
        {"object", "list"},
        {"created", static_cast<int64_t>(std::time(nullptr))},
        {"model", models_.model_name()},
        {"data", std::move(data)},
        {"usage",
         nlohmann::json{{"prompt_tokens", batch.prompt_tokens},
                        {"total_tokens", batch.prompt_tokens}}},
    }.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

// SystemOne helpers moved to systemone.h/.cpp (shared with vllm_c.cpp C ABI).

ApiServer::DispatchResult ApiServer::handle_ner(
    const std::string& request_body) const {
  if (!ner_) {
    return MakeError(500, "InternalServerError",
                    "The model does not support NER");
  }
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                    std::string("invalid JSON body: ") + e.what());
  }
  if (!body.is_object()) {
    return MakeError(400, "BadRequestError", "request body must be an object");
  }
  // model: honoured like every other serving handler.
  if (body.contains("model") && body["model"].is_string() &&
     !models_.is_base_model(body["model"].get<std::string>())) {
    return MakeError(404, "NotFoundError",
                    "The model `" + body["model"].get<std::string>() +
                        "` does not exist.");
  }
  // text: required string.
  if (!body.contains("text") || !body["text"].is_string()) {
    return MakeError(400, "BadRequestError",
                    "text is required and must be a string");
  }
  const std::string text = body["text"].get<std::string>();
  // labels: required array of strings.
  if (!body.contains("labels") || !body["labels"].is_array()) {
    return MakeError(400, "BadRequestError",
                    "labels is required and must be an array of strings");
  }
  std::vector<std::string> labels;
  for (const nlohmann::json& item : body["labels"]) {
    if (!item.is_string()) {
     return MakeError(400, "BadRequestError",
                      "labels must be an array of strings");
    }
    labels.push_back(item.get<std::string>());
  }
  if (labels.empty()) {
    return MakeError(400, "BadRequestError",
                    "labels must contain at least one string");
  }
  // threshold: optional, default 0.5.
  float threshold = 0.5F;
  if (body.contains("threshold") && body["threshold"].is_number()) {
    threshold = body["threshold"].get<float>();
  }
  // max_width: optional, default 12 (0 = use model default).
  int64_t max_width = 12;
  if (body.contains("max_width") && body["max_width"].is_number_integer()) {
    max_width = body["max_width"].get<int64_t>();
  }

  try {
    const NerResult result = ner_(text, labels, threshold, max_width);
    nlohmann::json entities = nlohmann::json::array();
    for (const NerEntity& e : result.entities) {
     entities.push_back(nlohmann::json{
         {"label", e.label},
         {"text", e.text},
         {"start", e.start},
         {"end", e.end},
         {"confidence", e.confidence},
     });
    }
    static std::atomic<uint64_t> ner_counter{0};
    DispatchResult r;
    r.body = nlohmann::json{
       {"id", "ner-" + std::to_string(ner_counter.fetch_add(1))},
       {"object", "ner"},
       {"created", static_cast<int64_t>(std::time(nullptr))},
       {"model", models_.model_name()},
       {"entities", std::move(entities)},
       {"usage",
        nlohmann::json{{"prompt_tokens", result.prompt_tokens},
                       {"total_tokens", result.prompt_tokens}}},
    }.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

ApiServer::DispatchResult ApiServer::handle_score(
    const std::string& request_body) const {
  if (!score_) {
    return MakeError(500, "InternalServerError",
                    "The model does not support scoring");
  }
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                    std::string("invalid JSON body: ") + e.what());
  }
  if (!body.is_object()) {
    return MakeError(400, "BadRequestError", "request body must be an object");
  }
  // model: honoured like every other serving handler.
  if (body.contains("model") && body["model"].is_string() &&
     !models_.is_base_model(body["model"].get<std::string>())) {
    return MakeError(404, "NotFoundError",
                    "The model `" + body["model"].get<std::string>() +
                        "` does not exist.");
  }
  // context: required string.
  if (!body.contains("context") || !body["context"].is_string()) {
    return MakeError(400, "BadRequestError",
                    "context is required and must be a string");
  }
  const std::string context = body["context"].get<std::string>();
  // options: required array of strings.
  if (!body.contains("options") || !body["options"].is_array()) {
    return MakeError(400, "BadRequestError",
                    "options is required and must be an array of strings");
  }
  std::vector<std::string> options;
  for (const auto& opt : body["options"]) {
    if (!opt.is_string()) {
      return MakeError(400, "BadRequestError",
                      "each option must be a string");
    }
    options.push_back(opt.get<std::string>());
  }
  if (options.empty()) {
    return MakeError(400, "BadRequestError",
                    "options must not be empty");
  }

  auto start = std::chrono::steady_clock::now();
  try {
    ScoreResult result = score_(context, options);
    auto end = std::chrono::steady_clock::now();
    double latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    nlohmann::json probs = nlohmann::json::array();
    for (float p : result.probabilities) {
      probs.push_back(R4(p));
    }
    DispatchResult r;
    r.body = nlohmann::json{
        {"model", models_.model_name()},
        {"probabilities", std::move(probs)},
        {"winner", result.winner},
        {"confidence", R4(result.confidence)},
        {"usage", nlohmann::json{{"input_tokens", result.prompt_tokens},
                                 {"output_tokens", 0}}},
        {"latency_ms", R2(latency_ms)},
    }.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

ApiServer::DispatchResult ApiServer::handle_systemone(
    const std::string& request_body) const {
  // MODEL-NIMBLE: the request-level seam owns parsing and validation, because
  // its request contract (openjev's) is not ParseSystemOneBody's.
  if (systemone_request_) {
    nlohmann::ordered_json body;
    try {
      body = nlohmann::ordered_json::parse(request_body);
    } catch (const std::exception& e) {
      return MakeError(400, "BadRequestError",
                       std::string("invalid JSON body: ") + e.what());
    }
    const auto start = std::chrono::steady_clock::now();
    try {
      nlohmann::ordered_json result = systemone_request_(body);
      const double latency_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
      const bool named = body.is_object() && body.contains("model") &&
                         body["model"].is_string();
      nlohmann::ordered_json out = nlohmann::ordered_json::object();
      out["model"] = named ? body["model"].get<std::string>() : models_.model_name();
      out["answers"] = std::move(result["answers"]);
      out["usage"] = std::move(result["usage"]);
      out["latency_ms"] = R2(latency_ms);
      DispatchResult r;
      r.body = out.dump();
      return r;
    } catch (const std::invalid_argument& e) {
      return MakeError(400, "BadRequestError", e.what());
    } catch (const std::exception& e) {
      return MakeError(500, "InternalServerError", e.what());
    }
  }
  if (!ner_ && !decision_) {
    return MakeError(500, "InternalServerError",
                    "The model does not support SystemOne");
  }
  nlohmann::ordered_json body;
  try {
    body = nlohmann::ordered_json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                    std::string("invalid JSON body: ") + e.what());
  }
  auto parsed = ParseSystemOneBody(body);
  if (!parsed.ok) {
    return MakeError(parsed.error_status, parsed.error_type, parsed.error_msg);
  }
  // Decision path (MODEL-LAYA): one forward per question.
  if (decision_) {
    auto start = std::chrono::steady_clock::now();
    try {
      nlohmann::json answers = nlohmann::json::object();
      int64_t total_tokens = 0;
      for (const auto& q : parsed.questions) {
        auto result = decision_(parsed.text, q.type, q.instructions,
                                  RenderDecisionOptions(q));
        total_tokens += result.prompt_tokens;
        answers[q.id] = BuildSystemOneAnswerDecision(q, result);
      }
      auto end = std::chrono::steady_clock::now();
      double latency_ms =
         std::chrono::duration<double, std::milli>(end - start).count();
      DispatchResult r;
      r.body = nlohmann::json{
         {"model", parsed.model.empty() ? models_.model_name() : parsed.model},
         {"answers", std::move(answers)},
         {"usage", nlohmann::json{{"input_tokens", total_tokens},
                                  {"output_tokens", 0}}},
         {"latency_ms", R2(latency_ms)},
      }.dump();
      return r;
    } catch (const std::exception& e) {
      return MakeError(500, "InternalServerError", e.what());
    }
  }
  // NER path (MODEL-GLINER25): one NER call, answers from shared result.
  auto start = std::chrono::steady_clock::now();
  try {
    const NerResult result = ner_(parsed.text, parsed.all_labels,
                                 parsed.threshold, parsed.max_width);
    auto end = std::chrono::steady_clock::now();
    double latency_ms =
       std::chrono::duration<double, std::milli>(end - start).count();
    nlohmann::json answers = nlohmann::json::object();
    for (const auto& q : parsed.questions) {
     answers[q.id] = BuildSystemOneAnswer(q, result);
    }
    DispatchResult r;
    r.body = nlohmann::json{
       {"model", parsed.model.empty() ? models_.model_name() : parsed.model},
       {"answers", std::move(answers)},
       {"usage", nlohmann::json{{"input_tokens", result.prompt_tokens},
                                {"output_tokens", 0}}},
       {"latency_ms", R2(latency_ms)},
    }.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

// POST /v1/systemone/permute (kev serve.py:systemone_permute). Re-runs one
// choice question under n_perm option orders. GLiNER's NER is order-invariant,
// so probabilities are stable across permutations; the API shape matches kev.
ApiServer::DispatchResult ApiServer::handle_systemone_permute(
    const std::string& request_body) const {
  if (!ner_ && !decision_) {
    return MakeError(500, "InternalServerError",
                    "The model does not support SystemOne");
  }
  nlohmann::ordered_json body;
  try {
    body = nlohmann::ordered_json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                    std::string("invalid JSON body: ") + e.what());
  }
  if (!body.is_object()) {
    return MakeError(400, "BadRequestError", "request body must be an object");
  }
  if (!body.contains("request") || !body["request"].is_object()) {
    return MakeError(400, "BadRequestError",
                    "request is required and must be an object");
  }
  auto parsed = ParseSystemOneBody(body["request"]);
  if (!parsed.ok) {
    return MakeError(parsed.error_status, parsed.error_type, parsed.error_msg);
  }
  if (!body.contains("question") || !body["question"].is_string()) {
    return MakeError(400, "BadRequestError",
                    "question is required and must be a string");
  }
  std::string question_id = body["question"].get<std::string>();
  const SystemOneQuestion* target = nullptr;
  for (const auto& q : parsed.questions) {
    if (q.id == question_id) { target = &q; break; }
  }
  if (!target) {
    return MakeError(400, "BadRequestError",
                    "question '" + question_id + "' not found");
  }
  if (target->type != "choice") {
    return MakeError(400, "BadRequestError",
                    "question must be a choice question");
  }
  int n_perm = body.value("n_perm", 6);
  int seed = body.value("seed", 0);
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  nlohmann::json runs = nlohmann::json::array();
  std::vector<float> min_prob(target->keys.size(), 1.0F);
  std::vector<float> max_prob(target->keys.size(), 0.0F);
  std::string first_choice;
  bool argmax_stable = true;
  for (int i = 0; i < n_perm; ++i) {
    // Shuffle the key order, then build the matching label order so
    // the NER call uses option_text labels, not bare key names.
    std::vector<size_t> idx(target->keys.size());
    for (size_t j = 0; j < idx.size(); ++j) idx[j] = j;
    if (i > 0) std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<std::string> order_keys, order_labels;
    for (size_t j : idx) {
     order_keys.push_back(target->keys[j]);
     order_labels.push_back(target->labels[j]);
    }
    auto start = std::chrono::steady_clock::now();
    std::vector<float> scores;
    if (decision_) {
     auto dr = decision_(parsed.text, "choice", target->instructions,
                         order_labels);
     scores = dr.scores;
    } else {
     NerResult result = ner_(parsed.text, order_labels, parsed.threshold,
                              parsed.max_width);
     for (const auto& label : order_labels) {
       float max_conf = 0.0F;
       for (const auto& e : result.entities) {
         if (e.label == label) max_conf = std::max(max_conf, e.confidence);
       }
       scores.push_back(max_conf);
     }
    }
    auto end = std::chrono::steady_clock::now();
    double latency_ms =
       std::chrono::duration<double, std::milli>(end - start).count();
    std::vector<float> probs = Softmax(scores);
    size_t argmax = 0;
    for (size_t j = 1; j < probs.size(); ++j) {
     if (probs[j] > probs[argmax]) argmax = j;
    }
    nlohmann::json prob_dist = nlohmann::json::object();
    for (size_t j = 0; j < order_keys.size(); ++j) {
     prob_dist[order_keys[j]] = R2(probs[j]);
     for (size_t k = 0; k < target->keys.size(); ++k) {
       if (order_keys[j] == target->keys[k]) {
         min_prob[k] = std::min(min_prob[k], probs[j]);
         max_prob[k] = std::max(max_prob[k], probs[j]);
         break;
       }
     }
    }
    std::string choice = order_keys[argmax];
    if (i == 0) first_choice = choice;
    else if (choice != first_choice) argmax_stable = false;
    runs.push_back(nlohmann::json{
       {"order", order_keys},
       {"probabilities", std::move(prob_dist)},
       {"choice", choice},
       {"latency_ms", R2(latency_ms)},
    });
  }
  nlohmann::json spread_json = nlohmann::json::object();
  for (size_t k = 0; k < target->keys.size(); ++k) {
    spread_json[target->keys[k]] = R2(max_prob[k] - min_prob[k]);
  }
  DispatchResult r;
  r.body = nlohmann::json{
     {"runs", std::move(runs)},
     {"argmax_stable", argmax_stable},
     {"spread", std::move(spread_json)},
  }.dump();
  return r;
}

// POST /v1/systemone/separate (kev serve.py:systemone_separate). Answers each
// question in its own NER call (N passes). Response shape matches /v1/systemone.
ApiServer::DispatchResult ApiServer::handle_systemone_separate(
    const std::string& request_body) const {
  if (!ner_ && !decision_) {
    return MakeError(500, "InternalServerError",
                    "The model does not support SystemOne");
  }
  nlohmann::ordered_json body;
  try {
    body = nlohmann::ordered_json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                    std::string("invalid JSON body: ") + e.what());
  }
  auto parsed = ParseSystemOneBody(body);
  if (!parsed.ok) {
    return MakeError(parsed.error_status, parsed.error_type, parsed.error_msg);
  }
  auto start = std::chrono::steady_clock::now();
  try {
    nlohmann::json answers = nlohmann::json::object();
    int64_t total_tokens = 0;
    for (const auto& q : parsed.questions) {
     if (decision_) {
       auto dr = decision_(parsed.text, q.type, q.instructions,
                           RenderDecisionOptions(q));
       total_tokens += dr.prompt_tokens;
       answers[q.id] = BuildSystemOneAnswerDecision(q, dr);
     } else {
       // One NER pass per question, so the per-question label set is the
       // NER label set (the instruction for a noul question), not kev's
       // "no"/"yes" option texts.
       const NerResult result = ner_(parsed.text, NerLabels(q),
                                     parsed.threshold, parsed.max_width);
       total_tokens += result.prompt_tokens;
       answers[q.id] = BuildSystemOneAnswer(q, result);
     }
    }
    auto end = std::chrono::steady_clock::now();
    double latency_ms =
       std::chrono::duration<double, std::milli>(end - start).count();
    DispatchResult r;
    r.body = nlohmann::json{
       {"model", parsed.model.empty() ? models_.model_name() : parsed.model},
       {"answers", std::move(answers)},
       {"usage", nlohmann::json{{"input_tokens", total_tokens},
                                {"output_tokens", 0}}},
       {"latency_ms", R2(latency_ms)},
    }.dump();
    return r;
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }
}

ApiServer::DispatchResult ApiServer::handle_audio_speech(
    const std::string& request_body) const {
  // OpenAI's createSpeech, extended with the two MUSIC inputs. See
  // speech_api.h for why `lyrics` and `description` are separate fields.
  if (!synthesizer_) {
    return MakeError(500, "InternalServerError", "No speech synthesizer configured.");
  }
  ::vllm::openai::SpeechRequest request;
  try {
    request = ::vllm::openai::ParseSpeechRequest(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError", e.what());
  }
  // THE REFUSAL BEFORE STAGING. `requires_reference_audio()` exists so a server
  // can answer this without loading or synthesizing anything; a family with no
  // text-only synthesis must not be handed a request it can only fail.
  if (speech_capabilities_.requires_reference_audio && request.reference_audio.empty()) {
    return MakeError(400, "BadRequestError",
                     "The loaded speech family '" + speech_capabilities_.family +
                         "' has no text-only synthesis: `reference_audio` (a data: URL "
                         "carrying a 16-bit PCM mono WAV) is required.");
  }
  ::vllm::openai::SpeechResponse response;
  try {
    response = synthesizer_(request);
  } catch (const std::exception& e) {
    // The family's refusal reaches the client verbatim: it names the field or
    // the missing stage, which a generic 500 body would throw away.
    return MakeError(500, "InternalServerError", e.what());
  }
  if (response.wav.empty()) {
    return MakeError(500, "InternalServerError", "The speech family returned no audio.");
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "audio/wav";
  out.body = std::move(response.wav);
  return out;
}

ApiServer::DispatchResult ApiServer::handle_videos(
    const std::string& request_body) {
  // vLLM-Omni's ASYNC video endpoint: validate, enqueue, and return the job id
  // immediately. Generation is minutes-long (a 50-step denoise over the packed
  // video+audio sequence), so answering inline would hold an HTTP worker for the
  // whole run -- which is exactly why upstream splits async from /sync.
  if (!video_runner_) {
    return MakeError(500, "InternalServerError", "No video runner configured.");
  }
  ::vllm::openai::VideoRequest request;
  try {
    request = ::vllm::openai::ParseVideoRequest(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError", e.what());
  }

  const std::string id =
      video_jobs_.Create(request.model, video_model_warning(request));
  std::thread worker([this, id, request]() {
    try {
      video_jobs_.MarkRunning(id);
      video_jobs_.MarkSucceeded(id, video_runner_(request));
    } catch (const std::exception& e) {
      // A runner throw is a FAILED job, not a crashed server: the thread must
      // never let an exception escape (std::terminate) and must always leave the
      // job in a terminal state, or a poller would wait on "running" forever.
      try {
        video_jobs_.MarkFailed(id, e.what());
      } catch (const std::exception&) {
      }
    } catch (...) {
      try {
        video_jobs_.MarkFailed(id, "unknown error");
      } catch (const std::exception&) {
      }
    }
  });
  {
    std::lock_guard<std::mutex> lock(video_workers_mutex_);
    video_workers_.push_back(std::move(worker));
  }

  ::vllm::openai::VideoJob job;
  video_jobs_.Get(id, &job);
  return VideoJsonOk(::vllm::openai::VideoJobStatusJson(job));
}

ApiServer::DispatchResult ApiServer::handle_videos_sync(
    const std::string& request_body) {
  // The SYNCHRONOUS twin: run to completion on the calling worker and answer with
  // the terminal job record. Same runner, same failure mapping -- only the
  // waiting differs.
  if (!video_runner_) {
    return MakeError(500, "InternalServerError", "No video runner configured.");
  }
  ::vllm::openai::VideoRequest request;
  try {
    request = ::vllm::openai::ParseVideoRequest(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError", e.what());
  }

  const std::string id =
      video_jobs_.Create(request.model, video_model_warning(request));
  try {
    video_jobs_.MarkRunning(id);
    video_jobs_.MarkSucceeded(id, video_runner_(request));
  } catch (const std::exception& e) {
    video_jobs_.MarkFailed(id, e.what());
    return MakeError(500, "InternalServerError", e.what());
  }
  ::vllm::openai::VideoJob job;
  video_jobs_.Get(id, &job);
  return VideoJsonOk(::vllm::openai::VideoJobStatusJson(job));
}

ApiServer::DispatchResult ApiServer::handle_video_status(
    const std::string& job_id) const {
  ::vllm::openai::VideoJob job;
  if (!video_jobs_.Get(job_id, &job)) {
    return MakeError(404, "NotFoundError", "Unknown video job: " + job_id);
  }
  return VideoJsonOk(::vllm::openai::VideoJobStatusJson(job));
}

ApiServer::DispatchResult ApiServer::handle_video_content(
    const std::string& job_id) const {
  // OpenAI's GET /v1/videos/{video_id}/content. Without it a caller can start and
  // poll a job but never FETCH the result over HTTP, which makes the endpoint
  // unusable to anyone without a filesystem view of the server.
  ::vllm::openai::VideoJob job;
  if (!video_jobs_.Get(job_id, &job)) {
    return MakeError(404, "NotFoundError", "Unknown video job: " + job_id);
  }
  if (job.status == ::vllm::openai::VideoJobStatus::kFailed) {
    return MakeError(500, "InternalServerError",
                     "Video job " + job_id + " failed: " + job.error);
  }
  if (job.status != ::vllm::openai::VideoJobStatus::kSucceeded) {
    // A pending job must NEVER answer with bytes: a partially muxed file would
    // reach the client as a valid-looking, truncated MP4.
    return MakeError(409, "ConflictError",
                     std::string("Video job ") + job_id + " is not finished (status: " +
                         ::vllm::openai::VideoJobStatusName(job.status) +
                         "); poll GET /v1/videos/" + job_id + " until it succeeds");
  }

  std::ifstream file(job.output_path, std::ios::binary);
  if (!file) {
    return MakeError(500, "InternalServerError",
                     "Video job " + job_id + " succeeded but its output is not "
                     "readable: " + job.output_path);
  }
  std::string bytes((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  if (!file.eof() && file.fail()) {
    return MakeError(500, "InternalServerError",
                     "Video job " + job_id + " output could not be read in full: " +
                         job.output_path);
  }

  DispatchResult out;
  out.status = 200;
  out.content_type = "video/mp4";
  out.body = std::move(bytes);
  return out;
}

ApiServer::DispatchResult ApiServer::handle_metrics() const {
  // serve/instrumentator/metrics.py:82 — the prometheus text exposition served
  // by make_asgi_app(registry). The PrometheusResponse content type is
  // "text/plain; version=0.0.4; charset=utf-8".
  DispatchResult out;
  out.status = 200;
  out.content_type = v1::metrics::kContentTypeLatest;
  out.body = (metrics_ != nullptr) ? metrics_->Expose() : std::string();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_tokenize(
    const std::string& request_body) const {
  // serve/tokenize/api_router.py:46 (tokenize) over the TokenizeRequest union
  // (serve/tokenize/protocol.py:156): TokenizeCompletionRequest{prompt} OR
  // TokenizeChatRequest{messages, ...}. vLLM discriminates on the body shape
  // (pydantic Union) and, for the chat form, renders the model chat template
  // then tokenizes (serve/tokenize/serving.py:70-124). The response is
  // {count, max_model_len, tokens, token_strs} for both forms (:119).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  if (tokenizer_ == nullptr) {
    return MakeError(500, "InternalServerError", "No tokenizer configured.");
  }

  const bool return_token_strs = body.value("return_token_strs", false);
  std::string prompt;
  // add_special_tokens default differs by form: True for the completion form
  // (protocol.py:28), False for the chat form (protocol.py:78 — the chat
  // template already emits the model's special tokens).
  bool add_special_tokens;

  if (body.contains("messages")) {
    // ── TokenizeChatRequest (serve/tokenize/protocol.py:50). Render the
    // messages through the SAME chat-template seam create_chat_completion
    // tokenizes through (chat_.prompt_fn()), then tokenize — never reinvent
    // template rendering here (serve/tokenize/serving.py:84 preprocess_chat).
    if (!body.at("messages").is_array()) {
      return MakeError(400, "BadRequestError",
                       "`messages` must be an array.");
    }
    // check_generation_prompt (protocol.py:120-128): the two are exclusive.
    const bool add_generation_prompt =
        body.value("add_generation_prompt", true);
    const bool continue_final_message =
        body.value("continue_final_message", false);
    if (add_generation_prompt && continue_final_message) {
      return MakeError(400, "BadRequestError",
                       "Cannot set both `continue_final_message` and "
                       "`add_generation_prompt` to True.");
    }
    add_special_tokens = body.value("add_special_tokens", false);

    std::vector<ChatMessage> messages;
    std::vector<ChatCompletionToolsParam> tools;
    try {
      messages = body.at("messages").get<std::vector<ChatMessage>>();
      if (auto it = body.find("tools");
          it != body.end() && it->is_array()) {
        tools = it->get<std::vector<ChatCompletionToolsParam>>();
      }
    } catch (const std::exception& e) {
      return MakeError(400, "BadRequestError",
                       std::string("Invalid request: ") + e.what());
    }

    // build_chat_params folds add_generation_prompt / continue_final_message
    // into the template kwargs (protocol.py:130-146). Our ChatPromptFn seam
    // renders through the `add_generation_prompt` gate; continue_final_message
    // (open-ended final turn) suppresses the generation header, so it maps to
    // add_generation_prompt=false here.
    const bool render_generation_prompt =
        add_generation_prompt && !continue_final_message;
    try {
      if (chat_ == nullptr) {
        return MakeError(500, "InternalServerError",
                         "tokenize: the chat form needs the chat template of a "
                         "text-generation server (transcription-only server)");
      }
      // chat_template_kwargs: the tokenize chat form carries it too
      // (serve/tokenize/protocol.py:97,138), and it must render through the
      // same kwargs create_chat_completion would use or the two disagree.
      nlohmann::ordered_json template_kwargs = nlohmann::ordered_json::object();
      if (auto it = body.find("chat_template_kwargs");
          it != body.end() && it->is_object()) {
        template_kwargs = nlohmann::ordered_json::parse(it->dump());
      }
      prompt = chat_->prompt_fn()(messages, render_generation_prompt, tools,
                                  template_kwargs);
    } catch (const std::exception& e) {
      return MakeError(400, "BadRequestError",
                       std::string("Chat template render failed: ") + e.what());
    }
  } else {
    // ── TokenizeCompletionRequest (serve/tokenize/protocol.py:24).
    if (!body.contains("prompt") || !body.at("prompt").is_string()) {
      return MakeError(400, "BadRequestError",
                       "`prompt` (string) is required.");
    }
    prompt = body.at("prompt").get<std::string>();
    add_special_tokens = body.value("add_special_tokens", true);
  }

  // SERVE-REQUEST-LENGTH-GUARD (#1541). THE surface this row exists for:
  // /tokenize needs no engine and no model, and there is no authentication
  // anywhere in src/vllm/entrypoints/, so an anonymous caller reaches the
  // tokenizer here more cheaply than anywhere else. Measured on the FINAL
  // prompt, after the chat form's template render, because that string is
  // exactly what the encode below is handed.
  if (auto refusal = refuse_oversized_prompt(prompt.size())) {
    return *refusal;
  }
  std::vector<int32_t> ids = add_special_tokens
                                 ? tokenizer_->EncodeWithSpecialTokens(prompt)
                                 : tokenizer_->Encode(prompt);

  nlohmann::json resp;
  resp["count"] = ids.size();
  resp["max_model_len"] = max_model_len_;
  resp["tokens"] = ids;
  if (return_token_strs) {
    std::vector<std::string> token_strs;
    token_strs.reserve(ids.size());
    for (int32_t id : ids) token_strs.push_back(tokenizer_->TokenText(id));
    resp["token_strs"] = token_strs;
  } else {
    resp["token_strs"] = nullptr;
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = resp.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_detokenize(
    const std::string& request_body) const {
  // serve/tokenize/api_router.py:73 (detokenize) over DetokenizeRequest
  // (serve/tokenize/protocol.py:166) → DetokenizeResponse{prompt}.
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  if (!body.contains("tokens") || !body.at("tokens").is_array()) {
    return MakeError(400, "BadRequestError",
                     "`tokens` (array of token ids) is required.");
  }
  if (tokenizer_ == nullptr) {
    return MakeError(500, "InternalServerError", "No tokenizer configured.");
  }
  std::vector<int32_t> ids;
  try {
    for (const auto& t : body.at("tokens")) {
      ids.push_back(t.get<int32_t>());
    }
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid token id: ") + e.what());
  }
  nlohmann::json resp;
  resp["prompt"] = tokenizer_->Decode(ids);
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = resp.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_reset_prefix_cache(
    bool reset_running_requests, bool reset_external) const {
  // serve/dev/cache/api_router.py:20 → {"success": bool}.
  bool success = false;
  if (reset_prefix_cache_) {
    try {
      success = reset_prefix_cache_(reset_running_requests, reset_external);
    } catch (const std::exception& e) {
      return MakeError(500, "InternalServerError", e.what());
    }
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json{{"success", success}}.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_server_info() const {
  // serve/dev/server_info/api_router.py:43 — the three-key server_info shape.
  // vllm_config is rendered as a string; vllm_env/system_env are objects. This
  // bounded server exposes version + served model rather than the full config.
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  nlohmann::json info;
  info["vllm_config"] = std::string("served_model_name=") + models_.model_name();
  info["vllm_env"] = nlohmann::json::object();
  info["system_env"] =
      nlohmann::json{{"vllm_cpp_version", version_}};
  out.body = info.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_tokenizer_info() const {
  // serve/tokenize/api_router.py:95-108 (attach_router → get_tokenizer_info,
  // registered only when app.state.args.enable_tokenizer_info_endpoint) →
  // serve/tokenize/serving.py:154-160 get_tokenizer_info →
  // TokenizerInfo(tokenizer, chat_template).to_dict() (:164-195) →
  // TokenizerInfoResponse (serve/tokenize/protocol.py:185, ConfigDict
  // extra="allow", required `tokenizer_class`).
  //
  // vLLM emits the HF `tokenizer_config.json` init_kwargs verbatim (minus the
  // vocab_file/merges_file paths) plus `tokenizer_class` and optional
  // `chat_template`. We surface EXACTLY the fields our byte-level /
  // SentencePiece BPE tokenizer can genuinely back. Fields vLLM emits that our
  // tokenizer does not carry are OMITTED (never fabricated) and NAMED in
  // specs/utility-endpoints.md: the raw `chat_template` string (our chat
  // template lives in the ChatPromptFn render seam, not the tokenizer), the HF
  // init_kwargs (clean_up_tokenization_spaces / add_bos_token /
  // model_input_names / padding-truncation defaults — not parsed), and the
  // added-token `normalized` / `single_word` flags (not stored on SpecialToken).
  if (tokenizer_ == nullptr) {
    return MakeError(500, "InternalServerError", "No tokenizer configured.");
  }
  nlohmann::json info;
  // `tokenizer_class` (REQUIRED by TokenizerInfoResponse): the genuine BPE
  // family (the HF `tokenizers` library's own class names for these two
  // byte-level / SentencePiece BPE tokenizers).
  info["tokenizer_class"] = tokenizer_->IsSentencePiece()
                                ? "SentencePieceBPETokenizer"
                                : "ByteLevelBPETokenizer";
  if (max_model_len_ > 0) info["model_max_length"] = max_model_len_;
  info["vocab_size"] = tokenizer_->VocabSize();
  if (tokenizer_->BosId() >= 0) info["bos_token_id"] = tokenizer_->BosId();
  if (tokenizer_->EosId() >= 0) info["eos_token_id"] = tokenizer_->EosId();
  // `added_tokens_decoder` mirrors the HF tokenizer_config.json map id →
  // {content, special, lstrip, rstrip}. `normalized`/`single_word` are not
  // tracked by our SpecialToken, so they are omitted (named above).
  nlohmann::json added = nlohmann::json::object();
  for (const vllm::tok::SpecialToken& t : tokenizer_->AddedTokens()) {
    added[std::to_string(t.id)] = {{"content", t.text},
                                   {"special", t.special},
                                   {"lstrip", t.lstrip},
                                   {"rstrip", t.rstrip}};
  }
  info["added_tokens_decoder"] = added;
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = info.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_abort_requests(
    const std::string& request_body) const {
  // serve/dev/rlhf/api_router.py:94-138 (abort_requests): parse the body,
  // extract `request_ids`; a non-empty list aborts exactly those (external)
  // ids via engine.abort(request_ids); an empty/missing list aborts ALL
  // in-flight requests. Response {"status":"aborted","aborted":<count>}. A
  // malformed body → 400 {"detail":"Invalid JSON format"}; an abort failure →
  // 500 {"error": "..."} (both mirror the upstream router shapes exactly).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception&) {
    DispatchResult err;
    err.status = 400;
    err.content_type = "application/json";
    err.body = nlohmann::json{{"detail", "Invalid JSON format"}}.dump();
    return err;
  }
  std::vector<std::string> request_ids;
  if (auto it = body.find("request_ids");
      it != body.end() && it->is_array()) {
    try {
      request_ids = it->get<std::vector<std::string>>();
    } catch (const std::exception& e) {
      DispatchResult err;
      err.status = 400;
      err.content_type = "application/json";
      err.body =
          nlohmann::json{{"detail",
                          std::string("Invalid request_ids: ") + e.what()}}
              .dump();
      return err;
    }
  }
  int aborted = 0;
  if (abort_requests_) {
    try {
      aborted = abort_requests_(request_ids);
    } catch (const std::exception& e) {
      DispatchResult err;
      err.status = 500;
      err.content_type = "application/json";
      err.body =
          nlohmann::json{
              {"error", std::string("Failed to abort requests: ") + e.what()}}
              .dump();
      return err;
    }
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body =
      nlohmann::json{{"status", "aborted"}, {"aborted", aborted}}.dump();
  return out;
}

void ApiServer::register_routes() {
  httplib::Server& server = impl_->server;

  // Write a DispatchResult onto an httplib::Response — either the full JSON body
  // or a chunked text/event-stream (SSE), matching upstream's JSONResponse vs
  // StreamingResponse(media_type="text/event-stream").
  auto write = [](const DispatchResult& result, httplib::Response& res) {
    res.status = result.status;
    if (result.streaming) {
      if (result.sse_stream != nullptr) {
        // W2 live StreamingResponse: one provider invocation pulls one
        // per-request collector output. A slow/disconnected client occupies
        // only its httplib worker; AsyncLLM keeps batching other requests.
        std::shared_ptr<SseStream> stream = result.sse_stream;
        res.set_chunked_content_provider(
            result.content_type,
            [stream](size_t /*offset*/, httplib::DataSink& sink) -> bool {
              try {
                std::string chunk;
                if (!stream->next(chunk)) {
                  sink.done();
                  return true;
                }
                if (!sink.write(chunk.data(), chunk.size())) {
                  stream->abort();
                  return false;
                }
                return true;
              } catch (...) {
                // MID-FLIGHT WITNESS: a provider exception here means the live
                // stream died after headers were already sent (the client sees
                // a truncated body, not a 500). Rethrow-to-inspect so the raw
                // cause reaches stderr before the abort.
                try {
                  std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                  std::cerr << "sse: stream aborted mid-flight: " << e.what()
                            << "\n";
                } catch (...) {
                  std::cerr << "sse: stream aborted mid-flight: unknown error\n";
                }
                stream->abort();
                return false;
              }
            },
            [stream](bool success) {
              if (!success) stream->abort();
            });
        return;
      }

      // Legacy synchronous compatibility/test seam: write the precomputed
      // chunks exactly as before.
      auto chunks = std::make_shared<std::vector<std::string>>(result.sse_chunks);
      res.set_chunked_content_provider(
          result.content_type,
          [chunks](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            for (const std::string& chunk : *chunks) {
              if (!sink.write(chunk.data(), chunk.size())) return false;
            }
            sink.done();
            return true;
          });
    } else {
      res.set_content(result.body, result.content_type);
    }
  };

  // TASK-CONDITIONAL (mirrors vLLM registering the generate routes only when
  // "generate" is in supported_tasks, api_server.py:255-265): a serving-less
  // (transcription-only) server has no completion/chat handlers, so the two
  // generate routes are NOT registered and answer 404.
  if (completion_ != nullptr) {
    server.Post("/v1/completions",
                [this, write](const httplib::Request& req, httplib::Response& res) {
                  write(handle_completions(req.body), res);
                });
  }
  if (chat_ != nullptr) {
    server.Post("/v1/chat/completions",
                [this, write](const httplib::Request& req, httplib::Response& res) {
                  write(handle_chat_completions(req.body), res);
                });
  }
  server.Get("/v1/models",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_models(), res);
             });
  server.Get("/health",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_health(), res);
             });
  server.Get("/version",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_version(), res);
             });

  // ── C8 additive routes: each registered only when its backing is attached,
  // so a default-constructed server is byte-identical to before. /ping and
  // /server_info are read-only liveness/introspection and always present. ─────
  server.Get("/ping",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_ping(), res);
             });
  server.Post("/ping",
              [this, write](const httplib::Request&, httplib::Response& res) {
                write(handle_ping(), res);
              });
  server.Get("/server_info",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_server_info(), res);
             });

  if (embedder_) {
    // Embeddings (ARCH-ONE-SURFACE ROW 6). Registered ONLY when an embedder is
    // attached (task-conditional, the api_server.py:255-265 supported_tasks
    // mirror), so a text server answers 404 at the route table — and an
    // embedding server, having no completion_/chat_ handlers, answers 404 on
    // the generate routes the same way.
    server.Post("/v1/embeddings",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_embeddings(req.body), res);
                });
  }

  if (ner_) {
    // NER (MODEL-GLINER25, ABI v27). Registered ONLY when a NER callback is
    // attached, so a text/embedding server answers 404 at the route table.
    server.Post("/v1/ner",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_ner(req.body), res);
                });
  }
  if (systemone_request_ && !ner_ && !decision_) {
    // MODEL-NIMBLE: /v1/systemone only. permute and separate are defined over
    // the per-question callbacks and stay unregistered (404) here.
    server.Post("/v1/systemone",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_systemone(req.body), res);
                });
  }
  if (ner_ || decision_) {
    // kev / System One-compatible endpoints. Backed by NER (GLiNER2.5) or
    // decision (Laya) -- the handlers route to the correct callback.
    server.Post("/v1/systemone",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_systemone(req.body), res);
                });
    server.Post("/v1/systemone/permute",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_systemone_permute(req.body), res);
                });
    server.Post("/v1/systemone/separate",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_systemone_separate(req.body), res);
                });
  }

  if (score_) {
    // cua-s1-forms score (MODEL-CUA-S1-FORMS). Registered ONLY when a score
    // callback is attached, so a non-cua-s1 server answers 404 at the route
    // table.
    server.Post("/v1/score",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_score(req.body), res);
                });
  }

  if (transcriber_) {
    // Parakeet ASR (ARCH-ONE-SURFACE ROW 1). Registered ONLY when a
    // transcriber is attached, so a text server answers 404 exactly as before.
    // The multipart shape mirrors vLLM's create_transcriptions
    // (speech_to_text/transcription/api_router.py:31): the audio arrives as
    // the `file` upload, `response_format` as an ordinary form field.
    server.Post("/v1/audio/transcriptions",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  if (!req.form.has_file("file")) {
                    write(MakeError(400, "BadRequestError",
                                    "multipart/form-data with a `file` upload "
                                    "is required"),
                          res);
                    return;
                  }
                  write(handle_audio_transcriptions(
                            req.form.get_file("file").content,
                            req.form.get_field("response_format")),
                        res);
                });
  }

#ifdef VLLM_WITH_DIARIZATION
  if (diarizer_) {
    server.Post("/v1/audio/diarizations",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  if (!req.form.has_file("file")) {
                    write(MakeError(400, "BadRequestError",
                                    "multipart/form-data with a `file` upload "
                                    "is required"),
                          res);
                    return;
                  }
                  write(HandleAudioDiarizations(*this,
                            req.form.get_file("file").content,
                            req.form.get_field("response_format")),
                        res);
                });
  }

  if (sas_) {
    server.Post("/v1/audio/sas",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  if (!req.form.has_file("file")) {
                    write(MakeError(400, "BadRequestError",
                                    "multipart/form-data with a `file` upload "
                                    "is required"),
                          res);
                    return;
                  }
                  write(HandleAudioSas(*this,
                            req.form.get_file("file").content,
                            req.form.get_field("response_format")),
                        res);
                });
  }
#endif

  if (synthesizer_) {
    // Speech + music (W6 of #672). Registered ONLY when a synthesizer is
    // attached, so a text server answers 404 exactly as before. The body is
    // JSON and the RESPONSE is audio/wav bytes, which is OpenAI's own shape for
    // createSpeech.
    server.Post("/v1/audio/speech",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_audio_speech(req.body), res);
                });
  }

  if (video_runner_) {
    // MiniMax-H3. Registered ONLY when a runner is attached, so a server built
    // without video support answers 404 exactly as before.
    server.Post("/v1/videos",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_videos(req.body), res);
                });
    server.Post("/v1/videos/sync",
                [this, write](const httplib::Request& req,
                              httplib::Response& res) {
                  write(handle_videos_sync(req.body), res);
                });
    // Registered BEFORE the bare-id pattern so the intent is readable in one
    // place; the two cannot collide in any case, since `[^/]+` stops at the '/'
    // and httplib full-matches the path.
    server.Get(R"(/v1/videos/([^/]+)/content)",
               [this, write](const httplib::Request& req,
                             httplib::Response& res) {
                 write(handle_video_content(req.matches[1]), res);
               });
    server.Get(R"(/v1/videos/([^/]+))",
               [this, write](const httplib::Request& req,
                             httplib::Response& res) {
                 write(handle_video_status(req.matches[1]), res);
               });
  }
  if (metrics_ != nullptr) {
    server.Get("/metrics",
               [this, write](const httplib::Request&, httplib::Response& res) {
                 write(handle_metrics(), res);
               });
  }
  if (tokenizer_ != nullptr) {
    server.Post(
        "/tokenize",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          write(handle_tokenize(req.body), res);
        });
    server.Post(
        "/detokenize",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          write(handle_detokenize(req.body), res);
        });
  }
  if (reset_prefix_cache_) {
    server.Post(
        "/reset_prefix_cache",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          // Query params default false (cache/api_router.py:22-26).
          const bool reset_running =
              req.has_param("reset_running_requests") &&
              req.get_param_value("reset_running_requests") == "true";
          const bool reset_external =
              req.has_param("reset_external") &&
              req.get_param_value("reset_external") == "true";
          write(handle_reset_prefix_cache(reset_running, reset_external), res);
        });
  }
  // GET /tokenizer_info registered only when a tokenizer is attached AND the
  // info endpoint is enabled — mirrors vLLM gating it behind
  // enable_tokenizer_info_endpoint (serve/tokenize/api_router.py:95).
  if (tokenizer_ != nullptr && tokenizer_info_enabled_) {
    server.Get(
        "/tokenizer_info",
        [this, write](const httplib::Request&, httplib::Response& res) {
          write(handle_tokenizer_info(), res);
        });
  }
  // POST /abort_requests registered only when the abort callback is attached
  // (serve/dev/rlhf/api_router.py:94).
  if (abort_requests_) {
    server.Post(
        "/abort_requests",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          write(handle_abort_requests(req.body), res);
        });
  }
}

bool ApiServer::listen(const std::string& host, int port) {
  register_routes();
  return impl_->server.listen(host, port);
}

int ApiServer::bind_to_any_port(const std::string& host) {
  register_routes();
  return impl_->server.bind_to_any_port(host);
}

bool ApiServer::serve() { return impl_->server.listen_after_bind(); }

void ApiServer::stop() { impl_->server.stop(); }

bool ApiServer::is_running() const { return impl_->server.is_running(); }

size_t ApiServer::http_worker_count() const {
  return impl_->http_worker_count;
}

void ConfigureUtilityEndpoints(ApiServer& server,
                               const vllm::tok::Tokenizer& tokenizer,
                               int64_t max_model_len, vllm::v1::AsyncLLM& engine,
                               const UtilityEndpointOptions& options) {
  // /tokenize + /detokenize — ON by default whenever a tokenizer exists (vLLM
  // always calls attach_tokenize_router, serve/tokenize/api_router.py:36,62).
  server.set_tokenizer(&tokenizer, max_model_len);

  // /tokenizer_info — OFF unless --enable-tokenizer-info-endpoint, mirroring
  // vLLM's enable_tokenizer_info_endpoint gate (serve/tokenize/api_router.py:95;
  // cli_args.py:140 default False). When off, the route is never registered → 404.
  if (options.enable_tokenizer_info_endpoint) {
    server.set_tokenizer_info_enabled(true);
  }

  // /abort_requests — DEV-mode gated. vLLM only registers the dev/rlhf router
  // under `if envs.VLLM_SERVER_DEV_MODE` (api_server.py:238; envs.py:157 default
  // 0), so it stays 404 in a default production server. Under
  // --enable-server-dev-mode we wire the abort callback to the live engine:
  // explicit request_ids are aborted via AsyncLLM::abort (async_llm.h:115), and
  // the returned "aborted" count is the drop in unfinished requests (abort()
  // synchronously removes the request states under the output-processor lock, so
  // the delta is exact). The empty-request_ids "abort ALL" contract is a NAMED
  // RESIDUAL: AsyncLLM exposes no active-request-id enumeration, so empty ids
  // abort nothing and report 0 rather than fabricate an all-abort we cannot reach.
  if (options.enable_server_dev_mode) {
    server.set_abort_requests(
        [&engine](const std::vector<std::string>& request_ids) -> int {
          if (request_ids.empty()) {
            return 0;  // abort-ALL residual: no active-request-id accessor.
          }
          const int before = engine.get_num_unfinished_requests();
          engine.abort(request_ids);
          const int after = engine.get_num_unfinished_requests();
          return before > after ? before - after : 0;
        });
  }

  // /metrics is not wired HERE because it is not a utility endpoint: the caller
  // owns the PrometheusStatLogger's lifetime and attaches it to both frontends
  // plus set_metrics_logger (server_main.cpp). Since #277 that logger IS live
  // on the AsyncLLM serving path, so /metrics reports real counts.
  //
  // /reset_prefix_cache stays deliberately unwired: reset_prefix_cache() lives
  // only on the scheduler's KVCacheManager, mutated exclusively on the
  // EngineCore engine thread, with no thread-safe RPC to reach it (see the
  // header block comment + specs/{utility,admin}-endpoints.md). It stays 404.
}

}  // namespace vllm::entrypoints::openai

namespace vllm {

// The SERVER half of the one-definition-rule instrument declared in
// `include/vllm/http_transport_abi.h`. It is defined HERE, in the translation
// unit that owns the listener, so the reading is that unit's own.
HttpTransportAbi ApiServerHttpTransportAbi() {
  HttpTransportAbi abi;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  abi.tls = true;
#else
  abi.tls = false;
#endif
  abi.result_size = sizeof(httplib::Result);
  abi.client_connection_size = sizeof(httplib::ClientConnection);
  return abi;
}

// Read from the TYPE of the listener this file constructs, not from a constant
// beside it, so replacing that member with `httplib::SSLServer` flips this to
// true and the case that asserts it false turns red.
bool ApiServerListenerIsTls() {
  return !std::is_same_v<entrypoints::openai::ApiServerListener,
                         httplib::Server>;
}

}  // namespace vllm
