// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 (the route
// shapes) + the per-endpoint api_router modules
//   - completion/api_router.py:34    POST /v1/completions
//   - chat_completion/api_router.py:40 POST /v1/chat/completions
//   - models/api_router.py:20        GET  /v1/models
//   - serve/instrumentator/health.py:22 GET /health
//   - serve/instrumentator/basic.py:53  GET /version
//
// SCOPE (M3.1 Task 4 / T0): the thin HTTP transport over the Task-2 serving
// handlers. The OpenAI-protocol logic (parse → SamplingParams → engine →
// response/SSE) already lives in serving_{completion,chat}; this file only
// binds it to routes over the vendored cpp-httplib (third_party/httplib.h).
//
// DEPENDENCY DEVIATION (recorded): cpp-httplib is a header-only MIT HTTP
// TRANSPORT library (the same choice as llama.cpp's server) — NOT a compute/ML
// dependency, so it is consistent with the no-pytorch / no-ggml rule. It is
// gated behind the CMake option VLLM_CPP_SERVER.
//
// The route DISPATCH (parse body → handler → status + body / SSE chunks) is
// decoupled from the socket via the handle_* methods below, so the request
// handling is unit-testable WITHOUT binding a port; listen() wires the same
// methods onto an httplib::Server.
#ifndef VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
#define VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/entrypoints/openai/systemone.h"
#include "vllm/entrypoints/openai/video_api.h"
#include "vllm/entrypoints/openai/speech_api.h"
#include "vllm/multimodal/parakeet_transcription.h"
#ifdef VLLM_WITH_DIARIZATION
#include "vllm/multimodal/diarization.h"
#endif

namespace vllm::tok {
class Tokenizer;
}  // namespace vllm::tok

namespace vllm::v1::metrics {
class PrometheusStatLogger;
}  // namespace vllm::v1::metrics

namespace vllm::v1 {
class AsyncLLM;
}  // namespace vllm::v1

namespace vllm::entrypoints::openai {

// The HTTP api_server. Holds non-owning references to the serving handlers +
// model registry (constructed + owned by the caller; they outlive it).
class ApiServer {
 public:
  enum class HttpWorkerPoolMode {
    kCapacityFixed,
    // Diagnostic same-binary fallback for A/B attribution only.
    kLegacyDynamic,
  };

  // A live SSE response occupies one cpp-httplib worker while it waits on its
  // collector. Keep enough fixed workers for every scheduler-visible stream,
  // plus a small bounded reserve for health/discovery/control requests.
  static constexpr size_t kDefaultMaxConcurrentStreams = 8;
  static constexpr size_t kControlWorkerHeadroom = 4;

  ApiServer(OpenAIServingCompletion& completion, OpenAIServingChat& chat,
            OpenAIServingModels& models, std::string version,
            size_t max_concurrent_streams = kDefaultMaxConcurrentStreams,
            HttpWorkerPoolMode worker_pool_mode =
                HttpWorkerPoolMode::kCapacityFixed);
  // TASK-CONDITIONAL construction without the text-generation serving stack
  // (ARCH-ONE-SURFACE ROW 1): a server for a SupportsTranscription-ONLY model
  // (Parakeet) has no AsyncLLM to build OpenAIServingCompletion/Chat around, so
  // /v1/completions and /v1/chat/completions are simply NOT REGISTERED (404),
  // mirroring vLLM registering the generate routes only when "generate" is in
  // supported_tasks (api_server.py:255-265). Attach the transcription seam via
  // set_transcriber; /v1/models, /health, /version and the other utility
  // routes behave as before.
  ApiServer(OpenAIServingModels& models, std::string version,
            size_t max_concurrent_streams = kDefaultMaxConcurrentStreams,
            HttpWorkerPoolMode worker_pool_mode =
                HttpWorkerPoolMode::kCapacityFixed);
  ~ApiServer();

  ApiServer(const ApiServer&) = delete;
  ApiServer& operator=(const ApiServer&) = delete;

  // The socket-free result of dispatching one request (shared by the HTTP layer
  // and unit tests). Streaming uses either the legacy precomputed vector or
  // W2's live pull source.
  struct DispatchResult {
    int status = 200;
    bool streaming = false;
    std::string content_type = "application/json";
    std::string body;                     // populated when !streaming
    std::vector<std::string> sse_chunks;  // populated when streaming
    std::shared_ptr<SseStream> sse_stream;  // live AsyncLLM streaming path
  };

  // Route dispatch (parse the JSON body → the Task-1 request → the Task-2
  // handler → a DispatchResult). On a malformed body → 400; an unknown model →
  // 404; an internal failure → 500 — each carrying the OpenAI ErrorResponse
  // JSON. These are what listen()'s handlers call, and what the tests exercise
  // without a socket.
  DispatchResult handle_completions(const std::string& request_body);
  DispatchResult handle_chat_completions(const std::string& request_body);
  DispatchResult handle_models() const;
  DispatchResult handle_health() const;
  DispatchResult handle_version() const;

  // ── C8 utility + observability surface (SERVE-METRICS /
  // SERVE-UTILITY-ENDPOINTS). Each is ADDITIVE and OPT-IN: a route is only
  // registered when its backing dependency is configured below, so a server
  // constructed without them behaves byte-identically to before. ──────────────

  // GET /metrics — Prometheus text-0.0.4 exposition
  // (serve/instrumentator/metrics.py). Registered only when a metrics logger is
  // attached. Returns 404 (route absent) otherwise; the handler itself returns
  // the exposition with the prometheus content type.
  // POST /v1/videos       -> enqueue, return {id, status} immediately
  // POST /v1/videos/sync  -> run to completion, return the .mp4 path
  // GET  /v1/videos/{id}  -> job status
  // GET  /v1/videos/{id}/content -> the finished MP4 bytes (video/mp4)
  // POST /v1/audio/transcriptions (ARCH-ONE-SURFACE ROW 1). Mirror of vLLM's
  // speech_to_text/transcription/api_router.py:31 `create_transcriptions` +
  // serving.py:29 OpenAIServingTranscription: multipart `file` upload (16-bit
  // PCM mono WAV here — the extractor refuses anything else loudly), optional
  // `response_format` = "json" (default, -> TranscriptionResponse
  // {"text": ...}) or "text" (-> the raw transcript, text/plain).
  // verbose_json / srt / vtt are NAMED RESIDUALS -> 400. Registered ONLY when
  // a transcriber is attached (the video_runner precedent); the route lambda
  // extracts the upload and calls this with the raw file bytes.
  DispatchResult handle_audio_transcriptions(
      const std::string& file_bytes, const std::string& response_format) const;
#ifdef VLLM_WITH_DIARIZATION
  DispatchResult handle_audio_diarizations(
      const std::string& file_bytes, const std::string& response_format) const;
  DispatchResult handle_audio_sas(
      const std::string& file_bytes, const std::string& response_format) const;
#endif

  // POST /v1/embeddings (ARCH-ONE-SURFACE ROW 6). Mirror of vLLM's
  // pooling/embed/api_router.py:28 `create_embedding` over the
  // EmbeddingCompletionRequest shape (embed/protocol.py:34: `model`, `input`
  // as ONE string or an ARRAY of strings) and the EmbeddingResponse shape
  // (embed/protocol.py:173-185: id "embd-...", object "list", data rows
  // {index, object:"embedding", embedding:[...]}, usage prompt/total tokens).
  // Token-array inputs, `dimensions` (matryoshka) and `encoding_format:
  // "base64"` are NAMED RESIDUALS -> 400. Registered ONLY when an embedder is
  // attached (the transcriber precedent), so a text server answers 404 at the
  // route table.
  DispatchResult handle_embeddings(const std::string& request_body) const;

  // POST /v1/ner (MODEL-GLINER25, ABI v27). GLiNER-native NER endpoint:
  // request carries `text` + `labels` (entity types) + optional `threshold`
  // (default 0.5) and `max_width` (default 12). Response returns extracted
  // entities as spans with text, label, start/end character offsets, and
  // confidence. Registered ONLY when a NER callback is attached (the embedder
  // precedent), so a text server answers 404 at the route table.
  DispatchResult handle_ner(const std::string& request_body) const;

  // POST /v1/systemone (MODEL-GLINER25). Jev / System One-compatible endpoint
  // mirroring the kev project's serve.py. `state` is the text (or any JSON) to
  // extract from; `questions` is a map of id → Question, each with type "noul",
  // "choice", or "score". The SAME NER callback backs this and /v1/ner. One NER
  // pass serves all questions (the separate endpoint does N passes).
  DispatchResult handle_systemone(const std::string& request_body) const;

  // POST /v1/systemone/permute (kev serve.py:systemone_permute). Re-runs one
  // choice question under n_perm option orders and reports per-order
  // probabilities, argmax stability, and spread.
  DispatchResult handle_systemone_permute(
      const std::string& request_body) const;

  // POST /v1/systemone/separate (kev serve.py:systemone_separate). Answers each
  // question in its own NER call (N passes). Response shape matches
  // /v1/systemone.
  DispatchResult handle_systemone_separate(
      const std::string& request_body) const;

  // POST /v1/score (MODEL-CUA-S1-FORMS). Takes a context string + a list of
  // option strings, returns per-option probabilities + winner. Registered ONLY
  // when a score callback is attached, so a non-cua-s1 server answers 404.
  DispatchResult handle_score(const std::string& request_body) const;

  // POST /v1/audio/speech (W6 of #672). OpenAI's createSpeech spelling, with
  // the two MUSIC inputs (`lyrics`, `description`) as ADDITIONAL named fields
  // — see speech_api.h for why they are not one `input` behind a separator.
  // Returns the RIFF/WAVE bytes with content type audio/wav; `voice`, `speed`,
  // streaming and a non-wav `response_format` are NAMED RESIDUALS -> 400.
  // Registered ONLY when a synthesizer is attached (the transcriber precedent),
  // so a text server answers 404 at the route table exactly as before.
  //
  // The reference-clip refusal happens HERE, before the runner is called: the
  // attached family's `requires_reference_audio` is known without synthesizing,
  // which is the whole reason SpeechEngine exposes it.
  DispatchResult handle_audio_speech(const std::string& request_body) const;

  DispatchResult handle_videos(const std::string& request_body);
  DispatchResult handle_videos_sync(const std::string& request_body);
  DispatchResult handle_video_status(const std::string& job_id) const;
  // OpenAI's download endpoint. Unknown id -> 404; a job that has not finished ->
  // 409 naming its status (NEVER a truncated file); a failed job -> 500 carrying
  // the failure. Only a succeeded job yields bytes, and only the whole file.
  DispatchResult handle_video_content(const std::string& job_id) const;

  DispatchResult handle_metrics() const;
  // POST /tokenize, POST /detokenize (serve/tokenize/api_router.py). Registered
  // only when a tokenizer is attached. /tokenize accepts BOTH forms of the
  // TokenizeRequest union (serve/tokenize/protocol.py:156): the raw
  // TokenizeCompletionRequest{prompt} and the TokenizeChatRequest{messages, ...}
  // — the chat form renders through chat_.prompt_fn() (the same model chat
  // template create_chat_completion uses) then tokenizes. Schemas match vLLM.
  DispatchResult handle_tokenize(const std::string& request_body) const;
  DispatchResult handle_detokenize(const std::string& request_body) const;
  // POST /reset_prefix_cache (serve/dev/cache/api_router.py) → {"success": b}.
  DispatchResult handle_reset_prefix_cache(bool reset_running_requests,
                                           bool reset_external) const;
  // GET /ping (serve/sagemaker/api_router.py) — liveness, mirrors /health.
  DispatchResult handle_ping() const;
  // GET /server_info (serve/dev/server_info/api_router.py).
  DispatchResult handle_server_info() const;
  // GET /tokenizer_info (serve/tokenize/api_router.py:95-108 attach_router +
  // serving.py:154-160 get_tokenizer_info → TokenizerInfoResponse). Gated in
  // vLLM behind `enable_tokenizer_info_endpoint`; we mirror that with the
  // tokenizer-info flag below (default off → the route is not registered → 404).
  // Surfaces the tokenizer_config.json-equivalent fields our BPE tokenizer can
  // genuinely back (tokenizer_class, model_max_length, vocab_size,
  // bos/eos_token_id, added_tokens_decoder); fields vLLM emits that our
  // tokenizer does not carry are OMITTED + named in the spec.
  DispatchResult handle_tokenizer_info() const;
  // POST /abort_requests (serve/dev/rlhf/api_router.py:94-138) → {"status":
  // "aborted","aborted":<count>}. Parses {request_ids:[...]}; empty/missing ids
  // means "abort all in-flight". Wired to the injected abort callback below.
  DispatchResult handle_abort_requests(const std::string& request_body) const;

  // Attach the Prometheus stat logger backing GET /metrics (non-owning; must
  // outlive the server). Enables the /metrics route.
  void set_metrics_logger(const v1::metrics::PrometheusStatLogger* logger) {
    metrics_ = logger;
  }
  // Attach the video generation+mux runner backing POST /v1/videos and
  // POST /v1/videos/sync (MiniMax-H3). ADDITIVE and OPT-IN like the routes
  // above: absent runner => routes unregistered => 404, byte-identical to a
  // server built without video support.
  //
  // The runner performs generation AND muxing and returns the .mp4 path, or
  // throws to fail the job. The LIBRARY NEVER SPAWNS A PROCESS -- the ffmpeg
  // invocation lives in examples/ (developer-ratified 2026-08-03), which is
  // precisely why this enters as a callback rather than a built-in.
  void set_video_runner(::vllm::openai::VideoRunner runner) {
    video_runner_ = std::move(runner);
  }

  // Attach the transcription seam backing POST /v1/audio/transcriptions.
  // ADDITIVE and OPT-IN like the video runner above: absent => route
  // unregistered => 404, byte-identical to a server without ASR. The callback
  // wraps the ONE library entry
  // (vllm::multimodal::ParakeetTranscriber::TranscribeWavBytes) — the SAME
  // seam vllm_transcribe drives — so HTTP and FFI cannot drift.
  using TranscribeFn =
      std::function<vllm::multimodal::ParakeetTranscription(
          const uint8_t* wav_bytes, size_t num_bytes)>;
  void set_transcriber(TranscribeFn transcriber) {
    transcriber_ = std::move(transcriber);
  }

#ifdef VLLM_WITH_DIARIZATION
  // Attach the diarization seam backing POST /v1/audio/diarizations (ABI v30).
  // ADDITIVE and OPT-IN: absent => route unregistered => 404, byte-identical to
  // a server without diarization. The callback wraps the parakeet.cpp C-API.
  using DiarizeFn =
      std::function<std::vector<vllm::multimodal::SpeakerSegment>(
          const uint8_t* wav_bytes, size_t num_bytes)>;
  void set_diarizer(DiarizeFn diarizer) {
    diarizer_ = std::move(diarizer);
  }

  // Attach the SAS seam backing POST /v1/audio/sas (speaker-attributed ASR).
  // Runs both ASR and diarization on the same audio and merges the results.
  using SasFn =
      std::function<vllm::multimodal::SpeakerAttributedASR(
          const uint8_t* wav_bytes, size_t num_bytes)>;
  void set_sas(SasFn sas) {
    sas_ = std::move(sas);
  }
#endif

  // Attach the embedding seam backing POST /v1/embeddings (ARCH-ONE-SURFACE
  // ROW 6). ADDITIVE and OPT-IN like the transcriber above: absent => route
  // unregistered => 404, byte-identical to a server without pooling. The
  // callback wraps the ONE engine path (LoadedEngine -> LLMEngine::embed ->
  // the registry forward + PoolingRunner step) — the SAME path vllm_embed
  // drives — so HTTP and FFI cannot drift. Returns one embedding per input
  // (input order) + the total prompt token count for the usage block; throws
  // to fail the request (-> 500).
  struct EmbeddingBatch {
    std::vector<std::vector<float>> embeddings;
    int64_t prompt_tokens = 0;
  };
  using EmbedFn =
      std::function<EmbeddingBatch(const std::vector<std::string>& inputs)>;
  void set_embedder(EmbedFn embedder) { embedder_ = std::move(embedder); }

  // Attach the NER seam backing POST /v1/ner and POST /v1/systemone
  // (MODEL-GLINER25, ABI v27). ADDITIVE and OPT-IN like the embedder above:
  // absent => routes unregistered => 404, byte-identical to a server without
  // a GLiNER2.5 model. The callback wraps the ONE library seam
  // (Gliner2NerInference) — the SAME path vllm_gliner_ner drives — so HTTP
  // and FFI cannot drift. Returns extracted entities + prompt token count;
  // throws to fail the request (-> 500).
  // Result types are defined in systemone.h and aliased here so the
  // callback typedefs and server_main.cpp keep their existing names.
  using NerEntity = systemone::NerEntity;
  using NerResult = systemone::NerResult;
  using NerFn = std::function<NerResult(
      const std::string& text, const std::vector<std::string>& labels,
      float threshold, int64_t max_width)>;
  void set_ner(NerFn ner) { ner_ = std::move(ner); }

  // Attach the score seam backing POST /v1/score for a cua-s1-forms model
  // (MODEL-CUA-S1-FORMS, Phase 4). ADDITIVE and OPT-IN like ner_ above:
  // absent => the /v1/score route is not registered (404). When set, the
  // handler runs one forward (context + options → probabilities) and returns
  // the per-option distribution + winner. The callback wraps the ONE library
  // seam (CuaS1ScoreInference) so HTTP and FFI cannot drift.
  using ScoreResult = systemone::ScoreResult;
  using ScoreFn = std::function<ScoreResult(
      const std::string& context,
      const std::vector<std::string>& options)>;
  void set_score(ScoreFn score) { score_ = std::move(score); }
  // Attach the decision seam backing POST /v1/systemone for a Laya model
  // (MODEL-LAYA, Phase 4). ADDITIVE and OPT-IN like ner_ above: absent => the
  // SystemOne handler routes to ner_ (GLiNER2.5) or 404. When set, the
  // SystemOne handler runs one decision forward per question and returns
  // per-option scores + escalate logits instead of NER entities. The callback
  // wraps the ONE library seam (LayaInference) so HTTP and FFI cannot drift.
  using DecisionResult = systemone::DecisionResult;
  using DecisionFn = std::function<DecisionResult(
      const std::string& state, const std::string& qtype,
      const std::string& instructions,
      const std::vector<std::string>& options)>;
  void set_decision(DecisionFn decision) { decision_ = std::move(decision); }

  // MODEL-NIMBLE: a REQUEST-level SystemOne seam, for a model whose per-field
  // prompt embeds the whole schema and so cannot run one question at a time.
  // ADDITIVE and OPT-IN: absent => handle_systemone is unchanged. The callback
  // receives the parsed body and returns {"answers": {...}, "usage": {...}};
  // a std::invalid_argument it throws is the request's fault (HTTP 400). It
  // wraps the ONE library seam (NimbleDecide) that vllm_decide also calls.
  using SystemOneRequestFn =
      std::function<nlohmann::ordered_json(const nlohmann::ordered_json& body)>;
  void set_systemone_request(SystemOneRequestFn fn) {
    systemone_request_ = std::move(fn);
  }

  // Attach the speech/music synthesis seam backing POST /v1/audio/speech (W6 of
  // #672). ADDITIVE and OPT-IN like the embedder above: absent => route
  // unregistered => 404, byte-identical to a server without a speech model. The
  // callback wraps the ONE library seam (multimodal::SpeechEngine::Synthesize)
  // — the SAME seam vllm_synthesize drives — so HTTP and FFI cannot drift.
  //
  // `capabilities` is what the server knows WITHOUT synthesizing, so a request
  // can be refused before the weights stage. It is taken with the runner rather
  // than separately, because a runner without them would make the route guess.
  using SynthesizeFn =
      std::function<::vllm::openai::SpeechResponse(const ::vllm::openai::SpeechRequest&)>;
  void set_synthesizer(SynthesizeFn synthesizer,
                       ::vllm::openai::SpeechCapabilities capabilities) {
    synthesizer_ = std::move(synthesizer);
    speech_capabilities_ = std::move(capabilities);
  }

  // Attach the tokenizer + max_model_len backing /tokenize and /detokenize
  // (non-owning; must outlive the server). ALSO derives the request-boundary
  // prompt-length bound below, because that bound is a property of exactly this
  // pair and of nothing else. Defined out of line: it reads the tokenizer's
  // vocabulary, and this header only forward-declares the type.
  void set_tokenizer(const vllm::tok::Tokenizer* tokenizer,
                     int64_t max_model_len);

  // SERVE-REQUEST-LENGTH-GUARD (#1541). The largest prompt, in BYTES, this
  // server will hand to the tokenizer; 0 means unbounded.
  //
  // DERIVED, never configured: `max_model_len * tokenizer.MaxTokenBytes()`. The
  // token texts of an encode concatenate back to the input, so a prompt of B
  // bytes costs at least B / MaxTokenBytes() tokens; anything above this bound
  // therefore exceeds max_model_len tokens and is already unservable. The bound
  // thus refuses nothing InputProcessor::ValidatePromptLen would have accepted
  // -- it only moves the refusal AHEAD of the encode that pays for it. Zero
  // when no tokenizer is attached, or when max_model_len <= 0, which is the
  // same "no context length is known" state ValidatePromptLen early-outs on.
  size_t max_prompt_bytes() const { return max_prompt_bytes_; }
  // Enable GET /tokenizer_info (mirrors vLLM's `enable_tokenizer_info_endpoint`
  // CLI flag: off by default, so the route is absent → 404 unless enabled AND a
  // tokenizer is attached).
  void set_tokenizer_info_enabled(bool enabled) {
    tokenizer_info_enabled_ = enabled;
  }
  // Attach the abort callback backing POST /abort_requests. Mirrors
  // EngineClient.abort(request_ids); an empty id list means "abort all
  // in-flight". Returns the number of requests aborted (the response's
  // "aborted" count).
  void set_abort_requests(
      std::function<int(const std::vector<std::string>&)> abort_requests) {
    abort_requests_ = std::move(abort_requests);
  }
  // Attach the prefix-cache reset callback backing POST /reset_prefix_cache.
  // Signature mirrors EngineClient.reset_prefix_cache(reset_running_requests,
  // reset_external) → success.
  void set_reset_prefix_cache(
      std::function<bool(bool, bool)> reset_prefix_cache) {
    reset_prefix_cache_ = std::move(reset_prefix_cache);
  }

  // Bind host:port, register the routes and serve until stop() (blocking).
  // Returns false if the bind fails.
  bool listen(const std::string& host, int port);

  // Bind to an OS-assigned ephemeral port (port 0) on `host`; returns the bound
  // port (or -1 on failure). Follow with serve() to run the loop. Used by the
  // smoke test to avoid a fixed-port race.
  int bind_to_any_port(const std::string& host);
  // Run the accept loop after a successful bind_to_any_port (blocking).
  bool serve();
  // Signal the accept loop to stop (thread-safe); listen()/serve() then return.
  void stop();
  // True once the server is accepting connections (poll before issuing
  // requests against a bind_to_any_port + serve() server on another thread).
  bool is_running() const;
  // Configured fixed worker count. Exposed for startup diagnostics and the
  // transport-capacity regression; it is max_concurrent_streams + headroom,
  // or zero when the diagnostic legacy-dynamic mode is selected.
  size_t http_worker_count() const;

  // Expose the diarizer and SAS callbacks for the route handlers.
#ifdef VLLM_WITH_DIARIZATION
  DiarizeFn diarizer_callback() const { return diarizer_; }
  SasFn sas_callback() const { return sas_; }
#endif

 private:
  // Null in the serving-less (transcription-only) construction: the generate
  // routes are then not registered, and direct handler dispatch reports the
  // NotImplementedError mirror (see handle_completions).
  OpenAIServingCompletion* completion_ = nullptr;
  OpenAIServingChat* chat_ = nullptr;
  OpenAIServingModels& models_;
  std::string version_;

  // Opt-in C8 backings (all nullptr/empty by default → routes not registered).
  const v1::metrics::PrometheusStatLogger* metrics_ = nullptr;
  ::vllm::openai::VideoRunner video_runner_;
  TranscribeFn transcriber_;
#ifdef VLLM_WITH_DIARIZATION
  DiarizeFn diarizer_;
  SasFn sas_;
#endif
  EmbedFn embedder_;
  NerFn ner_;
  ScoreFn score_;
  DecisionFn decision_;
  SystemOneRequestFn systemone_request_;
  SynthesizeFn synthesizer_;
  ::vllm::openai::SpeechCapabilities speech_capabilities_;
  mutable ::vllm::openai::VideoJobStore video_jobs_;
  // Background workers for the ASYNC endpoint. Joined in ~ApiServer, which is
  // why they are joinable threads and not detached: a detached worker would
  // outlive `this` and write into a destroyed job store.
  std::vector<std::thread> video_workers_;
  std::mutex video_workers_mutex_;
  const vllm::tok::Tokenizer* tokenizer_ = nullptr;
  int64_t max_model_len_ = 0;
  size_t max_prompt_bytes_ = 0;  // see max_prompt_bytes(); 0 == unbounded
  size_t max_token_bytes_ = 0;   // the derivation's second factor, for the message

  // SERVE-REQUEST-LENGTH-GUARD (#1541). nullopt when `prompt_bytes` is within
  // max_prompt_bytes() (or the bound is unset); otherwise the 400
  // BadRequestError NAMING the limit. It REFUSES. It never truncates, because a
  // silently shortened prompt returns model output for text the caller did not
  // send.
  std::optional<DispatchResult> refuse_oversized_prompt(
      size_t prompt_bytes) const;
  bool tokenizer_info_enabled_ = false;
  std::function<bool(bool, bool)> reset_prefix_cache_;
  std::function<int(const std::vector<std::string>&)> abort_requests_;

  // The non-fatal note a `model` naming something we do not serve earns (empty
  // when absent or matching). See the definition for why it is not a rejection.
  std::string video_model_warning(
      const ::vllm::openai::VideoRequest& request) const;

  // Opaque httplib::Server (pimpl keeps third_party/httplib.h out of this
  // header — only api_server.cpp and the smoke test pull it in).
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void register_routes();
};

// ── Production wiring of the opt-in C8 utility/admin endpoints (SERVE-METRICS /
// SERVE-UTILITY-ENDPOINTS / SERVE-ADMIN) from the LIVE engine + tokenizer,
// mirroring vLLM 0.26 @ 555967922's PER-ENDPOINT default gating. This is the
// SINGLE seam shared by the production server (examples/server/main.cpp) and its
// integration gate, so the gate exercises the exact wiring main() runs — the
// setters are never called twice or diverge between the two.
//
// Per-endpoint default gating (vLLM file:line ↔ ours):
//   /tokenize,/detokenize — ON by default when a tokenizer exists. vLLM's
//     build_app always calls register_vllm_serve_api_routers →
//     attach_tokenize_router (serve/__init__.py:11-31; api_server.py:222), which
//     registers POST /tokenize + /detokenize
//     (serve/tokenize/api_router.py:36,62). → set_tokenizer.
//   /tokenizer_info — OFF unless enable_tokenizer_info_endpoint
//     (serve/tokenize/api_router.py:95 gates attach on
//     app.state.args.enable_tokenizer_info_endpoint; cli_args.py:140 default
//     False). → our --enable-tokenizer-info-endpoint flag (opts.tokenizer_info).
//   /abort_requests — DEV-mode gated. vLLM registers the dev/rlhf router ONLY
//     under `if envs.VLLM_SERVER_DEV_MODE` (api_server.py:238-240 →
//     register_vllm_dev_api_routers → dev/rlhf/api_router.py:94); VLLM_SERVER_DEV_MODE
//     defaults 0 (envs.py:157,1350). → our --enable-server-dev-mode flag
//     (opts.server_dev_mode). Explicit-id abort routes through AsyncLLM::abort;
//     the abort-ALL (empty request_ids) path is a NAMED RESIDUAL — AsyncLLM
//     exposes no active-request-id enumeration, so empty ids abort nothing and
//     report 0 (explicit-id abort is the supported production path).
//   /metrics — wired by the CALLER, not here, because the caller owns the
//     PrometheusStatLogger's lifetime: server_main.cpp constructs one under
//     --enable-metrics, attaches it to both frontends (including the production
//     AsyncLLM, whose output handler folds each step's SchedulerStats +
//     IterationStats into it since #277 — specs/async-metrics.md) and hands it
//     to set_metrics_logger. Without --enable-metrics the route stays 404.
//   /reset_prefix_cache — NOT wired (deliberate, honest residual).
//     reset_prefix_cache() lives only on the scheduler's KVCacheManager, mutated
//     exclusively on the EngineCore engine thread with no thread-safe RPC.
//     Attaching a backing from main.cpp would be a fabricated wiring that
//     never reaches the live engine, so it is left unwired and named in
//     specs/{utility,admin}-endpoints.md.
struct UtilityEndpointOptions {
  bool enable_tokenizer_info_endpoint = false;  // --enable-tokenizer-info-endpoint
  bool enable_server_dev_mode = false;          // mirrors VLLM_SERVER_DEV_MODE
};

void ConfigureUtilityEndpoints(ApiServer& server,
                               const vllm::tok::Tokenizer& tokenizer,
                               int64_t max_model_len, vllm::v1::AsyncLLM& engine,
                               const UtilityEndpointOptions& options);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
