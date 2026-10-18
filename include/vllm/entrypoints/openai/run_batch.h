// Ported from: vllm/entrypoints/openai/run_batch.py @ 555967922 (vLLM
// 0.26.0.dev0). The OFFLINE Batch API runner: read a JSONL of OpenAI-format
// requests (BatchRequestInput), dispatch each line to the matching serving
// handler, collect BatchRequestOutput rows (custom_id echoed, per-line error
// isolation), and write the response JSONL.
//
// SCOPE (SERVE-BATCH-API, W1 — the first CPU-buildable brick): the ORCHESTRATOR
// over the existing OpenAIServingChat handler. It does NOT reimplement any
// generation — `/v1/chat/completions` lines drive
// OpenAIServingChat::create_chat_completion (the same handler
// api_server.cpp::handle_chat_completions drives), mirroring vLLM's
// endpoint_registry mapping url -> openai_serving_chat.create_chat_completion
// (run_batch.py:722-731,844-847).
//
// SCHEMA (run_batch.py:148-228):
//   BatchRequestInput  { custom_id, method, url, body }
//   BatchResponseData  { status_code=200, request_id, body: AllResponse|null }
//   BatchRequestOutput { id, custom_id, response|null, error|null }
// The per-line id is "vllm-<uuid>", the response request_id "vllm-batch-<uuid>"
// (run_batch.py:511-519,546-563).
//
// RECORDED DEVIATION (per-line isolation): upstream `run_batch` does NOT wrap
// `BatchRequestInput.model_validate_json` in try/except, so a MALFORMED line
// aborts the whole job (test_run_batch.py::test_completions_invalid_input asserts
// returncode != 0). This runner instead isolates a bad line into an error row
// and continues — the robust offline-batch shape the task requires. The
// per-HANDLER error isolation (an exception from the serving handler -> an error
// row) IS faithful to upstream run_request (run_batch.py:534-537). The vLLM
// abort-on-bad-line contract, if mirrored, would live in the (unbuilt) CLI
// wrapper's exit code; see specs/batch-api.md § Deviations.
//
// DEFERRED / RESIDUALS (named; spec § Work breakdown):
//   - /v1/embeddings, /score, /rerank, /v1/audio/{transcriptions,translations}
//     dispatch: no serving handler wired yet -> the "does not support endpoint"
//     error row (mirrors handler_getter -> None, run_batch.py:600-603).
//   - concurrency: W1 runs lines sequentially through the sync handler (each
//     create_chat_completion runs to completion); upstream's asyncio.gather
//     overlap over one AsyncLLM is a later brick.
//   - file I/O: local paths only; the http(s) GET/PUT and data-URL fetch vLLM
//     supports (run_batch.py:334-505) is a named later brick.
//   - the `vllm run-batch` CLI + BatchFrontendArgs wiring (run_batch.py:232-297).
#ifndef VLLM_ENTRYPOINTS_OPENAI_RUN_BATCH_H_
#define VLLM_ENTRYPOINTS_OPENAI_RUN_BATCH_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_models.h"

namespace vllm::entrypoints::openai {

// The body of the response (run_batch.py:202-210). `body` holds the serialized
// AllResponse (a ChatCompletionResponse at W1); null when the request errored.
struct BatchResponseData {
  int status_code = 200;
  std::string request_id;
  std::optional<nlohmann::json> body;
};

// The per-line object of the batch output/error file (run_batch.py:213-228).
struct BatchRequestOutput {
  std::string id;         // "vllm-<uuid>"
  std::string custom_id;  // echoed from the input line
  std::optional<BatchResponseData> response;
  // Either a plain string message (make_error_request_output, run_batch.py:518)
  // or an ErrorResponse object (run_request's ErrorResponse branch,
  // run_batch.py:562); null on success.
  std::optional<nlohmann::json> error;
};

void to_json(nlohmann::json& j, const BatchResponseData& d);
void to_json(nlohmann::json& j, const BatchRequestOutput& o);

// The batch runner. Holds non-owning pointers to the serving handlers (like
// vLLM's endpoint_registry built from the init_app_state serving objects,
// run_batch.py:687-777). At W1 only chat is wired; `models` is optional (null
// skips check_model). The handlers are constructed + owned by the caller and
// must outlive the runner.
class RunBatch {
 public:
  explicit RunBatch(OpenAIServingChat* chat,
                    OpenAIServingModels* models = nullptr);

  // Dispatch every line of `input_jsonl` (blank lines skipped, run_batch.py:
  // 808-811) and collect the output rows IN ORDER (run_batch.py:806-847).
  std::vector<BatchRequestOutput> RunLines(const std::string& input_jsonl);

  // As RunLines, serialized back to a JSONL string (one row per line, trailing
  // newline per line, mirroring write_local_file, run_batch.py:354-356).
  std::string Run(const std::string& input_jsonl);

  // Dispatch a single JSONL line to a BatchRequestOutput. Exposed for
  // fine-grained testing.
  BatchRequestOutput RunLine(const std::string& request_json);

 private:
  // `body_json` is the chat body's serialization from an ORDER-PRESERVING
  // parse of the original line text: the nlohmann::json `body` sorts object
  // keys, and RestoreToolSchemaOrder re-reads `tools` from this text so the
  // request document's schema key order survives into the prompt.
  BatchRequestOutput DispatchChat(const std::string& custom_id,
                                  const nlohmann::json& body,
                                  const std::string& body_json);

  OpenAIServingChat* chat_;
  OpenAIServingModels* models_;
};

// Local-file runner: read the JSONL at `input_path`, run it, write the response
// JSONL to `output_path` (mirrors read_file + write_local_file for the local
// path, run_batch.py:334-356). Returns the number of output rows written.
// Throws std::runtime_error if a file cannot be opened. s3://http(s) fetch is a
// named residual.
size_t RunBatchFile(RunBatch& runner, const std::string& input_path,
                    const std::string& output_path);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_RUN_BATCH_H_
