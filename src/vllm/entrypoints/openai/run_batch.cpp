// Ported from: vllm/entrypoints/openai/run_batch.py @ 555967922.
// See include/vllm/entrypoints/openai/run_batch.h for scope + recorded
// deviations.
#include "vllm/entrypoints/openai/run_batch.h"

#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

namespace {

// random_uuid() (vllm/utils) == uuid4().hex, 32 hex chars. Uniqueness only,
// mirroring make_tool_call_id (tool_parsers/abstract.cpp:58-68).
std::string RandomUuidHex() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hi << std::setw(16)
      << lo;
  return oss.str();
}

// make_error_request_output (run_batch.py:508-520): status 400, error is a
// plain string message, no body.
BatchRequestOutput MakeErrorRow(const std::string& custom_id,
                                const std::string& error_msg) {
  BatchRequestOutput out;
  out.id = "vllm-" + RandomUuidHex();
  out.custom_id = custom_id;
  BatchResponseData rd;
  rd.status_code = 400;  // HTTPStatus.BAD_REQUEST
  rd.request_id = "vllm-batch-" + RandomUuidHex();
  out.response = std::move(rd);
  out.error = nlohmann::json(error_msg);
  return out;
}

// run_request's ErrorResponse branch (run_batch.py:554-563) fed by
// create_error_response(e): status_code = error.code, body = null, error = the
// ErrorResponse object.
BatchRequestOutput MakeErrorResponseRow(const std::string& custom_id,
                                        const std::string& message,
                                        const std::string& type, int code) {
  ErrorResponse err;
  err.error.message = message;
  err.error.type = type;
  err.error.code = code;

  BatchRequestOutput out;
  out.id = "vllm-" + RandomUuidHex();
  out.custom_id = custom_id;
  BatchResponseData rd;
  rd.status_code = code;
  rd.request_id = "vllm-batch-" + RandomUuidHex();
  out.response = std::move(rd);
  out.error = nlohmann::json(err);  // ADL to_json(ErrorResponse)
  return out;
}

}  // namespace

void to_json(nlohmann::json& j, const BatchResponseData& d) {
  j = nlohmann::json{{"status_code", d.status_code},
                     {"request_id", d.request_id},
                     {"body", d.body.has_value() ? *d.body
                                                  : nlohmann::json(nullptr)}};
}

void to_json(nlohmann::json& j, const BatchRequestOutput& o) {
  j = nlohmann::json{{"id", o.id}, {"custom_id", o.custom_id}};
  j["response"] =
      o.response.has_value() ? nlohmann::json(*o.response) : nlohmann::json(nullptr);
  j["error"] = o.error.has_value() ? *o.error : nlohmann::json(nullptr);
}

RunBatch::RunBatch(OpenAIServingChat* chat, OpenAIServingModels* models)
    : chat_(chat), models_(models) {}

BatchRequestOutput RunBatch::DispatchChat(const std::string& custom_id,
                                          const nlohmann::json& body,
                                          const std::string& body_json) {
  // Mirrors run_request(openai_serving_chat.create_chat_completion, ...):
  // check_type_for_url validates the body as a ChatCompletionRequest
  // (run_batch.py:175-176); a validation failure surfaces through
  // create_error_response as an ErrorResponse row.
  if (chat_ == nullptr) {
    // handler_getter -> None (run_batch.py:600-603).
    return MakeErrorRow(
        custom_id, "Model does not support endpoint: /v1/chat/completions");
  }

  ChatCompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeErrorResponseRow(custom_id,
                                std::string("Invalid request body: ") + e.what(),
                                "BadRequestError", 400);
  }
  // Tool schemas keep the request document's key order (the pinned renderer
  // dumps them into the prompt with sort_keys=False); the nlohmann::json
  // parse above sorts object keys, so re-read `tools` order-preserving from
  // the body's ORIGINAL text. The argument is a string: passing the json
  // object itself would implicitly convert through get<std::string>() and
  // throw type_error.302 on every object body (ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN).
  RestoreToolSchemaOrder(body_json, request);

  // check_model (chat_completion/serving.py; api_server.cpp:186-190).
  if (models_ != nullptr && !models_->check_model(request.model)) {
    return MakeErrorResponseRow(
        custom_id,
        "The model `" + request.model.value_or("") + "` does not exist.",
        "NotFoundError", 404);
  }

  ChatCompletionResult result;
  try {
    result = chat_->create_chat_completion(request);
  } catch (const std::exception& e) {
    // create_error_response(e) (run_batch.py:537).
    return MakeErrorResponseRow(custom_id, e.what(), "InternalServerError", 500);
  }

  // A streamed request is invalid in the batch context (run_batch.py:564-567).
  if (result.streaming) {
    return MakeErrorRow(custom_id, "Request must not be sent in stream mode");
  }

  // The AllResponse success branch (run_batch.py:545-553).
  BatchRequestOutput out;
  out.id = "vllm-" + RandomUuidHex();
  out.custom_id = custom_id;
  BatchResponseData rd;
  rd.status_code = 200;
  rd.request_id = "vllm-batch-" + RandomUuidHex();
  rd.body = nlohmann::json(*result.response);  // ADL to_json(ChatCompletionResponse)
  out.response = std::move(rd);
  out.error = std::nullopt;  // null
  return out;
}

BatchRequestOutput RunBatch::RunLine(const std::string& request_json) {
  // BatchRequestInput.model_validate_json (run_batch.py:813). RECORDED
  // DEVIATION: upstream lets a validation error abort the job; here a bad line
  // becomes an error row so the batch continues (see run_batch.h).
  nlohmann::json line;
  try {
    line = nlohmann::json::parse(request_json);
  } catch (const std::exception& e) {
    return MakeErrorRow("", std::string("Invalid JSON in batch line: ") + e.what());
  }
  if (!line.is_object()) {
    return MakeErrorRow("", "Batch request line must be a JSON object");
  }

  // custom_id is a required field (run_batch.py:157); isolate its absence.
  std::string custom_id;
  if (auto it = line.find("custom_id");
      it != line.end() && it->is_string()) {
    custom_id = it->get<std::string>();
  } else {
    return MakeErrorRow("", "Batch request line is missing a string 'custom_id'");
  }

  if (auto it = line.find("url"); it == line.end() || !it->is_string()) {
    return MakeErrorRow(custom_id,
                        "Batch request line is missing a string 'url'");
  }
  const std::string url = line.at("url").get<std::string>();

  auto body_it = line.find("body");
  if (body_it == line.end()) {
    return MakeErrorRow(custom_id, "Batch request line is missing 'body'");
  }
  const nlohmann::json& body = *body_it;

  // Endpoint dispatch (run_batch.py:722-777,815-842). W1: only chat is wired.
  const auto ends_with = [](const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };
  if (url == "/v1/chat/completions") {
    // The chat body is the line's top-level "body" member. RestoreToolSchemaOrder
    // re-reads `tools` from the body's ORIGINAL text order (the pinned
    // renderer dumps schemas into the prompt with sort_keys=False), and the
    // nlohmann::json parse above sorted its keys — so re-serialize the body
    // from an order-preserving parse of the original line text.
    // ordered_json::dump keeps insertion order, so this text carries the
    // request document's key order (ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN).
    const nlohmann::ordered_json ordered_line =
        nlohmann::ordered_json::parse(request_json);
    return DispatchChat(custom_id, body, ordered_line.at("body").dump());
  }
  // Registered endpoint keys (run_batch.py:732-777) whose serving handler is not
  // wired here yet -> the "does not support endpoint" error (handler_getter ->
  // None, run_batch.py:600-603). Named residuals; see specs/batch-api.md.
  if (url == "/v1/embeddings" || ends_with(url, "/score") ||
      ends_with(url, "/rerank") || url == "/v1/audio/transcriptions" ||
      url == "/v1/audio/translations") {
    return MakeErrorRow(custom_id, "Model does not support endpoint: " + url);
  }
  // URL not matched at all (run_batch.py:832-842).
  return MakeErrorRow(
      custom_id,
      "URL " + url +
          " was used. Supported endpoints: /v1/chat/completions."
          " (/v1/embeddings, /score, /rerank, /v1/audio/transcriptions,"
          " /v1/audio/translations are named residuals; see"
          " specs/batch-api.md.)");
}

std::vector<BatchRequestOutput> RunBatch::RunLines(
    const std::string& input_jsonl) {
  std::vector<BatchRequestOutput> rows;
  std::istringstream stream(input_jsonl);
  std::string line;
  while (std::getline(stream, line)) {
    // Skip empty lines (run_batch.py:808-811). Strip trailing CR (CRLF files).
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // Strip surrounding whitespace for the emptiness check.
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos) continue;  // blank line
    rows.push_back(RunLine(line));
  }
  return rows;
}

std::string RunBatch::Run(const std::string& input_jsonl) {
  const std::vector<BatchRequestOutput> rows = RunLines(input_jsonl);
  std::ostringstream out;
  for (const BatchRequestOutput& row : rows) {
    out << nlohmann::json(row).dump() << "\n";  // write_local_file (:354-356)
  }
  return out.str();
}

size_t RunBatchFile(RunBatch& runner, const std::string& input_path,
                    const std::string& output_path) {
  std::ifstream in(input_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("run_batch: cannot open input file: " + input_path);
  }
  std::ostringstream buf;
  buf << in.rdbuf();

  const std::vector<BatchRequestOutput> rows = runner.RunLines(buf.str());

  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("run_batch: cannot open output file: " +
                             output_path);
  }
  for (const BatchRequestOutput& row : rows) {
    out << nlohmann::json(row).dump() << "\n";
  }
  return rows.size();
}

}  // namespace vllm::entrypoints::openai
