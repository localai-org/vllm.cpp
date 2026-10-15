// Bounded public C-ABI image chat vehicle. No internal engine headers.
#include "vllm.h"
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using json = nlohmann::json;
void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
std::string Error() {
  const char* text = vllm_last_error();
  return text ? text : "unknown C-API error";
}
struct Stream {
  json chunks = json::array();
  int terminal = 0;
  std::string error;
};
bool Collect(const char* delta, bool finished, void* opaque) {
  auto& state = *static_cast<Stream*>(opaque);
  try {
    if (finished) {
      Require(delta && !*delta, "terminal callback must have empty text");
      ++state.terminal;
    } else {
      Require(state.terminal == 0 && delta && *delta, "invalid callback order/payload");
      state.chunks.push_back(json::parse(delta));
    }
    return true;
  } catch (const std::exception& error) {
    state.error = error.what();
    return false;
  }
}
void Run(const char* model, const char* request_file, int depth, json& report) {
  std::ifstream input(request_file);
  Require(input.good(), "cannot open request bundle");
  const json requests = json::parse(input);
  Require(requests.is_array() && !requests.empty() && requests.size() <= 8, "one to eight bounded requests required");
  vllm_model_params params = vllm_model_params_default();
  params.model_path = model;
  params.block_size = 1600;
  params.num_blocks = 180;
  params.max_model_len = 8192;
  params.max_num_seqs = 1;
  params.max_num_batched_tokens = 1600;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = 2;  // Explicit OFF, not model default.
  params.gpu_memory_utilization = 0.;
  params.limit_mm_per_prompt = "{\"image\":2,\"video\":0}";
  const std::string spec = "{\"method\":\"mtp\",\"num_speculative_tokens\":3}";
  if (depth) params.speculative_config = spec.c_str();
  vllm_engine* handle = nullptr;
  const auto loaded = vllm_engine_load(&params, &handle);
  Require(loaded == VLLM_OK, "engine load: " + Error());
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(handle, &vllm_engine_free);
  report = {{"status", "FAIL"}, {"spec_depth", depth}, {"abi_version", VLLM_ABI_VERSION},
              {"scope", "direct C1 public C ABI ordinary/streamed PNG/JPEG chat; no HTTP, tensor or speed parity"},
              {"cases", json::array()}};
  for (const auto& entry : requests) {
    json body = entry.at("body");
    Require(body.value("n", 1) == 1 && body.at("max_tokens").get<int>() > 0 &&
            body.at("max_tokens").get<int>() <= 128, "bounded single-choice request required");
    body["stream"] = false;
    const std::string ordinary_body = body.dump();
    char* response_text = nullptr;
    const auto completed = vllm_chat(engine.get(), ordinary_body.c_str(), &response_text);
    Require(completed == VLLM_OK, "ordinary chat: " + Error());
    std::unique_ptr<char, decltype(&vllm_string_free)> owned(response_text, &vllm_string_free);
    Require(response_text != nullptr, "missing ordinary response");
    const auto response = json::parse(response_text);
    Require(response.at("choices").size() == 1, "wrong ordinary choice count");
    const auto choice = response.at("choices").at(0);
    const std::string text = choice.at("message").at("content").get<std::string>();
    const auto usage = response.at("usage");
    Require(!text.empty() && usage.at("prompt_tokens").get<int>() > 0 &&
            usage.at("completion_tokens").get<int>() > 0 &&
            usage.at("total_tokens").get<int>() == usage.at("prompt_tokens").get<int>() +
                                                    usage.at("completion_tokens").get<int>(), "invalid ordinary text/usage");
    body["stream"] = true;
    body["stream_options"] = {{"include_usage", true}};
    const std::string streamed_body = body.dump();
    Stream stream;
    const auto streamed = vllm_chat_stream(engine.get(), streamed_body.c_str(), &Collect, &stream);
    Require(streamed == VLLM_OK, "stream chat: " + Error());
    Require(stream.error.empty() && stream.terminal == 1 && !stream.chunks.empty(), "invalid terminal stream callback: " + stream.error);
    std::string streamed_text;
    json streamed_usage;
    int usage_count = 0, finish_count = 0;
    const auto identity = stream.chunks.at(0).at("id");
    for (const auto& chunk : stream.chunks) {
      Require(chunk.at("id") == identity && chunk.at("model") == response.at("model"), "wrong callback identity/model");
      if (chunk.contains("usage") && !chunk.at("usage").is_null()) {
        streamed_usage = chunk.at("usage"); ++usage_count;
      }
      for (const auto& delta : chunk.at("choices")) {
        Require(delta.at("index") == 0, "wrong stream choice index");
        const auto& fields = delta.at("delta");
        if (fields.contains("content") && !fields.at("content").is_null()) streamed_text += fields.at("content").get<std::string>();
        if (!delta.at("finish_reason").is_null()) {
          Require(delta.at("finish_reason") == choice.at("finish_reason"), "ordinary/stream finish differs");
          ++finish_count;
        }
      }
    }
    Require(streamed_text == text && usage_count == 1 && streamed_usage == usage && finish_count == 1,
            "ordinary/stream output or usage differs");
    report["cases"].push_back({{"name", entry.at("name")}, {"ordinary_response", response},
        {"streamed_text", streamed_text}, {"streamed_usage", streamed_usage},
        {"callback_chunks", stream.chunks.size()}, {"terminal_callbacks", stream.terminal},
        {"ordinary_stream_equal", true}});
  }
  vllm_spec_acceptance counters{};
  Require(vllm_engine_spec_acceptance(engine.get(), &counters) == VLLM_OK, "missing speculation counters");
  Require(depth ? counters.drafts_proposed > 0 : counters.drafts_proposed == 0, "unexpected/missing actual proposals");
  report["drafts_proposed"] = counters.drafts_proposed;
  report["drafts_accepted"] = counters.drafts_accepted;
  report["drafted_request_steps"] = counters.drafted_request_steps;
  engine.reset();
  report["engine_freed"] = true;
  report["status"] = "PASS";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: native-vision-chat MODEL_DIR REQUEST_BUNDLE_JSON OUTPUT_JSON 0|3\n";
    return 2;
  }
  if (std::filesystem::exists(argv[3])) {
    std::cerr << "output exists; preserve previous evidence\n";
    return 2;
  }
  json report;
  int code = 1;
  try {
    const int depth = std::stoi(argv[4]);
    Require(depth == 0 || depth == 3, "depth must be zero or three");
    Run(argv[1], argv[2], depth, report); code = 0;
  } catch (const std::exception& error) { report["status"] = "FAIL"; report["error"] = error.what(); }
  std::ofstream output(argv[3]);
  if (!output.good()) { std::cerr << "cannot write output\n"; return 2; }
  output << report.dump(2) << '\n';
  std::cout << json{{"status", report.value("status", "FAIL")}, {"error", report.value("error", "")},
                   {"drafts_proposed", report.value("drafts_proposed", int64_t{0})}}.dump() << '\n';
  return code;
}
