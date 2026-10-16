#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sys/resource.h>

#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/config/speculative.h"
#include "vt/xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/exl3_grouped.h"
#include "vt/breakable_graph.h"
#include "vllm/v1/core/sched/scheduler.h"

namespace {
std::vector<int32_t> Ints(const vllm::StTensor& tensor) {
  REQUIRE((tensor.dtype == "I32" || tensor.dtype == "I64"));
  const size_t width = tensor.dtype == "I32" ? 4 : 8;
  std::vector<int32_t> out(tensor.nbytes / width);
  for (size_t i = 0; i < out.size(); ++i) {
    int64_t value = 0;
    if (width == 8) std::memcpy(&value, tensor.data + i * width, width);
    else { int32_t narrow; std::memcpy(&narrow, tensor.data + i * width, width); value = narrow; }
    REQUIRE(value >= 0);
    REQUIRE(value <= INT32_MAX);
    out[i] = static_cast<int32_t>(value);
  }
  return out;
}

std::vector<float> Floats(const vllm::StTensor& tensor) {
  REQUIRE((tensor.dtype == "F16" || tensor.dtype == "F32"));
  const size_t width = tensor.dtype == "F16" ? 2 : 4;
  std::vector<float> out(tensor.nbytes / width);
  for (size_t i = 0; i < out.size(); ++i) {
    if (width == 4) std::memcpy(&out[i], tensor.data + i * width, width);
    else { uint16_t bits; std::memcpy(&bits, tensor.data + i * width, width); out[i] = vt::F16ToF32(bits); }
  }
  return out;
}

void Compare(const char* label, const std::vector<float>& got,
              const vllm::StTensor& expected, bool exact) {
  if (const char* output = std::getenv("VT_B70_EXL3_MTP_OUTPUT")) {
    const auto path = std::filesystem::path(output) / (std::string(label) + ".f32.bin");
    REQUIRE_FALSE(std::filesystem::exists(path));
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(got.data()), got.size() * sizeof(float));
    file.close();
    REQUIRE(file.good());
  }
  const auto want = Floats(expected);
  REQUIRE(got.size() == want.size());
  double error = 0, norm = 0;
  size_t different = 0;
  bool finite = true;
  for (size_t i = 0; i < got.size(); ++i) {
    finite &= std::isfinite(got[i]) && std::isfinite(want[i]);
    const double delta = double(got[i]) - want[i];
    error += delta * delta; norm += double(want[i]) * want[i];
    different += got[i] != want[i];
  }
  REQUIRE(finite);
  REQUIRE(norm > 0);
  const double relative = std::sqrt(error / norm);
  std::cout << "REAL_EXL3_MTP stage=" << label << " relative=" << relative
            << " different=" << different << '\n';
  if (exact) CHECK(different == 0);
  else {
    CHECK(relative < 2e-3);  // existing bounded native block comparison budget
    size_t failed = 0, first = got.size();
    for (size_t i = 0; i < got.size(); ++i) {
      // Same pointwise band as xpu_test::Close; collect rather than abort so
      // the independent head/global-ID checks run even if forward fails.
      if (std::abs(got[i] - want[i]) > .003f + .01f * std::abs(want[i])) {
        if (failed++ == 0) first = i;
      }
    }
    std::cout << "REAL_EXL3_MTP pointwise_failed=" << failed << " first=" << first << '\n';
    CHECK(failed == 0);
  }
}

// Read-only evidence after completed synchronous engine generation. The
// native draft's chosen-ID download has drained the model queue; this probe
// queue never overlaps model execution and does not establish in-flight safety.
nlohmann::json SnapshotGdnStates(const std::vector<vllm::GdnStateCache>& states,
                                vt::Queue& queue, const std::string& prefix,
                                const char* baseline_prefix, int request) {
  auto& backend = vt::GetBackend(queue.device);
  nlohmann::json snapshots = nlohmann::json::array();
  REQUIRE(states.size() == 48);
  for (size_t layer = 0; layer < states.size(); ++layer) {
    for (const std::string kind : {"conv", "ssm"}) {
      const auto& state = kind == "conv" ? states[layer].conv_state : states[layer].ssm_state;
      CAPTURE(layer);
      CAPTURE(kind);
      REQUIRE(state.IsContiguous());
      CHECK(state.dtype == (kind == "conv" ? vt::DType::kF16 : vt::DType::kF32));
      const size_t row_bytes = size_t(state.stride[0]) * vt::SizeOf(state.dtype);
      REQUIRE(row_bytes > 0);
      const std::string suffix = "-r" + std::to_string(request) + "-gdn" +
                                 std::to_string(layer) + "-" + kind + ".bin";
      const std::string path = prefix + suffix;
      REQUIRE_FALSE(std::filesystem::exists(path));
      std::ofstream file(path, std::ios::binary);
      REQUIRE(file.good());
      std::ifstream reference;
      if (baseline_prefix) {
        const std::string reference_path = std::string(baseline_prefix) + suffix;
        REQUIRE(std::filesystem::file_size(reference_path) == row_bytes * state.shape[0]);
        reference.open(reference_path, std::ios::binary);
        REQUIRE(reference.good());
      }
      std::vector<uint8_t> row(row_bytes), expected(baseline_prefix ? row_bytes : 0);
      size_t different_bytes = 0;
      for (int64_t slot = 0; slot < state.shape[0]; ++slot) {
        CAPTURE(slot);
        backend.Copy(queue, row.data(), static_cast<const uint8_t*>(state.data) +
                     slot * row_bytes, row_bytes);
        backend.Synchronize(queue);
        file.write(reinterpret_cast<const char*>(row.data()), row.size());
        if (baseline_prefix) {
          reference.read(reinterpret_cast<char*>(expected.data()), expected.size());
          REQUIRE(reference.good());
          size_t changed = 0;
          for (size_t i = 0; i < row.size(); ++i) changed += row[i] != expected[i];
          different_bytes += changed;
          CHECK(changed == 0);
        }
      }
      file.close();
      REQUIRE(file.good());
      snapshots.push_back({{"layer", layer}, {"kind", kind}, {"dtype", vt::Name(state.dtype)},
          {"slots", state.shape[0]}, {"bytes", row_bytes * state.shape[0]},
          {"file", std::filesystem::path(path).filename().string()},
          {"baseline_compared", baseline_prefix != nullptr}, {"different_bytes", different_bytes}});
    }
  }
  return snapshots;
}
}  // namespace

TEST_CASE("XPU EXL3 public engine: target filters and EOS during verification") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* depth = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  if (!model || !output || !depth) std::exit(77);
  const int k = std::stoi(depth);
  REQUIRE((k == 0 || k == 1 || k == 3));
  REQUIRE_FALSE(std::filesystem::exists(output));
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 4352;
  params.max_num_batched_tokens = 4096;
  params.max_num_seqs = 1;
  params.num_blocks = 24;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  if (k) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":" + std::to_string(k) + "}");
  auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  const auto& raw = engine->config().raw;
  const auto& text_config = raw.contains("text_config") ? raw.at("text_config") : raw;
  const int32_t eos = text_config.at("eos_token_id").get<int32_t>();
  constexpr int32_t allowed = 271;
  vllm::SamplingParams sampling;
  const char* temperature = std::getenv("VT_B70_EXL3_ENGINE_TEMPERATURE");
  sampling.temperature = temperature ? std::stod(temperature) : 0;
  sampling.top_k = 20;
  sampling.top_p = .8;
  sampling.min_p = .05;
  sampling.seed = 931;
  sampling.max_tokens = 8;
  sampling.min_tokens = 3;
  sampling.allowed_token_ids = std::vector<int32_t>{allowed, eos};
  sampling.logit_bias = std::map<int32_t, float>{{eos, 100}};
  sampling.output_kind = vllm::RequestOutputKind::kCumulative;
  const bool processors = std::getenv("VT_B70_EXL3_ENGINE_PROCESSORS") != nullptr;
  std::vector<std::vector<int32_t>> callback_histories;
  std::vector<int32_t> expected_ids{allowed, allowed, allowed, eos};
  if (processors) {
    sampling.allowed_token_ids = std::vector<int32_t>{allowed, 1206, eos};
    sampling.bad_words_token_ids = std::vector<std::vector<int32_t>>{{allowed, allowed}};
    sampling.presence_penalty = .25;
    sampling.frequency_penalty = 2;
    sampling.repetition_penalty = 1.1;
    sampling.logits_processor = {
        +[](const int32_t* ids, int32_t count, float* logits, int32_t vocab, void* opaque) {
          auto& histories = *static_cast<std::vector<std::vector<int32_t>>*>(opaque);
          histories.emplace_back();
          if (count) histories.back().assign(ids, ids + count);
          if (vocab > 1206) {
            if (std::isfinite(logits[271])) logits[271] = 22;
            if (std::isfinite(logits[1206])) logits[1206] = 19;
          }
        }, &callback_histories};
    expected_ids = {allowed, 1206, allowed, eos};
  }
  nlohmann::json result = {{"depth", k}, {"eos", eos},
                          {"temperature", sampling.temperature},
                          {"top_k", 20}, {"top_p", .8}, {"min_p", .05}, {"seed", 931},
                          {"history_processors", processors},
                          {"requests", nlohmann::json::array()}};
  for (int i = 0; i < 2; ++i) {
    const auto proposed = engine->runner().spec_drafts_proposed();
    const auto out = engine->engine().generate(
        "A gardener plants 12 tomato plants in each of 3 rows. How many plants?",
        sampling, "filters-eos-" + std::to_string(i));
    REQUIRE(out.outputs.size() == 1);
    const auto& completion = out.outputs[0];
    result["requests"].push_back({{"ids", completion.token_ids}, {"text", completion.text},
                                  {"finish_reason", completion.finish_reason.value_or("")},
                                  {"proposed", engine->runner().spec_drafts_proposed() - proposed}});
    if (processors) result["callback_histories"] = callback_histories;
    std::ofstream file(output);
    file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
    CHECK(out.finished);
    CHECK(completion.finish_reason == "stop");
    CHECK(completion.token_ids == expected_ids);
    CHECK_FALSE(engine->engine().has_unfinished_requests());
    if (k) CHECK(engine->runner().spec_drafts_proposed() > proposed);
  }
  if (processors) {
    CHECK_FALSE(callback_histories.empty());
    CHECK(std::find(callback_histories.begin(), callback_histories.end(),
                    std::vector<int32_t>{allowed, 1206}) != callback_histories.end());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine: C1 short decode graph boundary") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* state_prefix = std::getenv("VT_B70_EXL3_ENGINE_STATE_PREFIX");
  REQUIRE(model != nullptr);
  REQUIRE(output != nullptr);
  REQUIRE(state_prefix != nullptr);
  REQUIRE_FALSE(std::filesystem::exists(output));
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 2048;
  params.max_num_batched_tokens = 1024;
  params.max_num_seqs = 1;
  params.num_blocks = 8;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  std::string text;
  for (int i = 0; i < 100; ++i)
    text += "A garden has twelve tomato plants in each row. Explain how to count "
            "the plants and water them evenly during a warm summer. ";
  auto prompt = loaded->tokenizer().Encode(text);
  REQUIRE(prompt.size() >= 956);
  prompt.resize(956);
  vllm::SamplingParams sampling;
  sampling.temperature = 0;
  sampling.max_tokens = 12;
  sampling.ignore_eos = true;
  sampling.output_kind = vllm::RequestOutputKind::kCumulative;
  const char* graph_setting = std::getenv("VT_B70_EXL3_ENGINE_GRAPH");
  const bool graph_requested = graph_setting && std::stoi(graph_setting) == 1;
  nlohmann::json result = {{"prompt_ids", prompt}, {"graph_requested", graph_requested},
      {"steps", nlohmann::json::array()}, {"ids", nlohmann::json::array()}};
  vt::ResetGraphBreakStats();
  loaded->engine().add_request("short-decode-boundary", prompt, sampling);
  bool replay_below = false, replay_above = false;
  std::set<int> lengths;
  for (int cycle = 0; loaded->engine().has_unfinished_requests() && cycle < 20; ++cycle) {
    const auto before = vt::GetGraphBreakStats();
    const auto outputs = loaded->engine().step();
    const auto after = vt::GetGraphBreakStats();
    const auto& runner = loaded->runner();
    const int length = runner.last_attn_meta().max_seq_len;
    const bool replayed = after.replays > before.replays;
    lengths.insert(length);
    replay_below |= length <= 960 && replayed;
    replay_above |= length > 960 && replayed;
    result["steps"].push_back({{"length", length},
        {"input_rows", runner.last_step().input_token_ids.size()},
        {"captures", after.segments_captured}, {"replays", after.replays},
        {"replayed", replayed}});
    for (const auto& request : outputs) {
      REQUIRE(request.outputs.size() == 1);
      result["ids"] = request.outputs.front().token_ids;
      if (request.finished) result["finish_reason"] = request.outputs.front().finish_reason;
    }
  }
  CHECK_FALSE(loaded->engine().has_unfinished_requests());
  CHECK(result["ids"].size() == 12);
  for (int length : {959, 960, 961, 962}) CHECK(lengths.count(length) == 1);
  if (graph_requested) {
    CHECK(replay_below);
    CHECK(replay_above);
  }
  if (const char* baseline = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(baseline);
    REQUIRE(file.good());
    const auto reference = nlohmann::json::parse(file);
    CHECK(result["prompt_ids"] == reference["prompt_ids"]);
    CHECK(result["ids"] == reference["ids"]);
  }
  xpu_test::Queue probe(vt::DeviceType::kXPU);
  result["gdn_states"] = SnapshotGdnStates(loaded->runner().gdn_state(), probe.q,
      state_prefix, std::getenv("VT_B70_EXL3_ENGINE_STATE_BASELINE_PREFIX"), 0);
  loaded.reset();
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  std::ofstream file(output);
  file << result.dump(2) << '\n';
  file.close();
  REQUIRE(file.good());
}

TEST_CASE("XPU EXL3 public engine: autonomous fixed MTP depth and request reuse") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* depth = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  if (!model || !output || !depth) {
    std::cerr << "requires VT_B70_EXL3_MODEL/ENGINE_OUTPUT/ENGINE_DEPTH\n";
    std::exit(77);
  }
  const int k = std::stoi(depth);
  REQUIRE((k == 0 || k == 1 || k == 3));
  REQUIRE_FALSE(std::filesystem::exists(output));
  // Optional F2 qualification uses exactly the frozen serving prompt, with
  // three real W8A8 prefill chunks. Ordinary short reuse remains unchanged.
  const char* prefill_workload = std::getenv("VT_B70_EXL3_ENGINE_PREFILL_WORKLOAD");
  std::vector<int32_t> prefill_ids;
  if (prefill_workload) {
    std::ifstream file(prefill_workload);
    REQUIRE(file.good());
    const auto task = nlohmann::json::parse(file);
    REQUIRE(task.at("schema") == "b70-f2-fixed-prefill-prompt-v1");
    prefill_ids = task.at("prompt_ids").get<std::vector<int32_t>>();
    REQUIRE(prefill_ids.size() == 4096);
    REQUIRE(k == 3);
  }
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 4352;
  params.max_num_batched_tokens = prefill_workload ? 1600 : 4096;
  params.max_num_seqs = 1;
  params.num_blocks = 24;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  if (k) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":" + std::to_string(k) + "}");
  const char* state_prefix = std::getenv("VT_B70_EXL3_ENGINE_STATE_PREFIX");
  const char* state_baseline = std::getenv("VT_B70_EXL3_ENGINE_STATE_BASELINE_PREFIX");
  std::unique_ptr<xpu_test::Queue> state_probe;
  if (state_prefix) {
    REQUIRE(std::string(std::getenv("VT_ASYNC_RUNNER") ? std::getenv("VT_ASYNC_RUNNER") : "") == "0");
    state_probe = std::make_unique<xpu_test::Queue>(vt::DeviceType::kXPU);
  }
  REQUIRE((!prefill_workload || state_probe));
  auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  if (prefill_workload) {
    // Deterministically initialize every allocated row, including spare and
    // provisional slots. Full snapshot comparison never reads uninitialized
    // bytes. First-use reset and subsequent request reuse still run normally.
    auto& backend = vt::GetBackend(state_probe->q.device);
    for (const auto& state : engine->runner().gdn_state()) {
      backend.Memset(state_probe->q, state.conv_state.data, 0x35, state.conv_state.Bytes());
      backend.Memset(state_probe->q, state.ssm_state.data, 0x29, state.ssm_state.Bytes());
    }
    backend.Synchronize(state_probe->q);
  }
  vllm::SamplingParams sampling;
  sampling.temperature = 0;
  sampling.max_tokens = 64;
  if (prefill_workload) sampling.ignore_eos = true;
  sampling.output_kind = vllm::RequestOutputKind::kCumulative;
  const std::string prompt =
      "A gardener plants 12 tomato plants in each of 3 rows. "
      "How many tomato plants are there? Explain briefly.";
  nlohmann::json result = {{"depth", k}, {"prompt", prompt},
                          {"requests", nlohmann::json::array()}};
  if (prefill_workload) {
    result["prompt_ids"] = prefill_ids;
    result["prefill_chunk_tokens"] = 1600;
    result["all_recurrent_rows_initialized"] = true;
  }
  const char* graph_setting = std::getenv("VT_B70_EXL3_ENGINE_GRAPH");
  const bool graph_requested = graph_setting && std::stoi(graph_setting) == 1;
  result["graph_requested"] = graph_requested;
  vt::ResetGraphBreakStats();
  std::vector<int32_t> first;
  for (int request = 0; request < 2; ++request) {
    const auto& runner = engine->runner();
    const int64_t calls = runner.spec_mtp_propose_calls();
    const int64_t forwards = runner.spec_mtp_draft_decode_forwards();
    const int64_t varied = runner.spec_mtp_proposals_with_varied_drafts();
    const int64_t proposed = runner.spec_drafts_proposed();
    const int64_t accepted = runner.spec_drafts_accepted();
    const std::string request_id = "native-reuse-" + std::to_string(request);
    const auto out = prefill_workload
        ? engine->engine().generate(prefill_ids, sampling, request_id)
        : engine->engine().generate(prompt, sampling, request_id);
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const auto& ids = out.outputs.front().token_ids;
    result["requests"].push_back({
        {"ids", ids}, {"text", out.outputs.front().text},
        {"propose_calls", runner.spec_mtp_propose_calls() - calls},
        {"decode_forwards", runner.spec_mtp_draft_decode_forwards() - forwards},
        {"varied_proposals", runner.spec_mtp_proposals_with_varied_drafts() - varied},
        {"proposed", runner.spec_drafts_proposed() - proposed},
        {"accepted", runner.spec_drafts_accepted() - accepted},
        {"proposed_by_depth_cumulative", runner.spec_drafts_proposed_by_depth()},
        {"accepted_by_depth_cumulative", runner.spec_drafts_accepted_by_depth()}});
    const auto graph_stats = vt::GetGraphBreakStats();
    result["requests"].back()["graph_stats"] = {
        {"segments_captured", graph_stats.segments_captured}, {"replays", graph_stats.replays},
        {"device_bytes", vt::xpu::GetMemoryInfo().graph_device_bytes}};
    if (state_prefix) {
      CHECK(runner.gdn_state_slots() == k + 1);
      result["requests"].back()["gdn_states"] = SnapshotGdnStates(
          runner.gdn_state(), state_probe->q, state_prefix, state_baseline, request);
    }
    // Retain real IDs and counters even when a functional check fails.
    std::ofstream file(output);
    file << result.dump(2) << '\n';
    file.close();
    REQUIRE(file.good());
    CHECK(ids.size() == 64);
    if (graph_requested) {
      CHECK(graph_stats.segments_captured > 0);
      CHECK(graph_stats.replays > 0);
    }
    if (request == 0) first = ids;
    else CHECK(ids == first);
    if (k) {
      CHECK(runner.spec_mtp_propose_calls() > calls);
      CHECK(runner.spec_mtp_draft_decode_forwards() - forwards ==
            (runner.spec_mtp_propose_calls() - calls) * (k - 1));
      CHECK(runner.spec_drafts_proposed_by_depth().size() == static_cast<size_t>(k));
      if (k == 3) CHECK(runner.spec_mtp_proposals_with_varied_drafts() > varied);
    } else CHECK(runner.spec_mtp_propose_calls() == 0);
  }
  if (const char* reference = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(reference);
    REQUIRE(file.good());
    const auto baseline = nlohmann::json::parse(file);
    for (const auto& request : result["requests"])
      CHECK(request["ids"] == baseline["requests"][0]["ids"]);
  }
  if (k == 3 && std::getenv("VT_B70_EXL3_ENGINE_LIFECYCLE")) {
    result["lifecycle"] = nlohmann::json::array();
    const auto save = [&] {
      std::ofstream file(output);
      file << result.dump(2) << '\n';
      file.close();
      REQUIRE(file.good());
    };
    const auto check_prefix = [&](const vllm::RequestOutput& out) {
      REQUIRE(out.finished);
      REQUIRE(out.outputs.size() == 1);
      const auto& ids = out.outputs[0].token_ids;
      REQUIRE(ids.size() <= first.size());
      CHECK(std::equal(ids.begin(), ids.end(), first.begin()));
      CHECK_FALSE(engine->engine().has_unfinished_requests());
    };
    for (int limit : {1, 2, 3, 4, 5}) {
      auto short_sampling = sampling;
      short_sampling.max_tokens = limit;
      const auto out = engine->engine().generate(prompt, short_sampling,
                                                 "limit-" + std::to_string(limit));
      REQUIRE(out.outputs.size() == 1);
      result["lifecycle"].push_back({{"case", "limit"}, {"limit", limit},
                                    {"ids", out.outputs[0].token_ids}});
      save();
      check_prefix(out);
      CHECK(out.outputs[0].token_ids.size() == static_cast<size_t>(limit));
    }
    auto stop_sampling = sampling;
    stop_sampling.stop = {"tomato plants"};
    stop_sampling.include_stop_str_in_output = true;
    const auto stop = engine->engine().generate(prompt, stop_sampling, "stop-string");
    REQUIRE(stop.outputs.size() == 1);
    result["lifecycle"].push_back({{"case", "stop-string"},
                                  {"ids", stop.outputs[0].token_ids},
                                  {"text", stop.outputs[0].text}});
    save();
    check_prefix(stop);
    CHECK(stop.outputs[0].text.ends_with("tomato plants"));
    CHECK(stop.outputs[0].token_ids.size() < first.size());
    auto token_stop_sampling = sampling;
    token_stop_sampling.stop_token_ids = {first.at(4)};
    const auto token_stop = engine->engine().generate(prompt, token_stop_sampling,
                                                      "stop-token");
    REQUIRE(token_stop.outputs.size() == 1);
    result["lifecycle"].push_back({{"case", "stop-token"},
                                  {"ids", token_stop.outputs[0].token_ids}});
    save();
    check_prefix(token_stop);
    CHECK(token_stop.outputs[0].token_ids.size() <= 5);
    // First step fills target/draft state, second performs a provisional
    // verification. Abort at that boundary before scheduling another step.
    engine->engine().add_request("cancel-provisional", prompt, sampling);
    engine->engine().step();
    const int64_t verified_before = engine->runner().spec_drafts_proposed();
    engine->engine().step();
    CHECK(engine->runner().spec_drafts_proposed() > verified_before);
    engine->engine().abort_request("cancel-provisional");
    CHECK_FALSE(engine->engine().has_unfinished_requests());
    const auto reused = engine->engine().generate(prompt, sampling, "reuse-after-cancel");
    REQUIRE(reused.outputs.size() == 1);
    result["lifecycle"].push_back({{"case", "reuse-after-cancel"},
                                  {"ids", reused.outputs[0].token_ids}});
    save();
    check_prefix(reused);
    CHECK(reused.outputs[0].token_ids == first);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine R08: simultaneous C4/C2 unequal requests") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* depth = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  if (!model || !output || !depth) std::exit(77);
  const int k = std::stoi(depth);
  REQUIRE((k == 0 || k == 3));
  const char* graph_setting = std::getenv("VT_B70_EXL3_ENGINE_GRAPH");
  const bool graph_requested = graph_setting && std::stoi(graph_setting) == 1;
  const bool graph_screen = std::getenv("VT_B70_EXL3_ENGINE_GRAPH_SCREEN") != nullptr;
  REQUIRE_FALSE(std::filesystem::exists(output));
  const char* state_prefix = std::getenv("VT_B70_EXL3_ENGINE_STATE_PREFIX");
  const char* state_baseline = std::getenv("VT_B70_EXL3_ENGINE_STATE_BASELINE_PREFIX");
  std::unique_ptr<xpu_test::Queue> state_probe;
  if (state_prefix) {
    REQUIRE(k == 3);
    REQUIRE(std::string(std::getenv("VT_ASYNC_RUNNER") ? std::getenv("VT_ASYNC_RUNNER") : "") == "0");
    state_probe = std::make_unique<xpu_test::Queue>(vt::DeviceType::kXPU);
  }
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 256;
  params.max_num_batched_tokens = 128;
  params.max_num_seqs = 4;
  params.num_blocks = 32;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  if (k) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  vt::ResetGraphBreakStats();
  auto& engine = loaded->engine();
  const std::array<std::string, 4> prompts{
      "A gardener plants 12 tomato plants in each of 3 rows. How many tomato plants are there? Explain briefly.",
      "What is 7 + 5? Answer briefly.",
      "Write a short story about a fox that finds a red umbrella beside a quiet river.",
      "Explain why tomato plants need sunlight. Use two simple sentences."};
  // Full concurrency must last beyond the two slot warmups. Both control
  // arms use these same caps; shorter historical R08 limits remain default.
  const std::array<int, 4> limits = graph_screen ? std::array<int, 4>{16, 20, 24, 31}
                                               : std::array<int, 4>{8, 12, 16, 23};
  nlohmann::json result = {{"depth", k}, {"graph_requested", graph_requested},
      {"graph_screen", graph_screen}, {"groups", nlohmann::json::array()}};
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  for (int concurrency : {4, 2}) {
    nlohmann::json group = {{"concurrency", concurrency}, {"steps", nlohmann::json::array()},
                            {"requests", nlohmann::json::object()}};
    result["groups"].push_back(group);
    auto& current = result["groups"].back();
    const auto proposed = loaded->runner().spec_drafts_proposed();
    for (int req = 0; req < concurrency; ++req) {
      vllm::SamplingParams sampling;
      sampling.temperature = 0;
      sampling.max_tokens = limits[req];
      sampling.ignore_eos = graph_screen;
      sampling.output_kind = vllm::RequestOutputKind::kCumulative;
      engine.add_request("c" + std::to_string(concurrency) + "-req-" + std::to_string(req),
                          prompts[req], sampling);
    }
    bool witnessed_batch = false, witnessed_full_verify = false, witnessed_graph_batch = false;
    save();
    for (int step = 0; engine.has_unfinished_requests() && step < 128; ++step) {
      const auto before_graph = vt::GetGraphBreakStats();
      const auto outputs = engine.step();
      const auto after_graph = vt::GetGraphBreakStats();
      const auto& runner = loaded->runner();
      const auto& input = runner.last_step();
      const int count = runner.last_forward_num_reqs();
      std::vector<std::string> ids;
      for (int row = 0; row < count; ++row) {
        REQUIRE(runner.input_batch().req_ids[row].has_value());
        ids.push_back(*runner.input_batch().req_ids[row]);
      }
      current["steps"].push_back({{"requests", ids}, {"num_reqs", count},
          {"logical_forward_tokens", runner.last_forward_num_actual_tokens()},
          {"logical_logit_rows", runner.last_forward_rows()},
          {"qsl", input.query_start_loc}, {"seq_lens", input.seq_lens},
          {"cu_num_logits", input.cu_num_logits},
          {"drafts_per_req", input.num_draft_tokens_per_req},
          {"graph_captures", after_graph.segments_captured},
          {"graph_replays", after_graph.replays},
          {"graph_replayed_this_step", after_graph.replays > before_graph.replays}});
      witnessed_batch |= count == concurrency;
      witnessed_full_verify |= count == concurrency &&
          runner.last_forward_rows() == concurrency * 4;
      witnessed_graph_batch |= count == concurrency &&
          runner.last_forward_rows() == concurrency * 4 &&
          after_graph.replays > before_graph.replays;
      for (const auto& request : outputs) if (request.finished) {
        REQUIRE(request.outputs.size() == 1);
        const auto& completion = request.outputs[0];
        current["requests"][request.request_id] = {
            {"ids", completion.token_ids}, {"text", completion.text},
            {"finish_reason", completion.finish_reason.value_or("")}};
      }
      save();
    }
    CHECK_FALSE(engine.has_unfinished_requests());
    CHECK(witnessed_batch);
    current["witnessed_genuine_batch"] = witnessed_batch;
    current["witnessed_full_verification"] = witnessed_full_verify;
    current["witnessed_full_batch_graph_replay"] = witnessed_graph_batch;
    current["proposed"] = loaded->runner().spec_drafts_proposed() - proposed;
    if (state_prefix) {
      REQUIRE(loaded->runner().gdn_state_slots() == 16);
      // Snapshot every allocated recurrent row, including completed and spare
      // owners, after the native sampler has completed the producer queue.
      current["gdn_states"] = SnapshotGdnStates(loaded->runner().gdn_state(),
          state_probe->q, state_prefix, state_baseline, concurrency);
    }
    save();
    CHECK(current["requests"].size() == static_cast<size_t>(concurrency));
    if (k) {
      CHECK(witnessed_full_verify);
      CHECK(loaded->runner().spec_drafts_proposed() > proposed);
    }
    if (graph_requested) CHECK(witnessed_graph_batch);
    else CHECK(vt::GetGraphBreakStats().replays == 0);
    for (int req = 0; req < concurrency; ++req) {
      const auto& ids = current["requests"].at("c" + std::to_string(concurrency) + "-req-" + std::to_string(req)).at("ids");
      CHECK_FALSE(ids.empty());
      CHECK(ids.size() <= static_cast<size_t>(limits[req]));
    }
  }
  if (const char* baseline = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(baseline); REQUIRE(file.good());
    const auto reference = nlohmann::json::parse(file);
    for (size_t group = 0; group < result["groups"].size(); ++group)
      for (const auto& [id, request] : result["groups"][group]["requests"].items()) {
        CAPTURE(id);
        CHECK(request["ids"] == reference["groups"][group]["requests"][id]["ids"]);
      }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine R09 capacity: single native long request and resources") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* length = std::getenv("VT_B70_EXL3_ENGINE_CAPACITY_PROMPT");
  if (!model || !output || !length) std::exit(77);
  const int prompt_tokens = std::stoi(length);
  REQUIRE((prompt_tokens == 131072 || prompt_tokens == 261120));
  const int output_tokens = prompt_tokens == 261120 ? 1024 : 8;
  REQUIRE_FALSE(std::filesystem::exists(output));
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 262144;
  params.max_num_batched_tokens = 1600;
  params.max_num_seqs = 4;  // account for four recurrent owners, one long request
  params.num_blocks = 180;  // target/draft storage plus bounded state identities
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = true;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  const auto& kv = loaded->kv_cache_config();
  REQUIRE(kv.mtp_draft_shares_target_pages);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  CHECK(kv.kv_cache_groups[0].layer_names.size() == 17);
  CHECK(vllm::v1::KVBytesPerBlock(kv) == 55705600);
  std::string passage;
  for (int repeat = 0; repeat < prompt_tokens / 16 + 32; ++repeat)
    passage += "A fox found a red umbrella beside a quiet river. The rain stopped, and the fox carried it home. ";
  auto prompt = loaded->tokenizer().Encode(passage);
  REQUIRE(prompt.size() >= size_t(prompt_tokens));
  prompt.resize(prompt_tokens);
  nlohmann::json result = {{"prompt_ids", prompt}, {"prompt_tokens", prompt_tokens},
      {"requested_output_tokens", output_tokens}, {"max_model_len", params.max_model_len},
      {"max_num_seqs", params.max_num_seqs}, {"num_blocks", params.num_blocks},
      {"depth", 3}, {"prefix_caching", true}, {"steps", nlohmann::json::array()}};
  const auto memory = [&] {
    const auto info = vt::xpu::GetMemoryInfo();
    struct rusage usage{};
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
    const uint64_t rss = uint64_t(usage.ru_maxrss) * 1024;
    REQUIRE(info.peak_allocated_bytes <= (uint64_t(32) << 30));
    REQUIRE(rss <= (uint64_t(32) << 30));
    return nlohmann::json{{"backend_live_device_bytes", info.allocated_bytes},
        {"backend_peak_device_bytes", info.peak_allocated_bytes},
        {"graph_device_bytes", info.graph_device_bytes}, {"pinned_host_bytes", info.pinned_bytes},
        {"driver_free_known", info.free_known}, {"driver_free_bytes", info.free_bytes},
        {"exl3_workspace_bytes", info.exl3_workspace_bytes},
        {"gdn_workspace_bytes", info.gdn_workspace_bytes},
        {"native_gdn_workspace_bytes", info.native_gdn_workspace_bytes},
        {"attention_workspace_bytes", info.attention_workspace_bytes}, {"host_peak_rss_bytes", rss}};
  };
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  result["loaded_memory"] = memory(); save();
  vllm::SamplingParams sampling;
  sampling.temperature = 0;
  sampling.max_tokens = output_tokens;
  sampling.ignore_eos = true;  // exact output-count capacity, not EOS/quality qualification
  sampling.output_kind = vllm::RequestOutputKind::kCumulative;
  auto& engine = loaded->engine();
  engine.add_request("capacity", prompt, sampling);
  int emitted = 0;
  for (int step = 0; engine.has_unfinished_requests() &&
                     step < (prompt_tokens + 1599) / 1600 + output_tokens + 32; ++step) {
    const auto outputs = engine.step();
    const auto& input = loaded->runner().last_step();
    REQUIRE(loaded->runner().last_forward_num_reqs() == 1);
    REQUIRE(loaded->runner().input_batch().req_ids.front().has_value());
    CHECK(*loaded->runner().input_batch().req_ids.front() == "capacity");
    REQUIRE_FALSE(input.positions.empty());
    CHECK(input.positions.back() < params.max_model_len);
    for (const auto& request : outputs) {
      CHECK(request.request_id == "capacity");
      REQUIRE(request.outputs.size() == 1);
      emitted = static_cast<int>(request.outputs[0].token_ids.size());
      if (request.finished) {
        result["ids"] = request.outputs[0].token_ids;
        result["text"] = request.outputs[0].text;
      }
    }
    result["steps"].push_back({{"positions_begin", input.positions.front()},
        {"positions_end", input.positions.back()}, {"seq_lens", input.seq_lens},
        {"qsl", input.query_start_loc}, {"logical_logit_rows", loaded->runner().last_forward_rows()},
        {"emitted_tokens", emitted}, {"memory", memory()}});
    save();
  }
  REQUIRE_FALSE(engine.has_unfinished_requests());
  REQUIRE(result.contains("ids"));
  CHECK(result["ids"].size() == size_t(output_tokens));
  CHECK(result["steps"][0]["positions_begin"] == 0);
  CHECK(loaded->runner().spec_drafts_proposed() > 0);
  result["spec_drafts_proposed"] = loaded->runner().spec_drafts_proposed();
  result["after_request_memory"] = memory();
  CHECK(vt::GetReferenceTierHits() == 0);
  loaded.reset();
  result["after_engine_release_memory"] = memory();
  save();
}

TEST_CASE("XPU EXL3 public engine R09 prefix: joint native MTP3 cold warm state") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* prefix = std::getenv("VT_B70_EXL3_ENGINE_PREFIX");
  if (!model || !output || !prefix) std::exit(77);
  const bool caching = std::stoi(prefix) != 0;
  const char* depth_setting = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  const int depth = depth_setting ? std::stoi(depth_setting) : 3;
  REQUIRE((depth == 0 || depth == 3));
  const char* state_output = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_STATE_OUTPUT");
  const char* state_baseline = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_STATE_BASELINE");
  REQUIRE((!state_baseline || state_output));
  // Optional completed-step evidence only; never overlaps model execution.
  std::unique_ptr<xpu_test::Queue> probe;
  if (state_output) probe = std::make_unique<xpu_test::Queue>(vt::DeviceType::kXPU);
  const char* graph_setting = std::getenv("VT_B70_EXL3_ENGINE_GRAPH");
  const bool graph_requested = graph_setting && std::stoi(graph_setting) == 1;
  const bool graph_screen = std::getenv("VT_B70_EXL3_ENGINE_GRAPH_SCREEN") != nullptr;
  const char* output_setting = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_OUTPUT_TOKENS");
  const int output_tokens = output_setting ? std::stoi(output_setting) : (graph_screen ? 16 : 8);
  REQUIRE((output_tokens >= 1 && output_tokens <= 64));
  const bool pair_only = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_PAIR_ONLY") != nullptr;
  const bool lifecycle = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_LIFECYCLE") != nullptr;
  const char* prompt_length = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_PROMPT");
  const int prompt_tokens = prompt_length ? std::stoi(prompt_length) : 3201;
  REQUIRE(prompt_tokens >= 3201);
  REQUIRE(prompt_tokens <= 65536);  // larger admission is a separate focused step
  const bool long_case = prompt_tokens > 3201;
  const bool long_lifecycle = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_LONG_LIFECYCLE") != nullptr;
  REQUIRE_FALSE((long_case && lifecycle));
  REQUIRE_FALSE((long_lifecycle && !long_case));
  REQUIRE_FALSE((pair_only && (lifecycle || long_lifecycle)));
  REQUIRE_FALSE(std::filesystem::exists(output));
  vllm::entrypoints::EngineParams params;
  params.max_model_len = std::max(4096, prompt_tokens + 64);
  params.max_num_batched_tokens = 1600;  // exact snapshot boundary per chunk
  params.max_num_seqs = 1;
  params.num_blocks = std::max(64, 3 * ((prompt_tokens + 64 + 1599) / 1600) + 16);
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = caching;
  if (pair_only) {
    REQUIRE(depth == 3);
    REQUIRE(prompt_tokens == 32768);
    REQUIRE(output_tokens == 64);
    params.max_model_len = 262144;
    params.max_num_seqs = 4;
    params.num_blocks = 180;
  }
  if (depth) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  else params.block_size = 1600;  // fixed page protocol, independent of auto sizing
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  REQUIRE(loaded->block_size() == 1600);
  REQUIRE(loaded->speculative_config().has_value() == (depth > 0));
  auto& engine = loaded->engine();
  vt::ResetGraphBreakStats();
  std::string passage;
  for (int repeat = 0; repeat < prompt_tokens / 16 + 32; ++repeat)
    passage += "A fox found a red umbrella beside a quiet river. The rain stopped, and the fox carried it home. ";
  auto tokens = loaded->tokenizer().Encode(passage);
  if (const char* ids_path = std::getenv("VT_B70_EXL3_ENGINE_PREFIX_PROMPT_IDS")) {
    std::ifstream file(ids_path); REQUIRE(file.good());
    tokens = nlohmann::json::parse(file).at("prompt_token_ids").get<std::vector<int32_t>>();
  }
  REQUIRE(tokens.size() >= size_t(prompt_tokens + (pair_only ? 0 : 16)));
  nlohmann::json result = {{"prefix_caching", caching}, {"depth", depth},
      {"lifecycle", lifecycle}, {"prompt_tokens", prompt_tokens},
      {"num_blocks", params.num_blocks}, {"output_tokens", output_tokens},
      {"graph_requested", graph_requested}, {"graph_screen", graph_screen},
      {"pair_only", pair_only}, {"max_model_len", params.max_model_len},
      {"max_num_seqs", params.max_num_seqs}, {"block_size", loaded->block_size()},
      {"requests", nlohmann::json::array()}};
  const auto memory = [&] {
    const auto info = vt::xpu::GetMemoryInfo();
    struct rusage usage{};
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
    const uint64_t rss = uint64_t(usage.ru_maxrss) * 1024;  // Linux KiB
    CHECK(info.peak_allocated_bytes <= (uint64_t(32) << 30));
    CHECK(rss <= (uint64_t(32) << 30));
    return nlohmann::json{{"backend_live_device_bytes", info.allocated_bytes},
        {"backend_peak_device_bytes", info.peak_allocated_bytes},
        {"graph_device_bytes", info.graph_device_bytes}, {"pinned_host_bytes", info.pinned_bytes},
        {"driver_free_known", info.free_known}, {"driver_free_bytes", info.free_bytes},
        {"exl3_workspace_bytes", info.exl3_workspace_bytes},
        {"gdn_workspace_bytes", info.gdn_workspace_bytes},
        {"native_gdn_workspace_bytes", info.native_gdn_workspace_bytes},
        {"attention_workspace_bytes", info.attention_workspace_bytes}, {"host_peak_rss_bytes", rss}};
  };
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  for (int req = 0; req < (pair_only ? 2 : (long_case ? (long_lifecycle ? 5 : 3) : (lifecycle ? 10 : 5))); ++req) {
    std::vector<int32_t> prompt(tokens.begin(), tokens.begin() + prompt_tokens + (req == 2 ? 16 : 0));
    if (req == 3) prompt[1600] = prompt[1600] == 0 ? 1 : 0;
    if (long_lifecycle && req == 3) prompt[0] = 6;  // distinct64K prefix evicts old snapshots/pages
    if (req == 5) prompt[2000] = prompt[2000] == 0 ? 1 : 0;
    if (req >= 6 && req <= 8) prompt[0] = req;  // three new prefix identities force LRU eviction
    vllm::SamplingParams sampling;
    sampling.temperature = 0;
    sampling.max_tokens = output_tokens;
    sampling.ignore_eos = true;
    sampling.output_kind = vllm::RequestOutputKind::kCumulative;
    const std::string name = "prefix-" + std::to_string(req);
    result["requests"].push_back({{"request_id", name}, {"prompt_ids", prompt},
        {"steps", nlohmann::json::array()}});
    auto& current = result["requests"].back();
    const auto before_request = vt::GetGraphBreakStats();
    engine.add_request(name, prompt, sampling);
    for (int step = 0; engine.has_unfinished_requests() && step < (prompt_tokens + 1599) / 1600 + output_tokens + 8; ++step) {
      const auto outputs = engine.step();
      const auto graph_stats = vt::GetGraphBreakStats();
      const auto& input = loaded->runner().last_step();
      current["steps"].push_back({{"positions_begin", input.positions.front()},
          {"seq_lens", input.seq_lens}, {"qsl", input.query_start_loc},
          {"logical_logit_rows", loaded->runner().last_forward_rows()}, {"memory", memory()},
          {"graph_captures", graph_stats.segments_captured}, {"graph_replays", graph_stats.replays},
          {"producer_event", loaded->runner().last_spec_hidden().producer_ready_event != nullptr}});
      if (graph_setting && !graph_requested) CHECK(graph_stats.replays == 0);
      for (const auto& request : outputs) if (request.finished) {
        REQUIRE(request.outputs.size() == 1);
        current["ids"] = request.outputs[0].token_ids;
        current["text"] = request.outputs[0].text;
        CHECK(request.outputs[0].token_ids.size() == size_t(output_tokens));
      }
      save();
    }
    CHECK_FALSE(engine.has_unfinished_requests());
    REQUIRE(current.contains("ids"));
    if (state_output && req < 2) {
      current["gdn_states"] = SnapshotGdnStates(loaded->runner().gdn_state(), probe->q,
          state_output, state_baseline, req);
    }
    const auto after_request = vt::GetGraphBreakStats();
    current["graph_replays_this_request"] = after_request.replays - before_request.replays;
    if (graph_requested) CHECK(after_request.replays > before_request.replays);
    const int first = current["steps"][0]["positions_begin"];
    if (!caching || req == 0 || (long_lifecycle && req == 3)) CHECK(first == 0);
    // A mutation at position1600 invalidates later snapshots, but the first
    // unchanged page may still be reused. It must never skip the mutation.
    if (caching && req == 3 && !long_lifecycle) {
      if (depth) CHECK(first == 0);
      else CHECK((first == 0 || first == 1600));
    }
    if (caching && (req == 1 || req == 2)) CHECK(first >= 1600);
    if (pair_only && req == 1) CHECK(first == (caching ? 30400 : 0));
    if (long_lifecycle && req == 4) CHECK(first == 0);
    if (req >= 6) CHECK(first == 0);
    if (req == 5) CHECK(first == (caching && !depth ? 1600 : 0));
    if (req == 1 || req == 4 || req == 9) CHECK(current["ids"] == result["requests"][0]["ids"]);
  }
  if (lifecycle || long_lifecycle) {
    std::vector<int32_t> prompt(tokens.begin(), tokens.begin() + prompt_tokens);
    vllm::SamplingParams sampling;
    sampling.temperature = 0;
    sampling.max_tokens = output_tokens;
    sampling.ignore_eos = true;
    engine.add_request("prefix-cancel", prompt, sampling);
    const auto provisional = engine.step();
    // A cold first chunk has no completion yet; a warm one-token tail emits.
    if (caching) CHECK_FALSE(provisional.empty());
    else CHECK(provisional.empty());
    result["cancel_first_position"] = loaded->runner().last_step().positions.front();
    CHECK(result["cancel_first_position"] == (caching ? (prompt_tokens - 1) / 1600 * 1600 : 0));
    engine.abort_request("prefix-cancel");
    CHECK_FALSE(engine.has_unfinished_requests());
    engine.add_request("prefix-after-cancel", std::move(prompt), sampling);
    for (int step = 0; engine.has_unfinished_requests() && step < (prompt_tokens + 1599) / 1600 + output_tokens + 8; ++step) {
      const auto outputs = engine.step();
      for (const auto& request : outputs) {
        CHECK(request.request_id != "prefix-cancel");
        if (request.finished) {
          REQUIRE(request.outputs.size() == 1);
          result["after_cancel_ids"] = request.outputs[0].token_ids;
        }
      }
    }
    CHECK_FALSE(engine.has_unfinished_requests());
    REQUIRE(result.contains("after_cancel_ids"));
    CHECK(result["after_cancel_ids"] == result["requests"][0]["ids"]);
    result["after_cancel_memory"] = memory();
    CHECK(result["after_cancel_ids"].size() == size_t(output_tokens));
    save();
  }
  if (const char* baseline = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(baseline); REQUIRE(file.good());
    const auto reference = nlohmann::json::parse(file);
    for (size_t i = 0; i < result["requests"].size(); ++i) {
      CAPTURE(i);
      CHECK(result["requests"][i]["ids"] == reference["requests"][i]["ids"]);
    }
    if (result.contains("after_cancel_ids"))
      CHECK(result["after_cancel_ids"] == reference["after_cancel_ids"]);
  }
  const auto graph_stats = vt::GetGraphBreakStats();
  result["graph_stats"] = {{"captures", graph_stats.segments_captured}, {"replays", graph_stats.replays}};
  if (graph_requested) {
    CHECK(graph_stats.segments_captured > 0);
    CHECK(graph_stats.replays > 0);
  }
  if (depth) CHECK(loaded->runner().spec_drafts_proposed() > 0);
  else {
    CHECK(loaded->runner().spec_drafts_proposed() == 0);
    CHECK(loaded->runner().spec_drafts_accepted() == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  loaded.reset();
  result["after_engine_release_memory"] = memory();
  CHECK(result["after_engine_release_memory"]["graph_device_bytes"] == 0);
  save();
}

TEST_CASE("XPU EXL3 public engine R08 pages: simultaneous exact boundary continuation") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* depth = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  if (!model || !output || !depth) std::exit(77);
  const int k = std::stoi(depth);
  REQUIRE((k == 0 || k == 3));
  REQUIRE_FALSE(std::filesystem::exists(output));
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 1664;
  params.max_num_batched_tokens = 4096;
  params.max_num_seqs = 2;
  params.num_blocks = 32;  // two pages/request plus target/draft/spec reservations
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  if (!k) params.block_size = 1600;  // keep the declared boundary protocol
  if (k) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  REQUIRE(loaded->block_size() == 1600);
  auto& engine = loaded->engine();
  const std::array<std::string, 2> passages{
      "A fox found a red umbrella beside a quiet river. The rain stopped, and the fox carried it home. ",
      "The gardener planted tomatoes in rows. Every morning she watered the soil and counted the new leaves. "};
  nlohmann::json result = {{"depth", k}, {"block_size", 1600}, {"groups", nlohmann::json::array()}};
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  const std::array<std::array<int, 2>, 2> lengths{{{1599, 1601}, {1600, 1599}}};
  for (int group = 0; group < 2; ++group) {
    result["groups"].push_back({{"prompt_lengths", lengths[group]},
        {"prompt_ids", nlohmann::json::object()}, {"steps", nlohmann::json::array()},
        {"requests", nlohmann::json::object()}});
    auto& current = result["groups"].back();
    for (int req = 0; req < 2; ++req) {
      std::string text;
      for (int repeat = 0; repeat < 100; ++repeat) text += passages[req];
      auto ids = loaded->tokenizer().Encode(text);
      REQUIRE(ids.size() >= size_t(lengths[group][req]));
      ids.resize(lengths[group][req]);
      const std::string name = "page-" + std::to_string(group) + "-" + std::to_string(req);
      current["prompt_ids"][name] = ids;
      vllm::SamplingParams sampling;
      sampling.temperature = 0;
      sampling.max_tokens = 8;
      sampling.ignore_eos = true;
      sampling.output_kind = vllm::RequestOutputKind::kCumulative;
      engine.add_request(name, std::move(ids), sampling);
    }
    bool witnessed_verify = false;
    std::set<int64_t> seen_boundary_positions;
    for (int step = 0; engine.has_unfinished_requests() && step < 24; ++step) {
      const auto outputs = engine.step();
      const auto& runner = loaded->runner();
      const auto& input = runner.last_step();
      const auto& attention = runner.last_attn_meta();
      const int count = runner.last_forward_num_reqs();
      std::vector<std::string> names;
      for (int row = 0; row < count; ++row) names.push_back(*runner.input_batch().req_ids[row]);
      current["steps"].push_back({{"requests", names}, {"qsl", input.query_start_loc},
          {"seq_lens", input.seq_lens}, {"positions", input.positions},
          {"logical_logit_rows", runner.last_forward_rows()},
          {"block_table", attention.block_table_tensor}, {"block_cols", attention.block_table_num_cols},
          {"slot_mapping", attention.slot_mapping}});
      std::set<int64_t> unique_slots;
      for (int row = 0; row < count; ++row) {
        for (int token = input.query_start_loc[row]; token < input.query_start_loc[row + 1]; ++token) {
          const int64_t position = input.positions[token];
          const auto block = attention.block_table_tensor.at(
              row * attention.block_table_num_cols + position / 1600);
          CHECK(attention.slot_mapping.at(token) == block * int64_t(1600) + position % 1600);
          CHECK(unique_slots.insert(attention.slot_mapping[token]).second);
          if (position == 1599 || position == 1600) seen_boundary_positions.insert(position);
        }
      }
      witnessed_verify |= count == 2 && runner.last_forward_rows() == 8;
      for (const auto& request : outputs) if (request.finished) {
        REQUIRE(request.outputs.size() == 1);
        const auto& completion = request.outputs[0];
        current["requests"][request.request_id] = {{"ids", completion.token_ids},
            {"text", completion.text}, {"finish_reason", completion.finish_reason.value_or("")}};
        CHECK(completion.token_ids.size() == 8);
        CHECK(completion.finish_reason == "length");
      }
      save();
    }
    CHECK_FALSE(engine.has_unfinished_requests());
    CHECK(current["requests"].size() == 2);
    CHECK(seen_boundary_positions == std::set<int64_t>{1599, 1600});
    if (k) CHECK(witnessed_verify);
    current["witnessed_C2_verify"] = witnessed_verify;
    current["boundary_positions"] = seen_boundary_positions;
    save();
  }
  if (const char* baseline = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(baseline); REQUIRE(file.good());
    const auto reference = nlohmann::json::parse(file);
    for (size_t group = 0; group < result["groups"].size(); ++group)
      for (const auto& [id, request] : result["groups"][group]["requests"].items()) {
        CAPTURE(id);
        CHECK(request["ids"] == reference["groups"][group]["requests"].at(id)["ids"]);
      }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine R08 lifecycle: continuous turnover and state isolation") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  const char* depth = std::getenv("VT_B70_EXL3_ENGINE_DEPTH");
  if (!model || !output || !depth) std::exit(77);
  const int k = std::stoi(depth);
  REQUIRE((k == 0 || k == 3));
  REQUIRE_FALSE(std::filesystem::exists(output));
  // Independent copy queue used only at completed synchronous engine steps.
  // It outlives the engine and never operates concurrently with model work.
  xpu_test::Queue probe(vt::DeviceType::kXPU);
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 256;
  params.max_num_batched_tokens = 128;
  params.max_num_seqs = 4;
  params.num_blocks = 32;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  if (k) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  auto& engine = loaded->engine();
  const char* graph_setting = std::getenv("VT_B70_EXL3_ENGINE_GRAPH");
  const bool graph_requested = graph_setting && std::stoi(graph_setting) == 1;
  vt::ResetGraphBreakStats();
  auto& backend = vt::GetBackend(probe.q.device);
  const std::array<std::string, 5> prompts{
      "Write a short story about a fox that finds a red umbrella beside a quiet river.",
      "What is 7 + 5? Answer briefly.",
      "A gardener plants 12 tomato plants in each of 3 rows. How many tomato plants are there? Explain briefly.",
      "Explain why tomato plants need sunlight. Use two simple sentences.",
      "Hello"};
  const std::array<int, 5> limits{48, 12, 13, 40, 17};
  const std::array<std::string, 5> names{"long", "early-eos", "medium", "cancel", "replacement"};
  const auto add = [&](int index) {
    vllm::SamplingParams sampling;
    sampling.temperature = 0;
    sampling.max_tokens = limits[index];
    sampling.output_kind = vllm::RequestOutputKind::kCumulative;
    engine.add_request(names[index], prompts[index], sampling);
  };
  nlohmann::json result = {{"depth", k}, {"steps", nlohmann::json::array()},
      {"requests", nlohmann::json::object()}, {"cancelled_id", names[3]},
      {"sentinel_layers", {0, static_cast<int>(loaded->runner().gdn_state().size()) - 1}}};
  result["graph_requested"] = graph_requested;
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  struct Sentinel { void* data; std::vector<uint8_t> bytes; };
  std::vector<Sentinel> sentinels;
  const auto poison_spare_slots = [&](int active_base) {
    sentinels.clear();
    const auto& states = loaded->runner().gdn_state();
    REQUIRE_FALSE(states.empty());
    for (size_t layer : {size_t(0), states.size() - 1}) {
      for (const auto& state : {states[layer].conv_state, states[layer].ssm_state}) {
        const size_t bytes = size_t(state.stride[0]) * vt::SizeOf(state.dtype);
        REQUIRE(bytes > 0);
        for (int slot = 0; slot < loaded->runner().gdn_state_slots(); ++slot) {
          // Compact slot zero is a real owner, not a null/sentinel slot.
          if (slot >= active_base && slot < active_base + k + 1) continue;
          auto* destination = static_cast<uint8_t*>(state.data) + slot * bytes;
          sentinels.push_back({destination, std::vector<uint8_t>(bytes, uint8_t(0x29 + slot))});
          backend.Copy(probe.q, destination, sentinels.back().bytes.data(), bytes);
        }
      }
    }
    backend.Synchronize(probe.q);
    REQUIRE_FALSE(sentinels.empty());
  };
  const auto check_sentinels = [&] {
    for (const auto& sentinel : sentinels) {
      std::vector<uint8_t> actual(sentinel.bytes.size());
      backend.Copy(probe.q, actual.data(), sentinel.data, actual.size());
      backend.Synchronize(probe.q);
      CHECK(actual == sentinel.bytes);
    }
  };
  std::map<std::string, int> stable_bases;
  bool cancelled = false, saw_mixed = false, protected_tail = false;
  int cancel_base = -1, replacement_base = -1;
  int transition = 0;
  const std::array<int, 4> desired{1, 4, 2, 1};
  add(0);
  for (int step = 0; engine.has_unfinished_requests() && step < 96; ++step) {
    const auto before_graph = vt::GetGraphBreakStats();
    const auto outputs = engine.step();
    const auto graph_stats = vt::GetGraphBreakStats();
    const auto& runner = loaded->runner();
    const auto& input = runner.last_step();
    const auto& gdn = runner.last_gdn_meta();
    const int count = runner.last_forward_num_reqs();
    if (transition < 4 && count == desired[transition]) ++transition;
    std::vector<std::string> ids;
    std::map<std::string, int> bases;
    size_t spec_row = 0, ordinary_row = 0;
    for (int row = 0; row < count; ++row) {
      REQUIRE(runner.input_batch().req_ids[row].has_value());
      const auto& id = *runner.input_batch().req_ids[row];
      ids.push_back(id);
      int base;
      if (gdn.spec_sequence_masks && (*gdn.spec_sequence_masks)[row]) {
        REQUIRE(gdn.spec_state_indices_tensor.has_value());
        base = gdn.spec_state_indices_tensor->at(spec_row++ * (k + 1));
      } else {
        REQUIRE(gdn.non_spec_state_indices_tensor.has_value());
        base = gdn.non_spec_state_indices_tensor->at(ordinary_row++);
      }
      bases[id] = base;
      if (stable_bases.contains(id)) CHECK(stable_bases.at(id) == base);
      else stable_bases[id] = base;
      if (id == names[3]) cancel_base = base;
      if (id == names[4]) replacement_base = base;
      if (cancelled) CHECK(id != names[3]);
    }
    std::set<int> unique;
    for (const auto& [id, base] : bases) CHECK(unique.insert(base).second);
    saw_mixed |= gdn.num_spec_decodes > 0 && gdn.num_prefills > 0;
    result["steps"].push_back({{"requests", ids}, {"num_reqs", count}, {"state_bases", bases},
        {"logical_forward_tokens", runner.last_forward_num_actual_tokens()},
        {"logical_logit_rows", runner.last_forward_rows()}, {"qsl", input.query_start_loc},
        {"seq_lens", input.seq_lens}, {"drafts_per_req", input.num_draft_tokens_per_req},
        {"spec_decodes", gdn.num_spec_decodes}, {"prefills", gdn.num_prefills},
        {"previous_accepted_tokens", gdn.num_accepted_tokens.value_or(std::vector<int32_t>{})}});
    result["steps"].back()["graph_stats"] = {
        {"captures", graph_stats.segments_captured}, {"replays", graph_stats.replays}};
    if (gdn.num_spec_decodes > 0 && gdn.num_prefills > 0) {
      CHECK(runner.last_spec_hidden().producer_ready_event == nullptr);
      CHECK(graph_stats.replays == before_graph.replays);
      CHECK(graph_stats.segments_captured == before_graph.segments_captured);
    }
    if (graph_setting && !graph_requested) CHECK(graph_stats.replays == 0);
    for (const auto& request : outputs) {
      if (cancelled) CHECK(request.request_id != names[3]);
      if (!request.finished) continue;
      REQUIRE(request.outputs.size() == 1);
      const auto& completion = request.outputs[0];
      result["requests"][request.request_id] = {{"ids", completion.token_ids},
          {"text", completion.text}, {"finish_reason", completion.finish_reason.value_or("")}};
    }
    if (protected_tail) check_sentinels();
    if (step == 0) {
      CHECK(count == 1);
      // Poison unused slots before they are assigned to fresh requests. Their
      // native control agreement checks actual zero/reset on first use.
      poison_spare_slots(bases.at(names[0]));
      for (int req = 1; req <= 3; ++req) add(req);
    }
    if (step == 2) {
      CHECK(count == 4);
      CHECK_FALSE(result["requests"].contains(names[3]));
      engine.abort_request(names[3]);
      cancelled = true;
      add(4);
    }
    if (step > 2 && count == 1 && !protected_tail) {
      REQUIRE(ids == std::vector<std::string>{names[0]});
      poison_spare_slots(bases.at(names[0]));
      protected_tail = true;
    }
    save();
  }
  CHECK_FALSE(engine.has_unfinished_requests());
  CHECK(transition == 4);
  CHECK(cancelled);
  CHECK(protected_tail);
  CHECK(cancel_base >= 0);
  CHECK(replacement_base == cancel_base);
  CHECK(result["requests"].size() == 4);
  CHECK_FALSE(result["requests"].contains(names[3]));
  CHECK(result["requests"][names[1]]["finish_reason"] == "stop");
  if (k) {
    CHECK(saw_mixed);
    CHECK(loaded->runner().spec_drafts_proposed() > 0);
  }
  result["witnessed_ordered_1_4_2_1"] = transition == 4;
  result["witnessed_mixed_prefill_spec"] = saw_mixed;
  result["replacement_reused_cancelled_state_base"] = replacement_base == cancel_base;
  result["sentinel_tail_checked"] = protected_tail;
  // Repeat the replacement alone after other owners have left. Its generated
  // IDs must not depend on the preceding request or mixed-batch position.
  vllm::SamplingParams standalone;
  standalone.temperature = 0;
  standalone.max_tokens = limits[4];
  engine.add_request("replacement-alone", prompts[4], standalone);
  result["standalone_steps"] = nlohmann::json::array();
  bool control_finished = false;
  for (int step = 0; engine.has_unfinished_requests() && step < 32; ++step) {
    const auto outputs = engine.step();
    const auto stats = vt::GetGraphBreakStats();
    const auto& runner = loaded->runner();
    result["standalone_steps"].push_back({{"step", step},
        {"positions", runner.last_step().positions},
        {"captures", stats.segments_captured}, {"replays", stats.replays},
        {"producer_event", runner.last_spec_hidden().producer_ready_event != nullptr}});
    for (const auto& request : outputs) {
      if (!request.finished) continue;
      REQUIRE(request.outputs.size() == 1);
      control_finished = true;
      result["replacement_alone_ids"] = request.outputs[0].token_ids;
    }
  }
  REQUIRE(control_finished);
  CHECK(result["requests"][names[4]]["ids"] == result["replacement_alone_ids"]);
  // Preserve the one-token reset above. A separate genuine multi-token
  // prefill is needed to witness graph -> eager -> graph: Hello itself is
  // eligible for the single-row graph even when its owner is freshly reset.
  if (graph_setting) {
    const bool preceding_graph = vt::GetGraphBreakStats().replays > 0;
    vllm::SamplingParams transition_sampling = standalone;
    transition_sampling.max_tokens = 12;
    transition_sampling.ignore_eos = true;
    engine.add_request("transition-prefill", prompts[2], transition_sampling);
    result["transition_steps"] = nlohmann::json::array();
    bool transition_finished = false, eager_prefill = false, resumed_replay = false;
    for (int step = 0; engine.has_unfinished_requests() && step < 32; ++step) {
      const auto before = vt::GetGraphBreakStats();
      const auto outputs = engine.step();
      const auto stats = vt::GetGraphBreakStats();
      const auto& runner = loaded->runner();
      if (step == 0) {
        CHECK(runner.last_step().input_token_ids.size() > 1);
        eager_prefill = !runner.last_spec_hidden().producer_ready_event &&
            stats.replays == before.replays && stats.segments_captured == before.segments_captured;
        CHECK(eager_prefill);
      }
      resumed_replay |= stats.replays > before.replays;
      result["transition_steps"].push_back({{"step", step},
          {"positions", runner.last_step().positions},
          {"captures", stats.segments_captured}, {"replays", stats.replays},
          {"producer_event", runner.last_spec_hidden().producer_ready_event != nullptr}});
      for (const auto& request : outputs) {
        if (!request.finished) continue;
        REQUIRE(request.outputs.size() == 1);
        transition_finished = true;
        result["transition_ids"] = request.outputs[0].token_ids;
      }
    }
    REQUIRE(transition_finished);
    CHECK(result["transition_ids"].size() == 12);
    result["witnessed_graph_eager_graph"] = preceding_graph && eager_prefill && resumed_replay;
  }
  const auto stats = vt::GetGraphBreakStats();
  result["graph_stats"] = {{"captures", stats.segments_captured}, {"replays", stats.replays}};
  if (graph_requested) {
    CHECK(stats.segments_captured > 0);
    CHECK(stats.replays > 0);
    CHECK(result["witnessed_graph_eager_graph"].get<bool>());
  } else if (graph_setting) CHECK(stats.replays == 0);
  save();
  if (const char* baseline = std::getenv("VT_B70_EXL3_ENGINE_BASELINE")) {
    std::ifstream file(baseline); REQUIRE(file.good());
    const auto reference = nlohmann::json::parse(file);
    for (const auto& [id, request] : result["requests"].items()) {
      CAPTURE(id);
      CHECK(request["ids"] == reference["requests"].at(id)["ids"]);
    }
    if (result.contains("transition_ids"))
      CHECK(result["transition_ids"] == reference["transition_ids"]);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine R10: paired GPU outputs survive consumers and retirement") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  if (!model || !output) std::exit(77);
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE(vt::GraphCaptureEnabled());
  REQUIRE(std::string(std::getenv("VT_ASYNC_RUNNER") ? std::getenv("VT_ASYNC_RUNNER") : "") == "0");
  // Public MTP sampling completes its queue before step() returns. This tests
  // real GPU ownership across completed steps, not pending cancellation.
  // Both independent consumer queues outlive the engine and retained leases.
  xpu_test::Queue first_consumer(vt::DeviceType::kXPU);
  xpu_test::Queue second_consumer(vt::DeviceType::kXPU);
  const auto download = [](const vt::Tensor& tensor, vt::Queue& queue) {
    REQUIRE(tensor.IsContiguous());
    REQUIRE(tensor.device.type == vt::DeviceType::kXPU);
    std::vector<uint8_t> bytes(tensor.Bytes());
    auto& backend = vt::GetBackend(queue.device);
    backend.Copy(queue, bytes.data(), tensor.data, bytes.size());
    backend.Synchronize(queue);
    return bytes;
  };
  struct Held {
    std::string owner;
    std::shared_ptr<void> storage;
    vt::Tensor hidden, logits;
    std::vector<uint8_t> hidden_bytes, logits_bytes;
  };
  std::vector<Held> held;
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 4352;
  params.max_num_batched_tokens = 4096;
  params.max_num_seqs = 1;
  params.num_blocks = 24;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  auto& engine = loaded->engine();
  const std::string prompt =
      "A gardener plants 12 tomato plants in each of 3 rows. "
      "How many tomato plants are there? Explain briefly.";
  vllm::SamplingParams sampling;
  sampling.temperature = 0;
  sampling.max_tokens = 64;
  sampling.ignore_eos = true;
  sampling.output_kind = vllm::RequestOutputKind::kCumulative;
  const auto control = engine.generate(prompt, sampling, "lease-control");
  REQUIRE(control.finished);
  REQUIRE(control.outputs.size() == 1);
  REQUIRE(control.outputs[0].token_ids.size() == 64);
  // Reuse captured engine caches. Held generations force new destinations.
  // No fixture continuation drives the native execution.
  vt::ResetGraphBreakStats();
  engine.add_request("held-output", prompt, sampling);
  nlohmann::json result = {{"scope", "completed_native_steps_two_consumer_queues"},
      {"control_ids", control.outputs[0].token_ids}, {"steps", nlohmann::json::array()},
      {"held", nlohmann::json::array()}};
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  bool finished = false;
  size_t post_hold_steps = 0;
  for (int step = 0; engine.has_unfinished_requests() && step < 96; ++step) {
    const auto before = vt::GetGraphBreakStats();
    const auto outputs = engine.step();
    const auto stats = vt::GetGraphBreakStats();
    const auto& runner = loaded->runner();
    const auto& hidden = runner.last_spec_hidden();
    const auto& logits = runner.last_forward_logits();
    result["steps"].push_back({{"step", step}, {"positions", runner.last_step().positions},
        {"input_ids", runner.last_step().input_token_ids}, {"rows", logits.rows},
        {"captures", stats.segments_captured}, {"replays", stats.replays}});
    for (const auto& snapshot : held) {
      CAPTURE(snapshot.owner);
      CHECK(download(snapshot.hidden, second_consumer.q) == snapshot.hidden_bytes);
      CHECK(download(snapshot.logits, second_consumer.q) == snapshot.logits_bytes);
      if (hidden.storage) CHECK(hidden.tensor.data != snapshot.hidden.data);
      if (logits.device_storage) CHECK(logits.device_tensor.data != snapshot.logits.data);
    }
    if (held.size() == 2) ++post_hold_steps;
    if (held.size() < 2 && hidden.producer_ready_event && logits.rows == 4 &&
        (stats.replays > before.replays || stats.segments_captured > before.segments_captured)) {
      REQUIRE(hidden.storage != nullptr);
      REQUIRE(logits.device_storage != nullptr);
      CHECK_FALSE(logits.non_owning_view);
      CHECK_FALSE(hidden.storage.owner_before(logits.device_storage));
      CHECK_FALSE(logits.device_storage.owner_before(hidden.storage));
      CHECK(hidden.storage.get() == hidden.tensor.data);
      CHECK(logits.device_storage.get() == logits.device_tensor.data);
      CHECK(hidden.tensor.dtype == vt::DType::kF16);
      CHECK(hidden.tensor.shape[0] == 4);
      CHECK(hidden.tensor.shape[1] == 5120);
      CHECK(logits.device_tensor.shape[1] == 248320);
      hidden.WaitReady(first_consumer.q);
      Held snapshot;
      snapshot.owner = held.empty() ? "hidden_only" : "logits_only";
      snapshot.storage = held.empty() ? hidden.storage : logits.device_storage;
      snapshot.hidden = hidden.tensor;
      snapshot.logits = logits.device_tensor;
      snapshot.hidden_bytes = download(hidden.tensor, first_consumer.q);
      snapshot.logits_bytes = download(logits.device_tensor, first_consumer.q);
      result["held"].push_back({{"owner", snapshot.owner}, {"step", step},
          {"hidden_bytes", snapshot.hidden_bytes.size()}, {"logits_bytes", snapshot.logits_bytes.size()}});
      held.push_back(std::move(snapshot));
    }
    for (const auto& request : outputs) {
      if (!request.finished) continue;
      REQUIRE(request.outputs.size() == 1);
      finished = true;
      result["held_ids"] = request.outputs[0].token_ids;
      CHECK(request.outputs[0].token_ids == control.outputs[0].token_ids);
    }
    save();
  }
  CHECK(finished);
  CHECK_FALSE(engine.has_unfinished_requests());
  REQUIRE(held.size() == 2);
  CHECK(post_hold_steps >= 3);
  const auto final_stats = vt::GetGraphBreakStats();
  CHECK(final_stats.segments_captured >= 2);  // real retirement/re-capture
  CHECK(final_stats.replays > 0);
  result["captures"] = final_stats.segments_captured;
  result["replays"] = final_stats.replays;
  loaded.reset();  // destroys graphs, weights, caches and producer queue
  for (const auto& snapshot : held) {
    CAPTURE(snapshot.owner);
    CHECK(download(snapshot.hidden, second_consumer.q) == snapshot.hidden_bytes);
    CHECK(download(snapshot.logits, second_consumer.q) == snapshot.logits_bytes);
  }
  result["preserved_after_engine_destruction"] = true;
  held.clear();  // release buffers/events before consumer queue teardown
  CHECK(vt::GetReferenceTierHits() == 0);
  save();
}

TEST_CASE("XPU EXL3 runner R10: pending cancellation and paired consumer lifetime") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  if (!model || !output) std::exit(77);
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE(vt::GraphCaptureEnabled());
  REQUIRE(std::string(std::getenv("VT_ASYNC_RUNNER") ? std::getenv("VT_ASYNC_RUNNER") : "") == "0");
  // Real production runner/scheduler split, with native sampling/feedback
  // until cancellation. The loader's unused runner never executes the model.
  // Producer/consumer queues outlive all borrowers of the loaded weights.
  xpu_test::Queue producer(vt::DeviceType::kXPU);
  xpu_test::Queue consumer(vt::DeviceType::kXPU);
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 256;
  params.max_num_batched_tokens = 128;
  params.max_num_seqs = 1;
  params.num_blocks = 24;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = false;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  const auto spec = loaded->speculative_config();
  REQUIRE(spec.has_value());
  auto draft = loaded->loaded_model().BuildMtpDraft(loaded->config());
  REQUIRE(draft != nullptr);
  vllm::SchedulerConfig scheduler_config;
  scheduler_config.max_num_seqs = 1;
  scheduler_config.max_model_len = 256;
  scheduler_config.max_num_batched_tokens = 128;
  scheduler_config.enable_chunked_prefill = true;
  scheduler_config.watermark = 0.;
  vllm::v1::Scheduler scheduler(scheduler_config, loaded->kv_cache_config(),
      loaded->block_size(), false, nullptr, spec);
  auto runner = std::make_unique<vllm::v1::GPUModelRunner>(
      loaded->config(), loaded->loaded_model(), loaded->kv_cache_config(),
      producer.q, 1, 256, 128, spec, std::move(draft));
  const auto prompt = loaded->tokenizer().Encode(
      "A gardener plants 12 tomato plants in each of 3 rows. "
      "How many tomato plants are there? Explain briefly.");
  REQUIRE_FALSE(prompt.empty());
  const auto add = [&](const std::string& id, int limit) {
    vllm::SamplingParams sampling;
    sampling.temperature = 0;
    sampling.max_tokens = limit;
    sampling.ignore_eos = true;
    scheduler.add_request(std::make_unique<vllm::v1::Request>(id, prompt, sampling, 0.));
  };
  const auto complete = [&](const vllm::v1::SchedulerOutput& scheduled) {
    const auto sampled = runner->sample_tokens(std::nullopt);
    auto updated = scheduler.update_from_output(scheduled, sampled);
    if (auto drafts = runner->take_draft_token_ids()) scheduler.update_draft_token_ids(*drafts);
    return updated;
  };
  const auto generate = [&](const std::string& id) {
    std::vector<int32_t> ids;
    for (int step = 0; scheduler.get_num_unfinished_requests() && step < 48; ++step) {
      auto scheduled = scheduler.schedule();
      REQUIRE(scheduled.total_num_scheduled_tokens > 0);
      REQUIRE_FALSE(runner->execute_model(scheduled).has_value());
      const auto updated = complete(scheduled);
      for (const auto& request : updated.outputs) {
        CHECK(request.request_id == id);
        ids.insert(ids.end(), request.new_token_ids.begin(), request.new_token_ids.end());
      }
    }
    CHECK(scheduler.get_num_unfinished_requests() == 0);
    return ids;
  };
  add("pending-control", 16);
  const auto control = generate("pending-control");
  REQUIRE(control.size() == 16);
  add("pending-cancel", 64);
  for (int step = 0; step < 6; ++step) {
    auto scheduled = scheduler.schedule();
    REQUIRE(scheduled.total_num_scheduled_tokens > 0);
    REQUIRE_FALSE(runner->execute_model(scheduled).has_value());
    complete(scheduled);
  }
  REQUIRE(vt::GetGraphBreakStats().replays > 0);
  // Allocate the consumer's private destinations before submitting the forward.
  // Device-to-device copies are asynchronous; ordinary host copies would wait.
  xpu_test::Buffer hidden_copy(consumer.q, vt::DType::kF16, {4, 5120});
  xpu_test::Buffer logits_copy(consumer.q, vt::DType::kF32, {4, 248320});
  auto& backend = vt::GetBackend(producer.q.device);
  auto consumer_done = std::shared_ptr<vt::Event>(
      new vt::Event(backend.CreateEvent(true)), [&](vt::Event* event) {
        backend.DestroyEvent(*event); delete event;
      });
  auto scheduled = scheduler.schedule();
  REQUIRE(scheduled.total_num_scheduled_tokens == 4);
  const auto before = vt::GetGraphBreakStats();
  REQUIRE_FALSE(runner->execute_model(scheduled).has_value());
  const auto stats = vt::GetGraphBreakStats();
  CHECK(stats.replays > before.replays);
  auto hidden = runner->last_spec_hidden();
  auto logits = runner->last_forward_logits();
  REQUIRE(hidden.producer_ready_event != nullptr);
  REQUIRE(hidden.storage != nullptr);
  REQUIRE(logits.device_storage != nullptr);
  REQUIRE(hidden.tensor.Bytes() == hidden_copy.bytes);
  REQUIRE(logits.device_tensor.Bytes() == logits_copy.bytes);
  const bool producer_pending = !backend.QueryEvent(*hidden.producer_ready_event);
  CHECK(producer_pending);  // required witness; a completed call is no substitute
  hidden.WaitReady(consumer.q);
  backend.Copy(consumer.q, hidden_copy.tensor.data, hidden.tensor.data, hidden_copy.bytes);
  backend.Copy(consumer.q, logits_copy.tensor.data, logits.device_tensor.data, logits_copy.bytes);
  backend.RecordEvent(*consumer_done, consumer.q);
  scheduler.finish_requests("pending-cancel", vllm::v1::RequestStatus::kFinishedAborted);
  const bool still_pending_at_abort = !backend.QueryEvent(*hidden.producer_ready_event);
  CHECK(still_pending_at_abort);
  CHECK(scheduler.get_num_unfinished_requests() == 0);
  // Abandon this forward without sampling/proposing. The next real execute
  // consumes finished_req_ids, resets the freed recurrent owner, then prefill
  // reuses the KV allocation. Retained outputs force fresh graph destinations.
  add("pending-replacement", 16);
  const auto replacement = generate("pending-replacement");
  CHECK(replacement == control);
  backend.SynchronizeEvent(*consumer_done);
  const auto expected_hidden = hidden_copy.download();
  const auto expected_logits = logits_copy.download();
  const auto download = [&](const vt::Tensor& tensor) {
    std::vector<unsigned char> result(tensor.Bytes());
    backend.Copy(consumer.q, result.data(), tensor.data, result.size());
    backend.Synchronize(consumer.q);
    return result;
  };
  CHECK(download(hidden.tensor) == expected_hidden);
  CHECK(download(logits.device_tensor) == expected_logits);
  runner.reset();  // drains pending producers before releasing runner caches
  loaded.reset();  // destroys graph cache/weights; leased outputs remain valid
  CHECK(download(hidden.tensor) == expected_hidden);
  CHECK(download(logits.device_tensor) == expected_logits);
  hidden = {}; logits = {};
  nlohmann::json result = {{"scope", "real_runner_execute_sample_split_pending_cancel"},
      {"producer_pending_before_cancel", producer_pending},
      {"producer_pending_after_abort", still_pending_at_abort},
      {"consumer_uses_producer_event", true}, {"no_sampling_of_cancelled_forward", true},
      {"control_ids", control}, {"replacement_ids", replacement},
      {"hidden_bytes", hidden_copy.bytes}, {"logits_bytes", logits_copy.bytes},
      {"pending_forward_replay_delta", stats.replays - before.replays}};
  std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 public engine R11: held-out capability task manifest") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* tasks_path = std::getenv("VT_B70_EXL3_CAPABILITY_TASKS");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  if (!model || !tasks_path || !output) std::exit(77);
  REQUIRE_FALSE(std::filesystem::exists(output));
  std::ifstream tasks_file(tasks_path);
  REQUIRE(tasks_file.good());
  const auto manifest = nlohmann::json::parse(tasks_file);
  REQUIRE(manifest.at("schema") == "b70-exl3-r11-capability-v1");
  REQUIRE(manifest.at("tasks").size() == 8);
  const char* selected = std::getenv("VT_B70_EXL3_CAPABILITY_CASE");
  // One fixed launch geometry, including four recurrent owners and the full
  // supported context. This is a correctness screen; step wall times below
  // include sampling/host work and are not target-forward latency or tok/s.
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 262144;
  params.max_num_batched_tokens = 1600;
  params.max_num_seqs = 4;
  params.num_blocks = 180;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = true;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  const auto& tokenizer = loaded->tokenizer();
  const auto template_text = vllm::entrypoints::LoadChatTemplateFromConfig(
      (std::filesystem::path(model) / "tokenizer_config.json").string());
  const std::string bos = tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : "";
  const std::string eos = tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : "";
  const auto render = vllm::entrypoints::MakeChatTemplatePromptFn(template_text, bos, eos);
  const nlohmann::ordered_json kwargs = {{"enable_thinking", false}};
  nlohmann::json result = {{"schema", "b70-exl3-r11-native-output-v1"},
      {"launch", {{"max_model_len", 262144}, {"max_num_batched_tokens", 1600},
        {"max_num_seqs", 4}, {"num_blocks", 180}, {"kv_cache_dtype", "fp8"},
        {"prefix_caching", true}, {"mtp_depth", 3}}},
      {"chat_template_kwargs", kwargs}, {"tasks", nlohmann::json::array()}};
  const auto save = [&] {
    std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
  };
  bool found = false;
  for (const auto& task : manifest.at("tasks")) {
    const std::string id = task.at("id");
    if (selected && id != selected) continue;
    found = true;
    CAPTURE(id);
    const std::vector<vllm::entrypoints::openai::ChatMessage> messages = {
        {"system", manifest.at("system").get<std::string>()},
        {"user", task.at("prompt").get<std::string>()}};
    const auto prompt = tokenizer.EncodeWithSpecialTokens(render(messages, true, {}, kwargs));
    REQUIRE(prompt.size() >= task.value("min_prompt_tokens", size_t(1)));
    REQUIRE(prompt.size() <= task.value("max_prompt_tokens", size_t(65536)));
    const int max_tokens = task.at("max_tokens");
    REQUIRE(max_tokens > 0);
    vllm::SamplingParams sampling;
    sampling.temperature = 0.;
    sampling.max_tokens = max_tokens;
    sampling.output_kind = vllm::RequestOutputKind::kCumulative;
    if (task.contains("json_schema")) {
      vllm::StructuredOutputsParams structured;
      structured.json = task.at("json_schema").dump();
      sampling.structured_outputs = structured;
    }
    auto& engine = loaded->engine();
    const auto proposed = loaded->runner().spec_drafts_proposed();
    const auto accepted = loaded->runner().spec_drafts_accepted();
    result["tasks"].push_back({{"id", id}, {"kind", task.at("kind")},
        {"prompt_ids", prompt}, {"sampling", {{"temperature", 0.}, {"max_tokens", max_tokens},
        {"ignore_eos", false}, {"structured", task.contains("json_schema")}}},
        {"steps", nlohmann::json::array()}});
    auto& current = result["tasks"].back();
    const auto began = std::chrono::steady_clock::now();
    engine.add_request(id, prompt, sampling);
    bool finished = false;
    int64_t emitted = 0;
    for (int step = 0; engine.has_unfinished_requests() &&
         step < int(prompt.size() / 1600) + max_tokens + 32; ++step) {
      const auto started = std::chrono::steady_clock::now();
      const auto outputs = engine.step();
      const auto ended = std::chrono::steady_clock::now();
      int64_t current_emitted = emitted;
      for (const auto& request : outputs) {
        REQUIRE(request.outputs.size() == 1);
        const auto& completion = request.outputs.front();
        current_emitted = completion.token_ids.size();
        if (!request.finished) continue;
        finished = true;
        current["ids"] = completion.token_ids;
        current["text"] = completion.text;
        current["finish_reason"] = completion.finish_reason.value_or("");
      }
      const double elapsed = std::chrono::duration<double>(ended - began).count();
      if (!current.contains("first_token_wall_s") && current_emitted > 0)
        current["first_token_wall_s"] = elapsed;
      current["steps"].push_back({{"step", step},
          {"step_wall_s", std::chrono::duration<double>(ended - started).count()},
          {"wall_s", elapsed}, {"new_emitted_tokens", current_emitted - emitted},
          {"positions", loaded->runner().last_step().positions},
          {"logical_forward_tokens", loaded->runner().last_forward_num_actual_tokens()},
          {"logical_logit_rows", loaded->runner().last_forward_rows()}});
      emitted = current_emitted;
      save();
    }
    REQUIRE(finished);
    REQUIRE_FALSE(engine.has_unfinished_requests());
    current["proposed"] = loaded->runner().spec_drafts_proposed() - proposed;
    current["accepted"] = loaded->runner().spec_drafts_accepted() - accepted;
    const auto mem = vt::xpu::GetMemoryInfo();
    CHECK(mem.peak_allocated_bytes <= (uint64_t(32) << 30));
    current["backend_peak_device_bytes"] = mem.peak_allocated_bytes;
    CHECK(vt::GetReferenceTierHits() == 0);
    save();
  }
  REQUIRE(found);
  loaded.reset();
  result["after_release_graph_bytes"] = vt::xpu::GetMemoryInfo().graph_device_bytes;
  CHECK(result["after_release_graph_bytes"] == 0);
  save();
}


TEST_CASE("XPU EXL3 public engine R11: frozen serving timing workload") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* workload_path = std::getenv("VT_B70_EXL3_SERVING_WORKLOAD");
  const char* selected = std::getenv("VT_B70_EXL3_SERVING_CASE");
  const char* output = std::getenv("VT_B70_EXL3_ENGINE_OUTPUT");
  if (!model || !workload_path || !selected || !output) std::exit(77);
  REQUIRE_FALSE(std::filesystem::exists(output));
  // Operator tracing is a separate diagnostic, never part of serving timing.
  REQUIRE(std::getenv("VT_SPEC_TRACE") == nullptr);
  REQUIRE(std::getenv("VT_XPU_EXL3_TRACE") == nullptr);
  REQUIRE(std::getenv("VT_OP_PROVIDER_TRACE") == nullptr);
  const bool profile = std::getenv("VT_XPU_PROFILE") &&
      std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  const bool host_profile = std::getenv("VT_XPU_HOST_PROFILE") &&
      std::string(std::getenv("VT_XPU_HOST_PROFILE")) == "1";
  std::ifstream workload_file(workload_path); REQUIRE(workload_file.good());
  const auto workload = nlohmann::json::parse(workload_file);
  REQUIRE(workload.at("schema") == "b70-exl3-r11-serving-workload-v1");
  nlohmann::json task;
  for (const auto& candidate : workload.at("cases"))
    if (candidate.at("id") == selected) task = candidate;
  REQUIRE_FALSE(task.is_null());
  REQUIRE(task.at("cache_state") == "cold");
  const int depth = task.value("mtp_depth", 3);
  REQUIRE((depth == 0 || depth == 3));
  const int context_limit = task.value("max_model_len", 262144);
  REQUIRE(context_limit > 0);
  REQUIRE(context_limit <= 262144);
  const auto& specs = task.at("requests");
  REQUIRE(specs.size() >= 1);
  REQUIRE(specs.size() <= 4);
  vllm::entrypoints::EngineParams params;
  params.max_model_len = context_limit;
  params.max_num_batched_tokens = 1600;
  params.max_num_seqs = 4;
  params.num_blocks = 180;
  if (!depth) params.block_size = 1600;  // fixed primary page, target-only isolation
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = true;
  if (depth) params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":3}");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  REQUIRE_FALSE(loaded->runner().use_async_scheduling());
  REQUIRE(loaded->speculative_config().has_value() == (depth != 0));
  REQUIRE(loaded->block_size() == 1600);
  if (profile) (void)vt::xpu::DrainProfileEvents();
  if (host_profile) (void)vt::xpu::DrainHostProfileRecords();
  const char* warmup_env = std::getenv("VT_B70_EXL3_SERVING_WARMUP");
  REQUIRE((warmup_env == nullptr || std::string(warmup_env) == "0" ||
           std::string(warmup_env) == "1"));
  const bool warmup = warmup_env && std::string(warmup_env) == "1";
  int warmup_cycles = 0;
  if (warmup) {
    // Match the pinned producer's explicit request warmup before a cold-prefix
    // measurement. Real public requests warm the selected prefill/decode/MTP
    // shapes; their tokens never enter the scored generation or timing.
    std::map<std::string, int64_t> counts;
    std::set<std::string> added, done;
    const auto add_warmup = [&] {
      for (const auto& spec : specs) {
        const std::string id = spec.at("id");
        if (added.count(id)) continue;
        if (spec.contains("after_emitted")) {
          const auto& trigger = spec.at("after_emitted");
          if (counts[trigger.at("request").get<std::string>()] <
              trigger.at("tokens").get<int64_t>()) continue;
        }
        vllm::SamplingParams sampling;
        sampling.temperature = 0.; sampling.ignore_eos = true;
        sampling.max_tokens = 32;
        sampling.output_kind = vllm::RequestOutputKind::kCumulative;
        loaded->engine().add_request("warmup-" + id,
            spec.at("prompt_ids").get<std::vector<int32_t>>(), sampling);
        added.insert(id);
      }
    };
    add_warmup();
    while (loaded->engine().has_unfinished_requests() && warmup_cycles < 4096) {
      ++warmup_cycles;
      for (const auto& request : loaded->engine().step()) {
        REQUIRE(request.outputs.size() == 1);
        const std::string id = request.request_id.substr(7);
        counts[id] = request.outputs.front().token_ids.size();
        if (request.finished) {
          REQUIRE(counts[id] == 32);
          REQUIRE(request.outputs.front().finish_reason.value_or("") == "length");
          done.insert(id);
        }
      }
      if (profile) (void)vt::xpu::DrainProfileEvents();
      if (host_profile) (void)vt::xpu::DrainHostProfileRecords();
      add_warmup();
    }
    REQUIRE(done.size() == specs.size());
    REQUIRE_FALSE(loaded->engine().has_unfinished_requests());
    REQUIRE(loaded->scheduler().kv_cache_manager->reset_prefix_cache());
  }
  nlohmann::json result = {{"schema", "b70-exl3-r11-serving-output-v1"},
      {"case", task.at("id")}, {"cache_state", "cold"},
      {"profiled", profile || host_profile}, {"device_profile", profile},
      {"launch", {{"max_model_len", context_limit}, {"max_num_batched_tokens", 1600},
        {"max_num_seqs", 4}, {"num_blocks", 180}, {"kv_cache_dtype", "fp8"},
        {"prefix_caching", true}, {"mtp_depth", depth}, {"attention_page_tokens", loaded->block_size()}}},
      {"sampling", {{"temperature", 0.}, {"ignore_eos", true}}},
      {"startup_state", warmup ? "request_warmup_o32_then_prefix_reset" :
                                "constructor_only_native_lazy_work_in_first_request"},
      {"warmup", {{"enabled", warmup}, {"cycles", warmup_cycles},
                  {"output_tokens_per_request", warmup ? 32 : 0}}},
      {"requests", nlohmann::json::object()}, {"cycles", nlohmann::json::array()},
      {"device_profile_records", nlohmann::json::array()},
      {"host_profile_records", nlohmann::json::array()}};
  // Bound pending diagnostic events to one engine cycle. Drain/serialization
  // happen after its wall timestamp and only in the separate profiling arm;
  // they remain visible in that arm's end-to-end time and host gaps.
  const auto drain_profiles = [&](int cycle) {
    if (profile)
      for (const auto& r : vt::xpu::DrainProfileEvents())
        result["device_profile_records"].push_back({{"cycle", cycle}, {"stage", r.stage}, {"matrix", r.matrix},
            {"queue", r.queue_id}, {"start_ns", r.start_ns}, {"end_ns", r.end_ns},
            {"stream_span", r.stream_span}, {"host_submit_ns", r.host_submit_ns}});
    if (host_profile)
      for (const auto& r : vt::xpu::DrainHostProfileRecords())
        result["host_profile_records"].push_back({{"cycle", cycle}, {"stage", r.stage}, {"queue", r.queue_id},
            {"start_ns", r.start_steady_ns}, {"end_ns", r.end_steady_ns},
            {"copy_bytes", r.copy_bytes}, {"caller_address", r.caller_address}});
  };
  const auto began = std::chrono::steady_clock::now();
  const auto seconds = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  };
  std::map<std::string, int64_t> emitted;
  std::set<std::string> admitted, finished;
  const auto admit = [&] {
    for (const auto& spec : specs) {
      const std::string id = spec.at("id");
      if (admitted.count(id)) continue;
      if (spec.contains("after_emitted")) {
        const auto& trigger = spec.at("after_emitted");
        if (emitted[trigger.at("request").get<std::string>()] <
            trigger.at("tokens").get<int64_t>()) continue;
      }
      const auto prompt = spec.at("prompt_ids").get<std::vector<int32_t>>();
      const int cap = spec.at("output_tokens");
      REQUIRE_FALSE(prompt.empty());
      REQUIRE(prompt.size() + cap <= size_t(context_limit));
      vllm::SamplingParams sampling;
      sampling.temperature = 0.; sampling.ignore_eos = true;
      sampling.max_tokens = cap;
      sampling.output_kind = vllm::RequestOutputKind::kCumulative;
      const double arrived = seconds();
      loaded->engine().add_request(id, prompt, sampling);
      result["requests"][id] = {{"prompt_ids", prompt}, {"output_limit", cap},
          {"admitted_at_s", arrived}, {"observations", nlohmann::json::array()}};
      admitted.insert(id);
    }
  };
  admit();
  for (int cycle = 0; loaded->engine().has_unfinished_requests() && cycle < 4096; ++cycle) {
    const auto& runner = loaded->runner();
    const int64_t proposed_before = runner.spec_drafts_proposed();
    const int64_t accepted_before = runner.spec_drafts_accepted();
    const auto graph_before = vt::GetGraphBreakStats();
    const double start = seconds();
    const auto outputs = loaded->engine().step();
    const double end = seconds();
    if (profile || host_profile) drain_profiles(cycle);
    int64_t emitted_this_cycle = 0;
    for (const auto& request : outputs) {
      REQUIRE(request.outputs.size() == 1);
      const auto& completion = request.outputs.front();
      const int64_t count = completion.token_ids.size();
      const int64_t delta = count - emitted[request.request_id];
      REQUIRE(delta >= 0);
      emitted_this_cycle += delta;
      emitted[request.request_id] = count;
      auto& trace = result["requests"][request.request_id];
      if (delta > 0)
        trace["observations"].push_back({{"at_s", end}, {"tokens", count}, {"new_tokens", delta}});
      if (request.finished) {
        trace["finished_at_s"] = end;
        trace["ids"] = completion.token_ids;
        trace["finish_reason"] = completion.finish_reason.value_or("");
        finished.insert(request.request_id);
      }
    }
    const auto& step = runner.last_step();
    const auto& dense = runner.input_batch().req_ids;
    nlohmann::json positions = nlohmann::json::object();
    for (size_t row = 0; row + 1 < step.query_start_loc.size(); ++row) {
      REQUIRE(dense[row].has_value());
      const std::string& id = *dense[row];
      const int offset = step.query_start_loc[row];
      REQUIRE(offset >= 0);
      REQUIRE(size_t(offset) < step.positions.size());
      positions[id] = step.positions[offset];
      auto& trace = result["requests"][id];
      if (!trace.contains("first_scheduled_position"))
        trace["first_scheduled_position"] = step.positions[offset];
    }
    const auto graph_after = vt::GetGraphBreakStats();
    result["cycles"].push_back({{"cycle", cycle}, {"start_s", start}, {"end_s", end},
        {"cycle_wall_s", end - start}, {"emitted_tokens", emitted_this_cycle},
        {"proposed", runner.spec_drafts_proposed() - proposed_before},
        {"accepted", runner.spec_drafts_accepted() - accepted_before},
        {"positions", positions}, {"logical_forward_tokens", runner.last_forward_num_actual_tokens()},
        {"logical_logit_rows", runner.last_forward_rows()},
        {"graph_captures", graph_after.segments_captured - graph_before.segments_captured},
        {"graph_replays", graph_after.replays - graph_before.replays}});
    if (profile || host_profile) {
      // This is the metadata consumed by the completed target forward, also
      // when its kernels replay a graph. Dispatch/capture logs cannot count
      // those replays. No tensor downloads or model/graph-policy changes here.
      const auto& am = runner.last_attn_meta();
      const auto& gm = runner.last_gdn_meta();
      REQUIRE(am.num_reqs == runner.last_forward_num_reqs());
      REQUIRE(am.query_start_loc.size() == size_t(am.num_reqs + 1));
      REQUIRE(am.query_start_loc.front() == 0);
      REQUIRE(am.query_start_loc.back() == am.num_actual_tokens);
      REQUIRE(am.seq_lens.size() == size_t(am.num_reqs));
      std::vector<int32_t> query_lengths;
      std::vector<std::string> request_ids;
      for (int row = 0; row < am.num_reqs; ++row) {
        const int32_t length = am.query_start_loc[row + 1] - am.query_start_loc[row];
        REQUIRE(length > 0); REQUIRE(dense[row].has_value());
        query_lengths.push_back(length); request_ids.push_back(*dense[row]);
      }
      result["cycles"].back()["target_shape"] = {
          {"request_ids", request_ids}, {"query_start_loc", am.query_start_loc},
          {"query_lengths", query_lengths}, {"seq_lens", am.seq_lens},
          {"draft_tokens_per_request", step.num_draft_tokens_per_req},
          {"num_prefills", gm.num_prefills}, {"num_decodes", gm.num_decodes},
          {"num_spec_decodes", gm.num_spec_decodes},
          {"scope", "Actual logical target metadata; eager kernel times and whole graph replay times are separate"}};
    }
    admit();
  }
  result["end_to_end_wall_s"] = seconds();
  REQUIRE(finished.size() == specs.size());
  for (const auto& spec : specs) {
    const auto& trace = result["requests"].at(spec.at("id").get<std::string>());
    CHECK(trace.at("ids").size() == spec.at("output_tokens").get<size_t>());
    CHECK(trace.at("finish_reason") == "length");
    CHECK(trace.at("first_scheduled_position") == 0);
  }
  if (profile || host_profile) drain_profiles(-1);
  const auto memory = vt::xpu::GetMemoryInfo();
  result["backend_peak_device_bytes"] = memory.peak_allocated_bytes;
  result["backend_live_device_bytes"] = memory.allocated_bytes;
  result["graph_device_bytes"] = memory.graph_device_bytes;
  result["w8a8_workspace_bytes"] = memory.w8a8_workspace_bytes;
  result["w8a8_preparation_bytes"] = memory.w8a8_preparation_bytes;
  result["fp16_silu_table_bytes"] = memory.fp16_silu_table_bytes;
  result["w8a8_panel_columns"] = vt::Exl3W8A8ModelPanelColumns();
  CHECK(result["backend_peak_device_bytes"].get<uint64_t>() <= (uint64_t(32) << 30));
  struct rusage usage{};
  REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
  result["host_peak_rss_bytes"] = uint64_t(usage.ru_maxrss) * 1024;
  CHECK(result["host_peak_rss_bytes"].get<uint64_t>() <= (uint64_t(32) << 30));
  CHECK(vt::GetReferenceTierHits() == 0);
  if (!depth) {
    CHECK(loaded->runner().spec_drafts_proposed() == 0);
    CHECK(loaded->runner().spec_drafts_accepted() == 0);
  }
  loaded.reset();
  result["after_release_graph_bytes"] = vt::xpu::GetMemoryInfo().graph_device_bytes;
  CHECK(result["after_release_graph_bytes"] == 0);
  std::ofstream file(output); file << result.dump(2) << '\n'; file.close(); REQUIRE(file.good());
}

TEST_CASE("XPU EXL3 real MTP: original P128 draft and selected compact head") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixture = std::getenv("VT_B70_EXL3_MTP_FIXTURE");
  const char* subset_path = std::getenv("EXL3_DRAFT_VOCAB");
  if (!model || !fixture || !subset_path) {
    std::cerr << "Set VT_B70_EXL3_MODEL, VT_B70_EXL3_MTP_FIXTURE and EXL3_DRAFT_VOCAB.\n";
    std::exit(77);
  }
  // Queue outlives all resident weights, result owners and working allocations.
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
  REQUIRE(config.hidden_size == 5120);
  REQUIRE(config.vocab_size == 248320);
  std::vector<vllm::SafetensorsFile> shards;
  for (const auto* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((std::filesystem::path(model) / name).string()));
  auto weights = vllm::LoadQwen3_5MTP(shards, config, vllm::Qwen3_5MTPKind::kDense,
                                    vt::DeviceType::kXPU);
  std::ifstream subset_file(subset_path);
  REQUIRE(subset_file.good());
  const auto subset = nlohmann::json::parse(subset_file);
  weights.draft_head_exl3 = vllm::LoadExl3DraftHead(shards, subset, config.vocab_size);
  REQUIRE(weights.IsExl3());
  REQUIRE(weights.dense_layers.size() == 1);
  REQUIRE(weights.final_norm.dtype == vt::DType::kF16);
  REQUIRE(weights.draft_head_exl3.weight.OutFeatures() == 65536);
  vllm::Qwen3_5DenseWeights target;
  const vllm::TensorResolver get = [&](const std::string& name) -> const vllm::StTensor& {
    return shards[0].Get(name);
  };
  target.embed_tokens = vllm::dense_loaders::LoadF16Direct(get, "model.language_model.embed_tokens.weight");
  // This bounded call never constructs/uploads a dense or full target head.
  vllm::Qwen3_5MTPModel draft(weights, target, config);
  const auto oracle = vllm::SafetensorsFile::Open(fixture);
  const auto token_map = Ints(oracle.Get("token_map"));
  REQUIRE(token_map.size() == 65536);
  CHECK(std::memcmp(weights.draft_head_exl3.token_ids.bytes.data(), token_map.data(), 65536 * 4) == 0);

  xpu_test::Buffer head_input(gpu.q, vt::DType::kF16, {1, 5120});
  head_input.upload(oracle.Get("head_hidden").data);
  auto isolated_logits = draft.ComputeLogits(head_input.tensor, gpu.q);
  REQUIRE(isolated_logits.rows == 1);
  REQUIRE(isolated_logits.vocab == 65536);
  std::vector<float> isolated(65536);
  auto& backend = vt::GetBackend(gpu.q.device);
  backend.Copy(gpu.q, isolated.data(), isolated_logits.device_tensor.data, isolated.size() * 4);
  backend.Synchronize(gpu.q);
  Compare("identical_head_input", isolated, oracle.Get("compact_logits"), true);
  CHECK(draft.SelectDraftTokens(isolated_logits, gpu.q) == Ints(oracle.Get("global_argmax")));

  const auto ids = Ints(oracle.Get("input_ids"));
  const auto positions = Ints(oracle.Get("positions"));
  const auto rows32 = Ints(oracle.Get("selected_rows"));
  REQUIRE(ids.size() == 128);
  REQUIRE(positions.size() == 128);
  REQUIRE(rows32 == std::vector<int32_t>{127});
  xpu_test::Buffer input(gpu.q, vt::DType::kF16, {128, 5120});
  input.upload(oracle.Get("target_hidden").data);
  // Native-owned KV with the observed physical page/block/slot identities.
  std::filesystem::path record_path(fixture);
  record_path.replace_extension(".json");
  std::ifstream record_file(record_path);
  REQUIRE(record_file.good());
  const auto record = nlohmann::json::parse(record_file);
  const auto& active = record.at("attention_metadata").at("active_values");
  const int32_t block = active.at("block_table_first_column").at("values").at(0).at(0).get<int32_t>();
  REQUIRE(block >= 0);
  const int64_t blocks = int64_t(block) + 1;
  xpu_test::Buffer storage(gpu.q, vt::DType::kI8, {blocks * 1600 * 4 * 512});
  std::vector<uint8_t> poison(storage.bytes, 0x7f);
  storage.upload(poison.data());
  vllm::PagedKvCache cache;
  cache.data = storage.tensor.data; cache.dtype = vt::DType::kI8;
  cache.num_blocks = blocks; cache.block_size = 1600;
  cache.num_kv_heads = 4; cache.head_size = 256;
  cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  vllm::v1::CommonAttentionMetadata metadata;
  metadata.num_reqs = 1; metadata.num_actual_tokens = 128;
  metadata.query_start_loc = metadata.query_start_loc_cpu = {0, 128};
  metadata.seq_lens = metadata.seq_lens_cpu = {128};
  metadata.num_computed_tokens_cpu = {0};
  metadata.max_query_len = 128; metadata.max_seq_len = 128;
  metadata.block_table_num_cols = 1; metadata.block_table_tensor = {block};
  metadata.slot_mapping = active.at("slot_mapping").at("values").get<std::vector<int64_t>>();
  auto hidden = draft.ForwardPaged(ids, positions, input.tensor, metadata, cache, gpu.q);
  REQUIRE(hidden.tensor.dtype == vt::DType::kF16);
  REQUIRE(hidden.tensor.shape[0] == 128);
  std::vector<uint16_t> hidden_bits(128 * 5120);
  backend.Copy(gpu.q, hidden_bits.data(), hidden.tensor.data, hidden_bits.size() * 2);
  backend.Synchronize(gpu.q);
  std::vector<float> hidden_values(hidden_bits.size());
  std::transform(hidden_bits.begin(), hidden_bits.end(), hidden_values.begin(), vt::F16ToF32);
  Compare("real_draft_forward", hidden_values, oracle.Get("draft_hidden"), false);

  const std::vector<int64_t> rows(rows32.begin(), rows32.end());
  const auto selected = draft.GatherHiddenRows(hidden.tensor, rows, gpu.q);
  auto logits = draft.ComputeLogits(selected.tensor, gpu.q);
  REQUIRE(logits.rows == 1);
  REQUIRE(logits.vocab == 65536);
  std::vector<float> got(65536);
  backend.Copy(gpu.q, got.data(), logits.device_tensor.data, got.size() * 4);
  backend.Synchronize(gpu.q);
  Compare("real_draft_compact_logits", got, oracle.Get("compact_logits"), false);
  CHECK(draft.SelectDraftTokens(logits, gpu.q) == Ints(oracle.Get("global_argmax")));
  CHECK(target.lm_head.Empty());
  CHECK(target.lm_head_exl3.Empty());
  CHECK(vt::GetReferenceTierHits() == 0);
}
