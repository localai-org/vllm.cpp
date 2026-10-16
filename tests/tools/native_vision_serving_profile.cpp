// Internal diagnostic: real C-API engine plus existing XPU request-boundary spans.
// Link one static runtime; do not expand the public C ABI or add product waits.
#include "vllm.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/models/qwen3_5_diagnostics.h"
#include "vt/xpu.h"
#ifdef VLLM_CPP_XPU_ONEDNN
#include "vt/xpu/xpu_gptq4.h"
#endif
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
std::string Error() {
  const char* error = vllm_last_error();
  return error ? error : "unknown C-API failure";
}
json Memory() {
  const auto m = vt::xpu::GetMemoryInfo(0);
  auto& b = vt::GetBackend(vt::DeviceType::kXPU);
  const auto p = vllm::Pool(b).stats();
  const auto persistent = vllm::Qwen3_5PersistentDecodePoolForDiagnostics(b).stats();
#ifdef VLLM_CPP_XPU_ONEDNN
  const auto onednn_scratch = vt::xpu::GetGptq4RuntimeStats(0).scratchpad_capacity_bytes;
#else
  const size_t onednn_scratch = 0;
#endif
#ifdef VT_MARLIN_NVFP4
  const auto aux = vllm::AuxPool(b).stats();
#else
  // This XPU build has no Marlin shared-expert aux-pool path.
  const vllm::DevicePool::Stats aux{};
#endif
  return {{"allocated_bytes", m.allocated_bytes}, {"pinned_bytes", m.pinned_bytes},
      {"free_pool_bytes", p.retained_bytes}, {"live_pool_blocks", p.live_blocks},
      {"pool_misses", p.misses}, {"graph_count", m.graph_count},
      {"graph_device_bytes", m.graph_device_bytes},
      {"onednn_queue_scratch_bytes", onednn_scratch},
      {"persistent_decode_pool", {{"free_bytes", persistent.retained_bytes},
        {"live_blocks", persistent.live_blocks}, {"misses", persistent.misses}}},
      {"aux_pool", {{"free_bytes", aux.retained_bytes}, {"live_blocks", aux.live_blocks}, {"misses", aux.misses}}},
      {"context_resident", {{"exl3", m.exl3_workspace_bytes}, {"gdn", m.gdn_workspace_bytes},
        {"native_gdn", m.native_gdn_workspace_bytes}, {"attention", m.attention_workspace_bytes},
        {"sampling", m.sampling_workspace_bytes}, {"w8a8", m.w8a8_workspace_bytes},
        {"w8a8_preparation", m.w8a8_preparation_bytes}, {"fp16_silu_table", m.fp16_silu_table_bytes},
        {"gated_silu_table", m.gated_silu_table_bytes}}}};
}
size_t ContextBytes(const json& row) {
  size_t sum = 0;
  for (const auto& value : row.at("context_resident")) sum += value.get<size_t>();
  return sum;
}
json Record(const vt::xpu::ProfileRecord& event) {
  Require(event.stream_span && event.end_ns > event.start_ns, "invalid executed queue span");
  return {{"stage", event.stage}, {"queue_id", event.queue_id},
      {"start_ns", event.start_ns}, {"end_ns", event.end_ns},
      {"duration_s", static_cast<double>(event.end_ns - event.start_ns) * 1e-9}};
}
void Run(const char* model, const char* requests_file, int depth, bool memory_only, json& report) {
  Require(std::getenv("VT_XPU_PROFILE") &&
          std::string(std::getenv("VT_XPU_PROFILE")) == (memory_only ? "0" : "1"),
          "diagnostic requires explicit VT_XPU_PROFILE=0 for memory, =1 for spans");
  for (const char* flag : {"VT_NATIVE_VISION_TRACE", "VT_PREFIX_SNAPSHOT_TRACE", "VT_XPU_HOST_PROFILE"}) {
    const char* value = std::getenv(flag);
    Require(!value || std::string(value) == "0", "disable tensor/host traces for this diagnostic");
  }
  std::ifstream input(requests_file);
  Require(input.good(), "cannot read frozen request bundle");
  const auto bundle = json::parse(input);
  Require(bundle.is_array() && !bundle.empty(), "request bundle required");
  const auto& entry = bundle.at(0);
  json body = entry.at("body");
  Require(body.at("messages").size() == 1 && body.at("messages").at(0).at("role") == "user",
          "one bounded initial user message required");
  int images = 0;
  for (const auto& part : body.at("messages").at(0).at("content")) {
    if (part.at("type") == "image_url") ++images;
    else Require(part.at("type") == "text", "unsupported content part");
  }
  Require(images == 1 && body.value("n", 1) == 1 && body.at("max_tokens").get<int>() > 0 &&
          body.at("max_tokens").get<int>() <= 128 && body.at("temperature").get<double>() == 0.,
          "one-image bounded greedy request required");
  body["stream"] = false;
  const std::string request = body.dump();
  vllm_model_params params = vllm_model_params_default();
  params.model_path = model;
  params.block_size = 1600;
  params.num_blocks = 180;
  params.max_model_len = 8192;
  params.max_num_seqs = 1;
  params.max_num_batched_tokens = 1600;
  params.kv_cache_dtype = "fp8";
  params.enable_prefix_caching = 2;
  params.gpu_memory_utilization = 0.;
  params.limit_mm_per_prompt = "{\"image\":1,\"video\":0}";
  const std::string spec = "{\"method\":\"mtp\",\"num_speculative_tokens\":3}";
  if (depth) params.speculative_config = spec.c_str();
  const json cold = memory_only ? Memory() : json();
  if (memory_only) {
    report["cold_before_model"] = cold;
    Require(cold.at("graph_count") == 0 && cold.at("live_pool_blocks") == 0 &&
            cold.at("persistent_decode_pool").at("live_blocks") == 0 &&
            cold.at("aux_pool").at("live_blocks") == 0,
            "memory diagnostic requires an isolated engine/backend");
  }
  vllm_engine* handle = nullptr;
  if (vllm_engine_load(&params, &handle) != VLLM_OK) throw std::runtime_error("load: " + Error());
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(handle, &vllm_engine_free);
  // Discard weight preparation events. Profile drains occur only between whole
  // requests in this diagnostic, never in the product prefill/decode hot path.
  const auto load_events = memory_only ? 0 : vt::xpu::DrainProfileEvents(0).size();
  report.update({{"spec_depth", depth}, {"fixture", entry.at("name")},
      {"expected", entry.at("expected")}, {"load_events_excluded", load_events},
      {"prefill_budget", 1600}, {"prefix_cache", "disabled"}, {"cases", json::array()}});
  json first_choices, first_usage;
  if (memory_only) report["model_loaded_before_first_image"] = Memory();
  json warm_baseline;
  for (int repeat = 0; repeat != (memory_only ? 6 : 4); ++repeat) {
    char* raw = nullptr;
    if (vllm_chat(engine.get(), request.c_str(), &raw) != VLLM_OK)
      throw std::runtime_error("chat: " + Error());
    std::unique_ptr<char, decltype(&vllm_string_free)> owned(raw, &vllm_string_free);
    Require(raw != nullptr, "missing actual response");
    const auto response = json::parse(raw);
    const auto usage = response.at("usage");
    const int prompt = usage.at("prompt_tokens").get<int>();
    const int output = usage.at("completion_tokens").get<int>();
    Require(prompt > 0 && prompt < 1600 && output > 0 && output <= body.at("max_tokens").get<int>() &&
        usage.at("total_tokens").get<int>() == prompt + output && response.at("choices").size() == 1,
        "single-chunk response usage/shape differs");
    if (repeat == 0) { first_choices = response.at("choices"); first_usage = usage; }
    Require(response.at("choices") == first_choices && usage == first_usage, "repeat response changed");
    if (memory_only) {
      // Test-only negative: genuinely grow the returned free pool at the first
      // additional warm request, with no outstanding use of this new block.
      // Larger than ALL retained free bytes guarantees a driver miss, rather
      // than accidentally borrowing an existing class. Never a serving mode.
      const char* grow = std::getenv("VT_B70_VISION_GROW_WARM_POOL");
      if (repeat == 3 && grow && grow[0] == '1') {
        auto& backend = vt::GetBackend(vt::DeviceType::kXPU);
        auto& pool = vllm::Pool(backend);
        const auto before = pool.stats();
        const size_t bytes = before.retained_bytes + 1;
        void* block = pool.Get(backend, bytes);
        pool.Put(backend, bytes, block);
        const auto after = pool.stats();
        report["injected_free_pool_growth"] = {{"requested_bytes", bytes},
            {"before_free_bytes", before.retained_bytes}, {"after_free_bytes", after.retained_bytes},
            {"before_misses", before.misses}, {"after_misses", after.misses},
            {"before_live_blocks", before.live_blocks}, {"after_live_blocks", after.live_blocks}};
        Require(after.retained_bytes > before.retained_bytes && after.misses > before.misses,
                "negative warm-pool allocation did not grow the free pool");
      }
      const auto memory = Memory();
      report["cases"].push_back({{"repeat", repeat}, {"response", response}, {"memory", memory}});
      if (repeat == 2) warm_baseline = memory;
      if (repeat > 2) Require(memory == warm_baseline, "fixed-shape warm memory/owners changed");
      continue; // No queue profiling drains or added synchronizations in memory mode.
    }
    const auto events = vt::xpu::DrainProfileEvents(0);
    json target = json::array(), draft = json::array();
    for (const auto& event : events) {
      if (event.stage == "runner_target_forward") target.push_back(Record(event));
      if (event.stage == "runner_mtp_draft") draft.push_back(Record(event));
    }
    Require(target.size() > 1, "missing target prefill/decode spans");
    Require(depth ? !draft.empty() : draft.empty(), "missing/unexpected draft proposal spans");
    // For this isolated C1/no-prefix prompt shorter than the fixed prefill
    // budget, the first forward is the sole target prefill. First proposal is
    // the shifted draft setup PLUS its two extra MTP3 draft-decode steps.
    const auto first_target = target.at(0);
    json first_draft;
    if (depth) {
      first_draft = draft.at(0);
      Require(first_target.at("queue_id") == first_draft.at("queue_id") &&
              first_target.at("end_ns").get<uint64_t>() <= first_draft.at("start_ns").get<uint64_t>(),
              "target/proposal spans overlap or use unrelated queue clocks");
    }
    report["cases"].push_back({{"repeat", repeat}, {"response", response},
        {"collected_events", events.size()}, {"target_spans", target}, {"draft_spans", draft},
        {"target_prefill", first_target}, {"initial_mtp_proposal", first_draft}});
  }
  vllm_spec_acceptance counters{};
  Require(vllm_engine_spec_acceptance(engine.get(), &counters) == VLLM_OK, "missing actual speculation counter");
  Require(depth ? counters.drafts_proposed > 0 : counters.drafts_proposed == 0, "missing/unexpected proposals");
  report["drafts_proposed"] = counters.drafts_proposed;
  report["drafts_accepted"] = counters.drafts_accepted;
  const auto memory = vt::xpu::GetMemoryInfo(0);
  report["backend_live_bytes"] = memory.allocated_bytes;
  report["backend_peak_bytes"] = memory.peak_allocated_bytes;
  engine.reset();
  report["engine_freed"] = true;
  if (memory_only) {
    const auto freed = Memory();
    report["after_engine_free"] = freed;
    Require(freed.at("graph_count") == cold.at("graph_count") &&
            freed.at("graph_device_bytes") == cold.at("graph_device_bytes"), "engine graph owners remain");
    Require(freed.at("live_pool_blocks") == cold.at("live_pool_blocks"), "engine pool owners remain");
    Require(freed.at("onednn_queue_scratch_bytes") == cold.at("onednn_queue_scratch_bytes"),
            "engine queue-scoped oneDNN scratch remains");
    for (const char* pool : {"persistent_decode_pool", "aux_pool"})
      Require(freed.at(pool).at("live_blocks") == cold.at(pool).at("live_blocks"),
              "engine persistent/aux pool owners remain");
    // Engine destruction completes outstanding consumers and drops cache/graphs.
    // Only this diagnostic teardown may release returned scratch.
    report["diagnostic_pool_released_bytes"] = vllm::Pool(vt::GetBackend(vt::DeviceType::kXPU)).Drain(
        vt::GetBackend(vt::DeviceType::kXPU));
    auto& backend = vt::GetBackend(vt::DeviceType::kXPU);
    report["diagnostic_persistent_decode_pool_released_bytes"] =
        vllm::Qwen3_5PersistentDecodePoolForDiagnostics(backend).Drain(backend);
#ifdef VT_MARLIN_NVFP4
    report["diagnostic_aux_pool_released_bytes"] = vllm::AuxPool(backend).Drain(backend);
#else
    report["diagnostic_aux_pool_released_bytes"] = 0;
#endif
    const auto final = Memory();
    report["after_diagnostic_pool_drain"] = final;
    const size_t cold_free_and_context = cold.at("free_pool_bytes").get<size_t>() +
        cold.at("persistent_decode_pool").at("free_bytes").get<size_t>() +
        cold.at("aux_pool").at("free_bytes").get<size_t>() + ContextBytes(cold);
    Require(cold.at("allocated_bytes").get<size_t>() >= cold_free_and_context,
            "cold owner accounting exceeds allocation");
    const size_t cold_noncontext = cold.at("allocated_bytes").get<size_t>() - cold_free_and_context;
    const size_t expected = cold_noncontext + ContextBytes(final);
    report["declared_no_model_baseline_bytes"] = expected;
    Require(final.at("allocated_bytes").get<size_t>() == expected, "unexplained allocation after engine teardown");
    Require(final.at("pinned_bytes") == cold.at("pinned_bytes"), "host USM owners remain");
    report["status"] = "PASS"; // Memory contract only, not numerical/semantic qualification.
  } else {
    report["status"] = "CAPTURED"; // Named facts and independent span checks run outside this tool.
  }
}
} // namespace
int main(int argc, char** argv) {
  if ((argc != 5 && argc != 6) || (argc == 6 && std::string(argv[5]) != "memory") ||
      std::filesystem::exists(argv[3])) {
    std::cerr << "usage: native-vision-serving-profile MODEL_DIR REQUEST_BUNDLE NEW_OUTPUT 0|3 [memory]\n";
    return 2;
  }
  json report = {{"status", "FAIL"}, {"scope", "profiled real-serving C1 target-prefill and initial MTP proposal queue spans; not trace-free speed or sum of kernel durations"}};
  int code = 1;
  try {
    const int depth = std::stoi(argv[4]);
    Require(depth == 0 || depth == 3, "depth must be zero or three");
    const bool memory_only = argc == 6;
    if (memory_only) report["scope"] = "C1 real-engine cold/load/first-use/six-request warm ownership and teardown; tracked backend only, not driver memory, numerical parity or performance";
    Run(argv[1], argv[2], depth, memory_only, report); code = 0;
  } catch (const std::exception& error) { report["error"] = error.what(); }
  std::ofstream output(argv[3]);
  if (!output.good()) return 2;
  output << report.dump(2) << '\n';
  std::cout << json{{"status", report.value("status", "FAIL")}, {"error", report.value("error", "")}}.dump() << '\n';
  return code;
}
