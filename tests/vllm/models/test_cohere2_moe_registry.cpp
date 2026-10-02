// Cohere2MoeForCausalLM REACHABILITY gate (spec .agents/specs/cohere2-moe.md,
// gate (d)). A tiny checkpoint and its config.json are written to temporary
// files, the config is read by LoadHfConfig, the model is loaded through
// `ModelRegistry::Load` under the released architecture string, and a greedy
// decode runs through `GPUModelRunner` (prefill, then one token per step through
// the paged cache, past the sliding window). The tokens must equal the greedy
// continuation of the independent torch transcription of the pinned
// cohere2_moe.py (scripts/cohere2-moe-ref.py, bf16 arm, full recompute per step).
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cohere2_moe_goldens.inc"
#include "cohere2_moe_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/cohere2_moe.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::SafetensorsFile;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;

namespace {

constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 8;
constexpr int kMaxModelLen = 32;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

struct Fixture {
  std::unique_ptr<c2m_tiny::TempFile> st;
  std::unique_ptr<c2m_tiny::TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  std::unique_ptr<vllm::LoadedModel> model;

  Fixture() {
    const HfConfig parsed = c2m_tiny::ConfigFromJson(c2m_golden::kConfig_a);
    const vllm::Cohere2MoeParams p = vllm::ParseCohere2MoeParams(parsed);
    st = std::make_unique<c2m_tiny::TempFile>(c2m_tiny::BuildSt(c2m_tiny::SynthTensors(p)),
                                              ".safetensors");
    cfg_json = std::make_unique<c2m_tiny::TempFile>(
        nlohmann::json::parse(c2m_golden::kConfig_a).dump(2), ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
    model = ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
  }
};

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;
  sp.PostInit();
  return sp;
}

std::vector<int32_t> RunnerGreedy(Fixture& fx, const std::vector<int32_t>& prompt, int steps) {
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  vt::Queue q = Q();
  GPUModelRunner runner(fx.cfg, *fx.model, kv, q, /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  const std::string id = "north";
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = prompt;
  nr.sampling_params = Greedy();
  // Out-of-order physical blocks so a block-table defect cannot hide.
  nr.block_ids = {std::vector<int>{5, 2}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = prompt;
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs.push_back(nr);
  so.num_scheduled_tokens[id] = static_cast<int>(prompt.size());
  so.total_num_scheduled_tokens = static_cast<int>(prompt.size());
  CHECK_FALSE(runner.execute_model(so).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  std::vector<int32_t> out{m1.sampled_token_ids[0][0]};
  int computed = static_cast<int>(prompt.size());
  int outputs = 1;
  for (int s = 1; s < steps; ++s) {
    SchedulerOutput sd;
    CachedRequestData cached;
    cached.req_ids = {id};
    cached.num_computed_tokens.push_back(computed);
    cached.num_output_tokens.push_back(outputs);
    cached.new_block_ids.emplace_back(std::nullopt);
    sd.scheduled_cached_reqs = std::move(cached);
    sd.num_scheduled_tokens[id] = 1;
    sd.total_num_scheduled_tokens = 1;
    CHECK_FALSE(runner.execute_model(sd).has_value());
    vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids.size() == 1);
    out.push_back(md.sampled_token_ids[0][0]);
    ++computed;
    ++outputs;
  }
  return out;
}

}  // namespace

TEST_CASE("cohere2_moe: the released architecture string loads through the registry") {
  Fixture fx;
  REQUIRE(fx.model != nullptr);
  CHECK(fx.model->registration().architecture == "Cohere2MoeForCausalLM");
}

TEST_CASE("cohere2_moe: GPUModelRunner greedy decode equals the pinned transcription") {
  Fixture fx;
  const std::vector<int32_t> prompt(std::begin(c2m_golden::kGreedyPrompt),
                                    std::end(c2m_golden::kGreedyPrompt));
  const std::vector<int32_t> want(std::begin(c2m_golden::kGreedyOut),
                                  std::end(c2m_golden::kGreedyOut));
  const std::vector<int32_t> got = RunnerGreedy(fx, prompt, static_cast<int>(want.size()));
  std::string gs, ws;
  for (int32_t t : got) gs += std::to_string(t) + " ";
  for (int32_t t : want) ws += std::to_string(t) + " ";
  MESSAGE("runner greedy: " << gs << "| reference: " << ws);
  CHECK(got == want);
}
