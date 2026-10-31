// SCRATCH (stage 3 evidence, not committed): Flint TT serving trace,
// attention-on-device (VT_FLINT_TT_DEVICE_ATTN=1) vs host-attention (=0).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <doctest/doctest.h>
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/flint_tt.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {
using vllm::HfConfig;
struct Step {
  std::vector<int32_t> positions;
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  std::vector<int32_t> logits_indices;
  vllm::v1::CommonAttentionMetadata meta;
  vllm::v1::GDNAttentionMetadata gdn;
  vllm::ModelForwardInput input;
  Step(const std::vector<int32_t>& ids, const HfConfig& config,
       vt::Queue& queue)
      : positions(ids.size()), input(MakeInput(ids, config, queue, *this)) {}
 private:
  static vllm::ModelForwardInput MakeInput(const std::vector<int32_t>& ids,
                                           const HfConfig& config,
                                           vt::Queue& queue, Step& self) {
    for (size_t i = 0; i < ids.size(); ++i)
      self.positions[i] = static_cast<int32_t>(i);
    self.meta.num_reqs = 1;
    self.meta.num_actual_tokens = static_cast<int>(ids.size());
    self.meta.max_query_len = static_cast<int>(ids.size());
    self.meta.max_seq_len = static_cast<int>(ids.size());
    self.meta.causal = true;
    return vllm::ModelForwardInput{.token_ids = ids,
                                   .positions = self.positions,
                                   .attn_meta = self.meta,
                                   .gdn_meta = self.gdn,
                                   .attn_kv = self.attn_kv,
                                   .gdn_state = self.gdn_state,
                                   .config = config,
                                   .queue = queue,
                                   .logits_indices = self.logits_indices,
                                   .num_reqs = 1};
  }
};
}  // namespace

TEST_CASE("scratch flint tt serving trace") {
  const char* model_path = std::getenv("FLINT_TT_W0_MODEL");
  if (!model_path) { MESSAGE("no model"); return; }
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (!tb) { MESSAGE("no device"); return; }
  vt::Queue q = tb->CreateQueue();
  vllm::GgufFile gguf = vllm::GgufFile::Open(model_path);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  reg.factory->parse_config(config);
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kTENSTORRENT);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  reg.factory->prepare(*model, config, q);
  std::vector<int32_t> prompt(static_cast<size_t>(32), 100);
  const int kSteps = 6, kReps = 1;
  for (int rep = 0; rep < kReps; ++rep) {
    double prefill_ms = 0, decode_ms = 0;
    std::vector<int32_t> ids = prompt;
    for (int step = 0; step < kSteps + 1; ++step) {
      Step s(ids, config, q);
      const auto t0 = std::chrono::steady_clock::now();
      vllm::ModelRegistry::Forward(*model, s.input);
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::fprintf(stderr, "[TRACE] step=%d %.1fms\n", step, ms);
      if (step == 0) prefill_ms = ms; else decode_ms += ms;
      ids.push_back(200 + step);
    }
    std::fprintf(stderr, "[TRACE] rep=%d device_attn=%s prefill32=%.1fms decode_avg=%.1fms\n",
                 rep, ([]{const char* e=std::getenv("VT_FLINT_TT_DEVICE_ATTN");return e&&e[0]=='1';}() ? "on" : "off"),
                 prefill_ms, decode_ms / kSteps);
  }
}
