// Flint on Tenstorrent — the STAGE-2 MODEL-LEVEL device-attention gate
// (spec .agents/specs/flint-p150-stage2-gate.md, issue
// ISSUE-LOCAL-01M6HF07NW0NQPVC5V21EGZJ8M, owning row FLINT-P150-ATTENTION
// stage 2).
//
// The W0 gate (test_flint_tt_w0.cpp) always runs the HOST attention arm — it
// never references VT_FLINT_TT_DEVICE_ATTN. This TU is the missing case: the
// SAME keep-quant paged recipe run TWICE through the registry — the host arm
// (env unset) and the device arm (VT_FLINT_TT_DEVICE_ATTN=1, read at forward
// time by DeviceAttnEnabled(), flint_tt.cpp:106, at the three model-level
// blocks flint_tt.cpp:745/:832/:914) — asserting the SAME drift-tolerant
// band (rel_rms < 0.50, max_abs < 64.0) for the device arm and RECORDING
// both arms' measured numbers + greedy chains. The device arm's numbers are
// the ADOPTION INPUTS, not a foregone conclusion: if the band holds the
// wiring default flips to ON, if not it stays env-gated default-off with
// the gap named (the spec forces no verdict).
//
// Card + operator-run under the gpu.lock mutex; skip (host legs) without
// FLINT_TT_W0_MODEL pointing at the published flint GGUF container.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/flint_tt.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, ModelForwardInput
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

constexpr const char* kPublishedGguf =
    "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf";
constexpr const char* kDeviceAttnEnv = "VT_FLINT_TT_DEVICE_ATTN";
constexpr int kGreedySteps = 8;
// THE DRIFT-TOLERANT BAND, the same fence the host arm's W0 gate asserts
// (test_flint_tt_w0.cpp:379-380): the ~20 percent cross-epoch drift record
// lives there; an absolute band tighter than the drift measures the EPOCH.
constexpr double kBandRelRms = 0.50;
constexpr double kBandMaxAbs = 64.0;

bool TenstorrentPresent() {
  return vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) != nullptr;
}

using vllm::HfConfig;

// Step owns the metadata and carries the input built over its own members —
// ModelForwardInput holds REFERENCES, so this must outlive it (the W0 /
// test_flint_gguf.cpp loop-scope pattern).
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
    for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); ++i)
      self.positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    self.meta.num_reqs = 1;
    self.meta.num_actual_tokens = static_cast<int>(ids.size());
    self.meta.max_query_len = static_cast<int>(ids.size());
    self.meta.max_seq_len = static_cast<int>(ids.size());
    self.meta.causal = true;
    return vllm::ModelForwardInput{
        .token_ids = ids,
        .positions = self.positions,
        .attn_meta = self.meta,
        .gdn_meta = self.gdn,
        .attn_kv = self.attn_kv,
        .gdn_state = self.gdn_state,
        .config = config,
        .queue = queue,
        .logits_indices = self.logits_indices,
        .num_reqs = 1,
    };
  }
};

struct ArmResult {
  double rel_rms = 0.0;
  double max_abs = 0.0;
  std::vector<int32_t> chain;
  std::string chain_str;
};

ArmResult RunArm(const std::vector<int32_t>& prompt_ids,
                 const std::vector<float>& cpu_first,
                 const HfConfig& config, vt::Queue& q,
                 vllm::LoadedModel& model, const char* label) {
  const auto forward = [&](const std::vector<int32_t>& ids) {
    Step s(ids, config, q);
    return vllm::ModelRegistry::Forward(model, s.input);
  };

  // First-token logits distance vs the CPU dequant reference — the SAME
  // metric the host arm's W0 gate measures.
  const vllm::ForwardLogits first = forward(prompt_ids);
  REQUIRE(first.host.size() == cpu_first.size());
  ArmResult r;
  double sum_sq = 0.0, sum_cpu_sq = 0.0;
  for (size_t i = 0; i < cpu_first.size(); ++i) {
    const double diff = double(first.host[i]) - double(cpu_first[i]);
    r.max_abs = std::max(r.max_abs, std::abs(diff));
    sum_sq += diff * diff;
    sum_cpu_sq += double(cpu_first[i]) * double(cpu_first[i]);
  }
  const double rms = std::sqrt(sum_sq / double(cpu_first.size()));
  r.rel_rms = rms / std::sqrt(sum_cpu_sq / double(cpu_first.size()));

  // The greedy chain: the arm's OWN tokens (adjudicated-divergent vs the CPU
  // row — RECORDED, not pinned; the W0 record explains why: the chain is
  // epoch-dependent and the distance legs carry the regression power).
  std::vector<int32_t> ids = prompt_ids;
  for (int step = 0; step < kGreedySteps; ++step) {
    const vllm::ForwardLogits l = forward(ids);
    const int64_t last = static_cast<int64_t>(ids.size() - 1) *
                         static_cast<int64_t>(config.vocab_size);
    const int32_t t = static_cast<int32_t>(
        std::max_element(l.host.begin() + last, l.host.end()) -
        (l.host.begin() + last));
    r.chain.push_back(t);
    r.chain_str += " " + std::to_string(t);
    ids.push_back(t);
  }
  std::fprintf(stderr,
               "[flint-tt-stage2-gate] arm=%s rel_rms=%.3e max_abs=%.3e "
               "chain=[%s ] forward_calls=%lld\n",
               label, r.rel_rms, r.max_abs, r.chain_str.c_str(),
               static_cast<long long>(vllm::FlintTTForwardCalls()));
  return r;
}

}  // namespace

TEST_CASE("flint TT stage-2 gate: the device-attention arm holds the host "
          "arm's drift-tolerant band at model level") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIP: no Tenstorrent device backend");
    return;
  }
  const char* keep_quant = std::getenv("VT_FLINT_KEEP_QUANT");
  const bool keep_quant_on = keep_quant != nullptr && keep_quant[0] == '1';
  REQUIRE(keep_quant_on);  // THE RECIPE IS THE KEEP-QUANT PAGED SERVING RECIPE
                           // — the gate means nothing on another config.
  // The gate owns the switch and pins BOTH arms explicitly — VT_FLINT_TT_DEVICE_ATTN=0
  // for the host shadow, =1 for the device arm — so the case is INVARIANT to
  // the adopted default (unset now means device attention since the
  // FLINT-P150-STAGE2-GATE flip).
  setenv(kDeviceAttnEnv, "0", 1);

  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "FlintForCausalLM");
  reg.factory->parse_config(config);
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kTENSTORRENT);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(
      config, source);
  REQUIRE(model != nullptr);

  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  reg.factory->prepare(*model, config, q);

  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromGguf(gguf);
  const std::vector<int32_t> prompt_ids =
      tok.Encode("The capital of France is");
  REQUIRE(!prompt_ids.empty());

  // THE CPU REFERENCE: the keep-quant arm's registry CPU forward refuses by
  // name, so the reference is the bf16 arm over the SAME quantized container
  // (LoadFlintFromGguf dequantizes at load) — the W0 frame, unchanged.
  const std::unique_ptr<vllm::FlintWeights> cpu_ref_weights =
      std::make_unique<vllm::FlintWeights>(
          vllm::LoadFlintFromGguf(gguf, config));
  std::vector<int32_t> pos(prompt_ids.size());
  for (size_t i = 0; i < pos.size(); ++i)
    pos[i] = static_cast<int32_t>(i);
  const std::vector<float> cpu_first = vllm::FlintModel::Forward(
      prompt_ids, pos, *cpu_ref_weights, config);
  REQUIRE(!cpu_first.empty());

  // THE HOST ARM (env pinned to 0 — the SinksAttention shadow, the
  // same-binary rollback of the adopted default): the same numbers the W0
  // gate measures, re-recorded here so the two arms sit side by side in one
  // log.
  const ArmResult host = RunArm(prompt_ids, cpu_first, config, q, *model,
                                "host");
  CHECK(host.max_abs < kBandMaxAbs);
  CHECK(host.rel_rms < kBandRelRms);

  // THE DEVICE ARM: set BEFORE the forwards — DeviceAttnEnabled() reads the
  // env at FORWARD time (flint_tt.cpp:106), so the same loaded model and
  // device context flip arms in place. The device attention scalars stage
  // LAZILY on this arm's first forward (flint_tt.cpp:618-626); the staging
  // shifts the pool/slot sequence for everything AFTER the host arm — the
  // recorded allocation-order hazard — which is why the host arm runs
  // FIRST here, matching the serving recipes and the pre-adoption gate
  // history.
  setenv(kDeviceAttnEnv, "1", 1);
  const ArmResult device = RunArm(prompt_ids, cpu_first, config, q, *model,
                                  "device");
  unsetenv(kDeviceAttnEnv);

  // THE MODEL-LEVEL GATE: the device arm holds the SAME drift-tolerant band
  // the host arm holds. This is the deliverable; its verdict is the
  // adoption input (spec: flip the default if it holds, keep it OFF and
  // name the gap if not).
  CHECK(device.max_abs < kBandMaxAbs);
  CHECK(device.rel_rms < kBandRelRms);

  // THE CHAINS ARE RECORDED, NOT PINNED (the W0 adjudicated-divergent
  // record): each arm's per-leg chain stays in the log for the evidence.
  (void)host.chain;
  (void)device.chain;
}
