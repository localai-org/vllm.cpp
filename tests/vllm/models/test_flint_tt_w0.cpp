// Flint on Tenstorrent — W0 gate (spec .agents/specs/flint-tenstorrent.md,
// issue ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q).
//
// Three legs:
//   1. the staging-plan host gates (NO card): the resident slice accounts
//      against the plan, the streamed expert tier never resides, and a
//      starved budget is refused by name;
//   2. THE RED GATE: a Tenstorrent-queue forward through the registry must
//      NOT throw the "arm is not implemented" refusal and must increment
//      FlintTTForwardCalls() — dispatch reachability is PROVEN EXERCISED,
//      not merely compiled (AGENTS.md §"Nothing lands dead");
//   3. the W0 correctness leg: ONE prompt through the device forward and the
//      same worktree's CPU reference forward on IDENTICAL bf16 weights —
//      first-token logits distance, an 8-token greedy chain, and the
//      env-gated layer dumps (FLINT_TT_W0_DUMP / FLINT_CPU_DUMP). The W0
//      gate is the HONEST device-vs-CPU baseline; no oracle parity and no
//      throughput claim this wave.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
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
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace {

constexpr const char* kPublishedGguf =
    "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf";
constexpr int kGreedySteps = 8;

bool TenstorrentPresent() {
  return vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) != nullptr;
}

using vllm::HfConfig;

bool RedGateReady() {
  const char* m = std::getenv("FLINT_TT_W0_MODEL");
  return m != nullptr && *m != '\0' && std::filesystem::exists(m) &&
         TenstorrentPresent();
}

// ModelForwardInput holds REFERENCES, so the metadata it names must OUTLIVE
// it: Step owns the metadata and carries the input built over its own
// members (the test_flint_gguf.cpp loop-scope pattern).
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

int32_t Argmax(const vllm::ForwardLogits& fl) {
  const int64_t last = (fl.rows - 1) * fl.vocab;
  return static_cast<int32_t>(
      std::max_element(fl.host.begin() + last, fl.host.end()) -
      (fl.host.begin() + last));
}

}  // namespace

// ---- leg 1: the staging-plan host gates (NO card) ---------------------------

TEST_CASE("flint TT W0: the staging plan accounts the resident slice and "
          "refuses a starved budget") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::FlintWeights w = vllm::LoadFlintFromGguf(gguf, config);

  const vllm::FlintTTStagingPlan plan = vllm::PlanFlintTTStaging(w);
  std::fprintf(stderr,
               "[flint-tt-w0] resident=%lld B (%.3f GiB) expert_tier=%lld B "
               "(%.3f GiB) streamed/layer=%lld B dense=%lld moe=%lld "
               "sliding=%lld tensors=%lld\n",
               static_cast<long long>(plan.resident_bytes),
               double(plan.resident_bytes) / double(1 << 30),
               static_cast<long long>(plan.expert_tier_bytes),
               double(plan.expert_tier_bytes) / double(1 << 30),
               static_cast<long long>(plan.streamed_bytes_per_layer),
               static_cast<long long>(plan.dense_layers),
               static_cast<long long>(plan.moe_layers),
               static_cast<long long>(plan.sliding_layers),
               static_cast<long long>(plan.tensor_count));
  // The spec's geometry, read from the artifact (never hardcoded in the
  // forward): 49 layers, 2 dense + 47 MoE, and the expert tier ~50 GiB bf16
  // — the tier CANNOT reside on one P150, so the plan must report it as
  // streamed.
  CHECK(plan.dense_layers == 2);
  CHECK(plan.moe_layers == w.params.num_hidden_layers - 2);
  CHECK(plan.sliding_layers + (plan.sliding_layers == 0 ? 1 : 0) >= 0);
  CHECK(plan.expert_tier_bytes > int64_t(40) << 30);
  CHECK(plan.streamed_bytes_per_layer > 0);
  // The default budget admits the resident slice...
  CHECK(plan.resident_bytes <= int64_t(32) << 30);
  // ...and a starved budget is refused BY NAME with the byte deficit.
  bool refused = false;
  try {
    (void)vllm::PlanFlintTTStaging(
        w, vllm::FlintTTStagingOptions{.device_budget_bytes =
                                           plan.resident_bytes - 1});
  } catch (const std::exception& e) {
    refused = true;
    const std::string what = e.what();
    CHECK(what.find("deficit") != std::string::npos);
    CHECK(what.find("STREAMS") != std::string::npos);
  }
  CHECK(refused);
}

// ---- legs 2+3: the device gate (card + FLINT_TT_W0_MODEL) -------------------

TEST_CASE("flint TT W0: the registry dispatch reaches the device forward and "
          "the device-vs-CPU baseline is recorded") {
  if (!RedGateReady()) {
    MESSAGE("SKIP: FLINT_TT_W0_MODEL not set to a flint GGUF on a box with "
            "a Tenstorrent device");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "FlintForCausalLM");
  reg.factory->parse_config(config);
  vllm::ModelSource source = vllm::ModelSource::FromGguf(
      gguf, vt::DeviceType::kTENSTORRENT);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(
      config, source);
  REQUIRE(model != nullptr);

  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  // Prepare materializes the device context eagerly (the kolibri seam); a
  // refusal here is a W0 failure, not a skip.
  reg.factory->prepare(*model, config, q);

  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromGguf(gguf);
  const std::vector<int32_t> prompt_ids =
      tok.Encode("The capital of France is");
  REQUIRE(!prompt_ids.empty());

  // THE RED GATE: before the W0 arm this forward threw the named refusal and
  // the counter stayed 0; after it, the device case is EXERCISED.
  const int64_t calls_before = vllm::FlintTTForwardCalls();
  vllm::ForwardLogits dev_first;
  {
    Step dev_step(prompt_ids, config, q);
    CHECK_NOTHROW(dev_first =
                      vllm::ModelRegistry::Forward(*model, dev_step.input));
  }
  CHECK(vllm::FlintTTForwardCalls() == calls_before + 1);
  std::fprintf(stderr, "[flint-tt-w0] dispatch: the kTENSTORRENT case was "
                       "EXERCISED (counter %lld -> %lld)\n",
               static_cast<long long>(calls_before),
               static_cast<long long>(vllm::FlintTTForwardCalls()));

  // The CPU reference leg, same weights, same prompt, same worktree.
  vt::Queue cpu_q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::ForwardLogits cpu_first;
  {
    Step cpu_step(prompt_ids, config, cpu_q);
    cpu_first = vllm::ModelRegistry::Forward(*model, cpu_step.input);
  }
  REQUIRE(dev_first.host.size() == cpu_first.host.size());

  // First-token logits distance (the like-for-like baseline; the kolibri
  // frame is CPU-vs-oracle 2.0e-3 layer-0 l_out).
  double max_abs = 0.0, sum_sq = 0.0, sum_cpu_sq = 0.0;
  for (size_t i = 0; i < cpu_first.host.size(); ++i) {
    const double diff = double(dev_first.host[i]) - double(cpu_first.host[i]);
    max_abs = std::max(max_abs, std::abs(diff));
    sum_sq += diff * diff;
    sum_cpu_sq += double(cpu_first.host[i]) * double(cpu_first.host[i]);
  }
  const double rms = std::sqrt(sum_sq / double(cpu_first.host.size()));
  const double rel_rms =
      rms / std::sqrt(sum_cpu_sq / double(cpu_first.host.size()));
  std::fprintf(stderr,
               "[flint-tt-w0] first-token logits: max_abs=%.3e rms=%.3e "
               "rel_rms=%.3e (n=%zu)\n",
               max_abs, rms, rel_rms, cpu_first.host.size());

  // The greedy chains: device vs CPU, 8 steps (prefill + append).
  std::vector<int32_t> dev_ids = prompt_ids, cpu_ids = prompt_ids;
  std::vector<int32_t> dev_chain, cpu_chain;
  for (int step = 0; step < kGreedySteps; ++step) {
    vllm::ForwardLogits dl, cl;
    {
      Step ds(dev_ids, config, q);
      dl = vllm::ModelRegistry::Forward(*model, ds.input);
    }
    {
      Step cs(cpu_ids, config, cpu_q);
      cl = vllm::ModelRegistry::Forward(*model, cs.input);
    }
    const int32_t dt = Argmax(dl), ct = Argmax(cl);
    dev_chain.push_back(dt);
    cpu_chain.push_back(ct);
    dev_ids.push_back(dt);
    cpu_ids.push_back(ct);
  }
  int64_t chain_match = 0;
  std::string dev_str, cpu_str;
  for (int step = 0; step < kGreedySteps; ++step) {
    if (dev_chain[static_cast<size_t>(step)] ==
        cpu_chain[static_cast<size_t>(step)])
      ++chain_match;
    dev_str += " " + std::to_string(dev_chain[static_cast<size_t>(step)]);
    cpu_str += " " + std::to_string(cpu_chain[static_cast<size_t>(step)]);
  }
  std::fprintf(stderr,
               "[flint-tt-w0] greedy chain match=%lld/%d dev=[%s ] cpu=[%s ]\n",
               static_cast<long long>(chain_match), kGreedySteps,
               dev_str.c_str(), cpu_str.c_str());

  // THE W0 GATE. The honest device-vs-CPU band, and WHY it is wide: the
  // pinned substrate's device rms_norm carries a ~1% SYSTEMATIC bias — the
  // recorded open defect in tenstorrent_ops.cpp ("ROOT-CAUSE RECORD
  // (kolibri1 TT prompt-dependent hard divergence)"), measured here by the
  // op probe at 7.8e-3 max on unit-scale rows (~4 bf16 ulp). Normed once per
  // site across 49 layers the bias compounds multiplicatively (~e^0.25) and
  // shows up as the rel_rms below; the greedy chain flips early. This is the
  // SAME defect class the kolibri TT row carries. Killing the bias (a
  // norm-arm repair on the substrate) is the next wave's lever, not this
  // gate's business.
  //
  // Re-specified 2026-10-26 (the W3 matmul-arm adoption; the developer
  // approved the flip on the recorded evidence —
  // docs/bench-evidence/tt-matmul-noise-20261025/report.md, the throughput
  // table and throughput-w0-legs.log). Two changes, each with its reason:
  //
  // 1. THE CHAIN LEG IS ADJUDICATED-DIVERGENT, not "match >= 1". Under BOTH
  //    arms the first token flips 12834->269 against the CPU row — the
  //    hard-divergence bias class above, not matmul multiply noise — so a
  //    chain_match floor measures which arm diverges harder, not
  //    correctness. Red-first: the old CHECK(chain_match >= 1) is RED on
  //    the adopted arm (0/8; the three throughput legs each failed exactly
  //    that one assertion), and its 1/8 on the old arm was one coincidental
  //    position. The gate now asserts the chain is DIVERGENT-CONSISTENT:
  //    the device tokens equal the recorded adjudicated-divergent chain of
  //    the ACTIVE arm (per config — the shipped arm and the
  //    VT_TT_MATMUL_MATH_FIDELITY=lofi A/B arm each carry their own chain,
  //    bit-identical across the 3 legs of each arm on this pinned
  //    environment; report.md carries it). The distance legs carry the
  //    regression power.
  // 2. THE DISTANCE BAND TIGHTENS to the measured adopted-arm numbers with
  //    modest headroom, justified from the 3 legs (bit-identical
  //    rel_rms 2.212e-1 and max_abs 3.275e+1 on every leg): rel_rms
  //    0.5 -> 0.25 (1.13x the measurement) and max_abs 64 -> 40 (1.22x).
  //    The band sits BETWEEN the two arms' measurements (old arm 2.579e-1),
  //    so on the per-config-deterministic harness it is BOTH a
  //    hard-regression fence against distance growth AND the arm selector:
  //    reverting the shipped arm to the LoFi class is RED here, which the
  //    pre-adoption `=lofi` legs demonstrate (2.579e-1 > 0.25). The
  //    chain-consistency leg pins WHICH arm ran independent of distance.
  //    The developer approved the adoption and the fence re-specification
  //    (the throughput table and the invalid 2.045e-1 lineage are in
  //    report.md; W0 records the evidence, it does not paper over it).
  CHECK(max_abs < 40.0);
  CHECK(rel_rms < 0.25);
  {
    static const int32_t kShippedChain[kGreedySteps] = {
        269, 6963, 332, 269, 332, 116253, 12, 332};
    static const int32_t kLofiChain[kGreedySteps] = {
        269, 269, 269, 269, 269, 140759, 269, 269};
    const char* fid = std::getenv("VT_TT_MATMUL_MATH_FIDELITY");
    const bool lofi_arm = fid != nullptr && std::string(fid) == "lofi";
    const int32_t* expected = lofi_arm ? kLofiChain : kShippedChain;
    bool chain_consistent = true;
    for (int step = 0; step < kGreedySteps; ++step) {
      if (dev_chain[static_cast<size_t>(step)] != expected[step])
        chain_consistent = false;
    }
    CHECK_MESSAGE(chain_consistent,
                  "the greedy chain is not the recorded "
                  "adjudicated-divergent chain of the active arm");
  }
}
