// Flint on Tenstorrent (Blackhole P150) — W0 device forward declarations.
//
// (MODEL-TEXT-FLINT device arm, spec .agents/specs/flint-tenstorrent.md,
// issue ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q.) The kolibri1 TT campaign is
// the pattern: one registry, a kTENSTORRENT case, a device context built at
// prepare, and a device forward the run PROVES it reached (FlintTTForwardCalls
// counts entries, not compilations).
//
// W0 POLARITY. The device legs are embedding, every RMSNorm, and every linear
// projection (q/k/v/o, dense MLP, router, shared expert, routed experts,
// lm_head) over bf16-resident weights. The host legs — per-head q/k RMSNorm,
// the partial-RoPE pattern, the sinks softmax with the sliding window,
// sigmoid top-k routing, the MoE combine, and the residual adds — are named
// in the spec and owed to the next wave. The routed-expert tier (~50 GiB
// bf16) cannot reside on one P150 (the kolibri 384-expert wall again), so
// the selected experts STREAM from the host bytes per layer.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/flint.h"

namespace vt {
class Backend;
struct Queue;
}

namespace vllm {

struct FlintTTStagingOptions {
  // P150 DRAM per chip. Overridable so tests exercise the refusal.
  int64_t device_budget_bytes = int64_t(32) << 30;
};

// Device-free byte accounting over the bf16 FlintWeights: the RESIDENT slice
// (embed, lm_head, per-layer attention projections, norms, dense MLPs, shared
// experts, router) and the STREAMED expert tier. The resident slice must fit
// the budget; the expert tier never resides, so it is reported, never refused.
struct FlintTTStagingPlan {
  int64_t resident_bytes = 0;
  int64_t expert_tier_bytes = 0;      // the full routed tier (never staged)
  int64_t streamed_bytes_per_layer = 0;  // top_k experts' 3 projections
  int64_t tensor_count = 0;
  int64_t dense_layers = 0;
  int64_t moe_layers = 0;
  int64_t sliding_layers = 0;
};

// Computes the plan. Throws std::runtime_error when the resident slice
// exceeds the device budget (named with the byte deficit).
FlintTTStagingPlan PlanFlintTTStaging(
    const FlintWeights& weights,
    const FlintTTStagingOptions& options = {});

// Opaque device context: the staged resident weights plus the queue they
// were staged on. Built once at prepare on a Tenstorrent queue. The impl is
// wide open within vllm/ (the forward reads the staged buffers); the type is
// still the ownership seam — exactly one builder, exactly one owner.
class FlintTTDeviceContext {
 public:
  struct Impl;
  FlintTTDeviceContext();
  ~FlintTTDeviceContext();
  FlintTTDeviceContext(FlintTTDeviceContext&&) noexcept;
  FlintTTDeviceContext& operator=(FlintTTDeviceContext&&) noexcept;
  FlintTTDeviceContext(const FlintTTDeviceContext&) = delete;
  FlintTTDeviceContext& operator=(const FlintTTDeviceContext&) = delete;

  // The staged resident bytes (accounting, for evidence).
  int64_t resident_bytes() const;

  std::unique_ptr<Impl> impl_;
};

// Builds the context: uploads the resident slice on `queue`'s device.
std::unique_ptr<FlintTTDeviceContext> BuildFlintTTDeviceContext(
    vt::Backend& backend, vt::Queue& queue, const FlintWeights& weights);

// The W0 device forward: ONE contiguous sequence, the same contract the CPU
// reference arm enforces. Returns [t, vocab] f32 logits.
std::vector<float> ForwardFlintTT(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& positions, const FlintWeights& weights,
    FlintTTDeviceContext& ctx);

// The dispatch-reachability counter: every ForwardFlintTT entry increments
// it. A test asserts >= 1 after a device forward so the registry case is
// proven EXERCISED, not merely compiled.
int64_t FlintTTForwardCalls();

// The keep-quant dispatch counter: every block-carrying weight MatmulWeight
// routes to vt::MatmulBT (-> kMatmulBTQuant) increments it. The keep-quant
// gate asserts > 0 after a device forward — blocks CONSUMED, not carried;
// the bf16 arm asserts == 0 (no block dispatch on that config).
int64_t FlintTTQuantDispatchCalls();

}  // namespace vllm
