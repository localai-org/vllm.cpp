// Kolibri-1 — Tenstorrent B2b-ii streaming MoE, the host half
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md
// ### B2 scope — B2b addendum, slice ii; issue
// ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN).
//
// Slice ii replaces the B2b-i routed-expert refusal with the streaming
// path: FP8_E4M3 slot buffers on device consuming
// `Kolibri1TTExpertSlotPolicy` verbatim, the routed path end to end
// (router readback -> host remap -> fetch list -> fetch executor fills
// the slots -> Touch()), slot swaps on the ContentChangedSince reset
// lane, and the B1 per-token stream bound asserted at RUNTIME.
//
// This header is the HOST half, device-free and testable without a card
// (the addendum's device-free test lane):
//
//  - `PlanKolibri1TTSlotPool`: the device slot-pool plan. One slot = one
//    (layer, expert) pair; the slot's FP8 payload is the three routed
//    projections' `Fp8BlockWeight::packed` bytes VERBATIM, row-major,
//    concatenated gate/up/down; the f32 scale grids stay HOST-side and
//    stage with the slot (§ B2 scope). The per-layer slot capacity is
//    the B1 streaming plan's hot_experts divided across the layers —
//    the policy itself is built through
//    `PlanKolibri1TTExpertSlotPolicy` UNCHANGED, against the per-layer
//    residual share, so the TOTAL pool (layers x per-layer pool) still
//    fits the B1 device residual by construction and the policy's own
//    refusal fires on any over-request.
//  - `BuildKolibri1TTSlotFetchList`: the fetch executor's host half —
//    the logical-expert -> slot remap over the B2a dispatch plan, the
//    fetch-list construction (host source pointers per slot), and the
//    byte accounting against the B1 per-token stream bound. A step
//    whose stream_bytes EXCEEDS the bound throws LOUDLY by name — it
//    never silently degrades.
//  - `Kolibri1TTStreamBoundGuard`: the per-step accumulator the forward
//    charges each layer's fetch against; same loud refusal.
//  - `Kolibri1TTSlotShadow`: the host-side shadow of the staged slot
//    bytes — the readback pivot for slot-content verification (this
//    tt-metal has NO extract_shard, addendum lines 396-404), and the
//    eviction-hook integration that keeps the shadow exact across slot
//    swaps.
//  - `Kolibri1TTRouterLogitsReadback`: the router readback dtype pivot
//    (the f32 logits row downloads whole; no shard extraction), named
//    so the readback path is a contract and not an inline detail.
//
// The reset lane: `Kolibri1TTExpertSlotPolicy::Fingerprint`/
// `ContentChangedSince` (kolibri1_tt.h:233) drive the graph reset
// exactly as the GDN churn fix (qwen3_5.cpp TT-GDN-SLOT-CHURN,
// ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W, #3404). The forward records
// the fingerprint when it builds its step state and re-checks it every
// step; a swap flips the predicate and invalidates the memoized slot
// dequants. `Kolibri1TTSlotEpoch` is that recording, device-free.
#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_weights.h"

namespace vllm {

// ---- The device slot-pool plan ----------------------------------------------

struct Kolibri1TTSlotPoolPlan {
  // The policy, built UNCHANGED through PlanKolibri1TTExpertSlotPolicy
  // against the per-layer residual share.
  Kolibri1TTExpertSlotPolicy policy;
  int64_t layers = 0;
  // Per-layer slot capacity == policy.capacity == the B1 hot_experts
  // divided across the layers (>= 1; a plan that cannot afford one slot
  // per layer refuses by name).
  int64_t capacity_per_layer = 0;
  // One slot's FP8 payload (the 3 packed projections concatenated,
  // verbatim) and its HOST-side f32 scale grids.
  int64_t packed_bytes_per_slot = 0;
  int64_t scale_bytes_per_slot = 0;
  // Per-layer device pool == policy.slot_pool_bytes; the TOTAL device
  // pool; the total host-side scale bytes.
  int64_t per_layer_pool_bytes = 0;
  int64_t total_pool_bytes = 0;
  int64_t total_scale_bytes = 0;
};

// Plans the slot pool over fully-loaded kolibri1 weights. Refuses by
// name when:
//  - a routed expert weight is missing or not fp8-block,
//  - the per-expert byte math (3 packed projections + 3 f32 scale
//    grids) does not sum to the shape's expert_bytes,
//  - the residual cannot afford ONE slot per layer.
Kolibri1TTSlotPoolPlan PlanKolibri1TTSlotPool(
    const Kolibri1Weights& weights, const Kolibri1TTStreamingPlan& streaming);

// The routed expert's slot payload view over the loaded weights: the
// concatenated packed bytes (gate, up, down — row-major, verbatim) and
// the concatenated f32 scale grids. The views point INTO the loaded
// weights (no copy); they stay valid for the weights' lifetime.
struct Kolibri1TTExpertSlotPayload {
  const uint8_t* packed_host = nullptr;
  int64_t packed_bytes = 0;
  const float* scale_host = nullptr;
  int64_t scale_bytes = 0;
};

Kolibri1TTExpertSlotPayload Kolibri1TTRoutedExpertPayload(
    const Kolibri1Weights& weights, int64_t layer, int64_t expert);

// ---- The fetch executor's host half -----------------------------------------

struct Kolibri1TTSlotFetchJob {
  int64_t layer = 0;
  int64_t expert = 0;
  int64_t slot = 0;
  // The host source (the loaded weights) and the device destination
  // offset within the layer's pool (slot * packed_bytes_per_slot).
  const uint8_t* packed_host = nullptr;
  int64_t packed_bytes = 0;
  const float* scale_host = nullptr;
  int64_t scale_bytes = 0;
  int64_t device_offset = 0;
};

struct Kolibri1TTSlotFetchList {
  std::vector<Kolibri1TTSlotFetchJob> jobs;
  int64_t stream_bytes = 0;  // == the dispatch plan's stream_bytes
};

// Builds the fetch list for one layer's dispatch: the missed experts in
// dispatch order, each with its suggested slot, host payload, and device
// offset. `per_token_stream_bytes` is the B1 ceiling; a dispatch whose
// stream_bytes exceeds it throws LOUDLY by name (never a silent
// degradation) — the guard below is the per-step accumulator for the
// whole-token bound.
Kolibri1TTSlotFetchList BuildKolibri1TTSlotFetchList(
    const Kolibri1TTExpertSlotPolicy& policy,
    const Kolibri1TTLayerDispatch& dispatch,
    const Kolibri1Weights& weights,
    const Kolibri1TTSlotPoolPlan& pool,
    int64_t per_token_stream_bytes);

// The per-step stream-bound accumulator: the forward charges every
// layer's fetch; the whole token must stay under the B1
// per_token_stream_bytes. Exceeding it throws by name — the loud
// failure the addendum requires.
class Kolibri1TTStreamBoundGuard {
 public:
  explicit Kolibri1TTStreamBoundGuard(int64_t per_token_stream_bytes)
      : bound_(per_token_stream_bytes) {}
  void Charge(int64_t layer, int64_t bytes);  // throws on exceed
  int64_t charged() const { return charged_; }
  int64_t bound() const { return bound_; }

 private:
  int64_t bound_ = 0;
  int64_t charged_ = 0;
};

// ---- The slot shadow (the readback pivot's reference side) -------------------

// Host-side shadow of the staged slot bytes. The device readback of a
// slot's payload is verified BYTE-EXACT against this shadow (no
// extract_shard on this tt-metal — the shadow IS the pivot's reference).
// The eviction hook keeps it exact across slot swaps: the policy's
// SetEvictHook clears the evicted slot's shadow and the reset-lane
// fingerprint is re-recorded by the forward.
class Kolibri1TTSlotShadow {
 public:
  Kolibri1TTSlotShadow(const Kolibri1TTExpertSlotPolicy& policy,
                       int64_t packed_bytes_per_slot);
  ~Kolibri1TTSlotShadow();

  // Installs the eviction hook on the policy (one hook at a time).
  void AttachEvictHook();
  // Records the staged bytes for (layer, slot). Overwrites.
  void Record(int64_t layer, int64_t slot, const uint8_t* packed,
              int64_t bytes);
  // Byte-exact comparison of a device readback against the shadow.
  // Throws by name on any mismatch (a silent diff would defeat the
  // verification). Returns the bytes compared.
  int64_t VerifyReadback(int64_t layer, int64_t slot,
                         const uint8_t* packed, int64_t bytes) const;
  bool Has(int64_t layer, int64_t slot) const;
  // Number of slots carrying a shadow (the fill count, for the gates).
  int64_t FilledSlots() const { return filled_; }

 private:
  void Clear(int64_t layer, int64_t slot);
  const Kolibri1TTExpertSlotPolicy* policy_ = nullptr;
  int64_t packed_bytes_per_slot_ = 0;
  std::vector<std::vector<std::vector<uint8_t>>> bytes_;
  int64_t filled_ = 0;
};

// ---- The reset-lane recording ------------------------------------------------

// The graph-reset recording for one forward instance: the fingerprint
// taken when the step state was built, re-checked with
// ContentChangedSince every step. A slot swap flips it; an identical
// re-selection does not (the GDN churn semantics).
struct Kolibri1TTSlotEpoch {
  uint64_t fingerprint = 0;
  int64_t resets = 0;  // diagnostics: how often the predicate fired

  void Record(const Kolibri1TTExpertSlotPolicy& policy) {
    fingerprint = policy.Fingerprint();
  }
  // True when the policy content changed since the recording — the
  // caller must reset the memoized slot dequants (the graph-reset lane).
  bool ConsumeIfChanged(const Kolibri1TTExpertSlotPolicy& policy) {
    if (!policy.ContentChangedSince(fingerprint)) return false;
    ++resets;
    fingerprint = policy.Fingerprint();
    return true;
  }
};

// ---- The router readback pivot ------------------------------------------------

// The router readback on this tt-metal: the f32 logits buffer downloads
// WHOLE (a dtype pivot to f32 at the GEMM, then a flat D2H copy — the
// b2bi precedent). There is NO extract_shard on this tt-metal
// (addendum lines 396-404): any per-shard readback is refused by name.
// `download(count, dst)` performs the flat D2H copy of `count` f32
// values; this wrapper is the named contract the forward and the tests
// both go through.
template <typename DownloadFn>
std::vector<float> Kolibri1TTRouterLogitsReadback(DownloadFn download,
                                                  int64_t count) {
  if (count <= 0) {
    throw std::runtime_error(
        "kolibri1-tt B2b-ii: router readback of a non-positive count — "
        "refusing instead of returning an empty readback");
  }
  std::vector<float> out(static_cast<size_t>(count));
  download(count, out.data());
  return out;
}

}  // namespace vllm
