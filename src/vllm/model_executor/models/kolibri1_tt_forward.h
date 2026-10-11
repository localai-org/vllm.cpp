// Kolibri-1 — Tenstorrent B2b-i dense-resident device forward (private
// header, MODEL-TEXT-kolibri-1-tenstorrent, spec
// .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i; issue
// ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D).
//
// Slice i runs the RESIDENT non-expert set entirely on device with the
// routed-expert tier ABSENT: attention (hybrid geometry — 40 sliding-window
// layers at window 513, 10 full-attention RNoPE layers, two-group KV,
// per-head q/k RMS norms), the four sandwich norms, the router (bf16
// [384,2560] gate, f32 e_score_correction_bias, sigmoid-logit-add,
// top-6-of-384, the CPU row's f32 compute path), the shared expert, and
// embed + untied lm_head, with on-device sampling through the landed decode
// seam (ModelRegistry::Forward, vt::GreedyArgmax). The routed path is
// REFUSED BY NAME (B2b-ii owns it); the refusal fires whenever the router
// selects routed experts and the decode proceeds with the shared expert
// only — which is why the golden chains (full-model decodes) cannot be
// replayed in this slice and the 141/145 token gate stays owed until
// B2b-ii.
//
// The forward mirrors the CPU row's op sequence op-for-op
// (kolibri1_forward.cpp) — the same vt ops, the same order, the same
// shapes — over the resident slice. The ONE deliberate compute difference:
// the CPU row dequants each fp8-block projection per call (its documented
// R1 disposition); this slice dequants ONCE per weight into a
// device-resident bf16 buffer (the same DequantRowsBf16 bytes, memoized —
// the b2i bring-up verified the device GEMM consumes exactly these bf16
// dequants of the byte-verified staged fp8). The device fp8-block GEMM that
// would consume the staged FP8_E4M3 operands in tile layout is the B2b
// COMPUTE wave's, not this slice's (addendum § FP8/trace constraints).
//
// Backend-agnostic TU: it uses only vt ops and the shared residency seam
// (dense_attn::ResidentWeight), so it compiles in every build; it REFUSES
// a non-Tenstorrent queue by name at its boundary.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/kolibri1_tt_stream.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vt/backend.h"
#include "vt/tensor.h"

namespace vllm {

// ---- The B2b-i routed-expert refusal (slice i's named refusal) -------------
//
// The router's top-6-of-384 output always requests ROUTED experts; in this
// slice the routed tier is absent by design, so the MoE block fires this
// refusal BY NAME (counted, message printed once per process) and the step
// proceeds with the shared expert only. The refusal is an observable,
// counted event — NOT a throw — because the slice's completion condition is
// one greedy decode COMPLETING on the card with the refusal active (the
// goldens cannot be replayed without the routed experts; the 141/145 token
// gate stays owed to B2b-ii, per the addendum's gate ordering).

// The refusal message for one layer's router request: names the missing
// part, the owning slice, the row, and the issue. Pure function of its
// inputs (host-side testable, no card).
std::string Kolibri1TTRoutedExpertRefusalMessage(
    int64_t layer, const std::vector<int32_t>& requested_ids);

// Process-wide count of fired routed-expert requests (one per MoE block per
// step — layers x steps for a full decode). Read by the gate to show the
// refusal firing by name in the gate configuration.
int64_t Kolibri1TTRoutedExpertRefusalCount();
// Test seam: zeroes the counter (and the once-per-process message latch).
void Kolibri1TTResetRoutedExpertRefusalCount();

// ---- The device-resident compute context -----------------------------------
//
// The memoized bf16 dequants of the resident slice's fp8-block projections
// (attention q/k/v/o + the shared expert's gate/up/down), dequanted ONCE
// per weight with the CPU row's DequantRowsBf16 (the R1 disposition) into
// backend allocations held for the model's lifetime. The bf16 modules
// (router gate, norms, embed, lm_head, q/k norms, router bias) ride the
// shared dense_attn::ResidentWeight residency seam instead (memoized in the
// OwnedTensor's d_dev), exactly like every other device model.
struct Kolibri1TTResidentDeviceContext {
  struct Layer {
    vt::Tensor q, k, v, o;               // attention [N, K] bf16
    vt::Tensor sh_gate, sh_up, sh_down;  // shared expert [N, K] bf16
  };
  std::vector<Layer> layers;
  // Owns the dequant allocations the layer views point into (freed through
  // the backend when the context dies, in order).
  std::vector<std::shared_ptr<void>> keepalive;

  int64_t projections = 0;      // fp8 projections dequanted (350 on the real ckpt)
  int64_t uploaded_bytes = 0;   // bf16 bytes uploaded to the backend
  double build_seconds = 0.0;   // wall clock of the one-time build
  bool built = false;
};

// Builds the context: dequants every resident fp8-block projection host-side
// (threaded over output rows through the ONE pool, bit-identical to the CPU
// row's per-call dequant by the pool determinism contract) into backend
// allocations. Throws std::runtime_error on a malformed projection. The
// returned context owns its allocations (freed through the backend).
std::unique_ptr<Kolibri1TTResidentDeviceContext>
BuildKolibri1TTResidentDeviceContext(vt::Backend& backend, vt::Queue& queue,
                                     const Kolibri1Weights& weights);

// The checked accessor for the device context a ModelRegistry::Load +
// prepare produced: the B2b device waves (and their gates) read the model's
// context back out through this seam instead of re-building it, so the
// per-op agreement battery verifies the PRODUCTION residency the decode
// runs. Builds lazily on first use (prepare builds it eagerly). Refuses
// by name when `model` is not a Kolibri1ForCausalLM load.
Kolibri1TTResidentDeviceContext& Kolibri1LoadedModelTTContext(
    LoadedModel& model, vt::Queue& queue);

// ---- The B2b-ii streaming-MoE device context (slice ii) ---------------------

// The device half of the streaming MoE: the FP8_E4M3 slot pool staged on
// the card (capacity per layer = the pool plan's capacity_per_layer; the
// slot payload = the three routed projections' packed bytes VERBATIM,
// row-major, concatenated gate/up/down; the f32 scale grids stay
// HOST-side), the slot shadow (the readback pivot's reference), the
// per-step stream-bound guard, and the reset-lane epoch. Built once per
// model over the B1 streaming plan's refusals (inherited unchanged).
struct Kolibri1TTStreamingDeviceContext {
  Kolibri1TTSlotPoolPlan pool;
  // Per-layer pool bases (backend allocations, freed through the backend
  // in order).
  std::vector<void*> pool_base;
  std::vector<std::shared_ptr<void>> keepalive;
  // The slot shadow is built against the policy INSIDE `pool`; it must
  // be constructed after the pool and owns the eviction hook.
  std::unique_ptr<Kolibri1TTSlotShadow> shadow;
  Kolibri1TTSlotEpoch epoch;
  int64_t per_token_stream_bytes = 0;  // the B1 ceiling this context enforces
  // The per-step stream-bound accumulator; re-created at each forward
  // step (one guard charge window per token).
  std::unique_ptr<Kolibri1TTStreamBoundGuard> guard;

  // Gate/evidence counters.
  int64_t slot_fills = 0;              // fetches staged (bytes below)
  int64_t staged_bytes = 0;            // fp8 bytes staged host -> device
  int64_t readback_verified_bytes = 0; // slot bytes verified vs the shadow
  int64_t swap_fills = 0;              // fills that evicted a resident
  int64_t memo_hits = 0;               // dequant memo hits
  int64_t dram_free_before_staging = 0;
  int64_t dram_free_after_staging = 0;
  bool built = false;

  // The memoized bf16 dequants of resident slots, keyed (layer, expert).
  // The reset lane clears it whenever ContentChangedSince fires; a memo
  // entry is also invalid when the expert's slot moved.
  struct SlotDequant {
    int64_t slot = -1;
    std::shared_ptr<void> owner;
    vt::Tensor gate, up, down;  // bf16 [n, k] device views
  };
  std::map<std::pair<int64_t, int64_t>, SlotDequant> memo;
};

// Builds the streaming context: the B1 streaming plan over the REAL
// checkpoint's byte shape (device_budget_bytes, kv_reserve_bytes from the
// options), the slot pool plan, and the per-layer device pool allocations.
// Throws std::runtime_error on every inherited refusal (concurrency,
// host tier, device residual) and on a pool that cannot be allocated.
std::unique_ptr<Kolibri1TTStreamingDeviceContext>
BuildKolibri1TTStreamingDeviceContext(vt::Backend& backend, vt::Queue& queue,
                                      const Kolibri1Weights& weights,
                                      int64_t device_budget_bytes,
                                      int64_t kv_reserve_bytes);


// The checked accessor for the streaming context a load produced (null
// when VT_KOLIBRI1_TT_B2II_STREAM=0). The device gates read the
// streaming counters through this seam.
Kolibri1TTStreamingDeviceContext* Kolibri1LoadedModelTTStreamContext(
    LoadedModel& model, vt::Queue& queue);

// ---- The B2b-i forward ------------------------------------------------------

// Runs one forward step of the dense-resident slice on a Tenstorrent queue:
// embedding -> N sandwich-norm layers (sliding: RoPE + window; full: RNoPE;
// GQA with per-head qk-norm) -> MoE on EVERY layer (router on device, f32
// sigmoid-logit-add top-6-of-384, the routed-expert refusal fired BY NAME,
// the shared expert computed) -> final norm -> untied lm_head. `ctx` is the
// model's device-resident compute context (built once; see above).
//
// `multi_kv` resolves each layer's PagedKvCache by layer name; null falls
// back to `attn_kv[layer]`. Logits gather follows `logits_indices` when it
// is a strict subset of the step. Refuses a non-Tenstorrent queue by name
// (the CPU row owns the CPU arm; every other device is owed).
ForwardLogits ForwardKolibri1TTResidentForward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Kolibri1Weights& weights,
    const MultiKvCacheIndex* multi_kv, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices,
    Kolibri1TTResidentDeviceContext& ctx,
    Kolibri1TTStreamingDeviceContext* streaming = nullptr);

// ---- SCRATCH DEBUG (layer-1 moe localization): the router diagnostic -------
//
// The CPU arm stashes its last router logits in
// `g_kolibri1_dbg_route_logits` (kolibri1_forward.cpp) tagged with the
// layer it came from in `g_kolibri1_dbg_route_layer`; the same pattern
// carries the layer-0 dhn stash. The TT arm's diagnostic compares
// against the CPU reference ONLY when the layer identity AND the element
// count match — a process-global "last CPU layer" alone cannot establish
// layer-by-layer parity, and an unconditional index into the global
// reads out of bounds when the TT arm runs alone (the global starts
// empty) or after a shorter CPU batch.
}  // namespace vllm

// Defined at file scope in kolibri1_forward.cpp (global namespace).
extern std::vector<float> g_kolibri1_dbg_route_logits;
extern int64_t g_kolibri1_dbg_route_layer;
extern std::vector<uint16_t> g_kolibri1_dbg_dhn;
extern int64_t g_kolibri1_dbg_dhn_layer;

namespace vllm {

struct Kolibri1DebugRouteDiff {
  double maxdiff_dev = 0;  // device logits vs the host reference (same inputs)
  double maxdiff_cpu = 0;  // device logits vs the CPU arm's stashed reference
  bool cpu_compared = false;
};

// The MoeBlock router diagnostic: the HOST-REFERENCE leg (device values vs
// host-computed reference logits for the SAME inputs) stands on its own and
// is bounds-checked over the two local vectors. The optional CPU leg runs
// only when the stashed CPU reference exists, its layer identity matches
// `layer`, and its size matches `logits`. Never indexes the CPU global
// without those checks.
Kolibri1DebugRouteDiff Kolibri1DebugRouteCompare(
    int64_t layer, const std::vector<float>& logits,
    const std::vector<float>& href);

}  // namespace vllm
