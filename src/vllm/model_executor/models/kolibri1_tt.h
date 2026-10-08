// Kolibri-1 — Tenstorrent (Blackhole P150) weight-staging plan (wave A).
//
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md,
// issue ISSUE-LOCAL-01M49K2SN7E3EH60K2FK7T88JX.) The registry entry stays
// the single landed REGISTER_VLLM_MODEL in kolibri1_registry.cpp — the
// registry is device-agnostic and the TT forward dispatch is a later wave.
// This TU owns the DEVICE-FREE half of the TT port: the per-tensor staging
// decisions over the landed Kolibri1Weights, the byte accounting against a
// P150's 32 GiB, and the refusals the plan must raise before any device is
// touched.
//
// FP8 DECISION (spec §FP8): the fp8-block projections stage NATIVE
// FP8_E4M3 — bytes verbatim from Fp8BlockWeight::packed, row-major (the
// tt-metal pin constrains FP8_E4M3 to RM at tensor creation,
// ttnn/core/tensor/py_to_tt_tensor.cpp:58-59) — and the f32 scale grids
// stay beside them. Blackhole LLK support: data_format_derive.h:81,
// llk_pack_tile_api.h:75, tensix_types.h:230. Dequant-at-load is rejected:
// it doubles the staged bytes and the P150 consumes fp8 natively.
//
// THE 384-EXPERT WALL (spec §math): the routed experts alone are
// 50 × 384 × 3,933,120 B ≈ 70.33 GiB and the whole model ≈ 73.6 GiB —
// a single 32 GiB P150 cannot hold it. PlanKolibri1TTStaging computes the
// accounting per tensor and REFUSES a single-device full-model plan by
// name with the byte deficit; the mesh/expert-parallel layout is owed to
// the device wave and is deliberately not decided here.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kolibri1.h"
#include "vllm/model_executor/models/kolibri1_weights.h"

namespace vllm {

// Device dtype of one staged tensor.
enum class Kolibri1TTDType {
  kFp8E4M3,  // native: fp8 bytes verbatim, f32 scale grid beside
  kBf16,     // router gate, norms, embed, lm_head (bf16 in the checkpoint)
  kF32,      // the router's e_score_correction_bias (widened at load)
};

// One staged tensor with its device dtype and on-device byte cost
// (weights only; scale grids are accounted separately in the plan totals).
struct Kolibri1TTTensor {
  std::string name;
  Kolibri1TTDType dtype;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bytes = 0;  // device bytes for the packed operand
  int64_t scale_bytes = 0;  // f32 scale grid bytes (0 for bf16 tensors)
  // Where the staged bytes live on the host (device-free bookkeeping).
  // The B2b-i resident planner fills them for every tensor it plans and
  // the device staging consumes exactly these bytes; the wave-A planner
  // fills them on the projections it shares helpers with.
  const void* host = nullptr;        // packed operand bytes, verbatim
  const void* scale_host = nullptr;  // f32 scale grid bytes (fp8 only)
  int64_t scale_rows = 0;
  int64_t scale_cols = 0;
};

struct Kolibri1TTStagingPlan {
  Kolibri1Params params;
  std::vector<Kolibri1TTTensor> tensors;

  int64_t total_bytes = 0;        // packed operands + scale grids
  int64_t fp8_bytes = 0;          // subset staged FP8_E4M3
  int64_t bf16_bytes = 0;         // subset staged BF16
  int64_t tensor_count = 0;

  // The hybrid geometry this plan asserts: 10 full-attention (RNoPE) and
  // 40 sliding-window layers, window 513, uniform head_dim 128.
  int64_t full_layers = 0;
  int64_t swa_layers = 0;
};

struct Kolibri1TTStagingOptions {
  // P150 DRAM per chip. Overridable so tests can exercise the refusal.
  int64_t device_budget_bytes = int64_t(32) << 30;
  // 0 = single device. >0 = mesh chip count; the plan divides NOTHING
  // automatically — wave A only verifies the plan FITS the mesh total and
  // records per-chip accounting as owed (the sharding layout is the device
  // wave's decision).
  int64_t mesh_chips = 0;
};

// Plans the TT staging of fully-loaded kolibri1 weights: every fp8-block
// projection → FP8_E4M3 native (bytes from Fp8BlockWeight::packed), every
// bf16 module → BF16. Asserts the checkpoint's hybrid geometry (two KV
// groups, full-rotary head_dim 128) and refuses:
//  - a weights tree with an empty or non-fp8 expert (an incomplete load),
//  - a single-device plan whose total exceeds the device budget (the
//    refusal names the byte deficit and the owed mesh row),
//  - a mesh plan that exceeds mesh_chips × device_budget_bytes.
// Throws std::runtime_error on every refusal.
Kolibri1TTStagingPlan PlanKolibri1TTStaging(
    const Kolibri1Weights& weights,
    const Kolibri1TTStagingOptions& options = {});

// ---- Wave B1: single-P150 expert streaming (host-side policy only) ----
//
// The full routed-expert tier (50 × 384 × 3,933,120 B ≈ 70.33 GiB) cannot
// reside on one P150, so decode STREAMS experts from a host-RAM tier keyed
// on router output (spec §"Single-P150 expert streaming — design"). This
// plan records the byte policy of that mode and refuses infeasible
// operating points before any device or tier is touched. Device-free: no
// allocation, no slot store, no ttnn — those are B2/B3 (spec §"Out").

// The checkpoint geometry the streaming policy accounts over. Defaults are
// the shipped kolibri1 checkpoint's byte math (spec § byte-math table);
// every field is overridable so tests can exercise the refusals.
struct Kolibri1TTStreamingShape {
  int64_t layers = 50;
  int64_t experts = 384;
  int64_t topk = 6;
  // One routed expert, one layer: fp8 bytes + f32 scale grids
  // (3×1,310,720 + (4×20 + 4×20 + 20×4)×4 = 3,933,120).
  int64_t expert_bytes = 3933120;
  int64_t attention_bytes = 1703936000;      // 50 × 34,078,720, always resident
  int64_t shared_expert_bytes = 196656000;   // 50 × 3,933,120, always resident
  int64_t router_bytes = 98304000;           // 50 × bf16 [384,2560]
  int64_t norm_bytes = 2 << 20;              // ~2 MiB
  int64_t embed_head_bytes = 1310720000;     // bf16 [128000,2560] × 2
};

struct Kolibri1TTStreamingOptions {
  // P150 DRAM per chip.
  int64_t device_budget_bytes = int64_t(32) << 30;
  // Host-RAM tier capacity for the full routed-expert tier. The NVMe tier
  // is a pluggable leaf B2/B3 owe; a host refusal NAMES it, never spills.
  int64_t host_budget_bytes = int64_t(96) << 30;
  // KV cache bytes reserved on device before the hot set is sized.
  int64_t kv_reserve_bytes = 0;
  // Decode concurrency operating point (concurrent tokens per step).
  int64_t concurrency = 1;
  // Refuse (or warn, best-effort) when the expected per-layer touched
  // fraction of the expert space exceeds this.
  double touched_fraction_threshold = 0.25;
  // true = warn in the plan instead of refusing at high concurrency.
  bool best_effort = false;
};

struct Kolibri1TTStreamingPlan {
  // Resident non-expert components (attention, shared expert, router,
  // norms, embed + untied head).
  int64_t resident_bytes = 0;
  // device budget − resident − KV reserve.
  int64_t device_residual_bytes = 0;
  // Hot set derived from the residual: floor(residual / expert_bytes).
  int64_t hot_experts = 0;
  int64_t hot_set_bytes = 0;
  // The full routed-expert tier the host tier must hold.
  int64_t host_required_bytes = 0;
  // layers × topk × expert_bytes: the per-token miss-stream bound.
  int64_t per_token_stream_bytes = 0;
  // At 60 GB/s and 25 GB/s (derived numbers, not measurements).
  double stream_ms_low = 0.0;
  double stream_ms_high = 0.0;
  // Expected distinct-expert fraction touched per layer:
  // 1 − (1 − topk/experts)^concurrency.
  double touched_fraction_per_layer = 0.0;
  // Non-empty only in best-effort mode above the threshold.
  std::string concurrency_warning;
};

// Computes the streaming byte policy and refuses:
//  - resident components alone exceeding the device budget (names the
//    deficit),
//  - the full routed-expert tier exceeding the host budget (names the
//    deficit AND the NVMe leaf as owed — never a silent spill),
//  - a concurrency whose touched fraction exceeds the declared threshold
//    (names the (1−f) × routed-bytes per-step I/O consequence), unless
//    best_effort asks for a warning instead.
// Throws std::runtime_error on every refusal.
Kolibri1TTStreamingPlan PlanKolibri1TTStreaming(
    const Kolibri1TTStreamingShape& shape,
    const Kolibri1TTStreamingOptions& options = {});

// ---- Wave B2a: the TT-native expert slot store policy, the MoE dispatch
// plan, and the graph-reset lane (device-free; spec §"B2 scope") ----
//
// The device tier of the streaming design: a fixed-capacity pool of
// FP8_E4M3 expert slot buffers (packed bytes verbatim, row-major; scale
// grids f32 host-side). This is the POLICY half only — slot tables,
// eviction, dispatch, and the reset predicate. No allocation, no ttnn:
// the device buffers themselves are B2b.

struct Kolibri1TTSlotPolicyOptions {
  int64_t layers = 50;
  int64_t experts = 384;
  // One slot = one expert, one layer (fp8 + f32 scale grids, § byte math).
  int64_t expert_bytes = 3933120;
  // The streaming plan's device residual: the budget the slot pool must
  // fit (device budget − resident − KV reserve).
  int64_t device_residual_bytes = 0;
  // Requested hot-set size in experts. Capacity resolves to this, capped
  // by the residual; exceeding the residual refuses by name.
  int64_t requested_hot_experts = 0;
};

// The per-layer logical-expert → slot remap over a fixed-capacity slot
// pool. Touch() pins an expert into a slot (reassigning the
// least-recently-used slot on overflow and firing the eviction hook);
// SlotFor() reads the table (-1 = miss). Fingerprint() /
// ContentChangedSince() are the reset-lane predicate: the GDN
// slot-churn semantics (qwen3_5.cpp TT-GDN-SLOT-CHURN,
// ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W) — a slot whose logical expert
// changed flips the predicate, an identical re-selection does not.
struct Kolibri1TTExpertSlotPolicy {
  int64_t layers = 0;
  int64_t experts = 0;
  // One slot = one expert, one layer (fp8 + f32 scale grids).
  int64_t expert_bytes = 0;
  // Slots per layer; the streaming plan's hot_experts.
  int64_t capacity = 0;
  // capacity × expert_bytes: the device bytes the pool occupies.
  int64_t slot_pool_bytes = 0;

  using EvictHook = std::function<void(int64_t layer, int64_t expert,
                                       int64_t slot)>;

  // Registers the eviction callback. One hook at a time.
  void SetEvictHook(EvictHook fn) { hook_ = std::move(fn); }

  // Pins `expert` on `layer`: returns its slot, evicting the LRU resident
  // when the layer's pool is full (hook fires with the freed triple).
  // Re-pinning a resident expert returns its slot and evicts nothing.
  int64_t Touch(int64_t layer, int64_t expert);
  // The slot holding `expert` on `layer`, or -1 on a miss.
  int64_t SlotFor(int64_t layer, int64_t expert) const;
  int64_t ResidentCount(int64_t layer) const;
  // The slot a fetch would fill next: the first free slot, else the LRU
  // slot (what Touch() would hand out, read-only for the dispatch plan).
  int64_t SuggestSlotFor(int64_t layer) const;

  // The reset-lane predicate: an order-independent hash of every layer's
  // slot→logical-expert table.
  uint64_t Fingerprint() const;
  bool ContentChangedSince(uint64_t recorded_fingerprint) const {
    return Fingerprint() != recorded_fingerprint;
  }

  // Layer-scope access for the dispatch plan and the device stage (B2b).
  const std::vector<int64_t>& TableFor(int64_t layer) const;

 private:
  // table_[layer][slot] = logical expert id, or -1 when the slot is free.
  std::vector<std::vector<int64_t>> table_;
  // slot → recency stamp, monotonic; lower = older.
  std::vector<std::vector<uint64_t>> stamp_;
  uint64_t clock_ = 0;
  EvictHook hook_;

  friend Kolibri1TTExpertSlotPolicy PlanKolibri1TTExpertSlotPolicy(
      const Kolibri1TTSlotPolicyOptions&);
};

// Builds the policy: capacity = min(requested_hot_experts,
// residual/expert_bytes); refuses by name when the requested hot set
// exceeds the device residual (the deficit in the message).
Kolibri1TTExpertSlotPolicy PlanKolibri1TTExpertSlotPolicy(
    const Kolibri1TTSlotPolicyOptions& options);

// One step's MoE dispatch for a set of layers. Pure policy: reads the
// slot tables, never mutates the policy — the caller performs the fetch
// (filling fetch_slots) and then Touch()es the fetched experts.
struct Kolibri1TTLayerDispatch {
  int64_t layer = 0;
  // Requested experts already resident, with their slots.
  std::vector<int64_t> resident_experts;
  std::vector<int64_t> resident_slots;
  // DISTINCT requested experts missing from the pool (router repeats
  // collapse), and the slots the fetch must fill.
  std::vector<int64_t> missed_experts;
  std::vector<int64_t> fetch_slots;
};

struct Kolibri1TTDispatchPlan {
  std::vector<Kolibri1TTLayerDispatch> layers;
  // misses × expert_bytes (misses × 3,933,120 B on the real checkpoint):
  // this step's stream byte bound, always ≤ the B1 per-token ceiling.
  int64_t stream_bytes = 0;
};

// Dispatches one layer's router output (6-of-384 ids, repeats allowed)
// against the current slot tables.
Kolibri1TTDispatchPlan PlanKolibri1TTMoEDispatch(
    const Kolibri1TTExpertSlotPolicy& policy,
    const std::vector<std::pair<int64_t, std::vector<int64_t>>>&
        layer_requests);

// The all-miss worst case over every layer with one top-k set — the plan
// the B1 per_token_stream_bytes ceiling bounds.
Kolibri1TTDispatchPlan PlanKolibri1TTMoEDispatchAllLayers(
    const Kolibri1TTExpertSlotPolicy& policy, int64_t layers,
    const std::vector<int64_t>& topk_ids);

// ---- Wave B2b-i: the dense-resident device slice (spec §"B2 scope — B2b
// addendum", slice i) ----
//
// B2b-i runs the RESIDENT non-expert components entirely on device with
// the routed-expert tier absent (streamed in B2b-ii): attention (q/k/v/o
// fp8-block), the shared expert (fp8-block), the router (bf16 gate + f32
// bias), the norms (bf16, incl. the per-head q/k norms and the final
// norm), and embed + untied lm_head (bf16). This plan reuses the wave-A
// per-tensor dtype decisions verbatim over exactly that slice and asserts
// the hybrid geometry like the wave-A plan. It refuses ONLY when the
// resident slice alone exceeds the device budget — the single-device
// FULL-model refusal (wave A, §math) must NOT fire here: the resident
// slice is what fits one P150 (~3.09 GiB of the ~73.6 GiB full model).
// The routed experts are never enumerated, so the plan also runs over a
// weights tree whose expert tier was never loaded. Device-free like every
// plan in this TU: the device staging itself is the TT backend's seam
// (vt::tenstorrent, tenstorrent_staging.cpp), driven by the host pointers
// this plan records per tensor.

// Per-component byte accounting of the resident slice. The operand bytes
// match the B1 streaming shape's committed numbers exactly (attention
// 1,703,936,000; shared 196,608,000 fp8 + 48,000 grids = the B1
// shared_expert_bytes 196,656,000; router gate 98,304,000; embed+head
// 1,310,720,000); the grids and the router bias are itemized beside them
// so the grand total is exact rather than the B1 shape's ~2 MiB norm
// approximation.
struct Kolibri1TTResidentComponentBytes {
  int64_t attention_fp8_bytes = 0;    // q/k/v/o packed operands
  int64_t attention_scale_bytes = 0;  // their f32 scale grids
  int64_t shared_expert_fp8_bytes = 0;
  int64_t shared_expert_scale_bytes = 0;
  int64_t router_gate_bytes = 0;   // bf16 [num_experts, hidden] per layer
  int64_t router_bias_bytes = 0;   // f32 [num_experts] per layer
  int64_t norm_bytes = 0;          // bf16: sandwich + q/k norms + final
  int64_t embed_head_bytes = 0;    // bf16 embed + untied lm_head
  int64_t total_bytes = 0;
};

struct Kolibri1TTResidentStagingPlan {
  Kolibri1Params params;
  // The resident slice only, in plan order; every tensor carries its host
  // source pointers (host / scale_host) for the device staging.
  std::vector<Kolibri1TTTensor> tensors;

  int64_t total_bytes = 0;  // == components.total_bytes
  int64_t fp8_bytes = 0;    // packed fp8 operand bytes (FP8_E4M3)
  int64_t scale_bytes = 0;  // f32 scale grid bytes staged beside them
  int64_t bf16_bytes = 0;   // bf16 modules (router gate, norms, embed, head)
  int64_t f32_bytes = 0;    // the router bias (f32)
  int64_t tensor_count = 0;

  Kolibri1TTResidentComponentBytes components;
  int64_t full_layers = 0;
  int64_t swa_layers = 0;
};

// Plans the B2b-i resident slice over fully-loaded (or resident-only)
// kolibri1 weights: every fp8-block projection → FP8_E4M3 native (bytes
// from Fp8BlockWeight::packed, row-major, verbatim), the router gate,
// norms, embed and lm_head → BF16, the router bias → F32. Asserts the
// checkpoint's hybrid geometry (two KV groups, full-rotary head_dim 128)
// exactly like the wave-A plan, and refuses:
//  - a non-zero mesh_chips (the resident slice plans a single P150; the
//    mesh / expert-parallel staging layout is owed, spec ## Owed),
//  - a resident slice whose total exceeds the device budget (the refusal
//    names the byte deficit and that these components have no host tier).
// Throws std::runtime_error on every refusal.
Kolibri1TTResidentStagingPlan PlanKolibri1TTResidentStaging(
    const Kolibri1Weights& weights,
    const Kolibri1TTStagingOptions& options = {});

}  // namespace vllm
