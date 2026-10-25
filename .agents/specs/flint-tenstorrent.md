# Flint on Tenstorrent — W0: device dispatch + first device-side correctness baseline

Row: MODEL-TEXT-FLINT (the flint-cpu-gguf spec's `## Owed` names "The device
arm (Tenstorrent), a separate row"; the roadmap does not yet carry a dedicated
`MODEL-TEXT-FLINT-TENSTORRENT` row — the same roadmap gap wave 2 noted. This
spec starts that arm under the canonical MODEL-TEXT-FLINT row).

Issue: ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q.

## Scope

W0 is dispatch + first device-side correctness evidence. NOT speed, NOT oracle
parity.

- The registry dispatch: `FlintForCausalLM` currently serves one arm, the CPU
  reference forward (`FlintModel::Forward`), and a Tenstorrent queue falls
  through to host code with no device dispatch at all. W0 adds the
  `kTENSTORRENT` case in `flint.cpp`'s forward, named refusals for every other
  non-CPU device (the kolibri1_registry.cpp:328-361 pattern), and a prepare-time
  residency build on a Tenstorrent queue.
- A REAL device forward over the bf16-resident `FlintWeights` (the GGUF
  loader's `LoadFlintFromGguf` already dequants to bf16; the safetensors arm
  lands bf16 too). Device legs: embedding, per-layer input/post-attention
  RMSNorm, every linear projection (q/k/v/o, dense MLP gate/up/down, router,
  shared expert, routed experts, lm_head) through `vt::MatmulBT` on
  device-resident bf16 weights, SwiGLU product via `vt::MoeSiluMul`, residual
  adds via `vt::Add`. Host legs at W0 (each named, each owed): per-head q/k
  RMSNorm, the partial-RoPE pattern, the sinks softmax with the sliding
  window, sigmoid top-k routing/renormalisation, and the MoE combine.
- The routed experts do NOT reside: the full expert tier is ≈50 GiB bf16
  (256 experts x 4 projections x 47 MoE layers) against a 32 GiB P150 — the
  kolibri 384-expert wall again. W0 STREAMS the selected experts per layer
  (top_k + shared) from the host bytes, like the kolibri streaming precedent.
- Geometry read from the artifact, never hardcoded: `sliding_window`
  (513), the `attention.sliding_window_pattern` / `rope_pattern` KVs that say
  which layers carry RoPE, per-head sinks, `expert_gating_func = 2`
  (sigmoid + renorm), `expert_weights_norm = true`, `leading_dense_block_count
  = 2`. All of these already flow through `FlintParams`; the device forward
  consumes the same parsed struct — no second parser.

## Upstream anchors

- vLLM implements nothing for this architecture; the reference is the pinned
  llama-cpp-flint oracle (`1b44e30775dcb2f93952116b05f89e5bd797fb00`,
  `src/models/flint.cpp`) — this wave does NOT gate against it.
- Local CPU reference: `src/vllm/model_executor/models/flint.cpp:314-565`
  (`FlintModel::Forward`).
- Pattern precedent: `src/vllm/model_executor/models/kolibri1_registry.cpp`
  (the `kTENSTORRENT` case at :341), `kolibri1_tt_forward.cpp`
  (`LinearBTDevice`, `ExpertMlp`, staging), `src/vt/tenstorrent/
  tenstorrent_ops.cpp` (op vocabulary).
- Pinned substrate: /tmp/umdtrial-install + /tmp/tt-metal-umdtrial; P150 is a
  LOCAL non-fleet device — every device run under `flock -x` on
  `${GPU_LOCK:-$HOME/gpu.lock}`.

## Design

New TU `src/vllm/model_executor/models/flint_tt.cpp` +
`include/vllm/model_executor/models/flint_tt.h`:

- `FlintTTStagingPlan`: device-free byte accounting over `FlintWeights` — the
  resident slice (embed, lm_head, per-layer attn projections, norms, sinks,
  dense MLPs, shared experts, router) plus the streamed expert tier; refuses a
  resident slice that exceeds the device budget by name. Overridable budget so
  tests exercise the refusal device-free.
- `FlintTTDeviceContext`: the staged resident weights (bf16) + accounting;
  built at prepare on a Tenstorrent queue, lazily otherwise (the kolibri seam).
- `ForwardFlintTT(...)`: one stateless full-sequence step (matching the CPU
  reference arm's shape — no paged KV cache at W0; the CPU arm refuses >1
  sequence and so does the device arm). q/k/v and router matmuls request f32
  outputs (the CPU reference keeps the head stream and the router logits f32);
  the normed stream and residual stream round to bf16 as the CPU reference
  rounds them. Layer-0/few-layer activation dumps behind an env flag, tagged
  the same as the CPU forward's matching env-gated dump, so the
  device-vs-CPU distance is readable per stage.
- `flint.cpp`: dispatch switch on `input.queue.device.type`; `FlintTTForward
  Calls()` counter so the run proves the device case was exercised, not merely
  compiled (reachability).

## Tests

- `tests/vllm/models/test_flint_tt_w0.cpp`:
  - device-free staging-plan accounting + refusal (runs on any host);
  - the RED gate: a Tenstorrent-queue forward through the registry must NOT
    throw the "no Tenstorrent arm" refusal and must increment
    `FlintTTForwardCalls()` — fails before the implementation (captures the
    refusal), passes after. Gated on a present device and the
    `FLINT_TT_W0_MODEL` env; SKIPs with a message otherwise;
  - the W0 correctness leg: one prompt, device vs the same worktree's CPU
    forward on the SAME bf16 weights — max-abs and RMS distance on first-token
    logits, an 8-token greedy chain comparison, and the env-gated layer dumps.
- Red evidence: `docs/bench-evidence/flint-tt-w0-20261025/red.txt`.

## Gates / evidence

The W0 gate is HONEST device-vs-CPU only. No oracle parity claim, no
throughput claim, no ceiling. The recorded numbers are the like-for-like
baseline (device vs the CPU reference on identical bf16 weights); the frame
that makes them meaningful is the kolibri precedent (CPU-vs-oracle 2.0e-3 on
layer-0 l_out). Evidence lands in
`docs/bench-evidence/flint-tt-w0-20261025/report.md` with the exact build and
run recipe, env, clock state, and mutex.

## Owed

- The device attention leg: head-norm, RoPE, sinks softmax and the sliding
  window on-device (dense_attn / paged arm) — W1+.
- The paged-engine integration (the CPU arm's same owed item).
- The keepquant device arm for the APEX/k-quant containers (IQ2_S, IQ3_S,
  IQ4_XS, Q4_K, Q5_K, Q6_K — all have TT int8-dot arms already).
- The token gate vs the llama-cpp-flint oracle.
- The performance trace (and the roadmap row itself:
  MODEL-TEXT-FLINT-TENSTORRENT).
- Expert streaming memoisation; at W0 every layer re-uploads its selected
  experts (correctness vehicle only).

## Stop conditions

Return NEEDS_CONTEXT if the P150 is unusable or the pinned substrate is gone.
Do not widen to speed work. Do not gate against the oracle this wave.
