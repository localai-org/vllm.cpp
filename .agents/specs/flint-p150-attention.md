# SPEC — `FLINT-P150-ATTENTION` — Flint attention on the Tenstorrent device: sinks, RoPE, sliding window through the AttnBlock seam

- Owning row: `FLINT-P150-ATTENTION` (matrix record ID `BACKEND-TENSTORRENT-FLINT-P150-ATTN`)
- Kind: feature (the last big port of the Flint P150 campaign)
- Issue: created in the same flow (row-scoped local issue)

## Where the port stands

The Flint P150 device arm (W0 #3508, W1 #3510) runs rms_norm and the
matmuls on device; attention remains HOST-side — `SinksAttention`
(`flint_tt.cpp:126`) computes sinks+RoPE+window on CPU and the activations
cross to device around it. This row moves attention onto the device
through the shared `dense_attn::AttnBlock` seam, completing the port.

## What Flint attention IS (all verified in-tree/artifact)

- Per-head attention **sinks** (`softmax_type: learnable`, one logit per
  head, softmax over [scores, sink] taking the first half) — the host
  reference is `SinksAttention`.
- **QK-norm**: RMSNorm over the HEAD dim on q and k (the W1 exact-norm arm
  already runs norms on device).
- **Hybrid window**: 35 sliding layers (window 513, WITH RoPE) + 14 full
  layers (NO RoPE — `no_rope_layer_indices`), the ISWA split.
- Head dim 128, 32 heads / 4 KV, GQA.

## Scope

Move Flint attention (per-head learnable sinks, head-dim QK RMSNorm,
RoPE-as-configured, the 513-token sliding window) from the host
`SinksAttention` leg onto the device attention path of the TT arm, in three
gated stages: single-layer probe, all-layers wiring, perf trace. Non-goals:
no CPU-path change, no CUDA arm, no new tt-metal kernels, no perf adoption
without the trace.

## Upstream chain

vLLM implements no Flint arm. The host reference is this tree's own
`flint.cpp` chain (`head_norm`, `RopeNeoxHalfSplit`, `SinksAttention`),
mirrored by `flint_tt.cpp`'s host legs. Device ops come from the pinned
tt-metal/ttnn library through `vt::` (`RmsNorm`, `RopeNeox`,
`ReshapeAndCache`, `PagedAttention`), the same composition kolibri1's TT
`AttentionBlock` drives.

## Our baseline

Today the TT arm runs attention HOST-side (`flint_tt.cpp:126`): q/k/v come
down, sinks+norm+RoPE+window run on CPU, attn_out goes back up around the
device norms and matmuls. The non-routed-attribution row's table prices this
crossing; the W0 gate (rel_rms 2.579e-1, greedy 1/8) is the honest model
baseline.

## Port map

- `src/vt/tenstorrent/tenstorrent_paged.cpp`: per-head `attn_sink`
  denominator seeding (mirrors cpu_paged_attn.cpp:218-225) + device SDPA
  arms decline a sink by name.
- `src/vllm/model_executor/models/flint_tt.cpp`: device attention legs
  (stage 2 wiring, env-gated), host shadow retained.
- `tests/vllm/models/test_flint_tt_attention_probe.cpp`: stage 1.

## Tests to port

No upstream tests exist (vLLM has no Flint). The gate is the stage-1 probe
(both window modes, 32 per-head slices, W2-derived band) and the full-model
keep-quant band gate (`test_flint_tt_w0` contract shape against the CPU
reference and the recorded W0 baseline).

## Dependencies

Base has W0/W1/W2 landed. Device recipe: the standing P150 mutex/reset/env
discipline. Claims file `.agents/claims/CLAIM-FLINT-P150-ATTENTION.md`.

## Work breakdown

Stage 1 probe -> stage 2 wiring + model gate -> stage 3 perf trace, one
commit series each, stage-scoped.

## Risks/decisions

Band gate, never bit-exact (the W2 record); per-head slices AND a model-level
gate both required (the SDPA-2-heads and PA-correct-but-output-wrong scars);
STOP conditions in the issue if the seam lacks a primitive or divergence is
structural.

## Known hazards from campaign history (each a prior session's scar)

- The substrate matmul noise (2.4e-2 single-product, ~23x the tf32 bound,
  dtype-independent, enters at the first post-norm q/k/v projections —
  FLINT-TT-W2) means **the device attention gate is a BAND gate, never
  bit-exact**. Bit-exactness against the host shadow is not the contract.
- Prior TT attention sessions hit: SDPA computing only 2 heads
  (`tt-sdpa-only-2-heads`), PA "correct but output wrong"
  (`tt-pa-correct-but-output-wrong`), device PA never verified
  end-to-end (`tt-device-pa-never-verified`). Each scar narrows this
  spec's verification: per-head output slices must be checked
  individually, and a device PA that passes unit-level checks must still
  pass the model-level token gate before "works" is claimed.

## Stages (each gated, each a commit series)

1. **Single-layer probe**: one Flint attention layer on device (sinks +
   head-dim QK-norm + RoPE-as-configured + 513 window) vs the host
   `SinksAttention` shadow, per-head output slices, band gate derived
   from the W2 substrate-noise record. The layer runs BOTH window modes
   (sliding and full/no-RoPE) — the mode flip is where prior attention
   ports broke.
2. **All-layers wiring**: `FlintForwardTT` routes attention through
   `AttnBlock` on device; the full-model keep-quant band gate (the
   `test_flint_gguf` contract shape, vs the llama.cpp goldens) is the
   model-level proof. The host `SinksAttention` stays as the shadow and
   for any layer the device cannot serve.
3. **Perf trace**: attention-on-device vs host-attention serving trace
   (the ~88% non-routed attribution row's table is the baseline for what
   attention costs today) — recorded, no adoption verdict forced: if
   device attention is slower at these shapes, the record says so and
   the wiring stays behind its env.

## Band discipline

The band for stage 1 is derived from the W2 substrate record (not
invented): the noise floor is the measured single-product deviation
propagated through the attention reduction. If the device attention
cannot hold ANY sane band (i.e. divergence is structural — a wrong
kernel, not noise), that is a defect finding: STOP, report, repair
separately.

## Non-goals

- No CPU-path change (the host kernel stays; it is the shadow).
- No CUDA arm. No new tt-metal kernels unless the seam lacks one —
  prefer composing existing device ops; a new kernel needs its own
  row-level adjudication.
- No perf adoption without the stage-3 trace.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure); device legs under the standing recipe (mutex, reset, envs,
claims file, governor logged, no timeout guards); the full-model
keep-quant band gate; the existing flint_tt suites green.

## Git integration

One pull request, `row/FLINT-P150-ATTENTION`, spec committed first,
stage-scoped commits.
