ID: ISSUE-LOCAL-01M6HF07NW0NQPVC5V21EGZJ8M
Title: The model-level device-attention gate at the fixed base, then the adoption decision
Row: FLINT-P150-STAGE2-GATE
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-03
Updated: 2026-11-03
Closed: 2026-11-03

## Problem

FLINT-P150-ATTENTION stage 2 STOPPED because the W0 model-level band gate was allocation-order-sensitive (neutral DBuf edits moved it 2.683e-1 to 7.224e-1). The blocker is cured: FLINT-TT-DENSE-MLP-STAGING (fd15425a1) fixed EnsureHostBytes to write the shadow's dtype; all four lifetime permutations now read 2.683e-1. Deliverable: a model-level gate case running the keep-quant paged recipe with VT_FLINT_TT_DEVICE_ATTN=1 asserting the same drift-tolerant band (rel_rms < 0.50, max_abs < 64.0) plus the device arm's own chain consistency; then the recorded adoption decision (default flip if the band holds, env-gated default-off with the measured gap if not -- stage 3 recorded device attention roughly neutral at 53.0 vs 55.2 s/step, so the numbers decide). Spec: .agents/specs/flint-p150-stage2-gate.md

## Resolution

ADOPTED, closed 2026-11-03. The model-level gate case
(`tests/vllm/models/test_flint_tt_stage2_gate.cpp`) ran the keep-quant paged
recipe twice (host arm pinned `VT_FLINT_TT_DEVICE_ATTN=0`, device arm `=1`;
the gate pins both arms explicitly so the flip cannot move its measurements)
at 3
reps on the P150: both arms held the drift-tolerant band (host rel_rms
2.198e-1 / max_abs 3.335e+1; device 2.200e-1 / 3.956e+1 vs the 0.50 / 64.0
fence; reps bit-identical), the host W0 gate (green BOTH as opt-out and
under the adopted default), the single-layer probe and
`test_flint_paged` 208/208 all green on the same runs. Adoption basis
recorded in `docs/bench-evidence/flint-p150-stage2-gate-20261101/README.md`:
band holds + serving trace roughly neutral (53.0 vs 55.2 s/step) →
`DeviceAttnEnabled()` flipped to default ON with `=0` as the opt-out;
ENVIRONMENT.md and USAGE.md updated in the same change. No serving number
changed, so no docs/benchmarks/ entry moved. Named finding for the next
wave: identical device-attention forwards are history-dependent across
pool/slot sequences (device-after-device measured 5.050e-1, device-first
2.540e-2); the gate pins both arms explicitly and the serving path never
toggles arms.
