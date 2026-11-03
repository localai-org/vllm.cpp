# SPEC — `FLINT-P150-STAGE2-GATE` — the model-level device-attention gate at the fixed base, then the adoption decision

- Owning row: `FLINT-P150-ATTENTION` (stage-2 STOP completion; this row IS that stage's completion, filed as its own row because the blocker it waited on — FLINT-TT-DENSE-MLP-STAGING — is now landed)
- Issue: created in the same flow (row-scoped local issue)

## Why now

FLINT-P150-ATTENTION stage 2 STOPPED because the W0 model-level band gate
was allocation-order-sensitive: neutral DBuf lifetime edits moved the band
between 2.683e-1 and 7.224e-1, so no model-level gate could hold. The
blocker is cured: FLINT-TT-DENSE-MLP-STAGING (landed fd15425a1) fixed
`EnsureHostBytes` to write the shadow's dtype, and all four lifetime
permutations now read 2.683e-1. The stage-2 wiring exists env-gated
(`VT_FLINT_TT_DEVICE_ATTN=1`, flint_tt.cpp:745/:832/:914); the
single-layer probe is 64/64 per-head slices in both window modes.

## Deliverable

1. **The model-level device-attention gate case**: a new TEST_CASE in
   `tests/vllm/models/test_flint_tt_w0.cpp` (or a sibling TU, house
   pattern) that runs the keep-quant paged serving recipe with
   `VT_FLINT_TT_DEVICE_ATTN=1` set, and asserts the SAME drift-tolerant
   band the host arm's gate asserts (rel_rms < 0.50, max_abs < 64.0) plus
   the device arm's own adjudicated-divergent chain consistency. Record
   the device arm's measured rel_rms/max_abs and greedy chain in the
   evidence — these are the adoption inputs, not a foregone conclusion.
2. **The adoption decision, recorded**: if the device arm holds the band,
   the wiring flips to default-on (the env-gate stays as an opt-out);
   if it does not, the wiring stays env-gated default-off and the
   measured numbers name the gap — no adoption verdict is forced by the
   spec (FLINT-P150-ATTENTION stage 3 recorded device attention roughly
   neutral at 53.0 vs 55.2 s/step, so the default flip is a policy
   choice the numbers should make).
3. **Evidence**: docs/bench-evidence/flint-p150-stage2-gate-20261101/ —
   the host-arm and device-arm runs side by side (same prompt, same
   artifact, same recipe), the band assertions, the chains, and the
   adoption decision with its basis.

## The recipe

The keep-quant paged serving recipe (docs/USAGE.md matched-pool section;
artifact /mnt/models/flint-GGUF/flint-Q4_K_M.gguf, sha 20c89707...):
VT_FLINT_KEEP_QUANT=1, pool matched to the taskset, taskset -c 32-63,
VT_FLINT_TT_DEVICE_ATTN=1 for the device arm (unset for the host arm).
3-rep medians, idle-gated, claims file, governor logged, NO timeout
guards.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure); the new gate case green; the host-arm W0 gate unchanged green;
test_flint_paged 208/208; the single-layer probe 64/64.

## Stop conditions

- The device arm exceeds the band (rel_rms >= 0.50 or max_abs >= 64.0)
  → record the numbers, keep the wiring env-gated default-off, name the
  gap: that is a finding, not a failure of the row (the layer-level
  probe is healthy; the model-level gap is the finding).
- Device wedge surviving two resets → STOP, report.

## Git integration

One pull request, `row/FLINT-P150-STAGE2-GATE`, spec committed first.
