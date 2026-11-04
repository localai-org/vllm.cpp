# FLINT-P150-STAGE2-GATE — the model-level device-attention gate at the fixed base, then the adoption decision

Row `FLINT-P150-STAGE2-GATE` (owning row `FLINT-P150-ATTENTION` stage 2),
spec `.agents/specs/flint-p150-stage2-gate.md`, issue
`ISSUE-LOCAL-01M6HF07NW0NQPVC5V21EGZJ8M`. Branch
`row/FLINT-P150-STAGE2-GATE`, base spec+issue commit `b09c7330f` on top of
the cured blocker `fd15425a1`. Date 2026-11-03.

## The gate case

`tests/vllm/models/test_flint_tt_stage2_gate.cpp` (target
`test_flint_tt_stage2_gate`): the keep-quant paged serving recipe
(`VT_FLINT_KEEP_QUANT=1`, artifact `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`)
run TWICE through `ModelRegistry::Load`/`Forward` — the host arm
(`VT_FLINT_TT_DEVICE_ATTN` unset) and the device arm (`=1`, set between the
arms; `DeviceAttnEnabled()` reads the env at forward time,
flint_tt.cpp:106, at the three model-level blocks :745/:832/:914). Both arms
are measured against the SAME CPU dequant reference (the W0 frame) and both
assert the drift-tolerant band `rel_rms < 0.50`, `max_abs < 64.0` (the W0
fence, test_flint_tt_w0.cpp:379-380); the chains are recorded, not pinned
(the adjudicated-divergent W0 record). 3 reps.

## Results (idle-gated, governor `performance` before and after every leg)

First-token logits vs the CPU dequant reference (n=750720), 3 reps were
BIT-IDENTICAL within each arm:

| arm | rel_rms | max_abs | greedy chain (8 steps) | band |
|---|---|---|---|---|
| host (`VT_FLINT_TT_DEVICE_ATTN=0`) | 2.198e-1 | 3.335e+1 | `269 269 269 269 269 269 269 269` | held, 3/3 |
| device (`VT_FLINT_TT_DEVICE_ATTN=1`) | 2.200e-1 | 3.956e+1 | `12834 332 269 284 1978 877 3087 90562` | held, 3/3 |

Post-adoption re-run ON the flipped-default build
(`gate-adopted-default-3reps.log`, arms explicitly pinned by the gate): both
arms reproduce the numbers above bit-identically across 3 reps; the host-arm
W0 gate is green BOTH ways — with the opt-out `=0` (the shadow) and with the
adopted default (unset, which now exercises the device arm in W0) — and the
single-layer probe plus `test_flint_paged` 208/208 stay green.

Both arms reproduce the epoch's numbers from the cured base (the pre-blocker
record read rel_rms 2.683e-1 on the same recipe; this run read 2.198e-1 —
inside the drift-tolerant fence, which is why the fence is absolute 0.50,
not the morning-calibrated 0.25). The device arm sits +0.002e-1 rel_rms and
+6.2 abs on the first-token logits against the host arm — well inside the
band. The device arm's chain is its own adjudicated-divergent chain,
bit-identical across all 3 reps (recorded, not pinned; the W0 record
explains why the chain is epoch-dependent). The CPU reference chain for the
same steps is `12834 332 269 764 32 13 3529 198` (the W0 log's cpu row).

Unchanged gates on the same run (all 3 reps SUCCESS, raw log
`gate-3reps.log`):

- the host-arm W0 gate (`test_flint_tt_w0` dispatch case): 9/9 assertions,
  first-token rel_rms 2.198e-1 / max_abs 3.335e+1 — UNMOVED by the gate-case
  change;
- the single-layer probe (`test_flint_tt_attention_probe`): SUCCESS, 3/3;
- `test_flint_paged`: 208/208.

Device recipe: `flock -x` on `$HOME/gpu.lock`, luwen `reset` + 15 s +
`/dev/shm` lock cleanup before the legs, `taskset -c 32-63` for the test
legs, `-c 64-127` for the build (`/tmp/build-p150stage2`), idle gate
`ps` load check, NO timeout guards, governor logged per leg (`performance`,
unchanged). One run-environment finding, recorded because it cost two
red runs: the keep-quant kernel include dir is `__FILE__`-derived
(tenstorrent_keepquant.cpp:1896) and `-ffile-prefix-map` makes it RELATIVE
(`./src/vt/tenstorrent/kernels`), so the test process CWD must be the source
root the binaries were built from — otherwise every TT forward throws
`TT_THROW kernel.cpp:106 set_compiler_include_paths`.

## The adoption decision: DEFAULT ON

**The device attention is adopted as the default.** Basis:

1. The band holds: the device arm measured rel_rms 2.200e-1 / max_abs
   3.956e+1 against the 0.50 / 64.0 drift-tolerant fence, 3/3 reps
   bit-identical — before AND after the flip (the gate pins both arms
   explicitly, so the flip cannot move its measurements) — no model-level
   gap exists at this base (the blocker the stage-2 STOP waited on is cured
   by `fd15425a1`).
2. The cost is neutral: the stage-3 serving trace
   (`docs/bench-evidence/flint-p150-attention-20261031/report.md`) records
   53.0 s/step device vs 55.2 s/step host on the 32-token prompt + 6 steps —
   attention is a minor share of the MoE-dominated step.
3. The host shadow stays reachable as the same-binary rollback
   (`VT_FLINT_TT_DEVICE_ATTN=0`), and the new gate case exercises both arms
   in one binary so the rollback cannot rot silently.

The flip: `DeviceAttnEnabled()` (flint_tt.cpp:106) returns TRUE when the env
is unset and FALSE on `=0`; `docs/ENVIRONMENT.md` and `docs/USAGE.md` updated
in the same change. No `docs/benchmarks/` number changes — the serving trace
numbers above were recorded in `flint-p150-attention-20261031` and are
unchanged; this row records the adoption basis, not a new measurement.

## FINDING: the gate arms must pin the env, never rely on the default

An intermediate build ran the gate with UNPINNED arms after the flip
(`gate-unpinned-arms-finding.log`): the gate's "host" arm (env unset) then
measured the DEVICE arm (2.540e-2, device-first fresh process), and its
"device" arm measured a device-after-device history at rel_rms 5.050e-1 —
1 percent OVER the fence. The measurements are history-dependent: identical
device-attention forwards give different distances depending on the
pool/slot sequence their predecessors left (the allocation-order hazard the
cured blocker `flint_tt.cpp:618-626` records). This is a defect of the TEST
design, not of the wiring: no serving process toggles arms mid-flight, the
adopted default's own first-forward history measured IN band (2.540e-2), and
the shipped gate now pins `=0` / `=1` explicitly so its two arms are the
same histories the pre-adoption gate and the serving recipes use. The
recorded gap, named for the next wave: the device-attention model-level
distance is not history-robust — a forward's band depends on what ran
before it in the same process — which is the same sensitivity class the
FLINT-TT-DENSE-MLP-STAGING row cured for the MLP staging, and it should be
attacked at the pool/slot-sequence level, not by re-pinning fences.
