# Flint on Tenstorrent — W3: the substrate matmul-noise characterization and the exactness attempt

Row: BACKEND-TENSTORRENT.
Issue: ISSUE-LOCAL-01M5TB3983MW41T4NPBZ5ZYF0E.
Prior waves: W0 (flint-tenstorrent.md — the device arm, rel_rms 2.579e-1),
W1 (rms_norm repair — the per-op norm exactness landed, model number
unchanged), W2 (flint-tt-w2.md — the tf32 hypothesis killed; the measured
per-op noise is operand-dtype-INDEPENDENT ~2.5e-2 rel_rms vs host double,
multiply-path dominated, integers exact).

Branch `row/TT-MATMUL-W3`. Base `041615b8c` (head of `row/FLINT-TT-W2`;
`a1b94d9ba` in the original prose is a pre-rewrite SHA of that stack — the
history was rewritten to `b4b743d4a` + `041615b8c`, unit head `c746ea51f`),
which contains origin/main `3090f069bda47a09e6ef809a86b0c8b61a4f21c6` plus
the four unmerged W2 commits). The W2 work is the W3 substrate — the micro
test and the bisection instrument live there — so W3 bases on it rather
than bare origin/main. This is a deliberate base deviation from the wave
brief and is recorded in the W3 report.

## Scope

CHARACTERIZE first; the exactness route is chosen BY the map, not by
preference.

1. CHARACTERIZE the noise (extend the W2 micro into a W3 scratch micro):
   - Shape: K sweeps at fixed M/N (is the 2.4e-2 per-product constant,
     K-dependent, or shape-quantized?), M and N sweeps, and the
     tile-quantized vs non-tile-multiple question. Does the q/k/v
     projection shape sit at a bad point in the map?
   - Staging/layout: TILE vs ROW_MAJOR staging of the operands.
   - fp32_dest_acc_en on/off (f32 out vs bf16 out through the same
     ttnn call).
   - Accumulation order if the pin exposes a selectable program config.
   - Re-run the W2 D (MatmulBT) and E (bf16-out) arms on the same map so
     the isolated rel~1.2 uncorrelated reading gets a mechanism or is
     pinned as a staging/layout artifact.
2. ATTEMPT EXACTNESS, route picked by the map:
   a. A composed-exact matmul arm for the SMALL set of shapes the Flint
      model needs (the kolibri GDN multiply+sum SFPU precedent; note
      ttnn::sum is the tf32 tile-reduce and ttnn::mean is the exact f32
      path). If exact composition is impossible on this pin, prove it
      with the map and record the floor.
   b. A local tt-metal-pin patch ONLY if the map isolates a fixable
      kernel defect (accumulator or padding class). Provenance recorded.
   c. If the noise is silicon-real (reproduces on a plain ttnn matmul
      outside our stack), the deliverable is the map plus a minimal
      upstream-filing draft recorded locally, NOT filed.
3. MODEL VERDICT: rerun the W0 harness
   (FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf)
   before/after whatever lands. Tighten the W0 band ONLY if rel_rms
   actually drops; otherwise record why.

## Red-first

Where a gate changes (a new exact-arm CHECK in the micro), show it red
against the current noisy arm and green with the new arm, on device.

## Upstream anchors

- ttnn::operations::matmul::matmul at the pinned substrate (TT_METAL_HOME
  ~/Sources/tt/tt-metal-pin, runtime /tmp/umdtrial-install).
- src/vt/tenstorrent/tenstorrent_ops.cpp MatmulKernel / MatmulBTKernel
  (the only two matmul entry points our stack reaches ttnn through).

## Tests and gates

- tests/vllm/models/test_flint_tt_w3_noise.cpp (scratch diagnostic TU,
  the W2 micro pattern): the shape/layout/dest-acc map, the D/E re-runs,
  and — if an exact arm lands — its CHECK.
- tests/test_flint_tt_w0: the before/after model verdict.

## Evidence

docs/bench-evidence/tt-matmul-noise-20261025/report.md: the noise map
table, the attempt and its verdict, before/after W0 numbers, recipe,
environment, patch provenance if route (b).

## Risks and stop conditions

- A tt-metal rebuild is long; only started if the map indicts a kernel
  defect (route b). Budgeted with a monitor; the pinned substrate
  (/tmp/umdtrial-install, /tmp/tt-metal-umdtrial) is never wiped.
- The box is shared: taskset + nice, and no device run overlaps a -j 32
  build. Every device run under `exec 9>$HOME/gpu.lock; flock -x 9`.
- Stop: the map shows the noise is silicon-real and the composed-exact
  route is proven impossible on this pin; the record then carries the
  floor, the filing draft, and the unchanged-W0 explanation.

## Outcome (2026-10-26)

Measured: the noise is the pinned ttnn DEFAULT math fidelity (LoFi)
multiply truncation — flat ~2.3e-2 rel_rms on the micro across every
K/M/N/layout axis, per-product probe bit-exact, reproducing on a plain
ttnn matmul with no vt staging. The exactness arm (HiFi4 + approx off +
fp32_dest_acc_en) measures 3.3e-4 rel_rms, frac_bad 0 — ~70x on the micro
— and on the W0 model it improves first-token rel_rms 2.579e-1 -> 2.212e-1
(~14%) at no decode-throughput cost (3 alternating full-wall legs each,
`throughput-w0-legs.log`: walls 853.4-882.7s vs 846.1-857.9s, ~1.4% faster
on the leg mean — a tie, not claimed as a speedup).

Rejected: the ORIGINAL no-flip record. Its "shipped 2.045e-1" came from an
uncommitted binary (log cites a W0 band of 0.25 at test lines that exist in
no commit); the committed-head rerun invalidates it, and with the true
numbers the accuracy premise of the no-flip call inverts. The greedy chain
is adjudicated-divergent on BOTH arms (the first token flips 12834->269
either way — the recorded hard-divergence bias class), so the flip costs
nothing the old band could see.

Why each default has its value: the developer APPROVED the adoption on the
committed-head evidence, so the shipped matmul default is the exactness
arm and `VT_TT_MATMUL_MATH_FIDELITY=lofi` restores the pre-adoption arm
for same-binary A/B (the knob inverted). The W0 gate is re-specified to
the adjudicated-divergent form — chain consistency per config (each arm's
recorded chain, bit-identical across its 3 legs) plus a tightened distance
fence, rel_rms 0.5 -> 0.30 (1.36x the measured 2.212e-1) and max_abs
64 -> 40 (1.22x the measured 3.275e+1); the old arm's distance numbers
(2.579e-1 / 3.200e+1) stay inside that fence, so it is a regression fence,
not an arm selector.
