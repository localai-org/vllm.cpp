ID: ISSUE-LOCAL-01M4YVEP000ENCRECVB0S6FDDY
Title: kolibri1 TT host-free ON arm produces more teacher-forced hard flips than OFF (40 vs 17)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-15
Updated: 2026-10-21
Closed: -

## Problem

The #3454 readjudication attributes the kolibri1 P150 hard-flip count to device bf16 1-ULP drift x router gain, with the ARM DELTA (host-free ON 40 teacher-forced HARD flips vs OFF 17) left as the largest single lever: the host-free path's shadow handling (WeightViewShadows/Bfp8WeightShadows reconciliation) is suspected to introduce more drift than the eager path. The #3452 UAF fixes (memo eviction order + shadow sweeps at UnregisterHostBuffer/MarkScratchAcquired) landed after that measurement; this issue owns re-measuring the delta on the current base, bisecting which host-free-specific path carries the extra drift (rope/residual/PA force-host knobs, expert-weight shadow residency), and fixing it if it is a defect (stale shadow serve) or pricing it precisely if it is inherent precision difference.

## Resolution

- 2026-10-15 (row/tt-kolibri-shadow-recon): the delta REPRODUCES exactly on the
  current base (ON 40 / OFF 17 TF HARD, gate 5/2). The shadow-reconciliation
  hypothesis is FALSIFIED: both arms share the shadow maps and the #3452
  sweeps, and the first diverging stage in the run is `L0 dhn` — the
  input_layernorm — on BIT-IDENTICAL inputs (emb hash `dd76406a`, gam sum
  858.502781 both arms): host f32 loop 9.846392 vs device 9.765238 (ratio
  0.9918). `VT_TT_NORM_DEBUG` proves the device norm's inputs are bit-identical
  (`dev_x`/`norm_f32` −20.613504, `wf32` 858.502781) while `dev_y` reads
  9.777145: the deficit is in the tt-metal rms_norm kernel ON THE MODEL'S
  ACTIVATION DISTRIBUTION — the committed micro (random inputs, same shape/eps,
  patched `fp32_acc=true` substrate, verified installed) measures 0.9997 clean.
  Bisect: `VT_TT_FORCE_HOST_ROPE` byte-identical flip set (ruled out);
  `VT_TT_FORCE_HOST_PA` identical chain (ruled out); `VT_TT_FORCE_HOST_RESIDUAL`
  reads 7 HARD — WORSE — because the host round-trips desync the device shadow
  (the recorded 1e37 collapse class), so it is not a clean probe and the "host
  residual is more accurate" reading of the pre-#3452 bisect is dead.
  `VT_TT_RMSNORM_COMPOSED=1` re-confirmed substrate-blocked in-model
  (`Invalid subtile broadcast type`, binary_ng_device_operation.cpp:224).
  MECHANISM: host-free forces the device rms_norm at T=1 for capture; that
  kernel carries a ~0.7% systematic input-dependent bias on kolibri1
  activations; 50 layers x 2-3 norm sites compounds into multi-nat router-logit
  shifts and the 40-vs-17 arm delta. Evidence:
  docs/bench-evidence/kolibri1-readj/shadow-recon-bisect-20261015.md.
  NEXT LEVER: substrate characterization — feed the in-model emb/gam row
  through the micro to split input-dependence from program provenance
  (program-cache/config under the model call pattern), then fix tt-metal or
  force the clean config in-model. STAYS OPEN (the delta is not fixed; it is
  re-attributed).

2026-10-21: STILL TRUE. The record's own 2026-10-15 entry stands: the arm delta
REPRODUCES exactly (host-free ON 40 teacher-forced HARD flips against OFF 17),
the shadow-reconciliation hypothesis is falsified, and the attribution is the
tt-metal rms_norm bias on kolibri's activation distribution. Nothing since
then removes the bias: the tt-metal upstream head probed on 2026-10-19
(2500c1d8f63) reads 1.0072 -> 0.9809 on the largest-magnitude captured row,
1.0077 -> 1.0033 on the next, and 0.9984 -> 0.9970 on the clean row, so the
input-dependence is not improved upstream
(`ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF`).

What settles it: a host-free ON/OFF re-measurement after the tt-metal rms_norm
substrate is fixed, which is a P150 measurement.

## Addendum 2026-10-25: the upstream filing is tt-metal#62339

The rms_norm compute-config finding has an OPEN upstream filing:
https://github.com/tenstorrent/tt-metal/issues/62339 (filed 2026-10-22 —
the selection site documents no accuracy contract; default 0.9809 vs host
1.0 on the captured rows). The Flint TT W0 device-vs-CPU baseline
(rel_rms 2.58e-01, docs/bench-evidence/flint-tt-w0-20261025/) attributes
its divergence to this same ~1% systematic bias compounding over 49
layers, with the matmul arms clean at 1.9e-3 — the filing now carries a
second independent model witness. No response upstream yet.
