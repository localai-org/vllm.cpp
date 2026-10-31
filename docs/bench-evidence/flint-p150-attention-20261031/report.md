# FLINT-P150-ATTENTION — stage 1 green, stage 2 wired and STOPPED at the model gate, stage 3 recorded

Row `FLINT-P150-ATTENTION` (matrix record `BACKEND-TENSTORRENT-FLINT-P150-ATTN`),
spec `.agents/specs/flint-p150-attention.md`, issue
`ISSUE-LOCAL-01M687MYRKDS21SBESBG012KN7`. Branch `row/FLINT-P150-ATTENTION`.
Date 2026-10-31.

## Stage 1 — the single-layer probe (GREEN, committed)

`test_flint_tt_attention_probe`: one Flint attention layer's core on device
(vt::RmsNorm over [T*H, 128] for the head-dim QK norm, vt::RopeNeox on the
sliding layers, ReshapeAndCache + vt::PagedAttention with the 513 window and
the per-head sink) vs the host SinksAttention shadow, BOTH window modes, ALL
32 head slices individually. Band = 3x the W2 record's 2.64e-2 single-product
rel_rms = 7.9e-2.

| mode | worst head | rel_rms | mean | slices |
|---|---|---|---|---|
| SLIDING t=600 (window 513 clipped), RoPE | 6 | 5.35e-3 | 4.60e-3 | 32/32 under band |
| FULL t=96, no RoPE | 17 | 3.31e-3 | 2.89e-3 | 32/32 under band |

Log `probe-green.log` (64 per-head lines). The seam primitive this needed —
the per-head attention sink — landed in the TT kPagedAttention host oracle
(mirroring cpu_paged_attn.cpp:218-225) with both ttnn SDPA arms declining a
sink by name; no new tt-metal kernel.

## Stage 2 — all-layers wiring (WIRED, model gate NOT held, STOPPED)

`ForwardFlintTT` grew the device attention leg behind `VT_FLINT_TT_DEVICE_ATTN=1`
(default OFF — the host shadow is byte-identical and the W0 gate reproduces
its recorded band with the wiring landed: 2.683e-1). The leg composes
device ops (RmsNorm, RopeNeox, WriteKvCache, PagedAttention with window+sink,
o_proj on device) with no host round trip for the attention output.

Model-level result: the W0 gate PASSES once (first-token rel_rms 4.825e-1 <
the 5.0e-1 band, `w0-device-attn.log`) but the number is NOT STABLE against
allocation-order changes that are neutral to the arm:

| DBuf-lifetime variant | host-shadow arm | device-attention arm |
|---|---|---|
| committed (layer-scope o_proj operands) | 2.683e-1 | 4.825e-1 (one run), 3.068 (another) |
| else-scoped o_proj operands | 7.224e-1 | — |
| + build-time scalar staging | 7.224e-1 | — |

The attention legs themselves stay healthy in every variant (layer-0 stage
sums q/k/v/attn_out/ao match the host arm within 0.4%; the probe holds 64/64
slices). The instability comes from the PRE-EXISTING dense-MLP device defect
the W2 record already flagged (report.md §3 reading 2: dense gate_proj
outliers, ~1000x sum deviation, "the next site to chase"): the FLINT-P150
model sits downstream of that staging every layer, so no model-level band can
hold. STOP per the spec's stop condition; the blocker is filed as
`ISSUE-LOCAL-01M697FEEHXV9Y4CTYEJGRYHQR` (same directory, row
FLINT-P150-ATTENTION). The wiring lands as a STAGED SLICE: default OFF, the
row keeps owning the wiring, adoption is owed on that issue.

## Stage 3 — the serving trace (RECORDED, no adoption verdict)

Keep-quant arm (`VT_FLINT_KEEP_QUANT=1`), published Q4_K_M, 32-token prompt +
6 greedy steps, one rep per mode, P150, load average ~3-4 (idle gate held):

| leg | prefill32 | decode avg/step |
|---|---|---|
| host attention (`VT_FLINT_TT_DEVICE_ATTN=0`) | 49.5 s | 55.2 s |
| device attention (`=1`) | 48.8 s | 53.0 s |

Attention is a minor share of the MoE-dominated ~50 s step at these shapes;
the ~2 s/step delta is a tie-class observation, not a claimed speedup. The
wiring stays env-gated default OFF. Logs `trace-kq-off.log`,
`trace-kq-on.log`.

## Recipe and environment

Worktree `/tmp/flint-p150-attn`, build `/tmp/build-p150attn`
(`-G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_TENSTORRENT=ON`,
CMAKE_PREFIX_PATH=/tmp/umdtrial-install/lib64/cmake;/tmp/umdtrial-install/share/cmake),
builds `taskset -c 64-127 ninja -j 64`, serving legs `taskset -c 32-63`.
Device: the LOCAL P150, every device run under
`exec 9>$HOME/gpu.lock; flock -x 9`, luwen reset + 15 s before the first run
of a series, `rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory`;
pinned substrate env: LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:
/tmp/umdtrial-install/libexec/tt-metalium,
TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium,
TT_METAL_HOME=~/Sources/tt/tt-metal-pin (env-tt-common.sh NOT sourced).
Governor: `performance` (logged, unchanged). Model artifact:
`/mnt/models/flint-GGUF/flint-Q4_K_M.gguf` (the published container pin in
docs/USAGE.md).

## Deviations

- Stage 2's "full-model keep-quant band gate is the model-level proof" is NOT
  met — the gate passed once but cannot hold against allocation-order
  perturbations; recorded as the stop above, not waived.
- The stage-2 commits are a staged slice per AGENTS.md §"Nothing lands dead":
  the wiring is reachable only behind `VT_FLINT_TT_DEVICE_ATTN=1`; the owning
  row keeps the wiring; the blocker issue names the owed adoption.
- The W2-era scratch diagnostics reproduced two readback traps on this pin
  (f32 host vectors staged into bf16 DBufs; device-shadow commit vs raw
  DBuf::Download) — documented here because both mimic "wrong kernel" during
  bisection.
