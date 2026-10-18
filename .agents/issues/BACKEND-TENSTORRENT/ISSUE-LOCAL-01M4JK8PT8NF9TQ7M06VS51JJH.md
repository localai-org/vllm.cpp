ID: ISSUE-LOCAL-01M4JK8PT8NF9TQ7M06VS51JJH
Title: tt host-free decode corrupts kolibri1 activations on P150
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: 2026-10-10

## Problem

On P150 (thalia, /tmp/umdtrial-install), kolibri1 B2b-ii gate: host-free decode ON (default, VT_TT_HOST_FREE_DECODE unset) scores ARGMAX 0/8 with all hard flips and nat gaps -6..-12 (junk tokens); host-free OFF (VT_TT_HOST_FREE_DECODE=0) scores 26/33 with 7 hard flips. The host-free captured-decode path therefore corrupts activations somewhere on device. Prior suspects from the B2b-ii record: device rope, ReshapeAndCache, residual norm paths. Also verify the earlier rms_norm f32-shadow fix covers the host-free path's norm calls.

## Resolution

2026-10-10 (root-cause session, worktree /tmp/vllm-tt-hostfree, branch
row/tt-hostfree-rootcause, stack /tmp/umdtrial-install):

- The kolibri1 TT decode never captures (no Warm*/driver hooks), so the ON
  arm is the EAGER host-free path. Its live op deltas vs OFF at T=1 are: the
  device residual RmsNorm arm (RmsNormKernel), forced device rope at T=1
  (PreferDeviceRope), and the general device-resident dataflow. The device
  RAC (TryReshapeAndCacheDeviceDecode) and device PA decode arms decline for
  kolibri1 (no warmed RacIdx; block_size 16 fails the %32 TILE guard), so
  the B2b-ii record's RAC/PA suspects are NOT on this model's ON path.
- LOCALIZED with env-gated per-stage dumps (VT_KOLIBRI1_TT_STAGE_DUMP on
  both the CPU and TT rows, VT_TT_NORM_DEBUG inside RmsNormKernel): the
  first diverging stage is the L0 residual RmsNorm output of the FIRST step;
  TT stage sums flip between correct (9.51) and corrupted (26.34 /
  1e30-scale garbage / NaN) ACROSS RUNS depending on which reads execute —
  a state-dependent stale-device-shadow serve, not a numeric drift. The
  per-head q/k norms, projections, and rope outputs verified correct.
- FIXED (one concrete instance): the embedding DBuf died at the end of the
  embedding scope while `hidden` kept referencing its pooled block; the
  recycled block's Memset + shadow-slot reuse desynced the embedding's TT
  device shadow. Hold the embedding buffer alive for the whole step
  (kolibri1_tt_forward.cpp). This moved the first step's SCRATCH argmax
  from junk (127907) to near-tie (3990).
- RESIDUE (open): the corruption persists past the fix — with BOTH host
  levers forced (VT_TT_FORCE_HOST_ROPE + VT_TT_FORCE_HOST_RESIDUAL, new
  bisection toggles) the gate is still 0/8 (4 near-ties, 4 hard), so a
  third host-free delta remains in the general eager device-residency
  dataflow: some producer leaves a slot device-current (or host-current)
  while the other side holds stale/garbage bytes, and the host-free ON arm
  is the first consumer that trusts the shadow instead of re-staging from
  host. DBuf::Download (stage dumps) and EnsureDevice2D (op staging) read
  DIFFERENT bytes from the same slot in the same run — the bookkeeping
  invariant "host_current XOR device_current with both sides equal at every
  mark" is violated somewhere between CommitDeviceLogical2D, Memset's
  MemsetDeviceFill/MarkHostWritten arms, and the pool's
  OnScratchBlockAcquired reservation. Owed: a slot-coherency audit with a
  red-first focused test (stage a buffer, memset it, commit a device op,
  read via both paths, assert equality).
- Gate verdicts this session (same binary, tt-smi reset before the
  session): host-free ON 0/8 (8 hard); OFF 26/33 with 7 hard flips — the
  B2b-ii record's baseline reproduced exactly; ON + both host levers 0/8
  (4 near-ties, 4 hard).

2026-10-10 (completion session, same worktree/branch):

- MECHANISM (two invariant breaks past the embedding-lifetime fix):
  1. CopyDeviceDeviceIfCapture (tenstorrent_residency.cpp, the eager
     host-free D2D copy lane) accepted copies whose DESTINATION was an
     INTERIOR pointer of a registered pool block. FindSlot resolves an
     interior pointer to the BASE slot, so the kolibri1 expert
     gather/scatter row copies (base + row*rb; kolibri1_tt_forward.cpp
     :593/:871) installed a whole-slot device shadow for one row at a
     foreign offset, left the slot's host bytes stale, and clobbered the
     recorded geometry — the next EnsureDevice2D refreshed the whole
     buffer from the last row's shadow. FIXED: the lane now requires
     d->host == dst, the guard the eager resident lane
     (CopyDeviceDeviceIfResident) already had. Alone this moved ON from
     8 hard flips to 4 near-ties + 4 hard.
  2. PreferDeviceRope forced the DEVICE rope at T=1 for the whole
     host-free mode, but the R1 force exists ONLY so the captured region
     carries no host op. The eager T=1 device rope replaced the q/k slot
     shadow (CommitDevice2D) while the rope's own async
     slice/multiply/add/concat ops still read the previous storage —
     tt-metal reused the freed blocks under the in-flight reads, and the
     decode consumed garbage rope I/O. Pure-timing sensitive: a mutex +
     stderr print at rope entry scored 26/33; a post-commit drain did
     not (the free precedes it). Retired-shadow retention (the
     retired_pts pattern) is memory-correct but OOMs this device — the
     dequant slot pool owns DRAM to within ~0.6 MB/bank, and any
     retained or extra transient tips the 2.5 MB working alloc into
     fragmentation. FIXED at the invariant level: the force is scoped to
     an ACTIVE capture (tt_capture_active()), which is the case that
     justifies it; the eager decode takes the host apply the >=64-row
     heuristic already prefers. Captured decode keeps the device rope.
- VERDICTS after (same binary, tt-smi reset before each run, env
  TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium):
  host-free ON 26/33 (near-ties 1, hard 6) — the OFF arm's verdict;
  OFF 26/33 (hard 7) — no regression. The gate's own >=141 check stays
  red for BOTH arms on the shared 7-flip residue, which the B2b-ii
  record owns and this issue does not chase.
- TT unit suite (tests/test_tenstorrent_backend, full): 103/105, the two
  failures are exactly the known pre-existing kGdnDecode and
  kMatmulBTQuantGrouped cases. The binary segfaults during teardown
  AFTER printing the full summary (post-device-close static
  destruction); all case results are recorded before it.
- REJECTED: retired-shadow ring (OOMs; +100-400 KB retained tips the
  margin); in-place rope shadow write via ttnn::copy (deterministic OOM
  from copy churn); post-commit queue drain (does not close the
  free-before-drain window); NO_EAGER_ZEROFILL lever (harmless — the
  memset arms are not on the corruption path once fix 1 lands).
- ENV note for the next session: run the gate with
  TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium (not
  the source checkout) and tt-smi -r 0 before each gate run; back-to-back
  heavy runs bus-error in UMD TLB/ARC paths without the reset.

