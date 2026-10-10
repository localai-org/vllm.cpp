ID: ISSUE-LOCAL-01M4JK8PT8NF9TQ7M06VS51JJH
Title: tt host-free decode corrupts kolibri1 activations on P150
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: -

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
