# SPEC — `QWEN38-DFLASH2-STAGE-B` — the captured verify trace serves DFlash2: the aux-tap D2D geometry fix and the GDN commit-or-restore rollback

- Owning row: `QWEN38-DFLASH2-STAGE-B` (record ID `BACKEND-TENSTORRENT-QWEN38-DFLASH2-STAGE-B`)
- Parent row: `QWEN38-DFLASH2` (stage 0 study: docs/design/qwen38-dflash2-integration-20261102.md; stage A: docs/bench-evidence/qwen38-dflash2-stage-a-20261103/)
- Issue: created in the same flow (row-scoped local issue)

## What stage B decides and delivers

Stage A's verdict is **COMPUTE-BOUND** (device tail 6623 ms = 99.9% of TPOT;
host wall/step 3.7 ms): DFlash2 stages B–D proceed. Stage B makes the
captured verify trace able to serve speculative decode correctly on the
P150. Two prerequisite defects stand between stage A's trace contract and a
correct on-card verify replay, plus one new capture arm:

1. **The aux-tap D2D re-shadow defect** (the general seam bug). The qwen3.5
   dense decode aux taps write per-layer `[1,H]` clones into a persistent
   `[S,H*taps]` DBuf slot. On the TT capture lane,
   `CopyDeviceDeviceIfCapture` (src/vt/tenstorrent/tenstorrent_residency.cpp:1490)
   allocates a fresh **src-shaped** device tensor and installs it as the
   destination slot's shadow (:1558-1574), clobbering the slot's recorded
   `[S,H*taps]` geometry. The captured trace bakes reads at the recorded
   geometry, so tap columns read back pool residue — documented as a
   pre-existing defect at qwen3_5.cpp:12430-12441 (`VT_TT_GDN_REPLAY_TAP`
   comment). DFlash2's drafter consumes exactly these taps, so the seam must
   be correct before any drafter work.
2. **The GDN rollback**. The captured trace bakes content-keyed state
   gathers at capture time. A rejected speculative step must restore
   pre-step GDN state: a two-buffer commit-or-restore scheme, with the
   `gdn_idx_changed` reset lane (full re-gather) as the fallback when the
   fast restore cannot apply.
3. **The exact-shape spec verify capture arm**. Stage A proved the
   `(S, 9, spec=true)` verify slot key stable for 197+ replays and
   fixed-address persistent inputs work as built. Stage B plumbs the
   spec-shape verify capture so the arm the DFlash2 loop replays is the
   target's own traced verify. Shape-count risk: `VT_SPEC_GRAPH_MAX_QLENS`
   (default 2) bounds how many distinct qlen shapes may capture; the spec
   arm must live inside that budget or the knob's default moves with a
   recorded reason.

## Scope

IN: the `CopyDeviceDeviceIfCapture` geometry-preserving fix (the shared
seam bug — fixed in this flow, not deferred); the GDN two-buffer
commit-or-restore with the reset-lane fallback; the exact-shape spec verify
capture arm on the dense decode driver (qwen3_5.cpp Qwen3_5DenseDecodeGraph,
Step :12396 — the DENSE arm, not the MoE driver at :11677); the replay-tap
probe reading real tap data when `VT_TT_GDN_REPLAY_TAP` is set; the
evidence directory; the stage-B gate measurement (greedy token identity
traced vs eager on the identical workload). OUT: the drafter checkpoint and
buckets (stages C/D), the on-card accept/commit move, any product default
change — the fix changes no default behavior; the spec arm is
reached through the existing spec-decode entry points.

## Upstream chain

vLLM mirrors the serving-path behavior (spec decode accept rule: greedy
accept is the target's argmax — divergence is a wrong implementation, STOP).
The DFlash2 on-card verify-loop structure is the dflash2-vision bundle
(`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` @
`7855cf0d2982064350991bdb8e1ec1db82f914b3`), quoted as provenance, never a
mirror source (oracle registry `vllm-tt-plugin`, gateable = no). Weights:
incoai/Qwen3.8-27B-DFlash2 (drafter) and the pinned Qwen3.8-27B target —
stages C pins these in docs/USAGE.md; stage B runs the target only.

## Port map

| Upstream | Local anchor | Note |
|---|---|---|
| bundle `dflash2_serving.py:898-1053` (three captured traces, fixed-address buffers) | `src/vt/tenstorrent/tenstorrent_residency.cpp:1490` `CopyDeviceDeviceIfCapture` | provenance; the D2D arm we fix |
| bundle `model.py:2549` (target's own traced verify) | `src/vllm/model_executor/models/qwen3_5.cpp:12186/12396` dense decode driver | the verify arm stage B captures |
| bundle `dflash2.py:532-543` (8-row `[anchor, MASK×7]` draft over residual taps) | `Qwen3_5AuxTaps` (qwen3_5.h:222), ForwardDeviceMultiTap :9393/:10966 | the taps the fix makes readable |
| — | `src/vt/tenstorrent/tenstorrent_backend.cpp:77` | the Copy dispatch that reaches the fixed arm |

## Design

### Fix 1: geometry-preserving D2D copy (capture lane)

In `CopyDeviceDeviceIfCapture`, after the existing guards (:1516-1538), when
the destination slot `d` already holds `device_current` shadow of the SAME
logical numel (byte size and dtype already proven equal by
`ShadowLogicalBytes(*s->device) == bytes` and dtype match), do NOT allocate
a src-shaped clone. Instead reshape the source shadow to the destination
shadow's logical shape in-place (`ttnn` reshape view, no data movement) and
`ttnn::copy` INTO the existing destination tensor, leaving `d->device`,
`d->dev_rows`, `d->dev_cols` untouched. Allocate the fresh src-shaped
shadow only when the destination has no current shadow (today's path, which
the #2294 eager lane and the #1625 lending case depend on). Keep every
existing decline guard byte-for-byte: base-pointer check (:1527), byte
contract (:1538), inertness live-read (:1500-1501).

Geometry match on numel: a `[1,H]` clone into a `[S,H*taps]` slot is
legal iff `1*H == S*H*taps` OR the copy's `bytes` equals the slot's
recorded geometry byte size — the tap writer copies the FULL buffer per
step on the DFlash2 lane, so the reshape is exact; a partial-row copy
(the kolibri1 `[1,h]` staging case) fails `ShadowLogicalBytes == bytes`
against the slot geometry and keeps today's path.

### Fix 2: GDN two-buffer commit-or-restore

The captured trace bakes content-keyed state gathers at capture time, so a
rejected step's out-of-place commit invalidates the next replay's baked
read address (the all-zero-logits signature stage A recorded). Before the
spec step, the dense decode driver stages the GDN state slots' shadows into
shadow copies (device-side, the same persistent-slot mechanism the aux DBuf
uses). On accept: free the shadow copies (commit = keep the step's own
out-of-place result; the baked addresses for the NEXT step re-resolve
through the existing content-key path). On reject: copy the shadow copies
back over the live slots before the next replay. When the shadow-copy
restore cannot apply (slot evicted, geometry changed), fall back to the
`gdn_idx_changed` reset lane: force the full re-gather the eager path
already performs. The env knob for the shadow-copy staging is
`VT_TT_SPEC_STATE_SHADOW` (default on for the spec arm; the knob exists so
the fallback lane stays testable).

### Arm 3: the spec verify capture

The dense decode driver requests the verify capture at the spec shape
`(S, 9, spec=true)` — the stage-A slot key, stable across replays. The
capture region covers the verify forward only (the bundle's structure:
draft, verify, extend are separate traces; stage B captures verify only).
The replay-tap probe (`VT_TT_GDN_REPLAY_TAP`) prints the tap-derived
checksums from the now-correct seam, replacing stage A's "recorded as
defect, not printed as data" caveat.

## Tests to port / write

1. **Red-before for fix 1** (host-runnable): extend the capture-lane D2D
   copy test in `tests/vt/test_tenstorrent_backend.cpp` (the inertness-guard
   case lives there) with a geometry-mismatch case: write a `[1,H]` src
   into a slot recorded `[S,H*taps]` under a captured copy; assert the
   slot's recorded geometry is unchanged and the copied bytes read back at
   the recorded geometry. Red on current head, green after fix 1.
2. **Rollback unit**: a host-side test that accepts and rejects a spec step
   and asserts GDN state slot bytes after each; the fallback lane asserted
   by evicting the shadow copy. Red before fix 2 lands (the reset lane alone
   cannot restore — it re-gathers stale content-keyed state), green after.
3. **Device gate** (P150, lease + mutex per `.agents/environment.md`):
   greedy decode token-identity, traced spec arm vs eager, identical
   workload (batch 1, input 32, 200 tokens, governor performance, 3-rep
   medians). Greedy token divergence from the eager arm = wrong
   implementation, STOP. MiMoV2 sink gate mandatory on kernel-region
   changes; dtype bit-identity contract including the windowed leg.

## Risks/decisions

- **The reshape-view copy may not exist for the shadow's layout/memory
  config on tt-metal at our pin.** If `ttnn::reshape` on the pinned
  tt-metal declines the shadow's tiled layout, fall back to allocating the
  destination-shaped tensor and copying into it, then moving it into
  `d->device` with the RECORDED geometry preserved (the record must name
  the served geometry either way). Record which arm shipped in the spec's
  `## Outcome`.
- **`VT_SPEC_GRAPH_MAX_QLENS` budget**: the spec shape consumes one slot of
  the default 2. If the serving path needs both for non-spec shapes, the
  default moves to 3 in the same change with the reason beside it.
- **The two near-identical decode drivers** (MoE :11677, dense :12186):
  stage B touches ONLY the dense arm. The implementer anchors every edit to
  the `Qwen3_5DenseDecodeGraph`/`Step` symbols, not line numbers alone.

## Non-goals

The drafter (stage C), the bucket system with hysteresis (stage D), the
on-card accept/commit move, GGUF quant arms of the drafter, any sampling
path beyond greedy for this stage's gate.

## Gates

1. Focused: the two new host tests red-before, green-after (capture the red
   output in the evidence directory).
2. Full: `scripts/agent-preflight.sh --staged`, the checker battery,
   `python3 scripts/agent-integration.py --base origin/main`.
3. Device: the greedy token-identity gate above, evidence under
   `docs/bench-evidence/qwen38-dflash2-stage-b-<date>/` (flat per-run
   layout: `<run-id>/<file>.log`), governor logged per leg, idle gate
   before each leg (load < 16, `ps aux --sort=-%cpu`).
4. Reachability: the fixed D2D arm must be reached from the capture path
   the gate exercises — the reviewer deletes the
   `tenstorrent_backend.cpp:77` dispatch in a scratch copy and the focused
   gate must go red.

## Git integration

One pull request (the recorded default; no split answer recorded for this
row). Branch `row/QWEN38-DFLASH2-STAGE-B`, worktree `/tmp/q38-stage-b`,
push to `mine`.

## Stop conditions

- Greedy token divergence traced vs eager: STOP, the accept rule is the
  target's argmax — divergence means a wrong implementation, not a tuning
  gap.
- The reshape arm unbuildable at the tt-metal pin AND the destination-shaped
  allocation fallback also declining: record the blocker, leave the seam
  defect open in the issue, do not merge a partial fix that still clobbers
  geometry on any lane.
- Device unavailability (no fleet lease, no idle window): the host gates
  land; the device gate stays PENDING in the issue, never waived silently.
