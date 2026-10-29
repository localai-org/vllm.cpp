# SPEC — `TT-KOLIBRI1-READBACK` — eliminate the per-fill readback pivot from the kolibri-1 P150 slot path

- Owning row: `TT-KOLIBRI1-READBACK`
- Kind: performance
- Issue: created in the same flow (row-scoped local issue)

## Scope

The B2b-ii expert-slot path in `kolibri1_tt_forward.cpp` and its shadow in
`kolibri1_tt_stream.cpp`. Host code only; no tt-metal, no kernel change, no
baseline/ABI change. The kolibri-1 P150 decode step stays bit-identical.

## Problem

The 2026-10-18/21 P150 decode measurements (548.5 ms/step median, 1.823
tok/s) carry ~19 slot fills per decode step. Each fill runs
H2D (3.9 MiB packed payload) → **D2H readback → blocking `Synchronize` →
byte-exact verify** (`kolibri1_tt_forward.cpp:733-743`), and each memo miss
re-reads the slot from the device a second time with its own
D2H + `Synchronize` (`kolibri1_tt_forward.cpp:816-827`). Two blocking
round-trips per touched expert per step sit on the critical path, and the
`Synchronize` drains the queue mid-layer, serializing the H2D of the next
fill behind the D2H of the last.

## Ground truth in the tree (what the design stands on)

1. `Kolibri1TTSlotShadow::Record` (`kolibri1_tt_stream.cpp:247-259`) keeps
   the FULL slot bytes host-side (`cell.assign(...)`), invalidated through
   the evict hook (`:229-245`). The host already holds authoritative bytes
   for every filled slot.
2. `stage_expert` (`kolibri1_tt_forward.cpp:703-748`) builds `slot_image`
   host-side BEFORE the H2D, so the compute path never needed the device to
   tell it what it staged.
3. `VerifyReadback` (`kolibri1_tt_stream.cpp:261-279`) compares the D2H
   bytes against the shadow — it proves the H2D landed intact ("no
   extract_shard on this tt-metal" made the round trip the only
   introspection tool). It is an H2D-integrity gate, not a data path.

## Design

Two independent levers, landed as separate commits, measured separately:

**Lever 1 (pure win, no gate loss): the memo-miss path consumes the
shadow's stored bytes.** Add `Kolibri1TTSlotShadow::Bytes(layer, slot)` and
dequant from it instead of re-reading the device
(`kolibri1_tt_forward.cpp:816-827` deleted). The dequant input is
byte-identical: at fill time `VerifyReadback` proved device == shadow, the
shadow is invalidated on every eviction, and the slot-pool policy refuses a
`SlotFor` on a slot the shadow does not hold. Nothing weakens: the
integrity proof still runs at fill time.

**Lever 2 (gate-preserving deferral): stop draining per fill.** The fill
path keeps every D2H verification but drops the per-fill
`be.Synchronize(d.q)` — the D2H copies enqueue on the queue and ONE
drain point per layer (after the fetch loop, before the compute loop)
settles them. The bytes-compares are unchanged; only their timing moves.
A corrupt stage is still caught before any expert computes over it — the
drain precedes the compute loop by construction. If the queue contract
does not allow N outstanding D2H copies into distinct host buffers (the
page-locked-host requirement is a CUDA-side rule, but the tt-metal
`Copy`-into-heap path may hold its own), fall back to one drain per
K fill with K measured, and record the fallback.

**Explicitly out of scope:** removing the verification entirely, moving
the expert dequant/GEMM on-device (the owed fp8-block device dots), the
device-router default flip, and the residency drain
(`#3452` `drain=true` inside `MarkScratchAcquired`) — that one is a
separate suspect with its own measurement and gets its own row if this one
does not close the gap.

## Risks

- The shadow's host copies double the host memory for staged slots
  (already the case today — this change adds nothing).
- `VT_CHECK` failure in `VerifyReadback` currently fires at fill time,
  inside the fetch loop; after Lever 2 it fires at the drain point. The
  refusal message and the `readback_verified_bytes` accounting are
  unchanged, so the existing stage-dump evidence keeps its meaning.
- Steady-state memo hits (measured 100%) mask lever 1's value; the
  measurement must be taken on a churn-heavy schedule (long context,
  wide token routing) where misses actually occur.

## Tests

- Red-before: measure decode ms/step on the P150 recipe (host thalia,
  `taskset -c 32-63`, tt-metal pin a585e5744a8, idle-host gate: load < 16,
  `gpu.lock` held) with `VT_KOLIBRI1_TT_STAGE_STATS=1` recording
  `slot_fills`, and a profile split of H2D/D2H/Synchronize time per fill.
  No timeout guards on the device runs.
- The existing bit-exactness contract is the gate: the decode bench's
  per-step logits must stay byte-identical to the pre-change arm on the
  same prompt (the memo/dequant path is deterministic and the inputs are
  the same bytes). A/B the same binary pair twice; a non-identical step
  is a defect, not drift.
- Mutation: an implementer mutation that corrupts the staged bytes
  (flip one byte of the H2D source after the copy) must still be caught
  by the drain-point verification — the reviewer proves the gate still
  detects the defect it existed for.

## Gates

- `test_kolibri1_tt_decode_bench` byte-identical steps vs base, twice.
- The B2b-ii unit suites (`test_kolibri1_tt*`) green.
- Preflight green modulo the known host `check-windows-portability.py`
  dotnet breakage.
- Remote: the TT-lane CI green (device benches do not run in CI; the
  device evidence lands under `docs/bench-evidence/` with the recipe).

## Evidence

- P150 numbers and the readback-pivot suspect list:
  2026-10-21 measurement (548.5 ms/step), memo stats 100% hit, the
  per-fill pivot ~19/step.
- Code anchors: `kolibri1_tt_forward.cpp:703-748` (fill),
  `:816-827` (memo miss), `kolibri1_tt_stream.cpp:247-279`
  (shadow Record/VerifyReadback), evict hook `:229-245`.

## Stop conditions

- If the per-fill D2H+Synchronize measures < 5% of step time, lever 2 is
  not worth its complexity — land lever 1 alone and record why.
- If the tt-metal queue refuses outstanding D2H copies into distinct host
  destinations, take the K-fill batching fallback; if that also fails,
  land lever 1 and close lever 2 with the refusal evidence.
- If byte-identical steps do not reproduce (any step diverges), stop:
  that is a correctness defect in the change, and the row fails red-first.

## Outcome

Landed 2026-10-29 as 90fe546b9 (PR #3517, squash; levers 51a82216a + d9b5544/d81dd34fb lineage). Measured: 3358.1 -> 3184.6 ms/step (-5.2% same-session A/B, 3-run medians), byte-identical greedy chains 3/3 per arm. Review MERGEABLE (6 claims, 5 mutations); the review's key finding is recorded in the evidence README: the old fill-time verify was a tautology and lever 2's ordering is the first genuine device-vs-host compare. Rejected: removing or weakening any verification (the M1 mutation proved a silent fallback corrupts 4846 fills with zero errors — the drain-point verify is load-bearing). Defaults: best-effort K-doc; the memo consumes shadow bytes; the drain is per layer. Left open: the absolute anchoring (548 vs 3358) + first green device-golden record, owed under ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9; a stderr/counter-assertion gate for readback-verified (candidate); the M3 reachability characterization (drain deletion caught only by the stats counter).

## Git integration

One pull request, `row/TT-KOLIBRI1-READBACK`, spec committed before the
implementation. Both levers in the same PR as separate commits so the
measurement can attribute each.
