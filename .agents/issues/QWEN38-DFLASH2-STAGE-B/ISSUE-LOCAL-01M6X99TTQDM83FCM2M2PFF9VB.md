ID: ISSUE-LOCAL-01M6X99TTQDM83FCM2M2PFF9VB
Title: DFlash2 stage B: the TT D2D capture lane clobbers the destination slot's recorded geometry, so residual taps read back pool residue
Row: QWEN38-DFLASH2-STAGE-B
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-08
Updated: 2026-11-08
Closed: 2026-11-10

## Problem

CopyDeviceDeviceIfCapture (tenstorrent_residency.cpp:1490) allocates a fresh src-shaped device tensor and installs it as the destination slot's shadow (:1558-1574) even when the destination already owns a persistent device shadow of equal numel but different geometry. The qwen3.5 dense decode aux taps write per-layer [1,H] clones into a persistent [S,H*taps] DBuf slot (qwen3_5.cpp:12430-12441 VT_TT_GDN_REPLAY_TAP comment); the captured trace then bakes reads at the slot's recorded [S,H*taps] geometry while the shadow holds a [1,H] tensor, so tap columns read back pool residue. DFlash2's drafter consumes exactly these taps, so the seam must be correct before stage B's verify capture. Stage B also owes the GDN two-buffer commit-or-restore and the exact-shape spec verify capture arm. Spec: .agents/specs/qwen38-dflash2-stage-b.md

## Resolution

- 2026-11-08 (implementer, row/QWEN38-DFLASH2-STAGE-B @ 49f04cb17): the three
  stage-B units shipped on the branch, host gates green.
  - Fix 1 (this issue's seam defect): CopyDeviceDeviceIfCapture serves the
    destination's RECORDED geometry when the dst slot holds an equal-numel
    device shadow; the src-shaped-install path survives unchanged for the
    no-shadow destinations (#2294/#1625). Red-before:
    docs/bench-evidence/qwen38-dflash2-stage-b-host/red-geometry-mismatch-d2d-copy.log
    (the consume-at-recorded-geometry matmul threw the a/b inner-dimension
    mismatch — the clobbered [1, H] record served). Green:
    green-focused-tests.log (5/5 with the #2294 and inertness-guard cases).
  - Which reshape arm shipped: the spec's named FALLBACK — ttnn::copy into
    the live dst shadow's storage segfaulted at the tt-metal pin, so the arm
    allocates a dst-shaped tensor, copies the reshaped source into it, and
    installs it with the recorded geometry preserved.
  - Fix 2: the dense decode driver stages GDN state shadows before a captured
    spec step (VT_TT_SPEC_STATE_SHADOW, default on); accept frees, reject
    restores at the recorded geometry; the restore declines on
    eviction/geometry change and joins the gdn_idx_changed reset lane. Red:
    red-gdn-spec-state-shadow-rollback.log (via the -DVT_TEST_GDN_SHADOW_STUBS
    reproduction build — the seam did not exist on the head); green in the
    focused set.
  - Arm 3: the (S, 9, spec=true) verify capture is the existing keyed-ring
    spec arm; the replay-tap probe now prints per-tap FNV checksums from the
    corrected seam. The spec shape consumes one qlen of the default
    VT_SPEC_GRAPH_MAX_QLENS=2 budget; the default did not move.
  - Host build repair: the branch's tt-metal shim pointed at the
    experimental spec-subtree include layout and the 17-arg
    chunk_gated_delta_rule; the local durable pin (a585e5744a8) carries the
    restored paths and the 18-arg symbol. Commit 9996ce94d aligns the branch
    with the pin it must link (host-build repair, not a pin advance).
  - PENDING: the P150 device gate (greedy token identity, traced spec arm vs
    eager) — no lease held this session; not waived. The checker battery and
    scripts/agent-preflight.sh --staged run recorded in the row evidence.

- 2026-11-08 (fresh implementer, row/QWEN38-DFLASH2-STAGE-B @ 91622cf06):
  fresh-review FAIL, four findings, all repaired. Each repair is its own
  commit; each mutation proof applied the baseline-restoring change in the
  tree, reran, and restored the source byte-for-byte (git checkout) after
  the red run.
  - Finding 1 (blocking), commit 3bf41335f: the committed geometry case
    detected nothing — its only consumer (the device matmul) serves
    flat-equal bytes, so a clobbered [1, 32] record still produced an
    identical result and the case stayed green under the mutation. The case
    (tests/vt/test_tenstorrent_backend.cpp:6215) now asserts DIRECTLY on the
    slot's served-geometry record via DeviceShadowExact
    (tenstorrent_residency.cpp:1089): the record must still name [S, TAPS]
    and must not name the source's [1, S*TAPS]. Red under the baseline
    src-shaped-install mutation (tenstorrent_residency.cpp:1552 →
    `if (false && ...)`): red-recorded-geometry-d2d-copy-mutation.log; green
    at head: green-recorded-geometry-d2d-copy.log. The stale
    red-geometry-mismatch-d2d-copy.log / green-focused-tests.log pair is
    from an earlier test iteration and is superseded (README note).
  - Finding 2, commit 41aa1e846: the dtype-equality arm
    (tenstorrent_residency.cpp:1587) had no test. New case stages a
    persistent f32 [2, 16] dst shadow of equal logical numel and copies a
    bf16 [1, 32] src: the dtype check must decline the fast path and the
    record must name the served src-shaped geometry. Red with the dtype
    check mutated out: red-shadow-dtype-mismatch-mutation.log; green:
    green-shadow-dtype-decline.log.
  - Finding 3, commit 059d17be6: the base-pointer guard
    (tenstorrent_residency.cpp:1529, ISSUE-LOCAL-01M4JK8PT8NF9TQ7M06VS51JJH)
    had no test. New case copies a 64-byte [1, 32] src into an INTERIOR dst
    (base + one row's stride) in the capture/host-free lane: the D2D arm
    must decline (stages_avoided_device_copy == 0), the host fallback must
    write the interior host bytes, and no device shadow may survive the
    partial host write. Red with `if (d->host != dst) return false;`
    mutated out: red-interior-pointer-guard-mutation.log; green:
    green-interior-pointer-decline.log.
  - Finding 4, commit 91622cf06: the head/baseline attribution was
    irreconcilable because the earlier mapping applied baseline error line
    numbers to the head source. Reconciled (failing-cases-reconciled.md):
    13 cases fail at BOTH 9996ce94d and the repaired head, 3 fail only at
    head, 0 only at baseline; head = 16 failing of 112 in BOTH reruns.
    PagedAttention pure-decode: reproduces in both full-suite runs, passes
    3/3 in isolation — order/state-dependent, NOT a random flake; open for
    bisection. kRmsNormGated committed-gate-shadow: CORRECTED 2026-11-08 —
    NOT a branch regression; it fails byte-identically in isolation at the
    pre-branch baseline 9996ce94d (rms-base.log), so it is pre-existing and
    suite-context dependent; filed as ISSUE-LOCAL-01M6Y0TC5D24YGF74H0TQ2VAN5
    on row BACKEND-TENSTORRENT-GDN, which owns the op family. The three
    head-only names above are pre-existing suite-context items, not
    regressions owned by this row.
  - This issue does NOT close: fix 1's seam is correct and now detected,
    the P150 token-identity gate LANDED 2026-11-08: six legs
    (defaults x3 captured, eager x3), byte-identical greedy token stream
    across all six (sha256 d49c53d2...), warm TPOT within 0.6%
    (6606.84 vs 6648.14 ms), capture engagement proven by the
    instrumented proof leg (docs/bench-evidence/qwen38-dflash2-stage-b-20261108/).
    MiMoV2 sink gates 178/51/67 PASS; preflight green except the two
    known host gaps. The kRmsNormGated defect is real but predates this
    branch and is owned by BACKEND-TENSTORRENT-GDN.

-

## Resolution

LANDED 2026-11-10 via PR #3553 (merged; evidence commit `34fbb6154` on origin/main, re-verified reachable). Device gate PASS: byte-identical greedy token streams across all six legs (defaults x3 captured, eager x3; one sha256 across the six --output-token-ids files), warm TPOT within 0.6% of eager (6606.84 vs 6648.14 ms). Capture engagement proven by the instrumented proof leg. Mutation-proven detection for both fixes; second-pass review PASS over the full series. Evidence: docs/bench-evidence/qwen38-dflash2-stage-b-20261108/ and qwen38-dflash2-stage-b-host/. Owned follow-ups remain recorded above (PagedAttention order-dependence, teardown segfault; the kRmsNormGated defect re-homed to BACKEND-TENSTORRENT-GDN as 01M6Y0TC5D24YGF74H0TQ2VAN5).
