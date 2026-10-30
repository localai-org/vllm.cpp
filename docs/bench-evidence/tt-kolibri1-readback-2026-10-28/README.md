# TT-KOLIBRI1-READBACK — P150 decode evidence (host thalia, 2026-10-28)

Recipe: `test_kolibri1_tt_decode_bench`, 128-token prefill walk, 64 greedy
decode steps, `taskset -c 32-63`, tt-metal pin `a585e5744a8` substrate at
`/tmp/umdtrial-install`, device reset (`luwen reset` + shm lock cleanup)
before each leg, `flock -x $HOME/gpu.lock` held across each leg,
`VT_KOLIBRI1_TT_STAGE_STATS=1 VT_KOLIBRI1_TT_MEMO_STATS=1`. Idle-host gate:
each run waits for 1-minute load < 16 before starting; the run's own thread
pool drives load back up during the run (self-inflicted, verified via
`ps` — no foreign CPU hogs during legs). Only same-session A/B ratios are
trusted. 3 runs per arm, median reported.

| Arm | decode s (3 runs) | median | ms/step | prefill median s | CHAIN |
|---|---|---|---|---|---|
| baseline (3284412c0) | 214.919 / 216.426 / 211.108 | 214.919 | 3358.1 | 320.019 | identical |
| lever 1 (51a82216a) | 205.903 / 205.615 / 207.358 | 205.903 | 3217.2 | 300.757 | identical |
| lever 1+2 | 203.009 / 203.812 / 201.830 | 203.812 | 3184.6 | 292.596 | identical |

All arms produce the identical greedy CHAIN
(`10,32,325,8676,...,286,` — byte-identical decode steps, the spec's
correctness gate). The 2026-10-21 spec figure (548.5 ms/step) was not
reproduced on this box/session; this row's comparisons are all
same-session A/B against the baseline leg above.

- Lever 1 (memo miss dequants from the shadow): −141 ms/step vs baseline
  (−4.2%).
- Lever 2 (one drain per layer): −33 ms/step vs lever 1 (−1.0%);
  −173 ms/step total vs baseline (−5.2%).
- The `lever2-INVALID-*.log` leg is quarantined: an accidental
  `git checkout` that reverted the reviewer mutation also wiped the
  uncommitted lever-2 edits, so that leg re-measured lever-1 code.
  The true lever-1+2 leg is `lever12.log`.

## Mutation RED evidence

**Correction (fresh reviewer):** `mutation-red-drain-verify.log` was taken on
LEVER-1 code — it predates the lever-2 commit by roughly two hours — so the
gate it exercises is the lever-1 readback compare, not the lever-2
drain-point ordering. The reviewer reproduced the equivalent RED on the true
head `d9b55d5e4`: with the mutation applied, one device byte is corrupted
before the readback enqueue and the run aborts at the first fill
(`kolibri1_tt_stream.cpp:272`).

`mutation-red-drain-verify.log`: with lever 1+2 in the tree, a scratch-copy
mutation writes one flipped byte into the staged DEVICE slot after the H2D
lands (single-byte H2D of `byte ^ 0xFF` after a `Synchronize`). The
drain-point verification fires:

```
ERROR: vt: kolibri1-tt B2b-ii: DEVICE READBACK of layer 0 slot 0 does NOT
match the host shadow byte-for-byte ... kolibri1_tt_stream.cpp:272
[doctest] test cases: 1 | 0 passed | 1 failed
```

Two earlier mutation attempts (flipping the HOST source byte after the H2D
enqueue, with and without an intervening sync) did NOT fire, correctly: they
leave the device bytes — the thing the gate verifies — intact. The gate
detects device-vs-shadow divergence, which is the H2D-integrity failure
mode it exists for. The mutation was reverted byte-for-byte afterward
(`git checkout` of the file; the lever-2 edits were re-applied and
re-verified).

## Correctness claims, restated (fresh reviewer)

The claim "all `test_kolibri1_tt*` suites green on every arm" was green with
DEVICE LEGS SKIPPED: the ctest runs completed in 0.29 s with no
`VT_KOLIBRI1_TT_B2II_MODEL` set, so no device leg executed. The device golden
gate is red at base AND head with byte-identical signatures (flip-record md5
`59746f64...` on both) — pre-existing environment drift, not a regression of
this row, filed as `ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9` on row
MODEL-TEXT-kolibri-1-tenstorrent. The CHAIN byte-identity claims in the table
above were re-verified by the reviewer directly from the raw
baseline/lever1/lever12 logs.

## Gate-history nuance

The pre-change fill-time verify was a tautology: it `Record`ed the device
buffer after the sync and then compared that same buffer against itself, so it
could never fail. The genuine compare lived only in the memo-miss re-read.
Lever 2's `Record`-before-enqueue ordering replaces the tautology with a
strictly stronger device-vs-host compare at the drain point. Net effect:
strengthened.

## Notes

- The tt-metal queue accepted N outstanding D2H copies into distinct host
  destinations; no K-fill batching fallback was needed.
- `Record` moved before the D2H enqueue in lever 2, so the drain-point
  compare is device bytes vs HOST image. This is strictly stronger than the
  pre-change ordering (which recorded post-sync device bytes and then
  compared the shadow against the same buffer — a tautology). No byte-compare
  was removed.
- Memo stats on this schedule: steady-state decode runs ~100% memo hits with
  1–5 fills/layer/step continuing (the churn comes from slot swaps, not
  memo misses), which is why lever 1's measured win exceeds what the
  steady-state miss rate alone would predict — the memo-miss arm also
  carries the prefill walk's misses.

> EVIDENCE-LOST (2026-10-30): `lever12.log`, `mutation-red-drain-verify.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
