# TT-ANCHORING — the 6x kolibri-1 P150 discrepancy resolved; the device golden gate root-caused (host thalia, 2026-10-30)

Row: `MODEL-TEXT-kolibri-1-tenstorrent`. Issue:
`ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9`. Session: `row/TT-ANCHORING`,
worktree `/tmp/tt-anchoring`, head `8d1eb0daa` (origin/main).

## Question

The kolibri-1 P150 decode bench measured 548.5 ms/step on 2026-10-21
(`3ce14f35c`) and 3358.1 ms/step on 2026-10-28 (`3284412c0`, the readback
baseline). ~6x. Which of governor state, host load, tt-metal/driver state, or
a real code regression explains it? And what is the root-cause class of the
device golden-argmax gate red?

## Method

Every leg: CPU governor logged at start, idle gate (1-min load < 16) before
start, device reset (`luwen reset` + 15 s + shm lock cleanup) inside
`flock -x $HOME/gpu.lock`, bench `taskset -c 32-63`, 3 reps per cell,
median reported. Decode = 64 greedy steps after a 128-token prefill;
ms/step = decode_s / 64. Governor swept explicitly: all 128 CPUs pinned
`performance` for the perf cells, `schedutil` for the schedutil cell, pinned
back after. Governor state at session start was `schedutil` on every CPU —
someone reset it since the 2026-10-28 pin (`governor-state-at-start.txt`);
re-pinned to `performance` before cell 1.

Binaries:
- `head` = `/tmp/build-anchoring`, `8d1eb0daa`, Release, TENSTORRENT=ON.
- `base1028` = `/tmp/build-readback` (the 2026-10-28 session's own build,
  2026-10-28 15:48, TENSTORRENT=ON; the readback baseline binary at
  `3284412c0` lineage). Provenance: name + mtime only — the 10-28 session
  did not record the exact source SHA into the build dir, so this cell is
  "the binary that measured 3358.1", not a fresh reconstruction.
- `base1021` = fresh build of `3ce14f35c` (the 10-21 refresh's exact tree) at
  `/tmp/build-baseline-1021` — a true reconstruction of the 548.5 vintage.

Substrate unchanged across every cell: tt-metal pin `a585e5744a8` runtime at
`/tmp/umdtrial-install`, weights `/mnt/models/Aleph-Alpha/Kolibri-1`.

## The attribution table

| Cell | Tree | Governor | Load at leg start (1m) | decode s (3 reps) | median ms/step | tok/s | Verdict |
|---|---|---|---|---|---|---|---|
| head-perf | `8d1eb0daa` | performance | 9.4 / 14.0 / 15.2 | 62.31 / 63.98 / 62.94 | **983.5** | 1.017 | — |
| head-sched | `8d1eb0daa` | schedutil | 14.1 / 14.2 / 14.0 | 65.36 / 61.71 / 64.69 | **1010.8** | 0.989 | +2.8% vs perf — governor is NOT the discrepancy |
| base1028-perf | `3284412c0` binary | performance | 15.9 / 15.7 / 15.8 | 201.67 / 207.09 / 230.58 | **3235.8** | 0.309 | reproduces the 10-28 3358.1 within 4% |
| base1021-perf | `3ce14f35c` | performance | 14.1 / 14.1 / 14.1 | 37.10 / 34.85 / 35.53 | **555.1** | 1.802 | reproduces the 10-21 548.5 within 1.2% |

Contamination marks: `base1028-perf-r1` and `-r3` started with a foreign
`rustc` (114–122% CPU, another session's build on cores 64-127) — their load
averages are inflated, but the bench is pinned to 32-63 and the r1 value sits
inside the cell's spread, so the median stands. r3's 230.58 s is the outlier
the median absorbs.

## Verdicts

1. **The 548 number reproduces** on today's host, today's driver, today's
   substrate, at `3ce14f35c` under `performance`: 555.1 ms/step median
   (544.5–579.8 spread), 1.80 tok/s. The 10-21 record was real.
2. **The 3358 number reproduces too**: the very same `build-readback` binary
   that produced it measures 3235.8 ms/step today under a pinned `performance`
   governor and an idle host. The 10-28 session was NOT measuring noise: its
   binary is genuinely ~5.8x slower than the 10-21 tree.
3. **The 6x is a REAL CODE REGRESSION** that entered between `3ce14f35c`
   (2026-10-21) and the readback baseline `3284412c0` — 50 commits, 10-21 to
   10-28. Neither the governor (2.8% effect, swept both states) nor host load
   (both vintage numbers reproduce on today's clean host) explains any of it.
4. **The regression is PARTIALLY repaired at head**: `8d1eb0daa` measures
   983.5 ms/step. The FIX-PAGED-REGRESSION re-gate commits (`f37d01230`,
   `e5dc3de3a`, landed after the 10-28 session) are the plausible repair —
   they are the only default-flipping code between `3284412c0` and head —
   but this session did not bisect the repair, so the attribution of the
   3235.8→983.5 step is inferred, not measured.
5. **A residual 1.77x gap remains at head** (983.5 vs 555.1). Under the
   session's stop-condition threshold ("> 2x the 548 number") this does not
   trigger a stop, and the head CHAIN is internally byte-stable, but it is a
   real, unexplained gap that a row must own. Owed below.

## The device golden-argmax gate: red at BOTH eras, root cause = substrate, not code

`test_kolibri1_tt_b2ii` device gate (the 141/145 argmax gate), fresh
`luwen` reset + tt-metal cache clear before every leg, `performance`
governor:

| Leg | Binary | Chain verdict | Teacher-forced hard flips | flip-record md5 |
|---|---|---|---|---|
| b2ii-golden-head-1 | head | 3/11 match, 8 flips (3 near, 5 hard) | 41 | `80dbd40f` |
| b2ii-golden-head-2 | head | **7/15 match, 8 flips (2 near, 6 hard)** | 39 | `16862d46` |
| b2ii-golden-base1021-1 | `3ce14f35c` | 3/11 match, 8 flips (3 near, 5 hard) | 40 | `900b44ac` |

- The head-2 signature **is the 10-28 signature** (7/15 chain, 6 hard flips),
  reproduced today with a fresh reset and cache clear.
- The `3ce14f35c` binary — the tree the 548.5 vintage was measured on — is
  EQUALLY red on the same substrate today (40 hard flips vs 41). The drift is
  therefore **independent of our code**: same class, same magnitude, same
  prompt/step concentration at both code eras.
- **Root-cause class: substrate-level numeric divergence in the device
  execution path** (tt-metal pin `a585e5744a8` runtime / device+driver
  state), NOT host, NOT governor, NOT our code, NOT the readback path (the
  staged bytes are readback-verified byte-exact: 18.3 GB staged =
  18.3 GB verified, zero refusals, in every leg).
- Flip characterization: divergences concentrate at early steps (0–5) of
  specific prompts ('Ein gutes Buch…' step 0 gap −8.97 nats, 'The capital of
  Australia is' step 0 gap −11.07 nats, 'counting: one two three…' steps
  0–3). The (prompt, step) POSITION pattern is stable across runs and eras;
  the flipped TOKEN at a position is not (cross-run md5s differ). Gaps of
  3–12 nats are far outside fp8-quantization band noise — the CPU fp8 arm
  passes 141/145 against the same goldens — so the device compute of the
  prefill/first-step path diverges materially from the host reference.
- Not the flip RECORD the 10-28 reviewer computed (`59746f64…`): the record
  is not byte-stable across runs (three different md5s today), so that md5
  was a per-run value all along, not a signature to reproduce.
- Substrate-swap attempt: the only other install on the box is
  `/tmp/umdtrial-new-install` = the 2026-09-25 pin `d20b8e27f29`. The head
  binary fails against it at load (`undefined symbol:
  ttnn::transformer::paged_scaled_dot_product_attention_decode…` — the head
  tree calls a symbol the older ttnn does not carry), so the pin-vs-driver
  split would need a rebuild against the older substrate. Attempted,
  recorded, not pursued in this session.

## Issue disposition

`ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9` stays OPEN (the gate never went
green; close-only-if-green). Updated with the root-cause class above.

## Owed

- The residual 1.77x head-vs-1021 decode gap (983.5 vs 555.1 ms/step) and the
  `3ce14f35c`→`3284412c0` 5.8x regression + its partial repair: a bisect row
  must own the exact culprit commit(s) and confirm the repair attribution.
  Not a product fix inside this measurement row.
- The golden-argmax substrate divergence (early-step, 3–12 nat gaps) needs a
  substrate-side investigation (pin advance or device path trace) — the row
  owns it; this session pinned the class and reproduced it across eras.

## Raw logs

Every leg's complete raw log is committed beside this README (`*.log`),
including the governor state at session start. The failed
substrate-swap leg (`b2ii-golden-head-newpin-1.log`) is the ABI-mismatch
record.
