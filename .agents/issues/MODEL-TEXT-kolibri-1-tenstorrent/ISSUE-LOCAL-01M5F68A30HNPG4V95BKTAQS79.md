ID: ISSUE-LOCAL-01M5F68A30HNPG4V95BKTAQS79
Title: kolibri1 P150: the recorded performance numbers are a second vintage and are presented as current
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The Kolibri-1 P150 performance numbers carried by the tree are a second vintage, and the records that carry them do not say which vintage they are: the decode rate, the per-step decode cost and the prompt-prefill time were measured on earlier trees, some of them while the host was busy, and a reader cannot tell that from the text. A clean three-run measurement on 2026-10-21 (origin/main 3ce14f35c, Release build at /tmp/build-p150-bench, host thalia, P150 Blackhole, taskset -c 32-63, tt-metal substrate pin a585e5744a8, model /mnt/models/Aleph-Alpha/Kolibri-1) reads decode 1.82585 / 1.81767 / 1.82310 tok/s (median 1.823), a decode phase of 35.05 / 35.21 / 35.11 s over the bench's 64 incremental steps after a 128-token prompt prefill (median 0.548 s per token), a 128-token prompt prefill of 146.97 / 131.11 / 124.89 s (median 131.1 s; the spread is the prompt forward's load-time variance), and a band-gate verdict that reproduces the standing one byte-identically (host-free ON 3/8 sequences ON, worst in-band divergence -1.74672 nats against the 2.5-nat band edge, margin 0.753 nats). This box carries a ~4 percent single-run noise floor, so a bare tok/s without its run spread is not a measurement: every refreshed number is reported as a three-run set with its median. The refresh must (a) put the current number and its run spread in the record that owns the quantity, (b) keep the superseded vintages with their dates instead of overwriting them, (c) correct the harness description, since the decode bench times 64 incremental steps after a 128-token prompt prefill and the prefill number is for that 128-token prompt, and (d) record what did NOT change and why, so that a reader does not infer that the CPU-side work of 2026-10-19..21 moved the P150 number: the bf16-dot adoption and the dequant-cache default are CPU-path changes, the f32-mean reduction repair is a Tenstorrent-path change that already landed before the standing 2026-10-19 band-gate verdict was recorded, and the P150 path still carries the tt-metal rms_norm deviation that the upstream filing is preparing.

## Resolution

- FIXED AS A RECORDS CHANGE (2026-10-21, `row/P150-NUMBERS-REFRESH`): the
  tree's P150 performance numbers are refreshed to the clean 2026-10-21
  three-run measurement, and every record that quotes them now says which
  vintage it carries. The refreshed records are:
  - `.agents/specs/kolibri-tt.md` `## Now`: the current decode rate, decode
    phase, per-step cost and 128-token prompt prefill, each as a three-run
    set with its median, plus the harness correction, the superseded
    vintages, and the note on what did not change.
  - `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX.md`:
    the band-gate re-measurement, byte-identical to the 2026-10-19 verdict,
    and the note on what did not change and why.
  - `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M54CJVSXAW32PB9AVKGJG2RH.md`:
    the current per-step decode cost, with the superseded vintages and the
    note that the priced fix is not on today's tree, so the number is not a
    verdict on it.
  - `docs/gap-kolibri1-tt-captured-decode.md`: the current P150 table in
    front of the 2026-10-11 study table, which stays as the superseded
    vintage.
  - `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M4N8Y7XPDC74SED0WEDN0YFR.md`,
    `.agents/issues/MODEL-TEXT-kolibri-1-tenstorrent/ISSUE-LOCAL-01M4NZA9B3DBKS3MD0M4TGRVES.md`
    and `.agents/issues/MODEL-TEXT-kolibri-1-tenstorrent/ISSUE-LOCAL-01M4NS6D2GCD5WXBFN24XGXYQ4.md`:
    a dated pointer from the 2026-10-11 A/B numbers to the current one.
  - `docs/bench-evidence/kolibri1-tt-p150-refresh-20261021.md`: the
    evidence document, with the run conditions, the three-run table, the
    band-gate table, the noise floor, the superseded vintages, and the
    reproduction recipe.

  WHAT DID NOT CHANGE, AND WHY: the band-gate verdict did not change, and
  it reproduces byte-identically, so the correctness quantity is stable. The
  CPU-side work of 2026-10-19 to 2026-10-21 did not move it: the bf16-dot
  adoption (`61d333e52`) and the CPU dequant-cache default (`d9e669582`,
  `e451be705`, `3ce14f35c`) are CPU-path changes, and a P150 number cannot
  move because of them. A reader must not infer that the P150 number moved
  because the CPU work landed. The f32-mean reduction repair (`4b3a0091c`)
  is a Tenstorrent-path change and not a CPU-path change, and it landed
  before the 2026-10-19 band-gate verdict was recorded, so that verdict and
  today's run stand on the same TT path. The P150 path still carries the
  tt-metal rms_norm deviation (ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF, with
  the upstream probe ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF), and that
  filing is the open one.

  WHAT DID CHANGE, AND WHAT IS OPEN: the decode speed moved. Today's
  per-step cost is 548.5 ms, while the same decode bench read about
  1556 ms/step on 2026-10-18 (`53c6a4666`), so the step time fell by about
  2.8x in three days. The tree carries no attribution for the fall. The
  leading hypothesis is the f32-mean reduction repair (`4b3a0091c`,
  2026-10-19), the only Tenstorrent-path product change in that window,
  which replaced the pin's tf32 FPU tile-reduce `ttnn::sum` with the
  accurate full-fp32 `ttnn::mean` in the GDN decode dots and the grouped
  keep-quant tails. The second candidate is the host condition, because the
  2026-10-17 and 2026-10-18 sessions recorded host load and concurrent
  builds, while today's three runs are on a quiet host. The attribution
  stays OPEN, and the next traceable step is a same-session A/B of
  `4b3a0091c`'s parent against `3ce14f35c` under the quiet-host recipe.

  NOT CLAIMED: no host-free OFF arm number, because the OFF arm was not
  re-measured in this session, and no movement of the decode rate, because
  it has no earlier vintage on this tree.
