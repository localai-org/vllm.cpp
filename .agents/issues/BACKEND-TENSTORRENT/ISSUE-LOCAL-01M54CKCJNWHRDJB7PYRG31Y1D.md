ID: ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D
Title: kolibri1 P150: chunked prefill measured 2.08x faster but fails the band-class equivalence — 27 hard flips, worst 10.2 nats; adoption blocked, instrument landed opt-in
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-17
Closed: -

## Problem

The chunked-prefill near-tie adjudication instrument (VT_KOLIBRI1_TT_CHUNK_ADJUDICATE=1 in test_kolibri1_tt_b2ii.cpp) walked all 8 gate goldens serial and chunked (8-token prompt forwards), decode teacher-forced on the golden chain: chunked-vs-serial argmax flips 77 (near-ties 50, HARD 27, worst gap 10.21 nats vs the 2.5-nat band), hard flips on 6 of 8 prompts including decode position 0. Prefill total serial 139.4 s vs chunked 67.1 s (x2.08) — the speed is real but the greedy chain moves materially, so adoption is NOT justified by the committed equivalence evidence; the honest blocker is recorded. The bench knob (VT_KOLIBRI1_TT_DECODE_BENCH_PREFILL_CHUNK) and the instrument stay opt-in.

## Resolution

- 2026-10-17 (row/tt-kolibri-p150-resume, post-#3452 deterministic chain, pinned
  device recipe): the adjudication REPRODUCES the blocker. Chunked-vs-serial
  flips 78 (near-ties 47, HARD 31, worst gap 9.957 nats vs the 2.5-nat band),
  hard flips on all 8 gate prompts including decode position 0 on 4 of them;
  prefill total serial 146.3 s vs chunked 71.0 s (x2.06). The speed is real;
  the band-class equivalence precondition fails at many positions, so
  adoption stays blocked exactly as recorded above.

