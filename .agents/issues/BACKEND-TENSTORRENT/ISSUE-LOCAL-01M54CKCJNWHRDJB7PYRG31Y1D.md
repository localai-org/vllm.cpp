ID: ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D
Title: kolibri1 P150: chunked prefill measured 2.08x faster but fails the band-class equivalence — 27 hard flips, worst 10.2 nats; adoption blocked, instrument landed opt-in
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-18
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
- 2026-10-18 (row/tt-kolibri-p150-next): the precision-vs-logic discriminator
  the blocker owed. Instrument: `SCRATCH dbg chunk sweep`
  (VT_KOLIBRI1_TT_CHUNK_SWEEP=1, test_kolibri1_tt_b2ii.cpp) — one gate
  golden, chunk sizes 2/4/8/16, last-prefill-row plus 5 teacher-forced decode
  rows against the serial reference. VERDICT: inherent multi-token prefill
  kernel precision, NOT chunk logic. With a SINGLE chunk (plen 6, one
  6-token forward, zero chunk boundaries — no ordering, slot mapping, or
  cross-chunk mask in play) the last-prefill row already deviates
  rms(d)=1.44 nats, max|d|=6.8 logits; growing the chunk count does not
  scale the deviation (chunk 2/4/8/16 d=0 rms: 1.44/2.12/1.85/1.67). The
  teacher-forced decode rows over the chunked KV stay argmax-identical to
  serial at nearly every position with sub-band gaps (worst seen 1.94 nats,
  inside the 2.5-nat band). So the KV content is approximately right and the
  divergence is born in the multi-token prefill-path kernels' numerics vs
  the decode-path kernels — the same bf16-precision class as the rms_norm
  deviation (ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF), amplified through 50
  layers and the router. Chunked prefill stays opt-in; any future fix is a
  tt-metal prefill-kernel precision change, not a walk change.

