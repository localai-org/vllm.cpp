ID: ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D
Title: kolibri1 P150: chunked prefill measured 2.08x faster but fails the band-class equivalence — 27 hard flips, worst 10.2 nats; adoption blocked, instrument landed opt-in
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-22
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

  SUPERSEDED 2026-10-22 (row/TT-CHUNKED-NULL-ARM): the root-cause reading
  of this entry is withdrawn. The direction of this entry survives. The
  comparison it rests on had no same-mode null arm, so the flip set it
  measured is contaminated by the chunked path's own nondeterminism and
  cannot be read as a precision signature. The band-class equivalence
  blocker re-confirmed on 2026-10-21 below is unaffected by this
  supersession. The corrected entry is the last one in this section, and it
  carries the new record ISSUE-LOCAL-01M5HHEV4DWX0D4S5E955VE0AN.

2026-10-21: STILL TRUE. The band-class equivalence blocker is unchanged and was
re-confirmed after this record's own entry: `93e37e703` (PR #3468) re-ran the
adjudication and found 78 flips (47 near-ties, 31 HARD, worst gap 9.957 nats
against the 2.5-nat band), hard flips on all 8 gate prompts including decode
position 0 on 4 of them, with prefill 146 s serial against 71 s chunked (x2.06).
The chunked-prefill knob stays opt-in.

What settles it: a tt-metal multi-token prefill-kernel precision change, or an
adopted narrower band. The instrument is already in the tree
(`VT_KOLIBRI1_TT_CHUNK_ADJUDICATE=1` and `VT_KOLIBRI1_TT_CHUNK_SWEEP=1` in
`tests/vllm/models/test_kolibri1_tt_b2ii.cpp`), so a re-measurement is one P150
run.

- 2026-10-22 (row/TT-CHUNKED-NULL-ARM, new record
  ISSUE-LOCAL-01M5HHEV4DWX0D4S5E955VE0AN): the root cause this record stated
  is corrected, and the "What settles it" condition above is superseded with
  it. The 2026-10-18 comparison had NO same-mode null arm. It compared the
  chunked path against the serial path without measuring what the chunked
  path does to ITSELF, so its flip set carries no noise floor and cannot
  carry a precision reading.

  The null arm, measured 2026-10-21 on P150 (origin/main 3ce14f35c, build
  /tmp/build-tt-taps, harness tests/vllm/models/test_kolibri1_tt_taps.cpp,
  the 6-token teacher-forced prompt 5249, 25079, 74493, 15774, 121598,
  116026, model /mnt/models/Aleph-Alpha/Kolibri-1, taskset -c 32-63): the
  CHUNKED prefill path is NOT reproducible run to run, and the SERIAL walk
  is. Three chunked repeats in one process read the same token 42046 with
  the top-1 logit at 6.930672, then 7.048378, then 6.747927 nats. The
  `p3-warm` arm, the THIRD chunked process, read 42046 at 3.597120, then
  6.964228, then 6.917763, a 3.4-nat spread at the first repeat. The second
  chunked process, `p2-warm`, reproduced `p1-cold` exactly, so that spread is
  not a device-state difference. Two serial repeats
  were byte identical, 74493 at 8.055458 both times. A device reset with
  tt-smi -r 0 plus a kernel-cache clear changed nothing between the cold and
  the warm chunked arm, so this is not a cold or warm artefact. The
  per-layer taps localise it: the divergence is first observed at the last
  dumped query row, layer 0 attn_out_pre_norm row 5, where rms(d) is 6.96e-05
  and max |d| is 9.766e-04, while the query row 0 of layer 0 stays bit
  identical at every tap point. The dumps carry the query rows 0 and 5 only,
  so rows 1 to 4 have no reading and the birth point inside layer 0 is not
  localised. At that row, 1296 of 2560 elements differ, 822 by 1 to 2 bf16
  steps, 474 by 3 or more; the max element difference 9.766e-04 is two bf16
  steps at the element that carries it. It grows
  with depth: l_out rms(d) is 1.35e-03 at layer 0 and 3.24 at layer 49, max
  |d| 21.6. The router's expert ORDER flips in 36 of 50 layers between two
  chunked runs. Of those 36 layers, 17 keep the same six experts in a
  different order and 19 change the expert set.

  VERDICT (corrected): the chunked-vs-serial divergence is 5x to 123x the
  chunked path's own run-to-run spread (reproduced 2026-10-22), so the
  divergence is NOT born in the walk logic, and that part of the 2026-10-18
  verdict survives. The flip set the 2026-10-18 sweep measured is
  contaminated by the path's own nondeterminism, so it CANNOT be read as a
  precision signature. The verdict
  "inherent multi-token prefill kernel precision" is NOT ESTABLISHED by the
  evidence this record cites. The earlier verdict overstated what the
  evidence supported: it read a flip set as a precision signature without a
  noise floor, and the 1.44-nat single-forward deviation it measured is the
  same order as the path's own run-to-run spread.

  A reproduction check on 2026-10-22, over the tap artifacts that are still
  on disk and with no device and no model run, bounds the per-layer
  chunk-vs-serial to run-to-run ratio at 5x to 123x, and the layer-0
  attn_out_pre_norm anchor recomputes at rms(d) 4.74e-03 against a
  run-to-run 6.96e-05, a 68x ratio. The 600x upper bound
  and the 2.17e-02 anchor of the 2026-10-21 measurement do not reproduce
  there and are owed to be reconciled. The direction of the verdict does not
  change: the chunk-vs-serial difference is far larger than the path's own
  spread at every layer.

  What would establish the 2026-10-18 claim: a reproducible chunked prefill
  path. Then the chunk sweep's numbers mean what they were taken to mean,
  and a re-run of the sweep with a same-mode null arm decides the
  precision-versus-logic question. The instrument now carries that null arm:
  the SCRATCH chunk sweep runs the same chunked walk twice per chunk size
  and prints the chunked-versus-chunked difference beside the
  chunked-versus-serial one. Chunked prefill stays opt-in.

  Owed, carried by ISSUE-LOCAL-01M5HHEV4DWX0D4S5E955VE0AN and by the row
  spec .agents/specs/kolibri-tt.md: the chunked prefill path's
  reproducibility, the re-run sweep with its null arm, and the mechanism
  hunt. The measurement and its provenance are in
  docs/bench-evidence/kolibri1-tt-chunked-null-arm-20261022.md.
