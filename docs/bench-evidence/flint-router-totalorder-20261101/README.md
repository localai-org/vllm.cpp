# FLINT-ROUTER-TOTALORDER — the router tie-break, the measured tie frequency, the parallel top-k it unlocks

Row: `FLINT-ROUTER-TOTALORDER` · spec:
`.agents/specs/flint-router-totalorder.md` · issue
`ISSUE-LOCAL-01M6M9VMVN9JP3XSG7BW8RZTKN` · date 2026-11-04 · worktree
`/tmp/flint-router-order`, branch `row/FLINT-ROUTER-TOTALORDER` (spec commit
`690ba4518`; commits `f88dcfa12` comparator, `96fbb5335` tie stats,
`baf44d49e` parallel top-k). Host: thalia (aarch64, 128 cores), governor
`performance` AS FOUND (logged per leg). Builds CPUs 64-127 (`-j 64`), bench
legs `taskset -c 32-63`, pool matched `VLLM_CPP_CPU_THREADS=32`/`16`.
Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`. Legs:
200 completions, `--input-len 128 --output-len 32 --seed 0 --temperature 0
--ignore-eos` (default), the keep-quant paged serving recipe
(`VT_FLINT_KEEP_QUANT=1`). `16t`/`32t` = CPU thread count AND request
concurrency (pool matched), per the house recipe.

## 1. The comparator change (commit `f88dcfa12`)

Both flint.cpp selection sites (the reference forward and the
`FlintMoeBlock` host top-k) order by score descending, then by LOWEST expert
index on an exact f32 tie — a total order. The score-only comparator left
the order among equal elements (and so the selected expert set feeding the
gather) unspecified.

llama.cpp note for the record, no behavioral claim: its CPU router selection
(`ggml_argsort_top_k`) orders by value only, so its tie order is likewise
unspecified; its CUDA MoE top-k kernel (`ggml/src/ggml-cuda/topk-moe.cu`)
already ranks with `other > score || (other == score && g < lane)` — the
same lowest-index tie-break this row pins. Nothing upstream is violated.

## 2. The equivalence gate (the falsification harness, repurposed)

`tests/test_flint_router_topk_equiv` (the FLINT-CPU-DESCENT-2 harness) is now
the equivalence gate. ReferenceRouting = the landed serial total-order
selection; CandidateRouting = the parallel-shaped CHUNKED-MERGE selection
(six 64-expert chunks, bounded-insertion local top-k per chunk, partial_sort
merge under the SAME total order) — the combine step a parallel selection
must get right, well-posed only because the comparator is a total order.
Cases, all green after the fix (15582 assertions):

- randomized logits, 4 seeds × 64 reps, single + random batches to t=32;
- tie-saturated logits (5-value corpus; saturated ±100/0/30 corpus), 128+128 reps;
- boundary-tie batches (3-value corpus, mass ties at the k boundary), 64 reps;
- an explicit lowest-index-wins assertion at the selection boundary;
- the falsification RECORD (documentation, not a gate): under the superseded
  score-only comparator, libstdc++ partial_sort disagrees with the total
  order on **128/128** tie-saturated reps.

Production shape: `FlintMoeBlock` runs the per-row selection through the
host row seam (`host_parallel::ForOutputRows`). The seam partitions token
rows only; each row runs the identical serial instruction sequence, so the
parallel result is bit-identical to the serial one by construction
(host_parallel.h's determinism contract). The size guard (`kMinParallelWork`
= 65536) keeps the selection INLINE at the serving batch sizes measured here
(t·384 < 65536 up to t=170): decode pays no pool kick.

## 3. The tie frequency over the real serving run

Instrument: `VT_FLINT_ROUTER_TIE_STATS=1` (default-off; aggregated in a
leaky static, one `[FLINT-ROUTER-TIE-STATS]` summary line at process exit —
a function-local static would be destroyed before the atexit printer, which
is exactly why the first samples printed mangled and the counter holds a
leaky singleton). Per router call (one token × layer) it classifies:
duplicates anywhere among the 384 sigmoid scores, duplicates within the
top-6 head, and the BOUNDARY tie (`score[5] == score[6]`) — the only tie
class that can change the selected expert set.

**The authoritative leg** (`ab-after-32-stats.log`, 200 completions, 32t,
this branch, parallel build):

| counter | value | rate |
|---|---:|---|
| router calls | 924,173 | |
| calls with ≥1 tie anywhere | 22,087 | 2.39% |
| duplicate scores anywhere | 22,115 | |
| duplicates inside the top-6 head | 4 | |
| **boundary ties (selection-changing)** | **2** | 2.2 ppm |

Boundary-tie samples (the total order admits the LOWER index):

- `expert_73=expert_147@bits_1050326934`
- `expert_89=expert_227@bits_1044172901`

Earlier legs agree within batching noise (after-32: 891,395 calls / 22,450
tie calls / 4 top-k / 2 boundary; after-16: 1,033,864 / 23,748 / 2 / 2; a
24-completion leg: 143,870 calls / 0 top-k / 0 boundary).

**Outcome: TIES OCCUR.** The comparator change is therefore NOT a provable
no-op on real inputs: on the two boundary inputs of this leg, the selected
6th expert changed from the unspecified partial_sort choice to the
lowest-index expert (73 over 147, 89 over 227). Per the spec this is the
NUMERICS ADJUDICATION — lowest-index-wins vs the unspecified status quo —
**FLAGGED FOR THE OPERATOR** with the counts above. The keep-quant serving
gate stayed UNMOVED across the change (section 5), so the measured effect on
the gated chains is nil; the adjudication governs whether the pin is
accepted as the reference behavior.

## 4. Router phase before/after (VT_FLINT_PHASE_TIMING, no instruments)

Interleaved ABAB legs, per-forward `batch=<t>` phase lines, medians (ms).
`moe_router` = router matmul + host top-k. The box drifted between windows
(up to ~20% on untouched phases — e.g. `before-32-nostats` vs
`before-32-clean`, same binary), so ONLY same-window pairs are evidence;
the first pair is the clean one.

| leg | window | moe_router | moe_routed | wall | verdict |
|---|---|---:|---:|---:|---|
| ab-before-32 (base) | A | 152.377 | 805.5 | 1297.5 | — |
| ab-after-32 (row) | A | 152.716 | 799.9 | 1293.7 | **tie (+0.2%)** |
| ab-after-32b (row) | B | 152.592 | 797.6 | 1294.0 | — |
| ab-before-32b (base) | B | 117.880 | 636.7 | 1036.3 | window shifted; not comparable |
| ab-before-16 (base) | C | 7.163 | 262.6 | 470.8 | — |
| ab-after-16 (row) | C | 7.335 | 259.1 | 468.7 | **tie (wall −0.4%)** |

**Finding, honest against the spec's premise**: the router phase did NOT
regress and did NOT improve at 16t/32t serving shapes — a measured tie. The
spec's "partial_sort doubles at 32t / the parallel lever is worth ~2-5%"
premise attributed to the selection what is actually the router matmul's
thread-dispatch behavior; and at these batch sizes the size guard keeps the
selection inline (t·384 < 65536), so there is nothing to speed up. The
parallel seam engages above ~170-token batches (large prefill chunks), where
it parallelizes across token rows. The FLINT-CPU-DESCENT-2 STOP is still
resolved: the selection is now deterministic, and any future parallel
selection is provable — but the lever has no measured win at serving
concurrency, and the row does not claim one. A first-sweep set of legs ran
with the tie-stats full-sort enabled and showed the instrument's own cost
(moe_router 3.97→7.23ms at 16t with stats on); the stats env is
measurement-only and default-off.

## 5. The serving gate (keep-quant paged token gate)

`test_flint_gguf` on the FINAL build (parallel seam landed, comparator +
stats): **53/60 greedy tokens exact, 7 near-tie, 0 hard, worst gap 2.48648**
— IDENTICAL to the pre-row baseline (53/60, 0 hard, worst 2.48648). UNMOVED.

## 6. The standing gates (all green on the final build)

| gate | result |
|---|---|
| `test_flint_router_topk_equiv` (equivalence) | 4 cases, 15582 assertions, PASS |
| `test_flint_forward` (sink folding + dtype bit-identity) | 110 assertions PASS |
| `test_flint_paged` | **208/208** PASS |
| `test_flint_paged_engine` (real-weights continuous batches) | 11 assertions PASS |
| `test_mimov2_forward` / `test_mimov2_w1` / `test_mimov2_w2` (MiMoV2 sink gates) | 178/51/67 PASS |
| `check-agent-record` | OK |
| `check-env-doc` (VT_FLINT_ROUTER_TIE_STATS row) | OK (486 vars) |
| `check-commit-style --range origin/main..HEAD` | OK |
| `check-commit-trailers --range origin/main..HEAD --filled` | OK |

## Log inventory

- `ab-*.log` — the interleaved evidence legs (headers carry governor + load).
- `before-*.log`, `after-*.log` (non-ab) — first-sweep and instrument-on
  legs; retained for completeness. `before-32.log` ran while a same-box
  build finished (declared superseded by `ab-*`); the non-ab `after-*` legs
  carry the tie-stats instrument cost in their phase lines and one
  mangled-sample print (the static-destruction bug, fixed in `baf44d49e`).
- `tie-samples-short.log` — the 24-completion stats smoke.
- Every log header records governor (`performance` AS FOUND) and
  `loadavg(pre)`; `loadavg(post) ≈ 32-34` on a 32t leg is the bench itself
  saturating its pinned CPUs.

## Deviations

- The task asked for a claims-file RUNNING line; the record checker
  (`check-agent-record.py`) requires claim row IDs to carry a known matrix
  prefix and no FLINT prefix exists — the FLINT rows live in their issues
  and specs. The row position is recorded in the spec's `## Now` and a
  dated note in the row issue instead.
- The spec's performance premise did not survive measurement (section 4);
  the parallel top-k landed for its determinism/provability value with the
  no-win recorded rather than claimed.
