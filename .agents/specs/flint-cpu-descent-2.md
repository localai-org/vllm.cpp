# SPEC — `FLINT-CPU-DESCENT-2` — the remaining CPU levers from the attribution table, post-NEON-dot

- Owning row: `FLINT-CPU-DESCENT-2`
- Kind: performance (the CPU descent continues; the attribution table names what's left)
- Issue: created in the same flow (row-scoped local issue)

## Where the CPU descent stands

The keep-quant paged serving arm: 1.20 → 5.51 (pool match) → 7.01
(sliding decomposition) tok/s @16t, with the NEON q4_K dot adding +6.5%
(serving) and 2.11x on the raw dot. The gated llama.cpp floor is 26.29
tok/s @16t — the residual is ~3.5x, ceiling-free.

The attribution table (TT-NONROUTED-ATTRIBUTION, attention now fixed
by FLINT-SLIDING-ATTN) names what remains, in value order:

1. **Router host top-k** (2.6% @16t, doubling to ~5% @32t): the host
   `partial_sort` top-k is serial and doubles at 32t. Lever: parallel or
   radix-select top-k, or move the selection onto the device (the TT
   path has SigmoidBiasTopK; the CPU path needs its own).
2. **MoE shared expert + combine** (7.4%): the shared expert runs
   per-token like the routed tier but is the SAME weights every token —
   a candidate for batching/caching across the step.
3. **lm_head** (3.1%): the logits projection over the full vocab — a
   candidate for the same NEON-dot treatment the routed tier got
   (the lm_head weight is not Q4_K-quantized in this artifact, so the
   NEON dot does not apply directly; check what class it is and what
   kernel serves it).

The routed phase (33.1%) keeps its named lever: the NEON dot landed;
the grouped-GEMM follow-up (kMatmulBTQuantGrouped) is a separate row if
this row's table shows the routed phase still dominant after 1+2+3.

## Deliverable

Stage 1 (micro attribution, post-NEON-dot): re-run the attribution
table with the NEON dot landed — the shares have moved (the routed
phase shrank; the fixed phases grew as a share). Record the new table.
Stage 2 (adoption): the top lever from the new table, implemented
through the shared seam, gates green, serving numbers before/after at
16/32t. Repeat if the table re-ranks — the row closes when the named
levers are exhausted or the residual is attributed elsewhere.

## The correctness contract (unchanged)

The keep-quant serving gate (53/60, 0 hard, worst 2.486) is the
correctness gate on every change; the dtype bit-identity suite; MiMoV2
sink gates on kernel-region changes; test_flint_paged 208/208.

## Measurement discipline

Bench zone taskset -c 32-63, idle-gated (1-min load < 16 + ps), claims
file, governor logged, 3-rep medians, NO timeout guards. The NEON-dot
row's serving legs used interleaved same-binary A/B — the same
discipline applies.

## Non-goals

- No TT-path change. No new quant formats. No window-semantics change.
- No adoption without the serving gate green.

## Git integration

One pull request, `row/FLINT-CPU-DESCENT-2`, spec committed first,
stage-scoped commits.
