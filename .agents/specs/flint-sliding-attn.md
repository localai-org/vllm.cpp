# SPEC — `FLINT-SLIDING-ATTN` — sliding attention is 37% of the serving step and grows 1.94x with sequence length despite a bounded window: diagnose, then fix

- Owning row: `FLINT-SLIDING-ATTN`
- Kind: performance (diagnose-then-fix; the top CPU lever per TT-NONROUTED-ATTRIBUTION)
- Issue: created in the same flow (row-scoped local issue)

## The measured facts (confirmed independently twice)

- The keep-quant paged serving step spends **37% in attention-sliding**
  (35 layers, window 513, WITH RoPE) — `docs/bench-evidence/tt-nonrouted-attribution-20261031/`.
- Within one 200-step serving run, the sliding phase grows
  **46.7 → 90.8 ms/step (1.94x)** from seq ~16 to seq ~207, while the
  routed phase over the same buckets is FLAT (ratio 0.994). Reproduced by
  the implementer and independently by the fresh reviewer.
- The kernel's window math is correct and bounded:
  `jmin = max(0, p - window_left)`, `probs` sized to the window
  (`src/vt/cpu/cpu_paged_attn.cpp:337-343`). Per-token key work is
  O(window), not O(seqlen). **So the growth is not the window math.**

## Hypotheses to discriminate (diagnose stage)

- **H1 — KV append/table maintenance inside the timer scope**: the phase
  timer may include the per-step KV page append + page-table/btab
  maintenance, which genuinely grows with the sequence. Fix: move it out
  of the timed region / out of the attention call; the attention gate
  stays green.
- **H2 — arena-scatter cache misses**: the ~9 pages a windowed gather
  touches become increasingly scattered across the K/V arena as the
  sequence allocates pages, so cache/TLB misses per windowed gather grow
  even though the key count is constant. Fix candidates: window-page
  compaction at append; per-sequence page locality; prefetch. Each is a
  real kernel change — measure before adopting.
- **H3 — batch-1 serial attention**: `ParallelForRows` splits by TOKEN
  rows; at serving concurrency 1 there is one row, so all 32 heads x 513
  keys x 128 dims run on ONE thread while the rest of the pool idles.
  Fix: parallelize inside the row (per-head split) when rows < pool
  width. This could be a large constant-factor win independent of the
  seq-length growth — and the two effects may compound.

The diagnose stage must attribute the 1.94x growth to H1/H2/H3 (or a
named fourth) with per-phase and per-thread evidence (the phase timers +
a thread-occupancy profile of the attention region), BEFORE any fix is
adopted. Multiple hypotheses may share the attribution.

## Fix stage

Adopt the fix for the attributed cause(s) through the shared kernel —
H1 is a timer/scope change; H3 is a `ParallelForRows` split change in the
paged kernel (head-parallel when rows cannot fill the pool); H2 is a
kernel/arena change and is the most invasive (its adoption may be a
follow-up row if the diagnosis lands late in this one). Every fix keeps
the token gates green: `test_flint_paged` 208/208, the dtype-dispatch
bit-identity contract (`test_ops_paged_attn_dtype`), and the keep-quant
serving gate (53/60, 0 hard — re-pinned only if the fix legitimately
changes tokens, which H1/H3 must not).

## Measured target

Stage-1 attribution + the H3 fix alone should move the 37% phase
materially at batch 1 (the serial-row observation predicts up to ~16x on
the attention region for the parallel part). Record serving tok/s at 16t
and 32t before/after; the residual after this row is the NEXT row's
baseline (no ceiling).

## Non-goals

- No TT-path change. No new quant formats. No window-math change (the
  window semantics are correct).

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure); `test_flint_paged`, `test_ops_paged_attn_dtype`,
`test_flint_gguf` serving gate; MiMoV2's sink-bearing gate if the kernel
region is touched (the sink-seeding history makes that gate mandatory on
any kernel-region change). Serving legs: bench zone, idle-gated, claims
file, governor logged.

## Git integration

One pull request, `row/FLINT-SLIDING-ATTN`, spec committed first,
diagnose-stage commit before any fix commit.

## Outcome

2026-10-31. Attribution: the growth is the UNCLIPPED-WINDOW keys=p+1 term
(window left 512 never clips below p=513, so keys grow 17 -> 208 over the
measured run) executed on ONE thread — the named fourth, amplified by H3.
H1 refuted (timer intercept constant 39.3-41.0 ms across reps; append/slot
map are O(1 token)); H2 refuted (key count not constant in range; slope stays
linear to +/-0.15% while pages touched grow 2 -> 14 blocks). Decisive H3
evidence: the 1-thread control reproduces the slope within 2% (0.2574 vs
0.2630 ms/step) with a 4x larger intercept (GEMVs parallelize, the kernel
does not), and the 35/14 sliding-to-full slope ratio is 2.506 vs 2.50.

Fix: the paged CPU kernel's parallel axis moved from token rows to
(token, head) units (cpu_paged_attn.cpp); per-head key order untouched —
bit-identical by construction, no gate re-pinned. Per-key slope 0.263 ->
0.027 ms/step @16t; sliding late/early ratio 2.06 -> 1.11 @16t and 2.07 ->
1.10 @32t; decode 5.48 -> 7.01 tok/s @16t and 4.91 -> 6.83 tok/s @32t
(3-rep medians, docs/bench-evidence/flint-sliding-attn-20261031/). Gates:
test_flint_paged 208/208, dtype bit-identity 172, test_flint_gguf 184,
MiMoV2 sink gates 178/51/67 — all PASS. Rejected: any timer-scope change
(H1 refuted), any arena/page-locality change (H2 unproven in range; a
follow-up row may measure it past seq 513 where the window binds). The
residual growth is the same key term at its parallel floor; it plateaus at
seq 513 by design. Evidence-dir contamination record: thalia carries a
resident foreign single-cpu test loop inside bench zone 32-63; every kept
leg is canary-gated (early-step median reproduces the untouched intercept).
