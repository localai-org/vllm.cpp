# SPEC — `FLINT-SLIDING-PAGESCAN` — the sliding paged kernel's cost tracks pages scanned, not window-resident keys: diagnose, then fix

- Owning row: `FLINT-SLIDING-PAGESCAN`
- Kind: performance (diagnose-then-fix; the top CPU lever per TT-NONROUTED-ATTRIBUTION and its fresh review)
- Issue: created in the same flow (row-scoped local issue)

## The measured facts (confirmed independently twice: implementer + fresh reviewer)

- Attention is **50.3%** of the Flint keep-quant serving step
  (TT-NONROUTED-ATTRIBUTION table): sliding 37.0% + full 13.3%.
- **Sliding attention grows with sequence length despite the bounded
  window**: ~42 ms at seq 20 → ~67 ms at seq 200 (row), and the fresh
  review independently reproduced **46.7 → 90.8 ms (1.94x)** while the
  routed phase over the same buckets is FLAT (0.994 ratio).
- The window math is bounded (`jmin = max(0, p - window_left)`,
  `probs` sized to the window — cpu_paged_attn.cpp:337-343). Per-token
  KEY work is O(window).
- FLINT-SLIDING-ATTN landed the (token, head)-unit parallel decomposition
  (growth ratio 2.06 → 1.11 @16t) — but the residual 1.11 ratio and the
  absolute cost still grow with sequence length. The window should bound
  them; it does not fully.

## The hypothesis

The paged kernel's cost tracks **pages scanned**, not window-resident
keys. Mechanisms to discriminate:

- **M-A — the btab walk**: the page-table indirection
  (`btab[r * bt_row + (j / block_size) * bt_col]`) is computed per KEY
  (`cpu_paged_attn.cpp:372-374`), inside the per-token loop. As the
  sequence grows, the WINDOW spans more pages (window 513 / block 32 ≈
  17 pages per token at late sequence vs ~1 at early), so the btab walk
  grows linearly with the window span even though the key count is
  bounded. Fix: hoist the per-token page-table slice (compute the
  block indices once per token, then index per key) or cache the
  page-resolved base pointers per window.
- **M-B — the K/V page walk across the arena**: the windowed gather
  touches ~17 scattered pages; as the arena grows, those pages are
  increasingly far apart (cache/TLB misses per constant key count).
  FLINT-SLIDING-ATTN refuted this in the *pre-fix* kernel (the slope was
  linear while pages touched grew 2 → 14 blocks — scatter would have
  curved it); the post-fix (token, head) decomposition may have changed
  the access pattern enough to re-expose it. Evidence: cache-miss
  counters (perf stat) on the attention region at seq 20 vs seq 200.
- **M-C — the residual is not attention at all**: the phase timer's
  scope includes the KV append or the RoPE re-application (both grow
  with sequence length); the attention kernel itself is flat. Diagnose
  first (the attribution method from TT-NONROUTED-ATTRIBUTION applies).

## Deliverable

1. **Diagnose**: attribute the residual 1.11x growth to M-A/M-B/M-C
   with counters (perf stat cache-miss/instruction mix on the attention
   region at seq 20 vs seq 200; the phase timers already exist,
   VT_FLINT_PHASE_TIMING). The FLINT-SLIDING-ATTN row's method applies.
2. **Fix the attributed mechanism** through the shared kernel:
   M-A is a hoist (cheap, bit-identical — the per-key ORDER is
   unchanged, only the btab lookup moves); M-B is a kernel/arena change
   (invasive — measure first, may defer); M-C is a timer-scope fix
   (behavior-neutral). The dtype bit-identity contract
   (test_ops_paged_attn_dtype, INCLUDING the windowed path at
   test:352-359) is HARD: M-A must be bit-identical per-head; the
   suite's windowed leg is the proof.
3. **Verify**: the sliding phase's seq-length growth ratio at the fixed
   base (the 1.11 residual) must drop materially; the serving tok/s at
   16/32t recorded before/after; the token gate (53/60, 0 hard, worst
   2.486) unmoved or re-adjudicated with the measured record.

## Non-goals

- No TT-path change. No new quant formats. No window-semantics change
  (the window math is correct). No adoption of M-B without the counter
  evidence.

## Gates

check-agent-record, style, trailers, env-doc (if any new env), preflight
(named host dotnet failure); test_ops_paged_attn_dtype (the windowed
leg), test_flint_paged 208/208, test_flint_gguf serving gate, MiMoV2
sink gates. Serving legs: bench zone, idle-gated, claims file, governor
logged, 3-rep medians, NO timeout guards.

## Stop conditions

- The residual 1.11x attributes to a mechanism outside M-A/M-B/M-C →
  record the attribution, STOP: the row delivered the diagnosis.
- M-A's hoist cannot hold the dtype bit-identity (it should — the
  per-key order is unchanged) → STOP, the decomposition is wrong.
- M-B's fix is required but cannot hold the bit-identity → record and
  defer: the arena change needs its own adjudication.

## Git integration

One pull request, `row/FLINT-SLIDING-PAGESCAN`, spec committed first,
diagnose commit before any fix commit.
