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

## Record

Matrix record: `KERNEL-ATTN-PAGED-SLIDING` (kernel-matrix), owned by
`CLAIM-FLINT-SLIDING-PAGESCAN` while this row runs.

## Work breakdown

- W1: reproduce the residual growth at the row head (attention-sliding,
  seq ~20 vs ~200, 3 reps, VT_FLINT_PHASE_TIMING=1).
- W2: discriminate M-A / M-B / M-C with counters and timer-scope reading.
- W3: fix the attributed mechanism through `src/vt/cpu/cpu_paged_attn.cpp`
  (or the timer scope in `flint.cpp` for M-C).
- W4: verify — gates, the residual ratio, the serving tok/s at 16/32t.

## Risks/decisions

- The windowed dtype bit-identity (test_ops_paged_attn_dtype:352-359) is
  HARD: any kernel change keeps the per-key order.
- M-B's fix is invasive and may defer with recorded evidence.
- M-C is behavior-neutral: timers only.

## Port map

- `vllm/v1/attention/backends/flash_attention.py` sliding-window metadata
  and the windowed mask (window semantics this kernel mirrors).
- The windowed paged decode path (`vt::PagedAttention`, window_size).
- No new upstream surface: this row diagnoses and tunes the already-ported
  kernel; the reference behavior is unchanged.

## Tests to port

- None new: the windowed leg of `tests/vt/test_ops_paged_attn_dtype.cpp`
  (:352-359) already pins the bit-identity contract this row must hold.
- `tests/vllm/models/test_flint_paged.cpp` and the `test_flint_gguf`
  keep-quant serving gate re-run unchanged.

## Dependencies

- `src/vt/cpu/cpu_paged_attn.cpp` (the kernel),
  `src/vllm/model_executor/models/flint.cpp` (the phase timers, M-C only).
- No dependency change.

## Scope

Diagnose the residual ~1.11x sequence-length growth of the sliding paged
attention phase post-FLINT-SLIDING-ATTN, attribute it to exactly one of
M-A (per-key btab walk) / M-B (arena scatter) / M-C (timer scope), then fix
the attributed mechanism. Excluded: TT path, quant formats, window semantics.

## Upstream chain

- vLLM sliding-window attention metadata and the windowed mask
  (`vllm/v1/attention/backends/flash_attention.py`) at the parity pin
  recorded in `.agents/upstream-sync.md`.
- The pinned-vLLM paged attention is the behavioral reference; this row
  changes no reference behavior.

## Our baseline

- Residual growth ~1.11x (seq ~20 → ~200) post-FLINT-SLIDING-ATTN at 16t;
  the routed phase is flat (0.994). Prior evidence:
  `docs/bench-evidence/flint-sliding-attn-20261031/` and
  `docs/bench-evidence/tt-nonrouted-attribution-20261031/`.
- Gates at base: test_ops_paged_attn_dtype 172/172, test_flint_paged
  208/208, test_flint_gguf keep-quant 53/60 (0 hard, worst 2.486).

## Outcome

Attribution verdict (2026-11-04, evidence
[flint-sliding-pagescan-20261101](../../docs/bench-evidence/flint-sliding-pagescan-20261101/)):
the residual ~1.14x growth is the UNCLIPPED-WINDOW key count (29 → 208 keys
below the seq-513 clip), not pages; M-B is refuted by the flat clipped regime
(attn_sliding slope −0.00001 ms/step over steps 520–690 with keys constant at
513), M-C by reading the timer scope (flint.cpp:1063 — every non-kernel term
is O(batch)) plus the equal growth of the RoPE-free attn_full phase. The M-A
hoist (per-unit page-slice + page-major walk, bit-identical) landed and
measured PERF-NEUTRAL: the per-key cost is the dot/exp/V math, so the ratio
does not drop (1.14 → 1.14 @16t; tok/s 7.35 → 7.38 @16t, 6.94 → 6.84 @32t,
noise). The row stops per its stop condition: the diagnosis is the delivery;
the growth plateaus at seq 513 by correct window behavior. Gates unchanged:
dtype 172/172 incl. the windowed leg, test_ops_paged_attn 1838/1838, NEON
15,200,901 assertions, test_flint_paged 208/208, keep-quant 53/60 (0 hard,
worst 2.486), MiMoV2 sink gates 178/51/67. Rejected: whole-process perf stat
as scatter evidence (model load contaminates the counters); the invasive M-B
arena fix (mechanism refuted, nothing to fix).
