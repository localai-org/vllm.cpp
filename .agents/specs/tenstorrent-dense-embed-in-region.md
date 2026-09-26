# Spec: the dense driver's embedding goes inside the captured region — the replay arm serves

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-26).
Issue: `ISSUE-LOCAL-01M3FF1DJA10C7DBY17QGZACSV` (this row closes it).
Follow-up to: `ISSUE-LOCAL-01M3FF1`'s STOP-FINDING record (#3323) and the
decompose (#3322).
Git integration: one pull request (spec + the pattern port + the
increments + gates), branch `row/TT-DENSE-EMBED-IN-REGION`.

## Problem

The dense replay arm has never served: the #2469 port shipped the
boundary predicate without the two post-replay `expected_cur_pos`
increments (masking the arm), and un-gating it (#3323's record) revealed
the defect beneath — the dense capture EXCLUDES the embedding (region =
`DenseForwardLayers` only, `qwen3_5.cpp:12526`), so every replay step
runs `DenseEmbedInto` EAGERLY while the trace is live: a fresh `dids`
alloc (`qwen3_5.cpp:9849`), `EmbeddingKernel`'s own device alloc
(`tenstorrent_ops.cpp:219-221`), an eager `ttnn::embedding` program, and
a shadow copy — the documented replay-corrupting class
(`tenstorrent_paged.cpp:2361-2364`). The device desyncs at the first
served replay and never recovers (deterministic, two legs).

## The repair (port the proven pattern)

`qwen3.cpp`'s lane serves replays with a zero-allocation step
(`qwen3.cpp:953-955`, the #2469 era's proven design):
- the embedding runs INSIDE the captured region;
- the token ids stage via `WarmDecodeIds` (persistent per-n device
  buffers, allocation-free refresh — the redesign's W3 discipline);
- `EmbedDeviceIdsInto`'s capture-safe variant handles the embedding
  inside the trace.

Port that shape to `Qwen3_5DenseDecodeGraph`:
1. Extend the dense capture region to include the embedding step (the
   capture scope covers `DenseEmbedInto`'s device work), staging the ids
   via the persistent `WarmDecodeIds` pattern rather than per-step
   fresh allocations.
2. The eager path keeps its current behavior (the change is
   capture-arm-shaped; the eager stream must not move).
3. ON TOP: the two `expected_cur_pos` increments from #3323's session
   (`qwen3_5.cpp:12397` do_replay arm, `:12625` capture-tail arm — the
   two lines are recorded in the issue; the prior session's copy sits
   in a stash at /tmp/fix-replay-port if useful, but they are two lines
   — re-applying from the anchors is simpler).

## Gates

1. **Replays serve**: the anchor leg's `VT_DECODE_GRAPH_STATS` shows
   served replays (replays track steps in steady state), no mid-request
   boundary resets — #3323's green mechanical state, now WITHOUT the
   desync.
2. **The decisive correctness gate**: the fixed tokens equal the EAGER
   stream — byte-identical to main's current anchor tokens (the
   recorded sha256 `13c3f70b…`); NOT the desync's `a79a66bc…` zeros. If
   eager and served-replay diverge in bound values (both coherent, no
   zeros), STOP and report — that is an eager-vs-replay numerics
   adjudication (which side matches the oracle), a separate row.
3. Device suite 92/92 / 525,723 (the arm had zero coverage — add ONE
   served-replay test if the harness allows; if not, name the gap).
4. **TPOT re-measured** on the genuinely replay-served arm (the honest
   baseline the perf campaign needs; no optimization claim — the
   number with its decomposition is the deliverable).
5. Standard gates.

## Risks

- The embedding's device work may not be capture-safe as-is
  (tt-metal's mid-trace constraints) — the capture-safe embedding
  variant exists for qwen3.cpp's lane; reuse it, do not invent.
- The ids staging must be allocation-free in the replay step; a miss
  fatals loudly (the divergence detector).

## Non-goals

- No eager-path numerics change; no dispatch-cost work (that is the
  next perf row); no tt-metal change.

## Stop conditions

- The embedding cannot enter the region without a tt-metal-level
  impossibility → stop, record (the arm stays eager-served; the
  increments stay out; the perf plan adjusts).
- Served-replay tokens diverge from eager (both coherent) → stop,
  adjudication row.
