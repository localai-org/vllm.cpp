# Spec: the dense driver's embedding goes inside the captured region — the replay arm serves

Row: `BACKEND-TENSTORRENT`. State: STOPPED (2026-09-27) — see `## Now`.
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

## Now

2026-09-27: **STOPPED at the second defect beneath the un-gating — the port
landed, the increments did not.** The embedding DID enter the region (no
tt-metal impossibility): the capture scope covers `DenseEmbedInto`'s device
work, the replayed embedding is byte-correct, and the still-masked arm's
anchor leg is byte-identical to main's `13c3f70b…`. The increments are proven
mechanically correct (red-first: replays==captures on the base tree;
replays>captures with them) but the FIRST served replay of the anchor model
writes 248,320/248,320 exact-zero logits and the process never recovers —
neither written stop condition fires exactly (this is the zeros class, not the
coherent divergence), so the stop-condition remedy is applied by analogy and
named here: **the arm stays eager-served, the increments stay out** (landing
them would unmask the anchor arm into the zero-desync). The defect is isolated
past this row's scope: the dense region's replay is broken on BOTH of its
models — the anchor (quantized) replays exact zeros, and the bf16
Qwen3.5-9B's first served replay diverges COHERENTLY from eager
([220,16,220,220] vs [220,16,220,16], the stop-condition-2 adjudication
class) — while the qwen3-0.6B classic-dense captured gate is green on the
same pin (176 byte-correct served replays), so the pin's trace-replay
machinery works and the culprit is in the machinery the two dense models
share and the classic-dense lane lacks (the GDN layer path, the fused
preamble, and/or the unified-KV PA decode, with the quant arms amplifying
to zeros on the anchor) — owned by
ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS with the full evidence in
docs/bench-evidence/tt-dense-embed-in-region-20260927.md. The increments and
the served-replay test land with that fix.

## Outcome

- **What shipped**: the in-region embedding port only — the replay arm
  refreshes `WarmDecodeIds` instead of embedding eagerly while the trace is
  live; the capture arm stages the ids, runs the R4 dummy pass, and captures
  `EmbedDeviceIdsInto` + `DenseForwardLayers` + `CaptureDecodePosAdvance` as
  one kFull segment (qwen3_5.cpp:12505-12521, :12647-12670, :12706-12721).
  The eager/cold path is untouched. The still-masked arm's capture steps now
  embed in-trace — reached on every capture, byte-preserving.
- **What was measured**: the replayed embedding byte-identical to the eager
  embedding of the same token (the `[DENSE-DUMP]` probe); the capture-launch
  tokens correct in every leg; the still-masked anchor leg byte-identical to
  main's anchor (`13c3f70b…`); the bf16 9B's served replay diverging
  COHERENTLY from eager ([220,16,220,220] vs [220,16,220,16] — the
  adjudication class, corrected from this session's earlier coherence-only
  misread); the qwen3-0.6B captured gate green on the same pin (176
  byte-correct served replays). The served-arm TPOT is VOID (neither dense
  model can serve a correct replay); the masked arm's TPOT is unchanged
  from the decompose's class (see the bench-evidence doc's gates table).
- **What was rejected and why**: landing the increments — they unmask the
  anchor arm into the zero-desync (user-visible brokenness; "never trade
  correctness for throughput"); landing the served-replay test — it is red on
  the masked tree by design (`CHECK(replays > captures)` fails at
  replays==captures), and a permanently-red gate is not a gate; the
  layer-count bisection of the region — the conv-shadow serveability gate
  refuses to capture any truncated model (captures=0 for every K<64), so the
  axis cannot localize the zero (recorded so the follow-up does not retry it).
- **Why each default has its value**: the ids stage through `WarmDecodeIds`
  (not a fresh per-step buffer) because the replay step must be
  allocation-free around a live trace; the R4 dummy pass stays because
  tt-metal refuses new binaries mid-trace; the embedding runs inside the
  scope (not before it) because a replay must re-embed the refreshed ids —
  the whole point of the row.
