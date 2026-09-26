ID: ISSUE-LOCAL-01M3FF1DJA10C7DBY17QGZACSV
Title: The dense captured arm never serves a replay: the #2469 continuation port is missing its two post-replay cur_pos increments, so every decode step re-warms eagerly (67.0 s/request of the 35 s/token wall)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-27
Closed: 2026-09-27

## Problem

Port row: [tenstorrent-dense-embed-in-region](../../specs/tenstorrent-dense-embed-in-region.md) (2026-09-26).

The 27B step decompose (docs/bench-evidence/tt-27b-step-decompose-20260926.md, VT_TT_STEP_PHASES instrument) measured the captured arm's decode as [cold 33.9 s -> capture 5.7 s (+31.1 s trace wait) -> cold 33.2 s] per request, with boundary=1 at EVERY decode step and VT_DECODE_GRAPH_STATS reporting '4 total replays' for a 12-step run — every capture's own launch, never a served replay. Mechanism: Qwen3_5DenseDecodeGraph::Step updates s.expected_cur_pos = seq_lens[0]-1 (qwen3_5.cpp:12400) and checks (seq_lens[0]-1) == s.expected_cur_pos (qwen3_5.cpp:12395), but the #2469 port from qwen3.cpp is missing the two post-replay increments — qwen3.cpp:971 ('++s.expected_cur_pos; // the replay's in-trace plus_one advanced cur_pos', the replay branch) and qwen3.cpp:1118 ('++s.expected_cur_pos; // the capture step's launch ran the trace once', the capture-tail branch). The predicate is therefore false at every mid-request step, tt_boundary fires, s.graph.Reset() (qwen3_5.cpp:12437) destroys the freshly captured trace, and the slot collapses to cold — two of three steps run the full 64-layer forward EAGERLY at ~33.5 s of host dispatch each (the device work is 1.27 s; the legD probe measured gdn 20.3 ms / fa 18.6 ms per layer). The fix is the two missing increments in the dense driver's do_replay branch and capture-tail branch, mirroring qwen3.cpp:971/:1118; expected value ~-6% TPOT directly (the third step becomes a 31.1 s trace wait instead of a 33.2 s eager pass) and it is the precondition for every captured-arm lever to be measurable (INT8DOT re-measure, M=2 batching, the trace-replay per-command overhead that owns the remaining 31.1 s).

## Resolution

**2026-09-27 (the TT-DENSE-EMBED-IN-REGION session): the issue's premise is
falsified and its remaining work moves to
ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS.** The two increments are confirmed
correct and mechanically sufficient to make replays SERVE (red-first: the base
tree reads replays==captures, the increments read replays>captures — measured
on-device), but they are NOT sufficient to make the arm produce tokens: with
the embedding moved inside the captured region (byte-correct at replay,
proven), the FIRST served replay of the anchor model writes 248,320/248,320
exact-zero logits — the `a79a66bc` class — and the process never recovers. The
defect is NOT the increments, NOT the eager embedding (the prior session's
attribution is falsified: the in-region port produces the identical signature),
and NOT the dense driver's frame alone: the dense region's replay is broken
on BOTH of its models — the anchor (quantized) replays exact zeros, and the
bf16 Qwen3.5-9B's first served replay diverges COHERENTLY from eager
([220,16,220,220] vs [220,16,220,16] — the adjudication class) — while the
qwen3-0.6B classic-dense lane replays byte-correct on the same pin (176
served replays), so the culprit is in the machinery the two dense models
share and the classic-dense lane lacks (the GDN layer path, the fused
preamble, and/or the unified-KV PA decode, with the quant arms amplifying to
zeros on the anchor) — the full chain in the new issue and
docs/bench-evidence/tt-dense-embed-in-region-20260927.md.

Per the spec's stop-condition remedy ("the arm stays eager-served; the
increments stay out"), the two increments (qwen3.cpp:971/:1118's mirrors in
the do_replay arm and the capture-tail arm of Qwen3_5DenseDecodeGraph::Step)
stay OUT of the tree — landing them would unmask the anchor arm into the
zero-desync. What LANDS from this row: the in-region embedding port (the
capture scope covers DenseEmbedInto's device work; the eager/cold path
unchanged; the still-masked arm's anchor leg byte-identical to main's
`13c3f70b…`), and the record. The increments land with the quant-path replay
fix, gated by the served-replay test this session wrote (red on the masked
tree at `CHECK(replays > captures)`; its design is recorded in the new issue).
