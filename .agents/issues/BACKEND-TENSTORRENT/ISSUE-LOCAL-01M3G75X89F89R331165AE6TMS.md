ID: ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS
Title: The dense qwen3.5 captured region's replay is broken on both of its models: the anchor (quantized) replays exact zeros, the bf16 9B replays coherently-wrong tokens — the first served replay of either never matches eager
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

TT-DENSE-EMBED-IN-REGION isolation finding (2026-09-27, thalia P150, pin d20b8e27): with the embedding moved inside the captured region (byte-correct at replay: the replayed hidden equals the eager embedding of the same token) and the two expected_cur_pos increments landing (replays serve mechanically, red-first proven), the FIRST served replay of the dense 27B region writes 248,320/248,320 exact-zero logits; the final-norm hidden tap (dnorm, the lm_head input) reads tmin=tmax=0.0 at replay while the same trace's capture-launch reads sane values (-14.9..22.0). The zero originates INSIDE the 64-layer region — not the embedding (proven byte-correct), not the lm_head (the tap is the zero point). After the first zeroed replay the process is PERMANENTLY broken: every later generate — including eager cold steps — embeds correctly (hid0 nonzero) but writes zero logits; the layer-count bisection axis is unusable because the conv-shadow serveability gate refuses to capture any truncated model (captures=0 for every K<64 on both the 27B and the 9B; K=64 captures and desyncs), so the culprit cannot be narrowed by layers.

THE DISCRIMINATOR (measured the same session, CORRECTED by the eager adjudication): the bf16 Qwen3.5-9B (dense GDN-hybrid, 32 layers — the SAME driver, the SAME capture path, the SAME in-region port + increments) ALSO cannot serve correct replays: a fresh-process served-9B leg reads [220,16,220,220] against the eager-9B reference [220,16,220,16] — the first replay-served token diverges COHERENTLY (both sane, no zeros), the spec's adjudication class, and the served logits are a different distribution than the correct step's (probe: argmax 494, first5 [4.5 7.34 3.02 2.36 0.80]) — a stale-or-wrong replay input. An earlier same-session reading ('the 9B serves correctly') checked only coherence and was wrong. The qwen3-0.6B classic-dense captured gate is green on the same pin (125/125, 176 served replays at 0.068 ms, byte-correct), so the pin's trace-replay machinery works. The corrected culprit set: the machinery the two dense models share and the classic-dense lane lacks — the GDN layer path (kGdnDecode/kCausalConv1dUpdate/kGdnPostConv/kRmsNormGated/kSigmoidGateBf16), the fused preamble (kAttnQkNormRopeGate + the per-step cos|sin refill), and/or the unified-KV PA decode — with the quantized arms (the keep-quant decode path / the forced int8-dot GEMMs, tenstorrent_keepquant.cpp:965-973) AMPLIFYING the coherent divergence into the exact-zero desync on the anchor.

NEXT (the follow-up row): per-op isolation inside the quant path — the keep-quant word-decode ops vs the int8-dot GEMMs — then the fix; the two expected_cur_pos increments and the served-replay test (red-first on the masked tree: replays==captures) land with that fix per the TT-DENSE-EMBED-IN-REGION spec's stop-condition remedy. Evidence: docs/bench-evidence/tt-dense-embed-in-region-20260927.md; logs /tmp/embed-region/ (green-served-replay, green-dump2, green-dump3, sweep, sweep2, dense9b-replay, qwen3-capture-gate, red-served-replay3).

## Resolution

-
