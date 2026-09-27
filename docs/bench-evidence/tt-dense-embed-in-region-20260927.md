# TT dense embed-in-region — the replay arm's port, the defect beneath it, and the bf16/quant discriminator (2026-09-27)

Branch `row/TT-DENSE-EMBED-IN-REGION` @ `ca0643fa1` (the spec commit) + this
row's change. Host: personal Tenstorrent Blackhole P150a workstation
(aarch64); every device command under `flock -x $HOME/gpu.lock`, `luwen reset`
(+15 s, retried 3x), `source ~/Sources/tt/env-tt-common.sh`. vllm.cpp: Release
Ninja build `/tmp/row-embed-region/build` (`-DVLLM_CPP_TENSTORRENT=ON`, the
tt-metal-pin cmake prefix), full tree green. tt-metal: the recorded pin
`vllm-cpp-pin/20260925` (`d20b8e27f29`), build-linkage and runtime trees
verified at the same commit. Model: the anchor APEX I-Nano
(`/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`,
11,240,605,152 bytes); the bf16 discriminator is `Qwen3.5-9B-hf`
(`Qwen3_5ForConditionalGeneration`, 32 layers, GDN-hybrid dense, 19.3 GB
safetensors). Recipe: the #3323 anchor leg verbatim (`--num-prompts 4
--output-len 4 --concurrency 1 --seed 0 --temperature 0 --ignore-eos`, fixture
`tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`) with
`VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1`. Logs under
`/tmp/embed-region/`.

## The port (landed) — the in-region embedding

`Qwen3_5DenseDecodeGraph`'s capture scope now COVERS `DenseEmbedInto`'s device
work — the `qwen3.cpp` lane's proven shape (qwen3.cpp:953-955 replay arm,
:998-1016 capture staging, :1041-1051 in-scope embed), ported verbatim:

- the replay arm refreshes the persistent ids (`WarmDecodeIds`, allocation-free)
  instead of running `DenseEmbedInto` eagerly while the trace is live
  (qwen3_5.cpp:12505-12521);
- the capture arm stages the ids outside the scope, runs the R4 dummy pass
  (program cache), captures `EmbedDeviceIdsInto` + `DenseForwardLayers` +
  `CaptureDecodePosAdvance` in one kFull segment (qwen3_5.cpp:12647-12670,
  :12706-12721);
- the eager/cold path is untouched (the change is capture-arm-shaped).

**Proven correct in isolation**: the replayed embedding is byte-identical to
the eager embedding of the same token (the `[DENSE-DUMP]` probe: the replayed
`hid0=[0.0049 0.0009 0.0009 0.0009]` equals the cold step's embedding of
token 220 exactly), and the capture-launch tokens are correct in every leg
(`[220,17,220…]` — the anchor row 0 prefix).

## The two increments (NOT landed — the spec's stop-condition remedy)

The two post-replay `++s.expected_cur_pos` increments (qwen3.cpp:971/:1118's
mirrors) make replays serve mechanically — red-first proven: on the base tree
the served-replay probe reads `replays(1) == segments_captured(1)` and fails
`CHECK(replays > captures)`; with the increments `replays(2) > captures(1)`.
**But the first served replay of the ANCHOR model emits token 0 and the
process never recovers** (below), so the increments stay OUT per the spec's
remedy ("the arm stays eager-served; the increments stay out"); the two-line
diff is recorded in ISSUE-LOCAL-01M3FF1DJA10C7DBY17QGZACSV and lands with the
region-replay fix (ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS).

## The defect beneath (the row's stop) — the dense region's replay is broken on both of its models

With the port + the increments, the anchor (APEX I-Nano, quantized) leg's first
served replay writes **248,320/248,320 exact-zero logits** (vmin=vmax=0.0,
argmax 0) — the `a79a66bc` signature. Isolation chain (all measured, logs
cited):

1. **Not the embedding**: the replayed hidden is byte-correct (above); the
   prior session's attribution (the eager embedding while traced) is
   falsified — the in-region port produces the identical zero signature.
2. **Not the lm_head**: the `dnorm` hidden tap (the lm_head input, wired
   through the region's `hidden_tap` copy) reads `tmin=tmax=0.0` at replay
   while the SAME trace's capture-launch reads sane values (-14.9..22.0) — the
   64-layer region's output is the zero.
3. **Not the pin's trace-replay machinery**: the `qwen3-0.6B` captured gate is
   GREEN on the same pin (125/125 assertions, **176 served replays** at
   0.068 ms/replay, tokens byte-correct against the committed capture goldens)
   — the `qwen3.cpp` region replays fine.
4. **Not the dense driver's replay mechanics — CORRECTED 2026-09-27 by the
   eager adjudication: the 9B replay ALSO diverges.** An earlier reading of
   this session ("the 9B serves correctly") checked only COHERENCE (no zeros)
   and was WRONG: the fresh-process served-9B leg reads `[220,16,220,220]`
   against the eager-9B reference `[220,16,220,16]` — the first
   replay-served token diverges **coherently** (both sane, no zeros), the
   spec's stop-condition-2 adjudication class. The served replay's logits
   are a different distribution than the correct step's (the probe read
   argmax 494 / first5 `[4.5 7.34 3.02 2.36 0.80]` at one served step) — a
   stale-or-wrong replay input, not a numerics tie.
5. **The corrected discriminator**: the dense qwen3.5 region's replay is
   broken on BOTH of its models — coherent-wrong on the bf16 9B, exact-zero
   on the quantized anchor — while the qwen3-0.6B classic-dense lane (no
   GDN layers, no fused preamble, no unified-KV PA, no keep-quant) replays
   byte-correct on the same pin. The culprit is in the machinery the two
   dense models share and the classic-dense lane lacks: the GDN layer path
   (kGdnDecode / kCausalConv1dUpdate / kGdnPostConv / kRmsNormGated /
   kSigmoidGateBf16), the fused preamble (kAttnQkNormRopeGate + the
   per-step cos|sin refill), and/or the unified-KV PA decode path — with
   the quantized arms (the keep-quant decode / the forced int8-dot GEMMs)
   AMPLIFYING the coherent divergence into the exact-zero desync on the
   anchor. The per-op isolation inside that set is the follow-up row's
   work (ISSUE-LOCAL-01M3G75X…).

Supporting negatives: the layer-count bisection axis is unusable — the
conv-shadow serveability gate refuses to capture any truncated model
(`captures=0` for every K<64 on both the 27B and the 9B; K=64 captures and
desyncs), so the region cannot be narrowed by layers; the truncation sweep
also showed the desync is PROCESS-PERMANENT (after one zeroed replay, every
later generate — including eager cold steps — embeds correctly but writes
zero logits: the ids=[0] rows in the K-sweep logs).

## Gates

| Gate | Verdict |
|---|---|
| Anchor leg byte-identity (the port-only, still-masked arm) | **PASS** — tokens sha256 `13c3f70b3611ff6b59fef413357704ca2bcd84251c72baff450a935de1a049cb`, byte-identical to main's recorded anchor (`[[220,17,220,17],[220,17,220,17],[220,16,220,17],[220,16,220,16]]`); the still-masked signature intact (`4 total replays` = the four captures' own launches, `boundary=1` at every decode step) |
| TPOT (the honest number) | mean **34,642.92 ms** (median 34,632.68) vs main's masked-arm 34,651.60 — **-0.03%, noise-level: the port is perf-neutral on the masked arm**. Decomposition unchanged from the decompose's class: per request [cold 33.9 s (body 33.1) → capture 4.7 s wall + the 31.1 s trace-wait landing in the following step's `gap_prev` → cold 33.2 s]; the ported capture step's embed phase is **0.3 ms** (the `WarmDecodeIds` staging + the dummy pass) where the pre-port step ran the eager `DenseEmbedInto` |
| Served-replay byte-identity (the decisive gate) | **FAILS on BOTH dense models** — the anchor (quantized) replays exact zeros; the bf16 9B replays coherently-wrong tokens ([220,16,220,220] vs the eager [220,16,220,16], a fresh-process leg) — the spec's stop-condition-2 adjudication class; the row stops, the increments stay out, the served-arm TPOT is VOID |
| Device suite `test_tenstorrent_backend` | **PASS — 92/92 cases, 525,723/525,723 assertions** (identical to the standing bar; log `suite-port`) |

## Pre-existing conditions named (not this row's debt)

- The 27B **Q4_K_M** checkpoint-gated lane OOMs at LOAD on this pin
  (268,222,464 B `ttnn::where` output in `DecodeKeepQuantWordsF32` vs
  62,769,408 B free) — the same OOM the int8dot session recorded on
  2026-09-23 ("the 27B gate is unreachable on this build/tt-metal revision
  regardless of the lever"); the near-tie 27B gate cannot run.
- A default `max_num_seqs` (32) drives the GDN unified pool to a geometry
  whose first capture hits the pinned tt-metal's "trace buffer overlaps with
  DRAM activity" fatal at 98.4% bank occupancy; the anchor leg's own engine
  parameters (max_num_seqs 1, block 32, 256 blocks, auto-fit max_model_len)
  capture cleanly — the served-replay probe mirrors them.
