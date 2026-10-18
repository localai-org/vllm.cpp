# llama.cpp oracle pin advanced to `11fe0215`: gateability on the unsloth Qwen3.8-27B Q4_K_M (2026-10-03)

Oracle: `llama-cpp` (`.agents/oracles/llama-cpp.md`). Prior pin
`10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`, pinned 2026-08-16); new
pin `11fe02151f79c41d0d4af7da708755d73b9c0da6` (llama.cpp main, 2026-10-04
dated tree), pinned 2026-10-03.

## Why the pin advanced

`b10451` REFUSES the artifact
`/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf`
(`missing tensor 'blk.64.ssm_conv1d.weight'` — the artifact's `blk.64` MTP
block carries no SSM tensors and the `qwen35.nextn_predict_layers` KV override
is inert at the pin), which left the TT-KEEPQUANT-INT8DOT gate-1 band without
a denominator
([tt-int8dot-flip-live-gates-20261003.md](tt-int8dot-flip-live-gates-20261003.md)).

## Load evidence at `11fe0215` (loads and generates, 2026-10-03)

- Fresh shallow clone at `11fe02151f79c41d0d4af7da708755d73b9c0da6`, CPU-only
  build: `cmake -S /tmp/llamacpp-main -B /tmp/llamacpp-main/build
  -DGGML_NATIVE=ON && cmake --build ... --parallel 16` on the 128-core aarch64
  host.
- `llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -st --no-jinja -p "The capital of
  France is" -n 4 -ngl 0 --no-warmup`: LOADS, no `blk.64` / SSM /
  missing-tensor error anywhere, generates
  `\n<think>` + `\n` + `The` + ` user`. A chat prompt ("hi") also loads and
  generates. Log: `/tmp/llamacpp-main-load-test.log`; full probe record
  `/tmp/llamacpp-mtp-verdict.md`.
- `llama-cli` needs `LD_LIBRARY_PATH=<build>/bin` on this fresh build
  (unrelated rpath issue, `mtmd_get_cap_from_file`).
- Decode speed on this host: ~0.8 tok/s CPU incremental. Teacher-forced
  full-sequence batch decode (the gate's denominator path) is far cheaper
  than incremental.

## Attribution

Every gate recorded against `b10451` stays attributed to it. From 2026-10-03
the 27B Q4_K_M band-gate denominator is `11fe0215`: `b10451` refuses the
artifact, `11fe0215` loads it.

## Gateability

`gateable = yes`: the oracle demonstrably builds and runs the model on the
exact artifact the TT-KEEPQUANT band gate needs.
