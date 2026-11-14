# Tenstorrent keep-quant 27B decode (capture-default)

Qwen3.8-27B-Q4_K_M on the P150, with the keep-quant int8-dot decode
path (`VT_TT_KEEPQUANT_INT8DOT=1`). The first end-to-end completion of
the 27B bench on the Tenstorrent backend, after fixing seven layers of
capture-time program-cache misses in the decode graph trace capture
(commits `26b88846a`, `69b0d4f8e`, `ead93289b` on
`row/BACKEND-TT-KEEPQUANT-W3`).

## Results

| Model | Batch | Prompts | In/Out | TTFT (s) | TPOT (s) | Output tok/s | Total tok/s | Duration (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q4_K_M | 2 | 2 | 128/32 | 64.5 | 2.0 | 0.49 | 2.45 | 131 |
| Qwen3.8-27B-Q4_K_M | 2 | 2 | 128/32 | 1389 | 29.7 | 0.03 | 0.14 | 2326 |

Both runs: `--num-prompts 2 --input-len 128 --output-len 32 --concurrency 2
--num-blocks 64 --max-num-batched-tokens 64 --seed 0`, 64 generated tokens,
RC=0, 912/916 JIT cache hits (27B) / 691/695 (0.8B).

## Reproduce

```sh
flock "$HOME/gpu.lock" bash -c '
  ~/Sources/tt/luwen/target/release/reset && sleep 15 &&
  cd /tmp/row-tt-w3 && source ~/Sources/tt/env-tt-common.sh &&
  export VT_TT_KEEPQUANT_INT8DOT=1 &&
  timeout -k 10 3600 ./build/examples/vllm-bench \
    --model /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0
'
```

## The 2026-11-05 re-bench (row `QWEN38-P150-REBENCH`)

First current-head measurement since the 09-14 leg. Same recipe and
artifact, `VT_TT_KEEPQUANT_INT8DOT=1`, head `702b19167`, governor swept
(the kolibri1 anchoring method: 3 reps per cell, medians, idle gate,
fresh resets), full logs in
[`../bench-evidence/qwen38-p150-rebench-20261102/`](../bench-evidence/qwen38-p150-rebench-20261102/README.md).

| Leg | Governor | TTFT (s) | TPOT (s) | Output tok/s | Total tok/s | Duration (s) | JIT hits |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-09-14 (baseline) | unrecorded | 1389 | 29.7 | 0.03 | 0.14 | 2326 | 912/916 |
| 2026-11-05 median | performance | 1203.6 | 25.2 | 0.03 | 0.16 | 1996 | 819/819 |
| 2026-11-05 median | schedutil | 1215.5 | 25.5 | 0.03 | 0.16 | 2015 | 819/819 |

−13.4% TTFT and −15.0% TPOT vs 09-14, far beyond the ≤0.1% warm-cell
spread; the governor accounts for only ~1.0%, so the delta is real code
delta (the pool-match, NEON dots, sliding-attention, and slot-serve
fixes named in the evidence README). Warm JIT hit rate is 100%; the
first leg after a build measures 20.3% on a cold `~/.cache/tt-metal-cache`
and is not comparable to a warm baseline.

## Limitations

- 2 prompts only (smoke test); multi-request throughput not measured
- No token-exact gate (correctness not verified beyond RC=0)
- Single seed (0); determinism not tested
- TTOT includes device-resident decode-graph replay overhead
- The 27B prefill (TTFT) dominates the run (~23 min for 128 tokens)

## The 2026-11-13 high-bit arm (row `TT-STAGING-FOOTPRINT`)

UD-Q6_K_XL serves on one P150 after the row-page staging repack (the
64-B-per-block grid padded Q6_K rows x1.22 / Q8_0 rows x1.88 past the
board; the repack stages one row per 64-B accessor page, pad <= 63 B
per row). Artifact:
`unsloth/Qwen3.8-27B-GGUF` @ `ce331efc48d5722b4b8041f72793e530eae18d9b`,
`Qwen3.8-27B-UD-Q6_K_XL.gguf`, 25,299,061,664 B, sha256
`701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917`,
local path
`/mnt/models/unsloth-qwen3.8-27B-gguf-v2/`. Mix: Q8_0 53.5% / Q6_K
42.2% / Q5_K 4.2% / F32 — all registered types, no IQ family.

| Arm | KV pool | Captured? | TPOT (ms) | Tokens | Notes |
|---|---|---|---:|---:|---|
| Q4_K_M (baseline) | 64 blocks | yes (capture-default) | 6607 median | 200 | the standing arm |
| UD-Q6_K_XL | 8 blocks | NO — eager arm (`VT_TT_DECODE_CAPTURE=0`) | 5340.9 median | 64 | ~19% faster TPOT; pool caps max_model_len at 96 |

3 reps per arm, deterministic token IDs (UD-Q6_K_XL sha256
`b36bc61a...` x3; the Q4_K_M stream byte-identical to its recorded
`d49c53d2...` under the repacked staging — the change does not perturb
the existing arm). The captured arm at KV 8 OOMs at trace capture by
~80 MB (fragmentation-bound); the recorded lever is the tied-embed dual
residency. Evidence:
[`../bench-evidence/qwen38-q6k-vs-q4km-20261113/`](../bench-evidence/qwen38-q6k-vs-q4km-20261113/README.md)
and
[`../bench-evidence/tt-staging-footprint-20261113/`](../bench-evidence/tt-staging-footprint-20261113/).
