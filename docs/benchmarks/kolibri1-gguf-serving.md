# Kolibri-1 GGUF (APEX I-Micro) on CPU, serving path

The mudler APEX k-quant conversion of Kolibri-1 served through the production
engine (`vllm-bench`, full `AsyncLLM`/V1 stack), against the fp8 safetensors
arm on the SAME box and the SAME workload. Measured 2026-10-30, the shared
aarch64 128-core dev box, bench zone `taskset -c 32-63`, governor
`performance` on all CPUs (read 17:00Z), idle-gated before every leg (1-min
load < 16 plus a process scan; the per-leg gate is in the evidence logs), at
this branch's head; 3-rep medians, `--num-prompts 8 --input-len 128
--output-len 32 --concurrency 4 --seed 0 --temperature 0`, decode-throughput
axis. Artifacts: I-Micro sha256 `3edfaf09…c90e0e`, fp8 dir
`/mnt/models/Aleph-Alpha/Kolibri-1` — the pin and recipe are in
[USAGE](../USAGE.md#kolibri-1-from-a-gguf-k-quant).

| Threads | fp8 safetensors | GGUF I-Micro | ratio |
|---:|---:|---:|---:|
| 16 | 4.46 tok/s (4.44/4.46/4.48) | **4.97 tok/s** (5.01/4.97/4.97) | 1.11× |
| 32 | 5.77 tok/s (5.77/5.76/5.80) | **6.55 tok/s** (6.49/6.55/6.58) | 1.14× |

Honesty constraints on every quotable number above:

- The tier is **APEX I-Micro** specifically. The tier sweep measured the
  BIGGER I-Mini tier FARTHER from the fp8 oracle on the token cross-check
  (25/33 exact vs 37/44, worst topk 5.03 vs 3.245) — do not generalize the
  numbers to other tiers.
- The gate is the **developer-adjudicated quantized-arm record** (2026-10-30,
  Option A: 37/44 exact greedy positions, worst topk diff 3.245, 2 hard flips
  vs the fp8 goldens), NOT a 2.5-band token-exact gate. The fp8 2.5 band
  governs same-weights kernel comparisons only.
- **No quantization-matched floor exists**: the `llama-cpp-kolibri1` oracle
  pin (`b10451`) predates kolibri1 support in llama.cpp (PRs #30673/#30865),
  so the llama.cpp-denominator comparison is a named follow-up. The 1.1×
  figure is the GGUF-vs-fp8 INTERNAL ratio, both sides ours, same workload.

The GGUF arm serving faster than fp8 is consistent with its load-time
dequant-to-bf16 design: both arms run the same forward on bf16 weights, and
the k-quant container moves fewer bytes from storage/page-cache into RAM.

Recipe, gate logs, and the full disposition:
[the evidence report](../bench-evidence/kolibri1-gguf-arm-20261030/README.md).
