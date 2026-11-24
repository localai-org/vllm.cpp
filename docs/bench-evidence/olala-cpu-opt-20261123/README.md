# olala bf16 CPU optimization A/B — bounded pool + row-parallel GEMVs, 2026-11-23

Axis `p150-olala-bf16` follow-up leg: the CPU decode/prefill cells of the
published baseline
([olala-bf16-cpu-ab-20261123](../olala-bf16-cpu-ab-20261123/README.md))
re-measured after the two optimization commits on
`row/NEW-CANDIDATE-PORTS-olala-bench`. Same artifact, same prompt, same
sampling as the baseline leg; only the vllm.cpp side changed.

## Commits and per-lever deltas (3 legs each, median)

| Stage | Commit | Decode tok/s | Prompt-eval tok/s | Step delta |
|---|---|---:|---:|---|
| Published baseline | `6e5500d62` | 0.3990 | 0.4579 | — |
| Lever 1+2: bounded worker pool + output-row-parallel GEMVs | `d02eff903` | 1.8053 | 1.9247 | **4.52x / 4.20x** |
| Lever 3: profiled waste removal (dead dot, B/C-rms hoist, mimo tables, prefill attention row-parallel) | `9a720e64e` | 1.8908 | 2.0534 | **1.048x / 1.067x** |
| Cumulative | | **1.8908** | **2.0534** | **4.74x / 4.49x** |

Fork (llama.cpp OVHai olala @ `bd20ca9b`): 2.158 decode / 2.615 prefill.
Remaining gap: 1.14x decode, 1.27x prefill — **still open, no ceiling
declared**.

A thread-count sweep on the pool commit measured 8/16/24/32/48/64/96/128
threads at 0.40/0.75/1.03/1.19/1.69/1.80/1.83/1.78 decode tok/s; 96 is the
default (`OLALA_HOST_THREADS` overrides).

## Rejected and why

- **Barrier fusion** (merging same-input GEMVs into single pool barriers:
  MoE 17->4, Mamba 2->1, V-step 5->1): measured **0.95 tok/s decode** —
  interleaving different weight tensors across worker chunks destroys the
  streaming locality each single-tensor site had. Fully reverted; the
  per-site barriers stay.
- **NEON f32x4 Mamba recurrence** (mistral.rs lever 4): not attempted. The
  remaining 1.14x gap's next traceable hypothesis is instead the per-row
  scalar double FADD dependency chain (every row is a serial `s += w*x`
  chain the compiler cannot vectorize without reordering, and reordering
  would break the bit-identical id gate) — a lever that needs an
  exactness-preserving design (e.g. pairwise trees proven id-neutral on all
  four quant arms) before it can land.

## Parity

The 64-id greedy stream is byte-identical to the published baseline at
every stage (id prefix `151644,77091,151646,...` unchanged through index
57, the documented fork-divergence point). `test_olala_paged_engine` 18/18
EXACT and `test_olala_ops` 305/305 (including the q6_k/q8_0 exact and
q5_k_m near-tie per-arm gates) in Release AND bounds-checked Debug after
every lever.

## Host

Ampere-1a, 128 threads, sequential legs, load < 0.08 per core at
measurement. Same build recipe as the baseline (`/tmp/bench-build`,
Release, `VLLM_CPP_TENSTORRENT=OFF`).

## Files

- `vllm-olala-bench-final-stdout.log` — final state, 3 legs + full 64-id stream
- `vllm-olala-bench-pool-only-stdout.log` — lever 1+2 stage, 3 legs
- `model-sha256.txt` — the shared artifact hash
