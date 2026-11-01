# G5 AVX2 K-quant tier — Flare GSQ evidence (2026-10-31)

Row `QUANT-GGUF-CIQ-GEMM`, spec
[`gguf-compute-in-quant-gemm.md`](../../specs/gguf-compute-in-quant-gemm.md)
§"G5 plan". The change adds the x86 AVX2 arms of llama.cpp b10451
`ggml/src/ggml-cpu/arch/x86/quants.c:2018` (Q4_K), `:2406` (Q6_K) and `:3364`
(IQ3_S) as a runtime-selected tier, with the portable kernels unchanged as the
universal fallback.

## Artifact and host

| Item | Value |
|---|---|
| Model | `models/Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M.gguf` |
| Model size / sha256 | 3,980,479,744 B / `2607106302272f4af6fd7f1c4aa6af0f21c356a972af3c2bb6412ad00fe48681` |
| Tensors | 234 IQ3_S, 104 Q6_K, 42 Q4_K |
| Candidate binary sha256 | `2f3b4c5d023cad0e225f5096f6fcfefcec6535dc6faf9e40c96bc267d00da0fa` |
| CPU | Xeon E5-2696 v3, 18 cores / 36 threads, AVX2 (no AVX-512) |
| Threads | `VLLM_CPP_CPU_THREADS=18`, `OMP_NUM_THREADS=4` |

Host was shared with other agents for every run. Each JSON records the
`/proc/loadavg` at the end of its run; the A/B below interleaves the arms so the
drift applies to both.

## Same-binary A/B (the kernel contribution)

One binary, one command, `VT_CPU_QUANT_AVX2` is the only difference. Greedy
32-token completion, one 8-token warmup first, `keep_quant=1`.

```sh
python3 .agents/evidence/quant-g5-avx2/bench.py \
  --server /tmp/g5-obj/lumenengine-server \
  --model /mnt/openwebui/AIKAR/Lumen-3.8-Flare/models/Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M.gguf \
  --port 11603 --out .agents/evidence/quant-g5-avx2/ab-<arm>-<n>.json
```

| Arm | Run | Request 1 (s) | Request 2 (s) | Mean (s) | loadavg |
|---|---|---|---|---|---|
| `VT_CPU_QUANT_AVX2=portable` | 1 | 16.874 | 16.630 | 16.75 | 18.72 |
| `VT_CPU_QUANT_AVX2=portable` | 3 | 16.536 | 16.689 | 16.61 | 18.01 |
| `VT_CPU_QUANT_AVX2=avx2` | 2 | 6.158 | 6.188 | 6.17 | 17.51 |
| `VT_CPU_QUANT_AVX2=avx2` | 4 | 6.403 | 6.459 | 6.43 | 16.00 |

Mean of means: portable **16.68 s**, AVX2 **6.30 s** → **2.65×** decode
speedup from the three kernels, on this host, under identical contention. The
candidate served both arms. Files: `final-<arm>-<n>.json`.

An earlier 4-arm A/B on the pre-cleanup binary
(`0d6dc8f8…`, identical kernels, selector accepted an undocumented `0` alias)
measured 16.61 s portable against 6.36 s AVX2 (2.61×); files `ab-<arm>-<n>.json`.

All four runs emitted exactly:

```text
 6, 7, 8, 9, 10, ...\n\n**Explanation:** This is a simple sequence of consecutive integers, where each
```

## Before/after against the installed server

| Arm | Request 1 (s) | Request 2 (s) | Startup (s) | loadavg |
|---|---|---|---|---|
| Installed `bin/lumenengine-server` (portable only) — `baseline-20261031.json` | 14.390 | 14.760 | 26.0 | 9.96 |
| G5 candidate, first run (concurrent 12-job test build) — `avx2-20261031.json` | 12.217 | 11.703 | 30.0 | 27.40 |
| G5 final artifact, A/B mean — see above | 6.30 | | 27.0 | 16.0–18.7 |
| llama.cpp b10451 `llama-server -t 18 -tb 18 -fa on` | 3.91 | 4.24 | 3.0 | recorded in the spec |

The installed binary and the two A/B arms are different builds, so the
same-binary table above is the defensible number; the installed-server row is
the user-visible before state at a quieter moment.

## Focused gate

`/tmp/g5-test-build` is a CPU-only `Release` build with `-Werror`:

```sh
cmake -S . -B /tmp/g5-test-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_METAL=OFF \
  -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_FLASH_ATTN=OFF \
  -DVLLM_CPP_MARLIN=OFF -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_HF_DOWNLOAD=OFF
ninja -j12 test_ops_quant_dot
```

| Log | Tier | Result |
|---|---|---|
| `g5-test-default.log` | `auto` (AVX2 on this host) | 35/35 cases, 285,691 assertions, SUCCESS |
| `g5-test-portable.log` | `VT_CPU_QUANT_AVX2=portable` | 35/35 cases, 285,691 assertions, SUCCESS |
| `g5-test-forced.log` | `VT_CPU_QUANT_AVX2=avx2` | 35/35 cases, 285,691 assertions, SUCCESS |

The G5 case itself ran (not skipped): `-tc='*G5*' --success` reports 1 case,
13,882 assertions for q4_K, q6_K and iq3_s, `wide != portable`, production
`BlockVecDot` routing, and both arms within `1e-5 * ref.l1` of an independent
f64 dequantize-then-dot reference for 1, 3 and 5 blocks.

## Mutation coverage (fresh review, 2026-10-31)

Each mutation ran in a scratch copy, was built with `ninja test_ops_quant_dot`,
and was restored byte-for-byte afterwards (sha256 confirmed after every run):

| Mutation | Location | Observed red |
|---|---|---|
| IQ3_S sign `xor/sub` -> `xor/add` | `cpu_quant_dot_avx2.cpp:380,385` | 6 failures at `test_ops_quant_dot.cpp:765,769` for `name iq3_s`, nblocks 1/3/5 |
| IQ3_S weight `2*ls+1` -> `2*ls+2` | `cpu_quant_dot_avx2.cpp:393-394` | 6 failures at the same two checks |
| production `BlockVecDot` arm removed | `cpu_quant_dot.cpp:1267-1275` | 3 routing failures at `test_ops_quant_dot.cpp:730` for q4_K/q6_K/iq3_s |

The first mutation is the one the pre-repair case missed: `kWeightCases` held
no IQ3_S row, so the G5 filter never reached IQ3_S's numeric battery.

## How the candidate binary was produced

The Flare loader (`gemma4_gguf_weights.cpp`) lives on the GSQ worktree, whose
HEAD is not a descendant of this row's base, so the row branch cannot build the
server by itself yet. The candidate swaps only the two changed objects into the
existing static archive and relinks:

1. compile `cpu_quant_dot.cpp` (default flags) and `cpu_quant_dot_avx2.cpp`
   (`-mavx2 -mf16c`) with the recorded recipe in
   `/tmp/flare-gsq-repair-build/gemma4_gguf_weights-compile.json`;
2. `ar r` both into a copy of `libvllm.a` (902 members vs 901);
3. link with `/tmp/flare-gsq-repair-build/server-link.json`.

All shared quant frontend files (`cpu_quant_dot.cpp`, `quant.h`,
`cpu_quant_blocks.h`, `cpu_quant_iq_tables.h`) were byte-identical between the
GSQ worktree HEAD and this row's base before the change.
