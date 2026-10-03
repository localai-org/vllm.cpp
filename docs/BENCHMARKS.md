# Benchmarks

## At a glance: W5/W6 green; validated release artifacts pending

| Reference | Workload | Headline | Tokens |
|---|---|---|---|
| **Weight load (`ENG-LOAD-DIRECT-UPLOAD`, #150)** | Qwen3.6-27B bf16 (50.098 GiB), GB10 Vulkan, same binary both arms, interleaved under one GPU lock | **Load phase 1.54x warm** (19.27 -> 12.48 s), **1.61x cold** (52.62 -> 32.75 s); bytes moved **100.196 -> 81.260 GiB**. Every ON leg beat every OFF leg | byte-identical; 6/6 token-exact |
| **Structured state record (active)** | v1 scalar + relational + Git-history contracts | No benchmark. At `776c56f1`: 157 imports = 3,231,342 exact bytes; append preserved all 156 wrappers/rows. 95 tests: validator/core 44 (checker 20 + core 24), NOW 18, migration 22, cutover 11. New raw-row mutation guard. | n/a |
| **Binary release matrix (ACTIVE; required W1-W11/W13 implemented in #196)** | Eight primary CPU/CUDA/Vulkan/Metal/MLX host tuples | Adaptive x86 tiers, Vulkan 35/35 + cross-device 11/11, and metadata/mutation gates green. **PENDING:** hosted full matrix, matching hardware, tagged publish | n/a |
| **Binary release delivery topology** | #196: read-only build/verify, OIDC attest, protected publish; generated indexes and explicit handoff-authenticated assets | Fixes the zero-binary release path by attaching all eight archive/checksum/provenance triplets plus indexes. Hosted proof pending; W12 diagnostics optional | n/a |
| **Container images (ACTIVE)** | `ENG-RELEASE-CONTAINERS`: GHCR lanes `-cuda`/`-vulkan`/`-cpu` ([spec](../.agents/specs/container-images.md)) | cpu linux/amd64 image **783 MB**; SIGTERM shutdown **0.25 s, exit 0** (was 30 s then SIGKILL 137, #312). `PENDING`: cuda/vulkan sizes, arm64, GB10 in-container `/health` | n/a |
| **Developer agent entry point (implemented)** | `DOCS-AGENT-PROTOCOL-ENTRYPOINT`: public contribution guide + synchronized, mutation-gated pre-claim intake rule | Rebased documentation/protocol only; benchmark void | n/a |
| **ARCH audit: ABI is text-only** | 4 capabilities (H3 video, Laguna, Kimi-Linear, DeepSeek-V4) reachable only from `examples/`, none registry-backed. No gate asks whether a CONSUMER can reach a capability. Documentation only |
| **DSR fix: server TU profiler guards (2026-08-09)** | **No number owed:** comments only. #189 moved the server body into the shared layer with its 5 `VT_BENCH_PROFILE_CONTROL` guards, taking DSR 32 -> 37; they are `DSR-ALLOW`'d per site, baseline unchanged at 32 |
| **DSR fix: async readback capability (2026-08-08)** | **No number owed**: behavior-neutral (CPU/CUDA async-ON, discrete non-CUDA async-OFF, unchanged); moves a `kCUDA` check onto `Backend`, unblocking red CI on #127/#154/#155 |
| **`ROAD-V1-MEM` M1+M2 (2026-08-08)** | KV auto-sizing CPU brick: `--kv-cache-memory` sizes the pool from a byte budget via the group-aware `KVBytesPerBlock` divisor (ABI v16, CPU-gated). M3 profile run dgx-gated |
| **Record/checker repair 2026-08-07–08** | Gates fixed. Public: `VT_GEMMA4_EXPERT_VRAM_MB` (positive-MiB LRU cap; unset/0 unlimited), `VT_SERVER_MAX_{PROMPT_CHARS,NEW_TOKENS}` (200000/4096; 0 disables); nine Gemma4/ROCm tuners internal. No runtime/perf change. |
| **Qwen3.8 Flash Next / TensorFold gap** | `dgx:gpu0`, required GGUF and MLX-MTP artifacts | **BLOCKED_MISSING_ARTIFACTS:** no timing or ratio; committed GGUF manifest gives `BLOCKED_NO_MTP_WEIGHTS` ([detail](benchmarks/qwen38-tensorfold-gap.md)) | not run |
| **vLLM** | Qwen3.6-27B NVFP4 `unsloth` @`890bdef7`, GB10 | ahead 4.5% at c1, **tie** at c2 to c32 | identical |
| **vLLM** | Qwen3.6-27B NVFP4 `nvidia` @`0893e160` (ModelOpt `modelopt_mixed`), GB10 | **0.8289x to 0.8639x, BEHIND** at c1 to c8 (canonical 2026-08-10; confirms the prior 0.843-0.861x). Gap fully ATTRIBUTED | near-tie |
| **vLLM** | Qwen3.6-35B-A3B NVFP4 `nvidia` @`491c2f1e`, GB10 | **CANONICAL 2026-08-11 @`348c265d`: 0.918x-0.972x** over c1-c32 (first c16/c32); best c4 0.9719. Supersedes the ad-hoc grid | near-tie |
| **vLLM** | DeepSeek-V2-Lite (MLA), GB10 | 0.86x to 0.95x throughput, TTFT wins at c4/c8 | identical |
| **vLLM** | Laguna-S-2.1 NVFP4 (118B/8B MoE), GB10 | **parity+, 1.03x** (44.46 vs 43.10 tok/s, byte-exact, default config; bf16 weights now device-resident) | near-tie |
| **llama.cpp** | Qwen3.5-2B GGUF, CPU aarch64 | 20-core Arm/i8mm: prefill **1.18x ahead**, decode tie, memory parity. RPi5/A76: vllm.cpp is **0.461x prefill / 0.653x decode+E2E**, but uses **24.2% less RSS** | byte-identical on both Arm lanes |
| **MLX-LM** | Qwen3-0.6B, Apple M4 | 97.6% warm total, prefill ahead | near-tie |
| **DwarfStar** | DeepSeek-V4-Flash GGUF, GB10 | **beats ds4, 1.144x** (18.69 vs 16.33 tok/s, byte-exact, default config) | n/a, GGUF peer |
| **vLLM** | Kimi-Linear-48B-A3B, GB10 | no binding number: the published checkpoint is tiktoken-only, so it cannot drive the warm-server harness | golden 122/128, near-tie profile |
| **Muse Glimmer 30B (#268)** | no vLLM denominator (pin lacks `muse_glimmer`); SECONDARY llama.cpp, same GGUF, idle GB10 | **vLLM axis is an OPEN GAP.** vs llama.cpp ([#333](../.agents/specs/muse-glimmer.md) §14): prefill **tie 0.997x**, decode **0.232x**, RSS 1.92x | coherent, NOT token-exact |

Reading the ratios: throughput is ours/reference, latency is reference/ours, so
**1.0 or higher is a win** everywhere on this page. Which architecture each number
is measured on, and the per-architecture correctness gate behind it, is the
registry-bound list in [FEATURES.md](FEATURES.md).

## Benchmark detail index

| Benchmark ID | Disposition | Detail |
|---|---|---|
| `qwen38-tensorfold-gap` | `BLOCKED_MISSING_ARTIFACTS`; no number | [Qwen3.8 TensorFold gap](benchmarks/qwen38-tensorfold-gap.md) |

## vLLM, online serving

The binding comparison. vLLM runs its **production graphed config**, never
`--enforce-eager`, because the graphed config is the honest denominator.

| Model | Quant | vLLM pin | Axes passing | Disposition |
|---|---|---|---:|---|
| Qwen3.6-27B | NVFP4 (`unsloth` @`890bdef7`) | 0.25.0 | **115/124** | Effective parity-or-better, two-grid totality. Revision-PINNED (the gate no longer lets `readdir` choose): @`ccdaab7e` is the same repo name re-quantized to FP8 W8A8 throughout, not NVFP4 |
| Qwen3.6-27B | NVFP4 (`nvidia` @`0893e160`, ModelOpt `modelopt_mixed`) | 0.25.0 | 0/4 | **BEHIND, uniformly 0.85x** on decode throughput (was 0.72x before the FP8 tower fix); greedy continuation IDENTICAL to vLLM. A different model from the `unsloth` row (NVFP4 MLP + FP8 W8A8 GDN/attn tower) |
| Qwen3.6-35B-A3B | NVFP4 `modelopt_mixed` | 0.25.0 | 2/18 | 3-rep grid 2026-08-05 @`1ea26427`: 0.93-1.03x (c4 wins), c16 0.93x. Both c16 levers A/B'd NEG: drain event -1.9%, mirror 0.999x. ★ probe found a prod async batch-1 greedy DEGENERATION bug the mirror fixes |
| DeepSeek-V2-Lite | bf16 MLA | 0.25.0 | 4/25 | Attributed miss, row stays `ACTIVE` |
| Qwen3.5-4B | bf16 direct-load | 0.26.0.dev0 | **1.0283x tput, `PENDING`** | OPEN: TTFT/TPOT/E2E 1.085/1.017/1.029x, VRAM +118.7 MiB ([data](bench-evidence/qwen35-4b-sm120-main-20260807.md)) |

### GDN prefill kernels by GPU

| GPU | Workload and basis | vllm.cpp | vLLM | Ratio | Status |
|---|---|---:|---:|---:|---|
| RTX 5070 Ti (`sm_120`) | Qwen3.5-4B BF16 c32: conv / post-conv | 234.605→**219.506**; 122.587 ms | 145.421; 108.035 ms | 1.509x; 1.135x | Opt-in, byte-exact: K4 **6.436%**, tile **1.859x** ([conv](../.agents/specs/sm120-qwen35-conv-channel-tile-2026-08-08.md), [tile](../.agents/specs/sm120-qwen35-postconv-token-tile-2026-08-08.md)) |
| GB10 (`sm_121a`) | Qwen3.6-27B NVFP4, historical normalized prefill | 0.43 us/token/layer | 0.18 us/token/layer | **2.39x slower** | Directional only: unequal token clusters, older pin ([ledger](../.agents/parity-ledger.md)) |
| GB10 (`sm_121a`) | Qwen3.6-35B NVFP4, later local kernel A/B | 321.148 us c1; 960.313 us c6 | - | `PENDING` | Register vs tiled improved 4.7%/7.3%; no paired vLLM denominator ([record](../.agents/specs/gdn-prefill-conv-reg-2026-07-18.md)) |
| Jetson Thor (`sm_110`), AGX Orin (`sm_87`) | No matched GDN workload | - | - | `PENDING` | Runtime correctness only; no causal-conv speed trace |

### Qwen3.6-27B by concurrency

Medians of three interleaved repetitions, 1,024 in / 128 out, cache off, closed
loop. Output is token-for-token identical to vLLM at every point. The `nvidia`
ModelOpt table that follows is a DIFFERENT checkpoint on a DIFFERENT axis and
must not be compared against this one: a 7-token prompt scored on OUTPUT tokens
per second, with method, attribution and the owed 1,024 in / 128 out axis in
[the benchmark record](../.agents/benchmark-record.md).

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** TOTAL tok/s | **86.05** | 159.68 | 292.34 | 508.77 | 801.76 | 1095.01 |
| vLLM TOTAL tok/s | 82.32 | 158.03 | 290.31 | 505.46 | 789.16 | 1076.25 |
| **Ratio** | **1.045x** | 1.011x | 1.007x | 1.007x | 1.016x | 1.017x |
| Axes passing | 20/20 | 20/20 | 18/20 | 15/20 | 19/20 | 18/20 |

TOTAL tokens per second (prompt plus output), not a decode rate: c1's 86.05 is
about 10 tok/s of decode plus about 940 of prefill. The `nvidia` table below
counts OUTPUT only, hence its c1 of 10.41.
We are nominally ahead at all six, but only c1 means anything. Our run-to-run
noise band is 0.5% and c2 through c32 land between 0.7% and 1.7%, so **treat
those five as ties**, not as wins. The nine axes that fail in both grids are one
tradeoff, not nine problems: our synchronous deterministic forward loses on
low-concurrency *median* decode and TTFT, and wins the corresponding *tail* and
the same metric at higher concurrency (c8 p99 ITL 0.86x, but 1.055x at c16 and
1.078x at c32).

### Qwen3.6-27B NVFP4 `nvidia` @`0893e160` by concurrency (ModelOpt)

| Concurrency | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| **vllm.cpp** tok/s (canonical 2026-08-10) | 9.4201 | 17.2474 | 29.5132 | 46.7061 |
| vLLM 0.25.0 tok/s (canonical) | 11.3646 | 20.3858 | 34.6041 | 54.0616 |
| **Ratio POST-LEVER (BINDING, main @`348c265d`)** | **0.8384x** | **0.9637x** | **0.9545x** | **0.9670x** |
| ours tok/s post-lever | 9.366 | 19.529 | 32.870 | 51.753 |
| Ratio pre-lever (same recipe, superseded) | 0.8289x | 0.8461x | 0.8529x | 0.8639x |
| Levers landed | packed NVFP4 `lm_head` + merged GDN fp8 qkvz. Both ACTIVE by kernel signature: `cutlass_80_tensorop` ABSENT, split `nvjet_64x128x128` replaced by merged `192x48x128` | | |
| OPEN: c1 did not move | c2-c8 gained ~10 points, c1 only +0.010 though both levers execute there. The pre-lever attribution sized them AT c1, so it mis-assigned c1 | | |
| TPOT / TTFT ratio (canonical) | 1.2245 / 1.0077 | 1.1902 / 1.1111 | 1.2313 / 0.9722 | 1.2250 / 0.9992 |
| Prior ad-hoc ratio (superseded, consistent) | 0.847x | 0.861x | 0.853x | 0.843x |
| Before the FP8 tower fix | 8.76 | 17.07 | 33.01 | 62.13 |
| Noise band, measured BEFORE any delta | ±0.03% c1 back-to-back; 0.29-1.85% leg-to-leg with reload, drifting down on BOTH arms so it cancels in the ratio | | |
| c16, c32 | NOT MEASURED. Both canonical attempts void: denominator contended mid-timing once, host OOM-reboot once | | |
| Startup, cold to `/health` | 33.38 s vs vLLM 182.41 s = **5.46x faster** | | |
| Peak host RSS | 21.10 vs 13.09 GiB = **1.612x, BELOW FLOOR, open gap** | | |
| Peak GPU memory | PENDING: `nvidia-smi` returns N/A on this unified-memory part | | |
| Step attribution (nsys, node-level, both arms same tool) | ours 98.906 vs vLLM 81.577 ms/step, 99.2/99.3% GPU-busy; lm_head 8.6414 + fp8 tower 7.6068 + splitK 0.0532 + other 1.0279 = **17.3292 vs measured 17.3292** | | |
| Lever 1, `lm_head` | ships U8/NVFP4 (0.666 GiB), we read 2.368 GiB BF16: **+1.702 GiB/step**, 11.183 ms. Marlin efficiency is EQUAL (207.9 vs 210.0 GiB/s), only bytes differ | | |
| Lever 2, GDN fp8 in_proj | identical 6.7188 GiB/step both arms; ours 96 GEMMs at 165.9 GiB/s vs vLLM 48 merged qkvz at 204.3; `in_proj_qkv` at **129.3 vs 213.6 GiB/s** | | |
| OPEN: host-memory-state sensitivity | same binaries read c1 0.7604 pre-reboot vs 0.8289 post; vLLM barely moved. Protocols also differed, variables not separated | | |
| Spread, ours / vLLM | 1.000 / 1.006 | 1.009 / 1.069 | 1.001 / 1.001 | 1.005 / 1.003 |
| Method | medians of 3, warm servers, one `flock`, greedy, `ignore_eos` so both emit exactly 128 tokens, `--gpu-memory-utilization 0.55 --max-model-len 4096`, vLLM in its production graphed config | | | |
| Tokens | greedy continuation IDENTICAL between engines, captured from the same warm processes as these numbers | | | |
| Reading | still a real loss, not a tie, and still FLAT across the sweep, so batching and the scheduler are not the cause | | | |
| Roof | 20.42 GiB over about 273 GB/s: vLLM's 79 ms/token is about 95% of the bandwidth limit, ours 96 ms/token about 78% | | | |
| Peak host RSS | 21.0 GiB, down from 24.2 GiB, because the FP8 tower is no longer expanded to BF16 | | | |

#### NVFP4 `lm_head` kept packed (`PERF-27B-LMHEAD-FP4`, #213)

| Axis | Packed (`VT_LMHEAD_FP4=1`) | Dequant (`=0`) | Result |
|---|---:|---:|---|
| Peak host RSS | 19.36 GiB | 21.06 GiB | **-1.70 GiB**, but measured BEFORE `ENG-LOAD-DIRECT-UPLOAD` (#150) made `LoadCtNvfp4Raw` borrow mmap'd bytes, which moves the RSS accounting; re-measurement OWED |
| Peak host RSS, non-CUDA | | | **Arithmetic, not measured.** A backend with no fp4 GEMM keeps packed + one bf16 operand: Vulkan **-1.70 GiB** (it used to stage a host bf16 head *and* a device copy), plain CPU **+0.67 GiB**. See `docs/USAGE.md` |
| Greedy continuation | identical to the dequant leg, byte for byte | | SOLID |
| `test_qwen27_paged_engine` | 235/235 | 235/235 | unchanged |
| tok/s, leg A / leg B | 11.197 / 11.193 | 9.418 / 10.163 | **INDICATIVE ONLY** |
| Reading, throughput | packed faster in all four legs; packed legs agree to 0.04%, dequant legs disagree by 7.9% | | DIRECTION established, MAGNITUDE not |
| Owed | binding grid: 3 reps per leg, order-alternated, c1/c2/c4/c8, medians of per-rep medians, before any ratio is quoted | | PENDING |
| Method | same model and revision as the table above, same-binary A/B on GB10 | | |
| Provenance | the row's fresh reviewer, gate checkpoint `nvidia/Qwen3.6-27B-NVFP4`@`0893e1606ff3d5f97a441f405d5fc541a6bdf404` | | |

### Qwen3.6-35B-A3B by concurrency

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s | 65.6 | 93.6 | 142.9 | 197.6 | 256.2 | 323.1 |
| vLLM tok/s | 67.0 | 99.9 | 150.6 | 211.3 | 272.9 | 333.6 |
| **Ratio** | **0.979x** | 0.937x | 0.949x | 0.935x | 0.939x | 0.969x |
| Mean TPOT | 0.978x | 0.945x | 0.943x | 0.938x | 0.930x | 0.967x |
| Mean TTFT | 0.972x | **0.872x** | 0.970x | 0.965x | 0.969x | 0.968x |
| Our CoV | 0.39% | 0.26% | 0.59% | 0.60% | 0.37% | 0.41% |
| vLLM CoV | 0.62% | 0.35% | 0.81% | 0.57% | 0.50% | 0.35% |

**There is no isolated c2/c8 weakness.** Binding grid at `a0fa12c7`
(2026-08-10), 3 reps, binding-eligible 12/12; the prior c2 0.87x / c8 0.92x
"weak cells" came from a different harness and were never comparable. The
deficit is a flat mid-band (c2 to c16 within 0.935x-0.949x, CoV under 0.81% on
both engines) and is entirely MARGINAL per-token work, since our FIXED per-step
cost already beats vLLM (9.02 ms against 9.43 ms). Two glue levers land against
it, both default ON: the fused shared-expert gate_up sink off the MoE-marlin
route (**+1.31% c8 / +1.38% c4**, `VT_MARLIN_DENSE_PAIR`) and the shared
`down_proj` emitted as bf16 rather than f32 (**+2.05% c8 / +0.79% c4**,
bit-identical, `VT_SHARED_DOWN_BF16`).

The SiLU lever is NEGATIVE ([spec](../.agents/specs/moe-silu-vectorize.md)):
the 9.2x per-launch gap that motivated it was a MEAN over a bimodal kernel
(min 1.34 us, max 979 us) and our decode-phase SiLU is already faster than
vLLM's, so the remaining band is unattributed. The
marlin block-size lever is REFUTED (block 8 at c8 is 1.16% SLOWER against a
+0.29% control), and a per-launch marlin gap is WITHDRAWN as cross-tool
uncertainty. Memory passes decisively: peak PSS **3.81x**, peak GPU **1.40x**.
Detail in `.agents/benchmark-record.md`.

**Device-resident sampled tokens on integrated (`VT_ASYNC_DEVICE_MIRROR`) A/B'd
2026-08-06, speed-NEUTRAL** (same-binary): c16 OFF median 2305.8 vs ON 2303.3
(0.999x), c32 2928.9 vs 2919.1 (0.997x), bands overlap. It is a drain MOVE (relocate
the drain past the host prep, not remove it), so it overlaps only the small host
prep; the drain still serializes GPU input staging, so c16 does not recover. The
drain REMOVAL, `VT_ASYNC_EXECUTOR` Option A (`row/SERVE-ASYNC-OPTION-A`, default-OFF,
input H2D staged out of capture, the faithful vLLM structure), is GREEN + RED-reproducing
but binding-A/B speed-NEUTRAL (c8/c16/c32 +0.0-0.1%), so the c16 gap is NOT the async
input path. Detail in `.agents/benchmark-record.md`.

**But the mirror FIXES a shipping correctness bug, so it is now DEFAULT ON
(ROW-SERVE-ASYNC-LLM, 2026-08-06).** Baseline async (AsyncLLM depth-2) batch-1 greedy
decode degenerated into nondeterministic token-0 garbage; the mirror is deterministic
and coherent. The missing gate now exists, `test_qwen36_async_serving` (depth-2
AsyncLLM, batch-1 + concurrency, token-exact vs the SACRED oracle): RED on `=0`, GREEN
on the default. c16 re-checked on the default: 2312.9/2303.9/2294.4 (median
**2303.9**), c32 2942.7 (no regression). Root cause + file:line in the benchmark
record. The same P0 hit classic dense `Qwen3ForCausalLM` (quant-independent), fixed by `ROW-SERVE-ASYNC-DENSE-MIRROR` (see the MXFP4 Qwen3-8B row). The intake-drain lever likewise measured NEUTRAL (2026-08-06, `VT_INTAKE_DRAIN` A/B 3+3 reps): admitting during the forward wait collapses intake -91% but shifts it into queued, arrival-to-scheduled invariant, so the recorded INTAKE term is an attribution boundary over a GPU-bound prefill wait, not reducible; lever reverted, byte-exact `VT_LOOP_TRACE` probe kept.

### DeepSeek-V2-Lite (MLA)

Medians of 3 reps, 1,024 in / 128 out. The vLLM arm runs `--moe-backend triton`,
which is its best **stable** graphed config on GB10: the auto-selected FlashInfer
CUTLASS MoE backend hard-rebooted the box five times. The substitution does not
flatter us, we lose against it.

| Concurrency | Output tok/s ours / vLLM | Ratio | Median TTFT | Median TPOT |
|---:|---|---:|---:|---:|
| 1 | 33.18 / 38.17 | 0.87x | 0.95x | 0.90x |
| 2 | 52.63 / 55.27 | 0.95x | 0.88x | **1.03x** |
| 4 | 70.36 / 81.51 | 0.86x | **1.04x** | 0.86x |
| 8 | 102.37 / 116.35 | 0.88x | **1.12x** | 0.86x |

Peak memory is the decisive win: **31.38 GiB against vLLM's 68.5 GiB**, with the
caveat that vLLM pre-reserves a fixed fraction up front while we allocate the KV
blocks the workload needs. Real difference in operating footprint, not evidence
of a lower per-token KV cost.

### Laguna-S-2.1 (NVFP4)

`poolside/Laguna-S-2.1-NVFP4`, 118B total / ~8B active MoE, 48 layers,
256 experts, ~67 GiB. Both arms NVFP4, single request, batch 1, GB10. The local
checkpoint directory is named `laguna-xs-nvfp4`, which is where the "Laguna-XS"
label in earlier revisions of this page came from; the measured model is S-2.1.

| Arm | Decode tok/s | Ratio |
|---|---:|---:|
| vLLM NVFP4, graphed | 43.10 | 1.00x |
| **vllm.cpp NVFP4**, resident decode + CUDA graph | **44.46** | **1.03x** |

The gap is CLOSED (2026-08-04, same-tool nsys graph-node tracing on both
engines). Root cause: the bf16 M=1 projection GEMVs (o_proj, qkv, router,
dense, lm_head) read their weights from GB10 UNIFIED/ATS host memory (a
`w.View()` device retag, no `cudaMalloc` staging), which the GPU reads slower
than true device memory. `VT_LAGUNA_RESIDENT_BF16W` (default-ON parity enabler)
stages every projection device-resident (one H2D copy at load); byte-exact
(ids bit-identical to the `=0` retag arm). Binding new-default decode is 44.46
tok/s (median-of-3), 1.03x vs vLLM's graphed 43.10.

Per-call the residency recovers o_proj 194 to 131 us (about 168 to 249 GB/s),
qkv 245 to 225 us, and lm_head 2410 to 1620 us/call. This supersedes the earlier
invocation / bf16-output-`cublasGemmEx` framing (measured a wash): the
invocation was never the cause, the weight's memory RESIDENCY was. Full
forensics in `.agents/benchmark-record.md`.

Memory: the roughly 2.6 GB of bf16 device copies sit in the 119 GB GB10 unified
pool; peak host RSS is unchanged at about 40.7 GiB (device/unified allocations
are not counted in `ru_maxrss`), so there is no memory regression.

## Memory

Qwen3.6-27B NVFP4, GB10, whole serving window.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.

## llama.cpp, CPU

| RPi5/A76 arm (R4-R6, `GATING`) | Result |
|---|---|
| Scope | 4-core A76, DotProd, no i8mm; 20-core binding arm does NOT transfer. buildx/QEMU-built, Pi-executed unthrottled; hashes pinned in the [campaign spec](../.agents/specs/rpi5-cortex-a76-cpu-optimization.md) |
| Assembly vs compiler SDOT (one GCC 13.3 binary) | AAPCS64 leaf wall +3.66% M1/T1, +5.08% M128/T1, +3.69% M128/T4, with 9.74-10.24% fewer instructions; M1/T4 is the -2.43% residual ([assembly evidence](bench-evidence/rpi5-a76-q8-dot-20260806.md)) |
| 64-token Qwen model gate | Byte-identical across x86, portable, SDOT and assembly arms; asm vs SDOT median TTFT -1.55%, TPOT neutral, E2E -0.13%; vs portable TTFT -33.40%, E2E -2.67%. Cortex-A76+DotProd selects assembly by default |
| Same-file llama.cpp floor (pp17/tg64) | **NOT MET on speed**: prefill 12.81 vs 27.77 tok/s (0.461x), decode 2.55 vs 3.91 (0.653x), E2E 26,018.39 vs 16,998.49 ms ([competitor evidence](bench-evidence/rpi5-a76-llamacpp-20260806.md)) |
| Peak RSS | **2.841 vs 3.747 GiB, 24.2% less**; 3 clean unthrottled reps; same-text 64-token greedy output byte-identical after trailing-newline normalization |
| #293 C1 | [T4 decode 2.149x; floor unchanged](bench-evidence/rpi5-a76-output-wait-c1-main-20260811.md) |

Same GGUF file both arms, `dgx.casa` GB10 aarch64 (20 cores), idle, 3 reps,
llama.cpp `237ad9b96` built fresh on the same host.

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| Prefill | **223.8 tok/s** | 177.3 | **1.18x** | **PASS** |
| Decode | 24.7 tok/s | 25.4 | 0.97x | tie |
| Peak memory | 2.83 GiB | 2.80 GiB | 1.01x | **PARITY** |

Decode lands inside llama.cpp's own run-to-run spread, and the memory difference
is 30 MiB on a 2.8 GiB working set. Prefill is the only axis with a real gap and
it goes our way. Output tokens are **byte-identical** to llama.cpp's greedy
decode and to our own CPU reference path. Single-stream only: we have not
measured concurrent serving against llama.cpp's server.

## MLX-LM, Apple M4

Qwen3-0.6B, warm, batch 1, 6 interleaved runs.

| Axis | vllm.cpp | MLX-LM | Ratio |
|---|---:|---:|---:|
| Prefill TTFT | **524.5 ms** | 532.6 ms | **1.015x** |
| Decode | 27.23 tok/s | 27.85 | 0.978x |
| Warm total | 24.37 tok/s | 24.96 | 0.976x |

The 2.4% is a real gap, not noise: our spread was 0.12% and MLX-LM's 0.34%. All
of it sits in decode, 0.81 ms per token. Indicative rather than binding: two
models, 18 of 75 ops native, and the 97.6% needs the optional MLX GEMM provider
shape-gated to prefill (95.9% on the default build).

## DwarfStar, GGUF

DeepSeek-V4-Flash cannot run on vLLM on a single GB10 at all: every
vLLM-loadable checkpoint is 156 GB or larger against a 119 GiB unified pool, so
the only quant that fits is extreme-low-bit GGUF, which vLLM cannot load here.
GGUF was forced by the hardware. A policy-correct vLLM comparison needs 2x GB10
Sparks with TP2 and is owed.

| Engine | Quant | Decode tok/s | Ratio |
|---|---|---:|---:|
| DwarfStar (`ds4`) | IQ2_XXS mixed | 16.33 | 1.00x |
| **vllm.cpp** (default) | same GGUF | **16.28** | **0.997x, parity** |
| **vllm.cpp** (`VT_V4_RESIDENT_W` default-ON) | same GGUF | **18.69** | **1.144x, byte-exact** |

The default arm is parity, measured same-session clean (2026-08-04, single-load
steady both arms); the earlier 15.87/96% and 17.13 figures are superseded.

Weight residency is the beat-path (2026-08-05, `VT_V4_RESIDENT_W`, default-ON). Env var allowlisted (env-doc gate green).
The dense Q8_0 MLA/shared-expert/lm_head projection tower is read from the GGUF
mmap over ATS/unified memory, which the GB10 GPU reads about 20% slower per-GEMV
than `cudaMalloc`'d device memory. Staging that ~6 GiB tower device-resident once
at load (same bytes, same kernel, same invocation) lifts decode 16.23 to 18.69
(median-of-3, drop_caches), generated ids byte-identical. It is the same lever
that took Laguna to vLLM parity+ (`VT_LAGUNA_RESIDENT_BF16W`).

PEAK RESIDENT is flat at 86.68 GiB in both arms: the staged copy is additive but
the clean mmap file pages evict under the unified pool, so net usage does not
grow. An nsys A/B (identical instance counts) confirms the mechanism is
residency-bound, not latency-bound: per-launch time drops about 20% on every
dense Q8_0 kernel (`QuantDotGemmQ8_0Kernel` 184 to 147 µs, `Q8_0GroupDiagKernel`
212 to 166 µs, `Q8_0PairKernel` 74 to 60 µs). This corrects the earlier
"per-launch GEMV parity / Q8_0 weight-stream floor" framing: our GEMV was
ATS-bound, not at ds4 parity.

Phase-2 staged the routed-expert slabs too (the ~70 GiB IQ2/Q2_K bulk,
`VT_V4_RESIDENT_EXPERTS`, first-touch `cudaMalloc` plus immediate
`madvise(MADV_DONTNEED)` per slab so the transient stays ~flat). It was **measured
NEGATIVE (2026-08-05) and is HELD default-OFF** as a characterized artifact.
Same-binary median-of-3, warm-cancelled steady, drop_caches: OFF (Phase-1) **19.43
tok/s** vs ON **18.76 tok/s** (0.966x, ~3.4% slower), generated ids byte-identical
(md5 equal across all 6 runs), PEAK RESIDENT flat at 86.6 GiB. The move itself
works: host RSS drops 86 to 14 GiB as the mmap pages are reclaimed.

The regression matches the roofline. Unlike the dense Q8_0 tower (63% of DRAM
peak, bandwidth-bound), the grouped-MoE `QuantDotGemmGrouped<IQ2_XXS>`/`<Q2_K>`
kernels run at only ~19-24% of peak (dequant/latency-bound), so weight residency,
a bandwidth lever, cannot help them. It also adds a large one-time graph-capture
cost, and pinning the 70 GiB as `cudaMalloc` (vs evictable mmap file cache) cuts
the unified-pool reclaimable headroom from ~103 to ~30 GiB avail. The lever stays
in the tree, default-OFF, for reproducibility; detail in the benchmark record.

## Speculative decoding

| Speculator | Model | Result | Status |
|---|---|---|---|
| MTP | Qwen3.6-27B NVFP4 | token-identical to vLLM MTP, **~4% faster at c1**; on-par at c2-c8 | `DONE` |
| DFlash | Qwen3.6-27B NVFP4 | **2.9x over spec-off** (10.16 → 29.32 tok/s), at/above vLLM DFlash-on (**1.003x**, non-overlapping bands) | `DONE` |
| n-gram | Qwen3.6-27B NVFP4 | draft-free (`SPEC-NGRAM`); 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted (correctness only, no speed row yet) | `DONE` |
| DSpark | Qwen3.6-35B-A3B NVFP4 + RedHatAI dspark draft, k=8 | Token-identical to spec-off. Warm c1 **1.15x** at **20.8% acceptance vs upstream 7.7%**. nsys: verify runs EAGER (0 graph launches vs 1/step) and the MoE costs 1.7x more GPU per token at T=9 | `ACTIVE` |
| Breadth (EAGLE1/3, suffix, ngram-gpu, dynamic-k, ...) | n/a | enumerated from vLLM source + `INVENTORIED` 2026-08-06 (`.agents/specs/spec-decode-inventory.md`), unmeasured | `INVENTORIED` |

## How we measure


Record dates are CI-guarded: structured state event timestamps and ordered
indexes are validated by `check-state-record`, so scoreboard stamps remain
traceable. The
review protocol behind these numbers is guarded the same way: the reviewer and
implementer sub-agent prompts are tracked artifacts checked by
`check-protocol-consistency` (orchestration harness step 5/5), and
`check-gate-commands` pins the 25 record rows that name a gate command able to
FAIL. That pin is exact, not shrink-only: gaining a gate command reddens it too,
so the set is never re-pinned silently in either direction. Since 2026-08-07,
a PR verified green merges in that same session (disposition rule).

**Hardware.** NVIDIA GB10 / DGX Spark (sm_121a) for CUDA, `dgx.casa` aarch64 for
CPU, Apple M4 for Metal. GB10's 119 GiB pool is unified, so host and device
memory compete; end-to-end wall-clock on a cold page cache is unusable there,
and steady-state per-step timing or `nsys` GPU-busy is the anchor. The
2026-08-06 #77-slip tree-revert changed no benchmark content or number.

**Oracle pin.** vLLM 0.26.0.dev0 (`55596792`) plus transformers 5.14.1, built from
source for sm_121a. Speed figures labelled 0.25.0 are the last binding run; the
engine is unchanged by the pin advance and a 0.26 re-benchmark is pending.
Correctness re-validated bit-identical across the advance, zero golden drift.

**Protocol.** Greedy, closed loop, three interleaved repetitions per point, one
`flock` across the whole series, same-binary A/B for every lever, cold legs
discarded. Workload equivalence between arms is audited, not assumed: batch cap,
token budget, context, corpus bytes, KV and SSM dtypes, kernel family, and
graphed decode all match, and the audit is
[recorded](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md). The
2026-08-04/08 governance checkpoints (record/CI substrate, anchor backfill,
operator/helper W0-W5, upstream/device inventory, onboarding probe, the
review-hardened `agent-start.py` entrypoint, and CI-bound review-until-PASS
policy) touched no engine code and moved no number: **NOT APPLICABLE**, nothing
to reproduce.

The PR #28 sanitizer repair is also NOT APPLICABLE to performance: both full
333-test CPU detector lanes pass after merging upstream `main`, while the
ASan+UBSan build footprint falls from 93 GiB to 5.7 GiB and TSan occupies
1.9 GiB. Reproduce with the sanitizer
CTest commands preserved in the structured state evidence. The 2026-08-06 live-state audit and the 2026-08-08 state-record migration plus range-gate and stale-reference repair are likewise NOT APPLICABLE: bookkeeping, record checkers, and prose. No engine code, kernel, or number on this page changed.

**Vocabulary.** *Token-exact* means our output ids equal the reference's, byte
for byte. *Near-tie* means the reference's own greedy decode is not deterministic
at this precision, so the gate is distributional: our output must fall inside the
set the reference produces across K runs. *Tie* means the difference is inside
the measured run-to-run noise band, which is 0.5% on GB10 and 0.12% to 0.34% on
M4. We never publish a partial, contended, or stale-denominator number as
binding, and when a denominator turns out to be wrong we correct every ratio
built on it rather than keeping the flattering one.

**CPU elementwise GEMM, wide x86 tiers (2026-08-07).** INDICATIVE ONLY, not binding: the x86 dev box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. The AVX-512 tier measures 1.56x to 2.83x over SSE2 on the elementwise micro-kernels, byte-identically. A binding number needs a qualified x86 host, which the project does not have.

**CPU elementwise GEMM, transpose-free `[K,N]` path (2026-08-07).** On dgx aarch64 the `[K,N]` path beats `[N,K]` by 1.16x to 1.30x, byte-identically. The x86 arm is INDICATIVE ONLY, not binding: that box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. `VT_CPU_MATMUL_STEAL` ships default OFF and is NOT measured; it must justify itself by measurement and may measure neutral.

## Open gaps

| Track | Status | Next gate |
|---|---|---|
| Surface coverage (`ARCH-ONE-SURFACE`) | **CORRECTNESS COMPLETE:** #139 restores DSR 32 (`kcuda=0`) via registry/name resolution; ABI-v14 selection unchanged; no speed claim | Selector 2/2·11 plus execution-bound CMake/File-API/CTest + CI/preflight + manifest-integrity guard 52/52; CPU platform/loader/C-ABI tests green; CUDA A/B remains residual |
| 35B prefill TTFT | 0.93x to 0.98x at every concurrency (2026-08-05) | Attribute the residual, then close |
| 35B low-batch MoE decode | CLOSED at low batch (c1 0.975x, c4 wins); c16 0.93x. `VT_ASYNC_DEVICE_MIRROR` **default ON for correctness**. `VT_ASYNC_EXECUTOR` Option A (H2D out of capture) A/B'd speed-NEUTRAL | c16 lever is prefill glue (task #61), not the decode drain. `test_qwen36_async_serving` GREEN |
| CPU keep-quant MoE decode | **No number owed**: correctness-only P0. The grouped keep-quant GEMM read activations as f32 whatever their dtype, so CPU MoE decode emitted token-0 garbage from `b4f5610a` (2026-07-31) | Speed unmeasured and unclaimed; `test_ops_quant_dot` GREEN (150224 assertions) |
| DeepSeek-V2-Lite MLA | Attributed miss, `ACTIVE` | Throughput at every concurrency |
| Qwen3.5 upstream throughput levers (roadmap C10) | NOT MEASURED. vLLM's 2026-08-06 25K tok/s/GPU is a GB200/NVLink72 disaggregated cluster result, not comparable to one GB10, and is NOT adopted as our bar | Advance the parity pin past `555967922` so the referenced PRs exist, re-capture goldens at zero drift, then port the GDN prefill kernel |
| DeepSeek-V4-Flash | **Parity with ds4 (0.997x)** | Optional beat-path: f16 tensor-core DSA/router (near-tie class) |
| DeepSeek-V4-Flash vs vLLM | Infeasible on one Spark | 2x GB10 with TP2 over the NCCL seam |
| Tensor parallelism (task #287 spike, 2026-08-08) | **No number owed** (`benchmark_binding=false`): records-only scope at pin `555967922` ([spec](../.agents/specs/tensor-parallelism-spike.md)). TP-W1 landed (group table, 6/6); TP-W2..W7 gate on token-exactness, not speed | Perf gate is TP-W6: at or above vLLM TP=2 on every axis, same 2-GPU box (PENDING-HW) |
| Multimodal image, audio, video | Correctness gated, speed unmeasured | Per-modality speed grids |
| `/v1/videos` OpenAI + ONE-SURFACE ROW 2 | **No number owed:** ABI-v12 device selection is backend-dispatch plumbing; generation math and speed paths are unchanged | DSR 34→32; baseline/allowlist unchanged; 25/25 checker mutations; CPU fold 6/137, including one-queue/device-provenance mutations |
| Qwen3-dense decode CUDA-graph | Token-exact pass, ~4.3% e2e directional | Steady-state per-step tok/s |
| Kimi-Linear-48B-A3B (KDA+MLA+MoE) | **RUNNER FOLD LANDS (ROW 7, §21, #122): engine==CLI 128/128 byte-identical; vs golden 122/128 (near-tie profile); FA2 MLA default-ON; SACRED green.** Server 19.0 tok/s wall; CLI 18.93 reproduced | vLLM ~21 (#111 floor; in-session re-measure ABORTED by GB10 reboot at util 0.82, §21): **~0.90×**, >= vLLM NOT met; residual = KDA host islands + grouped MoE + decode graph |
| vLLM 0.26 re-benchmark | Pending | Re-run the binding grids on the advanced pin |
| MiniMax-H3 FP4 speed (W-FP4a) | **Measured GB10 (`row/H3-FP4-GPU-E2E`).** Marlin W4A16 byte-exact vs bf16; fp4 a memory win, 0.8x bf16/forward. Real-ckpt fp4-resident e2e RUNS (mp4/wav) | fp4 speed CLOSED. bf16-vs-quant A/B: ENCODER half MEASURED (§8.15), DiT half NOT (no bf16 render exists). Detail: benchmark-record + spec §8 |
| MiniMax-H3 render coherence (`row/H3-RENDER-CLOSE` #77) | **CLOSED: a COHERENT scene on GB10.** #70/#74 white was wrong-PARTITION usage (t2va on the ref2va ckpt); t2va on the FL2VA GGUF renders a prompt-matched orange cat (adj-cos 0.95 vs 0.06, no patch-grid) | Verified first: t2va inputs byte-exact vs upstream; CUDA device==host at seq 1920. Follow-up `H3-TASK-PARTITION-GUARD`: the task/partition mismatch now RAISES 1:1 with `_resolve_task` (spec §8.6-8.7) |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`) | **fl2va COHERENT; ref2va assembly bug FIXED+gated.** vision→cond scatter gated; ref2va block-dim double-division fixed + RED-first gated (128 vs 512) + a permanent ref2va DiT-forward rung (§8.10) | grid RE-ATTRIBUTED: with the fix ref2va grids in fp4 AND bf16, and t2va with no refs on the ref2va NVFP4 also grids while FL2VA-GGUF renders, so it is the **NVFP4 checkpoint/loader**, NOT assembly/fp4 (§8.10) |
| MiniMax-H3 Thor render speed (sm_110, no FA2) | **34.6 s/step** at 864x480/124f/50 steps on Q4_K_M, **16.6x** off 574.5 (render ~28 min, was ~8 h). Landed: warp-per-query, chunked warp reduce-scatter (1.76x), bf16 `mma.sync` (9.82x) | Shared-memory K/V tiling (23% SLOWER) and register Q-blocking (-0.8%) both measured and REVERTED: memory traffic is not the bound (one head's K+V is 3.9 MB against 32 MB of L2) |
| MiniMax-H3 render duration (audio halving) | **CLOSED**: a 124-frame render silently muxed as **61**. The decoded audio ran half the video's duration and the muxer passes `-shortest`. Every structural check passed: shapes were self-consistent, just halved | Gated on the duration invariant (latent steps / 40 Hz equals the video duration), not on shape self-consistency, which a halved pipeline satisfies |
| MiniMax-H3 quantization floor | **Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude outliers 3-bit cannot hold; a controlled A/B (same prompt, seed, code) turned a murky lattice-covered silhouette into a photoreal close-up | Per-tensor mixed precision, if a smaller footprint is ever owed |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`, `row/H3-NVFP4-LOADER-DIFF`, `row/H3-NVFP4-STREAM-DIFF`) | **fl2va COHERENT; ref2va NVFP4 nibble loader bug FIXED (byte-verified); grid residual DIAGNOSED §8.12** | Activation diff + fingerprints (PR #95): NO load-path defect; all weights/islands/RoPE quant-noise-close to the coherent GGUF; grid = community-NVFP4 quant fidelity, not a loader fix. See benchmark-record |
| MiniMax-H3 encoder quantization (`H3-ENC-BF16-COND-DIFF`) | **Measured Thor (`d1085374`).** Q4_K_M vs bf16 encoder, same 233-token prompt, same forward: rel RMS **0.0340** (0.0685 excl. sink), per-token cosine mean **0.99745** / min 0.909, median rotation **3.5°** | NOT a scale change (best rescale 0.0340->0.0328). Same energy as a ONE-WORD prompt edit but DIFFUSE: 232/233 tokens rotate vs 172/233 untouched. Render A/B owed. Detail: benchmark-record |
| MXFP4 Qwen3-8B (W4A16 Marlin) | **`KERNEL-MARLIN-DENSE-EXEC` x3 (dense-ON default): c1 1.020, c2/c4/c8 0.962/0.966/0.969, GPU mem 2.63x less** (beats #51 1.005/0.925/0.939/0.953 EVERY axis); #44 3/3, 32B-NVFP4A16 6/6; -Werror test-guard fixes x2 | **VT_MARLIN_DENSE default-ON** (+951us). `FLASH-PTXAS` #82: cuModule A/B ties our+vLLM PTX across ptxas 13.0/13.2/driver-JIT (~144us); +10us is engine CONTEXT not codegen, no ptxas lever/flip (retires #75) |
| Vulkan vs llama.cpp Vulkan (`BENCH-VK-LLAMA`) | **Both arms measured, same weights.** 0.6B @128-in/32-out: llama.cpp Vulkan **11,956** pp / **174.8** tg; ours **575** pp / **66.6** tg | Decode **8.59 -> 91.7 t/s** (**10.7x**), 6/6 exact; **2.62x** off llama.cpp at matched shape. CUDA arm unblocked. 27B: fallbacks **11->5**. paged_attn batching REFUTED ([plan](../.agents/specs/bench-27b-five-way.md)) |
| Vulkan load memory (`BACKEND-VULKAN-LOADMEM`) | **The load held the model TWICE.** 27B bf16, GB10, `VT_ADOPT_DEVICE_BYTES` A/B: **VmRSS 100.759 -> 53.413 GiB**, MemAvailable floor 13.85 -> 47.3 of 119.6. Device bytes identical. [Detail](../.agents/benchmark-record.md) | Load-phase peak (host build), and the page cache that tracks copied bytes 1:1 |
| Memory footprint vs declared workload (`ROAD-V1-MEM`, #83) | **Never measured, and not measurable today**: there is no auto-sizing to compare against, because the KV pool is a hand-typed `--num-blocks`, so "what the run actually needed" has no number | Once M1's `MemoryBudget` lands: predicted-vs-actual bytes per allocation class, then peak footprint ours-auto vs vLLM at its 0.9 default on the same model and config |
| Startup latency (cold to first `/health`) | **36.51 s vs vLLM 0.25.0's 221.51 s = 6.07x** (medians of 3, 27B-NVFP4, GB10). PROVISIONAL: 3 of 6 legs contended, repeat killed by a host reboot. [Detail](../.agents/benchmark-record.md) | Uncontended 3-rep re-run on a quiet box |
| Speculation depth (`ROAD-V1-D3-SPEC-K`, #81) | **Never measured, MTP is k=1** (our port covers vLLM's k=1 branch only), so no acceptance-vs-depth curve exists | k=2..4 three-way greedy gate, then the c1/c>1 A/B + the per-workload (prose vs code) acceptance-vs-depth curve any dynamic or adaptive depth policy needs |
| Vulkan vs llama.cpp Vulkan (`BENCH-VK-LLAMA`) | 25 NATIVE (+8 GDN). **27B prefill 21.5x**; decode **4.36 vs 4.35, MET** (7 clean legs). Smart barriers skip 19.8%/tok, GPU -1.09 ms; e2e 8/12, unresolved. OFF. [source](../benchmarks/demo/vulkan_27b_llamacpp.json) | `VK-C` coopmat A/B on Thor (`VT_VULKAN_COOPMAT=0` A/Bs it): **11.1x-32.9x** vs our UNTILED scalar kernel, not vs a competent GEMM. `VK-E`: llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96` on dgx, same GGUF, three columns |
| ROCm (`BACKEND-GATE-ROCM-VLLM` / `-SGLANG`) | **NOT APPLICABLE: no number measured, claimed or owed.** W0 ctest-green on 4 gfx archs (#41); gfx1201 hipBLAS + Gemma-4 MoE (#140, contributor) ran M0/M1 on 2× R9700, our side CPU-link-verified only. No AMD HW here | The approach-(b) fix (PENDING community) unblocks the first APU model run (M2); the gate becomes a same-box vLLM-ROCm oracle once a model runs ([#41](https://github.com/mudler/vllm.cpp/issues/41)); floor: vLLM |
| Tenstorrent Blackhole (`BACKEND-TENSTORRENT`) | **NOT APPLICABLE (speed).** Correctness: OPT-125m STRICT 6/6 e2e on real hardware. Qwen3-0.6B has a device-specific golden and short 4-token warm smoke (~0.28 tok/s), not a completed speed run | Full 16x16 Qwen3 gate, then device-resident tensors + `ttnn::sdpa_decode` before any performance comparison. [Spec](../.agents/specs/tenstorrent-backend.md) |
| Prompt logprobs (`SAMPLE-PROMPT-LOGPROBS`, #223) | **NO number measured, claimed or owed.** Correctness-only, CPU. Upstream ships this path explicitly unoptimized (`gpu_model_runner.py:5622-5623`); a step where no request asks is unchanged | Floor if one is ever wanted: vLLM's own `prompt_logprobs=k`, same model and prompt |
| `logprobs_mode` (`SAMPLE-LOGPROB-TOKEN-IDS`, #238) | **NO number measured, claimed or owed.** Correctness-only, CPU. One [n, vocab] device->host copy per step when a processed mode is engaged, nothing when not | Nothing to close: observation modes, not a path vLLM optimizes either |
| SGLang floor arms | Never ran | Both arms of the SGLang comparison |
| Embeddings on the ONE surface (ROW 6, `LlamaModel` + `vllm_embed` + `/v1/embeddings`) | **NO number measured, claimed or owed.** Correctness-gated only, CPU: the 2026-08-08 fold (engine path == direct registry path, f64 LAST+normalize reference on the committed fixture) is plumbing, no speed claim | A REAL embedding checkpoint (e5-mistral class) + a same-box `vllm.LLM(task="embed")` oracle; only then does an embed-throughput bar exist |
| Parakeet/FastConformer ASR (P1-P4 + ONE-SURFACE fold ROW 1) | **NO number measured, claimed or owed.** Correctness-gated only, CPU f32; the 2026-08-07 surface fold (`vllm_transcribe`, `/v1/audio/transcriptions`) is transcript-byte-identical plumbing, no speed claim. | Floor is `parakeet.cpp`, same clip and box; needs a CUDA provider and a pretrained checkpoint |
| Async serving correctness (#323) | **FIXED**: the decode graph replayed stale HOST token ids, degenerating concurrent requests past slot 0 (classic-dense, graph on = default). Graph declines while the mirror is live; async 7/7, SACRED 184/184 | Graph to read ids at REPLAY |
| Ampere consumer (`sm_86`, RTX 3090 class) | **No number owed; no such board here.** 2026-08-06 build-verify: 7/7 FA2 TUs 0-warn, real `sm_86` SASS. [Detail](../.agents/benchmark-record.md) | External RTX 3090 report. Floor is llama.cpp on that card (GGUF, not our Blackwell-only NVFP4 grid) |
| Pre-Ampere breadth (Turing `sm_75` / Volta `sm_70` / Pascal) | **No number owed; nothing runs on these arches.** 2026-08-06 `sm_75`: 20/20 TUs PASS (0 err/warn), WMMA bodies + all 3 selectors arch-gated; GB10 SASS byte-identical. [Detail](../.agents/benchmark-record.md) | Full-library LINK at `sm_75` + `cuobjdump` SASS, then a build-supported row. The fp16 `fattn` port is speed-only now; its floor when a card exists is llama.cpp on that card |

## Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Same GGUF both arms: ours `-DVLLM_CPP_VULKAN=ON`, llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96` via `llama-bench`; clean legs only, one `flock $HOME/gpu.lock`. GEMV sweep: `benchmarks/vulkan_gemv_ab.cpp` |

Build flags, environment variables, and the full gate list are in
[BUILD.md](BUILD.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
