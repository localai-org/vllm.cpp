# vLLM, online serving

The binding comparison. vLLM runs its **production graphed config**, never
`--enforce-eager`. Every `0.25.0` row below is **SUPERSEDED and OPTIMISTIC**, on
two counts that both flattered us: the harness enforced the `0.25.0` ROLLBACK
and *raised* on the pin until 2026-08-12
([#520](https://github.com/mudler/vllm.cpp/issues/520)), and the same runs never
passed `--language-model-only`, so the denominator ran vLLM's UNFUSED
QK-norm+RoPE+gate path
([#414](https://github.com/mudler/vllm.cpp/issues/414)). Nothing is withdrawn.
The first series free of both, at the pin, graphed, and at a pinned clock is in
[the benchmark record](../../.agents/benchmark-record.md).

**THE PIN MOVED UNDER THESE ROWS, AND THEY HAVE NOT BEEN RE-MEASURED.** The
parity pin advanced to <!--pin:commit-->`a7c23ac96d`<!--/pin--> on 2026-09-22
([#3320](https://github.com/mudler/vllm.cpp/pull/3320)), from `e126687a9a`, which it reached on 2026-09-03
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)). Every row below that
says "at the pin" was measured against a pin TWO advances back, `555967922`, with
FlashInfer `0.6.15.post1`. **FlashInfer is `0.6.18` on the oracle this tree has
actually built and on every pin since, and the step is STILL NOT discharged** --
the `parity-pin` block reads `0.6.15.post1` and the `0.6.18` leg is REFUSED by
`online_gate.py`, so no row here has a `0.6.18` denominator. It is on
the executed path of these numbers on both sides — it is the NVFP4 GEMM under the
denominator and the CUTLASS source tree our own arm compiles against. **Four rows
here owe a re-measurement**: the 27B NVFP4 `nvidia` and 35B-A3B NVFP4 rows, the
Qwen3.8-27B bf16 c4 row, and the cold-to-`/health` startup row, which is also
where `nvidia-cutlass-dsl` `4.6.0` to `4.6.2` can move a published number.
Owed by [#2818](https://github.com/mudler/vllm.cpp/issues/2818). The advance
preceded the re-measurement by a developer ruling that inverts the sync cycle's
step order; **a red on the re-measurement requires reverting the pin, not
withdrawing these rows on their own.** Nothing here is withdrawn today.

| Model | Quant | vLLM oracle | Axes passing | Disposition |
|---|---|---|---:|---|
| Qwen3.6-27B | NVFP4 (`unsloth` @`890bdef7`) | 0.25.0 ROLLBACK, SUPERSEDED | **115/124** | Effective parity-or-better, two-grid totality. Revision-PINNED (the gate no longer lets `readdir` choose): @`ccdaab7e` is the same repo name re-quantized to FP8 W8A8 throughout, not NVFP4 |
| Qwen3.6-27B | NVFP4 (`nvidia` @`0893e160`, ModelOpt `modelopt_mixed`) | 0.25.0 ROLLBACK, SUPERSEDED | 0/6 | BEHIND, uniformly 0.94x on decode, flat c1-c32, but **VOID as a ratio** (#520, #414). At the pin, clocks pinned: **0.976x c1 / 0.946x c4 TPOT**. Different model from the `unsloth` row (NVFP4 MLP + FP8 W8A8 tower) |
| Qwen3.6-35B-A3B | NVFP4 `modelopt_mixed` | 0.25.0 ROLLBACK, SUPERSEDED | 2/18 | 3-rep grid 2026-08-05 @`1ea26427`: 0.93-1.03x, VOID as ratios (#520, #414). At the pin, clocks pinned: **0.995x c1 / 0.946x c4 TPOT**. ★ probe found a prod async batch-1 greedy DEGENERATION bug the mirror fixes |
| DeepSeek-V2-Lite | bf16 MLA | 0.25.0 ROLLBACK, SUPERSEDED | 4/25 | Attributed miss, row stays `ACTIVE` |
| Qwen3.5-4B | bf16 direct-load | 0.26.0.dev0, UNFUSED and vision-tower-carrying denominator ([#1345](https://github.com/mudler/vllm.cpp/issues/1345)) | **1.0283x tput, `PENDING` and OPTIMISTIC** | OPTIMISTIC on all axes (#414). TTFT/TPOT/E2E 1.085/1.017/1.029x; VRAM **+118.7 MiB** OPEN, not like-for-like: the oracle built the tower the flag elides ([data](../bench-evidence/qwen35-4b-sm120-main-20260807.md)) |
| Qwen3.8-27B | bf16 (@`1d4bf0f2`) | 0.26.0.dev0 at the pin, graphed; c4 at a pinned 2184 MHz, the 2026-08-19 re-measure SAMPLED only | **1 of 3 concurrency cells** | Token gate PASSES. c4 like-for-like: tput **0.963x**, ITL **1.008x**. c1 re-measured COMPLETE, pairing DISCARDED on clock spread; c8 vLLM denominator NOT MEASURABLE ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |

### GDN prefill kernels by GPU

| GPU | Workload and basis | vllm.cpp | vLLM | Ratio | Status |
|---|---|---:|---:|---:|---|
| RTX 5070 Ti (`sm_120`) | Qwen3.5-4B BF16 c32: conv / post-conv | 234.605→**219.506**; 122.587 ms | 145.421; 108.035 ms | 1.509x; 1.135x | Opt-in, byte-exact: K4 **6.436%**, tile **1.859x** ([conv](../../.agents/specs/sm120-qwen35-conv-channel-tile-2026-08-08.md), [tile](../../.agents/specs/sm120-qwen35-postconv-token-tile-2026-08-08.md)) |
| GB10 (`sm_121a`) | Qwen3.6-27B NVFP4, historical normalized prefill | 0.43 us/token/layer | 0.18 us/token/layer | **2.39x slower** | Directional only: unequal token clusters, older pin ([ledger](../../.agents/parity-ledger.md)) |
| GB10 (`sm_121a`) | Qwen3.6-35B NVFP4, later local kernel A/B | 321.148 us c1; 960.313 us c6 | - | `PENDING` | Register vs tiled improved 4.7%/7.3%; no paired vLLM denominator ([record](../../.agents/specs/gdn-prefill-conv-reg-2026-07-18.md)) |
| Jetson Thor (`sm_110`), AGX Orin (`sm_87`) | No matched GDN workload | - | - | `PENDING` | Runtime correctness only; no causal-conv speed trace |

### Qwen3.6-27B by concurrency

Medians of three interleaved repetitions, 1,024 in / 128 out, cache off, closed
loop. Output is token-for-token identical to vLLM at every point. The `nvidia`
ModelOpt table that follows is a DIFFERENT checkpoint on a DIFFERENT axis and
must not be compared against this one: a 7-token prompt scored on OUTPUT tokens
per second, with method, attribution and the owed 1,024 in / 128 out axis in
[the benchmark record](../../.agents/benchmark-record.md).

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

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s (canonical 2026-08-11) | 10.756 | 19.232 | 32.365 | 50.520 | 69.040 | 84.064 |
| vLLM 0.25.0 tok/s (canonical 2026-08-11) | 11.250 | 20.153 | 34.281 | 53.666 | 73.114 | 89.706 |
| Ratio, SUPERSEDED and OPTIMISTIC (main @`348c265d`, rollback oracle #520, unfused denominator #414) | 0.9561x | 0.9543x | 0.9441x | 0.9414x | 0.9443x | 0.9371x |
| **BINDING at the pin, graphed, `--language-model-only`, clocks 2184 MHz: decode TPOT** | **0.976x** | - | **0.946x** | - | VOID #577 | - |
| Same series: decode tput / prefill TTFT (c4 TTFT is a median) | 0.973x / 0.944x | - | 0.958x / 0.982x | - | VOID #577 | - |
| Levers `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1` (both DEFAULT OFF), same-binary A/B: decode / TTFT | 1.007x / 1.048x | - | 1.012x / 1.041x | - | 1.027x / - | - |
| Driver verdict | `{"gate_pass": false}`; first six-point grid here. Levers `lm_head`+qkvz ACTIVE by kernel signature | | | | | |
| OPEN, and what it retracts | vLLM reproduces to 1.0%, OURS moved +14.8% at c1; lottery REFUTED (6 loads, 1.0046x); build diff leads. "c1 did not move" withdrawn: c1 was noise-dominated ([#349](https://github.com/mudler/vllm.cpp/issues/349)) | | | | | |

| Noise band, measured BEFORE any delta | ±0.03% c1 back-to-back; 0.29-1.85% leg-to-leg with reload, drifting down on BOTH arms so it cancels in the ratio | | |
| Startup, cold to `/health` | 33.38 s vs vLLM 182.41 s = **5.46x faster** | | |
| Peak host RSS | 21.10 vs 13.09 GiB = **1.612x, BELOW FLOOR, open gap** | | |
| Peak GPU memory | PENDING: `nvidia-smi` returns N/A on this unified-memory part | | |
| Step attribution (nsys, node-level, both arms same tool) | ours 98.906 vs vLLM 81.577 ms/step, 99.2/99.3% GPU-busy; lm_head 8.6414 + fp8 tower 7.6068 + splitK 0.0532 + other 1.0279 = **17.3292 vs measured 17.3292** | | |
| Lever 1, `lm_head` | ships U8/NVFP4 (0.666 GiB), we read 2.368 GiB BF16: **+1.702 GiB/step**, 11.183 ms. Marlin efficiency is EQUAL (207.9 vs 210.0 GiB/s), only bytes differ | | |
| Lever 2, GDN fp8 in_proj | identical 6.7188 GiB/step both arms; ours 96 GEMMs at 165.9 GiB/s vs vLLM 48 merged qkvz at 204.3; `in_proj_qkv` at **129.3 vs 213.6 GiB/s** | | |
| Lever 2b, packed GDN decode (#365) | separate c1 in16/out256 harness: 0.977x -> 0.984x, INDICATIVE not binding (arms not interleaved); see the record | | | | | |
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
| **BINDING at the pin, graphed, `--language-model-only`, clocks 2184 MHz: decode TPOT** | **0.995x** | - | **0.946x** | - | NOT ESTABLISHED, 6.8% spread | - |
| Same series: decode tput / prefill TTFT (c4 TTFT is a mean) | 0.987x / 0.920x | - | 0.944x / 0.849x | - | 0.930x / NOT ESTABLISHED, 15.0% spread | - |
| Ratio, SUPERSEDED and OPTIMISTIC @`348c265d` (rollback oracle #520, unfused denominator #414) | 0.9708x | 0.9293x | 0.9719x | 0.9183x | 0.9264x | 0.9377x |
| **vllm.cpp** tok/s @`348c265d` | 65.262 | 93.012 | 140.174 | 193.477 | 251.490 | 311.897 |
| vLLM tok/s @`348c265d` | 67.223 | 100.088 | 144.226 | 210.679 | 271.479 | 332.606 |
| Ratio, superseded @`a0fa12c7` | 0.979x | 0.937x | 0.949x | 0.935x | 0.939x | 0.969x |
| Mean TPOT @`a0fa12c7` | 0.978x | 0.945x | 0.943x | 0.938x | 0.930x | 0.967x |
| Mean TTFT @`a0fa12c7` | 0.972x | **0.872x** | 0.970x | 0.965x | 0.969x | 0.968x |
| Our CoV @`a0fa12c7` | 0.39% | 0.26% | 0.59% | 0.60% | 0.37% | 0.41% |
| vLLM CoV @`a0fa12c7` | 0.62% | 0.35% | 0.81% | 0.57% | 0.50% | 0.35% |

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

The SiLU lever is NEGATIVE ([spec](../../.agents/specs/moe-silu-vectorize.md)):
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

### Qwen3.8-27B (bf16) by concurrency

| Axis | c1 | c4 | c8 |
|---|---:|---:|---:|
| Requests completed, ours / vLLM | 5,5,5 / 6,6,6 of 6 | 24,24,24 / 24,24,24 of 24 | 36,37,36 / 48,48,48 of 48 |
| Output token throughput | NOT ESTABLISHED as a ratio; both arms' absolutes are re-measured below | **0.963x** | NOT ESTABLISHED as a ratio; ours is re-measured below and vLLM has no leg |
| Total token throughput | NOT ESTABLISHED | **0.918x** | NOT ESTABLISHED |
| Status of the two withheld cells | SUPERSEDED 2026-08-19 by the re-measure rows below: the cause was our SSE keepalive frame and our arm now completes every request ([#931](https://github.com/mudler/vllm.cpp/issues/931)) | (cell stands) | SUPERSEDED: our arm is complete, and the missing half is now vLLM's, for a different reason ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |
| Median ITL, over completed only | 1.013x | **1.008x** | 1.021x |
| Median TPOT, over completed only | 1.014x | 0.980x | 0.925x |
| Median TTFT, over completed only | 0.733x | 0.881x | 1.268x |
| Median E2EL, over completed only | 1.003x | 0.974x | 0.983x |
| ours / vLLM output tok/s | 2.37 / 3.50 | 15.01 / 15.58 | 15.96 / 27.85 |
| ours / vLLM median TPOT ms | 220.6 / 223.6 | 239.0 / 234.3 | 261.1 / 241.4 |
| Cold start to first `/health` | **53 s vs 780 s = 14.7x**, medians of 3 (ours 53/53/53, vLLM 786/780/771), NOT like-for-like readiness (next two rows) | same binary | same binary |
| Why that ratio is not like-for-like | ours answers `/health` on process liveness only (`api_server.cpp:286-294`); no dummy run, no kernel warmup, decode CUDA graph captures lazily on first use | vLLM warms up and captures before it serves (`gpu_worker.py:697-708`, `api_server.py:780-785`) | 53 s is "weights loaded", 780 s is "warmed and graph-captured" |
| What our readiness signal defers | first request TTFT **91.613 s**, the same with the SSE keepalive on and off, so it is genuine first-inference cost, not the [#931](https://github.com/mudler/vllm.cpp/issues/931) defect | cell stands as measured; a like-for-like comparison has not been taken | forensics in `.agents/benchmark-record.md` |
| Host memory after warmup | **42.5 vs 110.1 GiB = 2.59x**, but vLLM's is set by `--gpu-memory-utilization 0.85` pre-reserving KV, so it is what the configured engine holds, not what the model needs | | |
| **RE-MEASURED 2026-08-19, absolutes only** ([#915](https://github.com/mudler/vllm.cpp/issues/915), [#979](https://github.com/mudler/vllm.cpp/issues/979)) | ours **4.4040 tok/s** (CV 0.039%), vLLM **4.2835 tok/s** (CV 0.033%) | (cell stands) | ours **22.6402 tok/s** (CV 0.205%); vLLM **NOT MEASURABLE**, below |
| Requests completed on the re-measure | ours 6,6,6 of 6; vLLM 6,6,6 of 6; `failed=0` every leg | (cell stands) | ours 48,48,48 of 48, `failed=0`; 162/162 over the whole series |
| Median TPOT / ITL on the re-measure | ours 218.11 / 216.56 ms; vLLM 228.36 / 226.86 ms | (cell stands) | ours 250.57 / 232.83 ms |
| **Why no c1 ratio is quoted** | `gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three pairings. The ratio is OWED, not withheld for being unflattering | (cell stands) | no vLLM leg exists to pair |
| What the clock gate actually refused | cross-arm rule PASSED: same boot, both arms 2489 MHz, median offset **0.0%**. The within-run rule FAILED on both arms | (cell stands) | ours 12.92-14.99% spread, same breach |
| Within-run SM-clock spread, 5% ceiling | ours 13.58 / 26.36 / 14.34%, vLLM 10.16 / 17.48 / 18.52%; `SwThermal` in every window, `HwSlowdown+HwThermal` in one | (cell stands) | **all six of our legs and all three vLLM legs breached it** |
| Clocks were SAMPLED, never pinned | `nvidia-smi -lgc` returns `LGC_RC=4`, permission denied, as root inside an `rc` lease. Every prior pinned figure used the retired `ssh`+`flock` path | (cell stands) | same lease, same refusal |
| **vLLM c8 denominator** | (c1 landed) | (cell stands) | **NOT MEASURABLE on this box at the recorded config**, which is the answer and not a gap. Not a claim that vLLM is defective |
| Evidence for that verdict | (c1 landed: vLLM's own c1 cold start was 426 s and all three legs ran) | (cell stands) | `/health` at 373 s, then the worker was lost during warmup; the KV reservation took **48,715 MB in one 4 s window** (58,453 -> 9,738); last value 6,261 MB; death inside one 2 s sample |
| Why no watchdog can guard it | (c1 landed) | (cell stands) | ~6-7 GB headroom at `--gpu-memory-utilization 0.85 --max-num-batched-tokens 8192`; 12,000 MB kills a healthy server and 5,000 MB is never reached in time. Every other fix is an engine knob |
| Settled: the HOST rebooted | (c1 landed) | (cell stands) | OBSERVED: `boot_id` moved `3fd9745a-...` -> `64c495a3-...` in a later leased job, so not a lost pod. DERIVED, only if it does not virtualize `/proc`: boot at or before 10:41:47.6Z, one-sided. Causation NOT traced |
| #915's own-arm debt, DISCHARGED | prior 2.37 -> 4.4040 tok/s while median TPOT moved 220.6 -> 218.11 | (cell stands) | prior 15.96 -> 22.6402 while TPOT moved 261.1 -> 250.57. Different boots, so read the SHAPE, not the percentage |
| What that shape proves | throughput moved a lot, per-token barely at all; qualitative, the boots differ: the throughput axis was dividing live tokens by a wall still holding dead requests ([#931](https://github.com/mudler/vllm.cpp/issues/931)) | (cell stands) | withholding 0.677x rather than publishing it was correct, and this is the evidence for that |
| A divergence found in the raw files | our `usage.prompt_tokens` reports 5,942 where vLLM reports 6,144 on identical prompts; 19 of 48 short at c8 | (cell stands) | `output_lens` is `[128]xN` on both arms, so TPOT and ITL stand. Total-token throughput is CORRUPTED, and output throughput is BIASED UP, next row |
| Size of that bias, which EXCEEDS our own stated precision | if the prompts were truly truncated the wall is short by ~0.22 s of 174.39 s, about **0.13%**, beside a published CV of 0.039% | (cell stands) | ~1.1-1.6 s of 271.0 s, about **0.4-0.6%**, beside a published CV of 0.205%. A lower bound: shorter context also cheapens decode |
| Are the two TTFT medians still comparable? | YES, by evidence and not by luck: our two short prompts are the two LOWEST TTFTs in all three reps, so both arms' medians fall on 1024-token requests | (cell stands) | no vLLM leg exists to compare against |
| Why only c4 counts | `output_throughput` divides tokens by a wall duration that still contains the dead request, so a cell where one arm dropped requests is withheld, not quoted | 3 paired reps, clocks 2184 MHz | token gate: 4/7 strict, 3 exact fp32 ties ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |

### Qwen3.8-27B quantized arms, both gates PENDING and no number quoted

| Arm | Oracle, and why that one | Gate | Blocked on |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` (`unsloth` @`fe1e2a23`) | llama.cpp `b10451`, the ONLY comparator: vLLM has no in-tree GGUF reader at pin `555967922` and SGLang's aliases miss `qwen3_5` ([#979](https://github.com/mudler/vllm.cpp/issues/979)). Its oracle, never its mirror | **PENDING**, no number | [#857](https://github.com/mudler/vllm.cpp/issues/857): `.agents/oracles/llama-cpp.md` records the pin `gateable = no` and #857 owes that measurement |
| `unsloth/Qwen3.8-27B-NVFP4` @`7d6f8d4d` | vLLM `555967922`, the MIRROR and primary oracle, which runs this format | **PENDING**, no number | [#1185](https://github.com/mudler/vllm.cpp/issues/1185): the pinned oracle builds, installs and imports in an `rc` lease, but RUNNING a model there is untested |
| Why no number may be quoted yet | Correctness precedes speed and neither arm loads: the "NVFP4" file is `mixed-precision` with ZERO `*.input_scale` tensors, and the Q4_K_M file's block 64 is the MTP drafter (`nextn_predict_layers = 1`) | spec'd, not implemented | [#821](https://github.com/mudler/vllm.cpp/issues/821), [spec](../../.agents/specs/qwen38-27b-quant-arms.md) |
| What a token gate alone would miss | Each arm owes a resident-bytes assertion beside its tokens: a Q4_K_M arm that silently dequantizes to bf16 passes every token gate and defeats the point of the arm | owed, not measured | [#821](https://github.com/mudler/vllm.cpp/issues/821) |

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
