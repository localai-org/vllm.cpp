ID: ISSUE-GH-1334
Title: MiniMax-Music3's vocoder is **53.6 s of a 161.6 s run (33.2 %)** after [#1238](https://github.com/mudler/vllm.cpp/issues/1238), against 12.0 % before it, and #1238's depth A/B measured it at 53.6 s on BOTH legs — it is a FIXED term that grows as a share of every other improvement. **Both arms are slow for one cause.** `vt::cpu::Conv1dKernel` (`src/vt/cpu/cpu_conv1d_general.cpp::Conv1dKernel`) computes each output cell with ONE f64 accumulator swept over `(ic ascending, k ascending)`, which for a Music3 residual unit is `384 * 7 = 2688` STRICTLY DEPENDENT f64 additions per output element: no instruction-level parallelism, no vectorisation at any width, and a measured 1.7-2.0 GMAC/s per core = ~2.8-3.0 cycles per multiply-accumulate on a 5.0 GHz Zen 5 whose `fadd` latency is 3. The CUDA provider loses by a DIFFERENT mechanism with the same cause (§15.9: 3.552 s device vs 2.983 s host at latent length 20; §13.10's per-stage ratios flat at 0.37-0.40x across a 150x span of work): one accumulator per cell means it is not latency-bound but f64-RATE bound, and Thor's consumer Blackwell runs fp64 at a fraction of its fp32 rate. **The f64 stays.** §13.2 records it as what all four consumers' goldens were taken with and what makes the CUDA provider `memcmp`-identical; narrowing to torch's f32 re-gates four shipped models and is left OWED. **The chain breaks bit-identically instead**: one f64 accumulator per cell over a TILE of output positions with the `(ic, k)` sweep hoisted outside, so every cell receives the identical sequence of IEEE-754 double additions of the identical products in the identical order, interleaved across independent cells rather than serialised — a scheduling change, not an arithmetic one, so `memcmp` survives by construction. Measured single-threaded at the real vocoder shapes, `-ffp-contract=off`, `memcmp`-identical on every case: **3.4x-6.5x** per stage. Spec [`minimax-music3.md`](../../specs/minimax-music3.md) §18
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: perf
GitHub: 1334
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:441`

### Frozen archive evidence

> | [#1334](https://github.com/mudler/vllm.cpp/issues/1334) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | MiniMax-Music3's vocoder is **53.6 s of a 161.6 s run (33.2 %)** after [#1238](https://github.com/mudler/vllm.cpp/issues/1238), against 12.0 % before it, and #1238's depth A/B measured it at 53.6 s on BOTH legs — it is a FIXED term that grows as a share of every other improvement. **Both arms are slow for one cause.** `vt::cpu::Conv1dKernel` (`src/vt/cpu/cpu_conv1d_general.cpp::Conv1dKernel`) computes each output cell with ONE f64 accumulator swept over `(ic ascending, k ascending)`, which for a Music3 residual unit is `384 * 7 = 2688` STRICTLY DEPENDENT f64 additions per output element: no instruction-level parallelism, no vectorisation at any width, and a measured 1.7-2.0 GMAC/s per core = ~2.8-3.0 cycles per multiply-accumulate on a 5.0 GHz Zen 5 whose `fadd` latency is 3. The CUDA provider loses by a DIFFERENT mechanism with the same cause (§15.9: 3.552 s device vs 2.983 s host at latent length 20; §13.10's per-stage ratios flat at 0.37-0.40x across a 150x span of work): one accumulator per cell means it is not latency-bound but f64-RATE bound, and Thor's consumer Blackwell runs fp64 at a fraction of its fp32 rate. **The f64 stays.** §13.2 records it as what all four consumers' goldens were taken with and what makes the CUDA provider `memcmp`-identical; narrowing to torch's f32 re-gates four shipped models and is left OWED. **The chain breaks bit-identically instead**: one f64 accumulator per cell over a TILE of output positions with the `(ic, k)` sweep hoisted outside, so every cell receives the identical sequence of IEEE-754 double additions of the identical products in the identical order, interleaved across independent cells rather than serialised — a scheduling change, not an arithmetic one, so `memcmp` survives by construction. Measured single-threaded at the real vocoder shapes, `-ffp-contract=off`, `memcmp`-identical on every case: **3.4x-6.5x** per stage. Spec [`minimax-music3.md`](../../specs/minimax-music3.md) §18 | perf |

## Resolution

-
