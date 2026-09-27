ID: ISSUE-GH-2283
Title: **The DeepSeek-V4 carried tower's BF16-sourced half is still widened to f32 (~2.62 GiB), and W1d's ~97.7 GiB projection has never been observed.** Filed 2026-08-29 because W1d ([#2186](https://github.com/mudler/vllm.cpp/issues/2186), landed `c9ad53fee`) CLOSED its issue while `.agents/specs/model-dsv4-exl3.md` `## Owed` still pointed two live entries at it -- a reader following either landed on a closed issue. **(1)** The 108.59 -> ~97.7 GiB figure is arithmetic on the measured 108.59 / 26.64 split, not a load anyone has watched complete; the last real measurement (2026-08-28, `dgx:gpu0`, worker `rc-worker-4b8lj`, tree `525d2b991`) REFUSED, and nothing has re-run since. It falls due as an `rc`-leased `dgx:gpu0` measurement against the staged 100 GB artifact, and a load that completes is still not a forward that runs (#1961, #1970, #1976). **(2)** The carried tower's other half -- norms, embeddings, router, `BF16` on disk, 2.621 GiB -> 5.24 GiB at f32 -- is untouched, and the same "Inherit vLLM defaults" argument applies verbatim. NOT folded into W1d deliberately: W1d's nine fields had three consuming functions and one device vtable entry, while this half is read by the sampler and lm_head paths too (the embedding is held twice on device, #1946), so it is a wave with its own gate. Build on what W1d left: `vllm::HostBf16`, the inlined `vllm::HostBf16ToF32` (out-of-line `vt::BF16ToF32` + no LTO would cost a call per element in the GEMV inner loop) with its exhaustive 65536-pattern agreement case, `Dot`'s bf16 overload, generic `MatVec`/`Gemm`/`expert_f32`/`GroupedOutputLora`, and a `DeepseekV4HostResidentBytes` that now reads each field's own `value_type` under a mutation-proven gate. Spec [model-dsv4-exl3.md](../../specs/model-dsv4-exl3.md) `## Owed`
Row: MODEL-DSV4-EXL3
State: UNKNOWN
Kind: bug
GitHub: 2283
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:886`

### Frozen archive evidence

> | [#2283](https://github.com/mudler/vllm.cpp/issues/2283) | `MODEL-DSV4-EXL3` | **The DeepSeek-V4 carried tower's BF16-sourced half is still widened to f32 (~2.62 GiB), and W1d's ~97.7 GiB projection has never been observed.** Filed 2026-08-29 because W1d ([#2186](https://github.com/mudler/vllm.cpp/issues/2186), landed `c9ad53fee`) CLOSED its issue while `.agents/specs/model-dsv4-exl3.md` `## Owed` still pointed two live entries at it -- a reader following either landed on a closed issue. **(1)** The 108.59 -> ~97.7 GiB figure is arithmetic on the measured 108.59 / 26.64 split, not a load anyone has watched complete; the last real measurement (2026-08-28, `dgx:gpu0`, worker `rc-worker-4b8lj`, tree `525d2b991`) REFUSED, and nothing has re-run since. It falls due as an `rc`-leased `dgx:gpu0` measurement against the staged 100 GB artifact, and a load that completes is still not a forward that runs (#1961, #1970, #1976). **(2)** The carried tower's other half -- norms, embeddings, router, `BF16` on disk, 2.621 GiB -> 5.24 GiB at f32 -- is untouched, and the same "Inherit vLLM defaults" argument applies verbatim. NOT folded into W1d deliberately: W1d's nine fields had three consuming functions and one device vtable entry, while this half is read by the sampler and lm_head paths too (the embedding is held twice on device, #1946), so it is a wave with its own gate. Build on what W1d left: `vllm::HostBf16`, the inlined `vllm::HostBf16ToF32` (out-of-line `vt::BF16ToF32` + no LTO would cost a call per element in the GEMV inner loop) with its exhaustive 65536-pattern agreement case, `Dot`'s bf16 overload, generic `MatVec`/`Gemm`/`expert_f32`/`GroupedOutputLora`, and a `DeepseekV4HostResidentBytes` that now reads each field's own `value_type` under a mutation-proven gate. Spec [model-dsv4-exl3.md](../../specs/model-dsv4-exl3.md) `## Owed` | bug |

## Resolution

-
