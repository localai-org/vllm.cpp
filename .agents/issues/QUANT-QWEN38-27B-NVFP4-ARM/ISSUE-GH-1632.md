ID: ISSUE-GH-1632
Title: **W6's NVFP4 token gate named [#1185](https://github.com/mudler/vllm.cpp/issues/1185) as the authority it waits on, and #1185 closed on 2026-08-18 as local-only** -- it tracked one operator's machines rather than a defect here -- so five sites pointed a reader at an issue that reports "closed" without reporting "cleared": `docs/FEATURES.md`, and the spec's `**Related:**` header, wave table, blockers section, `## Owed` list and `## Now`. **The blocker did not close with the issue, and it is not the one the citations described.** The pinned oracle `5559679229bc961848b121ccdeaa8fa5d79bec98` DOES build, install, import and GENERATE TOKENS inside an `rc` lease on `dgx:gpu0` (2026-08-18), which kills the "a model run is untested" clause those sites carried, and #1213 killed the "a lease cannot produce a runtime" premise underneath it. It survived at `max_num_batched_tokens` 512, `max_model_len` 512 and `gpu_memory_utilization` 0.30 on a ~20 GiB model, where the recorded denominator for this family is 8192 and 2048; `AGENTS.md` §Gates requires vLLM's PRODUCTION configuration as the denominator, so a reduced-`mnbt` arm is a different engine setup rather than a smaller measurement, and `gpu_memory_utilization` is a REFUTED lever (`.agents/specs/mtp-k-gt-1.md`: 0.75 thrashed 42 minutes, 0.30 rebooted the box). The named next levers are `max_num_batched_tokens` and `cudagraph_capture_sizes`, one at a time. The second half is the bytes: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`@`36f717a2` is ~20.4 GiB over four shards and is not mirrored where a lease can read it, which is also why its sha256 is recorded as unpaid. Same shape as [#1613](https://github.com/mudler/vllm.cpp/issues/1613) for the block-wise FP8 gate. FIXED IN FLOW: all five citations now name this issue, and the loader is untouched -- W5's accounting and cross-check need no lease and no oracle. Spec [`qwen38-27b-quant-arms.md`](../../specs/qwen38-27b-quant-arms.md), parent [#821](https://github.com/mudler/vllm.cpp/issues/821)
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: UNKNOWN
Kind: gap
GitHub: 1632
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:585`

### Frozen archive evidence

> | [#1632](https://github.com/mudler/vllm.cpp/issues/1632) | `QUANT-QWEN38-27B-NVFP4-ARM` | **W6's NVFP4 token gate named [#1185](https://github.com/mudler/vllm.cpp/issues/1185) as the authority it waits on, and #1185 closed on 2026-08-18 as local-only** -- it tracked one operator's machines rather than a defect here -- so five sites pointed a reader at an issue that reports "closed" without reporting "cleared": `docs/FEATURES.md`, and the spec's `**Related:**` header, wave table, blockers section, `## Owed` list and `## Now`. **The blocker did not close with the issue, and it is not the one the citations described.** The pinned oracle `5559679229bc961848b121ccdeaa8fa5d79bec98` DOES build, install, import and GENERATE TOKENS inside an `rc` lease on `dgx:gpu0` (2026-08-18), which kills the "a model run is untested" clause those sites carried, and #1213 killed the "a lease cannot produce a runtime" premise underneath it. It survived at `max_num_batched_tokens` 512, `max_model_len` 512 and `gpu_memory_utilization` 0.30 on a ~20 GiB model, where the recorded denominator for this family is 8192 and 2048; `AGENTS.md` §Gates requires vLLM's PRODUCTION configuration as the denominator, so a reduced-`mnbt` arm is a different engine setup rather than a smaller measurement, and `gpu_memory_utilization` is a REFUTED lever (`.agents/specs/mtp-k-gt-1.md`: 0.75 thrashed 42 minutes, 0.30 rebooted the box). The named next levers are `max_num_batched_tokens` and `cudagraph_capture_sizes`, one at a time. The second half is the bytes: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`@`36f717a2` is ~20.4 GiB over four shards and is not mirrored where a lease can read it, which is also why its sha256 is recorded as unpaid. Same shape as [#1613](https://github.com/mudler/vllm.cpp/issues/1613) for the block-wise FP8 gate. FIXED IN FLOW: all five citations now name this issue, and the loader is untouched -- W5's accounting and cross-check need no lease and no oracle. Spec [`qwen38-27b-quant-arms.md`](../../specs/qwen38-27b-quant-arms.md), parent [#821](https://github.com/mudler/vllm.cpp/issues/821) | gap |

## Resolution

-
