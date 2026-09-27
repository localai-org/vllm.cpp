ID: ISSUE-GH-1365
Title: Our c1 arm has a **reproducible ~4 s TTFT outlier on request 3 of every leg** that the pinned oracle does not have. Found 2026-08-19 in the raw `vllm bench serve --save-detailed` files of the Qwen3.8-27B bf16 c1/c8 re-measure ([#915](https://github.com/mudler/vllm.cpp/issues/915), [#979](https://github.com/mudler/vllm.cpp/issues/979)), `out/bench-20260819T035148Z/`. Concurrency 1, so the six requests are strictly serialized, and index 2 reads 3.981 / 3.924 / 4.006 / 3.955 s across the warmup leg and all three reps against 0.73-0.93 s for every other request in the same leg — FOUR legs of four, always the same index. Request 3 carries a 1024-token prompt exactly as requests 4, 5 and 6 do, so prompt length does not separate it, and the 77.005 s first value in the warmup leg is the separate known first-inference cost behind a liveness-only `/health`. The oracle `0.1.dev1+g555967922` on the byte-identical client invocation, same box and same lease (`out/vllm-20260819T095758Z/`), has no such point: 18 requests across three legs, every TTFT between 0.834 and 1.015 s. **Nothing published is wrong**: the repository quotes the MEDIAN and labels it, and the median of six averages ranks three and four, which the outlier never occupies. It moves the MEAN — ours 1347.6 / 1372.6 / 1365.4 ms against vLLM's 873.3 / 883.4 / 900.2 ms while the medians read 883.78 against 876.4 — and it costs wall time, request 4 starting 31.63 s after request 3 where every other gap is ~28.4 s, so roughly 3.1 s of the 174.39 s c1 wall. A reproducible outlier at a FIXED request index in four legs of four is a behaviour rather than noise, which is why it is filed rather than left in the record. NOT fixed in flow: the finding row writes no product code and holds no GPU, and the cause is deliberately not chased. Owed under `## Owed` in [qwen38-27b-bf16-gate.md](../../specs/qwen38-27b-bf16-gate.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1365
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:449`

### Frozen archive evidence

> | [#1365](https://github.com/mudler/vllm.cpp/issues/1365) | — | Our c1 arm has a **reproducible ~4 s TTFT outlier on request 3 of every leg** that the pinned oracle does not have. Found 2026-08-19 in the raw `vllm bench serve --save-detailed` files of the Qwen3.8-27B bf16 c1/c8 re-measure ([#915](https://github.com/mudler/vllm.cpp/issues/915), [#979](https://github.com/mudler/vllm.cpp/issues/979)), `out/bench-20260819T035148Z/`. Concurrency 1, so the six requests are strictly serialized, and index 2 reads 3.981 / 3.924 / 4.006 / 3.955 s across the warmup leg and all three reps against 0.73-0.93 s for every other request in the same leg — FOUR legs of four, always the same index. Request 3 carries a 1024-token prompt exactly as requests 4, 5 and 6 do, so prompt length does not separate it, and the 77.005 s first value in the warmup leg is the separate known first-inference cost behind a liveness-only `/health`. The oracle `0.1.dev1+g555967922` on the byte-identical client invocation, same box and same lease (`out/vllm-20260819T095758Z/`), has no such point: 18 requests across three legs, every TTFT between 0.834 and 1.015 s. **Nothing published is wrong**: the repository quotes the MEDIAN and labels it, and the median of six averages ranks three and four, which the outlier never occupies. It moves the MEAN — ours 1347.6 / 1372.6 / 1365.4 ms against vLLM's 873.3 / 883.4 / 900.2 ms while the medians read 883.78 against 876.4 — and it costs wall time, request 4 starting 31.63 s after request 3 where every other gap is ~28.4 s, so roughly 3.1 s of the 174.39 s c1 wall. A reproducible outlier at a FIXED request index in four legs of four is a behaviour rather than noise, which is why it is filed rather than left in the record. NOT fixed in flow: the finding row writes no product code and holds no GPU, and the cause is deliberately not chased. Owed under `## Owed` in [qwen38-27b-bf16-gate.md](../../specs/qwen38-27b-bf16-gate.md) | bug |

## Resolution

-
