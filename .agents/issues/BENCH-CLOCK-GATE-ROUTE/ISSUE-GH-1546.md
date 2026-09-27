ID: ISSUE-GH-1546
Title: **`gpu_clock_state.compare_clock_records` bounds the cross-arm MEDIAN offset and nothing bounds the difference in EXCURSION BURDEN between the arms.** The two are independent on the 2026-08-19 Qwen3.8-27B bf16 c1 evidence: `median_offset_pct` is exactly **0.0** on all three pairings while the arms time-weighted mean-clock cost differs by **0.020 / 0.153 / 0.103** points rep for rep (ours 0.136 / 0.402 / 0.204 against vLLM 0.116 / 0.249 / 0.307, re-derived from the raw `*.samples.json` at `/mnt/nas_share/rc/q38bf16/out/`). The median cannot see the excursion population, which is exactly the part that does NOT cancel between the arms and therefore the part that transfers into the ratio. Proposed: one ADDITIVE term holding the two arms mean-clock cost within the same physics ceiling `MAX_CROSS_ARM_OFFSET_PCT` already rests on, which encodes what `7e07bbc91` measured -- a workload-generated excursion appears in BOTH arms, a GPU-state defect appears in ONE. Explicit non-goal: this must NOT become a route to re-scoring the nine discarded windows, which carry two independent refusals and stay `DISCARD`. Decided in [clock-gate-route.md](../../specs/clock-gate-route.md)
Row: BENCH-CLOCK-GATE-ROUTE
State: UNKNOWN
Kind: gap
GitHub: 1546
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:547`

### Frozen archive evidence

> | [#1546](https://github.com/mudler/vllm.cpp/issues/1546) | `BENCH-CLOCK-GATE-ROUTE` | **`gpu_clock_state.compare_clock_records` bounds the cross-arm MEDIAN offset and nothing bounds the difference in EXCURSION BURDEN between the arms.** The two are independent on the 2026-08-19 Qwen3.8-27B bf16 c1 evidence: `median_offset_pct` is exactly **0.0** on all three pairings while the arms time-weighted mean-clock cost differs by **0.020 / 0.153 / 0.103** points rep for rep (ours 0.136 / 0.402 / 0.204 against vLLM 0.116 / 0.249 / 0.307, re-derived from the raw `*.samples.json` at `/mnt/nas_share/rc/q38bf16/out/`). The median cannot see the excursion population, which is exactly the part that does NOT cancel between the arms and therefore the part that transfers into the ratio. Proposed: one ADDITIVE term holding the two arms mean-clock cost within the same physics ceiling `MAX_CROSS_ARM_OFFSET_PCT` already rests on, which encodes what `7e07bbc91` measured -- a workload-generated excursion appears in BOTH arms, a GPU-state defect appears in ONE. Explicit non-goal: this must NOT become a route to re-scoring the nine discarded windows, which carry two independent refusals and stay `DISCARD`. Decided in [clock-gate-route.md](../../specs/clock-gate-route.md) | gap |

## Resolution

-
