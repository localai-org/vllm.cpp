ID: ISSUE-GH-682
Title: The 35B mnbt sweep gave our arm 8,192 tokens of KV (256 blocks x 32, no `--num-blocks`) and the pin 1,819,368 — a 222x asymmetry that suppressed the very batching the sweep was varying; measured, our arm could not form a wave >2048 tokens at EITHER budget until KV was provisioned (spec [`perf-chunked-prefill-budget-2026-08-13.md`](../../specs/perf-chunked-prefill-budget-2026-08-13.md) §Runtime)
Row: SERVE-GATE-ONLINE
State: UNKNOWN
Kind: bug
GitHub: 682
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:156`

### Frozen archive evidence

> | [#682](https://github.com/mudler/vllm.cpp/issues/682) | `SERVE-GATE-ONLINE` | The 35B mnbt sweep gave our arm 8,192 tokens of KV (256 blocks x 32, no `--num-blocks`) and the pin 1,819,368 — a 222x asymmetry that suppressed the very batching the sweep was varying; measured, our arm could not form a wave >2048 tokens at EITHER budget until KV was provisioned (spec [`perf-chunked-prefill-budget-2026-08-13.md`](../../specs/perf-chunked-prefill-budget-2026-08-13.md) §Runtime) | bug |

## Resolution

-
