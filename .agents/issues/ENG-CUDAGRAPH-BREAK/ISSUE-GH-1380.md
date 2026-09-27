ID: ISSUE-GH-1380
Title: A speculative decode-graph capture does a `cudaMalloc` inside the captured region and throws on `thor:gpu0` (sm_110), and the queue is POISONED afterwards. Located to the SECOND parity-ring slot: slot 0 cold, slot 1 cold, slot 0 captures (`captured()` true, `replay_count()` 1), slot 1's capture throws `cudaMalloc: operation not permitted when stream is capturing`, and the next step fails with `embedding: operation failed due to a previous error during capture` without opening a scope. So a REPLAY is unreachable on a speculative shape there, on a default-ON path (`VT_SPEC_DECODE_GRAPH`). The driver's own pre-grow names this case and covers only the retained `[S, vocab]` logits. Found by [#1374](https://github.com/mudler/vllm.cpp/issues/1374) and PRE-EXISTING — the case drives the driver directly, bypassing the predicate, and the five migrated drivers read 2066 assertions / 0 differing on the same binary. NOT fixed in flow: it is a device-level allocation defect on a path W6 did not write, it needs `dgx`/sm_121a and a real checkpoint to scope, and `AGENTS.md` routes a surprising fix to the normal row, spec and fresh-review path. Owned by row `ENG-CUDAGRAPH-BREAK`, under `## Owed` in [`eng-cudagraph-break.md`](../../specs/eng-cudagraph-break.md)
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1380
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:464`

### Frozen archive evidence

> | [#1380](https://github.com/mudler/vllm.cpp/issues/1380) | `ENG-CUDAGRAPH-BREAK` | A speculative decode-graph capture does a `cudaMalloc` inside the captured region and throws on `thor:gpu0` (sm_110), and the queue is POISONED afterwards. Located to the SECOND parity-ring slot: slot 0 cold, slot 1 cold, slot 0 captures (`captured()` true, `replay_count()` 1), slot 1's capture throws `cudaMalloc: operation not permitted when stream is capturing`, and the next step fails with `embedding: operation failed due to a previous error during capture` without opening a scope. So a REPLAY is unreachable on a speculative shape there, on a default-ON path (`VT_SPEC_DECODE_GRAPH`). The driver's own pre-grow names this case and covers only the retained `[S, vocab]` logits. Found by [#1374](https://github.com/mudler/vllm.cpp/issues/1374) and PRE-EXISTING — the case drives the driver directly, bypassing the predicate, and the five migrated drivers read 2066 assertions / 0 differing on the same binary. NOT fixed in flow: it is a device-level allocation defect on a path W6 did not write, it needs `dgx`/sm_121a and a real checkpoint to scope, and `AGENTS.md` routes a surprising fix to the normal row, spec and fresh-review path. Owned by row `ENG-CUDAGRAPH-BREAK`, under `## Owed` in [`eng-cudagraph-break.md`](../../specs/eng-cudagraph-break.md) | bug |

## Resolution

-
