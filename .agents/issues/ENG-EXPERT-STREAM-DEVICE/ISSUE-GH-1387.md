ID: ISSUE-GH-1387
Title: `docs/FEATURES.md`'s routed-expert-streaming row still read "CPU keep-quant towers only" after W0c made the seam take the slot arm on `is_cpu()` OR `host_memory_is_device_addressable()` and W0d made the load-time fit refusal drop those towers from its bound. `AGENTS.md` routes a feature-surface change to that page, and the change that moved the surface did not write it. `scripts/check-doc-checkpoint.py` said so, on commit `939755f99` of `row/ENG-EXPERT-STREAM-DEVICE-W0`: a measurement was appended to `.agents/benchmark-record.md` with no `docs/FEATURES.md` edit beside it. The PAGE is FIXED IN FLOW while repairing the fresh review of [#1377](https://github.com/mudler/vllm.cpp/pull/1377): the row now names both the device arm ([#1124](https://github.com/mudler/vllm.cpp/issues/1124)) and the residency condition ([#1378](https://github.com/mudler/vllm.cpp/issues/1378)), and the observability detail it displaced to stay inside the 220-character cell budget is stated at `docs/USAGE.md:4598-4620`. The GATE is NOT fixed and needs a decision: the checker walks a range one COMMIT at a time, so once a commit is published on a branch that may not be force-pushed, no later commit can make it green, and `scripts/agent-preflight.sh` keeps reporting `doc-checkpoint range` red on this branch until it merges. The squashed commit that lands on `main` carries both paths and passes. Changing the walk is checker semantics and needs its own row, spec and red-first evidence per `AGENTS.md` "Changing the rules or a checker", so it is not folded in here. Spec [`expert-stream-device-slots.md`](../../specs/expert-stream-device-slots.md)
Row: ENG-EXPERT-STREAM-DEVICE
State: UNKNOWN
Kind: bug
GitHub: 1387
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:475`

### Frozen archive evidence

> | [#1387](https://github.com/mudler/vllm.cpp/issues/1387) | `ENG-EXPERT-STREAM-DEVICE` | `docs/FEATURES.md`'s routed-expert-streaming row still read "CPU keep-quant towers only" after W0c made the seam take the slot arm on `is_cpu()` OR `host_memory_is_device_addressable()` and W0d made the load-time fit refusal drop those towers from its bound. `AGENTS.md` routes a feature-surface change to that page, and the change that moved the surface did not write it. `scripts/check-doc-checkpoint.py` said so, on commit `939755f99` of `row/ENG-EXPERT-STREAM-DEVICE-W0`: a measurement was appended to `.agents/benchmark-record.md` with no `docs/FEATURES.md` edit beside it. The PAGE is FIXED IN FLOW while repairing the fresh review of [#1377](https://github.com/mudler/vllm.cpp/pull/1377): the row now names both the device arm ([#1124](https://github.com/mudler/vllm.cpp/issues/1124)) and the residency condition ([#1378](https://github.com/mudler/vllm.cpp/issues/1378)), and the observability detail it displaced to stay inside the 220-character cell budget is stated at `docs/USAGE.md:4598-4620`. The GATE is NOT fixed and needs a decision: the checker walks a range one COMMIT at a time, so once a commit is published on a branch that may not be force-pushed, no later commit can make it green, and `scripts/agent-preflight.sh` keeps reporting `doc-checkpoint range` red on this branch until it merges. The squashed commit that lands on `main` carries both paths and passes. Changing the walk is checker semantics and needs its own row, spec and red-first evidence per `AGENTS.md` "Changing the rules or a checker", so it is not folded in here. Spec [`expert-stream-device-slots.md`](../../specs/expert-stream-device-slots.md) | bug |

## Resolution

-
