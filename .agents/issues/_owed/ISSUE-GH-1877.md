ID: ISSUE-GH-1877
Title: NVFP4 35B greedy decode is not run-to-run reproducible at c1 for requests after the first: two identical invocations (same binary, env, seed 777, temp 0) diverge at tokens 21 and 457 on requests 2-3, while request 1 is byte-identical and the bf16 35B null is clean on the same harness. Sibling of #1283 (which was c16); candidates: marlin/grouped-MoE atomics, per-process fp8 plan selection, cross-request state. Makes arm-identity-at-depth undecidable on this checkpoint. Found by the GDN-MOE-PACKED-BA speed A/B; listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1877
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:703`

### Frozen archive evidence

> | [#1877](https://github.com/mudler/vllm.cpp/issues/1877) | — | NVFP4 35B greedy decode is not run-to-run reproducible at c1 for requests after the first: two identical invocations (same binary, env, seed 777, temp 0) diverge at tokens 21 and 457 on requests 2-3, while request 1 is byte-identical and the bf16 35B null is clean on the same harness. Sibling of #1283 (which was c16); candidates: marlin/grouped-MoE atomics, per-process fp8 plan selection, cross-request state. Makes arm-identity-at-depth undecidable on this checkpoint. Found by the GDN-MOE-PACKED-BA speed A/B; listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md) | bug |

## Resolution

-
