ID: ISSUE-GH-2151
Title: **`cb28167c9` (#2148) reverted the reorder-threshold wiring on an uncontrolled measurement, so the revert comes back out.** The revert's stated reason — the wiring "costs 38% at c=8" — compared two builds run hours apart, in commit order, on a drifting box, with neither arm re-measured. An interleaved A/B with a terminal control (`A B A B A B A`, one lease, one hour, 2026-08-28) settles it: the arm WITHOUT the wiring (`16ebcac4b`) read 56.22, 51.29 and 36.82 out tok/s, and the arm WITH it (`5e9d81dad`) read 34.66, 35.49, 43.30 and 78.86 — the wired arm holds both the LOWEST and the HIGHEST reading in the set, so no ordering between the builds exists and the 38% was an artifact of the instrument. The wiring is justified without any throughput claim: `SpecAsDecodeReorderThreshold` (`include/vllm/v1/attention/backend.h:180-184`) mirrors upstream's `1 + (parallel_drafting ? 2 : 1) * k` and reaches only the spec-as-decode classification at `:201`, while the reorder the value exists to bound takes the declaration default of 1, against upstream's `_may_reorder_batch` passing `decode_threshold=self.reorder_batch_threshold` (`gpu_model_runner.py:1126-1130` @ pin `5559679229`). Restores #2138 byte-for-byte and repairs `## Now`, `## Outcome` and `## Owed` in [specs/reorder-threshold-wiring.md](../../specs/reorder-threshold-wiring.md), whose `## WITHDRAWN` section carried the same false premise. The instrument defect the run exposed is larger than this row and is tracked separately by [#2152](https://github.com/mudler/vllm.cpp/issues/2152)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 2151
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:834`

### Frozen archive evidence

> | [#2151](https://github.com/mudler/vllm.cpp/issues/2151) | `SPEC-DFLASH2` | **`cb28167c9` (#2148) reverted the reorder-threshold wiring on an uncontrolled measurement, so the revert comes back out.** The revert's stated reason — the wiring "costs 38% at c=8" — compared two builds run hours apart, in commit order, on a drifting box, with neither arm re-measured. An interleaved A/B with a terminal control (`A B A B A B A`, one lease, one hour, 2026-08-28) settles it: the arm WITHOUT the wiring (`16ebcac4b`) read 56.22, 51.29 and 36.82 out tok/s, and the arm WITH it (`5e9d81dad`) read 34.66, 35.49, 43.30 and 78.86 — the wired arm holds both the LOWEST and the HIGHEST reading in the set, so no ordering between the builds exists and the 38% was an artifact of the instrument. The wiring is justified without any throughput claim: `SpecAsDecodeReorderThreshold` (`include/vllm/v1/attention/backend.h:180-184`) mirrors upstream's `1 + (parallel_drafting ? 2 : 1) * k` and reaches only the spec-as-decode classification at `:201`, while the reorder the value exists to bound takes the declaration default of 1, against upstream's `_may_reorder_batch` passing `decode_threshold=self.reorder_batch_threshold` (`gpu_model_runner.py:1126-1130` @ pin `5559679229`). Restores #2138 byte-for-byte and repairs `## Now`, `## Outcome` and `## Owed` in [specs/reorder-threshold-wiring.md](../../specs/reorder-threshold-wiring.md), whose `## WITHDRAWN` section carried the same false premise. The instrument defect the run exposed is larger than this row and is tracked separately by [#2152](https://github.com/mudler/vllm.cpp/issues/2152) | bug |

## Resolution

-
