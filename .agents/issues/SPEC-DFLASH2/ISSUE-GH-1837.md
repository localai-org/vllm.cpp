ID: ISSUE-GH-1837
Title: **DFlash2 propose round-trips ~25 MB and ~10 full-queue syncs per step because the `final_out` host contract takes every step off the D13 paged+graph draft forward.** Measured 2026-08-24 on the #1574 workload: step 141.5 ms vs vLLM ~127 / SGLang ~121 at equal acceptance, and the whole remaining gap is propose overhead the in-code debt note in `runner.cpp::propose_drafts_block` already names. W8 mirrors the merged upstream `_generate_draft` (`b389ac2946`): the block forward hands out DEVICE logits+hidden, the selector runs TopK/edges/walk device-to-device, and only the K draft token ids come back. Wave spec [dflash2-device-propose.md](../../specs/dflash2-device-propose.md); the GPU TPOT number is owed there, operator-run
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1837
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:689`

### Frozen archive evidence

> | [#1837](https://github.com/mudler/vllm.cpp/issues/1837) | `SPEC-DFLASH2` | **DFlash2 propose round-trips ~25 MB and ~10 full-queue syncs per step because the `final_out` host contract takes every step off the D13 paged+graph draft forward.** Measured 2026-08-24 on the #1574 workload: step 141.5 ms vs vLLM ~127 / SGLang ~121 at equal acceptance, and the whole remaining gap is propose overhead the in-code debt note in `runner.cpp::propose_drafts_block` already names. W8 mirrors the merged upstream `_generate_draft` (`b389ac2946`): the block forward hands out DEVICE logits+hidden, the selector runs TopK/edges/walk device-to-device, and only the K draft token ids come back. Wave spec [dflash2-device-propose.md](../../specs/dflash2-device-propose.md); the GPU TPOT number is owed there, operator-run | perf |

## Resolution

-
