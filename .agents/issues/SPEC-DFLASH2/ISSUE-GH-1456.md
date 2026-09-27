ID: ISSUE-GH-1456
Title: **The GB10 oracle DOES have a FLASH_ATTN denominator: the arch measurement stands, the conclusion drawn from it does not.** Measured 2026-08-21 by W6 on `dgx:gpu0` through an `rc` lease, with the very wheel #1456 was filed about (`vllm-0.1.dev1+g66e5414c6`, sha256 `fbc247ab...`). A capture that exported `VLLM_ATTENTION_BACKEND=TRITON_ATTN` got `FLASH_ATTN` anyway and RAN: `Using FlashAttention version 2`, 54.87 GiB loaded, CUDA graphs captured (PIECEWISE 5/5, FULL 1/1, plus the DFlash2 speculator's own), 4 x 64 coherent tokens, speculation live at 209 accepted of 350 drafted and mean acceptance length 5.00. No `cudaErrorUnsupportedPtxVersion`. Consistent with the `sm_80`/`sm_75` SASS finding rather than contradicting it: `sm_80` PTX JITs FORWARD, and that error is the OPPOSITE failure (PTX newer than the driver). So `FA_USABLE=0` in the staged `FA-CONSTRAINT.txt` was inferred from emitted arches, never observed from a run, and is the thing to reconcile. A SECOND trap found in the same run and recorded so nobody repeats it: **`VLLM_ATTENTION_BACKEND` does not exist at this revision** -- grepping every `.py` in the wheel returns nothing; the knob is `EngineArgs.attention_backend` (`arg_utils.py:706`) folded into `AttentionConfig.backend` (`:2382`), so the old export selects NOTHING and auto-selection wins silently, letting a run record one backend while executing another. NOT reconciled in flow: W6 does not substitute a denominator the developer declared, and takes both arms instead, each named in its own golden. Owed under `## Owed` O22 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: verification
GitHub: 1456
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:587`

### Frozen archive evidence

> | [#1456](https://github.com/mudler/vllm.cpp/issues/1456) | `SPEC-DFLASH2` | **The GB10 oracle DOES have a FLASH_ATTN denominator: the arch measurement stands, the conclusion drawn from it does not.** Measured 2026-08-21 by W6 on `dgx:gpu0` through an `rc` lease, with the very wheel #1456 was filed about (`vllm-0.1.dev1+g66e5414c6`, sha256 `fbc247ab...`). A capture that exported `VLLM_ATTENTION_BACKEND=TRITON_ATTN` got `FLASH_ATTN` anyway and RAN: `Using FlashAttention version 2`, 54.87 GiB loaded, CUDA graphs captured (PIECEWISE 5/5, FULL 1/1, plus the DFlash2 speculator's own), 4 x 64 coherent tokens, speculation live at 209 accepted of 350 drafted and mean acceptance length 5.00. No `cudaErrorUnsupportedPtxVersion`. Consistent with the `sm_80`/`sm_75` SASS finding rather than contradicting it: `sm_80` PTX JITs FORWARD, and that error is the OPPOSITE failure (PTX newer than the driver). So `FA_USABLE=0` in the staged `FA-CONSTRAINT.txt` was inferred from emitted arches, never observed from a run, and is the thing to reconcile. A SECOND trap found in the same run and recorded so nobody repeats it: **`VLLM_ATTENTION_BACKEND` does not exist at this revision** -- grepping every `.py` in the wheel returns nothing; the knob is `EngineArgs.attention_backend` (`arg_utils.py:706`) folded into `AttentionConfig.backend` (`:2382`), so the old export selects NOTHING and auto-selection wins silently, letting a run record one backend while executing another. NOT reconciled in flow: W6 does not substitute a denominator the developer declared, and takes both arms instead, each named in its own golden. Owed under `## Owed` O22 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md) | verification |

## Resolution

-
