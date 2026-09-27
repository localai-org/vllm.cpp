ID: ISSUE-GH-915
Title: Qwen3.8-27B (bf16): the token gate and every speed axis are OWED — the checkpoint loads and runs, and has never been adjudicated or measured. [`porting-inventory.md`](../../porting-inventory.md) deviation 17 already recorded that `Qwen/Qwen3.8-27B`@`1d4bf0f2` "fits GB10 and loads on current `main` with no code change" and that "its own token-exact gate is OWED and unrun", but no issue named either axis. Loading is not a token and a token is not a throughput number. [#821](https://github.com/mudler/vllm.cpp/issues/821) owns the NVFP4 and Q4_K_M arms of the same checkpoint and covers neither. Spec: [`qwen38-27b-bf16-gate.md`](../../specs/qwen38-27b-bf16-gate.md)
Row: MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation
State: UNKNOWN
Kind: verification
GitHub: 915
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:253`

### Frozen archive evidence

> | [#915](https://github.com/mudler/vllm.cpp/issues/915) | `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` | Qwen3.8-27B (bf16): the token gate and every speed axis are OWED — the checkpoint loads and runs, and has never been adjudicated or measured. [`porting-inventory.md`](../../porting-inventory.md) deviation 17 already recorded that `Qwen/Qwen3.8-27B`@`1d4bf0f2` "fits GB10 and loads on current `main` with no code change" and that "its own token-exact gate is OWED and unrun", but no issue named either axis. Loading is not a token and a token is not a throughput number. [#821](https://github.com/mudler/vllm.cpp/issues/821) owns the NVFP4 and Q4_K_M arms of the same checkpoint and covers neither. Spec: [`qwen38-27b-bf16-gate.md`](../../specs/qwen38-27b-bf16-gate.md) | verification |

## Resolution

-
