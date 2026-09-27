ID: ISSUE-GH-1599
Title: **A non-string `model` is silently dropped, and the required-key message that follows names the key as MISSING when the user gave it.** Measured at `31cefe631` + `e2a9e035d`: `{"vllm_cpp":{"drafter_chain":[{"method":"mtp","model":123}]}}` parses, with `entry[0].draft_model_path` empty; `{..."method":"dflash","model":123}` refuses with `requires a "model" key naming the draft checkpoint`, pointing away from the actual mistake. `src/vllm/config/speculative.cpp:230-232` reads the key only when `is_string()` already holds, so every other JSON type takes the silent branch. **INHERITED, not introduced**: the top level has the identical shape at `:526-528` and behaves identically — `{"method":"dflash","model":123}` reports the key as missing too — so the chain entry is faithful to the landed contract and repairing only the entry would leave two spellings of one rule disagreeing. The repair belongs to both together: judge PRESENCE, then TYPE, and say "must be a string" when a value of the wrong type was given, which is #1160's polarity applied to value type instead of key presence. NOT fixed in `SPEC-DRAFTER-CHAIN` W1: it moves a landed top-level refusal that other suites assert on. Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1`
Row: SPEC-DRAFTER-CHAIN
State: UNKNOWN
Kind: bug
GitHub: 1599
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:564`

### Frozen archive evidence

> | [#1599](https://github.com/mudler/vllm.cpp/issues/1599) | `SPEC-DRAFTER-CHAIN` | **A non-string `model` is silently dropped, and the required-key message that follows names the key as MISSING when the user gave it.** Measured at `31cefe631` + `e2a9e035d`: `{"vllm_cpp":{"drafter_chain":[{"method":"mtp","model":123}]}}` parses, with `entry[0].draft_model_path` empty; `{..."method":"dflash","model":123}` refuses with `requires a "model" key naming the draft checkpoint`, pointing away from the actual mistake. `src/vllm/config/speculative.cpp:230-232` reads the key only when `is_string()` already holds, so every other JSON type takes the silent branch. **INHERITED, not introduced**: the top level has the identical shape at `:526-528` and behaves identically — `{"method":"dflash","model":123}` reports the key as missing too — so the chain entry is faithful to the landed contract and repairing only the entry would leave two spellings of one rule disagreeing. The repair belongs to both together: judge PRESENCE, then TYPE, and say "must be a string" when a value of the wrong type was given, which is #1160's polarity applied to value type instead of key presence. NOT fixed in `SPEC-DRAFTER-CHAIN` W1: it moves a landed top-level refusal that other suites assert on. Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1` | bug |

## Resolution

-
