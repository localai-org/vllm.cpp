ID: ISSUE-GH-1600
Title: **A misspelled or mis-cased `vllm_cpp` on a chain-only document is answered with `a string "method" is required` — the one key such a document must not have.** Measured at `31cefe631` + `e2a9e035d`: `{"VLLM_CPP":{"drafter_chain":[...]}}`, a bare `{"drafter_chain":[...]}` and `{"vllm_cp":{...}}` all return that message. Each document names every speculator it wants; the engine demands the one key that D7 makes MUTUALLY EXCLUSIVE with a chain, and the engine says so itself when the spelling is right — so the user is told to add the key that would then be refused. Mechanism: `has_chain = doc.contains("vllm_cpp")` is false, so the method requirement at `src/vllm/config/speculative.cpp:353-379` fires before the key-admission loop at `:400-429`, which is where the unknown name would have been reported by name with the accepted list. **The ORDERING is inherited and deliberate** — #1160 put the method check first so an unsupported method is the first error a user sees — but the document CLASS that hits it is invented by this wave: before the chain existed, a document with no `method` was simply an incomplete vLLM document and the message was right. `.agents/specs/drafter-chain.md` D9 argues at length that the user must not be misled here, and this is the one shape where the landed code misleads. Two candidate repairs, neither chosen: report an unadmitted top-level key BEFORE the method requirement when the document carries no `method` at all, or judge the extension key case-insensitively for the DIAGNOSTIC only. Both move a landed error ordering that `tests/vllm/config/test_speculative_unknown_keys.cpp` and this row's own regression case assert on, so it needs its own row, a red-before test and green-after evidence. Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1`
Row: SPEC-DRAFTER-CHAIN
State: UNKNOWN
Kind: bug
GitHub: 1600
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:565`

### Frozen archive evidence

> | [#1600](https://github.com/mudler/vllm.cpp/issues/1600) | `SPEC-DRAFTER-CHAIN` | **A misspelled or mis-cased `vllm_cpp` on a chain-only document is answered with `a string "method" is required` — the one key such a document must not have.** Measured at `31cefe631` + `e2a9e035d`: `{"VLLM_CPP":{"drafter_chain":[...]}}`, a bare `{"drafter_chain":[...]}` and `{"vllm_cp":{...}}` all return that message. Each document names every speculator it wants; the engine demands the one key that D7 makes MUTUALLY EXCLUSIVE with a chain, and the engine says so itself when the spelling is right — so the user is told to add the key that would then be refused. Mechanism: `has_chain = doc.contains("vllm_cpp")` is false, so the method requirement at `src/vllm/config/speculative.cpp:353-379` fires before the key-admission loop at `:400-429`, which is where the unknown name would have been reported by name with the accepted list. **The ORDERING is inherited and deliberate** — #1160 put the method check first so an unsupported method is the first error a user sees — but the document CLASS that hits it is invented by this wave: before the chain existed, a document with no `method` was simply an incomplete vLLM document and the message was right. `.agents/specs/drafter-chain.md` D9 argues at length that the user must not be misled here, and this is the one shape where the landed code misleads. Two candidate repairs, neither chosen: report an unadmitted top-level key BEFORE the method requirement when the document carries no `method` at all, or judge the extension key case-insensitively for the DIAGNOSTIC only. Both move a landed error ordering that `tests/vllm/config/test_speculative_unknown_keys.cpp` and this row's own regression case assert on, so it needs its own row, a red-before test and green-after evidence. Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1` | bug |

## Resolution

-
