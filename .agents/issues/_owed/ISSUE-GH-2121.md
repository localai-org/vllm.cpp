ID: ISSUE-GH-2121
Title: **`AsyncLLM::add_request_wave` does not fan out `n > 1`.** Both overloads (`src/vllm/v1/engine/async_llm.cpp:133,166`) register every input with `request_index=0` and no `ParentRequest`, so a wave entry carrying `n > 1` is served as `n == 1` — the same defect [#1816](https://github.com/mudler/vllm.cpp/issues/1816) records for the single-request overloads. The wave is a LOCAL extension with no upstream counterpart and no OpenAI route reaches it; its only caller is `examples/bench/bench_core.h:222,225`, which is why it is excluded from #1816's fix rather than folded into it. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2121
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:830`

### Frozen archive evidence

> | [#2121](https://github.com/mudler/vllm.cpp/issues/2121) | — | **`AsyncLLM::add_request_wave` does not fan out `n > 1`.** Both overloads (`src/vllm/v1/engine/async_llm.cpp:133,166`) register every input with `request_index=0` and no `ParentRequest`, so a wave entry carrying `n > 1` is served as `n == 1` — the same defect [#1816](https://github.com/mudler/vllm.cpp/issues/1816) records for the single-request overloads. The wave is a LOCAL extension with no upstream counterpart and no OpenAI route reaches it; its only caller is `examples/bench/bench_core.h:222,225`, which is why it is excluded from #1816's fix rather than folded into it. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed` | bug |

## Resolution

-
