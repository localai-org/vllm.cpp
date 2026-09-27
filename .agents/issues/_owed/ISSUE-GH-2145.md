ID: ISSUE-GH-2145
Title: **The parallel-sampling fan-out DEEP-copies the prompt `n` times where upstream's copy is shallow.** Both fan-out sites build each child with `EngineCoreRequest child = request;` (`src/vllm/v1/engine/async_llm.cpp` `PublishParallelSampling`, `src/vllm/v1/engine/llm_engine.cpp` `FanOutParallelSampling`), and `EngineCoreRequest::prompt_token_ids` is a `std::vector<int32_t>` held BY VALUE (`include/vllm/v1/engine/types.h:79`), so each of the `n` children owns a full copy of the prompt and `Request::FromEngineCoreRequest` makes a second one per child — `O(n * prompt_len)` bytes moved before the first token is scheduled. Upstream copies ZERO prompt tokens: `copy(request)` (`vllm/v1/engine/async_llm.py:393`, `vllm/v1/engine/llm_engine.py:283` @ pin `5559679229`) is SHALLOW, every child references the same list object, and the last child reuses the parent outright. The comment on both of our lines claimed the copy "shares the prompt token ids", which is FALSE; [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s pull request corrects the comment and points here, and does NOT fix the cost, because the cheap mirror is a shared immutable token buffer on `EngineCoreRequest` that every engine path reads — a types-level change with its own review surface. No correctness effect; the cost scales with prompt length times `n`, so it is invisible on the short-prompt suites. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2145
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:831`

### Frozen archive evidence

> | [#2145](https://github.com/mudler/vllm.cpp/issues/2145) | — | **The parallel-sampling fan-out DEEP-copies the prompt `n` times where upstream's copy is shallow.** Both fan-out sites build each child with `EngineCoreRequest child = request;` (`src/vllm/v1/engine/async_llm.cpp` `PublishParallelSampling`, `src/vllm/v1/engine/llm_engine.cpp` `FanOutParallelSampling`), and `EngineCoreRequest::prompt_token_ids` is a `std::vector<int32_t>` held BY VALUE (`include/vllm/v1/engine/types.h:79`), so each of the `n` children owns a full copy of the prompt and `Request::FromEngineCoreRequest` makes a second one per child — `O(n * prompt_len)` bytes moved before the first token is scheduled. Upstream copies ZERO prompt tokens: `copy(request)` (`vllm/v1/engine/async_llm.py:393`, `vllm/v1/engine/llm_engine.py:283` @ pin `5559679229`) is SHALLOW, every child references the same list object, and the last child reuses the parent outright. The comment on both of our lines claimed the copy "shares the prompt token ids", which is FALSE; [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s pull request corrects the comment and points here, and does NOT fix the cost, because the cheap mirror is a shared immutable token buffer on `EngineCoreRequest` that every engine path reads — a types-level change with its own review surface. No correctness effect; the cost scales with prompt length times `n`, so it is invisible on the short-prompt suites. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed` | bug |

## Resolution

-
