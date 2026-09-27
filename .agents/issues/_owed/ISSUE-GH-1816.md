ID: ISSUE-GH-1816
Title: **`AsyncLLM` -- the engine every OpenAI HTTP route runs on -- never fans a request out into `n` children, so every `n > 1` request to the production server is silently served as `n = 1`.** `LLMEngine` does fan out (`llm_engine.cpp:151` `FanOutParallelSampling`, 1:1 `llm_engine.py:280-291`: a shared `ParentRequest`, `n` children named `{idx}_{parent}` with `n == 1` params and `seed + idx`, aggregated back into one `RequestOutput`), and all three `AsyncLLM::add_request` overloads instead pass `request_index=0, parent=nullptr` at `async_llm.cpp:79,120,278`. Measured over a real socket on `bacb71109`: `{"n":2,"temperature":1.0,"seed":7}` returns ONE choice and `completion_tokens` counts one sequence; the identical body without `prompt_logprobs` behaves the same, so the fan-out is the variable. The gap stayed invisible because the covered engine is not the served one -- `test_serving.cpp` ("serving_completion: n>1 returns n indexed, deterministic choices") gates the property over the SYNC engine. `best_of` is affected too: `serving_completion.cpp` asks the engine for `best_of` children and ranks them with `SelectBestOf`, and there is nothing to rank. Found while landing [#1815](https://github.com/mudler/vllm.cpp/issues/1815) and NOT fixed there -- porting the fan-out onto `AsyncLLM` touches the abort path and the streaming `RequestOutputKind` handling, so it needs its own row, spec and fresh review rather than an in-flow repair. Owed under [prompt-logprobs.md](../../specs/prompt-logprobs.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1816
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:670`

### Frozen archive evidence

> | [#1816](https://github.com/mudler/vllm.cpp/issues/1816) | — | **`AsyncLLM` -- the engine every OpenAI HTTP route runs on -- never fans a request out into `n` children, so every `n > 1` request to the production server is silently served as `n = 1`.** `LLMEngine` does fan out (`llm_engine.cpp:151` `FanOutParallelSampling`, 1:1 `llm_engine.py:280-291`: a shared `ParentRequest`, `n` children named `{idx}_{parent}` with `n == 1` params and `seed + idx`, aggregated back into one `RequestOutput`), and all three `AsyncLLM::add_request` overloads instead pass `request_index=0, parent=nullptr` at `async_llm.cpp:79,120,278`. Measured over a real socket on `bacb71109`: `{"n":2,"temperature":1.0,"seed":7}` returns ONE choice and `completion_tokens` counts one sequence; the identical body without `prompt_logprobs` behaves the same, so the fan-out is the variable. The gap stayed invisible because the covered engine is not the served one -- `test_serving.cpp` ("serving_completion: n>1 returns n indexed, deterministic choices") gates the property over the SYNC engine. `best_of` is affected too: `serving_completion.cpp` asks the engine for `best_of` children and ranks them with `SelectBestOf`, and there is nothing to rank. Found while landing [#1815](https://github.com/mudler/vllm.cpp/issues/1815) and NOT fixed there -- porting the fan-out onto `AsyncLLM` touches the abort path and the streaming `RequestOutputKind` handling, so it needs its own row, spec and fresh review rather than an in-flow repair. Owed under [prompt-logprobs.md](../../specs/prompt-logprobs.md) `## Owed` | bug |

## Resolution

-
