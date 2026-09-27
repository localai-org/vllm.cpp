ID: ISSUE-GH-2120
Title: **`/v1/chat/completions` streaming collapses `n > 1` onto one choice's parser and text state.** `ChatSseStream` (`src/vllm/entrypoints/openai/serving_chat.cpp:305-548`) holds `previous_text_`, `previous_num_tokens_` and `tools_streamed_` as scalars, ONE `parser_`/`engine_parser_`/`reasoning_parser_` instance for the whole response, emits the role frame for index 0 only (`:385`), and reads `response.outputs.front()`. Upstream keeps every one of those per choice index (`vllm/entrypoints/openai/chat_completion/serving.py:404-802` at pin `5559679229`). Repairing it is a parser-lifetime change with its own review surface, not a repair of the engine fan-out, so it is explicitly OUT of [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s scope. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2120
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:829`

### Frozen archive evidence

> | [#2120](https://github.com/mudler/vllm.cpp/issues/2120) | — | **`/v1/chat/completions` streaming collapses `n > 1` onto one choice's parser and text state.** `ChatSseStream` (`src/vllm/entrypoints/openai/serving_chat.cpp:305-548`) holds `previous_text_`, `previous_num_tokens_` and `tools_streamed_` as scalars, ONE `parser_`/`engine_parser_`/`reasoning_parser_` instance for the whole response, emits the role frame for index 0 only (`:385`), and reads `response.outputs.front()`. Upstream keeps every one of those per choice index (`vllm/entrypoints/openai/chat_completion/serving.py:404-802` at pin `5559679229`). Repairing it is a parser-lifetime change with its own review surface, not a repair of the engine fan-out, so it is explicitly OUT of [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s scope. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed` | bug |

## Resolution

-
