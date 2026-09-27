ID: ISSUE-GH-2119
Title: **`/v1/completions` streaming drops every choice past the first**: `CompletionSseStream::next` (`src/vllm/entrypoints/openai/serving_completion.cpp:92`) reads `response.outputs.front()` and formats one SSE choice from it, while `RequestOutputCollector::Merge` (`src/vllm/v1/engine/output_processor.cpp:96-118`) keeps distinct `index` completions side by side in ONE frame whenever the producer outruns the consumer. Upstream flattens the samples one choice per chunk and asserts it (`tests/entrypoints/openai/completion/test_completion.py:419-424`, `test_parallel_streaming` at pin `5559679229`); the SYNC path in the same file already loops all outputs (`serving_completion.cpp:286`), so the defect is specific to the async SSE source. Latent until [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s fan-out makes a second output reachable, so FIXED in that row's pull request. Spec: [async-parallel-sampling.md](../../specs/async-parallel-sampling.md)
Row: SAMPLE-N-ASYNC
State: UNKNOWN
Kind: bug
GitHub: 2119
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:828`

### Frozen archive evidence

> | [#2119](https://github.com/mudler/vllm.cpp/issues/2119) | `SAMPLE-N-ASYNC` | **`/v1/completions` streaming drops every choice past the first**: `CompletionSseStream::next` (`src/vllm/entrypoints/openai/serving_completion.cpp:92`) reads `response.outputs.front()` and formats one SSE choice from it, while `RequestOutputCollector::Merge` (`src/vllm/v1/engine/output_processor.cpp:96-118`) keeps distinct `index` completions side by side in ONE frame whenever the producer outruns the consumer. Upstream flattens the samples one choice per chunk and asserts it (`tests/entrypoints/openai/completion/test_completion.py:419-424`, `test_parallel_streaming` at pin `5559679229`); the SYNC path in the same file already loops all outputs (`serving_completion.cpp:286`), so the defect is specific to the async SSE source. Latent until [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s fan-out makes a second output reachable, so FIXED in that row's pull request. Spec: [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) | bug |

## Resolution

-
