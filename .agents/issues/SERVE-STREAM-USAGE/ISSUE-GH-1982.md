ID: ISSUE-GH-1982
Title: **`ChatSseStream::next` writes the `/v1/chat/completions` role frame before it reads anything from the engine, so `vllm bench serve --backend openai-chat` stamps TTFT on an empty frame and our TTFT through that harness is an HTTP round trip, not a time to first token.** Upstream builds the role chunk under `if first_iteration:` inside `async for res in result_generator:` (`vllm/entrypoints/openai/chat_completion/serving.py:477,487`) and says why at `:484-486`: an exception in the generator "needs to be sent as the FIRST response". `vllm/benchmarks/lib/endpoint_request_func.py:404-408` guards on the presence of `choices`, not on non-empty `delta.content`, and our role frame carries `delta.content = ""` with no `usage`. vLLM and SGLang order the frame after the first result, so their rows on the same harness are honest and only ours is not; this blocks the #1574 three-engine TTFT row. `.agents/specs/stream-options.md` scoped the buffering to continuous usage on purpose and both its passages are corrected here. Fixed by removing the `usage_.include_continuous_usage` guard around the first-result buffering loop, so the default path buffers too. Spec: [`specs/chat-role-frame-ordering.md`](../../specs/chat-role-frame-ordering.md)
Row: SERVE-STREAM-USAGE
State: UNKNOWN
Kind: bug
GitHub: 1982
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:761`

### Frozen archive evidence

> | [#1982](https://github.com/mudler/vllm.cpp/issues/1982) | `SERVE-STREAM-USAGE` | **`ChatSseStream::next` writes the `/v1/chat/completions` role frame before it reads anything from the engine, so `vllm bench serve --backend openai-chat` stamps TTFT on an empty frame and our TTFT through that harness is an HTTP round trip, not a time to first token.** Upstream builds the role chunk under `if first_iteration:` inside `async for res in result_generator:` (`vllm/entrypoints/openai/chat_completion/serving.py:477,487`) and says why at `:484-486`: an exception in the generator "needs to be sent as the FIRST response". `vllm/benchmarks/lib/endpoint_request_func.py:404-408` guards on the presence of `choices`, not on non-empty `delta.content`, and our role frame carries `delta.content = ""` with no `usage`. vLLM and SGLang order the frame after the first result, so their rows on the same harness are honest and only ours is not; this blocks the #1574 three-engine TTFT row. `.agents/specs/stream-options.md` scoped the buffering to continuous usage on purpose and both its passages are corrected here. Fixed by removing the `usage_.include_continuous_usage` guard around the first-result buffering loop, so the default path buffers too. Spec: [`specs/chat-role-frame-ordering.md`](../../specs/chat-role-frame-ordering.md) | bug |

## Resolution

-
