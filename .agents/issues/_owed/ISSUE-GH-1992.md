ID: ISSUE-GH-1992
Title: **Neither `ChatSseStream::next` nor `CompletionSseStream::next` converts an engine exception into a `data: {"error": …}` frame, so a streaming request that fails is a truncated 200 and the cause reaches only `stderr`.** Upstream yields the error frame and then `data: [DONE]` from the generator's `except GenerationError` / `except Exception` arms (`vllm/entrypoints/openai/chat_completion/serving.py:827-833` at the pin `555967922`), and that frame is what makes the first-iteration ordering at `:484-486` mean anything: the role chunk is built inside the loop so an exception can be the FIRST response, which needs a response to exist. Ours propagates out of `next()` into the cpp-httplib chunked content provider (`src/vllm/entrypoints/openai/api_server.cpp::ApiServer::register_routes`), which logs `sse: stream aborted mid-flight:` and aborts, so a client cannot tell a failed request from a short one. Found while fixing [#1982](https://github.com/mudler/vllm.cpp/issues/1982) and NOT fixed in that flow: upstream's `try` wraps the whole generator, so the frame is owed for mid-stream failures on both endpoints, and that is a different blast radius needing its own red-first cases for the payload shape, the trailing `[DONE]` and the separate `GenerationError` converter. Owed by [`specs/chat-role-frame-ordering.md`](../../specs/chat-role-frame-ordering.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1992
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:762`

### Frozen archive evidence

> | [#1992](https://github.com/mudler/vllm.cpp/issues/1992) | — | **Neither `ChatSseStream::next` nor `CompletionSseStream::next` converts an engine exception into a `data: {"error": …}` frame, so a streaming request that fails is a truncated 200 and the cause reaches only `stderr`.** Upstream yields the error frame and then `data: [DONE]` from the generator's `except GenerationError` / `except Exception` arms (`vllm/entrypoints/openai/chat_completion/serving.py:827-833` at the pin `555967922`), and that frame is what makes the first-iteration ordering at `:484-486` mean anything: the role chunk is built inside the loop so an exception can be the FIRST response, which needs a response to exist. Ours propagates out of `next()` into the cpp-httplib chunked content provider (`src/vllm/entrypoints/openai/api_server.cpp::ApiServer::register_routes`), which logs `sse: stream aborted mid-flight:` and aborts, so a client cannot tell a failed request from a short one. Found while fixing [#1982](https://github.com/mudler/vllm.cpp/issues/1982) and NOT fixed in that flow: upstream's `try` wraps the whole generator, so the frame is owed for mid-stream failures on both endpoints, and that is a different blast radius needing its own red-first cases for the payload shape, the trailing `[DONE]` and the separate `GenerationError` converter. Owed by [`specs/chat-role-frame-ordering.md`](../../specs/chat-role-frame-ordering.md) `## Owed` | bug |

## Resolution

-
