ID: ISSUE-GH-1681
Title: `POST /v1/chat/completions` answers HTTP 500 for the whole Qwen3.8 family because the vendored minja Jinja engine implements twelve of Jinja2's built-in tests and `undefined` is not one of them, so `{%- if enable_thinking is undefined or enable_thinking is true %}` throws at row 46 of the checkpoint's own template. `is true` was already present, so the first term was the only break. Fixed in flow together with the second half of the same defect: `enable_thinking` was set unconditionally by `apply_chat_template`, so even with `undefined` implemented the variable could never be undefined and the Qwen3.8 default would have been thinking-OFF against upstream's thinking-ON, and `ChatCompletionRequest` carried no `chat_template_kwargs` at all, so the `{"chat_template_kwargs":{"enable_thinking":false}}` body both competitor arms of [#1574](https://github.com/mudler/vllm.cpp/issues/1574) were measured with was silently ignored. Spec [`chat-template-jinja-undefined.md`](../../specs/chat-template-jinja-undefined.md), whose `## Owed` carries the twelve of Jinja2's thirty canonical built-in tests that stay unimplemented, each with the reason: nine need a grammar change because minja parses the right side of `is` as a bare identifier, `filter` and `test` need a name registry minja does not have, and `callable` can never be handed a callable because `BinaryOpExpr::do_evaluate` defers every binary operation whose left operand is one. No chat template of any checkpoint in `docs/USAGE.md` uses any of the twelve
Row: SERVE-CHAT-TEMPLATE
State: UNKNOWN
Kind: bug
GitHub: 1681
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:622`

### Frozen archive evidence

> | [#1681](https://github.com/mudler/vllm.cpp/issues/1681) | `SERVE-CHAT-TEMPLATE` | `POST /v1/chat/completions` answers HTTP 500 for the whole Qwen3.8 family because the vendored minja Jinja engine implements twelve of Jinja2's built-in tests and `undefined` is not one of them, so `{%- if enable_thinking is undefined or enable_thinking is true %}` throws at row 46 of the checkpoint's own template. `is true` was already present, so the first term was the only break. Fixed in flow together with the second half of the same defect: `enable_thinking` was set unconditionally by `apply_chat_template`, so even with `undefined` implemented the variable could never be undefined and the Qwen3.8 default would have been thinking-OFF against upstream's thinking-ON, and `ChatCompletionRequest` carried no `chat_template_kwargs` at all, so the `{"chat_template_kwargs":{"enable_thinking":false}}` body both competitor arms of [#1574](https://github.com/mudler/vllm.cpp/issues/1574) were measured with was silently ignored. Spec [`chat-template-jinja-undefined.md`](../../specs/chat-template-jinja-undefined.md), whose `## Owed` carries the twelve of Jinja2's thirty canonical built-in tests that stay unimplemented, each with the reason: nine need a grammar change because minja parses the right side of `is` as a bare identifier, `filter` and `test` need a name registry minja does not have, and `callable` can never be handed a callable because `BinaryOpExpr::do_evaluate` defers every binary operation whose left operand is one. No chat template of any checkpoint in `docs/USAGE.md` uses any of the twelve | bug |

## Resolution

-
