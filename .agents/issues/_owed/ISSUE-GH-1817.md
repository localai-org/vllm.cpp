ID: ISSUE-GH-1817
Title: **The two `ClampPromptLogprobs` call sites are reached but not measured: deleting one keeps the focused gate green.** Measured while landing [#1815](https://github.com/mudler/vllm.cpp/issues/1815) as mutation M6 -- removing `ClampPromptLogprobs(prompt_logprobs);` from `serving_completion.cpp` compiles (`compile_rc=0`, deletion confirmed by `git diff --stat`) and leaves `test_openai_api_server` at 70/70 and `test_openai_serving` at 59/59. The FUNCTION is gated directly (`test_protocol.cpp`, "ClampPromptLogprobs rewrites -inf to -9999.0 in place"); the call sites are not, so a refactor could drop one silently. The cause is the fixture rather than the test: prompt logprobs come from raw prompt logits with no sampling mask applied, and a `log_softmax` over finite float32 logits does not underflow to `-inf`, so nothing in the CPU tier produces the value the clamp exists for. Repair is a seam that lets a test hand `OpenAIServingCompletion` a `RequestOutput` carrying `-inf`, or a fixture whose prompt logits contain one -- NOT a widened assertion, because a green-on-deletion gate is the defect. Owed under [prompt-logprobs.md](../../specs/prompt-logprobs.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1817
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:671`

### Frozen archive evidence

> | [#1817](https://github.com/mudler/vllm.cpp/issues/1817) | — | **The two `ClampPromptLogprobs` call sites are reached but not measured: deleting one keeps the focused gate green.** Measured while landing [#1815](https://github.com/mudler/vllm.cpp/issues/1815) as mutation M6 -- removing `ClampPromptLogprobs(prompt_logprobs);` from `serving_completion.cpp` compiles (`compile_rc=0`, deletion confirmed by `git diff --stat`) and leaves `test_openai_api_server` at 70/70 and `test_openai_serving` at 59/59. The FUNCTION is gated directly (`test_protocol.cpp`, "ClampPromptLogprobs rewrites -inf to -9999.0 in place"); the call sites are not, so a refactor could drop one silently. The cause is the fixture rather than the test: prompt logprobs come from raw prompt logits with no sampling mask applied, and a `log_softmax` over finite float32 logits does not underflow to `-inf`, so nothing in the CPU tier produces the value the clamp exists for. Repair is a seam that lets a test hand `OpenAIServingCompletion` a `RequestOutput` carrying `-inf`, or a fixture whose prompt logits contain one -- NOT a widened assertion, because a green-on-deletion gate is the defect. Owed under [prompt-logprobs.md](../../specs/prompt-logprobs.md) `## Owed` | bug |

## Resolution

-
