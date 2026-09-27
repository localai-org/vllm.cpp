ID: ISSUE-GH-1355
Title: Our server reports 5,942 prompt tokens where vLLM reports 6,144 for the IDENTICAL generated prompts. Found 2026-08-19 in the raw `vllm bench serve --save-detailed` files of the Qwen3.8-27B bf16 re-measure ([#915](https://github.com/mudler/vllm.cpp/issues/915)), both arms driven by the byte-identical client invocation from the same pinned wheel `0.1.dev1+g555967922`, same dataset, same seed. `input_lens` is the SERVER-reported length — `vllm/benchmarks/lib/endpoint_request_func.py:247` overwrites `output.prompt_len` from the streamed `usage.prompt_tokens` — and reads `[915, 931, 1024, 1024, 1024, 1024]` for us against `[1024] x 6` for vLLM at c1, with 19 of 48 short (877-941) at c8, byte-identical across all three reps of each leg. Not the client: `_align_prompts_to_server_tokenizer` (`vllm/benchmarks/serve.py:74,2041-2044`) re-aligns against the server's own `/tokenize` and prints `WARNING: tokenizer mismatch` when it disagrees, and NEITHER arm printed it, so our `/tokenize` agreed on 1024 while our `usage.prompt_tokens` reported 915 for the same request. `output_lens` is `[128]xN` on both arms in every leg, so the campaign's output-throughput, TPOT and ITL figures are unaffected; `total_token_throughput` is affected, our c8 196.10 tok/s being computed over 47,072 input tokens where the intended workload is 49,152. TWO causes and the artifacts cannot separate them: under-reported usage, or a genuinely truncated prompt — and the second would mean the two arms did not run the same workload. A greedy token gate cannot see either, which is why it survived the gate on this checkpoint. NOT fixed in flow: the finding row writes no product code and holds no GPU. Owed under `## Owed` in [qwen38-27b-bf16-gate.md](../../specs/qwen38-27b-bf16-gate.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1355
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:448`

### Frozen archive evidence

> | [#1355](https://github.com/mudler/vllm.cpp/issues/1355) | — | Our server reports 5,942 prompt tokens where vLLM reports 6,144 for the IDENTICAL generated prompts. Found 2026-08-19 in the raw `vllm bench serve --save-detailed` files of the Qwen3.8-27B bf16 re-measure ([#915](https://github.com/mudler/vllm.cpp/issues/915)), both arms driven by the byte-identical client invocation from the same pinned wheel `0.1.dev1+g555967922`, same dataset, same seed. `input_lens` is the SERVER-reported length — `vllm/benchmarks/lib/endpoint_request_func.py:247` overwrites `output.prompt_len` from the streamed `usage.prompt_tokens` — and reads `[915, 931, 1024, 1024, 1024, 1024]` for us against `[1024] x 6` for vLLM at c1, with 19 of 48 short (877-941) at c8, byte-identical across all three reps of each leg. Not the client: `_align_prompts_to_server_tokenizer` (`vllm/benchmarks/serve.py:74,2041-2044`) re-aligns against the server's own `/tokenize` and prints `WARNING: tokenizer mismatch` when it disagrees, and NEITHER arm printed it, so our `/tokenize` agreed on 1024 while our `usage.prompt_tokens` reported 915 for the same request. `output_lens` is `[128]xN` on both arms in every leg, so the campaign's output-throughput, TPOT and ITL figures are unaffected; `total_token_throughput` is affected, our c8 196.10 tok/s being computed over 47,072 input tokens where the intended workload is 49,152. TWO causes and the artifacts cannot separate them: under-reported usage, or a genuinely truncated prompt — and the second would mean the two arms did not run the same workload. A greedy token gate cannot see either, which is why it survived the gate on this checkpoint. NOT fixed in flow: the finding row writes no product code and holds no GPU. Owed under `## Owed` in [qwen38-27b-bf16-gate.md](../../specs/qwen38-27b-bf16-gate.md) | bug |

## Resolution

-
