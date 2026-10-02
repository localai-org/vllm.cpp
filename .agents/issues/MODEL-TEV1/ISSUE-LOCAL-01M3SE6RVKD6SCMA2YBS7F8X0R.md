ID: ISSUE-LOCAL-01M3SE6RVKD6SCMA2YBS7F8X0R
Title: Async scheduling drops sample logprobs: AsyncLLM returns empty logprobs for every request
Row: MODEL-TEV1
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

With async scheduling on (the default on CPU and CUDA whenever the runner supports it), GPUModelRunner::sample_tokens_async calls sampler_.forward and discards its SamplerOutput, so ModelRunnerOutput.logprobs is never set on the async path. The scheduler's slice gate (scheduler.cpp:1353) then finds no logprobs, and every AsyncLLM request that asks for logprobs or logprob_token_ids finishes with an EMPTY SampleLogprobs. Measured on a synthetic Qwen3.5 dense LoadedEngine, CPU, 2026-09-30: logprobs=2 with max_tokens 1, 2, 3 returns 0 positions each; VT_ASYNC_SCHED=0 returns 1, 2, 3. Every OpenAI route runs on AsyncLLM, so /v1/chat/completions and /v1/completions logprobs are empty by default, and generative scoring (the Tev1 SystemOne lane) cannot read candidate logprobs. Upstream AsyncGPUModelRunnerOutput carries the sampler's logprobs_tensors and get_output returns them (gpu_model_runner.py:264-295, 320-325 @ 5559679229).

## Resolution

2026-09-30: GPUModelRunner's async sampling path now keeps the sampler's logprobs_tensors on the async output (skeleton.logprobs), as upstream AsyncGPUModelRunnerOutput does. Red then green: test_loaded_engine_dense 'async scheduling returns sample and explicit-id logprobs' failed (0 positions for 1 token) before and passes after. Once logprobs reached the server, a second defect surfaced: a top-k token that is part of a UTF-8 character made json::dump throw, and /v1/chat/completions with top_logprobs answered HTTP 500 with an empty body (measured on Tev1-0.8B, CPU, max_tokens 3 and 4). DecodedToken now applies SanitizeUtf8, which is upstream's tokenizer.decode(errors=replace); test_openai_logprobs 'a partial UTF-8 token serializes as U+FFFD' red then green. The pre-fix server behavior was inferred from the AsyncLLM measurement, not measured over HTTP. Suites green after: test_loaded_engine_dense 31/31, test_openai_logprobs 8/8, test_openai_serving 48/48, test_openai_api_server 103/103, test_runner 41/41, test_async_* all green, test_sampler, test_llm_engine.
