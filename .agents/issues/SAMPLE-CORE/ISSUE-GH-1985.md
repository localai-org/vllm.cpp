ID: ISSUE-GH-1985
Title: `generation_config.json` is read for `eos_token_id` only (`hf_config.cpp::ReadGenerationConfigEosIds`), so `Qwen/Qwen3.8-27B`'s shipped `top_k: 20` / `top_p: 0.95` never reach `SamplingParams` and `to_sampling_params` resolves omitted knobs straight to the neutral OpenAI defaults, which disable both filters. vLLM applies them through `ModelConfig.get_diff_sampling_param` -> `OpenAIServing*.default_sampling_params` -> `to_sampling_params`. Since `vllm bench serve` stopped sending `--temperature`, both engines sample at temperature 1.0 and vLLM draws from 20 candidates while we draw from 248,320: different sampling on two sides of a parity benchmark. Spec: [sample-gen-config-and-parallel-gumbel.md](../../specs/sample-gen-config-and-parallel-gumbel.md)
Row: SAMPLE-CORE
State: UNKNOWN
Kind: bug
GitHub: 1985
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:783`

### Frozen archive evidence

> | [#1985](https://github.com/mudler/vllm.cpp/issues/1985) | `SAMPLE-CORE` | `generation_config.json` is read for `eos_token_id` only (`hf_config.cpp::ReadGenerationConfigEosIds`), so `Qwen/Qwen3.8-27B`'s shipped `top_k: 20` / `top_p: 0.95` never reach `SamplingParams` and `to_sampling_params` resolves omitted knobs straight to the neutral OpenAI defaults, which disable both filters. vLLM applies them through `ModelConfig.get_diff_sampling_param` -> `OpenAIServing*.default_sampling_params` -> `to_sampling_params`. Since `vllm bench serve` stopped sending `--temperature`, both engines sample at temperature 1.0 and vLLM draws from 20 candidates while we draw from 248,320: different sampling on two sides of a parity benchmark. Spec: [sample-gen-config-and-parallel-gumbel.md](../../specs/sample-gen-config-and-parallel-gumbel.md) | bug |

## Resolution

-
