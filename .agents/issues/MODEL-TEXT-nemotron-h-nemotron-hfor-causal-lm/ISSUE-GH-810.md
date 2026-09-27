ID: ISSUE-GH-810
Title: NemotronH is not reachable through `include/vllm.h`: `GPUModelRunner::initialize_kv_cache` rebuilds the RECURRENT half of the allocation from `config_.linear_*` instead of the `MambaSpec` the model published, so every non-Qwen3.5 hybrid is refused by Qwen3.5's name at `runner.cpp:525`, and per-layer membership comes from `config_.layer_types[l] == "linear_attention"` rather than `KVCacheGroupSpec::layer_names`. Spec [`nemotron-h-abi-e2e.md`](../../specs/nemotron-h-abi-e2e.md); the attention half at `runner.cpp:539-607` is already spec-driven and is the model to mirror. Note the SAFETY constraint recorded there: neutering the check alone reaches a forward that ignores `attn_kv`/`gdn_state`/`num_reqs`, which is strictly more dangerous than the refusal
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 810
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:255`

### Frozen archive evidence

> | [#810](https://github.com/mudler/vllm.cpp/issues/810) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | NemotronH is not reachable through `include/vllm.h`: `GPUModelRunner::initialize_kv_cache` rebuilds the RECURRENT half of the allocation from `config_.linear_*` instead of the `MambaSpec` the model published, so every non-Qwen3.5 hybrid is refused by Qwen3.5's name at `runner.cpp:525`, and per-layer membership comes from `config_.layer_types[l] == "linear_attention"` rather than `KVCacheGroupSpec::layer_names`. Spec [`nemotron-h-abi-e2e.md`](../../specs/nemotron-h-abi-e2e.md); the attention half at `runner.cpp:539-607` is already spec-driven and is the model to mirror. Note the SAFETY constraint recorded there: neutering the check alone reaches a forward that ignores `attn_kv`/`gdn_state`/`num_reqs`, which is strictly more dangerous than the refusal | bug |

## Resolution

-
