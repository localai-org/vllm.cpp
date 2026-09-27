ID: ISSUE-GH-1857
Title: **The q>1 DFlash2 verify rides the PREFILL flash lane (`is_prefill = num_tokens > num_reqs`, `fa2_decode` hard-requires `num_tokens == num_reqs`), costing +9 ms/step from q=2 to q=9 on the #1574 K-ladder — the last attributed gap against SGLang (27.60 vs 25.07 tok/s at equal acceptance, ~109 vs ~122 ms/step, both paying the same ~20 ms draft).** W10 mirrors upstream's spec-as-decode: the reorder-threshold policy `1 + (parallel_drafting ? 2 : 1) * K` (`backend.py:718-736` @ `b389ac2946`, identical at the pin) classifies the runner's already-verified uniform verify length onto the decode class, the classification travels `CommonAttentionMetadata -> PagedAttentionArgs`, and a new ADDITIVE d256 launcher serves it with the exact presentation upstream `mha_fwd_kvcache` uses at seqlen_q>1 — batched split-KV, bottom-right causal against `seqused_k` (the draft mask with no new mask code), `set_params_splitkv` heuristic. The shipped q==1 arms and every unclassified batch stay dispatch-identical; `VT_FA2_SPEC_DECODE=0` restores the prefill route for a same-binary A/B. Wave spec [dflash2-spec-as-decode.md](../../specs/dflash2-spec-as-decode.md); the GPU step-time delta (the −8-9 ms claim), the GPU token gates and the first CUDA compile are owed there, operator-run
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1857
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:697`

### Frozen archive evidence

> | [#1857](https://github.com/mudler/vllm.cpp/issues/1857) | `SPEC-DFLASH2` | **The q>1 DFlash2 verify rides the PREFILL flash lane (`is_prefill = num_tokens > num_reqs`, `fa2_decode` hard-requires `num_tokens == num_reqs`), costing +9 ms/step from q=2 to q=9 on the #1574 K-ladder — the last attributed gap against SGLang (27.60 vs 25.07 tok/s at equal acceptance, ~109 vs ~122 ms/step, both paying the same ~20 ms draft).** W10 mirrors upstream's spec-as-decode: the reorder-threshold policy `1 + (parallel_drafting ? 2 : 1) * K` (`backend.py:718-736` @ `b389ac2946`, identical at the pin) classifies the runner's already-verified uniform verify length onto the decode class, the classification travels `CommonAttentionMetadata -> PagedAttentionArgs`, and a new ADDITIVE d256 launcher serves it with the exact presentation upstream `mha_fwd_kvcache` uses at seqlen_q>1 — batched split-KV, bottom-right causal against `seqused_k` (the draft mask with no new mask code), `set_params_splitkv` heuristic. The shipped q==1 arms and every unclassified batch stay dispatch-identical; `VT_FA2_SPEC_DECODE=0` restores the prefill route for a same-binary A/B. Wave spec [dflash2-spec-as-decode.md](../../specs/dflash2-spec-as-decode.md); the GPU step-time delta (the −8-9 ms claim), the GPU token gates and the first CUDA compile are owed there, operator-run | perf |

## Resolution

-
