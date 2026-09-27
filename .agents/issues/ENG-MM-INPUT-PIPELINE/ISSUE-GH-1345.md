ID: ISSUE-GH-1345
Title: [#414](https://github.com/mudler/vllm.cpp/issues/414) on a second surface: three in-process bench harnesses construct the vLLM oracle with `language_model_only` left at its `False` default and expose no way to set it -- `tools/bench/profile_vllm_online_gate.py:209` (the online gate's torch-profiler arm), `tools/bench/vllm_closed_loop_metrics.py:160` (the reference companion to `examples/bench/bench_core.h`) and `tools/bench/dump_vllm_tokens.py:34` (greedy oracle token capture). `EngineArgs.language_model_only` is an ordinary `LLM(...)` keyword (`vllm/engine/arg_utils.py:555,1691` at the pin), so pointed at a Qwen3.6 checkpoint each gets `text_only == False` and the UNFUSED QK-norm+RoPE+gate path against our fused one. For the two measurement harnesses that is the flattered denominator; for the token dump the direction is not automatic, because the two arms are not bit-identical and a golden captured on one can sit on the far side of a near-tie from the arm it gates. Deliberately OUTSIDE `scripts/check-oracle-denominator-flags.py`, which [#607](https://github.com/mudler/vllm.cpp/issues/607) wave L4 landed: all three take `--model` as a path, so no static rule can know whether a run points at a multimodal checkpoint, and a checker that cannot decide its own question is worse than none. Fix is to thread the knob through and record the RESOLVED value beside the measurement; the default is a denominator decision for the operator. NOT fixed in flow because it changes what three harnesses hand the oracle and the L4 wave could take no measurement to exercise it once. Also under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.6
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 1345
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:438`

### Frozen archive evidence

> | [#1345](https://github.com/mudler/vllm.cpp/issues/1345) | `ENG-MM-INPUT-PIPELINE` | [#414](https://github.com/mudler/vllm.cpp/issues/414) on a second surface: three in-process bench harnesses construct the vLLM oracle with `language_model_only` left at its `False` default and expose no way to set it -- `tools/bench/profile_vllm_online_gate.py:209` (the online gate's torch-profiler arm), `tools/bench/vllm_closed_loop_metrics.py:160` (the reference companion to `examples/bench/bench_core.h`) and `tools/bench/dump_vllm_tokens.py:34` (greedy oracle token capture). `EngineArgs.language_model_only` is an ordinary `LLM(...)` keyword (`vllm/engine/arg_utils.py:555,1691` at the pin), so pointed at a Qwen3.6 checkpoint each gets `text_only == False` and the UNFUSED QK-norm+RoPE+gate path against our fused one. For the two measurement harnesses that is the flattered denominator; for the token dump the direction is not automatic, because the two arms are not bit-identical and a golden captured on one can sit on the far side of a near-tie from the arm it gates. Deliberately OUTSIDE `scripts/check-oracle-denominator-flags.py`, which [#607](https://github.com/mudler/vllm.cpp/issues/607) wave L4 landed: all three take `--model` as a path, so no static rule can know whether a run points at a multimodal checkpoint, and a checker that cannot decide its own question is worse than none. Fix is to thread the knob through and record the RESOLVED value beside the measurement; the default is a denominator decision for the operator. NOT fixed in flow because it changes what three harnesses hand the oracle and the L4 wave could take no measurement to exercise it once. Also under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.6 | bug |

## Resolution

-
