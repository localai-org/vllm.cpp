ID: ISSUE-GH-2150
Title: **`ParentRequest::get_outputs` indexes `output_aggregator_` unchecked and dereferences a possibly-empty optional; upstream raises `IndexError` where we get UB.** `src/vllm/v1/engine/parallel_sampling.cpp:75-82` writes `output_aggregator_[idx] = ...` and later drains with `*slot`, both 1:1 with `vllm/v1/engine/parallel_sampling.py:100-126` @ pin `5559679229`, which is equally unchecked — but upstream's aggregator is a Python list, so an out-of-range index raises and an unfilled slot surfaces as `None`. Ours is `std::vector<std::optional<CompletionOutput>>`, where both are undefined behaviour. NOT a live bug: `idx` is `0..n-1` by construction from `get_child_info` and the vector is sized `n`, so a bounds check today would be DEAD CODE and is deliberately not added. Recorded because this UB is why [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s `request_index=0` mutation has no stable exit status (RC=135/139 full, RC=1 case-scoped across rounds) and had to be recorded as a signal rather than a number; a debug-configuration assertion or `.at()` is the likely shape of a fix, not a release-path branch. Found in the fresh review of #1816. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2150
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:832`

### Frozen archive evidence

> | [#2150](https://github.com/mudler/vllm.cpp/issues/2150) | — | **`ParentRequest::get_outputs` indexes `output_aggregator_` unchecked and dereferences a possibly-empty optional; upstream raises `IndexError` where we get UB.** `src/vllm/v1/engine/parallel_sampling.cpp:75-82` writes `output_aggregator_[idx] = ...` and later drains with `*slot`, both 1:1 with `vllm/v1/engine/parallel_sampling.py:100-126` @ pin `5559679229`, which is equally unchecked — but upstream's aggregator is a Python list, so an out-of-range index raises and an unfilled slot surfaces as `None`. Ours is `std::vector<std::optional<CompletionOutput>>`, where both are undefined behaviour. NOT a live bug: `idx` is `0..n-1` by construction from `get_child_info` and the vector is sized `n`, so a bounds check today would be DEAD CODE and is deliberately not added. Recorded because this UB is why [#1816](https://github.com/mudler/vllm.cpp/issues/1816)'s `request_index=0` mutation has no stable exit status (RC=135/139 full, RC=1 case-scoped across rounds) and had to be recorded as a signal rather than a number; a debug-configuration assertion or `.at()` is the likely shape of a fix, not a release-path branch. Found in the fresh review of #1816. Owed under [async-parallel-sampling.md](../../specs/async-parallel-sampling.md) `## Owed` | bug |

## Resolution

-
