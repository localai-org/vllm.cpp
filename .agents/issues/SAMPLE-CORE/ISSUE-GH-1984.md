ID: ISSUE-GH-1984
Title: `RandomSampleKernel` is launched `<<<n, 1>>>` and scans a 248,320-wide vocab on one thread per row, computing two `SplitMix64` rounds and an f64 `log` per element. Eleven lines above it the same file records that a single-block single-thread scan of a ~151k vocab cost ~7.5 ms/token, which is why greedy argmax was rewritten into `ArgmaxPartialKernel`/`ArgmaxFinalKernel`; the Gumbel draw never got that treatment. Upstream is whole-tensor (`vllm/v1/sample/ops/topk_topp_sampler.py::sample_with_exponential_noise`), so this is a mirror obligation. Reached by every non-greedy row through `ModelRunner::execute_model` -> `Sampler::forward` -> `vt::RandomSample`. Spec: [sample-gen-config-and-parallel-gumbel.md](../../specs/sample-gen-config-and-parallel-gumbel.md)
Row: SAMPLE-CORE
State: UNKNOWN
Kind: perf
GitHub: 1984
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:782`

### Frozen archive evidence

> | [#1984](https://github.com/mudler/vllm.cpp/issues/1984) | `SAMPLE-CORE` | `RandomSampleKernel` is launched `<<<n, 1>>>` and scans a 248,320-wide vocab on one thread per row, computing two `SplitMix64` rounds and an f64 `log` per element. Eleven lines above it the same file records that a single-block single-thread scan of a ~151k vocab cost ~7.5 ms/token, which is why greedy argmax was rewritten into `ArgmaxPartialKernel`/`ArgmaxFinalKernel`; the Gumbel draw never got that treatment. Upstream is whole-tensor (`vllm/v1/sample/ops/topk_topp_sampler.py::sample_with_exponential_noise`), so this is a mirror obligation. Reached by every non-greedy row through `ModelRunner::execute_model` -> `Sampler::forward` -> `vt::RandomSample`. Spec: [sample-gen-config-and-parallel-gumbel.md](../../specs/sample-gen-config-and-parallel-gumbel.md) | perf |

## Resolution

-
