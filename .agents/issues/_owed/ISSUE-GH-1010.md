ID: ISSUE-GH-1010
Title: An LTX-2.5 render emits **one** engine log line in 2.5 hours: `~/work/ltx25-e2e/ladder-075434/512x320_121f/run.log` on `dgx.casa` is 16 lines for a 2h29m run, 13 of them the CUDA container banner. Attributing 89% of a render's wall to a single-threaded phase ([#1009](https://github.com/mudler/vllm.cpp/issues/1009)) therefore required reading an external memory sampler's load-average column and correlating it with file mtimes — and neither that log nor the sampler's rows can be reached today ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)). Per-phase wall and peak memory are owed from the render path; the streaming chunk callback (`src/vllm/multimodal/ltx2_video.cpp:3262-3283 @ 332aed738`) already carries `chunk.first_frame` and is the natural site. No speedup — it is the precondition for measuring [#1007](https://github.com/mudler/vllm.cpp/issues/1007) and [#1009](https://github.com/mudler/vllm.cpp/issues/1009), and its absence is why the landed [#1008](https://github.com/mudler/vllm.cpp/issues/1008) has no speed number. It should land first. Lever 6. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 1010
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:287`

### Frozen archive evidence

> | [#1010](https://github.com/mudler/vllm.cpp/issues/1010) | — | An LTX-2.5 render emits **one** engine log line in 2.5 hours: `~/work/ltx25-e2e/ladder-075434/512x320_121f/run.log` on `dgx.casa` is 16 lines for a 2h29m run, 13 of them the CUDA container banner. Attributing 89% of a render's wall to a single-threaded phase ([#1009](https://github.com/mudler/vllm.cpp/issues/1009)) therefore required reading an external memory sampler's load-average column and correlating it with file mtimes — and neither that log nor the sampler's rows can be reached today ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)). Per-phase wall and peak memory are owed from the render path; the streaming chunk callback (`src/vllm/multimodal/ltx2_video.cpp:3262-3283 @ 332aed738`) already carries `chunk.first_frame` and is the natural site. No speedup — it is the precondition for measuring [#1007](https://github.com/mudler/vllm.cpp/issues/1007) and [#1009](https://github.com/mudler/vllm.cpp/issues/1009), and its absence is why the landed [#1008](https://github.com/mudler/vllm.cpp/issues/1008) has no speed number. It should land first. Lever 6. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-
