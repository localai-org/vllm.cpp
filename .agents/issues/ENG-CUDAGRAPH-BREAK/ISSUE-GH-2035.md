ID: ISSUE-GH-2035
Title: Seven decode-graph drivers (`qwen3.cpp`, `qwen3_moe.cpp`, `deepseek_v2.cpp`, `deepseek_v4.cpp`, `voxtral.cpp`, `laguna.cpp`, `qwen3_dflash.cpp`) open a `vt::GraphCaptureScope` with no `DevicePool::PreGrowForCapture` and no demand profile at all — only `qwen3_5.cpp` uses any of the #1380 machinery. Found while fixing #2029; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md)
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 2035
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:774`

### Frozen archive evidence

> | [#2035](https://github.com/mudler/vllm.cpp/issues/2035) | `ENG-CUDAGRAPH-BREAK` | Seven decode-graph drivers (`qwen3.cpp`, `qwen3_moe.cpp`, `deepseek_v2.cpp`, `deepseek_v4.cpp`, `voxtral.cpp`, `laguna.cpp`, `qwen3_dflash.cpp`) open a `vt::GraphCaptureScope` with no `DevicePool::PreGrowForCapture` and no demand profile at all — only `qwen3_5.cpp` uses any of the #1380 machinery. Found while fixing #2029; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md) | bug |

## Resolution

-
