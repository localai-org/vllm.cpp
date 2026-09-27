ID: ISSUE-GH-777
Title: Two GPU mutexes: the documented `flock $GPU_LOCK` took a DIFFERENT file from the harness scripts, so following the docs produced an unserialised run — it voided a standalone Marlin series (every absolute downgraded to an upper bound) and `flock` succeeds on the wrong path, so nothing at run time can see it. FIXED: `${GPU_LOCK:-$HOME/gpu.lock}` everywhere, pinned by `tests/scripts/test_gpu_lock_one_truth.py`, spec [`gpu-lock-one-truth.md`](../../specs/gpu-lock-one-truth.md)
Row: ENG-PREFLIGHT-COMPILES
State: UNKNOWN
Kind: bug
GitHub: 777
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:39`

### Frozen archive evidence

> | [#777](https://github.com/mudler/vllm.cpp/issues/777) | — | Two GPU mutexes: the documented `flock $GPU_LOCK` took a DIFFERENT file from the harness scripts, so following the docs produced an unserialised run — it voided a standalone Marlin series (every absolute downgraded to an upper bound) and `flock` succeeds on the wrong path, so nothing at run time can see it. FIXED: `${GPU_LOCK:-$HOME/gpu.lock}` everywhere, pinned by `tests/scripts/test_gpu_lock_one_truth.py`, spec [`gpu-lock-one-truth.md`](../../specs/gpu-lock-one-truth.md) | bug |

## Resolution

-
