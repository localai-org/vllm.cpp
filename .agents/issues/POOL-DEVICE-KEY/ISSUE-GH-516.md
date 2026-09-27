ID: ISSUE-GH-516
Title: `vllm::Pool()`'s free list is keyed by size class with NO DEVICE in the key, so a `cudaMalloc` block reaches a CPU `DBuf` (SIGSEGV) and a host block reaches a CUDA forward (uniform `0x7fff0000` NaN); spec [`pool-device-key.md`](../../specs/pool-device-key.md), lands through `row/MODEL-DIFFUSION-LTX25`
Row: POOL-DEVICE-KEY
State: UNKNOWN
Kind: bug
GitHub: 516
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:124`

### Frozen archive evidence

> | [#516](https://github.com/mudler/vllm.cpp/issues/516) | `POOL-DEVICE-KEY` | `vllm::Pool()`'s free list is keyed by size class with NO DEVICE in the key, so a `cudaMalloc` block reaches a CPU `DBuf` (SIGSEGV) and a host block reaches a CUDA forward (uniform `0x7fff0000` NaN); spec [`pool-device-key.md`](../../specs/pool-device-key.md), lands through `row/MODEL-DIFFUSION-LTX25` | bug |

## Resolution

-
