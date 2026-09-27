ID: ISSUE-GH-1692
Title: **[#1584](https://github.com/mudler/vllm.cpp/issues/1584) is FIXED by this row, and its gate is CPU-only: both production call sites need a GPU and neither arm was executed.** The fix adds `GetOpFallbackUncounted`, sharing one body with `GetOpFallback` so resolution order, the reference-tier install, the drain and every throw stay identical and only the `fetch_add` is conditional -- in its existing position, BEFORE the "nothing below" check, because a decline that throws is still a decline. The two SHAPE-GATED providers that hoist the lookup into a function-local static switch to it: `cuda_attention_cross.cu` `BlockedFallback` and `metal_mlx_provider.mm` `MlxFallback`. **Candidate 1 of #1584 -- drop the count from `GetOpFallback` -- was REJECTED and measured rather than argued:** it edits five per-call sites (`vulkan_ops.cpp:950,1067,1488,1509`, `tenstorrent_ops.cpp:1341`), one already gated on an exact value at `test_vulkan_backend.cpp:2901` (`after.declines == before.declines + 1`), and its failure mode for a future caller is silent UNDER-counting, the Risk 4 the seam exists to expose. Net blast radius is ZERO backends, not the four #1584 estimated; exactly two providers change what they report, by one. #1555's `WarmDeclineOnce` workaround is REMOVED. **What this issue owns is the missing EXECUTION:** `test_ops_attention_cross` on CUDA (20 cases, ALL 20 SKIP on a CPU-only build -- 32 assertions, every one the skip guard, so that suite gives the change no coverage here), `test_metal_backend` on a `VLLM_CPP_MLX` build, and the `.agents/reachability.md` mutation on `BlockedFallback()` / `MlxFallback()`. The CUDA COMPILE is covered by the `-DVLLM_CPP_CUDA=ON` `vllm`-target job in `.github/workflows/ci.yml`; the Metal `.mm` file is compiled by NO job in this repository, because MLX needs `MLX_ROOT`. Red-before/green-after and the mutation table in [op-provider-decline-exact.md](../../specs/op-provider-decline-exact.md)
Row: KERNEL-ACCEL-PROVIDER-DECLINE-EXACT
State: UNKNOWN
Kind: bug
GitHub: 1692
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:633`

### Frozen archive evidence

> | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) | `KERNEL-ACCEL-PROVIDER-DECLINE-EXACT` | **[#1584](https://github.com/mudler/vllm.cpp/issues/1584) is FIXED by this row, and its gate is CPU-only: both production call sites need a GPU and neither arm was executed.** The fix adds `GetOpFallbackUncounted`, sharing one body with `GetOpFallback` so resolution order, the reference-tier install, the drain and every throw stay identical and only the `fetch_add` is conditional -- in its existing position, BEFORE the "nothing below" check, because a decline that throws is still a decline. The two SHAPE-GATED providers that hoist the lookup into a function-local static switch to it: `cuda_attention_cross.cu` `BlockedFallback` and `metal_mlx_provider.mm` `MlxFallback`. **Candidate 1 of #1584 -- drop the count from `GetOpFallback` -- was REJECTED and measured rather than argued:** it edits five per-call sites (`vulkan_ops.cpp:950,1067,1488,1509`, `tenstorrent_ops.cpp:1341`), one already gated on an exact value at `test_vulkan_backend.cpp:2901` (`after.declines == before.declines + 1`), and its failure mode for a future caller is silent UNDER-counting, the Risk 4 the seam exists to expose. Net blast radius is ZERO backends, not the four #1584 estimated; exactly two providers change what they report, by one. #1555's `WarmDeclineOnce` workaround is REMOVED. **What this issue owns is the missing EXECUTION:** `test_ops_attention_cross` on CUDA (20 cases, ALL 20 SKIP on a CPU-only build -- 32 assertions, every one the skip guard, so that suite gives the change no coverage here), `test_metal_backend` on a `VLLM_CPP_MLX` build, and the `.agents/reachability.md` mutation on `BlockedFallback()` / `MlxFallback()`. The CUDA COMPILE is covered by the `-DVLLM_CPP_CUDA=ON` `vllm`-target job in `.github/workflows/ci.yml`; the Metal `.mm` file is compiled by NO job in this repository, because MLX needs `MLX_ROOT`. Red-before/green-after and the mutation table in [op-provider-decline-exact.md](../../specs/op-provider-decline-exact.md) | bug |

## Resolution

-
