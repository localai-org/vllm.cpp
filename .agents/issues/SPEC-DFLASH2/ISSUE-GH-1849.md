ID: ISSUE-GH-1849
Title: **The DFlash2 draft step costs a flat ~23 ms at EVERY K, and the two levers #1849 names resolve differently once read from the records.** Lever A (quantize the shared head) is ALREADY LANDED for the measured subject: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` stores `lm_head` as W4A16_NVFP4 g16 (header-verified 2026-08-21, quantization-matrix `QUANT-QWEN38-27B-NVFP4-ARM`), upstream computes with it packed through `lm_head.quant_method.apply`, and both our reads have been packed since #1628 — so the head traffic is ~2×0.72 GB not 2×2.54, the draft-phase floor re-derives to ~9 ms, and the unattributed residual GROWS to ~13-14 ms. Lever B (launch/sync trim) is counted in code at one replay + ~10 launches + ~76 B up / 64 B down + one sync — well under 0.5 ms, so the residual sits INSIDE kernels and needs on-box attribution. W9 lands `VT_SPEC_TRACE=2` (the `[spec-phase-dev]` pre/fwd/select/walk split) as the instrument, and borrow-first loading for the draft's shared bf16 embed+head (~5.1 GB host on the bf16 arm, ~2.5 GB on the r0b0tlab arm; memory only, no step-time claim). The bf16-target arm's 2×2.54 GB head reads are upstream's own serving dtype and stand as a recorded ceiling. Wave spec [dflash2-draft-fixed-cost.md](../../specs/dflash2-draft-fixed-cost.md); the K-ladder rerun, the `ncu`/`nsys` attribution and any step delta are owed there, operator-run
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1849
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:693`

### Frozen archive evidence

> | [#1849](https://github.com/mudler/vllm.cpp/issues/1849) | `SPEC-DFLASH2` | **The DFlash2 draft step costs a flat ~23 ms at EVERY K, and the two levers #1849 names resolve differently once read from the records.** Lever A (quantize the shared head) is ALREADY LANDED for the measured subject: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` stores `lm_head` as W4A16_NVFP4 g16 (header-verified 2026-08-21, quantization-matrix `QUANT-QWEN38-27B-NVFP4-ARM`), upstream computes with it packed through `lm_head.quant_method.apply`, and both our reads have been packed since #1628 — so the head traffic is ~2×0.72 GB not 2×2.54, the draft-phase floor re-derives to ~9 ms, and the unattributed residual GROWS to ~13-14 ms. Lever B (launch/sync trim) is counted in code at one replay + ~10 launches + ~76 B up / 64 B down + one sync — well under 0.5 ms, so the residual sits INSIDE kernels and needs on-box attribution. W9 lands `VT_SPEC_TRACE=2` (the `[spec-phase-dev]` pre/fwd/select/walk split) as the instrument, and borrow-first loading for the draft's shared bf16 embed+head (~5.1 GB host on the bf16 arm, ~2.5 GB on the r0b0tlab arm; memory only, no step-time claim). The bf16-target arm's 2×2.54 GB head reads are upstream's own serving dtype and stand as a recorded ceiling. Wave spec [dflash2-draft-fixed-cost.md](../../specs/dflash2-draft-fixed-cost.md); the K-ladder rerun, the `ncu`/`nsys` attribution and any step delta are owed there, operator-run | perf |

## Resolution

-
