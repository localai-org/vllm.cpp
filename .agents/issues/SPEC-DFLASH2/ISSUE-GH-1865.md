ID: ISSUE-GH-1865
Title: **W10's spec-as-decode lane never engages at runtime: the q=9 verify still runs `PagedFlashKernel` (nsys, 16 x 117 calls — the real prefill forward included) and `LaunchSpecDecodeFA2Bf16` never appears, with the on/off A/B speed-neutral.** Traced: the threading (runner -> `CommonAttentionMetadata` -> `PagedAttentionArgs`) is live — a probe at `vt::PagedAttention` under the production CPU fixture shows every uniform verify arriving classified — but the W10 admission dies on its bf16-query conjuncts because the model-side dtype selection (`FullAttnBlockPaged`) has NO spec-as-decode arm: the verify's bf16-ness rides the PREFILL lever's `Fa2PrefillOn()`, while the CUDA admission reads `Fa2SpecDecodeEnabled()`/`Fa2Decode*Enabled()` — two sides consulting different switches, and the profiled binary's FA2 arm was dark end to end (unstaged-CUTLASS configure prints `CUDA FA2 compiled-arch manifest: []` and builds green). Repair: the eligibility is extracted to a host-testable seam (`ClassifyDenseFa2`) and gains the spec arm reading the SPEC lane's own toggles; `vt::PagedAttention` counts classified arrivals so the W10 review's dead mutation (deleting the `pa_args.uniform_spec_query_len` threading) reds on a CPU box; a classified batch the CUDA dispatch cannot serve narrates ONCE to stderr naming the failed conjunct group. The nsys re-profile on a manifest-verified FA2 build and the moving A/B are owed, operator-run — wave spec [dflash2-spec-as-decode-repair.md](../../specs/dflash2-spec-as-decode-repair.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1865
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:702`

### Frozen archive evidence

> | [#1865](https://github.com/mudler/vllm.cpp/issues/1865) | `SPEC-DFLASH2` | **W10's spec-as-decode lane never engages at runtime: the q=9 verify still runs `PagedFlashKernel` (nsys, 16 x 117 calls — the real prefill forward included) and `LaunchSpecDecodeFA2Bf16` never appears, with the on/off A/B speed-neutral.** Traced: the threading (runner -> `CommonAttentionMetadata` -> `PagedAttentionArgs`) is live — a probe at `vt::PagedAttention` under the production CPU fixture shows every uniform verify arriving classified — but the W10 admission dies on its bf16-query conjuncts because the model-side dtype selection (`FullAttnBlockPaged`) has NO spec-as-decode arm: the verify's bf16-ness rides the PREFILL lever's `Fa2PrefillOn()`, while the CUDA admission reads `Fa2SpecDecodeEnabled()`/`Fa2Decode*Enabled()` — two sides consulting different switches, and the profiled binary's FA2 arm was dark end to end (unstaged-CUTLASS configure prints `CUDA FA2 compiled-arch manifest: []` and builds green). Repair: the eligibility is extracted to a host-testable seam (`ClassifyDenseFa2`) and gains the spec arm reading the SPEC lane's own toggles; `vt::PagedAttention` counts classified arrivals so the W10 review's dead mutation (deleting the `pa_args.uniform_spec_query_len` threading) reds on a CPU box; a classified batch the CUDA dispatch cannot serve narrates ONCE to stderr naming the failed conjunct group. The nsys re-profile on a manifest-verified FA2 build and the moving A/B are owed, operator-run — wave spec [dflash2-spec-as-decode-repair.md](../../specs/dflash2-spec-as-decode-repair.md) | bug |

## Resolution

-
