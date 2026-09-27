ID: ISSUE-GH-1900
Title: **A non-causal SWA layer drops its window in our attention kernels; upstream attends within it.** Our sliding-window lower bound is guarded on `causal && window > 0` (`src/vt/cpu/cpu_ops.cpp:2917,2994` plus nine sites in `src/vt/cuda/cuda_ops.cu`, mirrored by `DflashBlockPagedMaskOf`), so a layer resolved `(causal=false, sliding_window=W)` attends the FULL context. At the pin, `vllm/model_executor/models/qwen3_dflash.py:89-146` resolves the window and the causal flag as two INDEPENDENT answers and `:221-234` passes `per_layer_sliding_window` irrespective of `causal`; this repository's own loader already says the same in prose (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp:181-183`), so the kernels and that comment cannot both be right. Pre-existing and repo-wide, NOT introduced by #1890 — what W11 did was write it into a spec and a test header as a NORMATIVE claim without an upstream anchor, which is how it surfaced; both passages are corrected in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) to describe the byte-identity the battery measures rather than to assert the semantics are right. Live because SPEC-DFLASH2 W1 (#1314) made `(causal=false, window>0)` reachable and `z-lab/Qwen3.8-27B-DFlash2` declares exactly that, so it is the production path for the published draft checkpoint: our draft attends over more context than upstream's, which moves acceptance and is invisible to a correctness gate because speculative decoding verifies every draft. NOT FIXED IN FLOW: it moves every attention kernel in the tree and needs its own red-before evidence. Listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1900
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:719`

### Frozen archive evidence

> | [#1900](https://github.com/mudler/vllm.cpp/issues/1900) | `SPEC-DFLASH2` | **A non-causal SWA layer drops its window in our attention kernels; upstream attends within it.** Our sliding-window lower bound is guarded on `causal && window > 0` (`src/vt/cpu/cpu_ops.cpp:2917,2994` plus nine sites in `src/vt/cuda/cuda_ops.cu`, mirrored by `DflashBlockPagedMaskOf`), so a layer resolved `(causal=false, sliding_window=W)` attends the FULL context. At the pin, `vllm/model_executor/models/qwen3_dflash.py:89-146` resolves the window and the causal flag as two INDEPENDENT answers and `:221-234` passes `per_layer_sliding_window` irrespective of `causal`; this repository's own loader already says the same in prose (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp:181-183`), so the kernels and that comment cannot both be right. Pre-existing and repo-wide, NOT introduced by #1890 — what W11 did was write it into a spec and a test header as a NORMATIVE claim without an upstream anchor, which is how it surfaced; both passages are corrected in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) to describe the byte-identity the battery measures rather than to assert the semantics are right. Live because SPEC-DFLASH2 W1 (#1314) made `(causal=false, window>0)` reachable and `z-lab/Qwen3.8-27B-DFlash2` declares exactly that, so it is the production path for the published draft checkpoint: our draft attends over more context than upstream's, which moves acceptance and is invisible to a correctness gate because speculative decoding verifies every draft. NOT FIXED IN FLOW: it moves every attention kernel in the tree and needs its own red-before evidence. Listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) | bug |

## Resolution

-
