ID: ISSUE-GH-2088
Title: **A non-causal DFlash SWA layer runs with NO sliding window here, where upstream passes `per_layer_sliding_window` unconditionally.** Upstream resolves `(sliding_window, causal)` as two independent answers (`vllm/model_executor/models/qwen3_dflash.py:86-146` at pin `5559679229`) and hands the window to `Attention` with no reference to causality (`:229`), consuming `self.causal` one level out as attention metadata (`:234`, `:720`). Our tree conditions the window on causality in every kernel and in the paged seam's mask map — `qwen3_dflash_internal.h:125`, `cuda_ops.cu:1582`, `:1802`, `:1971`, `:2255`, `:2405`, `:2435`, `cpu_ops.cpp:2951`, `:3029` — so a declared `is_causal false` beside `sliding_attention` layers drops the window on all of them. That is every draft layer of the campaign subject, and it makes row 2 of the `.agents/specs/dflash2-draft-block-fa2.md` dispatch table unreachable from production. Acceptance-only, therefore invisible to a token gate, and the compute goes from `O(ctx x W)` to `O(ctx^2)` per query row. Owed: read the resolved `(causal, sliding_window)` pair off the draft's own `config.json` before writing code — the issue is INERT for a draft whose layers resolve `sliding_window == 0`. Listed under `## Owed` in [`specs/dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: correctness
GitHub: 2088
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:802`

### Frozen archive evidence

> | [#2088](https://github.com/mudler/vllm.cpp/issues/2088) | `SPEC-DFLASH2` | **A non-causal DFlash SWA layer runs with NO sliding window here, where upstream passes `per_layer_sliding_window` unconditionally.** Upstream resolves `(sliding_window, causal)` as two independent answers (`vllm/model_executor/models/qwen3_dflash.py:86-146` at pin `5559679229`) and hands the window to `Attention` with no reference to causality (`:229`), consuming `self.causal` one level out as attention metadata (`:234`, `:720`). Our tree conditions the window on causality in every kernel and in the paged seam's mask map — `qwen3_dflash_internal.h:125`, `cuda_ops.cu:1582`, `:1802`, `:1971`, `:2255`, `:2405`, `:2435`, `cpu_ops.cpp:2951`, `:3029` — so a declared `is_causal false` beside `sliding_attention` layers drops the window on all of them. That is every draft layer of the campaign subject, and it makes row 2 of the `.agents/specs/dflash2-draft-block-fa2.md` dispatch table unreachable from production. Acceptance-only, therefore invisible to a token gate, and the compute goes from `O(ctx x W)` to `O(ctx^2)` per query row. Owed: read the resolved `(causal, sliding_window)` pair off the draft's own `config.json` before writing code — the issue is INERT for a draft whose layers resolve `sliding_window == 0`. Listed under `## Owed` in [`specs/dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) | correctness |

## Resolution

-
