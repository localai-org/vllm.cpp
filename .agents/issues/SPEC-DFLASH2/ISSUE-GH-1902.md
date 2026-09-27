ID: ISSUE-GH-1902
Title: **W11's paged-seam guard never runs on a CUDA-graph REPLAY step, so the second refresh site is unguarded on every draft step after the first.** `detail::DflashBlockPagedAttention` re-derives the canonical `(slots, seq_ext)` pair from the store's own `ctx_len` and refuses a mismatch by name with no `kCPU` guard -- genuinely backend-independent, and it survives compiling out both `kCPU`-guarded reads beside it -- but the check is per CALL, not per STEP. On `st.g_state == 2` the driver calls `st.g_graph.Replay(queue)` and returns (`src/vllm/model_executor/models/qwen3_dflash.cpp:1636-1660`), so `ForwardPagedBody` and the guard with it are entered on the EAGER lane and on the ONE warm-then-capture step per request, and on NO replay step; those steps read persistent buffers refreshed by a SECOND production site (`:1627-1631`) with nothing downstream to check them. MEASURED by the fresh re-review of #1896: making that refresh skip on replay only (`if (st.g_state != 2) { ...Copy... }`), which on CUDA is a wrong answer at every step after the first, left ALL FOUR suites green (`decode_graph_seam` 4/4 23/23, `dflash2_draft` 43/43 449/449, `runner_reach` 8/8 162/162, `block_route` 13/13 30/30). Structurally invisible on CPU: the capture-capable CPU backend's `ReplayGraph` is a log push that executes nothing (`tests/vllm/models/decode_graph_seam_harness.h:117`), so the owed proof is DEVICE-SIDE. NOT FIXED IN FLOW: reading `g_seq_ext` back before `Replay` is a per-step D2H synchronisation on the path W11 exists to make faster, and moving the refresh inside the guarded function is impossible by construction because a replay never calls it and the buffer addresses are baked into the capture. The three places that overstated the guarantee are corrected. Listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1902
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:720`

### Frozen archive evidence

> | [#1902](https://github.com/mudler/vllm.cpp/issues/1902) | `SPEC-DFLASH2` | **W11's paged-seam guard never runs on a CUDA-graph REPLAY step, so the second refresh site is unguarded on every draft step after the first.** `detail::DflashBlockPagedAttention` re-derives the canonical `(slots, seq_ext)` pair from the store's own `ctx_len` and refuses a mismatch by name with no `kCPU` guard -- genuinely backend-independent, and it survives compiling out both `kCPU`-guarded reads beside it -- but the check is per CALL, not per STEP. On `st.g_state == 2` the driver calls `st.g_graph.Replay(queue)` and returns (`src/vllm/model_executor/models/qwen3_dflash.cpp:1636-1660`), so `ForwardPagedBody` and the guard with it are entered on the EAGER lane and on the ONE warm-then-capture step per request, and on NO replay step; those steps read persistent buffers refreshed by a SECOND production site (`:1627-1631`) with nothing downstream to check them. MEASURED by the fresh re-review of #1896: making that refresh skip on replay only (`if (st.g_state != 2) { ...Copy... }`), which on CUDA is a wrong answer at every step after the first, left ALL FOUR suites green (`decode_graph_seam` 4/4 23/23, `dflash2_draft` 43/43 449/449, `runner_reach` 8/8 162/162, `block_route` 13/13 30/30). Structurally invisible on CPU: the capture-capable CPU backend's `ReplayGraph` is a log push that executes nothing (`tests/vllm/models/decode_graph_seam_harness.h:117`), so the owed proof is DEVICE-SIDE. NOT FIXED IN FLOW: reading `g_seq_ext` back before `Replay` is a per-step D2H synchronisation on the path W11 exists to make faster, and moving the refresh inside the guarded function is impossible by construction because a replay never calls it and the buffer addresses are baked into the capture. The three places that overstated the guarantee are corrected. Listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) | bug |

## Resolution

-
