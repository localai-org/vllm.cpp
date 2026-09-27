ID: ISSUE-GH-2131
Title: **The GPU runner accepts exactly ONE `MambaSpec` group of exactly TWO shapes, so a PLE layer's third conv state is unrepresentable and W5c's KV-cache spec cannot be written against it.** Verified in the tree rather than relayed: `src/vllm/v1/worker/gpu/runner.cpp` asserts `mamba_spec->shapes.size() == 2 && mamba_spec->dtypes.size() == 2` ("runner: recurrent MambaSpec must contain conv then temporal state") and then reads `shapes[0]` as conv and `shapes[1]` as temporal, while the topology refusal a few lines above states the one-group rule in prose ("any number of non-eagle AttentionSpec groups and ONE MambaSpec group") and `gdn_group_id_` is a single scalar index, not a list. A `qwen4_exp` PLE layer carries THREE persistent recurrent streams — the GDN conv, the PLE dilated depthwise conv (`kernel_size = 4, dilation = 3`, a 9-deep ring buffer read at stride 3) and the int64 n-gram token history — so the two-shape assumption cannot express the states and the one-group assumption cannot address them separately. The blocker is in the ENGINE, not in the model registry, which is why it is its own issue rather than part of a model wave. Scope: generalise the recurrent-cache topology so a `MambaSpec` group can carry more than two shapes, or so more than one such group can exist, with per-state addressing, mirroring vLLM where it defines the behaviour; the existing Mamba/GDN arms must stay byte-identical, with `test_qwen27_paged_forward` and the qwen3.5 recurrent suites as the regression gate. Recorded under `## Owed` and `## Now` in [`specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md), which until now described the constraint in prose with no issue behind it
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: gap
GitHub: 2131
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:821`

### Frozen archive evidence

> | [#2131](https://github.com/mudler/vllm.cpp/issues/2131) | `MODEL-MM-QWEN4-EXP` | **The GPU runner accepts exactly ONE `MambaSpec` group of exactly TWO shapes, so a PLE layer's third conv state is unrepresentable and W5c's KV-cache spec cannot be written against it.** Verified in the tree rather than relayed: `src/vllm/v1/worker/gpu/runner.cpp` asserts `mamba_spec->shapes.size() == 2 && mamba_spec->dtypes.size() == 2` ("runner: recurrent MambaSpec must contain conv then temporal state") and then reads `shapes[0]` as conv and `shapes[1]` as temporal, while the topology refusal a few lines above states the one-group rule in prose ("any number of non-eagle AttentionSpec groups and ONE MambaSpec group") and `gdn_group_id_` is a single scalar index, not a list. A `qwen4_exp` PLE layer carries THREE persistent recurrent streams — the GDN conv, the PLE dilated depthwise conv (`kernel_size = 4, dilation = 3`, a 9-deep ring buffer read at stride 3) and the int64 n-gram token history — so the two-shape assumption cannot express the states and the one-group assumption cannot address them separately. The blocker is in the ENGINE, not in the model registry, which is why it is its own issue rather than part of a model wave. Scope: generalise the recurrent-cache topology so a `MambaSpec` group can carry more than two shapes, or so more than one such group can exist, with per-state addressing, mirroring vLLM where it defines the behaviour; the existing Mamba/GDN arms must stay byte-identical, with `test_qwen27_paged_forward` and the qwen3.5 recurrent suites as the regression gate. Recorded under `## Owed` and `## Now` in [`specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md), which until now described the constraint in prose with no issue behind it | gap |

## Resolution

-
