ID: ISSUE-GH-2193
Title: **dots3-note's grouped MoE arm re-introduced #237's address-keyed residency and cited the #237 fix as its warrant.** W5 shipped `Dots3NoteMoePtrsFor` as `static std::map<const Dots3NoteMoeWeights*, Dots3NoteMoePtrs> table; return table[key];` — the exact shape `ce2349dee` (2026-08-10) removed from `qwen3_5.cpp`, eighteen days earlier — under a comment claiming `qwen3_5.cpp`'s `MoeBf16Resident` relies on it. It does not: `MoeBf16ResidentFor` is `ResidentIn<MoeBf16Resident>(w->resident_bf16)` and `ResidentIn`'s own comment says it REPLACED that map because "keying on the weight's ADDRESS let a second engine inherit a freed engine's device pointers (issue #237)". Load A, destroy it, load B at A's address: `mr.ready` is already true, the upload is skipped, and every routed expert GEMM reads A's pointers. The buffers are deliberately never freed, so there is no crash and no error — B silently answers from A's experts, quieter than the #237 repro's zeroed token ids. Secondary: `table[key]` mutates a `std::map` under no lock where `ResidentIn` takes a mutex. FIXED IN FLOW in [#2187](https://github.com/mudler/vllm.cpp/pull/2187) (fresh-review F1): `Dots3NoteMoeWeights` gained a `ResidentSlot resident_moe`, the accessor builds into it under a mutex (the `laguna.cpp:497-507` shape), and `test_moe_resident_lifetime.cpp` gained four cases for this block. What is NOT gated, said plainly: `kMoeGroupedGemmBf16` is CUDA-only with no CPU reference tier and the accessor is file-local, so no CPU gate can call it — the cases pin that residency is a member of the weights, not the accessor body. Still owed: `deepseek_v2.cpp`'s `MoePtrs` (`04f5c01e7`, 2026-07-22) carries the same pre-#237 shape and is unswept debt on a SACRED path, not touched here. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md)
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 2193
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:850`

### Frozen archive evidence

> | [#2193](https://github.com/mudler/vllm.cpp/issues/2193) | `MODEL-MM-dots3-note` | **dots3-note's grouped MoE arm re-introduced #237's address-keyed residency and cited the #237 fix as its warrant.** W5 shipped `Dots3NoteMoePtrsFor` as `static std::map<const Dots3NoteMoeWeights*, Dots3NoteMoePtrs> table; return table[key];` — the exact shape `ce2349dee` (2026-08-10) removed from `qwen3_5.cpp`, eighteen days earlier — under a comment claiming `qwen3_5.cpp`'s `MoeBf16Resident` relies on it. It does not: `MoeBf16ResidentFor` is `ResidentIn<MoeBf16Resident>(w->resident_bf16)` and `ResidentIn`'s own comment says it REPLACED that map because "keying on the weight's ADDRESS let a second engine inherit a freed engine's device pointers (issue #237)". Load A, destroy it, load B at A's address: `mr.ready` is already true, the upload is skipped, and every routed expert GEMM reads A's pointers. The buffers are deliberately never freed, so there is no crash and no error — B silently answers from A's experts, quieter than the #237 repro's zeroed token ids. Secondary: `table[key]` mutates a `std::map` under no lock where `ResidentIn` takes a mutex. FIXED IN FLOW in [#2187](https://github.com/mudler/vllm.cpp/pull/2187) (fresh-review F1): `Dots3NoteMoeWeights` gained a `ResidentSlot resident_moe`, the accessor builds into it under a mutex (the `laguna.cpp:497-507` shape), and `test_moe_resident_lifetime.cpp` gained four cases for this block. What is NOT gated, said plainly: `kMoeGroupedGemmBf16` is CUDA-only with no CPU reference tier and the accessor is file-local, so no CPU gate can call it — the cases pin that residency is a member of the weights, not the accessor body. Still owed: `deepseek_v2.cpp`'s `MoePtrs` (`04f5c01e7`, 2026-07-22) carries the same pre-#237 shape and is unswept debt on a SACRED path, not touched here. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md) | bug |

## Resolution

-
