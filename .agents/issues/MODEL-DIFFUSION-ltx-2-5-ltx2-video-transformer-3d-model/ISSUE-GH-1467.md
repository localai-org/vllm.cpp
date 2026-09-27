ID: ISSUE-GH-1467
Title: **The ltx2 prompt->conditioning case no longer detects position renumbering, and post-`4712dac40` the instrument is INVERTED — the mutant is closer to the oracle than the correct code.** MEASURED in one build directory, CPU-only Release, x86_64, compile rc 0 on every arm and every source restored `sha256sum`-verified, as ratios to the propagated floor: before `4712dac40`, correct 0.565/0.688 and renumbered 0.831/**1.099** (the 1.10x the case's own note claims, and it reded); at `aeba0de6f`, correct **1.209/1.313** and renumbered 0.683/0.931 — so at the old `1.0x` bound the case reds the port and passes the defect. `4712dac40` is right; what it did here was raise this comparison's noise floor above the defect's signal, and no constant recovers a detection whose ordering has reversed. [#1458](https://github.com/mudler/vllm.cpp/issues/1458) restores a functioning instrument at a derived `2.0x` (its scatter-offset mutation still reds at 14.08x/24.06x) and does not claim this coverage. A repair owes an instrument with no bf16 accumulation between the defect and the assertion — the integer `positions` contract, or the f32 rope table `scripts/gen-ltx2-gemma-tower-goldens.py:363-375` already names as the right one for this class. Found in flow while gating #1458 (PR [#1461](https://github.com/mudler/vllm.cpp/pull/1461)); also listed under `## Owed` in [`ltx-2-5.md`](../../specs/ltx-2-5.md)
Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model
State: UNKNOWN
Kind: bug
GitHub: 1467
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:516`

### Frozen archive evidence

> | [#1467](https://github.com/mudler/vllm.cpp/issues/1467) | `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` | **The ltx2 prompt->conditioning case no longer detects position renumbering, and post-`4712dac40` the instrument is INVERTED — the mutant is closer to the oracle than the correct code.** MEASURED in one build directory, CPU-only Release, x86_64, compile rc 0 on every arm and every source restored `sha256sum`-verified, as ratios to the propagated floor: before `4712dac40`, correct 0.565/0.688 and renumbered 0.831/**1.099** (the 1.10x the case's own note claims, and it reded); at `aeba0de6f`, correct **1.209/1.313** and renumbered 0.683/0.931 — so at the old `1.0x` bound the case reds the port and passes the defect. `4712dac40` is right; what it did here was raise this comparison's noise floor above the defect's signal, and no constant recovers a detection whose ordering has reversed. [#1458](https://github.com/mudler/vllm.cpp/issues/1458) restores a functioning instrument at a derived `2.0x` (its scatter-offset mutation still reds at 14.08x/24.06x) and does not claim this coverage. A repair owes an instrument with no bf16 accumulation between the defect and the assertion — the integer `positions` contract, or the f32 rope table `scripts/gen-ltx2-gemma-tower-goldens.py:363-375` already names as the right one for this class. Found in flow while gating #1458 (PR [#1461](https://github.com/mudler/vllm.cpp/pull/1461)); also listed under `## Owed` in [`ltx-2-5.md`](../../specs/ltx-2-5.md) | bug |

## Resolution

-
