ID: ISSUE-GH-1343
Title: `vt::RmsNormPlusAdd` computes different numbers on different devices for a bf16 output. The ROCm arm (`src/vt/rocm/rocm_rmsnorm.hip`) keeps the whole chain in one f32 expression — `Store(orow, j, Load(xrow, j) * inv * wj + Load(arow, j))` — while the composed arm in `src/vt/fused_ops.cpp` calls `RmsNorm(q, out, x, w, args)` then `Add(q, out, out, addend)`, materializing `out` between them so the normalized value is rounded to bf16 BEFORE the addend is added. One `vt::` entry point, two numeric behaviours, selected by device. That is the shape AGENTS.md §Shared seams exists to prevent, and the project already holds `vt::Conv1d`/`ConvTranspose1d` to byte-identical host/CUDA providers as the standard. Invisible to a cross-device parity test with a bf16-eps tolerance, because the difference is at most 1 bf16 ULP per element. Found during the #1322 inventory. NOT fixed in flow: choosing the direction needs its own reading of the upstream fusion — either the composed arm grows a fused kernel that keeps f32 across the add, or the ROCm arm rounds — and only one matches upstream, so it owes a spec, a red-before test a tolerance cannot satisfy, and a fresh review. Owed under `## Owed` of [vt-act-round-polarity.md](../../specs/vt-act-round-polarity.md)
Row: VT-ACT-ROUND-POLARITY
State: UNKNOWN
Kind: bug
GitHub: 1343
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:440`

### Frozen archive evidence

> | [#1343](https://github.com/mudler/vllm.cpp/issues/1343) | `VT-ACT-ROUND-POLARITY` | `vt::RmsNormPlusAdd` computes different numbers on different devices for a bf16 output. The ROCm arm (`src/vt/rocm/rocm_rmsnorm.hip`) keeps the whole chain in one f32 expression — `Store(orow, j, Load(xrow, j) * inv * wj + Load(arow, j))` — while the composed arm in `src/vt/fused_ops.cpp` calls `RmsNorm(q, out, x, w, args)` then `Add(q, out, out, addend)`, materializing `out` between them so the normalized value is rounded to bf16 BEFORE the addend is added. One `vt::` entry point, two numeric behaviours, selected by device. That is the shape AGENTS.md §Shared seams exists to prevent, and the project already holds `vt::Conv1d`/`ConvTranspose1d` to byte-identical host/CUDA providers as the standard. Invisible to a cross-device parity test with a bf16-eps tolerance, because the difference is at most 1 bf16 ULP per element. Found during the #1322 inventory. NOT fixed in flow: choosing the direction needs its own reading of the upstream fusion — either the composed arm grows a fused kernel that keeps f32 across the add, or the ROCm arm rounds — and only one matches upstream, so it owes a spec, a red-before test a tolerance cannot satisfy, and a fresh review. Owed under `## Owed` of [vt-act-round-polarity.md](../../specs/vt-act-round-polarity.md) | bug |

## Resolution

-
