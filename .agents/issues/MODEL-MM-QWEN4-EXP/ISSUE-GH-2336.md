ID: ISSUE-GH-2336
Title: **The layer loop's remaining prerequisite is the PLE BLOCK and its GATE op, and the spec's `## Now` contradicted itself about that on `c0fa299b1` — two LIVE enumerations, which is #2288 in its seventh turn.** One paragraph said "NONE remain. The count is ZERO" of #2249's five prerequisites (true, and only about those five); eleven lines below, "What has no production shape yet is the PLE block, the GDN weight adapter onto `GdnLayerWeights`, the hyper-connection stream through the per-layer loop, and the loop itself". A wave dispatched to write the loop read the first and returned `NEEDS_DECISION`. Measured on `bd90b92b0`, the count moves BOTH ways. The PLE GATE (`modeling_qwen4_exp.py:1181-1182` — `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`, then `sigmoid(gate) * value.unsqueeze(-2)`) was OP-SIZED and nothing had ever named it: a `git grep` for `clamp_min`, `signed_sqrt` and `copysign` over `src/vt include/vt` returned ZERO lines and the only implementation was the host, file-local `SignedSqrtGate`. The DOT around it needs no new op (`vt::BatchedMatmul` over `[T*hc,1,H] x [T*hc,H,1]` VIEWS), and the multiply cannot reuse `vt::SigmoidGateBf16` (refuses by element count) or `vt::MulColVecF32` (per-COLUMN) because BOTH its operands broadcast — so ONE fused op is owed, not five. Two listed items are smaller than "missing production shape" implies: the GDN adapter is a nine-assignment FIELD COPY (with `output_gate_type` sigmoid-vs-silu, a lost `ResidentWeight::d_dev`, and an empty `in_proj_ba` as its real risks) and the hyper-connection widen is `vt::IndexSelect`. What is left is the PLE BLOCK — the LAST block seam, `PleForward` has zero cross-TU callers and `vt::RmsNormGroup` has zero production callers — and the loop. Split W5e-1 (the gate op), W5e-2 (the block), W5f (the loop). **W5e-1 LANDED: `vt::Qwen4ExpPleGate`, CPU arm, gated against section J of `qwen4_exp_ple_goldens.inc`, UNREACHED by design and recorded under `## Owed`; W5e-2 and W5f remain open on this issue.** Spec [qwen4-exp-flash-next.md](../../specs/qwen4-exp-flash-next.md)
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2336
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:907`

### Frozen archive evidence

> | [#2336](https://github.com/mudler/vllm.cpp/issues/2336) | `MODEL-MM-QWEN4-EXP` | **The layer loop's remaining prerequisite is the PLE BLOCK and its GATE op, and the spec's `## Now` contradicted itself about that on `c0fa299b1` — two LIVE enumerations, which is #2288 in its seventh turn.** One paragraph said "NONE remain. The count is ZERO" of #2249's five prerequisites (true, and only about those five); eleven lines below, "What has no production shape yet is the PLE block, the GDN weight adapter onto `GdnLayerWeights`, the hyper-connection stream through the per-layer loop, and the loop itself". A wave dispatched to write the loop read the first and returned `NEEDS_DECISION`. Measured on `bd90b92b0`, the count moves BOTH ways. The PLE GATE (`modeling_qwen4_exp.py:1181-1182` — `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`, then `sigmoid(gate) * value.unsqueeze(-2)`) was OP-SIZED and nothing had ever named it: a `git grep` for `clamp_min`, `signed_sqrt` and `copysign` over `src/vt include/vt` returned ZERO lines and the only implementation was the host, file-local `SignedSqrtGate`. The DOT around it needs no new op (`vt::BatchedMatmul` over `[T*hc,1,H] x [T*hc,H,1]` VIEWS), and the multiply cannot reuse `vt::SigmoidGateBf16` (refuses by element count) or `vt::MulColVecF32` (per-COLUMN) because BOTH its operands broadcast — so ONE fused op is owed, not five. Two listed items are smaller than "missing production shape" implies: the GDN adapter is a nine-assignment FIELD COPY (with `output_gate_type` sigmoid-vs-silu, a lost `ResidentWeight::d_dev`, and an empty `in_proj_ba` as its real risks) and the hyper-connection widen is `vt::IndexSelect`. What is left is the PLE BLOCK — the LAST block seam, `PleForward` has zero cross-TU callers and `vt::RmsNormGroup` has zero production callers — and the loop. Split W5e-1 (the gate op), W5e-2 (the block), W5f (the loop). **W5e-1 LANDED: `vt::Qwen4ExpPleGate`, CPU arm, gated against section J of `qwen4_exp_ple_goldens.inc`, UNREACHED by design and recorded under `## Owed`; W5e-2 and W5f remain open on this issue.** Spec [qwen4-exp-flash-next.md](../../specs/qwen4-exp-flash-next.md) | bug |

## Resolution

-
