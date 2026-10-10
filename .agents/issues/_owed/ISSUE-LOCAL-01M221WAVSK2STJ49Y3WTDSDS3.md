ID: ISSUE-LOCAL-01M221WAVSK2STJ49Y3WTDSDS3
Title: Refresh the Qwen3.8 public documentation
Row: -
State: OPEN
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-09
Updated: 2026-10-10
Closed: -

## Problem

The aarch64 NEON lane and the scalar body of vt::PagedAttention differ only in rounding policy: the scalar body builds with -ffp-contract=off (mul + add, two roundings), the NEON lane uses fused vfmaq (one rounding, like upstream vLLM and llama.cpp). The outputs agree to 1 bf16 ulp, but the kolibri1 decode bench pins the greedy chain byte-exactly, so the difference flips one near-tie token (0.197 nats gap) and fails the gate. Adjudication: docs/bench-evidence/kolibri1-neon-paged-attn-20261008.md; history #3425, #3433, #3435.

DECIDED (developer): compile the scalar body with fused multiply-add so both arms are identical by construction. The anchor stays byte-exact; the NEON lane then defaults on with no gate relaxation. Cost: the scalar arm is production for every CPU-served model, so CPU token goldens must be re-captured and re-adjudicated under the fused numerics (kolibri W3 + bench anchor first, then anything else that moves). Plan: spec first, then re-capture, then flip the NEON default.

## Resolution

2026-10-10: DECIDED — Option C, FMA-contract the scalar paged-attention reference (developer). The issue body now carries the decision, the accepted cost (re-capture and re-adjudicate all CPU token goldens under the fused numerics; kolibri W3 + bench anchor first), and the plan: spec -> re-capture -> default the NEON lane on once both arms are bit-identical. Spec is the next artifact.
