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

The aarch64 NEON lane of vt::PagedAttention agrees with the scalar body to 1 bf16 ulp (rounding policy only: scalar is -ffp-contract=off, NEON is fused like upstream). The kolibri1 decode bench pins the greedy chain byte-exactly, so one near-tie token flips (0.197 nats) and the gate fails. GitHub issue #3438 asks Ettore: B (near-tie instrument for the bench anchor, W3 methodology, running as a pilot) or C (FMA-contract the scalar reference, fleet-wide re-oracle, better long-run end state). Not on the table: making NEON bit-match the non-fused chain.

## Resolution

2026-10-10: DECIDED — Option C, FMA-contract the scalar paged-attention reference (developer). The issue body now carries the decision, the accepted cost (re-capture and re-adjudicate all CPU token goldens under the fused numerics; kolibri W3 + bench anchor first), and the plan: spec -> re-capture -> default the NEON lane on once both arms are bit-identical. Spec is the next artifact.
