ID: ISSUE-LOCAL-01M6VF2CZ6C0ZQA1G2Y5WJFVNH
Title: The DFlash2 adoption sketch for kolibri1 on the P150: the drafter fit, the checkpoint plan, the GO/NO-GO
Row: KOLIBRI1-DRAFTER-STUDY
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-07
Updated: 2026-11-07
Closed: -

## Problem

The developer picked kolibri1 as the first DFlash2 adoption target. kolibri1's P150 decode is already on-card with one host touch per step (the closest match to the bundle's loop); the measured baseline is 983.5 ms/step (1.02 tok/s); the attribution table is attention 50.3% / routed 33.1%. The drafter's real lift is a TRAINED CHECKPOINT (the DFlash2 drafter is qwen3.8-specific and does not transfer; kolibri1 is smaller — the drafter's overhead share is correspondingly larger). Deliverable: the adoption sketch (docs/design/kolibri1-drafter-study-20261107.md) — the TT decode structure today, the drafter fit, the checkpoint plan, the staged adoption, the GO/NO-GO with basis. Study reference: docs/design/tt-drafter-study-20261103.md on mine/row/TT-DRAFTER-STUDY. Device-free; no product code; the GO decision is the operator's with the sketch as basis. Spec: .agents/specs/kolibri1-drafter-study.md

## Resolution

-
