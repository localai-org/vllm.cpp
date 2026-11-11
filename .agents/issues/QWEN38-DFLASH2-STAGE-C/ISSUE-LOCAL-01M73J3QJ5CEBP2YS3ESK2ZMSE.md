ID: ISSUE-LOCAL-01M73J3QJ5CEBP2YS3ESK2ZMSE
Title: Stage C: port the DFlash2 block-diffusion drafter and wire the speculative loop through the captured verify arm
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-10
Updated: 2026-11-10
Closed: -

## Problem

Stage B's captured verify arm is speculative-ready but nothing drafts. The bundle's DFlash2 drafter (extra_code/qwen38/dflash2.py, DFlash2Draft/DFlash2Drafter: one 5-layer forward over the target's residual taps producing 8 fixed-shape rows [anchor, MASK x7]) is the missing producer; the modeled win at the bundle's acceptance is ~8.5x on the 6606.84 ms stage-B arm. Owed: W1 obtain/convert/pin the drafter weights (download needs developer go; converter + docs/USAGE.md pin + refuse-by-name), W2 component-validated drafter forward on TT (PCC band vs the bundle reference), W3 the loop through Qwen3_5DenseDecodeGraph (draft -> captured verify replay -> host greedy accept by token-id compare; reject rolls back via the stage-B GDN commit-or-restore; dense driver only), W4 the gate (greedy token identity vs plain decode, STOP on divergence; acceptance/TPOT table vs the stage-B baseline; the host-staging cost the modeled win assumed away must be attributed honestly). Spec: .agents/specs/qwen38-dflash2-stage-c.md

## Resolution

-
