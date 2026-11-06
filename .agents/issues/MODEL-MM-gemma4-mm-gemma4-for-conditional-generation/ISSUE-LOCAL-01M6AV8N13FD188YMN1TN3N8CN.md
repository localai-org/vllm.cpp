ID: ISSUE-LOCAL-01M6AV8N13FD188YMN1TN3N8CN
Title: Gemma4 MoE CPU forward throws in DualRmsNormPlusRes
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

The production Gemma4 MoE forward unconditionally calls vt::DualRmsNormPlusRes. Its CPU arm throws CPU compose not yet wired, so the compact CT production fixture cannot emit logits after loading. Implement the shared composed normalization and residual operation and verify its order and rounding through the production fixture.

## Resolution

2026-11-01: Shared CPU DualRmsNormPlusRes composition implemented with checked scratch sizing. BF16/F32, aliases, overflow metadata and malformed shapes pass; order, residual and size-guard mutations fail safely. Fresh review PASS d3e529b28. Operator reruns rebuilt CPU focused gate and real Gemma4 forward successfully. Lands with the fork CPU CT repair; upstream PR3540 remains Draft.
