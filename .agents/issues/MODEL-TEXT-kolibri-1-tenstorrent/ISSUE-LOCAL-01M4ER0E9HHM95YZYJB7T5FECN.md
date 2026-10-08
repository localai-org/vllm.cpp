ID: ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN
Title: B2b-ii: the streaming MoE on the P150 (slot pool, routed path, stream bound)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

B2b-i landed the dense-resident device forward with the routed-expert tier deliberately absent (the refusal fires by name 450x per decode). The goldens are full-model decodes and cannot be replayed until the routed experts run: the B2b-ii streaming path — FP8_E4M3 slot buffers consuming Kolibri1TTExpertSlotPolicy verbatim, router readback -> host remap -> fetch list -> fetch executor -> Touch(), slot swaps on the ContentChangedSince reset lane, the B1 per-token stream bound asserted at runtime with loud failure, the full 384-expert tier host-side — is owed, and with it the deferred 141/145 token gate.

## Resolution

-
