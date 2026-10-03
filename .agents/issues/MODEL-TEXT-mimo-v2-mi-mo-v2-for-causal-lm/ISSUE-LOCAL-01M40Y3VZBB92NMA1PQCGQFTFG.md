ID: ISSUE-LOCAL-01M40Y3VZBB92NMA1PQCGQFTFG
Title: runner-routing scanner misses cross-TU forward delegation — mimo_v2 classified NONE
Row: MODEL-TEXT-mimo-v2-mi-mo-v2-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: -

## Problem

scripts/check-runner-routing-consistency.py classifies a registered model from the registry TU's .forward hook. classify_with_helpers hops ONE level into a ForwardLogits-returning helper defined in the SAME TU (deepseek_v4's WrapV4DeviceLogits). mimo_v2's registry TU declares ForwardMiMoV2Device but defines it in mimo_v2.cpp, returning WrapDeviceLogits there; the local hop finds only the forward declaration, classification is NONE, and mimo_v2 silently drops into the exempt NONE bucket (test_private_device_wrapper_classifies_device fails: ['mimo_v2'] != []). dots3_note is unaffected because it delegates via Class::ForwardDevice, which collect_forward_device_bodies resolves cross-TU. Fix: extend the hop so a free function DECLARED in the registry TU but DEFINED in a sibling models/*.cpp is resolved there (one more level of the same semantics); keep NONE an error state.

## Resolution

-
