ID: ISSUE-LOCAL-01M4P2X9JQ50HW5D9JMYDS7MF5
Title: Gemma reasoning registry lacks prompt-aware adapter
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: 2026-10-11

## Problem

The reasoning factory does not register gemma4 and its ABC cannot forward prompt preparation. Add only the bounded registry and adapter surface; served prompt wiring remains separate.

## Resolution

2026-10-11: Bounded registry/ABC repair implemented after prospective5fdbca989.
Registry RED2 and forwarding RED4 preceded GREEN37/5789, six CPU targets,
28 metadata and53 header consumers. Four compiled sensitivity mutants fail.
Root verification and independent review remain owed; serving prompt wiring
and full readiness remain pending in gemma4-e4b-google-baseline.md T03/T10.
