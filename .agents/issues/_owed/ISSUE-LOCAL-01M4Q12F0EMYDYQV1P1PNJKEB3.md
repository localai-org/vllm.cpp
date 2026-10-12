ID: ISSUE-LOCAL-01M4Q12F0EMYDYQV1P1PNJKEB3
Title: Align Kolibri-1 public documentation with current code
Row: -
State: OPEN
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-12
Updated: 2026-10-12
Closed: -

## Problem

The feature row still describes dequant-at-load and lists the TT arm as owed. README news and USAGE omit the reachable Kolibri-1 path and its validation limits.

## Resolution

2026-10-12: The documentation branch corrects the loader and device claims,
adds CPU usage, and separates partial token agreement from full plugin parity.
The five scoped checkers and their 117 mutation tests pass. The issue remains
open pending independent review and upstream landing.
