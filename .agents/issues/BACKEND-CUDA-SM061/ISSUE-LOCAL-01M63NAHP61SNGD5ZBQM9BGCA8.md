ID: ISSUE-LOCAL-01M63NAHP61SNGD5ZBQM9BGCA8
Title: Reconcile legacy CUDA active-row gate records
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

The legacy CUDA campaign activates SM060, SM061, and SM070 but omits their runnable gate baseline, and the SM070 active row has no matching campaign commit evidence in the live audit.

## Resolution

29 October 2026: the expected runnable inventory now includes all three shared-spec rows.
The campaign record commit names SM060, SM061, and SM070, so the live audit can
trace the unmerged work. Both focused suites pass. Each inventory-deletion
mutation fails the existing exact-set assertion. Evidence:
[record repair](../../evidence/pascal-volta-enable/record-repair.md).
Independent review and landing remain pending.
