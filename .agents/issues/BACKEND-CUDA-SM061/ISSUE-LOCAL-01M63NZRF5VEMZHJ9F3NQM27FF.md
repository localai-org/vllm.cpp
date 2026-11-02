ID: ISSUE-LOCAL-01M63NZRF5VEMZHJ9F3NQM27FF
Title: Pin the legacy CUDA attention metadata contract
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

Independent mutation review found head geometry, block geometry, KV connector refusal and bfloat16 KV acceptance can change without failing legacy attention tests.

## Resolution

29 October 2026: add three registered-selector cases for geometry, storage,
and unsupported request modes. The complete host suite passes 25 cases and
148 assertions. Scratch predicate mutations fail the new checks. Evidence:
`.agents/evidence/pascal-volta-enable/metadata-repair/`.

The implementation is reviewable on the task branch. This issue stays OPEN
until the change lands. Full CUDA integration and real-model gates remain
separate obligations.
