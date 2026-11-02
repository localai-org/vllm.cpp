ID: ISSUE-LOCAL-01M63MYTWBYSRRCEMAY20QR08M
Title: Pin independent branding headers and legacy version
Row: ENG-LUMENENGINE-BRAND
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

The branding consumers include both public headers, so removing the forwarding include remains green. The command checker compares only alias equality, so changing both legacy version strings remains green. Independent review requires standalone header consumers and a historical version pin.

## Resolution

-

## Evidence

29 October 2026: standalone header callers and a historical version pin repair
both review findings. [Mutation and packaging evidence](../../evidence/lumenengine-brand/review-repair.md)
records red and green results. Independent review and landing remain pending.
