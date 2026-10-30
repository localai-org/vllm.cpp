ID: ISSUE-LOCAL-01M66JMQ68GXQADSAGMJMAXQYA
Title: TT P150 decode: 5.8x regression between 3ce14f35c and 3284412c0, partially repaired; residual 1.77x gap at head
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

TT-ANCHORING (2026-10-30, docs/bench-evidence/tt-anchoring-20261030/) measured the kolibri-1 P150 decode bench under pinned-performance governor and idle host: 3ce14f35c = 555.1 ms/step (reproduces the 548.5 record), the 3284412c0 readback baseline binary = 3235.8 ms/step (reproduces 3358.1), head 8d1eb0daa = 983.5 ms/step. A real ~5.8x code regression entered between 10-21 and the readback baseline (50 commits); the FIX-PAGED-REGRESSION re-gate commits (f37d01230, e5dc3de3a) plausibly repaired it to 983.5 (inferred, not bisected), leaving a real 1.77x gap. Governor and host load are excluded (both vintages reproduce on the clean host; governor swept both states, 2.8% effect). Owes: a bisect over 3ce14f35c..3284412c0 for the culprit commit(s), a confirmation bisect of the repair attribution, and ownership of the residual 983.5-vs-555.1 gap.

## Resolution

-
