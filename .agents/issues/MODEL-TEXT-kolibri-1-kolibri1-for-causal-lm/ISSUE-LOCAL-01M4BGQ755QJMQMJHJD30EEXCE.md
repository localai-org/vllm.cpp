ID: ISSUE-LOCAL-01M4BGQ755QJMQMJHJD30EEXCE
Title: kolibri-1 CPU: land the incremental production-shape bench (128-in, 64-out token anchor)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-07
Updated: 2026-10-07
Closed: 2026-10-07

## Problem

The row's only decode gate (test_kolibri1_w3) re-prefills the whole chain per step, so it cannot see the production incremental-decode geometry; a scratch bench caught the dequant-cache corruption that W3 could not, but nothing committed pins that shape. Also settle whether the morning cpu_paged_attn.cpp:165 block-table crash was a harness misconfiguration or a main bug.

## Resolution

Landed 2026-10-07: the morning crash was harness misconfiguration (8-block table for a 192-token shape), not a product bug; the committed test_kolibri1_decode_bench pins completion and the anchor token 109726. Evidence: docs/bench-evidence/kolibri1-incremental-baseline-20261007.md
