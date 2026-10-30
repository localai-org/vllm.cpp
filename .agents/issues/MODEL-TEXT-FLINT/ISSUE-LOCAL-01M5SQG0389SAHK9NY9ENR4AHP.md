ID: ISSUE-LOCAL-01M5SQG0389SAHK9NY9ENR4AHP
Title: the Flint synthetic gate cannot detect a mis-sized attention output or an embedding dtype corruption
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-25
Updated: 2026-10-25
Closed: 2026-10-25

## Problem

The wave-1 synthetic forward test used a config where hq*head_dim < hidden_size, so the t*h attn_out mis-sizing (heap overflow, ISSUE-LOCAL-01M5Q1PS772CPTMYJ02AMVZ3Y0) was invisible, and it never checked that the embedding lookup widens bf16 bits rather than memcpy-ing them into the f32 stream (ISSUE-LOCAL-01M5QBGBM3Z7NT3D9TA0K1HT7T). Both defects are fixed on main by the wave-2 change; this issue owes the regression tests that fail on the pre-fix code, proven red by mutation.

## Resolution

Closed 2026-10-25: the two detector cases land in commit 58a7704d1 on row/FLINT-GATE-HARDEN, each proven red against its own fix-site mutation (docs/bench-evidence/flint-gate-harden-20261025/: mutation A SIGABRT, mutation B 16/16 isolated mismatches) and green on main.
