ID: ISSUE-LOCAL-01M57MEBPZB5WP00GX1DWKMWDV
Title: BENCHMARKS.md carries two benchmark index tables; the checker red on main blocks every PR
Row: ENG-PUBLIC-DOC-PROJECTIONS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: 2026-10-18

## Problem

Commit 8111eea55 (PR #3339) added the '## Index' table required by check-benchmark-index.py without removing the pre-existing '## Benchmark detail index' table. The two tables list the same 16 benchmark IDs, so the checker reports 16 duplicate-ID errors on origin/main itself. The agent-record CI job runs the checker, so every open PR is red regardless of content. Repair: keep one table, carry the unique row (qwen38-tensorfold-gap) the old table holds, rerun the checker green.

## Resolution

2026-10-18: kept the #3339 '## Index' table, carried the unique qwen38-tensorfold-gap row into it, removed the duplicate 'Benchmark detail index' section. check-benchmark-index.py: 16 duplicate-ID errors on main -> OK on c4e9b2a87; check-site.py OK; check-agent-record.py OK. Landed as PR #3478.
