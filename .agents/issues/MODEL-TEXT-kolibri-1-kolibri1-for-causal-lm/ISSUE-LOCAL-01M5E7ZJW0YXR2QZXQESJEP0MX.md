ID: ISSUE-LOCAL-01M5E7ZJW0YXR2QZXQESJEP0MX
Title: Record the measured Kolibri-1 CPU performance and re-baseline the public benchmark index
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The row's public performance record does not exist: docs/BENCHMARKS.md has no Kolibri-1 CPU entry, docs/benchmarks/ has no Kolibri-1 detail page, and the dequant-cache default flip (PR #3488) leaves its decode and prefill benchmark rows owed to a node-3 window. The 12 tok/s campaign target has no published disposition, and the row spec has no Outcome section, so the tree carries no quotable number for the shipped Kolibri-1 CPU configuration.

## Resolution

2026-10-21: the row's public performance record now exists. docs/benchmarks/kolibri1-cpu-dequant-cache-default.md (new public page, with the weights record: Aleph-Alpha/Kolibri-1 @ 5bb7075ba22db733e3834b1c71d2245537cc14ed, 32 shards, 74 GiB; the bf16 sibling named as refused) plus its index row in docs/BENCHMARKS.md and a Kolibri-1, CPU section carrying the decode, prefill, thread-scaling and spec-decode rows; the dated evidence in docs/bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md; the spec's ## Outcome section. The campaign's 12 tok/s target is recorded as MET by the shipped default (spec decode 12.74-13.14 tok/s against 6.27-6.30 with the dequant cache off), with the host-specificity, the 4% noise floor and the 16 GiB-leg trade-off recorded beside it.
