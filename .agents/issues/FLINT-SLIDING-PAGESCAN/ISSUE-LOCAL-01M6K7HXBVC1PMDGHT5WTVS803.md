ID: ISSUE-LOCAL-01M6K7HXBVC1PMDGHT5WTVS803
Title: Sliding attention cost tracks pages scanned, not window-resident keys -- diagnose the residual 1.11x growth and fix the attributed mechanism
Row: FLINT-SLIDING-PAGESCAN
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-04
Updated: 2026-11-04
Closed: -

## Problem

Attention is 50.3% of the Flint keep-quant serving step (sliding 37.0% + full 13.3%), and sliding attention grows with sequence length despite the bounded window: 1.94x growth confirmed twice (implementer + fresh reviewer), reduced to a residual 1.11x by FLINT-SLIDING-ATTN's (token,head)-unit parallel decomposition -- but the residual still grows. The window math is bounded; the hypothesis is that cost tracks PAGES SCANNED, not window-resident keys: the btab page-table indirection is computed per key inside the per-token loop (cpu_paged_attn.cpp:372-374) and the window spans ~17 pages at late sequence vs ~1 early. Three mechanisms to discriminate: M-A the per-key btab walk (hoistable, bit-identical), M-B arena scatter (counter evidence needed), M-C timer scope including append/RoPE (behavior-neutral fix). Spec: .agents/specs/flint-sliding-pagescan.md

## Resolution

-
