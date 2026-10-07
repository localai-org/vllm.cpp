ID: ISSUE-LOCAL-01M47JKNK9RCZVDKSJ98820KY9
Title: Document the Kolibri1 CPU golden gate and remaining limits
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-06
Updated: 2026-10-06
Closed: 2026-10-06

## Problem

Public Kolibri1 documentation omits the committed W3 comparison, misstates dequantization timing, and lacks checkpoint limitations. README repeats a stale architecture count. W3 has no committed verdict, so parity cannot be promoted.

## Resolution

2026-10-06: README, FEATURES, and USAGE now describe the CPU forward, committed golden comparison, tokenizer blocker, and missing parity verdict. Source kolibri1_forward.cpp:65-113 proves per-projection dequantization. test_kolibri1_w3.cpp:237-244 skips execution without real weights; docs/bench-evidence/kolibri1-goldens-20261004.md has no committed verdict. Removed stale repeated README architecture counts. check-readme-structure.py, check-supported-models.py, and check-benchmark-index.py pass. No model or GPU execution; full preflight retains baseline tooling and fusion failures.
