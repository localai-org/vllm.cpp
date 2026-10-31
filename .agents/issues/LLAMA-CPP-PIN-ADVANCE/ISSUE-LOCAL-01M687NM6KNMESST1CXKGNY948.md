ID: ISSUE-LOCAL-01M687NM6KNMESST1CXKGNY948
Title: Advance the stock llama-cpp oracle pin past the Kolibri-1 support merge, reconciling every b10451 consumer
Row: LLAMA-CPP-PIN-ADVANCE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

The llama-cpp oracle pins stock release b10451, which predates Kolibri-1 support (#30673, 2026-10-17) and the router-precision removal (#30865, 2026-10-20 -- numerics-relevant). The KOLIBRI1-GGUF-ARM row named the advance as its follow-up so a quant-matched floor comparison becomes possible: llama.cpp serves the mudler APEX GGUFs directly (standard ggml types). Per upstream-sync: advance ONLY after reconciling every affected row and gate; the advanced pin must build and run the floor models on aarch64. Deliverables: first post-merge release tag identified and built; llama-cpp.md pin block + snapshot surfaces updated; every b10451 consumer inventoried and dispositioned (re-measure or re-anchor with dated note); gateability legs (kolibri1 APEX + one pre-existing floor model); the quant-matched kolibri1 comparison if it fits, else the named next step. Spec: .agents/specs/llama-cpp-pin-advance.md.

## Resolution

-
