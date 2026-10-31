ID: ISSUE-LOCAL-01M687NM6KNMESST1CXKGNY948
Title: Advance the stock llama-cpp oracle pin past the Kolibri-1 support merge, reconciling every b10451 consumer
Row: LLAMA-CPP-PIN-ADVANCE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

The llama-cpp oracle pins stock release b10451, which predates Kolibri-1 support (#30673, 2026-10-17) and the router-precision removal (#30865, 2026-10-20 -- numerics-relevant). The KOLIBRI1-GGUF-ARM row named the advance as its follow-up so a quant-matched floor comparison becomes possible: llama.cpp serves the mudler APEX GGUFs directly (standard ggml types). Per upstream-sync: advance ONLY after reconciling every affected row and gate; the advanced pin must build and run the floor models on aarch64. Deliverables: first post-merge release tag identified and built; llama-cpp.md pin block + snapshot surfaces updated; every b10451 consumer inventoried and dispositioned (re-measure or re-anchor with dated note); gateability legs (kolibri1 APEX + one pre-existing floor model); the quant-matched kolibri1 comparison if it fits, else the named next step. Spec: .agents/specs/llama-cpp-pin-advance.md.

## Resolution

- 2026-10-31, closed by the row's implementation on `row/LLAMA-CPP-PIN-ADVANCE`.
  Tag identified: release `b11765` = `9df0b73c707900b7277a46b5ae52ff7e7e567d13`
  (the #30865 merge commit; carries #30673; compare-API + fresh-clone verified,
  `docs/bench-evidence/llama-cpp-pin-advance-20261031/tag-identification.md`).
  Built clean on this aarch64 host (336 targets, GGML_NATIVE=ON, binary sha256
  recorded); gateability legs: kolibri1 APEX I-Micro loads and generates
  greedily, Qwen3.8-27B Q4_K_M loads and generates, flint correctly refuses at
  the stock pin. Pin advanced in `.agents/oracles/llama-cpp.md` with every
  consumer dispositioned (`reconciliation.md`): stale current-state surfaces
  re-stated, attribution-only surfaces kept, scoped `llama-cpp-kolibri1`
  re-anchored with a dated note. Quant-matched comparison taken: llama.cpp
  17.98/20.18 tok/s (16/32t) vs our arm 4.97/6.55 on the same I-Micro artifact,
  zone and threads - the floor is ~3x ahead, recorded not adjudicated. Neither
  stop condition fired.
