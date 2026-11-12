ID: ISSUE-LOCAL-01M697KQCH9PQ3NEMAPK4WQRVK
Title: Four review-exposed checker gaps: doc-sha-vs-disk, secondary pin values, oracle-file pin multiplicity, inline-run wiring escape
Row: CHECKER-HARDENING-2
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

Four fresh-review findings: (1) no checker validates sha256 pins recorded in docs/USAGE.md or bench-evidence reports against the files on disk (FLINT-CPU-PARITY M3; kolibri1-GGUF M2); (2) reverting the llama-cpp secondary pin block to b10451 is undetected -- check-oracle-pins binds only the primary vllm authority's parity spans (LLAMA-CPP-PIN-ADVANCE M1; PIN_SURFACES ~744); (3) two oracle files claiming the same pin sha is un-checked (LLAMA-CPP-PIN-ADVANCE M4 inverse); (4) inline single-line run: steps escape the AGENT_RECORD_CHECKER_STEPS pin (CHECKER-HARDENING F1; check-test-registration.py:1035 reads only literal-block steps). Spec: .agents/specs/checker-hardening-2.md. One row, four commits, each with its own red-first capture; any item needing beyond the stated shape splits out.

## Resolution

-
