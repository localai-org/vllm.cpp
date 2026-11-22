ID: ISSUE-LOCAL-01M8116014KEPHC3R26VTVZDQG
Title: olala bf16 token gate: first generated token diverges from llama-cpp-olala
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

Greedy decode on the pinned bf16 artifact diverges from the llama-cpp-olala oracle at the first generated token (705 vs 151644) and degrades into repeated 705. Evidence and both id sequences recorded in docs/bench-evidence/new-candidate-ports-olala-20261122/token-gate-bf16-20261122.txt. The fork golden harness is in the same directory. Config-level hypotheses already ruled out: final_norm, embedding normalization, prompt tokenization (now pinned to the real fork-tokenized ids).

## Resolution

-
