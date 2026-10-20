ID: ISSUE-LOCAL-01M5CRZ3J4CFX1KEJD6ZRE4A8R
Title: Cross-check kolibri1 CPU gates against upstream llama.cpp 62318cd57
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-20
Updated: 2026-10-20
Closed: 2026-10-20

## Problem

Our kolibri1 correctness record has two references (primary aleph-alpha-inference, torch transcription). Upstream llama.cpp merged Kolibri-1 support (62318cd57, #30673, 2026-10-17); the records need a third independent reference: pin it, gate it on the released checkpoint, and measure three-way agreement. No product code changes.

## Resolution

2026-10-20: llama.cpp 62318cd57 built CPU (clean tree), converted /mnt/models/Aleph-Alpha/Kolibri-1 fp8 to bf16 GGUF (903 tensors, 149069.2 MiB, sha256 76ddb05e458cce1a023d42cd4f13dc8e8152fbb6501c1f680918fc7d2a0fcefd), served and generated end-to-end (gateable). Three-way: llama.cpp top-1 vs primary 198/202 = 98.0% over the 202 aligned positions, identical to our committed production bf16-dot number; llama.cpp chain vs golden 126/256 = 49.2% (3/8 token-exact, near-tie re-ranks). Recorded in docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md; oracle pin advanced to 62318cd57 in .agents/oracles/llama-cpp.md.
