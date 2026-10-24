ID: ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E
Title: The flint GGUF arm: the published containers load and their greedy chains replay
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-24
Updated: 2026-10-24
Closed: -

## Problem

MODEL-TEXT-FLINT W1 (commit ea5c3dade, spec .agents/specs/flint-cpu-gguf.md) shipped the FlintForCausalLM registry, the CPU reference forward, and a synthetic-config gate, and its ## Owed names two gaps this row closes: the flint GGUF arm (the loader for the published containers) and the real-weights token gate (W1's gate ran on a synthetic config because the safetensors are on no box here). Every published container carries general.architecture = flint, and the top-level GGUF dispatch refuses them all as unsupported — a capability the published files name and no production entry point reaches. The llama-cpp-flint oracle (PR #30928 head 1b44e30775dcb2f93952116b05f89e5bd797fb00, pinned in .agents/oracles/llama-cpp-flint.md) demonstrably builds and runs the containers, so it is gateable and supplies the golden greedy chains the token gate replays.

## Resolution

-
