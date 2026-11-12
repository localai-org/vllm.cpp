ID: ISSUE-LOCAL-01M6AS6AFMM9G0D76SKPJGT9SH
Title: Gemma4 compressed-tensors F16 checkpoint fails in dense embedding loader
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

Lumen-3.8-Pulsar-it-qat-w4a16-ct declares dtype float16 and contains F16 embeddings. CPU CLI rejects the embedding in MaterializeBf16Source before loading mixed W4A16/W8A16 packed projections. Verify configured dtype, packed weights and production decode.

## Resolution

2026-11-01: The production CPU CT loader and registered forward accept F16 parameters plus compact mixed INT4/INT8 projections. Fresh mutation review PASS at d3e529b28; independent complete upstream CPU build and three focused tests pass. Operator rerun on the newly built library matches installed llama.cpp for all sixteen greedy tokens from [2,9259]; original no-BOS CLI workload also matches sixteen tokens. Canonical spec gemma4-ct-cpu-repack.md records artifact hashes, numerical limits and gate evidence. Lands in the authorized fork main integration; upstream PR3540 remains Draft.
