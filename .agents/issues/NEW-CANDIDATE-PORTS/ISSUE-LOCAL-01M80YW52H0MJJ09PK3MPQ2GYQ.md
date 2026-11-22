ID: ISSUE-LOCAL-01M80YW52H0MJJ09PK3MPQ2GYQ
Title: olala W2d-1: full-forward assembly smoke on the pinned bf16 artifact
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

W2d assembles OlalaGenerateHost (embed-normalized, 36 blocks, lm_head argmax) and OlalaVAttentionStepHost on the pinned Olala-v1.0-mini-7A1B bf16 GGUF. The smoke run caught three defects that unit tests missed: (1) OlalaGenerateHost appended the last-prefill argmax plus every generation step, growing token_ids by n_gen+1; (2) the V-attention step host indexed the current-token k/v with the signal-head index h instead of the GQA-mapped kvh, reading past the KV*D arrays (silent in Release, caught by the debug-build STL assertions); (3) an empty ymix scratch segfaulted in the mixer dispatch.

## Resolution

-
