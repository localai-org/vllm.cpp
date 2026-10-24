ID: ISSUE-LOCAL-01M5PSMQF5F4HNV86QZNPGHFMK
Title: The unsupported-architecture message test pins a stale literal without FlintForCausalLM
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

tests/vllm/test_model_loader_gguf.cpp's "FromModelDir rejects an unknown dense architecture before loading" case compares the registry's supported-architecture list against a string LITERAL that predates W1's FlintForCausalLM registration (commit ea5c3dade): the literal lists every registered architecture except FlintForCausalLM, so the CHECK_THROWS_WITH_AS equality fails on any tree that includes the flint registry TU. Found red at the base of row/MODEL-TEXT-FLINT-GGUF (base 7f842ee9b3c7) while running the flint GGUF dispatch gate; pre-existing on origin/main, not caused by this branch. Fix: add 'FlintForCausalLM' to the expected literal in its canonical sort position.

## Resolution

-
