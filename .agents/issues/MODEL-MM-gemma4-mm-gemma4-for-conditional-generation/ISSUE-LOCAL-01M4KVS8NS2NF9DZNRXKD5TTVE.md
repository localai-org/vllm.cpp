ID: ISSUE-LOCAL-01M4KVS8NS2NF9DZNRXKD5TTVE
Title: Gemma4 untied loader accepts absent or malformed output head
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: 2026-10-10

## Problem

LoadImpl accepts tie_word_embeddings=false without lm_head.weight. Forward then silently uses embeddings. Malformed head dimensions and nonboolean tie flags are also accepted. CPU synthetic red gate reproduces these cases; pinned vLLM gemma4.py constructs a vocab_size by hidden_size ParallelLMHead and ties only when configured.

## Resolution

10 October 2026: bounded output-head fix rejects missing/malformed untied heads and invalid tie flags. Real CPU loader gate passed 8 cases, 38 assertions, zero skips after expanded red with 8 failed assertions. Positive controls cover both loader overloads and BF16 transpose. Root verification and independent review of the implementation commit remain required. Parent milestone remains OPEN for tower/projector accounting and runtime gates.
