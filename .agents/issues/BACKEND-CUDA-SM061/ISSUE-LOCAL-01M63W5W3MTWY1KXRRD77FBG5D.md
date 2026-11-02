ID: ISSUE-LOCAL-01M63W5W3MTWY1KXRRD77FBG5D
Title: Load SmolLM and Llama GGUF tokenizer metadata
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

Published SmolLM2 and TinyLlama GGUF checkpoints are refused before CUDA completion because GGUF tokenizer smollm pretokenizer and llama SentencePiece metadata are unimplemented.

## Resolution

-
