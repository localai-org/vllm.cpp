ID: ISSUE-LOCAL-01M4FA8QB4RE33B14KGHK699S4
Title: Public ASR loader leaves combined diarization without a context
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

Source inspection at 124d484155b2a9df55c45be561ddd85d5f4a864d: src/capi/vllm_c.cpp:873-905 loads ParakeetTranscriber from an HF directory, then passes that directory to parakeet_capi_load without checking its result. CMakeLists.txt:1669-1671 pins parakeet.cpp 394d270fabb1d6125f05c772aa3ca078a574b19d; its src/parakeet_capi.cpp:140-164 requires a GGUF file, and :1057 returns null when the ASR context is null. Our src/capi/vllm_c.cpp:1722-1724 and :1788-1790 convert null combined results to VLLM_OK with zero utterances. Combined transcription is therefore unavailable through the normal public HF ASR load. Evidence is CPU-only source review, not an end-to-end audio run. Current authority covers documentation only. A runtime resolution needs separately scoped model loading changes and tests for valid combined output and failed context creation; do not fix runtime in this audit.

## Resolution

-
