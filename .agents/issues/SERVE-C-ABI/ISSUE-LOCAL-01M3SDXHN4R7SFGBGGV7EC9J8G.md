ID: ISSUE-LOCAL-01M3SDXHN4R7SFGBGGV7EC9J8G
Title: CI configure fails: parakeet.cpp FetchContent uses GIT_TAG main, the repository branch is master
Row: SERVE-C-ABI
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

ABI v30 (2833d6300) made VLLM_CPP_WITH_DIARIZATION ON by default and fetches mudler/parakeet.cpp with GIT_TAG main. That repository has no main branch (its default branch is master), so every CI lane fails at configure with 'fatal: invalid reference: main' (run 36726734816). A moving branch is also not a reproducible pin: parakeet.cpp master reworked parakeet_capi_transcribe_and_diarize and parakeet_capi_free_sas_results after the seam was written, so the seam does not compile against master either.

## Resolution

2026-09-30: FetchContent now pins parakeet.cpp to 394d270fabb1d6125f05c772aa3ca078a574b19d, the feat/diarization-sas head (ABI v8) that the seam fix a1884b809 compiled against; master 623a968 fails to compile diarization.cpp (transcribe_and_diarize and free_sas_results signatures changed). The same block also had three defects that the configure failure hid: dr_wav.h was not found as a subproject, ggml was linked shared (libvllm.so NEEDED libggml*.so.0 through a build-tree RUNPATH), and GGML_NATIVE forced -march=native. All three are fixed. Evidence on x86-64 CPU: default configure red before ('fatal: invalid reference: main'), green after; libvllm.so + vllm-server build with diarization ON (vllm_diarize_* exported, no ggml NEEDED) and OFF (stubs), and with VLLM_CPP_PARAKEET_CPP_DIR.
