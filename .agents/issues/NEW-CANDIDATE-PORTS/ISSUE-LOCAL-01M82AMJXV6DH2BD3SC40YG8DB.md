ID: ISSUE-LOCAL-01M82AMJXV6DH2BD3SC40YG8DB
Title: Q5_1 block_bytes tabulated as 21 instead of 24 in the GGUF reader
Row: NEW-CANDIDATE-PORTS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: 2026-11-22

## Problem

gguf_reader.cpp FindGgmlTraits case 7 tabulates Q5_1 as {32, 21} but llama.cpp ggml-common.h:237-249 defines block_q5_1 = f16 d (2) + f16 m (2) + uint8_t qh[4] (4) + uint8_t qs[QK5_1/2=16] (16) = 24 bytes (static_assert sizeof(block_q5_1) == 2*sizeof(ggml_half) + sizeof(uint32_t) + QK5_1/2). Every Q5_1 tensor nbytes is therefore ~12.5%% short (3 bytes per 32-element block, so R*3 bytes per R rows); consumer dequant that computes offsets from the tabulated nbytes reads shifted spans. GgufFile::Open still succeeds because the tensor span check only tests the upper bound against the data section. Correct case 7 to {32, 24} and update the two assertions (tests/vllm/models/test_inkling2_loop_gguf.cpp, tests/vllm/test_gguf.cpp). Found by the burn-vllm peer while cross-checking notes on PR localai-org/vllm.cpp#3586 (issue ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8); the W0 tensor walk's offsets came from the GGUF header directly and remain the trusted source.

## Resolution

FIXED 2026-11-22. `FindGgmlTraits` case 7 now returns `{32, 24, "Q5_1"}` with
the ggml-common.h:237-249 citation; the two test assertions were updated to 24
and a new assertion checks every type-7 tensor's `nbytes` against the real
artifact's header offsets (`numel/32 * 24`). Evidence:
`docs/bench-evidence/new-candidate-inkling2-q5fix-20261122/summary.md` and
`focused.log` — `ctest -R '^(test_inkling2_loop_gguf|test_gguf)$'` 2/2 passed;
`test_inkling2_loop_gguf` 11/11 cases, 86/86 assertions; 77-style skip intact.
The W0 spec's geometry statements were corrected from 21 to 24 in the same
change. Linkage: ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8 (the original W0 issue)
and PR localai-org/vllm.cpp#3586.
