ID: ISSUE-LOCAL-01M3RWMQD1RBNEGWZZRDSFTEPR
Title: test_capi 'vllm_gliner_ner runs end-to-end on the fixture' fails: vllm_engine_load returns 2 on tests/vllm/models/fixtures/gliner2_e2e
Row: MODEL-GLINER25
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

On origin/main b45a94273, CPU Release build (CUDA/HIP/Metal/Vulkan/TT off), tests/capi/test_capi.cpp:3003 fails: REQUIRE( vllm_engine_load(&mp, &eng) == VLLM_OK ) with values 2 == 0 for the gliner2_e2e fixture; vllm_last_error() reads only '1'. The other 72 test_capi cases pass. Reproduced 2026-09-30 in a clean worktree at that commit while verifying MODEL-NIMBLE; not caused by that change. Cause not diagnosed.

## Resolution

-
