ID: ISSUE-LOCAL-01M4FKVQSGAMQPES9YZ2WEFEK6
Title: test_capi v27 red in every CPU lane: vllm_engine_load of the gliner2 fixture fails on the 0-head derivation
Row: MODEL-GLINER25
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): tests/capi/test_capi.cpp:3003 'capi v27: vllm_gliner_ner runs end-to-end on the fixture' fails REQUIRE( vllm_engine_load(&mp, &eng) == VLLM_OK ) with the loader returning 2 (72 of 73 capi cases pass; only this one fails). The case loads the shared gliner2_e2e fixture (GLINER2_E2E_FIXTURE_DIR) through the C ABI; the engine load reaches LoadGliner2Weights -> InferEncoderParams, which derives num_attention_heads = hidden_size / 64 = 24 / 64 = 0 for the 24-wide fixture, so deberta_v2::Load throws 'hidden_size and num_attention_heads must be positive' and vllm_engine_load reports failure. Same root cause as the test_gliner2_e2e red state (tracked in its own issue); this issue tracks the C-ABI gate's red state. Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md). Fix: the shared InferEncoderParams repair (honor config.num_attention_heads when > 0, keep the /64 fallback); no C-ABI change needed.

## Resolution

CLOSED 2026-10-09 as a DUPLICATE of ISSUE-LOCAL-01M3RWMQD1RBNEGWZZRDSFTEPR (filed 2026-09-30, Row MODEL-GLINER25), which already tracked this exact test_capi.cpp:3003 failure. The diagnosis and the dated fix evidence are recorded on the older issue, which the same InferEncoderParams repair resolves.
