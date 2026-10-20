ID: ISSUE-LOCAL-01WKNP2PV339VJ6MPZ2V2QKMB1
Title: T09: Wire hooks and runner — flip mm_inputs_ready
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-20
Closed: 2026-10-20

## Problem

The Gemma4 multimodal path is fully implemented at the component level:
`EncodeMmGemma4` (image), `EncodeMmGemma4Audio` (audio), and
`EmbedMmGemma4` (merge) are all implemented and tested in isolation.
The `kGemma4MmHooks` factory sets `encode_mm` and `embed_mm` to these
functions.

However, `kGemma4MmHooks.mm_inputs_ready` is `false`
(`gemma4_audio_encoder.cpp:53`). The runner gates multimodal dispatch on
`ModelRegistry::SupportsMmInputs()`, which requires all three:
`mm_inputs_ready && encode_mm != nullptr && embed_mm != nullptr`.
Since `mm_inputs_ready` is false, the runner skips all multimodal
processing. The API path never processes audio/image/video.

## Resolution

Accepted — see spec section for evidence.
## Scope

Flip `mm_inputs_ready` from `false` to `true` in `kGemma4MmHooks`
so the runner dispatches multimodal inputs through the implemented
encode_mm → embed_mm → forward path.

Update the four existing `CHECK_FALSE(SupportsMmInputs(model))`
assertions that were sentinels for the gate being closed:

- `test_gemma4_e4b_loader.cpp:1110` — audio encoder test
- `test_gemma4_e4b_loader.cpp:1345` — audio merge test
- `test_gemma4_e4b_loader.cpp:1402` — readiness contract test
- `test_gemma4_e4b_vision_execution.cpp:103` — vision execution test

These become `CHECK(SupportsMmInputs(model))` because the model now
advertises MM support through the hooks.

The readiness contract test block
(`test_gemma4_e4b_loader.cpp:1404-1414`) tests the contract logic with
a default factory and stays unchanged.

Add a new RED-first test case that proves the factory path
(`kGemma4Factory`) carries the flipped flag through to
`SupportsMmInputs`. This test constructs a model through `kGemma4Factory`
(the factory the registry uses) and asserts
`SupportsMmInputs` returns true.

## Out of scope

Serving integration (T10). Full e2e API path with real weights.
Video frame sampling/extraction. Span overlap/provenance.
Original weights inference and per-capability quality gates.
Production device audio/BF16.
