ID: ISSUE-LOCAL-01GQ7HA9V4NTMVBB639P3ECRMP
Title: T10A: Register Gemma4 mm chat seam (refusing stub)
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

T09 flipped `mm_inputs_ready` to `true`, so the server now treats
Gemma4 as a multimodal model and calls
`MultiModalChatRegistry::MakeSeam(ctx)`. No registration exists for
`Gemma4ForConditionalGeneration`, so `RaiseForUnregistered` throws
and the catch block installs a generic refusing seam.

A registered factory with a Gemma4-specific refusal message is the
prerequisite for all serving-level multimodal work. It validates the
install context (tokenizer, config, prompt renderer, image codec)
and returns a seam that refuses multimodal requests with a clear
message naming the architecture and the pending modalities.

## Resolution

Accepted — see spec section for evidence.
## Scope

Create `src/vllm/entrypoints/openai/mm_chat_gemma4.cpp` with:
- A `MakeGemma4ChatSeam` factory that validates the install context
  and returns a `MultiModalChatSeam` whose `chat_fn` refuses
  multimodal requests with a clear message
- `REGISTER_VLLM_MM_CHAT(gemma4, "Gemma4ForConditionalGeneration", ...)`
- Add the source to `tests/CMakeLists.txt`

## Out of scope

Image/audio/video processing in the chat path. Full serving
integration. Real model weights. Placeholder expansion.
