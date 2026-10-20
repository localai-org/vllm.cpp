ID: ISSUE-LOCAL-01NZXQ4G5ZSQDVQVF39W9H5PZD
Title: T10B: Image processing in the Gemma4 chat seam
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

T10A registered the mm chat seam but the `chat_fn` refuses all
multimodal requests. Image processing through the serving path is
not implemented.

## Resolution

Accepted — see spec section for evidence.
## Scope

Upgrade `MakeGemma4ChatSeam` in `mm_chat_gemma4.cpp` to process
image content parts through the full serving path:

1. Factory resolves `boi_token_id`, `eoi_token_id`, `image_token_id`
   from `Gemma4SpecialTokenIds` (T03B, reads config.json)
2. Factory validates `prompt_fn` and `codec` are present
3. `chat_fn`:
   a. Walks messages, extracts image_url parts and text
   b. Replaces each message's content_parts with the text-only
      content (images become their `\t<|image|>` placeholder in
      the rendered prompt)
   c. Renders the prompt via `prompt_fn`
   d. Tokenizes via `tokenizer->EncodeWithSpecialTokens`
   e. For each image: decodes via `codec`, processes via
      `Gemma4ImageProcessor`
   f. Expands each `image_token_id` in prompt_ids to
      `[boi] + [image_token] * N + [eoi]` where N = num_soft_tokens
   g. Builds `MultiModalInputs` with `MultiModalFeatureSpec` per image
4. Audio and video content parts still refuse with a clear message

## Out of scope

Audio processing in the chat path. Video frame sampling/extraction.
Real model weights. Full e2e inference. The `default_output_length`
config field (C++ Gemma4VisionConfig does not carry it; the image
processor defaults to 280 which matches the E4B model).
