ID: ISSUE-LOCAL-01M4VH638BS1P682RC4QMQD22Z
Title: Read Gemma-4 special token IDs from config.json
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-13
Updated: 2026-10-13
Closed: -

## Problem

The C++ code does not type the Gemma-4 multimodal special token IDs (audio_token_id, image_token_id, video_token_id, boa/boi/eoa/eoi_token_id) from config.json. These fields exist in the real Google config.json fixture and are read by the Python model code (gemma4_mm.py) from config attributes. The C++ HfConfig struct has no typed fields for them; consumers access config.raw ad-hoc or receive them as function parameters without validating against config.json. There is no test that verifies the C++ loader reads and exposes these IDs with the correct values from the real fixture.

## Resolution

-
