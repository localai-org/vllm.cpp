ID: ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS
Title: Nemotron Nano VL / Omni: port the RADIO vision tower, the mlp1 projector and the dynamic-resolution image placeholder expansion
Row: MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

NemotronH_Nano_VL_V2 and NemotronH_Nano_Omni_Reasoning_V3 (vLLM nano_nemotron_vl.py @ e126687a9a) are not registered. The text backbone NemotronHForCausalLM exists, but the image path does not: no RADIO ViT (radio.py), no pixel-shuffle + mlp1 projector, no DynamicResolutionImageTiler preprocessing and no <img><image>*N</img> placeholder expansion. Audio (sound_encoder) and video are out of scope and stay refused by name.

## Resolution

-
