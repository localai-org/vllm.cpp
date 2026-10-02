ID: ISSUE-LOCAL-01M3RZ6SFVDJRX1Z5GZG1CKJ7G
Title: hf_config: a config.json carrying the Python JSON literal Infinity fails to parse
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

LoadHfConfig uses strict nlohmann JSON. HuggingFace configs are written by Python json.dump, which emits Infinity/NaN for non-finite floats. The released nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 config.json (rev ca9543b1) carries llm_config.time_step_limit = [0.0, Infinity], so the engine fails with a JSON parse error before any architecture code runs. transformers (json.loads) accepts it. Found while adding tests/vllm/models/fixtures/nemotron_nano_vl_v2_12b/config.json on feat/nemotron-omni-vl; not fixed there because it is outside that item's scope.

## Resolution

-
