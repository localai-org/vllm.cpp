ID: ISSUE-LOCAL-01M5JWVYRWXXKW994ZQ0BBCRZA
Title: Flint support: port RecursiveSI/flint (FlintForCausalLM) — CPU and GGUF arms
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-23
Updated: 2026-10-23
Closed: -

## Problem

Flint (RecursiveSI/flint @ 2a247be0c0aec895ccb0ac7ca92229ec6df37288, published 2026-10-21) is unimplemented in vllm.cpp. vLLM has no flint code; llama.cpp has it only in open PR ggml-org/llama.cpp#30928 (the model author's own, head 1b44e30775dcb2f93952116b05f89e5bd797fb00, MERGEABLE, CI green); transformers has it only on main (unreleased). The quantized artifacts are fetched and sha256-verified on this box (unsloth/flint-GGUF @ 6c2910f97fb36b5cb22bbdc8f014ab42656b0d14: UD-IQ1_M and UD-Q4_K_XL). The peer (mistral.rs, oc) has started a Flint implementation on the mistral.rs side. Spec: .agents/specs/flint-cpu-gguf.md (written with this issue; the row is declared there). First wave: the llama-cpp-flint oracle record (pin 1b44e30 with a gateability measurement), the FlintForCausalLM model TU, and the flint GGUF loader row. The Tenstorrent arm is a separate later row.

## Resolution

-
