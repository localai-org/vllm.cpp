ID: ISSUE-LOCAL-01M3RTV0R0TX47ZCYM4T94PP5E
Title: Serve Bespoke Labs Nimble 9B decision adapter through /v1/systemone and vllm_decide
Row: MODEL-NIMBLE
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

Ollama 0.35 serves bespokelabs/Bespoke-Nimble-9B as 'nimble' on /v1/systemone. The checkpoint is an unmerged PEFT LoRA (r=16, alpha=32, 248 target modules including the GatedDeltaNet in_proj_qkv/z/a/b and out_proj) on Qwen/Qwen3.5-9B @ c2022362. It answers each schema field with one forward: the last-position logits of the candidate letter tokens, softmax(logits / T). vllm.cpp runs the Qwen3.5 dense backbone but has no LoRA on the server path, no last-row candidate-logit readout, and no /v1/systemone lane with the openjev answer semantics (entropy confidence, unrounded probabilities, raw-JSON state) that Nimble's own server uses.

## Resolution

-
