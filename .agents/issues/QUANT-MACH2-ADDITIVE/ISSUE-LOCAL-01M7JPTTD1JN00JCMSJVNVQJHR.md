ID: ISSUE-LOCAL-01M7JPTTD1JN00JCMSJVNVQJHR
Title: No instrument records the Mach-2 logits, so the int5 head decode and the model output are ungated
Row: QUANT-MACH2-ADDITIVE
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: -

## Problem

The acceptance gate designed for this row reads next-token logits and per-layer activations instead of a token stream, because the model card states it loops off T=1.0/top_p=0.95/top_k=20 and a sampled stream is not reproducible. VT_Q4EXP_LAYER_FP already taps 14 per-layer tensors, but it stops at 'out': the logits are produced in qwen4_exp_registry.cpp and no instrument writes them. So the lm_head decode - int5 with group 64 - has no model-level gate at all, and neither does the final projection.

## Resolution

-
