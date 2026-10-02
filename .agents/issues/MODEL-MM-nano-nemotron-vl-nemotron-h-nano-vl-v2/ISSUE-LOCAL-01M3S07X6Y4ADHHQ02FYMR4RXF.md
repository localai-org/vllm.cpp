ID: ISSUE-LOCAL-01M3S07X6Y4ADHHQ02FYMR4RXF
Title: NemotronH loader: resolve the ModelOpt MIXED_PRECISION scheme per module so the Omni NVFP4 language tower loads
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

LoadNemotronHHostWeights enumerates Nemotron 3.5 Lightning's fixed scheme (NVFP4 experts and lm_head, FP8 mamba in/out). nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 @16993199 quantizes o_proj and the shared experts to FP8, ships input_scale on every NVFP4 expert, and keeps lm_head bf16. The structural gate (test_nano_nemotron_vl_registry STRUCTURAL on that index) reads 24103 language tensors shipped, 18157 claimed, 5946 unclaimed and 60 enumerated-but-not-shipped, so the load refuses. Upstream resolves the scheme per module from quantized_layers (modelopt MIXED_PRECISION).

## Resolution

-
