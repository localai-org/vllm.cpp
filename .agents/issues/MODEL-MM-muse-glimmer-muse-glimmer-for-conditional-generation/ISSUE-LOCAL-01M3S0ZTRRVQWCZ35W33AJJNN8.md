ID: ISSUE-LOCAL-01M3S0ZTRRVQWCZ35W33AJJNN8
Title: Muse Glimmer parity: a first reference run of the vision path on real tensors, and the GGUF text path token-exact against llama.cpp
Row: MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

docs/models/muse-glimmer.md records that the perception encoder has no reference check of any kind, that the GGUF k-quant agrees with llama.cpp on the first token and then diverges (quant drift or a second defect is open), and that the pinned vLLM oracle cannot load muse_glimmer. The last is stale: the parity pin e126687a9a registers MuseGlimmerForConditionalGeneration (vllm/model_executor/models/muse_glimmer.py). This issue owns: the vision tower, adapter and projection checked on the released tensors against a transcription of the pinned vLLM; the GGUF text greedy decode compared with the pinned llama.cpp oracle (b10451) on the same file with the divergence classified; and the record correction. The bf16 token gate against the pinned vLLM on a GPU lease stays owed.

## Resolution

-
