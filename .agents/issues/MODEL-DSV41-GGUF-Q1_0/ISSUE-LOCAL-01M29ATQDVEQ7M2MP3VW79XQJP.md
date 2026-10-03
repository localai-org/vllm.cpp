ID: ISSUE-LOCAL-01M29ATQDVEQ7M2MP3VW79XQJP
Title: The only DeepSeek-V4.1-Flash artifact that FITS one GB10 is degenerate by construction, so "run Q1_0 on the dgx" is unreachable with any published rung. `vcruz305/DeepSeek-V4.1-Flash-GGUF` `Q1_0` @ `543d86fd97` is 38.781 + 41.776 + 18.033 = 98.591 GiB against a 119 GiB unified pool, ~20.4 GiB clear -- and the size is the only part that works. The file stores `token_embd` ITSELF at ggml type 41, and a 1-bit block format keeps one scale per block and one sign bit per weight, so the embedding table returns plus-or-minus a single magnitude; the producer root-caused it on ggml-org/llama.cpp#28696 as "the file, not the runtime", measured 1.53 bits per weight over the backbone, and stated that Q2_K is the first rung with enough magnitude left. Q2_K is 246.349 GiB, twice this box, so across the published rungs "fits" and "can say anything" do not overlap. `deepseek41` is additionally in no released engine (absent from llama.cpp `b10451` and master; #28696 is OPEN, DRAFT and conversion-only; the runtime is an out-of-tree branch) and `vllm-gguf-plugin` has no DeepSeek adapter at all. A fitting non-degenerate rung is CONSTRUCTIBLE -- keep `token_embd` and `output` out of the 1-bit format and spend the headroom on them -- and that is a requantization question needing its own row, not scoped here
Row: MODEL-DSV41-GGUF-Q1_0
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

The developer asked that the Q1_0 rung run at least on the dgx; it fits and cannot produce meaningful output.

## Resolution

-
