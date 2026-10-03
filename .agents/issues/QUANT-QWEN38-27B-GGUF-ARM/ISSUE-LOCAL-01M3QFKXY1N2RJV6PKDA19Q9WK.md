ID: ISSUE-LOCAL-01M3QFKXY1N2RJV6PKDA19Q9WK
Title: The GDN column-permuted keep-quant path refuses by name on CUDA: kPermuteVHeads has CPU and ROCm arms only, and the op has no test anywhere
Row: QUANT-QWEN38-27B-GGUF-ARM
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

The T25 column-permuted GDN keep-quant path keeps out_proj as tiled-order Q5_K instead of expanding to bf16 (VT_GDN_COLPERM_KEEP_QUANT, src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:1136,1272; documented in docs/ENVIRONMENT.md). The path permutes the 4096-element GEMV input at runtime instead of the weight, and that gather runs through vt::PermuteVHeads from GdnOutProjMatmul (src/vllm/model_executor/models/qwen3_5.cpp:1810-1820). The op is registered on CPU (src/vt/cpu/cpu_ops.cpp:4471-4474) and ROCm (src/vt/rocm/rocm_ops.hip:242) only, so on CUDA vt::GetOp(kPermuteVHeads) refuses by name and the opt-in Q5_K arm cannot run on the one backend the released 27B GGUF arm is served from. The arm is also UNRECORDED as owed anywhere in .agents (no T25 or VT_GDN_COLPERM_KEEP_QUANT reference), which is why it is filed here rather than left to the next reader. Worse, the op has NO test on any backend: the CPU index mapping the CUDA kernel must match was itself unpinned, so a wrong mapping would have been indistinguishable from a right one. The kernel lands with this issue and the first coverage. Effect when enabled: ~4x less weight bandwidth for that GEMV (Q5_K ~5 MB vs bf16 20 MB per call, docs/ENVIRONMENT.md).

## Resolution

-
