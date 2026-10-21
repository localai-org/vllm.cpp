ID: ISSUE-LOCAL-01M2YXN1QEMAEY5W8QCKH76HTS
Title: TT: native BFP4/BFP8 weight residency for the 27B-class decode
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-20
Updated: 2026-10-21
Closed: -

## Problem

The f32-exact decode stack on the Tenstorrent P150 reaches ~0.028 tok/s on Qwen3.8-27B-APEX-I-Nano (B2, token-exact vs the llama.cpp b10451 greedy oracle), while Tenstorrent's native tt-metal pipeline reports ~50 tok/s on a single P150A with BFP4 weights / BFP8 KV / BF16 deltaNet state — a ~1800x decode gap. The root difference is FORMAT, not tuning: our path block-dequants GGUF weights and runs every decode GEMM through the SFPU f32-exact path (ttnn matmul truncates f32 operands to tf32 on Blackhole, measured; ComputeConfig does not lift it), while the native pipeline runs tensor-core BFP matmul where BFP precision IS the hardware's precision contract. This row commits the native BFP4/BFP8 weight-residency path: load-time conversion and device-resident ttnn matmul, with a near-tie correctness gate against the b10451 oracle on the quantized model.

## Resolution

2026-10-21: STILL TRUE. The native BFP4/BFP8 path is still not implemented.
`include/vllm/config/tt_weight_residency.h:10` keeps bfp4 in the reserved list
("RESERVED, not implemented; refused by name at the staging seam"), and
`src/vt/tenstorrent/tenstorrent_residency.cpp:605` still refuses it by name
("VT_TT_WEIGHT_RESIDENCY=bfp4: not implemented; staged ...").

The row's spec `.agents/specs/tenstorrent-bfp-weight-residency.md` is ACTIVE:
wave 1 (BFP8 weight residency) is merged, and its A/B identified per-call
compute as the bottleneck, but the remaining gap is still FORMAT -- our path
block-dequants GGUF weights and runs every decode GEMM through the SFPU
f32-exact path, where the native pipeline runs a tensor-core BFP matmul.

What settles it: the load-time conversion plus a device-resident ttnn matmul,
with the near-tie correctness gate against the b10451 oracle on the quantized
model. Not verifiable offline.
