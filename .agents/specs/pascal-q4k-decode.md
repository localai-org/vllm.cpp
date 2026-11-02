# Pascal Q4_K decode and weight-reusing prefill mapping

## Now

ACTIVE. Issue: `ISSUE-LOCAL-01M652EAJ85CNWG65FGPTZ8742`. Row:
`BACKEND-CUDA-SM061`. Base: `ca0b2ff73`.

## Scope and design

Two Pascal-only decode/prefill changes on the legacy CUDA lane:

1. Route dense single-row Q4_K decode through a new 32-lane DP4A lane. Every lane
   owns 8 of the 256 super-block values, so each lane issues two `__dp4a`
   instead of the generic kernel's eight-active-lane scalar walk. Prefill
   (`m > 1`), grouped experts, Volta, and newer GPU defaults are unchanged.
2. In the generic K-quant GEMM, map consecutive warps to the same output row
   `j` and consecutive activation rows `i`, so the four warps of a block share
   one weight row through L1. The per-output arithmetic is unchanged.

Preserve the public ABI, the existing `VT_ATTN_DECODE_OPT` behavior, and every
quantization dtype. `VT_PASCAL_Q4K_DECODE=0` restores the generic Q8_K lane.

## Upstream anchors

The pinned primary vLLM cannot execute this Pascal GGUF workload. The registered
secondary oracle is llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d`.
Port `ggml/src/ggml-cuda/vecdotq.cuh:918-975` (`vec_dot_q4_K_q8_1`) and the
Q4_K block layout in `ggml/src/ggml-quants.c:1530` (`dequantize_row_q4_K`).
Q8_1 preparation follows `quantize.cu:64-101`.

## Design detail

The Q4_K super-block is eight 32-value sub-blocks. Sub-block `sb` reads bytes
`(sb/2)*32 .. +32` of `qs` with shift `4*(sb&1)`, and its scale/min are the
6-bit pair from `scales[12]`. Lane `lane` owns sub-block `lane/4` and values
`8*(lane%4) .. +8` of it. The activation is `PascalQ8_1` per 32 values, the
same contract the Q5_0 and Q6_K lanes use. The per-lane contribution is
`a.d * (w.d*sc*dot - w.dmin*m*suma)` and the warp tree-reduces once.

## Risks and stop conditions

The Q8_1 activation lane matches llama.cpp's decode arithmetic but reduces the
super-block in a different order from the generic Q8_K lane. It can therefore
resolve a sub-milli-nat tie differently. The TinyLlama p16/t64 step-28 tie is
`< 0.001` nats; the reviewed register attention lane already resolves it the
same way by default. `VT_PASCAL_Q4K_DECODE=0` is the byte-identical rollback.

Stop on an ABI change, a non-Q4_K dtype change, or a prefill arithmetic change.

## Tests and gates

Add a registered-operation Q4_K decode regression that reconstructs the Q8_1
activation independently and compares `K = 256, 2048, 5632` at `N = 7`. Require
the focused quant suite to pass in both the default and the rollback
configuration. Record the GLM-5.3 2.4 GB MoE geometry as an environment blocker
on the 2 GB P620, as the preceding Q5_0 run already does.

Run the actual TinyLlama and SmolLM2 token gates on the P620. Record the two
near-tie axes and their measured gaps. Compile the changed translation unit for
sm60, sm61, and sm70.

## Evidence

Same-binary P620 A/B, 64 generated tokens, three repetitions:

| Model | Q4_K off | Q4_K on | Delta |
|---|---|---|---|
| TinyLlama p16 | 21.0 tok/s | 22.0 tok/s | +4.8% |
| SmolLM2 p16 | 40.1 tok/s | 40.3 tok/s | +0.5% |

The Q4_K decode kernel time fell from 868 ms to 406 ms in the nsys trace.
The prefill warp mapping moved TinyLlama p16/t16 from 12.8 to 13.8 tok/s
(+8%) and is token-neutral on every exact axis.

## Owed

Full-model performance acceptance beyond the two measured checkpoints, and the
Volta hardware gate, remain with the operator.
