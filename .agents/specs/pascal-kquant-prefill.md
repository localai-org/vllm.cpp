# Tiled K-quant prefill for Pascal DP4A

## Now

SPEC. Issue: `ISSUE-LOCAL-01M653NJXDM6B5KE8VAA2VP93X`. Row:
`BACKEND-CUDA-SM061`. Base: `780767bfd`. No implementation yet.

## Scope and design

The generic K-quant prefill GEMM maps one warp per output row and streams a
weight row per prompt token. On the P620 a 16-token TinyLlama trace spends
412 ms in the generic Q4_K prefill and 201 ms in the generic Q6_K prefill of a
1.25 s total. Add a tiled DP4A prefill lane that stages an `I x J` weight tile
once in shared memory and reuses it across the `J`-token tile.

Cover `m > 1` only. Start with Q4_K and Q6_K, the two dtypes this row's real
checkpoints use. Keep the single-row decode lanes, the generic K-quant kernel,
the grouped expert path, Volta, and every newer GPU default unchanged. Preserve
the public ABI and the existing environment overrides.

## Upstream anchors

Pinned secondary oracle: llama.cpp
`62318cd57dc63900ecb153272c0b10f78550855d`.

- `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh:100-110` (Q4_K) and
  `:136-146` (Q6_K) select `nthreads=256`, `occupancy=2`, `I=64`,
  `J in {8..64}`, and `MMQ_ITER_K=256`.
- `ggml/src/ggml-cuda/mmq.cuh:164-183` defines the config struct and the
  `rows_per_warp()` rule; `:386-403` defines the DP4A tile sizes.
- `ggml/src/ggml-cuda/mmq.cuh:595-599` and `:765-768` are the Q4_K and Q6_K
  dispatch arms.
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh:711` (Q4_K) and `:946` (Q6_K) stage
  the weight tile.
- `ggml/src/ggml-cuda/mmq-vec-dot.cuh:905` (Q4_K) and `:975` (Q6_K) are the
  DP4A tile dots.

Mirror the config table. Do not invent tile shapes.

## Design detail

Stage the weight tile as the oracle's DP4A SRAM layout, stage the activation
tile as `Q8_1` per 32 values, and loop `K` in 256-element iterations. Accumulate
in `int32` per lane with `__dp4a`, then fold the scale/min pair once per tile.
Use `I=64` and `J=64` for the wide projections and the oracle's smaller `J` for
narrow shapes. Fall back to the generic kernel when the shape or shared-memory
budget does not fit.

The tiled lane changes the float reduction order relative to the generic
kernel. Prefill output feeds the token gate, so the six exact axes must stay
exact or the near-tie policy must be extended explicitly.

## Tests and gates

Add a registered `MatmulBTQuant` prefill regression at `m in {2, 16, 32, 128}`,
`n in {7, 16, 5632}`, and `k in {2048, 5632}` for Q4_K and Q6_K, checked against
the CPU oracle and an f64 dequantized reference. Require the generic lane to
pass the same cases.

Add a same-binary throughput regression against the generic prefill and require
a substantial warmed speedup. Deleting the tiled dispatch must fail that
regression. Compile the changed translation unit for sm60, sm61, and sm70.

Run the focused quant suite and the eight-axis real-model gate on the P620.
Record an nsys trace that shows the tiled kernel replacing the generic prefill.

## Acceptance

The target is to cut the 613 ms TinyLlama p16 prefill below 200 ms while keeping
the six exact axes exact. The two documented near-ties remain governed by the
ratified tie policy.

## Risks and stop conditions

Shared memory on Pascal is 48 KiB per block; `I=64`, `J=64`, and the Q4_K/Q6_K
SRAM layouts must fit at the oracle's occupancy. Stop on an ABI change, a decode
path change, a non-K-quant dtype, a shared-memory overrun, or a new token
mismatch on an exact axis.

## Owed

Implementation, red-before evidence, fresh review, and operator gates remain
open.
