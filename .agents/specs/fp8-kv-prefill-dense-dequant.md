# fp8 KV prefill on the bf16 dispatch: one dense dequant per layer — ISSUE-LOCAL-01M3QG4WWC3X0C84M9PWQZAJ10

The fp8 KV cache read sent every prefill to the scalar CUDA-core flash kernel,
3.7x slower than the same kernel on a bf16 cache, because FA-2 admits only bf16
q/KV/out and the fp8 store presents f32. This is the prefill-performance half of
`KV-FP8`; W2 (issue [#1593](https://github.com/mudler/vllm.cpp/issues/1593))
landed the fp8 store and read-dequant for correctness.

Issue: [ISSUE-LOCAL-01M3QG4WWC3X0C84M9PWQZAJ10](../issues/KV-FP8/ISSUE-LOCAL-01M3QG4WWC3X0C84M9PWQZAJ10.md).
Owning row: `KV-FP8` ([engine-matrix.md](../engine-matrix.md)); the row's spec is
[fp8-kv-cache.md](fp8-kv-cache.md).

## Premise, grounded

| Where (line anchors at this branch's base, `b45a94273`) | What |
|---|---|
| `src/vt/cuda/cuda_paged_attn.cu:148-197` | The fp8 K/V read with the per-element dequant folded in (`Fp8E4M3ToF32Dev`), the W2 arm. |
| `src/vt/cuda/cuda_paged_attn.cu:2909`, `:2954` | The dispatch comments that name the split: the tensor-core ladder for bf16, the f32-q/out scalar arm otherwise. |
| `include/vllm/model_executor/models/kv_cache_route.h:40,73-78` | The store/read route that hands `kv_cache_dtype` to the backend. |
| `src/vllm/model_executor/models/qwen3_5.cpp` (the attention preamble) | The model-side arm this change routes. |
| `tests/vt/test_ops_paged_attn.cpp` | 33 cases on the base, including the W2 fp8 parity case. |

Measured on the 27B NVFP4 arm at 8k prefill: 28.0 s fp8 vs 7.8 s bf16 (3.7x),
25.1 s of 36 s in `PagedFlashKernel<float, unsigned char, …>`, against
2.9 ms/layer on the FA-2 split-KV kernel for a bf16 store.

## Design

Dequantize the fp8 cache ONCE per layer into a dense bf16 scratch, then run the
shipped bf16 dispatch on it:

- The scratch is **one block per request with `block_size = max_seq` and an
  identity block table**, so the kernel's paged address IS the dense address and
  no attention kernel changes.
- Each scratch allocation is owned by a scope guard from the moment it
  succeeds. The identity allocation and copy, the dequant launch, the bf16
  dispatch it feeds, and the vectors between them can all throw, and the guard
  frees on the SAME stream, so the free is ordered behind the work that reads
  the buffer. The success path still checks its own frees explicitly.
- The new shared `KvCachePresentsBf16` helper presents bf16 for an fp8 store, so
  FA-2 admits with zero cast kernels. It is model-agnostic: any model that gains
  an fp8 store inherits the CUDA path unchanged.
- `VT_ATTN_FP8_DENSE=0` restores the per-read dequant for a same-binary A/B.

**Two rejected levers are recorded rather than deleted**, because both are
plausible and neither survives the numbers:

1. Routing fp8 through the bf16 WMMA ladder with an fp8→bf16 cast inside the K/V
   staging (8k 28.0 → 9.1 s, `VT_ATTN_FP8_WMMA`). It keeps the per-element
   dequant on every re-stream, so the gap to bf16 (7.8 s) only narrows.
2. Converting the staging with `__nv_cvt_fp8_to_halfraw` instead of the software
   decode — bit-identical and cheaper, but still per re-stream.

The dense dequant removes the per-read dequant entirely for prefill, which is
why it is the shipped shape. The bench that separated kernel cost from engine
cost is `benchmarks/paged_attn_prefill_ab.cpp`, landed with this change.

## Tests

`tests/vt/test_ops_paged_attn.cpp` gains the fp8-dense parity case: the fp8 cache
against the f32 reference at **1.9e-6 max abs err** — the exact dequantized
values, tighter than the bf16-compute envelope — with the rest of the 33-case
suite unchanged. A second case injects the two failures a healthy device cannot
produce on demand (the identity-table allocation, and the identity-table copy
after both allocations succeeded) and asserts that the CUDA memory pool's used
bytes return to their pre-call value after the exception unwinds. It also
asserts the propagated message is the failing `Check`, so the guard's destructor
cannot mask the original exception.

## What this does NOT claim

- **No speed claim in this PR.** The numbers above are the author's measurement
  on the local `sm_120a` card with host embedding on (`VT_HOST_EMBEDDING=1`), and
  they are recorded as the shape's evidence, not as an operator gate. A
  same-binary A/B under the GPU lease is owed by the operator, as the helper
  template requires.
- The 20.5 GiB-vs-22.5 GiB peak comparison and the llama.cpp q8_0 denominator
  (39.9 s / 822 tok/s) are the author's run of the same recipe; they are not
  re-measured here.

## Gates

- `ctest --test-dir build -R test_ops_paged_attn` on a CUDA build
  (`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUTLASS_FETCH=ON` for FA-2).
- `VT_ATTN_FP8_DENSE=0` returns the base behaviour (the A/B arm).
- The full `ctest --test-dir build` on the same build.
- `scripts/agent-preflight.sh --staged`.

## Owed

- The operator's same-binary A/B under lease, with the 27B NVFP4 arm, both arms
  in one binary, and the recipe written into `docs/BENCHMARKS.md`.
- A `sm_120a` re-measurement: the author's numbers are from the local consumer
  card, and the fleet gate model runs elsewhere.
- Execution of the scratch-ownership case, which needs a CUDA device; the CUDA
  lane is its gate.
