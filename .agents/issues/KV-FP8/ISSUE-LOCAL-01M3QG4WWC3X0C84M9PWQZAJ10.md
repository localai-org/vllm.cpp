ID: ISSUE-LOCAL-01M3QG4WWC3X0C84M9PWQZAJ10
Title: fp8 KV prefill runs the scalar CUDA-core flash kernel at 3.7x the bf16 cache, because FA-2 admits only bf16 q/KV/out while the fp8 store presents f32
Row: KV-FP8
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

MEASURED on the 27B NVFP4 arm at 8k prefill: 28.0s with the fp8 KV cache against 7.8s with the same kernel on a bf16 cache, a 3.7x gap. nsys attributes it entirely to the kernel choice: 25.1s of 36s sits in `PagedFlashKernel<float, unsigned char, ...>`, the scalar CUDA-core arm, while the bf16 store runs the FA-2 split-KV kernel at 2.9 ms/layer. CAUSE: the model presents f32 q/out for a non-bf16 store (`GdnOutDType`/the KV-store route), and the vendored FA-2 admission requires bf16 q, bf16 KV and bf16 out, so every non-bf16 cache falls off the tensor-core ladder onto the per-element fp8 dequant. The W2 CUDA arm of this row (issue #1593) landed the fp8 store and the read dequant for CORRECTNESS, so this is the prefill-PERFORMANCE half the row never had. THE FIX: dequantize the fp8 cache ONCE per layer into a dense bf16 scratch and run the normal bf16 dispatch on it. The scratch is one block per request with block_size = max_seq and an identity block table, so the kernel s paged address IS the dense address and NO attention kernel changes. A shared `KvCachePresentsBf16` helper presents bf16 for an fp8 store, so FA-2 admits with zero cast kernels and any model that gains an fp8 store inherits the CUDA path unchanged. `VT_ATTN_FP8_DENSE=0` restores the per-read dequant for a same-binary A/B. TWO EARLIER SHAPES OF THE SAME LEVER WERE TRIED AND REJECTED, and the record keeps them: (a) routing fp8 through the bf16 WMMA ladder with an fp8->bf16 cast inside the K/V staging (8k 28.0 -> 9.1s) still pays a per-element dequant on every re-stream, and (b) converting the staging with `__nv_cvt_fp8_to_halfraw` made that conversion cheap (bit-identical) but did not remove it. The dense dequant removes it entirely for prefill. Numbers after: 32k prefill 34.9s / 938 tok/s against 36.4s / 901 for the bf16 store and 39.9s / 822 for llama.cpp s q8_0 KV, at 20.5 GiB against 22.5; 8k 7.6s / 1000 tok/s; decode with MTP n=3 38.6 vs 37.4. Op-level parity against the f32 reference is 1.9e-6 max abs err (the exact dequantized values), all 33 paged-attention cases pass, and benchmarks/paged_attn_prefill_ab.cpp is the isolated sweep that separated the kernel cost from the engine.

## Resolution

- 2026-09-30: review repairs on the W1 PR (mudler/vllm.cpp#3360). The dense bf16
  scratch and the identity table are owned by stream-aware scope guards, so the
  steps that can throw between the two allocations (identity copy, dequant
  launch, the bf16 dispatch, the vectors between them) release both buffers on
  the same stream. A new case in `test_ops_paged_attn` injects the identity
  allocation failure and the identity copy failure and checks the CUDA pool's
  used bytes return to the pre-call value without the guard masking the
  original exception. The benchmark header no longer claims a bf16 parity
  comparison it does not perform; parity is gated by the test suite.
