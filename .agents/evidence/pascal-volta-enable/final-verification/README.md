# Final local implementation verification

## Result

PASS for the completed local implementation gate. Product acceptance remains PENDING real-checkpoint correctness and every declared performance axis.

CUDA source head: `f81c17c8a6d33271eea968ed3d0bdfb88ff9cdf5`.
Combined source head: `011dbdc9781cf636825e2d76e6050c30a3180ccd`.
This evidence-only follow-up changes no product, test, default, lifecycle, or issue disposition.

## Independent gate

All GPU execution used `flock /home/user/gpu.lock` on the local Quadro P620, sm_61, 2048 MiB.

```sh
flock /home/user/gpu.lock ctest --test-dir /tmp/lumen-build-legacy -R '^test_(cuda_ops|cuda_backend|cuda_legacy_matmul|cuda_legacy_intrinsics|cuda_legacy_public|cuda_legacy_refusals|attn_validate_configuration)$' --output-on-failure
flock /home/user/gpu.lock /tmp/lumen-build-legacy/tests/test_cuda_quant_dot '--test-case-exclude=CUDA grouped keep-quant at GLM-5.3-Flash*'
```

Each independent command exits 0. CTest passes 7/7. Quantization passes 18 cases and 254192 assertions. The single named GLM published MoE geometry case remains PENDING a GPU larger than 2 GiB. Its parameters and assertions remain unchanged. The logs retain the exact exclusion.

The public synthetic loader test covers head dimensions 64, 128, and 256 with prompt lengths 1, 16, and 37. It compares CPU and CUDA greedy IDs, finite logits, and the absence of CPU reference fallback. This does not qualify a published checkpoint.

The operator independently runs the same final focused gates. The raw operator and independent logs are adjacent to this report.

## Complete Release build and local recipe

The full production library and nine focused binaries compile and link successfully. The final build ends at step 924/924 with exit 0.

```sh
cmake -S /tmp/lumen-pascal -B /tmp/lumen-build-legacy -DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/15
cmake --build /tmp/lumen-build-legacy --target test_cuda_ops test_cuda_backend test_cuda_quant_dot test_cuda_legacy_matmul test_cuda_legacy_capi test_cuda_legacy_intrinsics test_cuda_legacy_public test_cuda_legacy_refusals test_attn_validate_configuration -j4
```

The cache uses Release, CUDA 12.4, CUDA host GCC 13, C++ GCC 15, and `VLLM_CPP_CUDA_ARCHITECTURES=60;61;70`. An earlier local link selected GCC 13 libstdc++ before the GCC 15 runtime and lacked `__cxa_call_terminate`. The explicit executable linker search path corrects the local mixed-toolchain setup. No repository product change addresses that environment mismatch.

An earlier build stopped because duplicated review archives exhausted temporary storage quota. Removing generated duplicate archives preserved source, baseline artifacts, and logs. The resumed build completes. Neither interrupted build is recorded as passing.

`cuobjdump --list-elf` on the final `cuda_paged_attn.cu.o` lists actual sm_60, sm_61, and sm_70 cubins. Compilation does not establish P100 or Volta runtime behavior.

## Source composition and reviews

The final combined implementation preserves all earlier integration source ownership comparisons. The metadata follow-up changes the host selector test, its spec appendix, issue, and evidence. The final kernel, fixture, public test, and metadata test equal the final CUDA source tree byte-for-byte. The original public ABI header and export map equal the pinned task base byte-for-byte.

Independent source reviews are retained here:

- `initial-cuda-review.md` records the original FAIL and exact findings.
- `prefill-review.md` records the shared-memory repair PASS and deferred SiLU and EXL3 mutation results.
- `records-review.md` records the row inventory repair PASS.
- `metadata-review.md` records the metadata repair PASS, including all five prior surviving mutations.
- `branding-review.md` records the repaired compatible branding PASS.

The initial FAIL is preserved. Its findings have corresponding separate repair reviews. No new implementation appears in this evidence commit.

On the final combined source, `check-agent-record.py`, `check-surface-coverage.py`, and `cmake -P cmake/CudaArchFeaturesTest.cmake` each exit 0. The evidence writer invokes startup preflight with a 45-second bound and records exit 124 before long suites. Its partial output is not green. The full repository preflight is not green. Earlier clean-base rocprof and tools-suite failures remain separate from the local implementation gate. The default compute_121a configuration cannot compile with local CUDA 12.4. The CUDA syntax-only checker also has an nvcc scratch-object limitation. Actual full compilation supplies direct build evidence without weakening either checker.

## Pending acceptance

- Published-checkpoint token correctness, identical-workload traces, throughput, latency, and memory comparisons remain PENDING approved weights and operator measurements.
- P100 runtime remains PENDING hardware.
- Volta runtime and optimization remain TODO until GPU delivery, as the owning spec requires.
- The oversized GLM quantized MoE resource gate remains PENDING a GPU larger than 2 GiB.
- Windows and macOS execution remain unverified.
- The full-range whitespace check reports preserved raw compiler output trailing spaces in `compile-red.log` lines 2 and 90. Product files have no such finding.
- No row is marked DONE, no issue is closed, and no main merge or remote publication occurs.

The combined task branch is concrete and reviewable. These local results do not establish completion of the requested real-model optimization campaign.
