# ROCm quantized gather evidence

Row: `BACKEND-ROCM-QUANT-GATHER`.
Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).
The [committed spec](../../../.agents/specs/rocm-quant-gather.md) defines the gates.

The native operation, public dense control, and matching operation traces pass.
Pinned model qualification and the original upstream fixture gate remain pending.
This evidence makes no throughput, latency, model token parity, or performance claim.

## Identity and scope

The implementation started from spec commit `670e6d78ddf55231394748e0032939fd53dc56a5`.
The Q8_K materializer amendment preceded its code at `376223f000f85683f2881136e38ee52892d8d3a8`.
The reader amendment preceded its code at `2e5adfb85a255528bffa5d3cdb016efb2bab70d8`.
The prerequisite full-attention repair was independently reviewed at
`6a7bcb77637e66df34429208e3a4055e0945a875` and cherry-picked before the intended red run.
Its later runner expectation repair was imported as spec `8c9101fa4` and test commit `b46852d13`.
Those three dependency files equal donor `694f04525174d0c2dac8030ba8f1398f2683ec82` byte-for-byte.

Early captures identify uncommitted source with complete file and executable hashes.
Their hashes describe the measured stages, including defects repaired before the final implementation.
They are not claims that every early artifact equals the final commit.
The final commit and review receipts identify the final acceptance candidate.

All native GPU commands ran through the operator on the supplied gfx1100 host.
The operator held its recorded file mutex and checked input hashes around each run.
The two-device bounds test used visible devices `0,1`; other native runs used device `0`.
Helper builds and secondary oracle runs were CPU-only.
Every listed absolute path records this measured environment and is not a default.

The primary runtime uses vLLM `e126687a9a828d513c01a07cd69f025f27d63280` and plugin
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
It runs in the existing image
`sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.
The separate plugin artifact contains native extension
`592af79c210496e96eaa7bd409ae0a67e7de8c334aa9f782082c7639ba300138`.
The build recipe, log, and complete artifact hashes are retained here.
No existing oracle runtime was modified.

Stock llama.cpp is `10bf611e533d81f739128304991c5e133c6aebd8`.
The registered IQ1_XXXS fork is `36fe8e1cc7f2b3b8c92fdda0ab07600141921786`.
The [independent oracle audit](oracle-audit-report.md) verifies their source archives,
executing decoder chain, binaries, plugin artifact, and captured outputs.
That audit is separate from implementation mutation review.

## Red and green results

Raw `.log` filenames in this document refer to unchanged files inside `raw-logs.tar.gz`.
The archive preserves original whitespace and hashes.

| Gate | Observed result | Evidence |
|---|---|---|
| Initial public attempt | Wrong-reason failure in unused GDN state validation; not the intended red | `public-red.log` |
| Public attempt after reviewed prerequisite | Intended failure: zero compressed provider selections; 10 of 11 assertions passed | `public-red-after-prerequisite.log` and sealed recipe |
| Q8_K reader | Type 15 refused before materialization | `q8k-reader-red.log` |
| Q8_K materializer after reader repair | Reader passed; dot-only materializer guard refused decoder-only Q8_K | `materializer-second-red.log` |
| Q8_K reader and role-aware materializer | Two cases, 18 assertions passed; reader regression passed 36 cases, 133 assertions | `materializer-focused-green.log`, `reader-regressions-green.log` |
| Queue device binding | Ambient device 1 caused invalid stream use for queue 0 before the fix; 16 assertions passed after binding queue 0 | `ambient-device-witness-v1.log`, `ambient-device-witness-v2.log` |
| Native focused v2 | Seven executed CTests passed, including all 19 public model variants | `native-focused-v2.log`, operator receipts |
| Final native v4 | All 15 CTests executed and passed; primitive suite 9 cases and 11,784 assertions; public suite 2,850 assertions; required two-device case 16 assertions | `native-final-v4-focused-and-rocm-regressions.log`, `native-final-v4-required-ambient-device.log`, sealed commands and operator receipts |
| Native/primary/secondary operation comparison | All 76 outputs passed; 68 were byte-identical | `synthetic-parity-v2.json`, independent operator comparison |
| Applicable stock GET_ROWS shape | Width 256, five rows, four IDs; all six native outputs matched the pinned secondary result | `stock-shapes-v1-operator-results.json` |
| CPU-only configuration | Five CPU targets passed; three accelerator-only targets skipped explicitly; the imported runner suite passes 41 cases and 1,914 assertions | `cpu-configure.log`, `cpu-focused-green.log`, `cpu-runner-green.log` |

The model builder fixes seed `0x524f434d`, geometry, tokenizer metadata, and tensor values.
The two-regeneration manifest confirms all 38 model hashes, without changing fixture bytes between red and green.
The public path uses 19 codecs, two prompts, three repeats, and a fresh engine for each request and dense control.
It checks all four greedy token IDs and all 512 captured logits byte-for-byte.
The final public run additionally writes each completion's tokens and raw F32 logits with
`VT_ROCM_GATHER_PUBLIC_EVIDENCE`.

## Numerical oracle evidence

`native-operation-captures.tar.gz` preserves 76 native outputs and its report.
`primary-operation-captures.tar.gz` preserves 64 primary outputs and its report.
`secondary-operation-captures.tar.gz` preserves the six actual upstream F32 outputs.
The native BF16 secondary comparison applies explicit round-to-nearest-even to those F32 outputs.
It does not claim that either secondary graph executed BF16 output.

The primary gate preserves `atol=0.01`, `rtol=0.04`, widths 256/1024, and both admitted output dtypes.
The local ABI flattens the upstream two-by-two IDs; the primary harness restores that shape.
IQ1_M and F16 output exceed the shared local contracts and remain explicit exclusions.
The generated 128-row corpus does not replace the 32 original upstream GGUF fixtures.
All 32 were downloaded with the developer's authority and sha256-verified at
289,655,872 bytes. Their exported tensors (160 cases over 16 primary codecs)
pass the native-versus-plugin comparison: 320 outputs, 140 byte-exact, and every
remaining element inside the unchanged upstream tolerance.
The raw receipt is `gather-3113-repair/fixture-download-receipt.json` under
`/home/vikash/.cache/`; the comparison is
`original-fixture-coverage/compare/report.json` under
`/home/vikash/.cache/rdna3-gather-oracle-impl/`.

Eight nonidentical comparisons are confined to Q6_K and IQ3_S, across two widths and output dtypes.
Q6_K's primary decoder rounds intermediate products to F16; the local decoder uses F32 products.
IQ3_S's primary codebook contains effective values of 15.5 where the local/stock codebook contains 15.
All differences satisfy the unchanged upstream tolerance.
The [arithmetic diagnostic](oracle-audit-arithmetic-explanation.json) reproduces both executing expressions.
It explains the runtime outputs and does not replace them with a scalar oracle.

Stock GET_ROWS excludes Q8_K, so the harness calls its pinned `dequantize_row_q8_K` directly.
MXFP4 and IQ1_XXXS execute real upstream GET_ROWS graphs.
The additional stock-shape corpus preserves its applicable width 256, five rows, four IDs, and contiguous unbatched layout.
Its deterministic packed values and first/last/middle/repeated IDs adapt upstream's unrecorded random input.
The local suite explicitly refuses batched tables and strided IDs.
Upstream cases with batched tables, floating-only dtypes, or backward gradients exceed this gather ABI.

## Memory and matching traces

Both implementations used the same 32 packed tables, widths 256/1024, four I64 IDs, and requested F32/BF16 outputs.
Both used the exact existing rocprofv3 1.3.5 profiler and its isolated prefix.
The prefix contains profiler dependencies and excludes HIP/HSA/BLAS runtime replacements.
Version 2 adds allocation events to both sides; its inputs, executable, and harness equal version 1.
The trace-only primary harness omits the already-verified full-table dense control.
Every traced output equals the same implementation's untraced output byte-for-byte.

The native trace contains 64 fused gather kernels, with 32 F32 and 32 BF16 template instantiations.
Its 192 correlated `hipMalloc` events exactly match 32 packed tables, 32 ID arrays,
64 output arrays, and 64 error records of 16 bytes.
There is no extra native device allocation for selected packed rows or an F32 staging output.
The profiler's host runtime allocations are identified separately.

The primary trace contains 64 GPU packed-row selections and 64 native decoder launches.
Each output uses the requested F32 or BF16 decoder template.
Its caching allocator appears as a backing pool in the profiler.
Separate PyTorch telemetry measures each tensor operation: the live delta equals its output size;
the peak delta adds selected packed rows rounded to the observed 512-byte allocator unit.
Those tensor counters are not profiler allocation-size measurements.

The source chain in the independent audit and these executed GPU kernels establish selected-row native decode.
The trace harness contains no dense-control or host-dequantization call.
Provider counts are an additional reachability check, not the only evidence against host fallback.
A separate trace runs the complete public model test: 228 engines and 2,850 assertions passed.
All 456 preserved capture files contain exact dense-control and repeat token/logit bytes.
The operator verified every capture, kernel correlation, and device allocation independently.
The model trace executes 912 BF16 KV reads, 912 two-byte KV writes, and 456 direct BF16 quantized gathers.
It records 228 `hipMalloc` allocations of 16,384 bytes, matching one KV pool per engine.
The source allocates four pages of `16 * 1 * (64 + 64) * 2 = 4096` bytes each.
The allocator-to-KV attribution uses that source chain: CSV does not expose kernel argument pointers.
The observed cache template dtype and byte counts are preserved separately from this attribution.
`public-model-trace-captures.tar.gz` retains complete profiler CSVs, completion captures, and the scratch capture audit.
This establishes native BF16 cache storage at the original four-block setting.
The committed oracle amendment requires 16 blocks and a new matched memory capture.
The original traces retain their measured values.

## Initial qualification failures

The stock and fork bounded model attempts both abort on their first decode before emitting tokens.
The observed assertion is `ggml-backend.cpp:194::GGML_ASSERT(buffer)` through hybrid memory input setup.
An unused recurrent input is the source-based hypothesis, not a proven object-allocation diagnosis.
Their context request of 64 resolves to 256 cache cells through the pinned upstream padding rule.
Their BF16 KV payload is therefore 65,536 bytes, against the original spec's 16,384 bytes.
The committed amendment matches that physical capacity in the native and primary
arms. Both secondary model executions still require qualification under issue #3093.
The fork's independent real 2.4T qualification remains owed under issue #933.

The primary model attempt uses this row's exact Q4_0 fixture, SHA256
`663597e28852d64097f7a67542675e4575ee158d3c5e6e2143aadd1b4188f4af`.
The first attempt failed because Triton's default cache path was unwritable.
That operational failure is not a model compatibility result.
The second attempt supplies writable task-local cache paths without changing the model, HF config, or oracle code.
It reaches the pinned renderer and refuses `Qwen3_5TextConfig` where `Qwen3_5Config` is required.
It emits no tokens. The full traceback and sealed recipe preserve that actual model refusal.
The operator checked 340 input files before and after the second attempt.
That refusal predates the approved configuration amendment. The corrected v5
launch emits four tokens from the unchanged Q4_0 fixture with normal compilation
and graph capture. The [amendment evidence](oracle-amendment/README.md) retains
the intervening RoPE and cache-admission failures, successful launch, and receipts.
The full token matrix and matched model memory remain pending under issue #3093.
The final v4 operator audit confirms all 456 completion files equal the traced run byte-for-byte.
All 33 sealed source, library, and executable files remained unchanged during v4.
The row stays active until model execution, original fixtures, fresh mutation review,
and the operator's reviewed-head rerun meet their respective requirements.

## Full preflight

The first full staged run found two stale source citations after the materializer edit.
The corrected engine-matrix rows change only `SPEC-MTP-GGUF` and `SPEC-DFLASH-GGUF` line anchors.
Every unrelated row remains byte-identical, and the anchor baseline stays at 28.
The record gate then passes with 937 valid anchors, 28 stale anchors, and five pre-existing broken anchors.
The final full staged rerun exits 0 with no failures and five argument-dependent skips.
All 31 translation units in its host syntax scope compile.
This is not an all-green preflight, because a skipped check executes nothing.
The separate CPU ISA check passes against the actual CPU-only compile commands.
The ARM ISA, CUDA code-generation, and CUDA Triton artifact checks do not apply to this x86_64 HIP configuration.
Exact commit path classification runs after committing and accompanies the handoff.
Combined CUDA/HIP execution remains pending because no combined build was available.
The raw final log and explicit CPU ISA result are archived unchanged.
`preflight-final-staged-result.json` records the exact command, environment, staged tree, statuses, and skip dispositions.

## Reproduction

Build and run the focused commands in the spec under the operator's GPU authority.
Generate operation inputs with `rocm_quant_gather_capture generate DIRECTORY`.
Run `primary.py capture`, `secondary.cpp`'s `gather` mode, and native `capture` with the sealed corpus.
`compare.py` compares the native report against each independently executed oracle.
`stock_shapes.py` exports the applicable secondary geometry without downloading or changing packed values.
`audit_trace.py` verifies each trace against its untraced outputs and checks the memory evidence.

Exact argv, environment, source hashes, executable hashes, and operator receipts accompany each stage.
Downloaded upstream GGUF fixtures and generated model weights are not committed here.

> EVIDENCE-LOST (2026-10-30): `ambient-device-witness-v1.log`, `ambient-device-witness-v2.log`, `cpu-configure.log`,
> `cpu-focused-green.log`, `cpu-runner-green.log`, `materializer-focused-green.log`, `materializer-second-red.log`,
> `native-final-v4-focused-and-rocm-regressions.log`, `native-final-v4-required-ambient-device.log`,
> `native-focused-v2.log`, `public-red.log`, `public-red-after-prerequisite.log`, `q8k-reader-red.log`,
> `reader-regressions-green.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
