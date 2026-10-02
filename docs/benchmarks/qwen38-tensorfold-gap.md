# Qwen3.8 Flash Next: TensorFold gap

## Result

The 2026-09-29 DGX reproduction attempt is
`BLOCKED_MISSING_ARTIFACTS`. It publishes no performance number and no
cross-engine ratio.

A fresh successful audit lease then executed the committed bounded scan script
on `dgx:gpu0` (NVIDIA GB10, driver 580.173.02). The raw job ID is not retained;
the evidence stores only a SHA-256 lease fingerprint. The raw receipt establishes
absence only from the two documented staging paths and from `/workspace` to
depth 3 for the committed TensorFold, MiaAI, Vontra, MLX-MTP, and
Flash-Next-MTP patterns. It does not claim exhaustive host-wide absence. The
required artifacts/source were not found in that retained scope, so runner
artifact manifests and source revisions could not be populated and measurement
did not start.

## Consequence

Artifact hashes and launch environment files could not be created from measured
bytes. Therefore no server was started and the executable correctness gate did
not run. Serial decode, drafted decode, prefill and serving ladders, busy clock
windows, memory capture, and same-tool profiles are all
`NOT_RUN_PREREQUISITE`. Production vLLM remains the required named denominator
and is recorded as `NOT_RUN`: this blocked discovery did not stage or inspect a
runnable production-vLLM denominator.

TensorFold's publisher rates remain **unverified publisher claims**. They are
not local observations, a target derived from a local profile, or evidence for
an optimization. No ratio, speedup, winner, residual gap, or ceiling is stated.

## MTP artifact verdict

The vllm.cpp GGUF verdict is independently
`BLOCKED_NO_MTP_WEIGHTS`. The committed real-header manifest
`tests/vllm/models/qwen4_exp_gguf_manifest.inc` pins revision
`8bdc666649440e9bdc97e16f3f75782c98478ff5` and contains 1,224 tensor entries.
It has trunk blocks 0 through 47 and zero tensor names matching `mtp`, `nextn`,
`draft`, `eh_proj`, `enorm`, or `hnorm`. This verdict applies to that GGUF; it
does not claim anything about the absent TensorFold MLX checkpoint.

Consequently, native MTP cannot be implemented against the selected GGUF bytes.
Conversion is not authorized or justified here, and no speculative loader is
scoped for weights that are absent.

## Follow-on disposition

The execution-plan outcomes are explicit:

- **Task 5 / W2:** `NO_PORT_DECISION`. Incremental QSA is not implemented
  because Task 4 produced no profile proving repeated compressor work material.
- **Task 6 / W3:** `BLOCKED_NO_MTP_WEIGHTS`. The selected GGUF has no MTP head.
- **W4-W5:** `SKIPPED_NO_PROFILE`. Long-context selection, PLE, and prefill
  geometry have no measured bottleneck or correctness baseline.
- **Task 7 / W6:** `SYNTHESIS_COMPLETED_NO_PRODUCT_OPTIMIZATION`. This record is
  the completed synthesis; no profile-justified product change was retained.

The campaign can resume measurement only after the runner prerequisites are
available in the documented/configured staging scope. It must then rerun correctness
before timing rather than reusing this blocker as benchmark evidence.

## Evidence and validation

Versioned evidence:
`.agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z/`.

Validate it with:

```sh
python3 tools/bench/validate_qwen38_tensorfold_evidence.py \
  .agents/evidence/bench-qwen38-tensorfold-gap/latest
```
