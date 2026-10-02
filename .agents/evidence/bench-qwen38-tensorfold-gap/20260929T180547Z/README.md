# Qwen3.8 TensorFold gap blocker evidence

Status: `BLOCKED_MISSING_ARTIFACTS`.

A fresh successful repository `rc` audit lease executed the committed bounded scan script on `dgx:gpu0`; only a SHA-256 lease fingerprint is retained. The raw receipt found the two documented staging paths absent and zero matches under `/workspace` to depth 3 for the committed TensorFold/MiaAI/Vontra and MLX-MTP/Flash-Next-MTP patterns. This is not an exhaustive host-wide absence claim. Required artifacts/source were not found in that retained scope, so runner prerequisites could not be populated and measurement did not start.

Because prerequisites were absent, no server was started and no correctness, timing, clock, memory, ladder, or profiler result was produced. Every workload and both same-tool profiles are explicitly `NOT_RUN_PREREQUISITE`; there is no cross-engine ratio. Production vLLM remains the named denominator and is `NOT_RUN`: this blocked discovery did not stage or inspect a runnable denominator.

The MTP verdict is independently `BLOCKED_NO_MTP_WEIGHTS`, derived from the committed real-header manifest `tests/vllm/models/qwen4_exp_gguf_manifest.inc`: 1,224 tensor entries, trunk blocks 0 through 47, and zero tensor names matching `mtp`, `nextn`, `draft`, `eh_proj`, `enorm`, or `hnorm`. This says nothing about the absent TensorFold MLX checkpoint.

Task 5 records `NO_PORT_DECISION`; Task 6 is `BLOCKED_NO_MTP_WEIGHTS`; Task 7
completed synthesis as `SYNTHESIS_COMPLETED_NO_PRODUCT_OPTIMIZATION`.
