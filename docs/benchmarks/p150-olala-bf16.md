# p150-olala-bf16

| | Result |
|---|---|
| Subject | olala bf16 forward (`Olala-v1.0-mini-7A1B`), the W2e arms, decoded through the production paged engine |
| Disposition | **CPU-baseline leg published**; P150 device/traced cells `PENDING` (no traced device path for olala exists yet) |
| Identical workload | Prompt ids `[151644, 872, 198, 15588, 220, 151645]` (the W2d token-gate prompt, verified identical on both arms), greedy, 64 output tokens, batch 1, no BOS, context 512, same GGUF artifact sha256 `b266da8e…bef5747` |
| Denominator | llama.cpp OVHai olala fork @ `bd20ca9b`, CPU, raw ids via a timed golden harness (llama-bench does not know the `olala` arch; `llama-cli` `-st` wraps the prompt in the chat template and is rejected) |
| Precondition | `PUBLISHABLE-AFTER-W2E-ARMS` — W2e landed (`048c6b22d`); production reachability landed (`6ce35000c`, `test_olala_paged_engine.cpp` 18/18 EXACT vs the fork) |
| Evidence | [bench-evidence/olala-bf16-cpu-ab-20261123](../bench-evidence/olala-bf16-cpu-ab-20261123/README.md) |

Host: Ampere-1a, 128 threads, sequential legs on an idle-normalized box
(<0.03 load per core), 3 legs per arm, median + spread. Same artifact, same
prompt, same sampling both arms.

| Axis | vllm.cpp (host ops, scalar f32) | llama.cpp fork | Ratio | Result |
|---|---:|---:|---:|---|
| Prompt-eval (6-token prefill) | 0.4579 tok/s | 2.615 tok/s | 0.175x | **OPEN GAP** — 5.71x behind |
| Decode (64 tokens) | 0.3990 tok/s | 2.158 tok/s | 0.185x | **OPEN GAP** — 5.41x behind |
| Greedy-stream identity | 58 of the first 64 generated tokens identical; divergence at index 58 (fork `576`, ours `2055`) | — | — | expected greedy fp drift; binding token gate EXACT (18/18) |

This is the honest first baseline of the olala host-ops path, not a ceiling.
The decode gap names four traceable hypotheses in the evidence README: scalar
non-vectorized GEMV vs ggml's blocked kernels, per-step dispatch overhead (the
6-token prefill costing 13.1 s points at a large fixed per-call cost), no
bf16-native host path, and unlogged thread-pool configuration.

| Cell | State |
|---|---|
| CPU-baseline decode/prefill (this page) | measured 2026-11-23 |
| A1 traced device decode bucket ladder | `PENDING` — olala has no traced P150 path |
| A2 decode-vs-context ladder | `PENDING` — only the token-gate context is measured |
| A3 long-context TTFT ladder (128/2048/8192/32768) | `PENDING` |
| A8 weight-residency memory | `PENDING` — bf16 residency measured only as peak load time, not device bytes |
