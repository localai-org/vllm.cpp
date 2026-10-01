ID: ISSUE-LOCAL-01M3QETSTM8X8AJKBM9QCTGBKM
Title: The token table has no host-resident arm: every dense forward uploads [vocab, H] to the device even though the gather could run in host RAM
Row: ENG-HOST-EMBEDDING
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

A dense forward that owns a device embedding table pays [vocab, H] of device memory for it and gathers there. On the cards this project targets that is real: a 151k x 5120 bf16 table is 1.5 GiB of a 24 GiB pool, and on a tied-head model the table is kept for the lm_head GEMM regardless. The gather itself is tiny on the host: one row per id through the CPU vt::Embedding kernel, then one [T, H] copy to the device, which is llama.cpp`s ggml_get_rows shape (ggml/src/ggml-cpu/ops.cpp:4850 @ b10451) rather than vLLM`s device-side gather. Nothing in the tree offered this: `ResidentWeight`/`EmbedGather` always uploaded, so a text-only or memory-bound serve had no way to keep the table host-side. The change adds `VT_HOST_EMBEDDING=1` (host gather, one row per id, bf16/f16/f32 and every GGUF block format, bf16 byte-copy fast path) behind the existing `EmbedGather` seam, so every dense forward that owns a device table gets it for free and the device behaviour is unchanged when the flag is off or the host bytes are gone. It is vllm.cpp-original, so the secondary oracle is llama.cpp`s ggml_get_rows, and the correctness bar is the same pinned IQ4_NL/Q5_0 golden vectors the device-side vt::Embedding gate already uses.

## Resolution

- 2026-09-30: review repairs on the W1 PR (mudler/vllm.cpp#3356). The host arm's
  async-id override now routes through the shared `detail::ApplyDeviceTokenIds`
  body instead of its own unchecked Copy, so an override longer than the embed
  input is refused rather than written past the `[T]` buffer; two cases in
  `tests/vllm/models/test_host_embedding.cpp` pin the oversized refusal and the
  shorter-prefix tail preservation. The test's global initializer uses the
  portable `vllm_test::SetEnv` (`tests/support/test_env.h`) instead of POSIX
  `::setenv`, which an unconditional target cannot compile under MSVC.
