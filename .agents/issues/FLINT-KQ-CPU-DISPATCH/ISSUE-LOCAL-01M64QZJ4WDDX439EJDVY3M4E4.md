ID: ISSUE-LOCAL-01M64QZJ4WDDX439EJDVY3M4E4
Title: Flint CPU: the keep-quant tree is refused by the paged arm -- no CPU keep-quant serving arm exists
Row: FLINT-KQ-CPU-DISPATCH
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-30
Closed: 2026-10-30

## Problem

FLINT-CPU-PARITY (landed 19979acf9) discovered the CPU paged serving arm has no keep-quant consumer: VT_FLINT_KEEP_QUANT=1 loads a block tree whose only in-tree consumer is the Tenstorrent forward, and the paged arm died on an opaque matmul inner-dims mismatch -- now a named refusal at flint.cpp:868-881. The keep-quant arm is the real lever under the gated llama.cpp floor (26.29 tok/s @16t on flint-Q4_K_M.gguf, goldens + raw logs committed). Spec: .agents/specs/flint-kq-cpu-dispatch.md. The loader, tower slicer, per-dtype dispatch seam, and gguf dequant paths already exist (#3514 lineage) -- the arm is dispatch on the CPU paged forward, compute class chosen by measurement, red-first = the refusal test flips to serving in-band. shexp mapping verified.

## Resolution

- 2026-10-30: closed as fixed on `row/FLINT-KQ-CPU-DISPATCH`. The paged arm
  consumes the keep-quant tree through the per-dtype dispatch seam
  (`FlintLinearMatmul` + the lm_head call site -> `vt::MatmulBT` -> the CPU
  `kMatmulBTQuant` CIQ GEMM); the named refusal is gone. Serving gate:
  `test_flint_gguf` keep-quant paged token gate, 53/60 exact / 7 near-tie /
  0 hard flips, worst gap 2.486 < 2.5 nats against the llama.cpp goldens.
  Throughput measured and open: 1.20 tok/s @16t, 2.46 @32t vs the 26.29
  floor (CIQ per-expert path ~2.7x slower than the bf16-dequant arm at batch
  1; levers named in the evidence report). Evidence:
  docs/bench-evidence/flint-kq-cpu-20261030/report.md.
