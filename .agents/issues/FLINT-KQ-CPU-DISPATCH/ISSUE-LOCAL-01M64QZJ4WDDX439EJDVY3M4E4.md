ID: ISSUE-LOCAL-01M64QZJ4WDDX439EJDVY3M4E4
Title: Flint CPU: the keep-quant tree is refused by the paged arm -- no CPU keep-quant serving arm exists
Row: FLINT-KQ-CPU-DISPATCH
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

FLINT-CPU-PARITY (landed 19979acf9) discovered the CPU paged serving arm has no keep-quant consumer: VT_FLINT_KEEP_QUANT=1 loads a block tree whose only in-tree consumer is the Tenstorrent forward, and the paged arm died on an opaque matmul inner-dims mismatch -- now a named refusal at flint.cpp:868-881. The keep-quant arm is the real lever under the gated llama.cpp floor (26.29 tok/s @16t on flint-Q4_K_M.gguf, goldens + raw logs committed). Spec: .agents/specs/flint-kq-cpu-dispatch.md. The loader, tower slicer, per-dtype dispatch seam, and gguf dequant paths already exist (#3514 lineage) -- the arm is dispatch on the CPU paged forward, compute class chosen by measurement, red-first = the refusal test flips to serving in-band. shexp mapping verified.

## Resolution

-
