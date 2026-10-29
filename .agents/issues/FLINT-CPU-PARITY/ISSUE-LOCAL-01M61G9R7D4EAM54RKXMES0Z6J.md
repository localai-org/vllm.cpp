ID: ISSUE-LOCAL-01M61G9R7D4EAM54RKXMES0Z6J
Title: Flint CPU parity: register the llama.cpp flint oracle and measure the like-for-like Q4_K_M floor against our keepquant arm
Row: FLINT-CPU-PARITY
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-28
Closed: 2026-10-29

## Problem

The developer set llama.cpp as the peer floor for Flint CPU performance. llama.cpp PR #30928 (head 1b44e3077) is the only llama.cpp that loads flint; it is unregistered and its gateability measurement is untracked. Our keepquant Q4_K_M arm (landed 3284412c0) has no throughput number, and the only external reference points (mistral.rs 18.7 tok/s Q4_K_M, llama.cpp I-Mini 24.99/24.33 IQ-class) are not like-for-like denominators. Spec: .agents/specs/flint-cpu-parity.md. Deliverables: (1) gate the llama-cpp-flint oracle (build 1b44e3077, serve flint-Q4_K_M.gguf, golden + tg rates), (2) like-for-like floor on the identical artifact at our harness core/thread budget, (3) our keepquant serving number and the recorded gap; shexp expert_shared_count=2 slicing verified against the golden before any number is quoted. Kernel-speed work files its own row.

## Resolution

2026-10-29, this row's branch: the oracle is GATED — llama.cpp 1b44e3077 served
flint-Q4_K_M.gguf (sha 20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e)
on the 32-63 bench zone; `llama-cpp-flint` flipped to gateable=yes with evidence
docs/bench-evidence/flint-cpu-parity-20261029/. Floor: tg200 26.29 / tg400 25.66 tok/s at 16 t,
26.38 / 24.69 at 32 t. The shexp check verified our loader reads ONE double-width shared
expert (shared_expert_intermediate_size=1024, 2 slots of 512, never two experts). The
keepquant CPU serving number is NOT PRODUCIBLE: the block-carrying tree has no CPU
consumer — the reference forward refused it by name and the paged arm did NOT, dying on an
opaque matmul inner-dims mismatch; fixed here by moving the named audit into FlintForwardPaged
(keepquant-refusal.log). The keepquant-vs-floor gap stays unrecorded pending a CPU keep-quant
dispatch arm, which files its own row; the bf16 paged arm's context numbers (3.25/3.28 at 16 t,
2.33/2.50 at 32 t) are recorded in the evidence report as a bound, not a gap.
