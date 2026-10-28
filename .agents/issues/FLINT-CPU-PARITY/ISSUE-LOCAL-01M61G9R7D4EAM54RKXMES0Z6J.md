ID: ISSUE-LOCAL-01M61G9R7D4EAM54RKXMES0Z6J
Title: Flint CPU parity: register the llama.cpp flint oracle and measure the like-for-like Q4_K_M floor against our keepquant arm
Row: FLINT-CPU-PARITY
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-28
Closed: -

## Problem

The developer set llama.cpp as the peer floor for Flint CPU performance. llama.cpp PR #30928 (head 1b44e3077) is the only llama.cpp that loads flint; it is unregistered and its gateability measurement is untracked. Our keepquant Q4_K_M arm (landed 3284412c0) has no throughput number, and the only external reference points (mistral.rs 18.7 tok/s Q4_K_M, llama.cpp I-Mini 24.99/24.33 IQ-class) are not like-for-like denominators. Spec: .agents/specs/flint-cpu-parity.md. Deliverables: (1) gate the llama-cpp-flint oracle (build 1b44e3077, serve flint-Q4_K_M.gguf, golden + tg rates), (2) like-for-like floor on the identical artifact at our harness core/thread budget, (3) our keepquant serving number and the recorded gap; shexp expert_shared_count=2 slicing verified against the golden before any number is quoted. Kernel-speed work files its own row.

## Resolution

-
