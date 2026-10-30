ID: ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77
Title: Flint CPU paged-engine serving arm (MODEL-TEXT-FLINT)
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-26
Updated: 2026-10-27
Closed: 2026-10-27

## Problem

Flint serves on CPU only through the single-sequence reference forward. RESOLVED (2026-10-27, waves W-a..W-e on row/FLINT-PAGED): the paged arm serves the published Q4_K_M through the full LLMEngine with continuous batching; the synthetic gate gates the house band, the real-weights paged-engine gate adjudicates three continuously-arriving requests against the reference forward at margins 0.036-0.114 nats; the first CPU throughput numbers are recorded (docs/benchmarks/flint-cpu-paged.md). Three shared defects were found and fixed along the way (each closed with its own issue): the cpu_paged_attn sink seed, the missing flint.context_length mapping, and the multi-group slot-mapping write. Docs updated (USAGE/FEATURES/benchmarks).

## Resolution

Landed 2026-10-27 on row/FLINT-PAGED in five committed waves (W-a KV-group gate; W-b paged arm + synthetic gate, which caught the shared cpu_paged_attn sink seed ISSUE-LOCAL-01M5YMB0XZ8723N0688YHMKP3N, the arm's double-added residual, and the in-place QK norm; W-c the real-weights paged-engine gate with continuous arrival, margins 0.036-0.114 nats inside the 2.5-nat band, which caught ISSUE-LOCAL-01M5YRQ1SWC93MDG52QZGDFBY4 (context_length) and ISSUE-LOCAL-01M5Z079R7RFSCPB1F4P9C94S5 (per-group slot mapping); W-d the first CPU throughput numbers (prefill 7.44 tok/s, decode 1.26 tok/s, docs/benchmarks/flint-cpu-paged.md); W-e USAGE/FEATURES/benchmark docs). Gates: test_flint_paged, test_flint_paged_engine, test_flint_gguf, test_flint_forward, test_ops_paged_attn(_neon), test_paged_attn_route, test_mimov2_* green; commit style + trailer checkers green; check-agent-record OK.
