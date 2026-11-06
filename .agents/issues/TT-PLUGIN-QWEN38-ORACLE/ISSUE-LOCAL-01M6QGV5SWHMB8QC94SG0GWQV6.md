ID: ISSUE-LOCAL-01M6QGV5SWHMB8QC94SG0GWQV6
Title: Pin the dflash2-vision tt-model bundle as a vllm-tt-plugin oracle instance for qwen3.8-27B
Row: TT-PLUGIN-QWEN38-ORACLE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-06
Updated: 2026-11-06
Closed: -

## Problem

mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4 @ 7855cf0d2982064350991bdb8e1ec1db82f914b3 is a self-contained tt-model serving bundle: vLLM + the tt-model plugin + custom kernels, DFlash2 speculative decoding, TWO p300c boards (4 Blackhole chips, 1x4 mesh). Greedy output is token-for-token exact to the target model — a correctness reference for OUR qwen3.8-27B TT path, which has none (the re-bench compared only against itself). Scope limits recorded up front: NOT a throughput denominator on different hardware (4 chips + spec decode + upstream kernels vs our 1 P150 + none + ours — the 2600x gap decomposes); NOT token-comparable across architectures (a 4-chip mesh's reduction order differs). The bundle CANNOT run on this host (hardware mismatch) — the honest gateability record is gateable=no with the named external resource (a 4x-Blackhole box). Deliverable: the oracle record + the AGENTS.md registry row + the re-bench cross-reference; the comparison is deferred until a gateable instance exists. Spec: .agents/specs/tt-plugin-qwen38-oracle.md

## Resolution

-
