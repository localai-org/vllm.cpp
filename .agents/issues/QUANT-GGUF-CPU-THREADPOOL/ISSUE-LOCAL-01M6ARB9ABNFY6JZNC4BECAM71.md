ID: ISSUE-LOCAL-01M6ARB9ABNFY6JZNC4BECAM71
Title: Reconcile CPU threadpool PR with upstream attention changes
Row: QUANT-GGUF-CPU-THREADPOOL
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

The CPU threadpool PR conflicts with upstream/main in the paged-attention work split. Preserve the PR query-row cache and upstream accumulator width while integrating upstream.

## Resolution

2026-11-01. Merged upstream/main 2359732b3563fc60bf216ee39c09b098c8e1eadb into the PR branch. Preserved the PR token query-row cache and upstream d_v accumulator allocation. Reconstructed the quantization matrix from upstream and reapplied only its threadpool row, with unrelated keyed rows byte-identical. The three focused paged-attention and threadpool suites passed at CPU thread counts 1, 3 and 20. Static archive linking initially hit disk quota; an equivalent thin archive and the generated linker commands produced the tested binaries.
