ID: ISSUE-LOCAL-01M82AGA1RXW8A4GRQHYYT67HE
Title: Correlate Lottolabs P150 trace-capture fixes with the RAC/PA latent defects
Row: QWEN38-DFLASH2-STAGE-C
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-25
Closed: 2026-11-25

## Problem

The Lottolabs Inkling-2 P150 runtime documents five fixes for a second-sequence trace-capture hang (allocate all persistent tensors in ctor, share the prefill gather index, route reshape through row-major, compile all decode buckets before capture, assert new_low_blocks empty). Our two latent TT issues (ISSUE-LOCAL-01M7JYHQJEANK9Z6S0SJWKXWBC RAC slot2/head2 garbage and ISSUE-LOCAL-01M7JYJ1WDJJ3M7E23VQCYS6KC program-cache-disabled PA garbage) plus the W28/W30/W31 writer evidence describe a different but overlapping lifetime/stale-address class. Write a written correlation that states, per fix, what our defect class predicts, whether the mechanism is the same, and whether it transfers to tenstorrent_ops.cpp/tenstorrent_residency.cpp or is inkling2-graph-specific. Evidence-based hypothesis only; no code change.

## Resolution

/tmp/stage-c-close/corr.txt
