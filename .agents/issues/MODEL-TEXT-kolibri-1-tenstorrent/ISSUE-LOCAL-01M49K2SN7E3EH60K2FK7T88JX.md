ID: ISSUE-LOCAL-01M49K2SN7E3EH60K2FK7T88JX
Title: Kolibri-1 Tenstorrent (P150) port — wave A: staging plan, FP8-block on-device decision
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-06
Updated: 2026-10-06
Closed: -

## Problem

the kolibri1 TT port: wave A registration/config/loader. The CPU row MODEL-TEXT-kolibri-1 is landed; nothing stages kolibri1 on Tenstorrent. Wave A (device-free): the TT staging plan over the landed FP8-block weights — native FP8_E4M3 per the tt-metal pin (blackhole LLK supports Fp8_e4m3), bf16 for router/norms/embed/head, byte accounting against the 32 GiB P150 (the 384x50 expert staging is 73.6 GiB, so a single-device full-model stage must refuse by name), GGUF refused by name. The device forward, mesh staging, and the oracle gateability measurement are owed waves.

## Resolution

-
