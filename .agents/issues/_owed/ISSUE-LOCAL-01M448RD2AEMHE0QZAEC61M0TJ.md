ID: ISSUE-LOCAL-01M448RD2AEMHE0QZAEC61M0TJ
Title: Measure aleph-alpha-inference oracle gateability on a GPU lease
Row: -
State: OPEN
Kind: bug
GitHub: 3396
Mirror: SYNCED
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

The aleph-alpha-inference oracle record (.agents/oracles/aleph-alpha-inference.md) pins 049a6a7bd240 but records gateable = no: the plugin has never demonstrably built and run Aleph-Alpha/Kolibri-1 (78.9 GB bf16, mirrored at /mnt/models/Aleph-Alpha/Kolibri-1). The measurement owed: vllm serve with the plugin at the pin on a GPU lease, greedy decode of a fixed prompt set, recorded per AGENTS.md 'When vLLM has no implementation' and 'Measure gateability'. Until then the oracle file cites this issue as evidence, and scripts/check-oracle-pins.py requires a gateable = no record to name its owing issue as #N.

## Resolution

-
