ID: ISSUE-LOCAL-01M3HB24YKDRXTXY85R4CG7VF6
Title: The capture-economics curve: TPOT vs output length on the serving arm
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

The serving arm's TPOT at output_len 4 is ~34 s/token, cold-dominated: the per-REQUEST fixed costs (cold body ~33 s, capture pass ~5.7 s, trace-wait ~31 s) dwarf the per-TOKEN served-replay cost (12 ms wall). Whether this matters is a function of generation length: at output_len 256 the fixed costs amortize over ~251 served steps and TPOT should collapse -- IF nothing grows with length (GDN state windows, KV pages, per-step keepquant repair transients). The curve (TPOT at output_len 4/16/64/256, phase mix per leg, the fitted TPOT(len) = fixed/len + per_token model) is the evidence the next optimization targets hang from. Spec: tenstorrent-capture-economics.md.

## Resolution

-
