ID: ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9
Title: TT device golden gate red at base and head with identical signatures — environment drift, never recorded green
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-28
Closed: -

## Problem

The readback row's fresh reviewer (2026-10-28) found the device golden-argmax test red at base 3284412c0 AND at head d9b55d5e4 with byte-identical failure signatures (7/15 chain, 6 hard flips; base and head flip-record md5 both 59746f64...). Identical md5 = pre-existing environment drift, not a row regression — but it means the device golden gate has never been recorded green at the current head/substrate, so the row's CHAIN claims rest on implementer raw logs rather than a green gate. Owes: an anchoring session that establishes whether the drift is substrate, governor, or box-state (the 548-vs-3358 ms/step discrepancy is the same investigation), then a recorded green device golden gate.

## Resolution

-
