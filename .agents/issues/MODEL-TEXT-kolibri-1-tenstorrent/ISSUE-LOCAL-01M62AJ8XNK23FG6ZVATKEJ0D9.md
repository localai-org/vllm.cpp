ID: ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9
Title: TT device golden gate red at base and head with identical signatures — environment drift, never recorded green
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-30
Closed: -

## Problem

The readback row's fresh reviewer (2026-10-28) found the device golden-argmax test red at base 3284412c0 AND at head d9b55d5e4 with byte-identical failure signatures (7/15 chain, 6 hard flips; base and head flip-record md5 both 59746f64...). Identical md5 = pre-existing environment drift, not a row regression — but it means the device golden gate has never been recorded green at the current head/substrate, so the row's CHAIN claims rest on implementer raw logs rather than a green gate. Owes: a...

## Resolution

- 2026-10-30 (row/TT-ANCHORING, evidence
  `docs/bench-evidence/tt-anchoring-20261030/`): gate re-run at head
  `8d1eb0daa` AND at the 2026-10-21 tree `3ce14f35c`, fresh luwen reset +
  tt-metal cache clear per leg, performance governor, idle host. RED at both
  eras, same class, same magnitude (39–41 teacher-forced hard flips; chain
  3/11 to 7/15; head-2 reproduces the recorded 7/15+6-hard signature).
  ROOT-CAUSE CLASS: substrate-level numeric divergence in the device
  execution path (tt-metal pin a585e5744a8 runtime / device+driver state) —
  NOT our code (both eras equally red), NOT governor (swept both states,
  2.8% on the bench), NOT host load, NOT the readback path (all staged bytes
  verify byte-exact). Flip positions stable across runs/eras (early steps of
  specific prompts, gaps 3–12 nats); flipped tokens are not run-stable, so
  the 59746f64 md5 was per-run, not a reproducible signature. Substrate swap
  to the only other install (2026-09-25 pin d20b8e27f29) is ABI-blocked
  (head calls a ttnn symbol the older pin lacks); recorded, not pursued.
  STAYS OPEN: the gate has never been recorded green; close when the
  substrate-side investigation lands a green (or a named pin/driver fix).

