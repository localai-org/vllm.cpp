ID: ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN
Title: B2b-ii: the streaming MoE on the P150 (slot pool, routed path, stream bound)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-09
Closed: -

## Problem

B2b-i landed the dense-resident device forward with the routed-expert tier deliberately absent (the refusal fires by name 450x per decode). The goldens are full-model decodes and cannot be replayed until the routed experts run: the B2b-ii streaming path — FP8_E4M3 slot buffers consuming Kolibri1TTExpertSlotPolicy verbatim, router readback -> host remap -> fetch list -> fetch executor -> Touch(), slot swaps on the ContentChangedSince reset lane, the B1 per-token stream bound asserted at runtime with loud failure, the full 384-expert tier host-side — is owed, and with it the deferred 141/145 token gate.

## Resolution

- 2026-10-09 (PARTIAL — the host half + device arm landed; the device
  GATES are blocked external): the streaming MoE's host half landed
  (slot-pool plan over `PlanKolibri1TTExpertSlotPolicy` unchanged, the
  fetch executor's host half, the LOUD stream-bound guard, the slot
  shadow / readback pivot, the eviction-hook integration, the
  `Kolibri1TTSlotEpoch` reset lane) with red-first device-free coverage
  in `test_kolibri1_tt_b2ii.cpp` (9/9), and the routed path REPLACED the
  B2b-i refusal in the production TT forward (slot pool staged FP8_E4M3
  verbatim; dispatch -> fetch -> stage -> readback-verify -> `Touch()`;
  memoized dequants cleared by `ContentChangedSince`; `vt::MoeCombine`).
  TT family 4/4; CPU battery green. The DEVICE legs (smoke, the 141/145
  token gate, the bench anchor) are OWED: the P150 board is in an
  external failure state (`cq_id 0 is out of range` reproduces on the
  UNCHANGED base commit and both pin lib generations; the documented
  recovery loop does not recover; dmesg shows a PCI rescan) — evidence
  and the full blocker record in
  `docs/bench-evidence/kolibri1-tt-b2ii-20261009.md` section 4. The
  token-gate leg is committed and runs on the first healthy card
  window. Issue stays OPEN on the device gates.
-
