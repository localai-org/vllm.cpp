ID: ISSUE-LOCAL-01M61DNDF79C6XMMB4WC8GESCN
Title: kolibri-1 P150: per-fill D2H readback + Synchronize on the slot path costs two blocking round-trips per expert per decode step
Row: TT-KOLIBRI1-READBACK
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-29
Closed: 2026-10-29

## Problem

The B2b-ii expert-slot path runs ~19 slot fills per decode step, each with H2D 3.9 MiB then a D2H readback + blocking Synchronize + byte-exact verify (kolibri1_tt_forward.cpp:733-743), and each memo miss re-reads the slot from the device a second time with its own D2H + Synchronize (:816-827). The Synchronize drains the queue mid-layer. Meanwhile the shadow already stores full slot bytes host-side (kolibri1_tt_stream.cpp:247-259, invalidated via the evict hook), so the memo-miss device re-read is redundant for compute, and the fill-time verification can defer its drain to one point per layer without weakening the byte-exact gate. Spec: .agents/specs/tt-kolibri1-readback.md. Levers: (1) memo-miss dequant consumes shadow Bytes(); (2) per-fill Synchronize becomes one drain per layer. Out of scope: removing verification, on-device expert compute, residency drain #3452.

## Resolution

Landed 2026-10-29 as 90fe546b9 (PR #3517, squash of row/TT-KOLIBRI1-READBACK). Both levers landed: the memo miss dequants from the shadow's stored bytes (lever 1) and one readback drain per layer with Record-before-enqueue ordering (lever 2, commit d81dd34fb) — the fresh review established the pre-existing fill-time verify was a tautology (compared a buffer against a shadow copied from itself) and the new ordering is the first genuine device-vs-host compare on this path, strictly stronger. Measured same-session A/B on the P150: 3358.1 → 3184.6 ms/step (-5.2%), byte-identical greedy chains 3/3 per arm. The absolute-number anchoring (548-vs-3358) and the first green device-golden record remain owed under ISSUE-LOCAL-01M62AJ8XNK23FG6ZVATKEJ0D9 (MODEL-TEXT-kolibri-1-tenstorrent).
