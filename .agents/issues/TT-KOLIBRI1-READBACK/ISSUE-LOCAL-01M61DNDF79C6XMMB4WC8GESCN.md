ID: ISSUE-LOCAL-01M61DNDF79C6XMMB4WC8GESCN
Title: kolibri-1 P150: per-fill D2H readback + Synchronize on the slot path costs two blocking round-trips per expert per decode step
Row: TT-KOLIBRI1-READBACK
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-28
Closed: -

## Problem

The B2b-ii expert-slot path runs ~19 slot fills per decode step, each with H2D 3.9 MiB then a D2H readback + blocking Synchronize + byte-exact verify (kolibri1_tt_forward.cpp:733-743), and each memo miss re-reads the slot from the device a second time with its own D2H + Synchronize (:816-827). The Synchronize drains the queue mid-layer. Meanwhile the shadow already stores full slot bytes host-side (kolibri1_tt_stream.cpp:247-259, invalidated via the evict hook), so the memo-miss device re-read is redundant for compute, and the fill-time verification can defer its drain to one point per layer without weakening the byte-exact gate. Spec: .agents/specs/tt-kolibri1-readback.md. Levers: (1) memo-miss dequant consumes shadow Bytes(); (2) per-fill Synchronize becomes one drain per layer. Out of scope: removing verification, on-device expert compute, residency drain #3452.

## Resolution

-
