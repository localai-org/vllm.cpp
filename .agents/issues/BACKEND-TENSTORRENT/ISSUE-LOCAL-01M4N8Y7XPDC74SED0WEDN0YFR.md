ID: ISSUE-LOCAL-01M4N8Y7XPDC74SED0WEDN0YFR
Title: kolibri1 TT: enable captured decode / dispatch batching
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-21
Closed: -

## Problem

kolibri1 P150 decode runs eager (~10 device ops/layer x 50 layers, ~960 ms/step, ~1 tok/s) because it lacks the Warm*/capture driver hooks that qwen3.5 implements; both device capture arms decline on kolibri geometry. Produce gap analysis and prototype.

## Resolution

2026-10-21: STILL TRUE as a gap; the study this issue asks for has landed.
`docs/gap-kolibri1-tt-captured-decode.md` is the committed gap analysis and
records the prototype's verdict: capture is NOT feasible for this model without
a MoE-arm redesign, because the per-layer router-logits download and the
host-driven streaming expert fetch with per-miss readback pivot verify sit
inside any candidate captured region, and a ttnn trace prohibits mid-region host
reads. The prototype (a router-bias host cache) showed no measurable win --
1152 ms/step against a 1077/1254 ms/step baseline, inside the run-to-run noise.

The gap itself is unchanged: both device capture arms still decline on kolibri
geometry through the %32 sdpa_decode TILE validation
(`src/vt/tenstorrent/tenstorrent_paged.cpp:352` and `:666`), so the decode stays
eager at about 960 ms/step. The three levers the study re-ranks (the
expert-fetch readback and verify cost, a device-side `vt::MoeRouterTopK` port,
and cache block_size 32 with a `WarmRacIdx` entry) are still open.

## Evidence (2026-10-11, gap-analysis session, branch row/tt-kolibri-capture)

Study committed as docs/gap-kolibri1-tt-captured-decode.md. Capture
verdict: NOT feasible for this model without a MoE-arm redesign — the
per-layer router-logits Download (kolibri1_tt_forward.cpp:478) and the
host-driven streaming expert fetch with per-miss readback pivot verify
(:693-701) sit inside any candidate captured region; ttnn trace
prohibits mid-region host reads. The %32 decline
(tenstorrent_paged.cpp:666/:352) is NOT a hard tt-metal constraint: it
is the sdpa_decode TILE validation, and block_size is our cache
allocation — block_size 32 on the TT arm un-declines it (plus a
WarmRacIdx entry for its shapes).

Prototype (router-bias host cache, removing 50 blocking per-step
syncs): NO measurable win — decode 1152 ms/step vs baseline
1077/1254 ms/step across two control runs (+-10% run-to-run noise;
greedy chains unreproducible run-to-run in all arms, consistent with
the row's 7-flip residue). Finding: the bias syncs wait only on the
tiny router GEMM tail. Re-ranked levers: (1) expert-fetch readback/
verify cost per slot miss, (2) device routing via a vt::MoeRouterTopK
TT port to remove the per-layer full-tail download, (3) cache
block_size 32 + WarmRacIdx to enable the device RAC/PA arms.
