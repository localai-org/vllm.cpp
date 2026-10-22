ID: ISSUE-LOCAL-01M5HHEV4DWX0D4S5E955VE0AN
Title: kolibri1 P150: the chunked prefill path is not reproducible run to run, so its flip set is not a measurement of precision
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: -

## Problem

The recorded chunked-prefill verdict (ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D) reads 'prefill-kernel precision, not walk logic', and it rests on a chunked-vs-serial comparison that never had a SAME-MODE NULL ARM. The null arm was measured on 2026-10-21 (P150, origin/main 3ce14f35c, /tmp/build-tt-taps, the 6-token chunked forward repeated in one process, and across processes with and without a device reset and a kernel-cache clear): the CHUNKED path is not reproducible and the SERIAL walk is. Chunked, three repeats in one process: the same token 42046 with the top-1 logit at 6.930672, then 7.048378, then 6.747927 nats - a ~0.3-nat move with an unchanged argmax; a second process in a different device state read 3.597120, 6.964228, 6.917763 - a 3.4-nat spread at the first repeat. Serial, two repeats: byte-identical (74493 at 8.055458 both). A device reset plus a kernel-cache clear changed nothing, so it is not a cold/warm artefact. The per-layer taps localise it: the divergence is born at layer 0's LAST query row (attn_out_pre_norm, row 5) where ~50% of the elements differ by 1-2 bf16 ULP (max 9.77e-04), and it grows through depth (l_out rms 3.24 at layer 49); the router's expert ORDER flips in 36 of 50 layers while the expert SETS are identical - a different summation order over the same experts. The chunk-vs-serial difference is 5x to 600x the run-to-run noise at every layer, so the recorded verdict's DIRECTION survives (the divergence is not born in the walk logic), but the flip set it measured cannot be read as a precision signature, and no chunked-vs-serial comparison is finer than the null arm until the path is reproducible. CONSEQUENCE: the chunked-prefill verdict must be corrected, the chunk sweep instrument must carry a same-mode null arm, and the three tap artifacts must be labelled by mode (serial deterministic, chunked not). The chunked path stays opt-in.

## Resolution

- 2026-10-22 (row/TT-CHUNKED-NULL-ARM): the null arm this record owed is
  measured, and the record that leaned on the missing arm is corrected.

  What landed: the corrected verdict in the chunked-prefill record
  (.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D.md,
  whose 2026-10-18 entry is now marked superseded with its historical text
  kept), and the null arm in the instrument. The SCRATCH chunk sweep
  (VT_KOLIBRI1_TT_CHUNK_SWEEP=1 in tests/vllm/models/test_kolibri1_tt_b2ii.cpp)
  now runs the SAME chunked walk twice per chunk size and prints the
  chunked-versus-chunked difference, and its ratio to the chunked-versus-
  serial difference, beside the chunked-versus-serial one. The instrument
  stays opt-in and changes no shipped default.

  VERDICT: the chunked-vs-serial divergence is 5x to 600x the chunked path's
  own run-to-run spread, so the divergence is NOT born in the walk logic.
  The flip set the 2026-10-18 chunk sweep measured is contaminated by the
  path's own nondeterminism, so it CANNOT be read as a precision signature.
  The verdict "inherent multi-token prefill kernel precision" is NOT
  ESTABLISHED by the evidence the record cites. The chunked path is not
  reproducible run to run: three repeats in one process read the same token
  42046 with the top-1 logit at 6.930672, then 7.048378, then 6.747927
  nats, and a second process in a different device state read 42046 at
  3.597120, then 6.964228, then 6.917763, a 3.4-nat spread at the first
  repeat, while two serial repeats were byte identical at token 74493 and
  8.055458 nats. The provenance, the per-layer localisation, and the labelled
  tap artifacts are in
  docs/bench-evidence/kolibri1-tt-chunked-null-arm-20261022.md.

  The tap artifacts are labelled by mode in the evidence file: the CHUNKED
  prefill dumps are NOT reproducible, the two SERIAL walk dumps are byte
  identical, and the CPU layer dump is a CPU reference and not a device
  walk. One refinement of the problem statement above: of the 36 layers
  whose router expert ORDER changes between two chunked runs, 17 keep the
  same six experts in a different order and 19 change the expert set.

  Owed, and still open: the chunked prefill path's reproducibility, a re-run
  of the chunk sweep with the null arm the instrument now carries, and the
  mechanism hunt for the divergence the taps localise. The taps show the
  shape of that divergence, born at layer 0's last query row at one to two
  bf16 ULP and growing with depth, with the router's expert order flipping
  between two runs of the same path. The measurements show the shape and not
  the cause, so no mechanism is claimed here.
