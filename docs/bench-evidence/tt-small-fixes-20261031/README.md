# TT-SMALL-FIXES evidence — 2026-10-31

Row `TT-SMALL-FIXES`, spec `.agents/specs/tt-small-fixes.md`, issue
`ISSUE-LOCAL-01M686P391579S3SRN675V2XT7`. Two fixes, one capture each.

## Fix 1 — B2BI/B2II env silent-skip

- RED: `red-b2ii-silent-skip.log` — at HEAD `762098300`, build
  `/tmp/build-ttsmallfix` (`-DVLLM_CPP_TENSTORRENT=ON`, Release), the blackhole
  card present, standing recipe (mutex `~/gpu.lock`, `luwen reset`, tt-metalium
  env), and ONLY `VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1`
  set (the documented spelling, real checkpoint): the device leg prints
  `SKIPPED: VT_KOLIBRI1_TT_B2BI_MODEL not set` and the suite reports
  `10 | 10 passed | 0 failed` / `SUCCESS!` — the phantom green.
- GREEN: `green-b2ii-device-leg.log` — same build and recipe at the fix, same
  single env var: the leg now RUNS (`[kolibri1-tt-b2bi] device leg start:
  model=...`); 203 assertions pass, RC=0.
- GREEN (refusal arm): `green-b2ii-bogus-path-refusal.log` — bogus
  `VT_KOLIBRI1_TT_B2II_MODEL=/nonexistent/bogus-checkpoint`: the leg refuses BY
  NAME (`SKIPPED: /nonexistent/bogus-checkpoint is not a Kolibri-1 checkpoint`)
  instead of silently skipping, which also proves the variable is read.