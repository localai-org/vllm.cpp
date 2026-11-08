# QWEN38-DFLASH2-STAGE-B host evidence

Note on superseded logs: `red-geometry-mismatch-d2d-copy.log` and
`green-focused-tests.log` correspond to an EARLIER test iteration of the
capture-lane D2D recorded-geometry case. The committed test relied on the
device matmul consumer, which the fresh review showed cannot detect a
clobbered record (the device path serves flat-equal bytes). Both logs are
superseded by:

- `green-recorded-geometry-d2d-copy.log` — the rewritten case (asserts the
  slot's dev_rows/dev_cols record via DeviceShadowExact) green at head.
- `red-recorded-geometry-d2d-copy-mutation.log` — the same case with the fix
  mutated back to the baseline src-shaped install
  (tenstorrent_residency.cpp:1552 `if (d->device_current && ...)` ->
  `if (false && ...)`): both recorded-geometry assertions fail. The source
  was restored byte-for-byte after the run.

Run recipe: `build/tests/test_tenstorrent_backend --test-case="*capture-lane
D2D copy preserves an equal-numel dst slot's recorded geometry*"` with
`TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` at the local tt-metal pin, under the
`${HOME}/gpu.lock` mutex, on the P150 (/dev/tenstorrent/0).

## Log index (2026-11-08 fresh-review repair series)

- `green-recorded-geometry-d2d-copy.log` /
  `red-recorded-geometry-d2d-copy-mutation.log` — finding 1: the rewritten
  recorded-geometry case (asserts dev_rows/dev_cols via DeviceShadowExact).
- `green-shadow-dtype-decline.log` /
  `red-shadow-dtype-mismatch-mutation.log` — finding 2: the dtype-mismatch
  decline case.
- `green-interior-pointer-decline.log` /
  `red-interior-pointer-guard-mutation.log` — finding 3: the interior
  destination-pointer decline case.
- `failing-cases-reconciled.md` + `full-suite-head-repaired{,-rep2}.log` —
  finding 4: the reconciled failing-case attribution. Supersedes the
  head/baseline pairing presented by `full-suite.log` and
  `baseline-failing-cases.log` (see the reconciled file for why).
