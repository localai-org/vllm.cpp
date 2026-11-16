ID: ISSUE-LOCAL-01M7JHW5N40PWJCRG5BAKQN7Y8
Title: TT paged-attention device-SDPA small-window defect: deterministic garbage at seq=2/5/7 — the first isolated reproduction of the DFlash2 degeneracy's compute class
Row: TT-PA-SMALL-WINDOW
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: -

## Problem

The QWEN38-DFLASH2-STAGE-C degeneracy hunt (20+ waves) reproduced the compute class IN ISOLATION (67d041d76 on row/QWEN38-DFLASH2-STAGE-C): one raw RAC write + one raw PA decode yields deterministic garbage at seq=2 (4.86e33) and seq=5/7 (2.99717e+32, IDENTICAL bytes), clean at seq=1/3/4/6/8/64; the Copy-seeded (host-path) arm is clean at every seq; the RAC's writes land in all 40 harness reps — the defect is on the PA read/decode side at small seq. The engine's DFlash2 degeneracy (the daa113b5 NaN-argmax stream, the acceptance multiple unmeasurable) is this defect reached through the engine's allocator/program-cache state; the engine's own PA windows are never this small (out-of-engine form — why 20 waves of engine-only probing could not isolate it). Owed: W1 characterize (the seq/geometry boundary, the identical-bytes stale-read mechanism, the focused reproduction as a committed test), W2 root-cause (our small-window kernel branch vs an upstream tt-metal SDPA defect — the indexed_fill displaced-write precedent says the pinned substrate carries real write-path bugs), W3 fix + gates: the focused red-before/green-after, the [gdn]/PA failing-SETS unchanged, and the stage-C row's verdict gates (Q4_K_M KV64 drafter-ON captured 3x d49c53d2 + TPOT on/off = THE ACCEPTANCE MULTIPLE; UD-Q6_K_XL KV8 b36bc61a; drafter-OFF 6/6 stands). Spec: .agents/specs/tt-pa-small-window.md

## Resolution

- 2026-11-16 (W1, characterization): the boundary is mapped and it moves
  with process state — the stage-C seq-parity set (2/5/7) is that process's
  fingerprint, not the defect's property. Invariants that travel: the
  garbage is always in the second KV group's heads (2/3), never
  bf16-rounded, and its bytes are recycled memory (RISC-V instructions,
  RTA words, NaNs). The Copy-seeded arm garbages too — the RAC
  write-visibility hypothesis is falsified. Evidence:
  `docs/bench-evidence/tt-pa-small-window-20261116/` (flat).
- 2026-11-16 (W2, root cause): OURS, in the commit path. The
  `VT_TT_PA_DUMP=1` bisect shows the raw `sdpa_decode` output clean at
  every call including the garbage ones; `CommitDeviceLogical2D` moved the
  bf16 SDPA output onto the caller's f32-contract output slot and
  `EnsureHostBytes` (tenstorrent_residency.cpp:1587-1597) wrote the host
  bytes at the shadow's 2-byte width into the 4-byte buffer — the f32
  consumer's second half (heads 2/3) read recycled bytes. The bf16 re-read
  of the same buffer matches the oracle to ~1e-3 on every garbage row. The
  upstream tt-metal SDPA kernel and the writer-L1-fence candidate are both
  falsified (the fence was applied, changed nothing, reverted).
- 2026-11-16 (W3, fix): `PaOutTypedForCommit` in tenstorrent_paged.cpp —
  the decode and prefill commit arms typecast the SDPA output to OUT's
  dtype before the commit; unhandled dtypes decline to the host scatter.
  Green: the RAC-seeded sweep 0 garbage / 64 (was garbage at seq=1),
  the Copy-seeded sweep 0 / 64 (was 31/64), payload tracking, fresh-cache
  per seq, and the program-cache-disabled arm all clean; the existing PA
  suites (5 cases) and the RAC bit-exact arm unchanged. Evidence in the
  same directory (`w3-*-GREEN.log`, `w3-*-regression.log`).
- Owed to the stage-C row: the consumption gates (Q4_K_M KV64 drafter-ON
  3 reps token identity + TPOT vs 6607; UD-Q6_K_XL KV8 vs b36bc61a +
  TPOT vs 5340.9; drafter-OFF 6/6) run from /tmp/q38-stage-c after this
  row's fix merges there.
