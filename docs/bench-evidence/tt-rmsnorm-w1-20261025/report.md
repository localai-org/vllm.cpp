# Tenstorrent rms_norm substrate repair W1 (2026-10-25)

Spec: .agents/specs/tt-rmsnorm-repair.md (commit 022f4198c, before the
implementation). Implementation: 7305b695e. Branch row/TT-RMSNORM-W1, base
b1597b6e2 (origin/main + the W0 branch — the W0 harness was needed; the
deviation is in the spec).

## The repair

The composed f32 chain (x^2 -> mean -> +eps -> rsqrt -> scale -> gamma) was
always micro-exact; what the two refused repairs could not deliver was the
row scale INTO the model. W1's delivery: the inv as a 3D-broadcast multiply
([rows,1,1] against [rows,1,d], the tenstorrent_gdn.cpp:640-667 delta-rule
pattern). The refused plain 2D multiply leaked the [rows,1] operand's
padded-tile garbage into every column; the 3D reshape hits a broadcast path
that does not. BIT-EXACT vs the CPU arm on every captured in-model row
(normrow-000..007 + random: sum ratio 1, max_abs 0, 0 differing elements),
where the pinned ttnn::rms_norm reads 0.9899-1.0077 input-dependently.

Arm inventory (VT_TT_RMSNORM_COMPOSED; VT_TT_RMSNORM_TTNN=1 restores the
pinned arm):

- 1 repeat+same-shape multiply — refused in-model (binary_ng "Invalid
  subtile broadcast type"); kept as the reachability proof only.
- 2 3D-broadcast multiply — DEFAULT; bit-exact on every captured row.
- 3 ROW_MAJOR inv multiply — also exact (no tile padding to leak).
- 4 matmul outer product — diagnostic; tf32-bounded per the gdn record
  (b67b30fd: ttnn::sum is the tf32 tile-reduce, ttnn::mean the f32 SFPU
  path — the reduce side of this arm uses mean).

## The honest model-level verdict

- Flint W0 harness (same binary A/B): rel_rms 2.579e-1 (old) -> 2.568e-1
  (new). UNCHANGED at the model level.
- Kolibri device token gate (test_kolibri1_tt_b2ii -tc="*device token
  gate*"): OLD arm argmax 3/141, 5 hard flips, 40 instruction flips;
  NEW arm 3/141, 6, 42. The gate was ALREADY collapsed at this level
  (the recorded prompt-dependent hard divergence); the new arm is
  equivalent within noise. NO REGRESSION.

THE W0 ATTRIBUTION IS FALSIFIED: per-op rms_norm exactness does not move
either model's device-vs-CPU number. The ~1% bias compounding over 49
layers was not what dominates the model-level divergence. Named next
suspect (W2): ttnn::matmul truncates f32 operands to tf32 on this pin —
every f32-operand matmul in the device forward loses ~2^-11 and
compounds; the flint_tt arm's operand dtypes need the audit.

## Recipe

- Host: aarch64 Ampere-1a, P150 (local non-fleet device, $HOME/gpu.lock
  flock around every run, ~/Sources/tt/luwen/target/release/reset before
  the first run, pinned substrate /tmp/umdtrial-install +
  TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin).
- Micro red: the recorded tt/host2=1.0072 on the replayed normrow-000
  (matches the compute-config record and tt-metal#62339).
- Flint: FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf
  test_flint_tt_w0 (recipe: ../flint-tt-w0-20261025/report.md).
- Kolibri: VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1
  ./tests/test_kolibri1_tt_b2ii -tc="*device token gate*".
- Logs: /tmp/w1-kolibri-after.log (new arm), /tmp/w1-kolibri-ttnn.log
  (old arm).

## Implementer note

The W1 implementer subagent was interrupted by its API rate limit after
the arms, the in-model reachability proof, and the flint A/B; the
coordinator completed the kolibri A/B, committed the implementation
verbatim (7305b695e), and wrote this report from the logs. The fresh
review is still owed on the whole unit.

## REVIEW ADDENDUM (2026-10-25, reviewer minors discharged)

- BAND DEFERRAL STATED: the W0 gate band (test_flint_tt_w0.cpp:280-281,
  max_abs<64 / rel_rms<0.5) is deliberately NOT tightened by this change —
  the honest baseline is unchanged, so tightening now would only dress the
  number; the band tightens when W2 (the matmul tf32 suspect) closes and
  the model-level distance actually drops.
- RAW LOGS PRESERVED: w1-kolibri-after.log, w1-kolibri-ttnn.log and the
  Flint A/B green.txt are committed beside this report; /tmp paths do not
  survive a reboot.
- BUILD RECIPE PINNED: configure with VLLM_CPP_TENSTORRENT=ON — the AUTO
  setting silently omits the device model tests.
- ISSUE RECORDS: the A/B falsification is recorded in the owning issues
  (see the records commit); ISSUE-LOCAL-01M4YVEP000ENCRECVB0S6FDDY carries
  the dated entry.

> EVIDENCE-LOST (2026-10-30): `w1-kolibri-after.log`, `w1-kolibri-ttnn.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
