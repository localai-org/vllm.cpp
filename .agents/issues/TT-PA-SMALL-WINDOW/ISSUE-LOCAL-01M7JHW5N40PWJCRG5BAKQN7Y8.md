ID: ISSUE-LOCAL-01M7JHW5N40PWJCRG5BAKQN7Y8
Title: TT paged-attention device-SDPA small-window defect: deterministic garbage at seq=2/5/7 — the first isolated reproduction of the DFlash2 degeneracy's compute class
Row: TT-PA-SMALL-WINDOW
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: 2026-11-16

## Problem

The QWEN38-DFLASH2-STAGE-C degeneracy hunt (20+ waves) reproduced the compute class IN ISOLATION (67d041d76 on row/QWEN38-DFLASH2-STAGE-C): one raw RAC write + one raw PA decode yields deterministic garbage at seq=2 (4.86e33) and seq=5/7 (2.99717e+32, IDENTICAL bytes), clean at seq=1/3/4/6/8/64; the Copy-seeded (host-path) arm is clean at every seq; the RAC's writes land in all 40 harness reps — the defect is on the PA read/decode side at small seq. The engine's DFlash2 degeneracy (the daa113b5 NaN-argmax stream, the acceptance multiple unmeasurable) is this defect reached through the engine's allocator/program-cache state; the engine's own PA windows are never this small (out-of-engine form — why 20 waves of engine-only probing could not isolate it). Owed: W1 characterize (the seq/geometry boundary, the identical-bytes stale-read mechanism, the focused reproduction as a committed test), W2 root-cause (our small-window kernel branch vs an upstream tt-metal SDPA defect — the indexed_fill displaced-write precedent says the pinned substrate carries real write-path bugs), W3 fix + gates: the focused red-before/green-after, the [gdn]/PA failing-SETS unchanged, and the stage-C row's verdict gates (Q4_K_M KV64 drafter-ON captured 3x d49c53d2 + TPOT on/off = THE ACCEPTANCE MULTIPLE; UD-Q6_K_XL KV8 b36bc61a; drafter-OFF 6/6 stands). Spec: .agents/specs/tt-pa-small-window.md

## Resolution

FIXED 2026-11-16: the bf16 SDPA output committed on an f32-contract out; PaOutTypedForCommit typecasts the commit to OUT's dtype in both arms. Focused red/green (0 garbage / 64 swept, was garbage at seq=1), all five characterization arms green, the W22 mix harness pa_bad 5/40 -> 0/40, the W22 discriminator 1.43e38 -> 0.00121, the pure-decode PA suite case cleared. The consumption gates ran at the merge e41f9fbbb: the acceptance multiple stays UNMEASURABLE (the drafter-ON stream remains the degenerate e171d6b1 signature, 3/3) — the engine's DFlash2 degeneracy is NOT this defect; the OFF arms byte-identical in band. Two latent defects newly exposed, filed under the spec's ## Owed
