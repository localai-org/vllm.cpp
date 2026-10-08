ID: ISSUE-LOCAL-01M4D2KFZ8H7B39XE75XYE0AYY
Title: Kolibri-1 TT B2b-i first slice: device bring-up — resident staging on P150 + one verified device op (embedding + first resident projection)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: CLOSED
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: 2026-10-08

## Problem

The B2b device-bound arm of the kolibri1 TT port (spec .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice B2b-i) has no device bring-up: no TT build links against the pin, the resident non-expert slice (~3.09 GiB: attention 1.587, embed+head 1.221, router 0.094, shared expert 0.188, norms ~0.002) is not staged on the card, and no device op is verified against the CPU row. This slice delivers: (1) a TT-enabled build that links (the stale-_ttnncpp.so stub workaround) and starts on the P150; (2) the resident slice staged on device per the wave-A dtype decision (fp8-block projections native FP8_E4M3 row-major packed bytes verbatim; router/norms/embed/head bf16) with byte accounting printed and matching the plan, and the single-device full-model refusal NOT firing for the resident slice; (3) ONE verified device op end-to-end (embedding lookup + the first resident projection on device) compared against the CPU row's output for the same input; (4) a ctest target whose host-side staging checks run without the card, with the device leg documented; (5) evidence under docs/bench-evidence/. Stop conditions: card not visible after reset; stub workaround fails for the main binary; resident staging does not fit 32 GiB.

## Resolution

Landed 2026-10-08 on branch `row/kolibri-tt-b2i` (pushed to `mine`; no PR —
helper handoff). What landed: the shared staging seam
(`vt::tenstorrent::StageResidentOperands` / `ReadbackStagedOperandF32` /
`ReadbackStagedOperandBytes`, `src/vt/tenstorrent/tenstorrent_staging.cpp`),
the resident-slice planner (`PlanKolibri1TTResidentStaging` over the
production registry's loaded weights via the new
`Kolibri1LoadedModelWeights` accessor), and `test_kolibri1_tt_b2i` (host-side
staging gates under ctest with no card + the env-gated device leg).

Device leg PASS on the P150 on BOTH builds — the pin build against the
fresh `/tmp/pin-build` lib64 (smoke `SMOKE_RC=0`, 36,880/36,880 assertions,
86.7 s) and the non-pin tt-metal d20b8e27f29 + gdn stub (73.9 s): real fp8
checkpoint through
`ModelRegistry::Load` (18.8 s); resident plan 3,311,163,520 B (3.084 GiB,
753 tensors) staged as 1103 operands in 2.54 s on chip 0 (dram_free_after
30,833,363,968 B of 34,178,731,008 B); the full-model single-device refusal
fires while the resident slice does not; all 1103 staged operands verified
byte-exact by readback (mismatches=0 — raw-byte fp8 readback; the ttnn host
FP8→F32 pivot flushes subnormals and the weights contain 28,829 of them in
layer-0 q_proj alone); embedding bit-exact (15360 elements, mismatches=0);
layer-0 q_proj GEMM within its stated envelope (8 bf16 unit roundoffs per
unit term-magnitude sum plus store rounding; measured worst ratio 1.375,
envelope_violations=0 — the earlier 2-ulp premise was falsified by
measurement). 36880/36880 assertions, `Status: SUCCESS`, 73.9 s total.

Gates: TT pin compile clean (pin source via the fresh `/tmp/pin-build`
lib64; the stale-lib64 prerequisite resolved for the pin-side build, the
durable copy into the pin tree owed); host-side cases 6/6 (204 assertions)
on the TT build; CPU-only `ctest -R kolibri` 6/6; `check-agent-record`
rc=0; `check-env-doc` rc=0 (`VT_KOLIBRI1_TT_B2I_PROGRESS` allowlisted as
kernel-internal); `agent-preflight.sh --staged` rc=0. Evidence:
`docs/bench-evidence/kolibri1-tt-b2i-smoke-20261008.md`.

Remains owed (not this slice): the greedy decode (the slice's completion
condition), the token gate, the bench anchor, B2b-ii, and the durable
fresh-lib64 copy into the pin tree (the earlier pin device-op hangs were
the stale in-tree libs, not the pin source — no pin bump owed). The #1486-class `env -i` segfault is filed separately as
`ISSUE-LOCAL-01M4D5W0HSEPNP26AT30CEP874` (known residue, not a blocker).
