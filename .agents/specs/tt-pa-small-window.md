# SPEC — `TT-PA-SMALL-WINDOW` — the paged-attention device-SDPA small-window defect: deterministic garbage at seq=2/5/7, isolated at last

- Owning row: `TT-PA-SMALL-WINDOW` (record ID `BACKEND-TENSTORRENT-PA-SMALL-WINDOW`)
- Trigger: the QWEN38-DFLASH2-STAGE-C degeneracy hunt's W22 wave grew the shared-cache harness to engine state and — for the first time — **reproduced the compute class in isolation** (commit `67d041d76` on `row/QWEN38-DFLASH2-STAGE-C`): one raw RAC write + one raw PA decode yields deterministic garbage at seq=2 (4.86e33) and seq=5/7 (2.99717e+32 — IDENTICAL bytes), clean at seq=1/3/4/6/8/64; the Copy-seeded (host-path) arm is clean at every seq. The engine's DFlash2 degeneracy (the `daa113b5` NaN-argmax stream) is this defect reached through the engine's allocator/program-cache state.

## The isolated reproduction (the wave's deliverable, already committed on the stage-C branch)

`67d041d76`'s harness: the GEMM trio on one h (bfp8-staged weights), the
embed (kEmbeddingQuant Q8_0), RAC+PA at the engine's KV geometry, the DBuf
hold — 40 reps. The trio/embed/fill/gather/hold legs are 0-bad; the **PA
ingredient is RED**, and stripping the mix entirely keeps it RED. The RAC's
writes land in all 40 reps — the defect is in the PA read/decode side at
small seq. The engine's own PA windows are never this small (out-of-engine
form), which is why the engine-only hunt could not isolate it for 20 waves.

## Waves

- **W1 — characterize**: the defect's boundary — which seq values garbage
  (2, 5, 7 clean at 1/3/4/6/8/64? — map the pattern), which geometry
  (heads/dims/KV layout), Copy-seeded vs RAC-seeded (the host-path arm is
  clean — is the defect in reading RAC-WRITTEN bytes, or in the PA decode's
  small-window kernel path generally?), and whether the identical bytes at
  seq=5/7 pin a stale-read mechanism (the same bytes = the same wrong source
  buffer). Deliverable: the characterization table + the minimal
  reproduction as a focused test, committed.
- **W2 — root cause**: trace the PA decode path at the failing seq (the
  device-SDPA small-window arm — which kernel serves it, what it reads,
  where the garbage enters). Candidates: a small-window kernel branch with a
  wrong page walk; an upstream tt-metal SDPA defect at small windows (the
  pinned substrate — the indexed_fill displaced-write precedent says this
  substrate carries real write-path bugs); an alignment/padding assumption
  that holds at large windows only.
- **W3 — fix + gates**: the minimal fix (ours or a tt-metal patchset commit
  — the patchset rebase context: our pin carries local fixes already, the
  layernorm run-args precedent). Gates: the focused reproduction
  red-before/green-after; the [gdn]/PA suites' failing-SETS unchanged; the
  engine gate — Q4_K_M KV64 drafter-ON captured 3 reps token identity vs
  `d49c53d2...` + TPOT on/off vs 6607 = **THE ACCEPTANCE MULTIPLE** (the
  stage-C row's verdict number, unmeasurable until this lands); UD-Q6_K_XL
  KV8 vs `b36bc61a...` + TPOT vs 5340.9; drafter-OFF 6/6 stands.

## Scope

IN: the PA small-window defect (characterize, root-cause, fix), the
focused tests, the tt-metal patchset commit if upstream. OUT: the DFlash2
loop changes (stage-C owns them), any other PA behavior change.

## Risks/decisions

- If the root cause is upstream tt-metal and unworkaround-able on our
  side: the pinned substrate carries the bug — a patchset commit is the
  precedent-backed path (the layernorm fix shipped that way); record the
  upstream issue text for the eventual pin advance.
- The engine's PA windows are never this small — the DFlash2 verify's
  windows hit the defect through the ALLOCATOR/PROGRAM-CACHE state, not
  the window size directly; the W1 characterization must explain the
  bridge (what the engine's state changes about the small-window path's
  inputs).

## Stop conditions

- W1 shows the reproduction is NOT deterministic (flaky): record honestly —
  the hunt returns to the engine with a flakiness instrument instead.
- The fix requires re-architecting the PA path: NEEDS_DECISION with the
  numbers.

## Git integration

One pull request; worktree `/tmp/tt-pa`; push to `mine`. The stage-C row
consumes this row's fix for its acceptance gate.

## Now

DONE 2026-11-16. The defect is characterized (the seq-parity set is process
state, the invariants are the second-KV-group heads, never-computed bytes,
recycled content), root-caused (OURS: the bf16 SDPA output committed on an
f32-contract output slot; EnsureHostBytes writes at the shadow's width —
the upstream kernel, the RAC write, and the writer-L1-fence candidate are
all falsified), and fixed (PaOutTypedForCommit typecasts the SDPA output to
OUT's dtype in both commit arms). All five characterization arms green; the
existing PA/RAC suites unchanged. The stage-C consumption gates run from
/tmp/q38-stage-c after this branch merges there.

## Owed

- `ISSUE-LOCAL-01M7JYHQJEANK9Z6S0SJWKXWBC` — the TT batched-decode RAC
  device lane's second-slot second-head garbage under rich suite state
  (exposed by this row's consumption gate; the fix shifted the allocator/JIT
  state that gated it).
- `ISSUE-LOCAL-01M7JYJ1WDJJ3M7E23VQCYS6KC` — the program-cache-disabled PA
  decode's value-level garbage under rich suite state (the characterization
  arm that stays red there; out of the production path's reach — the engine
  never disables the cache).

## Outcome

- Measured: the focused sweep (RAC-seeded, 64 seqs) red-before garbage at
  seq=1 (1.43305e+38, byte-identical to the stage-C discriminator's main
  arm), green-after 0 garbage / 64 with max_abs 0.002238 (the bf16
  envelope); the Copy-seeded sweep 31/64 garbage red-before, 0 green-after.
- The mechanism check that pinned it: re-reading the garbage rows' output
  buffer as bf16 pairs matches the oracle to ~1e-3 while the f32 read shows
  1e38 — the values were correct, the write width was not.
- Rejected: the RAC-write-visibility hypothesis (the Copy-seeded arm
  garbages too); the upstream tt-metal SDPA kernel (dev_out clean at every
  call, env-dump bisect); the program-cache run-args class (worse with the
  cache off; the buffer bindings re-patch on every hit at this pin); the
  writer-kernel Blackhole L1-cache fence (applied to the pin tree, changed
  nothing — reverted).
- Defaults: the fix typecasts on device rather than declining f32 outs to
  the host scatter, because the f32 A/B path is a registered production
  arm and the host scatter would forfeit the device-resident o_proj input;
  the typecast is capture-safe by the W4 doctrine (the eager warmup runs
  the same out dtype). f16 outs decline to the host scatter: ttnn carries
  no f16 dtype at this pin.
- The stage-C seq-parity fingerprint (2/5/7) is recorded as a state
  fingerprint of that process, so the hunt's "which seq" question is closed
  as malformed: the invariants travel, the set does not.
- The stage-C consumption gates (2026-11-16, from /tmp/q38-stage-c at the
  merge e41f9fbbb): Q4_K_M KV64 drafter-ON ×3 — token-ids sha256
  `e171d6b1…` all three reps = the recorded DEGENERATE signature, TPOT
  9924.84 ms (the degenerate loop's band, not a number) — **THE ACCEPTANCE
  MULTIPLE REMAINS UNMEASURABLE**: the engine's DFlash2 degeneracy is NOT
  this defect. UD-Q6_K_XL KV8 drafter-OFF: `b36bc61a…` BYTE-IDENTICAL, TPOT
  5370.57 vs the recorded 5340.9 (+0.55%, the rerun band); its drafter-ON
  leg stays capacity-blocked (the standing named blocker). Q4_K_M drafter-OFF:
  `d49c53d2…` BYTE-IDENTICAL, TPOT 6642.71 vs 6607 (+0.54%, the band). The
  W22 mix harness: pa_bad 5/40 → **0/40** — every op 0-bad in all 40 reps.
  The W22 discriminator: fully green (the main arm 1.43305e+38 → 0.00121).
  The suite ledger (the honest comparison, full suite at each head):
  pre-merge 17 failed → post-merge 16 failed; CLEARED by the fix: the
  pure-decode PA case, the W22 harness, the W22 discriminator; UNCHANGED
  pre-existing: kGdnStateGather/Scatter (the ScatterRowsExact guard's
  refusal, present at 67d041d76 in isolation); NEWLY EXPOSED (filed under
  ## Owed): the RAC device lane's second-slot second-head garbage (suite
  context only) and the cache-disabled decode's value garbage (suite
  context only). The suite teardown segfault (GraphTracker::is_enabled at
  process exit, after the summary) reproduces at BOTH heads — pre-existing,
  not this row's.
