# QWEN38-DFLASH2-STAGE-C W3 — the speculative loop: wiring + the two named device blockers (2026-11-15)

Row: `QWEN38-DFLASH2-STAGE-C`, issue
`ISSUE-LOCAL-01M73J3QJ5CEBP2YS3ESK2ZMSE`, spec
`.agents/specs/qwen38-dflash2-stage-c.md`. Head at measurement: the W3
commit on `row/QWEN38-DFLASH2-STAGE-C` (745f5edde + W3).

## What W3 wired (all behind `VT_DFLASH_DRAFT`, default OFF)

- The DIRECT-ARGMAX draft arm (`src/vllm/v1/worker/gpu/runner.cpp`,
  `DflashDraftDirectArm` + `dflash_direct_arm_`): a DFlash2-classified draft
  proposes through the bundle's `dflash2.py` `has_selector == false` branch
  (full-vocab argmax per block row over the W2-validated block forward) instead
  of the candidate selector + path walk. Default OFF is byte-identical: the
  selector lane, its refusals (`RefuseDflash1ArgmaxOnDflash2Block`,
  `RefuseQuantizedDflash2LmHead`) and its device residency are untouched.
- The GGUF shared head is now KEPT PACKED
  (`qwen3_5_gguf_weights.cpp::LoadGgufSharedEmbedAndHeadBf16`) and the draft's
  block-logits GEMM takes the target's own block-quant `MatmulBT` dispatch
  (`DflashLogitsF32D`'s keep-quant arm) — the #1628 ADMITTED state, and the
  rebind (`BindDflashDraftSharedHead`, model_loader.cpp) gives the draft the
  target's resident head owner, so no second head is uploaded (2.44 GB bf16
  twin + 1.04 GB packed copy avoided; banner line in the logs).
- The draft's own bf16 weights are staged AT WIRING time
  (`GPUModelRunner::PreStageDflashWeights`), the `PrepareBf16Resident` pattern:
  a lazy first-propose stage at a 27B footprint allocated against a fragmented,
  ~99-percent-full bank set and OOM'd; the same bytes land fine straight after
  the weights load.
- `kIndexSelect` on Tenstorrent (`tenstorrent_ops.cpp`): the row gather the
  accepted-prefix context append needs (the scatter twin `kIndexCopy` landed in
  W2). One-hot matmul composition, exact in bf16, refuses f32 by name.
- Verify/accept/GDN-rollback: the engine's existing machinery — the greedy
  verify is `vt::GreedyRejectionSample` (token-id compare cascade; commit =
  longest accepted prefix + the first differing verify token), the reject
  length `num_rejected` drives the scheduler rollback and the stage-B
  `StageGdnStateShadow`/`RestoreGdnStateShadow` commit-or-restore rolls the
  GDN state back. Host-side focused gate extended:
  `tests/vllm/v1/spec_decode/test_rejection_sampler.cpp` — two new k=8
  DFlash2-shaped cases (full accept commits draft+bonus; reject at depth 3
  commits prefix+target argmax with `num_rejected = 5`), 21/21 cases,
  299+ assertions green.

## Blocker 1 (RESOURCE): the 27B k-quant target + a resident bf16 drafter does not fit one P150

The board is 8 banks x 4,272,341,376 B. With UD-Q6_K_XL served (the TT-
STAGING-FOOTPRINT row-page repack, eager arm KV 8 — the only servable config
for that arm), the target + its keep-quant row words + the embed dual
residency already fill DRAM to within ~100 MB (that row's own record: the
captured arm at KV 8 OOM'd by ~80 MB). The drafter adds ~3.5 GB bf16 +
~1.05 GB lazy keep-quant stagings. Even with the head share (-3.5 GB) and the
direct arm (-254 MB selector codebooks), every KV size fails:

- KV 8: the 3-group spec admission check cannot fit one request
  (`auto-fit ... not enough GPU memory available to serve even a single
  token`, w3-probe2.log).
- KV 16: OOM at a 94.7 MB `EnsureKeepQuantRowWords` staging — and drafter-OFF
  KV 16 eager ALSO OOMs (w3-off16.log): KV 16 is over the line with no drafter
  at all.
- KV 32/48/64: OOM at the same stagings, 20-60 MB short, fragmentation-bound
  (largest free block < the request), w3-nb32/nb48/da16/da64/w3-pre32.log.

The spec's own baseline (stage B, `## Our baseline`) is Q4_K_M KV 64 — on that
target the drafter loads and the engine reaches the spec steps.

## Blocker 2 (PORT GAP): the TT GDN spec verify kernels are not registered

On Q4_K_M KV 32/64 with the drafter ON, the loop runs — draft (direct-argmax
arm), the (S, 9, spec) verify forward, sample — and dies at the FIRST GDN
spec op:

`vt: causal_conv1d_spec_update: conv_state must be f32, or bf16 on CUDA
(src/vt/ops.cpp:2401)` — `kCausalConv1dSpecUpdate` has no Tenstorrent
kernel (CPU + CUDA only; `tenstorrent_ops.cpp` registers `kCausalConv1dUpdate`
but not the spec variant), and the dtype check refuses TT outright. Stage B
captured the spec arm but never ran the GDN spec path end-to-end. The
port of `kCausalConv1dSpecUpdate` (and whatever follows it in the GDN spec
segmentation) is the remaining W3 work; without it no token-identity or TPOT
number exists for the loop on this device.

One probe got PAST the memory wall with `VT_TT_WEIGHT_RESIDENCY=bfp8` (the
draft's bf16 weights typecast to BFP8_B in place, dropping the bf16 staging)
and ran the spec steps up to that op (w3-bfp32.log) — recorded as evidence
that the memory side closes on Q4_K_M.

## Gates this wave

- Focused: rejection_sampler 21/21 (the two new k=8 cascade cases green;
  mutation red owed to the fresh reviewer per the row discipline).
- Device gate (W3-d): NOT RUN — blocked by blockers 1 and 2 above; the greedy
  token-identity STOP condition never got a workload to serve.
- `--staged` preflight: see the commit message for the run on the exact staged
  state.

## Continuation (2026-11-15, same day — the W3 kernel landed)

The staged W3 delta was audited coherent and carried forward; the missing
kernel is ported. State on this base (pre-embed-fix, 745f5edde + W3):

1. `kCausalConv1dSpecUpdate` is PORTED to Tenstorrent
   (tenstorrent_gdn.cpp, the transposed-shadow composition of the decode
   kernel's proven primitives). The gather indices split by content
   stability: slot/cu_seqlens-derived indices are host-built and
   content-cached (GdnConvIdxEntry's discipline); the acceptance-derived
   indices (the window read offset and the roll's state-part rows) are
   derived ON DEVICE from the uploaded acceptance tile, because a
   host-built index would bake one step's acceptance into a captured graph
   and silently corrupt every later replay. Focused gate:
   tests/vt/test_tenstorrent_backend.cpp kCausalConv1dSpecUpdate, 39/39
   assertions EXIT 0 on the P150 vs the CPU f32 oracle — captured shape
   out max_abs=0 (f32 MAC) / 1.5e-8 (silu one ulp), rolling continuation
   3 steps <= 3e-8 with the state bit-exact, mixed seqlens <= 3e-8, NULL
   slot byte-exact, bf16-state cache bit-exact. RED before: the same case
   refuses by name with the kernel absent (w3-specconv-red.log). Green:
   w3-specconv-green.log.
2. ops.cpp's spec-update dtype contract now asks
   Backend::SupportsCompressedConvState instead of naming CUDA — the
   hard-coded name refused the TT spec verify before its kernel could
   resolve (blocker 2's exact line).
3. probe-1.log (Q4_K_M KV 64, drafter-ON, direct arm): the head-share
   banners fire (embed 2542796800 B + head 1042944000 B saved), the draft
   pre-stages at wiring, and the first propose still OOMs — 178 MB short
   with 21.5 MB free. On THIS base the drafter beside the Q4_K_M target at
   KV 64 does not fit; blocker 1 stands here and closes with the embed fix
   on main (PR #3563, ~2.4 GB freed).
4. The captured-replay arm of the spec verify is named-and-refused by the
   kernel's own check for now: the acceptance metadata has no persistent
   device refresh on TT (the driver's StageSpecStepInputs memcpy leaves
   the slot host-current with no device shadow, and tt-metal records no
   writes in-region), so a captured spec verify would replay one step's
   num_accepted forever. Until the driver-side in-place device refresh
   lands (the WarmRopeCosSin copy_to_device pattern), the spec verify runs
   on the eager arm (VLLM_CPP_CUDAGRAPH=0), which is also the arm the
   token-identity gate needs.

## What the next wave owes

1. The device gates on the restacked base (the embed fix merged): the
   Q4_K_M KV-64 drafter-ON greedy token identity vs the recorded
   no-drafter stream d49c53d2…, the TPOT ON vs OFF table, the drafter-OFF
   byte-identity, and the UD-Q6_K_XL drafter arm.
2. The GDN spec segment's next op: kGdnSpecDecode is also CPU+CUDA only —
   the loop's next by-name refusal once the conv op resolves. The TT port
   follows GdnDecodeStepComposed (initial state from column num_accepted-1,
   T sequential composed steps, per-timestep slot snapshots).
3. The driver-side TT spec-metadata in-place device refresh (the
   WarmRopeCosSin pattern) for the captured spec verify arm.

## W3 finish (2026-11-15, the OOM root cause + the gate disposition)

The Q4_K_M KV-64 drafter-ON crash (`w3-q4km-on.log`, `probe-1.log`) is
ROOT-CAUSED, and it is neither of the two hypotheses the leg named.

**Root cause: capacity, not staging.** The exact error is a TT DRAM
allocator OOM — `Not enough space to allocate 178257920 B` (the draft's
`ffn_down` [5120, 17408] bf16) inside `EnsureMatmulWeightDevice` under the
draft trunk's first forward — with the board at 4250752960/4272341376 B
per bank (99.5%). The ledger, measured with `VT_TT_STEP_PHASES=1`
(`w3f-diag-off-env.txt` legs): the Q4_K_M target at KV 64 sits at
**5366 MiB free**, stable from decode step 3 (the GDN layers'
bf16-expanded projections alone are 10.4 GiB staged: `[10240,5120] x48`,
`[6144,5120] x48`, `[5120,6144] x48` — the kTransformedWeight bf16
expansion the loader documents). The drafter's bf16 trunk stages ~3.3 GiB
of DRAM at the first propose (`VT_STAGING_TRACE`: `gate_up [34816,5120]
x5`, `down [5120,17408] x5`) plus ~2 GiB of zeros primes and scratch
retention — the fifth layer's `ffn_down` lands 6.9 MiB short.

**The two prior legs each had only ONE of the two shares active**, which
is why both died the same way:
- `w3-q4km-on.log` (06:16, pre-embed base): embed shared via the bf16 twin
  (2542796800 B saved), the head bind silently skipped
  (`shared_lm_head` null → no banner) → the draft's own bf16 head resident.
- `probe-1.log` (10:51, post-embed base): head shared (1042944000 B
  saved), the embed share DECLINED by the dtype-equality guard ("the
  draft read bf16 [248320, 5120] and the target holds q4_K") → the
  draft's own 2.44 GiB bf16 table resident.

**The fix (this wave):**
1. The W3 share-gate block-quant arm (`BindDflashDraftSharedEmbed`,
   model_loader.cpp — the uncommitted edit, audited and completed): a
   block-quant target table now shares with the draft's float table — the
   draft's four embed lookups decode the packed rows on the fly
   (RefreshQuantEmbedRows), so the aliasing hazard the f16 case names does
   not exist. Both banners now fire together
   (`w3f-on-probe.stderr.log`: embed 2542796800 B + head 1042944000 B).
   `IsFloat` moved to dtype.h/dtype.cpp (the ops.cpp anonymous-namespace
   duplicate collided with it); exactly one `if (!same)` guard remains.
2. `VT_TT_WEIGHT_RESIDENCY=bfp8` on every DRAFTER-ON leg (the row's
   recorded memory lever, now measured): the draft's trunk stages as
   BFP8_B (~1.55 GiB, the bf16 staging dropped) and the wall CLOSES.
   Scope verified: `EnsureBfp8WeightDevice` refuses non-bf16 (the
   target's packed keep-quant weights) by name and falls back to the
   normal staging; the GDN layers stage via `EnsureDevice2D` directly and
   never route through `EnsureMatmulWeightDevice` — so the TARGET's
   numerics are bit-identical and the lossless verify keeps the emitted
   tokens the target's greedy stream; only acceptance/TPOT can move.
3. `VT_DFLASH_DRAFT=1` RESTORED on every drafter-ON leg — the untracked
   `run-w3-finish-gate.sh` had dropped it (the committed `run-gate.sh`
   has it), so the probe legs silently ran the SELECTOR arm and died at
   the selector's f32 sample-row gather (`speculator.cpp:217` →
   `kIndexSelect: only bf16 rows are supported`) instead of the direct
   arm's path. The direct arm downloads the block logits and argmaxes on
   host; the selector never runs.

**Verified past the wall** (`w3f-diag-on-direct.log`: direct arm + bfp8 +
both shares): the draft's full trunk stages, the first propose completes,
the verify runs — and the loop dies at the NEXT named blocker, exactly
the one this README's continuation already owed:

```
vt: no kernel for op GdnSpecDecode (id 12) on device tenstorrent (type 6)
```

**Gates this run:**
- (b) Q4_K_M drafter-OFF, knob defaults: `w3f-off-{1,2,3}` — token-ids
  sha256 `d49c53d23ac5bc4180096b30ebe3fb6990ba6f6eff7d6098760685f9035019ff`
  = BYTE-IDENTICAL to the recorded stream, 3/3 legs; TPOT
  6626.97 / 6627.91 / 6626.89 ms vs the recorded 6607 (0.30%, the same
  band the G4 reproduction recorded: 6624 vs 6607).
- (a) Q4_K_M KV-64 drafter-ON: BLOCKED at the verify by the
  `kGdnSpecDecode` TT port gap (the refusal above, captured in
  `w3f-diag-on-direct.log`). The memory wall is closed and the draft
  proposes; the verify's GDN spec segment has no TT kernel. The port
  follows `GdnDecodeStepComposed` (the W2-validated exactness-scrubbed
  step): initial state from column `num_accepted-1`, T sequential
  composed steps, per-timestep slot snapshots, one split-block scatter.
  It is the next wave's first item and needs the fresh-implementer /
  fresh-reviewer cycle per the row discipline.
- (c) UD-Q6_K_XL KV-8 drafter-ON: blocked by the same gap (the rebase
  onto lo-main is ALREADY in this branch's base — `13542b40a` is an
  ancestor of HEAD and the fork's main equals it; the embed fix
  `47847ad9c` landed before the branch's W1).

## W4 kernel wave (2026-11-15, the fresh-implementer wave: kGdnSpecDecode + the sampler it uncovered)

Head at measurement: the W4 commits on `row/QWEN38-DFLASH2-STAGE-C`
(c6f4042c7 + the GDN spec kernel + the sampler port + the EnsureHostBytes
repair). Issues: the row issue's W4 entry,
`ISSUE-LOCAL-01M7FTQR1HT8B61DVC00Y5AHZ8` (the sampler port gap),
`ISSUE-LOCAL-01M7FSHPRTFND4Q75M5MNBAX3W` (the bf16 state readback, closed),
`ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H` (the divergence bisect, open).

### Shipped

1. **`kGdnSpecDecode` on Tenstorrent** (`tenstorrent_gdn.cpp`,
   `GdnSpecDecodeKernel`) — op id 12, the refusal `w3f-diag-on-direct.stderr.log`
   named. The multi-row GDN decode step the verify's GDN segment issues.
   Composition, per `GdnDecodeStepComposed` (the W2-validated exactness-scrubbed
   step the single-row `kGdnDecode` runs):
   - initial state from column `num_accepted-1` of the request's slot row —
     ONE exact `GatherRowsExact` in the SPLIT shadow geometry (each slot
     expands into its `sf` block rows; the one-hot matmul alternative rounds
     its inputs to tf32, measured 2.4e-4, and the initial state's error
     compounds through the delta-rule correction the wide-range arm pins);
   - T sequential composed steps per request; the per-timestep q/k/v/g/beta
     are CONTIGUOUS row slices of the served streams (request i's tokens are
     rows `[cs_i, cs_i+1)` in batch order — the pure-spec contract), so the
     recurrence needs no per-timestep index build;
   - per-timestep state snapshots accumulated as device rows and landed in
     ONE split-block `ScatterRowsDevice` (last-writer-wins = the oracle's
     sequential store order); a NULL snapshot column skips only its store;
   - bf16 STORAGE semantics at the snapshot boundary (the register chain
     stays f32 — the oracle's `StoreF32` boundary);
   - NULL-init requests zero their out rows and store NOTHING (the oracle's
     :114-115 — no snapshots even from live columns);
   - the captured arm refuses BY NAME (the acceptance metadata has no
     persistent device refresh — the conv spec kernel's named gap); the spec
     verify runs eager (`VLLM_CPP_CUDAGRAPH=0`).
   - Commit-or-restore composition: the kernel commits through the SAME
     `CommitDeviceLogical2D` slot mechanism `kGdnDecode` uses, so the stage-B
     `StageGdnStateShadow`/`RestoreGdnStateShadow` clones cover its advance —
     a rejected verify tail restores the pre-step shadow this kernel's commit
     replaced. The rollback identity is pinned BIT-EXACT (below).
2. **`kGreedyRejectionSample` on Tenstorrent**
   (`ISSUE-LOCAL-01M7FTQR1HT8B61DVC00Y5AHZ8`) — the by-name refusal that
   follows (id 107): the loop got past the GDN spec step and died at the
   sampler. Host-staged (the `kGreedyArgmax` precedent), ported 1:1 from the
   CPU reference — bit-exact vs the CPU oracle by construction.
3. **The bf16 GDN state-cache readback repaired**
   (`ISSUE-LOCAL-01M7FSHPRTFND4Q75M5MNBAX3W`, closed) — `EnsureHostBytes`'
   generic tail lost the width-adaptive bf16 arm to f4725780b; the
   kGdnDecode bf16-state arm's final readback was failing on the BASE
   (pre-existing; no wave changes in its path).

### Focused gate (the P150, mutex + reset per leg)

- RED before (`w4-gdnspec-red.log`): the test case refuses by name —
  `no kernel for op GdnSpecDecode (id 12) on device tenstorrent` — the same
  refusal the engine's verify hit (`w3f-diag-on-direct.stderr.log:1609`).
- GREEN after (`w4-gdnspec-green.log`, 4/4 test cases EXIT 0):
  - captured shape (ns=1, T=9, nat=3, slots at a non-zero offset):
    Hk16/Hv32 out max_abs 6.5e-9 state 3.0e-8; Hk16/Hv48 out 5.6e-9 state
    3.0e-8 (tol 0.02 — the composed-step envelope);
  - rolling continuation, 3 spec steps with VARYING acceptance (nat 1/2/3),
    each step's init reading the previous step's committed snapshot column:
    out <= 9.3e-9, state <= 3.7e-8;
  - mixed seqlens (ragged cu_seqlens, disjoint slot rows): out 7.5e-9, state
    2.2e-8;
  - NULL-init: out 4.7e-9, state 1.5e-8, the NULL request's out rows
    explicitly ZERO, its live snapshot columns NOT stored;
  - bf16-state chain (2 steps, the emulated oracle: the CPU step on the
    bf16-rounded state + the storage rounding at the commit): out 9.3e-9,
    state 1.2e-7 / 6.1e-5 (the bf16 storage band);
  - **THE ROLLBACK IDENTITY, bit-exact** — the spec kernel's snapshot at
    column j == the native `kGdnDecode` chain's state after j+1 tokens (same
    device, same composed step, zero tolerance): out 0/9216, state
    0/1179648 bit mismatches. This is the property the stage-B
    commit-or-restore rides: a rejected verify tail rolls the state back to
    a slot a later step can resume from.
  - the sampler: full-accept / reject-at-depth-2 / two-requests, all three
    BIT-EXACT vs the CPU oracle.
- The kGdnDecode + kCausalConv1dSpecUpdate cases rerun in the same log:
  green (the EnsureHostBytes repair un-breaks the kGdnDecode bf16-state arm:
  cache bit mismatches 0/98304).

### Device gates

- **(a) Q4_K_M KV-64 drafter-ON: STOP at the token-identity gate.** The loop
  RUNS END-TO-END for the first time on this row (`w4-on-probe`: exit 0, 24
  tokens, the acceptance telemetry live — 24/32 proposed accepted) and the
  emitted stream is `[220, 0, 0, ...]` vs the recorded greedy stream
  `[220, 16, 23066, ...]` (d49c53d2…). THE SPEC'S STOP CONDITION. The probe
  evidence locates the degeneracy UPSTREAM of the GDN spec kernels
  (`w4-diag-spec.stderr.log`, `VT_TT_GDN_SPEC_DEBUG`):
  - the conv-spec kernel's x stream — the spec tokens' qkvz projection — has
    NaN at the FIRST call (step 0, GDN layer 0: `x_asum=nan`) and collapses
    to a frozen ~43.3553 sum at GDN layer 8+ in EVERY step (steps 0-3),
    while the state-window reads stay real (T_asum ~6-15k);
  - the GDN spec kernel's gathered state is real (s_asum 0.6-9.5k) and its
    received activations are EXACTLY zero (q_asum=0, v_asum=0) for the
    collapsed layers — the zero outs are a downstream consequence;
  - the acceptance telemetry plus the all-zero ids fit BOTH argmaxes
    degenerating to token 0 (the NaN-argmax signature: NaN comparisons false
    -> best stays 0) — the verify's block logits AND the draft's block
    logits are NaN-poisoned;
  - the GDN spec kernel and the sampler are exonerated: their focused gates
    are bit-exact, the metadata (nat0/init0/slots) is sane every call, and
    the first NaN precedes any GdnSpecDecode call.
  `ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H` owns the bisect (the embed of the
  draft's proposals -> the qkvz projection GEMM -> the conv-spec x; the bfp8
  weight-staging lever is the prime suspect). The no-bfp8 A/B is
  capacity-blocked on this base: the draft's bf16 trunk OOMs at the same
  178257920 B ffn_down allocation even at KV 32 (`w4-nobfp8-probe.*`).
  **The acceptance multiple is UNMEASURABLE until the divergence lands** —
  a degenerate loop's TPOT (the probe's 9970 ms at 24 tokens) is not a
  number, and recording it as the multiple would be dishonest.
- **(b) UD-Q6_K_XL KV-8 drafter-ON: BLOCKED by the same divergence** (the
  same verify path; running it would measure the same degenerate loop).
- **(c) drafter-OFF byte-identity, both arms: PASS, 6/6 legs** — the knob
  default unchanged by this wave's changes:
  - Q4_K_M KV-64 captured, 3 reps: token-ids sha256
    `d49c53d23ac5bc4180096b30ebe3fb6990ba6f6eff7d6098760685f9035019ff` =
    BYTE-IDENTICAL to the recorded stream 3/3 (`w4-off-{1,2,3}`); TPOT
    6628.03 / 6626.55 / 6627.64 ms vs the recorded 6607 (0.32%, the same
    band the W3 rerun recorded: 6626.89-6627.91).
  - UD-Q6_K_XL KV-8 captured, 3 reps: token-ids sha256
    `b36bc61a59828c1ddf05fbabf62ca0a491c652d50f7ec06bee86d4b17ad18bcc` =
    BYTE-IDENTICAL to the recorded stream 3/3 (`w4-q6kl-off-{1,2,3}`, 64
    tokens — the 8-block pool caps max_model_len at 96); TPOT 5325.37 /
    5326.25 / 5325.98 ms vs the recorded eager 5341.31 / the embed row's
    captured band 5318.76-5329.15.

### Recipe

`run-w4-gate.sh` (this dir): the committed run-w3-finish-gate.sh's discipline
(file mutex, luwen reset + 15 s + /dev/shm cleanup per leg, idle gate,
taskset 32-63 bench / 64-127 builds, run FROM THE REPO ROOT, INT8DOT=1) with
`VT_DFLASH_DRAFT=1` restored on the drafter-ON legs and
`VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8` per the W3 disposition.

## W6-W13 bisect (2026-11-15 evening — the verify's stream death localized)

All legs Q4_K_M KV64, seed 0, degenerate stream `daa113b5` unless noted.
Instrumentation: `VT_TT_GDN_SPEC_DEBUG` probes (serve-time readbacks, slot
states, per-layer asum/nan/inf/zero) — pure diagnostics, uncommitted.

### The stream's shape
- The good drafter-ON reference `d49c53d2` = the DRAFTER-OFF stream
  (`w3f-off-{1,2,3}`): spec-decode token identity means the verify must
  reproduce the greedy stream byte-for-byte.
- Degenerate = `[220, 0, 0, ...]`: token #0 (from the clean prefill) correct;
  every verified token after = 0 — the verify's logits collapse to argmax-0.
- The poison is live in the FIRST spec step. Its T=9 geometry (9 rows, not a
  tile multiple) is unique to the drafter-ON path: prefill T=32 = exactly one
  tile; decode T=1 pads to one tile.

### The layer traces (per staging arm)
- w7 (default device-residual norms): layers 0-9 clean; l=10+ hidden=0,
  res=garbage (9.46e41) forever.
- w8/w9 (`VT_TT_FORCE_HOST_RESIDUAL=1`): layers 0-2 clean; l=3+ (the first
  full-attn layer) res=inf, hidden=0. Mixer outputs: FA layers read
  clean-looking, GDN block outputs all-zero from l=4.
- w12 (`VT_TT_HOST_FREE_DECODE=0`): onset at l=1. **The onset layer moves with
  the staging arm — the poison is pool-state dependent, not op-fixed.**

### The op-level exonerations (standalone tests, all GREEN at engine geometry)
- `kCausalConv1dSpecUpdate` sweep F (C=10240, widened state_len=11, T=9,
  slots=9, nat=1, NaN-seeded out): max_abs 2.98e-08, nonfinite=0.
- Sweep G (shifted-address second call — call 1's buffers held alive so call
  2 lands at different addresses): max_abs 2.98e-08. The stale-address
  replay hypothesis is dead for this op.
- New stride test (row-strided bf16 view upload): EnsureDevice2D REFUSES
  row-strided views outright (residency.cpp:930) — the engine never uploads
  through it.
- `GdnSpecDecodeKernel` checksums: clean at every T=9 call (nat0=1, no NaN).
- The gated norm's operand probes (`[RNG-IN]`): x and g clean through the
  spec step's first 8 GDN layers.
- `VT_TT_ALLOC_TRACE` slot-census alias scan (w10): ZERO duplicate device
  addresses across all slots — no double-hand.
- w13 `[DRAFT_LOGITS]`: propose #1 CLEAN (asum 4.94e6, 0 NaN) — the drafter
  is a VICTIM, not the poisoner; proposes #2-3 read 100% NaN (the verify's
  poisoned aux taps). The poisoner is the first verify's own forward.

### The z-gate collapse signature (w11 `[SPEC-OUT]` / `[RNG-IN]`)
- g#55 (the spec step's 8th GDN layer, l=9): the z gate reads asum 3.62e40
  with 109 Infs, from CLEAN h (22227). From g#56 (l=10): the gate reads
  ALL-ZEROS for the rest of the run.
- The z = the merged/split qkvz GEMM's gate output; h clean + z inf/garbage
  pins the poison to the qkvz GEMM's output commit or the serve that reads it.
- `[QKVZ-OUT]` mixed reads: call 50 (l=2) clean 62540; call 51+ (l=4+)
  8.67e40-9.68e40 garbage — the qkvz GEMM's committed output reads garbage
  while its input h reads clean, in BOTH staging arms.

### Where this leaves the bisect
Every op is GREEN in isolation at the engine geometry (47/47 assertions);
the engine dies with a moving onset. The probes cannot distinguish a stale
clean read from a fresh write (the codebase's own dual-read control exists
for exactly this: `TrustDump`'s two-pass verified read, and the
`post_input_norm_DIRECT` control at qwen3_5.cpp:8457). The next instrument
is the trusted dual-read applied at the spec step's qkvz commit + mixer
output — it splits "the download patterns diverge" from "a kernel
difference" before any fix is written. The `[CONV-SPEC-STAGE]` fill-stage
garbage (every 3rd fill, identical bytes) remains the strongest single
anomaly: the cached zeros base reads ZEROS at fill time while the fill's
output reads recycled garbage — but the standalone fill test at the same
geometry is GREEN, so the engine's allocator/cache state participates.

### Runners (this dir)
`run-w6-pinpoint.sh` (w7 base), `run-w8-hostresid.sh`, `run-w9-mixer.sh`,
`run-w10-alias.sh` (+VT_TT_ALLOC_TRACE), `run-w11-spectok.sh`,
`run-w12-hostfree-off.sh`, `run-w13-draftlog.sh`. Each = the W4 discipline
(file mutex, luwen reset + 15 s + /dev/shm cleanup, idle gate, taskset
32-63 bench / 64-127 builds, FROM THE REPO ROOT, INT8DOT=1,
VLLM_CPP_CUDAGRAPH=0, VT_TT_WEIGHT_RESIDENCY=bfp8, VT_DFLASH_DRAFT=1).

## W20 the decisive instrument (2026-11-16 — the trusted dual-read: the alias theory FALSIFIED)

All legs Q4_K_M KV64, seed 0, degenerate `daa113b5` (the w14-w19 TrustDump
runs — 826/962/1154 dumps — are the same verdicts this section's reads
cross-check; the qkvz commit-side dumps they intended NEVER FIRED: the
dead agent's TrustDump calls sat AFTER `return out;` in BOTH ProjectGdnQkvzRaw
arms — unreachable dead code, fixed this wave).

### The verdict

**The task's working hypothesis ("the COMMITTED bytes are correct at commit
time but the SERVE reads different bytes at the same address") is
FALSIFIED.** The read-only trusted dual-read (`DebugSlotTrustRead`, new —
two independent to_vector passes, NO slot mutation, so the consumer's serve
path stays pristine) at the qkvz COMMMIT side reads the poison:

```
[TRUST-SLOT] qkvz-z-c54 agree=1 asum=36816.6  addr=4f4c6f00   clean
[TRUST-SLOT] qkvz-z-c55 agree=1 asum=3.75443e+40 inf=113 addr=4f4c6f00   GARBAGE
[TRUST-SLOT] qkvz-z-c56 agree=1 asum=0 zero=55296 addr=4f4c6f00   all zeros
```

The SAME garbage (asum 3.75443e+40, 113 Infs) the serve-side reads
([GSERVE-TRUST] at the gated norm, [RNG-IN]) — commit and serve AGREE. The
serve is exonerated; the commit is the convict.

### The commit is garbage PRE-commit, with verified-clean operands

`[ZGEMM-PRECOMMIT]` (new — the trusted dual-read of the matmul's own output
INSIDE MatmulBTKernel, before any commit machinery): c55's output reads
asum 3.75443e+40 inf=113 — Identical to the commit-side and serve-side
reads. The operands, dual-read at the same site ([MMBT9]): the activation h
(a_asum 22227.6 — matches [QKVZ-IN]) and the staged weight (w_asum 396486 —
byte-identical at the same position of every spec step, same device
address) — BOTH CLEAN, BOTH STABLE. The matmul computed garbage from clean
operands; c56's all-zero input produced a correct all-zero output through
the same program — the compute works; the failure is state-dependent.

### The alias/block-recycle theory: falsified by intervention

- The commit ledger ([TT-COMMIT], new): EVERY commit replaces an
  already-freed old shadow (old_live=0) — the output blocks cycle through
  the allocator's free list call to call (the z slot: 4f4c6f00, 4ca6af00; a
  legal-looking recycle).
- THE INTERVENTION: a 3-deep output keep-alive in MatmulBTKernel
  (VT_TT_GEMM_OUT_KEEPALIVE=0 rolls back) — the current call's output block
  can never be a recent recycle of its own geometry's blocks. **The poison
  MOVED (clean through the first 12 spec-step z GEMMs, then garbage with a
  BIGGER signature: asum ~9e42, ~2500 Infs) instead of disappearing — the
  recycled-block content is NOT the mechanism.** (The keep-alive stays in
  the tree: it is harmless, it moved the onset later, and the first emitted
  token changed 220→23066 — the recorded stream's third token — but it is
  NOT the fix and the stream stays degenerate: e171d6b1.)
- The garbage's bit signature (asum 1e38-1e42, Infs, no NaNs until late) is
  consistent with BFP8/bf16 bit-pattern reinterpretation, but no staging
  lifecycle reproduces it (below).

### Isolated reproductions: ALL GREEN (the poison needs the engine's op mix)

Three new focused cases in tests/vt/test_tenstorrent_backend.cpp, all at the
engine's exact z geometry (M=9 partial tile, K=5120, N=6144, bf16
activation/weight/out, the weight staged through the REAL EnsureBfp8WeightDevice
cold path via the new Bfp8StageForTest surface):

1. `w20-bfp8-m9.log` — the bfp8-staged weight at M=9 AND M=32 vs the f32
   CPU oracle: max_rel 0.75 percent, 0 non-finite. **The W3 scope claim
   ("the GDN layers stage via EnsureDevice2D directly and never route
   through EnsureMatmulWeightDevice") is WRONG — MatmulBTKernel stages EVERY
   weight, so VT_TT_WEIGHT_RESIDENCY=bfp8 has been degrading the TARGET's
   bf16 GDN in_proj weights since W3 — but the bfp8 matmul itself is
   numerically FINE at the spec geometry (the degradation is the expected
   BFP8 band, not the poison).**
2. `w20-bfp8-warm.log` — 200 warm repeated calls (program-cache hits + the
   allocator's per-call output Alloc/Free cycle): 0/200 bad.
3. `w20-bfp8-unsynced.log` — the engine's staging lifecycle (a ~33 MB
   bf16_dev temporary created, typedcast on device ASYNC, dropped with NO
   sync, then the matmul) x 50 fresh weights: 0/50 bad.

### Where this leaves the bisect

**SHARPENED (the w20 log's [SPEC-OUT] core series, read post-run): the
recurrence's core output is ~ZERO at the spec calls** — asum 120 (the first
call) then 1.2 / 1.18 / 0.9 / ... with EXACTLY 24576 zeros (= 4 of 9 rows)
per call, later NaN — while the recurrence's q/k/v inputs (the pc_q/pc_v
trusted dumps) are CLEAN at every pre-collapse spec call (no zero rows in
any [9x16] dump before seq 795; the all-zero dumps start at the post-
collapse steps). A ~zero core from clean q/k/v points at the recurrence's
UNPROBED inputs: the INITIAL-STATE gather (column num_accepted-1 of the
slot row — the prefill's committed state) and the BA outputs (araw/braw →
g/beta; the sba_* dumps only cover the prefill). The core ~0 explains the
stream's degeneracy on its own (the GDN blocks contribute noise-direction
only → the verify's logits diverge from token 1), and the z GEMM's 1e40
read at c55 is a SECOND anomaly on top (it is what collapses h to zero at
l=10). The W4 focused gate tested nat=1/2/3 and the initial-state column
read — but with TEST-built inputs; the ENGINE's state-cache content at the
gather is unprobed.

The poison lives in the interaction the isolated harness cannot see: the
engine's INTERLEAVED op mix — the conv-spec fill's `indexed_fill` (whose
cached-zeros-base/clone anomaly stands: the fill's output reads recycled
bytes while the base reads zeros, and the fill's output buffer is the
3.3 MB allocation whose freed block the 110 KB z output splits into), the
GDN recurrence's gathers/scatters, and the paged attention — all sharing
the tt-metal program cache and the DRAM allocator with the matmuls. The
strongest remaining hypothesis: a tt-metal-pin program-cache/runtime-arg
staleness under interleaved programs (the kernel writing a PREVIOUS call's
buffer while the current output reads uninitialized recycled bytes — which
explains the keep-alive intervention moving the poison, the identical-bytes
periodicity, and the staging-arm-dependent onset). The next instrument is
the interleaved-op focused case: matmul + indexed_fill + gather through the
SHARED program cache in one harness — scoped, well-defined, and the first
reproduction attempt that varies the op MIX rather than the op.

### Runners/tests this wave

`run-w20-trust2.sh` (the W6 discipline + VT_DUMP_TRUST + VT_TT_ALLOC_TRACE),
the w20-trust2/ dump dir, the three focused logs above. New instrumentation
(all env-gated, default-off): DebugSlotTrustRead, [XSERVE-TRUST],
[GSERVE-TRUST], [MMBT9], [ZGEMM-PRECOMMIT], [TT-COMMIT], [I8DOT-OUT], the
dead-code TrustDump placements fixed, the 3-deep GEMM output keep-alive.
