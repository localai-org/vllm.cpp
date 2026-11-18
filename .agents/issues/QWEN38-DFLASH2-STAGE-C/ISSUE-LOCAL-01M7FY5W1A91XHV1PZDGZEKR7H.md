ID: ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H
Title: The drafter-ON verify forward degenerates upstream of the GDN spec kernels: the conv-spec x stream collapses at GDN layer 8 with NaNs — the emitted stream is [220, 0, 0, ...] (the W4 token-identity STOP)
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-16
Closed: -

## Problem

With kGdnSpecDecode (id 12) and kGreedyRejectionSample (id 107) ported, the drafter-ON leg runs to completion and the emitted stream degenerates. W21 (2026-11-16): two mechanisms found and fixed in-engine — (1) ttnn::indexed_fill miswrites multi-replacement scatters (the k-th replacement row lands at id[k]+k; survives a program-cache bust) — the conv-spec x scatter replaced with a bit-exact broadcast-mask multiply; (2) ttnn::copy into the resident state shadow does not land — the spec state commit installs the committed tensor directly (eager-only path). Validated: the conv output's zero rows gone, the state persists across verify steps (the gathered state == the expected snapshot byte-exact). The recurrence, the gather, and the BA outputs are proven faithful (CPU reference max_abs 4.5e-07). The stream is still degenerate [220,0,0,...]: the remaining poison is the GDN block output garbage at l>=1 (mixer asum 2e40) with clean gated-norm operands, plus the z GEMM 1e40 at c54+ — the out-of-place op WRITE path misaddresses under the engine's allocator/program-cache state. Next: the shared-cache harness extended with the norm multiply + the recycle pattern.

## Resolution

-

## W6-W13 bisect (2026-11-15 evening — the follow-on session's record)

The W4 "collapses at GDN layer 8" framing is WRONG in one respect and right in
a sharper one: the collapse is NOT op-fixed — its onset layer MOVES with the
staging arm (w7 device-res: l=10; w8/w9 forced-host-residual: l=4, the first
full-attn layer; w12 host-free-off: l=1). The poison is POOL-STATE dependent.
What stays true: the poison is live in the FIRST spec step (T=9 — the only
non-tile-multiple geometry in the workload), and the verify's logits collapse
to argmax-0 (the all-zero accepted stream).

New evidence (all in docs/bench-evidence/qwen38-dflash2-stage-c-w3-20261115/,
README.md's W6-W13 section carries the full record):
- `[DRAFT_LOGITS]` (w13): propose #1 CLEAN (asum 4.94e6, 0 NaN); proposes #2-3
  read 100% NaN. The drafter is a VICTIM (it reads the verify's poisoned aux
  taps), not the poisoner.
- `[SPEC-OUT]`/`[RNG-IN]` (w11): the z gate reads asum 3.62e40 + 109 Infs at
  the spec step's 8th GDN layer (l=9) from CLEAN h (22227); from l=10 the gate
  reads ALL-ZEROS for the rest of the run. `[QKVZ-OUT]` mixed reads garbage
  (8.67e40+) from call 51 (l=4) while `[QKVZ-IN]` h reads clean — in BOTH
  staging arms. The poison sits between the qkvz GEMM's clean input and its
  garbage-read output.
- `[CONV-SPEC-STAGE]` (w7b): the conv-spec fill's cached zeros base reads
  ZEROS at fill time while the fill's output reads recycled garbage (identical
  bytes asum=1.34712e40 + 36 NaNs at every first-GDN-after-FA call). tt-metal
  indexed_fill = out-of-place clone + row writes (indexed_fill.hpp:11-15), so
  the output garbage is NOT the base's content.
- `[TT-SLOT-ALIAS]` census scan (w10): ZERO duplicate device addresses across
  all slots — no double-hand.

Op-level exonerations (standalone tests, GREEN at the engine geometry,
47/47 assertions): `kCausalConv1dSpecUpdate` sweep F (C=10240, widened
state_len=11, T=9, slots=9, nat=1, NaN-seeded out: max_abs 2.98e-08,
nonfinite=0) and sweep G (shifted-address second call: the call-1 buffers
held alive so call 2 lands elsewhere — max_abs 2.98e-08). The stale-address
replay hypothesis is dead for this op. `GdnSpecDecodeKernel` checksums clean
at every T=9 call. EnsureDevice2D REFUSES row-strided views outright
(residency.cpp:930) — a new test pins that contract.

## W20 (2026-11-16): the trusted dual-read — the alias theory FALSIFIED

The decisive instrument ran (README.md's W20 section carries the full
record). Verdicts:

- **The stale-download/serve-alias hypothesis is FALSIFIED.** The read-only
  trusted dual-read at the qkvz COMMIT side reads the same garbage the
  serve reads (asum 3.75443e+40, 113 Infs at the first spec step's l=9,
  call 55). The matmul's own output reads garbage PRE-commit
  ([ZGEMM-PRECOMMIT]) with BOTH operands dual-read clean and stable at the
  same site ([MMBT9]: h asum 22227.6, the staged weight byte-identical
  across steps).
- **The block-recycle alias is falsified by intervention.** A 3-deep output
  keep-alive in MatmulBTKernel (the current output's block can never be a
  recent recycle) MOVED the poison (onset later, bigger: ~9e42, ~2500 Infs)
  instead of removing it. The recycled-block content is not the mechanism.
  The keep-alive stays (harmless; onset later; first token 220→23066) but
  the stream stays degenerate (e171d6b1).
- **Three isolated reproductions at the exact engine geometry are ALL
  GREEN** (tests/vt/test_tenstorrent_backend.cpp, w20-bfp8-{m9,warm,unsynced}
  .log): the bfp8-staged weight at M=9/M=32 vs the f32 oracle (max_rel
  0.75 percent), 200 warm program-cache-hit calls, 50 unsynced-staging
  lifecycle reps. The W3 scope claim that the target's GDN weights never
  route through EnsureMatmulWeightDevice is WRONG (they do; bfp8 degrades
  them since W3) but the bfp8 matmul itself is numerically fine at the spec
  geometry.
- The dead agent's qkvz commit-side TrustDump calls were unreachable dead
  code (after `return out;` in both ProjectGdnQkvzRaw arms) — why waves
  w14-w19 captured no qkvz commit. Fixed.

## Next instrument

**SHARPENED: the recurrence's core output is ~ZERO at the spec calls**
(asum 120 → 1.2 → 0.9, exactly 4 of 9 rows zero, later NaN) with CLEAN
q/k/v (the pc_q/pc_v trusted dumps have no zero rows pre-collapse). The
recurrence's UNPROBED inputs are the next probes: the INITIAL-STATE gather
(column num_accepted-1 — the prefill's committed state as the ENGINE holds
it) and the BA outputs (araw/braw → g/beta; the sba_* dumps cover only the
prefill). The core ~0 explains the degenerate stream on its own; the z
GEMM's 1e40 read at c55 is a second anomaly (it collapses h at l=10).

Beyond that, the poison needs the engine's INTERLEAVED op mix: the conv-spec fill's
indexed_fill (the cached-zeros-base/clone anomaly stands), the GDN
recurrence's gathers/scatters, and the paged attention, all sharing the
tt-metal program cache and the DRAM allocator with the matmuls. Strongest
hypothesis: a tt-metal-pin program-cache/runtime-arg staleness under
interleaved programs (the kernel writing a previous call's buffer while the
current output reads uninitialized recycled bytes — fits the keep-alive
intervention moving the poison, the identical-bytes periodicity, and the
staging-arm-dependent onset). Next step: the interleaved-op focused case
(matmul + indexed_fill + gather through the SHARED program cache in one
harness) — the first reproduction that varies the op MIX, not the op — plus
the state-gather/BA trusted reads at the spec calls.

## W21 (2026-11-16): the probes ran — two real defects found, fixed, and the write-path class named

The raw-f32 dump instrumentation (VT_TT_SPEC_DUMP, this wave; the w21-specdump
/ w21-mask / w21-bust / w21-fix / w21-fix2 / w21-snap / w21-iscat dump dirs)
answered both probes and found the poison's first two CONCRETE mechanisms.

**PROBE 2 (the BA outputs): CLEAN.** At the first spec call (l=0), the
committed g/beta match the composed CPU math
(-exp(a_log)·softplus(a+dt_bias), sigmoid(b)) to max_abs 3.0e-07 / 1.3e-07
over all 432 (token, head) pairs; g ∈ [-8.31, -8.7e-07], beta ∈
[0.028, 0.998]. The a/b raws, the projections, and the GdnPostConv gate chain
are exonerated.

**PROBE 1 (the initial-state gather): the GATHER IS FAITHFUL — the STATE
PERSISTENCE was the defect.** The gathered S_all is byte-identical to a direct
shadow read of the same slot (gs1_S_all == gs1_slot0_direct), and the
recurrence run on the dumped inputs reproduces the TT output to max_abs
4.5e-07 (the CPU composed reference, threaded per head). But the state the
gather reads is NOT the state the previous step's recurrence produced. The
VERIFIED pairings (the W22 record correction — an earlier draft of this
paragraph paired the expected snapshot with the l=1 step-1 gather, the wrong
layer's number):
- the step-1 l=0 gather = 11984.7 (w21-specdump.stderr.log:596, gs0_S_all) —
  the l=0 PREFILL state, correctly served at step 1;
- the step-1 l=1 gather = 301.568 (w21-specdump.stderr.log:616, gs1_S_all) —
  the l=1 layer's TRUE small prefill state (the model's own small state at
  layers >= 1), not a defect;
- the step-2 l=0 gather PRE-fix = 10946.6 (w21-specdump.stderr.log:1754) —
  neither the init 11984.7 nor the expected snapshot: the copy's half-landed
  residue;
- the expected post-token-0 snapshot = 11550.9 (the recurrence run on the
  dumped inputs), and the POST-fix step-2 l=0 gather reads exactly it
  (w21-fix.stderr.log:1530).
And the cache shadow read DIRECTLY after the commit copy equals its PRE-copy
content (w21-snap/gs0_cache_post.f32 asum 11984.7 = the init state, vs the
scatter's own source rows gs0_snap_next.f32 asum 102778; gs1 the same shape
301.568 vs 2603.31). **ttnn::copy into the resident shadow does not land in
the engine.**

**THE CONV-SPEC FILL'S MISWRITE (the W7 anomaly, now resolved to a law):** the
dumped fill output shows the k-th replacement row landing at id[k]+k — ids
[0,9,18,27,36,45,54,63,72] wrote x[0..8] at blocks
[0,10,20,30,40,50,60,70,72]. The displaced writes zero out the conv outputs at
tokens 4-7 (the whole conv window reads the never-written base) — the exact
"4 of 9 rows zero" the [SPEC-OUT] core series showed, at EVERY layer, from the
FIRST spec call. The served x (the qkvz mixed) is clean at the same calls
(commit-side trust read zero=0; the serve probe agrees). **ttnn::indexed_fill
miswrites multi-replacement scatters in the engine, and the miswrite SURVIVES
a full program-cache bust** (the w21-bust arm: a fresh zeros base + a fresh
ids tensor with a unique shape per call — a guaranteed program miss — same
displacement). Single-replacement fills (the decode conv, every isolated
test) are fine; the strided-id multi-replacement form breaks.

**The two fixes (this wave, both validated bit-exact):**
1. The conv-spec x scatter is a broadcast-mask multiply:
   xT[t, s·C+c] = x[t, c] · (s == slot(t)) — exact f32 elementwise, the same
   traffic class as the fill, no indexed_fill. Validated: the conv output's
   zero rows are GONE (the w21-mask dumps: the slot-0 blocks == x for all 9
   tokens; the out zero rows = []), and the first spec call's core went from
   asum 120 with 24576 zeros to 305 with 0 zeros.
2. The spec state commit installs the committed tensor as the slot's shadow
   directly (CommitDeviceLogical2D replaces s->device) instead of
   ttnn::copy into the resident shadow — legal because the spec kernel is
   eager-only (the capture refusal at its head). Validated: the step-2 l=0
   state now persists across verify steps (the post-fix gather = 11550.9 =
   the EXACT expected post-token-0 snapshot, w21-fix.stderr.log:1530; the
   pre-fix gather read 10946.6, w21-specdump.stderr.log:1754; steps 3-4 =
   11603 / 11464.8, evolving).

Also corrected: the W20 record's "no zero rows in any [9x16] dump before seq
795" is WRONG — the spec pc_q/pc_v/spec_conv dumps carry exact-zero rows at
tokens 5,6,7 in 192/192 dumps from the first spec call (they were the conv
fill's displaced writes, downstream). And the small ssm states at layers >= 1
(asum ~300-460 vs 11984.7 at l=0) are the model's TRUE behavior — the
drafter-OFF legs' byte-identical stream proves the prefill states, so the
small spec cores at l>=1 (asum 3.9, the true math on the true state) are not
themselves the degeneracy.

**Where the hunt stands:** the stream is STILL degenerate [220, 0, 0, ...]
(the w21-fix sha daa113b5 — the original W4 signature, not the keep-alive
variant). The remaining poison shows as the GDN block's output garbage at
l >= 1 (the [MIXER-OUT] l=1 asum 2.00072e+40, no NaN/Inf/zeros — the
bit-reinterpretation signature) with CLEAN gated-norm operands ([RNG-IN] x =
the core's bytes, g = the z's bytes, both clean), and as the z GEMM's 1e40
commit at c54+ (the W20 record). The gated norm's output and the out_proj are
the next unprobed stage. The class is now NAMED: **the out-of-place op WRITE
path misaddresses under the engine's allocator/program-cache state** — the
fill (displaced), the copy (no-land), and the mixer garbage are three
symptoms. The next instrument is the shared-cache harness (matmul +
indexed_fill + gather through ONE program cache in one process, the W20
disposition's named case) extended with the norm's multiply and the
engine's alloc/free recycle pattern.
**W21 gates + the harness (2026-11-16, same wave):** the drafter-OFF
byte-identity gate holds at 6/6 (Q4_K_M d49c53d2 ×3 at TPOT 6626.89-6627.43
vs the recorded 6607; UD-Q6_K_XL b36bc61a ×3 at 5325.97-5331.21) — the two
fixes are inert off the spec path. The drafter-ON token identity STILL FAILS
(3/3 degenerate; the degenerate stream's exact bytes follow the
instrumentation's timing — e171d6b1 probe-free vs daa113b5 probed — the
degeneracy is the stable fact), so **the acceptance multiple remains
UNMEASURABLE**. The UD-Q6_K_XL drafter-ON leg stays capacity-blocked (the KV-8
pool + the draft trunk; the standing W3/W4 blocker). The W20 disposition's
shared-cache harness RAN and is GREEN (40 reps of matmul + multi-replacement
indexed_fill + gather through one shared program cache with the recycle
churn, 0 bad) — the engine-only nature of the poison stands; the harness
needs the richer engine state (the bfp8 staging lifecycle, the embed, the
paged attention, the DBuf pool pattern) and the gated norm's output + the
out_proj are the next unprobed engine stages.

## W22 (2026-11-16): PROBE 3/4 clean — the poison's entry is the mixed qkvz GEMM's COMPUTE (call 54, l=8); the harness reproduces the class in isolation

The four W21-review repairs landed first (the dead same_spec scaffolding
collapsed; the probe-1 record's layer pairing corrected; the GDN-suite
failing-NAMES baseline committed — the 15-vs-16 discrepancy resolves as the
RAC capture-safety case FLOATING with the box's device state, the stable
set is 15; the ScatterRowsExact strided-form refusal + its red/green test).
The env-doc gate repair (VT_TT_SPEC_DUMP / VT_TT_CONV_SPEC_FILL_MASK
documented; FRESH/BUST allowlisted) rode the same wave.

**PROBE 3 (the gated norm's own write): CLEAN.** [NORM-PRECOMMIT] ==
[TRUST-SLOT] norm-commit-N at the spec geometry (rows=432, d=128), both
calls, dual-read stable, zero=0: the norm's committed shadow IS the norm's
output (284.18 / 246.648).

**PROBE 4 (the out_proj's own write): CLEAN.** [OUTPROJ-PRECOMMIT] ==
[TRUST-SLOT] outproj-commit-N at M=9, K=6144, N=5120 (775100 / 445293,
agree=1). The norm's and the out_proj's writes LAND — the two unprobed
writes between the clean operands and the mixer output are exonerated.

**The poison's entry, localized:** the first garbage in the run is
[QKVZ-OUT] call=54 mixed_asum=inf (w22-probe.stderr.log:1022) — the MIXED
qkvz GEMM (N=10240) at the first spec step's l=8. [MIXED-PRECOMMIT] (new)
reads the SAME garbage pre-commit (asum 3.46266e+40, inf=108, out_addr
b754800 = the [TT-COMMIT]'s address = the [XSERVE-TRUST]'s serve/owner
address) with the [MMBT9] operands dual-read verified clean at the same
site, while the SPLIT z GEMM on the SAME h (N=6144, the same call) is
clean (asum 21911.1). The WRITE path is exonerated (commit == pre-commit ==
serve); **the COMPUTE is the convict** — the W20 z-GEMM class (garbage from
clean operands) on the mixed arm. The onset moved l=1 → l=8 across the W21
fixes (l=0..7 are clean end to end in this run); the FA layers downstream
of the collapse stay correct ([MIXER-OUT] l=11/15 clean). B3's write-path
fix + the acceptance gates therefore do NOT run: there is no write path to
fix — the compute class is a tt-metal-pin program-cache/runtime-args
investigation, and the acceptance multiple remains UNMEASURABLE (the
stream still degenerate, e171d6b1).

**B4 — the harness grew and REPRODUCES the class in isolation** (the new
case after the W21 harness; `w22-harness.log`, `w22-disc.log`): the GEMM
trio on ONE h (mixed/z/outproj, bfp8-staged weights), the embed
(kEmbeddingQuant Q8_0), RAC+PA at the engine's KV geometry (D=128, block
32, the seq advancing), and the DBuf pool's persistent-buffer hold — the
trio/embed/fill/gather/hold are 0-bad in 40 reps, but the PA ingredient is
RED, and stripping the mix entirely keeps it RED: **ONE raw RAC write at
slot 0 then ONE raw PA decode gives deterministic garbage at seq=2
(4.86e+33), seq=5 and seq=7 (2.99717e+32 — the IDENTICAL bytes at both),
while seq=1,3,4,6,8 are clean**; the Copy-seeded arm (the host read path)
is clean, seq=64 is clean, and the RAC's writes land in all 40 reps
(rac_bad=0). This is the FIRST isolated reproduction of the compute-class
misaddressing — at an out-of-engine form (the engine's PA windows are never
that small): a real PA device-SDPA defect at seq=2/5/7, the next row's
lever, not this poison's mechanism.
