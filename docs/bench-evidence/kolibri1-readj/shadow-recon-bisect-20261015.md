# kolibri1 P150 host-free shadow-reconciliation lever — bisect outcome (2026-10-15)

Worktree `row/tt-kolibri-shadow-recon` (base: origin/main 50d425bec, includes the
#3452 memo-eviction + shadow-sweep fixes). Gate: `test_kolibri1_tt_b2ii
--test-case='*device token gate*'`, `VT_KOLIBRI1_TT_STAGE_DUMP=1`, model
`/mnt/models/Aleph-Alpha/Kolibri-1`, clean `tt-smi -r 0` + cache clear per leg.
TF HARD counts from the near-tie instrument line (parse_adjudication.py).

## The arm delta reproduces exactly on the current base

| arm | gate HARD | argmax chain | TF instrument |
|---|---|---|---|
| OFF (`VT_TT_HOST_FREE_DECODE=0`) | 2 | 12/20, flips 8 (6 near, 2 hard) | 44 tf flips (27 near, **17 HARD**), worst 8.65 nats |
| ON (host-free default) | 5 | 3/11, flips 8 (3 near, 5 hard) | 69 tf flips (29 near, **40 HARD**), worst 10.62 nats |

40/17 — identical to the #3454 readjudication. The #3452 UAF fixes did not
close the delta (expected: they fixed crashes, not drift).

## Bisect matrix (forced-host knobs, ON arm)

| leg | gate HARD | outcome |
|---|---|---|
| ON + `VT_TT_FORCE_HOST_ROPE=1` | 5 | flip set byte-identical to ON — rope ruled out |
| ON + `VT_TT_FORCE_HOST_RESIDUAL=1` | 7 | WORSE; host round-trips desync the device shadow (the recorded 1e37 collapse class, tenstorrent_ops.cpp:466-478) — NOT a clean probe, and it falsifies "host residual is the accurate path" |
| ON + both | 7 | identical to +fres |
| ON + `VT_TT_FORCE_HOST_PA=1` | 5 | chain identical (3/11, 5 hard) — PA ruled out |

## Mechanism (stage-trace localization, forward 0 layer 0)

The first diverging stage in the entire run is `L0 dhn` (the input_layernorm),
on BIT-IDENTICAL inputs: `L-2 emb` hash `dd76406a` and `gam` sum 858.502781 in
both arms. OFF (host f32 loop, `rmsnorm-host call=1`) outputs sum 9.846392;
ON (device add-norm, forced at T=1 for capture) outputs sum 9.765238 — ratio
0.9918, the 0.9899-class systematic bias.

`VT_TT_NORM_DEBUG=1` on the ON arm shows the device norm's INPUTS are
bit-identical (`dev_x`/`norm_f32` sum −20.613504, `wf32` sum 858.502781) while
`dev_y` sums 9.777145 — so the deficit is in the tt-metal rms_norm kernel ON
THE MODEL'S ACTIVATION DISTRIBUTION. The committed micro
("SCRATCH dbg rmsnorm micro", random nd(0,0.5) inputs, same [1,2560], same eps,
patched substrate with `fp32_acc=true`) measures 0.9997 — clean. The bias is
input-distribution-dependent, not a config or staging difference.

The composed f32 repair is still substrate-blocked IN-MODEL: re-confirmed today
(`VT_TT_RMSNORM_COMPOSED=1` gate leg) — `TT_THROW: Invalid subtile broadcast
type` from binary_ng (binary_ng_device_operation.cpp:224) on the
`repeat([rows,1] -> [rows,d])` row scale.

## Verdict

The shadow-reconciliation hypothesis is FALSIFIED as the lever: both arms share
WeightViewShadows/Bfp8WeightShadows and their #3452 sweeps, and the arm delta
originates upstream of any shadow traffic, at the first add-norm. The host-free
arm is noisier because it forces the device rms_norm at T=1 (capture requires
no host ops), and that kernel carries a ~0.7% systematic, input-dependent bias
on kolibri1's activations; 50 layers x 2-3 norm sites compounds it into
multi-nat router-logit shifts and 40 vs 17 teacher-forced hard flips.

## Fix options priced

1. **Substrate characterization (recommended next lever).** The micro is clean
   where the model is not; find the in-model program difference (program-cache
   provenance, compute-config, CB data format under the model's call pattern)
   and either fix tt-metal or force the clean config in-model. Cheap to probe:
   the micro passes the same shapes/eps, so a reproducing in-model input dump
   (`emb` row + `gam`) fed through the micro isolates input-dependence vs
   program provenance in one run.
2. **Composed f32 chain** — blocked by the binary_ng subtile broadcast refusal;
   needs an upstream broadcast primitive or a matmul-shaped row scale. Priced
   before (ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN), still refused on this stack.
3. **Host add-norm inside host-free** — falsified: the host round-trips desync
   the device shadow (1e37 collapse, measured 7 HARD today).
