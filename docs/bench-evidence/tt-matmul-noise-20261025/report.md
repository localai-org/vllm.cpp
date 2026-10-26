# W3 — the substrate matmul-noise characterization and the exactness-attempt verdict

Date: 2026-10-26. Branch `row/TT-MATMUL-W3`, spec commit `15fdfd5ce` (spec
`.agents/specs/tt-matmul-noise.md`), base `a1b94d9ba` (head of
`row/FLINT-TT-W2`, which is PR #3511 — open, not merged; the stack is
deliberate and recorded in the spec). Issue
`ISSUE-LOCAL-01M5TB3983MW41T4NPBZ5ZYF0E`.

## Verdict (the spec's route decision)

The noise is the pinned ttnn **default math fidelity (LoFi) multiply
truncation**. It reproduces on a plain ttnn matmul with no vt staging
(`test_flint_tt_w3_plain`, route (b) evidence: LoFi+f32-dest arm reads
rel_rms 2.71e-2, frac_bad 1.76e-2 — the W2 class, outside our stack), and it
is fixable by a compute-kernel-config selection (route (a)):

| arm | micro rel_rms (host-double ref) | frac_bad |
|---|---|---|
| shipped (LoFi + approx, f32-dest on the f32-out arm) | 2.3e-2 … 2.6e-2 | ~2e-2 |
| **HiFi4 + approx off + fp32_dest_acc_en** | **3.3e-4** | **0** |

The exactness arm is a **~70x micro improvement and was NOT adopted**: it
does not move the W0 model number (below — it measures slightly WORSE), so
per the W1/W2 discipline the shipped arm is unchanged. `F32AccKernelConfig()`
carries the map's conclusions in-source; `VT_TT_MATMUL_MATH_FIDELITY=hifi4`
selects the experimental arm (scratch-micro and A/B only), `=lofi` pins the
shipped behavior explicitly.

This is NOT an upstream-filing case: the behavior is the documented meaning
of `math_fidelity` at the pin (`compute_kernel_config.hpp` defaults
`LoFi` + `math_approx_mode=true`), not a defect. The fix is config
selection, which remains available behind the env and can be revisited if a
later wave changes the exactness economics.

## The W2 anomaly correction

W2's report (and PR #3511) records an open anomaly: MatmulBT/bf16-out arms
reading uncorrelated (rel≈1.2) against the f32-out arm. **The anomaly is a
TEST-SIDE STAGING BUG in the W2 micro, not a substrate defect.** The W2 micro
fed the bt arms [K,N] data staged as {N,K}. W3's map re-runs all four arms
on the SAME host-generated values with correct staging (pathcmp case): with
correct staging the bt arms track the non-bt arms on every shape. W3
supersedes the W2 record on this point; cross-reference into #3511 at merge.

One REAL staging-class observation survives, precisely named under "Open
defects".

## The noise map (HiFi4 exact arm, host-double reference, exact bf16 operands)

rel_rms is flat in every axis — the residual is per-op constant, not
accumulation growth — and the single-product probe is bit-exact:

| sweep | values | rel_rms | note |
|---|---|---|---|
| K (M=1,N=512) | 32…2560 | 2.85e-4 … 3.46e-4 | K-independent |
| M (K=N=512) | 1…32 | 2.85e-4 … 3.24e-4 | M-independent |
| N (M=1,K=512) | 32…2048 | 2.59e-4 … 4.33e-4 | N-independent |
| tiles/off-tile | {516,516},{520,512}… | 3.04e-4 … 3.40e-4 | no tile-quantization signature |
| model q/k/v | {1,2048,6144} | 3.10e-4 | not at a bad point |
| per-product (x=e_k) | k=0,256,511 | **0** | a single multiply is bit-exact |

The same map on the shipped arm reads ~2.3e-2 … 2.6e-2 rel_rms flat,
frac_bad ~2e-2 — the W2 number, reproduced
(`noise-map-shipped-arm.log`, `red-first-shipped-arm.log`).

The residual 3.1e-4 class on the exact arm: per-product exact, K-independent,
so it is not multiply truncation and not accumulation growth; scattered bad
elements (up to frac_bad 4.9e-4 at M=5 K=2048) sit at scattered positions
with ~1e-4 absolute error on near-zero references. Recorded as the observed
floor of the HiFi4+f32-dest arm at this pin; not tile-boundary-localized.

## The plain-ttnn repro (does it reproduce outside vt?)

`test_flint_tt_w3_plain.cpp` — raw `ttnn::operations::matmul::matmul` via
tt-metal's own `Tensor::from_vector -> to_device`, no vt staging, M=1 K=512
N=512:

| arm | rel_rms | frac_bad |
|---|---|---|
| default (LoFi, approx on, no f32 dest) | 4.04e-3 | 3.9e-3 |
| HiFi4, approx off, no f32 dest | 1.50e-2 | 7.8e-3 |
| **HiFi4 + approx off + fp32_dest_acc_en** | **3.28e-4** | **0** |
| LoFi + fp32_dest_acc_en | 2.71e-2 | 1.76e-2 |
| HiFi2 + fp32_dest_acc_en | 4.04e-3 | 3.9e-3 |

The noise reproduces on plain ttnn (LoFi+f32-dest row = the W2 class) and
the HiFi4 row matches our vt exact arm exactly (3.3e-4): the mechanism and
the fix are both substrate-level, and vt staging adds nothing on either side.

## Red-first (spec requirement)

The exactness CHECKs in `test_flint_tt_w3_noise` (rel_rms<1e-3,
max_abs<1e-3, plus the MatmulBT case) are **red** against the shipped arm
(`-tc="*exactness*"`: rel_rms 2.62e-2, all CHECKs fail, Status: FAILURE) and
**green** under `VT_TT_MATMUL_MATH_FIDELITY=hifi4` (all assertions pass).
They skip with a pointer to this report when the experimental arm is not
selected, because the shipped arm is intentionally left on LoFi; the red
behavior of the bound itself is on record (`red-first-shipped-arm-checks-
firing.log`: rel_rms 2.62e-2, all CHECKs fail). Run the pair with
`-tc="*dest-acc arm*,*MatmulBT arm*"`. Logs:
`red-first-shipped-arm-checks-firing.log`, `green-hifi4-arm.log`.

## W0 model verdict (the no-flip decision, honest)

`FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`,
`test_flint_tt_w0`, first-token logits vs the CPU row:

| arm | rel_rms | max_abs | greedy chain | gate |
|---|---|---|---|---|
| before (shipped arm, unchanged; the W2 base state) | **2.045e-1** | 30.2 | 1/8 | PASS |
| HiFi4 experimental | 2.181e-1 | 32.6 | 1/8 | PASS |
| LoFi + NO f32-dest (an arm this tree never shipped; early A/B) | 2.634e-1 | 30.4 | 0/8 | FAIL (chain_match) |

The exactness arm improves the micro 70x and does NOT improve the model —
first-token rel_rms moves 2.045e-1 -> 2.181e-1 (slightly worse), chain
1/8 on both. **No flip.** The W0 band stays at 0.5. The model's residual is
dominated by the recorded prompt-dependent hard-divergence bias class
(tenorrent_ops.cpp root-cause record; kolibri TT carries the same class),
not by matmul multiply noise — the map kills "multiply noise" as the
first-order model-level term.

## No-regress leg

Kolibri1 TT B2b-ii device token gate (`VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1`,
`-tc="*device token gate*"`): **fails identically on the shipped arm and the
experimental arm** (argmax_matches 3 vs the >=141 bound; hard_flips 5;
instr_hard_flips 40 — bit-identical numbers across arms,
`kolibri-token-gate.log`, `kolibri-token-gate-lofi-arm.log`). The failure is
pre-existing at the W3 base `a1b94d9ba` and is not attributable to W3 (W3
changes no executed default behavior). Flagged for the kolibri/TT row owner;
W3 does not repair it.

## Open defects named by the map (not fixed here)

1. **bf16-out staging at M=1**: every bf16-OUT arm at M=1 reads uncorrelated
   (rel_rms ≈ 1.2, all K/N swept) while f32-out at M=1 and bf16-out at M>=5
   (rel_rms 1.7e-3 = correct bf16 output rounding) are clean. The class is
   in the output readback/staging of a [1,N] bf16 result after
   CommitDevice2D, not in the compute config. Filed as
   ISSUE-LOCAL-01M5TWFY79FWJN2BDPKYQZ3VXS. The model path uses the f32-out
   arm, so nothing shipped is reached by it today.
2. **Residual scattered bad elements on the exact arm** (above): the
   observed 3.1e-4 floor, mechanism unnamed.
3. **Order-dependent bf16-out staging corruption**: with the exact arm
   selected, the case-7 allocations ({5,2048,2048} f32 out) followed by the
   case-8 bf16-out arm corrupt the readback (rel_rms 1.22, frac_bad 0.91 —
   `green-run-case8-staging-corruption.log`); the same case passes in the
   shipped-arm run and standalone. Same family as defect 1 (output
   readback/staging), order- and size-dependent. Both are recorded in
   ISSUE-LOCAL-01M5TWFY79FWJN2BDPKYQZ3VXS.

## Recipe

```sh
exec 9>$HOME/gpu.lock; flock -x 9
source ~/Sources/tt/env-tt-common.sh
export TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
~/Sources/tt/luwen/target/release/reset && sleep 15   # first run only
cmake --build /tmp/tt-matmul-w3-bld --target test_flint_tt_w3_noise \
  test_flint_tt_w3_plain test_flint_tt_w0 -j 8
/tmp/tt-matmul-w3-bld/tests/test_flint_tt_w3_noise -s                       # shipped map
/tmp/tt-matmul-w3-bld/tests/test_flint_tt_w3_noise -tc="*exactness*"        # red on shipped
VT_TT_MATMUL_MATH_FIDELITY=hifi4 /tmp/tt-matmul-w3-bld/tests/test_flint_tt_w3_noise \
  -tc="*dest-acc arm*,*MatmulBT arm*" -s                                       # green
/tmp/tt-matmul-w3-bld/tests/test_flint_tt_w3_noise \
  -tc="*dest-acc arm*,*MatmulBT arm*" -s                                      # red (bound firing on record; skips on the shipped arm today)
/tmp/tt-matmul-w3-bld/tests/test_flint_tt_w3_plain -s
FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  /tmp/tt-matmul-w3-bld/tests/test_flint_tt_w0
```

Environment: local P150 (blackhole), firmware bundle 19.15.0, KMD 2.11.1,
Gentoo host, kernel 7.0.1-gentoo, doctest 2.5.2, worktree `/tmp/tt-matmul-w3`
at the commit that carries this report, build `/tmp/tt-matmul-w3-bld`
(VLLM_CPP_TENSTORRENT=ON). Shared box; every device run under the gpu.lock
mutex. Raw logs beside this report.
