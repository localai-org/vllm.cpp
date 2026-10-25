# The Tenstorrent rms_norm substrate repair — an f32-exact device arm reachable in the model context (W1)

Row: BACKEND-TENSTORRENT.
Issues: ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF (the kernel-precision record,
canonical for this work) and ISSUE-LOCAL-01M4YVEP000ENCRECVB0S6FDDY (the
arm-delta record it re-attributes into). Upstream: tenstorrent/tt-metal#62339
(open, no response). Device evidence base:
`docs/bench-evidence/tt-rms-norm-compute-config-20261021.md` (the probe, the
raw matrix log, and the reconstruction live there).

## Problem

The pinned tt-metal `ttnn::rms_norm` deviates from the host-f64 oracle
SYSTEMATICALLY and INPUT-DEPENDENTLY on real in-model activation rows:
0.9809..1.0077 on the captured kolibri rows depending on the kernel's compute
config (`fp32_acc`/`approx_mode`), never exact. Normed once per norm site
across ~50 layers, this compounds into multi-nat router-logit shifts and the
recorded prompt-dependent hard flips (40 vs 17) and the flint W0
device-vs-CPU baseline (rel_rms 2.58e-01, 1/8 greedy).

Two repairs are already REFUSED by the substrate and must not be re-tried
blindly (full record in `src/vt/tenstorrent/tenstorrent_ops.cpp`
RmsNormKernel):

1. Composed f32 chain (x^2 -> mean -> +eps -> rsqrt -> scale) is EXACT in the
   micro-test (ratio 0.9998), but its [rows,1] row scale cannot reach the
   model context: plain multiply broadcasts padded-tile garbage (sporadic
   1e37 logits), `BcastOpDim::W` is numerically wrong (1.019),
   `BcastOpDim::H` is refused, and repeat + same-shape multiply is refused
   by binary_ng ("Invalid subtile broadcast type").
2. Serving norms from the host f32 loop desyncs the device shadow (gate
   collapsed 26/33 -> 0/8, 1e37 garbage).

## Scope

W1 makes ONE f32-exact device rms_norm arm reachable in the MODEL context.
Not speed. Not an upstream tt-metal fix landing.

Candidate routes, cheapest first, each gated by a micro-test AND a
model-context probe before adoption:

- **A1 — ROW_MAJOR scale broadcast.** Serve the composed chain's [rows,1]
  scale as a ROW_MAJOR-layout tensor (no tile padding to leak) into the
  multiply.
- **A2 — matmul outer product.** Materialize the row scale as
  matmul(inv [rows,1], ones [1,d]) -> [rows,d] (matmul measured clean at
  1.9e-3), then a same-shape f32 multiply.
- **B — a LOCAL patch on the pinned tt-metal fork**
  (`~/Sources/tt/tt-metal-pin`): force the f32-exact accumulation inside the
  rms_norm compute kernel (`layernorm.cpp`). If used, the patch is recorded
  as a LOCAL patch on a pinned oracle (file, diff, rebuild+install steps),
  never silently upstream-equivalent. The pinned substrate
  (`/tmp/umdtrial-install`, `/tmp/tt-metal-umdtrial`) must survive.
- **C — binary_ng subtile-broadcast patch**, only if A1/A2 fail and C is
  shown to be an unimplemented (not wrong) path.

## Design

The repair lands in `RmsNormKernel` (tenstorrent_ops.cpp) behind an
env-gated arm first (`VT_TT_RMSNORM_COMPOSED=2` for A1, `=3` for A2), is
proven red-first against the existing micro
(`test_kolibri1_tt_b2ii` "SCRATCH dbg rmsnorm micro" with
`VT_TT_RMSNORM_MICRO_INPUT` replaying the captured normrow-000), then the
winning arm becomes the device default with the previous arm kept behind
`VT_TT_RMSNORM_TTNN=1` for A/B. The reduce side uses `ttnn::mean`
(f32-exact, commit b67b30fd) rather than `ttnn::sum` (tf32 tile-reduce).

## Tests and gates

1. Micro-test EXACT: "SCRATCH dbg rmsnorm micro", both random and replayed
   normrow-000 arms, tt/cpu and tt/host2 ratios within bf16 rounding of 1.
2. Model-context leg: `tests/test_flint_tt_w0` with
   `FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf` —
   the W0 baseline is rel_rms 2.58e-01 and greedy 1/8; record the after
   numbers honestly whatever they are.
3. No regression: the kolibri TT gate (`test_kolibri1_tt_b2ii`) recorded
   numbers are the floor.

## Risks

- The multiply that consumes a ROW_MAJOR [rows,1] may go through the same
  broken bcast path (A1 falsified) — the micro decides.
- matmul's own 1.9e-3 error may exceed the current arm on some rows (A2
  traded exactness for reachability) — compare per-row ratios against the
  incumbent arm, not only against 1.0.
- A pin patch (B) invalidates substrate comparability with the recorded
  evidence chain; if adopted, the report must name the exact rebuilt
  install and re-measure the BEFORE numbers on it too.

## Evidence

`docs/bench-evidence/tt-rmsnorm-w1-20261025/report.md`: before/after
device-vs-CPU numbers on BOTH flint and kolibri legs, recipe, env, patch
provenance if route B.

## Stop conditions

- If every route is falsified by the micro or the model context, record the
  falsification per route in the report and leave the issue open with the
  next lever named. No arm ships without (1) and (2).

## Git integration

One pull request (repository default; no recorded developer preference for
this row).

## Owed

- The W0 gate band (max_abs<64 / rel_rms<0.5) stays untightened until W2
  (the matmul f32->tf32 operand suspect) closes; tightening now would dress
  an unchanged number.
- W2: audit the device forward's matmul operand dtypes against the tf32
  truncation; the W1 A/B falsified the rms_norm attribution, so the model
  gap's driver is still open.
