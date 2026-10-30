# Tenstorrent f32 reduction sweep: every f32-exact reduction site in the TT backend (2026-10-21)

Row: `MODEL-TEXT-kolibri-1-tenstorrent` (sweep of the shared Tenstorrent
backend). Branch `row/TT-F32-EXACT-SWEEP` @ `891c4f27d` (origin/main).
Host: personal Tenstorrent Blackhole P150a workstation; device work under
`flock -x $HOME/gpu.lock`; the pinned substrate `/tmp/umdtrial-install`
(tt_umd 0.9.12 era, built 2026-10-09) untouched; build `/tmp/build-sweep`
(Release, `-DVLLM_CPP_TENSTORRENT=ON`, `-j 4`).

## Question

The tt-metal stack advance moved an f32 `ttnn::sum` onto the **tf32 FPU
tile-reduce**, so "a sum in f32" is no longer exact: it truncates its inputs
to tf32 (2^-11 relative). Commit `4b3a0091c` (PR #3481) repaired the two
exact-f32 DOT sites it found (the composed GDN decode step and the grouped
keep-quant decode). This sweep asks the remaining question: **where else in
the Tenstorrent backend does an f32-exact reduction exist, and which of those
sites are genuinely wrong?**

## The pin's own rule (upstream anchors)

- `include/ttnn/operations/reduction/generic/generic_reductions.hpp:44-52` —
  `ttnn::mean` carries `fast_and_approximate_mode = false`: "When false
  (default), fp32 mean reduces on the accurate SFPU path (full fp32); true
  selects the faster tf32 FPU path." `ttnn::sum` has no such parameter.
- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:30-32`
  — "Only ttnn.mean exposes fast_and_approximate_mode ... `True` uses the
  faster FPU path (inputs truncated to TF32, higher ULP error). The accurate
  path requires a compute_kernel_config with `fp32_dest_acc_en=True`."
- `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:127-130` —
  the dispatch: `use_sfpu_fp32_mean = !fast_and_approximate_mode && dtype ==
  FLOAT32 && reduce_math == AVG && arch != QUASAR && config.fp32_dest_acc_en`.
  The default config sets `fp32_dest_acc_en` (init_device_compute_kernel_config,
  `default_fp32_acc=true`), so a plain `ttnn::mean` takes the accurate path.
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp:14` — "Fast keeps fp32
  on the FPU/GMPOOL path (inputs truncated to tf32 — faster, lossy);
  Accurate routes fp32 SUM through the SFPU for full-fp32 accumulation."
- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp:262-290` —
  SUM dispatches to the FPU tile-reduce; AVG (mean) dispatches to the SFPU
  path, so the only f32-exact reduction the API exposes is `ttnn::mean`.

## The audit table

Every reduction site in `src/vt/tenstorrent/` (grep `ttnn::sum|ttnn::mean|
reduce|sum(`), classified. "TF32 COST" is the deviation the tf32 tile-reduce
contributes, measured on the P150 against an f64 host reference on the same
shapes (a device probe: the reduction chain, the norm chain, and the
one-hot matmul copy).

| # | Site | What it reduces | Class | TF32 COST (measured) | Verdict |
|---|---|---|---|---|---|
| 1 | `tenstorrent_ops.cpp:1290` — `kAttnQkNormRopeGate` q/k row norm | `sum(x^2)` over Dh, f32 | norm statistic, **result kept at f32** (the model's `dq3`/`dk3` are `DType::kF32`, `qwen3_5.cpp:5877-5878`) | 2.0e-4 relative on the row scale; the case measured 3.5e-4..5.2e-4 abs, **35-52 % of its 1e-3 floor** | **(a) FIXED here** |
| 2 | `tenstorrent_gdn.cpp:67` — `kGdnPostConv` l2 lambda | `sum(x^2)` over Dk, f32 | norm statistic, result stored **bf16** in the model path (`actdt`, `qwen3_5.cpp:4678-4683`) | 2.7e-5 abs of the site's 1.115e-3 total deviation (2.4 %); the total is the documented bf16 tile upload (`TileSpecOf` is BFLOAT16, `tenstorrent_residency.cpp:163-166`) | (b) tolerant — 16x below the bf16 store round (2^-8), 0.13 % of the 0.02 abs envelope |
| 3 | `tenstorrent_gdn.cpp:1301` — `kL2Norm` | `sum(x^2)` over D, f32 | norm statistic, result stored **bf16** in the model path | same as #2 (the same upload) | (b) tolerant — same reason; the declared envelope is 0.02 abs against a measured 1.115e-3 |
| 4 | `tenstorrent_ops.cpp:696` — the composed rms-norm arm (`VT_TT_RMSNORM_COMPOSED=1`, **not** the default) | `sum(x^2)` over D, f32 | norm statistic, result stored **bf16** in the model path | 2.0e-4 relative, 16x below the bf16 store round | (b) tolerant — **but** the arm's own comment calls it "the same chain the CPU oracle runs (x^2 -> mean -> ...)" (`tenstorrent_ops.cpp:646-648`), and the 1-line `ttnn::mean(sq, {1}, true)` swap (the arm already divides by `d` afterwards) would make that claim true; the arm is env-gated and no case covers it, so it is left for the arm's own unit |
| 5 | `tenstorrent_gdn.cpp:652,661` — the composed GDN decode dots | signed dot over Dk, f32 | dot product, state kept at f32 | 6.1e-3 (K=128) / 3.3e-3 (K=5120) relative — the cancellation-amplified signature | (a) already fixed by `4b3a0091c` (mean·K); re-measured here as 8.2e-7 / 4.5e-6 |
| 6 | `tenstorrent_keepquant.cpp:1421,1492` — the grouped keep-quant decode dots | signed dot over K, f32 | dot product | worst_rel 0.1307 (per `4b3a0091c`) | (a) already fixed by `4b3a0091c` (mean·K) |
| 7 | `tenstorrent_gdn.cpp:752` — the one-hot state-index gather (`ttnn::matmul(oh, cache2d)`) | a degenerate dot (0*x + v) over the slot dim, f32 | state copy, the ssm state is f32 on device | 1.74e-3 relative on the copy | (b) tolerant **in a different hazard class** — this is the f32 matmul's tf32 MACs, not a ttnn reduction, and the code already documents the trade-off (`tenstorrent_gdn.cpp:517-528`: "its matmul rounds the gathered state to tf32, which sits inside the decode step's own tf32 envelope"). The indexed arm's 0.02 abs envelope absorbs it. **The comment at `tenstorrent_gdn.cpp:1474-1476` ("exact 0/1 one-hot matmuls (0*x + v == v in f32)") is stale on this pin** — the same file states the opposite at :1493-1497 |
| 8 | `kernels/keepquant_kernel_code.h` — the keep-quant dot kernels | the per-block dot | integer accumulation | n/a | (c) not a ttnn reduction: the per-block accumulator is integer end to end and the f32 fold is in registers at full precision |
| 9 | `tenstorrent_ops.cpp:693`, `tenstorrent_gdn.cpp:1386` — `ttnn::rms_norm`; `tenstorrent_ops.cpp:408` — `ttnn::layer_norm` | the fused norm's own reduction | fused kernel | n/a | (c) not a ttnn reduction — the reduction is inside the pin's fused kernel. The rms_norm's ~1 % in-model bias is a separate tracked issue (ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF) |

### Why #2, #3 and #4 are tolerant and #1 is not

The discriminating question is not "does the truncation fit the envelope" —
it is **the precision at which the site keeps its result**:

- #2, #3, #4 store the normalized row at **bf16** in the model path, whose
  store round is 2^-8 = 3.9e-3 — 16x coarser than the 2^-11 = 4.9e-4 the
  reduction loses. The truncation is invisible under the store round, and
  the sites' measured deviations are dominated by other, documented causes
  (the bf16 tile upload for #2/#3).
- #1 keeps the row at **f32** and feeds it straight into the attention, so
  the reduction's truncation IS the op's dominant error: the device case
  measured 3.5e-4..5.2e-4 abs against a 1e-3 floor it had carved out of a
  5.4e-3 defect, i.e. the truncation was eating 35-52 % of the budget.

## The fix (site #1)

`src/vt/tenstorrent/tenstorrent_ops.cpp`, the `norm_rope` lambda of
`AttnQkNormRopeGateKernel`:

```cpp
-    ttnn::Tensor sq = ttnn::multiply(x, x);
-    ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
-    ttnn::Tensor denom =
-        ttnn::add(ttnn::multiply(s, 1.0f / static_cast<float>(dh)), na.eps);
-    ttnn::Tensor inv = ttnn::rsqrt(denom);
+    ttnn::Tensor sq = ttnn::multiply(x, x);
+    ttnn::Tensor m = ttnn::mean(sq, ttsl::SmallVector<int>{1}, true);
+    ttnn::Tensor denom = ttnn::add(m, na.eps);
+    ttnn::Tensor inv = ttnn::rsqrt(denom);
```

`ttnn::mean` takes the accurate full-fp32 SFPU path, and the CPU oracle's
chain is `rsqrt(mean(x^2)+eps)` (`cpu_ops.cpp:1707`) — the same mean scale
this arm now uses, where the sum arm applied `eps` at the sum scale. The arm
is also one op SHORTER than the sum arm (no `* (1/dh)`), which is what keeps
the op's remaining chain inside this stack's L1 buffer budget: the
`mean·K` form (mean, then rescale by Dh) makes the op's final
`ttnn::concat` read clobbered buffers on this pin — the normed row and the
rope's products are then exact (5e-9) while the concatenated output is off
by 0.25-0.43 and the committed q/k output is garbage (max_abs 1.95). The
mean-scale form keeps the op's op count at or below the sum arm's and the
case is green.

### Red-first evidence (P150, deterministic, 2/2 identical runs)

`tests/vt/test_tenstorrent_backend.cpp:5370-5382` — the case's f32-out abs
floor is tightened from 1e-3 to **1e-5** (14x the measured post-fix
headroom of 7.2e-7, 540x below the 5.4e-3 bf16-typecast defect the floor
also separates).

- RED — the original `ttnn::sum` arm with the tightened floor:
  `kAttnQkNormRopeGate` **12 of 24 assertions fail**; the f32-out arms
  measure q/k max_abs 3.46e-4/3.69e-4 (T=1), 4.15e-4/5.25e-4 (rot=128),
  2.28e-4/2.12e-4 (in=bf16), 2.84e-4/2.83e-4 (T=65), 2.80e-4/2.40e-4
  (Hq=32), 3.20e-5/2.96e-5 (gemma=false) against the 1e-5 floor.
- GREEN — the `ttnn::mean` arm: **24/24 assertions pass**; the same arms
  measure 4.77e-7/2.38e-7, 5.96e-7/4.77e-7, 4.77e-7/2.38e-7, 7.15e-7/4.77e-7,
  4.77e-7/4.77e-7, 4.47e-8/2.24e-8. The bf16-out arms become the oracle's
  exact RNE round (0/0 for in=bf16, 1.22e-4/2.44e-4 for in=f32), which the
  op's own doctrine (`tenstorrent_ops.cpp:1077-1082`) claims a bf16 out
  should be.

### Before / after against the CPU f32 oracle (the case's own MESSAGE)

| Arm | Before (sum/tf32) | After (mean/SFPU) |
|---|---|---|
| T=1 Hq=4 rot=64 gemma in=f32 out=f32 | q 3.457e-4, k 3.691e-4 | q 4.768e-7, k 2.384e-7 |
| T=3 rot=128 gemma in=f32 out=f32 | q 4.153e-4, k 5.248e-4 | q 5.960e-7, k 4.768e-7 |
| T=3 gemma in=bf16 out=f32 | q 2.279e-4, k 2.120e-4 | q 4.768e-7, k 2.384e-7 |
| T=65 gemma in=bf16 out=f32 | q 2.844e-4, k 2.832e-4 | q 7.153e-7, k 4.768e-7 |
| T=3 Hq=32 in=bf16 out=f32 | q 2.799e-4, k 2.401e-4 | q 4.768e-7, k 4.768e-7 |
| gemma=false in=f32 out=f32 | q 3.202e-5, k 2.958e-5 | q 4.470e-8, k 2.235e-8 |
| in=bf16 out=bf16 | q 3.906e-3, k 1.953e-3 | q 0, k 0 |
| in=f32 out=bf16 | q 2.906e-3, k 2.648e-3 | q 1.221e-4, k 2.441e-4 |

## Suite counts (P150, default order)

- BEFORE (this tree, `891c4f27d`): **108 cases — 107 passed, 1 failed**;
  **791,119 assertions — 791,118 passed, 1 failed**. The single failure is
  the pre-recorded 2261-class RAC flake
  (`test_tenstorrent_backend.cpp:2550`, the batched decode PagedAttention
  capture-safety case), which passes isolated 3/3 and varies run to run.
- AFTER (this change, the mean arm): **108 cases — 107 passed, 1 failed**;
  **791,119 assertions — 791,117 passed, 2 failed**. The failing case is the
  same pre-recorded 2261-class RAC flake
  (`test_tenstorrent_backend.cpp:2550-2551`): this run's values are
  126/128 on K and 126/128 on V (the pre-fix baseline of the same day
  measured 123/128 K and 128/128 V), and the case **passes isolated 3/3**
  with this binary. The variation is the documented one, not a new failure.
- The `kAttnQkNormRopeGate` case itself: **24/24 assertions pass**, and its
  arms measure as tabulated above.
- `test_kolibri1_tt` (host suite, same tree): **18/18 cases, 257/257
  assertions**.

## Owed / next steps

- The composed rms-norm arm (#4) can take the same one-line `ttnn::mean`
  swap when its own unit lands; the arm is env-gated and no case covers it
  today.
- The stale comment at `tenstorrent_gdn.cpp:1474-1476` ("exact 0/1 one-hot
  matmuls (0*x + v == v in f32)") contradicts `tenstorrent_gdn.cpp:517-528`
  and `:1493-1497` on this pin.
- The `mean·K` form (mean, then rescale by K) is NOT safe on this pin in a
  chain that ends in a `ttnn::concat`: it makes the concat read clobbered
  L1 buffers. Sites that need the sum of a signed product should check the
  op's op count and buffer budget before adopting it.
