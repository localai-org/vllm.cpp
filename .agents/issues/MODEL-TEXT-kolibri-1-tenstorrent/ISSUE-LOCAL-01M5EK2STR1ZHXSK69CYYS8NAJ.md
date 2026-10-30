ID: ISSUE-LOCAL-01M5EK2STR1ZHXSK69CYYS8NAJ
Title: the pinned tt-metal moved f32 ttnn::sum onto the tf32 tile-reduce — the attention q/k norm computes its f32 norm statistic at 2^-11
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The tt-metal stack this backend builds against (tt_umd 0.9.12 era, /tmp/umdtrial-install) routes an f32 ttnn::sum onto the tf32 FPU tile-reduce, so a 'sum in f32' is no longer exact: the pin's own headers document the split (generic_reductions.hpp: mean's default fast_and_approximate_mode=false is 'the accurate SFPU path (full fp32)', true is 'the faster tf32 FPU path', and sum has no SFPU arm; reduce_helpers_common.hpp:14 'Fast keeps fp32 on the FPU/GMPOOL path (inputs truncated to tf32 — faster, lossy)'). Commit 4b3a0091c repaired the two exact-f32 DOT sites (the composed GDN decode step and the grouped keep-quant decode) with the mean*K pattern. The sweep for the remaining sites measured the unattributed reductions on the P150: the kAttnQkNormRopeGate q/k norm (src/vt/tenstorrent/tenstorrent_ops.cpp:1290) reduces x^2 with ttnn::sum over a 2-D f32 [T*Hq, Dh] tile and keeps the result at f32 (the model's dq3/dk3 are DType::kF32, qwen3_5.cpp:5877-5878), so the tf32 truncation IS the op's dominant error: the device suite measures q max_abs 3.46e-4..4.15e-4 and k max_abs 3.69e-4..5.25e-4 against the 1e-3 abs floor the case documents (test_tenstorrent_backend.cpp:5378), i.e. 35-52 percent of the budget. A direct device probe against an f64 host reference on the same shapes measures the sum path at 2.0e-4 relative (the 2^-11 signature) and the mean*K path at 1.4e-7 relative. The other three f32 sum sites are NOT defects and are recorded in the audit (docs/bench-evidence/tt-f32-reduction-sweep-20261021/report.md): the GdnPostConv l2 and the kL2Norm keep their rows at bf16 in the model path (actdt, qwen3_5.cpp:4678-4683) and their measured deviation is dominated by the bf16 tile upload (1.115e-3 measured, of which the tf32 sum contributes 2.7e-5), and the composed rmsnorm arm is the non-default diagnostic (VT_TT_RMSNORM_COMPOSED=1) whose output is stored bf16.

## Resolution

FIXED in this change: the q/k row statistic reduces through ttnn::mean (the pin's accurate full-fp32 SFPU path) instead of ttnn::sum (the tf32 FPU tile-reduce), and the eps lands at the CPU oracle's mean scale (cpu_ops.cpp:1707). The arm is one op shorter than the sum arm, which keeps the op's chain inside this stack's L1 buffer budget - the mean rescaled by Dh (the mean.K form) makes the op's final ttnn::concat read clobbered buffers on this pin. Red-first (P150, deterministic): with the case's f32-out floor tightened 1e-3 -> 1e-5, the original sum arm fails 12/24 assertions (the f32-out arms measure q/k max_abs 3.46e-4..5.25e-4 against the 1e-5 floor). Green: 24/24 assertions pass and the same arms measure 2.24e-8..7.15e-7 (the bf16-out arms become the oracle's exact RNE round). The sweep's full audit, with the measured classification of every f32-exact reduction site in the TT backend, is docs/bench-evidence/tt-f32-reduction-sweep-20261021/report.md; the three other ttnn::sum sites are recorded there as tolerant (their rows are stored bf16 in the model path) with their numbers, and the composed rms-norm arm's one-line mean swap is named as its own unit's next step.
