ID: ISSUE-LOCAL-01M590MZYN28EA6DEKFFA8GNVY
Title: the tt-metal pin advance moved f32 ttnn::sum onto the tf32 FPU tile-reduce — the composed GDN decode step lost its f32-exact dot
Row: BACKEND-TENSTORRENT-GDN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-19
Updated: 2026-10-19
Closed: -

## Problem

After the tt-metal stack advance (252ca04f8, tt_umd 0.9.12 era /tmp/umdtrial-install, built 2026-10-09) the wide-range arm of kGdnDecode fails deterministically on the P150: state max_abs 0.0051, max_rel 1.83 vs tol 0.002/abs 1e-5 (test_tenstorrent_backend.cpp:4126; 3/3 identical runs, isolated and in-suite). GDN product code is unchanged since the pin advance; the composed step's exact-f32 construction is broadcast multiply + ttnn::sum (tenstorrent_gdn.cpp GdnDecodeStepComposed), and on the new pin f32 ttnn::sum lands on the tf32 FPU tile-reduce — the same truncation class 8ab35b966 documented for ttnn::matmul. The error lands in the small-magnitude delta-rule terms, the exact signature that commit pins the arm for. The VT_TT_GDN_DECODE=chunked diagnostic degrades further (max_abs 0.246), ruling out a composed-only defect.

## Resolution

FIXED in this repair: the dot reductions route through ttnn::mean(dim, keep_dim) - Dk instead of ttnn::sum. The pin's mean documents its default as the accurate full-fp32 SFPU path (generic_reductions.hpp), so the reduction is exact again: state rel_rms 3.6e-5 -> 3.2e-8, max_rel 1.83 -> 9.1e-4, focused arm green 2026-10-19. No assertion re-pinned.
