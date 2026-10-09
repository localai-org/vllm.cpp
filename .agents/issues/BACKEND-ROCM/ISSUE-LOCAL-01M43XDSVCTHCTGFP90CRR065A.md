ID: ISSUE-LOCAL-01M43XDSVCTHCTGFP90CRR065A
Title: Records rot on rocm-gfx11-exl3-perf: dangling SHAs, wrong closed-issue root cause, stale spec claims
Row: BACKEND-ROCM
State: OPEN
Kind: cleanup
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

(1) The Outcome 2026-10-04 anchors cite four SHAs that are not ancestors of HEAD (369348d21, bef2d0d5f, 852816b18, edd7765b6 = dangling pre-rebase; landed equivalents are 68f239609, ef05e04cd, be089bbea, 87de2c0ff). (2) ISSUE-LOCAL-01M41E6SXYTRPR7H25XNPAHSTE is CLOSED with the wrong root cause ('bf16-in Hadamard load arm not reproducing CastF16K bit-exactly'); cbd773c7e's own message shows the real cause was the want_recon window handing bf16 operands to the f16-only recon kernel at 32<m<=144; if the recorded cause were true the unchanged fold would corrupt again. (3) gfx1101-exl3-decode-latency.md design item 4 says prefill keeps GdnScanCoopK; the tree serves varlen prefill through the fused scan by default since 921dceccb. (4) The 2026-10-02 'hard ceiling statement' bullet (zero-cost decode caps at ~35 tok/s) is falsified by the 960.8 GB/s read benchmark but carries no correction marker, unlike the Outcome blockquote above it. (5) docs/ENVIRONMENT.md VT_EXL3_GEMV_FOLD_OUT row says the epilogue keeps 'separate __fmul_rn ops' but the code uses plain multiplies (the substance is right: no contractible pair, -ffp-contract=off); the spec's Owed line about pinning the dot arm 'the way the GEMV fold pins its epilogue with __fmul_rn' references a mechanism that does not exist in the code.

## Resolution

-
