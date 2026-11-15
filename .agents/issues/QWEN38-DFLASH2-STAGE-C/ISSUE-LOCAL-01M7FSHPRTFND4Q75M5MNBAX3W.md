ID: ISSUE-LOCAL-01M7FSHPRTFND4Q75M5MNBAX3W
Title: EnsureHostBytes refuses the bf16 GDN state cache readback the documented contract promises
Row: QWEN38-DFLASH2-STAGE-C
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

f4725780b (FLINT-TT-DENSE-MLP-STAGING) made EnsureHostBytes' generic tail write the shadow's dtype width only: an f32 shadow now requires the host buffer to fit n*4 f32 bytes. The GDN ssm cache shadow is ALWAYS f32 (EnsureGdnCacheDevice stages f32 regardless of the host cache dtype, tenstorrent_gdn.cpp:410-416), and a production bf16 mamba_ssm cache registers its slot at n*2 bytes — so any download of a device-current bf16 GDN state cache throws 'host buffer too small for the f32 shadow' instead of rounding the stored truth through bf16 on the way out. The conv-transposed arm of the same function kept the width-adaptive form (f32 when it fits, else bf16-rounded), and tenstorrent_gdn.cpp:410-416 still documents the rounding as existing. Red on the base (no wave changes in the path): tests/vt/test_tenstorrent_backend.cpp kGdnDecode bf16-state arm fails at its final readback (the kGdnSpecDecode wave's sweep E hits the same wall). Fix: restore the width-adaptive bf16 arm in the generic tail — f32 write when the slot fits n*4, else bf16-rounded write when it fits n*2 (the conv-transposed arm's exact form); the #1922 rule (never write f32 into a sub-f32 buffer) is preserved because the f32 branch still requires n*4.

## Resolution

Fixed in the kGdnSpecDecode wave 2026-11-15: the width-adaptive bf16 arm restored in EnsureHostBytes' generic tail (f32 when the slot fits n*4, else the bf16-rounded write — the conv-transposed arm's form; the #1922 rule survives). Green on the P150: the kGdnDecode bf16-state readback 0/98304 bit mismatches and the kGdnSpecDecode bf16-state sweep, 4/4 test cases EXIT 0 (w4-gdnspec-green.log).
