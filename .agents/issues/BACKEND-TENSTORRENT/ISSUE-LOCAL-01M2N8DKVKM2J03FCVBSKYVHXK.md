ID: ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK
Title: APEX e2e: keep-quant decode OOM — chunk policy misses the TILE-tiling blowup
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The APEX 27B e2e decode dies in MatmulBTQuantGroupedKernel -> DecodeKeepQuantWordsF32 -> repair -> to_layout(TILE): the repair lambda tiles {B,16,16} product planes to {B,32,32}, a 4x blowup the 256 MiB plane budget does not count. The ceil(N/8) trace term forces 31,040-row chunks on the head [248320,5120]; the largest single TILE alloc is then 2.5 GB (318 MB/bank) against a 238 MB largest free block. An env-lever probe (VT_TT_KEEPQUANT_CHUNK_BYTES=64MiB, chunk 3,276) completed the first full e2e generation (exit 0, 64/64 tokens), proving the attribution. The default policy must account for tiling so the knob is not required.

## Resolution

2026-10-21: FALSIFIED BY THE TREE. The tiling-aware cap this issue asks
for landed as `5d891f6d6` ("fix(BACKEND-TENSTORRENT): the keep-quant chunk
tiling cap, sticky per weight", 2026-09-16), whose body closes this issue
by name ("Closes ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK") and carries the
green-after evidence: the full backend suite 77/77 with 524,495 assertions,
and the APEX 27B e2e probe WITHOUT the `VT_TT_KEEPQUANT_CHUNK_BYTES` knob,
exit 0 with 64/64 generated tokens. The knob the issue said was required is
no longer required.

The cap is in this tree at `src/vt/tenstorrent/tenstorrent_keepquant.cpp:1339`
(`chunk = std::min(chunk, std::max<int64_t>(plane_bytes / (K * 16), 1))`),
with the divisor named in the comment at `:1313`, and the spec
`.agents/specs/tenstorrent-apex-keepquant-chunk-tiling.md` records the
attribution and the gates.

