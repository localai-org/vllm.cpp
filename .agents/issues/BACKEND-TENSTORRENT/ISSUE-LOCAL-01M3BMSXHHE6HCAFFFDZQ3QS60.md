ID: ISSUE-LOCAL-01M3BMSXHHE6HCAFFFDZQ3QS60
Title: GDN sub-batch shapes overflow L1 circular buffers at higher request counts (2016260 B > 1572864 B)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-25
Updated: 2026-10-21
Closed: -

## Problem

With the BH core-bound decomposition landed (row/TT-GDN-BATCH-SPLIT), the 27B APEX decode at c=4/16 prompts loads past the original assert and serves ~36 min, then tt-metal throws per-program L1 circular-buffer overflow (2016260 B > 1572864 B, chunk_gdn phased program at the BH' sub-shape) in a retry loop. Distinct from the core-count assert (fixed) and the DRAM retention (other row): this bounds the decomposition's serving depth at higher concurrent batch counts. Candidate owners: the sub_b choice interacting with the factory's static CB sizing at the sub-shape, or a tt-metal factory limit at BH' values. Evidence: /tmp/gdn-c4.log legs 2026-09-25, commit body 9e7d073fd; c=4/4-prompt completes, so the overflow is count-dependent.

## Resolution

2026-10-21: STILL TRUE; not verifiable offline. The L1 circular-buffer overflow
at the BH' sub-shape is unchanged in the tree. The decomposition that reaches
it (`d482c6585`, at `src/vt/tenstorrent/tenstorrent_gdn.cpp:288-319` and
`:823-848`) still picks `sub_b = max(1, cores / hv)` with no per-program
circular-buffer sizing, and the commit that landed it records the overflow as
"a distinct, newly visible device limit, recorded as the open gap" -- this
record.

What settles it: a P150 run at c=4 and c=16 on the current pin with the
tt-metal cache cleared, which is where the overflow was first seen. The
core-count assert this record was distinct from is closed by `d482c6585` and
recorded as `ISSUE-LOCAL-01M3AF30AE0KR6M7JN8D9BA46B`.
