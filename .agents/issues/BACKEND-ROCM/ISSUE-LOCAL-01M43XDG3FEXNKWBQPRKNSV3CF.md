ID: ISSUE-LOCAL-01M43XDG3FEXNKWBQPRKNSV3CF
Title: Served greedy-continuation probe is presented as a gate but is not committed
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

After e1982b8aa proved unit tiers stay green while served decode corrupts, the row spec (gfx1100-exl3-decode-60tps.md Outcome 2026-10-04 'Gates:') lists 'the 128-token greedy continuation is byte-identical with VT_EXL3_GEMV_FOLD_OUT=0 vs =1' -- but no test, script, or golden commits this; the probe clients live only in ~/agent-artifacts. The repo already has the committed form (scripts/deepseek-v2-dgx-gate.sh gates served greedy by md5 against tests/parity/goldens/). The 2026-10-04 decode-loop corruption (ISSUE-LOCAL-01M43XCSPS01QDG7D65Q8NKHQ7) is precisely the class this uncommitted gate would have caught at review time: '8 runs, one md5' compared builds to each other, and all of them were wrong. Also record: the served probe must compare against an independent reference (oracle or a verified-good build), not only knob A/B.

## Resolution

-
