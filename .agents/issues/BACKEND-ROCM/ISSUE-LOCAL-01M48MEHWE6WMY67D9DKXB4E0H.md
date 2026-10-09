ID: ISSUE-LOCAL-01M48MEHWE6WMY67D9DKXB4E0H
Title: qwen3_5 vision out_hidden_size fallback lost in the ad4c6c42b merge delta
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-06
Updated: 2026-10-06
Closed: 2026-10-06

## Problem

PR #3400 commit ad4c6c42b5d1 read the vision tower geometry from vision_config but dropped the v.out_hidden_size = config.hidden_size fallback that upstream kept when it landed the same fix (db5300836). A config without a vision_config block therefore fell to the struct default 2560 instead of the text hidden size, and test_qwen3_5_moe_vision.cpp:411/413 plus test_qwen3_5_dense_vision.cpp:391 fail on every CPU lane (2560 != 2048/5120). These were the only two PR-caused failures across build-test-cpu, sanitize-cpu (address,undefined) and sanitize-cpu (thread); the other 13 failing suites fail identically on pristine upstream/main (verified against main CI logs at e67a071a6 and 0fbd7c994).

## Resolution

Fixed on rocm-gfx11-exl3-perf (PR #3400), the commit titled 'fix(qwen3_5): restore the out_hidden_size fallback the merge delta lost': restored the fallback line plus upstream's comment; the PR's norm_eps float read stays. test_qwen3_5_moe_vision 7/7 and test_qwen3_5_dense_vision 9/9 pass locally (Release, x86); full ctest 816/829 with the 13 failures byte-identical to pristine upstream/main at 0fbd7c994.
