ID: ISSUE-LOCAL-01M7XH1MTYH6QQHDKT31WGB76Y
Title: GLM4 #2839 forward repair: drive the SACRED gate from 69/128 to 128/128 against the committed vLLM goldens (snapshot now local, T3 runnable on CPU)
Row: MODEL-TEXT-GLM4-MOE-LITE-GATE-2839
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: -

## Problem

The committed-goldens STRICT case (T2) is red at 69/128 and the live SACRED case (T3) reproduces the same divergence on this host (prompt 0 tok 14: ours 9851 vs vLLM 279). The forward was 8/8 at d85fd04fb, so this is a regression to locate and repair, not a bar to re-derive — exit 1 of the two licensed exits, no weakening of T2. New since the spec: the pinned snapshot zai-org/GLM-4.7-Flash @ 7dd20894a642a0aa287e9827cb1a1f7f91386b67 (62.5 GB, MIT) is downloaded in the local HF cache (developer-authorized), so T3 runs on this host CPU (~5 min per battery) via test_glm4_moe_lite_paged_engine --test-case=*SACRED*; iterate with VT_GLM_DUMP_IDS=1 + scripts/glm4-moe-lite-neartie-gap.py (CPU numerics drift from the dgx-captured our_ids, so the anchor REQUIRE at :387 fails locally — host artifact, not the bug). Divergence shape: prompt 7 exact, prompt 0 diverges at pos 14, prompts 1-6 at pos 4-6 then cascade; neartie_gap_mnats.npy all-zero — a real logits difference, not a tie. Prime suspect: 48a05674b (GLM router fp32) and other commits in d85fd04fb..HEAD touching the glm4/shared path. Plan: reproduce live on CPU, bisect d85fd04fb..HEAD on the failing cells, repair, drive T2/T3 to 128/128, record pinned weights (repo+rev+sha256) in docs/USAGE.md, close #2839.

## Resolution

-
