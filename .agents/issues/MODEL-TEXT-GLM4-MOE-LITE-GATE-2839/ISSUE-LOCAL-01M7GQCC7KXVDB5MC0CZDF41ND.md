ID: ISSUE-LOCAL-01M7GQCC7KXVDB5MC0CZDF41ND
Title: The teacher-forced gap is stored in integer millinats, so a tie and a 0.0004-nat deficit are the same number
Row: MODEL-TEXT-GLM4-MOE-LITE-GATE-2839
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

scripts/glm4-moe-lite-neartie-gap.py computes gap = max(0, argmax_logprob - our_logprob) and stores int(round(gap * 1000)) in neartie_gap_mnats.npy, so every gap below half a millinat is committed as 0. At the seven first-divergence positions of the GLM-4.7-Flash goldens the prefix is the oracle's, so its teacher-forced argmax must be its own greedy token and the gap must be positive; the committed array says 0 at all 128 cells, which cannot distinguish 'the oracle agrees with our token' from 'the oracle prefers its own token by 0.0004 nats'. K1 needs one number at full resolution. The change stores the raw float64 gap in nats plus the teacher-forced argmax id beside the existing millinat array, so the committed artifact keeps its current meaning and the decisive comparison becomes readable.

## Resolution

K1 landed at 13542b40a + this commit. scripts/glm4-moe-lite-neartie-gap.py now keeps neartie_gap_mnats.npy byte-identical in meaning and adds teacher_forced_gap_nats.npy (float64 nats, unrounded) and teacher_forced_argmax.npy (int32), plus a printed count of positions whose positive gap the millinat array commits as 0 and of positions where the oracle's teacher-forced argmax is our token. gap_rows is pure and unit-tested in tests/scripts/test_glm4_neartie_gap.py: red before, 6 of 6 failing with no attribute gap_rows; green after, 6 of 6 passing, including 0.0004 nats stored as 0 millinats and 0.0004 nats. Registered in scripts/agent-preflight.sh and .github/workflows/ci.yml behind the existing numpy guard; check-test-registration.py green. No golden was regenerated and no assertion moved: the next capture is what produces the full-resolution artifact.
