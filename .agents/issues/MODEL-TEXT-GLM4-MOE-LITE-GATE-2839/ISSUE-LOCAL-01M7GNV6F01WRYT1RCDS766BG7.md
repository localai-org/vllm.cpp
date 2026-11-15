ID: ISSUE-LOCAL-01M7GNV6F01WRYT1RCDS766BG7
Title: The gap array's millinat resolution hides whether the GLM4-MoE-Lite divergences are ties
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

neartie_gap_mnats.npy is int32 millinats, floored at zero by scripts/glm4-moe-lite-neartie-gap.py:92 and rounded at :93. A gap of 0 therefore means 'under half a millinat', not 'exactly tied', and the committed array is 0 in all 128 cells. At each of the seven first-divergence positions the prefix is identical to the oracle's, so vLLM's teacher-forced argmax must be its own greedy token; measured, it is OUR token, and our token sits inside vLLM's top-20 in every case with no 99_999_000 sentinel. That is vLLM's greedy-decode path and its prefill prompt_logprobs path disagreeing at sub-millinat resolution, not evidence that the forward is wrong. The STRICT bar was selected from five greedy runs that share one numeric path, so they cannot detect that disagreement. Recorded so the campaign does not start kernel bisection before testing the oracle's own path determinism.

## Resolution

Recorded on row/MODEL-TEXT-GLM4-MOE-LITE-GATE-2839 at 13542b40a in .agents/specs/glm4-moe-lite-gate-2839.md section '2026-11-12 - the gap array's resolution cannot carry this argument', with O5 naming the two outstanding measurements. No code change: neartie_gap_mnats.npy is int32 millinats floored at zero by scripts/glm4-moe-lite-neartie-gap.py:92 and rounded at :93, so a committed 0 means under half a millinat; the array is 0 in all 128 cells; no 99999000 sentinel exists, so our token was inside the oracle top-20 at all seven first divergences, where the oracle's teacher-forced argmax must be its own greedy token. The measurement therefore reads as vLLM greedy-decode against prefill prompt_logprobs disagreement at sub-millinat resolution, which the K=5 self-determinism experiment cannot see. The STRICT assertion is untouched and T2 stays red.
