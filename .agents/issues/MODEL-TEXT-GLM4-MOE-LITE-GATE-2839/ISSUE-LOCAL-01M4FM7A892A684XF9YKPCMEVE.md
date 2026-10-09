ID: ISSUE-LOCAL-01M4FM7A892A684XF9YKPCMEVE
Title: test_glm4_moe_lite_paged_engine STRICT case is red in EVERY lane by design — classified, not a sanitizer-lane residue (#2839)
Row: MODEL-TEXT-GLM4-MOE-LITE-GATE-2839
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

MEASURED in THREE builds, byte-identical values: the case 'glm4-moe-lite
committed goldens: the oracle is deterministic, so the bar is STRICT' fails
CHECK(exact_positions == N * T) with 69 == 128 and CHECK(exact_prompts == N)
with 1 == 8 (per-prompt mismatch [2,6,6,12,12,11,10,0], 8/10 assertions) — in
the normal Release build (/tmp/build-b, -O3), in a no-build-type (-O0)
NON-sanitizer build (/tmp/build-b-o0), and in the sanitize lane (/tmp/build-san,
address,undefined, CI env with VT_POOL_BYPASS set). The case compares
committed artifacts only
(tests/parity/goldens/glm4_moe_lite_greedy/{greedy_ids,our_ids,greedy_dist,neartie_gap_mnats}.npy)
— it runs no engine and no build-dependent numerics, so it fails identically
in every lane, with the bypass set and unset. CLASSIFICATION: this is NOT a
sanitizer-lane residue and NOT an over-tight tolerance to repair here. It is
the deliberately-red gate landed by f3f8f71d8 ('THIS CHANGE MAKES A GATE FAIL.
That is the result, not a regression.'): .agents/specs/glm4-moe-lite-gate-2839.md
records T2 'Expected RED at 69/128' and O1 'T2 is RED as landed, and it must
stay red until one of the two exits above'. The test's own header comment
(tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp:167-171) states 'THIS
CASE IS EXPECTED TO FAIL, and that is the point. Do not soften it.' The two
admissible exits are a forward repair that makes it 128/128, or a re-capture of
neartie_gap_mnats.npy on the checkpoint plus an explicit ratification of a
distributional bar over a deterministic oracle; BOTH need the 31.2B
zai-org/GLM-4.7-Flash snapshot and a GPU, neither exists on this host, and the
spec's stop condition is 'Stop and report NEEDS_DECISION before weakening T2'
— weakening or skipping the assertion is neither admissible nor this
campaign's work. At f3cd97e3 (2026-09-19) the lane was 100% green only because
the STRICT case did not exist yet (f3f8f71d8 landed after); the case has been
red in every lane since its landing, by design. This issue records the
measurement and the classification so the sanitize-lane residue campaign does
not mistake it for a detector finding, and so the exit remains owned by #2839 /
this row. Action taken: none on the test — the gate stays red everywhere,
exactly as its spec requires. Part of the CI-MAIN-CTEST-RESIDUE sanitize sweep
(group B): this test is on the residue list of
`.agents/specs/ci-main-ctest-residue.md`.

## Resolution

Investigation complete 2026-10-09 on row/ci-sanitize-residue-b; no code change, and none is admissible here (filed in f171ffc44). The STRICT case compares committed artifacts only (tests/parity/goldens/glm4_moe_lite_greedy .npy) and runs no engine, so it fails identically in every lane, re-measured on a clean rebuild in all four trees (Release /tmp/build-b, -O0 /tmp/build-b-o0, address,undefined /tmp/build-san and thread /tmp/build-tsan, the sanitizer trees with the exact CI env): positions 69/128, prompts exact 1/8, per-prompt mismatch [2,6,6,12,12,11,10,0], 8/10 assertions, rc=1, byte-identical everywhere. It is the deliberately-red gate from f3f8f71d8 and .agents/specs/glm4-moe-lite-gate-2839.md (T2 'Expected RED at 69/128'; O1 'T2 is RED as landed, and it must stay red until one of the two exits above'; stop condition 'Stop and report NEEDS_DECISION before weakening T2'; the test's own header: 'THIS CASE IS EXPECTED TO FAIL, and that is the point. Do not soften it.'). The two admissible exits — a forward repair that makes it 128/128, or a re-capture of neartie_gap_mnats.npy on the checkpoint plus an explicit ratification of a distributional bar over a deterministic oracle — both need the 31.2B zai-org/GLM-4.7-Flash snapshot and a GPU, neither available on this host. The gate STAYS RED in every lane, exactly as its spec requires; this issue is closed as classified (not a sanitizer-lane residue, not an over-tight tolerance), and the exit remains owned by #2839 / row MODEL-TEXT-GLM4-MOE-LITE-GATE-2839. Logs /tmp/rb-test_glm4_moe_lite_paged_engine.log, /tmp/ro0-test_glm4_moe_lite_paged_engine.log, /tmp/rsan-test_glm4_moe_lite_paged_engine.log, /tmp/rtsan-test_glm4_moe_lite_paged_engine.log.
