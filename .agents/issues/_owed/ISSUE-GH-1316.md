ID: ISSUE-GH-1316
Title: `scripts/main-baseline.py` renders a scheduled run that executed ZERO jobs as `RED` with all 11 covered jobs `missing`, so `NEWEST BASELINE: RED at <sha>` names a tree the run never checked out. Measured at `origin/main` `250db75a2`: runs `32206456661` and `32140419182` both return `startedAt: null` for every job, because GitHub cancelled them while they were pending in the single `ci-schedule-refs/heads/main-mudler/vllm.cpp` group, whose queue holds one run ([#274](https://github.com/mudler/vllm.cpp/issues/274)). Fail-closed, and the `missing (expected, never ran)` line is accurate about the jobs; the defect is the verdict word, because a run that executed nothing is NOT RUN rather than RED, and the newest verdict should fall through to the newest run that actually ran. NOT fixed in flow: `test_an_expected_job_the_payload_never_mentions_is_red` and `test_a_narrowed_run_reports_red_and_names_what_never_ran` deliberately assert missing-is-red so a narrowed run cannot pass, and separating "narrowed" from "never started" changes what the verdict means, which owes its own spec, red-before evidence and a fresh reviewer. Owed under `## Owed` of [baseline-lane-eviction.md](../../specs/baseline-lane-eviction.md), which removes the only observed producer of a zero-job run
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1316
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:435`

### Frozen archive evidence

> | [#1316](https://github.com/mudler/vllm.cpp/issues/1316) | — | `scripts/main-baseline.py` renders a scheduled run that executed ZERO jobs as `RED` with all 11 covered jobs `missing`, so `NEWEST BASELINE: RED at <sha>` names a tree the run never checked out. Measured at `origin/main` `250db75a2`: runs `32206456661` and `32140419182` both return `startedAt: null` for every job, because GitHub cancelled them while they were pending in the single `ci-schedule-refs/heads/main-mudler/vllm.cpp` group, whose queue holds one run ([#274](https://github.com/mudler/vllm.cpp/issues/274)). Fail-closed, and the `missing (expected, never ran)` line is accurate about the jobs; the defect is the verdict word, because a run that executed nothing is NOT RUN rather than RED, and the newest verdict should fall through to the newest run that actually ran. NOT fixed in flow: `test_an_expected_job_the_payload_never_mentions_is_red` and `test_a_narrowed_run_reports_red_and_names_what_never_ran` deliberately assert missing-is-red so a narrowed run cannot pass, and separating "narrowed" from "never started" changes what the verdict means, which owes its own spec, red-before evidence and a fresh reviewer. Owed under `## Owed` of [baseline-lane-eviction.md](../../specs/baseline-lane-eviction.md), which removes the only observed producer of a zero-job run | bug |

## Resolution

-
