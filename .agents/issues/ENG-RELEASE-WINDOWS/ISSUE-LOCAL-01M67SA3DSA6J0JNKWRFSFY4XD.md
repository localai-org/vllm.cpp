ID: ISSUE-LOCAL-01M67SA3DSA6J0JNKWRFSFY4XD
Title: windows-msvc-vulkan is RED on every PR: flint.cpp shadows 'v' and /WX makes it an error
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: 2026-10-30

## Problem

The Windows Vulkan job fails to compile on every pull request:

```
src/vllm/model_executor/models/flint.cpp(529,18): warning C4456: declaration of 'v' hides previous local declaration
src/vllm/model_executor/models/flint.cpp(653,18): warning C4456: declaration of 'v' hides previous local declaration
src/vllm/model_executor/models/flint.cpp(529,18): error C2220: the following warning is treated as an error
```

Both diagnostics cite the outer declaration at `flint.cpp(402,24)`, the
attention value buffer `std::vector<float> v(static_cast<size_t>(t) * hkv * dh);`.
The two inner `for (float v : x) s += v;` loops that compute a `cpu_dump`
checksum shadow it. GCC and Clang do not warn on this construct, so it built
green on Linux and landed.

Introduced by `090c536fb` (`feat(flint): the W0 Tenstorrent device arm — dispatch
plus the first device-vs-CPU baseline`). Present at `main` `c6c0d8674`.

## Measured evidence

2026-10-30, run [41145791168](https://github.com/localai-org/vllm.cpp/actions/runs/41145791168),
job [124370468409](https://github.com/localai-org/vllm.cpp/actions/runs/41145791168/job/124370468409),
observed on pull request #3529, whose diff touches no C++, no CMake and no
PowerShell:

- `windows-msvc-vulkan` — `fail`, 34m46s. The failing step is `Build and execute
  the native Windows Vulkan focused gate`, which runs
  `scripts/build-windows-release.ps1 -Backend vulkan`; it ends with
  `##[error]Process completed with exit code 1`. The preceding
  `-ContractTest` step passed (`Windows PowerShell/CRT contract tests OK`), so
  the lane's own contract tests are sound and this is a pure compile break.
- The same job failed on six unrelated branches opened before #3529 —
  `row/ENG-RECORD-CONFLICT-SURFACES-CITE-BUCKET-2` (#3367),
  `row/ENG-RECORD-CONFLICT-SURFACES-INTAKE-TRIAGE` (#3474),
  `row/POLICY-ISSUE-INTAKE`, `row/GATE-ISSUE-ARCHIVE-RESTORE`,
  `row/BACKEND-VULKAN-VKI-REBAR` and `row/ORACLE-PIN-SURFACES-rebase` — and has
  no recorded success in the last 14 `pull_request` runs sampled.
- `main` never contradicts it because `main` never runs the lane: the job's
  `if:` admits only `pull_request`, `schedule` and `workflow_dispatch`.

## Why this is a repeat

Same defect class as ISSUE-GH-2404, where `glm5_next_weights.cpp` shadowed `v`
and `/WX` turned C4456 into C2220, fixed by `5f230020f`. Same cost: a PR-only
gate with no `main` baseline converts a landed Windows break into a red charged
to whichever innocent branch runs CI next.

## Fix

Rename the two loop variables, behaviour-preserving. They are now `xv` rather
than the `v_x` named when this record was filed: `xv` is the spelling this file
already uses for an element of `x` (`const float xv = normed[...]` in the router
and lm_head loops), so the three checksum loops read as one idiom.

The repair rides with this record in one change, because it was filed from the
run log by the session that could reproduce it under `cl`.

## Resolution

2026-10-30: fixed. The two `cpu_dump` checksum loops at `flint.cpp` 529 and 653 renamed their induction variable from `v` to `xv`, so neither hides the attention value buffer declared at 402. The loop `v` at 385 is outside the layer scope and was already unambiguous, so it is untouched.

Measured with the lane's own warning settings, `cl /Zs /W4 /WX /permissive-` (VS 2022 17.x, MSVC 14.44.35207), against both trees:

```
BEFORE (main c6c0d8674 -> this record's HEAD)
flint.cpp(529): warning C4456: declaration of 'v' hides previous local declaration
flint.cpp(402): note: see declaration of 'v'
flint.cpp(653): warning C4456: declaration of 'v' hides previous local declaration
flint.cpp(529): error C2220: the following warning is treated as an error
rc=2

AFTER (this change)
rc=0, no diagnostics
```

The rename is behaviour-preserving: both sites are float sums of `x` for a stderr checksum under `cpu_dump`.
