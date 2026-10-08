# Public documentation for ABI 30 audio interfaces

## Now

Documentation audit of `SERVE-C-ABI` at base `d780204dd`.
Issue: ISSUE-LOCAL-01M3JZF0TM67WEV1V63ZTJ00VQ.
The underlying capability lifecycle and benchmark dispositions do not change.

## Scope and source inventory

| Surface | Source | Documentation action |
|---|---|---|
| Current ABI | `include/vllm.h:389` | Replace obsolete current-version statements with 30 |
| Diarization and speaker-attributed ASR | `include/vllm.h:1116`; `src/capi/vllm_c.cpp:1532` | Explain separate handles, PCM format, result cleanup, and disabled-build errors |
| Build dependency | `CMakeLists.txt:1621` | Document default ON, pinned dependency, OFF option, and local-source override |
| HTTP attachment | `include/vllm/entrypoints/openai/api_server.h:286`; `src/vllm/entrypoints/openai/api_server.cpp:1869` | Distinguish conditional callbacks from bundled-server availability |
| ASR handle | `src/capi/vllm_c.cpp:868` | Inspect the actual loader before suggesting a combined-ASR recipe |

The source change is `2833d6300`. This task documents existing local APIs.
It ports no upstream behavior or tests and needs no GPU or model downloads.

## Design and owned files

Update README news and its current ABI statement, docs/BUILD.md,
docs/USAGE.md, docs/reference/c-api.md, docs/reference/server.md, and the
relevant public C API row in docs/FEATURES.md. Keep prose brief and practical.
Document unsupported or unwired paths explicitly. Do not imply runtime parity,
working model artifacts, or benchmark results from header declarations.
Use a short C example only if every call and ownership rule is source-grounded.

## Tests and gates

Read every changed claim against the header, C implementation, build files,
and server wiring. Run the existing README structure and public-document
checks applicable to these pages, plus their mutation suites. Check local link
targets and `git diff --check`. An independent reviewer checks the immutable
implementation commit. The operator repeats focused verification.

Full preflight has host-tool failures at the base. Preserve its log and
report these separately. Do not change checkers, product code, lifecycle
matrices, or benchmark numbers to make this documentation repair pass.

## Risks and stop conditions

A declaration is not proof of runtime availability. In particular, check
whether the bundled server ever installs diarization callbacks and whether
SAS has a runnable ASR loading recipe. Explain limits rather than inventing
flags or verified artifacts. Do not repair implementation defects in this row.

## Git integration

One fork pull request containing the spec commit before the documentation
commit. User authorizes pushing the task branch and opening an upstream PR.
No merge authority, external compute, or GPU execution is granted.

## Outcome

The documentation names ABI 30, result ownership, and loading limits. Source
inspection establishes API availability only. This refresh preserves upstream
news and the decision API documentation. No capability lifecycle or benchmark
result changes.

| Claim | Source evidence |
|---|---|
| Diarization defaults to ON and pins parakeet.cpp to `394d270fabb1d6125f05c772aa3ca078a574b19d` | `CMakeLists.txt:1626-1664` |
| The local override builds its source tree without checking its revision | `CMakeLists.txt:1646-1651` |
| Dependency libraries are static and `GGML_NATIVE` defaults to OFF | `CMakeLists.txt:1632-1645` |
| Separate diarization handle, PCM contract, and result fields | `include/vllm.h:1116-1190` |
| Engine and result cleanup, including disabled-build behavior | `src/capi/vllm_c.cpp:937-940`, `1532-1670`, `1824-1832` |
| ASR directory is passed to both loaders without checking the second result | `src/capi/vllm_c.cpp:895-908` |
| Combined WAV path assumes a 44-byte header and 16 kHz | `src/capi/vllm_c.cpp:1698-1721` |
| Empty dependency results return success without utterances | `src/capi/vllm_c.cpp:1722-1725`, `1788-1791` |
| HTTP registration requires callbacks and the build define | `src/vllm/entrypoints/openai/api_server.cpp:1869-1906` |
| Bundled startup never installs either callback | `rg -n 'set_diarizer|set_sas' src examples include` finds only the two header definitions |

No model, GPU, or benchmark execution is required for this documentation change.
The combined-ASR API has no verified loading recipe in this audit, so the docs
do not offer one. The HTTP routes remain unavailable in the bundled server.
The original draft's unpinned dependency and incomplete local-build warnings
are obsolete: the current CMake code pins and builds both dependency paths.

Focused verification: README structure, surface coverage, quickstart recipes,
and agent record checks PASS. The README mutation suite passes 19 tests, and
the surface coverage suite passes 46 tests. All 18 added local links and their
anchors resolve. `git diff --check` passes. Independent review is pending.

The operator baseline reports unrelated host-tool failures in release suites:
missing `file`, unavailable GNU `getconf`, and a subprocess that removes the
library path needed by the provisioned Python. These remain outside scope.
