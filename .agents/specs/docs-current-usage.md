# Public documentation refresh

## Scope

Correct the public documentation for C ABI 30 and the landed Kolibri-1 CPU
path. This is a documentation-only audit. It changes no model lifecycle state,
implementation, default, or benchmark result.

## Sources and design

| Surface | Binding sources | Planned correction |
|---|---|---|
| C ABI | `include/vllm.h`, `src/vllm/c_api.cpp`, audio entrypoints and tests | Current version, diarization calls, ownership, and limits |
| Kolibri-1 | Model registration and loader, CLI parser, `.agents/specs/kolibri-1-cpu.md`, `.agents/specs/kolibri-tt.md` | Readable model recipe with checkpoint pin, CPU usage, evidence, and accelerator limits |
| Public overview | The sources above | Concise news and links in README, feature table, usage guide, and model index |

Read the actual sources before writing each claim. Preserve historical evidence
in existing specs. Do not publish a new speed comparison. Do not download model
weights or run GPU work. The existing model specs own upstream behavior and
performance gates. This audit does not port behavior or tests.

## Verification

Run the README structure checker and its mutation tests, supported-models
checker, benchmark-index checker, record checker, and `git diff --check`.
Check changed relative links and each documented identifier against source.
Request independent review of the immutable documentation commit. The operator
reruns the focused checks before pushing the exact reviewed SHA to the fork.

Full preflight is attempted. Record missing host tools or baseline failures
separately from documentation validation. Do not weaken a checker.

## Risks and stop conditions

Synthetic tests and CPU token evidence do not establish accelerator parity.
A C ABI declaration does not establish server activation or an HTTP route.
Exclude any claim whose implementation or recorded evidence cannot be found.

## Owed

- ISSUE-LOCAL-01M4F9SK0DPVRCV7E7SYBCV47K owns this documentation correction.
