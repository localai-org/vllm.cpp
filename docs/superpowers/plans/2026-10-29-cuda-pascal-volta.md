# Pascal and Volta CUDA enablement implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Implement the approved pascal and volta cuda enablement design.

**Architecture:** Keep existing engine entry points and public interfaces. Add capability-gated implementation or additive aliases behind those entry points. Commit evidence and receive independent mutation review.

**Tech Stack:** C++20, CMake, CUDA 12.4 where applicable, Python test harnesses.

**Spec:** `.agents/specs/pascal-volta-enable.md`

## Global constraints

- Preserve existing API, ABI, commands, configuration keys, build options, and newer GPU behavior.
- Follow the committed spec and `AGENTS.md`; do not invent oracle paths or preferences.
- No push, merge, service changes, or large asset downloads without existing authorization.
- Build with at most four jobs and claim the required GPU exclusion before execution.
- Full gate failures need a precise baseline comparison, never weakened checkers.

## Review focus

- Existing consumers must compile and link after additive changes.
- Capability dispatch must not execute unsupported instructions or silently use CPU fallback.
- Installed consumers must receive every advertised new header and target.
- Actual commands and model entry points must reach the implementation, beyond unit-level construction.
- Unsupported arms and hardware-unverified results must be explicit.

### Task 1: Implement and verify pascal and volta cuda enablement

**Files:**
- Modify only the owning paths required by the spec: cmake/CudaArchFeatures.cmake, src/vt/cuda/, tests/vt/, tests/CMakeLists.txt, docs/BUILD.md, docs/FEATURES.md, docs/USAGE.md.
- Add focused production-entry-point tests and evidence under `.agents/evidence/pascal-volta-enable/`.
- Update the owning issue, spec, and exact matrix rows when their records change.

**Interfaces:**
- Consumes the existing public C API and CMake build/install entry points.
- Produces the same public behavior plus the additive capability described in the spec.

- [ ] Read the owning spec, current source, test harness, and matching pinned source where applicable.
- [ ] Declare the exact test cases and write the smallest failing production-entry-point regression.
- [ ] Run it and capture the intended red result before implementation.
- [ ] Implement the minimum complete change within the spec and preserve existing paths.
- [ ] Run focused gates, remove each claimed guarantee in scratch, and record failures and byte-for-byte restoration.
- [ ] Run the spec's full scoped gate and repository preflight; distinguish proven baseline failures.
- [ ] Commit implementation and dated evidence with the canonical issue reference and required trailers.
- [ ] Return immutable head, changed files, red/green commands, mutation evidence, remaining external blockers, and scoped gate recipe for independent review.
