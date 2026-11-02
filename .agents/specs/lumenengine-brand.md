# LumenEngine compatible branding

Row: `ENG-LUMENENGINE-BRAND`
Issue: `ISSUE-LOCAL-01M63H60ES6WTJFPTMQBVSZWXY`

## Now

ACTIVE. CPU packaging implementation is verified on 29 October 2026. Fresh review and the operator gate remain pending. User approved compatible branding on 29 October 2026. Base: `db1d71c9b7352a07983b408b2e755e853b4534f0`.

## Scope

Rename maintained product branding to `LumenEngine`. Add new public names as aliases where useful. Preserve every existing public API, ABI, command, configuration key, build option, library filename, target, and header. Existing integrations must continue to work.

## Upstream chain

This is original packaging and branding, not an upstream behavior port. Preserve vLLM names for upstream comparisons, source attribution, and pinned oracle records. Preserve URLs and historical evidence. Public API anchors: `include/vllm.h`, `CMakeLists.txt:1513` for `vllm::vllm`, `CMakeLists.txt:3127` for the shared library, and `CMakeLists.txt:3208` for install exports. Inspect CLI and server entry points before choosing new aliases.

## Design and port map

Keep the engine implementation and existing symbols. Update maintained README, build/usage/API documentation, product help and server identity text where user-facing. Introduce a forwarding `include/lumenengine.h` including `vllm.h`, CMake aliases `LumenEngine::LumenEngine` and `LumenEngine::shared` for the shared C ABI library in both build and install trees, and CLI aliases for existing shipped tools. Installed and build-tree consumers must both work. Do not rename CMake project variables or existing artifacts to simulate a rebrand. New names must reach the same implementations, rather than duplicate code.

Keep existing version output and machine-consumed response formats compatible. Human-readable product descriptions may include the new name. Do not blindly replace `vllm` in model identifiers, upstream paths, test fixtures, configuration namespaces, function names, or source history. Do not erase original authorship or licenses. Avoid mass policy-document edits.

## Tests to port

No upstream tests apply to original packaging. Add tests invoking the actual production build/install and executable entry points. Prove old and new headers compile in C and C++; old and new CMake targets link; installed headers and targets resolve; old executables remain and new aliases execute the same behavior. Existing public symbols and struct layouts must not change.

## Gates

Full scoped gate: CPU configure/build of the shipped CLI and server when dependencies exist, CMake consumer build/install tests for both names, C/C++ forwarding-header compile tests, CLI help/version comparisons, existing ABI smoke tests, documentation/build contract checkers, and `scripts/agent-preflight.sh --quiet`. Use at most four build jobs. Record red-before, focused-green, full-gate results, mutation failures, and any unchanged baseline failures. Delete the production alias/install call in reviewer scratch and require its regression gate to fail.

## Work breakdown

1. Add failing real-consumer and command tests.
2. Add compatible aliases and maintained product branding.
3. Verify build-tree and installed consumers, document compatibility, and obtain fresh mutation review.

## Risks and stop conditions

Do not change ABI versions or layouts for branding. Existing output consumed by scripts must remain compatible. If an alias cannot be portable across supported operating systems, use an existing build-system mechanism or record the exact limitation; do not silently replace an existing command. Stop on changes outside this scope. Leave external publication pending; no push or merge is authorized.

## Integration

Use one PR with this spec committed before implementation. Issue and exact evidence links are added before implementation begins. Evidence belongs in `.agents/evidence/lumenengine-brand/`.

## Packaging decision

Both LumenEngine targets select the shared C ABI library. This keeps build-tree and installed consumers consistent without exporting private static backend dependencies. Existing `vllm::vllm` remains the static build-tree target. A new `find_package(LumenEngine CONFIG)` package exposes the shared aliases and `vllm::shared`. The package preserves a target already defined by its caller.

## Review repairs

Independent review identified two test blind spots. Compile each public header
in a separate C11 and C++20 caller. Pin legacy server version bytes independently
of alias equality. See [repair evidence](../evidence/lumenengine-brand/review-repair.md).
The repairs are tracked by `ISSUE-LOCAL-01M63MYTWBYSRRCEMAY20QR08M`.

## Evidence

See [the CPU packaging report](../evidence/lumenengine-brand/SUMMARY.md) for red-before tests, installed consumer results, command compatibility, mutations, and external limits. The issue remains open until landing.

## Our baseline

At `8408d5035`, `include/vllm.h` is the installed C ABI header and the only
pinned public header. CMake defines `vllm::vllm` and `vllm::shared` for build
consumers. It installs the libraries without a CMake package. The CLI, server,
and benchmark retain their `vllm-` names and help identity.
The red-before tests in the evidence report exercise these missing new names.

## Port map

| Surface | Local anchor | Change |
|---|---|---|
| Public header | `include/lumenengine.h` | Forward to unchanged `vllm.h` |
| Build targets | `CMakeLists.txt` | Add shared C ABI aliases |
| Installed targets | `cmake/LumenEngineConfig.cmake.in` | Add a relocatable package |
| Command names | `examples/CMakeLists.txt` | Copy the existing executables |
| Human help | CLI, benchmark, and server usage functions | Name LumenEngine |
| Public header inventory | `scripts/check-surface-coverage.py` | Pin both installed headers |

## Dependencies

Use the existing C++20 compiler, C11 compiler, CMake, vendored headers, and
Python standard library. This packaging work adds no runtime dependency.
CPU verification disables optional diarization and network downloads.
Native Windows and macOS runtime checks remain unavailable in this environment.
