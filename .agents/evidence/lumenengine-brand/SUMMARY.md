# LumenEngine compatible branding evidence

Row: `ENG-LUMENENGINE-BRAND`
Issue: `ISSUE-LOCAL-01M63H60ES6WTJFPTMQBVSZWXY`
Date: 29 October 2026
Spec-first commit: `8408d5035`

## Environment

Local Linux x86_64, GCC 15.2.0, CMake 4.2.3. CPU-only packaging gate, no model,
GPU, remote service, or oracle workload. Build parallelism is at most 4 jobs.
The gate selects Release with `-O0 -DNDEBUG` because it measures compatibility,
not throughput. Diarization and Hugging Face downloads are disabled to avoid
external dependency downloads. The vendored HTTP server remains enabled.

## Red before implementation

- `cc -std=c11 -I include -fsyntax-only tests/branding/consumer.c` exits 1 because
  `lumenengine.h` does not exist. See `red-header.log`.
- `cmake -S tests/branding -B /tmp/lumen-brand-build -DENGINE_SOURCE=/tmp/lumen-brand -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_WITH_DIARIZATION=OFF -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_HF_DOWNLOAD=OFF -DCMAKE_BUILD_TYPE=Release`
  exits 1 because both LumenEngine targets are missing. See `red-targets.log`.
- The first configuration with examples enabled also exposed existing
  `CMAKE_SOURCE_DIR` assumptions. Project-relative example paths repair this
  external-consumer failure without changing standalone paths.
- After building the original production commands,
  `python3 tests/branding/check_commands.py /tmp/lumen-brand-build/engine/examples`
  exits 1 for the missing `lumenengine-cli` alias. See `red-command-alias.log`.
- After building aliases, the same command exits 1 for missing product identity.
  All 3 real `--help` commands exit 0 and lack `LumenEngine` before the help edits.
  See `red-help.log` and `red-all-help.log`.
- `python3 scripts/check-surface-coverage.py` exits 1 after adding the installed
  forwarding header, because its old inventory pins only `vllm.h`.
  See `red-surface-pin.log`. Only the inventory and fixtures change.

## Focused and full CPU gate

`python3 tests/branding/run_gate.py /tmp/lumen-brand-build` exits 0.
See `full-gate.log` for each exact nested command and its results.

- Real `add_subdirectory` consumers compile and run with both headers, the old
  static target, the old shared target, and both new shared target names.
- 7 build-tree CTest cases pass, including the existing `test_dlopen.cpp` ABI
  smoke and existing `VerifyExports.cmake` symbol-export gate.
- A real installation moves to another prefix before `find_package` configures
  external C11 and C++20 consumers. A repeated package lookup also succeeds.
- All 6 installed CTest cases pass. The imported new targets remain shared.
- Old and new CLI, server, and benchmark commands have identical bytes and
  matching help output after normalizing `argv[0]`, in both trees.
- Server machine version output remains byte-identical to
  `legacy-server-version.txt`: `vllm.cpp 0.0.3 c-abi=30`.
- Direct C11 and C++20 syntax checks pass for the forwarding consumer.
- `include/vllm.h` and `cmake/vllm_export.map` remain byte-for-byte unchanged.

Repository checks:

- `python3 tests/scripts/test_check_surface_coverage.py`: 47 tests pass.
- `python3 tests/scripts/test_checker_text.py`: 33 tests pass.
- `python3 scripts/check-test-registration.py`: exits 0.
- `python3 scripts/check-readme-structure.py`: exits 0.
- `python3 scripts/check-quickstart-recipes.py`: exits 0.
- `python3 scripts/check-build-runtime-deps.py`: exits 0.
- `python3 scripts/check-env-doc.py`: exits 0.
- `scripts/agent-preflight.sh --staged --quiet`: output is recorded in
  `/tmp/lumen-brand-preflight-final.log`. It is still running at the implementation
  commit. Its gate remains pending; the handoff report records its eventual exit.
  This is not a claim that the entire repository preflight is green.

## Routine CTest registration

`tests/CMakeLists.txt` registers `test_lumenengine_header_cpp` and
`test_lumenengine_header_c` against the existing shared library. When the examples
and server are enabled, `examples/CMakeLists.txt` registers
`test_lumenengine_commands`. These tests use the already-built engine and do not
start a recursive build. The external gate retains install and relocation checks.

With `VLLM_CPP_BUILD_TESTS=ON`, both header tests and the command test pass.
See `root-tests.log`. Enabling tests in the external-consumer build exposed one
existing absolute-root Vulkan-loader test source path. Changing it to
`PROJECT_SOURCE_DIR` preserves the standalone path and permits configuration.

## Mutations

The focused checks fail for each deliberate defect and pass after restoration.
See `mutation-header.log`, `mutation-build-targets.log`,
`mutation-installed.log`, and `mutation-surface-pin.log`.

- Remove the forwarding include in a scratch header and compile a real C caller.
- Delete either build-tree LumenEngine alias in a scratch source root.
- Redirect either build-tree LumenEngine alias to the static library.
- Delete either installed public header.
- Delete either installed LumenEngine target alias from the package config.
- Delete each installed command alias.
- Replace the help identity in both spellings of each command. This passes the
  binary-parity assertion and fails the independent help-identity assertion.
- Revert the surface inventory to either single-header set.

Every mutated artifact is restored byte-for-byte. Scratch source roots do not
modify production sources. Installed mutation artifacts use a separate prefix.
The restored installed command gate exits 0.

## Limits and handoff

Native Windows and macOS execution is unverified. The implementation uses CMake
copies rather than symlinks. No model behavior or performance claim is made.
The existing server-only release component remains unchanged. General install
adds the new command names and the library package.

Fresh review and the operator's independent gate remain required before landing.
The issue remains open because no merge or remote publication is authorized.

## Proven environment baseline failure

`test_rocprof_attach_preflight.ControllerPreflight.test_reservation_allocation_metadata_is_refused`
fails for `invalid-page`. The restricted child Python 3.14 process raises
`MemoryError` while importing `ctypes`, before it can emit the expected
`invalid system page size` diagnostic. The branding tree and an exact-spec-base
scratch fixture both exit 1 for this case. The relevant test, production script,
patch, and upstream fixtures have no diff from `8408d5035`.

Commands and raw output are in `rocprof-current.log` and `rocprof-baseline.log`.
The operator independently reproduced the same failure at clean base
`db1d71c9b7352a07983b408b2e755e853b4534f0`.
This baseline remains failing. No assertion or fixture restriction is weakened.

## Inventory reconciliation

The owning engine inventory and `CLAIM-ENG-LUMENENGINE-BRAND` record `ACTIVE`.
The spec records baseline, port map, dependencies, and pending review.
All unrelated engine inventory bytes remain unchanged. Only the directly
invalidated subsection and total summary cells change with the new row.
`python3 scripts/check-agent-record.py` exits 0. The 106-test record suite passes.
See `green-record-check.log` and `green-record-suite.log`.

The running staged preflight observed the briefly incomplete new inventory row
before its structured spec and claim were added. Its record-suite red is repaired
and the full affected suite is rerun green. This is separate from the unchanged
ROCm environment baseline failure.

## Final resumed verification

The final package gate selects `^branding_` tests explicitly. Reusing its build
folder after the routine root-test check exposed stale CTest files when switching
`VLLM_CPP_BUILD_TESTS` back off. The unfiltered run discovered unrelated, unbuilt
engine tests. This test selection keeps all seven declared build-tree cases and
all six installed cases; the final run passes each case. No product behavior or
assertion changes for this harness repair.

`python3 scripts/check-tree-compiles.py --build-dir /tmp/lumen-brand-treecheck --jobs 4`
exits 0: all five affected host translation units compile. The build directory
was configured with CUDA, optional diarization, and Hugging Face downloads off.
See `tree-compiles-cpu.log`. The default preflight compile configuration detects
CUDA 12.4 and requests the existing default `compute_121a`, which that toolkit
cannot compile. This environment configuration remains a failing default gate;
the CPU package build and explicit host syntax gate pass.

The final routine root CTest rerun passes all three header/command cases. The
final surface and checker-text suites pass 47 and 33 tests. All nine recorded
artifact hashes still match the final build. The staged preflight is pending
while its mutation suites finish; no passing preflight is claimed.
