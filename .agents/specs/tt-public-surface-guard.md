# TT public surface guard — every `vt::tenstorrent` symbol callable from an always-compiled TU must live in the public header with a non-TT stub

Row: `BACKEND-TENSTORRENT`. Issue: ISSUE-LOCAL-01M7W2ZKR8HRBYJF6HCPP46AGH.

## Why

`VLLM_CPP_TENSTORRENT` defaults to OFF, and the default configure compiles
almost the whole tree: every file under `src/` except `src/vt/tenstorrent/**`
and the sources added inside `if(VLLM_CPP_TENSTORRENT)` in the top-level
CMakeLists (`src/vllm/platforms/tenstorrent.cpp`, the three
`src/vt/tenstorrent/` TUs, the ops object library), plus every test registered
OUTSIDE the TT blocks in `tests/CMakeLists.txt` (609 of 622 today — the
`*_tt` tests are registered inside `if(VLLM_CPP_TENSTORRENT)`, but
`test_qwen38_dflash2_drafter_tt` and the parity suites are not).

The established pattern that keeps those TUs linkable is the public header
`src/vt/tenstorrent/tenstorrent_device.h` (ttnn-free) declaring each symbol
under `#ifdef VLLM_CPP_TENSTORRENT` with an inline no-op in the `#else` branch
(the "WarmPagedKvShadow pattern", tenstorrent_device.h:142-166). Device-agnostic
TUs may then call `vt::tenstorrent::X` unguarded — a runtime `DeviceType` check
gates the behavior, and the stub satisfies the linker.

The litter this spec retires is every symbol that SKIPPED the pattern:

1. `vt::tenstorrent::EnsureHost` is declared only in the INTERNAL header
   (`tenstorrent_internal.h:412-413`, which pulls ttnn headers) and defined
   only in `tenstorrent_residency.cpp` (TT-only). The drafter test, which must
   compile without ttnn, forward-declared it by hand
   (`tests/vllm/models/test_qwen38_dflash2_drafter_tt.cpp:74-77`) and calls it
   at :787 — so every CPU-only lane that links that always-registered test
   dies with `undefined reference to vt::tenstorrent::EnsureHost(Tensor&)`
   (observed red on `build-newest-gcc`).
2. The public header itself declares nine symbols UNCONDITIONALLY whose
   definitions exist only in TT-only TUs: `RegisterHostBuffer`,
   `UnregisterHostBuffer`, `MarkHostWritten`, `MarkScratchAcquired`,
   `EnsureHostBytes`, `CopyDeviceDeviceIfCapture`, `MemsetDeviceIfCapture`,
   `MemsetDeviceFill`, `CopyDeviceDeviceIfResident`
   (tenstorrent_device.h:167-207). Safe today only because no always-compiled
   TU references them; one reference and the same link break returns.

## Scope

In scope:

- `EnsureHost(Tensor&)` / `EnsureHost(const Tensor&)` move from
  `tenstorrent_internal.h` to `tenstorrent_device.h`, declared under
  `#ifdef VLLM_CPP_TENSTORRENT` with `#else` inline no-ops
  (`inline void EnsureHost(Tensor&) {}` / `inline void EnsureHost(const Tensor&) {}`).
  `tenstorrent_internal.h` includes the public header, so TT TUs see the same
  declarations unchanged.
- The nine unconditional declarations in the public header become
  `#ifdef`/`#else` pairs with no-op stubs (`void f(...) {}` /
  `bool f(...) { return false; }` per existing stub semantics —
  `ConvShadowServeable` already stubs to `true`, the "serveable/absent"
  polarity; for these nine the safe non-TT value is the no-op/false one,
  matching what the functions do when no device exists: nothing).
- The drafter test drops its hand-rolled `namespace vt::tenstorrent { ... }`
  forward-declaration block and includes
  `"vt/tenstorrent/tenstorrent_device.h"` instead (the header is ttnn-free by
  design; qwen3.cpp / qwen3_5.cpp already include it in every build).
- New checker `scripts/check-tt-public-surface.py` (picked up automatically by
  the `agent-preflight.sh` discovered sweep of `scripts/check-*.py` and by CI's
  checker block) enforcing, over the always-compiled TU set:
  - R1: no unguarded `#include` of any header under `src/vt/tenstorrent/`
    other than `tenstorrent_device.h` (internal headers pull ttnn);
  - R2: no unguarded re-opening of `namespace vt::tenstorrent` to declare
    symbols (forward-declaration litter — declare in the public header with a
    stub instead);
  - R3: every unguarded `vt::tenstorrent::X` reference (and
    `using vt::tenstorrent::X;` / `using namespace vt::tenstorrent;`) has a
    non-TT-branch declaration in `tenstorrent_device.h`;
  - R4 (header self-consistency): every symbol declared in the TT-positive
    branch of `tenstorrent_device.h` has a matching `#else` stub, and no
    symbol is declared unconditionally without an unconditional definition.
- Unit test `tests/scripts/test_check_tt_public_surface.py` with a synthetic
  fixture tree: red when a gap exists (unguarded ref without stub, internal
  include, namespace re-opening, unconditional dangler), green when clean.

Out of scope: guarding the 98 already-unguarded-but-stubbed call sites in
qwen3.cpp / qwen3_5.cpp / model_loader.cpp / the parity tests (the stub pattern
makes them legal; churning them buys nothing), the TT-only TUs
(`src/vt/tenstorrent/**`, `test_tenstorrent_backend.cpp`, the 13 TT-block
tests), and any numeric behavior.

## Design

The checker is pure Python, no build, no GPU, mirroring the other
`scripts/check-*.py` gates:

- Always-compiled set: walk `src/`, `include/`, `examples/` excluding
  `src/vt/tenstorrent/` and the files registered inside exact-condition
  `if(VLLM_CPP_TENSTORRENT)` blocks of the top-level `CMakeLists.txt`; plus the
  test sources registered outside TT blocks in `tests/CMakeLists.txt`
  (parsed with a small CMake conditional scanner, condition string compared
  exactly so `if(VLLM_CPP_TENSTORRENT STREQUAL "AUTO")` and `else()` branches
  are not misread).
- Guard tracking per C++ file: a stack over `#if/#ifdef/#ifndef/#elif/#else/#endif`
  where a frame is TT-positive iff its expression references
  `VLLM_CPP_TENSTORRENT` positively (`#ifdef X`, `#if defined(X)`,
  `#if !defined(...)` handled with negation). A reference is "unguarded"
  iff no enclosing frame is TT-positive in the current branch. Comments and
  string literals are stripped before matching (same discipline as
  check-device-leakage.py).
- Public-header parse: same guard tracking over `tenstorrent_device.h`;
  collect (a) symbols with declarations in non-TT branches (the stub surface
  — used by R3) and (b) all declared symbols for R4 (TT-branch decl without
  `#else` stub, or unconditional decl, is a violation).
- Output: one `file:line: symbol` per violation plus the cure ("add an #else
  stub in src/vt/tenstorrent/tenstorrent_device.h, or guard the call site with
  #ifdef VLLM_CPP_TENSTORRENT"); exit 1 on any violation, 0 otherwise, `--report`
  for the full list.

Red-before / green-after: the checker run on the pre-fix tree must exit 1 and
name `test_qwen38_dflash2_drafter_tt.cpp:787` (`EnsureHost`) plus the nine
header danglers; after the fix it exits 0. The link proof is building
`test_qwen38_dflash2_drafter_tt` in a CPU-only (`VLLM_CPP_TENSTORRENT=OFF`)
build directory.

## Gates

- `python3 scripts/check-tt-public-surface.py` exits 0 on the fixed tree;
  exits 1 on the pre-fix tree (red-before evidence in the commit body).
- `python3 -m unittest tests/scripts/test_check_tt_public_surface.py` green
  (synthetic red fixtures + green fixtures, both directions).
- `cmake --build <cpu-build-dir> --target test_qwen38_dflash2_drafter_tt -j8`
  links in `/tmp/build-parsers-review` (`VLLM_CPP_TENSTORRENT=OFF`).
- `scripts/agent-preflight.sh --staged` green (the discovered sweep picks the
  new checker up).
- A TT build still compiles: the moved declarations are unchanged for
  `VLLM_CPP_TENSTORRENT=ON` (tenstorrent_internal.h includes the public
  header), verified by a syntax-only pass over `src/vt/tenstorrent/` via the
  existing TT build dir if available, else by inspection + the header being
  include-order independent.

## Risks

- Moving `EnsureHost` into the public header changes its declaration context:
  `Tensor` there resolves to `vt::Tensor` via the existing
  `namespace vt { struct Tensor; }` forward block — signatures must match the
  internal header's exactly (they do: `void EnsureHost(Tensor& t);`).
- The nine no-op stubs change CPU-build link behavior from "undefined if
  referenced" to "defined no-op" — strictly safer; no CPU code path calls
  them (the TT backend registers only in TT builds), and if one ever does, the
  no-op is the correct semantics (no device shadow exists without a device).
- The CMake scanners are text heuristics, like every other checker here: a
  `target_sources` call split across lines with the file path on the next line
  would be missed. Mitigation: the TT block today is a single contiguous
  `target_sources(...)` call; the checker's docstring says so, and R3's
  stub-surface rule is the load-bearing half (a missed TT-only file only
  over-scans, and over-scanning a file with no TT refs is a no-op).

## Stop conditions

- If the checker flags a site that is genuinely TT-only but registered
  unconditionally by mistake, fix the REGISTRATION (move the test into the
  `if(VLLM_CPP_TENSTORRENT)` block) rather than adding a stub — a stub is for
  symbols the device-agnostic layer legitimately calls behind a runtime check.
- If R4 flags a public symbol whose non-TT no-op semantics are not obvious,
  stop and ask rather than guessing the stub's return value.

## Now

Filed 2026-11-20 from the red `build-newest-gcc` lane on main (14e06b7bc).
Landed on branch `fix/ensurehost-cpu-link` (pushed to `mine`): spec+issue
`a5b7b18fa`, fix `7880f6528`, issue closure `33ef6ad8a`. Checker green
(0 violations / 96 stubbed symbols), unit tests 9/9, CPU-only link+run of
the drafter test proven in `/tmp/build-tt-surface`. Awaiting merge; the red
lane on main (build-newest-gcc) goes green once it lands.

## Owed

- None.

## Git integration

One pull request (default; no recorded split preference for this row).
