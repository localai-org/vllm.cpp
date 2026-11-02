#!/usr/bin/env python3
"""Fail if an example reaches past the public ABI/library surface into internal headers.

The target invariant (developer-directed 2026-08-07): **`examples/*` are THIN CLIENTS
of the public surface, never of internal headers.** A capability must be usable through
the packaged library the way a downstream consumer uses it — the flat C ABI
(`include/vllm.h`, also exposed by the `lumenengine.h` forwarding header),
which `make install` ships (`install(FILES ...)`; the packaged `vllm_shared`
exposes those installed headers and is version-scripted to export only the
`vllm_*` C ABI, `CMakeLists.txt:1712-1730`). An example that instead `#include`s the
internal C++ tree (`vllm/...`, `vt/...`, `src/...`) is coupling to engine internals: it
proves the capability is reachable ONLY by re-implementing against private headers, the
"CLI-only capability" anti-pattern this project keeps rediscovering per model (Laguna
keep-quant decode via `laguna-gen`, DeepSeek-V4 via `deepseek-v4-gen`, the ENTIRE
MiniMax-H3 diffusion pipeline via `minimax-h3-gen`, Kimi-Linear's paged-incremental
decode). The remedy is never to keep a parallel CLI implementation; it is to grow the
public ABI so the example can be a client.

This is the STRUCTURAL, lint-able half of the surface-coverage audit
(`.agents/specs/surface-coverage-2026-08-07.md`). It is deliberately a coarse FLOOR,
like check-fusion-consistency.py / check-runner-routing-consistency.py. It has TWO axes.

Axis 1 — EXAMPLE INCLUDE BOUNDARY (structural): whether each example is a public-surface
client or an internal-reacher, so a new CLI-only model driver cannot land silently.

Axis 2 — C-ABI CAPABILITY REACHABILITY (advertised vs real): every capability the public
FEATURES page advertises must EITHER name a C-ABI entry point that actually exists in
include/vllm.h, OR be an explicit embedder-unreachable row backed by a fold-tracked
allowlist entry. This makes "the flat C ABI is the ONE way an embedder reaches a
capability" a checked property, not a hope.

Three checks, all mutation-testable via pure functions:

(1) **PUBLIC SURFACE DERIVED + PINNED.** The public header set is DERIVED from
    `CMakeLists.txt`'s `install(FILES ...)` rules (`public_headers_from_cmake`), not
    hardcoded — that is the source of truth for what a downstream consumer gets. It is
    checked against the pinned `EXPECTED_PUBLIC` so a change to the installed/exported
    surface reds until re-derived; the boundary can never drift out from under the lint.

(2) **EXAMPLE INCLUDE BOUNDARY (axis 1).** Every top-level unit under `examples/` (one
    per immediate subdirectory) that `#include`s an internal header MUST carry an entry in
    scripts/example-abi-allowlist.txt — the TRANSITION TRACKER: each entry names the fold
    row that will grow the ABI capability and rewrite the example as a client (order:
    grow the ABI entry point, rewrite the example against `vllm.h`, delete the parallel
    implementation). There are NO permanent exemptions — dev/diagnostic tools are tracked
    too. A NEW internal-reaching example with no entry reds the gate. The converse is
    enforced too: an allowlisted unit that no longer reaches internal headers (folded, now
    a clean ABI client) is a STALE entry and reds — removing it is the enforcement gate
    closing, exactly like the runner-routing allowlist. A shrink-only ratchet
    (`MAX_INTERNAL_REACHING`) caps the count of internal-reachers so it can only fall as
    folds land: a new one cannot be admitted without consciously raising the ceiling.
    `examples/cli` (the `vllm-cli` reference client — `#include "vllm.h"` only) is the
    clean baseline and must NEVER be allowlisted.

(3) **C-ABI CAPABILITY REACHABILITY (axis 2).** The marked abi-capability-table in
    docs/FEATURES.md is bound to include/vllm.h: a `reachable` row must name at least one
    C-ABI symbol that EXISTS in the header (an aspirational ABI claim reds); an
    `embedder-unreachable` row must name no live surface and carry a fold-tracked entry in
    scripts/abi-capability-allowlist.txt (a CLI/server-only capability that is not tracked
    reds). A stale allowlist entry (matching no embedder-unreachable row) reds.

The validation logic is pure functions (`internal_includes_by_unit`, `parse_allowlist`,
`boundary_errors`, `public_surface_pinned`, `capability_table`, `capability_errors`) so it
is unit- and mutation-testable (tests/scripts/test_check_surface_coverage.py), mirroring
check-runner-routing-consistency.py. Per AGENTS.md, never weaken this checker to make an
internal-reaching example or an unreachable capability pass: grow the ABI + fold, or add a
conscious, reviewable allowlist entry with its fold row.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# `scripts/` is sys.path[0] when this file is RUN, but not when a test loads it by
# path, so pin it either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker_text import strip_comments  # noqa: E402,F401  (re-exported by name)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
EXAMPLES_CMAKE = EXAMPLES_DIR / "CMakeLists.txt"
ALLOWLIST = ROOT / "scripts/example-abi-allowlist.txt"
CMAKELISTS = ROOT / "CMakeLists.txt"

# Axis 2 (capability reachability): the public C ABI header + the FEATURES table that
# advertises which capabilities an embedder can reach through it, + the embedder-
# unreachable tracker.
VLLM_H = ROOT / "include/vllm.h"
FEATURES = ROOT / "docs/FEATURES.md"
CAP_ALLOWLIST = ROOT / "scripts/abi-capability-allowlist.txt"
CAP_BEGIN = "<!-- abi-capability-table:begin -->"
CAP_END = "<!-- abi-capability-table:end -->"
REACHABLE = "reachable"
UNREACHABLE = "embedder-unreachable"

# The public surface is NOT hardcoded — it is DERIVED from the CMake install rules
# (`public_headers_from_cmake`), because that is the source of truth for what a
# downstream consumer actually gets. EXPECTED_PUBLIC is the pinned expectation the
# derived set is checked against, so the boundary can never drift silently: today
# `make install` ships `vllm.h` and the compatible `lumenengine.h` forwarding
# header. Both expose the same versioned C ABI.
EXPECTED_PUBLIC = frozenset({"vllm.h", "lumenengine.h"})

# Ratchet: the number of example units still reaching internal headers may only DECREASE
# as folds land (developer-directed 2026-08-07: no permanent exemptions, the allowlist
# shrinks). A new internal-reacher cannot be added without consciously RAISING this
# ceiling — a reviewable red flag, which is the point. Lower it as each example is folded
# onto the public ABI. 8 since the ROW 7 Kimi-Linear runner fold (kimi_linear_gen
# became a clean vllm.h client of the v13 vllm_complete_tokens after the
# paged-runner fold); 9 since the ROW 2 MiniMax-H3 video fold (2026-08-08:
# minimax_h3_gen AND minimax_h3_mux became clean vllm.h clients of the v12
# vllm_video_* slice); 11 after the ROW 1 Parakeet fold (2026-08-07:
# parakeet_transcribe); before it, 12 = server + the 4 spec-named drivers +
# minimax_h3_mux + parakeet_transcribe + the 5 dev/diagnostic tools;
# examples/cli is the clean baseline.
# COUPLED: the ratchet claims in .agents/specs/surface-coverage-2026-08-07.md and the
# state log, and the equality pin in tests/scripts/test_check_surface_coverage.py, must
# move in the SAME change as this constant.
MAX_INTERNAL_REACHING = 9

# An include INTO the internal C++ tree: `#include "vllm/..."` (the engine/model/layer
# headers under include/vllm/, NOT the flat public "vllm.h"), `#include "vt/..."` (the
# vt:: runtime), or `#include "src/..."`. BOTH the quoted `"..."` and the angle `<...>`
# forms are matched: because vllm/vllm_shared PUBLIC-include the whole include/ dir
# (CMakeLists.txt), `#include <vllm/config.h>` resolves and compiles for every example
# just like the quoted form, so a lint that only saw `"..."` would miss it. The trailing
# slash is load-bearing: `vllm.h` (public) has no slash and never matches; `vllm/...`
# (internal) always does.
_INTERNAL_INCLUDE = re.compile(r'#\s*include\s*["<]((?:vllm|vt|src)/[^">]+)[">]')

# `install(FILES <headers...> DESTINATION ...)` — the rules that define the public,
# shipped header set. Used to DERIVE the public surface rather than assume it.
_INSTALL_FILES = re.compile(
    r"install\s*\(\s*FILES\s+(.*?)\s+DESTINATION", re.IGNORECASE | re.DOTALL
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# `strip_comments` — an include named in a comment must never count as a real
# internal reach — is imported above from scripts/checker_text.py, the one copy this
# family shares. It blanks comments in place instead of collapsing them, so offsets
# and line numbers survive; for the `findall` uses below that is indistinguishable
# from the private copy it replaces.


def internal_includes(text: str) -> list[str]:
    """Every internal-tree header this source text `#include`s (comments stripped)."""
    return _INTERNAL_INCLUDE.findall(strip_comments(text))


def unit_of(path: Path, examples_dir: Path) -> str:
    """The example UNIT a source file belongs to: `examples/<top-level-subdir>`. A file
    directly under examples/ is its own unit `examples/<stem>`."""
    rel = path.relative_to(examples_dir)
    top = rel.parts[0]
    if len(rel.parts) == 1:
        top = path.stem
    return f"examples/{top}"


# An `add_executable(<target> <source> ...)` and a `target_include_directories(<target> ...)`
# call — an example can breach the boundary not only by `#include "vllm/..."` but by being
# GRANTED an internal include directory (`-I ${CMAKE_SOURCE_DIR}/src`), after which a bare
# `#include "foo.h"` resolves into the internal tree without ever matching _INTERNAL_INCLUDE.
_ADD_EXECUTABLE = re.compile(r"add_executable\s*\(\s*([A-Za-z0-9_.-]+)\s+([^\s)]+)")
_INCLUDE_DIRS_CALL = re.compile(
    r"target_include_directories\s*\(\s*([A-Za-z0-9_.-]+)\b(.*?)\)", re.DOTALL
)
# A grant into the internal source/header trees at the PROJECT root: `${CMAKE_SOURCE_DIR}/src`
# or `.../include`. `${CMAKE_CURRENT_SOURCE_DIR}/bench` (an example's OWN dir) never matches.
_INTERNAL_GRANT = re.compile(r"SOURCE_DIR\}/(?:src|include)\b")


def internal_include_dir_grant_units(
    cmake_text: str, examples_dir_name: str = "examples"
) -> dict[str, list[str]]:
    """Example UNITS whose CMake target is GRANTED an internal include directory (an `-I`
    into ${CMAKE_SOURCE_DIR}/src or /include). Scans examples/CMakeLists.txt. Only targets
    whose source lives under the examples/ tree count (a benchmarks/ source is out of scope).
    Maps unit -> [grant descriptions]. This is the second breach vector besides #include."""
    target_src: dict[str, str] = {}
    for m in _ADD_EXECUTABLE.finditer(cmake_text):
        target_src[m.group(1)] = m.group(2).strip().strip('"')
    units: dict[str, list[str]] = {}
    for m in _INCLUDE_DIRS_CALL.finditer(cmake_text):
        target, body = m.group(1), m.group(2)
        if not _INTERNAL_GRANT.search(body):
            continue
        src = target_src.get(target)
        if not src or "SOURCE_DIR" in src or src.startswith(("/", "$")):
            continue  # source outside the examples subtree (e.g. an absolute benchmarks/ path)
        top = src.split("/")[0]
        if top.endswith((".cpp", ".cc", ".cxx", ".c")):
            top = Path(top).stem
        unit = f"{examples_dir_name}/{top}"
        units.setdefault(unit, []).append(f"{target}: -I into the internal tree")
    return units


def internal_includes_by_unit(examples_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Map example UNIT -> [(source file rel-path, internal header included), ...] for
    every internal include under examples/. A unit absent from the map is a clean ABI
    client (no internal reach)."""
    hits: dict[str, list[tuple[str, str]]] = {}
    if not examples_dir.is_dir():
        return hits
    for path in sorted(examples_dir.rglob("*")):
        if path.suffix not in (".cpp", ".h", ".hpp", ".cc", ".cxx"):
            continue
        found = internal_includes(read(path))
        if not found:
            continue
        unit = unit_of(path, examples_dir)
        rel = path.relative_to(ROOT).as_posix()
        for header in found:
            hits.setdefault(unit, []).append((rel, header))
    return hits


@dataclass(frozen=True)
class AllowEntry:
    """One transition-tracker row: an internal-reaching example accepted as a KNOWN,
    TRACKED state while its capability is folded into the public ABI."""
    unit: str
    fold: str      # the tracking row-ID that grows the ABI + rewrites the example
    reason: str


def parse_allowlist(text: str) -> tuple[dict[str, AllowEntry], list[str]]:
    """Parse scripts/example-abi-allowlist.txt. Each non-comment line is
    `examples/<unit> | fold=<ROW-ID> | <reason>`. Returns (entries, errors); a malformed
    line is an error so a transition tracker can never silently lose its fold pointer."""
    entries: dict[str, AllowEntry] = {}
    errors: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            errors.append(
                f"malformed allowlist line (need `unit | fold=<ROW-ID> | reason`): {raw.strip()}"
            )
            continue
        unit, fold_field, reason = parts[0], parts[1], "|".join(parts[2:]).strip()
        if not fold_field.startswith("fold="):
            errors.append(f"allowlist line 2nd field must be `fold=<ROW-ID>`: {raw.strip()}")
            continue
        fold = fold_field[len("fold="):].strip()
        if not fold:
            errors.append(f"allowlist entry for {unit} has an empty fold row: {raw.strip()}")
            continue
        if not reason:
            errors.append(f"allowlist entry for {unit} has no reason: {raw.strip()}")
            continue
        if unit in entries:
            errors.append(f"duplicate allowlist entry for {unit}")
            continue
        entries[unit] = AllowEntry(unit=unit, fold=fold, reason=reason)
    return entries, errors


def boundary_errors(
    reaching: set[str], allowlisted: dict[str, AllowEntry]
) -> list[str]:
    """Drift between the internal-reaching example units and the transition tracker.
    Empty == the boundary check passes.
      - an internal-reaching unit NOT allowlisted -> a new CLI-only capability landed;
      - an allowlisted unit that no longer reaches internal headers -> a STALE entry
        (folded; remove it to close the enforcement gate)."""
    errors: list[str] = []
    for unit in sorted(reaching - set(allowlisted)):
        errors.append(f"uncovered:{unit}")
    for unit in sorted(set(allowlisted) - reaching):
        errors.append(f"stale:{unit}")
    return errors


def public_headers_from_cmake(cmake_text: str) -> set[str]:
    """DERIVE the public header set from CMakeLists.txt's `install(FILES ...)` rules —
    the source of truth for what a downstream consumer gets (installed to INCLUDEDIR, so
    spelled by basename). Returns the set of installed header basenames (e.g. {'vllm.h'})."""
    headers: set[str] = set()
    for m in _INSTALL_FILES.finditer(cmake_text):
        for tok in m.group(1).replace("\n", " ").split():
            tok = tok.strip().strip('"')
            if tok.endswith((".h", ".hpp")):
                headers.add(Path(tok).name)
    return headers


def public_surface_pinned(cmake_text: str) -> bool:
    """True if the DERIVED public header set (from the CMake install rules) still equals
    the pinned expectation. If the installed surface changes, this reds so EXPECTED_PUBLIC
    is re-derived and the include-boundary stays anchored to what ships, never drifts."""
    return public_headers_from_cmake(cmake_text) == set(EXPECTED_PUBLIC)


# --- Axis 2: FEATURES C-ABI capability coverage vs the actual ABI header ---------------

_BACKTICK = re.compile(r"`([A-Za-z0-9_]+)`")


def _norm(label: str) -> str:
    """Normalize a capability label for a stable join key: strip, drop backticks,
    lowercase, collapse internal whitespace."""
    return " ".join(label.replace("`", "").lower().split())


@dataclass(frozen=True)
class CapRow:
    label: str          # verbatim col-1 text
    surface: str        # col-2 text (C-ABI symbols, or a dash)
    status: str         # reachable | embedder-unreachable | (other -> error)
    symbols: tuple[str, ...]  # backticked tokens parsed from `surface`


def capability_table(features_text: str) -> list[CapRow] | None:
    """Parse the marked abi-capability-table from docs/FEATURES.md, or None if the
    markers are absent (the caller turns that into an explicit error so a deleted table
    can never read as 'no gaps')."""
    start = features_text.find(CAP_BEGIN)
    end = features_text.find(CAP_END)
    if start == -1 or end == -1 or end < start:
        return None
    block = features_text[start + len(CAP_BEGIN):end]
    rows: list[CapRow] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if all(set(c) <= set("-: ") for c in cells):  # separator
            continue
        if _norm(cells[0]) in ("capability", ""):      # header
            continue
        rows.append(
            CapRow(
                label=cells[0],
                surface=cells[1],
                status=_norm(cells[2]),
                symbols=tuple(_BACKTICK.findall(cells[1])),
            )
        )
    return rows


def declared_abi_symbols(vllm_h_text: str) -> set[str]:
    """Every identifier token DECLARED in the public C-ABI header. Comments are stripped
    FIRST so a symbol that survives only in a doc comment does not read as declared — else
    deleting the `vllm_complete` declaration would stay green because the prose at
    vllm.h:259/:309 mentions the token. Used to verify a `reachable` row names a symbol/
    field that actually exists in the code (no aspirational ABI)."""
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", strip_comments(vllm_h_text)))


def capability_errors(
    features_text: str,
    vllm_h_text: str,
    cap_allowlist: dict[str, AllowEntry],
) -> list[str]:
    """Bind the FEATURES C-ABI capability table to reality: a `reachable` row must name
    at least one C-ABI symbol that EXISTS in include/vllm.h; an `embedder-unreachable`
    row must name no live surface and carry a fold-tracked allowlist entry. Every
    allowlist entry must match an embedder-unreachable row (no stale)."""
    errors: list[str] = []
    rows = capability_table(features_text)
    if rows is None:
        return [
            "docs/FEATURES.md is missing the abi-capability-table markers "
            f"({CAP_BEGIN} ... {CAP_END}); they bind the public C-ABI capability claims "
            "to include/vllm.h and the embedder-unreachable tracker"
        ]
    symbols = declared_abi_symbols(vllm_h_text)
    seen_unreachable: set[str] = set()
    for row in rows:
        key = _norm(row.label)
        if row.status == REACHABLE:
            if not row.symbols:
                errors.append(
                    f"capability '{row.label}' is marked reachable but names no C-ABI "
                    "symbol in backticks; name the include/vllm.h entry point(s)"
                )
                continue
            missing = [s for s in row.symbols if s not in symbols]
            if missing:
                errors.append(
                    f"capability '{row.label}' claims C-ABI symbol(s) not declared in "
                    f"include/vllm.h: {', '.join(missing)}; add the entry point or fix "
                    "the row (an aspirational ABI claim)"
                )
        elif row.status == UNREACHABLE:
            seen_unreachable.add(key)
            if row.symbols:
                errors.append(
                    f"capability '{row.label}' is marked embedder-unreachable but names "
                    f"a C-ABI symbol ({', '.join(row.symbols)}); mark it reachable or "
                    "remove the surface"
                )
            if key not in cap_allowlist:
                errors.append(
                    f"capability '{row.label}' is embedder-unreachable with no entry in "
                    "scripts/abi-capability-allowlist.txt; add one naming its fold row "
                    "(the ARCH-ONE-SURFACE order-of-work), or grow the C-ABI so the row "
                    "becomes reachable"
                )
        else:
            errors.append(
                f"capability '{row.label}' has status '{row.status}'; expected "
                f"'{REACHABLE}' or '{UNREACHABLE}'"
            )
    for key in sorted(set(cap_allowlist) - seen_unreachable):
        errors.append(
            f"scripts/abi-capability-allowlist.txt entry '{key}' matches no "
            "embedder-unreachable row in the FEATURES abi-capability-table (stale; the "
            "capability became reachable — remove it)"
        )
    return errors


def main() -> int:
    rc = 0

    # (1) The public surface the boundary is drawn around must still be what ships.
    if not public_surface_pinned(read(CMAKELISTS)):
        rc = 1
        derived = sorted(public_headers_from_cmake(read(CMAKELISTS)))
        print(
            "ERROR: the DERIVED public header set from CMakeLists.txt's install(FILES ...) "
            f"rules is {derived or '(none)'}, not the pinned {sorted(EXPECTED_PUBLIC)}. The "
            "installed/exported surface changed; re-derive EXPECTED_PUBLIC in check-surface-"
            "coverage.py so the example include-boundary stays anchored to what downstream "
            "consumers actually get.",
            file=sys.stderr,
        )

    reaching_map = internal_includes_by_unit(EXAMPLES_DIR)
    # A CMake -I grant into the internal tree is a breach even with no matching #include.
    grant_map = (
        internal_include_dir_grant_units(read(EXAMPLES_CMAKE))
        if EXAMPLES_CMAKE.exists() else {}
    )
    for unit, grants in grant_map.items():
        reaching_map.setdefault(unit, []).append(("examples/CMakeLists.txt", grants[0]))
    reaching = set(reaching_map)

    # Shrink-only ratchet: the count of internal-reaching example units may only fall.
    if len(reaching) > MAX_INTERNAL_REACHING:
        rc = 1
        print(
            f"ERROR: {len(reaching)} example units reach internal headers, over the "
            f"{MAX_INTERNAL_REACHING} ratchet ceiling — the allowlist may only SHRINK. A "
            "new internal-reaching example is the exact defect this gate stops: fold it "
            "onto the public ABI instead of raising the ceiling. Lower MAX_INTERNAL_REACHING "
            "in check-surface-coverage.py only when an example is folded, never to admit a "
            "new one.",
            file=sys.stderr,
        )
    allow_text = read(ALLOWLIST) if ALLOWLIST.exists() else ""
    allowlisted, allow_errors = parse_allowlist(allow_text)

    for err in allow_errors:
        rc = 1
        print(f"ERROR: scripts/example-abi-allowlist.txt: {err}", file=sys.stderr)

    errors = boundary_errors(reaching, allowlisted)
    uncovered = [e.split(":", 1)[1] for e in errors if e.startswith("uncovered:")]
    stale = [e.split(":", 1)[1] for e in errors if e.startswith("stale:")]

    if uncovered:
        rc = 1
        print(
            "ERROR: example(s) reach past the public ABI surface (include/vllm.h) into "
            "the internal C++ tree (vllm/..., vt/..., src/...) and are NOT on "
            "scripts/example-abi-allowlist.txt — the CLI-only-capability anti-pattern "
            "(AGENTS.md: examples are thin clients of the public surface, never internal "
            "headers):",
            file=sys.stderr,
        )
        for unit in uncovered:
            sample = reaching_map[unit][0]
            more = f" (+{len(reaching_map[unit]) - 1} more)" if len(reaching_map[unit]) > 1 else ""
            print(f"  - {unit}: {sample[0]} includes \"{sample[1]}\"{more}", file=sys.stderr)
        print(
            "Grow the public ABI (include/vllm.h) so the example can be a client, then "
            "rewrite it against vllm.h and delete the parallel implementation; or, while "
            "that fold is scheduled, add the unit to scripts/example-abi-allowlist.txt "
            "with `| fold=<ROW-ID> | <reason>` naming its tracking row.",
            file=sys.stderr,
        )
    if stale:
        rc = 1
        print(
            "ERROR: scripts/example-abi-allowlist.txt has entries for example(s) that no "
            "longer include internal headers — the fold landed; remove the stale "
            "transition-tracker entry (this is the enforcement gate closing):",
            file=sys.stderr,
        )
        for unit in stale:
            print(f"  - {unit} (now a clean public-ABI client)", file=sys.stderr)

    # (3) Capability reachability: FEATURES C-ABI capability table vs include/vllm.h.
    cap_allow_text = read(CAP_ALLOWLIST) if CAP_ALLOWLIST.exists() else ""
    cap_allow, cap_allow_errors = parse_allowlist(cap_allow_text)
    for err in cap_allow_errors:
        rc = 1
        print(f"ERROR: scripts/abi-capability-allowlist.txt: {err}", file=sys.stderr)
    cap_errors = capability_errors(read(FEATURES), read(VLLM_H), cap_allow)
    if cap_errors:
        rc = 1
        print(
            "ERROR: the docs/FEATURES.md C-ABI capability coverage table drifted from "
            "the actual C ABI (include/vllm.h) / the embedder-unreachable tracker:",
            file=sys.stderr,
        )
        for err in cap_errors:
            print(f"  - {err}", file=sys.stderr)

    if rc == 0:
        clean = "examples/cli"
        n_units = len({unit_of(p, EXAMPLES_DIR)
                       for p in EXAMPLES_DIR.rglob("*")
                       if p.suffix in (".cpp", ".h", ".hpp", ".cc", ".cxx")})
        rows = capability_table(read(FEATURES)) or []
        n_reach = sum(1 for r in rows if r.status == REACHABLE)
        n_unreach = sum(1 for r in rows if r.status == UNREACHABLE)
        print(
            f"OK (surface-coverage): {n_units} example unit(s); {len(reaching)} reach "
            f"internal headers ({len(allowlisted)} transition-tracked with a fold row), "
            f"the rest are public-ABI clients ({clean} is the clean baseline). C-ABI "
            f"capability table: {n_reach} reachable, {n_unreach} embedder-unreachable "
            f"({len(cap_allow)} fold-tracked)."
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
