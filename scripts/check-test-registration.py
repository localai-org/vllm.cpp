#!/usr/bin/env python3
"""Require critical regression tests to remain buildable and CTest-registered.

The CPU CI lane builds every target and runs ``ctest``, but both promises become
vacuous when a regression test is accidentally removed from ``tests/CMakeLists.txt``.
This tree gate pins the small set of tests whose review explicitly requires a
non-vacuous registration guard.  It also verifies that the shared helper still
creates an executable *and* registers that executable with CTest.

It pins CTest LABELS for the same reason, and the reason is measured rather than
assumed.  A gate whose preconditions no runner has is invoked by label -- the
documented recipe is ``ctest -L gpu`` -- and ``ctest -L`` prints
``No tests were found!!!`` and returns **0** when the label selects nothing
(CMake 3.28.3).  So a renamed or deleted label turns a documented gate into a
command that measures nothing while reporting success, which is the exact defect
class the labelled gate itself exists to close.  ``REQUIRED_LABEL_SELECTIONS``
holds the expected selection as a literal in this file, never read back out of
``tests/CMakeLists.txt``, because a checker that reads its expectation from the
file it checks is a tautology.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_CMAKE = ROOT / "tests/CMakeLists.txt"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"
MUTATION_SUITE = ROOT / "tests/scripts/test_check_test_registration.py"
MUTATION_MANIFEST = ROOT / "tests/scripts/check_test_registration_mutations.txt"
MUTATION_MANIFEST_SHA256 = (
    "4589e7a96d839a8309fd0e5799cc7a652b22a9d568d56151e2a1b8feea5d02ee"
)

REQUIRED_TESTS = {
    "test_device_selection": "vllm/entrypoints/test_device_selection.cpp",
}

# label -> the EXACT set of CTest test names it may select.  Exact, not a floor:
# a floor cannot see a second test that quietly joins a lane whose whole purpose
# is that a human runs it deliberately inside a lease.  Adding a labelled gate is
# a one-line addition here and is meant to be a deliberate record.
REQUIRED_LABEL_SELECTIONS = {
    "gpu": (
        "test_minimax_music3_depth_arm_real",
        "test_minimax_music3_device_arm_real",
    ),
}

# Python suites whose registration in BOTH lanes is pinned: the name in the
# preflight `SUITES` array, and the exact command a CI step runs.  `SUITES` is a
# bare Bash array that nothing measures, so before this every entry in it was
# deletable at rc=0 -- measured on `test_sglang_lease_identity` and reproduced on
# an unrelated control entry (#1833).  "Registered in TWO places, deliberately",
# which `.github/workflows/ci.yml` documents, bought nothing in either direction.
#
# EXPLICIT AND EXACT, in the shape `REQUIRED_TESTS` above uses for C++, and
# deliberately NOT the symmetric population rule ("every `tests/scripts/test_*.py`
# in both lanes").  That rule is correct and it reds on 23 suites at once -- 11
# preflight-only and 12 CI-only at the time of writing -- each of which is a
# separate classification, some of them lane-correct on purpose.  Forcing 23
# unreviewed classifications through one change, or lowering it to a floor that
# admits them, would make a mute switch of a real gate.  Those 23 belong to #408
# and #1509; this set is the mechanism they can widen rather than replace.
#
# `check-test-registration` and `test_check_test_registration` are NOT listed
# here.  They keep their own hardcoded checks below, whose diagnostics are what
# the M9-M12 mutation cases match on.
REQUIRED_SUITE_REGISTRATIONS = {
    "test_sglang_lease_identity": "tests/scripts/test_sglang_lease_identity.py",
    # #2553. Added the moment the suite reached both lanes, which is the only
    # moment it costs nothing: it was preflight-only, so the CI step landing
    # beside this line would otherwise have been deletable at rc=0 -- the exact
    # #1833 finding this set exists to answer. It recomputes the sha256 of the
    # #1864 reference render and the #2514 Phase A latents, the evidence
    # `.agents/oracles/ltx-2.md` cites for `gateable = yes`.
    #
    # This is the widening the seeding comment above anticipates, one classified
    # suite at a time. It is NOT the symmetric population rule, which still
    # belongs to #408 and #1509.
    "test_ltx2_oracle_goldens": "tests/scripts/test_ltx2_oracle_goldens.py",
    # #2877, added in the change that first gave the suite a lane at all. It had
    # landed in NEITHER lane, and the tool it gates had never been run on a real
    # fingerprint: `taps=N END` is a cumulative counter and the comparator read
    # it per-step, so it refused every genuine multi-step capture while eleven
    # single-step cases -- the only shape its fixtures could express -- passed.
    # Pinning it here is the same one-suite-at-a-time widening #2553 made, and
    # the two named cases in the mutation suite are what make it a semantic
    # change rather than a fixture edit.
    "test_q4exp_layerfp_diff": "tests/scripts/test_q4exp_layerfp_diff.py",
}

def _without_line_comments(text: str) -> str:
    """Remove ``#`` comments while preserving quoted ``#`` characters."""

    cleaned: list[str] = []
    for line in text.splitlines():
        quoted = False
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quoted:
                kept.append(char)
                escaped = True
                continue
            if char == '"':
                quoted = not quoted
                kept.append(char)
                continue
            if char == "#" and not quoted:
                break
            kept.append(char)
        cleaned.append("".join(kept))
    return "\n".join(cleaned)


def _configure(
    source_dir: Path, build_dir: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Configure CMake after requesting its semantic target codemodel."""

    query = build_dir / ".cmake/api/v1/query/codemodel-v2"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _codemodel_targets(
    build_dir: Path,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    """Return one configuration and its targets from the CMake File API.

    Multi-config generators describe several artifact paths.  Release is the
    CI contract and must be selected consistently for both the codemodel and
    the subsequent CTest query.
    """

    reply = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        return None, {}
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    codemodel_file = index["reply"]["codemodel-v2"]["jsonFile"]
    codemodel = json.loads((reply / codemodel_file).read_text(encoding="utf-8"))
    configurations = codemodel.get("configurations", [])
    if not configurations:
        return None, {}
    configuration = next(
        (entry for entry in configurations if entry.get("name") == "Release"),
        configurations[0],
    )
    targets: dict[str, dict[str, object]] = {}
    for summary in configuration.get("targets", []):
        detail = json.loads((reply / summary["jsonFile"]).read_text(encoding="utf-8"))
        targets[summary["name"]] = detail
    name = configuration.get("name")
    return (name if isinstance(name, str) and name else None), targets


def _ctest_tests(
    build_dir: Path, configuration: str | None
) -> dict[str, dict[str, object]]:
    """Return configured CTest command/properties keyed by test name.

    CTest omits ``command`` from JSON when a target executable does not exist
    yet.  The caller materializes disposable placeholders at the File-API
    artifact paths before asking for this document, so the command is the
    resolved executable path rather than an uninterpreted CMake token.
    """

    command = ["ctest", "--test-dir", str(build_dir)]
    if configuration is not None:
        command.extend(["-C", configuration])
    command.append("--show-only=json-v1")
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return {
        test["name"]: {
            "command": test.get("command", []),
            "properties": test.get("properties", []),
        }
        for test in document.get("tests", [])
        if isinstance(test.get("name"), str)
    }


def _cmake_truthy(value: object) -> bool:
    """Interpret CMake's documented false constants; unknown values fail closed."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().upper()
    return normalized not in {
        "",
        "0",
        "FALSE",
        "OFF",
        "NO",
        "N",
        "IGNORE",
        "NOTFOUND",
    } and not normalized.endswith("-NOTFOUND")


# Debug-symbol side artifacts, which a multi-config generator lists BESIDE the
# binary it produced.  Measured on MSVC 19.44 / CMake 4.4.2: the codemodel
# describes `test_device_selection` as type EXECUTABLE with TWO artifacts,
# `tests/Release/test_device_selection.exe` and `.../test_device_selection.pdb`.
# Only the first is the file CTest runs, and requiring the artifact LIST to hold
# exactly one entry therefore reported a configured, correctly registered target
# as having no executable at all.  `.dSYM` is the same shape from Xcode, and is
# spelled lowercase here because the comparison lowercases the path.
_DEBUG_SYMBOL_SUFFIXES = (".pdb", ".dsym", ".dwarf")


def _target_artifact(build_dir: Path, detail: dict[str, object]) -> Path | None:
    """Resolve the single configured executable artifact for a File-API target.

    "Exactly one" still means one *executable*: a target that genuinely
    produces two binaries is refused, as before.  Only the artifacts that are
    not the binary are set aside.
    """

    artifacts = detail.get("artifacts", [])
    if detail.get("type") != "EXECUTABLE" or not isinstance(artifacts, list):
        return None
    paths = [
        entry.get("path")
        for entry in artifacts
        if isinstance(entry, dict)
        and not str(entry.get("path", "")).lower().endswith(_DEBUG_SYMBOL_SUFFIXES)
    ]
    if len(paths) != 1 or not isinstance(paths[0], str):
        return None
    return (build_dir / paths[0]).resolve()


def _materialize_ctest_targets(
    build_dir: Path, targets: dict[str, dict[str, object]], required: dict[str, str]
) -> dict[str, Path]:
    """Make unbuilt configured target paths resolvable to CTest JSON.

    Only the disposable configure directory is touched.  Existing artifacts are
    never replaced; absent artifacts become empty executable placeholders long
    enough for ``ctest --show-only=json-v1`` to resolve target-name commands.
    """

    artifacts: dict[str, Path] = {}
    for target in required:
        detail = targets.get(target)
        if detail is None:
            continue
        artifact = _target_artifact(build_dir, detail)
        if artifact is None:
            continue
        artifacts[target] = artifact
        if artifact.exists():
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
        artifact.chmod(0o700)
    return artifacts


def _configured_contract_errors(
    source_dir: Path,
    build_dir: Path,
    required: dict[str, str],
    extra_args: list[str] | None = None,
) -> list[str]:
    """Ask CMake/CTest what exists instead of interpreting CMake source text."""

    configured = _configure(source_dir, build_dir, extra_args)
    if configured.returncode != 0:
        transcript = configured.stdout + configured.stderr
        errors = ["CMake configure failed while proving required test registration"]
        for target in sorted(required):
            errors.append(f"missing required test target {target} in configured codemodel")
        if "already exists" in transcript:
            for target in sorted(required):
                errors.append(f"required test target {target} is registered 2 times")
        errors.append(
            "vllm_cpp_add_test does not create an executable with its configured sources"
        )
        return errors

    configuration, targets = _codemodel_targets(build_dir)
    artifacts = _materialize_ctest_targets(build_dir, targets, required)
    tests = _ctest_tests(build_dir, configuration)
    errors: list[str] = []
    for target, expected_source in sorted(required.items()):
        detail = targets.get(target)
        if detail is None:
            errors.append(f"missing required test target {target} in configured codemodel")
            errors.append(
                "vllm_cpp_add_test does not create an executable with its configured sources"
            )
            continue
        actual_sources = {
            Path(source["path"]).as_posix() for source in detail.get("sources", [])
        }
        if expected_source not in actual_sources:
            actual = ", ".join(sorted(actual_sources)) or "<no sources>"
            errors.append(
                f"required test target {target} must compile {expected_source}; got {actual}"
            )
        artifact = artifacts.get(target)
        if artifact is None:
            errors.append(
                f"required test target {target} has no single configured executable artifact"
            )
        if target not in tests:
            errors.append(
                f"required test target {target} is not registered with CTest; "
                "vllm_cpp_add_test does not register that executable with CTest"
            )
        elif artifact is not None:
            test = tests[target]
            command = test.get("command", [])
            if not isinstance(command, list):
                command = []
            actual_command: Path | None = None
            if len(command) == 1:
                candidate = Path(command[0])
                actual_command = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (build_dir / candidate).resolve()
                )
            if actual_command != artifact:
                rendered = shlex.join(command) if command else "<unresolved command>"
                errors.append(
                    f"CTest test {target} must execute configured target {target} exactly; "
                    f"got {rendered}"
                )
            properties = test.get("properties", [])
            if not isinstance(properties, list):
                properties = []
            disabled = [
                prop.get("value")
                for prop in properties
                if isinstance(prop, dict) and prop.get("name") == "DISABLED"
            ]
            if any(_cmake_truthy(value) for value in disabled):
                errors.append(f"CTest test {target} must not be DISABLED")
    return errors


def _test_labels(test: dict[str, object]) -> set[str]:
    """Return the CTest LABELS of one ``--show-only=json-v1`` test entry."""

    properties = test.get("properties", [])
    if not isinstance(properties, list):
        return set()
    labels: set[str] = set()
    for prop in properties:
        if not isinstance(prop, dict) or prop.get("name") != "LABELS":
            continue
        value = prop.get("value")
        if isinstance(value, str):
            labels.add(value)
        elif isinstance(value, list):
            labels.update(entry for entry in value if isinstance(entry, str))
    return labels


def _label_selection_errors(
    tests: dict[str, dict[str, object]], selections: dict[str, tuple[str, ...]]
) -> list[str]:
    """Compare the configured LABELS against this file's literal pin.

    The diagnostic names BOTH sides of the comparison in words.  ``ctest -L``
    reports success over an empty selection, so a reader who is handed only a
    return code cannot tell a passing lane from an absent one.
    """

    errors: list[str] = []
    for label, expected in sorted(selections.items()):
        selected = sorted(
            name for name, test in tests.items() if label in _test_labels(test)
        )
        wanted = sorted(expected)
        if selected == wanted:
            continue
        errors.append(
            f"ctest -L {label} selects {len(selected)} test(s) "
            f"[{', '.join(selected) if selected else '<none>'}]; "
            f"REQUIRED_LABEL_SELECTIONS in scripts/check-test-registration.py pins "
            f"{len(wanted)} [{', '.join(wanted) if wanted else '<none>'}]. "
            "Compared the LABELS property of the CONFIGURED tree against the literal "
            "pin in this checker, not against tests/CMakeLists.txt text. "
            "An empty selection is the dangerous case: ctest -L returns 0 over it"
        )
    return errors


def _configured_label_errors(
    source_dir: Path,
    build_dir: Path,
    selections: dict[str, tuple[str, ...]],
    extra_args: list[str] | None = None,
) -> list[str]:
    """Ask CMake/CTest which tests a label selects, then compare with the pin."""

    configured = _configure(source_dir, build_dir, extra_args)
    if configured.returncode != 0:
        return [
            "CMake configure failed while proving CTest label selection; "
            "no label could be read, which fails closed"
        ]
    configuration, _ = _codemodel_targets(build_dir)
    return _label_selection_errors(_ctest_tests(build_dir, configuration), selections)


def label_errors(
    cmake_text: str, selections: dict[str, tuple[str, ...]] | None = None
) -> list[str]:
    """Return violations of the CTest label-selection contract."""

    if selections is None:
        selections = REQUIRED_LABEL_SELECTIONS
    with tempfile.TemporaryDirectory(prefix="vllm-label-unit-") as temporary:
        root = Path(temporary)
        (root / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        for source in {
            *REQUIRED_TESTS.values(),
            "vllm/entrypoints/other.cpp",
            "parity/test_minimax_music3_device_arm_real.cpp",
            "parity/test_minimax_music3_depth_arm_real.cpp",
        }:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int label_guard_dummy;\n", encoding="utf-8")
        return _configured_label_errors(root, root / "build", selections)


def registration_errors(
    cmake_text: str, required: dict[str, str] | None = None
) -> list[str]:
    """Return violations of the executable + CTest registration contract."""

    if required is None:
        required = REQUIRED_TESTS
    with tempfile.TemporaryDirectory(prefix="vllm-registration-unit-") as temporary:
        root = Path(temporary)
        (root / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        for source in {
            *required.values(),
            "vllm/entrypoints/other.cpp",
        }:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int registration_guard_dummy;\n", encoding="utf-8")
        return _configured_contract_errors(root, root / "build", required)


# ---------------------------------------------------------------------------
# The registration must agree with the LINK (#1883).
#
# `if(VLLM_CPP_SERVER)` in the top-level CMakeLists decides which translation
# units enter `libvllm`.  A test target registered OUTSIDE that guard is still
# configured when the flag is OFF, and CMake is happy, because nothing about a
# missing definition is visible until `ld` runs.  So the failure lands at link
# time on a configuration CI exercises and no configure-only gate can see.
#
# This check is STATIC and derives BOTH sides.  It reads the gated
# `target_sources` out of the top-level file, resolves each one's declaring
# header, walks the tree's own `#include "..."` graph, and refuses any
# `vllm_cpp_add_test` target outside the guard whose sources reach one of those
# headers.  Nothing here is transcribed from the file it checks: the gated set
# is read, not listed, so adding a seventh gated translation unit extends the
# gate instead of aging it out.
#
# What the proxy CANNOT see, stated because an unwritten limit gets quoted as
# though it were none:
#   * it matches the declaring HEADER, not the symbol, so a hand-declared
#     `extern` or a macro-built call is invisible to it;
#   * it reads `#include "..."` only, and would miss an angle-bracket include of
#     a repository header (the tree uses quotes for its own headers);
#   * it can fire on an include that no longer carries a call.  That is still a
#     defect worth naming -- an unused include of a conditionally compiled
#     header -- and the fix is to drop the include or move the target, never to
#     widen this check;
#   * it says nothing about whether the suite is REACHED.  That is
#     `.agents/reachability.md`'s question, and this gate does not answer it.
SERVER_GUARD_CONDITION = "VLLM_CPP_SERVER"

_CMAKE_COMMAND = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CMAKE_ARGUMENT = re.compile(r'"[^"]*"|[^\s()]+')
_QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)
_SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")
_HEADER_SUFFIXES = (".h", ".hpp", ".hh")


def _cmake_commands(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line, lowercased name, raw argument text)`` per invocation.

    Arguments are joined across physical lines, because this tree wraps a
    `vllm_cpp_add_test` call onto a second line often enough that a line-wise
    reader would miss the source of every wrapped target.
    """

    text = _without_line_comments(text)
    commands: list[tuple[int, str, str]] = []
    position = 0
    while True:
        match = _CMAKE_COMMAND.search(text, position)
        if match is None:
            return commands
        index = match.end()
        depth = 1
        quoted = False
        escaped = False
        while index < len(text) and depth:
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
            index += 1
        commands.append(
            (
                text.count("\n", 0, match.start()) + 1,
                match.group(1).lower(),
                text[match.end() : index - 1],
            )
        )
        position = index


def _cmake_arguments(raw: str) -> list[str]:
    return [argument.strip('"') for argument in _CMAKE_ARGUMENT.findall(raw)]


def _guarded_commands(text: str) -> list[tuple[int, str, list[str], bool]]:
    """Yield each command with whether an enclosing `if` tests the guard.

    `else()` CLEARS the frame's flag rather than inverting it: the else branch
    of `if(VLLM_CPP_SERVER)` is exactly the un-gated one, so treating it as
    gated would hide a target that only exists when the server is OFF.
    """

    scanned: list[tuple[int, str, list[str], bool]] = []
    stack: list[bool] = []
    for line, name, raw in _cmake_commands(text):
        if name == "if":
            stack.append(SERVER_GUARD_CONDITION in raw)
            continue
        if name in {"else", "elseif"}:
            if stack:
                stack[-1] = SERVER_GUARD_CONDITION in raw if name == "elseif" else False
            continue
        if name == "endif":
            if stack:
                stack.pop()
            continue
        scanned.append((line, name, _cmake_arguments(raw), any(stack)))
    return scanned


def _include_spelling(relative: str) -> str:
    """The path a `#include "..."` would name for a repository-relative path.

    An `#include "..."` directive is spelled with forward slashes whatever the
    host is, so the key this returns has to be that spelling on every host.  The
    separator is normalized first because `str(Path)` on Windows renders a
    relative path with backslashes, and none of the prefixes below ever matched
    one: each caller of this function built its key, or its `include/`
    candidate, from a `str()`, so on Windows `_gated_declaring_headers` resolved
    1 of the 6 gated translation units in this tree and 0 in the suite's own
    fixtures.  A guard that cannot name the header it forbids cannot fire.
    """

    normalized = relative.replace("\\", "/")
    for base in ("include/", "src/", "tests/"):
        if normalized.startswith(base):
            return normalized[len(base) :]
    return normalized


def _without_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _gated_declaring_headers(root: Path) -> dict[str, str]:
    """Map each gated translation unit's declaring header to that unit."""

    top = root / "CMakeLists.txt"
    if not top.is_file():
        return {}
    headers: dict[str, str] = {}
    for _, name, arguments, gated in _guarded_commands(
        top.read_text(encoding="utf-8", errors="replace")
    ):
        if name != "target_sources" or not gated:
            continue
        for argument in arguments:
            if not argument.endswith(_SOURCE_SUFFIXES):
                continue
            stem = Path(argument)
            candidates = [
                root / "include" / _include_spelling(str(stem.with_suffix(suffix)))
                for suffix in _HEADER_SUFFIXES
            ] + [root / stem.with_suffix(suffix) for suffix in _HEADER_SUFFIXES]
            for candidate in candidates:
                if candidate.is_file():
                    spelling = _include_spelling(
                        str(candidate.resolve().relative_to(root.resolve()))
                    )
                    headers.setdefault(spelling, argument)
                    break
    return headers


def _repository_includes(root: Path) -> dict[str, list[str]]:
    """The tree's own quoted-include graph, keyed by include spelling."""

    graph: dict[str, list[str]] = {}
    for base in ("include", "src", "tests"):
        directory = root / base
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.suffix not in _SOURCE_SUFFIXES + _HEADER_SUFFIXES:
                continue
            spelling = _include_spelling(str(path.relative_to(root)))
            graph.setdefault(
                spelling,
                _QUOTED_INCLUDE.findall(
                    _without_c_comments(path.read_text(encoding="utf-8", errors="replace"))
                ),
            )
    return graph


def _reaching_spellings(
    graph: dict[str, list[str]], gated: dict[str, str]
) -> dict[str, str]:
    """Map an include spelling to the gated header it TRANSITIVELY reaches.

    Transitive because `downloader.h` includes `hf_hub.h`: a test that names
    only the former still needs the latter's definitions at link time, and a
    direct-include check would call that clean.  The VALUE is the gated header,
    never the gated source, so the caller can name both in one diagnostic.
    """

    reaching = {header: header for header in gated}
    changed = True
    while changed:
        changed = False
        for spelling, includes in graph.items():
            if spelling in reaching:
                continue
            for include in includes:
                if include in reaching:
                    reaching[spelling] = reaching[include]
                    changed = True
                    break
    return reaching


def server_guard_errors(root: Path = ROOT) -> list[str]:
    """Refuse a test target that links a gated unit from outside the guard."""

    tests_cmake = root / "tests/CMakeLists.txt"
    if not tests_cmake.is_file():
        return ["tests/CMakeLists.txt is missing"]
    gated = _gated_declaring_headers(root)
    if not gated:
        return [
            "no `if("
            + SERVER_GUARD_CONDITION
            + ")` target_sources were found in the top-level CMakeLists, so this "
            "guard would pass vacuously"
        ]
    reaching = _reaching_spellings(_repository_includes(root), gated)

    errors: list[str] = []
    for line, name, arguments, guarded in _guarded_commands(
        tests_cmake.read_text(encoding="utf-8", errors="replace")
    ):
        if name != "vllm_cpp_add_test" or guarded or not arguments:
            continue
        target = arguments[0]
        for source in arguments[1:]:
            path = root / "tests" / source
            if not path.is_file():
                continue
            for include in _QUOTED_INCLUDE.findall(
                _without_c_comments(path.read_text(encoding="utf-8", errors="replace"))
            ):
                header = reaching.get(include)
                if header is None:
                    continue
                errors.append(
                    f"tests/CMakeLists.txt:{line}: {target} is registered outside "
                    f"`if({SERVER_GUARD_CONDITION})`, but tests/{source} reaches "
                    f"{header}, declared by {gated[header]}, which is compiled "
                    f"only inside that guard. A -D{SERVER_GUARD_CONDITION}=OFF "
                    "build configures and then fails to link. Move the "
                    "registration inside the guard"
                )
                break
    return errors


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _literal_block(lines: list[str], header_index: int) -> list[str]:
    parent_indent = _indent(lines[header_index])
    raw: list[str] = []
    for candidate in lines[header_index + 1 :]:
        if not candidate.strip():
            raw.append("")
            continue
        if _indent(candidate) <= parent_indent:
            break
        raw.append(candidate)
    nonblank = [line for line in raw if line.strip()]
    if not nonblank:
        return []
    content_indent = min(_indent(line) for line in nonblank)
    return [line[content_indent:] if line.strip() else "" for line in raw]


def _yaml_mapping(line: str, indent: int, sequence: bool = False) -> tuple[str, str] | None:
    """Parse one direct YAML mapping key in the workflow's structural subset."""

    if _indent(line) != indent:
        return None
    content = line[indent:]
    if sequence:
        if not content.startswith("-"):
            return None
        content = content[1:].lstrip()
        if not content:
            return None
    match = re.match(r"^(?P<key>'[^']*'|\"[^\"]*\"|[^:#]+?)\s*:\s*(?P<value>.*)$", content)
    if match is None:
        return None
    key = match.group("key").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1]
    return key.strip(), match.group("value").strip()


def _unconditional_ci_run_blocks(text: str) -> list[list[str]]:
    """Return literal run blocks owned by unconditional Actions jobs/steps.

    This is deliberately a narrow GitHub-Actions structural parser, not a YAML
    implementation: it recognizes the canonical ``jobs -> job -> steps -> -``
    hierarchy and direct ``if``/``run`` fields.  A run block in prose, a sibling
    mapping, or a conditional job/step never enters the result.
    """

    lines = text.splitlines()
    blocks: list[list[str]] = []
    jobs_index = next(
        (i for i, line in enumerate(lines) if line == "jobs:"), None
    )
    if jobs_index is None:
        return blocks

    job_starts = [
        i
        for i in range(jobs_index + 1, len(lines))
        if (mapping := _yaml_mapping(lines[i], 2)) is not None
        and mapping[1] == ""
    ]
    for job_pos, job_start in enumerate(job_starts):
        job_end = job_starts[job_pos + 1] if job_pos + 1 < len(job_starts) else len(lines)
        job_lines = lines[job_start + 1 : job_end]
        job_fields = {
            mapping[0]
            for line in job_lines
            if (mapping := _yaml_mapping(line, 4)) is not None
        }
        if "if" in job_fields:
            continue
        steps_offset = next(
            (
                i
                for i, line in enumerate(job_lines)
                if _yaml_mapping(line, 4) == ("steps", "")
            ),
            None,
        )
        if steps_offset is None:
            continue
        steps_start = job_start + 1 + steps_offset + 1
        step_starts = [
            i
            for i in range(steps_start, job_end)
            if _indent(lines[i]) == 6 and lines[i][6:].startswith("-")
        ]
        for step_pos, step_start in enumerate(step_starts):
            step_end = (
                step_starts[step_pos + 1]
                if step_pos + 1 < len(step_starts)
                else job_end
            )
            step_fields: dict[str, tuple[str, int]] = {}
            first = _yaml_mapping(lines[step_start], 6, sequence=True)
            if first is not None:
                step_fields[first[0]] = (first[1], step_start)
            for index in range(step_start + 1, step_end):
                mapping = _yaml_mapping(lines[index], 8)
                if mapping is not None:
                    step_fields[mapping[0]] = (mapping[1], index)
            if {"if", "continue-on-error", "shell"} & step_fields.keys():
                continue
            run = step_fields.get("run")
            if run is not None and re.fullmatch(r"\|[-+]?", run[0]):
                run_index = run[1]
                blocks.append(_literal_block(lines, run_index))
    return blocks


def _direct_commands(block: list[str]) -> list[list[str]] | None:
    """Parse a literal block that contains only direct, unconditional commands."""

    commands: list[list[str]] = []
    for line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line != line.lstrip():
            return None
        try:
            argv = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            return None
        if not argv or any(token in {";", "&&", "||", "|", "&"} for token in argv):
            return None
        commands.append(argv)
    return commands


def _ci_has_active_guard_step(ci_text: str) -> bool:
    expected = [
        ["python3", "scripts/check-test-registration.py"],
        ["python3", "tests/scripts/test_check_test_registration.py"],
    ]
    return any(
        _direct_commands(block) == expected
        for block in _unconditional_ci_run_blocks(ci_text)
    )


def _active_ci_commands(ci_text: str) -> set[tuple[str, ...]]:
    commands: set[tuple[str, ...]] = set()
    for block in _unconditional_ci_run_blocks(ci_text):
        parsed = _direct_commands(block)
        if parsed is not None:
            commands.update(tuple(command) for command in parsed)
    return commands


def _bash_array_values(text: str, name: str) -> list[str] | None:
    """Read one top-level Bash array assignment by its actual variable name."""

    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == f"{name}=("]
    if len(starts) != 1:
        return None
    values: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() == ")":
            return values
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            return None
        values.extend(tokens)
    return None


def _trace_preflight_commands(text: str) -> tuple[int, list[tuple[str, ...]]]:
    """Execute preflight with Python/Git shims and return Python argv traces."""

    with tempfile.TemporaryDirectory(prefix="vllm-preflight-trace-") as temporary:
        root = Path(temporary)
        script = root / "scripts/agent-preflight.sh"
        script.parent.mkdir(parents=True)
        script.write_text(text, encoding="utf-8")
        script.chmod(0o700)
        (root / ".agents").mkdir()
        (root / ".agents/NOW.md").write_text("trace-only\n", encoding="utf-8")
        shim_dir = root / "shim"
        shim_dir.mkdir()
        trace = root / "python.trace"
        python = shim_dir / "python3"
        python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\0' \"$@\" >> \"$VLLM_REGISTRATION_TRACE\"\n"
            "printf '\\0' >> \"$VLLM_REGISTRATION_TRACE\"\n",
            encoding="utf-8",
        )
        python.chmod(0o700)
        git = shim_dir / "git"
        git.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        git.chmod(0o700)
        environment = os.environ.copy()
        environment["PATH"] = f"{shim_dir}{os.pathsep}{environment.get('PATH', '')}"
        environment["VLLM_REGISTRATION_TRACE"] = str(trace)
        result = subprocess.run(
            ["bash", str(script), "--quiet", "--no-require-role"],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        raw = trace.read_bytes() if trace.exists() else b""
    invocations = []
    for record in raw.split(b"\0\0"):
        if not record:
            continue
        invocations.append(
            tuple(token.decode("utf-8") for token in record.split(b"\0") if token)
        )
    return result.returncode, invocations


def _preflight_execution_errors(text: str) -> list[str]:
    returncode, invocations = _trace_preflight_commands(text)
    errors: list[str] = []
    checker = ("scripts/check-test-registration.py",)
    suite = ("tests/scripts/test_check_test_registration.py",)
    if invocations.count(checker) != 1:
        errors.append("preflight does not execute CHECKERS through its checker loop")
    if invocations.count(suite) != 1:
        errors.append("preflight does not execute SUITES through its suite loop")
    if returncode != 0:
        errors.append(f"instrumented preflight execution failed with rc={returncode}")
    return errors


def wiring_errors(preflight_text: str, ci_text: str) -> list[str]:
    """Return missing preflight/CI wiring for this checker and its mutations."""

    preflight_source = preflight_text
    preflight_text = _without_line_comments(preflight_source)
    errors: list[str] = []
    checkers = _bash_array_values(preflight_text, "CHECKERS")
    suites = _bash_array_values(preflight_text, "SUITES")
    if checkers is None or "check-test-registration" not in checkers:
        errors.append("check-test-registration is missing from preflight CHECKERS")
    if suites is None or "test_check_test_registration" not in suites:
        errors.append("test_check_test_registration is missing from preflight SUITES")
    errors.extend(_preflight_execution_errors(preflight_source))
    active_ci_commands = _active_ci_commands(ci_text)
    if ("python3", "scripts/check-test-registration.py") not in active_ci_commands:
        errors.append("check-test-registration is missing from the explicit CI checker step")
    if (
        "python3",
        "tests/scripts/test_check_test_registration.py",
    ) not in active_ci_commands:
        errors.append("test_check_test_registration is missing from the CI mutation suite")
    if not _ci_has_active_guard_step(ci_text):
        errors.append(
            "CI guard step must contain the checker and mutation suite as direct active commands"
        )
    for suite, command in sorted(REQUIRED_SUITE_REGISTRATIONS.items()):
        if suites is None or suite not in suites:
            errors.append(f"pinned suite {suite} is missing from preflight SUITES")
        if ("python3", command) not in active_ci_commands:
            errors.append(f"pinned suite {suite} is missing from the CI suite lane")
    return errors


def _method_calls(method: ast.AST) -> list[ast.Call]:
    return [call for call in ast.walk(method) if isinstance(call, ast.Call)]


def mutation_suite_integrity_errors(
    source: str,
    manifest_text: str | None = None,
    *,
    manifest_path: Path = MUTATION_MANIFEST,
) -> list[str]:
    """Pin the suite/manifest pair from this independent production layer."""

    errors: list[str] = []
    if manifest_path.resolve() != MUTATION_MANIFEST.resolve():
        errors.append("mutation suite must use the canonical manifest path")
    if manifest_text is None:
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            return errors + [f"mutation manifest is unreadable: {exc}"]
    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if digest != MUTATION_MANIFEST_SHA256:
        errors.append("mutation manifest differs from the production-pinned digest")
    manifest = {
        line
        for line in manifest_text.splitlines()
        if line and not line.startswith("#")
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return errors + [f"mutation suite is not valid Python: {exc}"]
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    actual = {name for name in methods if name.startswith("test_M")}
    if actual != manifest:
        errors.append("mutation inventory differs from fixed manifest")

    for name in sorted(actual & manifest):
        calls = {
            call.func.attr
            for call in _method_calls(methods[name])
            if isinstance(call.func, ast.Attribute)
        }
        if not {
            "assert_error",
            "assert_wiring_error",
            "assert_server_guard_error",
            "assertTrue",
        } & calls:
            errors.append(f"{name} has no semantic outcome assertion")

    m42 = methods.get("test_M42_byte_identical_alternate_manifest_path_fails")
    m42_path_assertion = False
    if m42 is not None:
        for call in _method_calls(m42):
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and call.func.attr == "assertEqual"
                and len(call.args) == 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "errors"
                and isinstance(call.args[1], ast.List)
                and len(call.args[1].elts) == 1
                and isinstance(call.args[1].elts[0], ast.Constant)
                and call.args[1].elts[0].value
                == "mutation suite must use the canonical manifest path"
            ):
                continue
            m42_path_assertion = True
            break
    if not m42_path_assertion:
        errors.append(
            "test_M42 must assert the exact canonical-manifest path diagnostic"
        )

    for name, production_call in {
        "assert_error": "registration_errors",
        "assert_wiring_error": "wiring_errors",
        # Pinned exactly like its two siblings: a helper that stopped calling
        # the production function would leave its mutation cases asserting
        # against nothing while still reading as covered (#1883).
        "assert_server_guard_error": "server_guard_errors",
    }.items():
        method = methods.get(name)
        if method is None:
            errors.append(f"{name} helper is missing")
            continue
        calls = _method_calls(method)
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "assertTrue"
            for call in calls
        ):
            errors.append(f"{name} has no direct semantic assertion")
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "mod"
            and call.func.attr == production_call
            for call in calls
        ):
            errors.append(f"{name} does not call {production_call}")

    wrapper = methods.get("_suite_integrity_errors")
    wrapper_ok = False
    if wrapper is not None:
        for call in _method_calls(wrapper):
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "mod"
                and call.func.attr == "mutation_suite_integrity_errors"
            ):
                continue
            path_keywords = [kw for kw in call.keywords if kw.arg == "manifest_path"]
            wrapper_ok = (
                len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "source"
                and len(path_keywords) == 1
                and isinstance(path_keywords[0].value, ast.Attribute)
                and isinstance(path_keywords[0].value.value, ast.Name)
                and path_keywords[0].value.value.id == "mod"
                and path_keywords[0].value.attr == "MUTATION_MANIFEST"
            )
    if not wrapper_ok:
        errors.append("suite integrity wrapper does not use the canonical manifest")

    integrity = methods.get("test_suite_integrity_contract_is_pinned")
    if integrity is None:
        errors.append("required mutation-suite integrity method is missing")
    else:
        calls = _method_calls(integrity)
        called_wrapper = any(
            isinstance(call.func, ast.Name) and call.func.id == "_suite_integrity_errors"
            for call in calls
        )
        asserted = any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "assertEqual"
            for call in calls
        )
        if not (called_wrapper and asserted):
            errors.append("mutation-suite integrity method has no pinned assertion")
    return errors


# Every option in this tree that can put a `FetchContent_Declare` on the
# CONFIGURE path, so that the CPU-only configure below reaches no network at
# all.  Three sites, three guards:
#   * `VLLM_CPP_CUTLASS_FETCH`  -> cutlass,     CMakeLists.txt:577  (default OFF)
#   * `VLLM_CPP_WITH_DIARIZATION` -> parakeet.cpp, CMakeLists.txt:1595 (default ON)
#   * `VLLM_CPP_BUILD_BORINGSSL` -> boringssl,  CMakeLists.txt:2740  (default OFF)
#
# The middle one is the defect this list answers.  It is the only one that
# defaults ON, and a gate whose configure dies on a failed git checkout then
# reports the CONSEQUENCE -- 'missing required test target
# test_device_selection in configured codemodel' -- about a target that was
# configured all along, naming the tree instead of the host.  The other two are
# listed so that a fourth site cannot join them silently: the suite counts the
# `FetchContent_Declare` calls in the tree and fails when that count and this
# tuple's length disagree.
#
# Nothing this gate measures needs any of the three.  The required targets are
# the device selector and two labelled music3 arms, none of which references
# `VLLM_WITH_DIARIZATION`, CUTLASS headers, or the HF transport.
NETWORK_FETCH_OPTIONS = (
    "VLLM_CPP_CUTLASS_FETCH",
    "VLLM_CPP_WITH_DIARIZATION",
    "VLLM_CPP_BUILD_BORINGSSL",
)

# The CPU-only configure this gate measures the shipped tree with.  Pinned here
# rather than spelled at the call site so a test can assert what it asks for.
CPU_ONLY_CONFIGURE_ARGS = (
    "-DVLLM_CPP_CUDA=OFF",
    "-DVLLM_CPP_HIP=OFF",
    "-DVLLM_CPP_VULKAN=OFF",
    "-DVLLM_CPP_METAL=OFF",
    "-DVLLM_CPP_MLX=OFF",
    "-DVLLM_CPP_TRITON=OFF",
    *(f"-D{option}=OFF" for option in NETWORK_FETCH_OPTIONS),
    "-DVLLM_CPP_BUILD_TESTS=ON",
    "-DVLLM_CPP_BUILD_EXAMPLES=OFF",
    "-DVLLM_CPP_SERVER=OFF",
    "-DCMAKE_BUILD_TYPE=Release",
)


def check_tree(root: Path = ROOT) -> list[str]:
    paths = {
        "tests/CMakeLists.txt": root / "tests/CMakeLists.txt",
        "scripts/agent-preflight.sh": root / "scripts/agent-preflight.sh",
        ".github/workflows/ci.yml": root / ".github/workflows/ci.yml",
        "tests/scripts/test_check_test_registration.py": root
        / "tests/scripts/test_check_test_registration.py",
        "tests/scripts/check_test_registration_mutations.txt": root
        / "tests/scripts/check_test_registration_mutations.txt",
    }
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        return [f"required registration-guard input is missing: {path}" for path in missing]

    configure_args = list(CPU_ONLY_CONFIGURE_ARGS)
    with tempfile.TemporaryDirectory(prefix="vllm-registration-tree-") as temporary:
        build_dir = Path(temporary) / "build"
        registration = _configured_contract_errors(
            root,
            build_dir,
            {
                target: f"tests/{source}"
                for target, source in REQUIRED_TESTS.items()
            },
            configure_args,
        )
        # The labelled gate is registered unconditionally, so the CPU-only
        # configure above already knows about it: no accelerator is needed to
        # read what `ctest -L gpu` would select.
        registration += _configured_label_errors(
            root, build_dir, REQUIRED_LABEL_SELECTIONS, configure_args
        )
    integrity = mutation_suite_integrity_errors(
        paths["tests/scripts/test_check_test_registration.py"].read_text(encoding="utf-8"),
        paths["tests/scripts/check_test_registration_mutations.txt"].read_text(encoding="utf-8"),
        manifest_path=paths["tests/scripts/check_test_registration_mutations.txt"],
    )
    # STATIC, and deliberately so: the failure it names lands at LINK time on a
    # `-DVLLM_CPP_SERVER=OFF` build, which the configure above reaches and
    # cannot see, and which costs a full `libvllm` compile to reproduce.
    guard = server_guard_errors(root)
    return registration + guard + wiring_errors(
        paths["scripts/agent-preflight.sh"].read_text(encoding="utf-8"),
        paths[".github/workflows/ci.yml"].read_text(encoding="utf-8"),
    ) + integrity


def main() -> int:
    errors = check_tree()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    selection = ", ".join(
        f"-L {label} -> {len(names)} [{', '.join(sorted(names))}]"
        for label, names in sorted(REQUIRED_LABEL_SELECTIONS.items())
    )
    print(
        "OK: required regression tests have executable + CTest registration, "
        f"the configured tree matches the pinned label selection ({selection}), "
        f"no test target outside `if({SERVER_GUARD_CONDITION})` reaches a "
        "translation unit compiled only inside it, "
        "and the guard is wired into preflight/CI "
        f"together with {len(REQUIRED_SUITE_REGISTRATIONS)} pinned suite(s) "
        f"[{', '.join(sorted(REQUIRED_SUITE_REGISTRATIONS))}] in BOTH lanes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
