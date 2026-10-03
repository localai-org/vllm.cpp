#!/usr/bin/env python3
"""Fail closed when the native Windows server contract regresses.

This is deliberately a source-contract gate on non-Windows hosts. Native MSVC
compile and runtime evidence are separate release gates; this checker prevents
the known POSIX-only or baseline-contaminating shapes from reaching them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_CPP = (
    "src/vllm/entrypoints/openai/server_main.cpp",
    "src/vllm/platform/process.cpp",
    "src/vllm/platform/console_shutdown.cpp",
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
    "src/vllm/v1/kv_offload/fs_io.cpp",
)

POSIX_PATTERNS = (
    r"^\s*#\s*include\s*<(?:arpa/inet|netdb|netinet/[^>]+|sys/socket|sys/stat|sys/types|sys/wait|unistd|fcntl)\.h>",
    r"(?<![A-Za-z0-9_.>])(?:fork|execvp|waitpid|pipe|read|write|open|close|fsync|pread|pwrite|getpid|stat)\s*\(",
)


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
UNSUPPORTED_TIER_FILTER = (
    "--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"
)
UNSUPPORTED_TIER_DIAGNOSTIC = "unknown x86 ISA tier 'amx'"
WINDOWS_EXCLUDED_SOURCE = "src/vt/cuda/nvfp4_persistent_cache.cpp"
POWERSHELL_AUDIT_PATH_ENV = "VLLM_CPP_POWERSHELL_AUDIT_PATH"


def _has_central_msvc_nominmax(cmake: str) -> bool:
    """Return whether root CMake defines NOMINMAX before creating targets."""
    active = re.sub(r"(?m)#.*$", "", cmake)
    first_target = re.search(
        r"(?im)^\s*(?:add_library|add_executable|add_subdirectory)\s*\(",
        active,
    )
    contract = re.search(
        r"(?ims)^\s*if\s*\(\s*MSVC\s*\)\s*$"
        r"(?:(?!^\s*endif\b).)*?"
        r"^\s*add_compile_definitions\s*\([^)]*\bNOMINMAX\b[^)]*\)",
        active,
    )
    return contract is not None and (
        first_target is None or contract.start() < first_target.start()
    )


def _local_nominmax_definitions(active_source: str) -> list[tuple[int, bool]]:
    """Return active NOMINMAX definition lines and whether absence-guarded."""
    definitions: list[tuple[int, bool]] = []
    absence_guards: list[bool] = []
    for number, line in enumerate(active_source.splitlines(), 1):
        match = re.match(r"\s*#\s*(\w+)(.*)$", line)
        if match is None:
            continue
        kind, tail = match.group(1), match.group(2).strip()
        guard = bool(
            re.fullmatch(r"NOMINMAX", tail) if kind == "ifndef" else
            kind in {"if", "elif"} and re.fullmatch(
                r"!\s*defined\s*(?:\(\s*NOMINMAX\s*\)|NOMINMAX)", tail
            )
        )
        if kind in {"if", "ifdef", "ifndef"}:
            absence_guards.append(guard)
        elif kind in {"else", "elif"}:
            if absence_guards:
                absence_guards[-1] = guard if kind == "elif" else False
        elif kind == "endif":
            if absence_guards:
                absence_guards.pop()
        elif kind == "define" and re.match(r"NOMINMAX\b", tail):
            definitions.append((number, any(absence_guards)))
    return definitions


def windows_excluded_sources(root: Path) -> set[str]:
    """Return sources proven by CMake to be absent specifically on WIN32."""
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    blocks = re.findall(r"(?ms)^if\(NOT WIN32\)\s*$\n(.*?)^endif\(\)\s*$", text)
    matches = [block for block in blocks if WINDOWS_EXCLUDED_SOURCE in block]
    if len(matches) != 1 or not re.search(
            rf"target_sources\s*\(\s*vllm\s+PRIVATE\s+{re.escape(WINDOWS_EXCLUDED_SOURCE)}\s*\)",
            matches[0]):
        return set()
    return {WINDOWS_EXCLUDED_SOURCE}


def _project_header_closure(root: Path, sources: set[str],
                            include_roots: set[Path]) -> set[str]:
    """Resolve every project-local quoted include reachable from sources."""
    closure = set(sources)
    pending = list(sources)
    include_re = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
    while pending:
        relative = pending.pop()
        source = root / relative
        if not source.is_file() or source.suffix.lower() not in CPP_SUFFIXES:
            continue
        for included in include_re.findall(source.read_text(encoding="utf-8")):
            candidates = [source.parent / included]
            candidates.extend(directory / included for directory in include_roots)
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                    found = resolved.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                if resolved.is_file() and found not in closure:
                    closure.add(found)
                    pending.append(found)
                    break
    return closure


def _is_fetch_content_source(relative: str) -> bool:
    """True for a source under a FetchContent populate directory.

    `_deps/` is where CMake Materializes third-party dependencies; a build
    directory configured INSIDE the source tree (the Windows release build's
    `build-pr-windows-vulkan/`) therefore places vendored code such as the
    parakeet_cpp ggml under the root, and the codemodel walk used to carry it
    into the scan. Those files are not this project's; the portability
    contract audits OUR sources reaching Windows (see
    ISSUE-LOCAL-01M3Z28HG59E1W8DYC2JJ9EVDD).
    """
    return "_deps" in Path(relative).parts


def _load_codemodel_sources(root: Path, build_dir: Path) -> set[str]:
    replies = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(replies.glob("index-*.json"))
    if not indexes:
        raise RuntimeError(f"{build_dir}: CMake file-api codemodel reply is missing")
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    reply = index.get("reply", {})
    codemodel_entry = next(
        (value for key, value in reply.items() if key.startswith("codemodel-v2")),
        None,
    )
    if not isinstance(codemodel_entry, dict) or "jsonFile" not in codemodel_entry:
        raise RuntimeError(f"{build_dir}: CMake codemodel-v2 reply is missing")
    codemodel = json.loads(
        (replies / codemodel_entry["jsonFile"]).read_text(encoding="utf-8")
    )
    configurations = codemodel.get("configurations", [])
    if not configurations:
        raise RuntimeError(f"{build_dir}: CMake codemodel has no configuration")
    targets: dict[str, dict] = {}
    target_files: dict[str, Path] = {}
    for target in configurations[0].get("targets", []):
        target_id = target.get("id")
        json_file = target.get("jsonFile")
        if target_id and json_file:
            targets[target_id] = target
            target_files[target_id] = replies / json_file
    roots = [target_id for target_id, value in targets.items()
             if value.get("name") == "server"]
    if len(roots) != 1:
        raise RuntimeError(
            f"{build_dir}: expected one shipped server target, found {len(roots)}"
        )
    pending = roots[:]
    seen: set[str] = set()
    sources: set[str] = set()
    include_roots = {root, root / "include", root / "src"}
    while pending:
        target_id = pending.pop()
        if target_id in seen:
            continue
        seen.add(target_id)
        data = json.loads(target_files[target_id].read_text(encoding="utf-8"))
        for group in data.get("compileGroups", []):
            for include in group.get("includes", []):
                path = Path(include.get("path", ""))
                absolute = path if path.is_absolute() else root / path
                try:
                    resolved = absolute.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                include_roots.add(resolved)
        for source in data.get("sources", []):
            path = Path(source.get("path", ""))
            absolute = path if path.is_absolute() else root / path
            try:
                sources.add(absolute.resolve().relative_to(root).as_posix())
            except ValueError:
                pass
        pending.extend(
            dependency["id"] for dependency in data.get("dependencies", [])
            if dependency.get("id") in targets
        )
    sources -= windows_excluded_sources(root)
    sources = {s for s in sources if not _is_fetch_content_source(s)}
    return _project_header_closure(root, sources, include_roots)


def shipped_server_sources(root: Path, build_dir: Path | None,
                           source_manifest: Path | None = None) -> set[str]:
    if source_manifest is not None:
        data = json.loads(source_manifest.read_text(encoding="utf-8"))
        sources = {str(item) for item in data.get("sources", [])}
        sources -= windows_excluded_sources(root)
        sources = {s for s in sources if not _is_fetch_content_source(s)}
        return _project_header_closure(root, sources, {root, root / "include", root / "src"})
    if build_dir is not None:
        return _load_codemodel_sources(root, build_dir.resolve())
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to derive the shipped-server source set")
    with tempfile.TemporaryDirectory(prefix="vllm-windows-codemodel-") as temp:
        generated = Path(temp)
        query = generated / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "codemodel-v2").touch()
        command = [
            "cmake", "-S", str(root), "-B", str(generated), "-G", "Ninja",
            "-DVLLM_CPP_BUILD_TESTS=OFF", "-DVLLM_CPP_BUILD_EXAMPLES=ON",
            "-DVLLM_CPP_SERVER=ON", "-DVLLM_CPP_CUDA=OFF",
            "-DVLLM_CPP_HIP=OFF", "-DVLLM_CPP_METAL=OFF",
            "-DVLLM_CPP_MLX=OFF", "-DVLLM_CPP_TRITON=OFF",
            "-DVLLM_CPP_VULKAN=OFF", "-DCMAKE_BUILD_TYPE=Release",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "CMake configure failed while deriving shipped-server sources:\n" +
                result.stdout + result.stderr
            )
        return _load_codemodel_sources(root, generated)


def windows_possible_lines(text: str):
    """Yield (line number, line) for branches that can compile on Windows."""
    possible = True
    # (parent possible, condition known, condition possible on Windows)
    stack: list[tuple[bool, bool, bool]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if re.match(r"#\s*ifdef\s+_WIN32\b", stripped):
            stack.append((possible, True, True))
            possible = possible and True
        elif re.match(r"#\s*ifndef\s+_WIN32\b", stripped):
            stack.append((possible, True, False))
            possible = False
        elif re.match(r"#\s*if\s+defined\s*\(\s*_WIN32\s*\)", stripped):
            stack.append((possible, True, True))
            possible = possible and True
        elif re.match(r"#\s*if\b.*!\s*defined\s*\(\s*_WIN32\s*\)", stripped):
            stack.append((possible, True, False))
            possible = False
        elif (re.match(r"#\s*if\b", stripped) and
              re.search(r"\b(?:__GNUC__|__clang__)\b", stripped) and
              not re.search(r"\b_MSC_VER\b", stripped)):
            stack.append((possible, True, False))
            possible = False
        elif (re.match(r"#\s*if(?:n?def)?\b", stripped) and
              re.search(r"\b(?:__unix__|__APPLE__)\b", stripped) and
              not re.search(r"\b_WIN32\b", stripped)):
            stack.append((possible, True, False))
            possible = False
        elif re.match(r"#\s*if(?:n?def)?\b", stripped):
            stack.append((possible, False, possible))
        elif re.match(r"#\s*else\b", stripped) and stack:
            parent, known, condition = stack[-1]
            possible = parent and (not condition if known else True)
        elif re.match(r"#\s*elif\b", stripped) and stack:
            parent, _, _ = stack[-1]
            if (re.search(r"\b(?:__GNUC__|__clang__)\b", stripped) and
                    not re.search(r"\b_MSC_VER\b", stripped)):
                possible = False
            else:
                possible = parent
        elif re.match(r"#\s*endif\b", stripped) and stack:
            parent, _, _ = stack.pop()
            possible = parent
        elif possible:
            yield number, line


C_FAMILY_COMPILE_LANGUAGES = {"C", "CXX"}


def _generator_expression_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) for every balanced `$<...>` generator expression."""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    index = 0
    while index < len(text):
        if text.startswith("$<", index):
            stack.append(index)
            index += 2
            continue
        if text[index] == ">" and stack:
            spans.append((stack.pop(), index + 1))
        index += 1
    return spans


def msvc_cxx_flag_text(cmake_text: str) -> str:
    """Return the flag text that can reach an MSVC C/C++ translation unit.

    Two things are removed, both of which satisfied the old substring test
    without compiling anything (#774):

    * `#` comments. `CMakeLists.txt` says `/W4 /WX` in prose, and a policy
      cannot be satisfied by prose.
    * Generator expressions whose `COMPILE_LANGUAGE` names only languages
      outside C/C++. `$<$<COMPILE_LANGUAGE:OBJCXX>:/WX>` is the Metal backend,
      which never compiles under MSVC, so it answers for nothing here. A
      generator expression naming no language at all is KEPT: it does reach
      C/C++.

    Spans are blanked rather than cut so reported offsets stay meaningful,
    matching `without_set_source_properties`.
    """
    active = re.sub(r"(?m)#.*$", "", cmake_text)
    out = list(active)
    for start, end in _generator_expression_spans(active):
        languages: set[str] = set()
        for mention in re.finditer(
            r"COMPILE_LANGUAGE\s*:\s*([^>]*)", active[start:end]
        ):
            languages |= {
                name.strip().upper()
                for name in mention.group(1).split(",")
                if name.strip()
            }
        if languages and not (languages & C_FAMILY_COMPILE_LANGUAGES):
            out[start:end] = " " * (end - start)
    return "".join(out)


def has_msvc_flag(text: str, flag: str) -> bool:
    """Return whether `flag` appears as a WHOLE token, not as a substring.

    `"/WX" in "/WX-"` is True and `/WX-` DISABLES warnings-as-errors; `"/W4" in
    "/W44996"` is True and `/W44996` sets one warning to level 4 rather than
    raising the level. Both satisfied the old test (#774). Matching stays
    case-sensitive because `cl` is: `/w` and `/W4` are different flags.
    """
    return re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])", text
    ) is not None


def without_set_source_properties(text: str) -> str:
    """Remove balanced set_source_files_properties commands."""
    lowered = text.lower()
    needle = "set_source_files_properties("
    out = list(text)
    start = 0
    while True:
        at = lowered.find(needle, start)
        if at < 0:
            break
        depth = 0
        end = at
        while end < len(text):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        out[at:end] = " " * (end - at)
        start = end
    return "".join(out)


def project_targets(text: str) -> set[str]:
    """Targets this project DECLARES, by `add_library` / `add_executable`.

    A target that appears nowhere in this set came from somewhere else --
    `FetchContent_MakeAvailable` is the case that matters here -- so this
    project's warning policy was never applied to it and cannot be negated on
    it.
    """
    return {
        match.group(1)
        for match in re.finditer(
            r"(?im)^\s*add_(?:library|executable)\s*\(\s*([A-Za-z0-9_.:+-]+)", text
        )
    }


def foreach_bindings(text: str) -> dict[str, set[str]]:
    """Map each `foreach(VAR a b c)` loop variable to the names it takes."""
    bindings: dict[str, set[str]] = {}
    for match in re.finditer(
        r"(?im)^\s*foreach\s*\(\s*([A-Za-z0-9_]+)([^)]*)\)", text
    ):
        bindings.setdefault(match.group(1), set()).update(
            token for token in match.group(2).split() if token
        )
    return bindings


def _target_is_foreign(token: str, targets: set[str],
                       bindings: dict[str, set[str]]) -> bool:
    """True only when the token PROVABLY names targets this project never declares.

    The fail-safe direction is deliberate: anything unresolved stays in scope,
    so an unbound `${...}` -- which could name a project target -- still answers
    for the policy.
    """
    variable = re.fullmatch(r"\$\{([A-Za-z0-9_]+)\}", token)
    if variable:
        names = bindings.get(variable.group(1))
        if not names:
            return False
        return all(name not in targets for name in names)
    if re.fullmatch(r"[A-Za-z0-9_.:+-]+", token):
        return token not in targets
    return False


def without_foreign_target_compile_options(text: str) -> str:
    """Blank `target_compile_options()` calls on targets this project does not own.

    The MSVC `/W4 /WX` policy is asserted on the flags that reach THIS project's
    C/C++ compile. `target_compile_options(<target> PRIVATE ...)` reaches only
    `<target>`, so a `/w` on a vendored, fetched target is not a negation of the
    policy -- it is the vendored code being kept off this project's -Werror
    path, which `CMakeLists.txt` says in a comment beside it.

    Reading that `/w` as project-wide is what made `windows-msvc-cpu` red on
    `main` and on every pull request (#1649), and it also red
    `test_real_tree_msvc_warning_policy_reaches_the_cxx_compile` in this
    checker's own suite.

    Spans are blanked rather than cut so reported offsets stay meaningful,
    matching `without_set_source_properties` and `msvc_cxx_flag_text`.
    """
    targets = project_targets(text)
    bindings = foreach_bindings(text)
    needle = "target_compile_options("
    lowered = text.lower()
    out = list(text)
    start = 0
    while True:
        at = lowered.find(needle, start)
        if at < 0:
            break
        depth = 0
        end = at
        while end < len(text):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        argument = text[at + len(needle):end].strip()
        first = argument.split(None, 1)[0] if argument else ""
        if first and _target_is_foreign(first, targets, bindings):
            out[at:end] = " " * (end - at)
        start = end
    return "".join(out)


def source_properties(text: str, source: str) -> str:
    """Return the balanced set_source_files_properties command for source."""
    lowered = text.lower()
    needle = "set_source_files_properties("
    start = 0
    while True:
        at = lowered.find(needle, start)
        if at < 0:
            return ""
        depth = 0
        end = at
        while end < len(text):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        command = text[at:end]
        if source in command:
            return command
        start = end


def require_file(root: Path, relative: str, errors: list[str]) -> str:
    path = root / relative
    if not path.is_file():
        errors.append(f"{relative}: required Windows portability surface is missing")
        return ""
    return path.read_text(encoding="utf-8")


def without_cpp_comments(text: str) -> str:
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    return re.sub(r"//.*", "", text)


_CPP_INERT = re.compile(
    r'''(?P<block>/\*.*?\*/)|'''
    r'''(?P<line>//[^\n]*)|'''
    r'''(?P<raw>R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)")|'''
    r'''(?P<string>"(?:\\.|[^"\\])*")|'''
    r'''(?P<char>'(?:\\.|[^'\\])*')''',
    re.DOTALL,
)


def without_cpp_comments_and_literals(text: str) -> str:
    """Blank inert C++ text while preserving offsets and line boundaries."""
    return _CPP_INERT.sub(
        lambda match: "".join("\n" if char == "\n" else " "
                              for char in match.group(0)),
        text,
    )


# A `noreturn` attribute on a function whose return type is not `void` is MSVC
# C4646, which `/W4 /WX` promotes to C2220 and which fails the whole project's
# compile. GCC and Clang accept it silently, so no POSIX lane can see it, and
# `windows-msvc-cpu` / `windows-msvc-vulkan` are pull-request-only jobs with no
# `main` baseline (#584). #1829 reached `main` through exactly that hole.
#
# FOUR spellings carry the same meaning to a Windows toolchain, and the defect
# is the SHAPE rather than the syntax that states it. C4646's own text names
# `__declspec(noreturn)`, so a detector reading only the C++11 attribute would
# miss the one spelling the warning is named after:
#
#   [[noreturn]], [[gnu::noreturn]]   C++11 attribute, vendor namespace optional
#   __declspec(noreturn)              MSVC's own spelling, and C4646's wording
#   __attribute__((noreturn))         clang-cl and MinGW
#   _Noreturn                         C11, which MSVC accepts in C mode
#
# The tree carries zero occurrences of the last three today. They are matched so
# that the rule covers the class it claims to cover, because a detector whose
# claim is wider than its regex is this repository's signature failure.
_NORETURN_ATTRIBUTE = re.compile(
    r"""\[\[\s*(?:(?:gnu|clang|msvc)\s*::\s*)?(?:noreturn|__noreturn__)\s*\]\]"""
    r"""|__declspec\s*\(\s*noreturn\s*\)"""
    r"""|__attribute__\s*\(\s*\(\s*(?:__)?noreturn(?:__)?\s*\)\s*\)"""
    r"""|\b_Noreturn\b"""
)
# An attribute construct that the return-type read must step OVER rather than
# stop at. `__declspec(dllexport)` -- what `VLLM_API` in `include/vllm.h`
# expands to on Windows -- carries parentheses of its own, and a scan that took
# those for the declarator's would read the export macro as the function name.
# Cited by macro name rather than by line: a line anchor in a comment goes stale
# the next time that header gains a declaration.
_ATTRIBUTE_OPENERS = ("__declspec", "__attribute__")
_TEMPLATE_PREFIX = re.compile(r"\s*template\s*<")
_LANGUAGE_LINKAGE = re.compile(r'extern\s+"C(?:\+\+)?"')
# Everything a declaration may carry BEFORE its return type. Deliberately no
# export/attribute-macro allowlist: an unknown all-caps token stays part of the
# return type and reds this gate, which is the fail-closed direction this
# checker is built on. No `noreturn` in this tree carries such a macro.
# `_Noreturn` is BOTH a spelling matched above and a token stripped here: the
# match consumes only the first one, so a second `_Noreturn` written after it
# must not leak into the return type.
_DECL_SPECIFIERS = frozenset({
    "static", "inline", "constexpr", "consteval", "extern", "virtual",
    "explicit", "friend", "mutable", "thread_local", "_Noreturn",
})
# The first-party C-family tree. NOT `shipped_server_sources`: `windows-msvc-cpu`
# builds the tests too, so a test translation unit carrying this shape reds the
# same lane, and unlike the POSIX rules there is no guarded/unguarded judgement
# to make -- the shape is ill-formed under MSVC wherever it appears. `.cu`/`.cuh`
# are in because nvcc hands host code to `cl.exe` on Windows.
NORETURN_SCAN_DIRS = (
    "src", "include", "tests", "examples", "tools", "benchmarks", "scripts",
)
NORETURN_SCAN_SUFFIXES = CPP_SUFFIXES | {".cu", ".cuh", ".hip", ".inc"}
NORETURN_SCAN_EXCLUDED_DIR = "third_party"


def _attribute_span(text: str, index: int) -> int | None:
    """End offset of the attribute construct starting at `index`, else `None`.

    `[[...]]` closes on its bracket pair. `__declspec(...)` and
    `__attribute__((...))` close on a BALANCED paren scan, because
    `__attribute__((visibility("default")))` nests and a non-greedy regex would
    leave a stray `)` behind to be read as part of a return type.
    """
    if text.startswith("[[", index):
        close = text.find("]]", index + 2)
        return close + 2 if close != -1 else None
    for opener in _ATTRIBUTE_OPENERS:
        if not text.startswith(opener, index):
            continue
        cursor = index + len(opener)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            return None
        depth = 0
        while cursor < len(text):
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        return None
    return None


def _strip_attributes(head: str) -> str:
    """Blank every attribute construct in `head`, preserving its length."""
    out: list[str] = []
    index = 0
    while index < len(head):
        end = _attribute_span(head, index)
        if end is not None:
            out.append(" " * (end - index))
            index = end
            continue
        out.append(head[index])
        index += 1
    return "".join(out)


def _strip_template_prefix(head: str) -> str:
    """Drop a leading `template <...>`, which is not part of the return type."""
    match = _TEMPLATE_PREFIX.match(head)
    if match is None:
        return head
    depth = 0
    for index in range(match.end() - 1, len(head)):
        if head[index] == "<":
            depth += 1
        elif head[index] == ">":
            depth -= 1
            if depth == 0:
                return head[index + 1:]
    return head


def _statement_start(active: str, index: int) -> int:
    """Offset just past the `;`, `{`, `}` or label `:` that precedes `index`.

    `::` is never a label, so a qualified return type stays whole. Angle
    brackets are deliberately NOT tracked here: `_strip_template_prefix` removes
    the one construct that needs them, and tracking them would let an unbalanced
    `>` from a comparison swallow the statement boundary.
    """
    cursor = index - 1
    while cursor >= 0:
        char = active[cursor]
        if char in ";{}":
            return cursor + 1
        if char == ":" and not (
            (cursor and active[cursor - 1] == ":")
            or (cursor + 1 < len(active) and active[cursor + 1] == ":")
        ):
            return cursor + 1
        cursor -= 1
    return 0


def _declarator_head(active: str, start: int) -> tuple[str | None, int]:
    """Text between a LEADING attribute and the declarator's `(`, plus that offset.

    Angle-bracket depth is tracked so a template argument list cannot end the
    scan early, and an attribute construct is stepped over whole so that its own
    parentheses cannot be taken for the declarator's. Reaching `;`, `{` or `}`
    first means the attribute does not lead a function declaration, and the
    caller then tries the trailing position before reporting the site unparsed.
    """
    depth = 0
    index = start
    while index < len(active):
        skipped = _attribute_span(active, index)
        if skipped is not None:
            index = skipped
            continue
        char = active[index]
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif depth == 0:
            if char == "(":
                return active[start:index], index
            if char in ";{}":
                return None, index
        index += 1
    return None, len(active)


def _trailing_declarator(active: str, attribute_start: int) -> tuple[str | None, int]:
    """Head and declarator `(` for an attribute written AFTER the declarator.

    `void die(int code) __attribute__((noreturn));` is the ordinary GNU
    placement, and read forwards it finds `;` before any `(`. Reported as
    unparsed it would be a false red on correct `void` code, so the declaration
    is read backwards instead: the `)` immediately before the attribute closes
    the parameter list, and the head runs from the statement start to its `(`.
    """
    index = attribute_start - 1
    while index >= 0 and active[index].isspace():
        index -= 1
    if index < 0 or active[index] != ")":
        return None, -1
    depth = 0
    while index >= 0:
        if active[index] == ")":
            depth += 1
        elif active[index] == "(":
            depth -= 1
            if depth == 0:
                return active[_statement_start(active, index):index], index
        index -= 1
    return None, -1


def _declaration_words(head: str) -> list[str]:
    head = _strip_attributes(head)
    head = _LANGUAGE_LINKAGE.sub(" ", head)
    head = _strip_template_prefix(head)
    # `*` and `&` become their own tokens so `void*` cannot read as `void`.
    head = head.replace("*", " * ").replace("&", " & ")
    return [word for word in head.split() if word not in _DECL_SPECIFIERS]


def _has_trailing_void_return(active: str, open_paren: int) -> bool:
    depth = 0
    for index in range(open_paren, len(active)):
        if active[index] == "(":
            depth += 1
        elif active[index] == ")":
            depth -= 1
            if depth == 0:
                tail = active[index + 1:index + 200]
                stop = re.search(r"[;{]", tail)
                if stop is not None:
                    tail = tail[:stop.start()]
                return re.search(r"->\s*void\s*$", tail.strip()) is not None
    return False


def noreturn_nonvoid_sites(source: str) -> list[tuple[int, str, str]]:
    """(line, declared return type, spelling) per `noreturn` on a non-void function.

    Comments and literals are blanked with offsets and newlines preserved, so the
    reported line is the source line and a `noreturn` written in prose or in a
    string is not a site. A constructor or destructor has no return type and is
    accepted; `auto` is accepted only with a trailing `-> void`. Every spelling
    `_NORETURN_ATTRIBUTE` matches is read in the leading position, and the GNU
    one is read in its trailing position as well, because that is where it is
    normally written.
    """
    active = without_cpp_comments_and_literals(source)
    sites: list[tuple[int, str, str]] = []
    for match in _NORETURN_ATTRIBUTE.finditer(active):
        line = active.count("\n", 0, match.start()) + 1
        spelling = " ".join(match.group(0).split())
        head, paren = _declarator_head(active, match.end())
        if head is None:
            head, paren = _trailing_declarator(active, match.start())
        if head is None:
            sites.append((line, "<unparsed declaration>", spelling))
            continue
        words = _declaration_words(head)
        if "operator" in words:
            words = words[:words.index("operator") + 1]
        if len(words) < 2:
            continue  # constructor / destructor: no return type to be wrong.
        return_type = " ".join(words[:-1])
        if return_type == "void":
            continue
        if return_type == "auto" and _has_trailing_void_return(active, paren):
            continue
        sites.append((line, return_type, spelling))
    return sites


def noreturn_scan_paths(root: Path) -> list[Path]:
    """Every first-party C-family file this rule reads, sorted and deduplicated."""
    paths: set[Path] = set()
    for directory in NORETURN_SCAN_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in NORETURN_SCAN_SUFFIXES:
                continue
            if NORETURN_SCAN_EXCLUDED_DIR in path.relative_to(root).parts:
                continue
            paths.add(path)
    return sorted(paths)


def noreturn_nonvoid_errors(root: Path) -> list[str]:
    """Refuse every `noreturn` spelling on a non-void return type under `root`."""
    errors: list[str] = []
    for path in noreturn_scan_paths(root):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"{path}: unreadable while scanning for [[noreturn]]: {exc}")
            continue
        # Both cases, because `_Noreturn` is C11's spelling and a
        # case-sensitive `"noreturn"` fast path would skip the file carrying it
        # -- a hole that opens the moment the detector is widened past the
        # C++11 attribute. These two substrings cover all four spellings.
        if "noreturn" not in source and "Noreturn" not in source:
            continue
        relative = path.relative_to(root).as_posix()
        for line, return_type, spelling in noreturn_nonvoid_sites(source):
            errors.append(
                f"{relative}:{line}: {spelling} on a non-void return type "
                f"'{return_type}' is MSVC C4646, and /W4 /WX makes it C2220"
            )
    return errors


def has_shell_process_launch(text: str) -> bool:
    """Reject active shell APIs and cmd executable literals, not inert tokens."""
    active = without_cpp_comments_and_literals(text)
    if re.search(
        r"\b(?:system|popen|_popen|ShellExecute[AW]?)\s*\(", active
    ):
        return True

    for token in _CPP_INERT.finditer(text):
        literal = token.group("string")
        if literal is not None:
            value = literal[1:-1]
        else:
            raw = token.group("raw")
            if raw is None:
                continue
            delimiter = token.group("delimiter") or ""
            value = raw[len('R"' + delimiter + '('):-len(')' + delimiter + '"')]
        if re.search(r"(?i)(?:^|[\\/])cmd(?:\.exe)?(?:\s|$)", value):
            return True
    return False


def _active_powershell(text: str) -> str:
    """Strip comments and literal `if ($false) { ... }` blocks locally.

    Native Windows additionally parses the file with PowerShell's AST below;
    this fallback keeps the Linux checker fail-closed for the mutations we own.
    """
    text = re.sub(r"(?s)<#.*?#>", "", text)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines)
    text = re.sub(r"(?is)if\s*\(\s*\$false\s*\)\s*\{.*?\}", "", text)
    return re.sub(r"(?is)if\s*\(\s*\$ContractTest\s*\)\s*\{.*?\}", "", text)


def _powershell_syntax(text: str) -> str:
    """Blank PowerShell comments and strings while preserving source offsets."""
    out = list(text)
    index = 0
    quote = ""
    block_comment = False
    while index < len(text):
        if block_comment:
            if text.startswith("#>", index):
                out[index:index + 2] = "  "
                block_comment = False
                index += 2
            else:
                if text[index] != "\n":
                    out[index] = " "
                index += 1
            continue
        if quote:
            if quote == '"' and text[index] == "`" and index + 1 < len(text):
                out[index:index + 2] = "  "
                index += 2
                continue
            if text[index] == quote:
                if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                    out[index:index + 2] = "  "
                    index += 2
                    continue
                out[index] = " "
                quote = ""
                index += 1
                continue
            if text[index] != "\n":
                out[index] = " "
            index += 1
            continue
        if text.startswith("<#", index):
            out[index:index + 2] = "  "
            block_comment = True
            index += 2
            continue
        if text[index] == "#":
            while index < len(text) and text[index] != "\n":
                out[index] = " "
                index += 1
            continue
        if text[index] in {"'", '"'}:
            quote = text[index]
            out[index] = " "
            index += 1
            continue
        index += 1
    return "".join(out)


def _powershell_tokens(text: str) -> list[tuple[str, str]]:
    """Tokenize the small PowerShell contract without trusting spellings.

    Comments and continuation whitespace are inert, while quoted values remain
    available for binding the required filter and diagnostic.  This is the
    cross-platform view; on Windows the native PowerShell parser still owns the
    syntax check and fake-runner execution below.
    """
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("`\r\n", index):
            index += 3
            continue
        if text.startswith("`\n", index):
            index += 2
            continue
        if text.startswith("<#", index):
            end = text.find("#>", index + 2)
            if end < 0:
                return [("error", "unterminated block comment")]
            index = end + 2
            continue
        if text[index] == "#":
            end = text.find("\n", index + 1)
            index = len(text) if end < 0 else end + 1
            continue
        if text[index] in {"'", '"'}:
            quote = text[index]
            start = index
            index += 1
            while index < len(text):
                if quote == "'" and text.startswith("''", index):
                    index += 2
                    continue
                if quote == '"' and text[index] == "`" and index + 1 < len(text):
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    tokens.append(("string", text[start:index]))
                    break
                index += 1
            else:
                return [("error", "unterminated string")]
            continue
        if text.startswith("2>&1", index):
            tokens.append(("symbol", "2>&1"))
            index += 4
            continue
        if text.startswith("::", index):
            tokens.append(("symbol", "::"))
            index += 2
            continue
        if text.startswith("@(", index):
            tokens.append(("symbol", "@("))
            index += 2
            continue
        if text[index] == "$":
            if index + 1 < len(text) and text[index + 1] == "{":
                end = text.find("}", index + 2)
                if end < 0:
                    return [("error", "unterminated variable")]
                value = "$" + text[index + 2:end]
                index = end + 1
            else:
                match = re.match(r"\$[A-Za-z_][A-Za-z0-9_:]*", text[index:])
                if match is None:
                    tokens.append(("symbol", "$"))
                    index += 1
                    continue
                value = match.group(0)
                index += len(value)
            tokens.append(("variable", value.lower()))
            continue
        if text[index] == "@" and index + 1 < len(text):
            match = re.match(r"@[A-Za-z_][A-Za-z0-9_]*", text[index:])
            if match is not None:
                value = match.group(0)
                tokens.append(("splat", value.lower()))
                index += len(value)
                continue
        match = re.match(r"-[A-Za-z][A-Za-z0-9-]*", text[index:])
        if match is not None:
            value = match.group(0)
            tokens.append(("operator", value.lower()))
            index += len(value)
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_-]*", text[index:])
        if match is not None:
            value = match.group(0)
            tokens.append(("word", value.lower()))
            index += len(value)
            continue
        match = re.match(r"\d+", text[index:])
        if match is not None:
            value = match.group(0)
            tokens.append(("number", value))
            index += len(value)
            continue
        tokens.append(("symbol", text[index]))
        index += 1
    return tokens


def _powershell_string_value(token: tuple[str, str]) -> str | None:
    if token[0] != "string" or len(token[1]) < 2:
        return None
    quote = token[1][0]
    value = token[1][1:-1]
    if quote == "'":
        return value.replace("''", "'")
    return re.sub(r"`(.)", r"\1", value, flags=re.DOTALL)


def _powershell_exact_string_value(token: tuple[str, str]) -> str | None:
    """Return exact literals only when cross-platform semantics are known."""
    if token[0] != "string" or len(token[1]) < 2:
        return None
    # PowerShell maps backtick escapes such as `t and `n to control
    # characters.  Fail closed here instead of erasing the escape on hosts
    # where the native PowerShell parser is unavailable.
    if token[1][0] == '"' and "`" in token[1][1:-1]:
        return None
    return _powershell_string_value(token)


def _safe_powershell_message(token: tuple[str, str]) -> bool:
    """Allow inert diagnostic strings, but no executable subexpression."""
    return token[0] == "string" and "$(" not in token[1]


def _validate_exact_unsupported_tier_probe(
        body: str, errors: list[str]) -> list[tuple[str, str]]:
    tokens = _powershell_tokens(body)
    string = ("string", "*")
    expected: list[tuple[str, str]] = [
        ("word", "param"), ("symbol", "("),
        ("symbol", "["), ("word", "parameter"), ("symbol", "("),
        ("word", "mandatory"), ("symbol", ")"), ("symbol", "]"),
        ("symbol", "["), ("word", "string"), ("symbol", "]"),
        ("variable", "$tiertest"), ("symbol", ","),
        ("symbol", "["), ("word", "scriptblock"), ("symbol", "]"),
        ("variable", "$runner"), ("symbol", ")"),
        ("variable", "$arguments"), ("symbol", "="), ("symbol", "@("),
        ("string", "FILTER"), ("symbol", ")"),
        ("word", "if"), ("symbol", "("), ("variable", "$null"),
        ("operator", "-eq"), ("variable", "$runner"), ("symbol", ")"),
        ("symbol", "{"),
        ("variable", "$probeoutput"), ("symbol", "="), ("symbol", "@("),
        ("symbol", "&"), ("variable", "$tiertest"),
        ("splat", "@arguments"), ("symbol", "2>&1"), ("symbol", ")"),
        ("variable", "$probeexitcode"), ("symbol", "="),
        ("variable", "$lastexitcode"), ("symbol", "}"),
        ("word", "else"), ("symbol", "{"),
        ("variable", "$proberesult"), ("symbol", "="), ("symbol", "&"),
        ("variable", "$runner"), ("variable", "$tiertest"),
        ("variable", "$arguments"),
        ("variable", "$probeoutput"), ("symbol", "="), ("symbol", "@("),
        ("variable", "$proberesult"), ("symbol", "."), ("word", "output"),
        ("symbol", ")"),
        ("variable", "$probeexitcode"), ("symbol", "="),
        ("symbol", "["), ("word", "int"), ("symbol", "]"),
        ("variable", "$proberesult"), ("symbol", "."), ("word", "exitcode"),
        ("symbol", "}"),
        ("word", "if"), ("symbol", "("), ("variable", "$probeexitcode"),
        ("operator", "-ne"), ("number", "1"), ("symbol", ")"),
        ("symbol", "{"), ("word", "throw"), string, ("symbol", "}"),
        ("variable", "$diagnostic"), ("symbol", "="),
        ("variable", "$probeoutput"), ("operator", "-join"), string,
        ("word", "if"), ("symbol", "("), ("variable", "$diagnostic"),
        ("operator", "-notmatch"), ("symbol", "["), ("word", "regex"),
        ("symbol", "]"), ("symbol", "::"), ("word", "escape"),
        ("symbol", "("), ("string", "DIAGNOSTIC"), ("symbol", ")"),
        ("symbol", ")"), ("symbol", "{"), ("word", "throw"), string,
        ("symbol", "}"),
    ]

    def matches(actual: tuple[str, str], wanted: tuple[str, str]) -> bool:
        if wanted == string:
            return _safe_powershell_message(actual)
        if wanted == ("string", "FILTER"):
            return _powershell_exact_string_value(actual) == UNSUPPORTED_TIER_FILTER
        if wanted == ("string", "DIAGNOSTIC"):
            return (_powershell_exact_string_value(actual) ==
                    UNSUPPORTED_TIER_DIAGNOSTIC)
        return actual == wanted

    def sequence_matches(wanted_tokens: list[tuple[str, str]]) -> bool:
        return (len(tokens) == len(wanted_tokens) and
                all(matches(actual, wanted)
                    for actual, wanted in zip(tokens, wanted_tokens)))

    if not sequence_matches(expected):
        errors.append(
            "build-windows-release.ps1: exact unsupported-tier probe body is required"
        )
    return tokens


def _validate_exact_amx_refusal_block(
        text: str, errors: list[str]) -> None:
    tokens = _powershell_tokens(text)
    starts = [
        index for index in range(len(tokens) - 2)
        if (tokens[index] == ("variable", "$env:vt_cpu_matmul_tier") and
            tokens[index + 1] == ("symbol", "=") and
            _powershell_exact_string_value(tokens[index + 2]) == "amx")
    ]
    valid = False
    if len(starts) == 1:
        start = starts[0]
        expected = [
            ("variable", "$env:vt_cpu_matmul_tier"), ("symbol", "="),
            tokens[start + 2],
            ("word", "invoke-unsupportedtierprobe"),
            ("operator", "-tiertest"), ("variable", "$tiertest"),
            ("symbol", "}"), ("word", "finally"),
            ("symbol", "{"),
            ("variable", "$env:vt_cpu_matmul_tier"), ("symbol", "="),
            ("variable", "$savedtier"), ("symbol", "}"),
        ]
        valid = tokens[start:start + len(expected)] == expected
    if not valid:
        errors.append(
            "build-windows-release.ps1: AMX refusal must use only the isolated "
            "unsupported-tier probe; exact AMX refusal block is required"
        )


def _powershell_function_body(text: str, name: str) -> str:
    syntax = _powershell_syntax(text)
    match = re.search(
        rf"\bfunction\s+{re.escape(name)}\s*\{{", syntax, re.IGNORECASE
    )
    if match is None:
        return ""
    opening = syntax.find("{", match.start(), match.end())
    depth = 0
    for offset in range(opening, len(syntax)):
        if syntax[offset] == "{":
            depth += 1
        elif syntax[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:offset]
    return ""


def _powershell_scriptblock_assignment_body(text: str, variable: str) -> str:
    syntax = _powershell_syntax(text)
    match = re.search(
        rf"\${re.escape(variable)}\s*=\s*\{{", syntax, re.IGNORECASE
    )
    if match is None:
        return ""
    opening = syntax.find("{", match.start(), match.end())
    depth = 0
    for offset in range(opening, len(syntax)):
        if syntax[offset] == "{":
            depth += 1
        elif syntax[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:offset]
    return ""


def _ordered_matches(text: str, stages: tuple[tuple[str, str], ...],
                     errors: list[str], label: str) -> None:
    offsets: list[int] = []
    for description, pattern in stages:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match is None:
            errors.append(
                f"build-windows-release.ps1: missing active {description} in {label}"
            )
            return
        offsets.append(match.start())
    if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        errors.append(
            "build-windows-release.ps1: native gate order must be "
            "configure/codemodel -> checker -> build/focused tests -> install -> "
            "CRT audit -> help/tier/server smokes"
        )


def _validate_powershell_ast_order(commands: list[dict],
                                   errors: list[str]) -> None:
    stages = (
        ("codemodel query", r"\bNew-Item\b.*codemodel-v2"),
        ("configure", r"\bInvoke-Checked\s+cmake\b.*(?:\"-S\"|\s-S\s)"),
        ("portability checker", r"\bInvoke-Checked\s+python\b.*check-windows-portability\.py"),
        ("build", r"\bInvoke-Checked\s+cmake\b.*\"--build\""),
        ("focused tests", r"\bInvoke-Checked\b.*tests[/\\]Release[/\\]\$test"),
        ("install", r"\bInvoke-Checked\s+cmake\b.*\"--install\""),
        ("CRT audit", r"\bInvoke-CrtAudit\b"),
        ("live --help smoke", r"\bInvoke-Checked\s+\$server\b.*--help"),
        ("forced-tier smoke", r"\bInvoke-UnsupportedTierProbe\b.*\$tierTest\b"),
        ("server smoke harness", r"\bInvoke-Checked\s+python\b.*\$smokeHarness"),
    )
    offsets: list[int] = []
    for description, pattern in stages:
        candidates = [
            int(item.get("offset", -1)) for item in commands
            if re.search(pattern, item.get("text", ""), re.IGNORECASE | re.DOTALL)
        ]
        if not candidates:
            errors.append(
                f"build-windows-release.ps1: missing active {description} in PowerShell AST"
            )
            return
        offsets.append(min(candidates))
    if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        errors.append(
            "build-windows-release.ps1: native gate order must be "
            "configure/codemodel -> checker -> build/focused tests -> install -> "
            "CRT audit -> help/tier/server smokes"
        )


def _validate_powershell_source_order(text: str, errors: list[str]) -> None:
    active = _active_powershell(text)
    pipeline = active.find("codemodel-v2")
    if pipeline >= 0:
        active = active[pipeline:]
    stages = (
        ("codemodel query", r"codemodel-v2"),
        ("configure", r"(?:^\s*cmake\s+-S\b|^\s*\"-S\"\s*,\s*\$SourceDir)"),
        ("portability checker", r"check-windows-portability\.py"),
        ("build", r"(?:^\s*cmake\s+--build\b|\"--build\"\s*,\s*\$BuildDir)"),
        ("focused tests", r"(?:^\s*foreach\s*\(\s*\$test\b|^\s*&\s+\"\$BuildDir/tests/test_openai_api_server\.exe\")"),
        ("install", r"(?:^\s*cmake\s+--install\b|\"--install\"\s*,\s*\$BuildDir)"),
        ("CRT audit", r"^\s*Invoke-CrtAudit\b"),
        ("live --help smoke", r"^\s*Invoke-Checked\s+\$server\b[^\n]*--help"),
        ("forced-tier smoke", r"^\s*Invoke-UnsupportedTierProbe\b[^\n]*\$tierTest\b"),
        ("server smoke harness", r"^\s*Invoke-Checked\s+python\b[^\n]*\$smokeHarness"),
    )
    _ordered_matches(active, stages, errors, "source contract")


def _cpp_braced_body(source: str, opening: int) -> tuple[str, int] | None:
    """Return a balanced braced body and its closing-brace offset."""
    if opening < 0 or opening >= len(source) or source[opening] != "{":
        return None
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:offset], offset
    return None


def _cpp_function_body_span(source: str,
                            signature: str) -> tuple[str, int, int] | None:
    active = _cpp_structural_view(source)
    search_from = 0
    while True:
        match = re.search(signature, active[search_from:])
        if match is None:
            return None
        match_start = search_from + match.start()
        match_end = search_from + match.end()
        parameters_open = active.rfind("(", match_start, match_end)
        if parameters_open < 0:
            return None

        depth = 0
        parameters_close = -1
        for offset in range(parameters_open, len(active)):
            if active[offset] == "(":
                depth += 1
            elif active[offset] == ")":
                depth -= 1
                if depth == 0:
                    parameters_close = offset
                    break
        if parameters_close < 0:
            return None

        suffix_depth: list[str] = []
        declaration = False
        pairs = {"(": ")", "[": "]"}
        offset = parameters_close + 1
        while offset < len(active):
            char = active[offset]
            requires = re.match(r"requires\b", active[offset:])
            if not suffix_depth and requires is not None:
                expression = offset + requires.end()
                while expression < len(active) and active[expression].isspace():
                    expression += 1
                if expression < len(active) and active[expression] == "(":
                    depth = 0
                    for cursor in range(expression, len(active)):
                        if active[cursor] == "(":
                            depth += 1
                        elif active[cursor] == ")":
                            depth -= 1
                            if depth == 0:
                                expression = cursor + 1
                                break
                    else:
                        return None
                    while expression < len(active) and active[expression].isspace():
                        expression += 1
                if expression < len(active) and active[expression] == "{":
                    requires_body = _cpp_braced_body(active, expression)
                    if requires_body is None:
                        return None
                    offset = requires_body[1] + 1
                    continue
            if char in pairs:
                suffix_depth.append(pairs[char])
            elif suffix_depth and char == suffix_depth[-1]:
                suffix_depth.pop()
            elif not suffix_depth and char == ";":
                declaration = True
                search_from = offset + 1
                break
            elif not suffix_depth and char == "{":
                result = _cpp_braced_body(active, offset)
                if result is None:
                    return None
                return result[0], offset + 1, result[1]
            offset += 1
        if not declaration:
            return None


def _cpp_function_body(source: str, signature: str) -> str:
    result = _cpp_function_body_span(source, signature)
    return "" if result is None else result[0]


def _cpp_phase2_view(source: str) -> str:
    """Apply backslash-newline splicing without changing source offsets."""
    out = list(source)
    offset = 0
    while offset < len(source):
        if source[offset] != "\\":
            offset += 1
            continue
        end = offset + 1
        if source.startswith("\r\n", end):
            end += 2
        elif end < len(source) and source[end] == "\n":
            end += 1
        else:
            offset += 1
            continue
        out[offset:end] = " " * (end - offset)
        offset = end
    return "".join(out)


def _cpp_directive_view(source: str) -> str:
    """Apply phase 2 and blank comments while retaining pragma strings."""
    phase2 = _cpp_phase2_view(source)

    def replace(match: re.Match[str]) -> str:
        if match.group("block") is None and match.group("line") is None:
            return match.group(0)
        return "".join(
            "\n" if char == "\n" else " " for char in match.group(0)
        )

    return _CPP_INERT.sub(replace, phase2)


def _cpp_logical_directives(source: str, stop: int):
    """Yield directives after phase-2 splicing and phase-3 comments."""
    active = _cpp_directive_view(source[:stop])
    offset = 0
    for line in active.splitlines(keepends=True):
        end = offset + len(line)
        if re.match(r"\s*#", line):
            yield offset, end, line
        offset = end
    if offset < len(active):
        line = active[offset:]
        if re.match(r"\s*#", line):
            yield offset, len(active), line


def _cpp_structural_view(source: str) -> str:
    """Blank inert text and complete directives without moving offsets."""
    out = list(without_cpp_comments_and_literals(_cpp_phase2_view(source)))
    for start, end, _ in _cpp_logical_directives(source, len(source)):
        for offset in range(start, end):
            if out[offset] not in {"\r", "\n"}:
                out[offset] = " "
    return "".join(out)


def _cpp_condition_possibilities(expression: str,
                                 macros: frozenset[str]) -> tuple[bool, bool]:
    """Return whether a preprocessing expression may be true and false."""
    expression = expression.strip()
    while (expression.startswith("(") and expression.endswith(")") and
           expression.count("(") == expression.count(")")):
        expression = expression[1:-1].strip()
    if re.fullmatch(r"0+[uUlL]*", expression):
        return False, True
    if re.fullmatch(r"[1-9]\d*[uUlL]*", expression):
        return True, False
    defined = re.fullmatch(
        r"(!\s*)?defined\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|"
        r"([A-Za-z_]\w*))",
        expression,
    )
    if defined is not None:
        name = defined.group(2) or defined.group(3)
        if name in {"_WIN32", "_MSC_VER"}:
            value = True
        elif name in {"__GNUC__", "__clang__", "__unix__", "__APPLE__"}:
            value = False
        else:
            value = name in macros
        if defined.group(1):
            value = not value
        return value, not value
    if expression in {"_WIN32", "_MSC_VER"}:
        return True, False
    if expression in {"__GNUC__", "__clang__", "__unix__", "__APPLE__"}:
        return False, True
    return True, True


CppMacroState = tuple[
    frozenset[str], tuple[tuple[str, tuple[bool, ...]], ...]
]


def _cpp_macro_state(defined: set[str],
                     stacks: dict[str, list[bool]]) -> CppMacroState:
    return (
        frozenset(defined),
        tuple(sorted(
            (name, tuple(values)) for name, values in stacks.items() if values
        )),
    )


def _cpp_change_macro(state: CppMacroState, name: str,
                      defined: bool) -> CppMacroState:
    names = set(state[0])
    if defined:
        names.add(name)
    else:
        names.discard(name)
    return _cpp_macro_state(
        names, {key: list(values) for key, values in state[1]}
    )


def _cpp_push_macro(state: CppMacroState, name: str) -> CppMacroState:
    stacks = {key: list(values) for key, values in state[1]}
    stacks.setdefault(name, []).append(name in state[0])
    return _cpp_macro_state(set(state[0]), stacks)


def _cpp_pop_macro(state: CppMacroState, name: str) -> CppMacroState:
    stacks = {key: list(values) for key, values in state[1]}
    saved = stacks.get(name, [])
    # An unmatched pop restores implementation-owned state.  Treat the name
    # as possibly defined rather than assuming an unsafe alias disappeared.
    restored = saved.pop() if saved else True
    if not saved:
        stacks.pop(name, None)
    names = set(state[0])
    if restored:
        names.add(name)
    else:
        names.discard(name)
    return _cpp_macro_state(names, stacks)


def _split_cpp_states(states: set[CppMacroState], expression: str
                      ) -> tuple[set[CppMacroState], set[CppMacroState]]:
    true_states: set[CppMacroState] = set()
    false_states: set[CppMacroState] = set()
    for state in states:
        may_true, may_false = _cpp_condition_possibilities(expression, state[0])
        if may_true:
            true_states.add(state)
        if may_false:
            false_states.add(state)
    return true_states, false_states


def _active_cpp_macro_names_at(source: str, stop: int) -> set[str]:
    """Return macros possibly active on Windows at one exact source offset."""
    states: set[CppMacroState] = {_cpp_macro_state(set(), {})}
    stack: list[dict[str, set[CppMacroState] | bool]] = []
    for _, _, logical in _cpp_logical_directives(source, stop):
        directive = re.match(r"\s*#\s*([A-Za-z_]\w*)(.*)", logical,
                             re.DOTALL)
        if directive is None:
            continue
        command = directive.group(1).lower()
        argument = directive.group(2).strip()
        if command in {"if", "ifdef", "ifndef"}:
            if command == "ifdef":
                expression = f"defined({argument.split()[0]})"
            elif command == "ifndef":
                expression = f"!defined({argument.split()[0]})"
            else:
                expression = argument
            current, remaining = _split_cpp_states(states, expression)
            stack.append({
                "remaining": remaining,
                "completed": set(),
                "else_seen": False,
            })
            states = current
        elif command == "elif" and stack:
            frame = stack[-1]
            completed = frame["completed"]
            remaining = frame["remaining"]
            assert isinstance(completed, set) and isinstance(remaining, set)
            completed.update(states)
            states, new_remaining = _split_cpp_states(remaining, argument)
            frame["remaining"] = new_remaining
        elif command == "else" and stack:
            frame = stack[-1]
            completed = frame["completed"]
            remaining = frame["remaining"]
            assert isinstance(completed, set) and isinstance(remaining, set)
            completed.update(states)
            states = remaining
            frame["remaining"] = set()
            frame["else_seen"] = True
        elif command == "endif" and stack:
            frame = stack.pop()
            completed = frame["completed"]
            remaining = frame["remaining"]
            assert isinstance(completed, set) and isinstance(remaining, set)
            states = completed.union(states, remaining)
        elif command == "define":
            name = re.match(r"([A-Za-z_]\w*)", argument)
            if name is not None:
                states = {
                    _cpp_change_macro(state, name.group(1), True)
                    for state in states
                }
        elif command == "undef":
            name = re.match(r"([A-Za-z_]\w*)", argument)
            if name is not None:
                states = {
                    _cpp_change_macro(state, name.group(1), False)
                    for state in states
                }
        elif command == "pragma":
            pragma = re.fullmatch(
                r"(push_macro|pop_macro)\s*\(\s*\"([A-Za-z_]\w*)\"\s*\)",
                argument,
                re.DOTALL,
            )
            if pragma is not None:
                operation, name = pragma.groups()
                if operation == "push_macro":
                    states = {_cpp_push_macro(state, name) for state in states}
                else:
                    states = {_cpp_pop_macro(state, name) for state in states}
    return set().union(*(set(state[0]) for state in states)) if states else set()


def _validate_unsupported_tier_contract(text: str, errors: list[str]) -> None:
    active = _active_powershell(text)
    probe_body = _powershell_function_body(text, "Invoke-UnsupportedTierProbe")
    probe = _active_powershell(probe_body)
    contract = _active_powershell(_powershell_function_body(
        text, "Invoke-UnsupportedTierContractTests"
    ))
    good_runner = _active_powershell(
        _powershell_scriptblock_assignment_body(contract, "good")
    )
    if not probe:
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier probe helper"
        )
    if not contract:
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier fake-tool contract"
        )

    probe_tokens = _validate_exact_unsupported_tier_probe(probe_body, errors)

    def count_sequence(sequence: list[tuple[str, str]]) -> int:
        return sum(
            probe_tokens[index:index + len(sequence)] == sequence
            for index in range(len(probe_tokens) - len(sequence) + 1)
        )

    filter_literals = sum(
        _powershell_exact_string_value(token) == UNSUPPORTED_TIER_FILTER
        for token in probe_tokens
    )
    exact_filter_argument = any(
        probe_tokens[index:index + 3] == [
            ("variable", "$arguments"), ("symbol", "="), ("symbol", "@(")
        ] and index + 4 < len(probe_tokens) and
        _powershell_exact_string_value(probe_tokens[index + 3]) ==
        UNSUPPORTED_TIER_FILTER and
        probe_tokens[index + 4] == ("symbol", ")")
        for index in range(len(probe_tokens) - 4)
    )
    if filter_literals == 0:
        errors.append(
            "build-windows-release.ps1: missing active isolated "
            "unsupported-tier filter"
        )
    elif not exact_filter_argument:
        errors.append(
            "build-windows-release.ps1: unsupported-tier probe requires one "
            "exact unsupported-tier filter argument"
        )
    exit_check = [
        ("variable", "$probeexitcode"), ("operator", "-ne"),
        ("number", "1"),
    ]
    if count_sequence(exit_check) != 1:
        errors.append(
            "build-windows-release.ps1: missing active exact unsupported-tier exit status"
        )
    diagnostic_call = [
        ("variable", "$diagnostic"), ("operator", "-notmatch"),
        ("symbol", "["), ("word", "regex"), ("symbol", "]"),
        ("symbol", "::"), ("word", "escape"), ("symbol", "("),
    ]
    inline_diagnostic_call = [
        ("operator", "-notmatch"), ("symbol", "["), ("word", "regex"),
        ("symbol", "]"), ("symbol", "::"), ("word", "escape"),
        ("symbol", "("),
    ]
    diagnostic_bound = any(
        ((probe_tokens[index:index + len(diagnostic_call)] == diagnostic_call and
          index + len(diagnostic_call) < len(probe_tokens) and
          _powershell_exact_string_value(
              probe_tokens[index + len(diagnostic_call)]) ==
          UNSUPPORTED_TIER_DIAGNOSTIC) or
         (probe_tokens[index:index + len(inline_diagnostic_call)] ==
          inline_diagnostic_call and
          index + len(inline_diagnostic_call) < len(probe_tokens) and
          _powershell_exact_string_value(
              probe_tokens[index + len(inline_diagnostic_call)]) ==
          UNSUPPORTED_TIER_DIAGNOSTIC))
        for index in range(len(probe_tokens))
    )
    if not diagnostic_bound:
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier diagnostic"
        )
    if re.search(
            r"finally\s*\{[^}]*\$env:VT_CPU_MATMUL_TIER\s*=\s*\$savedTier",
            active, re.IGNORECASE | re.MULTILINE) is None:
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier "
            "environment restoration"
        )

    def invokes_tier_test(index: int) -> bool:
        if probe_tokens[index] != ("symbol", "&"):
            return False
        for token in probe_tokens[index + 1:index + 6]:
            if token[0] == "variable":
                return token[1].rsplit(":", 1)[-1].lstrip("$") == "tiertest"
            if token not in {("symbol", "("), ("symbol", "$"),
                             ("symbol", ")")}:
                return False
        return False

    direct_invocations = sum(
        invokes_tier_test(index) for index in range(len(probe_tokens))
    )
    exact_capture = [
        ("symbol", "@("), ("symbol", "&"),
        ("variable", "$tiertest"), ("splat", "@arguments"),
        ("symbol", "2>&1"), ("symbol", ")"),
    ]
    exact_direct = sum(
        probe_tokens[index:index + len(exact_capture)] == exact_capture
        for index in range(len(probe_tokens) - len(exact_capture) + 1)
    )
    if exact_direct != 1:
        errors.append(
            "build-windows-release.ps1: missing active merged unsupported-tier "
            "stdout/stderr capture"
        )
    if direct_invocations != 1 or exact_direct != 1:
        errors.append(
            "build-windows-release.ps1: unsupported-tier probe requires "
            "exactly one filtered process invocation"
        )

    runner_invocations = sum(
        probe_tokens[index:index + 2] == [
            ("symbol", "&"), ("variable", "$runner")
        ]
        for index in range(len(probe_tokens) - 1)
    )
    exact_runner_tokens = [
        ("variable", "$proberesult"), ("symbol", "="),
        ("symbol", "&"), ("variable", "$runner"),
        ("variable", "$tiertest"), ("variable", "$arguments"),
    ]
    exact_runner = sum(
        probe_tokens[index:index + len(exact_runner_tokens)] == exact_runner_tokens
        for index in range(len(probe_tokens) - len(exact_runner_tokens) + 1)
    )
    if runner_invocations != 1:
        errors.append(
            "build-windows-release.ps1: unsupported-tier probe requires "
            "exactly one fake-runner invocation"
        )
    elif exact_runner != 1:
        errors.append(
            "build-windows-release.ps1: fake runner requires exact "
            "unsupported-tier filter arguments"
        )

    for crash_status in ("134", "-1073741819", "3", "2"):
        if re.search(
                rf"ExitCode\s*=\s*{re.escape(crash_status)}\b", contract,
                re.IGNORECASE) is None:
            errors.append(
                "build-windows-release.ps1: unsupported-tier crash contract "
                f"is missing status {crash_status}"
            )

    fake_contract_requirements = (
        (r"\$calls\.Count\s*-ne\s*1\b", "exactly one recorded fake call"),
        (
            r"\$calls\s*\[\s*0\s*\]\.Arguments\.Count\s*-ne\s*1\b",
            "exactly one recorded fake argument",
        ),
        (
            r"\$calls\s*\[\s*0\s*\]\.Arguments\s*\[\s*0\s*\]\s*"
            r"-ne\s*(['\"])" + re.escape(UNSUPPORTED_TIER_FILTER) + r"\1",
            "exact recorded fake filter argument",
        ),
    )
    for pattern, description in fake_contract_requirements:
        if re.search(pattern, contract, re.IGNORECASE | re.MULTILINE) is None:
            errors.append(
                f"build-windows-release.ps1: missing active {description}"
            )
    if re.search(r"\$calls\.Add\s*\(", good_runner, re.IGNORECASE) is None:
        errors.append(
            "build-windows-release.ps1: missing active fake runner call recording"
        )
    good_calls = re.findall(
        r"Invoke-UnsupportedTierProbe\s+-TierTest\s+(['\"])"
        r"fake-tier-test\.exe\1\s+-Runner\s+\$good\b",
        contract,
        re.IGNORECASE,
    )
    if len(good_calls) != 1:
        errors.append(
            "build-windows-release.ps1: fake contract requires exactly one "
            "unsupported-tier probe invocation"
        )

    uncommented = re.sub(r"(?s)<#.*?#>", "", text)
    uncommented = "\n".join(
        line for line in uncommented.splitlines()
        if not line.lstrip().startswith("#")
    )
    contract_block = re.search(
        r"(?ms)^\s*if\s*\(\s*\$ContractTest\s*\)\s*\{(?P<body>.*?)^\s*\}",
        uncommented,
    )
    if (contract_block is None or
            re.search(r"(?m)^\s*Invoke-UnsupportedTierContractTests\s*$",
                      contract_block.group("body")) is None):
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier "
            "fake-tool contract"
        )

    _validate_exact_amx_refusal_block(active, errors)


def _finite_timeout_expression(console: str, expression: str) -> bool:
    expression = expression.strip()
    if re.fullmatch(r"\d+(?:[uUlL]*)", expression):
        return True
    if not re.fullmatch(r"[A-Za-z_]\w*", expression):
        return False
    definitions = re.findall(
        rf"\b(?:constexpr|const)\b[^;=]*\b{re.escape(expression)}\s*=\s*([^;]+);",
        console,
    )
    return (len(definitions) == 1 and
            re.fullmatch(r"\s*\d+(?:[uUlL]*)\s*", definitions[0]) is not None)


def _validate_bounded_drain(console: str, function: str, counter: str,
                            errors: list[str]) -> None:
    active_console = _cpp_structural_view(console)
    body_span = _cpp_function_body_span(
        console,
        rf"\bbool\s+{re.escape(function)}\s*\(",
    )
    body = "" if body_span is None else body_span[0]
    label = f"console_shutdown.cpp: {function} requires a finite timeout"
    if not body or re.search(r"\bINFINITE\b", body):
        errors.append(label)
        return

    starts = list(re.finditer(
        r"\b(?:const\s+)?ULONGLONG\s+start\s*=\s*"
        r"GetTickCount64\s*\(\s*\)\s*;",
        body,
    ))
    loop = re.search(
        rf"\bwhile\s*\(\s*{re.escape(counter)}\.load\s*\(\s*"
        r"std::memory_order_seq_cst\s*\)\s*!=\s*0\s*\)\s*\{",
        body,
    )
    if (len(starts) != 1 or loop is None or
            starts[0].start() >= (loop.start() if loop else 0)):
        errors.append(label)
        return

    opening = body.find("{", loop.start(), loop.end())
    loop_result = _cpp_braced_body(body, opening)
    if loop_result is None:
        errors.append(label)
        return
    loop_body, loop_close = loop_result
    timeout_branch = re.match(
        r"\s*if\s*\(\s*GetTickCount64\s*\(\s*\)\s*-\s*start\s*>=\s*"
        r"([A-Za-z_]\w*|\d+[uUlL]*)\s*\)"
        r"\s*(?:\{\s*)?return\s+false\s*;\s*(?:\}\s*)?",
        loop_body,
    )
    success_tail = body[loop_close + 1:]
    if (timeout_branch is None or
            not _finite_timeout_expression(active_console,
                                           timeout_branch.group(1)) or
            re.fullmatch(
                r"\s*return\s+true\s*;\s*", success_tail
            ) is None):
        errors.append(label)
        return

    assert body_span is not None
    trusted_fragments = (
        (
            body_span[1] + opening + 1 + timeout_branch.start(),
            timeout_branch.group(0),
        ),
        (body_span[1] + loop_close + 1, success_tail),
    )
    macro_collisions: set[str] = set()
    for fragment_start, fragment in trusted_fragments:
        for token in re.finditer(r"\b[A-Za-z_]\w*\b", fragment):
            if token.group(0) in _active_cpp_macro_names_at(
                    console, fragment_start + token.start()):
                macro_collisions.add(token.group(0))
    if macro_collisions:
        errors.append(
            f"console_shutdown.cpp: {function} trusted return-tail token "
            "must not be a macro "
            f"({', '.join(sorted(macro_collisions))})"
        )


def _validate_console_protocol(console: str, errors: list[str]) -> None:
    active_console = without_cpp_comments_and_literals(console)
    dispatch_span = _cpp_function_body_span(
        console, r"\bbool\s+DispatchControlEvent\s*\("
    )
    dispatch = "" if dispatch_span is None else dispatch_span[0]
    handler = _cpp_function_body(
        console, r"\bBOOL\s+WINAPI\s+ConsoleControlHandler\s*\("
    )
    stable_declarations = (
        re.search(r"std::atomic\s*<\s*WindowsHandlerState\s*\*\s*>\s+published", console),
        re.search(r"std::atomic\s*<\s*unsigned\s*>\s+entrants", console),
        re.search(r"std::atomic\s*<\s*unsigned\s*>\s+in_flight", console),
        re.search(
            r"static\s+auto\s*\*\s*registry\s*=\s*new\s+WindowsHandlerRegistry",
            console,
        ),
    )
    dispatch_steps = (
        r"entrants\.fetch_add\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"published\.load\s*\(\s*std::memory_order_seq_cst\s*\)",
        r"in_flight\.fetch_add\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"entrants\.fetch_sub\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"SetEvent\s*\(\s*state->stop_event\s*\)",
        r"in_flight\.fetch_sub\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
    )
    dispatch_offsets = []
    for pattern in dispatch_steps:
        match = re.search(pattern, dispatch)
        dispatch_offsets.append(-1 if match is None else match.start())
    teardown = _cpp_function_body(
        console, r"ConsoleShutdown::~ConsoleShutdown\s*\("
    ) or console
    teardown_steps = (
        r"published\.store\s*\(\s*nullptr\s*,\s*std::memory_order_seq_cst\s*\)",
        r"DrainEntrantsWithTimeout\s*\(",
        r"DrainInFlightWithTimeout\s*\(",
        r"(?:(?:win_state_|state)\.reset\s*\(\s*\))",
    )
    teardown_offsets = []
    for index, pattern in enumerate(teardown_steps):
        matches = list(re.finditer(pattern, teardown))
        match = matches[-1] if index == len(teardown_steps) - 1 and matches else (
            matches[0] if matches else None
        )
        teardown_offsets.append(-1 if match is None else match.start())
    ordered_dispatch = (
        all(offset >= 0 for offset in dispatch_offsets) and
        dispatch_offsets == sorted(dispatch_offsets)
    )
    ordered_teardown = (
        all(offset >= 0 for offset in teardown_offsets) and
        teardown_offsets == sorted(teardown_offsets)
    )
    retire = re.search(r"RetireHandlerState\s*\(", teardown)
    reset = re.search(r"(?:win_state_|state)\.reset\s*\(", teardown)
    timeout_retained = (
        re.search(r"if\s*\(\s*!\s*safe_to_close\s*\)", teardown) is not None and
        retire is not None and reset is not None and retire.start() < reset.start()
    )
    if (not all(stable_declarations) or not ordered_dispatch or
            not ordered_teardown or not timeout_retained):
        errors.append(
            "console_shutdown.cpp: stable event/in-flight handler lifetime protocol is required"
        )
    forbidden = r"RequestStop|\bimpl_|\bthis\b|std::mutex|condition_variable|\bstop_\s*\("
    if not dispatch or not handler or re.search(forbidden, dispatch + "\n" + handler):
        errors.append(
            "console_shutdown.cpp: OS handler may use only stable atomics and Win32 events"
        )
    final_decrements = list(re.finditer(
        r"state->in_flight\.fetch_sub\s*\(\s*1\s*,\s*"
        r"std::memory_order_seq_cst\s*\)",
        dispatch,
    ))
    macro_collisions: set[str] = set()
    if dispatch_span is not None and final_decrements:
        trusted_tail = dispatch[final_decrements[0].start():]
        trusted_start = dispatch_span[1] + final_decrements[0].start()
        for token in re.finditer(r"\b[A-Za-z_]\w*\b", trusted_tail):
            if token.group(0) in _active_cpp_macro_names_at(
                    console, trusted_start + token.start()):
                macro_collisions.add(token.group(0))
    if macro_collisions:
        errors.append(
            "console_shutdown.cpp: trusted final-tail token must not be a macro "
            f"({', '.join(sorted(macro_collisions))})"
        )
    final_tail_ok = False
    if len(final_decrements) == 1:
        resumed_declarations = list(re.finditer(
            r"\b(?:const\s+)?bool\s+resumed\s*=\s*[^;]+;",
            dispatch[:final_decrements[0].start()],
        ))
        final_tail = dispatch[final_decrements[0].end():]
        final_tail_ok = (
            len(resumed_declarations) == 1 and
            re.fullmatch(
                r"\s*;\s*return\s+resumed\s*;\s*",
                final_tail,
            ) is not None
        )
    if not final_tail_ok:
        errors.append(
            "console_shutdown.cpp: final in-flight decrement must be the last "
            "handler operation"
        )
    _validate_bounded_drain(
        console, "DrainEntrantsWithTimeout", "entrants", errors
    )
    _validate_bounded_drain(
        console, "DrainInFlightWithTimeout", "in_flight", errors
    )
    cleanup = _cpp_function_body(
        console, r"WindowsHandlerState::~WindowsHandlerState\s*\("
    ) or console
    if not all(re.search(rf"CloseHandle\s*\(\s*{event}\s*\)", cleanup)
               for event in ("stop_event", "quit_event")):
        errors.append(
            "console_shutdown.cpp: partial event creation cleanup must close every created handle"
        )


def _validate_powershell_ast(script: Path, errors: list[str]) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        return
    parser = r'''
$Path = [Environment]::GetEnvironmentVariable('__AUDIT_PATH_ENV__')
if ([string]::IsNullOrWhiteSpace($Path)) {
  [Console]::Error.WriteLine("PowerShell audit path is missing")
  exit 2
}
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) {
  $parseErrors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }
  exit 2
}
function Test-Dead([System.Management.Automation.Language.Ast]$Node) {
  $cursor = $Node
  while ($null -ne $cursor.Parent) {
    $parent = $cursor.Parent
    if ($parent -is [System.Management.Automation.Language.IfStatementAst]) {
      foreach ($clause in $parent.Clauses) {
        if ($clause.Item1.Extent.Text -match '^\s*\$false\s*$' -and
            $Node.Extent.StartOffset -ge $clause.Item2.Extent.StartOffset -and
            $Node.Extent.EndOffset -le $clause.Item2.Extent.EndOffset) {
          return $true
        }
      }
    }
    $cursor = $parent
  }
  return $false
}
$commands = @($ast.FindAll({
  param($node) $node -is [System.Management.Automation.Language.CommandAst]
}, $true) | Where-Object {
  if (Test-Dead $_) { return $false }
  $cursor = $_.Parent
  while ($null -ne $cursor) {
    if ($cursor -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
      return $false
    }
    $cursor = $cursor.Parent
  }
  return $true
} | ForEach-Object {
  [pscustomobject]@{ text = $_.Extent.Text; offset = $_.Extent.StartOffset }
})
$commands | ConvertTo-Json -Compress
'''
    parser = parser.replace("__AUDIT_PATH_ENV__", POWERSHELL_AUDIT_PATH_ENV)
    parser_environment = os.environ.copy()
    parser_environment[POWERSHELL_AUDIT_PATH_ENV] = str(script)
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", parser],
        text=True, capture_output=True, check=False, shell=False,
        env=parser_environment,
    )
    if result.returncode != 0:
        errors.append("build-windows-release.ps1: PowerShell AST parse failed: " +
                      (result.stderr.strip() or result.stdout.strip()))
        return
    try:
        decoded = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        errors.append(f"build-windows-release.ps1: PowerShell AST output invalid: {exc}")
        return
    if isinstance(decoded, dict):
        decoded = [decoded]
    command_text = "\n".join(item.get("text", "") for item in decoded)
    for description, pattern in (
        ("live --help smoke", r"(?s)Invoke-Checked\s+\$server.*?--help"),
        ("server smoke harness", r"(?s)Invoke-Checked\s+python.*?smokeHarness"),
        ("CRT audit", r"Invoke-CrtAudit"),
        ("isolated unsupported-tier smoke",
         r"Invoke-UnsupportedTierProbe\b.*?\$tierTest"),
        ("unsupported-tier fake-tool contract",
         r"Invoke-UnsupportedTierContractTests"),
    ):
        if not re.search(pattern, command_text, re.IGNORECASE):
            errors.append(f"build-windows-release.ps1: AST missing active {description}")
    _validate_powershell_ast_order(decoded, errors)
    contract = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script),
         "-SourceDir", str(script.parents[1]), "-ContractTest"],
        text=True, capture_output=True, check=False, shell=False,
    )
    if contract.returncode != 0:
        errors.append("build-windows-release.ps1: injected fake-tool contract failed: " +
                      contract.stdout + contract.stderr)


def check(root: Path, build_dir: Path | None = None,
          source_manifest: Path | None = None) -> list[str]:
    errors: list[str] = []
    # Runs BEFORE source discovery, and independently of it: this rule sweeps the
    # tree by glob and needs neither CMake nor a build directory, so it still
    # reports when the codemodel cannot be derived.
    errors.extend(noreturn_nonvoid_errors(root))
    try:
        source_paths = shipped_server_sources(root, build_dir, source_manifest)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"shipped-server source discovery failed: {exc}")
        source_paths = set()
    for required in REQUIRED_CPP:
        if required not in source_paths:
            errors.append(
                f"{required}: required implementation is not reachable from "
                "the shipped server target"
            )
    texts = {
        relative: require_file(root, relative, errors)
        for relative in sorted(source_paths)
        if Path(relative).suffix.lower() in CPP_SUFFIXES
    }
    cmake = require_file(root, "CMakeLists.txt", errors)
    warnings_path = root / "cmake/CompilerWarnings.cmake"
    # #1649: only the flags that reach THIS project's targets answer for the
    # policy. cmake/CompilerWarnings.cmake is kept WHOLE -- it is the policy
    # module, and it applies the flags through a function parameter (`${target}`)
    # that no caller-independent reading can resolve.
    warnings = without_foreign_target_compile_options(cmake)
    if warnings_path.is_file():
        warnings += "\n" + warnings_path.read_text(encoding="utf-8")
    build_script = require_file(root, "scripts/build-windows-release.ps1", errors)
    cpu_baseline = require_file(root, "src/vt/cpu/cpu_matmul_elem.cpp", errors)

    for relative, source in texts.items():
        active_source = without_cpp_comments(source)
        for number, line in windows_possible_lines(active_source):
            full_source_posix = (
                re.search(POSIX_PATTERNS[0], line) or
                re.search(r"(?<![A-Za-z0-9_])::stat\s*\(", line) or
                re.search(r"\bS_IS(?:DIR|REG)\s*\(", line)
            )
            platform_boundary = (
                relative in REQUIRED_CPP or
                relative.startswith("src/vllm/platform/")
            )
            scoped_posix = platform_boundary and any(
                re.search(pattern, line) for pattern in POSIX_PATTERNS
            )
            if full_source_posix or scoped_posix:
                errors.append(f"{relative}:{number}: unguarded POSIX include/call reaches Windows")
            if (re.search(
                    r"\b(?:CreateFileA|LoadLibraryA|MoveFileExA|DeleteFileA)\b",
                    line,
                ) or (platform_boundary and re.search(r"\.string\s*\(\)", line))):
                errors.append(f"{relative}:{number}: lossy Windows path conversion/API is forbidden")
            if re.search(r'\bf(?:open|reopen)\s*\([^)]*,\s*"[rwab+]*e"\s*[,)]', line):
                errors.append(
                    f"{relative}:{number}: glibc-only fopen 'e' "
                    f"(O_CLOEXEC) mode is invalid on MSVC and triggers __fastfail"
                )

    all_source = "\n".join(texts.values())
    if has_shell_process_launch(all_source):
        errors.append("server process launch: shell invocation is forbidden; execute argv directly")

    runtime_values = re.findall(
        r"CMAKE_MSVC_RUNTIME_LIBRARY\s+(?:\"([^\"]+)\"|([^\s\)]+))", cmake
    )
    runtime_values = [quoted or bare for quoted, bare in runtime_values]
    if not runtime_values or any(
        value not in {"MultiThreaded", "MultiThreaded$<$<CONFIG:Debug>:Debug>"}
        for value in runtime_values
    ):
        errors.append("CMakeLists.txt: exact static MSVC runtime (/MT) is required")
    global_options = without_set_source_properties(cmake)
    if re.search(r"(?i)/arch\s*:\s*AVX2", global_options):
        errors.append("CMakeLists.txt: global /arch:AVX2 contaminates the portable baseline")
    # Asserted on the flags that reach the C/C++ compile, by TOKEN. A substring
    # test passed a tree whose CXX arm said `/WX-` because the only bare `/WX`
    # left was on `$<COMPILE_LANGUAGE:OBJCXX>` (#774).
    cxx_warning_flags = msvc_cxx_flag_text(warnings)
    missing = [
        flag for flag in ("/W4", "/WX")
        if not has_msvc_flag(cxx_warning_flags, flag)
    ]
    if missing:
        errors.append(
            "CMakeLists.txt: MSVC /W4 /WX policy is required on the C/C++ "
            f"compile; missing {' '.join(missing)}"
        )
    # The disable spellings of the SAME two flags. Present-and-cancelled is a
    # different repair from absent, so it is a different error.
    negated = [
        flag for flag in ("/WX-", "/W0", "/w")
        if has_msvc_flag(cxx_warning_flags, flag)
    ]
    if negated:
        errors.append(
            "CMakeLists.txt: MSVC /W4 /WX policy is negated on the C/C++ "
            f"compile by {' '.join(negated)}"
        )
    if re.search(r"__attribute__\s*\(\(\s*target\s*\(\s*\"f16c\"", cpu_baseline):
        errors.append("cpu_matmul_elem.cpp: F16C must be isolated in a dedicated translation unit")
    f16c_properties = source_properties(cmake, "src/vt/cpu/cpu_matmul_elem_f16c.cpp")
    if not all(token in f16c_properties for token in ("/arch:AVX", "-mf16c")):
        errors.append("CMakeLists.txt: dedicated F16C translation unit needs compiler-specific ISA flags")
    if re.search(r"(?i)/arch\s*:\s*AVX2", f16c_properties):
        errors.append("CMakeLists.txt: F16C translation unit must not require AVX2")

    for relative, source in texts.items():
        active_source = without_cpp_comments(source)
        if any("__builtin_clzll" in line
               for _, line in windows_possible_lines(active_source)):
            errors.append(f"{relative}: non-portable compiler builtin reaches MSVC")

    central_nominmax = _has_central_msvc_nominmax(cmake)
    for relative, source in texts.items():
        active_source = without_cpp_comments_and_literals(source)
        definitions = _local_nominmax_definitions(active_source)
        for line, guarded in definitions:
            if not guarded:
                errors.append(
                    f"{relative}:{line}: unguarded source-local NOMINMAX is forbidden"
                )
        windows_header = re.search(
            r"(?m)^\s*#\s*include\s*<windows\.h>", active_source
        )
        if windows_header is None or central_nominmax:
            continue
        header_line = active_source.count("\n", 0, windows_header.start()) + 1
        if not any(guarded and line < header_line for line, guarded in definitions):
            errors.append(
                f"{relative}: NOMINMAX must be defined centrally or by an "
                "absence-guarded local fallback before windows.h"
            )

    server = "\n".join(texts.get(name, "") for name in REQUIRED_CPP[:3])
    for marker in ("CreateProcessW", "SetConsoleCtrlHandler"):
        if marker not in server:
            errors.append(f"server_main.cpp: required Win32 process/console support missing ({marker})")
    console = texts.get("src/vllm/platform/console_shutdown.cpp", "")
    _validate_console_protocol(console, errors)

    lmcache = texts.get("src/vllm/v1/kv_offload/lmcache/remote_client.cpp", "")
    if not all(marker in lmcache for marker in ("WSAStartup", "WSAGetLastError", "closesocket")):
        errors.append("remote_client.cpp: LMCache Winsock support is missing or silently disabled")
    if not re.search(r"(?s)if\s*\([^\)]*==\s*0\s*\).*?Close\s*\(\s*\).*?throw", lmcache):
        errors.append("remote_client.cpp: peer-close must invalidate the owned socket")

    fs_io = texts.get("src/vllm/v1/kv_offload/fs_io.cpp", "")
    if not all(marker in fs_io for marker in ("CreateFileW", "FlushFileBuffers", "MoveFileExW")):
        errors.append("fs_io.cpp: Windows KV filesystem support is missing or silently disabled")
    if "CREATE_NEW" not in fs_io:
        errors.append("fs_io.cpp: exclusive temporary creation requires CREATE_NEW")
    if not all(flag in fs_io for flag in ("MOVEFILE_REPLACE_EXISTING", "MOVEFILE_WRITE_THROUGH")):
        errors.append("fs_io.cpp: atomic publish requires replace and MOVEFILE_WRITE_THROUGH")

    required_script_markers = (
        "Visual Studio 17 2022",
        "Release",
        "bin/vllm-server.exe",
        "--help",
        "/health",
        "/version",
        "CTRL_BREAK_EVENT",
        "VT_CPU_MATMUL_TIER",
        '"portable"',
        '"avx2"',
        '"amx"',
    )
    active_script = _active_powershell(build_script)
    _validate_powershell_source_order(build_script, errors)
    _validate_unsupported_tier_contract(build_script, errors)
    for marker in required_script_markers:
        if marker not in active_script:
            errors.append(f"build-windows-release.ps1: required native CPU gate missing ({marker})")
    if not (re.search(r"(?i)\bdumpbin\b", active_script) and
            ('"/directives"' in active_script or
             re.search(r"(?i)\bdumpbin\s+/directives\b", active_script))):
        errors.append("build-windows-release.ps1: static-library CRT directive audit is missing")
    if not (re.search(r"(?i)\bdumpbin\b", active_script) and
            ('"/imports"' in active_script or
             re.search(r"(?i)\bdumpbin\s+/imports\b", active_script))):
        errors.append("build-windows-release.ps1: PE CRT import audit is missing")
    required_crt_rejections = (
        "VCRUNTIME", "MSVCP", "CONCRT", "UCRTBASE", "api-ms-win-crt-",
        "MSVCR", "LIBCMT",
    )
    if not all(token in active_script for token in required_crt_rejections):
        errors.append("build-windows-release.ps1: dynamic CRT rejection set is incomplete")
    if "--help" not in active_script:
        errors.append("build-windows-release.ps1: live --help smoke is missing")
    # Unit fixtures inject a generated source manifest and exercise the
    # cross-platform structural rules. Only the real repository script owns
    # the native AST/fake-tool execution contract.
    if source_manifest is None:
        _validate_powershell_ast(root / "scripts/build-windows-release.ps1", errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--test-source-manifest", type=Path,
        help="explicit source fixture for checker tests; normal runs use CMake codemodel",
    )
    args = parser.parse_args()
    errors = check(args.root.resolve(), args.build_dir, args.test_source_manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Windows portability contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
