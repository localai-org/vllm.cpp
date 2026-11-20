#!/usr/bin/env python3
"""Require every vt::tenstorrent symbol used by an always-compiled TU to be
public in src/vt/tenstorrent/tenstorrent_device.h with a non-TT stub.

WHY THIS EXISTS (ISSUE-LOCAL-01M7W2ZKR8HRBYJF6HCPP46AGH). VLLM_CPP_TENSTORRENT
defaults to OFF, and the default configure compiles almost the whole tree:
everything under src/ except src/vt/tenstorrent/** and the sources added
inside if(VLLM_CPP_TENSTORRENT) blocks of the top-level CMakeLists, plus every
test registered OUTSIDE the TT blocks in tests/CMakeLists.txt. The pattern
that keeps those TUs linkable is the public header tenstorrent_device.h
(ttnn-free) declaring each symbol under #ifdef VLLM_CPP_TENSTORRENT with an
inline no-op in the #else branch. A symbol that skips the pattern — declared
only in the ttnn-pulling internal header, or forward-declared ad hoc in a
test, or declared unconditionally in the public header with a TT-only
definition — is an undefined reference waiting for the next CPU-only lane:
that is exactly how test_qwen38_dflash2_drafter_tt.cpp broke build-newest-gcc
with vt::tenstorrent::EnsureHost.

THE RULES, over the always-compiled TU set (comment/string-stripped text,
preprocessor guard tracking):

  R1  no unguarded #include of a header under src/vt/tenstorrent/ other than
      tenstorrent_device.h (the internal headers pull ttnn headers).
  R2  no unguarded re-opening of `namespace vt::tenstorrent {` to declare
      symbols — declare them in the public header with a stub instead.
  R3  every unguarded vt::tenstorrent::X reference (and using-declaration of
      X) has X declared in a NON-TT branch of tenstorrent_device.h (the stub
      surface) or defined unconditionally in that header.
  R4  the public header is self-consistent: a symbol declared in a TT-positive
      branch has a matching #else stub, and no symbol is declared
      unconditionally without an unconditional (inline) definition in the
      header — an unconditional declaration whose definition lives in a
      TT-only TU is the dangler R3 cannot see until someone references it.

The CMake scanners are text heuristics, like every checker in scripts/: they
recognise exact-condition if(VLLM_CPP_TENSTORRENT) / elseif(...) blocks so
if(VLLM_CPP_TENSTORRENT STREQUAL "AUTO") and else() arms are not misread. The
TT blocks today are single contiguous target_sources(...) calls; a file path
split across lines inside one would be missed, which only over-scans (a
missed TT-only file is scanned as always-compiled; harmless unless it carries
unguarded TT refs, which R3 would then rightly flag).

Exit 0 clean, 1 violations, 2 usage/tree error. --root selects the tree
(default: the repo containing this script); --report prints every violation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PUBLIC_HEADER = "src/vt/tenstorrent/tenstorrent_device.h"
TT_DIR = "src/vt/tenstorrent/"
TT_MACRO = "VLLM_CPP_TENSTORRENT"

# ---------------------------------------------------------------------------
# C preprocessor guard tracking
# ---------------------------------------------------------------------------

_CPP_COND = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)")


def _expr_polarity(expr: str) -> tuple[bool, bool]:
    """(positive-branch is TT, negative/else-branch is TT) for a #if expr."""
    mentions = TT_MACRO in expr
    # negated form: '!defined(X)', '!X', 'not defined(X)'
    negated = bool(re.search(r"!\s*defined\s*\(\s*" + TT_MACRO, expr)) or bool(
        re.search(r"!\s*" + TT_MACRO + r"\b", expr)
    ) or bool(re.search(r"\bnot\s+defined\s*\(\s*" + TT_MACRO, expr))
    if not mentions:
        return False, False
    return not negated, negated


def guard_stack_scan(lines):
    """Yield (lineno, line, tt_guarded) per line: tt_guarded True when some
    enclosing frame is TT-positive in the current branch."""
    stack = []  # frames: [pos_tt, neg_tt, seen_else]
    for i, line in enumerate(lines):
        m = _CPP_COND.match(line)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ("if", "ifdef", "ifndef"):
                if d == "ifdef":
                    pos, neg = (TT_MACRO in rest), False
                elif d == "ifndef":
                    pos, neg = False, (TT_MACRO in rest)
                else:
                    pos, neg = _expr_polarity(rest)
                stack.append([pos, neg, False])
            elif d == "elif":
                if stack:
                    pos, neg = _expr_polarity(rest)
                    stack[-1] = [pos, neg, True]
            elif d == "else":
                if stack:
                    stack[-1][2] = True
            elif d == "endif":
                if stack:
                    stack.pop()
            yield i + 1, line, any((f[0] if not f[2] else f[1]) for f in stack)
            continue
        yield i + 1, line, any((f[0] if not f[2] else f[1]) for f in stack)


# ---------------------------------------------------------------------------
# CMake TT-block scanner (exact condition match only)
# ---------------------------------------------------------------------------

_CMAKE_COND = re.compile(r"^\s*(if|elseif|else|endif)\s*\((.*?)\)\s*(#.*)?$", re.IGNORECASE)
_CMAKE_FILE = re.compile(r"(src/[\w./-]+\.(?:cpp|h|hpp|cu|hip))")
_CMAKE_TEST = re.compile(r"vllm_cpp_add_test\(\s*(\S+?)\s+(\S+?)[\s)]")


def _cmake_tt_spans(path: str):
    """Yield (lineno, line, in_tt_block). in_tt_block is True inside a branch
    whose condition is exactly VLLM_CPP_TENSTORRENT."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    stack = []
    for i, line in enumerate(lines):
        m = _CMAKE_COND.match(line)
        if not m:
            yield i + 1, line, any(stack)
            continue
        d, rest = m.group(1).lower(), m.group(2).strip()
        is_tt = rest == TT_MACRO
        if d == "if":
            stack.append(is_tt)
        elif d == "elseif":
            if stack:
                stack[-1] = is_tt
        elif d == "else":
            if stack:
                stack[-1] = False  # else of an exact-TT if is non-TT
        elif d == "endif":
            if stack:
                stack.pop()
        yield i + 1, line, any(stack)


# ---------------------------------------------------------------------------
# comment / string stripping (keeps line structure)
# ---------------------------------------------------------------------------

def strip_comments_strings(text: str) -> str:
    out = []
    i, n = 0, len(text)
    state = None  # None, 'line', 'block', 'sq', 'dq', 'raw'
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                state = "line"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                out.append("  ")
                i += 2
                continue
            if c == '"':
                if i > 0 and text[i - 1] == "R" and text[i + 1 :].startswith("("):
                    state = "raw"
                    out.append("  ")
                    i += 1
                    continue
                state = "dq"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                state = "sq"
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
        elif state == "line":
            if c == "\n":
                state = None
                out.append(c)
            i += 1
        elif state == "block":
            if c == "*" and nxt == "/":
                state = None
                i += 2
                continue
            if c == "\n":
                out.append(c)
            i += 1
        elif state in ("dq", "sq"):
            if c == "\\":
                i += 2
                continue
            if (state == "dq" and c == '"') or (state == "sq" and c == "'"):
                state = None
            if c == "\n":
                out.append(c)
            i += 1
        elif state == "raw":
            if c == '"':
                # look ahead for the closing )" — approximate: end at next quote
                j = text.find('")', i)
                if j == -1:
                    state = None
                    i += 1
                else:
                    state = None
                    i = j + 2
                continue
            if c == "\n":
                out.append(c)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# always-compiled TU set
# ---------------------------------------------------------------------------

def always_compiled_files(root: str) -> tuple[list[str], list[str]]:
    """(always-compiled source files, tt-only source files) as repo-relative
    paths."""
    tt_src = set()
    main_cml = os.path.join(root, "CMakeLists.txt")
    if os.path.isfile(main_cml):
        for _, line, in_tt in _cmake_tt_spans(main_cml):
            if in_tt:
                for m in _CMAKE_FILE.finditer(line):
                    tt_src.add(m.group(1))
    tt_src.update(p for p in _walk(root, "src") if p.startswith(TT_DIR))
    always = []
    for base in ("src", "include", "examples"):
        for p in _walk(root, base):
            if p.startswith(TT_DIR) or p in tt_src:
                continue
            always.append(p)
    tests_cml = os.path.join(root, "tests", "CMakeLists.txt")
    if os.path.isfile(tests_cml):
        for _, line, in_tt in _cmake_tt_spans(tests_cml):
            if in_tt:
                continue
            m = _CMAKE_TEST.search(line)
            if m and m.group(2).endswith((".cpp", ".h", ".hpp")):
                p = "tests/" + m.group(2)
                if os.path.isfile(os.path.join(root, p)):
                    always.append(p)
    return sorted(set(always)), sorted(tt_src)


def _walk(root: str, base: str) -> list[str]:
    out = []
    b = os.path.join(root, base)
    if not os.path.isdir(b):
        return out
    for dirpath, dirnames, fnames in os.walk(b):
        dirnames[:] = [d for d in dirnames if d not in (".git", "build")]
        for fn in fnames:
            if fn.endswith((".cpp", ".h", ".hpp", ".cu", ".cc", ".cxx")):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return out


# ---------------------------------------------------------------------------
# public header parse: stub surface + declared symbols
# ---------------------------------------------------------------------------

_FUNC_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?(;|=|\{)")
_TYPE_DECL = re.compile(r"\b(?:struct|class|enum(?:\s+class)?)\s+([A-Za-z_]\w*)")


def _logical_statements(lines_with_tt):
    """Join physical lines into declaration statements (parens balanced and
    ending in ; { or }), keeping the TT-branch flag of the first line.
    Statements inside a function body (brace depth > 0) are skipped by the
    caller via depth tracking here: only depth-0 statements are yielded."""
    buf = []
    buf_tt = False
    buf_start = 0
    depth = 0
    for lineno, line, tt in lines_with_tt:
        if depth > 0:
            depth += line.count("{") - line.count("}")
            continue
        if _CPP_COND.match(line) or not line.strip():
            # preprocessor and blank/comment-only lines never start a
            # declaration; the branch flag latches at the first real line
            continue
        if not buf:
            buf_tt = tt
            buf_start = lineno
        buf.append(line)
        depth += line.count("{") - line.count("}")
        pdepth = sum(l.count("(") - l.count(")") for l in buf)
        joined = " ".join(buf)
        if pdepth <= 0 and re.search(r"[;{}]\s*$", joined):
            yield buf_start, joined, buf_tt
            buf = []
            # a namespace block does not hide declarations; a function or
            # struct body does (its lines are not declarations)
            if re.match(r"\s*namespace\b", joined):
                depth = 0
            else:
                depth = joined.count("{") - joined.count("}")
    if buf:
        yield buf_start, " ".join(buf), buf_tt


def parse_public_header(path: str):
    """Return (stub_names, danglers) for the public header.

    stub_names: identifiers with a DEFINITION (inline body) in a NON-TT
    branch or unconditionally — the linkable surface for CPU builds. A mere
    `;` declaration is never linkable by itself.
    danglers (R4): identifiers declared UNCONDITIONALLY with `;` and no
    inline definition — their definitions live in TT-only TUs, so any
    always-compiled reference is an undefined reference. Cure: wrap the
    declaration in #ifdef/#else with a no-op stub, or move it into a
    TT-only #ifdef section when no non-TT TU may reference it (R3 enforces
    that).
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = strip_comments_strings(text).splitlines()
    stub_names = set()
    uncond_decl = {}  # name -> has_unconditional_def
    for lineno, stmt, tt in _logical_statements(guard_stack_scan(lines)):
        if tt:
            continue  # TT-branch declarations are policed by R3 at the use site
        for m in _TYPE_DECL.finditer(stmt):
            stub_names.add(m.group(1))
            uncond_decl[m.group(1)] = True
        for m in _FUNC_DECL.finditer(stmt):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "return", "sizeof"):
                continue
            if m.group(2) == "{":
                stub_names.add(name)
                uncond_decl[name] = True
            else:
                uncond_decl.setdefault(name, False)
    danglers = sorted(n for n, ok in uncond_decl.items() if not ok)
    return stub_names, danglers


# ---------------------------------------------------------------------------
# TU scan: R1, R2, R3
# ---------------------------------------------------------------------------

_REF = re.compile(r"vt::tenstorrent::([A-Za-z_]\w*)")
_USING = re.compile(r"using\s+(?:namespace\s+)?vt::tenstorrent(?:::([A-Za-z_]\w*))?\s*;")
_INCLUDE_TT = re.compile(r'#\s*include\s*[<"][^>"]*vt/tenstorrent/([\w.]+)[>"]')
_NS_REOPEN = re.compile(r"\bnamespace\s+vt::tenstorrent\s*\{")


def scan_tu(root: str, rel: str, stub_names: set[str]):
    path = os.path.join(root, rel)
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    raw_lines = text.splitlines()
    lines = strip_comments_strings(text).splitlines()
    violations = []
    for (lineno, line, tt), raw in zip(guard_stack_scan(lines), raw_lines):
        if tt:
            continue
        # the include path is a string literal, which stripping erases —
        # match includes on the RAW line
        for m in _INCLUDE_TT.finditer(raw):
            if m.group(1) != "tenstorrent_device.h":
                violations.append((lineno, "R1", f"internal TT header include '{m.group(1)}' — include only vt/tenstorrent/tenstorrent_device.h (ttnn-free) or guard with #ifdef " + TT_MACRO))
        if _NS_REOPEN.search(line):
            violations.append((lineno, "R2", "re-opens namespace vt::tenstorrent to declare symbols — declare them in tenstorrent_device.h with an #else stub instead"))
        for m in _REF.finditer(line):
            name = m.group(1)
            if name not in stub_names:
                violations.append((lineno, "R3", f"vt::tenstorrent::{name} has no non-TT stub in {PUBLIC_HEADER} — add an #else no-op there or guard this site with #ifdef {TT_MACRO}"))
        for m in _USING.finditer(line):
            name = m.group(1)
            if name is None:
                violations.append((lineno, "R3", f"'using namespace vt::tenstorrent' — qualify each use and give it a stub in {PUBLIC_HEADER}"))
            elif name not in stub_names:
                violations.append((lineno, "R3", f"using vt::tenstorrent::{name}: no non-TT stub in {PUBLIC_HEADER}"))
    return violations


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--report", action="store_true", help="print every violation (default prints all too; kept for sweep symmetry)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    header = os.path.join(root, PUBLIC_HEADER)
    if not os.path.isfile(header):
        print(f"missing public header {PUBLIC_HEADER} under {root}")
        return 2
    stub_names, danglers = parse_public_header(header)
    always, tt_only = always_compiled_files(root)
    if not always:
        print(f"no always-compiled sources found under {root} — wrong --root?")
        return 2
    failures = []
    for rel in always:
        for lineno, rule, msg in scan_tu(root, rel, stub_names):
            failures.append(f"{rel}:{lineno}: [{rule}] {msg}")
    for name in danglers:
        failures.append(f"{PUBLIC_HEADER}: [R4] {name} is declared without a non-TT stub or an unconditional definition — its definition lives in a TT-only TU; add an #else no-op")
    for line in failures:
        print(line)
    print(f"check-tt-public-surface: {len(always)} always-compiled TUs, {len(tt_only)} TT-only, {len(stub_names)} stubbed symbols, {len(failures)} violation(s)")
    if failures:
        print("Cure: declare the symbol in " + PUBLIC_HEADER + " with an #else no-op stub, or guard the call site with #ifdef " + TT_MACRO + ".")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
