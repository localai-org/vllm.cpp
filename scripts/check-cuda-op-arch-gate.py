#!/usr/bin/env python3
"""Fail if a CUDA op that must exist on EVERY CUDA arch is registered from a
feature-gated translation unit.

THE DEFECT THIS EXISTS TO PREVENT ALREADY HAPPENED (issue #960, and #844 is the
same defect seen from the fallback's end).

`vt::QuantFp8Static`'s CUDA kernel is `out[i] = e4m3(x[i] * (1/scale))` — a plain
elementwise convert with no cutlass dependency of any kind. It lived in
`src/vt/cuda/cuda_matmul_fp8_cutlass.cu`, whose ONLY build gate is
`VT_CUTLASS_FP8_ARCHS`: CMake adds that TU to `_FP8_CUTLASS_SOURCES`, and hence
to the `vllm` target, only when the requested arch list intersects the
cutlass-fp8 feature cell. On sm_110 (Thor) that intersection is EMPTY, which is
the documented normal profile for the arch and not a misconfiguration. So the
kernel was simply not compiled, `OpId::kQuantFp8Static` was never registered for
`DeviceType::kCUDA`, and a CUDA queue asking for it resolved to the portable CPU
reference tier — which dereferenced device pointers and killed the process.

NOTHING REFUSED FIRST, which is what made it expensive. The op's GEMM partner
`kMatmulFp8CublasLt` is registered unconditionally in `cuda_matmul.cu`, so the
model-layer predicate that keys on it passed, the build looked complete, and the
crash arrived one call later inside a "correct but slow" fallback banner.

WHY A STRUCTURAL CHECK AND NOT ONLY A TEST. `tests/vt/test_ops_fp8_cpu.cpp` G4
asserts the registration at run time and is the stronger statement — it observes
the property that actually matters rather than a proxy for it. But it can only
speak on a host that BUILT the CUDA backend without cutlass-fp8, and no CI job
does: the GB10 gate host resolves `cutlass-fp8: ENABLED`, where the defect is
unreachable by construction, and every other job is CPU-only. This file runs in
the ordinary checker lane on every host, reads the build description rather than
a binary, and therefore fails at PR time on the machine of whoever moved the
registration back. Neither instrument subsumes the other; the runtime test is the
claim, this is the tripwire.

WHAT IS ASSERTED, per entry in `REQUIRED`, with no inference about what a kernel
"needs":

  (a) HOME       — the named source file is listed in the UNCONDITIONAL CUDA
                   source list: the `target_sources(vllm PRIVATE ...)` whose
                   enclosing `if()` stack is exactly `[VLLM_CPP_CUDA]`. A file
                   moved under any further condition (`if(VLLM_CPP_CUTLASS)`,
                   `if(VT_CUTLASS_FP8_ARCHS)`, an `else()` branch, a `foreach`)
                   is NOT in that list and fails here.
  (b) REGISTERED — that file contains exactly ONE
                   `RegisterOp(OpId::<name>, DeviceType::kCUDA, ...)`.
  (c) UNGUARDED  — that registration sits at preprocessor-conditional depth 0 in
                   the file, so wrapping it in `#ifdef VT_CUTLASS_FP8` — which
                   would restore the exact original behaviour while leaving the
                   TU in the unconditional list — is not a pass.
  (d) EXCLUSIVE  — no OTHER CUDA source registers the same OpId for kCUDA. A
                   second, gated registration would let a runtime test pass on a
                   cutlass host for the wrong reason and would make (a)-(c)
                   describe a copy that is not the one being selected.

TEXT THE COMPILER NEVER SEES IS NOT A PASS. The C++ side runs through
`checker_text.normalize_source`, so a commented-out or `#if 0`-ed registration
reads as absent, which is what it is. CMake `#` comments are stripped the same
way before the source list is parsed.

WHAT THIS DOES NOT DO, stated plainly. It does not decide which ops BELONG in the
required set — that is a judgement recorded in `REQUIRED` with its issue, one
line per op, and adding a feature-gated op to it would be wrong. It does not
check that the kernel is correct, that the arch can run it, or that any other op
is registered anywhere. And it reads CMake lexically: a source list assembled
through a variable or a `foreach` is invisible to it and would be reported as a
MISSING home rather than silently accepted.

Usage:
    python3 scripts/check-cuda-op-arch-gate.py [--report]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checker_text import blank_out, normalize_source  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# The ops whose CUDA registration must not depend on a CUDA FEATURE. One line per
# op: the OpId, the TU that owns it, and the issue that put it here. Keep this
# short and argued — an op with a genuinely arch-specific kernel (the cutlass
# GEMMs, Marlin, FA2) does NOT belong in it, because for those the feature gate is
# the correct behaviour and a missing registration is an honest refusal.
REQUIRED = (
    # #960: the static per-tensor fp8 activation quant. No cutlass dependency;
    # trapped in the cutlass-fp8 TU; unreachable on every non-cutlass-fp8 CUDA
    # arch, where it fell to the reference tier and segfaulted (#844).
    ("kQuantFp8Static", "src/vt/cuda/cuda_quant_fp8.cu", "#960"),
    # #1189 M1: the dynamic per-token, per-group fp8 activation quant. Same
    # property as its neighbour and the same consequence if it moves -- a max,
    # two divides and a hardware e4m3 convert, with no cutlass dependency, so a
    # CUDA queue must never fall through to the host reference tier and
    # dereference device pointers. It shares the TU deliberately.
    ("kQuantFp8Group", "src/vt/cuda/cuda_quant_fp8.cu", "#1189"),
)

# Where a stray duplicate registration could hide. Every CUDA-side source.
CUDA_SOURCE_GLOBS = ("src/vt/cuda/*.cu", "src/vt/cuda/*.cpp", "src/vllm/platforms/cuda.cpp")

_CMAKE_COMMENT = re.compile(r"#[^\n]*")
_CMAKE_TOKEN = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.M)
_CPP_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*", re.M)


def strip_cmake_comments(text: str) -> str:
    """Blank `#` comments, position-preserving (CMake has no block comment form
    we use). Bracket comments `#[[ ]]` are not used in this tree."""
    out = text
    for m in reversed(list(_CMAKE_COMMENT.finditer(text))):
        out = out[: m.start()] + blank_out(m.group(0)) + out[m.end() :]
    return out


def _command_span(text: str, open_paren: int) -> int:
    """Offset just past the `)` that closes the paren at `open_paren`."""
    depth = 0
    i = open_paren
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def unconditional_cuda_sources(cmake_text: str) -> set[str]:
    """Sources given to `target_sources(vllm PRIVATE ...)` under an `if()` stack of
    exactly `[VLLM_CPP_CUDA]` and no `foreach`/`function`/`macro` scope.

    The stack is tracked over the WHOLE file rather than by searching for a single
    block, so moving the call one level deeper — into `if(VLLM_CPP_CUTLASS)`, into
    an `else()`, into a `foreach()` — changes the answer, which is the motion this
    checker exists to notice.
    """
    text = strip_cmake_comments(cmake_text)
    if_stack: list[str] = []
    other_depth = 0
    sources: set[str] = set()

    for m in _CMAKE_TOKEN.finditer(text):
        cmd = m.group(1).lower()
        open_paren = m.end() - 1
        end = _command_span(text, open_paren)
        args = text[open_paren + 1 : end - 1]

        if cmd == "if":
            if_stack.append(args.strip())
        elif cmd == "elseif":
            if if_stack:
                if_stack[-1] = args.strip()
        elif cmd == "else":
            # An `else()` branch is NOT the `if()` condition. Marking it with a
            # sentinel keeps the stack depth right and keeps the branch out of the
            # unconditional set.
            if if_stack:
                if_stack[-1] = "!" + if_stack[-1]
        elif cmd == "endif":
            if if_stack:
                if_stack.pop()
        elif cmd in ("foreach", "while", "function", "macro"):
            other_depth += 1
        elif cmd in ("endforeach", "endwhile", "endfunction", "endmacro"):
            other_depth = max(0, other_depth - 1)
        elif cmd == "target_sources":
            if if_stack != ["VLLM_CPP_CUDA"] or other_depth != 0:
                continue
            words = args.split()
            if len(words) < 2 or words[0] != "vllm":
                continue
            for w in words[1:]:
                if w in ("PRIVATE", "PUBLIC", "INTERFACE"):
                    continue
                if "$" in w:  # a variable expansion: not a literal home
                    continue
                sources.add(w)
    return sources


def cuda_registrations(op: str, root: Path = REPO) -> dict[str, list[tuple[int, int]]]:
    """Every live `RegisterOp(OpId::<op>, DeviceType::kCUDA` in the CUDA sources.

    Returns {repo-relative path: [(line, preprocessor depth), ...]}.
    """
    pattern = re.compile(
        r"RegisterOp\s*\(\s*OpId::" + re.escape(op) + r"\s*,\s*DeviceType::kCUDA\b"
    )
    found: dict[str, list[tuple[int, int]]] = {}
    for glob in CUDA_SOURCE_GLOBS:
        for path in sorted(root.glob(glob)):
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = normalize_source(raw)
            hits = list(pattern.finditer(text))
            if not hits:
                continue
            # Preprocessor depth at each hit. Directive lines survive normalization
            # (`strip_preprocessor_disabled` blanks bodies, not directives), so a
            # `#if 0`-ed registration has already vanished from `text` above and a
            # real `#ifdef` is still counted here.
            depths: list[tuple[int, int]] = []
            marks = [(d.start(), d.group(1)) for d in _CPP_DIRECTIVE.finditer(text)]
            for h in hits:
                depth = 0
                for pos, kw in marks:
                    if pos >= h.start():
                        break
                    if kw in ("if", "ifdef", "ifndef"):
                        depth += 1
                    elif kw == "endif":
                        depth = max(0, depth - 1)
                depths.append((text.count("\n", 0, h.start()) + 1, depth))
            # POSIX spelling, always. `REQUIRED` names the home the way the
            # CMake source list spells it (forward slashes, because that list is
            # read as text), so a native-separator key misses the file the walk
            # just found: the home lookup reads 0 registrations and the
            # exclusivity loop reads the home as a SECOND file, and the gate
            # reports both halves of its own contradiction. `str()` is the
            # separator here on POSIX and a backslash on Windows, so this is
            # invisible on a POSIX host and in CI.
            found[path.relative_to(root).as_posix()] = depths
    return found


def check(report: bool = False, root: Path = REPO) -> list[str]:
    errors: list[str] = []
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    unconditional = unconditional_cuda_sources(cmake)
    if not unconditional:
        return [
            "CMakeLists.txt: found NO unconditional CUDA target_sources list. This "
            "checker cannot pass vacuously -- either the build description moved or "
            "the parser is broken; fix one of them."
        ]
    if report:
        print(f"unconditional CUDA sources under if(VLLM_CPP_CUDA): {len(unconditional)}")

    for op, home, issue in REQUIRED:
        regs = cuda_registrations(op, root)
        if report:
            print(f"{op} ({issue}) -> home {home}; registrations: {regs or '{}'}")

        # (a) HOME
        if home not in unconditional:
            errors.append(
                f"{op} ({issue}): {home} is NOT in the unconditional CUDA source list "
                f"(target_sources(vllm PRIVATE ...) directly under if(VLLM_CPP_CUDA)). "
                f"A CUDA arch outside the feature set now gets no kernel for this op "
                f"and falls to the portable CPU reference tier over DEVICE pointers."
            )

        # (b) REGISTERED and (c) UNGUARDED
        here = regs.get(home, [])
        if len(here) != 1:
            errors.append(
                f"{op} ({issue}): expected exactly ONE live "
                f"RegisterOp(OpId::{op}, DeviceType::kCUDA, ...) in {home}, found "
                f"{len(here)}. A commented-out or #if 0 registration reads as absent "
                f"here, which is what it is to the compiler."
            )
        else:
            line, depth = here[0]
            if depth != 0:
                errors.append(
                    f"{home}:{line}: {op} ({issue}): the CUDA registration sits at "
                    f"preprocessor-conditional depth {depth}. Guarding it re-creates "
                    f"the arch gate the unconditional TU was meant to remove."
                )

        # (d) EXCLUSIVE
        for path, hits in sorted(regs.items()):
            if path == home:
                continue
            lines = ", ".join(str(ln) for ln, _ in hits)
            errors.append(
                f"{path}:{lines}: {op} ({issue}) is ALSO registered for kCUDA here. "
                f"Its single home is {home}; a second registration lets a runtime "
                f"check pass on a feature-enabled host for the wrong reason."
            )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="print what was examined")
    ap.add_argument(
        "--root",
        default=str(REPO),
        help="repository root to examine (the checkout containing CMakeLists.txt)",
    )
    args = ap.parse_args()

    errors = check(report=args.report, root=Path(args.root))
    if errors:
        for e in errors:
            print(f"check-cuda-op-arch-gate: {e}", file=sys.stderr)
        print(
            f"\ncheck-cuda-op-arch-gate: FAIL ({len(errors)} error(s))",
            file=sys.stderr,
        )
        return 1
    print(f"check-cuda-op-arch-gate: OK ({len(REQUIRED)} op(s) pinned to an unconditional CUDA TU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
