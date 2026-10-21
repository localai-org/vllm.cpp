#!/usr/bin/env python3
"""The two diagnostic classes that the windows-msvc /WX lane punishes.

``windows-msvc-cpu`` and ``windows-msvc-vulkan`` build with ``/W4 /WX``, so
two diagnostic classes kill the build there while every POSIX lane stays
green.  Both have a host-side reproduction:

``C4456`` — "declaration of 'x' hides previous local declaration"
    A declaration inside a block that hides a local from an enclosing block
    of the SAME function.  GCC's and Clang's ``-Wshadow`` report the same
    defect at the same file, line and column, so the compile probe below is
    the RED: on the pre-fix source it prints ``kolibri1_forward.cpp:326:17:
    'prof' shadows a previous local``, which is the site the MSVC job log
    names.

``C4101`` — "unreferenced local variable"
    MSVC reports it for a local of an EMPTY class type.
    ``kolibri1_numa::PolicyGuard`` is empty off Linux
    (``kolibri1_numa.h``), so the two guard locals are unreferenced there
    while the Linux build has a class with a non-trivial constructor, and
    GCC's ``-Wunused-variable`` fires for neither.  No host compiler can
    express this one, so the portable statement of intent is the
    ``[[maybe_unused]]`` attribute and the source-level assertion below is
    what holds it.

The filter is exactly the C4456 class and nothing else.  MSVC splits the
shadow family four ways -- C4456 hides a previous LOCAL, C4457 hides a
function PARAMETER, C4458 hides a class MEMBER, C4459 hides a GLOBAL -- so
the probe matches the local class and leaves the other three alone.  The
member class in particular is common in this tree's headers and is not a
C4456.

The two host compilers differ in one place, and the probe takes that into
account instead of guessing.  Clang's ``-Wshadow`` has MSVC's scope: it
reports a local hiding a local in the same function, and it does NOT report
a lambda body hiding a local of the ENCLOSING function, because the lambda
body is its own function.  GCC's ``-Wshadow`` reports that lambda case too,
so a GCC-only hit in a lambda is a candidate rather than a verdict.  The
measured case is ``src/vllm/multimodal/ltx2_video.cpp``: GCC reports three
such hits, Clang reports none, and the MSVC lane compiled that file clean
(job 118938856642, 2026-10-20T23:23:15Z).

The probe compiles the PRODUCTION translation units with the production
include paths; it does not reconstruct the shape in a synthetic file.
"""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]

# Production translation units the windows-msvc lane compiles.  A local
# shadow in any of them is a C4456 on Windows and a green build on Linux.
PROBED_UNITS = (
    "src/vllm/model_executor/models/kolibri1_forward.cpp",
    "src/vllm/model_executor/models/kolibri1_tt_forward.cpp",
    "src/vllm/model_executor/models/kolibri1_weights.cpp",
)

# The C4101 class: a local of kolibri1_numa::PolicyGuard, which is an empty
# class off Linux.  Every such local must carry the attribute.
GUARD_DECL = re.compile(
    r"(?P<attr>\[\[maybe_unused\]\]\s+)?"
    r"(?:const\s+)?(?:kolibri1_numa::)?PolicyGuard\s+(?P<name>\w+)\s*;"
)

# GCC (>= 15) words the local-shadow class "shadows a previous local";
# Clang words it "shadows a local variable".  Both are C4456.  The member,
# parameter and global classes are different diagnostics and are not
# matched.
LOCAL_SHADOW = ("shadows a previous local", "shadows a local variable")

# The context line a compiler prints above a diagnostic, and the one GCC
# prints for a diagnostic inside a lambda body.
CONTEXT = re.compile(r"^.*: In (?P<ctx>.*):$")
DIAGNOSTIC = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<col>\d+): (?:warning|error):"
)


def _compilers():
    """The host C++ compilers, in probe order.

    Every one of them runs, because they do not agree on the lambda case.
    An empty list means the probe cannot run, and the caller says so.
    """
    found = []
    configured = os.environ.get("CXX")
    if configured:
        parts = shlex.split(configured)
        if parts:
            found.append(parts)
    # `c++` and `g++` are the same driver family on every system, so `c++`
    # is only a fallback for a host that has no `g++` at all.
    for name in ("clang++", "g++"):
        path = shutil.which(name)
        if path:
            found.append([path])
    if not any(entry[0].endswith("g++") for entry in found):
        path = shutil.which("c++")
        if path:
            found.append([path])
    return found


def _probe(compiler, unit):
    """Compile one production translation unit with -Wshadow.

    Returns (returncode, stderr).  -fsyntax-only keeps the probe to the
    front end, which is where the shadow diagnostic lives, and keeps it
    fast enough for a unit test.
    """
    command = list(compiler) + [
        "-std=c++20",
        "-fsyntax-only",
        "-Wshadow",
        "-I", str(ROOT / "src"),
        "-I", str(ROOT / "include"),
        "-I", str(ROOT / "third_party"),
        str(ROOT / unit),
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return done.returncode, done.stderr


def _relative(path):
    """The path as the probe spells it, relative to the repository root."""
    text = path.replace(str(ROOT) + "/", "").replace(str(ROOT) + "\\", "")
    return text.lstrip("./")


def _local_shadows(stderr, unit):
    """The local-shadow diagnostics the compiler reported for `unit`.

    Returns a mapping from a site to whether the site is only a GCC lambda
    candidate.  A site is a (file, line, column) triple, so two compilers
    report the same site the same way.

    A GCC hit inside a lambda body is a CANDIDATE, not a verdict: Clang and
    MSVC both scope the local class to one function, and a lambda body is
    its own function, so neither reports it.  A hit with Clang's wording is
    always a verdict, and a site any compiler called a plain local is a
    verdict too.
    """
    sites = {}
    in_lambda = False
    for line in stderr.splitlines():
        context = CONTEXT.match(line)
        if context:
            in_lambda = "lambda function" in context.group("ctx")
            continue
        wording = None
        for word in LOCAL_SHADOW:
            if word in line:
                wording = word
                break
        if wording is None:
            continue
        diagnostic = DIAGNOSTIC.match(line)
        if not diagnostic:
            continue
        if _relative(diagnostic.group("file")) != unit:
            continue
        site = (diagnostic.group("line"), diagnostic.group("col"))
        candidate = in_lambda and wording == "shadows a previous local"
        sites[site] = sites.get(site, True) and candidate
    return sites


class MsvcWxDiagnosticsTest(unittest.TestCase):
    """C4456 and C4101, the two classes /WX turns into a build failure."""

    @classmethod
    def setUpClass(cls):
        cls.compilers = _compilers()
        if not cls.compilers:
            raise unittest.SkipTest("no host C++ compiler")

    def test_no_local_shadows_in_probed_units(self):
        """C4456: no declaration hides a local in the probed units.

        The compile probe is the red.  On the pre-fix source the
        kolibri1_forward.cpp unit reports
        "kolibri1_forward.cpp:326:17: warning: declaration of 'prof' shadows
        a previous local [-Wshadow]", which is the same file, line and
        column the MSVC job log names.
        """
        for unit in PROBED_UNITS:
            with self.subTest(unit=unit):
                reported = {}
                for compiler in self.compilers:
                    returncode, stderr = _probe(compiler, unit)
                    # A unit that does not build on this host would make the
                    # shadow check vacuous, so a failed compile fails the
                    # test rather than passing it.
                    self.assertNotIn(
                        "error:",
                        stderr,
                        f"{unit} did not compile on this host with "
                        f"{compiler[0]}, so the shadow check would be "
                        f"vacuous:\n{stderr}",
                    )
                    self.assertEqual(
                        0,
                        returncode,
                        f"{unit} did not compile on this host with "
                        f"{compiler[0]}:\n{stderr}",
                    )
                    for site, candidate in _local_shadows(stderr, unit).items():
                        # A site one compiler calls a lambda candidate and
                        # another calls a plain local is a plain local.
                        reported[site] = reported.get(site, True) and candidate
                self.assertEqual(
                    {},
                    {
                        f"{unit}:{line}:{col}": site
                        for (line, col), site in reported.items()
                        if not site
                    },
                    f"{unit} hides a local, which is MSVC C4456 and a build "
                    f"failure under the windows-msvc /WX lane",
                )

    def test_policy_guard_locals_are_marked_maybe_unused(self):
        """C4101: a PolicyGuard local off Linux is an empty class.

        GCC's -Wunused-variable does not fire for a class with a
        non-trivial constructor, and PolicyGuard has one on Linux, so no
        host compiler can report this class.  The attribute is the portable
        statement of intent, and MSVC's C4101 punishes its absence.
        """
        sites = (
            "src/vllm/model_executor/models/kolibri1_dequant_cache.h",
            "src/vllm/model_executor/models/kolibri1_weights.cpp",
        )
        found = 0
        for site in sites:
            text = (ROOT / site).read_text(encoding="utf-8")
            for match in GUARD_DECL.finditer(text):
                found += 1
                with self.subTest(site=site, name=match.group("name")):
                    self.assertIsNotNone(
                        match.group("attr"),
                        f"{site}: PolicyGuard local "
                        f"'{match.group('name')}' lacks [[maybe_unused]]. "
                        f"PolicyGuard is an empty class off Linux, so MSVC "
                        f"reports C4101 and /WX makes it a build failure.",
                    )
        self.assertGreater(
            found,
            0,
            "no PolicyGuard local was found; the guard sites moved and this "
            "assertion no longer covers them",
        )


if __name__ == "__main__":
    if not _compilers():
        # A gate that cannot run must not report success.
        sys.stderr.write("no host C++ compiler: cannot run the probe\n")
        sys.exit(77)
    unittest.main()
