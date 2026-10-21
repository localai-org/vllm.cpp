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
    ``kolibri1_numa::PolicyGuard`` is empty off Linux (``kolibri1_numa.h``),
    so the two guard locals are unreferenced there.  GCC's and Clang's
    ``-Wunused-variable`` report exactly that class: an unreferenced local
    of a class with a trivial default constructor.  The probe therefore
    compiles the production translation units in the shape the lane
    compiles, and it names the C4101 site at the same file, line and column
    that MSVC names.

TWO SHAPES
    The probe compiles every probed translation unit twice.

    ``host``
        The shape this host builds.  The shadow class lives here.

    ``non-linux``
        The shape the lane builds.  ``-U__linux__`` makes
        ``kolibri1_numa.h`` take the same empty-guard branch that ``_WIN32``
        takes there, so the guard is the empty class the Windows lane
        compiles, and an unreferenced local of it is the C4101 class.  The
        shadow class is checked in this shape too, because the lane
        compiles this shape and not the host shape.

    The ``-Wunused-variable`` check runs in the non-Linux shape ONLY.  The
    lane does not compile the Linux-only code, so an unused local in the
    host shape is not a lane red, and a global check would report defects
    that the lane never sees.  The check is therefore scoped to the shape
    the lane compiles, which is the precise form of the class it guards.

WHAT COUNTS AS A SITE
    A diagnostic is a defect of this tree when its file is inside the
    repository.  A diagnostic in a system header is not, and the probe
    ignores it.  A diagnostic in a repository HEADER is: the lane compiles
    those headers, and MSVC names the header's line and column.  The C4101
    site in ``kolibri1_dequant_cache.h`` is a header site, so the scope of
    both classes is the whole repository and not the translation unit's own
    file.

    The probe compiles the PRODUCTION translation units with the production
    include paths; it does not reconstruct the shape in a synthetic file.
    It also checks that the probed units REACH every production
    ``PolicyGuard`` local, so the C4101 class cannot go quiet because a
    guard moved into a file that no probed unit compiles.

The filter for the shadow class is exactly the C4456 class and nothing else.
MSVC splits the shadow family four ways -- C4456 hides a previous LOCAL,
C4457 hides a function PARAMETER, C4458 hides a class MEMBER, C4459 hides a
GLOBAL -- so the probe matches the local class and leaves the other three
alone.  The member class in particular is common in this tree's headers and
is not a C4456.

The two host compilers differ in one place, and the probe takes that into
account instead of guessing.  Clang's ``-Wshadow`` has MSVC's scope: it
reports a local hiding a local in the same function, and it does NOT report
a lambda body hiding a local of the ENCLOSING function, because the lambda
body is its own function.  GCC's ``-Wshadow`` reports that lambda case too,
so a GCC-only hit in a lambda is a candidate rather than a verdict.  The
measured case is ``src/vllm/multimodal/ltx2_video.cpp``: GCC reports three
such hits, Clang reports none, and the MSVC lane compiled that file clean
(job 118938856642, 2026-10-20T23:23:15Z).
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

# The include roots the probe compiles with, in the compiler's search order.
INCLUDE_DIRS = ("src", "include", "third_party")

# Production translation units the windows-msvc lane compiles.  A local
# shadow in any of them, or in a header they include, is a C4456 on Windows
# and a green build on Linux.
PROBED_UNITS = (
    "src/vllm/model_executor/models/kolibri1_forward.cpp",
    "src/vllm/model_executor/models/kolibri1_tt_forward.cpp",
    "src/vllm/model_executor/models/kolibri1_weights.cpp",
)

# The shapes the probe compiles, and the flags each one carries.
#
# ``host`` is the shape this host builds.  ``non-linux`` is the shape the
# lane builds: ``-U__linux__`` selects the same empty-guard branch of
# ``kolibri1_numa.h`` that ``_WIN32`` selects there, and the unreferenced
# local of an empty class only exists in that shape.
SHAPES = ("host", "non-linux")
SHAPE_FLAGS = {
    "host": ("-Wshadow",),
    "non-linux": ("-Wshadow", "-Wunused-variable", "-U__linux__"),
}

# The C4101 class: a local of kolibri1_numa::PolicyGuard, which is an empty
# class off Linux.  Every such local must carry the attribute.
GUARD_DECL = re.compile(
    r"(?P<attr>\[\[maybe_unused\]\]\s+)?"
    r"(?:const\s+)?(?:kolibri1_numa::)?PolicyGuard\s+(?P<name>\w+)\s*;"
)

# The production trees the guard scan covers.  A test file is not part of
# the lane's build, so a guard declared in one is not the lane's business.
PRODUCTION_TREES = ("src", "include")

# GCC (>= 15) words the local-shadow class "shadows a previous local";
# Clang words it "shadows a local variable".  Both are C4456.  The member,
# parameter and global classes are different diagnostics and are not
# matched.
LOCAL_SHADOW = ("shadows a previous local", "shadows a local variable")

# The unreferenced-local class.  Both compilers word it "unused variable",
# and both spell the option "-Wunused-variable", so the match is on the
# space-separated wording and never on the option tag.
UNUSED_VARIABLE = ("unused variable",)

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


def _include_flags():
    """The include roots, as the compiler's search order."""
    flags = []
    for directory in INCLUDE_DIRS:
        flags += ["-I", str(ROOT / directory)]
    return flags


def _probe(compiler, unit, shape):
    """Compile one production translation unit in one shape.

    Returns (returncode, stderr).  -fsyntax-only keeps the probe to the
    front end, which is where both diagnostic classes live, and keeps it
    fast enough for a unit test.
    """
    command = list(compiler) + ["-std=c++20", "-fsyntax-only"]
    command += list(SHAPE_FLAGS[shape])
    command += _include_flags()
    command.append(str(ROOT / unit))
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return done.returncode, done.stderr


def _dependencies(compiler, unit):
    """The repository files the compiler reads for one translation unit.

    The compiler's own dependency list, not a text scan of the includes, so
    the answer follows the preprocessor's decisions.  The list is a
    preprocessing pass, and it costs a fraction of a compile.
    """
    command = list(compiler) + ["-std=c++20", "-M"]
    command += _include_flags()
    command.append(str(ROOT / unit))
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return set()
    text = done.stdout.replace("\\\n", " ")
    _, _, dependencies = text.partition(":")
    files = set()
    for entry in dependencies.split():
        relative = _repo_relative(entry)
        if relative:
            files.add(relative)
    return files


def _repo_relative(path):
    """The path relative to the repository root, or None when it is outside.

    A diagnostic in a system header is not this tree's defect.  A
    diagnostic in a repository file is, whether the file is a translation
    unit or a header the lane compiles.
    """
    text = os.path.normpath(path)
    root = str(ROOT)
    if text == root:
        return ""
    if text.startswith(root + os.sep):
        return text[len(root) + 1 :].replace(os.sep, "/")
    return None


def _local_shadows(stderr):
    """The local-shadow diagnostics the compiler reported.

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
        relative = _repo_relative(diagnostic.group("file"))
        if relative is None:
            continue
        site = (relative, diagnostic.group("line"), diagnostic.group("col"))
        candidate = in_lambda and wording == "shadows a previous local"
        sites[site] = sites.get(site, True) and candidate
    return sites


def _unused_locals(stderr):
    """The unreferenced-local diagnostics the compiler reported.

    Returns a mapping from a (file, line, column) site to the diagnostic
    text.  This is the C4101 class: MSVC reports an unreferenced local of
    an empty class, and both host compilers report the same class for an
    unreferenced local of a class with a trivial default constructor.
    """
    sites = {}
    for line in stderr.splitlines():
        if not any(word in line for word in UNUSED_VARIABLE):
            continue
        diagnostic = DIAGNOSTIC.match(line)
        if not diagnostic:
            continue
        relative = _repo_relative(diagnostic.group("file"))
        if relative is None:
            continue
        site = (relative, diagnostic.group("line"), diagnostic.group("col"))
        sites.setdefault(site, line.strip())
    return sites


def _guard_declarations():
    """Every production PolicyGuard local, as a mapping from file to sites.

    The scan covers the production trees, because the lane compiles them
    and a test file is not part of the lane's build.
    """
    declarations = {}
    for tree in PRODUCTION_TREES:
        base = ROOT / tree
        for path in sorted(base.rglob("*")):
            if path.suffix not in (".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            sites = []
            for number, line in enumerate(text.splitlines(), start=1):
                if "PolicyGuard" not in line:
                    continue
                match = GUARD_DECL.search(line)
                if match:
                    sites.append((number, match.group("name")))
            if sites:
                relative = _repo_relative(str(path))
                declarations[relative] = sites
    return declarations


class MsvcWxDiagnosticsTest(unittest.TestCase):
    """C4456 and C4101, the two classes /WX turns into a build failure."""

    @classmethod
    def setUpClass(cls):
        cls.compilers = _compilers()
        if not cls.compilers:
            raise unittest.SkipTest("no host C++ compiler")
        # One compile per (translation unit, compiler, shape).  The probe
        # is a pre-commit gate, so the compiles are shared by the tests
        # instead of repeated by each of them.
        cls.compiles = {}

    def _compiled(self, unit, compiler, shape):
        """The compile of one unit in one shape, run once per session.

        A unit that does not compile on this host would make the check
        vacuous, so a failed compile fails the test instead of passing it.
        """
        key = (unit, compiler[0], shape)
        if key not in self.compiles:
            self.compiles[key] = _probe(compiler, unit, shape)
        returncode, stderr = self.compiles[key]
        self.assertNotIn(
            "error:",
            stderr,
            f"{unit} did not compile on this host with {compiler[0]} in the "
            f"{shape} shape, so the check would be vacuous:\n{stderr}",
        )
        self.assertEqual(
            0,
            returncode,
            f"{unit} did not compile on this host with {compiler[0]} in the "
            f"{shape} shape:\n{stderr}",
        )
        return stderr

    def test_no_local_shadows_in_probed_units(self):
        """C4456: no declaration hides a local in the probed units.

        The compile probe is the red.  On the pre-fix source the
        kolibri1_forward.cpp unit reports
        "kolibri1_forward.cpp:326:17: warning: declaration of 'prof' shadows
        a previous local [-Wshadow]", which is the same file, line and
        column the MSVC job log names.  A header site counts too, because
        the lane compiles the headers and MSVC names their line and column.
        """
        reported = {}
        for shape in SHAPES:
            for unit in PROBED_UNITS:
                for compiler in self.compilers:
                    stderr = self._compiled(unit, compiler, shape)
                    for site, candidate in _local_shadows(stderr).items():
                        # A site one compiler calls a lambda candidate and
                        # another calls a plain local is a plain local.
                        reported[site] = reported.get(site, True) and candidate
        self.assertEqual(
            {},
            {
                f"{file}:{line}:{col}": site
                for (file, line, col), site in reported.items()
                if not site
            },
            "a declaration hides a local, which is MSVC C4456 and a build "
            "failure under the windows-msvc /WX lane",
        )

    def test_no_unreferenced_empty_class_locals(self):
        """C4101: no unreferenced local in the shape the lane compiles.

        The guard class is empty off Linux, so an unreferenced local of it
        is MSVC C4101.  The probe compiles the production units in the
        non-Linux shape, where the guard is that empty class, and both host
        compilers report the class with -Wunused-variable at the same file,
        line and column.

        The check is scoped to that shape on purpose.  The lane does not
        compile the Linux-only code, so an unused local in the host shape
        is not a lane red, and a global check would report defects that
        the lane never sees.
        """
        reported = {}
        for unit in PROBED_UNITS:
            for compiler in self.compilers:
                stderr = self._compiled(unit, compiler, "non-linux")
                for site, text in _unused_locals(stderr).items():
                    reported[site] = text
        self.assertEqual(
            {},
            {
                f"{file}:{line}:{col}": text
                for (file, line, col), text in reported.items()
            },
            "a local is unreferenced, which is MSVC C4101 for a local of an "
            "empty class and a build failure under the windows-msvc /WX lane",
        )

    def test_probed_units_reach_every_guard_declaration(self):
        """Every production PolicyGuard local is compiled by a probed unit.

        The C4101 check is only a check when the probed units reach the
        guard locals.  A guard that moved into a file that no probed unit
        includes would leave the class unobserved, so this test fails on
        that move and names the file to add to PROBED_UNITS.
        """
        declarations = _guard_declarations()
        self.assertGreater(
            len(declarations),
            0,
            "no production PolicyGuard local was found; the guard sites "
            "moved and the probe no longer covers them",
        )
        for compiler in self.compilers:
            reached = set()
            for unit in PROBED_UNITS:
                reached |= _dependencies(compiler, unit)
            for file, sites in declarations.items():
                with self.subTest(compiler=compiler[0], file=file):
                    self.assertIn(
                        file,
                        reached,
                        f"{file} declares a PolicyGuard local at "
                        f"{', '.join(str(line) for line, _ in sites)}, and no "
                        f"probed translation unit includes it, so the C4101 "
                        f"check cannot observe it.  Add a translation unit "
                        f"that includes it to PROBED_UNITS.",
                    )

    def test_policy_guard_locals_are_marked_maybe_unused(self):
        """C4101: a PolicyGuard local off Linux is an empty class.

        The compile probe above observes the class.  This assertion holds
        the attribute itself, which the compile probe cannot separate from
        a local that the code references: an unreferenced local of an empty
        class is the C4101 class, and the attribute is the portable
        statement that the local is unreferenced on purpose.
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
