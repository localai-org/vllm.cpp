#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-tt-public-surface.py.

ISSUE-LOCAL-01M7W2ZKR8HRBYJF6HCPP46AGH. The bug class: an always-compiled TU
(every source outside the if(VLLM_CPP_TENSTORRENT) blocks, plus every test
registered outside them) references a vt::tenstorrent symbol that has no
non-TT stub in the public header — the CPU-only link then fails with an
undefined reference, which is how the drafter test broke build-newest-gcc
with EnsureHost. TWO DIRECTIONS IN EVERY CASE: each red fixture (unguarded
reference without a stub, internal-header include, namespace re-opening,
unconditional header dangler) must exit 1 and name the rule, and the clean
fixture must exit 0 — a gate that fires on ordinary stubbed usage is the
defect, not the discipline, so the green tree carries unguarded references
to stubbed symbols on purpose.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-tt-public-surface.py"

MAIN_CMAKE = """\
project(fixture CXX)
add_library(vllm STATIC src/vllm/good.cpp)
if(VLLM_CPP_TENSTORRENT)
  target_sources(vllm PRIVATE
    src/vllm/platforms/tenstorrent.cpp
    src/vt/tenstorrent/x.cpp)
endif()
"""

TESTS_CMAKE = """\
vllm_cpp_add_test(test_always vllm/models/test_always.cpp)
if(VLLM_CPP_TENSTORRENT)
  vllm_cpp_add_test(test_tt_only vt/test_tt_only.cpp)
endif()
"""

PUBLIC_HEADER_GOOD = """\
#pragma once
namespace vt { struct Tensor; }
namespace vt::tenstorrent {
bool DeviceAvailable();
#ifdef VLLM_CPP_TENSTORRENT
void WarmThing(void* p);
#else
inline void WarmThing(void*) {}
inline bool DeviceAvailable() { return false; }
#endif
}
"""

PUBLIC_HEADER_DANGLER = PUBLIC_HEADER_GOOD.replace(
    "bool DeviceAvailable();\n", ""
).replace(
    "#ifdef VLLM_CPP_TENSTORRENT\nvoid WarmThing(void* p);\n#else\ninline void WarmThing(void*) {}\ninline bool DeviceAvailable() { return false; }\n#endif",
    "void WarmThing(void* p);\nbool DeviceAvailable();",
)

GOOD_CPP = """\
#include "vt/tenstorrent/tenstorrent_device.h"
void f() { vt::tenstorrent::WarmThing(nullptr); }
"""

GUARDED_CPP = """\
#include "vt/tenstorrent/tenstorrent_device.h"
void f() {
#ifdef VLLM_CPP_TENSTORRENT
  vt::tenstorrent::Missing();
#endif
}
"""

BAD_REF_CPP = """\
#include "vt/tenstorrent/tenstorrent_device.h"
void f() { vt::tenstorrent::Missing(); }
"""

BAD_INC_CPP = """\
#include "vt/tenstorrent/tenstorrent_internal.h"
void f() {}
"""

BAD_NS_CPP = """\
namespace vt::tenstorrent {
void EnsureHost(Tensor& t);
}
void f(vt::Tensor& t) { vt::tenstorrent::EnsureHost(t); }
"""

TT_ONLY_CPP = """\
// Registered inside if(VLLM_CPP_TENSTORRENT): never compiled by the default
// configure, so internal symbols are fine here.
#include "../../src/vt/tenstorrent/tenstorrent_internal.h"
void g() { vt::tenstorrent::Missing(); }
"""

TEST_ALWAYS_CPP = """\
#include "vt/tenstorrent/tenstorrent_device.h"
void h() { vt::tenstorrent::WarmThing(nullptr); }
"""

INTERNAL_HEADER = "#pragma once\nnamespace vt::tenstorrent { void Missing(); }\n"
PLATFORM_CPP = "// TT-only source\n"
X_CPP = "// TT-only source\n"


class CheckerCase(unittest.TestCase):
    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def fixture(self, files: dict[str, str], header: str = PUBLIC_HEADER_GOOD) -> Path:
        root = Path(tempfile.mkdtemp(prefix="vllm-tt-surface-"))
        base = {
            "CMakeLists.txt": MAIN_CMAKE,
            "tests/CMakeLists.txt": TESTS_CMAKE,
            "src/vt/tenstorrent/tenstorrent_device.h": header,
            "src/vt/tenstorrent/tenstorrent_internal.h": INTERNAL_HEADER,
            "src/vt/tenstorrent/x.cpp": X_CPP,
            "src/vllm/platforms/tenstorrent.cpp": PLATFORM_CPP,
            "src/vllm/good.cpp": GOOD_CPP,
            "tests/vllm/models/test_always.cpp": TEST_ALWAYS_CPP,
            "tests/vt/test_tt_only.cpp": TT_ONLY_CPP,
        }
        base.update(files)
        for name, content in base.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return root


class GreenTests(CheckerCase):
    def test_clean_tree_exits_zero(self):
        proc = self.run_checker(self.fixture({}))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("0 violation(s)", proc.stdout)

    def test_guarded_reference_is_legal(self):
        proc = self.run_checker(self.fixture({"src/vllm/guarded.cpp": GUARDED_CPP}))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_tt_only_tu_is_not_scanned(self):
        # test_tt_only.cpp (registered inside the TT block) references
        # vt::tenstorrent::Missing and includes the internal header; the
        # always-compiled set must not include it.
        proc = self.run_checker(self.fixture({}))
        self.assertNotIn("test_tt_only", proc.stdout)
        self.assertNotIn("tenstorrent.cpp", proc.stdout)
        self.assertNotIn("x.cpp", proc.stdout)


class RedTests(CheckerCase):
    def test_unguarded_reference_without_stub_is_r3(self):
        proc = self.run_checker(self.fixture({"src/vllm/bad_ref.cpp": BAD_REF_CPP}))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("bad_ref.cpp", proc.stdout)
        self.assertIn("[R3] vt::tenstorrent::Missing", proc.stdout)

    def test_reference_in_always_registered_test_is_r3(self):
        bad_test = TEST_ALWAYS_CPP.replace("WarmThing", "Missing")
        proc = self.run_checker(
            self.fixture({"tests/vllm/models/test_always.cpp": bad_test})
        )
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("test_always.cpp", proc.stdout)
        self.assertIn("[R3]", proc.stdout)

    def test_internal_header_include_is_r1(self):
        proc = self.run_checker(self.fixture({"src/vllm/bad_inc.cpp": BAD_INC_CPP}))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("[R1]", proc.stdout)
        self.assertIn("tenstorrent_internal.h", proc.stdout)

    def test_namespace_reopen_is_r2(self):
        proc = self.run_checker(self.fixture({"src/vllm/bad_ns.cpp": BAD_NS_CPP}))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("[R2]", proc.stdout)

    def test_unconditional_header_dangler_is_r4(self):
        proc = self.run_checker(self.fixture({}, header=PUBLIC_HEADER_DANGLER))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("[R4] WarmThing", proc.stdout)
        self.assertIn("[R4] DeviceAvailable", proc.stdout)

    def test_missing_header_is_usage_error(self):
        root = self.fixture({})
        (root / "src/vt/tenstorrent/tenstorrent_device.h").unlink()
        proc = self.run_checker(root)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
