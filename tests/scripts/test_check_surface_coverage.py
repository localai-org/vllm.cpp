#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-surface-coverage.py.

Two axes: (1) the example include-boundary lint (examples are thin clients of the public
C ABI, not internal-header reachers), and (2) the FEATURES C-ABI capability table bound to
include/vllm.h. Mirrors tests/scripts/test_check_runner_routing_consistency.py.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-surface-coverage.py"
SHARED_TEXT = ROOT / "scripts/checker_text.py"
SPEC = importlib.util.spec_from_file_location("check_surface_coverage", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

AllowEntry = mod.AllowEntry
CapRow = mod.CapRow


def entry(unit: str, fold: str = "ARCH-ONE-SURFACE", reason: str = "r") -> AllowEntry:
    return AllowEntry(unit=unit, fold=fold, reason=reason)


# ── Axis 1: example include boundary ──────────────────────────────────────────────────

class InternalIncludeTests(unittest.TestCase):
    def test_detects_internal_trees(self) -> None:
        text = (
            '#include "vllm.h"\n'
            '#include <string>\n'
            '#include "vllm/model_executor/models/laguna.h"\n'
            '#include "vt/backend.h"\n'
            '#include "src/foo.h"\n'
            '#include "bench_core.h"\n'
        )
        found = mod.internal_includes(text)
        self.assertIn("vllm/model_executor/models/laguna.h", found)
        self.assertIn("vt/backend.h", found)
        self.assertIn("src/foo.h", found)
        # The public flat header and a local sibling header are NOT internal.
        self.assertNotIn("vllm.h", found)
        self.assertFalse(any("bench_core.h" in h for h in found))

    def test_public_header_no_slash_never_matches(self) -> None:
        # "vllm.h" is public; only "vllm/..." (with slash) is internal.
        self.assertEqual(mod.internal_includes('#include "vllm.h"'), [])

    def test_commented_include_is_ignored(self) -> None:
        text = '// #include "vllm/internal.h"\n/* #include "vt/x.h" */\n'
        self.assertEqual(mod.internal_includes(text), [])

    def test_unit_of_top_level_subdir(self) -> None:
        p = ROOT / "examples/laguna_gen/main.cpp"
        self.assertEqual(mod.unit_of(p, ROOT / "examples"), "examples/laguna_gen")

    def test_unit_of_direct_file(self) -> None:
        p = ROOT / "examples/loose.cpp"
        self.assertEqual(mod.unit_of(p, ROOT / "examples"), "examples/loose")


class IncludeDirGrantTests(unittest.TestCase):
    def test_flags_internal_src_grant(self) -> None:
        cmake = (
            "add_executable(quant-gemm-bench quant_gemm_bench/main.cpp)\n"
            "target_include_directories(quant-gemm-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)\n"
        )
        self.assertIn("examples/quant_gemm_bench", mod.internal_include_dir_grant_units(cmake))

    def test_own_dir_grant_not_flagged(self) -> None:
        cmake = (
            "add_executable(vllm-bench bench/main.cpp)\n"
            "target_include_directories(vllm-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)\n"
        )
        self.assertEqual(mod.internal_include_dir_grant_units(cmake), {})

    def test_benchmarks_source_out_of_scope(self) -> None:
        # A grant to a target whose source is outside examples/ (absolute path) is skipped.
        cmake = (
            "add_executable(vulkan-gemm-ab ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_gemm_ab.cpp)\n"
            "target_include_directories(vulkan-gemm-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)\n"
        )
        self.assertEqual(mod.internal_include_dir_grant_units(cmake), {})


class ParseAllowlistTests(unittest.TestCase):
    def test_valid_entry(self) -> None:
        entries, errors = mod.parse_allowlist(
            "examples/laguna_gen | fold=ARCH-ONE-SURFACE | keep-quant decode CLI-only"
        )
        self.assertEqual(errors, [])
        self.assertIn("examples/laguna_gen", entries)
        self.assertEqual(entries["examples/laguna_gen"].fold, "ARCH-ONE-SURFACE")

    def test_comments_and_blanks_ignored(self) -> None:
        entries, errors = mod.parse_allowlist("# a comment\n\n")
        self.assertEqual((entries, errors), ({}, []))

    def test_missing_fold_field_errors(self) -> None:
        _, errors = mod.parse_allowlist("examples/x | ARCH-ONE-SURFACE | why")
        self.assertTrue(errors)

    def test_empty_fold_errors(self) -> None:
        _, errors = mod.parse_allowlist("examples/x | fold= | why")
        self.assertTrue(errors)

    def test_missing_reason_errors(self) -> None:
        _, errors = mod.parse_allowlist("examples/x | fold=ROW")
        self.assertTrue(errors)

    def test_duplicate_errors(self) -> None:
        _, errors = mod.parse_allowlist(
            "examples/x | fold=ROW | a\nexamples/x | fold=ROW | b"
        )
        self.assertTrue(errors)


class BoundaryErrorTests(unittest.TestCase):
    def test_reaching_and_allowlisted_is_clean(self) -> None:
        self.assertEqual(
            mod.boundary_errors({"examples/laguna_gen"},
                                {"examples/laguna_gen": entry("examples/laguna_gen")}),
            [],
        )

    def test_reaching_not_allowlisted_is_uncovered(self) -> None:
        # A new internal-reaching example with no entry => RED.
        self.assertEqual(
            mod.boundary_errors({"examples/new_gen"}, {}),
            ["uncovered:examples/new_gen"],
        )

    def test_allowlisted_not_reaching_is_stale(self) -> None:
        # A folded example that is now a clean client => the entry is stale => RED.
        self.assertEqual(
            mod.boundary_errors(set(), {"examples/done": entry("examples/done")}),
            ["stale:examples/done"],
        )

    def test_clean_baseline_never_allowlisted(self) -> None:
        # examples/cli reaches nothing, so it is neither uncovered nor stale.
        self.assertEqual(mod.boundary_errors(set(), {}), [])


class PublicSurfacePinnedTests(unittest.TestCase):
    def test_derives_from_install_rule(self) -> None:
        self.assertEqual(
            mod.public_headers_from_cmake("install(FILES include/vllm.h DESTINATION x)"),
            {"vllm.h"},
        )

    def test_derive_ignores_non_headers(self) -> None:
        self.assertEqual(
            mod.public_headers_from_cmake("install(FILES a.h b.txt c.hpp DESTINATION x)"),
            {"a.h", "c.hpp"},
        )

    def test_true_on_install_rule(self) -> None:
        self.assertTrue(
            mod.public_surface_pinned("install(FILES include/vllm.h include/lumenengine.h DESTINATION x)")
        )

    def test_false_when_either_public_header_is_missing(self) -> None:
        for header in ("vllm.h", "lumenengine.h"):
            self.assertFalse(mod.public_surface_pinned(
                f"install(FILES include/{header} DESTINATION x)"))

    def test_false_without_install_rule(self) -> None:
        self.assertFalse(mod.public_surface_pinned("add_library(vllm ...)"))

    def test_false_when_surface_drifts(self) -> None:
        # Installing an EXTRA header must red the pin (the boundary changed).
        self.assertFalse(
            mod.public_surface_pinned(
                "install(FILES include/vllm.h include/lumenengine.h include/vllm/extra.h DESTINATION x)"
            )
        )


# ── Axis 2: C-ABI capability reachability ─────────────────────────────────────────────

def _table(*rows: str) -> str:
    body = "\n".join(rows)
    return (
        f"{mod.CAP_BEGIN}\n"
        "| Capability | C-ABI surface | Embedder-reachable |\n"
        "|---|---|---|\n"
        f"{body}\n"
        f"{mod.CAP_END}\n"
    )


VLLM_H_STUB = "vllm_status vllm_complete(void); struct { int structured_json; };"


class CapabilityTableTests(unittest.TestCase):
    def test_parses_reachable_and_unreachable(self) -> None:
        rows = mod.capability_table(
            _table(
                "| Text completion | `vllm_complete` | reachable |",
                "| Embeddings / pooling | — | embedder-unreachable |",
            )
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].status, "reachable")
        self.assertEqual(rows[0].symbols, ("vllm_complete",))
        self.assertEqual(rows[1].status, "embedder-unreachable")
        self.assertEqual(rows[1].symbols, ())

    def test_missing_markers_returns_none(self) -> None:
        self.assertIsNone(mod.capability_table("no table here"))

    def test_header_and_separator_skipped(self) -> None:
        rows = mod.capability_table(_table("| Text | `vllm_complete` | reachable |"))
        self.assertEqual(len(rows), 1)


class DeclaredAbiSymbolsTests(unittest.TestCase):
    def test_finds_symbols(self) -> None:
        syms = mod.declared_abi_symbols(VLLM_H_STUB)
        self.assertIn("vllm_complete", syms)
        self.assertIn("structured_json", syms)
        self.assertNotIn("vllm_embed", syms)

    def test_comment_only_symbol_is_not_declared(self) -> None:
        # A token that survives ONLY in a doc comment must NOT read as declared — else
        # deleting the real declaration stays green because the prose still names it.
        header = "// vllm_complete streams tokens.\n/* see vllm_complete */\n"
        self.assertNotIn("vllm_complete", mod.declared_abi_symbols(header))
        errs = mod.capability_errors(
            _table("| Text | `vllm_complete` | reachable |"), header, {}
        )
        self.assertTrue(any("not declared" in e for e in errs))


class CapabilityErrorTests(unittest.TestCase):
    def test_reachable_existing_symbol_clean(self) -> None:
        errs = mod.capability_errors(
            _table("| Text | `vllm_complete` | reachable |"), VLLM_H_STUB, {}
        )
        self.assertEqual(errs, [])

    def test_reachable_missing_symbol_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Embeds | `vllm_embed` | reachable |"), VLLM_H_STUB, {}
        )
        self.assertTrue(any("not declared" in e for e in errs))

    def test_reachable_without_symbol_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Text | — | reachable |"), VLLM_H_STUB, {}
        )
        self.assertTrue(any("names no C-ABI symbol" in e for e in errs))

    def test_unreachable_allowlisted_clean(self) -> None:
        errs = mod.capability_errors(
            _table("| Embeddings / pooling | — | embedder-unreachable |"),
            VLLM_H_STUB,
            {"embeddings / pooling": entry("embeddings / pooling")},
        )
        self.assertEqual(errs, [])

    def test_unreachable_not_allowlisted_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Embeddings / pooling | — | embedder-unreachable |"),
            VLLM_H_STUB,
            {},
        )
        self.assertTrue(any("no entry in" in e for e in errs))

    def test_unreachable_with_symbol_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Embeds | `vllm_embed` | embedder-unreachable |"),
            VLLM_H_STUB,
            {"embeds": entry("embeds")},
        )
        self.assertTrue(any("names a C-ABI symbol" in e for e in errs))

    def test_stale_capability_allowlist_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Text | `vllm_complete` | reachable |"),
            VLLM_H_STUB,
            {"embeddings / pooling": entry("embeddings / pooling")},
        )
        self.assertTrue(any("matches no" in e for e in errs))

    def test_unknown_status_errors(self) -> None:
        errs = mod.capability_errors(
            _table("| Text | `vllm_complete` | maybe |"), VLLM_H_STUB, {}
        )
        self.assertTrue(any("expected" in e for e in errs))

    def test_missing_table_errors(self) -> None:
        errs = mod.capability_errors("no markers", VLLM_H_STUB, {})
        self.assertTrue(any("missing the abi-capability-table" in e for e in errs))


# ── The shipped tree stays green ──────────────────────────────────────────────────────

class ShippedTreeTests(unittest.TestCase):
    def test_boundary_green(self) -> None:
        reaching = set(mod.internal_includes_by_unit(mod.EXAMPLES_DIR))
        allow, errors = mod.parse_allowlist(mod.read(mod.ALLOWLIST))
        self.assertEqual(errors, [])
        self.assertEqual(mod.boundary_errors(reaching, allow), [])
        # examples/cli is the clean baseline and must never be allowlisted.
        self.assertNotIn("examples/cli", reaching)
        self.assertNotIn("examples/cli", allow)
        # The known CLI-only capability drivers ARE internal-reachers.
        # (parakeet_transcribe left this list when the ROW 1 fold made it a
        # clean vllm.h client; minimax_h3_gen + minimax_h3_mux left it when the
        # ROW 2 video fold did the same — asserted below instead.)
        for unit in ("examples/laguna_gen", "examples/deepseek_v4_gen"):
            self.assertIn(unit, reaching)
        for unit in ("examples/parakeet_transcribe", "examples/minimax_h3_gen",
                     "examples/minimax_h3_mux"):
            self.assertNotIn(unit, reaching)
            self.assertNotIn(unit, allow)

    def test_public_surface_pinned_green(self) -> None:
        self.assertTrue(mod.public_surface_pinned(mod.read(mod.CMAKELISTS)))

    def test_reaching_count_within_ratchet(self) -> None:
        reaching = set(mod.internal_includes_by_unit(mod.EXAMPLES_DIR))
        # include the CMake -I grants, exactly as main() does, so this tracks the ceiling.
        reaching |= set(mod.internal_include_dir_grant_units(mod.read(mod.EXAMPLES_CMAKE)))
        self.assertLessEqual(len(reaching), mod.MAX_INTERNAL_REACHING)

    def test_ratchet_ceiling_pinned_at_9(self) -> None:
        # EQUALITY pin: a ceiling bump (up OR down) must move this line + the ratchet claims in
        # the spec/state, so the change is test-visible and reviewed, never silent.
        # 9 = conscious operator exception 2026-08-08 (external contribution #65,
        # examples/cpu_kernel_bench dev-tool; re-shrinks when the bench folds).
        self.assertEqual(mod.MAX_INTERNAL_REACHING, 9)

    def test_capability_green(self) -> None:
        cap_allow, allow_errors = mod.parse_allowlist(mod.read(mod.CAP_ALLOWLIST))
        self.assertEqual(allow_errors, [])
        errs = mod.capability_errors(
            mod.read(mod.FEATURES), mod.read(mod.VLLM_H), cap_allow
        )
        self.assertEqual(errs, [])


# ── Subprocess enforcement seams: main()'s WIRING must be load-bearing ────────────────
# The pure-helper tests above pass even if main() forgets to CALL a helper. These run the
# real checker BINARY against a minimal fixture tree so a deletion of an enforcement seam
# in main() (the grant merge, the ratchet, the capability_errors call) turns a fixture RED
# to GREEN and fails here. Mirrors the reviewer's requirement that every seam be exercised
# end to end.

def _base_files(sym: str = "vllm_complete") -> dict[str, str]:
    """A minimal valid fixture tree the shipped checker passes (rc 0)."""
    return {
        "scripts/check-surface-coverage.py": CHECKER.read_text(encoding="utf-8"),
        # The checker imports its comment stripper from the shared helper beside it
        # (scripts/checker_text.py), so the fixture tree needs that file too — a
        # subprocess run of a scripts/ dir that lacks it dies on the import.
        "scripts/checker_text.py": SHARED_TEXT.read_text(encoding="utf-8"),
        "CMakeLists.txt": "install(FILES include/vllm.h include/lumenengine.h DESTINATION include)\n",
        "include/vllm.h": "VLLM_API vllm_status vllm_complete(void);\n",
        "docs/FEATURES.md": (
            "#### caps " + mod.CAP_BEGIN + "\n"
            "| Capability | C-ABI surface | Embedder-reachable |\n"
            "|---|---|---|\n"
            f"| Text | `{sym}` | reachable |\n"
            "| Embeddings | none | embedder-unreachable | " + mod.CAP_END + "\n"
        ),
        "scripts/abi-capability-allowlist.txt": "embeddings | fold=ROW | reason\n",
        "examples/cli/main.cpp": '#include "vllm.h"\n',
        "examples/foo_gen/main.cpp": '#include "vllm/x.h"\n',
        "examples/CMakeLists.txt": (
            "add_executable(cli cli/main.cpp)\n"
            "add_executable(foo-gen foo_gen/main.cpp)\n"
        ),
        "scripts/example-abi-allowlist.txt": "examples/foo_gen | fold=ROW | reason\n",
    }


def _run_fixture(files: dict[str, str]) -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(root / "scripts/check-surface-coverage.py")],
            capture_output=True, text=True,
        ).returncode


class SubprocessEnforcementTests(unittest.TestCase):
    def test_base_fixture_is_green(self) -> None:
        self.assertEqual(_run_fixture(_base_files()), 0)

    def test_grant_merge_seam_reds(self) -> None:
        # A CLEAN example (no internal #include) granted -I into src/ must red — this only
        # fires if main() merges the CMake-grant units into `reaching`.
        files = _base_files()
        files["examples/bar/main.cpp"] = '#include "vllm.h"\n'
        files["examples/CMakeLists.txt"] += (
            "add_executable(bar-gen bar/main.cpp)\n"
            "target_include_directories(bar-gen PRIVATE ${CMAKE_SOURCE_DIR}/src)\n"
        )
        self.assertNotEqual(_run_fixture(files), 0)

    def test_ratchet_seam_reds(self) -> None:
        # 13 internal-reachers, ALL allowlisted (so boundary_errors is empty) — only the
        # ratchet at main() can red this.
        files = _base_files()
        del files["examples/foo_gen/main.cpp"]
        cmake = "add_executable(cli cli/main.cpp)\n"
        allow = ""
        for i in range(mod.MAX_INTERNAL_REACHING + 1):
            files[f"examples/ex{i:02d}/main.cpp"] = '#include "vllm/x.h"\n'
            cmake += f"add_executable(ex{i:02d} ex{i:02d}/main.cpp)\n"
            allow += f"examples/ex{i:02d} | fold=ROW | r\n"
        files["examples/CMakeLists.txt"] = cmake
        files["scripts/example-abi-allowlist.txt"] = allow
        self.assertNotEqual(_run_fixture(files), 0)

    def test_capability_errors_seam_reds(self) -> None:
        # A reachable row naming a symbol absent from vllm.h must red — only fires if main()
        # calls capability_errors.
        self.assertNotEqual(_run_fixture(_base_files(sym="vllm_absent")), 0)


if __name__ == "__main__":
    unittest.main()
