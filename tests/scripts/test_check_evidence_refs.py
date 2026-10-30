#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-evidence-refs.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-evidence-refs.py"
SPEC = importlib.util.spec_from_file_location("check_evidence_refs", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_evidence_refs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_evidence_refs
SPEC.loader.exec_module(check_evidence_refs)

scan_report = check_evidence_refs.scan_report
basename_index = check_evidence_refs.basename_index
FILENAME_RE = check_evidence_refs.FILENAME_RE


def _fixture(tmp: str, report_text: str, tracked: list[str], disk: list[str]) -> list[tuple[str, int, str]]:
    base = Path(tmp)
    ev = base / "docs/bench-evidence/fixture"
    ev.mkdir(parents=True)
    (ev / "README.md").write_text(report_text, encoding="utf-8")
    for name in disk:
        target = ev / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok\n", encoding="utf-8")
    return scan_report(ev / "README.md", basename_index(tracked), base)


class FilenameTokenTests(unittest.TestCase):
    def test_matches_a_plain_log_name(self) -> None:
        self.assertEqual(FILENAME_RE.findall("see run.log for detail"), ["run.log"])

    def test_matches_the_log_txt_double_suffix(self) -> None:
        self.assertEqual(FILENAME_RE.findall("see run.log.txt"), ["run.log.txt"])

    def test_the_glob_mention_is_not_a_reference(self) -> None:
        # The loss note explains the `*.log` rule; that is not a filename.
        self.assertEqual(FILENAME_RE.findall("the global `*.log` rule"), [])

    def test_an_absolute_path_is_path_scoped_not_a_tree_reference(self) -> None:
        self.assertEqual(FILENAME_RE.findall("captured at /tmp/scratch/run.log"), ["/tmp/scratch/run.log"])


class ReferencedPresentTests(unittest.TestCase):
    def test_a_reported_log_that_is_tracked_and_on_disk_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(
                tmp,
                "raw logs: present.log\n",
                tracked=["docs/bench-evidence/fixture/present.log"],
                disk=["present.log"],
            )
        self.assertEqual(failures, [])


class ReferencedMissingTests(unittest.TestCase):
    def test_a_reported_log_absent_from_the_tree_fails(self) -> None:
        # The exact shape of the 2026-10-29 incident: the report asserts
        # captured logs and none are in the tree.
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(
                tmp,
                "raw logs: missing.log\n",
                tracked=[],
                disk=[],
            )
        self.assertEqual([(name := f[2], f[1]) for f in failures], [("missing.log", 1)])


class ReferencedUntrackedTests(unittest.TestCase):
    def test_a_reported_log_on_disk_but_untracked_fails(self) -> None:
        # On disk is not enough: the original incident lost files precisely
        # because they were never tracked.
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(
                tmp,
                "raw logs: on-disk-untracked.log\n",
                tracked=["docs/bench-evidence/fixture/other.log"],
                disk=["on-disk-untracked.log"],
            )
        self.assertEqual([f[2] for f in failures], ["on-disk-untracked.log"])


class UnreferencedExtraTests(unittest.TestCase):
    def test_an_extra_log_the_report_never_names_passes(self) -> None:
        # The floor polices the reference direction only.
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(
                tmp,
                "raw logs: present.log\n",
                tracked=["docs/bench-evidence/fixture/present.log"],
                disk=["present.log", "extra-unreferenced.log"],
            )
        self.assertEqual(failures, [])


class RecordedLossTests(unittest.TestCase):
    def test_an_evidence_lost_note_exempts_its_filenames(self) -> None:
        text = (
            "raw logs: recorded-loss.log\n"
            "\n"
            "> EVIDENCE-LOST (2026-10-30): `recorded-loss.log`\n"
            "> dropped by the old global ignore rule.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(tmp, text, tracked=[], disk=[])
        self.assertEqual(failures, [])

    def test_an_evidence_lost_note_exempts_filenames_on_wrapped_continuation_lines(self) -> None:
        # The 2026-10-30 loss notes wrap long filename lists across lines.
        text = (
            "> EVIDENCE-LOST (2026-10-30): `a.log`,\n"
            "> `b.log` and more\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(tmp, text, tracked=[], disk=[])
        self.assertEqual(failures, [])

    def test_the_exemption_ends_at_the_next_blank_line(self) -> None:
        text = (
            "> EVIDENCE-LOST (2026-10-30): `a.log`\n"
            "> reason text\n"
            "\n"
            "later prose claims b.log was captured\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(tmp, text, tracked=[], disk=[])
        self.assertEqual([f[2] for f in failures], ["b.log"])


class ShippedTreeTests(unittest.TestCase):
    def test_the_shipped_evidence_dirs_are_clean(self) -> None:
        # The green-after on the row's own change: with the whitelist and the
        # dated loss notes, every reference in docs/bench-evidence resolves.
        self.assertEqual(check_evidence_refs.run(), [])


class FencedOutputTests(unittest.TestCase):
    def test_a_filename_inside_a_fenced_excerpt_is_not_a_live_claim(self) -> None:
        # The row's own evidence README quotes the red-before output; those
        # names are recorded losses, not references.
        text = (
            "excerpt:\n"
            "```\n"
            "  README.md:3: quoted-missing.log\n"
            "```\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            failures = _fixture(tmp, text, tracked=[], disk=[])
        self.assertEqual(failures, [])



if __name__ == "__main__":
    unittest.main()
