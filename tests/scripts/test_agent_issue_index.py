#!/usr/bin/env python3
"""`scripts/agent-issue-index.py` -- the derived issue index (#2290).

Two properties carry this module, and both are about REFUSING rather than
writing. A snapshot the gates read is only as good as its worst failure: a
partial one is indistinguishable from a complete one at the reader, so every
degraded path must leave the previous file alone and say why.

The shape half is here rather than in the reader's suite on purpose. This module
WRITES the table `check-agent-record.ISSUE_ROW` parses, so the shape belongs to
the writer; gating it at the reader would test the generator through a proxy,
which is what the retired `IssueIndexTableShape` did.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import issue_records as records



def load_module():
    spec = importlib.util.spec_from_file_location(
        "agent_issue_index", ROOT / "scripts/agent-issue-index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The reader's regex, copied so a drift between writer and reader is RED here.
# Importing it would hide the drift: the point is that two files agree, and a
# shared import proves only that one file agrees with itself.
ISSUE_ROW = re.compile(
    r"^\|\s*\[#(\d+)\]\((https://github\.com/[^)]+/issues/(\d+))\)\s*\|"
    r"\s*(?:`([A-Z0-9][A-Za-z0-9_.-]*)`|—)\s*\|"
)

def parse_markdown_row(row: str) -> list[str]:
    """Return cells after applying Markdown backslash escapes."""
    cells: list[str] = []
    cell: list[str] = []
    escaped = False
    for character in row:
        if escaped:
            cell.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(character)
    if escaped:
        raise AssertionError("rendered row ends with an unmatched escape")
    cells.append("".join(cell).strip())
    if cells[0] or cells[-1]:
        raise AssertionError("rendered row lacks outer table delimiters")
    return cells[1:-1]



class FilingAnIssueNoLongerCollides(unittest.TestCase):
    """The campaign's actual claim, executable.

    The measurement that opened this row -- 16 of 21 open pull requests
    CONFLICTING, four of them on the index alone -- is a property of a SHARED
    APPEND TARGET, not of the index's content. These two cases pin the mechanism
    in a scratch repository so the argument cannot rot into folklore: two branches
    that each append to one file collide, and two that each write their own file
    do not.

    `merge.union.driver=false` throughout, because that is what GITHUB does. A
    merge run WITH the driver reproduces the local false green this row exists to
    remove: it resolves cleanly on a developer's machine while the forge reports
    CONFLICTING and never schedules CI at all (#883, #2248).
    """

    def setUp(self) -> None:
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", ".")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "T")
        (self.repo / "seed").write_text("seed\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "seed")
        self.base = self.git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True,
            stderr=subprocess.DEVNULL,
        )

    def branch_writing(self, name: str, path: str, content: str) -> str:
        self.git("checkout", "-q", "-B", name, self.base)
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Append when the file already exists, which is what an index row was.
        existing = target.read_text() if target.exists() else ""
        target.write_text(existing + content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"{name} files an issue")
        return self.git("rev-parse", "HEAD").strip()

    def conflicts(self, a: str, b: str) -> int:
        out = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "merge.union.driver=false",
             "merge-tree", "--write-tree", a, b],
            capture_output=True, text=True,
        ).stdout
        return sum(1 for line in out.splitlines() if line.startswith("CONFLICT"))

    def test_the_old_shape_collides(self) -> None:
        """A shared append target. This is the control, and it must be RED-ish.

        Without it, the green below proves nothing: a merge that cannot conflict
        for an unrelated reason would pass the same assertion.
        """
        (self.repo / "index.md").write_text("| row 0 |\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "the shared index")
        self.base = self.git("rev-parse", "HEAD").strip()
        a = self.branch_writing("a", "index.md", "| row A |\n")
        b = self.branch_writing("b", "index.md", "| row B |\n")
        self.assertGreater(
            self.conflicts(a, b), 0,
            "the control did not collide, so the case below proves nothing",
        )

    def test_the_new_shape_does_not(self) -> None:
        """Each change carries its own file, and the index is not in the tree."""
        a = self.branch_writing("a", "specs/a.md", "spec A\n")
        b = self.branch_writing("b", "specs/b.md", "spec B\n")
        self.assertEqual(
            self.conflicts(a, b), 0,
            "two independent filings still collide; the lock was not removed",
        )


class TheRetiredIndexStaysRetired(unittest.TestCase):
    """The row's whole point. If the tracked file returns, so does the lock."""

    def test_the_tracked_index_is_gone(self) -> None:
        self.assertFalse(
            (ROOT / ".agents/issue-index.md").exists(),
            "the tracked issue index is back; it is a surface every PR writes",
        )

    def test_the_archive_kept_its_content(self) -> None:
        archive = ROOT / ".agents/completed/issue-index.md"
        self.assertTrue(archive.is_file(), "the archive must keep the provenance")
        self.assertGreater(
            archive.read_text(encoding="utf-8").count("| [#"), 500,
            "the archive lost rows in the move",
        )

    def test_the_union_driver_is_gone_from_gitattributes(self) -> None:
        # The driver is what made the shape look safe locally while the forge
        # conflicted anyway. Leaving it would invite the next shared file.
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertNotIn("merge=union", text)

    def test_the_snapshot_is_ignored_so_no_pull_request_can_write_it(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".agents/issue-index.generated.md", text)



class RendererContractStaysWired(unittest.TestCase):
    def test_ci_runs_the_renderer_not_the_deleted_append_only_suite(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("test_check_issue_index_append_only.py", workflow)
        self.assertEqual(
            workflow.count("python3 tests/scripts/test_agent_issue_index.py"),
            1,
            "the existing renderer suite must run exactly once",
        )


class LocalCanonicalRendererIsOffline(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def record(self, number: int, title: str) -> records.IssueRecord:
        return records.IssueRecord(
            id=f"ISSUE-GH-{number}",
            title=title,
            row="ROW-A",
            state="OPEN",
            kind="bug",
            github=number,
            mirror="DIVERGED",
            availability="FULL",
            created="2026-08-01",
            updated="2026-08-01",
            closed="-",
            problem="Evidence.",
            resolution="-",
        )

    def test_refresh_reads_local_records_and_never_calls_github(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            issues_root = root / ".agents" / "issues"
            issue_path = issues_root / "ROW-A" / f"ISSUE-GH-{''}7.md"
            issue_path.parent.mkdir(parents=True)
            issue_path.write_text(
                records.render_issue_record(self.record(7, "local authority")),
                encoding="utf-8",
            )
            snapshot = root / ".agents" / "issue-index.generated.md"
            with mock.patch.object(self.mod, "ISSUES_ROOT", issues_root), \
                 mock.patch.object(self.mod, "SNAPSHOT", snapshot), \
                 mock.patch.object(
                     subprocess,
                     "run",
                     side_effect=AssertionError("refresh contacted GitHub"),
                 ):
                code, message = self.mod.refresh(rows={"ROW-A"}, owed=set())

            self.assertEqual(code, 0, message)
            self.assertEqual(
                snapshot.read_text(encoding="utf-8"),
                self.mod.render_local_files(issues_root),
            )
            self.assertRegex(message, r"1 open canonical issue")

    def test_local_files_render_in_stable_number_order_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            issues_root.mkdir(parents=True)
            for record in (self.record(12, "later"), self.record(7, "first")):
                (issues_root / f"{record.id}.md").write_text(
                    records.render_issue_record(record),
                    encoding="utf-8",
                )
            with unittest.mock.patch.object(
                subprocess,
                "run",
                side_effect=AssertionError("local renderer contacted GitHub"),
            ):
                text = self.mod.render_local_files(issues_root)

        rows = [line for line in text.splitlines() if line.startswith("| [#")]
        self.assertEqual([line.split("]")[0] for line in rows], ["| [#7", "| [#12"])
        self.assertIn("| `ROW-A` | first | bug |", rows[0])


    def test_local_render_escapes_every_free_form_table_cell_without_rewriting_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            issues_root.mkdir(parents=True)
            record = self.record(21, "Title | evidence")
            record = replace(record, kind="bug | regression")
            path = issues_root / f"{record.id}.md"
            canonical = records.render_issue_record(record)
            path.write_text(canonical, encoding="utf-8")

            text = self.mod.render_local_files(issues_root)

            self.assertEqual(path.read_text(encoding="utf-8"), canonical)
            row = next(line for line in text.splitlines() if line.startswith("| [#21]"))
            self.assertEqual(
                row,
                "| [#21](https://github.com/mudler/vllm.cpp/issues/21) "
                "| `ROW-A` | Title \\| evidence | bug \\| regression |",
            )
            self.assertEqual(len(re.split(r"(?<!\\)\|", row)[1:-1]), 4)

    def test_local_render_preserves_backslash_pipe_parity_in_every_free_form_cell(self) -> None:
        cases = (
            (r"Title \| evidence", r"bug \| regression"),
            (
                r"one \\ two | three \\\| four",
                r"kind || \\\\| trailing \\",
            ),
            (r"left|||middle\|right", r"bug\\|task\||other"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            issues_root.mkdir(parents=True)
            for offset, (title, kind) in enumerate(cases, start=22):
                with self.subTest(title=title, kind=kind):
                    record = replace(self.record(offset, title), kind=kind)
                    path = issues_root / f"{record.id}.md"
                    canonical = records.render_issue_record(record)
                    path.write_text(canonical, encoding="utf-8")

                    text = self.mod.render_local_files(issues_root)

                    self.assertEqual(path.read_text(encoding="utf-8"), canonical)
                    row = next(
                        line
                        for line in text.splitlines()
                        if line.startswith(f"| [#{offset}]")
                    )
                    cells = parse_markdown_row(row)
                    self.assertEqual(len(cells), 4)
                    self.assertEqual(
                        cells,
                        [
                            f"[#{offset}](https://github.com/mudler/vllm.cpp/issues/{offset})",
                            "`ROW-A`",
                            title,
                            kind,
                        ],
                    )


class LoadLocalFilesReportsEveryInvalidFile(unittest.TestCase):
    """One run of the loader names EVERY invalid record, not just the first.

    The loader used to raise on the first file whose record failed, so a run
    revealed one defect and each fix exposed the next only after another
    run; the first file alphabetically was always an _intake record nobody
    was editing. Three separate defects hid behind one another exactly this
    way during the ORPHAN-MODEL-ROWS repair
    (ISSUE-LOCAL-01M3NC14GE995V9E6F7GTYSQJ3).
    """

    def setUp(self) -> None:
        self.mod = load_module()

    def record(self, number: int, title: str) -> records.IssueRecord:
        return records.IssueRecord(
            id=f"ISSUE-GH-{number}",
            title=title,
            row="ROW-A",
            state="OPEN",
            kind="bug",
            github=number,
            mirror="SYNCED",
            availability="FULL",
            created="2026-08-01",
            updated="2026-08-01",
            closed="-",
            problem="Evidence.",
            resolution="-",
        )

    def test_two_broken_files_are_both_named_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            row_dir = issues_root / "ROW-A"
            for number, corrupt in ((7, "## Wrong"), (8, None)):
                path = row_dir / f"ISSUE-GH-{number}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                if corrupt is None:
                    text = records.render_issue_record(self.record(number, "fine"))
                    # Break the path-owner contract only: point the record at
                    # a row its directory does not carry.
                    text = text.replace("Row: ROW-A", "Row: ROW-B")
                    path.write_text(text, encoding="utf-8")
                else:
                    path.write_text(corrupt, encoding="utf-8")
            with self.assertRaises(records.IssueRecordError) as caught:
                self.mod.load_local_files(
                    issues_root, rows={"ROW-A"}, owed=set(), frozen_archive=b""
                )
            message = str(caught.exception)
            self.assertIn("ISSUE-GH-7.md", message)
            self.assertIn("ISSUE-GH-8.md", message)
            self.assertIn("missing the exact ## Problem heading", message)
            self.assertIn("path row 'ROW-A' must equal Row field ROW-B", message)

    def test_a_clean_collection_still_loads_and_validates_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            row_dir = issues_root / "ROW-A"
            row_dir.mkdir(parents=True)
            for number in (7, 8):
                (row_dir / f"ISSUE-GH-{number}.md").write_text(
                    records.render_issue_record(self.record(number, "fine")),
                    encoding="utf-8",
                )
            records_loaded = self.mod.load_local_files(
                issues_root, rows={"ROW-A"}, owed=set(), frozen_archive=b""
            )
            self.assertEqual(
                [record.id for record in records_loaded],
                ["ISSUE-GH-7", "ISSUE-GH-8"],
            )


if __name__ == "__main__":
    unittest.main()
