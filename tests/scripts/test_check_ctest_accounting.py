#!/usr/bin/env python3
"""Behaviour checks for the arm64-full ctest guard in ``.github/workflows/ci.yml``.

The guard used to be unreachable: GitHub runs ``run:`` blocks with ``bash -e``,
so a failing ctest aborted the step before the guard or its evidence printed.
It was also inert: it parsed ``out of N`` from ctest's summary, which is the
enumeration total. These tests execute the workflow's own ``run:`` block under
``bash -e`` with a fake ctest, so both shapes are detected as behaviour rather
than as text in the YAML.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/workflows/ci.yml"
CHECKER = ROOT / "scripts/check-ctest-accounting.py"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow_checker = _load("check_release_workflow", "scripts/check-release-workflow.py")
accounting = _load("check_ctest_accounting", "scripts/check-ctest-accounting.py")

PASS = "    Start 1: t_a\n1/2 Test #1: t_a .................   Passed    0.01 sec\n"
FAIL = "    Start 1: t_a\n1/1 Test #1: t_a .................***Failed    0.01 sec\n"
SKIP = (
    PASS
    + "    Start 2: t_b\n2/2 Test #2: t_b .................***Skipped   0.01 sec\n"
    + "\n50% tests passed, 0 tests failed out of 2\n\n"
    + "The following tests did not run:\n\t  2 - t_b (Skipped)\n"
)


def guard_script() -> str:
    document = workflow_checker.load_workflow_yaml(CI.read_text(encoding="utf-8"))
    steps = document["jobs"]["build-test-cpu-arm64-full"]["steps"]
    step = next(step for step in steps if step.get("name") == "Test")
    return str(step["run"])


class ParsingTests(unittest.TestCase):
    def test_counts_each_terminal_kind(self):
        text = PASS + SKIP.replace(PASS, "") + FAIL
        result = accounting.parse(text)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.accounted, 3)
        self.assertEqual(result.executed, 2)

    def test_no_terminal_lines_is_a_failure(self):
        result = accounting.parse("ctest produced nothing\n")
        self.assertEqual(result.accounted, 0)
        self.assertTrue(accounting.evaluate(result, 3))

    def test_a_skip_is_accounted_but_never_counted_as_executed(self):
        result = accounting.parse(SKIP)
        self.assertEqual(result.accounted, 2)
        self.assertEqual(result.executed, 1)
        self.assertEqual(accounting.evaluate(result, 2), [])
        self.assertTrue(accounting.evaluate(accounting.parse(PASS), 2))

    def test_duplicate_status_lines_are_refused(self):
        result = accounting.parse(PASS + PASS + PASS)
        self.assertEqual(result.accounted, 3)
        self.assertTrue(accounting.evaluate(result, 2))


class WorkflowGuardTests(unittest.TestCase):
    """Run the workflow's real Test step under the shell GitHub actually uses."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="arm64-ctest-guard-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        if CHECKER.exists():
            shutil.copy2(CHECKER, self.root / "scripts" / CHECKER.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "ctest"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            'cat "$CTEST_FIXTURE"\n'
            'exit "${CTEST_RC:-0}"\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.bin = bin_dir

    def run_guard(self, script_body: str, fixture: str, count: int, ctest_rc: int):
        fixture_path = self.root / "fixture.txt"
        fixture_path.write_text(fixture, encoding="utf-8")
        script = re.sub(
            r"\$\{\{\s*steps\.enumerate\.outputs\.count\s*\}\}",
            str(count),
            script_body,
        )
        script_path = self.root / "step.sh"
        script_path.write_text(script, encoding="utf-8")
        env = dict(os.environ)
        env["CTEST_FIXTURE"] = str(fixture_path)
        env["CTEST_RC"] = str(ctest_rc)
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        return subprocess.run(
            ["bash", "-e", str(script_path)],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def test_failing_ctest_still_prints_its_output_and_reaches_the_guard(self):
        completed = self.run_guard(guard_script(), FAIL, count=1, ctest_rc=8)
        self.assertIn("Test #1", completed.stdout, "cat ctest-run.txt never ran")
        self.assertIn("ACCOUNTED", completed.stdout)
        self.assertNotEqual(completed.returncode, 0)

    def test_fully_accounted_passing_run_is_accepted(self):
        completed = self.run_guard(guard_script(), PASS + PASS, count=2, ctest_rc=0)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("EXECUTED '2'", completed.stdout)

    def test_skip_is_reported_as_not_executed_when_ctest_passes(self):
        completed = self.run_guard(guard_script(), SKIP, count=2, ctest_rc=0)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("ACCOUNTED '2'", completed.stdout)
        self.assertIn("EXECUTED '1'", completed.stdout)
        self.assertIn("1 skipped entries are not passes", completed.stdout)

    def test_short_run_is_refused(self):
        completed = self.run_guard(guard_script(), PASS, count=2, ctest_rc=0)
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("run is not the suite", completed.stdout)

    def test_genuine_ctest_failure_is_not_masked_by_clean_accounting(self):
        completed = self.run_guard(guard_script(), FAIL, count=1, ctest_rc=8)
        self.assertIn("Test #1", completed.stdout)
        self.assertEqual(completed.returncode, 8, completed.stdout)


if __name__ == "__main__":
    unittest.main()
