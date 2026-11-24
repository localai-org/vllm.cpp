#!/usr/bin/env python3
"""Behaviour checks for the sanitize-cpu(thread) runner-death retry in ci.yml.

The retry must distinguish a killed ctest (no terminal summary) from a test
that actually failed (terminal summary, nonzero status). These tests execute
the workflow's own ``run:`` block under ``bash -e -o pipefail`` with a fake
ctest, so the guarantee is measured as behaviour rather than read as YAML text.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/workflows/ci.yml"
HELPER = ROOT / "scripts/ctest-run-complete.py"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow_checker = _load("check_release_workflow", "scripts/check-release-workflow.py")
helper = _load("ctest_run_complete", "scripts/ctest-run-complete.py")

FAKE_CTEST = """#!/bin/bash
# Deterministic stand-in for ctest; records every invocation by shard.
args="$*"
if [[ "$args" == *"-E"* ]]; then shard=main
elif [[ "$args" == *"test_ltx2_video"* ]]; then shard=ltx2
else shard=dots3; fi
printf '%s\\n' "$shard" >> "$FAKE_CTEST_SHARDS"
n=$(grep -c "^$shard$" "$FAKE_CTEST_SHARDS" || true)
if [[ "$FAKE_CTEST_MODE" == "real-failure" ]]; then
  echo "1/1 Test #1: fake ............***Failed    0.01 sec"
  echo "0% tests passed, 1 tests failed out of 1"
  exit 1
fi
if [[ "$FAKE_CTEST_MODE" == "death-twice" ]]; then exit 137; fi
if [[ "$FAKE_CTEST_MODE" == "death-once" && "$shard" == "main" && "$n" == "1" ]]; then exit 137; fi
echo "1/1 Test #1: fake ............   Passed    0.01 sec"
echo "100% tests passed, 0 tests failed out of 1"
exit 0
"""


def thread_step_run() -> str:
    document = workflow_checker.load_workflow_yaml(CI.read_text(encoding="utf-8"))
    steps = document["jobs"]["sanitize-cpu"]["steps"]
    step = next(s for s in steps if s.get("name") == "Test (thread, sharded with runner-death retry)")
    return str(step["run"])


def plain_step() -> dict:
    document = workflow_checker.load_workflow_yaml(CI.read_text(encoding="utf-8"))
    steps = document["jobs"]["sanitize-cpu"]["steps"]
    return next(s for s in steps if s.get("name") == "Test")


class DetectorTests(unittest.TestCase):
    def test_pass_and_fail_summaries_are_complete(self):
        self.assertTrue(helper.complete("100% tests passed, 0 tests failed out of 3\n"))
        self.assertTrue(helper.complete("0% tests passed, 1 tests failed out of 1\n"))
        self.assertTrue(helper.complete("Total Test time = 4.20 sec\n"))

    def test_truncated_log_is_incomplete(self):
        self.assertFalse(helper.complete(""))
        self.assertFalse(helper.complete("    Start 1: test_ltx2_video\n"))
        self.assertFalse(helper.complete("1/3 Test #1: fake .... Passed 0.01 sec\n"))

    def test_terminal_call_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "ctest.txt"
            log.write_text("100% tests passed, 0 tests failed out of 1\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(HELPER), str(log)])
            self.assertEqual(result.returncode, 0)
            log.write_text("    Start 1: fake\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(HELPER), str(log)])
            self.assertEqual(result.returncode, 1)


class WorkflowBehaviourTests(unittest.TestCase):
    def _run_fake(self, script: str, mode: str) -> tuple[int, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bin_dir = tmpdir / "bin"
            bin_dir.mkdir()
            ctest = bin_dir / "ctest"
            ctest.write_text(FAKE_CTEST, encoding="utf-8")
            ctest.chmod(0o755)
            shards = tmpdir / "shards.txt"
            env = dict(os.environ)
            env.update({
                "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                "FAKE_CTEST_MODE": mode,
                "FAKE_CTEST_SHARDS": str(shards),
            })
            result = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            shard_lines = shards.read_text().splitlines() if shards.exists() else []
            for label in ("main", "ltx2", "dots3"):
                (ROOT / f"ctest-{label}.txt").unlink(missing_ok=True)
            return result.returncode, shard_lines

    def test_real_failure_is_never_retried(self):
        rc, shards = self._run_fake(thread_step_run(), "real-failure")
        self.assertNotEqual(rc, 0)
        self.assertEqual(shards, ["main"])  # step aborts on the first real failure

    def test_runner_death_is_retried_once(self):
        rc, shards = self._run_fake(thread_step_run(), "death-once")
        self.assertEqual(rc, 0)
        self.assertEqual(shards, ["main", "main", "ltx2", "dots3"])

    def test_persistent_runner_death_fails_after_two_attempts(self):
        rc, shards = self._run_fake(thread_step_run(), "death-twice")
        self.assertNotEqual(rc, 0)
        self.assertEqual(shards, ["main", "main"])

    def test_address_undefined_step_is_unchanged(self):
        step = plain_step()
        self.assertEqual(step.get("if"), "matrix.lane != 'thread'")
        self.assertEqual(str(step["run"]), "ctest --test-dir build-sanitize --output-on-failure")


if __name__ == "__main__":
    unittest.main()
