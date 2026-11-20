#!/usr/bin/env python3
"""Behavior tests for the K1/K2 lease runner.

The runner exists so the two measurements that decide the GLM4-MoE-Lite
campaign are one command instead of two remembered command lines, and so they
never write into the tracked goldens. These tests pin the three things that can
go wrong silently: the plan that gets executed, the refusal when a tool is
absent, and the diff arithmetic that says whether the oracle still reproduces
its own capture.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/glm4-moe-lite-k1k2-run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("glm4_moe_lite_k1k2_run", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_plan_runs_k1_then_k2_against_the_scratch_copy(self):
        tools = {
            self.runner.TOOL_K1: Path("scripts/k1.py"),
            self.runner.TOOL_K2: Path("scripts/k2.py"),
        }
        plan = self.runner.plan_commands(
            python="python3", tools=tools,
            golden_dir="scratch", topk=20, max_num_seqs="1,4",
            model="zai-org/GLM-4.7-Flash", max_tokens=16)
        self.assertEqual([step["name"] for step in plan], ["k1", "k2"])
        k1, k2 = plan
        self.assertIn("--golden-dir", k1["argv"])
        self.assertEqual(k1["argv"][k1["argv"].index("--golden-dir") + 1],
                         "scratch")
        self.assertEqual(k1["argv"][k1["argv"].index("--max-tokens") + 1], "16")
        self.assertEqual(k2["argv"][k2["argv"].index("--max-num-seqs") + 1],
                         "1,4")
        self.assertEqual(k2["argv"][k2["argv"].index("--topk") + 1], "20")

    def test_missing_tool_names_the_file_and_the_pr_that_supplies_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            only_k2 = Path(tmp) / self.runner.TOOL_K2
            only_k2.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with self.assertRaises(self.runner.ToolMissing) as caught:
                self.runner.required_tools(Path(tmp))
        message = str(caught.exception)
        self.assertIn(self.runner.TOOL_K1, message)
        self.assertIn(self.runner.TOOL_K1_PROVIDER, message)


class DiffTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_identical_arrays_report_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = Path(tmp) / "committed"
            regenerated = Path(tmp) / "regenerated"
            committed.mkdir()
            regenerated.mkdir()
            for name in ("greedy_ids.npy", "the_gap.npy"):
                arr = np.arange(6, dtype="<i4").reshape(2, 3)
                np.save(committed / name, arr)
                np.save(regenerated / name, arr)
            diffs = self.runner.diff_artifacts(committed, regenerated,
                                               ["greedy_ids.npy", "the_gap.npy"])
        self.assertEqual([d["differing"] for d in diffs], [0, 0])
        self.assertEqual([d["first_differing"] for d in diffs], [None, None])
        self.assertTrue(all(d["present"] for d in diffs))

    def test_one_changed_cell_is_located(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = Path(tmp) / "committed"
            regenerated = Path(tmp) / "regenerated"
            committed.mkdir()
            regenerated.mkdir()
            base = np.zeros((2, 3), dtype="<i4")
            moved = base.copy()
            moved[1, 2] = 7
            np.save(committed / "greedy_ids.npy", base)
            np.save(regenerated / "greedy_ids.npy", moved)
            diffs = self.runner.diff_artifacts(committed, regenerated,
                                               ["greedy_ids.npy"])
        self.assertEqual(diffs[0]["differing"], 1)
        self.assertEqual(diffs[0]["first_differing"], [1, 2])

    def test_an_artifact_the_run_never_wrote_is_named_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = Path(tmp) / "committed"
            regenerated = Path(tmp) / "regenerated"
            committed.mkdir()
            regenerated.mkdir()
            np.save(committed / "teacher_forced_gap_nats.npy",
                    np.zeros((2, 3), dtype="<f8"))
            diffs = self.runner.diff_artifacts(
                committed, regenerated, ["teacher_forced_gap_nats.npy"])
        self.assertFalse(diffs[0]["present"])
        self.assertEqual(diffs[0]["differing"], 0)


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_a_clean_run_reports_reproduced(self):
        diffs = [{"name": "greedy_ids.npy", "present": True, "differing": 0,
                  "first_differing": None}]
        summary = self.runner.summarise(diffs, k1_sub_millinat=7,
                                        k2_verdict="ORACLE PATH-DEPENDENT")
        self.assertTrue(summary["reproduced"])
        self.assertEqual(summary["k1_sub_millinat"], 7)
        self.assertEqual(summary["k2_verdict"], "ORACLE PATH-DEPENDENT")

    def test_any_difference_or_absence_is_not_reproduced(self):
        diffs = [
            {"name": "a.npy", "present": True, "differing": 2,
             "first_differing": [0, 1]},
            {"name": "b.npy", "present": False, "differing": 0,
             "first_differing": None},
        ]
        summary = self.runner.summarise(diffs, k1_sub_millinat=0,
                                        k2_verdict="ORACLE PATH-STABLE")
        self.assertFalse(summary["reproduced"])

    def test_the_tracked_golden_directory_is_refused_as_scratch(self):
        """K1 rewrites artifacts in --golden-dir, so it must never be the track."""
        tracked = ROOT / "tests/parity/goldens/glm4_moe_lite_greedy"
        with self.assertRaises(SystemExit) as caught:
            self.runner.stage_goldens(tracked, tracked / "scratch")
        self.assertIn("tracked golden", str(caught.exception))

    def test_scratch_staging_copies_every_committed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "greedy_ids.npy").write_bytes(b"x")
            (source / "our_ids.i32").write_bytes(b"y")
            staged = self.runner.stage_goldens(source, Path(tmp) / "scratch")
            copied = sorted(p.name for p in staged.iterdir())
        self.assertEqual(copied, ["greedy_ids.npy", "our_ids.i32"])


if __name__ == "__main__":
    sys.exit(unittest.main())
