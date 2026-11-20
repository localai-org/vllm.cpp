#!/usr/bin/env python3
"""Behavior tests for K2: does vLLM's argmax depend on the path it ran through?

The goldens license the STRICT bar on five greedy runs that agree with each
other. Those five share one numeric path -- same prefill, same batch shape, same
decode kernels -- so they cannot see greedy-decode disagreeing with a
teacher-forced prefill. These tests pin the comparison that answers it, on
synthetic inputs, so the arithmetic is settled before any GPU is spent.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/glm4-moe-lite-path-determinism.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("glm4_moe_lite_path_determinism", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComparePathsTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def test_agreeing_paths_report_no_path_dependence(self):
        """The oracle keeps its own answer on both paths."""
        out = self.tool.compare_paths(
            greedy_ids=[[5, 6, 7]],
            teacher_argmax=[[5, 6, 7]],
            our_ids=[[5, 6, 9]],
        )
        self.assertEqual(out["disagree"], 0)
        self.assertFalse(out["path_dependent"])
        self.assertEqual(out["first_divergences"], [(0, 2)])
        self.assertEqual(out["flip_at_first_divergence"], 0)

    def test_argmax_moving_to_our_token_is_a_path_flip(self):
        """The whole question: teacher-forced, the oracle takes OUR token."""
        out = self.tool.compare_paths(
            greedy_ids=[[5, 6, 7]],
            teacher_argmax=[[5, 6, 9]],
            our_ids=[[5, 6, 9]],
        )
        self.assertEqual(out["disagree"], 1)
        self.assertTrue(out["path_dependent"])
        self.assertEqual(out["flip_at_first_divergence"], 1)
        self.assertEqual(out["oracle_reproduced_ours"], 1)

    def test_absent_distributions_are_not_a_comparison(self):
        """-1 means vLLM reported nothing there; that is not agreement."""
        out = self.tool.compare_paths(
            greedy_ids=[[5, 6, 7]],
            teacher_argmax=[[-1, -1, -1]],
            our_ids=[[5, 6, 9]],
        )
        self.assertEqual(out["compared"], 0)
        self.assertEqual(out["disagree"], 0)
        self.assertFalse(out["path_dependent"])
        self.assertEqual(out["flip_at_first_divergence"], 0)

    def test_an_exact_prompt_has_no_first_divergence(self):
        out = self.tool.compare_paths(
            greedy_ids=[[5, 6], [5, 6]],
            teacher_argmax=[[5, 6], [5, 9]],
            our_ids=[[5, 6], [5, 9]],
        )
        self.assertEqual(out["first_divergences"], [(1, 1)])
        self.assertEqual(out["flip_at_first_divergence"], 1)
        self.assertEqual(out["disagree"], 1)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(AssertionError):
            self.tool.compare_paths(
                greedy_ids=[[5, 6]],
                teacher_argmax=[[5]],
                our_ids=[[5, 6]],
            )


if __name__ == "__main__":
    sys.exit(unittest.main())
