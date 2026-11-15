#!/usr/bin/env python3
"""Behavior tests for K1: the teacher-forced gap at full resolution.

`neartie_gap_mnats.npy` is int32 MILLINATS, and the capture rounds `gap * 1000`,
so every gap below half a millinat is committed as 0. At the seven
first-divergence positions of the GLM-4.7-Flash goldens that single number is
the whole question: the prefix is the oracle's, so its teacher-forced argmax
must be its own greedy token and the gap must be positive. A 0 that means
"under half a millinat" cannot answer it.

These tests pin the float64 nat array and the argmax ids that K1 adds beside the
committed millinat array, and they pin the loss the millinat array still carries
so the reason the second artifact exists stays executable.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/glm4-moe-lite-neartie-gap.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("glm4_moe_lite_neartie_gap", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_logprobs(rows):
    """vLLM's `prompt_logprobs`: one entry per position, None or {id: obj}."""
    out = []
    for row in rows:
        if row is None:
            out.append(None)
        else:
            out.append({tid: SimpleNamespace(logprob=lp) for tid, lp in row.items()})
    return out


class GapRowsTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def test_exact_tie_stores_zero_in_both_resolutions(self):
        plp = prompt_logprobs([None, {5: -0.1, 7: -1.0}])
        mnats, nats, argmax = self.tool.gap_rows([1], [5], plp, 20)
        self.assertEqual(mnats, [0])
        self.assertEqual(nats, [0.0])
        self.assertEqual(argmax, [5])

    def test_sub_millinat_deficit_survives_only_in_the_nat_array(self):
        """K1's whole point: 0.0004 nats is 0 millinats."""
        plp = prompt_logprobs([None, {5: -0.1004, 7: -0.1}])
        mnats, nats, argmax = self.tool.gap_rows([1], [5], plp, 20)
        self.assertEqual(mnats, [0], "the committed array cannot see this gap")
        self.assertAlmostEqual(nats[0], 0.0004, places=9)
        self.assertGreater(nats[0], 0.0, "the oracle prefers its own token")
        self.assertEqual(argmax, [7], "and that token is the oracle's, not ours")

    def test_multi_nat_deficit_is_still_millinats(self):
        plp = prompt_logprobs([None, {5: -1.6, 7: -0.1}])
        mnats, nats, argmax = self.tool.gap_rows([1], [5], plp, 20)
        self.assertEqual(mnats, [1500])
        self.assertAlmostEqual(nats[0], 1.5, places=9)
        self.assertEqual(argmax, [7])

    def test_token_outside_topk_carries_the_sentinel_in_both_arrays(self):
        plp = prompt_logprobs([None, {7: -0.1}])
        mnats, nats, argmax = self.tool.gap_rows([1], [9], plp, 20)
        self.assertEqual(mnats, [self.tool.OUTSIDE_TOPK_MNATS])
        self.assertEqual(nats, [float(self.tool.OUTSIDE_TOPK_MNATS)])
        self.assertEqual(argmax, [7])

    def test_absent_position_reports_no_argmax(self):
        plp = prompt_logprobs([None, None])
        mnats, nats, argmax = self.tool.gap_rows([1], [5], plp, 20)
        self.assertEqual(argmax, [-1])
        self.assertEqual(mnats, [self.tool.OUTSIDE_TOPK_MNATS])

    def test_positions_are_offset_by_the_prompt_length(self):
        """A prompt of 3 shifts the first generated position to 3."""
        plp = prompt_logprobs([None, None, None, {5: -0.1, 7: -0.1}])
        mnats, nats, argmax = self.tool.gap_rows([1, 2, 3], [5], plp, 20)
        self.assertEqual(mnats, [0])
        self.assertEqual(argmax, [5])


if __name__ == "__main__":
    sys.exit(unittest.main())
