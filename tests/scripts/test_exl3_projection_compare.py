#!/usr/bin/env python3
"""Focused metric tests: relative norms, exact bits, and nonfinite rejection."""
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from compare_projection import LIMIT, bit_stats, metrics


class ProjectionMetricsTest(unittest.TestCase):
    def test_relative_norm_uses_entire_reference_not_mean_of_relative_errors(self):
        expected = struct.pack("<ff", 3, 4)
        actual = struct.pack("<ff", 3.3, 4.4)
        result = metrics(actual, expected, "F32")
        self.assertAlmostEqual(result["relative_norm"], 0.1, places=6)
        self.assertFalse(result["gate_pass"])
        self.assertEqual(result["first_bit_mismatch"], 0)

    def test_signed_zero_is_numerically_equal_but_not_bit_exact(self):
        result = metrics(struct.pack("<e", -0.0), struct.pack("<e", 0.0), "F16")
        self.assertEqual(result["relative_norm"], 0)
        self.assertTrue(result["gate_pass"])
        self.assertFalse(result["bit_exact"])
        self.assertEqual(result["bit_mismatches"], 1)

    def test_zero_reference_and_nonfinite_values_cannot_pass(self):
        self.assertFalse(metrics(struct.pack("<e", 1), struct.pack("<e", 0), "F16")["gate_pass"])
        for bad in (math.inf, -math.inf, math.nan):
            result = metrics(struct.pack("<f", bad), struct.pack("<f", 1), "F32")
            self.assertFalse(result["finite"])
            self.assertFalse(result["gate_pass"])

    def test_whole_tensor_bits_and_strict_threshold(self):
        self.assertEqual(bit_stats(b"\0\0\0\0", b"\0\0\1\0", 2)["first_bit_mismatch"], 1)
        self.assertFalse(metrics(struct.pack("<f", 1 + LIMIT * 2), struct.pack("<f", 1), "F32")["gate_pass"])
        self.assertTrue(metrics(struct.pack("<f", 1 + LIMIT / 2), struct.pack("<f", 1), "F32")["gate_pass"])


if __name__ == "__main__":
    unittest.main()
