"""Focused independent FP16 RoPE boundary checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from replay_attention_rope import half, rope_half


class RopeBoundaryTest(unittest.TestCase):
    def test_identity_preserves_unrotated_tail(self):
        self.assertEqual(rope_half([1, 2, 3, 4], [1, 0], 1, 1, 4, 2), [1, 2, 3, 4])

    def test_product_rounding_differs_from_single_final_round(self):
        a, b, c, s = half(.71), half(.37), half(.83), half(.54)
        result = rope_half([a, b], [c, s], 1, 1, 2, 2)
        self.assertNotEqual(result, [half(a*c - b*s), half(a*s + b*c)])
        self.assertEqual(result, [half(half(a*c) - half(b*s)), half(half(a*s) + half(b*c))])

    def test_wrong_shape_refuses(self):
        with self.assertRaises(ValueError):
            rope_half([1], [1, 0], 1, 1, 2, 2)


if __name__ == "__main__":
    unittest.main()
