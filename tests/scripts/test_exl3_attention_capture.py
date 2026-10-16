"""Focused guards against cold/padding reads and changed original execution."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from capture_attention import active_cache_addresses, observe_projection, attention_capture_step


class AttentionCaptureTest(unittest.TestCase):
    def test_selected_d29_is_one_step_and_reads_only_initialized_prefix(self):
        self.assertEqual(attention_capture_step(29, 0), ("d29", 1, 157, 156))
        self.assertEqual(attention_capture_step(-1, 0), ("p128", 128, 128, 0))
        self.assertEqual(attention_capture_step(-1, 1), ("d1", 1, 129, 128))
        for selected, count in [(29, 1), (-1, 2), (28, 0), (True, 0), (29, True)]:
            with self.assertRaises(ValueError): attention_capture_step(selected, count)
        addresses = active_cache_addresses(1600, 10, 157, [7], [7 * 1600 + 156], [156], max_seq_len=157)
        self.assertEqual(addresses[:156], [(7, p) for p in range(156)])
        self.assertEqual(addresses[156:], [(7, 156)])
        for maximum, length in [(129, 157), (157, 158), (158, 157), (True, 157)]:
            with self.assertRaises(ValueError):
                active_cache_addresses(1600, 10, length, [7], [7 * 1600 + length - 1],
                                       [length - 1], max_seq_len=maximum)

    def test_active_rows_use_actual_nonzero_physical_block(self):
        rows = active_cache_addresses(1600, 10, 129, [7], [7 * 1600 + 128], [128])
        self.assertEqual(len(rows), 129)
        self.assertEqual(rows[0], (7, 0))
        self.assertEqual(rows[-1], (7, 128))
        self.assertNotIn((7, 129), rows)

    def test_page_crossing_and_prefill_slots(self):
        positions = list(range(5))
        rows = active_cache_addresses(4, 8, 5, [6, 2], [24, 25, 26, 27, 8], positions)
        self.assertEqual(rows, [(6, 0), (6, 1), (6, 2), (6, 3), (2, 0)])

    def test_unwritten_capacity_wrong_mapping_and_out_of_bounds_rejected(self):
        invalid = [
            (1600, 10, 0, [7], [], []),
            (1600, 10, 130, [7], [11329], [129]),
            (1600, 10, 129, [10], [16128], [128]),
            (1600, 10, 129, [-1], [-1472], [128]),
            (1600, 10, 129, [7, 8], [11328], [128]),
            (1600, 10, 129, [7], [128], [128]),
            (1600, 10, 129, [7], [11327], [127]),
            (1600, 10, 129, [7], [], []),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                active_cache_addresses(*args)

    def test_projection_delegates_once_and_preserves_original_result(self):
        qkv, positions, result = object(), object(), tuple(object() for _ in range(4))
        calls, saved = [], []
        def original(*args, **kwargs):
            calls.append((args, kwargs)); return result
        observed = observe_projection(original, saved.append)
        self.assertIs(observed(qkv, positions=positions), result)
        self.assertEqual(calls, [((qkv,), {"positions": positions})])
        self.assertEqual(saved, [result])

    def test_original_failure_has_no_fabricated_capture(self):
        saved = []
        def original(*args):
            raise RuntimeError("original error")
        with self.assertRaisesRegex(RuntimeError, "original error"):
            observe_projection(original, saved.append)(object())
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
