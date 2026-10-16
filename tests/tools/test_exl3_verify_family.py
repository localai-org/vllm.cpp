import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(root / "exl3_reference"))
spec = importlib.util.spec_from_file_location("capture_verify_family", root / "exl3_reference/capture_verify_family.py")
family = importlib.util.module_from_spec(spec)
spec.loader.exec_module(family)


class VerifyFamilyTests(unittest.TestCase):
    def test_active_addresses_leave_page_and_head_padding_poisoned(self):
        keys = bytes([0x21]) * (4100 * 1024)
        values = bytes([0x35]) * len(keys)
        for name, layout in family.LAYOUTS.items():
            raw = family.cache_bytes(keys, values, 1601, name, poison=0x7f)
            page, row, head, _ = layout["strides"]
            for block, r in [(3, 0), (3, 1599), (10, 0)]:
                for h in range(4):
                    offset = block * page + r * row + h * head
                    self.assertEqual(raw[offset:offset + 256], keys[:256])
                    offset += layout["v_offset"]
                    self.assertEqual(raw[offset:offset + 256], values[:256])
            offset = 10 * page + row
            self.assertEqual(raw[offset:offset + 256], bytes([0x7f]) * 256)
            self.assertEqual(raw[0], 0x7f)  # Unused physical block.
            if name == "padded_interleaved":
                self.assertEqual(raw[3 * page + 1600 * row], 0x7f)

    def test_long_recipe_and_tail_perturbation_are_explicit(self):
        keys = b"".join(bytes([i % 255]) * 1024 for i in range(4100))
        values = bytes([0x34]) * len(keys)
        raw = family.cache_bytes(keys, values, 4801, "interleaved", mutate=True)
        layout = family.LAYOUTS["interleaved"]
        for r in (4099, 4100, 4797, 4798, 4800):
            base = family.BLOCK_IDS[r // 1600] * layout["strides"][0] + (r % 1600) * 2048
            self.assertEqual(raw[base], (r % 4100) % 255)
            self.assertEqual(raw[base + 256], 0x38 if r >= 4798 else 0x34)
        with self.assertRaises(ValueError):
            family.cache_bytes(keys, values, 262144, "interleaved")
        with self.assertRaises(ValueError):
            family.cache_bytes(keys[:-1], values, 4100, "interleaved")

    def test_requested_shapes_have_distinct_query_rows_and_physical_blocks(self):
        cases = family.family_cases()
        self.assertEqual(len(cases), 39)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        self.assertEqual(len(set(family.BLOCK_IDS)), family.BLOCKS)
        self.assertEqual(family.BLOCK_IDS[:3], [3, 10, 23])
        for rows in (2, 3, 4, 5):
            ids = family.query_row_ids(rows)
            self.assertEqual(len(ids), rows)
            self.assertTrue(all(0 <= i < 4 for i in ids))


if __name__ == "__main__":
    unittest.main()
