import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(root / "exl3_reference"))
spec = importlib.util.spec_from_file_location("capture_verify", root / "exl3_reference/capture_verify.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class VerifyCaptureTests(unittest.TestCase):
    def test_active_rows_do_not_include_unused_table_or_tail(self):
        self.assertEqual(capture.active_page_slices(4096, 1600, [7, 3, 9, -1]), [
            {"physical_block": 7, "active_rows": 1600},
            {"physical_block": 3, "active_rows": 1600},
            {"physical_block": 9, "active_rows": 896}])
        for length in (1599, 1600, 1601, 4799, 4800, 4801, 32768):
            pages = capture.active_page_slices(length, 1600, list(range(30)))
            self.assertEqual(sum(p["active_rows"] for p in pages), length)
            self.assertEqual(len(pages), (length + 1599) // 1600)

    def test_missing_active_blocks_and_wrong_page_are_refused(self):
        for length, page, table in [(3, 1600, [0]), (4096, 1664, [0, 1, 2]),
                                    (4096, 1600, [0, 1]), (4096, 1600, [0, -1, 2])]:
            with self.assertRaises(ValueError):
                capture.active_page_slices(length, page, table)


if __name__ == "__main__":
    unittest.main()
