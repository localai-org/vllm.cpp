#!/usr/bin/env python3
"""Host-only guards for the real grouped SmallM oracle capture."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
import capture_grouped as grouped
from extract_projection import digest, write_safetensors


class GroupCaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = []
        for source, n in enumerate((128, 256)):
            p = self.root / f"source{source}.safetensors"
            tensors = {"trellis": ("I16", [8, n//16, 64], bytes(8*(n//16)*128)),
                       "suh": ("F16", [128], b"\0\x3c"*128),
                       "svh": ("F16", [n], b"\0\x3c"*n),
                       "mul1": ("I32", [], struct.pack("<I", 0x83DCD12D))}
            for m in (1, 4):
                tensors[f"activation_m{m}_fp16"] = ("F16", [m, 128], bytes(m*128*2))
            write_safetensors(p, tensors, {})
            r = {"checkpoint": grouped.CHECKPOINT,
                 "projection": {"source_k": 128, "source_n": n, "bits": 4,
                                "codebook": "mul1", "column_start": 0, "column_count": n},
                 "fixture_sha256": digest(p.read_bytes()),
                 "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                   for name, t in tensors.items()}}
            p.with_suffix(".json").write_text(json.dumps(r))
            self.fixtures.append(p)

    def change(self, field, value):
        p = self.fixtures[1].with_suffix(".json")
        r = json.loads(p.read_text())
        r["projection"][field] = value
        p.write_text(json.dumps(r))

    def test_mixed_full_width_sources(self):
        _, k, bits, bounds = grouped.checked_group(self.fixtures)
        self.assertEqual((k, bits, bounds), (128, 4, [0, 128, 384]))

    def test_explicit_single_source_mode(self):
        _, k, bits, bounds = grouped.checked_group(self.fixtures[:1], single_source=True)
        self.assertEqual((k, bits, bounds), (128, 4, [0, 128]))
        for fixtures in ([], self.fixtures):
            with self.assertRaisesRegex(grouped.headers.InventoryError, "exactly one"):
                grouped.checked_group(fixtures, single_source=True)
        self.change("column_count", 128)
        with self.assertRaisesRegex(grouped.headers.InventoryError, "compatible full-width"):
            grouped.checked_group(self.fixtures[1:], single_source=True)

    def test_refuses_partial_sources_and_mixed_arithmetic(self):
        for key, bad in (("column_start", 128), ("column_count", 128),
                         ("source_k", 256), ("bits", 6), ("codebook", "mcg")):
            p = self.fixtures[1].with_suffix(".json")
            original = p.read_text()
            with self.subTest(key=key):
                self.change(key, bad)
                with self.assertRaisesRegex(grouped.headers.InventoryError, "compatible full-width"):
                    grouped.checked_group(self.fixtures)
            p.write_text(original)

    def test_refuses_changed_payload(self):
        with self.fixtures[1].open("r+b") as f:
            f.seek(-1, 2); f.write(b"\xff")
        with self.assertRaisesRegex(grouped.headers.InventoryError, "fixture SHA-256"):
            grouped.checked_group(self.fixtures)

    def test_refuses_wrong_checkpoint_and_single_source(self):
        p = self.fixtures[1].with_suffix(".json")
        r = json.loads(p.read_text())
        r["checkpoint"]["revision"] = "changed"
        p.write_text(json.dumps(r))
        with self.assertRaisesRegex(grouped.headers.InventoryError, "checkpoint mismatch"):
            grouped.checked_group(self.fixtures)
        with self.assertRaisesRegex(grouped.headers.InventoryError, "two sources"):
            grouped.checked_group(self.fixtures[:1])

    def test_refuses_wrong_image_and_overwrite_before_gpu_import(self):
        out = self.root / "capture.safetensors"
        with self.assertRaisesRegex(grouped.headers.InventoryError, "pinned production image"):
            grouped.capture(self.fixtures, out, self.root, "wrong")
        out.with_suffix(".json").write_text("keep")
        with self.assertRaisesRegex(grouped.headers.InventoryError, "overwrite"):
            grouped.capture(self.fixtures, out, self.root, grouped.IMAGE)
        self.assertEqual(out.with_suffix(".json").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
