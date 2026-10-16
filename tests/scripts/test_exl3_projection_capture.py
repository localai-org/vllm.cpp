#!/usr/bin/env python3
"""Focused host checks: oracle capture must reject changed inputs/runtime."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
import capture_projection as capture
from extract_projection import digest, write_safetensors


class CaptureIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "fixture.safetensors"
        self.tensors = {"activation_m1_fp16": ("F16", [1, 128], b"\0\x3c" * 128)}
        write_safetensors(self.fixture, self.tensors, {})
        self.receipt = {"fixture_sha256": digest(self.fixture.read_bytes()),
                        "tensor_hashes": {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                          for n, t in self.tensors.items()}}
        self.write_receipt()

    def write_receipt(self):
        self.fixture.with_suffix(".json").write_text(json.dumps(self.receipt))

    def test_valid_fixture_and_changed_packed_bytes(self):
        self.assertEqual(capture.checked_fixture(self.fixture), self.receipt)
        with self.fixture.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        with self.assertRaisesRegex(capture.headers.InventoryError, "fixture SHA-256"):
            capture.checked_fixture(self.fixture)

    def test_tensor_receipt_cannot_disagree_with_fixture(self):
        for field, bad, message in (("shape", [128, 1], "dtype/shape"),
                                    ("dtype", "BF16", "dtype/shape"),
                                    ("sha256", "changed", "tensor SHA-256")):
            original = self.receipt["tensor_hashes"]["activation_m1_fp16"][field]
            self.receipt["tensor_hashes"]["activation_m1_fp16"][field] = bad
            self.write_receipt()
            with self.assertRaisesRegex(capture.headers.InventoryError, message):
                capture.checked_fixture(self.fixture)
            self.receipt["tensor_hashes"]["activation_m1_fp16"][field] = original
        self.receipt["tensor_hashes"]["unexpected"] = {}
        self.write_receipt()
        with self.assertRaisesRegex(capture.headers.InventoryError, "tensor set"):
            capture.checked_fixture(self.fixture)

    def test_refuses_overwrite_and_wrong_image_before_gpu_import(self):
        output = self.root / "capture.safetensors"
        with self.assertRaisesRegex(capture.headers.InventoryError, "pinned production image"):
            capture.capture(self.fixture, output, self.root, "wrong-image")
        output.with_suffix(".json").write_text("keep")
        with self.assertRaisesRegex(capture.headers.InventoryError, "overwrite"):
            capture.capture(self.fixture, output, self.root, capture.IMAGE)
        self.assertEqual(output.with_suffix(".json").read_text(), "keep")

    def test_changed_library_is_rejected_before_gpu_import(self):
        (self.root / "exl3xpu").mkdir()
        (self.root / "exl3xpu/_C.so").write_bytes(b"wrong binary")
        with self.assertRaisesRegex(capture.headers.InventoryError, "runtime SHA-256"):
            capture.checked_runtime(self.root)

    def test_route_override_is_rejected(self):
        (self.root / "exl3xpu").mkdir()
        for name in ("_C.so", "ops.py"):
            (self.root / "exl3xpu" / name).write_bytes(b"test")
        with mock.patch.object(capture, "LIB_SHA", digest(b"test")), \
             mock.patch.object(capture, "OPS_SHA", digest(b"test")), \
             mock.patch.dict(os.environ, {"EXL3_TARGET_THREADS": "12"}, clear=True):
            with self.assertRaisesRegex(capture.headers.InventoryError, "overrides"):
                capture.checked_runtime(self.root)


if __name__ == "__main__":
    unittest.main()
