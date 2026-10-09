#!/usr/bin/env python3
"""Host tests for bit-preserving EXL3 fixture extraction and identity failures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("extract_projection", ROOT / "tools/exl3_reference/extract_projection.py")
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


class ProjectionExtractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "model"
        self.model.mkdir()
        self.output = root / "fixture.safetensors"
        self.reference = root / "reference.json"
        self.k, self.n = 128, 384
        self.parts = {}
        # Different K/N tile coordinates and both high/low I16 bits expose
        # flatten-and-slice errors and numerical casts masquerading as views.
        for bits in (4, 6):
            tr = b"".join(struct.pack("<H", (0x8000 + row * 997 + col * 71 + word) & 0xffff)
                          for row in range(8) for col in range(24) for word in range(16 * bits))
            for suffix, dtype, shape, raw in (
                ("trellis", "I16", [8, 24, 16 * bits], tr),
                ("suh", "F16", [128], b"\x00\xbc" * 128),
                ("svh", "F16", [384], b"".join(struct.pack("<H", i + 0x3000) for i in range(384))),
                ("mul1", "I32", [], struct.pack("<I", 2212286765)),
            ):
                self.parts[f"model.language_model.layers.0.mlp.gate{bits}.{suffix}"] = (dtype, shape, raw)
        self.write_model()

    def write_model(self):
        path = self.model / "model.safetensors"
        path.unlink(missing_ok=True)
        extract.write_safetensors(path, self.parts, {})
        index = {"metadata": {"total_size": sum(len(t[2]) for t in self.parts.values())},
                 "weight_map": {n: path.name for n in self.parts}}
        (self.model / "model.safetensors.index.json").write_text(json.dumps(index))
        (self.model / "config.json").write_text("{}")
        self.report = extract.headers.inventory(self.model)
        provenance = self.model / ".cache/huggingface/download"
        provenance.mkdir(parents=True, exist_ok=True)
        (provenance / "model.safetensors.metadata").write_text("test-revision\ntest-etag\n0\n")
        reference = {"reference_B": {"checkpoint": {
            "identity": {"model": "test/model", "revision": "test-revision"},
            "metadata_files": [{"path": str(self.model / n), "sha256": hashlib.sha256((self.model / n).read_bytes()).hexdigest()}
                               for n in ("config.json", "model.safetensors.index.json")],
            "header_inventory": {"shards": self.report["shards"]}}}}
        self.reference.write_text(json.dumps(reference))

    def test_first_last_blocks_preserve_every_tile_and_scale_bit(self):
        for bits in (4, 6):
            prefix = f"model.language_model.layers.0.mlp.gate{bits}"
            raw = self.parts[prefix + ".trellis"][2]
            for start in (0, 256):
                with self.subTest(bits=bits, start=start):
                    tensors, meta = extract.extract_projection(self.model, self.report, prefix, start, 128)
                    # Independently select each I16 word in 8 output tiles.
                    expected = b"".join(raw[((row * 24 + col) * 16 * bits + word) * 2:
                                          ((row * 24 + col) * 16 * bits + word) * 2 + 2]
                                        for row in range(8) for col in range(start // 16, start // 16 + 8)
                                        for word in range(16 * bits))
                    self.assertEqual(tensors["trellis"], ("I16", [8, 8, 16 * bits], expected))
                    self.assertEqual(tensors["svh"][2], self.parts[prefix + ".svh"][2][start * 2:(start + 128) * 2])
                    self.assertEqual(meta["source_n"], 384)

    def test_slice_reads_only_selected_packed_rows(self):
        original = extract.read_span
        calls = []
        def read(model, shard, begin, size):
            calls.append(size)
            return original(model, shard, begin, size)
        with mock.patch.object(extract, "read_span", side_effect=read):
            extract.extract_projection(self.model, self.report, "model.language_model.layers.0.mlp.gate6", 256, 128)
        self.assertEqual(calls, [4] + [128 // 16 * 32 * 6] * 8 + [256, 256])

    def test_rejects_partial_blocks_out_of_range_and_wrong_marker(self):
        prefix = "model.language_model.layers.0.mlp.gate4"
        for start, count in ((1, 128), (0, 127), (0, 0), (-128, 128), (384, 128), (False, 128)):
            with self.subTest(start=start, count=count):
                with self.assertRaisesRegex(extract.headers.InventoryError, "Hadamard128"):
                    extract.extract_projection(self.model, self.report, prefix, start, count)
        self.parts[prefix + ".mul1"] = ("I32", [], b"\0" * 4)
        self.write_model()
        with self.assertRaisesRegex(extract.headers.InventoryError, "marker bits"):
            extract.extract_projection(self.model, self.report, prefix, 0, 128)

    def test_export_roundtrip_hashes_and_m1_m4_inputs(self):
        result = extract.export(self.model, self.reference, "model.language_model.layers.0.mlp.gate4", 0, 384, self.output)
        report = extract.headers.read_shard_header(self.output)
        tensors = {t["name"]: t for t in report["tensors"]}
        for name, expected in result["tensor_hashes"].items():
            tensor = tensors[name]
            raw = extract.read_span(self.output.parent, report, tensor["data_offsets"][0], tensor["stored_bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
        self.assertEqual(tensors["activation_m1_fp16"]["shape"], [1, 128])
        self.assertEqual(tensors["activation_m4_fp16"]["shape"], [4, 128])
        self.assertEqual(result["oracle_outputs"]["status"], "pending")
        self.assertFalse(result["full_checkpoint_payload_verified"])
        self.assertEqual(json.loads(self.output.with_suffix(".json").read_text()), result)

    def test_rejects_changed_reference_metadata_revision_and_header(self):
        args = (self.model, self.reference, "model.language_model.layers.0.mlp.gate4", 0, 128, self.output)
        (self.model / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(extract.headers.InventoryError, "metadata hash mismatch"):
            extract.export(*args)
        self.write_model()
        provenance = self.model / ".cache/huggingface/download/model.safetensors.metadata"
        provenance.write_text("wrong-revision\netag\n0\n")
        with self.assertRaisesRegex(extract.headers.InventoryError, "revision mismatch"):
            extract.export(*args)
        self.write_model()
        reference = json.loads(self.reference.read_text())
        reference["reference_B"]["checkpoint"]["header_inventory"]["shards"][0]["header_sha256"] = "wrong"
        self.reference.write_text(json.dumps(reference))
        with self.assertRaisesRegex(extract.headers.InventoryError, "header hash mismatch"):
            extract.export(*args)
        self.assertFalse(self.output.exists())

    def test_never_overwrites_fixture_or_checkpoint(self):
        args = (self.model, self.reference, "model.language_model.layers.0.mlp.gate4", 0, 128)
        self.output.write_bytes(b"keep me")
        with self.assertRaisesRegex(extract.headers.InventoryError, "overwrite"):
            extract.export(*args, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep me")
        with self.assertRaisesRegex(extract.headers.InventoryError, "outside checkpoint"):
            extract.export(*args, self.model / "new.safetensors")


if __name__ == "__main__":
    unittest.main()
