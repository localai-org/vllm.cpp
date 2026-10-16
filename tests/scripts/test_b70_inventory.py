#!/usr/bin/env python3
"""Focused PR00 tests; synthetic shards never require a model runtime."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "b70_inventory", ROOT / "tools/b70_inventory.py"
)
inv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inv)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.embedding = "model.language_model.embed_tokens.weight"
        self.text = "model.language_model.layers.0.input_layernorm.weight"
        self.head = "lm_head.mul1"
        self.vision = "model.visual.patch_embed.proj.weight"
        self.mtp = "mtp.norm.weight"
        self.headers = {
            "first.safetensors": {
                self.embedding: self.tensor("BF16", [2, 3], 0, 12),
                self.text: self.tensor("BF16", [3], 12, 18),
                self.head: self.tensor("I32", [], 18, 22),
            },
            "second.safetensors": {
                self.vision: self.tensor("F16", [2], 0, 4),
                self.mtp: self.tensor("BF16", [3], 4, 10),
            },
        }
        self.index = {
            "metadata": {"total_size": 32},
            "weight_map": {
                name: shard
                for shard, header in self.headers.items()
                for name in header
            },
        }
        self.write_fixture()

    @staticmethod
    def tensor(dtype, shape, begin, end):
        return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end]}

    def write_shard(self, shard, header, payload_size):
        raw = header if isinstance(header, bytes) else json.dumps(header).encode()
        (self.model / shard).write_bytes(struct.pack("<Q", len(raw)) + raw
                                        + bytes(payload_size))

    def write_fixture(self):
        for shard, header in self.headers.items():
            self.write_shard(shard, header, 22 if shard == "first.safetensors" else 10)
        (self.model / "model.safetensors.index.json").write_text(json.dumps(self.index))

    def test_two_shards_classification_and_stable_order(self):
        report = inv.inventory(self.model)
        self.assertEqual(report["tensor_count"], 5)
        self.assertEqual(report["stored_bytes"], 32)
        self.assertEqual(report["errors"], [])
        expected = {"embedding": 12, "text": 6, "head": 4, "vision": 4, "mtp": 6,
                    "unclassified": 0}
        self.assertEqual({k: v["stored_bytes"] for k, v in report["groups"].items()},
                         expected)
        rows = report["tensors"]
        self.assertEqual([t["name"] for t in rows], sorted(self.index["weight_map"]))
        self.assertEqual(rows[0]["exl3_component"], "mul1")
        self.assertEqual(report["groups"]["mtp"]["dtypes"], {"BF16": 1})
        self.headers["first.safetensors"] = dict(
            reversed(list(self.headers["first.safetensors"].items())))
        self.write_fixture()
        # Header digests may change; tensor ordering and totals must not.
        self.assertEqual(inv.inventory(self.model)["tensors"], rows)

    def test_header_reads_never_reach_tensor_payload(self):
        path = self.model / "first.safetensors"
        raw = path.read_bytes()
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        reads = []

        class HeaderOnly(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                if size < 0 or self.tell() + size > header_end:
                    raise AssertionError("attempted to read tensor payload")
                return super().read(size)

        with mock.patch.object(Path, "open", return_value=HeaderOnly(raw)) as opened:
            inv.read_shard_header(path)
        opened.assert_called_once_with("rb", buffering=0)
        self.assertEqual(reads, [8, header_end - 8])

    def test_rejects_duplicate_header_and_index_keys(self):
        raw = json.dumps(self.headers["first.safetensors"])
        duplicate = json.dumps(self.embedding) + ":" + json.dumps(
            self.headers["first.safetensors"][self.embedding])
        self.write_shard("first.safetensors", (raw[:-1] + "," + duplicate + "}").encode(), 22)
        with self.assertRaisesRegex(inv.InventoryError, "duplicate.*embed_tokens"):
            inv.inventory(self.model)
        self.write_fixture()
        (self.model / "model.safetensors.index.json").write_text(
            '{"weight_map": {}, "weight_map": {}}')
        with self.assertRaisesRegex(inv.InventoryError, "duplicate.*weight_map"):
            inv.inventory(self.model)

    def test_rejects_duplicate_names_across_shards(self):
        self.headers["second.safetensors"][self.embedding] = self.tensor("BF16", [0], 10, 10)
        self.write_fixture()
        with self.assertRaisesRegex(inv.InventoryError, "multiple shards"):
            inv.inventory(self.model)

    def test_index_header_equality_in_both_directions(self):
        entry = self.headers["first.safetensors"][self.embedding]
        for remove_from in ("index", "header"):
            with self.subTest(remove_from=remove_from):
                self.index["weight_map"][self.embedding] = "first.safetensors"
                self.headers["first.safetensors"][self.embedding] = entry
                if remove_from == "index":
                    del self.index["weight_map"][self.embedding]
                else:
                    del self.headers["first.safetensors"][self.embedding]
                self.write_fixture()
                with self.assertRaisesRegex(inv.InventoryError, "index/header names.*embed_tokens"):
                    inv.inventory(self.model)

    def test_rejects_wrong_shard_assignment(self):
        self.index["weight_map"][self.embedding] = "second.safetensors"
        self.write_fixture()
        with self.assertRaisesRegex(inv.InventoryError, "wrong shard.*embed_tokens"):
            inv.inventory(self.model)

    def test_rejects_missing_shard(self):
        (self.model / "second.safetensors").unlink()
        with self.assertRaisesRegex(inv.InventoryError, "second.safetensors"):
            inv.inventory(self.model)

    def test_rejects_bad_header_lengths(self):
        for raw in (b"short", struct.pack("<Q", 999) + b"{}"):
            with self.subTest(raw=raw):
                (self.model / "first.safetensors").write_bytes(raw)
                with self.assertRaisesRegex(inv.InventoryError, "header"):
                    inv.inventory(self.model)
        self.write_fixture()
        with mock.patch.object(inv, "MAX_HEADER_BYTES", 16):
            with self.assertRaisesRegex(inv.InventoryError, "header.*limit"):
                inv.inventory(self.model)

    def test_rejects_malformed_json_and_metadata(self):
        for raw in (b'{', b'[]', b'{"__metadata__": null}',
                    b'{"__metadata__": {"format": 1}}',
                    b'{"__metadata__": {"format": NaN}}',
                    b'{"x": {"dtype": "F16", "dtype": "I16"}}'):
            with self.subTest(raw=raw):
                self.write_shard("first.safetensors", raw, 22)
                with self.assertRaises(inv.InventoryError):
                    inv.inventory(self.model)

    def test_rejects_missing_or_invalid_index_fields(self):
        for index in ({}, {"weight_map": []},
                      {**self.index, "metadata": {}},
                      {**self.index, "metadata": {"total_size": True}}):
            with self.subTest(index=index):
                (self.model / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaises(inv.InventoryError):
                    inv.inventory(self.model)

    def test_rejects_bad_tensor_fields(self):
        cases = [
            ({"data_offsets": [-1, 11]}, "offsets"),
            ({"data_offsets": [12, 0]}, "offsets"),
            ({"data_offsets": [0, 24]}, "offsets"),
            ({"data_offsets": [0, True]}, "offsets"),
            ({"data_offsets": [0]}, "offsets"),
            ({"shape": [7]}, "byte size"),
            ({"shape": [-1]}, "shape"),
            ({"shape": [1.5]}, "shape"),
            ({"shape": [True]}, "shape"),
            ({"shape": [2**63]}, "shape"),
            ({"shape": [2**62, 4]}, "overflows"),
            ({"shape": [2**62, 2]}, "overflows"),
            ({"dtype": "mystery"}, "dtype"),
            ({"dtype": []}, "dtype"),
        ]
        for fields, message in cases:
            with self.subTest(fields=fields):
                self.headers["first.safetensors"][self.embedding] = {
                    **self.tensor("BF16", [2, 3], 0, 12), **fields}
                self.write_fixture()
                with self.assertRaisesRegex(inv.InventoryError, message):
                    inv.inventory(self.model)

    def test_rejects_overlapping_spans(self):
        self.headers["first.safetensors"][self.text]["data_offsets"] = [10, 16]
        self.write_fixture()
        with self.assertRaisesRegex(inv.InventoryError, "overlap"):
            inv.inventory(self.model)

    def test_reports_padding_and_allows_zero_sized_tensor(self):
        self.headers["first.safetensors"][self.text] = self.tensor("BF16", [0], 12, 12)
        self.index["metadata"]["total_size"] -= 6
        self.write_fixture()
        report = inv.inventory(self.model)
        self.assertEqual(report["shards"][0]["unclaimed_data_bytes"], 6)
        self.assertEqual(report["stored_bytes"], 26)

    def test_rejects_index_total_mismatch(self):
        self.index["metadata"]["total_size"] += 1
        self.write_fixture()
        with self.assertRaisesRegex(inv.InventoryError, "total_size"):
            inv.inventory(self.model)

    def test_rejects_unsafe_shard_paths(self):
        for name in ("../outside.safetensors", "/tmp/outside.safetensors", ""):
            with self.subTest(name=name):
                self.index["weight_map"][self.embedding] = name
                self.write_fixture()
                with self.assertRaisesRegex(inv.InventoryError, "shard name"):
                    inv.inventory(self.model)

    def test_unknown_names_are_reported_and_cli_fails(self):
        self.headers["second.safetensors"]["unknown.weight"] = self.tensor("F16", [0], 10, 10)
        self.index["weight_map"]["unknown.weight"] = "second.safetensors"
        self.write_fixture()
        report = inv.inventory(self.model)
        self.assertEqual(report["groups"]["unclassified"]["tensor_count"], 1)
        self.assertIn("unknown.weight", report["errors"][0])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = inv.main([str(self.model), "--headers-only"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["errors"], report["errors"])
        self.assertIn("unknown.weight", err.getvalue())


class Exl3InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.header, storage = {}, {}
        offset = 0
        for prefix, bits in (("lm_head", 6), ("mtp.fc", 4)):
            stored = {}
            for part, dtype, torch_dtype, shape in (
                ("suh", "F16", "torch.float16", [128]),
                ("svh", "F16", "torch.float16", [256]),
                ("mul1", "I32", "torch.int32", []),
                ("trellis", "I16", "torch.int16", [8, 16, 16 * bits]),
            ):
                size = inv.DTYPE_BYTES[dtype]
                for dim in shape:
                    size *= dim
                name = prefix + "." + part
                self.header[name] = InventoryTest.tensor(dtype, shape, offset, offset + size)
                stored[name] = {"shape": shape, "dtype": torch_dtype, "n_bytes": size}
                offset += size
            if prefix == "lm_head":
                storage[prefix] = {"quant_format": "exl3", "bits_per_weight": bits,
                                   "mul1_multiplier": 2212286765, "stored_tensors": stored}
        self.size = offset
        self.quant = {"quant_method": "exl3", "codebook": "mul1", "tensor_storage": storage}
        (self.model / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "exl3", "codebook": "mul1"}}))
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": offset},
            "weight_map": {n: "model.safetensors" for n in self.header}}))
        self.write()

    def write(self):
        raw = json.dumps(self.header).encode()
        (self.model / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(self.size))
        (self.model / "quantization_config.json").write_text(json.dumps(self.quant))

    def test_matrix_dimensions_mtp_and_unlisted_tensors(self):
        report = inv.checkpoint_inventory(self.model)
        self.assertEqual([(m["k"], m["n"], m["bits"]) for m in report["matrices"]],
                         [(128, 256, 6), (128, 256, 4)])
        self.assertEqual(report["text_matrix_bit_counts"], {"6": 1})
        self.assertEqual(report["tensors_absent_from_quantization_metadata"],
                         sorted(n for n in self.header if n.startswith("mtp.")))
        self.assertEqual(report["plan_comparison"]["actual_text_head_matrices"], 1)
        self.assertEqual(len(report["findings"]), 2)
        self.assertFalse(report["loader_plan"]["vision"])
        self.assertFalse(report["loader_plan"]["mtp"])

    def test_rejects_metadata_bit_and_shape_disagreement(self):
        self.quant["tensor_storage"]["lm_head"]["bits_per_weight"] = 4
        self.write()
        with self.assertRaisesRegex(inv.InventoryError, "metadata mismatch"):
            inv.checkpoint_inventory(self.model)
        self.quant["tensor_storage"]["lm_head"]["bits_per_weight"] = 6
        self.quant["tensor_storage"]["lm_head"]["stored_tensors"]["lm_head.suh"]["shape"] = [64]
        self.write()
        with self.assertRaisesRegex(inv.InventoryError, "tensor_storage/header disagreement"):
            inv.checkpoint_inventory(self.model)

    def test_rejects_bad_mtp_scale_and_marker(self):
        self.header["mtp.fc.suh"]["shape"] = [64, 2]
        self.write()
        with self.assertRaisesRegex(inv.InventoryError, "scale shape/dtype"):
            inv.checkpoint_inventory(self.model)
        self.header["mtp.fc.suh"]["shape"] = [128]
        self.header["mtp.fc.mul1"]["dtype"] = "F32"
        self.write()
        with self.assertRaisesRegex(inv.InventoryError, "I32 scalar"):
            inv.checkpoint_inventory(self.model)

    def test_microbench_cli_contains_only_text_and_head(self):
        with tempfile.TemporaryDirectory() as dest:
            output = Path(dest) / "microbench.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(inv.main([str(self.model), "--microbench-output", str(output)]), 0)
            matrices = json.loads(output.read_text())["matrices"]
        self.assertEqual([m["name"] for m in matrices], ["lm_head"])

    def test_microbench_cannot_overwrite_checkpoint(self):
        output = self.model / "config.json"
        original = output.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(inv.main([str(self.model), "--microbench-output", str(output)]), 1)
        self.assertEqual(output.read_bytes(), original)

    def test_rejects_malformed_storage_dtype(self):
        self.quant["tensor_storage"]["lm_head"]["stored_tensors"]["lm_head.suh"]["dtype"] = []
        self.write()
        with self.assertRaisesRegex(inv.InventoryError, "invalid tensor_storage descriptor"):
            inv.checkpoint_inventory(self.model)


if __name__ == "__main__":
    unittest.main()
