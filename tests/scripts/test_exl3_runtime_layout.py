#!/usr/bin/env python3
"""Host layout checks: strides/offsets, alias storage and bounded metadata reads."""
import dataclasses
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from runtime_layout import attention_call_layout, describe, initialized_values, tensor_layout
from capture_runtime_layout import active_metadata


class Tensor:
    dtype, device = "torch.float16", "xpu:0"
    def __init__(self, shape=(2, 3), strides=(8, 1), offset=4, storage_bytes=64, values=None):
        self.shape, self.strides, self.offset = shape, strides, offset
        self.storage_bytes, self.values, self.reads = storage_bytes, values, 0
    def stride(self): return self.strides
    def storage_offset(self): return self.offset
    def element_size(self): return 2
    def untyped_storage(self):
        class Storage:
            def nbytes(inner): return self.storage_bytes
            def data_ptr(inner): return 4096
        return Storage()
    def data_ptr(self): return 4096 + 2 * self.offset
    def detach(self): return self
    def cpu(self): self.reads += 1; return self
    def tolist(self): return self.values
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)
    def __getitem__(self, index):
        if index != (slice(None), slice(None, 1)):
            raise AssertionError("only the initialized first block column may be read")
        self.first_column = Tensor(shape=(self.shape[0], 1), strides=self.strides,
                                   offset=self.offset, storage_bytes=self.storage_bytes,
                                   values=[[row[0]] for row in self.values])
        return self.first_column


class RuntimeLayoutTest(unittest.TestCase):
    def test_padded_view_span_and_offset_are_not_contiguous_logical_bytes(self):
        t = Tensor()
        r = tensor_layout(t)
        self.assertEqual(r["strides_elements"], [8, 1])
        self.assertEqual(r["logical_bytes"], 12)
        self.assertEqual(r["view_span_bytes"], 22)
        self.assertEqual(r["storage_offset_elements"], 4)
        self.assertEqual(t.reads, 0)

    def test_two_views_record_shared_storage_and_different_offsets(self):
        a, b = tensor_layout(Tensor()), tensor_layout(Tensor(offset=8))
        self.assertEqual(a["storage_data_ptr"], b["storage_data_ptr"])
        self.assertNotEqual(a["data_ptr"], b["data_ptr"])

    def test_scalar_and_empty_views(self):
        self.assertEqual(tensor_layout(Tensor(shape=(), strides=()))["logical_bytes"], 2)
        self.assertEqual(tensor_layout(Tensor(shape=(0, 3)))["view_span_bytes"], 0)

    def test_invalid_views_are_rejected_without_reading_payload(self):
        for t in (Tensor(strides=(-1, 1)), Tensor(offset=-1), Tensor(storage_bytes=29),
                  Tensor(shape=(2,), strides=(1, 2))):
            with self.assertRaises(ValueError): tensor_layout(t)
            self.assertEqual(t.reads, 0)

    def test_pointer_must_agree_with_owner_and_offset(self):
        t = Tensor()
        t.data_ptr = lambda: 1
        with self.assertRaisesRegex(ValueError, "pointer"): tensor_layout(t)

    def test_describing_nested_cache_never_reads_values(self):
        @dataclasses.dataclass
        class Group:
            cache: object
        t = Tensor()
        result = describe({"layers": [Group(t)]})
        self.assertIn("tensor_layout", result["layers"][0]["fields"]["cache"])
        self.assertEqual(t.reads, 0)

    def test_metadata_read_is_bounded_before_copy_and_rejects_nonfinite(self):
        t = Tensor(values=[[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "bound"): initialized_values(t, max_elements=5)
        self.assertEqual(t.reads, 0)
        self.assertEqual(initialized_values(t)["values"], t.values)
        t.values = [[math.nan]]
        with self.assertRaisesRegex(ValueError, "nonfinite"): initialized_values(t)

    def test_actual_attention_views_are_described_without_reading_cache_or_padding(self):
        q, k, v, out = (Tensor() for _ in range(4))
        table = Tensor(values=[[13, math.nan, math.nan], [17, math.nan, math.nan]])
        lengths = Tensor(shape=(2,), strides=(1,), values=[128, 64])
        scale = Tensor(shape=(2, 4), strides=(0, 0), values=[[1] * 4] * 2)
        record = attention_call_layout(dict(q=q, k=k, v=v, out=out, block_table=table,
                                            seqused_k=lengths, k_descale=scale))
        self.assertEqual(record["active_values"]["block_table_first_column"]["values"], [[13], [17]])
        self.assertEqual(record["arguments"]["k_descale"]["tensor_layout"]["strides_elements"], [0, 0])
        self.assertTrue(all(t.reads == 0 for t in (q, k, v, out, table)))
        self.assertEqual(table.first_column.reads, 1)

    def test_actual_attention_metadata_copies_are_bounded(self):
        lengths = Tensor(shape=(4097,), strides=(1,), offset=0, storage_bytes=8194)
        with self.assertRaisesRegex(ValueError, "bound"):
            attention_call_layout(dict(seqused_k=lengths))
        self.assertEqual(lengths.reads, 0)

    def test_active_slot_mapping_is_retained_without_reading_block_capacity(self):
        slots = Tensor(shape=(2,), strides=(1,), values=[20800, 20801])
        table = Tensor(values=[[13, math.nan, math.nan], [17, math.nan, math.nan]])
        result = active_metadata(SimpleNamespace(slot_mapping=slots, block_table=table))
        self.assertEqual(result["active_values"]["slot_mapping"]["values"], [20800, 20801])
        self.assertEqual(table.reads, 0)
        self.assertEqual(table.first_column.reads, 1)


if __name__ == "__main__":
    unittest.main()
