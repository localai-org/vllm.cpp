import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(root / 'exl3_reference'))
spec = importlib.util.spec_from_file_location('capture_verify_batch', root / 'exl3_reference/capture_verify_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class VerifyBatchTests(unittest.TestCase):
    def test_requests_have_disjoint_cache_addresses_and_mutation_is_local(self):
        keys = b''.join(bytes([i % 251]) * 1024 for i in range(4100))
        values = b''.join(bytes([(i + 13) % 251]) * 1024 for i in range(4100))
        for name, layout in batch.LAYOUTS.items():
            raw = batch.cache_bytes(keys, values, name, poison=0x7f, mutate_request=2)
            page, row, head, _ = layout['strides']
            blocks = [b for bs in batch.REQUEST_BLOCKS for b in bs]
            self.assertEqual(len(blocks), len(set(blocks)))
            for request, length in enumerate(batch.LENGTHS):
                for r in (0, length - 4, length - 3, length - 1):
                    for h in range(4):
                        offset = batch.REQUEST_BLOCKS[request][r // 1600] * page + (r % 1600) * row + h * head
                        source = (r + request * 337) % 4100
                        self.assertEqual(raw[offset:offset + 256], bytes([source % 251]) * 256)
                        expected = 0x38 if request == 2 and r >= length - 3 else (source + 13) % 251
                        self.assertEqual(raw[offset + layout['v_offset']:offset + layout['v_offset'] + 256], bytes([expected]) * 256)
                last_block = batch.REQUEST_BLOCKS[request][-1]
                inactive = last_block * page + (length % 1600) * row
                self.assertEqual(raw[inactive], 0x7f)
            self.assertEqual(raw[1 * page], 0x7f)  # Entire unreferenced page.
            if name == 'padded_interleaved':
                for b in blocks:
                    self.assertEqual(raw[b * page + 1600 * row], 0x7f)

    def test_permutation_preserves_request_queries_and_tables(self):
        base = b''.join(bytes([r]) * 12288 for r in range(4))
        cases = batch.batch_cases()
        self.assertEqual(len(cases), 9)
        for case in cases:
            for request, table in zip(case['request_ids'], batch.table_rows(case['request_ids'])):
                self.assertEqual(table[:len(batch.REQUEST_BLOCKS[request])], batch.REQUEST_BLOCKS[request])
                self.assertTrue(all(b == -1 for b in table[len(batch.REQUEST_BLOCKS[request]):]))
                q = batch.query_bytes(base, request)
                self.assertEqual([q[i * 12288] for i in range(4)], [(i + request) % 4 for i in range(4)])
            self.assertEqual(case['lengths'], [batch.LENGTHS[r] for r in case['request_ids']])
        with self.assertRaises(ValueError):
            batch.table_rows([0, 1, 2, 2])
        with self.assertRaises(ValueError):
            batch.cache_bytes(b'bad', b'bad', 'interleaved')


if __name__ == '__main__':
    unittest.main()
