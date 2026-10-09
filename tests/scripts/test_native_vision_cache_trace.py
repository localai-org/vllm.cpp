"""Focused cache-reader guards for the actual INFO-prefixed idle regression."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from native_vision_cache_trace import cache_record
from capture_native_exl3_vision_max_eviction import CacheTrace
from capture_native_exl3_vision_cancel import Trace as CancelTrace
from capture_native_exl3_vision_mixed_cancel import Trace as MixedCancelTrace


class CacheObservation(unittest.TestCase):
    def test_observed_info_prefix_preserves_complete_payload(self):
        record = {'event': 'idle', 'backend_allocated_bytes': 27766872048,
                  'scratch_pool_live_blocks': 6, 'scratch_pool_misses': 109}
        for prefix in ('', 'INFO '):
            line = prefix + 'NATIVE_VISION_CACHE ' + json.dumps(record) + '\n'
            self.assertEqual(cache_record(line), record)
            self.assertEqual(cache_record(line.encode()), record)

    def test_incremental_partial_line_is_not_counted_or_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.log'
            path.write_bytes(b'INFO NATIVE_VISION_CACHE {"event":"idle"')
            reader = CacheTrace(path)
            try:
                self.assertEqual(reader.read(), [])
                with path.open('ab') as out:
                    out.write(b'}\n')
                self.assertEqual(reader.read(), [{'event': 'idle'}])
                self.assertEqual(reader.read(), [{'event': 'idle'}])
                self.assertEqual(len(reader.records), 1)
            finally:
                reader.close()

    def test_cancel_readers_preserve_the_same_idle_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.log'
            path.write_text('INFO NATIVE_VISION_CACHE {"event":"idle","entries":2}\n')
            for reader_type in (CancelTrace, MixedCancelTrace):
                reader = reader_type(path)
                try:
                    reader.read()
                    reader.read()
                    self.assertEqual(reader.idle, [{'event': 'idle', 'entries': 2}])
                finally:
                    reader.file.close()

    def test_arbitrary_marker_inside_log_text_is_not_admitted(self):
        for prefix in ('DEBUG ', 'unrelated ', 'INFO unrelated ', 'INFO INFO '):
            self.assertIsNone(cache_record(prefix + 'NATIVE_VISION_CACHE {"event":"idle"}\n'))

    def test_malformed_and_non_cache_records_fail_closed(self):
        for payload in ('{', 'null', '[]', '{}', '{"event":"unknown"}'):
            with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                cache_record('INFO NATIVE_VISION_CACHE ' + payload + '\n')

    def test_duplicate_events_remain_visible_to_count_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.log'
            path.write_text('NATIVE_VISION_CACHE {"event":"idle"}\n'
                            'INFO NATIVE_VISION_CACHE {"event":"idle"}\n')
            reader = CacheTrace(path)
            try:
                self.assertEqual(len(reader.read()), 2)
                self.assertEqual(len(reader.records), 2)
            finally:
                reader.close()


if __name__ == '__main__':
    unittest.main()
