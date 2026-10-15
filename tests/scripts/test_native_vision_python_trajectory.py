"""Focused admission checks of a supplied complete Python GPU window.

Usage: python -O tests/scripts/test_native_vision_python_trajectory.py CAPTURE
The actual external witness is required; this does not run a GPU or MTP test.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from analyze_native_exl3_vision_python_trajectory import analyze

CAPTURE = Path(sys.argv.pop(1))


class PythonTrajectoryWitness(unittest.TestCase):
    def test_actual_complete_window(self):
        checksum = hashlib.sha256((CAPTURE / 'result.json').read_bytes()).hexdigest()
        result = analyze(CAPTURE, checksum)
        self.assertEqual(result['status'], 'DIAGNOSTIC')
        self.assertEqual(result['window_steps'], 8)
        self.assertEqual(result['full_vocabulary_size'], 248320)

    def negative(self, edit):
        # Copy metadata only; immutable head files are linked read-only unless
        # this specific negative replaces one link with its own private file.
        with tempfile.TemporaryDirectory(prefix='b70-python-window-') as tmp:
            root = Path(tmp)
            for path in CAPTURE.iterdir():
                if path.is_file() and path.name != 'result.json':
                    (root / path.name).symlink_to(path.resolve())
            report = json.loads((CAPTURE / 'result.json').read_text())
            edit(report, root)
            receipt = root / 'result.json'
            receipt.write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):
                analyze(root, hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_recorded_state_chain_negatives(self):
        self.negative(lambda a, _: a['frames']['c1'][0]['incoming'].pop('gdn0-ssm'))
        self.negative(lambda a, _: a['frames']['c1'][1]['incoming'].update({'gdn0-ssm': '0' * 64}))
        self.negative(lambda a, _: a['frames']['c4'][0]['controls']['identical_incoming_single_step']['incoming'].update({'gdn0-ssm': '0' * 64}))

    def test_masked_head_with_fresh_checksum_is_rejected(self):
        def edit(report, root):
            entry = report['frames']['c1_prefill']['head']
            path = root / entry['file']
            raw = bytearray(path.read_bytes())
            struct.pack_into('<e', raw, 0, float('-inf'))
            path.unlink()
            path.write_bytes(raw)
            entry['sha256'] = hashlib.sha256(raw).hexdigest()
        self.negative(edit)


if __name__ == '__main__':
    unittest.main()
