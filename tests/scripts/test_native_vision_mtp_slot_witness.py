"""Focused selector/owner admission negatives on an actual MTP capture.

Usage: python -O tests/scripts/test_native_vision_mtp_slot_witness.py CAPTURE
These mutate observed metadata in a private copy, not a live GPU selector.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from analyze_native_exl3_vision_adjacent_prefix import PREFIX
from analyze_native_exl3_vision_mtp_prefix import STATE_NAMES, load

CAPTURE = Path(sys.argv.pop(1))


class ActualMtpSlotWitness(unittest.TestCase):
    def test_supplied_committed_owner(self):
        record, blobs, _ = load(CAPTURE, 'prefix-0-commit.json', packet_prefix=PREFIX)
        self.assertEqual(record['mtp']['accepted_drafts'], 1)
        self.assertEqual(len(STATE_NAMES), 128)
        self.assertTrue(STATE_NAMES <= blobs.keys())

    def negative(self, filename, edit, anchor=False, message='wrong selected committed state slot/window'):
        with tempfile.TemporaryDirectory(prefix='b70-mtp-slot-') as tmp:
            root = Path(tmp)
            for path in CAPTURE.iterdir():
                if path.is_file():
                    (root / path.name).symlink_to(path.resolve())
            path = root / filename
            record = json.loads(path.read_text())
            edit(record)
            path.unlink()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, message):
                load(root, filename, packet_prefix=PREFIX, target_anchor=anchor)

    def test_wrong_selected_ssm_slot(self):
        self.negative('prefix-0-commit.json', lambda a: a['mtp'].update(selected_ssm_slot=a['mtp']['owner_slot']))

    def test_wrong_owner_pair(self):
        def edit(record):
            mtp = record['mtp']
            # A distinct valid owner slot, not an out-of-range synthetic index.
            masks = mtp['spec_sequence_masks']
            spec_row = sum(masks[:record['row']])
            slot = mtp['spec_state_indices'][spec_row * 4]
            record['gdn_slot'] = mtp['selected_ssm_slot'] = slot
        self.negative('prefix-0-commit.json', edit)

    def test_wrong_initialized_kv_page(self):
        def edit(record):
            row, cols = record['row'], record['kv_block_table_cols']
            record['kv_block_table'][row * cols] += 1
        self.negative('prefix-0-commit.json', edit, message='wrong KV write positions')

    def test_wrong_predicted_selector(self):
        self.negative('prefix-2-target-anchor.json',
                      lambda a: a['mtp'].update(predicted_accepted_drafts=1), anchor=True)


if __name__ == '__main__':
    unittest.main()
