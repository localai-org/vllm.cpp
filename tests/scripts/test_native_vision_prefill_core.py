"""Validate supplied real first-GDN witnesses without a GPU or NumPy.

Usage: python -O tests/scripts/test_native_vision_prefill_core.py RECEIPTS
Missing external evidence is an error; this is not an optional synthetic pass.
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(sys.argv[1])
del sys.argv[1]


def blob(root, entry):
    path = root / entry['file']
    if Path(entry['file']).name != entry['file']:
        raise RuntimeError('unsafe payload path')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise RuntimeError('payload identity changed')
    if 'bytes' in entry and len(raw) != entry['bytes']:
        raise RuntimeError('payload size changed')
    return raw


class PrefillCoreWitness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = json.loads((ROOT / 'n3-first-gdn-prefill-v1-result.json').read_text())
        cls.observed = json.loads((ROOT / 'n3-first-gdn-prefill-observed-v1-result.json').read_text())
        cls.reference = json.loads((ROOT / 'n3-python-prefill-core-v3/result.json').read_text())
        for value in (cls.native, cls.observed, cls.reference):
            if value['status'] != 'DIAGNOSTIC':
                raise RuntimeError('incomplete supplied witness')

    def test_actual_zero_state_prefill_anchors_and_identical_input(self):
        normalized = []
        for arm, case in enumerate(self.native['cases']):
            md = json.loads((ROOT / f'n3-image-checkpoint-v2-capture/prefix-{arm}.json').read_text())
            self.assertEqual(md['gdn_indices'][md['row']], case['selected_slot'])
            self.assertEqual(md['output_prefix'], [])
            self.assertEqual(case['selected_initial_state'], 'zero')
            self.assertEqual(case['actual_token_rows'], md['actual_token_rows'])
            normalized.append(blob(ROOT, case['normalized']))
            for label in ('conv', 'ssm'):
                actual = next(x for x in md['blobs'] if x['name'] == 'gdn0-' + label)
                self.assertEqual(blob(ROOT, case[label]), blob(ROOT / 'n3-image-checkpoint-v2-capture', actual))
        self.assertEqual(len(normalized), 2)
        self.assertEqual(len(normalized[0]), 224 * 5120 * 2)
        self.assertEqual(normalized[0], normalized[1])

    def test_observer_preserves_unobserved_execution(self):
        self.assertEqual(len(self.observed['cases']), 2)
        for plain, observed in zip(self.native['cases'], self.observed['cases']):
            for label in ('normalized', 'conv', 'ssm', 'first_image_query_output'):
                self.assertEqual(blob(ROOT, plain[label]), blob(ROOT, observed[label]))

    def test_installed_reference_core_geometry_and_native_disposition(self):
        root = ROOT / 'n3-python-prefill-core-v3'
        self.assertEqual(len(self.reference['cases']), 4)
        original = self.reference['cases'][0]
        for case in self.reference['cases'][1:]:
            for name in ('conv', 'ssm', 'core'):
                self.assertEqual(blob(root, original['payloads'][name]), blob(root, case['payloads'][name]))
        for name in ('conv', 'ssm'):
            self.assertEqual(blob(root, original['payloads'][name]), blob(ROOT, self.native['cases'][0][name]))
        # Native mixed Conv matches; mixed SSM/core retain a diagnosed mismatch.
        c4 = self.reference['cases'][1]
        self.assertEqual(blob(root, c4['payloads']['conv']), blob(ROOT, self.native['cases'][1]['conv']))
        self.assertNotEqual(blob(root, c4['payloads']['ssm']), blob(ROOT, self.native['cases'][1]['ssm']))
        manifest = (ROOT / 'n3-first-gdn-prefill-observed-v1/manifest.tsv').read_text().splitlines()
        for arm in (0, 1):
            line = next(x for x in manifest if x.split('\t')[:3] == [str(arm), '0', 'gdn_core'])
            fields = line.split('\t')
            raw = (ROOT / 'n3-first-gdn-prefill-observed-v1' / fields[-1]).read_bytes()
            self.assertEqual(len(raw), int(fields[-2]))
            begin = 0 if arm == 0 else 282 * 6144 * 2
            selected = raw[begin:begin + 224 * 6144 * 2]
            reference = blob(root, self.reference['cases'][arm]['payloads']['core'])
            (self.assertEqual if arm == 0 else self.assertNotEqual)(selected, reference)

    def test_computed_repair_matches_complete_reference_core_and_states(self):
        repaired = json.loads((ROOT / 'n3-gdn-mixed-producer-v2-result.json').read_text())
        observed = json.loads((ROOT / 'n3-gdn-mixed-producer-observed-v2-result.json').read_text())
        for result in (repaired, observed):
            self.assertEqual(result['status'], 'DIAGNOSTIC')
            self.assertTrue(result['computed_mixed_repair_control'])
            self.assertEqual(len(result['cases']), 2)
        root = ROOT / 'n3-gdn-mixed-producer-observed-v2'
        manifest = (root / 'manifest.tsv').read_text().splitlines()
        for arm in range(2):
            for label in ('normalized', 'conv', 'ssm', 'first_image_query_output'):
                self.assertEqual(blob(ROOT, repaired['cases'][arm][label]), blob(ROOT, observed['cases'][arm][label]))
            for label in ('conv', 'ssm'):
                self.assertEqual(blob(ROOT, repaired['cases'][arm][label]),
                                 blob(ROOT / 'n3-python-prefill-core-v3', self.reference['cases'][arm]['payloads'][label]))
            fields = next(x.split('\t') for x in manifest if x.split('\t')[:3] == [str(arm), '0', 'gdn_core'])
            raw = (root / fields[-1]).read_bytes()
            self.assertEqual(len(raw), int(fields[-2]))
            start = 0 if arm == 0 else 282 * 6144 * 2
            self.assertEqual(raw[start:start + 224 * 6144 * 2],
                             blob(ROOT / 'n3-python-prefill-core-v3', self.reference['cases'][arm]['payloads']['core']))
        # The old mixed anchor remains different; do not rewrite its history.
        self.assertFalse(repaired['cases'][1]['actual_ssm']['storage_exact'])


if __name__ == '__main__':
    unittest.main()
