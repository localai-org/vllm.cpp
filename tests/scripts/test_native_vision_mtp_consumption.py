"""Focused actual-trace checks; no GPU, logits or live selector mutation.

Usage: python -O tests/scripts/test_native_vision_mtp_consumption.py RECEIPT
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from analyze_native_exl3_vision_mtp_consumption import analyze

RECEIPT = Path(sys.argv.pop(1))


class ActualConsumptionWitness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(RECEIPT.read_text())
        cls.target_id = cls.report['responses'][0]['response']['id']
        cls.indices = [i for i, entry in enumerate(cls.report['trace'])
                       if entry['request_id'] in (cls.target_id, cls.target_id + '-0') and 'target_consumption' in entry]
        if len(cls.indices) < 3:
            raise RuntimeError('three actual target packets required')

    def test_actual_image_and_text_transitions(self):
        result = analyze(self.report)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual({x['label'] for x in result['requests']}, {'alpha', 'orbit'})
        self.assertEqual(result['requests'][0]['actual_accepted_lengths'], [0, 1, 2, 3])

    def test_wrong_consumption_negatives(self):
        def target(report):
            return report['trace'][self.indices[1]]
        mutations = {
            'consumed bonus counted as already computed': lambda a: target(a)['target_consumption'].update(
                computed_before=target(a)['target_consumption']['computed_before'] + 1),
            'duplicate logical position': lambda a: target(a)['target_consumption']['positions'].__setitem__(1,
                target(a)['target_consumption']['positions'][0]),
            'wrong incoming bonus token': lambda a: target(a)['target_consumption']['input_ids'].__setitem__(0, -1),
            'wrong previous state selector': lambda a: target(a)['target_consumption'].update(previous_num_sampled=0),
            'missing intermediate packet': lambda a: a['trace'].pop(self.indices[1]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = copy.deepcopy(self.report)
                mutate(report)
                with self.assertRaises(RuntimeError):
                    analyze(report)


if __name__ == '__main__':
    unittest.main()
