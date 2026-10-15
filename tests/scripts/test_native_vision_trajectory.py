"""No-GPU checks of supplied real trajectory witnesses and rejection semantics.

Usage: python -O tests/scripts/test_native_vision_trajectory.py CAPTURE PLAN RESULT
The external witness is required; absence is an error, never a synthetic pass.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from replay_native_exl3_vision_trajectory import validate_plan, validate_result

CAPTURE, PLAN, RESULT = map(Path, sys.argv[1:4])
del sys.argv[1:4]


class TrajectoryWitness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = validate_plan(CAPTURE, PLAN)
        cls.result = json.loads(RESULT.read_text())
        validate_result(cls.result, cls.plan)

    def test_supplied_full_witness(self):
        self.assertEqual(self.result['plan_sha256'], hashlib.sha256(PLAN.read_bytes()).hexdigest())
        validate_result(self.result, self.plan)

    def test_actual_first_step_state_anchors(self):
        # This supplied alpha packet has independent post-target observations.
        # Recompute the state hashes from bytes, not just metadata assertions.
        for key, name in [('c1_state_sha256', 'prefix-3.json'), ('c4_own_state_sha256', 'prefix-1.json')]:
            with self.subTest(arm=key):
                metadata = json.loads((CAPTURE / name).read_text())
                actual = {}
                for entry in metadata['blobs']:
                    if entry['name'].startswith(('gdn', 'attn')):
                        raw = (CAPTURE / entry['file']).read_bytes()
                        self.assertEqual(len(raw), entry['bytes'])
                        actual[entry['name']] = hashlib.sha256(raw).hexdigest()
                        self.assertEqual(actual[entry['name']], entry['sha256'])
                self.assertEqual(len(actual), 128)
                self.assertEqual(actual, self.result['steps'][0][key])

    def test_result_contract_negatives(self):
        # Supervisor negatives only: do not label these as mutated live MTP
        # selector/state-owner experiments, which require separate actual data.
        mutations = {
            'missing step': lambda x: x['steps'].pop(),
            'wrong forced token': lambda x: x['steps'][0].update(teacher_token=0),
            'wrong logical position': lambda x: x['steps'][0].update(position=0),
            'missing complete state witness': lambda x: x['steps'][0]['c1_state_sha256'].pop('gdn0-ssm'),
            'nonfinite head': lambda x: x['steps'][0]['evolving_shared_seed']['logits'].update(max_abs=float('nan')),
            'missing repeat coverage': lambda x: x['same_geometry_repeat']['steps'].pop(),
            'different moved-slot state': lambda x: x['same_geometry_moved_row_slot_page']['steps'][0]['states'].update(all_storage_exact=False),
            'missing poison check': lambda x: x['inactive_poisoned_bytes_checked'].update(c1=0),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                result = copy.deepcopy(self.result)
                mutate(result)
                with self.assertRaises((RuntimeError, KeyError)):
                    validate_result(result, self.plan)


if __name__ == '__main__':
    unittest.main()
