"""Host negative guards for the explicit six-request warm-owner contract."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from native_vision_warm_owners import WARM_FIELDS, check_warm_owners, check_warm_sequence


class WarmOwnerContract(unittest.TestCase):
    def observations(self):
        # Constructed counters test admission, not an actual GPU qualification.
        row = {key: 0 for key in WARM_FIELDS}
        row.update(backend_allocated_bytes=10000, scratch_pool_retained_bytes=1000,
                   scratch_pool_misses=8, scratch_pool_live_blocks=5,
                   backend_graph_count=2, backend_graph_device_bytes=128,
                   entries=1, embedding_bytes=512)
        rows = [deepcopy(row) for _ in range(6)]
        rows[0]['backend_allocated_bytes'] -= 100
        rows[0]['scratch_pool_retained_bytes'] -= 100  # First-use free-pool growth is declared.
        return rows

    def test_three_additional_observations_required(self):
        rows = self.observations()
        self.assertEqual(check_warm_owners(rows)['additional_same_shape_requests'], 3)
        with self.assertRaisesRegex(RuntimeError, 'three additional'):
            check_warm_owners(rows[:3])

    def test_growing_free_pool_fails_even_when_active_bytes_are_constant(self):
        rows = self.observations()
        rows[3]['backend_allocated_bytes'] += 64
        rows[3]['scratch_pool_retained_bytes'] += 64
        # Later records plateau again: checking only the final two must not pass.
        with self.assertRaisesRegex(RuntimeError, 'counter changed'):
            check_warm_owners(rows)

    def test_every_declared_owner_and_workspace_must_remain_stable(self):
        for key in WARM_FIELDS:
            with self.subTest(key=key):
                rows = self.observations()
                rows[4][key] += 1
                with self.assertRaisesRegex(RuntimeError, 'counter changed'):
                    check_warm_owners(rows)

    def test_missing_or_invalid_counter_fails_closed(self):
        for value in (None, -1, True, 1.0):
            with self.subTest(value=value):
                rows = self.observations()
                rows[-1]['scratch_pool_live_blocks'] = value
                with self.assertRaisesRegex(RuntimeError, 'invalid warm'):
                    check_warm_owners(rows)
        rows = self.observations()
        del rows[-1]['backend_sampling_workspace_bytes']
        with self.assertRaisesRegex(RuntimeError, 'invalid warm'):
            check_warm_owners(rows)

    def test_sequence_compares_like_phases_not_different_grids(self):
        small = self.observations()[-1]
        large = deepcopy(small)
        large['backend_allocated_bytes'] += 10000
        large['embedding_bytes'] += 5000
        rows = [deepcopy(row) for row in [large, small] * 5]
        result = check_warm_sequence(rows, 2, 2)
        self.assertEqual(result['additional_same_sequence_repetitions'], 3)
        with self.assertRaisesRegex(RuntimeError, 'three additional'):
            check_warm_sequence(rows[:-2], 2, 2)
        rows = deepcopy(rows)
        # Grow only the first phase of an intermediate warm wave, then recover.
        rows[4]['backend_allocated_bytes'] += 64
        rows[4]['scratch_pool_retained_bytes'] += 64
        with self.assertRaisesRegex(RuntimeError, 'counter changed.*phase 0'):
            check_warm_sequence(rows, 2, 2)


if __name__ == '__main__':
    unittest.main()
