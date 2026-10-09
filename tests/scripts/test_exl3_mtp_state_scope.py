#!/usr/bin/env python3
"""Capture admission: accepted snapshots, initialized history and live KV only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from mtp_state_scope import integrated_batch_reads, initialized_kv_addresses, speculative_state_reads
from capture_integrated_mtp import observe_prepared_batch


class MtpStateScopeTest(unittest.TestCase):
    def test_actual_batch_transition_retains_survivor_identity(self):
        for ids, lengths in ((["a", "b", "c", "d"], [1, 4, 2, 3]),
                             (["c", "d"], [2, 4]), (["d"], [3])):
            qsl = [0]
            for length in lengths:
                qsl.append(qsl[-1] + length)
            positions = [p for length in lengths for p in range(132 - length, 132)]
            plans = integrated_batch_reads(ids, [1000] * qsl[-1], [positions] * 3,
                                           qsl, [132] * len(ids))
            self.assertEqual([p["request_id"] for p in plans], ids)
            self.assertEqual([p["end"] - p["begin"] for p in plans], lengths)
            self.assertFalse(any(p["cold_prefill"] for p in plans))

    def test_cold_c4_requires_actual_all_four_prefill_rows(self):
        plans = integrated_batch_reads(["a", "b", "c", "d"], [1000] * 512,
                                      [list(range(128)) * 4] * 3,
                                      [0, 128, 256, 384, 512], [128] * 4)
        self.assertTrue(all(p["cold_prefill"] for p in plans))
        self.assertEqual(plans[-1]["begin"], 384)

    def test_invalid_active_batch_is_rejected(self):
        for ids, tokens, pos, qsl, seq in (
                (["a", "a"], [1000] * 8, [list(range(128, 132)) * 2] * 3, [0, 4, 8], [132, 132]),
                (["a"], [True] * 4, [list(range(128, 132))] * 3, [0, 4], [132]),
                (["a"], [1000] * 4, [list(range(128, 132))] * 3, [0, 3], [132]),
                (["a"], [1000] * 5, [list(range(128, 133))] * 3, [0, 5], [133]),
                (["a"], [1000] * 4, [list(range(128, 132))] * 3, [0, 4], [161]),
                (["a"], [1000] * 4, [list(range(127, 131))] * 3, [0, 4], [132]),
                (["a", "b"], [1000] * 132, [list(range(128)) + list(range(128, 132))] * 3,
                 [0, 128, 132], [128, 132])):
            with self.subTest(ids=ids, qsl=qsl, seq=seq):
                with self.assertRaises(ValueError):
                    integrated_batch_reads(ids, tokens, pos, qsl, seq)

    def test_batch_observer_preserves_call_and_result_identity(self):
        calls, observed = [], []
        result = object()
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        wrapper = observe_prepared_batch(original, observed.append)
        argument = object()
        self.assertIs(wrapper(argument, mode="live"), result)
        self.assertEqual(calls, [((argument,), {"mode": "live"})])
        self.assertEqual(observed, [result])

    def test_batch_observer_does_not_observe_failed_producer(self):
        observed = []
        def original():
            raise RuntimeError("producer failed")
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            observe_prepared_batch(original, observed.append)()
        self.assertEqual(observed, [])

    def test_first_q4_reads_only_completed_prefill_seed(self):
        plan = speculative_state_reads([0, 4], [[7, 8, 9, 10]], [1],
                                       12, {7: 3}, {7})[0]
        self.assertEqual(plan["ssm_before"], 7)
        self.assertEqual(plan["ssm_after"], [7, 8, 9, 10])
        self.assertEqual(plan["conv_before"], [0, 3])
        self.assertEqual(plan["conv_after"], [0, 6])

    def test_rejection_uses_previous_accepted_snapshot_and_history(self):
        plan = speculative_state_reads([0, 4], [[7, 8, 9, 10]], [3],
                                       12, {7: 6}, {7, 8, 9, 10})[0]
        self.assertEqual(plan["ssm_before"], 9)
        self.assertEqual(plan["conv_before"], [2, 5])

    def test_transition_uses_actual_slot_map_and_ragged_lengths(self):
        indices = [[12, 2, 9, 7], [4, 1, 10, 6], [3, 11, 13, 8], [0, 5, 14, 15]]
        for rows in (4, 2, 1):
            qsl = [0, 1, 5, 7, 10][:rows + 1]
            plan = speculative_state_reads(qsl, indices[:rows], [1, 4, 2, 3][:rows],
                                           16, {v[0]: 6 for v in indices}, set(range(16)))
            self.assertEqual(len(plan), rows)
            for row, read in enumerate(plan):
                length = qsl[row + 1] - qsl[row]
                self.assertEqual(read["ssm_after"], indices[row][:length])
                self.assertEqual(read["conv_after"], [0, 2 + length])

    def test_missing_consumed_snapshot_or_conv_history_is_rejected(self):
        for widths, slots in (({0: 6}, {0}), ({0: 3}, {0, 1, 2, 3})):
            with self.assertRaisesRegex(ValueError, "uninitialized"):
                speculative_state_reads([0, 4], [[0, 1, 2, 3]], [2], 4, widths, slots)

    def test_ownership_and_invalid_metadata_rejected(self):
        for qsl, indices, accepted, capacity in (
                ([0, 4, 8], [[0, 1, 2, 3], [3, 4, 5, 6]], [1, 1], 8),
                ([0, 4], [[0, 0, 2, 3]], [1], 4),
                ([0, 4], [[0, 1, 2, 4]], [1], 4),
                ([0, 5], [[0, 1, 2, 3]], [1], 4),
                ([1, 4], [[0, 1, 2, 3]], [1], 4),
                ([0, 4], [[0, 1, 2, 3]], [0], 4),
                ([0, 4], [[0, 1, 2, 3]], [True], 4)):
            with self.subTest(qsl=qsl, indices=indices, accepted=accepted):
                with self.assertRaises(ValueError):
                    speculative_state_reads(qsl, indices, accepted, capacity,
                                            {0: 6}, set(range(8)))

    def test_kv_before_excludes_new_and_previous_rejected_tail(self):
        args = (1600, 20, 132, [128, 129, 130, 131], [7],
                [7 * 1600 + p for p in range(128, 132)])
        before = initialized_kv_addresses(*args, "before")
        after = initialized_kv_addresses(*args, "after")
        self.assertEqual(before, [(7, p) for p in range(128)])
        self.assertEqual(after, [(7, p) for p in range(132)])

    def test_cold_prefill_never_reads_seed_kv(self):
        self.assertEqual(initialized_kv_addresses(1600, 20, 128, list(range(128)),
                         [7], [7 * 1600 + p for p in range(128)], "before"), [])

    def test_page_boundary_uses_only_active_columns(self):
        addresses = initialized_kv_addresses(64, 20, 132, [128, 129, 130, 131],
                                            [7, 3, 11], [704, 705, 706, 707], "after")
        self.assertEqual(addresses[63:65], [(7, 63), (3, 0)])
        self.assertEqual(addresses[-1], (11, 3))
        with self.assertRaisesRegex(ValueError, "pages"):
            initialized_kv_addresses(64, 20, 132, [128, 129, 130, 131],
                                     [7, 3, 11, None], [704, 705, 706, 707], "after")

    def test_unbounded_or_inconsistent_kv_metadata_rejected(self):
        for seq, positions, blocks, slots in ((161, [160], [0], [160]),
                                             (132, [128, 130, 131], [0], [128, 130, 131]),
                                             (132, [128, 129, 130, 131], [0], [0, 1, 2, 3]),
                                             (132, [128, 129, 130, 131], [20], [32128, 32129, 32130, 32131])):
            with self.assertRaises(ValueError):
                initialized_kv_addresses(1600, 20, seq, positions, blocks, slots, "after")


if __name__ == "__main__":
    unittest.main()
