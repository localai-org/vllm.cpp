"""EOS/count guards for the maximum-image warm-owner qualification."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from native_vision_completion import completion_length, committed_ids


def response(count=54, finish='stop'):
    return {'usage': {'prompt_tokens': 4149, 'completion_tokens': count,
                      'total_tokens': 4149 + count},
            'choices': [{'index': 0, 'finish_reason': finish}]}


class CompletionAccounting(unittest.TestCase):
    def test_real_recipe_eos_length_and_mtp_boundary(self):
        row = response()
        ids = [13] * 53 + [248046]
        self.assertEqual(completion_length(row, 4149), 54)
        self.assertEqual(committed_ids(row, 4149, ids + [22, 23, 24]), ids)
        row['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(RuntimeError):
            completion_length(row, 4149)

    def test_maximum_quota_length_is_still_required(self):
        self.assertEqual(len(committed_ids(response(64, 'length'), 4149, [13] * 64)), 64)
        for count in (0, -1, 65, 63):
            with self.subTest(count=count), self.assertRaises(RuntimeError):
                completion_length(response(count, 'length'), 4149)

    def test_wrong_usage_never_passes_as_regular_eos(self):
        for key, value in [('prompt_tokens', 4148), ('completion_tokens', True),
                           ('total_tokens', 4202), ('total_tokens', 4203.0)]:
            row = response()
            row['usage'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                completion_length(row, 4149)

    def test_stop_requires_actual_eos_at_the_public_boundary(self):
        for ids in ([13] * 54, [248046] + [13] * 52 + [248046],
                    [13] * 52 + [248046], [13] * 53 + [248046] + [13] * 4):
            with self.subTest(count=len(ids)), self.assertRaises(RuntimeError):
                committed_ids(response(), 4149, ids)

    def test_missing_or_unknown_finish_and_extra_choice_fail(self):
        for edit in ('finish', 'choice', 'usage'):
            row = copy.deepcopy(response())
            if edit == 'finish':
                row['choices'][0]['finish_reason'] = 'cancelled'
            elif edit == 'choice':
                row['choices'].append(copy.deepcopy(row['choices'][0]))
            else:
                row['usage']['extra'] = 1
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                completion_length(row, 4149)


if __name__ == '__main__':
    unittest.main()
