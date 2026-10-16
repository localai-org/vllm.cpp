"""Retain strict token counts while admitting current vLLM usage metadata."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/mm'))
from measure_native_exl3_text_regression import require_usage


class UsageAdmission(unittest.TestCase):
    def test_native_and_vllm_optional_details_preserve_exact_counts(self):
        for details in ({}, {'prompt_tokens_details': None, 'completion_tokens_details': None},
                        {'prompt_tokens_details': {'cached_tokens': 0}}):
            usage = dict(prompt_tokens=4096, completion_tokens=256, total_tokens=4352, **details)
            before = copy.deepcopy(usage)
            require_usage(usage, 4096, 256)
            self.assertEqual(usage, before)

    def test_optional_details_cannot_hide_wrong_or_missing_counts(self):
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            for value in (None, -1, 0, 4353):
                usage = {'prompt_tokens': 4096, 'completion_tokens': 256, 'total_tokens': 4352,
                         'prompt_tokens_details': None}
                if value is None:
                    del usage[key]
                else:
                    usage[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    require_usage(usage, 4096, 256)

    def test_integer_types_and_usage_object_remain_required(self):
        for usage in (None, [], {'prompt_tokens': 4096.0, 'completion_tokens': 256, 'total_tokens': 4352},
                      {'prompt_tokens': True, 'completion_tokens': 256, 'total_tokens': 4352}):
            with self.subTest(usage=usage), self.assertRaises(RuntimeError):
                require_usage(usage, 4096, 256)


if __name__ == '__main__':
    unittest.main()
