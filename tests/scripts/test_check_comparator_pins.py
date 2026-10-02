#!/usr/bin/env python3
"""Contract tests for the non-correctness comparator registry."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-comparator-pins.py"
SPEC = importlib.util.spec_from_file_location("check_comparator_pins", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_comparator_pins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_comparator_pins
SPEC.loader.exec_module(check_comparator_pins)

GOOD = """# Fixture

```comparator-pin
id = tensorfold
role = implementation-performance
upstream = https://github.com/ashhart/TensorFold
pin = 191188075bca56a7c71074a79375eb4c1cb22e1c
recipe_upstream = https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
recipe_pin = 856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e
scope = implementation organization and performance only
correctness = not-an-oracle
pinned_on = 2026-09-29
```
"""


class ComparatorPinTests(unittest.TestCase):
    def errors_for(self, text: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "tensorfold.md").write_text(text, encoding="utf-8")
            with mock.patch.object(check_comparator_pins, "COMPARATORS", registry):
                return check_comparator_pins.check_registry()

    def test_complete_non_correctness_record_passes(self) -> None:
        self.assertEqual(self.errors_for(GOOD), [])

    def test_missing_recipe_pin_fails(self) -> None:
        errors = self.errors_for(GOOD.replace(
            "recipe_pin = 856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e\n", ""
        ))
        self.assertTrue(any("recipe_pin" in error for error in errors))

    def test_correctness_role_fails(self) -> None:
        errors = self.errors_for(GOOD.replace(
            "correctness = not-an-oracle", "correctness = oracle"
        ))
        self.assertTrue(any("not-an-oracle" in error for error in errors))

    def test_repository_registry_is_clean(self) -> None:
        self.assertEqual(check_comparator_pins.main(), 0)


if __name__ == "__main__":
    unittest.main()
