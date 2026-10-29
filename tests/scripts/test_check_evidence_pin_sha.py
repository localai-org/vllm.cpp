#!/usr/bin/env python3
"""Unit checks for scripts/check-evidence-pin-sha.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-evidence-pin-sha.py"
SPEC = importlib.util.spec_from_file_location("check_evidence_pin_sha", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

SHA_A = "a" * 64
SHA_B = "b" * 64


class DistinctShasTests(unittest.TestCase):
    def test_finds_one_distinct_sha_across_repetition(self) -> None:
        text = f"sha256 `{SHA_A}`\nagain {SHA_A}\n"
        self.assertEqual(mod.distinct_shas(text), {SHA_A})

    def test_two_distinct_shas(self) -> None:
        self.assertEqual(len(mod.distinct_shas(f"{SHA_A} {SHA_B}")), 2)

    def test_shorter_and_longer_hex_do_not_match(self) -> None:
        self.assertEqual(mod.distinct_shas("0" * 40 + " " + "f" * 65), set())


class CheckPairTests(unittest.TestCase):
    def pin(self, sha: str | None = SHA_A) -> str:
        text = (
            "```oracle-pin\n"
            "id = fixture\n"
            "role = secondary\n"
            "evidence = docs/bench-evidence/fixture/report.md\n"
            "```\n"
        )
        if sha is not None:
            text += f"sha256 `{sha}` over the artifact\n"
        return text

    def test_agreeing_shas_pass(self) -> None:
        self.assertIsNone(mod.check_pair(self.pin(), f"sha256 `{SHA_A}`\n"))

    def test_disagreeing_shas_name_both_values(self) -> None:
        errors = mod.check_pair(self.pin(SHA_A), f"sha256 `{SHA_B}`\n")
        self.assertIsNotNone(errors)
        self.assertIn(SHA_A, errors)
        self.assertIn(SHA_B, errors)
        self.assertIn("disagree", errors)

    def test_two_shas_in_the_report_stay_unchecked(self) -> None:
        report = f"`{SHA_A}` and `{SHA_B}`\n"
        self.assertIsNone(mod.check_pair(self.pin(SHA_A), report))

    def test_two_shas_in_the_pin_stay_unchecked(self) -> None:
        pin = self.pin(SHA_A) + f"the prior artifact was `{SHA_B}`\n"
        self.assertIsNone(mod.check_pair(pin, f"sha256 `{SHA_A}`\n"))

    def test_a_report_without_a_sha_stays_unchecked(self) -> None:
        self.assertIsNone(mod.check_pair(self.pin(SHA_A), "no hash recorded\n"))



class PinEvidencePairsTests(unittest.TestCase):
    def test_only_bench_evidence_files_are_paired(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            oracles = base / ".agents/oracles"
            oracles.mkdir(parents=True)
            (oracles / "fixture.md").write_text(
                "```oracle-pin\n"
                "id = fixture\n"
                "role = secondary\n"
                "evidence = docs/bench-evidence/fixture/report.md\n"
                "```\n",
                encoding="utf-8",
            )
            (oracles / "issue.md").write_text(
                "```oracle-pin\n"
                "id = issue\n"
                "role = secondary\n"
                "evidence = #647\n"
                "```\n",
                encoding="utf-8",
            )
            (base / "docs/bench-evidence/fixture").mkdir(parents=True)
            (base / "docs/bench-evidence/fixture/report.md").write_text(
                f"sha256 `{SHA_B}`\n", encoding="utf-8"
            )
            pairs = mod.pin_evidence_pairs(oracles, base)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "fixture")
            self.assertEqual(pairs[0][1].name, "report.md")


if __name__ == "__main__":
    unittest.main()
