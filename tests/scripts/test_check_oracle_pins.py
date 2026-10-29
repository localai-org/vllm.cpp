#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-oracle-pins.py.

The checker's claim is that the oracle registry cannot rot: not into a record
missing its revision, not into a second `primary`, and not into an AGENTS.md
table that names a different set of oracles than the directory holds. This file
PERFORMS each of those mutations against a synthetic registry and fails if the
checker shrugs one off, which is what makes the claim testable rather than
asserted.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-oracle-pins.py"
SPEC = importlib.util.spec_from_file_location("check_oracle_pins", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_oracle_pins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_oracle_pins
SPEC.loader.exec_module(check_oracle_pins)


GOOD_SECONDARY = """# Fixture

```oracle-pin
id = fixture
role = secondary
upstream = https://github.com/example/fixture
scope = a path vLLM does not implement
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
"""

GOOD_PRIMARY = """# Fixture primary

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = everything vLLM implements
pin = 5559679229bc961848b121ccdeaa8fa5d79bec98
pin_label = 0.26.0.dev0
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/upstream-sync.md
```
"""


PARITY_RECORD = """# Fixture primary, paired with PARITY_SYNC

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = everything vLLM implements
pin = 1111111111111111111111111111111111111111
pin_label = 9.9.9rc1.dev1
pinned_on = 2026-09-04
gateable = yes
evidence = AGENTS.md
```
"""

PARITY_SYNC = """# Fixture authority

```parity-pin
vllm_commit = 1111111111111111111111111111111111111111
vllm_runtime_version = 9.9.9rc1.dev1+g1111111111
vllm_distribution_version = 9.9.9rc1.dev1+g1111111111.precompiled
flashinfer_version = 0.6.18
```
"""

# A secondary oracle for the SCOPING case: its `pin` and `pin_label` name a
# revision and a release PARITY_SYNC never mentions, which is the ordinary
# state of all thirteen of them. Reconciling this record against the authority
# reports it, so a registry-wide rule cannot stay green over this fixture.
UNRECONCILED_SECONDARY = """# Fixture secondary, pinned to something else entirely

```oracle-pin
id = fixture
role = secondary
upstream = https://github.com/example/fixture
scope = a path vLLM does not implement
pin = 3333333333333333333333333333333333333333
pin_label = 8.8.8
pinned_on = 2026-09-04
gateable = no
evidence = #647
```
"""


# A prose surface paired with PARITY_SYNC. The sentence around the spans is
# deliberately NOT minimal: it also names a second project's pin (`b10451`, six
# characters of [0-9a-f]) and a PRIOR vLLM pin, both of which a prose-parsing
# rule would have had to read and neither of which this rule may touch.
PIN_SURFACE = """# Fixture surface

Reference versions: vLLM <!--pin:label-->9.9.9rc1.dev1<!--/pin-->
(<!--pin:commit-->`1111111111`<!--/pin-->, the parity pin since 2026-09-03),
llama.cpp `b10451`. Rows below were read at the prior pin `555967922`.
"""


# The prose surfaces as `main` sees them: the five named paths, mirroring
# `PIN_SURFACES` in the checker, so the through-`main` cases exercise the same
# per-file requirement the real tree carries.
PIN_SURFACES_FIXTURE = {
    "NOW.md": PIN_SURFACE,
    "docs/FEATURES.md": PIN_SURFACE,
    "docs/benchmarks/how-we-measure.md": PIN_SURFACE,
    "docs/benchmarks/speculative-decoding.md": PIN_SURFACE,
    "docs/benchmarks/vllm-online-serving.md": PIN_SURFACE,
}


def pin_errors(text: str, required: bool = True) -> list[str]:
    """The pin-surface rule's complaints over one synthetic surface."""
    errors: list[str] = []
    parity = check_oracle_pins.parse_parity_pin("upstream-sync.md", PARITY_SYNC, errors)
    self_path = Path("NOW.md")
    check_oracle_pins.check_pin_surfaces(
        {self_path: text}, (self_path,) if required else (), parity, errors
    )
    return errors


def record(text: str, name: str = "fixture") -> check_oracle_pins.Record | None:
    """Parse one fixture, returning None when the block will not parse."""
    return check_oracle_pins.parse_record(Path(f"{name}.md"), text, [])


def errors_for(files: dict[str, str], agents_ids: list[str]) -> list[str]:
    """Run the whole registry check over a synthetic tree."""
    return check_oracle_pins.check_registry(
        {Path(f"{name}.md"): text for name, text in files.items()},
        agents_ids,
    )


class RecordParsingTests(unittest.TestCase):
    def test_wellformed_record_parses(self) -> None:
        parsed = record(GOOD_SECONDARY)
        assert parsed is not None
        self.assertEqual(parsed.fields["id"], "fixture")
        self.assertEqual(parsed.fields["gateable"], "no")

    def test_missing_block_is_reported(self) -> None:
        errs: list[str] = []
        self.assertIsNone(
            check_oracle_pins.parse_record(Path("fixture.md"), "# no block here\n", errs)
        )
        self.assertTrue(errs)

    def test_two_blocks_are_reported(self) -> None:
        errs: list[str] = []
        check_oracle_pins.parse_record(
            Path("fixture.md"), GOOD_SECONDARY + GOOD_SECONDARY, errs
        )
        self.assertTrue(any("exactly one" in e for e in errs))

    def test_every_required_key_is_load_bearing(self) -> None:
        # Dropping ANY single required key must be reported. A required-key list
        # that outgrows its enforcement is the failure this performs, not asserts.
        for key in check_oracle_pins.REQUIRED_KEYS:
            with self.subTest(key=key):
                mutated = "\n".join(
                    line
                    for line in GOOD_SECONDARY.splitlines()
                    if not line.startswith(f"{key} =")
                )
                errs = errors_for({"fixture": mutated}, ["fixture"])
                self.assertTrue(
                    any(key in e for e in errs),
                    f"dropping {key!r} was not reported: {errs}",
                )

    def test_empty_value_is_reported(self) -> None:
        mutated = GOOD_SECONDARY.replace("scope = a path vLLM does not implement", "scope =")
        errs = errors_for({"fixture": mutated}, ["fixture"])
        self.assertTrue(any("scope" in e for e in errs), errs)

    def test_unknown_key_is_reported(self) -> None:
        mutated = GOOD_SECONDARY.replace("gateable = no", "gateable = no\nnotes = smuggled")
        errs = errors_for({"fixture": mutated}, ["fixture"])
        self.assertTrue(any("notes" in e for e in errs), errs)


class RegistryConsistencyTests(unittest.TestCase):
    def test_clean_registry_passes(self) -> None:
        self.assertEqual(
            errors_for({"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY}, ["vllm", "fixture"]),
            [],
        )

    def test_id_must_match_filename(self) -> None:
        errs = errors_for({"renamed": GOOD_SECONDARY}, ["renamed"])
        self.assertTrue(any("filename" in e for e in errs), errs)

    def test_two_primaries_are_reported(self) -> None:
        second = GOOD_PRIMARY.replace("id = vllm", "id = usurper")
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "usurper": second}, ["vllm", "usurper"]
        )
        self.assertTrue(any("primary" in e for e in errs), errs)

    def test_missing_primary_is_reported(self) -> None:
        errs = errors_for({"fixture": GOOD_SECONDARY}, ["fixture"])
        self.assertTrue(any("primary" in e for e in errs), errs)

    def test_primary_must_be_vllm(self) -> None:
        usurper = GOOD_PRIMARY.replace("id = vllm", "id = sglang")
        errs = errors_for({"sglang": usurper}, ["sglang"])
        self.assertTrue(any("vllm" in e for e in errs), errs)

    def test_registry_entry_absent_from_agents_is_reported(self) -> None:
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY}, ["vllm"]
        )
        self.assertTrue(any("AGENTS.md" in e and "fixture" in e for e in errs), errs)

    def test_agents_entry_absent_from_registry_is_reported(self) -> None:
        errs = errors_for({"vllm": GOOD_PRIMARY}, ["vllm", "ghost"])
        self.assertTrue(any("ghost" in e for e in errs), errs)

    # Multiset, not set (CHECKER-HARDENING, FLINT-CPU-PARITY M4): the SAME row
    # admitted twice holds the same SET of ids, so the old set comparison
    # passed it. The failure must name the duplicated id and the counts.
    def test_duplicated_agents_registry_row_is_reported(self) -> None:
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY},
            ["vllm", "fixture", "fixture"],
        )
        self.assertTrue(
            any("fixture" in e and "2x" in e and "duplicated" in e for e in errs),
            errs,
        )

    def test_a_singly_admitted_row_stays_green(self) -> None:
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY},
            ["vllm", "fixture"],
        )
        self.assertEqual([e for e in errs if "fixture" in e], [])


class GateabilityTests(unittest.TestCase):
    def test_gateable_yes_needs_a_real_pin(self) -> None:
        # An oracle cannot be gateable against a revision it does not name.
        mutated = (
            GOOD_PRIMARY.replace(
                "pin = 5559679229bc961848b121ccdeaa8fa5d79bec98", "pin = UNPINNED"
            )
        )
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("UNPINNED" in e for e in errs), errs)

    def test_gateable_yes_needs_evidence_that_exists(self) -> None:
        mutated = GOOD_PRIMARY.replace(
            "evidence = .agents/upstream-sync.md",
            "evidence = .agents/specs/this-file-does-not-exist.md",
        )
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("evidence" in e for e in errs), errs)

    def test_gateable_yes_may_not_cite_an_issue_instead_of_evidence(self) -> None:
        # "#123" is a promise of measurement, not a measurement.
        mutated = GOOD_PRIMARY.replace("evidence = .agents/upstream-sync.md", "evidence = #647")
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("evidence" in e for e in errs), errs)

    def test_gateable_no_must_name_the_owing_issue(self) -> None:
        mutated = GOOD_SECONDARY.replace("evidence = #647", "evidence = not measured yet")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("issue" in e for e in errs), errs)

    def test_gateable_must_be_yes_or_no(self) -> None:
        mutated = GOOD_SECONDARY.replace("gateable = no", "gateable = partly")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("gateable" in e for e in errs), errs)

    def test_pinned_on_must_be_an_iso_date(self) -> None:
        mutated = GOOD_SECONDARY.replace("pinned_on = 2026-08-13", "pinned_on = last tuesday")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("pinned_on" in e for e in errs), errs)


class DeclaredOracleTests(unittest.TestCase):
    """A spec that opts into the `**Secondary oracle:**` syntax must name a real id."""

    def test_declared_id_must_be_registered(self) -> None:
        errs = check_oracle_pins.check_declarations(
            {Path("spec.md"): "**Secondary oracle:** `nonesuch`\n"},
            {"vllm", "sglang"},
        )
        self.assertTrue(any("nonesuch" in e for e in errs), errs)

    def test_registered_id_passes(self) -> None:
        self.assertEqual(
            check_oracle_pins.check_declarations(
                {Path("spec.md"): "**Secondary oracle:** `sglang`\n"},
                {"vllm", "sglang"},
            ),
            [],
        )

    def test_a_spec_that_never_declares_is_untouched(self) -> None:
        # No false positives on the hundreds of specs that mention llama.cpp or
        # SGLang in prose; only the opt-in syntax is enforced.
        self.assertEqual(
            check_oracle_pins.check_declarations(
                {Path("spec.md"): "we compared against llama.cpp and SGLang here\n"},
                {"vllm"},
            ),
            [],
        )


class ParityReconciliationTests(unittest.TestCase):
    """The vllm record's pin must agree with the AUTHORITY (#2829).

    Every case pairs a synthetic authority with a synthetic record, so nothing
    here moves when the real pin advances, and the expectation never comes from
    the file under test.
    """

    def reconcile(self, record_text: str, sync_text: str) -> list[str]:
        return check_oracle_pins.parity_errors(record_text, sync_text)

    def test_agreeing_surfaces_reconcile(self) -> None:
        self.assertEqual(self.reconcile(PARITY_RECORD, PARITY_SYNC), [])

    def test_a_stale_pin_is_reported(self) -> None:
        # THE #2829 SHAPE: the authority advanced and the record did not.
        mutated = PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        )
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("vllm_commit" in e for e in errs), errs)

    def test_a_stale_label_is_reported(self) -> None:
        mutated = PARITY_RECORD.replace("pin_label = 9.9.9rc1.dev1", "pin_label = 9.8.0")
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_truncated_label_is_reported(self) -> None:
        # A PROPER PREFIX of the right label. `startswith` accepts this and the
        # decomposition rule refuses it; that difference is the whole reason
        # the rule is stated as equality against the public version.
        mutated = PARITY_RECORD.replace("pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev")
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_label_carrying_the_local_segment_is_reported(self) -> None:
        # The other direction: the full runtime string is not the label either.
        mutated = PARITY_RECORD.replace(
            "pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev1+g1111111111"
        )
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_label_with_no_local_segment_in_the_authority_still_reconciles(self) -> None:
        # A pin taken from a tagged build reports a bare public version, and
        # the rule degrades to plain equality rather than going red on it.
        sync = PARITY_SYNC.replace(
            "vllm_runtime_version = 9.9.9rc1.dev1+g1111111111",
            "vllm_runtime_version = 9.9.9rc1.dev1",
        )
        self.assertEqual(self.reconcile(PARITY_RECORD, sync), [])

    def test_a_missing_authority_block_fails_closed(self) -> None:
        # Not a skipped rule. A checker that stops checking when it cannot find
        # its expectation is a mute switch.
        errs = self.reconcile(PARITY_RECORD, "# the block was renamed\n")
        self.assertTrue(any("parity-pin" in e for e in errs), errs)

    def test_two_authority_blocks_fail_closed(self) -> None:
        errs = self.reconcile(PARITY_RECORD, PARITY_SYNC + PARITY_SYNC)
        self.assertTrue(any("exactly one" in e for e in errs), errs)

    def test_an_unparsable_authority_line_is_reported(self) -> None:
        sync = PARITY_SYNC.replace("flashinfer_version = 0.6.18", "flashinfer 0.6.18")
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("is not `key = value`" in e for e in errs), errs)

    def test_an_unknown_authority_key_is_reported(self) -> None:
        sync = PARITY_SYNC.replace("flashinfer_version = 0.6.18", "torch_version = 2.13.0")
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("torch_version" in e for e in errs), errs)

    def test_a_duplicate_authority_key_is_reported(self) -> None:
        sync = PARITY_SYNC.replace(
            "flashinfer_version = 0.6.18",
            "flashinfer_version = 0.6.18\nflashinfer_version = 0.6.15",
        )
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("duplicate" in e for e in errs), errs)

    def test_an_empty_authority_value_is_reported(self) -> None:
        sync = PARITY_SYNC.replace(
            "vllm_commit = 1111111111111111111111111111111111111111", "vllm_commit ="
        )
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("vllm_commit" in e for e in errs), errs)

    def test_an_unreadable_record_fails_closed(self) -> None:
        errs = self.reconcile("# no oracle-pin block here\n", PARITY_SYNC)
        self.assertTrue(any("unchecked" in e for e in errs), errs)

    def test_public_version_strips_only_the_local_segment(self) -> None:
        cases = {
            "0.28.1rc1.dev132+ge126687a9": "0.28.1rc1.dev132",
            "0.28.1rc1.dev132+ge126687a9.precompiled": "0.28.1rc1.dev132",
            "0.29.0": "0.29.0",
        }
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(check_oracle_pins.public_version(given), expected)

    def test_the_live_surfaces_agree(self) -> None:
        # The reconciliation over the REAL two files, expectation from the
        # authority and value from the record, neither one a literal here.
        sync_text = check_oracle_pins.UPSTREAM_SYNC.read_text(encoding="utf-8")
        record_text = (check_oracle_pins.ORACLES / "vllm.md").read_text(encoding="utf-8")
        self.assertEqual(check_oracle_pins.parity_errors(record_text, sync_text), [])


class PinSurfaceTests(unittest.TestCase):
    """The prose surfaces are reconciled against the AUTHORITY (#2883).

    Every case asserts on the MESSAGE and not merely on the count, because
    "something was reported" does not distinguish the rule that fired.
    """

    def test_agreeing_surface_is_clean(self) -> None:
        self.assertEqual(pin_errors(PIN_SURFACE), [])

    def test_a_stale_revision_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("1111111111", "2222222222"))
        self.assertEqual(len(errs), 1)
        self.assertIn("is not a prefix of", errs[0])
        self.assertIn("2222222222", errs[0])

    def test_a_stale_label_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("9.9.9rc1.dev1<", "9.8.0<"))
        self.assertEqual(len(errs), 1)
        self.assertIn("does not match the public version", errs[0])
        self.assertIn("9.8.0", errs[0])

    def test_a_label_with_a_suffix_is_reported(self) -> None:
        # A CONTAINMENT rule passes this one: the expectation is a substring of
        # the value. Equality after the local-segment decomposition is not.
        errs = pin_errors(PIN_SURFACE.replace("9.9.9rc1.dev1<", "9.9.9rc1.dev1320<"))
        self.assertEqual(len(errs), 1)
        self.assertIn("does not match the public version", errs[0])

    def test_a_truncated_label_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("9.9.9rc1.dev1<", "9.9.9rc1.dev<"))
        self.assertEqual(len(errs), 1)
        self.assertIn("does not match the public version", errs[0])

    def test_the_two_character_abbreviation_is_reported(self) -> None:
        # The degenerate prefix #2883 names: `e1` is a prefix and is worthless.
        errs = pin_errors(PIN_SURFACE.replace("`1111111111`", "`11`"))
        self.assertEqual(len(errs), 1)
        self.assertIn("under the 8-character floor", errs[0])

    def test_the_floor_itself_is_accepted(self) -> None:
        # The boundary in the OTHER direction: eight is the shortest length this
        # tree writes, so it must pass or the rule fires on correct work.
        self.assertEqual(pin_errors(PIN_SURFACE.replace("`1111111111`", "`11111111`")), [])

    def test_seven_characters_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("`1111111111`", "`1111111`"))
        self.assertEqual(len(errs), 1)
        self.assertIn("under the 8-character floor", errs[0])

    def test_a_longer_prefix_is_accepted(self) -> None:
        # No FIXED length is imposed: eleven characters of the right revision is
        # correct, and a length rule pinned at ten would red on it.
        self.assertEqual(pin_errors(PIN_SURFACE.replace("`1111111111`", "`11111111111`")), [])

    def test_a_non_hexadecimal_revision_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("1111111111", "zzzzzzzzzz"))
        self.assertEqual(len(errs), 1)
        self.assertIn("is not a hexadecimal revision", errs[0])

    def test_an_unclosed_marker_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("`1111111111`<!--/pin-->", "`1111111111`"))
        joined = "\n".join(errs)
        self.assertIn("opener(s) but", joined)
        self.assertIn("carries no `<!--pin:commit-->` span", joined)

    def test_a_marker_kind_nobody_defined_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("pin:commit", "pin:sha"))
        joined = "\n".join(errs)
        self.assertIn("opener(s) but", joined)

    def test_an_empty_span_is_reported(self) -> None:
        errs = pin_errors(PIN_SURFACE.replace("`1111111111`", ""))
        joined = "\n".join(errs)
        self.assertIn("empty `commit` pin span", joined)

    def test_a_required_surface_with_no_marker_is_reported(self) -> None:
        errs = pin_errors("Reference versions: vLLM 9.9.9rc1.dev1 (`1111111111`).\n")
        self.assertEqual(len(errs), 1)
        self.assertIn("carries no `<!--pin:commit-->` span", errs[0])

    def test_a_label_only_surface_does_not_satisfy_the_requirement(self) -> None:
        errs = pin_errors("vLLM <!--pin:label-->9.9.9rc1.dev1<!--/pin-->.\n")
        self.assertEqual(len(errs), 1)
        self.assertIn("carries no `<!--pin:commit-->` span", errs[0])

    def test_an_unreadable_required_surface_is_reported(self) -> None:
        errors: list[str] = []
        parity = check_oracle_pins.parse_parity_pin("upstream-sync.md", PARITY_SYNC, errors)
        check_oracle_pins.check_pin_surfaces({}, (Path("NOW.md"),), parity, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("a declared pin surface, but it is unreadable", errors[0])

    def test_a_benign_prose_edit_does_not_fire(self) -> None:
        # THE "a gate that fires on correct work is the defect" CASE. The
        # paragraph is rewrapped, a sentence is added, and a THIRD mention of a
        # prior pin appears -- all outside the spans, all of which the rule must
        # ignore. `b10451` and `555967922` are already in PIN_SURFACE and are the
        # two tokens a prose-parsing rule got wrong.
        rewrapped = PIN_SURFACE.replace(
            "(<!--pin:commit-->`1111111111`<!--/pin-->, the parity pin since 2026-09-03),",
            "(<!--pin:commit-->`1111111111`<!--/pin-->,\nthe parity pin since 2026-09-03),",
        ) + (
            "\nEvery ratio here was captured at `e24d1b24` and at\n"
            "vLLM 0.26.0.dev0, and has not been re-validated.\n"
        )
        self.assertNotEqual(rewrapped, PIN_SURFACE)
        self.assertEqual(pin_errors(rewrapped), [])

    def test_an_unmarked_file_that_is_not_required_is_ignored(self) -> None:
        # The many files that name a prior pin on purpose, in miniature.
        self.assertEqual(
            pin_errors("The prior pin was `555967922`, and that stays true.\n", required=False),
            [],
        )

    def test_a_marker_at_the_start_of_a_line_is_reported(self) -> None:
        # THE RENDERING CASE, and the one this suite could not see. A column-1
        # HTML comment is a CommonMark type-2 HTML block, it interrupts the
        # paragraph, and the rest of the line is emitted raw -- so the span's
        # own value renders as literal backticks. `docs/FEATURES.md:46` shipped
        # this shape and three documents asserted it was invisible.
        broken = PIN_SURFACE.replace(
            "(<!--pin:commit-->`1111111111`<!--/pin-->, the parity pin",
            "\n<!--pin:commit-->`1111111111`<!--/pin-->, the parity pin",
        )
        self.assertNotEqual(broken, PIN_SURFACE)
        errs = pin_errors(broken)
        self.assertEqual(len(errs), 1)
        self.assertIn("starts its line", errs[0])
        # It names the LINE, because a reader has to find it.
        self.assertIn("NOW.md:5:", errs[0])

    def test_a_closing_marker_at_the_start_of_a_line_is_reported(self) -> None:
        # The closer breaks the same way, so the rule cannot look only at
        # openers.
        errs = pin_errors(
            PIN_SURFACE.replace("`1111111111`<!--/pin-->", "`1111111111`\n<!--/pin-->")
        )
        self.assertEqual(len(errs), 1)
        self.assertIn("starts its line", errs[0])

    def test_an_indented_marker_at_a_block_boundary_is_reported(self) -> None:
        # Four spaces after a blank line is an indented code block, and the
        # marker becomes VISIBLE text. Measured on `gh api /markdown`.
        errs = pin_errors(
            PIN_SURFACE.replace(
                "(<!--pin:commit-->", "\n\n    <!--pin:commit-->"
            )
        )
        self.assertTrue(any("starts its line" in e for e in errs), errs)

    def test_a_marker_that_merely_follows_text_is_accepted(self) -> None:
        # The benign direction of the same rule: one character of prose before
        # the marker is enough to keep it inline, and the rule must not fire.
        # Without this case the rule could be "never wrap near a marker".
        self.assertEqual(
            pin_errors(
                PIN_SURFACE.replace(
                    "(<!--pin:commit-->", "\n(<!--pin:commit-->"
                )
            ),
            [],
        )

    def test_the_live_surfaces_carry_no_line_initial_marker(self) -> None:
        # The rule applied to the tree, not to a fixture. This is the case that
        # would have caught `docs/FEATURES.md:46` before it landed.
        for path in check_oracle_pins.PIN_SURFACES:
            with self.subTest(surface=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(
                    [m.group(1) for m in check_oracle_pins.PIN_AT_LINE_START.finditer(text)],
                    [],
                )

    def test_a_mis_cased_marker_kind_is_reported(self) -> None:
        # `PIN_OPEN` was `[a-z]*` and could not express this, so beside a valid
        # span the opener and span counts agreed and NOTHING reported it.
        errs = pin_errors(PIN_SURFACE.replace("pin:label", "pin:LABEL"))
        self.assertEqual(len(errs), 1)
        self.assertIn("opener(s) but", errs[0])

    def test_a_marker_kind_outside_the_lowercase_class_is_reported(self) -> None:
        for kind in ("Commit", "commit2", "pin-commit"):
            with self.subTest(kind=kind):
                errs = pin_errors(PIN_SURFACE.replace("pin:commit", f"pin:{kind}"))
                self.assertTrue(any("opener(s) but" in e for e in errs), errs)

    def test_an_uppercase_revision_is_reported_for_its_case(self) -> None:
        # `E126687A` IS hexadecimal. Telling its author otherwise is false about
        # the value they wrote; the defect is the case, because the authority is
        # lowercase and the comparison is `startswith`.
        errs = pin_errors(PIN_SURFACE.replace("1111111111", "AAAAAAAAAA"))
        self.assertEqual(len(errs), 1)
        self.assertIn("hexadecimal but not lowercase", errs[0])
        self.assertNotIn("is not a hexadecimal revision", errs[0])

    def test_the_expectation_comes_from_the_authority_only(self) -> None:
        # A TAUTOLOGY GUARD. The rule opens no file: hand it an authority that
        # disagrees with the surface and it must report the surface, which it
        # cannot do if it is reading its expectation out of the surface.
        errors: list[str] = []
        other = PARITY_SYNC.replace("1111111111111111111111111111111111111111", "4" * 40)
        parity = check_oracle_pins.parse_parity_pin("upstream-sync.md", other, errors)
        path = Path("NOW.md")
        check_oracle_pins.check_pin_surfaces({path: PIN_SURFACE}, (path,), parity, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("is not a prefix of", errors[0])


class ParityThroughMainTests(unittest.TestCase):
    """The reconciliation is REACHED from the checker's entry point.

    Every other case here calls the rule directly, which proves the rule works
    and not that anything runs it. These three point `main` at a synthetic
    registry and assert on its return code, so deleting the call site in
    `main` turns this red -- the mutation AGENTS.md "Nothing lands dead" asks
    a reviewer to perform. The scoping case is here for the same reason: the
    scope is a property of `main`, so only `main` can be asked about it.
    """

    def agents_table(self, ids: tuple[str, ...]) -> str:
        """The AGENTS.md admissible-oracle table, naming exactly *ids*."""
        rows = "".join(f"| Fixture | `{oracle_id}` | a fixture |\n" for oracle_id in ids)
        return (
            "<!-- oracle-registry:begin -->\n"
            "| Oracle | Registry id | Reach for it when |\n"
            "|---|---|---|\n"
            f"{rows}"
            "<!-- oracle-registry:end -->\n"
        )

    def run_main(
        self,
        record_text: str,
        sync_text: str,
        others: dict[str, str] | None = None,
        surfaces: dict[str, str] | None = None,
        extra: dict[str, str] | None = None,
        declared_but_absent: tuple[str, ...] = (),
    ) -> int:
        """Drive `main` over a synthetic tree.

        *surfaces* replaces the prose pin surfaces, keyed by relative path;
        *extra* writes any other file under the root, which is how a declaration
        gets into the tree the declaration glob walks; *declared_but_absent*
        names surfaces `PIN_SURFACES` declares and nobody writes.
        """
        others = others or {}
        surfaces = PIN_SURFACES_FIXTURE if surfaces is None else surfaces
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracles = root / "oracles"
            oracles.mkdir()
            (oracles / "vllm.md").write_text(record_text, encoding="utf-8")
            for stem, text in others.items():
                (oracles / f"{stem}.md").write_text(text, encoding="utf-8")
            (root / "upstream-sync.md").write_text(sync_text, encoding="utf-8")
            (root / "AGENTS.md").write_text(
                self.agents_table(("vllm", *others)), encoding="utf-8"
            )
            for rel, text in {**surfaces, **(extra or {})}.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            with mock.patch.multiple(
                check_oracle_pins,
                # ROOT too: `evidence` existence, the declaration glob and the
                # pin surfaces' reported labels are all resolved against it, so
                # a half-patched module would read the real tree from inside a
                # synthetic one.
                ROOT=root,
                ORACLES=oracles,
                AGENTS_MD=root / "AGENTS.md",
                UPSTREAM_SYNC=root / "upstream-sync.md",
                DECLARATION_ROOTS=(root,),
                PIN_SURFACES=tuple(
                    root / rel for rel in (*surfaces, *declared_but_absent)
                ),
            ):
                return check_oracle_pins.main([])

    def test_main_passes_when_the_surfaces_agree(self) -> None:
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC), 0)

    def test_main_fails_when_the_surfaces_disagree(self) -> None:
        stale = PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        )
        self.assertEqual(self.run_main(stale, PARITY_SYNC), 1)

    def test_main_does_not_reconcile_a_secondary_oracle(self) -> None:
        # THE SCOPING TEST, and it has to go through `main`, because `main` is
        # where the scope lives: it selects the record to reconcile by filename
        # stem. `UNRECONCILED_SECONDARY` names a commit and a release the
        # authority never mentions, which is the ordinary state of the thirteen
        # secondary oracles, so a registry-wide version of this rule reports it
        # and returns 1. `check_registry` cannot be asked this question at
        # all, because `check_registry` never runs the parity rule. The live
        # `test_the_repository_registry_is_clean` does red under that same
        # mutation -- 26 parity errors over the 13 real secondaries -- but its
        # catch is incidental: it would evaporate if the registry ever held
        # `vllm.md` alone, and it reads whatever the tree happens to hold, so
        # it has no controlled fixture and cannot be the scoping test.
        self.assertEqual(
            self.run_main(
                PARITY_RECORD, PARITY_SYNC, {"fixture": UNRECONCILED_SECONDARY}
            ),
            0,
        )


    # ---- the prose surfaces are REACHED from `main` (#2883) ----------------

    def test_main_fails_when_a_prose_surface_is_stale(self) -> None:
        # Deleting `check_pin_surfaces(...)` from `main` turns this green, and
        # `test_main_passes_when_the_surfaces_agree` above proves the rest of
        # the fixture is clean, so the return code is attributable to this rule.
        stale = dict(PIN_SURFACES_FIXTURE)
        stale["docs/FEATURES.md"] = PIN_SURFACE.replace("1111111111", "2222222222")
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC, surfaces=stale), 1)

    def test_main_fails_when_a_prose_surface_loses_its_marker(self) -> None:
        unmarked = dict(PIN_SURFACES_FIXTURE)
        unmarked["NOW.md"] = "Pin: vLLM `1111111111` (9.9.9rc1.dev1).\n"
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC, surfaces=unmarked), 1)

    def test_main_fails_when_a_marker_starts_a_line(self) -> None:
        # The rendering rule, REACHED from the entry point. Deleting
        # `check_pin_surfaces(...)` from `main` turns this green too, and it is
        # the shape `docs/FEATURES.md:46` shipped: a marker at column 1 inside
        # a paragraph, which GitHub renders as a broken paragraph and literal
        # backticks. Nothing in the suite could see that class before.
        broken = dict(PIN_SURFACES_FIXTURE)
        broken["docs/FEATURES.md"] = PIN_SURFACE.replace(
            "(<!--pin:commit-->", "\n<!--pin:commit-->"
        )
        self.assertNotEqual(broken["docs/FEATURES.md"], PIN_SURFACE)
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC, surfaces=broken), 1)

    def test_main_fails_when_a_marker_kind_is_mis_cased(self) -> None:
        # The `[a-z]*` opener class made this rc 0 beside a valid commit span.
        mis_cased = dict(PIN_SURFACES_FIXTURE)
        mis_cased["docs/FEATURES.md"] = PIN_SURFACE.replace("pin:label", "pin:LABEL")
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC, surfaces=mis_cased), 1)

    def test_main_fails_when_a_declared_surface_does_not_exist(self) -> None:
        # A declared surface that was renamed or deleted must be REPORTED, not
        # skipped. `main` swallows the OSError and the rule supplies the reason,
        # so a surface cannot leave the tree and take its gate with it.
        self.assertEqual(
            self.run_main(PARITY_RECORD, PARITY_SYNC, declared_but_absent=("docs/GONE.md",)),
            1,
        )

    def test_main_ignores_prose_outside_a_span(self) -> None:
        # THE FALSE-POSITIVE CASE, through `main`. Every surface gains a prior
        # pin, a second project's pin and a rewrapped paragraph, and none of it
        # is inside a span. A rule that reads the sentence reds here.
        benign = {
            rel: text
            + "\nMeasured at the prior pin `555967922` against llama.cpp\n"
            + "`b10451`, and vLLM 0.26.0.dev0 at `e24d1b24`.\n"
            for rel, text in PIN_SURFACES_FIXTURE.items()
        }
        self.assertNotEqual(benign, PIN_SURFACES_FIXTURE)
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC, surfaces=benign), 0)

    # ---- the declaration check is REACHED from `main` (#2898) --------------

    def test_main_fails_on_a_declaration_naming_an_unregistered_oracle(self) -> None:
        # #2898: `check_declarations` had direct-call coverage at
        # DeclaredOracleTests and NOTHING proved `main` ran it -- deleting
        # `errors.extend(check_declarations(declarations, registry_ids))` left
        # all 42 tests green at rc 0. This case is the missing pin. It reds
        # under exactly that deletion and, because
        # `test_main_passes_when_a_declaration_names_a_registered_oracle`
        # below runs the SAME tree with only the id changed, the red is the
        # declaration rule and not the file's presence.
        self.assertEqual(
            self.run_main(
                PARITY_RECORD,
                PARITY_SYNC,
                extra={"specs/a-spec.md": "**Secondary oracle:** `nonesuch`\n"},
            ),
            1,
        )

    def test_main_passes_when_a_declaration_names_a_registered_oracle(self) -> None:
        # The POSITIVE CONTROL for the case above: same tree, same file, same
        # syntax, an id the synthetic registry does hold.
        self.assertEqual(
            self.run_main(
                PARITY_RECORD,
                PARITY_SYNC,
                extra={"specs/a-spec.md": "**Secondary oracle:** `vllm`\n"},
            ),
            0,
        )


class LiveRegistryTests(unittest.TestCase):
    def test_the_repository_registry_is_clean(self) -> None:
        self.assertEqual(check_oracle_pins.main([]), 0)

    def test_self_test_corpus_passes_in_both_directions(self) -> None:
        self.assertEqual(check_oracle_pins.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
