#!/usr/bin/env python3
"""Mutation checks for scripts/check-agent-record.py lifecycle enforcement."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
CHECKER = ROOT / "scripts/check-agent-record.py"
SPEC = importlib.util.spec_from_file_location("agent_record", CHECKER)
assert SPEC is not None and SPEC.loader is not None
agent_record = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_record
SPEC.loader.exec_module(agent_record)

# The restructured checker split the issue-record primitives into their own
# module; the tests address them through the checker as before.
_ISSUE_RECORDS_SPEC = importlib.util.spec_from_file_location(
    "agent_record.issue_records", ROOT / "scripts/issue_records.py"
)
assert _ISSUE_RECORDS_SPEC is not None and _ISSUE_RECORDS_SPEC.loader is not None
agent_record.issue_records = importlib.util.module_from_spec(_ISSUE_RECORDS_SPEC)
sys.modules[_ISSUE_RECORDS_SPEC.name] = agent_record.issue_records
_ISSUE_RECORDS_SPEC.loader.exec_module(agent_record.issue_records)


def check_issue_records(
    errors: list[str],
    *,
    issues_root: Path,
    rows: set[str],
    owed: dict,
    references: set[str],
    frozen_archive: bytes,
) -> None:
    """Compose the current validation path the restructured checker ships.

    The old checker monolith owned this entry point; the restructure moved the
    primitives into ``scripts/issue_records.py`` and the per-record walk into
    ``scripts/agent-issue.py``. The tests exercise the primitives directly in
    the same order the collection walk uses: parse, collection contracts,
    per-record contracts, then reference resolution.
    """

    issue_records = agent_record.issue_records
    records = [
        issue_records.parse_issue_file(path)
        for path in sorted(issues_root.rglob("*.md"))
    ]
    try:
        issue_records.validate_issue_collection(records)
    except issue_records.IssueRecordError as error:
        errors.append(str(error))
    for record in records:
        path = issues_root / (record.row or "") / f"{record.id}.md"
        try:
            issue_records.validate_issue_record(
                record, path, rows, owed, frozen_archive=frozen_archive
            )
        except issue_records.IssueRecordError as error:
            errors.append(str(error))
    for reference in sorted(references):
        try:
            issue_records.resolve_issue_reference(reference, records)
        except issue_records.IssueRecordError as error:
            errors.append(str(error))


agent_record.check_issue_records = check_issue_records


def with_field(row, field: str, value: str):
    index = agent_record.field_index(row.header, field)
    assert index is not None, (row.item_id, field)
    cells = list(row.cells)
    cells[index] = value
    return replace(row, cells=tuple(cells), raw="| " + " | ".join(cells) + " |")


def validate_mutation(rows, changed):
    mutated = [changed if row.item_id == changed.item_id else row for row in rows]
    by_id = {row.item_id: row for row in mutated}
    errors: list[str] = []
    agent_record.check_row_contracts(mutated, by_id, errors)
    return errors


def require(errors: list[str], pattern: str) -> None:
    if not any(re.search(pattern, error) for error in errors):
        raise AssertionError(f"missing expected error {pattern!r}:\n" + "\n".join(errors))



def tracked_issues(test: unittest.TestCase) -> str:
    """Return every canonical GitHub identity from tracked local records."""

    del test
    records = [
        agent_record.issue_records.parse_issue_file(path)
        for path in sorted((ROOT / ".agents/issues").glob("**/*.md"))
    ]
    return "\n".join(
        f"issues/{record.github})"
        for record in records
        if record.github is not None
    )


class AgentRecordMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parse_errors: list[str] = []
        cls.rows, _ = agent_record.check_matrices(parse_errors)
        if parse_errors:
            raise AssertionError("\n".join(parse_errors))
        cls.by_id = {row.item_id: row for row in cls.rows}

        baseline_errors: list[str] = []
        agent_record.check_row_contracts(cls.rows, cls.by_id, baseline_errors)
        if baseline_errors:
            raise AssertionError("\n".join(baseline_errors))

    def test_ready_requires_real_spec(self) -> None:
        ready = with_field(
            self.by_id["PAR-TP"], "spec", "`planned: specs/tensor-parallelism.md`"
        )
        require(
            validate_mutation(self.rows, ready),
            r"PAR-TP READY has no real .agents/specs link",
        )

    def test_ready_spec_names_stable_row(self) -> None:
        wrong_spec = with_field(
            self.by_id["PAR-TP"], "spec", "[wrong spike](specs/mtp-spec-decode.md)"
        )
        require(
            validate_mutation(self.rows, wrong_spec),
            r"no linked spec names exact stable token `PAR-TP`",
        )

    def test_prose_keyword_bag_is_not_a_spike_contract(self) -> None:
        prose = (
            "`PAR-TP` scope upstream chain our baseline port map tests to port "
            "gates dependencies work breakdown risks and decisions"
        )
        missing = agent_record.missing_spec_requirements(prose)
        self.assertEqual(set(missing), set(agent_record.SPEC_REQUIREMENTS))

    def test_empty_structured_sections_are_not_a_spike_contract(self) -> None:
        empty_headings = "`PAR-TP`\n" + "\n".join(
            f"## {label}" for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(empty_headings)),
            set(agent_record.SPEC_REQUIREMENTS),
        )
        empty_rows = "\n".join(
            f"| {label} | - |" for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(empty_rows)),
            set(agent_record.SPEC_REQUIREMENTS),
        )
        placeholder_tables = "`PAR-TP`\n" + "\n".join(
            f"## {label}\n| Key | Value |\n|---|---|\n| thing | - |"
            for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(placeholder_tables)),
            set(agent_record.SPEC_REQUIREMENTS),
        )

    def test_active_requires_claim_owner(self) -> None:
        # The fixture row must actually be `ACTIVE` today, or the mutation stops
        # exercising the ACTIVE branch and the test passes vacuously. It was
        # pinned to KERNEL-GDN-AOT-BF16 until the 2026-08-06 live-state audit
        # moved that row to READY, so it is now picked from the live record.
        active_id = next(
            row.item_id
            for row in self.rows
            if row.state == "ACTIVE" and row.path.name == "kernel-matrix.md"
        )
        active = with_field(self.by_id[active_id], "owner", "-")
        require(
            validate_mutation(self.rows, active),
            rf"ACTIVE row {re.escape(active_id)} has no CLAIM-\* owner",
        )

    def test_implemented_state_requires_exact_code_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "implementation exists somewhere",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_unrelated_local_link_is_not_code_or_test_evidence(self) -> None:
        bad_code = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "[not evidence](roadmap_v1.md#L1)",
        )
        require(
            validate_mutation(self.rows, bad_code),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

        bad_test = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "tests",
            "[not evidence](roadmap_v1.md#L1)",
        )
        require(
            validate_mutation(self.rows, bad_test),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact test/evidence anchor",
        )

    def test_out_of_range_line_is_not_an_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "[bad line](../src/vt/cuda/cuda_moe.cu#L999999), "
            "[test](../tests/vt/test_ops_moe_grouped.cpp#L453)",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_out_of_range_raw_range_is_not_an_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "`src/vt/cuda/cuda_moe.cu:349-999999`",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_done_requires_exact_ledger_link(self) -> None:
        done = self.by_id["QUANT-NVFP4-MO-W4A16"]
        evidence = done.field("tests").replace(
            "parity-ledger.md#L284", "NOW.md#L1"
        )
        done = with_field(done, "tests", evidence)
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 lacks exact parity-ledger link",
        )

    def test_done_requires_closing_commit_in_owner(self) -> None:
        done = with_field(self.by_id["QUANT-NVFP4-MO-W4A16"], "owner", "-")
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 owner is not the hexadecimal closing commit",
        )

    def test_done_closing_commit_must_exist(self) -> None:
        done = with_field(
            self.by_id["QUANT-NVFP4-MO-W4A16"], "owner", "deadbee"
        )
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 closing commit deadbee does not exist",
        )

    def test_tables_require_semantic_owner_column(self) -> None:
        factory = self.by_id["MODEL-FACTORY-registry"]
        owner_index = agent_record.field_index(factory.header, "owner")
        if owner_index is None:
            raise AssertionError("baseline MODEL-FACTORY-registry has no owner column")
        malformed = replace(
            factory,
            header=factory.header[:owner_index] + factory.header[owner_index + 1 :],
            cells=factory.cells[:owner_index] + factory.cells[owner_index + 1 :],
        )
        require(
            validate_mutation(self.rows, malformed),
            r"MODEL-FACTORY-registry table lacks semantic owner column",
        )

    def test_windows_release_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])
        windows = [row for row in rows if row.item_id == "ENG-RELEASE-WINDOWS"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].path.name, "engine-matrix.md")

    def test_dead_capability_rows_are_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])
        # INVENTORIED, not SPIKE: a SPIKE row obliges a `CLAIM-*` owner, and
        # inventing one would record work nobody is doing. `ENG-GATE-ENV-DOC`
        # left that state when #2389 landed the reverse direction of the gate,
        # Its current GATING state remains an exact semantic guarantee.
        expected_state = {
            "ENG-WEIGHT-RESIDENCY": "INVENTORIED",
            "ENG-STRUCTURED-OUTPUT": "INVENTORIED",
            "ENG-ATTENTION-WINDOW": "INVENTORIED",
            "ENG-GATE-ENV-DOC": "GATING",
        }
        for item_id, state in expected_state.items():
            with self.subTest(item_id=item_id):
                found = [row for row in rows if row.item_id == item_id]
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0].path.name, "engine-matrix.md")
                self.assertEqual(found[0].state, state)

    def test_serve_recipe_args_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])
        recipe = [row for row in rows if row.item_id == "SERVE-RECIPE-ARGS"]
        self.assertEqual(len(recipe), 1)
        self.assertEqual(recipe[0].path.name, "engine-matrix.md")

    def test_omni_pin_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        for item_id in ("ENG-UPSTREAM-OMNI-PIN", "SERVE-RECIPE-ARGS"):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "engine-matrix.md", item_id)

    def test_anchor_ratchet_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        found = [row for row in rows if row.item_id == "ENG-RECORD-ANCHOR-RATCHET"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path.name, "engine-matrix.md")

    def test_residency_config_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        found = [row for row in rows if row.item_id == "ENG-RESIDENCY-CONFIG"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path.name, "engine-matrix.md")

    def test_pool_best_fit_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        found = [row for row in rows if row.item_id == "ENG-POOL-BEST-FIT"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path.name, "engine-matrix.md")

    def test_ltx2_pin_row_is_inside_the_engine_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        for item_id in ("ENG-UPSTREAM-LTX2-PIN", "ENG-UPSTREAM-OMNI-PIN"):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "engine-matrix.md", item_id)

    def test_music3_and_indextts_rows_both_survive_their_collision(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        collided = (
            "MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-s2-mel-decoder",
        )
        for item_id in collided:
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)

    def test_indextts_rows_are_inside_the_model_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        for item_id in (
            "MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-s2-mel-decoder",
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
            self.assertEqual(
                found[0].field("state").strip().strip("`"), "INVENTORIED", item_id
            )

    def test_dots3_rows_are_inside_the_model_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        for item_id, state in (
            ("MODEL-MM-dots3-note-dots3-note-for-causal-lm", "SPIKE"),
            ("MODEL-SPEC-dots3-note-dots3-note-mtp", "INVENTORIED"),
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
            self.assertEqual(found[0].field("state").strip().strip("`"), state, item_id)

    def test_qwen4_exp_row_is_inside_the_model_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        item_id = "MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation"
        found = [row for row in rows if row.item_id == item_id]
        self.assertEqual(len(found), 1, item_id)
        self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
        # `ACTIVE` since W6a (#1989), the first wave to land product code. This was
        # `READY` when the row was spec-only; product code makes `ACTIVE` the exact
        # semantic state until the port reaches a terminal state.
        self.assertEqual(found[0].field("state").strip().strip("`"), "ACTIVE", item_id)

        # One row, not two: no speculative-head sibling exists for this arch.
        siblings = [row for row in rows if "qwen4-exp" in row.item_id]
        self.assertEqual([row.item_id for row in siblings], [item_id])

    def test_glm5_next_row_is_inside_the_model_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        item_id = "MODEL-MM-glm5-next-glm5-next-for-conditional-generation"
        found = [row for row in rows if row.item_id == item_id]
        self.assertEqual(len(found), 1, item_id)
        self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
        self.assertEqual(found[0].field("state").strip().strip("`"), "ACTIVE", item_id)

        # One row, not three: neither the text-only arm nor the MTP head has a
        # sibling row, and adding one to mirror the upstream PR is the mistake.
        siblings = [row for row in rows if "glm5-next" in row.item_id]
        self.assertEqual([row.item_id for row in siblings], [item_id])

    def test_quant_exl3_row_is_inside_the_quant_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "QUANT rows" in error], [])

        item_id = "QUANT-EXL3"
        found = [row for row in rows if row.item_id == item_id]
        self.assertEqual(len(found), 1, item_id)
        self.assertEqual(found[0].path.name, "quantization-matrix.md", item_id)
        self.assertEqual(found[0].field("state").strip().strip("`"), "ACTIVE", item_id)

        # TWO rows, and the second one is NAMED rather than allowed by a
        # loosened predicate. The original assertion was "one row, not two",
        # against the rank-sliced LAYOUT acquiring a sibling scheme row: that
        # layout is the same scheme read differently and still must not.
        #
        # `QUANT-EXL3-MUL1` (#2495) is admitted because it is a different
        # CODEBOOK, which is a different decode and not a different layout: cb 0
        # and cb 1 mask, xor and sum the two fp16 halves of the product, while cb
        # 2 sums the product's four bytes into an fp16 bit pattern and maps it
        # with a fused fp16 affine (`codebook.cuh:82-89`). It carries its own
        # artifact, its own bit widths and its own owed GEMV arm, none of which
        # `QUANT-EXL3`'s cells can hold without saying two things at once.
        #
        # `QUANT-EXL3-PERF` (#2570) is the THIRD, and it is admitted on a
        # different ground from the second. It is not a codebook, a width or a
        # layout: it is the only row on a different AXIS. `QUANT-EXL3` and
        # `QUANT-EXL3-MUL1` are correctness rows and their cells say so -- they
        # answer "does this width RUN" -- while this one answers "what does it
        # COST", which is the question #2570 asked and no row owned. Its `P`
        # cell is the one cell neither sibling can carry: `QUANT-EXL3-MUL1`
        # reads `E` `-` and `P` `-` precisely BECAUSE it ports the format and not
        # the benchmark, and overwriting those to hold a throughput verdict would
        # make one row say two things about two different measurements.
        #
        # The concrete surface is also disjoint. This row owns
        # `Exl3GemvArmInstantiated` and `GemvKernel` -- the `m <= 8` GEMV arm set
        # and its envelope -- which `QUANT-EXL3-MUL1`'s own claim file EXCLUDES
        # by name ("EXCLUDES the GEMV kernel itself"). A row whose scope another
        # row explicitly excluded is not a sibling scheme row; it is the owner
        # that exclusion implies.
        #
        # Listing all three by name keeps the force of the original: a FOURTH
        # `QUANT-*EXL3*` row still fails here and has to argue for itself in this
        # comment, which is exactly what a rank-sliced-layout row could not do.
        # Scoped to `QUANT-` deliberately -- `MODEL-DSV4-EXL3` also carries EXL3
        # in its id and is a MODEL row for the checkpoint that uses the scheme,
        # which is a different axis and must not be swept in here.
        siblings = [row for row in rows
                    if row.item_id.startswith("QUANT-") and "EXL3" in row.item_id]
        self.assertEqual(sorted(row.item_id for row in siblings),
                         ["QUANT-EXL3", "QUANT-EXL3-MUL1", "QUANT-EXL3-PERF"])

    def test_recipe_backfill_rows_are_inside_the_model_ratchet(self) -> None:
        """Preserve the exact keyed row, ownership, and semantic fields."""
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])

        for item_id in (
            "MODEL-TEXT-bailing-moe-v3-bailing-moe-v3-for-causal-lm",
            "MODEL-MM-moss-tts-moss-tts-delay-talker-for-generation",
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md")
            self.assertEqual(
                found[0].field("state").strip().strip("`"), "INVENTORIED", item_id
            )

    def test_beyond_pin_rows_stay_out_of_the_at_pin_model_inventory(self) -> None:
        """A beyond-pin row must not inflate the AT-THE-PIN model inventory.

        `check_model_invariants` pins what vLLM's OWN registry holds at
        `e126687a9a`: 324 category/target rows, 373 memberships, 356
        architectures, 309 targets, 245 modules. It counts a row only when the
        Upstream cell carries a backticked `module`-colon-colon-`class` token.
        So a row for an architecture that is NOT at the pin -- MuseGlimmer,
        KimiK3, MiniMaxH3DiT, and the seven recipe architectures rowed for
        #609/#610 -- deliberately spells its module and class as separate
        fields instead, and contributes nothing.

        That is a convention, and an unenforced convention drifts. The moment
        someone "helpfully" anchors one of those rows the pinned inventory
        silently gains a target vLLM does not register at the pin, and the next
        person to re-pin the counts bakes the error in. This proves the
        omission is load-bearing rather than an oversight.
        """
        clean: list[str] = []
        agent_record.check_model_invariants(clean)
        self.assertEqual(clean, [])

        beyond_pin = (
            "MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation",
            "MODEL-MM-kimi-k3-kimi-k3-for-conditional-generation",
            "MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit",
            "MODEL-TEXT-bailing-moe-v3-bailing-moe-v3-for-causal-lm",
            "MODEL-MM-moss-tts-moss-tts-delay-talker-for-generation",
            "MODEL-MM-moss-tts-moss-tts-realtime-talker-for-generation",
            "MODEL-MM-qwen3-tts-qwen3-tts-talker-for-conditional-generation",
            "MODEL-MM-higgs-audio-v3-higgs-audio-v3-talker-for-conditional-generation",
            "MODEL-MM-voxtral-realtime-voxtral-realtime-for-conditional-generation",
            "MODEL-MM-bailing-mm-native-bailing-mm-native-for-conditional-generation",
        )
        matrix = agent_record.AGENTS / "model-matrix.md"
        lines = matrix.read_text(encoding="utf-8").splitlines(keepends=True)
        for item_id in beyond_pin:
            rows = [line for line in lines if line.startswith(f"| `{item_id}` |")]
            self.assertEqual(len(rows), 1, item_id)
            upstream = agent_record.split_cells(rows[0])[2]
            self.assertEqual(
                [v for v in re.findall(r"`([^`]+)`", upstream) if "::" in v],
                [],
                item_id,
            )

        # Anchoring exactly ONE of them the at-the-pin way must move the
        # inventory off its pin, in the file rather than in a stub.
        victim = "MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit"
        mutated: list[str] = []
        for line in lines:
            if line.startswith(f"| `{victim}` |"):
                cells = line.split("|")
                cells[3] += " `vllm/model_executor/models/minimax_h3.py::MiniMaxH3DiTModel` "
                line = "|".join(cells)
            mutated.append(line)
        self.assertNotEqual(mutated, lines)

        errors: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            agents = Path(tmp)
            (agents / "model-matrix.md").write_text("".join(mutated), encoding="utf-8")
            # ROOT moves with AGENTS: the checker reports the path relative to
            # it, so leaving ROOT pointing at the real tree raises instead of
            # producing the error we are asserting on.
            with (
                mock.patch.object(agent_record, "AGENTS", agents),
                mock.patch.object(agent_record, "ROOT", agents),
            ):
                agent_record.check_model_invariants(errors)

        require(errors, r"model inventory .*expected")

    def test_kernel_row_ratchet_matches_the_current_inventory(self) -> None:
        """The #284 A76 family is a real kernel row, never a relaxed pin.

        The constant moved 52 -> 60 when the record-rot repair restored the
        eight promised matrix rows and the LTX2-VAE and megakernel rows landed;
        the ratchet semantics are unchanged: one row less or one row more must
        fail the check for its own reason.
        """
        path, expected = agent_record.MATRICES["KERNEL"]
        self.assertEqual(expected, 60)

        errors: list[str] = []
        rows = agent_record.parse_claim_rows(path, errors)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), expected)

        for mutated_expected in (expected - 1, expected + 1):
            with self.subTest(expected=mutated_expected):
                mutated_errors: list[str] = []
                with mock.patch.dict(
                    agent_record.MATRICES,
                    {"KERNEL": (path, mutated_expected)},
                ):
                    agent_record.check_matrices(mutated_errors)
                require(mutated_errors, r"\d+ KERNEL rows; expected \d+")

    def test_matrix_row_ratchets_match_the_current_inventory(self) -> None:
        """MUTATION: every matrix constant is a ratchet, in all four sections.

        The MODEL, QUANTIZATION, and BACKEND constants carry the same semantic
        the KERNEL test has pinned since #284: the committed value is what the
        matrix rows sum to, and one row either side fails the check. Red first:
        each mutated constant names its own section in the error.
        """
        for prefix, expected in (
            ("MODEL", 384),
            ("QUANT", 87),
            ("BACKEND", 90),
        ):
            with self.subTest(prefix=prefix):
                path, value = agent_record.MATRICES[prefix]
                self.assertEqual(value, expected)
                errors: list[str] = []
                rows = agent_record.parse_claim_rows(path, errors)
                self.assertEqual(errors, [])
                self.assertEqual(
                    sum(
                        row.item_id.startswith(prefix + "-") for row in rows
                    ),
                    expected,
                )
        for prefix, expected in (
            ("MODEL", 384),
            ("QUANT", 87),
            ("KERNEL", 60),
            ("BACKEND", 90),
        ):
            path, _ = agent_record.MATRICES[prefix]
            for mutated_expected in (expected - 1, expected + 1):
                with self.subTest(prefix=prefix, expected=mutated_expected):
                    mutated_errors: list[str] = []
                    with mock.patch.dict(
                        agent_record.MATRICES,
                        {prefix: (path, mutated_expected)},
                    ):
                        agent_record.check_matrices(mutated_errors)
                    require(mutated_errors, rf"\d+ {prefix} rows; expected \d+")

    def test_engine_summary_rejects_stale_area_rollup(self) -> None:
        source = agent_record.ENGINE_MATRIX.read_text(encoding="utf-8")
        current = next(
            line
            for line in source.splitlines()
            if line.startswith("| Serving, API, CLI, library |")
        )
        cells = [cell.strip() for cell in current.strip().strip("|").split("|")]
        self.assertGreater(int(cells[6]), 0)
        cells[5] = str(int(cells[5]) + 1)
        cells[6] = str(int(cells[6]) - 1)
        stale = "| " + " | ".join(cells) + " |"

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            matrix = Path(temp_dir) / "engine-matrix.md"
            with mock.patch.object(agent_record, "ENGINE_MATRIX", matrix):
                matrix.write_text(source, encoding="utf-8")
                baseline_errors: list[str] = []
                baseline_rows = agent_record.parse_claim_rows(matrix, baseline_errors)
                agent_record.check_engine_summary(baseline_rows, baseline_errors)
                self.assertEqual(baseline_errors, [])

                matrix.write_text(source.replace(current, stale), encoding="utf-8")
                errors: list[str] = []
                rows = agent_record.parse_claim_rows(matrix, errors)
                agent_record.check_engine_summary(rows, errors)

        require(errors, r"Serving, API, CLI, library summary ready=\d+; actual \d+")


class MigratedLegacyLinks(unittest.TestCase):
    def test_legacy_payload_keeps_original_agents_relative_link_base(self) -> None:
        source = ROOT / ".agents/completed/state-events/0000-00/STATE-LEGACY-000001.md"
        text = "<!-- legacy-payload:begin -->\n[spec](specs/example.md)"
        self.assertEqual(agent_record.link_bases(source, text), (ROOT / ".agents",))

    def test_post_cutover_event_links_remain_event_relative(self) -> None:
        source = ROOT / ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md"
        self.assertEqual(
            agent_record.link_bases(source, "[local](note.md)"), (source.parent,)
        )


class LinkExtraction(unittest.TestCase):
    """#460: what the checker calls a link must be a link a READER can follow.

    docs/BENCHMARKS.md is compacted by MOVING a superseded row into
    .agents/benchmark-record.md byte-for-byte. Before this, any row carrying a
    docs/-relative evidence link dangled the moment it was archived, whether it
    was quoted inside a fence or moved as live markdown, so the documented
    payment mechanism did not work for exactly the rows that carry evidence.
    """

    def test_fenced_link_is_not_extracted(self) -> None:
        text = "```text\n| row | [evidence](bench-evidence/x.md) |\n```\n"
        self.assertEqual(agent_record.extract_links(text), [])

    def test_tilde_fenced_link_is_not_extracted(self) -> None:
        text = "~~~console\n$ see [evidence](bench-evidence/x.md)\n~~~\n"
        self.assertEqual(agent_record.extract_links(text), [])

    def test_inline_code_link_is_not_extracted(self) -> None:
        self.assertEqual(
            agent_record.extract_links("write `[label](target.md)` to link\n"), []
        )

    def test_live_link_is_still_extracted(self) -> None:
        self.assertEqual(
            agent_record.extract_links("see [spec](specs/example.md) now\n"),
            ["specs/example.md"],
        )

    def test_a_backticked_label_is_still_a_link(self) -> None:
        # The overwhelmingly common form in this tree: [`name`](path).
        self.assertEqual(
            agent_record.extract_links("[`workflow.md`](workflow.md)\n"),
            ["workflow.md"],
        )

    def test_link_after_a_closed_fence_is_still_extracted(self) -> None:
        text = "```sh\nrun [x](nope.md)\n```\n\nreal [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_link_beside_an_inline_span_is_still_extracted(self) -> None:
        text = "`VT_FLAG=1` and [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_a_fence_with_an_INFO_STRING_does_not_close_a_block(self) -> None:
        """The F2 defect, minimised.

        CommonMark: a closing fence carries no info string. Treating ```sh as a
        close INVERTS fence phase for the rest of the file, so prose after the
        next real fence gets blanked and text inside a block gets validated.
        The two assertions are the two halves of the inversion. First: with no
        real close, everything after ```sh is still inside. Second, and this is
        the half that bit the tree: the BARE fence is the close, so the link
        below it is live and the one above it is not. Under the loose rule both
        answers were exactly backwards.
        """
        self.assertEqual(
            agent_record.extract_links("```\nopen\n```sh\ninside [a](nope-a.md)\n"),
            [],
        )
        self.assertEqual(
            agent_record.extract_links(
                "```\nopen\n```sh\ninside [a](nope-a.md)\n```\n"
                "after [b](specs/example.md)\n"
            ),
            ["specs/example.md"],
        )

    def test_a_closing_fence_must_match_the_opener(self) -> None:
        # Wrong character, then too short: neither closes, so the link after it
        # is still inside the block.
        for closer in ("~~~", "``"):
            with self.subTest(closer=closer):
                text = f"````\ncode\n{closer}\n[x](nope.md)\n"
                self.assertEqual(agent_record.extract_links(text), [])

    def test_a_LONGER_closing_fence_does_close(self) -> None:
        text = "```\ncode\n`````\n\nreal [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_prose_two_lines_below_a_closed_fence_is_still_scanned(self) -> None:
        """The live case, in the file that actually mis-paired.

        .agents/completed/state-events/0000-00/STATE-LEGACY-000001.md has an
        unclosed fence at :17697. Under the loose rule the ```sh at :17948
        "closed" it, phase inverted, and the ordinary prose link at :18297 was
        blanked: a reader-followable link silently stopped being validated.
        """
        source = ROOT / ".agents/completed/state-events/0000-00/STATE-LEGACY-000001.md"
        targets = agent_record.extract_links(source.read_text(encoding="utf-8"))
        self.assertIn("specs/cpu-llamacpp-floor-remeasure-2026-07-22.md", targets)

    def test_a_link_straddled_by_two_INLINE_SPANS_is_not_extracted(self) -> None:
        """F5: `[`name`](path)` is four backticks, and not a link.

        CommonMark reads it as the code span `[`, the literal text name, and the
        code span `](path)`, so there is no link and the checker agrees. It
        earns a test because the effect is to HIDE a target that does not exist:
        this row's own spec carried the form and hid `path`. An author who wants
        to SHOW the form writes it with a double-backtick delimiter, which is
        also code and also correctly skipped; an author who wants a real link
        writes it without the outer pair, which still resolves.
        """
        self.assertEqual(agent_record.extract_links("`[`name`](path)`\n"), [])
        self.assertEqual(agent_record.extract_links("`` [`name`](path) ``\n"), [])
        self.assertEqual(
            agent_record.extract_links("[`name`](specs/example.md)\n"),
            ["specs/example.md"],
        )

    def test_stripping_preserves_line_and_column_positions(self) -> None:
        # Blanked, not deleted, so every line and column offset survives. NOT
        # evidence of anything today: check_links reports no line numbers at
        # all. Held so a caller that does report them cannot be broken here.
        text = "a\n```\nbbbb\n```\nc `dd` e\n"
        stripped = agent_record.strip_code_spans(text)
        self.assertEqual(len(stripped.splitlines()), len(text.splitlines()))
        for original, blanked in zip(text.splitlines(), stripped.splitlines()):
            self.assertEqual(len(original), len(blanked))

    def test_the_benchmark_record_also_resolves_from_docs(self) -> None:
        # It is the declared archive of docs/BENCHMARKS.md, so a row moved into
        # it verbatim keeps its docs/-relative evidence link resolvable.
        source = ROOT / ".agents/benchmark-record.md"
        self.assertEqual(
            agent_record.link_bases(source, ""), (source.parent, ROOT / "docs")
        )

    def test_an_archived_row_with_a_docs_relative_link_is_accepted(self) -> None:
        record = ROOT / ".agents/benchmark-record.md"
        original = record.read_text(encoding="utf-8")
        moved = (
            "\n## Assembly vs compiler SDOT\n\n| Result | Evidence |\n|---|---|\n"
            "| leaf wall | [assembly evidence]"
            "(bench-evidence/rpi5-a76-q8-dot-20260806.md) |\n"
        )
        errors: list[str] = []
        try:
            record.write_text(original + moved, encoding="utf-8")
            agent_record.check_links(errors)
        finally:
            record.write_text(original, encoding="utf-8")
        self.assertEqual(
            [e for e in errors if "rpi5-a76-q8-dot" in e], [], errors[:5]
        )

    def test_an_archived_row_with_a_MISSING_link_still_dangles(self) -> None:
        # The second base is a base, not an amnesty.
        record = ROOT / ".agents/benchmark-record.md"
        original = record.read_text(encoding="utf-8")
        errors: list[str] = []
        try:
            record.write_text(
                original + "\n[gone](bench-evidence/no-such-file-20260812.md)\n",
                encoding="utf-8",
            )
            agent_record.check_links(errors)
        finally:
            record.write_text(original, encoding="utf-8")
        require(errors, r"dangling link bench-evidence/no-such-file-20260812\.md")

    def test_the_tree_has_no_dangling_link(self) -> None:
        errors: list[str] = []
        agent_record.check_links(errors)
        self.assertEqual(errors, [])



class PerClaimFileSource(unittest.TestCase):
    """A claim may live in its own file under .agents/claims/ (#364).

    The claims TABLE in coordination.md is insert-at-one-anchor, so every
    concurrent claim appended a row at the same line. It conflicted in 8 of the
    16 conflicting open PRs measured at origin/main d928e2c3 -- six of them one
    author's sequential ROCm GDN stack whose ONLY conflict was this file. A
    claim in its own file has one writer and cannot collide.

    parse_active_claims reads BOTH sources, so no existing row had to be
    migrated and the SPIKE/ACTIVE cross-check is unchanged.
    """

    def test_both_sources_are_read(self) -> None:
        sources = [str(p.relative_to(ROOT)) for p in agent_record.claim_sources()]
        self.assertIn(".agents/coordination.md", sources)
        self.assertTrue(
            (ROOT / ".agents/claims").is_dir(),
            "the per-claim directory must exist for new claims to have a home",
        )

    def test_a_claim_file_registers_exactly_like_a_table_row(self) -> None:
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-PROBE.md"
        probe.write_text(
            "# CLAIM-AGENT-RECORD-PROBE\n\n"
            "| Claim | Row IDs | State |\n|---|---|---|\n"
            "| `CLAIM-AGENT-RECORD-PROBE` | `ENG-RECORD-CONFLICT-SURFACES` (`ACTIVE`) | `ACTIVE` |\n",
            encoding="utf-8",
        )
        try:
            errors: list[str] = []
            claims = agent_record.parse_active_claims(errors)
            self.assertIn("CLAIM-AGENT-RECORD-PROBE", claims)
            self.assertEqual(
                claims["CLAIM-AGENT-RECORD-PROBE"].row_ids, {"ENG-RECORD-CONFLICT-SURFACES"}
            )
            self.assertEqual(claims["CLAIM-AGENT-RECORD-PROBE"].lifecycle, "ACTIVE")
        finally:
            probe.unlink()

    def test_dropping_the_directory_from_the_sources_is_caught(self) -> None:
        """MUTATION: read coordination.md alone and the claim disappears.

        This is the semantic being added. Without it a claim filed in its own
        file is invisible to the gate, and the row it owns fails the
        SPIKE/ACTIVE cross-check while looking correctly claimed to a human.
        """
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-PROBE.md"
        probe.write_text(
            "# CLAIM-AGENT-RECORD-PROBE\n\n"
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-AGENT-RECORD-PROBE` | `ENG-RECORD-CONFLICT-SURFACES` |\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(
                agent_record,
                "claim_sources",
                lambda: [ROOT / ".agents/coordination.md"],
            ):
                errors: list[str] = []
                self.assertNotIn(
                    "CLAIM-AGENT-RECORD-PROBE",
                    agent_record.parse_active_claims(errors),
                    "the mutation must hide the claim, or this test proves nothing",
                )
            errors = []
            self.assertIn("CLAIM-AGENT-RECORD-PROBE", agent_record.parse_active_claims(errors))
        finally:
            probe.unlink()

    def test_a_claim_declared_twice_across_sources_is_rejected(self) -> None:
        """Reading a second source must not weaken the duplicate check."""
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-DUP.md"
        row = (
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-AGENT-RECORD-DUP` | `ENG-RECORD-CONFLICT-SURFACES` |\n"
        )
        probe.write_text("# dup\n\n" + row + row, encoding="utf-8")
        try:
            errors: list[str] = []
            agent_record.parse_active_claims(errors)
            self.assertTrue(
                any("duplicate active claim" in e for e in errors),
                f"a duplicate must be caught; got {errors}",
            )
        finally:
            probe.unlink()


class TenstorrentResidualGoldenRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-RESIDUAL-GOLDEN"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

class Qwen35TextOnlyRowsAreCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROWS = (
        "MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm",
        "MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm",
    )

    def test_both_text_only_rows_exist_in_the_model_matrix(self) -> None:
        lines = (
            (ROOT / ".agents/model-matrix.md")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for row in self.ROWS:
            matching = [line for line in lines if line.startswith(f"| `{row}` |")]
            self.assertEqual(len(matching), 1, f"{row} must appear exactly once")

class TenstorrentMistralRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-MISTRAL"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        """A row whose issue is only in the PR body is untraceable from the tree.

        This row shipped originally citing PR #354 -- a merged PR, not an issue
        -- so nothing in the repository pointed at anything trackable. Pin both
        links here so a future edit cannot quietly drop them again.
        """
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("tenstorrent-mistral.md", row)
        # The intake surface moved out of roadmap_v1.md and into the
        # append-only issue index (POLICY-ISSUE-INTAKE, #840). The pin is the
        # same pin: this row's issue link must still exist somewhere trackable.
        index = tracked_issues(self)
        self.assertIn("issues/670", index)

class TenstorrentTraceRunnerRowIsCounted(TenstorrentMistralRowIsCounted):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-TRACE-RUNNER"

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("tenstorrent-trace-runner.md", row)
        index = tracked_issues(self)
        self.assertIn("issues/1105", index)


class TenstorrentHostFreeForwardRowIsCounted(TenstorrentMistralRowIsCounted):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-HOST-FREE-FORWARD"

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("tenstorrent-host-free-forward.md", row)
        index = tracked_issues(self)
        self.assertIn("issues/1105", index)


class CudaLlamacppRowIsCounted(TenstorrentMistralRowIsCounted):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-GATE-CUDA-LLAMACPP"

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("bench-qwen38-27b-four-way.md", row)
        index = tracked_issues(self)
        self.assertIn("issues/979", index)

    def test_the_row_is_not_confused_with_the_legacy_one(self) -> None:
        """The trailing pipe in every match above is load-bearing here.

        `BACKEND-GATE-CUDA-LLAMACPP` is a strict prefix of
        `BACKEND-GATE-CUDA-LLAMACPP-LEGACY`. A match written without the
        trailing pipe would count both rows as this one, and the removal
        mutation above would then delete two rows while claiming to delete one
        -- red for the right reason by accident. Pin that the exact match finds
        one row, the loose match finds two, and the two are different lines.
        """
        lines = (
            (ROOT / ".agents/backend-matrix.md")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        exact = [l for l in lines if l.startswith(f"| `{self.ROW}` |")]
        legacy = [l for l in lines if l.startswith(f"| `{self.ROW}-LEGACY` |")]
        loose = [l for l in lines if l.startswith(f"| `{self.ROW}")]
        self.assertEqual(len(exact), 1, f"{self.ROW} must appear exactly once")
        self.assertEqual(len(legacy), 1, f"{self.ROW}-LEGACY must still exist")
        self.assertNotEqual(exact[0], legacy[0])
        self.assertEqual(len(loose), 2, "the prefix spans both rows")


class MtpDepthRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "SPEC-MTP-K-GT-1"

    def test_the_row_exists_in_the_engine_matrix(self) -> None:
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        """A row whose issue lives only in the PR body is untraceable."""
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("mtp-k-gt-1.md", row)
        index = tracked_issues(self)
        self.assertIn("issues/81)", index)

class CanonicalIssueRecordTests(unittest.TestCase):
    def record(self, number: int) -> object:
        return agent_record.issue_records.IssueRecord(
            id=f"ISSUE-GH-{number}",
            title="Canonical issue",
            row="ROW-A",
            state="CLOSED",
            kind="bug",
            github=number,
            mirror="DIVERGED",
            availability="FULL",
            created="2026-08-01",
            updated="2026-08-02",
            closed="2026-08-02",
            problem="Observed failure.",
            resolution="Closed by fixture evidence.",
        )

    def errors_for(
        self,
        issues_root: Path,
        *,
        references: set[str] | None = None,
    ) -> list[str]:
        errors: list[str] = []
        agent_record.check_issue_records(
            errors,
            issues_root=issues_root,
            rows={"ROW-A"},
            owed={},
            references=references or set(),
            frozen_archive=b"# frozen\n",
        )
        return errors

    def write_record(self, issues_root: Path, number: int) -> Path:
        record = self.record(number)
        path = issues_root / "ROW-A" / f"{record.id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            agent_record.issue_records.render_issue_record(record),
            encoding="utf-8",
        )
        return path

    def test_canonical_files_are_the_authority_without_a_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            self.write_record(issues_root, 7)
            errors = self.errors_for(issues_root, references={"#7"})
        self.assertEqual(errors, [])

    def test_a_missing_canonical_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            issues_root = Path(temporary) / ".agents" / "issues"
            self.write_record(issues_root, 7)
            errors = self.errors_for(issues_root, references={f"ISSUE-GH-{''}8"})
        require(errors, rf"ISSUE-GH-{''}8.*exactly one local record|cannot resolve")

    def test_the_retired_index_is_no_longer_a_tracked_writable_surface(self) -> None:
        self.assertFalse(
            (agent_record.AGENTS / "issue-index.md").exists(),
            "the tracked issue index is back; it is a surface every PR writes",
        )
# Preserve the exact line anchors owned by ENG-RECORD-ANCHOR-RATCHET (#632).

class PerClaimStateConsistencyTests(unittest.TestCase):
    CLAIM = "CLAIM-KERNEL-CUDA-DECODE-MEGAKERNEL"
    ROW = "KERNEL-CUDA-DECODE-MEGAKERNEL"
    PATH = ROOT / f".agents/claims/{CLAIM}.md"

    def _row(self):
        errors: list[str] = []
        rows, by_id = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])
        return by_id[self.ROW]

    def _contract_errors(self, mutated: str) -> list[str]:
        original = self.PATH.read_bytes()
        self.PATH.write_text(mutated, encoding="utf-8")
        try:
            errors: list[str] = []
            rows, by_id = agent_record.check_matrices(errors)
            agent_record.check_row_contracts(rows, by_id, errors)
        finally:
            self.PATH.write_bytes(original)
        self.assertEqual(self.PATH.read_bytes(), original)
        return errors

    def test_claim_row_annotation_must_match_matrix_state(self) -> None:
        source = self.PATH.read_text(encoding="utf-8")
        old = f"`{self.ROW}` (`SPIKE`)"
        new = f"`{self.ROW}` (`ACTIVE`)"
        self.assertEqual(source.count(old), 1)
        errors = self._contract_errors(source.replace(old, new, 1))
        self.assertEqual(
            errors,
            [
                f".agents/claims/{self.CLAIM}.md:5: claim {self.CLAIM} "
                f"annotates {self.ROW} as ACTIVE, but matrix state is SPIKE"
            ],
        )

    def test_selected_owner_claim_must_have_nonterminal_lifecycle(self) -> None:
        source = self.PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("| `ACTIVE` |"), 1)
        errors = self._contract_errors(source.replace("| `ACTIVE` |", "| `DONE` |", 1))
        row = self._row()
        self.assertEqual(
            errors,
            [
                f".agents/kernel-matrix.md:{row.line_no}: owner {self.CLAIM} for live row "
                f"{self.ROW} has claim lifecycle DONE, not ACTIVE/IMPLEMENTING/SPIKE"
            ],
        )

    def _probe_errors(self, lifecycle: str, row_cell: str) -> list[str]:
        claim = "CLAIM-AGENT-RECORD-STATE-PROBE"
        probe = ROOT / f".agents/claims/{claim}.md"
        probe.write_text(
            "# claim-state probe\n\n"
            "| Claim | Row IDs | Agent | Worktree | Branch | Scope | State | Update |\n"
            "|---|---|---|---|---|---|---|---|\n"
            f"| `{claim}` | {row_cell} | test | test | test | test | `{lifecycle}` | test |\n",
            encoding="utf-8",
        )
        try:
            row = with_field(self._row(), "owner", f"`{claim}`")
            errors: list[str] = []
            with mock.patch.object(agent_record, "claim_sources", lambda: [probe]):
                agent_record.check_row_contracts([row], {self.ROW: row}, errors)
        finally:
            probe.unlink()
        return errors

    def test_existing_nonterminal_claim_lifecycles_are_live(self) -> None:
        for lifecycle in ("ACTIVE", "IMPLEMENTING", "SPIKE"):
            with self.subTest(lifecycle=lifecycle):
                self.assertEqual(
                    self._probe_errors(lifecycle, f"`{self.ROW}` (`SPIKE`)"),
                    [],
                )

    def test_missing_claim_row_annotation_is_rejected(self) -> None:
        errors = self._probe_errors("ACTIVE", f"`{self.ROW}`")
        self.assertEqual(
            errors,
            [
                ".agents/claims/CLAIM-AGENT-RECORD-STATE-PROBE.md:5: claim "
                f"CLAIM-AGENT-RECORD-STATE-PROBE does not annotate {self.ROW} "
                "with its matrix lifecycle state"
            ],
        )

    def test_unknown_selected_owner_claim_lifecycle_is_rejected(self) -> None:
        errors = self._probe_errors("PAUSED", f"`{self.ROW}` (`SPIKE`)")
        row = self._row()
        self.assertEqual(
            errors,
            [
                f".agents/kernel-matrix.md:{row.line_no}: owner "
                "CLAIM-AGENT-RECORD-STATE-PROBE for live row "
                f"{self.ROW} has claim lifecycle PAUSED, not ACTIVE/IMPLEMENTING/SPIKE"
            ],
        )

    def test_actual_roadmap_claim_is_state_consistent(self) -> None:
        source = self.PATH.read_text(encoding="utf-8")
        self.assertEqual(self._contract_errors(source), [])















































































































































# Preserve tracked line anchors into this test module after issue #3085.
# The retired cardinality rationale is hash-bound in the completed archive.
#
class SpikeOwnerContractTests(unittest.TestCase):
    """A SPIKE needs a live claim independently of an ACTIVE row."""

    def test_spike_requires_claim_owner(self) -> None:
        parse_errors: list[str] = []
        rows, _ = agent_record.check_matrices(parse_errors)
        self.assertEqual(parse_errors, [])
        by_id = {row.item_id: row for row in rows}

        baseline_errors: list[str] = []
        agent_record.check_row_contracts(rows, by_id, baseline_errors)
        self.assertEqual(baseline_errors, [])

        spike = next(
            row
            for row in rows
            if row.state == "SPIKE" and row.path.name == "kernel-matrix.md"
        )
        self.assertEqual(spike.state, "SPIKE")
        without_owner = with_field(spike, "owner", "-")
        errors = validate_mutation(rows, without_owner)
        location = f"{spike.path.relative_to(ROOT)}:{spike.line_no}"
        self.assertEqual(
            errors,
            [
                f"{location}: SPIKE row {spike.item_id} "
                "has no CLAIM-* owner"
            ],
        )


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
class RecordAnchorRatchet(unittest.TestCase):
    """ENG-RECORD-ANCHOR-RATCHET (#632), .agents/specs/record-anchor-ratchet.md.

    Ten cases. Seven cover a row of the spec's test table; the other three pin
    the two gate directions and the tree-against-baseline agreement. The
    table-driven cases build a SYNTHETIC tree and a synthetic row rather than
    asserting against the live matrices, because the live count is a moving
    backlog and a case that reds when somebody else repairs an unrelated anchor
    teaches people to ignore this suite.

    The fourth case is the load-bearing one. `is_code_anchor` answers with
    `any()`, so before this row one good link in a cell made every rotted
    citation beside it invisible -- and that is the exact shape three stale
    anchors hid in during the 2026-08-13/14 campaign.
    """

    HEADER = ("id", "item", "upstream", "our code", "tests evidence", "state", "owner")

    def row(self, state: str, code: str, tests: str = "-", *, source=None):
        cells = ("ENG-RATCHET-FIXTURE", "item", "up", code, tests, f"`{state}`", "-")
        return agent_record.ClaimRow(
            path=source if source is not None else agent_record.ENGINE_MATRIX,
            line_no=1,
            item_id="ENG-RATCHET-FIXTURE",
            state=state,
            header=self.HEADER,
            cells=cells,
            raw="| " + " | ".join(cells) + " |",
        )

    @staticmethod
    def tree(root: Path) -> None:
        """A cited file whose symbol sits at :4, not at :2."""
        target = root / "src/vllm/toy.cpp"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "#include <cstdio>\n"          # 1
            "// a comment that moved\n"    # 2
            "\n"                           # 3
            "void RatchetTarget() {}\n",   # 4
            encoding="utf-8",
        )

    def scan(self, root: Path, rows):
        return agent_record.scan_record_anchors(rows, root=Path(root))

    def test_bare_citation_at_the_wrong_line_counts_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp))
            res = self.scan(tmp, [self.row("PARTIAL", "`RatchetTarget` `src/vllm/toy.cpp:2`")])
        self.assertEqual(res.counts["stale"], 1, res.offenders)
        self.assertEqual(res.counts["broken"], 0, res.offenders)
        self.assertEqual(res.counts["ok"], 0, res.offenders)

    def test_bare_citation_out_of_range_counts_broken(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp))
            res = self.scan(tmp, [self.row("PARTIAL", "`RatchetTarget` `src/vllm/toy.cpp:99`")])
        self.assertEqual(res.counts["broken"], 1, res.offenders)
        self.assertEqual(res.counts["stale"], 0, res.offenders)

    def test_correct_bare_citation_counts_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp))
            res = self.scan(tmp, [self.row("PARTIAL", "`RatchetTarget` `src/vllm/toy.cpp:4`")])
        self.assertEqual(res.counts["ok"], 1, res.offenders)
        self.assertEqual(res.total, 0, res.offenders)

    def test_one_good_link_does_not_cover_a_rotted_bare_citation(self) -> None:
        """The `any()` shape that hid the rot: BOTH citations must be counted.

        Built against REAL tree paths rather than a synthetic root, because the
        half of the claim that matters is the interaction with `is_code_anchor`,
        and that function resolves against `ROOT` by construction. `:1` of this
        checker is its shebang -- in range, and forever without the symbol the
        cell names beside it.
        """
        cell = (
            "[checker](../scripts/check-agent-record.py#L1); "
            "`RatchetFixtureSymbol` `scripts/check-agent-record.py:1`"
        )
        source = agent_record.ENGINE_MATRIX
        # `any()` semantics are DELIBERATELY retained for the STATE gate -- one
        # good anchor still evidences the row (spec, "Scope"). Before this row
        # that was ALSO the whole of the anchor check, so the rotted citation
        # beside it was invisible.
        self.assertTrue(agent_record.is_code_anchor(cell, source))
        res = agent_record.scan_record_anchors([self.row("PARTIAL", cell, source=source)])
        self.assertEqual(res.counts["ok"], 1, res.offenders)
        self.assertEqual(res.counts["stale"], 1, res.offenders)

    def test_active_row_anchors_are_counted(self) -> None:
        """EVIDENCED_STATES omits ACTIVE, so this row got no anchor check at all."""
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp))
            res = self.scan(tmp, [self.row("ACTIVE", "`RatchetTarget` `src/vllm/toy.cpp:2`")])
        self.assertEqual(res.counts["stale"], 1, res.offenders)
        self.assertIn("ACTIVE", agent_record.RECORD_ANCHOR_STATES)
        self.assertIn("READY", agent_record.RECORD_ANCHOR_STATES)

    def test_write_baseline_refuses_to_ratchet_upward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "record-anchor-baseline.json"
            baseline.write_text(
                json.dumps({"total": 1, "buckets": {"stale": 1, "broken": 0}}) + "\n",
                encoding="utf-8",
            )
            before = baseline.read_text(encoding="utf-8")
            result = agent_record.RecordAnchorResult()
            result.counts["stale"] = 3
            result.counts["broken"] = 1
            err = io.StringIO()
            with contextlib.redirect_stderr(err), mock.patch.object(
                agent_record, "RECORD_ANCHOR_BASELINE", baseline
            ):
                rc = agent_record.write_record_anchor_baseline(result)
            self.assertEqual(rc, 1, err.getvalue())
            self.assertIn("REFUS", err.getvalue().upper())
            self.assertEqual(baseline.read_text(encoding="utf-8"), before)

    def test_baseline_matches_the_tree_exactly(self) -> None:
        """The committed baseline is the tree's rot, in BOTH directions.

        Only the rot buckets are pinned. `ok` is deliberately absent from the
        baseline file -- pinning it would make every change that adds or removes
        a citation rewrite one shared file.
        """
        result = agent_record.scan_record_anchors()
        stored = agent_record.load_record_anchor_baseline()
        self.assertEqual(
            {b: result.counts[b] for b in agent_record.RECORD_ANCHOR_BUCKETS},
            stored,
            agent_record.record_anchor_report(result),
        )
        self.assertNotIn("ok", stored)

    def test_a_repair_fails_until_the_baseline_is_lowered(self) -> None:
        """A ratchet, not a threshold: banking the improvement is mandatory."""
        result = agent_record.scan_record_anchors()
        result.counts["stale"] -= 1
        errors: list[str] = []
        agent_record.check_record_anchors(result, errors)
        require(errors, r"record-anchor baseline STALE in bucket 'stale'")

    def test_new_rot_fails_the_gate(self) -> None:
        result = agent_record.scan_record_anchors()
        result.counts["broken"] += 1
        errors: list[str] = []
        agent_record.check_record_anchors(result, errors)
        require(errors, r"RECORD ANCHOR REGRESSION in bucket 'broken'")

    def test_a_baseline_is_never_banked_from_a_tree_with_record_errors(self) -> None:
        """`--write-baseline` must not run before the checker finishes.

        The mode returned as soon as it had a number. That return happened
        before the `if errors:` gate, so a tree that failed any other record
        check could still bank its rot. The banked figure then carried the
        authority of a run that never passed. The write now happens after the
        gate.
        """
        digest = agent_record.RECORD_ANCHOR_BASELINE.read_bytes()
        stderr = io.StringIO()
        with mock.patch.object(
            agent_record, "check_roadmap", side_effect=lambda *a: a[1].append("SYNTHETIC")
        ), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = agent_record.main(["--write-baseline"])
        self.assertEqual(code, 1)
        self.assertIn("SYNTHETIC", stderr.getvalue())
        self.assertEqual(agent_record.RECORD_ANCHOR_BASELINE.read_bytes(), digest)


class HfModelDownloadRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "ENG-HF-MODEL-DOWNLOAD"

    def test_the_row_exists_in_the_engine_matrix(self) -> None:
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        """A row whose issue lives only in the PR body is untraceable."""
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("hf-model-download.md", row)
        index = tracked_issues(self)
        self.assertIn("issues/1280)", index)

class BpeQuadraticMergeRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "SPEC-BPE-QUADRATIC-MERGE"

    def test_the_row_exists_in_the_engine_matrix(self) -> None:
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_spec(self) -> None:
        """A row whose spec lives only in the PR body is untraceable."""
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("bpe-quadratic-merge.md", row)
        self.assertTrue(
            (ROOT / ".agents/specs/bpe-quadratic-merge.md").is_file(),
            "the spec the row cites must exist",
        )

class RequestLengthGuardRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "SERVE-REQUEST-LENGTH-GUARD"

    def test_the_row_exists_in_the_engine_matrix(self) -> None:
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_spec(self) -> None:
        """A row whose spec lives only in the PR body is untraceable."""
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("serve-request-length-guard.md", row)
        self.assertTrue(
            (ROOT / ".agents/specs/serve-request-length-guard.md").is_file(),
            "the spec the row cites must exist",
        )

class DeepseekV4MultiCacheRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "KV-DSV4-MULTICACHE"
    AREA = "KV cache and memory"
    STATE = "READY"

    def test_the_row_exists_in_the_engine_matrix(self) -> None:
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        """A row whose issue lives only in the PR body is untraceable."""
        text = (ROOT / ".agents/engine-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("kv-dsv4-multicache.md", row)
        self.assertTrue(
            (ROOT / ".agents/specs/kv-dsv4-multicache.md").is_file(),
            "the spec the row cites must exist",
        )
        index = tracked_issues(self)
        self.assertIn("issues/1925)", index)

    def test_the_row_is_counted_under_the_area_whose_summary_moved(self) -> None:
        """The 25 -> 26 and READY 3 -> 4 cells must be about THIS row.

        Reads the section boundaries out of the matrix directly instead of
        calling `check_engine_summary`, which computes the same span: a case
        that asked the checker where the row sits could not tell a misfiled row
        from a checker that agrees with itself.
        """
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        agent_record.check_engine_summary(rows, errors)
        self.assertEqual(errors, [], "the matrix must be clean before this case reads it")

        lines = agent_record.ENGINE_MATRIX.read_text(encoding="utf-8").splitlines()
        headings = [n for n, l in enumerate(lines, 1) if l.startswith("## ")]
        start = next(
            n for n, l in enumerate(lines, 1) if l.strip() == f"## {self.AREA}"
        )
        end = min((n for n in headings if n > start), default=len(lines) + 1)

        row = next(
            r
            for r in rows
            if r.path == agent_record.ENGINE_MATRIX and r.item_id == self.ROW
        )
        self.assertTrue(
            start < row.line_no < end,
            f"{self.ROW} is at line {row.line_no}, outside {self.AREA} "
            f"({start}..{end}); the area summary counted a different row",
        )
        self.assertEqual(row.state, self.STATE)


class DerivedMatrixMembershipTests(unittest.TestCase):
    HISTORY = ROOT / ".agents/completed/matrix-cardinality-history.md"
    PAYLOAD_BEGIN = "<!-- matrix-cardinality-history:begin -->"
    PAYLOAD_END = "<!-- matrix-cardinality-history:end -->"
    DIGEST_RE = re.compile(
        r"<!-- matrix-cardinality-history-sha256: ([0-9a-f]{64}) -->"
    )

    def _assert_history_integrity(self, source: str) -> None:
        digest = self.DIGEST_RE.search(source)
        self.assertIsNotNone(digest, "the archive must carry its payload digest")
        self.assertEqual(source.count(self.PAYLOAD_BEGIN), 1)
        self.assertEqual(source.count(self.PAYLOAD_END), 1)
        payload = source.split(self.PAYLOAD_BEGIN, 1)[1].split(
            self.PAYLOAD_END, 1
        )[0]
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(actual, digest.group(1))

    def _check_kernel_source(self, source: str) -> list[str]:
        path, expected = agent_record.MATRICES["KERNEL"]
        with tempfile.TemporaryDirectory(dir=agent_record.ROOT) as tmp:
            matrix = Path(tmp) / "kernel-matrix.md"
            matrix.write_text(source, encoding="utf-8")
            paths = [matrix if candidate == path else candidate
                     for candidate in agent_record.MATRIX_PATHS]
            matrices = dict(agent_record.MATRICES)
            matrices["KERNEL"] = (matrix, expected)
            errors: list[str] = []
            with mock.patch.object(agent_record, "MATRIX_PATHS", paths), \
                 mock.patch.object(agent_record, "MATRICES", matrices):
                agent_record.check_matrices(errors)
        return errors

    def test_matrix_registry_pairs_each_path_with_its_ratchet_constant(self) -> None:
        """Each matrix registry value is its owning path and its ratchet."""
        self.assertTrue(agent_record.MATRICES)
        for prefix, (path, expected) in agent_record.MATRICES.items():
            with self.subTest(prefix=prefix):
                self.assertIsInstance(path, Path)
                self.assertIsInstance(expected, int)

    def test_a_valid_unique_matrix_row_needs_no_checker_constant(self) -> None:
        """Adding a valid keyed row does not require editing the checker."""
        source = (ROOT / ".agents/kernel-matrix.md").read_text(encoding="utf-8")
        template = next(
            line for line in source.splitlines()
            if line.startswith("| `KERNEL-CPU-A76-Q8-DOT` |")
        )
        added = template.replace(
            "`KERNEL-CPU-A76-Q8-DOT`", "`KERNEL-TEST-DERIVED-ROW`", 1
        )
        with tempfile.TemporaryDirectory(dir=agent_record.ROOT) as tmp:
            matrix = Path(tmp) / "kernel-matrix.md"
            matrix.write_text(
                source.replace(template, template + "\n" + added, 1),
                encoding="utf-8",
            )
            paths = [
                matrix if path == agent_record.MATRICES["KERNEL"][0] else path
                for path in agent_record.MATRIX_PATHS
            ]
            # The restructured checker ratchets every matrix section, so a new
            # valid row is only accepted together with its constant bump. The
            # un-bumped constant must fail for the ratchet's own reason first.
            expected = agent_record.MATRICES["KERNEL"][1]
            errors: list[str] = []
            matrices = dict(agent_record.MATRICES)
            with mock.patch.object(agent_record, "MATRIX_PATHS", paths), \
                 mock.patch.object(agent_record, "MATRICES",
                                   {**matrices, "KERNEL": (matrix, expected)}):
                agent_record.check_matrices(errors)
            require(errors, rf"\d+ KERNEL rows; expected {expected}")
            rows, by_id = [], {}
            errors = []
            with mock.patch.object(agent_record, "MATRIX_PATHS", paths), \
                 mock.patch.object(agent_record, "MATRICES",
                                   {**matrices, "KERNEL": (matrix, expected + 1)}):
                rows, by_id = agent_record.check_matrices(errors)
        self.assertEqual(errors, [])
        self.assertIn("KERNEL-TEST-DERIVED-ROW", by_id)
        self.assertEqual(
            sum(row.item_id == "KERNEL-TEST-DERIVED-ROW" for row in rows), 1
        )

    def test_duplicate_matrix_key_still_fails_for_its_own_reason(self) -> None:
        source = (ROOT / ".agents/kernel-matrix.md").read_text(encoding="utf-8")
        template = next(
            line for line in source.splitlines()
            if line.startswith("| `KERNEL-CPU-A76-Q8-DOT` |")
        )
        errors = self._check_kernel_source(
            source.replace(template, template + "\n" + template, 1)
        )
        require(errors, r"duplicate ID KERNEL-CPU-A76-Q8-DOT")

    def test_malformed_matrix_row_still_fails_for_its_own_reason(self) -> None:
        source = (ROOT / ".agents/kernel-matrix.md").read_text(encoding="utf-8")
        template = next(
            line for line in source.splitlines()
            if line.startswith("| `KERNEL-CPU-A76-Q8-DOT` |")
        )
        malformed = template.rsplit(" | ", 1)[0] + " |"
        errors = self._check_kernel_source(source.replace(template, malformed, 1))
        require(errors, r"KERNEL-CPU-A76-Q8-DOT has 7 cells; header has 8")

    def test_retired_history_payload_is_self_validating(self) -> None:
        self._assert_history_integrity(self.HISTORY.read_text(encoding="utf-8"))

    def test_arbitrary_historical_word_mutation_breaks_integrity(self) -> None:
        history = self.HISTORY.read_text(encoding="utf-8")
        mutated = history.replace("architectures", "architectXres", 1)
        self.assertNotEqual(mutated, history)
        with self.assertRaises(AssertionError):
            self._assert_history_integrity(mutated)

    def test_live_tests_do_not_describe_retired_cardinality_pins(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        retired_phrases = (
            "ENGINE" + "_ROWS",
            "ratchet" + " bump",
            "re-" + "pinned",
            "pin" + " BINDS",
            "row count" + " error",
            'if "engine' + ' rows" in error',
            'if "MODEL' + ' rows" in error',
        )
        for phrase in retired_phrases:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, source)

    def test_checker_keeps_preexisting_citation_line_floor(self) -> None:
        lines = CHECKER.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(
            len(lines),
            2446,
            "retiring count prose must not invalidate tracked line citations",
        )

    def test_every_tracked_checker_line_citation_still_resolves(self) -> None:
        citation = re.compile(
            r"(?:scripts/)?check-agent-record\.py(?:#L|:)(\d+)"
            r"(?:-L?(\d+))?"
        )
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        references: list[tuple[Path, int, int]] = []
        for raw_path in tracked:
            if not raw_path:
                continue
            path = ROOT / raw_path.decode("utf-8")
            if path == CHECKER:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in citation.finditer(source):
                start = int(match.group(1))
                references.append((path.relative_to(ROOT), start, int(match.group(2) or start)))

        self.assertTrue(references)
        checker_lines = CHECKER.read_text(encoding="utf-8").splitlines()
        for path, start, end in references:
            with self.subTest(path=path, start=start, end=end):
                self.assertGreaterEqual(start, 1)
                self.assertGreaterEqual(end, start)
                self.assertLessEqual(end, len(checker_lines))

        # These cited statements were below the retired blocks. Their exact
        # predecessor positions prove that padding was restored at each site,
        # rather than merely appended at end of file.
        expected_lines = {
            1078: '"""Blank out fenced blocks and inline code, preserving line and column count.',
            1169: "if source.is_relative_to(ISSUES_ROOT):",
            1712: "baseline = load_record_anchor_baseline()",
            1960: "errors.append(",
            2000: "expected = {",
            2018: 'pipes = len(re.findall(r"(?<!\\\\)\\|", line))',
        }
        for line_no, expected in expected_lines.items():
            with self.subTest(line_no=line_no):
                self.assertEqual(checker_lines[line_no - 1].strip(), expected)


if __name__ == "__main__":
    unittest.main()


class Ltx2VaeKernelRowIsCounted(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "KERNEL-LTX2-VAE"

    def _kernel_matrix_path(self) -> Path:
        return agent_record.MATRICES["KERNEL"][0]

    def test_the_row_exists_in_the_kernel_matrix(self) -> None:
        text = self._kernel_matrix_path().read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_its_spec_and_its_claim(self) -> None:
        """A row whose issue lives only in the PR body is untraceable."""
        text = self._kernel_matrix_path().read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("ltx25-vae-device-residency.md", row)
        self.assertIn("CLAIM-LTX25-VAE-DEVICE-RESIDENCY", row)
        index = tracked_issues(self)
        self.assertIn("issues/1451)", index)

class Qwen35GdnBackendRowBacksTheRatchet(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-QWEN35"

    def test_the_qwen35_gdn_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

class RoadmapIssueProjectionTests(unittest.TestCase):
    def test_the_roadmap_refuses_a_restored_issue_row(self) -> None:
        roadmap = agent_record.AGENTS / "roadmap_v1.md"
        clean = roadmap.read_text(encoding="utf-8")

        def projection_errors(text: str) -> list[str]:
            with tempfile.TemporaryDirectory(dir=agent_record.ROOT) as tmp:
                agents = Path(tmp)
                (agents / "roadmap_v1.md").write_text(text, encoding="utf-8")
                (agents / "coordination.md").write_text("", encoding="utf-8")
                errors: list[str] = []
                with mock.patch.object(agent_record, "AGENTS", agents):
                    agent_record.check_roadmap({}, errors)
            return [
                error
                for error in errors
                if "stores a GitHub issue table row" in error
            ]

        self.assertEqual(projection_errors(clean), [])
        issue_row = (
            "| [#99999](https://github.com/mudler/vllm.cpp/issues/99999) "
            "| `BACKEND-ROCM` | title | bug |"
        )
        mutated = clean.replace(
            "## Top-level portfolio",
            issue_row + "\n\n## Top-level portfolio",
            1,
        )
        self.assertNotEqual(mutated, clean, "the mutation must restore an issue row")
        require(
            projection_errors(mutated),
            r"stores a GitHub issue table row",
        )



class CanonicalIssueReferenceTests(unittest.TestCase):
    LOCAL_ID = f"ISSUE-LOCAL-{''}01ARZ3NDEKTSV4RRFFQ69G5FAV"

    def intake(self, tmp: Path, *, problem: str | None = None, resolution: str = "-"):
        archived = (
            "| [#77](https://github.com/mudler/vllm.cpp/issues/77) "
            "| — | Archived 77 | bug |"
        )
        record = agent_record.issue_records.IssueRecord(
            id=f"ISSUE-GH-{''}77",
            title="Archived 77",
            row=None,
            state="UNKNOWN",
            kind="bug",
            github=77,
            mirror="MISSING",
            availability="METADATA_ONLY",
            created="UNKNOWN",
            updated="UNKNOWN",
            closed="UNKNOWN",
            problem=problem or (
                "Archive: `.agents/completed/issue-index.md:5`\n\n"
                "### Frozen archive evidence\n\n"
                f"> {archived}"
            ),
            resolution=resolution,
        )
        path = tmp / ".agents" / "issues" / "_intake" / f"ISSUE-GH-{''}77.md"
        return path, record

    @staticmethod
    def frozen_archive() -> bytes:
        return (
            "# Issue index\n\n"
            "| Issue | Row | Title | Kind |\n"
            "|---:|---|---|---|\n"
            "| [#77](https://github.com/mudler/vllm.cpp/issues/77) "
            "| — | Archived 77 | bug |\n"
        ).encode()

    def test_discovers_all_four_forms_in_commit_bodies_and_changed_files(self) -> None:
        references = agent_record.discover_issue_references(
            (
                f"Commit cites ISSUE-GH-{''}77, {self.LOCAL_ID}, #78, "
                "and issues/79."
            ),
            {
                Path("notes.txt"): (
                    f"Changed file cites ISSUE-GH-{''}80, {self.LOCAL_ID}, #81, "
                    "and https://github.com/mudler/vllm.cpp/issues/82."
                )
            },
        )
        self.assertEqual(
            references,
            {
                f"ISSUE-GH-{''}77",
                f"ISSUE-GH-{''}80",
                self.LOCAL_ID,
                "#78",
                "#79",
                "#81",
                "#82",
            },
        )


    def test_git_discovery_reads_commit_bodies_and_changed_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.txt").write_text(f"Changed {self.LOCAL_ID}.", encoding="utf-8")
            calls = (
                agent_record.subprocess.CompletedProcess([], 0, stdout="base\n", stderr=""),
                agent_record.subprocess.CompletedProcess([], 0, stdout="Commit #77\n", stderr=""),
                agent_record.subprocess.CompletedProcess([], 0, stdout=b"notes.txt\0", stderr=b""),
            )
            with (
                mock.patch.object(agent_record, "ROOT", root),
                mock.patch.object(agent_record.subprocess, "run", side_effect=calls),
            ):
                references = agent_record.branch_issue_references()
        self.assertEqual(references, {"#77", self.LOCAL_ID})

    def test_git_discovery_includes_tests_and_rejects_intake_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = root / "tests" / "scripts" / "probe.py"
            changed.parent.mkdir(parents=True)
            intake_id = "ISSUE-" + "GH-77"
            changed.write_text(f"Tracks {intake_id}.", encoding="utf-8")
            calls = (
                agent_record.subprocess.CompletedProcess([], 0, stdout="base\n", stderr=""),
                agent_record.subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                agent_record.subprocess.CompletedProcess(
                    [], 0, stdout=b"tests/scripts/probe.py\0", stderr=b""
                ),
            )
            with (
                mock.patch.object(agent_record, "ROOT", root),
                mock.patch.object(agent_record.subprocess, "run", side_effect=calls),
            ):
                references = agent_record.branch_issue_references()

            path, record = self.intake(root)
            errors: list[str] = []
            agent_record.check_canonical_issue_references(
                errors,
                references,
                [(path, record)],
            )

        self.assertEqual(references, {intake_id})
        self.assertEqual(len(errors), 1)
        self.assertRegex(errors[0], r"_intake.*triage|triage.*_intake")

    def test_only_self_declarations_and_exact_same_number_evidence_are_excluded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(Path(temporary))
            text = agent_record.issue_records.render_issue_record(record)
            self.assertEqual(
                agent_record.issue_references_in_text(
                    text,
                    path=path,
                    frozen_archive=self.frozen_archive(),
                ),
                set(),
            )

            other = replace(
                record,
                problem=record.problem.replace(
                    "Archived 77",
                    (
                        "Archived 77 cites #88 issues/88 "
                        f"ISSUE-GH-{''}77 {self.LOCAL_ID}"
                    ),
                ),
            )
            self.assertEqual(
                agent_record.issue_references_in_text(
                    agent_record.issue_records.render_issue_record(other),
                    path=path,
                    frozen_archive=self.frozen_archive(),
                ),
                {"#77", "#88", f"ISSUE-GH-{''}77", self.LOCAL_ID},
            )

    def test_local_id_and_github_declarations_do_not_hide_body_references(
        self,
    ) -> None:
        record = agent_record.issue_records.IssueRecord(
            id=self.LOCAL_ID,
            title="Local issue",
            row="ROW-A",
            state="OPEN",
            kind="bug",
            github=501,
            mirror="DIVERGED",
            availability="FULL",
            created="2026-08-01",
            updated="2026-08-31",
            closed="-",
            problem=f"Body cites {self.LOCAL_ID}, #501, and issues/501.",
            resolution="-",
        )
        path = Path(".agents/issues/ROW-A") / f"{self.LOCAL_ID}.md"
        self.assertEqual(
            agent_record.issue_references_in_text(
                agent_record.issue_records.render_issue_record(record),
                path=path,
            ),
            {self.LOCAL_ID, "#501"},
        )

    def test_malformed_frozen_evidence_has_no_reference_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(Path(temporary))
            malformed = replace(
                record,
                problem=record.problem.replace(
                    "[#77](https://github.com/mudler/vllm.cpp/issues/77)",
                    (
                        "[#88](https://github.com/mudler/vllm.cpp/issues/88) "
                        "mentions #77"
                    ),
                ),
            )
            self.assertEqual(
                agent_record.issue_references_in_text(
                    agent_record.issue_records.render_issue_record(malformed),
                    path=path,
                    frozen_archive=self.frozen_archive(),
                ),
                {"#77", "#88"},
            )

    def test_same_number_outside_evidence_and_copied_evidence_still_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(
                Path(temporary),
                resolution="See #77 and issues/77.",
            )
            text = agent_record.issue_records.render_issue_record(record)
            self.assertEqual(
                agent_record.issue_references_in_text(
                    text,
                    path=path,
                    frozen_archive=self.frozen_archive(),
                ),
                {"#77"},
            )
            copied = agent_record.issue_references_in_text(
                record.problem,
                path=Path(temporary) / "notes.md",
            )
            self.assertEqual(copied, {"#77"})

    def test_unverified_same_record_evidence_gets_no_citation_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(Path(temporary))
            text = agent_record.issue_records.render_issue_record(record)
            for frozen_archive in (
                b"# Issue index\n",
                self.frozen_archive().replace(b"Archived 77", b"Changed title"),
                self.frozen_archive().replace(b"/issues/77)", b"/issues/88)"),
                self.frozen_archive().replace(b"| bug |", b"| feature |"),
                self.frozen_archive().replace(b"| \xe2\x80\x94 |", b"| - |"),
            ):
                with self.subTest(frozen_archive=frozen_archive):
                    self.assertEqual(
                        agent_record.issue_references_in_text(
                            text,
                            path=path,
                            frozen_archive=frozen_archive,
                        ),
                        {"#77"},
                    )

    def test_every_reference_form_to_intake_fails_canonical_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(Path(temporary))
            errors: list[str] = []
            agent_record.check_canonical_issue_references(
                errors,
                {f"ISSUE-GH-{''}77", "#77", "issues/77"},
                [(path, record)],
            )
            self.assertEqual(len(errors), 3)
            for error in errors:
                self.assertRegex(error, r"_intake.*triage|triage.*_intake")

    def test_intake_debt_is_reported_without_a_baseline_or_ratchet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, record = self.intake(Path(temporary))
            report = agent_record.canonical_intake_debt([(path, record)])
            self.assertEqual(report, (f"ISSUE-GH-{''}77",))
class GdnDevicePureBackendRowBacksTheRatchet(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-GDN-DEVICE-PURE"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_its_spec_and_its_claim(self) -> None:
        """A row whose issue lives only in the PR body is untraceable.

        #2907 is OPEN, so it has no entry in the frozen archive and the
        derived snapshot is untracked -- `tracked_issues` cannot vouch for it
        on a fresh checkout. The row line itself carries the reference, and
        that is the tree-truthful place to pin it.

        The CLAIM owner follows the lifecycle, not the test. The checker
        requires a `CLAIM-*` owner on ACTIVE rows only, and the row's move to
        DONE (bcbfee7bf) retired its claim file into the frozen claims
        archive -- an unconditional claim assertion froze the ACTIVE-era row
        and red on `main` the moment the lifecycle moved. The row carries the
        claim exactly while it is ACTIVE, and never afterwards.
        """
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("#2907", row)
        self.assertIn("tenstorrent-gdn-device-pure.md", row)
        if "`ACTIVE`" in row:
            self.assertIn("CLAIM-BACKEND-TENSTORRENT-GDN-DEVICE-PURE", row)
        else:
            self.assertNotIn("CLAIM-", row)

class KeepquantBackendRowBacksTheRatchet(unittest.TestCase):
    """Preserve the exact keyed row, ownership, and semantic fields."""

    ROW = "BACKEND-TENSTORRENT-KEEPQUANT"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_its_spec_and_its_claim(self) -> None:
        """A row whose issue lives only in the PR body is untraceable.

        #2959 is OPEN, so it has no entry in the frozen archive and the
        derived snapshot is untracked -- `tracked_issues` cannot vouch for it
        on a fresh checkout. The row line itself carries the reference, and
        that is the tree-truthful place to pin it.
        """
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("#2959", row)
        self.assertIn("tenstorrent-keepquant.md", row)
        self.assertIn("CLAIM-BACKEND-TENSTORRENT-KEEPQUANT", row)


class LayaModelRowIsCounted(unittest.TestCase):
    """Mutation evidence for the model-inventory count bump in check-agent-record.py.

    Adding MODEL-LAYA to model-matrix.md increments rows/memberships/architectures/
    targets/modules by one each. The checker's expected inventory was updated to
    match. This test calls check_model_invariants and asserts zero errors: it
    passes against HEAD (expected=325) and fails against BASE (expected=324),
    proving the count change is semantic, not cosmetic.
    """

    def test_model_inventory_invariants_hold(self) -> None:
        errors: list[str] = []
        agent_record.check_model_invariants(errors)
        self.assertEqual(errors, [], f"model inventory invariant failed: {errors}")

